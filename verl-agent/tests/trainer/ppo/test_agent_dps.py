import itertools
import json
import os
import tempfile
import unittest

from agent_system.multi_turn_rollout.agent_dps import (
    AgentDPSNode,
    AgentDPSCurriculumCandidate,
    AgentDPSStateMemory,
    allocate_agent_dps,
    beta_event_utility,
    beta_mixed_utility,
    expected_mixed_gain,
    root_curriculum_key,
    select_agent_dps_curriculum,
)


class AgentDPSMemoryTests(unittest.TestCase):
    def test_root_curriculum_namespace_isolated_from_exact_branch_state(self):
        memory = AgentDPSStateMemory(decay=1.0)
        root = root_curriculum_key("task")
        exact = "exact-root-state"
        memory.observe_unique([(root, True, "bare")], step=0)
        memory.observe_unique([(exact, False, "branch")], step=0)

        root_probability, _, root_mass, _ = memory.posterior(root, step=0)
        exact_probability, _, exact_mass, _ = memory.posterior(exact, step=0)

        self.assertNotEqual(root, exact)
        self.assertEqual((root_mass, exact_mass), (1.0, 1.0))
        self.assertGreater(root_probability, 0.5)
        self.assertLess(exact_probability, 0.5)

    def test_posterior_query_does_not_create_or_mark_state_seen(self):
        memory = AgentDPSStateMemory(max_age_steps=2)
        probability, _, mass, visits = memory.posterior("unseen", 0)
        self.assertEqual((probability, mass, visits), (0.5, 0.0, 0))
        self.assertEqual(memory.states, {})

        memory.observe("seen", True, 0)
        memory.posterior("seen", 1)
        self.assertEqual(memory.states["seen"].last_seen_step, 0)
        memory.posterior("seen", 3)
        self.assertIn("seen", memory.states)
        self.assertAlmostEqual(memory.states["seen"].success_mass, 0.9**3)

    def test_memory_uses_soft_decay_instead_of_hard_horizon(self):
        memory = AgentDPSStateMemory(max_age_steps=2)
        memory.observe("state", True, 0)
        self.assertIn("state", memory.states)
        memory.metrics(2)
        self.assertIn("state", memory.states)
        memory.metrics(3)
        self.assertIn("state", memory.states)
        self.assertEqual(memory.pruned_total, 0)
        self.assertAlmostEqual(memory.states["state"].success_mass, 0.9**3)

    def test_recent_evidence_does_not_refresh_state(self):
        memory = AgentDPSStateMemory(max_age_steps=2)
        memory.observe("state", True, 0)
        self.assertTrue(memory.has_recent_evidence("state", 2))
        self.assertEqual(memory.states["state"].last_seen_step, 0)
        self.assertFalse(memory.has_recent_evidence("state", 3))

    def test_consumer_horizon_can_expire_replay_without_pruning_archive(self):
        memory = AgentDPSStateMemory(max_age_steps=32)
        memory.observe("rare", True, 0)

        self.assertTrue(memory.has_recent_evidence("rare", 8, max_age_steps=8))
        self.assertFalse(memory.has_recent_evidence("rare", 9, max_age_steps=8))
        self.assertIn("rare", memory.states)
        self.assertTrue(memory.has_recent_evidence("rare", 9))

    def test_revisited_state_compacts_soft_decayed_sufficient_statistics(self):
        memory = AgentDPSStateMemory(decay=1.0, max_age_steps=2)
        memory.observe("state", True, step=0)
        for step in range(1, 4):
            memory.observe("state", False, step=step)

        probability, _, mass, visits = memory.posterior("state", step=3)

        self.assertIn("state", memory.states)
        self.assertEqual(set(memory.states["state"].outcome_buckets), {3})
        self.assertEqual((mass, visits), (4.0, 4))
        self.assertAlmostEqual(probability, 2.0 / 6.0)

    def test_active_window_applies_age_decay_before_posterior(self):
        memory = AgentDPSStateMemory(decay=0.5, max_age_steps=8)
        memory.observe("state", True, step=0)
        memory.observe("state", False, step=2)

        probability, _, mass, visits = memory.posterior("state", step=2)

        self.assertAlmostEqual(memory.states["state"].success_mass, 0.25)
        self.assertAlmostEqual(memory.states["state"].failure_mass, 1.0)
        self.assertAlmostEqual(mass, 1.25)
        self.assertEqual(visits, 2)
        self.assertAlmostEqual(probability, 1.25 / 3.25)

    def test_hmm_combines_same_round_outcomes_and_predicts_partial_dynamics(self):
        memory = AgentDPSStateMemory(hmm_transition_decay=1.0)
        observations = [
            ("state", False, "failure-0"),
            ("state", True, "success-0"),
        ]
        memory.observe_unique(observations, step=0)
        memory.dps_belief("state", step=1)
        self.assertEqual(memory.states["state"].hmm_last_class, 1)

        memory.observe_unique(
            [("state", False, "failure-1"), ("state", True, "success-1")],
            step=1,
        )
        belief = memory.dps_belief("state", step=2)
        self.assertGreater(belief[1], belief[0])
        self.assertGreater(belief[1], belief[2])

    def test_hmm_transition_evidence_is_compacted_not_hard_deleted(self):
        memory = AgentDPSStateMemory(decay=1.0, max_age_steps=2, hmm_transition_decay=1.0)
        memory.observe("state", False, step=0)
        memory.dps_belief("state", step=1)
        memory.observe("state", False, step=1)
        memory.dps_belief("state", step=2)
        memory.observe("state", True, step=2)
        memory.dps_belief("state", step=3)
        memory.observe("state", True, step=3)
        memory.dps_belief("state", step=4)

        self.assertEqual(set(memory.states["state"].hmm_transition_buckets), {4})
        self.assertGreater(
            sum(sum(row) for row in memory.states["state"].hmm_transition_mass),
            0.0,
        )

    def test_graph_signal_uses_action_conditioned_future_values(self):
        memory = AgentDPSStateMemory(decay=1.0)
        for index in range(3):
            memory.observe("successor-good", True, step=0)
            memory.observe("successor-bad", False, step=0)
        memory.observe_transitions(
            [
                ("source", "take apple", "successor-good", True, "trajectory-good"),
                ("source", "open fridge", "successor-bad", False, "trajectory-bad"),
            ],
            step=0,
        )

        signal, dispersion, action_count, mass = memory.graph_signal("source", step=1)
        self.assertEqual(action_count, 2)
        self.assertEqual(mass, 2.0)
        self.assertGreater(signal, 0.0)
        self.assertGreater(dispersion, 0.0)

    def test_revisited_edge_compacts_soft_decayed_sufficient_statistics(self):
        memory = AgentDPSStateMemory(decay=1.0, max_age_steps=2)
        memory.observe_transitions(
            [("source", "take apple", "next", True, "trajectory-0")],
            step=0,
        )
        for step in range(1, 4):
            memory.observe_transitions(
                [("source", "take apple", "next", False, f"trajectory-{step}")],
                step=step,
            )

        edge = next(iter(memory.edges.values()))
        self.assertEqual(set(edge.evidence_buckets), {3})
        self.assertEqual(edge.visit_mass, 4.0)
        self.assertEqual(edge.success_mass, 1.0)
        self.assertEqual(edge.failure_mass, 3.0)

    def test_transition_contrast_uses_history_as_prior_and_current_once(self):
        same_edge = AgentDPSStateMemory(decay=1.0)
        different_edge = AgentDPSStateMemory(decay=1.0)
        for memory, action, next_state in (
            (same_edge, "open fridge", "next-a"),
            (different_edge, "take apple", "next-b"),
        ):
            memory.observe_unique(
                [("state", False, "historical-failure")], step=0
            )
            memory.observe_transitions(
                [
                    (
                        "state",
                        action,
                        next_state,
                        False,
                        "historical-failure",
                    )
                ],
                step=0,
            )

        current_outcomes = [(True, "current-success")]
        current_transitions = [
            ("open fridge", "next-a", True, "current-success")
        ]
        before_states = (
            same_edge.states["state"].success_mass,
            same_edge.states["state"].failure_mass,
        )
        before_edges = len(same_edge.edges)
        same = same_edge.transition_contrast_posterior(
            "state",
            1,
            factual_action="open fridge",
            factual_next_state="next-a",
            factual_outcome=True,
            current_outcomes=current_outcomes,
            current_transitions=current_transitions,
        )
        different = different_edge.transition_contrast_posterior(
            "state",
            1,
            factual_action="open fridge",
            factual_next_state="next-a",
            factual_outcome=True,
            current_outcomes=current_outcomes,
            current_transitions=current_transitions,
        )

        self.assertGreater(different.probability, same.probability)
        self.assertGreater(different.utility(4), same.utility(4))
        self.assertEqual(same.current_mass, 1.0)
        self.assertEqual(same.history_mass, 1.0)
        self.assertEqual(
            (
                same_edge.states["state"].success_mass,
                same_edge.states["state"].failure_mass,
            ),
            before_states,
        )
        self.assertEqual(len(same_edge.edges), before_edges)

    def test_saved_soft_statistics_round_trip_and_current_decay_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "agent_dps.json")
            memory = AgentDPSStateMemory(decay=0.9, max_age_steps=3)
            memory.observe("state", True, step=0)
            memory.observe("state", False, step=1)
            memory.observe_transitions([("state", "look", "next", False, "trajectory")], step=1)
            memory.task_sampler_state = {
                "schema_version": 1,
                "epoch": 2,
                "cursor": 17,
            }
            memory.save(path)

            loaded = AgentDPSStateMemory.load_or_create(path, decay=0.5, max_age_steps=1)
            probability, _, mass, visits = loaded.posterior("state", step=2)

            self.assertEqual(loaded.decay, 0.5)
            self.assertEqual(set(loaded.states["state"].outcome_buckets), {2})
            self.assertEqual((mass, visits), (0.95, 2))
            self.assertAlmostEqual(probability, 1.45 / 2.95)
            edge = next(iter(loaded.edges.values()))
            loaded.graph_signal("state", step=2)
            self.assertEqual(set(edge.evidence_buckets), {2})
            self.assertEqual(
                loaded.task_sampler_state,
                {"schema_version": 1, "epoch": 2, "cursor": 17},
            )

    def test_legacy_aggregate_archive_migrates_to_expiring_bucket(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "legacy_agent_dps.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "decay": 0.995,
                        "states": {
                            "state": {
                                "success_mass": 2.0,
                                "failure_mass": 1.0,
                                "last_step": 5,
                                "visit_count": 3,
                                "last_seen_step": 5,
                            }
                        },
                    },
                    handle,
                )

            loaded = AgentDPSStateMemory.load_or_create(path, decay=0.5, max_age_steps=2)

            self.assertTrue(loaded.dirty)
            self.assertEqual(loaded.decay, 0.5)
            self.assertEqual(loaded.states["state"].outcome_buckets, {5: [2.0, 1.0]})
            _, _, mass, visits = loaded.posterior("state", step=6)
            self.assertEqual((mass, visits), (1.5, 3))


class AgentDPSAllocatorTests(unittest.TestCase):
    def test_dynamic_curriculum_learns_repeated_partial_state(self):
        memory = AgentDPSStateMemory(hmm_transition_decay=0.9)
        for step in range(3):
            memory.observe_unique(
                [
                    ("frontier", False, f"frontier-f-{step}"),
                    ("frontier", True, f"frontier-s-{step}"),
                    ("hard", False, f"hard-{step}"),
                ],
                step=step,
            )
            memory.dps_belief("frontier", step + 1)
            memory.dps_belief("hard", step + 1)

        frontier = memory.dps_belief("frontier", 4)[1]
        hard = memory.dps_belief("hard", 4)[1]
        self.assertGreater(frontier, hard)
        selected, metrics = select_agent_dps_curriculum(
            [
                AgentDPSCurriculumCandidate(
                    "frontier", "frontier", frontier, 3.0
                ),
                AgentDPSCurriculumCandidate("hard", "hard", hard, 3.0),
            ],
            1,
            exploration_fraction=0.0,
            temperature=0.01,
            seed=7,
        )
        self.assertEqual(selected[0].task_ref, "frontier")
        self.assertGreater(metrics["agent_dps/root/frontier_selection_lift"], 0.0)

    def test_beta_predictive_objective_uses_evidence_not_only_mean_probability(self):
        # Both states have mean p=0.5, but the concentrated posterior is more
        # likely to produce a mixed group than the diffuse prior.  The root
        # allocator must use that posterior-predictive objective rather than
        # the legacy plug-in mean probability.
        self.assertGreater(beta_mixed_utility(20.0, 20.0, 2), beta_mixed_utility(1.0, 1.0, 2))
        nodes = [
            AgentDPSNode(
                state_key="diffuse",
                task_uid="task",
                anchor_id="diffuse",
                trajectory_id="trajectory-diffuse",
                probability=0.5,
                uncertainty=0.0,
                posterior_alpha=1.0,
                posterior_beta=1.0,
                tie_breaker=0.0,
            ),
            AgentDPSNode(
                state_key="concentrated",
                task_uid="task",
                anchor_id="concentrated",
                trajectory_id="trajectory-concentrated",
                probability=0.5,
                uncertainty=0.0,
                posterior_alpha=20.0,
                posterior_beta=20.0,
                tie_breaker=0.0,
            ),
        ]
        allocations, _, _ = allocate_agent_dps(
            nodes,
            budget=2,
            root_mode=True,
            initial_branch_size=2,
            min_new_branch=2,
            max_k=2,
        )
        self.assertEqual(len(allocations), 1)
        self.assertEqual(allocations[0].state_group_key, "concentrated")

    def test_hidden_factual_label_supports_one_branch_without_label_leakage(self):
        self.assertAlmostEqual(expected_mixed_gain(0.5, 1, existing_count=1), 0.5)
        nodes = [
            AgentDPSNode(
                state_key=key,
                task_uid="task",
                anchor_id=f"anchor-{key}",
                trajectory_id=f"trajectory-{key}",
                probability=0.5,
                uncertainty=0.0,
                existing_factual_count=1,
                tie_breaker=tie_breaker,
            )
            for key, tie_breaker in (("low", 0.1), ("high", 0.9))
        ]
        allocations, _, _ = allocate_agent_dps(
            nodes,
            budget=1,
            initial_branch_size=1,
            min_new_branch=1,
            max_k=1,
        )
        self.assertEqual(len(allocations), 1)
        self.assertEqual(allocations[0].state_group_key, "high")
        self.assertEqual(allocations[0].factual_count, 1)


    def test_branch_opening_is_atomic_and_locally_dense(self):
        nodes = [
            AgentDPSNode(
                state_key=f"state-{index}",
                task_uid="task",
                anchor_id=f"anchor-{index}",
                trajectory_id=f"trajectory-{index}",
                probability=0.5,
                uncertainty=0.0,
                existing_factual_count=1,
                tie_breaker=float(index),
            )
            for index in range(10)
        ]

        allocations, metrics, _ = allocate_agent_dps(nodes, budget=12, max_k=12)

        self.assertEqual(len(allocations), 3)
        self.assertEqual(sum(item.continuation_count for item in allocations), 12)
        self.assertTrue(all(item.continuation_count >= 4 for item in allocations))
        self.assertEqual(metrics["agent_dps/allocator/min_k"], 4.0)
        self.assertEqual(metrics["agent_dps/allocator/underdense_selected_states"], 0.0)

    def test_branch_budget_smaller_than_atomic_opening_is_not_spent(self):
        node = AgentDPSNode(
            state_key="state",
            task_uid="task",
            anchor_id="anchor",
            trajectory_id="trajectory",
            probability=0.5,
            uncertainty=0.0,
            existing_factual_count=1,
        )

        allocations, metrics, _ = allocate_agent_dps([node], budget=3, max_k=3)

        self.assertEqual(allocations, [])
        self.assertEqual(metrics["agent_dps/allocator/used_budget"], 0.0)
        self.assertEqual(metrics["agent_dps/allocator/unspent_budget"], 3.0)

    def test_top_m_uses_interaction_acquisition_before_marginal_allocation(self):
        nodes = [
            AgentDPSNode(
                state_key=f"state-{index}",
                task_uid="task",
                anchor_id=f"anchor-{index}",
                trajectory_id=f"trajectory-{index}",
                probability=0.5,
                uncertainty=0.0,
                acquisition_score=1.0 - index * 0.01,
                existing_factual_count=1,
                tie_breaker=float(index),
            )
            for index in range(10)
        ]

        allocations, metrics, _ = allocate_agent_dps(
            nodes,
            budget=12,
            max_k=12,
            candidate_top_m=2,
        )

        self.assertEqual(len(allocations), 2)
        self.assertEqual(sum(item.continuation_count for item in allocations), 12)
        self.assertEqual(
            {item.state_group_key for item in allocations},
            {"state-0", "state-1"},
        )
        self.assertEqual(metrics["agent_dps/allocator/prefiltered_states"], 8.0)
        self.assertEqual(metrics["agent_dps/allocator/acquisition_prefilter_enabled"], 1.0)
        self.assertEqual(metrics["agent_dps/allocator/mixed_objective_enabled"], 1.0)

    def test_transition_allocator_matches_global_knapsack_optimum(self):
        nodes = [
            AgentDPSNode(
                state_key=f"state-{index}",
                task_uid="task",
                anchor_id=f"anchor-{index}",
                trajectory_id=f"trajectory-{index}",
                probability=0.5,
                uncertainty=0.0,
                contrast_alpha=alpha,
                contrast_beta=beta,
                existing_factual_count=1,
                tie_breaker=float(index),
            )
            for index, (alpha, beta) in enumerate(
                ((3.0, 2.0), (1.0, 5.0), (0.25, 8.0))
            )
        ]
        allocations, metrics, _ = allocate_agent_dps(
            nodes,
            budget=12,
            initial_branch_size=4,
            min_new_branch=4,
            max_k=12,
            candidate_top_m=1,
        )
        allocated = {item.state_group_key: item.continuation_count for item in allocations}
        actual_score = sum(item.cumulative_utility for item in allocations)
        options = (0,) + tuple(range(4, 13))
        exhaustive = max(
            sum(
                0.0 if count == 0 else beta_event_utility(
                    nodes[index].contrast_alpha,
                    nodes[index].contrast_beta,
                    count,
                )
                for index, count in enumerate(counts)
            )
            for counts in itertools.product(options, repeat=len(nodes))
            if sum(counts) <= 12
        )

        self.assertAlmostEqual(actual_score, exhaustive)
        self.assertEqual(sum(allocated.values()), 12)
        self.assertEqual(metrics["agent_dps/allocator/posterior_optimal_knapsack"], 1.0)
        self.assertEqual(metrics["agent_dps/allocator/acquisition_prefilter_enabled"], 0.0)

    def test_root_acquisition_score_does_not_override_mixed_utility(self):
        nodes = [
            AgentDPSNode(
                state_key=f"root-{index}",
                task_uid=f"task-{index}",
                anchor_id=f"anchor-{index}",
                trajectory_id="",
                probability=0.5,
                uncertainty=0.0,
                acquisition_score=score,
                tie_breaker=tie,
            )
            for index, (score, tie) in enumerate(((0.01, 0.9), (100.0, 0.1)))
        ]
        allocations, _, _ = allocate_agent_dps(
            nodes,
            budget=5,
            initial_branch_size=2,
            min_new_branch=2,
            root_baseline=2,
            root_mode=True,
        )
        quotas = {item.state_group_key: item.continuation_count for item in allocations}
        self.assertGreater(quotas["root-0"], quotas["root-1"])

    def test_root_baseline_and_residual_are_global(self):
        nodes = [
            AgentDPSNode(
                state_key=f"s{index}",
                task_uid=f"task-{index}",
                anchor_id=f"anchor-{index}",
                trajectory_id="",
                probability=probability,
                uncertainty=0.0,
            )
            for index, probability in enumerate((0.5, 0.2, 0.8, 0.5))
        ]
        allocations, metrics, _ = allocate_agent_dps(
            nodes,
            budget=10,
            initial_branch_size=2,
            min_new_branch=2,
            max_k=10,
            root_baseline=2,
            root_mode=True,
        )
        self.assertEqual(sum(item.continuation_count for item in allocations), 10)
        self.assertTrue(all(item.continuation_count >= 2 for item in allocations))
        self.assertEqual(metrics["agent_dps/allocator/used_budget"], 10.0)
        self.assertEqual(len(allocations), 4)

    def test_task_coverage_and_caps_do_not_compete_with_global_utility(self):
        nodes = []
        for task_index in range(4):
            for state_index in range(8):
                nodes.append(
                    AgentDPSNode(
                        state_key=f"task-{task_index}-state-{state_index}",
                        task_uid=f"task-{task_index}",
                        anchor_id=f"anchor-{task_index}-{state_index}",
                        trajectory_id=f"trajectory-{task_index}-{state_index}",
                        probability=0.5,
                        uncertainty=0.0,
                        existing_factual_count=1,
                        tie_breaker=(1.0 if task_index == 0 else 0.0),
                    )
                )

        allocations, metrics, _ = allocate_agent_dps(
            nodes,
            budget=16,
            initial_branch_size=4,
            min_new_branch=4,
            max_k=4,
            task_min_coverage=True,
            task_max_budget_factor=1.5,
            task_diminishing_exponent=0.5,
        )
        by_task = {f"task-{index}": 0 for index in range(4)}
        for allocation in allocations:
            task = allocation.state_group_key.split("-state-", 1)[0]
            by_task[task] += allocation.continuation_count

        self.assertEqual(sum(by_task.values()), 16)
        self.assertEqual(metrics["agent_dps/allocator/task_min_coverage_enabled"], 0.0)
        self.assertEqual(metrics["agent_dps/allocator/task_budget_cap"], 0.0)
        self.assertEqual(metrics["agent_dps/allocator/mixed_objective_enabled"], 1.0)


if __name__ == "__main__":
    unittest.main()
