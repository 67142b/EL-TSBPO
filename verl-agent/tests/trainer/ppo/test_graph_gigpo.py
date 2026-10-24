import unittest

import numpy as np
import torch

from el_tsbpo.core_el_tsbpo import compute_el_tsbpo_advantage


class TestGraphGiGPO(unittest.TestCase):
    def _compute(
        self,
        *,
        token_rewards,
        step_rewards,
        masks,
        tasks,
        trajectories,
        states,
        actions,
        invalid_penalty=None,
    ):
        bsz = len(tasks)
        return compute_el_tsbpo_advantage(
            token_level_rewards=torch.tensor(token_rewards, dtype=torch.float32),
            response_mask=torch.tensor(masks, dtype=torch.float32),
            index=np.asarray(tasks, dtype=object),
            traj_index=np.asarray(trajectories, dtype=object),
            parent_node_id=np.asarray([f"p{i}" for i in range(bsz)], dtype=object),
            node_id=np.asarray([f"n{i}" for i in range(bsz)], dtype=object),
            advantage_mode="graph_gigpo",
            gigpo_step_rewards=torch.tensor(step_rewards, dtype=torch.float32),
            gigpo_anchor_obs=np.asarray(states, dtype=object),
            gigpo_action_text=np.asarray(actions, dtype=object),
            gigpo_mode="mean_norm",
            gigpo_step_advantage_w=1.0,
            invalid_action_penalty=(
                None
                if invalid_penalty is None
                else torch.tensor(invalid_penalty, dtype=torch.float32)
            ),
        )

    def test_sparse_states_use_gigpo_episode_fallback(self):
        advantages, _, components = self._compute(
            token_rewards=[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            step_rewards=[1.0, 0.0],
            masks=[[1, 1, 0], [1, 1, 0]],
            tasks=["task", "task"],
            trajectories=["success", "failure"],
            states=["state-a", "state-b"],
            actions=["go north", "go south"],
        )
        pair = 0.5
        expected = torch.tensor([[pair, pair, 0.0], [-pair, -pair, 0.0]])
        torch.testing.assert_close(advantages, expected)
        torch.testing.assert_close(components["trajectory"], expected)
        self.assertEqual(
            components["tree_diagnostics"]["graph_gigpo_fallback_rows"], 2
        )

    def test_action_edges_share_values_with_trajectory_centered_normalization(self):
        advantages, _, components = self._compute(
            token_rewards=[[0.0, 0.0]] * 3,
            step_rewards=[1.0, 0.0, 0.0],
            masks=[[1, 1]] * 3,
            tasks=["task"] * 3,
            trajectories=["t1", "t2", "t3"],
            states=["same state"] * 3,
            actions=["open door", "open door", "look"],
        )
        expected = torch.tensor(
            [[2.0 / 3.0] * 2, [-1.0 / 3.0] * 2, [-1.0 / 3.0] * 2]
        )
        torch.testing.assert_close(advantages, expected)
        self.assertAlmostEqual(
            components["tree_diagnostics"]["graph_gigpo_local_row_scalar_mean"],
            0.0,
            places=6,
        )
        self.assertTrue(torch.count_nonzero(components["trajectory"]) == 0)
        diagnostics = components["tree_diagnostics"]
        self.assertEqual(diagnostics["graph_gigpo_multi_action_states"], 1)
        self.assertEqual(diagnostics["graph_gigpo_unique_edges"], 2)

    def test_homogeneous_multi_action_state_has_zero_local_advantage(self):
        advantages, _, components = self._compute(
            token_rewards=[[0.0, 1.0], [0.0, 1.0]],
            step_rewards=[1.0, 1.0],
            masks=[[1, 1], [1, 1]],
            tasks=["task", "task"],
            trajectories=["t1", "t2"],
            states=["same state", "same state"],
            actions=["take apple", "take mug"],
        )
        torch.testing.assert_close(advantages, torch.zeros_like(advantages))
        self.assertEqual(
            components["tree_diagnostics"]["graph_gigpo_homogeneous_states"], 1
        )

    def test_repeated_state_rows_use_row_weighted_graph_baseline(self):
        # The first action is observed three times while the second is
        # observed once.  GIGPO's state baseline is over four observed turns,
        # not over two unique trajectories/actions.
        _, _, components = compute_el_tsbpo_advantage(
            token_level_rewards=torch.zeros(4, 1),
            response_mask=torch.ones(4, 1),
            index=np.asarray(["task"] * 4, dtype=object),
            traj_index=np.asarray(["t1", "t1", "t2", "t3"], dtype=object),
            parent_node_id=np.asarray(["p0", "p1", "p2", "p3"], dtype=object),
            node_id=np.asarray(["n0", "n1", "n2", "n3"], dtype=object),
            terminal_reward=np.asarray([10.0, 10.0, 10.0, 0.0]),
            trajectory_success=np.asarray([1.0, 1.0, 1.0, 0.0]),
            node_is_terminal=np.asarray([True] * 4),
            advantage_mode="graph_gigpo",
            gigpo_step_rewards=torch.tensor([10.0, 10.0, 10.0, 0.0]),
            gigpo_anchor_obs=np.asarray(["same"] * 4, dtype=object),
            gigpo_action_text=np.asarray(["open", "open", "open", "look"], dtype=object),
            gigpo_mode="mean_norm",
        )
        graph_values = components["sibling"][:, 0]
        torch.testing.assert_close(graph_values, torch.tensor([2.5, 2.5, 2.5, -7.5]))
        self.assertAlmostEqual(float(graph_values.mean()), 0.0, places=6)

    def test_discounted_return_is_used_once_and_padding_is_never_broadcast(self):
        gamma = 0.9
        advantages, _, _ = self._compute(
            token_rewards=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            # These are the canonical GIGPO G_t values produced before
            # advantage computation. Graph-GiGPO must not apply gamma again.
            step_rewards=[gamma, 0.0],
            masks=[[1, 1, 0], [1, 0, 0]],
            tasks=["task", "task"],
            trajectories=["success", "failure"],
            states=["same state", "same state"],
            actions=["go east", "go west"],
        )
        pair = gamma / 2.0
        expected = torch.tensor([[pair, pair, 0.0], [-pair, 0.0, 0.0]])
        torch.testing.assert_close(advantages, expected)

    def test_invalid_action_penalty_is_normalized_as_reward(self):
        advantages, _, components = self._compute(
            # The trainer has already subtracted 0.1 from row zero.
            token_rewards=[[0.0, -0.1, 0.0], [0.0, 0.0, 0.0]],
            step_rewards=[-0.1, 0.0],
            masks=[[1, 1, 0], [1, 1, 0]],
            tasks=["task", "task"],
            trajectories=["t1", "t2"],
            states=["same state", "same state"],
            actions=["invalid", "look"],
            invalid_penalty=[0.1, 0.0],
        )
        # The penalty directly changes only row zero's reward. Episode and
        # same-state baselines then turn that reward difference into a
        # centered relative advantage for both sampled actions.
        pair = 0.05
        expected = torch.tensor(
            [[-2.0 * pair, -2.0 * pair, 0.0], [2.0 * pair, 2.0 * pair, 0.0]]
        )
        torch.testing.assert_close(advantages, expected)
        torch.testing.assert_close(
            components["invalid_action"], torch.zeros_like(expected)
        )

    def test_invalid_action_penalty_enters_explicit_tree_return_before_normalization(self):
        advantages, _, components = compute_el_tsbpo_advantage(
            token_level_rewards=torch.tensor([[-0.1], [0.0]]),
            response_mask=torch.ones(2, 1),
            index=np.asarray(["task", "task"], dtype=object),
            traj_index=np.asarray(["invalid", "valid"], dtype=object),
            parent_node_id=np.asarray(["root", "root"], dtype=object),
            node_id=np.asarray(["invalid-leaf", "valid-leaf"], dtype=object),
            terminal_reward=np.asarray([0.0, 0.0]),
            trajectory_success=np.asarray([0.0, 0.0]),
            node_is_terminal=np.asarray([True, True]),
            advantage_mode="graph_gigpo",
            gigpo_step_rewards=torch.tensor([-0.1, 0.0]),
            gigpo_anchor_obs=np.asarray(["same-state", "same-state"], dtype=object),
            gigpo_action_text=np.asarray(["invalid", "look"], dtype=object),
            gigpo_mode="mean_norm",
            invalid_action_penalty=torch.tensor([0.1, 0.0]),
        )

        # Physical-subtree and graph returns both see the penalized reward
        # before mean normalization. No direct penalty or root episode
        # component is added afterwards.
        torch.testing.assert_close(
            components["tree_reward"].reshape(-1), torch.tensor([-0.1, 0.0])
        )
        torch.testing.assert_close(
            components["episode"].reshape(-1), torch.tensor([0.0, 0.0])
        )
        pair = 0.05
        torch.testing.assert_close(
            components["trajectory"].reshape(-1), torch.tensor([-pair, pair])
        )
        torch.testing.assert_close(
            components["sibling"].reshape(-1), torch.tensor([-pair, pair])
        )
        torch.testing.assert_close(
            advantages.reshape(-1), torch.tensor([-2.0 * pair, 2.0 * pair])
        )
        torch.testing.assert_close(
            components["invalid_action"], torch.zeros_like(advantages)
        )

    def test_empty_executed_action_is_a_distinct_graph_edge(self):
        advantages, _, components = self._compute(
            token_rewards=[[0.0, 0.0], [0.0, 0.0]],
            step_rewards=[0.0, 1.0],
            masks=[[1, 1], [1, 1]],
            tasks=["task", "task"],
            trajectories=["empty", "valid"],
            states=["same state", "same state"],
            actions=["", "look"],
        )
        pair = 0.5
        expected = torch.tensor([[-pair, -pair], [pair, pair]])
        torch.testing.assert_close(advantages, expected)
        self.assertEqual(
            components["tree_diagnostics"]["graph_gigpo_unique_edges"], 2
        )

    def test_missing_action_provenance_still_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "canonical action_text"):
            self._compute(
                token_rewards=[[0.0, 0.0]],
                step_rewards=[0.0],
                masks=[[1, 1]],
                tasks=["task"],
                trajectories=["missing"],
                states=["state"],
                actions=[None],
            )

    def test_all_quarantined_rows_return_zero_without_building_groups(self):
        advantages, _, components = compute_el_tsbpo_advantage(
            token_level_rewards=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            response_mask=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
            index=np.asarray(["task"], dtype=object),
            traj_index=np.asarray(["traj"], dtype=object),
            parent_node_id=np.asarray(["parent"], dtype=object),
            node_id=np.asarray(["node"], dtype=object),
            sample_weight=torch.zeros(1),
            advantage_mode="graph_gigpo",
            gigpo_step_rewards=torch.tensor([1.0]),
            gigpo_anchor_obs=np.asarray(["state"], dtype=object),
            gigpo_action_text=np.asarray(["look"], dtype=object),
            gigpo_mode="mean_norm",
        )
        torch.testing.assert_close(advantages, torch.zeros_like(advantages))
        self.assertEqual(
            components["tree_diagnostics"]["graph_gigpo_state_groups"], 0
        )

    def test_recursive_subtree_advantage_broadcasts_and_switches_at_nested_split(self):
        """A nested branch is a new rollout root, not another ancestor sum."""

        advantages, _, components = compute_el_tsbpo_advantage(
            token_level_rewards=torch.zeros(4, 1),
            response_mask=torch.ones(4, 1),
            index=np.asarray(["task"] * 4, dtype=object),
            traj_index=np.asarray(["a", "b", "a1", "a2"], dtype=object),
            parent_node_id=np.asarray(["root", "root", "a", "a"], dtype=object),
            node_id=np.asarray(["a", "b", "a1", "a2"], dtype=object),
            terminal_reward=np.asarray([0.0, 0.0, 10.0, 0.0]),
            trajectory_success=np.asarray([0.0, 0.0, 1.0, 0.0]),
            node_is_terminal=np.asarray([False, True, True, True]),
            advantage_mode="graph_gigpo",
            gigpo_step_rewards=torch.tensor([10.0, 0.0, 10.0, 0.0]),
            gigpo_anchor_obs=np.asarray(["root", "root", "a", "a"], dtype=object),
            gigpo_action_text=np.asarray(["go-a", "go-b", "open", "look"], dtype=object),
            gigpo_mode="mean_norm",
            graph_gigpo_segment_balance=False,
        )

        # Root children compare backed-up returns [5, 0], while children of
        # ``a`` compare [10, 0]. Mean normalization retains those reward
        # ranges. The nested split replaces, rather than adds to, the root
        # trajectory advantage.
        torch.testing.assert_close(
            components["trajectory"].reshape(-1),
            torch.tensor([2.5, -2.5, 5.0, -5.0]),
        )
        torch.testing.assert_close(
            components["episode"].reshape(-1),
            torch.zeros(4),
        )
        torch.testing.assert_close(
            components["recursive_subtree_delta"].reshape(-1),
            torch.zeros(4),
        )
        sibling = components["sibling"].reshape(-1)
        self.assertAlmostEqual(float(sibling[:2].sum()), 0.0, places=6)
        self.assertAlmostEqual(float(sibling[2:].sum()), 0.0, places=6)
        self.assertGreater(float(sibling[0]), 0.0)
        self.assertGreater(float(sibling[2]), 0.0)
        torch.testing.assert_close(
            advantages,
            components["trajectory"] + components["sibling"],
        )
        torch.testing.assert_close(
            components["tree_reward"].reshape(-1),
            torch.tensor([5.0, 0.0, 10.0, 0.0]),
        )
        self.assertEqual(
            components["tree_diagnostics"]["graph_gigpo_recursive_subtree_rows"],
            4,
        )
        self.assertEqual(
            components["tree_diagnostics"]["graph_gigpo_fallback_rows"], 0
        )

    def test_nearest_subtree_advantage_reaches_unary_suffix_turns(self):
        _, _, components = compute_el_tsbpo_advantage(
            token_level_rewards=torch.zeros(3, 1),
            response_mask=torch.ones(3, 1),
            index=np.asarray(["task"] * 3, dtype=object),
            traj_index=np.asarray(["success", "failure", "success"], dtype=object),
            parent_node_id=np.asarray(["root", "root", "success"], dtype=object),
            node_id=np.asarray(["success", "failure", "success-leaf"], dtype=object),
            terminal_reward=np.asarray([0.0, 0.0, 1.0]),
            trajectory_success=np.asarray([1.0, 0.0, 1.0]),
            node_is_terminal=np.asarray([False, True, True]),
            advantage_mode="graph_gigpo",
            gigpo_step_rewards=torch.tensor([1.0, 0.0, 1.0]),
            # Keep graph states singleton to isolate subtree broadcasting.
            gigpo_anchor_obs=np.asarray(["root-a", "root-b", "suffix"], dtype=object),
            gigpo_action_text=np.asarray(["go", "stop", "finish"], dtype=object),
            gigpo_mode="mean_norm",
            graph_gigpo_segment_balance=False,
        )

        # The success child has two turns while the failure child has one.
        # Ordinary GIGPO row measure does not reweight those turns.
        torch.testing.assert_close(
            components["trajectory"].reshape(-1),
            torch.tensor([0.5, -0.5, 0.5]),
        )
        torch.testing.assert_close(
            components["raw_trajectory"].reshape(-1),
            torch.tensor([0.5, -0.5, 0.5]),
        )
        torch.testing.assert_close(
            components["trajectory_policy_weight"],
            torch.ones(3),
        )
        self.assertEqual(
            components["tree_diagnostics"]["graph_gigpo_recursive_subtree_rows"],
            3,
        )

    def test_tree_rows_use_unweighted_gigpo_measure(self):
        _, _, components = compute_el_tsbpo_advantage(
            token_level_rewards=torch.zeros(4, 4),
            response_mask=torch.tensor(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            ),
            index=np.asarray(["task"] * 4, dtype=object),
            traj_index=np.asarray(["factual", "factual", "factual", "branch"], dtype=object),
            parent_node_id=np.asarray(["root", "f1", "f2", "root"], dtype=object),
            node_id=np.asarray(["f1", "f2", "f3", "b1"], dtype=object),
            terminal_reward=np.asarray([0.0, 0.0, 10.0, 0.0]),
            trajectory_success=np.asarray([1.0, 1.0, 1.0, 0.0]),
            node_is_terminal=np.asarray([False, False, True, True]),
            advantage_mode="graph_gigpo",
            gigpo_step_rewards=torch.tensor([10.0, 10.0, 10.0, 0.0]),
            # Singleton state groups isolate hierarchical trajectory credit.
            gigpo_anchor_obs=np.asarray(["s0", "s1", "s2", "s3"], dtype=object),
            gigpo_action_text=np.asarray(["a0", "a1", "a2", "b0"], dtype=object),
            gigpo_mode="mean_norm",
            graph_gigpo_segment_balance=False,
        )

        expected_weights = torch.ones(4)
        torch.testing.assert_close(
            components["trajectory_policy_weight"], expected_weights
        )
        row_lengths = torch.tensor([4.0, 2.0, 3.0, 1.0])
        objective_rows = components["trajectory"].sum(dim=-1) / row_lengths
        torch.testing.assert_close(objective_rows, torch.tensor([5.0, 5.0, 5.0, -5.0]))
        self.assertAlmostEqual(float(objective_rows.mean()), 2.5, places=6)
        diagnostics = components["tree_diagnostics"]
        self.assertEqual(diagnostics["graph_gigpo_segment_groups"], 1)
        self.assertEqual(diagnostics["graph_gigpo_segments"], 2)
        self.assertAlmostEqual(
            diagnostics["graph_gigpo_sibling_child_mass_relative_gap_max"],
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            diagnostics["graph_gigpo_weighted_trajectory_objective_mean"],
            2.5,
            places=6,
        )
        self.assertEqual(
            diagnostics["graph_gigpo_tree_policy_weight_enabled"], 0
        )

    def test_recursive_trajectory_mass_balances_sibling_children(self):
        advantages, _, components = compute_el_tsbpo_advantage(
            token_level_rewards=torch.zeros(4, 4),
            response_mask=torch.tensor(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            ),
            index=np.asarray(["task"] * 4, dtype=object),
            traj_index=np.asarray(["factual"] * 3 + ["branch"], dtype=object),
            parent_node_id=np.asarray(["root", "f1", "f2", "root"], dtype=object),
            node_id=np.asarray(["f1", "f2", "f3", "b1"], dtype=object),
            terminal_reward=np.asarray([0.0, 0.0, 10.0, 0.0]),
            trajectory_success=np.asarray([1.0, 1.0, 1.0, 0.0]),
            node_is_terminal=np.asarray([False, False, True, True]),
            advantage_mode="graph_gigpo",
            gigpo_step_rewards=torch.tensor([10.0, 10.0, 10.0, 0.0]),
            gigpo_anchor_obs=np.asarray(["s0", "s1", "s2", "s3"], dtype=object),
            gigpo_action_text=np.asarray(["a0", "a1", "a2", "b0"], dtype=object),
            gigpo_mode="mean_norm",
            graph_gigpo_segment_balance=True,
            graph_gigpo_policy_loss_agg_mode="seq-mean-token-mean",
        )

        weights = components["trajectory_policy_weight"]
        torch.testing.assert_close(
            weights,
            torch.tensor([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0]),
        )
        trajectory = components["trajectory"]
        turn_objectives = trajectory.sum(dim=-1) / torch.tensor(
            [4.0, 3.0, 2.0, 1.0]
        )
        torch.testing.assert_close(
            turn_objectives,
            torch.tensor([10.0 / 3.0, 10.0 / 3.0, 10.0 / 3.0, -10.0]),
        )
        # Equalized child mass makes the trajectory objective zero-sum while
        # preserving the group's total policy mass.
        self.assertAlmostEqual(float(turn_objectives.sum()), 0.0, places=5)
        diagnostics = components["tree_diagnostics"]
        self.assertEqual(diagnostics["graph_gigpo_segment_balanced_policy"], 1)
        self.assertEqual(diagnostics["graph_gigpo_segment_cap_hit_fraction"], 0.0)
        self.assertAlmostEqual(
            diagnostics["graph_gigpo_sibling_child_mass_relative_gap_max"],
            0.0,
            places=5,
        )
        self.assertAlmostEqual(
            diagnostics["graph_gigpo_raw_sibling_child_mass_relative_gap_max"],
            1.0,
            places=5,
        )

    def test_token_mean_segment_measure_uses_valid_token_count(self):
        _, _, components = compute_el_tsbpo_advantage(
            token_level_rewards=torch.zeros(3, 4),
            response_mask=torch.tensor(
                [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
            ),
            index=np.asarray(["task"] * 3, dtype=object),
            traj_index=np.asarray(["factual", "factual", "branch"], dtype=object),
            parent_node_id=np.asarray(["root", "f1", "root"], dtype=object),
            node_id=np.asarray(["f1", "f2", "b1"], dtype=object),
            terminal_reward=np.asarray([0.0, 10.0, 0.0]),
            trajectory_success=np.asarray([1.0, 1.0, 0.0]),
            node_is_terminal=np.asarray([False, True, True]),
            advantage_mode="graph_gigpo",
            gigpo_step_rewards=torch.tensor([10.0, 10.0, 0.0]),
            gigpo_anchor_obs=np.asarray(["s0", "s1", "s2"], dtype=object),
            gigpo_action_text=np.asarray(["a0", "a1", "b0"], dtype=object),
            gigpo_mode="mean_norm",
            graph_gigpo_segment_balance=True,
            graph_gigpo_policy_loss_agg_mode="token-mean",
        )
        # The factual child has six active response tokens, the branch has
        # one. Their unclipped balancing ratio is 7:1 and therefore bounded.
        weights = components["trajectory_policy_weight"]
        self.assertLessEqual(float(weights.max()), 6.0 + 1e-5)
        self.assertGreater(float(weights[2]), float(weights[0]))
        self.assertEqual(
            components["tree_diagnostics"][
                "graph_gigpo_segment_policy_measure_token"
            ],
            1,
        )

    def test_state_credit_uses_ordinary_row_weighted_gigpo_baseline(self):
        _, _, components = compute_el_tsbpo_advantage(
            token_level_rewards=torch.zeros(3, 1),
            response_mask=torch.ones(3, 1),
            index=np.asarray(["task"] * 3, dtype=object),
            traj_index=np.asarray(["factual", "factual", "branch"], dtype=object),
            parent_node_id=np.asarray(["root", "f1", "root"], dtype=object),
            node_id=np.asarray(["f1", "f2", "b1"], dtype=object),
            terminal_reward=np.asarray([0.0, 10.0, 0.0]),
            trajectory_success=np.asarray([1.0, 1.0, 0.0]),
            node_is_terminal=np.asarray([False, True, True]),
            advantage_mode="graph_gigpo",
            gigpo_step_rewards=torch.tensor([1.0, 1.0, 0.0]),
            gigpo_anchor_obs=np.asarray(["same-state"] * 3, dtype=object),
            gigpo_action_text=np.asarray(["go", "finish", "stop"], dtype=object),
            gigpo_mode="mean_norm",
            graph_gigpo_segment_balance=False,
        )

        weights = components["trajectory_policy_weight"]
        state_rows = components["sibling"].reshape(-1)
        # State groups are centered over observed turns, exactly as in GIGPO.
        self.assertAlmostEqual(float(state_rows.sum()), 0.0, places=6)
        self.assertGreater(float(state_rows[0]), 0.0)
        self.assertGreater(float(state_rows[1]), 0.0)
        self.assertLess(float(state_rows[2]), 0.0)
        torch.testing.assert_close(weights, torch.ones_like(weights))
        diagnostics = components["tree_diagnostics"]
        self.assertEqual(
            diagnostics["graph_gigpo_unified_tree_policy_measure"], 0
        )
        self.assertLessEqual(
            diagnostics["graph_gigpo_state_group_weighted_mean_abs_max"],
            1e-6,
        )

    def test_homogeneous_subtree_is_retained_with_zero_relative_credit(self):
        advantages, _, components = compute_el_tsbpo_advantage(
            token_level_rewards=torch.ones(2, 1),
            response_mask=torch.ones(2, 1),
            index=np.asarray(["task", "task"], dtype=object),
            traj_index=np.asarray(["a", "b"], dtype=object),
            parent_node_id=np.asarray(["root", "root"], dtype=object),
            node_id=np.asarray(["a", "b"], dtype=object),
            terminal_reward=np.asarray([1.0, 1.0]),
            trajectory_success=np.asarray([1.0, 1.0]),
            node_is_terminal=np.asarray([True, True]),
            advantage_mode="graph_gigpo",
            gigpo_step_rewards=torch.tensor([1.0, 1.0]),
            gigpo_anchor_obs=np.asarray(["root", "root"], dtype=object),
            gigpo_action_text=np.asarray(["left", "right"], dtype=object),
            gigpo_mode="mean_norm",
        )

        torch.testing.assert_close(advantages, torch.zeros_like(advantages))
        torch.testing.assert_close(
            components["rollout_weight"], torch.ones_like(components["rollout_weight"])
        )
        self.assertEqual(
            components["tree_diagnostics"][
                "graph_gigpo_recursive_homogeneous_subtree_groups"
            ],
            1,
        )
        self.assertEqual(
            components["tree_diagnostics"]["graph_gigpo_fallback_rows"], 0
        )

    def test_terminal_reward_scale_is_preserved_by_mean_normalization(self):
        advantages, _, components = compute_el_tsbpo_advantage(
            token_level_rewards=torch.zeros(2, 1),
            response_mask=torch.ones(2, 1),
            index=np.asarray(["task", "task"], dtype=object),
            traj_index=np.asarray(["success", "failure"], dtype=object),
            parent_node_id=np.asarray(["root", "root"], dtype=object),
            node_id=np.asarray(["success", "failure"], dtype=object),
            terminal_reward=np.asarray([10.0, 0.0]),
            # This field is only a binary diagnostic and must not override the
            # explicit environment reward above.
            trajectory_success=np.asarray([1.0, 0.0]),
            node_is_terminal=np.asarray([True, True]),
            advantage_mode="graph_gigpo",
            gigpo_step_rewards=torch.tensor([10.0, 0.0]),
            gigpo_anchor_obs=np.asarray(["root", "root"], dtype=object),
            gigpo_action_text=np.asarray(["finish", "stop"], dtype=object),
            gigpo_mode="mean_norm",
        )

        # The explicit environment reward remains 10/0 in the raw tree return,
        # while both relative components retain that scale after centering.
        pair = 5.0
        torch.testing.assert_close(
            components["trajectory"].reshape(-1), torch.tensor([pair, -pair])
        )
        torch.testing.assert_close(
            components["sibling"].reshape(-1), torch.tensor([pair, -pair])
        )
        torch.testing.assert_close(
            advantages.reshape(-1), torch.tensor([2.0 * pair, -2.0 * pair])
        )
        torch.testing.assert_close(
            components["tree_reward"].reshape(-1), torch.tensor([10.0, 0.0])
        )

    def test_branches_inherit_bare_root_episode_baseline(self):
        _, _, components = compute_el_tsbpo_advantage(
            token_level_rewards=torch.zeros(5, 1),
            response_mask=torch.ones(5, 1),
            index=np.asarray(["task"] * 5, dtype=object),
            traj_index=np.asarray(
                ["bare-good", "bare-bad", "branch-1", "branch-2", "branch-3"],
                dtype=object,
            ),
            bare_rollout_id=np.asarray(
                ["bare-good", "bare-bad", "bare-good", "bare-good", "bare-good"],
                dtype=object,
            ),
            parent_node_id=np.asarray(["p0", "p1", "p2", "p3", "p4"], dtype=object),
            node_id=np.asarray(["n0", "n1", "n2", "n3", "n4"], dtype=object),
            terminal_reward=np.asarray([10.0, 0.0, 0.0, 0.0, 0.0]),
            trajectory_success=np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
            node_is_terminal=np.asarray([True] * 5),
            advantage_mode="graph_gigpo",
            gigpo_step_rewards=torch.zeros(5),
            # Singleton graph states isolate the root episode signal.
            gigpo_anchor_obs=np.asarray([f"state-{i}" for i in range(5)], dtype=object),
            gigpo_action_text=np.asarray(["look"] * 5, dtype=object),
            gigpo_mode="mean_norm",
        )

        torch.testing.assert_close(
            components["episode"].reshape(-1),
            torch.tensor([
                5.0,
                -5.0,
                5.0,
                5.0,
                5.0,
            ]),
        )
        diagnostics = components["tree_diagnostics"]
        self.assertEqual(diagnostics["graph_gigpo_root_episode_trajectories"], 2)
        self.assertEqual(diagnostics["graph_gigpo_branch_inherited_episode_rows"], 3)

    def test_mean_norm_is_supported_for_hierarchical_graph_credit(self):
        advantages, _, components = compute_el_tsbpo_advantage(
            token_level_rewards=torch.zeros(2, 1),
            response_mask=torch.ones(2, 1),
            index=np.asarray(["task", "task"], dtype=object),
            traj_index=np.asarray(["a", "b"], dtype=object),
            parent_node_id=np.asarray(["root", "root"], dtype=object),
            node_id=np.asarray(["a", "b"], dtype=object),
            terminal_reward=np.asarray([10.0, 0.0]),
            node_is_terminal=np.asarray([True, True]),
            advantage_mode="graph_gigpo",
            gigpo_step_rewards=torch.tensor([10.0, 0.0]),
            gigpo_anchor_obs=np.asarray(["root", "root"], dtype=object),
            gigpo_action_text=np.asarray(["left", "right"], dtype=object),
            gigpo_mode="mean_norm",
            graph_gigpo_segment_balance=False,
        )
        torch.testing.assert_close(
            components["trajectory"].reshape(-1), torch.tensor([5.0, -5.0])
        )
        torch.testing.assert_close(
            components["sibling"].reshape(-1), torch.tensor([5.0, -5.0])
        )
        torch.testing.assert_close(
            advantages.reshape(-1), torch.tensor([10.0, -10.0])
        )
        torch.testing.assert_close(
            components["trajectory_policy_weight"], torch.ones(2)
        )
        diagnostics = components["tree_diagnostics"]
        self.assertEqual(diagnostics["graph_gigpo_mean_std_norm"], 0)
        self.assertEqual(
            diagnostics["graph_gigpo_tree_policy_weight_enabled"], 0
        )


if __name__ == "__main__":
    unittest.main()
