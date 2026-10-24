"""Check the physical-rollout contract shared by collection and state priors."""
from types import MethodType, SimpleNamespace
import unittest

from agent_system.multi_turn_rollout.agent_dps import AgentDPSStateMemory, state_key
from agent_system.multi_turn_rollout.dps_state_prior import StatePriorMemory, StatePriorPosterior
from agent_system.multi_turn_rollout.dps_survival import continuation_records


def trajectory(uid, *, start=0, length=3, success=False, done=True, truncated=False,
               parent=None):
    rows = [dict(traj_uid=uid, uid='task', turn_idx=turn, anchor_obs=f'state-{turn}',
                 dps_task_text='put the apple in the bowl', active_masks=True,
                 trajectory_success=float(success), dps_env_done=False,
                 dps_env_truncated=False)
            for turn in range(start, start + length)]
    rows[-1].update(dps_env_done=done, dps_env_truncated=truncated)
    if parent:
        for row in rows:
            row['el_tsbpo_bare_rollout_id'] = parent
    return rows


class ContinuationRecordTests(unittest.TestCase):
    def test_completed_bare_populates_all_state_priors_once(self):
        records = continuation_records(trajectory('bare', success=True), 50, 0)
        self.assertTrue(records[0]['terminal_observed'])
        self.assertEqual(records[0]['cluster'], 'bare')
        self.assertEqual(records[0]['suffixes'], [
            dict(state=state_key('task', f'state-{turn}'), depth=turn) for turn in range(3)])
        memory = StatePriorMemory()
        self.assertEqual(memory.observe(records, 0, 50), 1)
        self.assertEqual(memory.observe(records, 0, 50), 0)
        posterior = StatePriorPosterior(memory, 0, 50, particles=32)
        for turn in range(3):
            info = posterior.state_diagnostics('task', state_key('task', f'state-{turn}'), turn)
            self.assertEqual(info['current_success_mass'], 1)
            self.assertEqual(info['current_failure_mass'], 0)

    def test_complete_failure_and_horizon_timeout_are_both_terminal_evidence(self):
        cases = [
            (trajectory('failure'), 'failure'),
            (trajectory('failure-at-horizon', start=47, length=3), 'failure'),
            (trajectory('truncated', done=False, truncated=True), 'timeout'),
            (trajectory('horizon', start=47, length=3, done=False), 'timeout'),
        ]
        for rows, kind in cases:
            with self.subTest(kind=kind, uid=rows[0]['traj_uid']):
                record = continuation_records(rows, 50, 0, branch=True)[0]
                self.assertEqual(record['kind'], kind)
                self.assertTrue(record['terminal_observed'])
                memory = StatePriorMemory()
                self.assertEqual(memory.observe([record], 0, 50), 1)
                counts = next(iter(memory.pools.values()))['buckets']['0']
                self.assertEqual(next(iter(counts.values()))['failure'], 1)

    def test_incomplete_success_label_is_not_terminal_evidence(self):
        # A trajectory label can be broadcast to every row; a partial prefix
        # must not become a completed sample just because it carries it.
        for success in (False, True):
            rows = trajectory('partial', success=success, done=False)
            record = continuation_records(rows, 50, 0)[0]
            self.assertFalse(record['terminal_observed'])
            memory = StatePriorMemory()
            self.assertEqual(memory.observe([record], 0, 50), 0)
            self.assertEqual(memory.pools, {})
            self.assertEqual(memory.seen_ids, set())
            rows[-1]['dps_env_done'] = True
            self.assertEqual(memory.observe(continuation_records(rows, 50, 0), 0, 50), 1)

    def test_branch_continuations_share_parent_cluster_without_replayed_prefix(self):
        bare = trajectory('bare', success=True)
        branches = trajectory('branch-1', start=1, length=2, parent='bare')
        branches += trajectory('branch-2', start=2, length=1, success=True, parent='bare')
        memory = StatePriorMemory()
        self.assertEqual(memory.observe(continuation_records(bare, 50, 0), 0, 50), 1)
        records = continuation_records(branches, 50, 0, branch=True)
        self.assertEqual([r['cluster'] for r in records], ['bare', 'bare'])
        self.assertEqual([len(r['suffixes']) for r in records], [2, 1])
        self.assertEqual(memory.observe(records, 0, 50), 2)
        self.assertEqual(memory.clusters, {'0:bare': dict(step=0, paths=3)})
        self.assertEqual(len(memory.pools), 3)
        info = StatePriorPosterior(memory, 0, 50, particles=32).state_diagnostics(
            'task', state_key('task', 'state-2'), 2)
        self.assertEqual(info['current_success_mass'], 2)
        self.assertEqual(info['current_failure_mass'], 1)
        self.assertEqual(info['current_clusters'], 1)

    def test_quarantine_gaps_and_duplicate_turns_never_enter_history(self):
        for variant in ('quarantine', 'gap', 'duplicate'):
            rows = trajectory('bad')
            if variant == 'quarantine':
                rows[1]['el_tsbpo_quarantined'] = True
            elif variant == 'gap':
                rows.pop(1)
            else:
                rows.append(dict(rows[-1]))
            self.assertEqual(continuation_records(rows, 50, 0), [])

    def test_collector_stages_then_commits_bare_and_branch_with_semantic_registration(self):
        from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector

        class Similarity:
            stats = dict(encode_seconds=0.)

            def __init__(self):
                self.registrations = []

            def reset_stats(self):
                pass

            def register(self, memory, descriptors):
                self.registrations.append(list(descriptors))

        memory, similarity, summaries = AgentDPSStateMemory(), Similarity(), []
        collector = SimpleNamespace(
            config=SimpleNamespace(env=SimpleNamespace(max_steps=50)), _el_tsbpo_step=0,
            _get_agent_dps_memory=lambda: memory,
            _get_el_tsbpo_config=lambda: dict(agent_dps_branch_similarity=dict(enabled=True)),
            _get_agent_dps_branch_similarity=lambda: similarity,
            _write_agent_dps_allocation=lambda stage, summary: summaries.append((stage, summary)),
        )
        collect = MethodType(TrajectoryCollector._agent_dps_observe_continuations, collector)
        rows = trajectory('bare', success=True) + trajectory('partial', done=False)
        metrics = collect(rows)
        self.assertEqual(metrics['agent_dps/state_prior/bare_paths'], 1)
        self.assertEqual(metrics['agent_dps/state_prior/bare_incomplete'], 1)
        self.assertEqual(metrics['agent_dps/state_prior/bare_state_views'], 3)
        self.assertEqual(len(similarity.registrations[0]), 3)
        self.assertEqual(collect(rows)['agent_dps/state_prior/bare_paths'], 0)
        memory.survival_memory = collector._agent_dps_pending_state_prior.dump()
        metrics = collect(trajectory('branch', start=1, length=2, parent='bare'), branch=True)
        self.assertEqual(metrics['agent_dps/state_prior/branch_paths'], 1)
        committed = StatePriorMemory(memory.survival_memory)
        self.assertEqual(committed.clusters['0:bare']['paths'], 2)
        self.assertEqual(committed.seen_ids, {'bare', 'branch'})


if __name__ == '__main__':
    unittest.main()
