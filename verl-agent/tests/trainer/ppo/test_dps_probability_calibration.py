"""Probability, age, isolation and allocator contracts for calibrated DPS."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from types import MethodType
import unittest

import numpy as np

from agent_system.multi_turn_rollout.agent_dps import AgentDPSNode, AgentDPSStateMemory, allocate_agent_dps
from agent_system.multi_turn_rollout.dps_probability_calibration import (
    BetaMixtureForecast, ProbabilityCalibrator, audited_branch_groups, observe_groups,
)
from agent_system.multi_turn_rollout.dps_state_prior import StatePriorMemory, StatePriorPosterior


def row(task, success, failure, *, direct=(0., 0.), history=(0., 0.),
        semantic=(0., 0.), group='root'):
    return dict(task=task, group=group,
                history_success=history[0], history_failure=history[1],
                current_success=direct[0], current_failure=direct[1],
                direct_success=history[0]+direct[0], direct_failure=history[1]+direct[1],
                semantic_success=semantic[0], semantic_failure=semantic[1],
                success=success, failure=failure)


class ProbabilityTests(unittest.TestCase):
    def test_branch_group_requires_every_audited_terminal_suffix(self):
        from agent_system.multi_turn_rollout.agent_dps import state_key
        key = state_key('task', 'obs')
        selected = [dict(task_ref='task', state_group_key=key, allocated_k=4,
            state_prior=dict(depth=2, history_success_mass=0., history_failure_mass=0.,
                current_success_mass=1., current_failure_mass=0.,
                semantic_raw_success_mass=2., semantic_raw_failure_mass=1.))]
        rows = [dict(traj_uid=str(i), uid='task', turn_idx=2, anchor_obs='obs',
            active_masks=True, trajectory_success=float(i != 0), dps_env_done=True)
            for i in range(4)]
        records, skipped = audited_branch_groups(selected, rows, 50)
        self.assertEqual((records[0]['success'], records[0]['failure'], skipped), (3,1,0))
        self.assertEqual(records[0]['history_success'],0.)
        self.assertEqual(records[0]['current_success'],1.)
        rows[0]['dps_env_done'] = False
        self.assertEqual(audited_branch_groups(selected, rows, 50), ([],1))
        rows[0]['dps_env_done'] = True
        rows[0]['el_tsbpo_quarantined'] = True
        self.assertEqual(audited_branch_groups(selected, rows, 50), ([],1))
        # A wholly absent group has no legacy observed_count fields at all.
        self.assertEqual(audited_branch_groups(selected, [], 50), ([],1))

    def test_mixture_integrates_event_instead_of_mean_p(self):
        f = BetaMixtureForecast(np.array([.01, 10.]), np.array([10., .01]),
                                np.array([.5, .5]), {})
        self.assertAlmostEqual(f.mean, .5)
        self.assertLess(f.mixed_curve(6)[-1], .01)
        curve = f.mixed_curve(16)
        self.assertTrue(np.all(np.diff(curve) >= -1e-12))
        self.assertTrue(np.all(np.diff(curve, 2)[1:] <= 1e-12))

    def test_root_repeated_counts_learn_near_deterministic_population(self):
        archive = {}
        observe_groups(archive, 'root', [row(str(i), 6 if i%2 else 0, 0 if i%2 else 6)
                                       for i in range(80)], 0)
        model = ProbabilityCalibrator(archive, 1, 'root', 1.)
        forecast = model.forecast('new')
        self.assertLess(forecast.mixed_curve(6)[-1], .12)
        self.assertEqual(forecast.info['semantic_reliability'], 0.)

    def test_harmful_semantics_are_rejected_using_branch_counts(self):
        archive = {}
        records = [row(str(i), 8 if i%2 else 0, 0 if i%2 else 8,
                       direct=(1., 0.) if i%2 else (0., 1.),
                       semantic=(0., 4.) if i%2 else (4., 0.), group='state')
                   for i in range(100)]
        observe_groups(archive, 'branch', records, 0)
        forecast = ProbabilityCalibrator(archive, 1, 'branch', 1.).forecast(
            'new', semantic=(0., 4.), current=(1., 0.))
        self.assertTrue(forecast.info['semantic_identified'])
        self.assertLess(forecast.info['semantic_reliability'], .04)
        self.assertLess(forecast.contrast_probability(4, True), .15)

    def test_real_transfer_can_be_learned(self):
        rng, archive, records = np.random.default_rng(21), {}, []
        for i, p in enumerate(rng.beta(.5, .5, 250)):
            factual = int(rng.binomial(1, p))
            success = int(rng.binomial(12, p))
            records.append(row(str(i), success, 12-success,
                direct=(factual, 1-factual), semantic=(4*p, 4*(1-p))))
        observe_groups(archive, 'branch', records, 0)
        f = ProbabilityCalibrator(archive, 1, 'branch', 1.).forecast('new')
        self.assertGreater(f.info['semantic_reliability'], .2)

    def test_own_task_and_future_labels_are_excluded(self):
        archive = {}
        observe_groups(archive, 'root', [row(str(i), 6, 0) for i in range(12)], 0)
        original = ProbabilityCalibrator(archive, 2, 'root').forecast('target').mixed_curve(6)
        observe_groups(archive, 'root', [row('target', 0, 100)], 1)
        observe_groups(archive, 'root', [row('future', 0, 100)], 2)
        actual = ProbabilityCalibrator(archive, 2, 'root').forecast('target').mixed_curve(6)
        np.testing.assert_allclose(actual, original, rtol=0, atol=1e-12)

    def test_newer_outcomes_override_old_policy_without_hard_reset(self):
        archive = {}
        observe_groups(archive, 'root', [row(str(i), 0, 6) for i in range(30)], 0)
        observe_groups(archive, 'root', [row(str(i+30), 6, 0) for i in range(30)], 30)
        f = ProbabilityCalibrator(archive, 31, 'root', .8).forecast('new')
        self.assertGreater(f.mean, .85)

    def test_archive_deduplicates_and_survives_checkpoint(self):
        memory = AgentDPSStateMemory()
        records = [row('task', 6, 0, history=(2., 1.), semantic=(.3, .4))]
        self.assertEqual(observe_groups(memory.probability_calibration, 'root', records, 3), 1)
        self.assertEqual(observe_groups(memory.probability_calibration, 'root', records, 3), 0)
        with self.assertRaises(ValueError):
            observe_groups(memory.probability_calibration, 'root', [row('task', 0, 6)], 3)
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d)/'state.json')
            memory.save(p)
            loaded = AgentDPSStateMemory.load_or_create(p)
            self.assertEqual(loaded.probability_calibration, memory.probability_calibration)

    def test_root_allocator_uses_integrated_curve(self):
        nodes = [AgentDPSNode(state_key=str(i), task_uid=str(i), anchor_id=str(i),
            trajectory_id='', probability=.5, uncertainty=.1,
            posterior_alpha=1., posterior_beta=1., max_k=12,
            root_utility_curve=BetaMixtureForecast(np.array([a]), np.array([b]),
                np.ones(1), {}).mixed_curve(12)) for i, (a,b) in enumerate([(2., 2.), (100., .01)])]
        allocations, _, _ = allocate_agent_dps(nodes, 12, initial_branch_size=2,
            min_new_branch=2, max_k=12, root_baseline=2, root_mode=True)
        counts = {a.state_group_key:a.continuation_count for a in allocations}
        self.assertEqual(sum(counts.values()), 12)
        self.assertGreater(counts['0'], counts['1'])

    def test_joint_particles_apply_learned_transfer_without_duplicate_labels(self):
        memory, archive = StatePriorMemory(), {}
        memory.observe([dict(id='own', task='target', cluster='own', kind='success',
            terminal_observed=True, suffixes=[dict(state='s', depth=2)]),
            dict(id='donor', task='other', cluster='donor', kind='failure',
            terminal_observed=True, suffixes=[dict(state='d', depth=2)])], 0, 50)
        r = memory.pools['d:depth=2']['buckets']['0']['0:donor']
        semantic = SimpleNamespace(donors=lambda *_: (
            [('0:donor', r, 1.)], dict(semantic_mass=1.)))
        posterior = StatePriorPosterior(memory, 1, 50, decay=1., particles=8192,
            semantic_prior=semantic, probability_calibrator=ProbabilityCalibrator(archive, 1, 'branch'))
        info = posterior.state_diagnostics('target', 's', 2)
        self.assertEqual(info['semantic_raw_failure_mass'], 1.)
        self.assertEqual(info['semantic_failure_mass'], 0.)
        self.assertEqual(info['current_mass'], 0.)
        self.assertAlmostEqual(posterior.success_particles('target', 's', 2).mean(), .75, delta=.015)

    def test_semantic_weight_uncertainty_is_integrated_per_particle(self):
        memory = StatePriorMemory()
        paths = [dict(id='own', task='target', cluster='own', kind='success',
            terminal_observed=True, suffixes=[dict(state='s', depth=2)])]
        paths.extend(dict(id=f'donor{i}', task='other', cluster='donor', kind='failure',
            terminal_observed=True, suffixes=[dict(state='d', depth=2)]) for i in range(4))
        memory.observe(paths, 0, 50)
        donor = memory.pools['d:depth=2']['buckets']['0']['0:donor']
        semantic = SimpleNamespace(donors=lambda *_: (
            [('0:donor', donor, 1.)], dict(semantic_mass=4.)))
        count = 32768
        aa = np.tile([.2,2.], count//2)
        bb = np.tile([.8,1.], count//2)
        rho = np.tile([0.,1.], count//2)
        calibrator = SimpleNamespace(population_particles=lambda *_: (aa,bb,np.ones(count),rho,{}))
        posterior = StatePriorPosterior(memory, 0, 50, particles=count,
            semantic_prior=semantic, probability_calibrator=calibrator)
        samples = posterior.success_particles('target','s',2)
        exact = BetaMixtureForecast(np.array([1.2,3.]), np.array([.8,5.]),
                                    np.array([.5,.5]), {})
        self.assertAlmostEqual(samples.mean(), exact.mean, delta=.008)
        self.assertAlmostEqual(samples.var(), exact.variance, delta=.006)

    def test_stale_history_is_learned_separately_from_current_evidence(self):
        archive = {}
        records = [row(str(i), 12 if i%2 else 0, 0 if i%2 else 12,
            history=(0.,20.) if i%2 else (20.,0.),
            direct=(1.,0.) if i%2 else (0.,1.)) for i in range(80)]
        observe_groups(archive, 'branch', records, 0)
        forecast = ProbabilityCalibrator(archive, 1, 'branch', 1.).forecast(
            'target', (0.,20.), current=(1.,0.))
        self.assertTrue(forecast.info['history_identified'])
        self.assertLess(forecast.info['history_reliability'], .01)
        self.assertGreater(forecast.mean, .99)
        self.assertLess(forecast.contrast_probability(4, True), .03)

    def test_predictive_history_is_not_forcibly_discarded(self):
        rng, archive, records = np.random.default_rng(7), {}, []
        for i, p in enumerate(rng.beta(.5, .5, 160)):
            history = int(rng.binomial(20,p))
            current = int(rng.binomial(1,p))
            success = int(rng.binomial(12,p))
            records.append(row(str(i), success, 12-success,
                history=(history,20-history), direct=(current,1-current)))
        observe_groups(archive, 'branch', records, 0)
        forecast = ProbabilityCalibrator(archive, 1, 'branch', 1.).forecast('target')
        self.assertGreater(forecast.info['history_reliability'], .25)

    def test_history_particles_keep_current_bare_counts_unscaled(self):
        memory = StatePriorMemory()
        memory.observe([dict(id=f'history{i}', task='target', cluster='history',
            kind='failure', terminal_observed=True,
            suffixes=[dict(state='s', depth=2)]) for i in range(20)], 0, 50)
        memory.observe([dict(id='current', task='target', cluster='current',
            kind='success', terminal_observed=True,
            suffixes=[dict(state='s', depth=2)])], 1, 50)
        count = 16384
        eta = np.tile([0.,.5], count//2)
        aa, bb = np.full(count,.2), np.full(count,.8)
        model = SimpleNamespace(population_particles=lambda *_: (aa,bb,eta,np.zeros(count),{}))
        posterior = StatePriorPosterior(memory, 1, 50, decay=.8, particles=count,
                                        probability_calibrator=model)
        exact = BetaMixtureForecast(np.array([1.2,1.2]), np.array([.8,8.8]),
                                   np.array([.5,.5]), {})
        samples = posterior.success_particles('target','s',2)
        self.assertAlmostEqual(samples.mean(), exact.mean, delta=.012)
        self.assertAlmostEqual(samples.var(), exact.variance, delta=.012)
        info = posterior.state_diagnostics('target','s',2)
        self.assertEqual(info['history_failure_mass'],16.)
        self.assertEqual(info['history_effective_failure_mass'],4.)
        self.assertEqual(info['current_success_mass'],1.)

    def test_legacy_branch_merged_counts_are_not_guessed(self):
        legacy = dict(task='old', group='s', step=0, direct_success=5., direct_failure=1.,
                      semantic_success=0., semantic_failure=0., success=4, failure=0)
        model = ProbabilityCalibrator({'branch':{'old':legacy}}, 1, 'branch')
        self.assertEqual(model.diagnostics()['groups'],0.)
        self.assertEqual(model.diagnostics()['legacy_groups_skipped'],1.)
        with self.assertRaisesRegex(ValueError,'separate history/current'):
            observe_groups({},'branch',[legacy],1)
        archive = {}
        observe_groups(archive,'root',[legacy],1)
        self.assertEqual(next(iter(archive['root'].values()))['history_success'],5.)

    def test_root_cannot_condition_on_its_own_current_outcomes(self):
        with self.assertRaisesRegex(ValueError,'current rollout'):
            observe_groups({},'root',[row('t',6,0,direct=(1.,0.))],0)
        with self.assertRaisesRegex(ValueError,'current rollout'):
            ProbabilityCalibrator({},0,'root').forecast('t',current=(1.,0.))

    def test_collector_records_root_counts_before_branch_and_only_once(self):
        from omegaconf import OmegaConf
        from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
        memory, snapshots = AgentDPSStateMemory(), {}
        cfg = OmegaConf.create(dict(agent_dps_probability_calibration=True,
            agent_dps_root_similarity_enabled=False, agent_dps_allocation_diagnostics_dir='test'))
        collector = SimpleNamespace(config=OmegaConf.create(dict(env=dict(max_steps=50,
            rollout=dict(n=6)))), _el_tsbpo_step=125,
            _get_el_tsbpo_config=lambda: cfg, _get_agent_dps_memory=lambda: memory,
            _save_agent_dps_memory=lambda: None,
            _write_agent_dps_allocation=lambda name, data: snapshots.update({name:data}))
        for name in ('_get_agent_dps_probability_calibrator', '_agent_dps_root_parameters',
                     '_agent_dps_root_frontier'):
            setattr(collector, name, MethodType(getattr(TrajectoryCollector, name), collector))
        quotas, _, _ = TrajectoryCollector._allocate_agent_dps_roots(collector,
            ['obs']*3, [{'extra.gamefile':f'task{i}'} for i in range(3)], 18)
        rows = [dict(traj_uid=f'{i}-{j}', uid=f'task{i}', turn_idx=0, anchor_obs='obs',
            active_masks=True, trajectory_success=float(i%2), dps_env_done=True)
            for i in range(3) for j in range(int(quotas[i]))]
        first = TrajectoryCollector._agent_dps_observe_continuations(collector, rows)
        second = TrajectoryCollector._agent_dps_observe_continuations(collector, rows)
        self.assertEqual(first['agent_dps/probability/root/groups_committed'], 3.)
        self.assertEqual(second['agent_dps/probability/root/groups_committed'], 0.)
        self.assertEqual(len(memory.probability_calibration['root']), 3)
        self.assertTrue(all(r['current_success']==r['current_failure']==0.
                            for r in memory.probability_calibration['root'].values()))
        model = collector._get_agent_dps_probability_calibrator('branch')
        f = model.forecast('task0')
        self.assertEqual(f.info['source'], 'bare_warm_start')
        self.assertEqual(f.info['groups'], 2)
        self.assertEqual(f.info['semantic_reliability'], 0.)

    def test_invalid_and_inconsistent_feature_counts_are_rejected(self):
        r = row('t',4,0,history=(1.,0.),direct=(1.,0.))
        r['direct_success'] = 3.
        with self.assertRaisesRegex(ValueError,'disagrees'):
            observe_groups({},'branch',[r],0)
        with self.assertRaisesRegex(ValueError,'Invalid probability forecast'):
            ProbabilityCalibrator({},0,'branch').forecast('t',current=(float('nan'),0.))

    def test_probability_metrics_survive_quality_filter(self):
        from verl.trainer.ppo.metric_utils import _EL_TSBPO_QUALITY_METRICS
        for key in ['agent_dps/probability/root/brier',
                    'agent_dps/probability/root/semantic_reliability',
                    'agent_dps/probability/root/history_reliability',
                    'agent_dps/probability/branch/history_disabled_probability',
                    'agent_dps/probability/branch/legacy_groups_skipped',
                    'agent_dps/probability/branch/groups_committed']:
            self.assertIn(key, _EL_TSBPO_QUALITY_METRICS)


if __name__ == '__main__':
    unittest.main()
