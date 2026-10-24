"""EL-TSBPO dps state prior implementation."""
from __future__ import annotations

import copy
import hashlib
import math
from collections import defaultdict

import numpy as np


class StatePriorMemory:
    """Store bounded state outcome evidence across collection rounds."""

    def __init__(self, data=None):
        data = data or {}
        version = int(data.get('version', 1))
        if version > 3:
            raise ValueError(f'Unsupported branch state-prior version: {version}')
        self.max_steps = data.get('max_steps')
        self.seen_step = int(data.get('seen_step', -1))
        self.seen_ids = set(data.get('seen_ids', ()))
        self.pools = copy.deepcopy(data.get('state_pools', {}))
        self.clusters = copy.deepcopy(data.get('clusters', {}))
        self.migration = dict(data.get('migration', {}))


        self.semantic_keys = copy.deepcopy(data.get('semantic_keys', {}))
        if version < 3:
            paths = data.get('suffix_paths', {})


            self.migration = dict(source_version=version, migrated_paths=len(paths),
                                  dropped_origin_contexts=len(data.get('contexts', {})))
            for path_id, path in sorted(paths.items()):
                self._check_horizon(path['max_steps'])
                self._add_path(str(path_id), str(path['task']), int(path['step']),
                               str(path['cluster']), bool(path['success']), path['suffixes'])


            self.seen_ids.intersection_update(
                path_id.split(':', 1)[1] for path_id, path in paths.items()
                if int(path['step']) == self.seen_step and ':' in path_id)

    def _check_horizon(self, max_steps):
        max_steps = int(max_steps)
        if max_steps <= 0 or (self.max_steps is not None and int(self.max_steps) != max_steps):
            raise ValueError('Cannot reuse state priors with a different episode horizon')
        self.max_steps = max_steps

    def _add_path(self, path_id, task, step, cluster, success, views):
        unique = {}
        for view in views:
            state, depth = str(view['state']), int(view['depth'])
            if not state or not 0 <= depth < self.max_steps:
                raise ValueError('Invalid state-prior suffix/deadline')
            unique[(state, depth)] = None
        if not unique:
            return
        metadata = self.clusters.setdefault(cluster, dict(step=step, paths=0))
        if int(metadata['step']) != step:
            raise ValueError('Sampling cluster spans multiple collection steps')
        metadata['paths'] += 1
        signature = int(hashlib.sha256(path_id.encode()).hexdigest()[:32], 16)
        for state, depth in unique:
            key = f'{state}:depth={depth}'
            pool = self.pools.setdefault(key, dict(task=task, state=state, depth=depth, buckets={}))
            if pool['task'] != task:
                raise ValueError('Conflicting task identity in state-prior pool')
            row = pool['buckets'].setdefault(str(step), {}).setdefault(
                cluster, dict(success=0, failure=0, signature='0'))
            row['success' if success else 'failure'] += 1


            row['signature'] = format(int(row['signature'], 16) ^ signature, 'x')

    def observe(self, records, step, max_steps, *, max_entries=100000, max_age_steps=0):
        self._check_horizon(max_steps)
        step = int(step)
        if self.seen_step != step:
            self.seen_ids.clear()
            self.seen_step = step
        added = 0
        for record in records:
            uid = str(record['id'])
            if uid in self.seen_ids:
                continue
            if record['kind'] not in {'success', 'failure', 'timeout'}:
                raise ValueError('Unknown state-prior terminal kind')
            if record.get('terminal_observed', False) and record.get('suffixes'):
                self._add_path(f'{step}:{uid}', str(record['task']), step,
                               f"{step}:{record.get('cluster', uid)}",
                               record['kind'] == 'success', record['suffixes'])


                self.seen_ids.add(uid)
                added += 1
        self.prune(step, max_entries=max_entries, max_age_steps=max_age_steps)
        return added

    def prune(self, step, *, max_entries=100000, max_age_steps=0):
        entries = []
        for key, pool in self.pools.items():
            for source, rows in list(pool['buckets'].items()):
                if max_age_steps > 0 and int(step) - int(source) > max_age_steps:
                    del pool['buckets'][source]
                else:
                    entries.extend((int(source), cluster, key) for cluster in rows)
        excess = len(entries) - max(1, int(max_entries))
        if excess > 0:
            for source, cluster, key in sorted(entries)[:excess]:
                del self.pools[key]['buckets'][str(source)][cluster]
        used = set()
        for key, pool in list(self.pools.items()):
            pool['buckets'] = {s: rows for s, rows in pool['buckets'].items() if rows}
            if not pool['buckets']:
                del self.pools[key]
            else:
                used.update(c for rows in pool['buckets'].values() for c in rows)
        self.clusters = {c: row for c, row in self.clusters.items() if c in used}
        live_states = {pool['state'] for pool in self.pools.values()}
        self.semantic_keys = {s: keys for s, keys in self.semantic_keys.items() if s in live_states}

    def dump(self):
        return dict(version=3, max_steps=self.max_steps, seen_step=self.seen_step,
                    seen_ids=sorted(self.seen_ids), state_pools=self.pools,
                    clusters=self.clusters, migration=self.migration,
                    semantic_keys=self.semantic_keys,
                    )


class StatePriorPosterior:
    """Estimate continuation outcomes from historical and current evidence."""

    def __init__(self, memory, step, max_steps, decay=.9, particles=512, seed=17,
                 semantic_prior=None, empirical_bayes=False, probability_calibrator=None):
        memory._check_horizon(max_steps)
        self.memory, self.step = memory, int(step)
        self.max_steps, self.particles, self.seed = int(max_steps), int(particles), int(seed)
        self.decay = float(decay)
        if self.particles < 32 or not math.isfinite(self.decay) or not 0 < self.decay <= 1:
            raise ValueError('Invalid state-prior particle count or decay')
        self._priors, self._trees, self._weights, self._predictions = {}, {}, {}, {}
        self.semantic_prior = semantic_prior
        self.empirical_bayes = probability_calibrator is not None
        self.probability_calibrator = probability_calibrator
        self._calibration_draws = {}
        self.calibrator = None

    def _rng(self, key):
        digest = hashlib.sha256(f'{self.seed}:state-prior:{key}'.encode()).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], 'big'))

    def _cluster_weights(self, cluster, row, scale=1.):
        scale = np.asarray(scale, dtype=float)
        scale_key = (float(scale) if scale.ndim == 0 else hashlib.sha256(scale.tobytes()).hexdigest())
        cache_key = (cluster, row['signature'], row['success'], row['failure'], scale_key)
        if cache_key not in self._weights:
            total = int(self.memory.clusters[cluster]['paths'])
            counts = [row['success'], row['failure'],
                      total - row['success'] - row['failure']]
            if min(counts) < 0:
                raise ValueError('State-prior counts exceed their physical sampling cluster')
            tree_key = (cluster, scale_key)
            if tree_key not in self._trees:
                rng = self._rng(f'tree:{tree_key}')


                self._trees[tree_key] = (rng.gamma(total*scale, 1., self.particles)
                    if self.empirical_bayes else rng.exponential(size=self.particles))
            rng = self._rng(f'within:{cache_key}')
            weights = np.zeros((3, self.particles))
            for i, count in enumerate(counts):
                if count:
                    weights[i] = rng.gamma(count*scale if self.empirical_bayes else count,
                                           1., self.particles)
            weights /= np.maximum(weights.sum(axis=0), 1e-300)
            weights *= self._trees[tree_key]
            self._weights[cache_key] = weights[:2] * (1. if self.empirical_bayes else total)
        return self._weights[cache_key]

    def _predict(self, task, state, depth):
        key = (str(task), str(state), int(depth))
        if key in self._predictions:
            return self._predictions[key]
        base_alpha, base_beta, calibration = (.5, .5, {})
        eta, rho = 1., 1.
        if self.calibrator is not None:
            base_alpha, base_beta, calibration = self.calibrator.parameters(key[0])
        if self.probability_calibrator is not None:
            if key[0] not in self._calibration_draws:
                self._calibration_draws[key[0]] = self.probability_calibrator.population_particles(
                    key[0], self.particles, self.seed)
            aa, bb, eta, rho, calibration = self._calibration_draws[key[0]]
            base_alpha, base_beta = float(aa.mean()), float(bb.mean())
        if key[0] not in self._priors:
            shape = (np.stack([aa, bb]) if self.probability_calibrator is not None
                     else np.asarray([base_alpha, base_beta])[:, None])
            self._priors[key[0]] = self._rng(f'prior:{key[0]}').gamma(shape, 1., (2, self.particles))
        positive, negative = self._priors[key[0]].copy()
        history, current = np.zeros(2), np.zeros(2)
        history_clusters, current_clusters = set(), set()
        count = 0
        pool = self.memory.pools.get(f'{key[1]}:depth={key[2]}')
        if pool and pool['task'] != key[0]:
            raise ValueError('State-prior query task does not match evidence')
        buckets = sorted(pool['buckets'].items(), key=lambda item: int(item[0])) if pool else ()
        for source, rows in buckets:
            age = self.step - int(source)
            if age < 0:
                continue
            factor = self.decay ** age
            for cluster, row in sorted(rows.items()):
                raw = np.array([row['success'], row['failure']], dtype=float)
                if age:
                    history += factor * raw
                    history_clusters.add(cluster)
                else:
                    current += raw
                    current_clusters.add(cluster)


                scale = factor * eta if age else 1.
                sw, fw = self._cluster_weights(cluster, row, scale if self.empirical_bayes else 1.)
                positive += sw if self.empirical_bayes else factor * sw
                negative += fw if self.empirical_bayes else factor * fw
                count += int(raw.sum())
        shared = np.zeros(2)
        semantic_info = dict(semantic_neighbors=0, semantic_mass=0.,
                             semantic_max_similarity=0., semantic_clusters=0)
        if self.semantic_prior is not None:
            donors, semantic_info = self.semantic_prior.donors(key[1], key[2])
            for cluster, row, factor in donors:
                sw, fw = self._cluster_weights(cluster, row,
                    factor*rho if self.empirical_bayes else 1.)
                positive += sw if self.empirical_bayes else factor * sw
                negative += fw if self.empirical_bayes else factor * fw
                shared += factor * np.array([row['success'], row['failure']], dtype=float)
        raw_shared = shared.copy()
        shared *= float(np.mean(rho))
        effective_history = history * float(np.mean(eta))
        samples = positive / np.maximum(positive + negative, 1e-300)
        hmass, cmass = float(history.sum()), float(current.sum())
        info = dict(history_success_mass=float(history[0]), history_failure_mass=float(history[1]),
                    history_effective_success_mass=float(effective_history[0]),
                    history_effective_failure_mass=float(effective_history[1]),
                    current_success_mass=float(current[0]), current_failure_mass=float(current[1]),
                    base_prior_alpha=base_alpha, base_prior_beta=base_beta,
                    prior_alpha=base_alpha + float(effective_history[0] + shared[0]),
                    prior_beta=base_beta + float(effective_history[1] + shared[1]),
                    posterior_alpha=base_alpha + float(effective_history[0] + shared[0] + current[0]),
                    posterior_beta=base_beta + float(effective_history[1] + shared[1] + current[1]),
                    history_mass=hmass, current_mass=cmass, evidence_mass=hmass + cmass,
                    history_effective_mass=float(effective_history.sum()),
                    evidence_count=count, history_clusters=len(history_clusters),
                    current_clusters=len(current_clusters),
                    clusters=len(history_clusters | current_clusters),
                    evidence_missing=(count == 0 and shared.sum() == 0),
                    semantic_success_mass=float(shared[0]), semantic_failure_mass=float(shared[1]),
                    semantic_raw_success_mass=float(raw_shared[0]),
                    semantic_raw_failure_mass=float(raw_shared[1]),
                    **semantic_info, **calibration)


        a, b = info['posterior_alpha'], info['posterior_beta']
        info['epistemic_variance'] = a*b / ((a+b)**2 * (a+b+1.))
        info['aleatoric_variance'] = a*b / ((a+b) * (a+b+1.))
        if self.probability_calibrator is not None:
            info['semantic_raw_mass'] = float(raw_shared.sum())
            info['semantic_mass'] = float(shared.sum())
            info['epistemic_variance'] = float(samples.var())
            info['aleatoric_variance'] = float(np.mean(samples*(1-samples)))
            info['calibration_model'] = 'conditional_beta_mixture'
        self._predictions[key] = samples, info
        return samples, info

    def success_particles(self, task, context, depth, horizon=None, *, state_key=None):
        depth = int(depth)
        horizon = self.max_steps - depth if horizon is None else int(horizon)
        if not 0 <= depth < self.max_steps or horizon != self.max_steps - depth:
            raise ValueError('State priors must use their original finite deadline')
        state = state_key if state_key is not None else str(context).rsplit(':depth=', 1)[0]
        return self._predict(task, state, depth)[0].copy()

    def state_diagnostics(self, task, state, depth):
        return dict(self._predict(task, state, depth)[1])

    def diagnostics(self):
        calibration = (self.probability_calibrator.diagnostics() if self.probability_calibrator else
                       self.calibrator.diagnostics() if self.calibrator else {})
        return dict(states=len(self.memory.pools), clusters=len(self.memory.clusters),
                    buckets=sum(len(rows) for p in self.memory.pools.values()
                                for rows in p['buckets'].values()),
                    particles=self.particles,
                    prior_strength=(calibration.get('prior_strength', calibration.get('prior_strength_mean', 1.))
                                    if calibration.get('queried_tasks') else 1.),
                    semantic_enabled=self.semantic_prior is not None,
                    migrated_paths=self.memory.migration.get('migrated_paths', 0),
                    dropped_origin_contexts=self.memory.migration.get('dropped_origin_contexts', 0),
                    **(self.semantic_prior.diagnostics() if self.semantic_prior is not None else {}),
                    calibration=calibration,
                    inference=('conditional_calibrated_beta_mixture' if self.probability_calibrator else
                               'conditional_empirical_bayes_gamma_dirichlet'
                               if self.empirical_bayes else 'aggregate_state_prior_cluster_bootstrap'))
