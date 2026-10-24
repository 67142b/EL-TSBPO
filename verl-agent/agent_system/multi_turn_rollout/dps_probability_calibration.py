"""EL-TSBPO dps probability calibration implementation."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math

import numpy as np
from scipy.special import betaln, logsumexp


def audited_branch_groups(selected, rows, max_steps):
    from .dps_survival import continuation_records
    counts = defaultdict(dict)
    for record in continuation_records(rows, max_steps, 0, branch=True):
        if record['terminal_observed'] and record['suffixes']:
            first = record['suffixes'][0]
            key = (record['task'], first['state'], int(first['depth']))
            counts[key][record['id']] = record['kind'] == 'success'
    result, skipped = [], 0
    for record in selected:
        d = record['state_prior']
        key = (str(record['task_ref']), str(record['state_group_key']), int(d['depth']))
        values = counts.get(key, {})
        if len(values) != int(record['allocated_k']) or not values:
            skipped += 1
            continue
        success = sum(values.values())
        result.append(dict(task=key[0], group=f'{key[1]}:depth={key[2]}',
            history_success=d['history_success_mass'],
            history_failure=d['history_failure_mass'],
            current_success=d['current_success_mass'],
            current_failure=d['current_failure_mass'],
            direct_success=d['history_success_mass']+d['current_success_mass'],
            direct_failure=d['history_failure_mass']+d['current_failure_mass'],
            semantic_success=d['semantic_raw_success_mass'],
            semantic_failure=d['semantic_raw_failure_mass'],
            success=success, failure=len(values)-success))
    return result, skipped


def split_evidence(record, stage):
    keys = ('history_success', 'history_failure', 'current_success', 'current_failure')
    if all(key in record for key in keys):
        values = tuple(float(record[key]) for key in keys)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError('Invalid history/current calibration evidence')
        for name, value in zip(('direct_success', 'direct_failure'),
                               (values[0]+values[2], values[1]+values[3])):
            if name in record and not math.isclose(float(record[name]), value,
                                                  rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError('Combined direct evidence disagrees with history/current split')
        return values
    if any(key in record for key in keys):
        raise ValueError('Incomplete history/current calibration evidence')
    if stage == 'root':
        return float(record['direct_success']), float(record['direct_failure']), 0., 0.
    raise ValueError('Legacy branch calibration needs separate history/current evidence')


def observe_groups(archive, stage, records, step, *, max_entries=10000):
    if stage not in {'root', 'branch'}:
        raise ValueError('Unknown probability calibration stage')
    groups = archive.setdefault(stage, {})
    added = 0
    for raw in records:
        hs, hf, cs, cf = split_evidence(raw, stage)
        row = dict(step=int(step), task=str(raw['task']), group=str(raw['group']),
                   history_success=hs, history_failure=hf,
                   current_success=cs, current_failure=cf,
                   direct_success=hs+cs, direct_failure=hf+cf)
        if stage == 'root' and (cs or cf):
            raise ValueError('Root allocation cannot condition on current rollout outcomes')
        for key in ('semantic_success',
                    'semantic_failure', 'success', 'failure'):
            row[key] = float(raw[key])
        for value in (row[key] for key in ('history_success', 'history_failure',
                'current_success', 'current_failure', 'semantic_success',
                'semantic_failure', 'success', 'failure')):
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid probability calibration counts')
        if row['success'] + row['failure'] <= 0:
            continue
        if any(row[k] != int(row[k]) for k in ('success', 'failure')):
            raise ValueError('Observed terminal counts must be physical integers')
        key = f"{step}:{row['task']}:{row['group']}"
        if key in groups:
            if groups[key] != row:
                raise ValueError('An audited calibration group changed after commitment')
            continue
        groups[key] = row
        added += 1
    keep = max(1, min(10000, int(max_entries)))
    for key in sorted(groups, key=lambda k: (groups[k]['step'], k))[:-keep]:
        del groups[key]
    return added


class BetaMixtureForecast:

    def __init__(self, alpha, beta, weights, info):
        self.alpha, self.beta = np.asarray(alpha), np.asarray(beta)
        self.weights, self.info = np.asarray(weights), dict(info)
        self.mean = float(self.weights @ (self.alpha / (self.alpha + self.beta)))
        second = self.weights @ (self.alpha * (self.alpha + 1)
                                  / ((self.alpha + self.beta) * (self.alpha + self.beta + 1)))
        self.variance = max(0., float(second - self.mean**2))

    def mixed_curve(self, max_count):

        success = failure = np.ones_like(self.alpha)
        curve = [0.]
        for k in range(int(max_count)):
            success = success * (self.alpha + k) / (self.alpha + self.beta + k)
            failure = failure * (self.beta + k) / (self.alpha + self.beta + k)
            curve.append(float(np.clip(1-self.weights @ (success+failure), 0., 1.)))
        return tuple(curve)

    def contrast_probability(self, count, factual_success):
        same = (betaln(self.alpha+int(count), self.beta) if factual_success else
                betaln(self.alpha, self.beta+int(count))) - betaln(self.alpha, self.beta)
        return float(np.clip(-self.weights @ np.expm1(same), 0., 1.))

    def count_log_probability(self, success, failure):
        n = int(success+failure)
        combinatorial = math.lgamma(n+1)-math.lgamma(success+1)-math.lgamma(failure+1)
        return float(combinatorial + logsumexp(np.log(self.weights)
            + betaln(self.alpha+success, self.beta+failure)-betaln(self.alpha, self.beta)))


class ProbabilityCalibrator:
    """Calibrate root and branch probabilities from past outcomes."""

    def __init__(self, archive, step, stage, decay=.9, *,
                 history_grid=(0., .03125, .125, .5, 1.)):
        if stage not in {'root', 'branch'} or not 0 < float(decay) <= 1:
            raise ValueError('Invalid probability calibration model')
        self.step, self.stage, self.decay = int(step), stage, float(decay)
        coefficients = np.array([0., .03125, .125, .5, 1.])
        history_grid = np.asarray(history_grid, dtype=float)
        if (history_grid.ndim != 1 or not np.all(np.isfinite(history_grid))
                or not np.all((history_grid >= 0) & (history_grid <= 1))
                or 1. not in history_grid or len(np.unique(history_grid)) != len(history_grid)):
            raise ValueError('History grid must contain 1 and unique coefficients in [0,1]')
        mean, strength, history, rho = np.meshgrid(np.linspace(.025, .975, 25),
            np.geomspace(.001, 100., 31), history_grid, coefficients, indexing='ij')
        self.alpha = (mean*strength).ravel()
        self.beta = ((1-mean)*strength).ravel()
        self.history = history.ravel()
        self.rho = rho.ravel()
        self.legacy_groups_skipped = sum(
            int(r['step']) < self.step and 'history_success' not in r
            for r in archive.get('branch', {}).values()) if stage == 'branch' else 0
        self.scores = self._scores((r for r in archive.get(stage, {}).values()
                                   if int(r['step']) < self.step
                                   and (stage == 'root' or 'history_success' in r)), stage)
        self.warm = (self._scores(r for r in archive.get('root', {}).values()
                                 if int(r['step']) <= self.step)
                     if stage == 'branch' else None)
        self.cache = {}

    def _scores(self, source, stage='root'):
        rows = [r for r in source if self.decay**(self.step-int(r['step'])) >= 1e-8]
        multiplicity = Counter((r['task'], int(r['step'])) for r in rows)
        by_task, masses, counts, semantic, historical = {}, defaultdict(float), Counter(), Counter(), Counter()
        total = np.zeros_like(self.alpha)
        for r in rows:
            task, age = r['task'], self.step-int(r['step'])
            hs, hf, cs, cf = split_evidence(r, stage)
            a = self.alpha + self.history*hs + cs + self.rho*r['semantic_success']
            b = self.beta + self.history*hf + cf + self.rho*r['semantic_failure']
            weight = self.decay**age / multiplicity[task, int(r['step'])]
            score = weight * (betaln(a+r['success'], b+r['failure'])-betaln(a, b))
            total += score
            by_task.setdefault(task, np.zeros_like(score))[:] += score
            masses[task] += weight
            counts[task] += 1
            semantic[task] += int(r['semantic_success']+r['semantic_failure'] > 0)
            historical[task] += int(hs+hf > 0)
        return total, by_task, masses, counts, semantic, historical

    def parameters(self, task):
        task = str(task)
        if task in self.cache:
            return self.cache[task]
        data, source = self.scores, self.stage
        if len(set(data[1])-{task}) < 2 and self.warm is not None:
            data, source = self.warm, 'bare_warm_start'
        total, by_task, masses, counts, semantic, historical = data
        others = set(by_task)-{task}
        if len(others) < 2:
            result = (np.array([.5]), np.array([.5]), np.ones(1), np.array([0.]), np.ones(1),
                      dict(calibrated=False, source='cold', groups=0, tasks=len(others),
                           effective_task_rounds=0., semantic_identified=False,
                           history_identified=False))
        else:
            scores = total - by_task.get(task, 0.)
            identified = source == self.stage and sum(semantic[t] > 0 for t in others) >= 2
            history_identified = source == self.stage and sum(historical[t] > 0 for t in others) >= 2
            if not history_identified:
                scores = np.where(self.history == 1., scores, -np.inf)
            if not identified:

                scores = np.where(self.rho == 0., scores, -np.inf)
            weights = np.exp(scores-logsumexp(scores))

            active = weights > 0.
            result = (self.alpha[active], self.beta[active], self.history[active],
                      self.rho[active], weights[active],
                      dict(calibrated=True, source=source, groups=sum(counts[t] for t in others),
                           tasks=len(others), effective_task_rounds=sum(masses[t] for t in others),
                           semantic_identified=identified, history_identified=history_identified))
        self.cache[task] = result
        return result

    def forecast(self, task, history=(0., 0.), semantic=(0., 0.), *, current=(0., 0.)):
        if any(not math.isfinite(value) or value < 0
               for values in (history, semantic, current) for value in values):
            raise ValueError('Invalid probability forecast evidence')
        if self.stage == 'root' and any(current):
            raise ValueError('Root forecast cannot use current rollout evidence')
        a, b, eta, rho, weights, info = self.parameters(task)
        return BetaMixtureForecast(a+eta*history[0]+current[0]+rho*semantic[0],
                                  b+eta*history[1]+current[1]+rho*semantic[1], weights,
                                  {**info, **self._summary(a, b, eta, rho, weights)})

    def population_particles(self, task, count, seed):
        a, b, eta, rho, weights, info = self.parameters(task)
        digest = hashlib.sha256(f'{seed}:calibration:{task}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'big'))

        u = (np.arange(count)+rng.random(count))/count
        indices = np.searchsorted(np.cumsum(weights), np.minimum(u, 1-1e-15))
        rng.shuffle(indices)
        return (a[indices], b[indices], eta[indices], rho[indices],
                {**info, **self._summary(a, b, eta, rho, weights)})

    @staticmethod
    def _summary(a, b, eta, rho, weights):
        return dict(history_reliability=float(weights @ eta),
                    history_disabled_probability=float(weights @ (eta == 0.)),
                    semantic_reliability=float(weights @ rho),
                    semantic_disabled_probability=float(weights @ (rho == 0.)),
                    prior_strength=float(weights @ (a+b)),
                    prior_success=float(weights @ (a/(a+b))),
                    parameter_ess=float(1/(weights @ weights)))

    def diagnostics(self):
        answers = [self._summary(*v[:5]) for v in self.cache.values()]
        return dict(enabled=1., groups=sum(self.scores[3].values()),
                    tasks=len(self.scores[1]), effective_task_rounds=sum(self.scores[2].values()),
                    legacy_groups_skipped=float(self.legacy_groups_skipped),
                    queried_tasks=len(answers), calibrated_fraction=float(np.mean(
                        [v[5]['calibrated'] for v in self.cache.values()])) if answers else 0.,
                    **{key: float(np.mean([r[key] for r in answers])) if answers else 0.
                       for key in ('history_reliability', 'history_disabled_probability',
                                   'semantic_reliability', 'semantic_disabled_probability',
                                   'prior_strength', 'prior_success', 'parameter_ess')})
