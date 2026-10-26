"""EL-TSBPO dps survival implementation."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def terminal_kind(success, *, truncated=False, done=True, exhausted=False):
    if success:
        return 'success'
    if truncated or exhausted or not done:
        return 'timeout'
    return 'failure'


def continuation_records(rows, max_steps, prefix_length=0, *, branch=False):
    """Extract audited physical continuation paths."""
    from .agent_dps import state_key
    grouped = defaultdict(list)
    invalid = set()
    for row in rows:
        traj = str(row.get('traj_uid', '') or '')
        if not traj or not bool(row.get('active_masks', True)):
            continue
        if bool(row.get('el_tsbpo_quarantined', False)):
            invalid.add(traj)
        grouped[traj].append(row)
    records = []
    for traj, values in grouped.items():
        if traj in invalid:
            continue

        values = sorted(values, key=lambda r: int(r.get('turn_idx', 0)))
        depths = [int(r.get('turn_idx', 0)) for r in values]
        if depths != list(range(depths[0], depths[0] + len(depths))):
            continue
        first, last = values[0], values[-1]
        depth = depths[0]
        if not branch and depth != 0:
            continue
        if not 0 <= depth < max_steps or depths[-1] >= max_steps:
            continue
        task = str(first.get('uid', '') or '')
        if not task or not first.get('anchor_obs'):
            continue
        success = float(last.get('trajectory_success', 0) or 0) >= .5
        reason = str(last.get('el_tsbpo_terminal_reason', '') or '')
        done = bool(last.get('dps_env_done', reason in {'success', 'failure'}))
        truncated = bool(last.get('dps_env_truncated', False)) or reason == 'timeout'
        exhausted = depths[-1] + 1 == max_steps
        terminal_observed = done or truncated or exhausted
        kind = terminal_kind(success, truncated=truncated, done=done)
        key = state_key(task, first['anchor_obs'])
        records.append(dict(
            id=traj, task=task, depth=depth, length=len(values), kind=kind,
            terminal_observed=terminal_observed,
            cluster=str(first.get('el_tsbpo_bare_rollout_id')
                        or first.get('parent_traj_uid') or traj),
            suffixes=[dict(state=state_key(task, value['anchor_obs']), depth=turn)
                      for value, turn in zip(values, depths) if value.get('anchor_obs')],
        ))
    return records


def task_union_utility(q, counts):
    q = np.asarray(q, dtype=float)
    counts = np.asarray(counts, dtype=int)
    return float(1 - np.prod((1 - q) ** counts[:, None], axis=0).mean())


def _task_frontier(nodes, budget, opening, max_k, beam):
    particles = len(nodes[0].opposite_particles)
    remaining = np.full((budget + 1, beam, particles), np.inf)
    remaining[0, 0] = 1.
    traces = []
    for node in nodes:
        limit = min(int(max_k), budget)
        options = [0] + list(range(opening, limit + 1))
        candidates = np.full((len(options), budget + 1, beam, particles), np.inf)
        for i, count in enumerate(options):
            stop = budget + 1 - count
            np.multiply(remaining[:stop], (1 - np.asarray(node.opposite_particles)) ** count,
                        out=candidates[i, count:], where=np.isfinite(remaining[:stop]))
        candidates = candidates.transpose(1, 0, 2, 3).reshape(budget + 1, -1, particles)
        scores = candidates.mean(-1)
        chosen = np.argsort(scores, axis=1, kind='stable')[:, :beam]
        remaining = np.take_along_axis(candidates, chosen[:, :, None], axis=1)
        count = np.asarray(options)[chosen // beam]
        previous_beam = chosen % beam
        traces.append((count, previous_beam))
    utilities = 1 - remaining[:, 0].mean(-1)
    return utilities, traces


def allocate_task_union(nodes, budget, *, opening=4, max_k=8, beam=4):
    """Allocate branch samples to maximize expected mixed task coverage."""
    from .agent_dps import AgentDPSAllocation
    budget, opening, max_k, beam = int(budget), max(4, int(opening)), int(max_k), max(1, int(beam))
    by_task = defaultdict(list)
    for node in nodes:
        if node.opposite_particles:
            by_task[str(node.task_uid)].append(node)
    tables = []
    for task in sorted(by_task):
        ordered = sorted(by_task[task], key=lambda n: (-n.tie_breaker, n.state_key))
        utilities, trace = _task_frontier(ordered, budget, opening, max_k, beam)
        tables.append((task, ordered, utilities, trace))
    dp = np.full(budget + 1, -np.inf)
    dp[0] = 0.
    outer = []
    for _, _, utilities, _ in tables:
        next_dp = np.full_like(dp, -np.inf)
        choices = np.zeros(budget + 1, dtype=int)
        for count, utility in enumerate(utilities):
            if not np.isfinite(utility):
                continue
            scores = dp[:budget + 1 - count] + utility
            target = next_dp[count:]
            better = scores > target + 1e-12
            target[better] = scores[better]
            choices[count:][better] = count
        dp = next_dp
        outer.append(choices)
    used = max(range(budget + 1), key=lambda k: (round(float(dp[k]), 12), -k))
    cursor, plans = used, []
    for i in range(len(tables) - 1, -1, -1):
        task, ordered, utilities, trace = tables[i]
        assigned = int(outer[i][cursor])
        cursor -= assigned
        b, slot, counts = assigned, 0, [0] * len(ordered)
        for j in range(len(ordered) - 1, -1, -1):
            k = int(trace[j][0][b, slot])
            slot = int(trace[j][1][b, slot])
            counts[j], b = k, b - k
        if b != 0:
            raise RuntimeError('Broken task union allocation trace')
        if assigned:
            plans.append((task, ordered, counts, float(utilities[assigned])))
    allocations, details = [], []
    for task, ordered, counts, utility in sorted(plans, key=lambda x: x[0]):
        q = np.asarray([n.opposite_particles for n in ordered])
        recomputed = task_union_utility(q, counts)
        if abs(recomputed - utility) > 1e-9:
            raise RuntimeError('Task utility does not match selected plan')
        details.append(dict(task_ref=task, allocated_budget=sum(counts), predicted_activation=utility,
                            selected_states=sum(k > 0 for k in counts)))
        for index, (node, k) in enumerate(zip(ordered, counts)):
            if not k:
                continue
            node.current_k, node.opened = k, True
            local = task_union_utility(q[index:index + 1], [k])
            without = list(counts)
            without[index] = 0
            marginal = max(0., utility - task_union_utility(q, without))
            allocations.append(AgentDPSAllocation(
                anchor_id=node.anchor_id, trajectory_id=node.trajectory_id,
                continuation_count=k, predicted_mean=node.probability,
                predicted_uncertainty=math.sqrt(node.uncertainty), external_cost=float(k),
                score=marginal, cumulative_utility=local, marginal_utility=marginal,
                marginal_cost=float(k), marginal_efficiency=marginal / k,
                marginal_score=marginal, marginal_score_efficiency=marginal / k,
                state_group_key=node.state_key, factual_count=len(node.factual_outcomes),
                physical_factual_count=max(1, node.existing_factual_count)))
    prefix = 'agent_dps/allocator/'
    metrics = {prefix + k: float(v) for k, v in dict(
        budget=budget, used_budget=used, unspent_budget=budget-used,
        budget_utilization=used/max(1, budget), n_states=len(nodes),
        n_selected_states=len(allocations), task_union_enabled=1,
        predicted_new_mixed_tasks=sum(d['predicted_activation'] for d in details),
        task_covered_count=len(details), posterior_optimal_knapsack=0,
        task_union_beam_width=beam, atomic_opening_size=opening,
        min_k=min((a.continuation_count for a in allocations), default=0),
        max_k=max((a.continuation_count for a in allocations), default=0),
        mean_k=used/max(1,len(allocations)),
        mean_marginal_gain=sum(a.marginal_utility for a in allocations)/max(1,len(allocations)),
        predicted_created_contrast_sum=sum(a.cumulative_utility for a in allocations),
        predicted_created_contrast_rate=sum(a.cumulative_utility for a in allocations)/max(1,len(allocations)),
        bayesian_terminal_objective_enabled=1, prefilter_keep_fraction=1,
        acquisition_prefilter_enabled=0).items()}
    return allocations, metrics, details


def allocate_local_state_contrast(nodes, budget, *, opening=4, max_k=8,
                                  already_mixed_tasks=()):
    """Allocate globally for new contrast at each factual state group."""
    from .agent_dps import AgentDPSAllocation

    budget = max(0, int(budget))
    opening = max(4, int(opening))
    max_k = min(budget, int(max_k))
    mixed_tasks = {str(task) for task in already_mixed_tasks}
    eligible = []
    for node in sorted(nodes, key=lambda item: (-item.tie_breaker, item.state_key)):
        factual_labels = {float(value) >= .5 for value in node.factual_outcomes}
        if len(factual_labels) != 1 or not node.opposite_particles:
            continue
        opposite = np.clip(np.asarray(node.opposite_particles, dtype=np.float64), 0., 1.)
        if not np.isfinite(opposite).all():
            raise ValueError('Nonfinite local state contrast probability')
        if not np.any(opposite > 0.):
            continue
        k_values = np.arange(max_k + 1)
        gains = (1. - (1. - opposite[:, None]) ** k_values).mean(axis=0)
        eligible.append((node, opposite, gains))

    dp = np.full(budget + 1, -np.inf)
    dp[0] = 0.
    choices = []
    for _, _, gains in eligible:
        updated = dp.copy()
        chosen = np.zeros(budget + 1, dtype=np.int32)
        for count in range(opening, max_k + 1):
            scores = dp[:-count] + gains[count]
            better = scores > updated[count:] + 1e-12
            updated[count:][better] = scores[better]
            chosen[count:][better] = count
        dp = updated
        choices.append(chosen)

    used = max(range(budget + 1), key=lambda count: (round(float(dp[count]), 12), -count))
    cursor = used
    counts = [0] * len(eligible)
    for index in range(len(eligible) - 1, -1, -1):
        count = int(choices[index][cursor])
        counts[index] = count
        cursor -= count
    if cursor != 0:
        raise RuntimeError('Broken local state allocation trace')

    allocations = []
    selected_by_task = defaultdict(list)
    for (node, opposite, gains), count in zip(eligible, counts):
        if count == 0:
            continue
        utility = float(gains[count])
        node.current_k, node.opened = count, True
        selected_by_task[str(node.task_uid)].append((node, opposite, count, utility))
        allocations.append(AgentDPSAllocation(
            anchor_id=node.anchor_id, trajectory_id=node.trajectory_id,
            continuation_count=count, predicted_mean=node.probability,
            predicted_uncertainty=math.sqrt(node.uncertainty),
            external_cost=float(count), score=utility,
            cumulative_utility=utility, marginal_utility=utility,
            marginal_cost=float(count), marginal_efficiency=utility / count,
            marginal_score=utility, marginal_score_efficiency=utility / count,
            state_group_key=node.state_key, factual_count=len(node.factual_outcomes),
            physical_factual_count=max(1, node.existing_factual_count)))

    plans = []
    for task, selected in sorted(selected_by_task.items()):
        predicted_activation = 0.
        if task not in mixed_tasks:
            predicted_activation = task_union_utility(
                np.asarray([opposite for _, opposite, _, _ in selected]),
                [count for _, _, count, _ in selected])
        plans.append(dict(
            task_ref=task, allocated_budget=sum(count for _, _, count, _ in selected),
            predicted_activation=predicted_activation,
            predicted_new_mixed_states=sum(gain for _, _, _, gain in selected),
            already_mixed_bare=task in mixed_tasks,
            selected_states=len(selected)))

    prefix = 'agent_dps/allocator/'
    metrics = {prefix + key: float(value) for key, value in dict(
        budget=budget, used_budget=used, unspent_budget=budget - used,
        budget_utilization=used / max(1, budget), n_states=len(nodes),
        n_eligible_states=len(eligible), n_selected_states=len(allocations),
        task_union_enabled=1, local_state_contrast_enabled=1,
        predicted_new_mixed_states=sum(a.cumulative_utility for a in allocations),
        predicted_new_mixed_tasks=sum(plan['predicted_activation'] for plan in plans),
        task_covered_count=len(plans), posterior_optimal_knapsack=1,
        task_union_beam_width=0, atomic_opening_size=opening,
        min_k=min((a.continuation_count for a in allocations), default=0),
        max_k=max((a.continuation_count for a in allocations), default=0),
        mean_k=used / max(1, len(allocations)),
        mean_marginal_gain=sum(a.marginal_utility for a in allocations) / max(1, len(allocations)),
        predicted_created_contrast_sum=sum(a.cumulative_utility for a in allocations),
        predicted_created_contrast_rate=sum(a.cumulative_utility for a in allocations) / max(1, len(allocations)),
        bayesian_terminal_objective_enabled=1, prefilter_keep_fraction=1,
        acquisition_prefilter_enabled=0).items()}
    return allocations, metrics, plans
