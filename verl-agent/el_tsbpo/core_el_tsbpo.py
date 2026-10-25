"""Core EL-TSBPO credit assignment."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch


def _key(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return None if text in {"", "None", "nan"} else text


def _terminal_rewards(
    traj_index: Sequence,
    trajectory_success: Sequence | None,
    token_level_rewards: torch.Tensor,
) -> dict[str, float]:

    per_turn = token_level_rewards.detach().float().sum(dim=-1).cpu().numpy()
    result: dict[str, float] = {}
    for row, trajectory in enumerate(traj_index):
        tid = str(trajectory)
        if trajectory_success is not None and row < len(trajectory_success):
            value = float(trajectory_success[row])
        else:
            value = float(per_turn[row])
        result[tid] = max(result.get(tid, value), value)
    return result


def _zscore(values: Sequence[float], epsilon: float) -> list[float]:
    if len(values) < 2:
        return [0.0 for _ in values]
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=0))
    if not np.isfinite(std) or std <= epsilon:
        return [0.0 for _ in values]
    mean = float(array.mean())
    return [float((value - mean) / std) for value in values]


def _invalid_penalty_tensor(
    penalty: torch.Tensor | None,
    *,
    bsz: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if penalty is None:
        return torch.zeros(bsz, device=device, dtype=dtype)
    value = penalty.to(device=device, dtype=dtype).reshape(-1)
    if value.shape[0] != bsz:
        raise ValueError(
            "invalid_action_penalty must contain one value per rollout row"
        )
    return value.clamp_min(0.0)


def compute_tree_backups(
    *,
    parent_node_id: Sequence,
    node_id: Sequence,
    traj_index: Sequence,
    terminal_reward: Sequence | None = None,
    trajectory_success: Sequence | None = None,
    node_is_terminal: Sequence | None = None,
    active_weight: Sequence | torch.Tensor | None = None,
    edge_penalty: Sequence | torch.Tensor | None = None,
    gamma: float = 1.0,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Back up terminal rewards through the physical sampled tree."""

    try:
        gamma = float(gamma)
    except (TypeError, ValueError):
        gamma = 1.0
    if not np.isfinite(gamma):
        gamma = 1.0
    gamma = min(1.0, max(0.0, gamma))

    n_rows = len(node_id)
    if len(parent_node_id) != n_rows or len(traj_index) != n_rows:
        raise ValueError("tree metadata must have one entry per rollout row")
    if terminal_reward is not None and len(terminal_reward) != n_rows:
        raise ValueError("terminal_reward must have one entry per rollout row")
    if node_is_terminal is not None and len(node_is_terminal) != n_rows:
        raise ValueError("node_is_terminal must have one entry per rollout row")
    if active_weight is not None and len(active_weight) != n_rows:
        raise ValueError("active_weight must have one entry per rollout row")
    if edge_penalty is not None and len(edge_penalty) != n_rows:
        raise ValueError("edge_penalty must have one entry per rollout row")

    if active_weight is None:
        active = [True] * n_rows
    else:
        active = [
            float(
                active_weight[row].detach().item()
                if isinstance(active_weight, torch.Tensor)
                else active_weight[row]
            )
            > 0.0
            for row in range(n_rows)
        ]

    if edge_penalty is None:
        row_penalties = np.zeros(n_rows, dtype=np.float64)
    elif isinstance(edge_penalty, torch.Tensor):
        row_penalties = (
            edge_penalty.detach().float().cpu().numpy().reshape(-1).astype(np.float64)
        )
    else:
        row_penalties = np.asarray(edge_penalty, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(row_penalties)) or np.any(row_penalties < 0.0):
        raise ValueError("edge_penalty must contain finite non-negative values")

    explicit_terminal_rows = (
        terminal_reward is not None and node_is_terminal is not None
    )
    rewards_by_traj: dict[str, float] = {}
    terminal_trajectory_ids: set[str] = set()
    if explicit_terminal_rows:


        for row, trajectory in enumerate(traj_index):
            if not active[row] or row >= len(node_is_terminal):
                continue
            if not bool(node_is_terminal[row]):
                continue
            tid = str(trajectory)
            value = float(terminal_reward[row])
            rewards_by_traj[tid] = max(rewards_by_traj.get(tid, value), value)
            terminal_trajectory_ids.add(tid)


        for row, trajectory in enumerate(traj_index):
            if not active[row]:
                continue
            tid = str(trajectory)
            if tid in terminal_trajectory_ids:
                continue
            fallback = (
                float(trajectory_success[row])
                if trajectory_success is not None and row < len(trajectory_success)
                else float(terminal_reward[row])
            )
            rewards_by_traj[tid] = max(rewards_by_traj.get(tid, fallback), fallback)
    elif terminal_reward is not None:


        for row, trajectory in enumerate(traj_index):
            if not active[row]:
                continue
            tid = str(trajectory)
            value = float(terminal_reward[row])
            rewards_by_traj[tid] = max(rewards_by_traj.get(tid, value), value)
    else:
        rewards_by_traj = _terminal_rewards(
            traj_index,
            trajectory_success,
            torch.zeros((n_rows, 1), dtype=torch.float32),
        )

    children: dict[str, list[str]] = defaultdict(list)
    edge_rows: dict[tuple[str, str], list[int]] = defaultdict(list)
    known_nodes: set[str] = set()
    terminal_flags: dict[str, bool] = {}
    node_rewards: dict[str, float] = {}
    parent_by_child: dict[str, str] = {}
    penalty_by_edge: dict[tuple[str, str], float] = {}
    for row in range(n_rows):
        if not active[row]:
            continue
        parent = _key(parent_node_id[row])
        child = _key(node_id[row])
        if parent is None or child is None:
            continue
        previous_parent = parent_by_child.get(child)
        if previous_parent is not None and previous_parent != parent:
            raise ValueError(
                f"EL-TSBPO node {child!r} has multiple parents: "
                f"{previous_parent!r} and {parent!r}"
            )
        parent_by_child[child] = parent
        if (parent, child) in edge_rows:
            raise ValueError(
                f"duplicate active EL-TSBPO edge occurrence: {parent!r} -> {child!r}"
            )
        known_nodes.update((parent, child))
        if child not in children[parent]:
            children[parent].append(child)
        edge_rows[(parent, child)].append(row)
        penalty_by_edge[(parent, child)] = float(row_penalties[row])
        if node_is_terminal is not None and row < len(node_is_terminal):
            terminal_flags[child] = terminal_flags.get(child, False) or bool(
                node_is_terminal[row]
            )
        if (
            explicit_terminal_rows
            and row < len(node_is_terminal)
            and bool(node_is_terminal[row])
        ):
            row_reward = float(terminal_reward[row])
        else:
            row_reward = rewards_by_traj.get(str(traj_index[row]), 0.0)
        node_rewards[child] = max(
            node_rewards.get(child, float("-inf")),
            row_reward,
        )

    for child in known_nodes:
        if child not in children or not children[child]:
            terminal_flags[child] = True
    for node, is_terminal in terminal_flags.items():
        if is_terminal and children.get(node):
            raise ValueError(f"terminal EL-TSBPO node {node!r} also has children")

    discounted_tree_reward: dict[str, float] = {}
    discounted_edge_reward: dict[tuple[str, str], float] = {}
    descendant_count: dict[str, int] = {}
    pred_value: dict[str, float] = {}
    visiting: set[str] = set()

    def visit(node: str) -> tuple[float, int, float]:
        if node in discounted_tree_reward:
            return (
                discounted_tree_reward[node],
                descendant_count[node],
                pred_value[node],
            )
        if node in visiting:
            raise ValueError(f"cycle detected in EL-TSBPO tree at node {node!r}")
        visiting.add(node)
        child_ids = [child for child in children.get(node, []) if child in known_nodes]
        if not child_ids or terminal_flags.get(node, False):
            reward = float(node_rewards.get(node, 0.0))
            count = 1
            value = reward
        else:
            child_values = [visit(child) for child in child_ids]


            child_outcomes = []
            for child, item in zip(child_ids, child_values):
                outcome = (
                    item[0] if terminal_flags.get(child, False) else gamma * item[0]
                )
                discounted_edge_reward[(node, child)] = float(
                    outcome - penalty_by_edge.get((node, child), 0.0)
                )
                child_outcomes.append(outcome)
            reward = float(np.mean(child_outcomes))
            count = int(sum(item[1] for item in child_values))
            value = float(
                sum(item[1] * item[2] for item in child_values) / max(1, count)
            )
        visiting.remove(node)
        discounted_tree_reward[node] = reward
        descendant_count[node] = count
        pred_value[node] = value
        return reward, count, value

    child_nodes = {child for values in children.values() for child in values}
    roots = [node for node in known_nodes if node not in child_nodes]
    for root in roots:
        visit(root)
    for node in known_nodes:
        if node not in discounted_tree_reward:
            visit(node)


    credit_reward_by_node: dict[str, float] = {}

    def visit_credit(node: str) -> float:
        if node in credit_reward_by_node:
            return credit_reward_by_node[node]
        child_ids = [child for child in children.get(node, []) if child in known_nodes]
        if not child_ids or terminal_flags.get(node, False):
            value = float(node_rewards.get(node, 0.0))
        else:
            value = float(np.mean([visit_credit(child) for child in child_ids]))
        credit_reward_by_node[node] = value
        return value

    for node in known_nodes:
        visit_credit(node)

    edge_tree_reward = np.zeros(n_rows, dtype=np.float32)
    edge_discounted_tree_reward = np.zeros(n_rows, dtype=np.float32)
    credit_reward_by_edge: dict[tuple[str, str], float] = {}
    for (parent, child), rows in edge_rows.items():
        credit_value = float(
            credit_reward_by_node.get(child, node_rewards.get(child, 0.0))
            - penalty_by_edge.get((parent, child), 0.0)
        )
        discounted_value = float(
            discounted_edge_reward.get(
                (parent, child),
                discounted_tree_reward.get(child, node_rewards.get(child, 0.0)),
            )
        )
        credit_reward_by_edge[(parent, child)] = credit_value
        for row in rows:
            edge_tree_reward[row] = credit_value
            edge_discounted_tree_reward[row] = discounted_value

    penalized_rewards_by_traj = dict(rewards_by_traj)
    for row, trajectory in enumerate(traj_index):
        if not active[row]:
            continue
        tid = str(trajectory)
        penalized_rewards_by_traj[tid] = float(
            penalized_rewards_by_traj.get(tid, 0.0)
        ) - float(row_penalties[row])

    edge_advantage = np.zeros(n_rows, dtype=np.float32)
    parent_advantages: dict[str, dict[str, float]] = {}
    effective_parents = 0
    unary_edge_rows = 0
    zero_variance_edge_rows = 0
    for parent, child_ids in children.items():
        values = [
            float(
                credit_reward_by_edge.get(
                    (parent, child), credit_reward_by_node.get(child, 0.0)
                )
            )
            for child in child_ids
        ]
        normalized = _zscore(values, epsilon)
        if any(abs(value) > epsilon for value in normalized):
            effective_parents += 1
        elif len(child_ids) >= 2:
            zero_variance_edge_rows += sum(
                len(edge_rows[(parent, child)]) for child in child_ids
            )
        if len(child_ids) < 2:
            unary_edge_rows += sum(
                len(edge_rows[(parent, child)]) for child in child_ids
            )
        parent_advantages[parent] = {
            child: advantage for child, advantage in zip(child_ids, normalized)
        }
        for child, advantage in parent_advantages[parent].items():
            for row in edge_rows[(parent, child)]:
                edge_advantage[row] = advantage

    return {
        "tree_reward": edge_tree_reward,
        "discounted_tree_reward": edge_discounted_tree_reward,
        "edge_advantage": edge_advantage,
        "tree_reward_by_node": credit_reward_by_node,
        "discounted_tree_reward_by_node": discounted_tree_reward,
        "edge_reward_by_edge": credit_reward_by_edge,
        "discounted_edge_reward_by_edge": discounted_edge_reward,
        "pred_value_by_node": pred_value,
        "descendant_count_by_node": descendant_count,
        "children": dict(children),
        "edge_rows": dict(edge_rows),
        "parent_advantages": parent_advantages,
        "roots": roots,
        "effective_parent_count": effective_parents,
        "zero_signal_parent_count": len(children) - effective_parents,
        "trajectory_reward_by_id": dict(rewards_by_traj),
        "penalized_trajectory_reward_by_id": penalized_rewards_by_traj,
        "gamma": float(gamma),


        "diagnostics": {
            "active_edge_rows": int(sum(len(rows) for rows in edge_rows.values())),
            "effective_edge_rows": int(
                np.count_nonzero(np.abs(edge_advantage) > epsilon)
            ),
            "zero_signal_edge_rows": int(
                sum(len(rows) for rows in edge_rows.values())
                - np.count_nonzero(np.abs(edge_advantage) > epsilon)
            ),
            "unary_edge_rows": int(unary_edge_rows),
            "zero_variance_edge_rows": int(zero_variance_edge_rows),
            "effective_parent_count": int(effective_parents),
            "zero_signal_parent_count": int(len(children) - effective_parents),
            "invalid_penalty_rows": int(np.count_nonzero(row_penalties > 0.0)),
            "invalid_penalty_total": float(row_penalties.sum()),
        },
    }


def _normalize_relative_values(
    values: Sequence[float],
    *,
    mode: str,
    epsilon: float,
) -> list[float]:

    if not values:
        return []
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)) or len(array) < 2:
        return [0.0 for _ in values]
    baseline = float(array.mean())
    centered = array - baseline
    normalized_mode = str(mode).lower()
    if normalized_mode == "mean_norm":
        scale = 1.0
    elif normalized_mode == "mean_std_norm":
        scale = float(array.std(ddof=1)) if len(array) > 1 else 0.0
        if not np.isfinite(scale) or scale <= float(epsilon):
            return [0.0 for _ in values]
        scale += float(epsilon)
    else:
        raise ValueError(f"Unknown relative normalization mode: {mode}")
    return [float(value / scale) for value in centered]


def _graph_gigpo_recursive_advantage(
    *,
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence,
    traj_index: Sequence,
    bare_rollout_id: Sequence | None,
    anchor_obs: Sequence,
    action_text: Sequence,
    parent_node_id: Sequence,
    node_id: Sequence,
    terminal_reward: Sequence | None,
    trajectory_success: Sequence | None,
    node_is_terminal: Sequence | None,
    sample_weight: torch.Tensor | None,
    episode_advantage_w: float | None,
    step_advantage_w: float,
    recursive_advantage_w: float | None,
    mode: str,
    enable_similarity: bool,
    similarity_thresh: float,
    epsilon: float,
    gamma: float,
    invalid_action_penalty: torch.Tensor | None = None,
    use_tree_discounted_step_return: bool = False,
    segment_balance: bool = False,
    policy_loss_agg_mode: str = "seq-mean-token-sum",
    segment_weight_cap: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Compute recursive physical-tree and state-group advantages."""

    from gigpo import core_gigpo

    bsz = token_level_rewards.shape[0]
    if any(
        len(values) != bsz
        for values in (
            index,
            traj_index,
            anchor_obs,
            action_text,
            parent_node_id,
            node_id,
        )
    ):
        raise ValueError("Graph-GiGPO recursive metadata must align with rollout rows")
    if bare_rollout_id is not None and len(bare_rollout_id) != bsz:
        raise ValueError("Graph-GiGPO bare_rollout_id must align with rollout rows")
    if step_rewards.ndim != 1:
        step_rewards = step_rewards.reshape(bsz, -1).sum(dim=-1)
    if step_rewards.shape[0] != bsz:
        raise ValueError("Graph-GiGPO step_rewards must contain one value per row")
    normalized_mode = str(mode).lower()
    if normalized_mode not in {"mean_norm", "mean_std_norm"}:
        raise ValueError(f"Unknown Graph-GiGPO normalization mode: {mode}")

    device = token_level_rewards.device
    dtype = token_level_rewards.dtype
    if sample_weight is None:
        rollout_weight = torch.ones(bsz, device=device, dtype=dtype)
    else:
        rollout_weight = sample_weight.to(device=device, dtype=dtype).reshape(-1)
        if rollout_weight.shape[0] != bsz:
            raise ValueError("sample_weight must contain one value per rollout row")
    active = rollout_weight > 0

    penalty = _invalid_penalty_tensor(
        invalid_action_penalty,
        bsz=bsz,
        device=step_rewards.device,
        dtype=step_rewards.dtype,
    )


    credit_token_rewards = token_level_rewards
    credit_step_rewards = step_rewards

    backup_terminal_reward = terminal_reward
    if backup_terminal_reward is None and trajectory_success is None:


        backup_terminal_reward = (
            (credit_step_rewards + penalty).detach().float().cpu().numpy()
        )
    backups = compute_tree_backups(
        parent_node_id=parent_node_id,
        node_id=node_id,
        traj_index=traj_index,
        terminal_reward=backup_terminal_reward,
        trajectory_success=trajectory_success,
        node_is_terminal=node_is_terminal,
        active_weight=rollout_weight,
        edge_penalty=penalty,
        gamma=gamma,
        epsilon=epsilon,
    )
    tree_q = np.asarray(backups["tree_reward"], dtype=np.float64)
    discounted_q = np.asarray(backups["discounted_tree_reward"], dtype=np.float64)
    penalty_by_row = penalty.detach().float().cpu().numpy().astype(np.float64)


    represented_rows = np.zeros(bsz, dtype=bool)
    for rows in backups.get("edge_rows", {}).values():
        represented_rows[np.asarray(rows, dtype=np.int64)] = True
    q_by_row = credit_step_rewards.detach().float().cpu().numpy().astype(np.float64)
    q_by_row[represented_rows] = tree_q[represented_rows]
    graph_value_by_row = (
        credit_step_rewards.detach().float().cpu().numpy().astype(np.float64)
    )
    if use_tree_discounted_step_return:
        graph_value_by_row[represented_rows] = discounted_q[represented_rows]

    subtree_scalar = np.zeros(bsz, dtype=np.float32)
    segment_return_by_row = np.zeros(bsz, dtype=np.float32)
    subtree_available = np.zeros(bsz, dtype=bool)


    subtree_segment_group: list[tuple[str, str] | None] = [None] * bsz
    subtree_segment_child: list[str | None] = [None] * bsz
    parent_children: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in (
        torch.nonzero(active, as_tuple=False).reshape(-1).detach().cpu().tolist()
    ):
        parent = _key(parent_node_id[row])
        child = _key(node_id[row])
        if parent is None or child is None or not represented_rows[row]:
            continue
        group = (str(index[row]), parent)
        parent_children[group][child].append(row)

    subtree_nonzero_groups = 0
    subtree_homogeneous_groups = 0
    subtree_group_count = 0
    physical_children = backups.get("children", {})
    edge_rows = backups.get("edge_rows", {})
    terminal_values = backups.get("pred_value_by_node", {})

    def continuation(parent: str, child: str) -> tuple[list[int], str]:
        rows: list[int] = []
        while True:
            rows.extend(edge_rows[(parent, child)])
            next_children = physical_children.get(child, [])
            if len(next_children) != 1:
                return rows, child
            parent, child = child, next_children[0]

    for group, children in parent_children.items():
        child_ids = list(children)
        if len(child_ids) < 2:
            continue
        subtree_group_count += 1
        continuations = [continuation(group[1], child) for child in child_ids]
        child_values = [
            float(np.mean(float(terminal_values[end_node]) - penalty_by_row[rows]))
            for rows, end_node in continuations
        ]
        normalized = _normalize_relative_values(
            child_values, mode=normalized_mode, epsilon=epsilon
        )
        if any(abs(value) > epsilon for value in normalized):
            subtree_nonzero_groups += 1
        else:
            subtree_homogeneous_groups += 1
        for child, (rows, _), raw_value, advantage in zip(
            child_ids, continuations, child_values, normalized
        ):
            for row in rows:
                subtree_scalar[row] = float(advantage)
                segment_return_by_row[row] = float(raw_value)
                subtree_available[row] = True
                subtree_segment_group[row] = group
                subtree_segment_child[row] = child


    graph_scalar = np.zeros(bsz, dtype=np.float32)
    graph_available = np.zeros(bsz, dtype=bool)


    graph_contrast_available = np.zeros(bsz, dtype=bool)
    graph_cross_parent_available = np.zeros(bsz, dtype=bool)
    state_count = 0
    multi_action_states = 0
    homogeneous_states = 0
    unique_edge_count = 0
    edge_sample_counts: list[int] = []
    state_rows: dict[str, list[int]] = defaultdict(list)
    active_rows = torch.nonzero(active, as_tuple=False).reshape(-1)
    if active_rows.numel() > 0:
        active_numpy = active_rows.detach().cpu().numpy()
        state_uids = core_gigpo.build_step_group(
            np.asarray(anchor_obs, dtype=object)[active_numpy],
            np.asarray(index, dtype=object)[active_numpy],
            enable_similarity=enable_similarity,
            similarity_thresh=similarity_thresh,
            summarize=False,
        )
        state_actions: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        action_by_row: dict[int, str] = {}
        for local_pos, row in enumerate(active_numpy.tolist()):
            raw_action = action_text[row]
            if raw_action is None:
                raise ValueError(
                    "Graph-GiGPO requires canonical action_text for every active row"
                )
            action = str(raw_action).strip().lower() or "<empty_action>"
            state_uid = str(state_uids[local_pos])
            state_rows[state_uid].append(row)
            state_actions[state_uid][action][str(traj_index[row])].append(
                float(graph_value_by_row[row])
            )
            action_by_row[row] = action
        state_count = len(state_rows)
        for state_uid, action_map in state_actions.items():
            unique_edge_count += len(action_map)


            if len(state_rows[state_uid]) < 2:
                continue
            if len(action_map) >= 2:
                multi_action_states += 1


            edge_sample_counts.extend(
                len(trajectory_map) for trajectory_map in action_map.values()
            )
            row_values = [
                float(graph_value_by_row[row]) for row in state_rows[state_uid]
            ]
            normalized = _normalize_relative_values(
                row_values, mode=normalized_mode, epsilon=epsilon
            )
            for position, row in enumerate(state_rows[state_uid]):
                graph_scalar[row] = float(normalized[position])
            is_homogeneous = all(abs(value) <= epsilon for value in normalized)
            if is_homogeneous:
                homogeneous_states += 1
            physical_parents = {
                _key(parent_node_id[row])
                for row in state_rows[state_uid]
                if _key(parent_node_id[row]) is not None
            }
            is_cross_parent = len(physical_parents) > 1
            for row in state_rows[state_uid]:
                graph_available[row] = True
                graph_contrast_available[row] = not is_homogeneous
                graph_cross_parent_available[row] = is_cross_parent


    fallback_mask = active.detach().cpu().numpy() & ~subtree_available
    fallback_rows = np.flatnonzero(fallback_mask)


    root_id_by_row: list[str] = []
    for row in range(bsz):
        trajectory = str(traj_index[row])
        if bare_rollout_id is None:
            root_id_by_row.append(trajectory)
            continue
        raw_root = bare_rollout_id[row]
        root = "" if raw_root is None else str(raw_root)
        root_id_by_row.append(
            trajectory if root in {"", "None", "null", "nan"} else root
        )

    root_turn_returns: dict[str, list[float]] = defaultdict(list)
    terminal_rewards_by_traj = backups.get("trajectory_reward_by_id", {})
    per_turn_episode_score = (
        credit_token_rewards.detach().float().sum(dim=-1).cpu().numpy()
    )
    for row in (
        torch.nonzero(active, as_tuple=False).reshape(-1).detach().cpu().tolist()
    ):
        root = root_id_by_row[row]
        if str(traj_index[row]) != root:
            continue
        turn_return = (
            float(per_turn_episode_score[row])
            if terminal_reward is None
            else float(terminal_rewards_by_traj.get(root, 0.0)) - penalty_by_row[row]
        )
        root_turn_returns[root].append(turn_return)
    episode_rewards_by_id = {
        root: float(np.mean(values)) for root, values in root_turn_returns.items()
    }

    task_roots: dict[str, list[str]] = defaultdict(list)
    seen_task_root: set[tuple[str, str]] = set()
    for row in (
        torch.nonzero(active, as_tuple=False).reshape(-1).detach().cpu().tolist()
    ):
        task = str(index[row])
        root = root_id_by_row[row]
        pair = (task, root)
        if pair in seen_task_root:
            continue
        seen_task_root.add(pair)
        task_roots[task].append(root)
    episode_by_task_root: dict[str, dict[str, float]] = {}
    for task, roots in task_roots.items():
        normalized = _normalize_relative_values(
            [float(episode_rewards_by_id.get(root, 0.0)) for root in roots],
            mode=normalized_mode,
            epsilon=epsilon,
        )
        episode_by_task_root[task] = dict(zip(roots, normalized))
    episode_scalar = np.zeros(bsz, dtype=np.float32)
    for row in (
        torch.nonzero(active, as_tuple=False).reshape(-1).detach().cpu().tolist()
    ):
        episode_scalar[row] = episode_by_task_root.get(str(index[row]), {}).get(
            root_id_by_row[row], 0.0
        )
    fallback_scalar = episode_scalar * fallback_mask.astype(np.float32)
    fallback_tokens = (
        torch.as_tensor(fallback_scalar, device=device, dtype=dtype)[:, None]
        * response_mask
    )


    subtree_signal_available = subtree_available & (np.abs(subtree_scalar) > epsilon)
    trajectory_scalar = np.where(
        subtree_available,
        subtree_scalar,
        fallback_scalar,
    )


    graph_weight = 1.0
    graph_raw_scalar = graph_scalar.copy()
    response_turn_lengths = response_mask.sum(dim=-1)
    policy_active = active & (response_turn_lengths > 0)


    segment_groups: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    policy_rows = (
        torch.nonzero(policy_active, as_tuple=False).reshape(-1).detach().cpu().tolist()
    )
    for row in policy_rows:
        segment_group = subtree_segment_group[row]
        segment_child = subtree_segment_child[row]
        if segment_group is None or segment_child is None:
            segment_group = (str(index[row]), "<gigpo-root-fallback>")
            segment_child = root_id_by_row[row]
            subtree_segment_group[row] = segment_group
            subtree_segment_child[row] = segment_child
        segment_groups[segment_group][segment_child].append(row)


    trajectory_policy_weight = torch.ones(bsz, device=device, dtype=dtype)
    segment_group_id = torch.full((bsz,), -1, device=device, dtype=torch.long)
    segment_id = torch.full((bsz,), -1, device=device, dtype=torch.long)
    segment_turn_count = torch.zeros(bsz, device=device, dtype=dtype)
    segment_child_count = torch.zeros(bsz, device=device, dtype=dtype)
    active_turn_count = int(policy_active.sum().item())
    group_count = len(segment_groups)
    segment_count = 0
    child_masses_by_group: list[list[float]] = []
    response_length_np = response_turn_lengths.detach().float().cpu().numpy()
    sample_weight_np = rollout_weight.detach().float().cpu().numpy()
    normalized_loss_mode = str(policy_loss_agg_mode).lower().replace("_", "-")


    token_measure = normalized_loss_mode == "token-mean"
    try:
        weight_cap = float(segment_weight_cap)
    except (TypeError, ValueError):
        weight_cap = 6.0
    if not np.isfinite(weight_cap) or weight_cap < 1.0:
        weight_cap = 6.0
    raw_child_masses_by_group: list[list[float]] = []
    balanced_child_masses_by_group: list[list[float]] = []
    cap_hit_children = 0
    balanced_children = 0
    for group_ordinal, children in enumerate(segment_groups.values()):
        child_count = len(children)
        group_child_masses: list[float] = []
        raw_group_masses: list[float] = []
        for child_rows in children.values():
            current_segment_id = segment_count
            segment_count += 1
            if token_measure:
                effective_turn_mass = float(
                    sum(
                        float(sample_weight_np[row]) * float(response_length_np[row])
                        for row in child_rows
                    )
                )
            else:
                effective_turn_mass = float(
                    sum(float(sample_weight_np[row]) for row in child_rows)
                )
            for row in child_rows:
                segment_group_id[row] = group_ordinal
                segment_id[row] = current_segment_id
                segment_turn_count[row] = float(len(child_rows))
                segment_child_count[row] = float(child_count)
            group_child_masses.append(effective_turn_mass)
            raw_group_masses.append(effective_turn_mass)
        raw_child_masses_by_group.append(raw_group_masses)
        if segment_balance and child_count > 1:
            total_mass = float(sum(raw_group_masses))
            target_mass = total_mass / float(child_count)
            factors = [
                target_mass / max(float(mass), float(epsilon))
                if mass > float(epsilon)
                else 1.0
                for mass in raw_group_masses
            ]


            lower = 1.0 / weight_cap
            upper = weight_cap
            low_scale, high_scale = (
                0.0,
                weight_cap / max(min(factors, default=1.0), float(epsilon)),
            )
            for _ in range(60):
                scale = 0.5 * (low_scale + high_scale)
                projected = [
                    min(upper, max(lower, factor * scale)) for factor in factors
                ]
                projected_mass = float(
                    sum(
                        mass * factor
                        for mass, factor in zip(raw_group_masses, projected)
                    )
                )
                if projected_mass < total_mass:
                    low_scale = scale
                else:
                    high_scale = scale
            scale = 0.5 * (low_scale + high_scale)
            clipped = [min(upper, max(lower, factor * scale)) for factor in factors]
            balanced_children += child_count
            cap_hit_children += sum(
                abs(factor - lower) <= 1e-6 or abs(factor - upper) <= 1e-6
                for factor in clipped
            )
            balanced_masses = [
                mass * factor for mass, factor in zip(raw_group_masses, clipped)
            ]
            balanced_child_masses_by_group.append(balanced_masses)
            for factor, child_rows in zip(clipped, children.values()):
                for row in child_rows:
                    trajectory_policy_weight[row] = float(factor)
        else:
            balanced_child_masses_by_group.append(list(raw_group_masses))
        child_masses_by_group.append(group_child_masses)


    balanced_child_masses = balanced_child_masses_by_group


    policy_graph_scalar = np.zeros(bsz, dtype=np.float32)
    weighted_state_mean_abs: list[float] = []
    state_policy_weight = torch.ones_like(trajectory_policy_weight)
    policy_weight_np = state_policy_weight.detach().float().cpu().numpy()
    for rows in state_rows.values():
        if len(rows) < 2:
            continue
        row_values = [float(graph_value_by_row[row]) for row in rows]
        row_weights = [float(policy_weight_np[row]) for row in rows]
        normalized = _normalize_relative_values(
            row_values, mode=normalized_mode, epsilon=epsilon
        )
        for position, row in enumerate(rows):
            policy_graph_scalar[row] = float(normalized[position])
        mass = float(sum(row_weights))
        if mass > epsilon:
            weighted_state_mean_abs.append(
                abs(float(np.dot(row_weights, normalized) / mass))
            )
    graph_scalar = policy_graph_scalar


    graph_residual_mask = graph_contrast_available.copy()
    graph_residual_scalar = graph_scalar.copy()
    graph_residual_scalar[~graph_residual_mask] = 0.0

    raw_trajectory_tokens = (
        torch.as_tensor(trajectory_scalar, device=device, dtype=dtype)[:, None]
        * response_mask
    )
    episode_tokens = fallback_tokens
    subtree_delta_tokens = torch.zeros_like(raw_trajectory_tokens)
    physical_subtree_tokens = (
        torch.as_tensor(subtree_scalar, device=device, dtype=dtype)[:, None]
        * response_mask
    )


    raw_sibling_tokens = (
        torch.as_tensor(graph_residual_scalar, device=device, dtype=dtype)[:, None]
        * response_mask
    )
    sibling_policy_weight = state_policy_weight * graph_weight
    graph_tokens = raw_sibling_tokens * state_policy_weight[:, None]
    graph_raw_tokens = (
        torch.as_tensor(graph_raw_scalar, device=device, dtype=dtype)[:, None]
        * response_mask
    )
    weighted_physical_subtree = physical_subtree_tokens * rollout_weight[:, None]
    weighted_episode = episode_tokens * rollout_weight[:, None]
    weighted_subtree_delta = torch.zeros_like(raw_trajectory_tokens)
    raw_weighted_subtree = raw_trajectory_tokens * rollout_weight[:, None]
    weighted_subtree = (
        raw_trajectory_tokens
        * trajectory_policy_weight[:, None]
        * rollout_weight[:, None]
    )
    weighted_graph = graph_tokens * graph_weight * rollout_weight[:, None]
    fallback_tokens = weighted_episode
    advantages = weighted_subtree + weighted_graph
    effective_edge_weight = (advantages.abs().sum(dim=-1) > epsilon).to(
        dtype
    ) * rollout_weight
    graph_effective_weight = (weighted_graph.abs().sum(dim=-1) > epsilon).to(
        dtype
    ) * rollout_weight
    tree_reward = (
        torch.as_tensor(q_by_row, device=device, dtype=dtype)[:, None] * response_mask
    )
    discounted_reward = (
        torch.as_tensor(discounted_q, device=device, dtype=dtype)[:, None]
        * response_mask
    )
    denominator = (response_mask * rollout_weight[:, None]).sum().clamp_min(1.0).item()
    active_count = int(active.sum().item())
    subtree_rows = int(subtree_available.sum())
    graph_rows = int(graph_available.sum())
    tree_diagnostics = dict(backups.get("diagnostics", {}))
    objective_denominator = max(1, active_turn_count)
    objective_trajectory_row = weighted_subtree.sum(
        dim=-1
    ) / response_turn_lengths.clamp_min(1.0)
    objective_state_row = weighted_graph.sum(dim=-1) / response_turn_lengths.clamp_min(
        1.0
    )
    objective_combined_row = advantages.sum(dim=-1) / response_turn_lengths.clamp_min(
        1.0
    )
    active_policy_weights = trajectory_policy_weight[policy_active]
    policy_mass_by_row = trajectory_policy_weight * rollout_weight
    if token_measure:
        policy_mass_by_row = policy_mass_by_row * response_turn_lengths
    active_policy_mass = policy_mass_by_row[policy_active]
    if active_policy_weights.numel() > 0:
        policy_weight_sum = float(active_policy_weights.sum().item())
        policy_weight_square_sum = float(active_policy_weights.square().sum().item())
        policy_weight_ess = (
            policy_weight_sum
            * policy_weight_sum
            / max(float(epsilon), policy_weight_square_sum)
        )
        policy_weight_quantiles = torch.quantile(
            active_policy_weights.float(),
            torch.tensor([0.5, 0.95], device=active_policy_weights.device),
        )
    else:
        policy_weight_ess = 0.0
        policy_weight_quantiles = torch.zeros(2, device=device)
    if active_policy_mass.numel() > 0:
        policy_mass_sum = float(active_policy_mass.sum().item())
        policy_mass_square_sum = float(active_policy_mass.square().sum().item())
        policy_mass_ess = (
            policy_mass_sum
            * policy_mass_sum
            / max(float(epsilon), policy_mass_square_sum)
        )
    else:
        policy_mass_ess = 0.0
    segment_sizes = [
        len(rows) for children in segment_groups.values() for rows in children.values()
    ]
    child_mass_relative_gaps: list[float] = []
    raw_child_mass_relative_gaps: list[float] = []
    for masses in balanced_child_masses:
        if not masses:
            continue
        mass_mean = float(np.mean(masses))
        if mass_mean > 0.0:
            child_mass_relative_gaps.append(
                float((max(masses) - min(masses)) / mass_mean)
            )
    for masses in raw_child_masses_by_group:
        if not masses:
            continue
        mass_mean = float(np.mean(masses))
        if mass_mean > 0.0:
            raw_child_mass_relative_gaps.append(
                float((max(masses) - min(masses)) / mass_mean)
            )
    tree_diagnostics.update(
        {
            "tree_mode": "graph_gigpo_recursive",
            "tree_grpo_enabled": 0,
            "tree_gigpo_enabled": 0,
            "tree_gigpo_lite_enabled": 0,
            "gigpo_advantage_enabled": 0,
            "graph_gigpo_enabled": 1,
            "graph_gigpo_recursive_subtree": 1,
            "active_edge_rows": active_count,
            "effective_edge_rows": int((effective_edge_weight > 0).sum().item()),
            "zero_signal_edge_rows": int(
                ((active) & (effective_edge_weight <= 0)).sum().item()
            ),
            "actor_effective_edge_rows": int((effective_edge_weight > 0).sum().item()),
            "trajectory_effective_edge_rows": int(
                (weighted_subtree.abs().sum(dim=-1) > epsilon).sum().item()
            ),
            "sibling_effective_edge_rows": int(
                (graph_effective_weight > 0).sum().item()
            ),
            "graph_gigpo_state_groups": int(state_count),
            "graph_gigpo_multi_action_states": int(multi_action_states),
            "graph_gigpo_homogeneous_states": int(homogeneous_states),
            "graph_gigpo_unique_edges": int(unique_edge_count),
            "graph_gigpo_graph_rows": graph_rows,
            "graph_gigpo_recursive_subtree_rows": subtree_rows,
            "graph_gigpo_recursive_candidate_parent_groups": int(len(parent_children)),
            "graph_gigpo_recursive_subtree_groups": int(subtree_group_count),
            "graph_gigpo_recursive_nonzero_subtree_groups": int(subtree_nonzero_groups),
            "graph_gigpo_recursive_homogeneous_subtree_groups": int(
                subtree_homogeneous_groups
            ),
            "graph_gigpo_fallback_rows": int(len(fallback_rows)),
            "graph_gigpo_fallback_fraction": float(
                len(fallback_rows) / max(1, active_count)
            ),
            "graph_gigpo_mean_edge_samples": float(np.mean(edge_sample_counts))
            if edge_sample_counts
            else 0.0,
            "graph_gigpo_max_edge_samples": int(max(edge_sample_counts))
            if edge_sample_counts
            else 0,
            "graph_gigpo_local_row_scalar_mean": float(
                np.mean(graph_scalar[graph_available])
                if np.any(graph_available)
                else 0.0
            ),
            "graph_gigpo_recursive_weight": 1.0,
            "graph_gigpo_recursive_delta_weight": 0.0,
            "graph_gigpo_graph_weight": float(graph_weight),
            "sibling_advantage_weight": float(graph_weight),
            "trajectory_advantage_weight": 1.0,
            "graph_gigpo_mean_std_norm": int(normalized_mode == "mean_std_norm"),
            "graph_gigpo_hierarchical_trajectory": 1,
            "graph_gigpo_tree_discounted_step_return_enabled": int(
                bool(use_tree_discounted_step_return)
            ),
            "graph_gigpo_segment_balanced_policy": int(bool(segment_balance)),
            "graph_gigpo_unified_tree_policy_measure": int(bool(segment_balance)),
            "graph_gigpo_tree_policy_weight_enabled": int(bool(segment_balance)),
            "graph_gigpo_segment_policy_measure_token": int(token_measure),
            "graph_gigpo_segment_weight_cap": float(weight_cap),
            "graph_gigpo_segment_cap_hit_fraction": float(
                cap_hit_children / max(1, balanced_children)
            ),
            "graph_gigpo_segment_groups": int(group_count),
            "graph_gigpo_segments": int(segment_count),
            "graph_gigpo_segment_turn_count_mean": float(
                np.mean(segment_sizes) if segment_sizes else 0.0
            ),
            "graph_gigpo_segment_turn_count_max": int(
                max(segment_sizes) if segment_sizes else 0
            ),
            "graph_gigpo_trajectory_policy_weight_mean": float(
                active_policy_weights.mean().item()
                if active_policy_weights.numel()
                else 0.0
            ),
            "graph_gigpo_trajectory_policy_weight_p50": float(
                policy_weight_quantiles[0].item()
            ),
            "graph_gigpo_trajectory_policy_weight_p95": float(
                policy_weight_quantiles[1].item()
            ),
            "graph_gigpo_trajectory_policy_weight_max": float(
                active_policy_weights.max().item()
                if active_policy_weights.numel()
                else 0.0
            ),
            "graph_gigpo_trajectory_policy_weight_ess": float(policy_weight_ess),
            "graph_gigpo_trajectory_policy_mass_ess": float(policy_mass_ess),
            "graph_gigpo_sibling_child_mass_relative_gap_max": float(
                max(child_mass_relative_gaps) if child_mass_relative_gaps else 0.0
            ),
            "graph_gigpo_raw_sibling_child_mass_relative_gap_max": float(
                max(raw_child_mass_relative_gaps)
                if raw_child_mass_relative_gaps
                else 0.0
            ),
            "graph_gigpo_weighted_trajectory_objective_mean": float(
                objective_trajectory_row[policy_active].sum().item()
                / objective_denominator
            ),
            "graph_gigpo_state_objective_mean": float(
                objective_state_row[policy_active].sum().item() / objective_denominator
            ),
            "graph_gigpo_state_group_weighted_mean_abs_max": float(
                max(weighted_state_mean_abs) if weighted_state_mean_abs else 0.0
            ),
            "graph_gigpo_combined_objective_mean": float(
                objective_combined_row[policy_active].sum().item()
                / objective_denominator
            ),
            "graph_gigpo_additive_subtree_delta": 0,
            "gamma": float(gamma),
            "depth_neutral_sibling_credit": 1,
            "graph_gigpo_recursive_credit_uses_gamma": 0,
            "discounted_reward_gap_mean": float(
                (
                    (tree_reward - discounted_reward).abs()
                    * response_mask
                    * rollout_weight[:, None]
                )
                .sum()
                .item()
                / denominator
            ),
            "graph_gigpo_subtree_abs_mean": float(
                weighted_physical_subtree.abs().sum().item() / denominator
            ),
            "graph_gigpo_episode_abs_mean": float(
                weighted_subtree.abs().sum().item() / denominator
            ),
            "graph_gigpo_raw_episode_abs_mean": float(
                raw_weighted_subtree.abs().sum().item() / denominator
            ),
            "graph_gigpo_global_episode_abs_mean": float(
                weighted_episode.abs().sum().item() / denominator
            ),
            "graph_gigpo_recursive_delta_abs_mean": float(
                weighted_subtree_delta.abs().sum().item() / denominator
            ),
            "graph_gigpo_local_abs_mean": float(
                weighted_graph.abs().sum().item() / denominator
            ),
            "graph_gigpo_graph_raw_abs_mean": float(
                (graph_raw_tokens * graph_weight * rollout_weight[:, None])
                .abs()
                .sum()
                .item()
                / denominator
            ),
            "graph_gigpo_fallback_abs_mean": float(
                fallback_tokens.abs().sum().item() / denominator
            ),
            "graph_gigpo_subtree_graph_overlap_rows": int(
                np.count_nonzero(subtree_available & graph_available)
            ),
            "graph_gigpo_subtree_only_rows": int(
                np.count_nonzero(subtree_available & ~graph_available)
            ),
            "graph_gigpo_graph_only_rows": int(
                np.count_nonzero(graph_available & ~subtree_available)
            ),
            "graph_gigpo_graph_contrast_rows": int(
                np.count_nonzero(graph_contrast_available)
            ),
            "graph_gigpo_graph_residual_rows": int(
                np.count_nonzero(graph_residual_mask)
            ),
            "graph_gigpo_graph_suppressed_overlap_rows": 0,
            "graph_gigpo_cross_parent_rows": int(
                np.count_nonzero(graph_cross_parent_available)
            ),
            "graph_gigpo_cross_parent_overlap_rows": int(
                np.count_nonzero(
                    graph_cross_parent_available & subtree_signal_available
                )
            ),
            "graph_gigpo_global_episode_rows": int(len(fallback_rows)),
            "graph_gigpo_root_episode_trajectories": int(
                sum(len(roots) for roots in task_roots.values())
            ),
            "graph_gigpo_branch_inherited_episode_rows": int(
                sum(
                    1
                    for row in torch.nonzero(active, as_tuple=False)
                    .reshape(-1)
                    .detach()
                    .cpu()
                    .tolist()
                    if root_id_by_row[row] != str(traj_index[row])
                )
            ),
            "graph_gigpo_advantage_token_mean": float(
                (advantages * response_mask).sum().item() / denominator
            ),
            "graph_gigpo_advantage_abs_max": float(advantages.abs().max().item())
            if advantages.numel()
            else 0.0,
            "graph_gigpo_positive_edge_fraction": float(
                ((advantages * response_mask).sum(dim=-1) > epsilon).sum().item()
                / max(1, active_count)
            ),
            "graph_gigpo_negative_edge_fraction": float(
                ((advantages * response_mask).sum(dim=-1) < -epsilon).sum().item()
                / max(1, active_count)
            ),
        }
    )
    components = {
        "tree": advantages,


        "trajectory": weighted_subtree,
        "raw_trajectory": raw_weighted_subtree,
        "raw_trajectory_unweighted": raw_trajectory_tokens,
        "trajectory_segment_return": torch.as_tensor(
            segment_return_by_row, device=device, dtype=dtype
        ),
        "trajectory_policy_weight": trajectory_policy_weight,
        "trajectory_segment_group_id": segment_group_id,
        "trajectory_segment_id": segment_id,
        "trajectory_segment_turn_count": segment_turn_count,
        "trajectory_segment_child_count": segment_child_count,
        "episode": weighted_episode,
        "recursive_subtree_delta": weighted_subtree_delta,
        "sibling": weighted_graph,
        "raw_sibling": raw_sibling_tokens,
        "sibling_policy_weight": sibling_policy_weight,


        "invalid_action": torch.zeros_like(advantages),
        "tree_reward": tree_reward,
        "discounted_tree_reward": discounted_reward,
        "rollout_weight": rollout_weight,
        "effective_edge_weight": effective_edge_weight,
        "sibling_effective_edge_weight": graph_effective_weight,
        "tree_diagnostics": tree_diagnostics,
    }
    return advantages, advantages, components


def compute_el_tsbpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence,
    traj_index: Sequence,
    *,
    bare_rollout_id: Sequence | None = None,
    trajectory_success: Sequence | None = None,
    sample_weight: torch.Tensor | None = None,
    parent_node_id: Sequence | None = None,
    node_id: Sequence | None = None,
    terminal_reward: Sequence | None = None,
    node_is_terminal: Sequence | None = None,
    trajectory_advantage_weight: float = 1.0,
    gamma: float = 1.0,
    epsilon: float = 1e-6,
    advantage_mode: str = "graph_gigpo",
    gigpo_step_rewards: torch.Tensor | None = None,
    gigpo_anchor_obs: Sequence | None = None,
    gigpo_action_text: Sequence | None = None,
    gigpo_step_advantage_w: float = 1.0,
    gigpo_mode: str = "mean_norm",
    gigpo_enable_similarity: bool = False,
    gigpo_similarity_thresh: float = 0.95,
    invalid_action_penalty: torch.Tensor | None = None,
    graph_gigpo_use_tree_discounted_step_return: bool = True,
    graph_gigpo_segment_balance: bool = True,
    graph_gigpo_policy_loss_agg_mode: str = "seq-mean-token-sum",
    graph_gigpo_segment_weight_cap: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Assign physical tree and state-group credit to sampled turns."""
    if str(advantage_mode).lower() != "graph_gigpo":
        raise ValueError("EL-TSBPO supports graph_gigpo credit only")
    if parent_node_id is None or node_id is None:
        raise ValueError("Physical parent and child node IDs are required")
    if (
        gigpo_step_rewards is None
        or gigpo_anchor_obs is None
        or gigpo_action_text is None
    ):
        raise ValueError("Step rewards, state observations and actions are required")
    return _graph_gigpo_recursive_advantage(
        token_level_rewards=token_level_rewards,
        step_rewards=gigpo_step_rewards,
        response_mask=response_mask,
        index=index,
        traj_index=traj_index,
        bare_rollout_id=bare_rollout_id,
        anchor_obs=gigpo_anchor_obs,
        action_text=gigpo_action_text,
        parent_node_id=parent_node_id,
        node_id=node_id,
        terminal_reward=terminal_reward,
        trajectory_success=trajectory_success,
        node_is_terminal=node_is_terminal,
        sample_weight=sample_weight,
        episode_advantage_w=trajectory_advantage_weight,
        step_advantage_w=gigpo_step_advantage_w,
        recursive_advantage_w=None,
        mode=gigpo_mode,
        enable_similarity=gigpo_enable_similarity,
        similarity_thresh=gigpo_similarity_thresh,
        epsilon=epsilon,
        gamma=gamma,
        invalid_action_penalty=invalid_action_penalty,
        use_tree_discounted_step_return=graph_gigpo_use_tree_discounted_step_return,
        segment_balance=graph_gigpo_segment_balance,
        policy_loss_agg_mode=graph_gigpo_policy_loss_agg_mode,
        segment_weight_cap=graph_gigpo_segment_weight_cap,
    )


__all__ = ["compute_el_tsbpo_advantage", "compute_tree_backups"]
