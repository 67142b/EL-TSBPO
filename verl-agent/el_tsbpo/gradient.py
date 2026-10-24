"""EL-TSBPO gradient implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

_HASH_MODULUS = 2_147_483_647
_HASH_MULTIPLIER = 1_103_515_245
_HASH_SEED_MULTIPLIER = 12_345


@torch.no_grad()
def fixed_count_sketch(
    values: torch.Tensor,
    *,
    global_offset: int,
    dimension: int,
    seed: int,
    chunk_size: int = 1_048_576,
) -> torch.Tensor:

    dimension = max(1, int(dimension))
    flat = values.detach().reshape(-1)
    output = torch.zeros(dimension, dtype=torch.float32, device=flat.device)
    seed_term = (int(seed) % _HASH_MODULUS) * _HASH_SEED_MULTIPLIER
    for start in range(0, flat.numel(), max(1, int(chunk_size))):
        stop = min(flat.numel(), start + max(1, int(chunk_size)))
        indices = torch.arange(
            global_offset + start,
            global_offset + stop,
            dtype=torch.int64,
            device=flat.device,
        )
        indices = torch.remainder(indices, _HASH_MODULUS)
        hashed = torch.remainder(
            indices * _HASH_MULTIPLIER + seed_term,
            _HASH_MODULUS,
        )
        buckets = torch.remainder(hashed, dimension)
        signs = torch.where(
            torch.remainder(
                torch.div(hashed, dimension, rounding_mode="floor"), 2
            )
            == 0,
            1.0,
            -1.0,
        )
        output.scatter_add_(0, buckets, flat[start:stop].float() * signs)
    return output


def expanded_edge_indices(
    parent_node_ids: Sequence[Any] | None,
    group_memberships: Sequence[Any] | None,
    rollout_weight: Sequence[Any] | torch.Tensor,
) -> tuple[list[int], tuple[str, ...]]:

    if parent_node_ids is None or group_memberships is None:
        return [], ()
    if isinstance(rollout_weight, torch.Tensor):
        weight_values = rollout_weight.detach().reshape(-1).cpu().tolist()
    else:
        weight_values = rollout_weight
    if len(weight_values) != len(parent_node_ids) or len(group_memberships) != len(
        parent_node_ids
    ):
        raise ValueError("EL-TSBPO expansion metadata must align with rollout rows")
    indices: list[int] = []
    anchors: set[str] = set()
    for row, parent_value in enumerate(parent_node_ids):
        weight = weight_values[row]
        if float(weight) <= 0.0:
            continue
        parent = str(parent_value)
        memberships = group_memberships[row]
        if isinstance(memberships, Mapping):
            memberships = [memberships]
        if not isinstance(memberships, (list, tuple)):
            continue
        matched = any(
            isinstance(membership, Mapping)
            and str(membership.get("anchor_node_id", "")) == parent
            and bool(membership.get("group_id"))
            for membership in memberships
        )
        if matched:
            indices.append(row)
            anchors.add(parent)
    return indices, tuple(sorted(anchors))


def rotating_probe_tasks(
    task_uids: Sequence[Any],
    *,
    round_index: int,
    count: int,
) -> tuple[str, ...]:

    unique_tasks = tuple(dict.fromkeys(str(value) for value in task_uids))
    if len(unique_tasks) < 2 or int(count) <= 0:
        return ()
    reserve = min(int(count), len(unique_tasks) - 1)
    start = (max(0, int(round_index)) * reserve) % len(unique_tasks)
    return tuple(
        unique_tasks[(start + offset) % len(unique_tasks)]
        for offset in range(reserve)
    )


__all__ = [
    "expanded_edge_indices",
    "fixed_count_sketch",
    "rotating_probe_tasks",
]
