# Copyright 2025 NTU / verl-agent team. Licensed under Apache-2.0.
"""Stable identities shared by agent rollouts and optional samplers."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Optional

import numpy as np


def _to_python(value: Any) -> Any:
    """Convert tensors, arrays, and configs into JSON-compatible values."""

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def stable_hash(payload: Any, *, prefix: str = "sha256") -> str:
    """Hash canonical JSON so equal identities ignore formatting differences."""

    encoded = json.dumps(
        _to_python(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()}"


def _valid_indices(input_ids: Any, attention_mask: Any) -> tuple[list[int], list[int]]:
    ids = _to_python(input_ids)
    mask = _to_python(attention_mask)
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    if isinstance(mask, list) and mask and isinstance(mask[0], list):
        mask = mask[0]
    ids = list(ids or [])
    mask = list(mask or [])
    if len(mask) != len(ids):
        return ids, list(range(len(ids)))
    return ids, [idx for idx, item in enumerate(mask) if bool(item)]


def _select_positions(position_ids: Any, indices: list[int]) -> Any:
    positions = _to_python(position_ids)
    if positions is None:
        return None
    if isinstance(positions, list) and positions and isinstance(positions[0], list):
        if all(isinstance(row, list) for row in positions):
            if all(len(row) > max(indices, default=-1) for row in positions):
                return [[row[idx] for idx in indices] for row in positions]
    return [positions[idx] for idx in indices if idx < len(positions)]


def context_hash(
    input_ids: Any,
    attention_mask: Any = None,
    position_ids: Any = None,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    """Hash the actual unpadded model context and relevant runtime metadata."""

    ids, indices = _valid_indices(input_ids, attention_mask)
    return stable_hash(
        {
            "token_ids": [ids[idx] for idx in indices],
            "position_ids": _select_positions(position_ids, indices),
            "metadata": _to_python(metadata or {}),
        },
        prefix="ctx",
    )


def environment_snapshot_hash(
    *,
    gamefile: Any = None,
    action_history: Any = None,
    step_count: Any = None,
    seed: Any = None,
    rng_state: Any = None,
    anchor_obs: Any = None,
    admissible_commands: Any = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """Create a deterministic snapshot identity or an explicit incomplete one."""

    if action_history is None:
        normalized_history = []
    elif isinstance(action_history, (str, bytes)):
        normalized_history = [action_history]
    else:
        normalized_history = list(action_history)
    if admissible_commands is None:
        normalized_actions = []
    elif isinstance(admissible_commands, (str, bytes)):
        normalized_actions = [str(admissible_commands)]
    else:
        normalized_actions = sorted(
            str(command).strip().lower() for command in admissible_commands
        )
    return stable_hash(
        {
            "gamefile": gamefile,
            "action_history": normalized_history,
            "step_count": step_count,
            "seed": seed,
            "rng_state": rng_state,
            "anchor_obs": anchor_obs,
            "admissible_commands": normalized_actions,
            "extra": {} if extra is None else extra,
        },
        prefix="env",
    )


def env_key(task_scope: Any, snapshot_hash: str) -> str:
    """Environment-level key scoped to a task."""

    return stable_hash(
        {"task_scope": task_scope, "snapshot_hash": snapshot_hash},
        prefix="node",
    )


def environment_state_hash(
    *,
    task_scope: Any = None,
    gamefile: Any = None,
    anchor_obs: Any = None,
    admissible_commands: Any = None,
    step_count: Any = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """Hash the observable environment state without the full action prefix."""

    if admissible_commands is None:
        normalized_actions = []
    elif isinstance(admissible_commands, (str, bytes)):
        normalized_actions = [str(admissible_commands)]
    else:
        normalized_actions = sorted(
            str(command).strip().lower() for command in admissible_commands
        )
    return stable_hash(
        {
            "task_scope": task_scope,
            "gamefile": gamefile,
            "anchor_obs": anchor_obs,
            "admissible_commands": normalized_actions,
            "step_count": step_count,
            "extra": {} if extra is None else extra,
        },
        prefix="state",
    )


def decision_key(
    context_digest: str,
    snapshot_digest: str,
    *,
    runtime_state: Any = None,
) -> str:
    """Identity of a complete policy decision point."""

    return stable_hash(
        {
            "context_hash": context_digest,
            "snapshot_hash": snapshot_digest,
            "runtime_state": {} if runtime_state is None else runtime_state,
        },
        prefix="decision",
    )


def decode_config_hash(config: Any) -> str:
    """Fingerprint sampling settings that change the behavior distribution."""

    return stable_hash({} if config is None else config, prefix="decode")


def policy_group_key(
    decision_digest: str,
    policy_version: Any,
    decode_digest: str,
) -> str:
    """Identity used for exact-prefix sibling eligibility."""

    return stable_hash(
        {
            "decision_key": decision_digest,
            "policy_version": str(policy_version),
            "decode_config_hash": decode_digest,
        },
        prefix="policy-group",
    )


__all__ = [
    "stable_hash",
    "context_hash",
    "environment_snapshot_hash",
    "environment_state_hash",
    "env_key",
    "decision_key",
    "decode_config_hash",
    "policy_group_key",
]
