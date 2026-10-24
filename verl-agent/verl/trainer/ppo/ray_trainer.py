# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import hashlib
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_host_memory_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    filter_el_tsbpo_quality_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
try:
    from verl.utils.turn_diagnostics import CONFIG_KEYS, build_pre_update_dump, write_json
except ModuleNotFoundError as exc:
    if exc.name != "verl.utils.turn_diagnostics":
        raise
    CONFIG_KEYS = ()
    build_pre_update_dump = None
    write_json = None
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager
from gigpo import core_gigpo
from el_tsbpo import core_el_tsbpo
from el_tsbpo.gradient import expanded_edge_indices

from agent_system.multi_turn_rollout import TrajectoryCollector, adjust_batch


WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"
    GiGPO = 'gigpo'
    EL_TSBPO = 'el_tsbpo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics

def apply_invalid_action_penalty(
    data: DataProto,
    invalid_action_penalty_coef: float = 0.0,
    truncation_penalty_coef: float = 0.0,
):
    """Subtract output costs on their originating turns before normalization.

    The legacy function name is retained for callers. True token-budget
    truncation is identified from backend finish_reason, never padded length.
    Separate costs are logged; their sum is used by tree credit assignment.
    """
    def checked_coefficient(value, name):
        try:
            coefficient = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite non-negative number") from exc
        if not np.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number")
        return coefficient

    coefficient = checked_coefficient(invalid_action_penalty_coef, "invalid_action_penalty_coef")
    truncation_coefficient = checked_coefficient(truncation_penalty_coef, "truncation_penalty_coef")
    finish_reasons = data.non_tensor_batch.get('generation_finish_reason')
    if truncation_coefficient > 0.0 and finish_reasons is None:
        raise ValueError("truncation penalty requires generation_finish_reason from the rollout backend")
    if finish_reasons is not None:
        finish_reasons = np.asarray(finish_reasons, dtype=object).reshape(-1)
        if finish_reasons.size != len(data):
            raise ValueError("generation_finish_reason must have one value per rollout row")
        sample_weight = data.batch.get('el_tsbpo_sample_weight')
        active_rows = (
            np.ones(len(data), dtype=bool) if sample_weight is None
            else sample_weight.detach().cpu().numpy().reshape(-1) > 0
        )
        if truncation_coefficient > 0.0 and any(
            active and reason not in {'stop', 'length'}
            for active, reason in zip(active_rows, finish_reasons)
        ):
            raise ValueError("truncation penalty requires a known stop/length finish reason for every active turn")
    truncation_values = np.asarray(
        [reason == 'length' for reason in finish_reasons]
        if finish_reasons is not None else [False] * len(data),
        dtype=np.float32,
    )
    reward_tensor = data.batch['token_level_scores']
    # Keep the penalty on its originating turn for EL-TSBPO credit assignment.
    invalid_penalties = torch.zeros(
        len(data), dtype=reward_tensor.dtype, device=reward_tensor.device
    )
    truncation_penalties = torch.zeros_like(invalid_penalties)
    sample_weights = data.batch.get(
        'el_tsbpo_sample_weight',
        torch.ones(
            len(data),
            dtype=reward_tensor.dtype,
            device=reward_tensor.device,
        ),
    ).detach().float().cpu().numpy()
    sample_weights = np.asarray(sample_weights).reshape(-1)
    if sample_weights.size != len(data):
        raise ValueError("el_tsbpo_sample_weight must have one value per rollout row")
    sample_weights = np.where(
        np.isfinite(sample_weights),
        np.clip(sample_weights, 0.0, 1.0),
        0.0,
    )
    invalid_count = 0.0
    if 'step_rewards' in data.batch.keys():
        step_rewards = data.batch['step_rewards']
    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem

        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

        action_values = np.asarray(
            data_item.non_tensor_batch.get('is_action_valid', True),
            dtype=np.float32,
        ).reshape(-1)
        # One rollout row represents one environment action.  Be tolerant of
        # scalar numpy values and collated ``(1,)`` values, but never silently
        # broadcast a malformed multi-action row.
        if action_values.size != 1:
            raise ValueError(
                "is_action_valid must contain exactly one flag per rollout row"
            )
        valid_flag = bool(np.isfinite(action_values[0]) and action_values[0] >= 0.5)
        action_invalids = torch.tensor(
            0.0 if valid_flag else 1.0,
            dtype=torch.float32,
            device=prompt_ids.device,
        )
        penalty = coefficient * action_invalids
        effective_penalty = penalty * float(sample_weights[i])
        invalid_penalties[i] = effective_penalty.to(dtype=invalid_penalties.dtype)
        truncation_penalties[i] = (
            truncation_coefficient * float(truncation_values[i]) * float(sample_weights[i])
        )
        turn_penalty = effective_penalty + truncation_penalties[i]
        invalid_count += float(action_invalids.detach().float().item()) * float(sample_weights[i])
        # invalid action penalty
        # assert reward_tensor[i, valid_response_length - 1] != 0.0, f'i={i}'
        response_length = int(reward_tensor.shape[-1])
        if response_length > 0:
            valid_length = int(valid_response_length.detach().item())
            target_index = max(0, min(response_length - 1, valid_length - 1))
            reward_tensor[i, target_index] -= turn_penalty

        if 'step_rewards' in data.batch.keys():
            step_rewards[i] -= turn_penalty

    data.batch['invalid_action_penalty'] = invalid_penalties
    data.batch['truncation_penalty'] = truncation_penalties
    data.batch['turn_output_penalty'] = invalid_penalties + truncation_penalties
    valid_values = np.asarray(
        data.non_tensor_batch.get(
            'is_action_valid', np.ones(len(data), dtype=np.float32)
        ),
        dtype=np.float32,
    ).reshape(-1)
    if valid_values.size != len(data):
        raise ValueError("is_action_valid must have one flag per rollout row")
    format_values = np.asarray(
        data.non_tensor_batch.get('is_action_format_valid', valid_values),
        dtype=np.float32,
    ).reshape(-1)
    admissible_values = np.asarray(
        data.non_tensor_batch.get('is_action_admissible', valid_values),
        dtype=np.float32,
    ).reshape(-1)
    if format_values.size != len(data) or admissible_values.size != len(data):
        raise ValueError(
            "is_action_format_valid/is_action_admissible must have one flag per rollout row"
        )
    parent_values = np.asarray(
        data.non_tensor_batch.get('parent_traj_uid', [None] * len(data)),
        dtype=object,
    ).reshape(-1)
    branch_values = np.asarray(
        [value is not None and str(value) not in {"", "None", "null", "nan"}
         for value in parent_values],
        dtype=bool,
    )

    def weighted_ratio(values, mask=None):
        weights = sample_weights if mask is None else sample_weights[mask]
        selected = values if mask is None else values[mask]
        denom = float(weights.sum())
        return float(np.sum(selected * weights) / denom) if denom > 0.0 else 0.0

    weight_sum = float(sample_weights.sum())
    valid_action_ratio = (
        float(np.sum(valid_values * sample_weights) / weight_sum)
        if weight_sum > 0.0 else 0.0
    )
    effective_penalties = invalid_penalties[
        torch.as_tensor(sample_weights > 0.0, dtype=torch.bool, device=invalid_penalties.device)
    ]
    metrics = {
        'episode/valid_action_ratio': valid_action_ratio,
        'episode/invalid_action_rate': 1.0 - valid_action_ratio,
        'episode/invalid_action_count': invalid_count,
        'episode/invalid_action_penalty_mean': float(
            effective_penalties.detach().float().mean().item()
            if effective_penalties.numel() else 0.0
        ),
        'episode/invalid_action_penalty_max': float(
            effective_penalties.detach().float().max().item()
            if effective_penalties.numel() else 0.0
        ),
        'episode/action_format_valid_ratio': weighted_ratio(format_values),
        'episode/admissible_action_ratio': weighted_ratio(admissible_values),
        'episode/natural_valid_action_ratio': weighted_ratio(
            valid_values, ~branch_values
        ) if np.any(~branch_values) else 0.0,
        'episode/branch_valid_action_ratio': weighted_ratio(
            valid_values, branch_values
        ) if np.any(branch_values) else 0.0,
    }
    if finish_reasons is not None:
        metrics.update({
            'episode/truncation_rate': weighted_ratio(truncation_values),
            'episode/natural_truncation_rate': weighted_ratio(truncation_values, ~branch_values),
            'episode/branch_truncation_rate': weighted_ratio(truncation_values, branch_values),
        })
    return data, metrics


def _el_tsbpo_terminal_returns(data: DataProto) -> np.ndarray | None:
    """Return factual terminal outcomes without cross-turn penalty leakage.

    Invalid-action and truncation costs are kept on the turn that produced
    them by ``apply_invalid_action_penalty``.  Aggregating them into a
    trajectory terminal reward would make an invalid action alter every other
    turn's Tree/Agent-DPS signal, which is not the intended environment
    semantics.
    """

    metadata = data.non_tensor_batch
    terminal = metadata.get("el_tsbpo_terminal_reward")
    terminal_flags = metadata.get("el_tsbpo_node_is_terminal")
    trajectories = metadata.get("traj_uid")
    if terminal is None or terminal_flags is None or trajectories is None:
        return None

    terminal = np.asarray(terminal, dtype=np.float32).reshape(-1).copy()
    terminal_flags = np.asarray(terminal_flags, dtype=bool).reshape(-1)
    trajectories = np.asarray(trajectories, dtype=object).reshape(-1)
    if not (
        len(terminal) == len(data)
        and len(terminal_flags) == len(data)
        and len(trajectories) == len(data)
    ):
        raise ValueError("EL-TSBPO terminal metadata must align with rollout rows")

    # Keep this metric for log compatibility, but it must remain zero because
    # no per-trajectory aggregation is performed anymore.
    data.meta_info["el_tsbpo_invalid_penalty_total"] = 0.0
    return terminal

def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_rollout_logprob_audit_metrics(
    rollout_log_probs: torch.Tensor,
    actor_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
    clip_low: float = 0.8,
    clip_high: float = 1.2,
    prefix: str = "training/rollout_logprob_audit",
) -> Dict[str, float]:
    """Compare behavior and actor probabilities on actual training tokens.

    The reported ratio is ``pi_actor / pi_rollout``. Fully masked multi-turn
    tokens and zero-weight EL-TSBPO rows are excluded so padding cannot make
    the audit look artificially better or worse.
    """

    if rollout_log_probs.shape != actor_log_probs.shape:
        raise ValueError(
            "rollout/actor log-prob shape mismatch: "
            f"{tuple(rollout_log_probs.shape)} vs {tuple(actor_log_probs.shape)}"
        )
    if response_mask.shape != rollout_log_probs.shape:
        raise ValueError(
            "log-prob audit mask shape mismatch: "
            f"{tuple(response_mask.shape)} vs {tuple(rollout_log_probs.shape)}"
        )
    valid_mask = response_mask.to(dtype=torch.bool)
    if sample_weight is not None:
        if sample_weight.ndim != 1 or sample_weight.shape[0] != valid_mask.shape[0]:
            raise ValueError("log-prob audit sample_weight must have one value per row")
        valid_mask = valid_mask & (sample_weight > 0).unsqueeze(-1)

    finite_mask = torch.isfinite(rollout_log_probs) & torch.isfinite(actor_log_probs)
    nonfinite_count = int((valid_mask & ~finite_mask).sum().item())
    values_mask = valid_mask & finite_mask
    valid_count = int(values_mask.sum().item())
    metrics = {
        f"{prefix}/valid_tokens": float(valid_count),
        f"{prefix}/nonfinite_tokens": float(nonfinite_count),
    }
    if valid_count == 0:
        for name in (
            "abs_dlog_mean", "abs_dlog_p95", "abs_dlog_p99", "abs_dlog_max",
            "ratio_mean", "ratio_p01", "ratio_p50", "ratio_p99", "ratio_max",
            "ratio_outside_clipfrac", "prob_abs_diff_mean", "prob_abs_diff_max",
        ):
            metrics[f"{prefix}/{name}"] = 0.0
        return metrics

    rollout = rollout_log_probs[values_mask].detach().float()
    actor = actor_log_probs[values_mask].detach().float()
    delta = actor - rollout
    abs_delta = delta.abs()
    ratio = torch.exp(delta.clamp(min=-20.0, max=20.0))
    probability_diff = (torch.exp(actor) - torch.exp(rollout)).abs()

    def quantile(tensor: torch.Tensor, q: float) -> float:
        return float(torch.quantile(tensor, q).item())

    metrics.update({
        f"{prefix}/abs_dlog_mean": float(abs_delta.mean().item()),
        f"{prefix}/abs_dlog_p95": quantile(abs_delta, 0.95),
        f"{prefix}/abs_dlog_p99": quantile(abs_delta, 0.99),
        f"{prefix}/abs_dlog_max": float(abs_delta.max().item()),
        f"{prefix}/ratio_mean": float(ratio.mean().item()),
        f"{prefix}/ratio_p01": quantile(ratio, 0.01),
        f"{prefix}/ratio_p50": quantile(ratio, 0.50),
        f"{prefix}/ratio_p99": quantile(ratio, 0.99),
        f"{prefix}/ratio_max": float(ratio.max().item()),
        f"{prefix}/ratio_outside_clipfrac": float(
            ((ratio < float(clip_low)) | (ratio > float(clip_high))).float().mean().item()
        ),
        f"{prefix}/prob_abs_diff_mean": float(probability_diff.mean().item()),
        f"{prefix}/prob_abs_diff_max": float(probability_diff.max().item()),
    })
    return metrics


def _el_tsbpo_branch_mask(data: DataProto) -> np.ndarray:
    """Identify continuation rows from their factual tree parent."""

    parents = np.asarray(
        data.non_tensor_batch.get("parent_traj_uid", [None] * len(data)),
        dtype=object,
    ).reshape(-1)
    if parents.size != len(data):
        raise ValueError("parent_traj_uid must contain one value per rollout row")
    return np.asarray(
        [
            value is not None
            and str(value) not in {"", "None", "null", "nan"}
            for value in parents
        ],
        dtype=bool,
    )


def compute_el_tsbpo_source_policy_weights(
    data: DataProto,
    *,
    branch_share: float = 0.5,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Balance the bare and branch policy-gradient objectives by source.

    Agent-DPS continuations are conditionally sampled auxiliary evidence, not
    additional iid root rollouts.  The balance unit follows the actual policy
    aggregation: active tokens for ``token-mean``.  For sequence-mean modes,
    turns are first averaged inside each sampled continuation, continuations
    are averaged inside each source, and the two source means are mixed as
    ``(1-s) * L_bare + s * L_branch``. Increasing K or suffix length therefore
    changes an estimate, not its total coefficient.

    The returned row weights are applied to PPO advantages only. Rollout
    validity, reward, KL, and advantage construction retain their original
    masks and scales.
    """

    if not 0.0 <= float(branch_share) <= 1.0:
        raise ValueError("Graph-GiGPO branch policy share must be in [0, 1]")

    bsz = len(data)
    response_mask = data.batch.get("response_mask")
    if response_mask is None:
        response_mask = compute_response_mask(data)
    response_mask = response_mask.detach().float()
    if "loss_mask" in data.batch and "responses" in data.batch:
        response_width = int(data.batch["responses"].shape[-1])
        response_mask = data.batch["loss_mask"][:, -response_width:].detach().float()

    rollout_weight = data.batch.get(
        "el_tsbpo_sample_weight",
        torch.ones(bsz, device=response_mask.device, dtype=response_mask.dtype),
    ).detach().float().reshape(-1)
    active = rollout_weight > 0
    token_count = response_mask.sum(dim=-1) * active.to(response_mask.dtype)
    active = active & (token_count > 0)
    active_np = active.detach().cpu().numpy().astype(bool, copy=False)
    branch_np = _el_tsbpo_branch_mask(data)
    branch = torch.as_tensor(branch_np, device=response_mask.device, dtype=torch.bool)
    row_balanced = str(loss_agg_mode).startswith("seq-mean-token-")
    trajectory_values = data.non_tensor_batch.get("traj_uid")
    task_values = data.non_tensor_batch.get("uid")
    if trajectory_values is None or len(trajectory_values) != bsz:
        trajectory_ids = [f"row:{index}" for index in range(bsz)]
    else:
        trajectory_ids = [
            f"{str(task_values[index]) if task_values is not None and len(task_values) == bsz else ''}"
            f"::{str(value)}"
            for index, value in enumerate(trajectory_values)
        ]
    continuation_row_counts: dict[tuple[bool, str], int] = defaultdict(int)
    for index in range(bsz):
        if bool(active_np[index]):
            continuation_row_counts[(bool(branch_np[index]), trajectory_ids[index])] += 1
    continuation_count_values = np.ones(bsz, dtype=np.float32)
    for index in range(bsz):
        key = (bool(branch_np[index]), trajectory_ids[index])
        continuation_count_values[index] = float(
            max(1, continuation_row_counts.get(key, 1))
        )
    continuation_count_per_row = torch.as_tensor(
        continuation_count_values,
        device=response_mask.device,
        dtype=response_mask.dtype,
    )

    policy_weight = torch.zeros(
        bsz, device=response_mask.device, dtype=response_mask.dtype
    )
    total_tokens = float(token_count[active].sum().item())
    bare = active & ~branch
    branch_active = active & branch
    bare_tokens = float(token_count[bare].sum().item())
    branch_tokens = float(token_count[branch_active].sum().item())
    total_rows = float(active.sum().item())
    bare_continuations = len({
        trajectory_ids[index] for index in range(bsz)
        if bool(active_np[index]) and not bool(branch_np[index])
    })
    branch_continuations = len({
        trajectory_ids[index] for index in range(bsz)
        if bool(active_np[index]) and bool(branch_np[index])
    })
    if row_balanced and bare_continuations > 0 and branch_continuations > 0:
        bare_target = 1.0 - float(branch_share)
        branch_target = float(branch_share)
        policy_weight[bare] = (
            bare_target * total_rows / float(bare_continuations)
        ) / continuation_count_per_row[bare]
        policy_weight[branch_active] = (
            branch_target * total_rows / float(branch_continuations)
        ) / continuation_count_per_row[branch_active]
    elif row_balanced and bare_continuations > 0:
        policy_weight[bare] = (
            total_rows / float(bare_continuations)
        ) / continuation_count_per_row[bare]
    elif row_balanced and branch_continuations > 0:
        policy_weight[branch_active] = (
            total_rows / float(branch_continuations)
        ) / continuation_count_per_row[branch_active]
    elif bare_tokens > 0.0 and branch_tokens > 0.0:
        bare_target = 1.0 - float(branch_share)
        branch_target = float(branch_share)
        policy_weight[bare] = bare_target * total_tokens / bare_tokens
        policy_weight[branch_active] = branch_target * total_tokens / branch_tokens
    elif bare_tokens > 0.0:
        policy_weight[bare] = total_tokens / bare_tokens
    elif branch_tokens > 0.0:
        policy_weight[branch_active] = total_tokens / branch_tokens

    weighted_token_mass = token_count * policy_weight
    total_weighted_mass = float(weighted_token_mass.sum().item())
    bare_mass = float(weighted_token_mass[~branch].sum().item())
    branch_mass = float(weighted_token_mass[branch].sum().item())
    weighted_row_mass = active.to(response_mask.dtype) * policy_weight
    total_weighted_row_mass = float(weighted_row_mass.sum().item())
    bare_row_mass = float(weighted_row_mass[~branch].sum().item())
    branch_row_mass = float(weighted_row_mass[branch].sum().item())
    active_weights = policy_weight[active]
    metrics: Dict[str, float] = {
        "el_tsbpo/policy_weight/enabled": 1.0,
        "el_tsbpo/policy_weight/source_balanced": 1.0,
        "el_tsbpo/policy_weight/branch_share_target": float(branch_share),
        "el_tsbpo/policy_weight/balance_unit_is_row": float(row_balanced),
        "el_tsbpo/policy_weight/continuation_balanced": float(row_balanced),
        "el_tsbpo/policy_weight/bare_continuations": float(bare_continuations),
        "el_tsbpo/policy_weight/branch_continuations": float(branch_continuations),
        "el_tsbpo/policy_weight/raw_bare_row_fraction": (
            float(bare.sum().item()) / max(1.0, float(active.sum().item()))
        ),
        "el_tsbpo/policy_weight/raw_branch_row_fraction": (
            float(branch_active.sum().item()) / max(1.0, float(active.sum().item()))
        ),
        "el_tsbpo/policy_weight/raw_bare_token_fraction": (
            bare_tokens / total_tokens if total_tokens > 0.0 else 0.0
        ),
        "el_tsbpo/policy_weight/raw_branch_token_fraction": (
            branch_tokens / total_tokens if total_tokens > 0.0 else 0.0
        ),
        "el_tsbpo/policy_weight/bare_token_mass_fraction": (
            bare_mass / total_weighted_mass if total_weighted_mass > 0.0 else 0.0
        ),
        "el_tsbpo/policy_weight/branch_token_mass_fraction": (
            branch_mass / total_weighted_mass if total_weighted_mass > 0.0 else 0.0
        ),
        "el_tsbpo/policy_weight/bare_row_mass_fraction": (
            bare_row_mass / total_weighted_row_mass
            if total_weighted_row_mass > 0.0 else 0.0
        ),
        "el_tsbpo/policy_weight/branch_row_mass_fraction": (
            branch_row_mass / total_weighted_row_mass
            if total_weighted_row_mass > 0.0 else 0.0
        ),
        "el_tsbpo/policy_weight/row_weight_mean": (
            float(active_weights.mean().item()) if active_weights.numel() else 0.0
        ),
        "el_tsbpo/policy_weight/row_weight_min": (
            float(active_weights.min().item()) if active_weights.numel() else 0.0
        ),
        "el_tsbpo/policy_weight/row_weight_max": (
            float(active_weights.max().item()) if active_weights.numel() else 0.0
        ),
    }
    continuation_masses_by_key: dict[tuple[bool, str], float] = defaultdict(float)
    policy_weight_np = policy_weight.detach().cpu().numpy()
    for index in range(bsz):
        if bool(active_np[index]):
            continuation_masses_by_key[
                (bool(branch_np[index]), trajectory_ids[index])
            ] += float(policy_weight_np[index])
    for source_name, source_value in (("bare", False), ("branch", True)):
        continuation_masses = [
            value for (source_key, _), value in continuation_masses_by_key.items()
            if source_key == source_value
        ]
        masses = np.asarray(continuation_masses, dtype=np.float64)
        mass_mean = float(masses.mean()) if masses.size else 0.0
        metrics[f"el_tsbpo/policy_weight/{source_name}_continuation_mass_mean"] = mass_mean
        metrics[f"el_tsbpo/policy_weight/{source_name}_continuation_mass_cv"] = float(
            masses.std() / mass_mean if masses.size and mass_mean > 0.0 else 0.0
        )
    if "advantages" in data.batch and total_weighted_mass > 0.0:
        advantages = data.batch["advantages"].detach().float()
        weighted_mask = response_mask * policy_weight[:, None]
        metrics["el_tsbpo/policy_weight/weighted_advantage_mean"] = float(
            (advantages * weighted_mask).sum().item() / total_weighted_mass
        )
        for source, source_rows in (("bare", ~branch), ("branch", branch)):
            source_mask = weighted_mask * source_rows[:, None].to(weighted_mask.dtype)
            source_mass = float(source_mask.sum().item())
            metrics[f"el_tsbpo/policy_weight/{source}_advantage_mean"] = (
                float((advantages * source_mask).sum().item() / source_mass)
                if source_mass > 0.0 else 0.0
            )
        token_counts = response_mask.sum(dim=-1)
        row_advantages = (
            (advantages * response_mask).sum(dim=-1)
            / token_counts.clamp_min(1.0)
        )
        objective_mass = weighted_row_mass if row_balanced else weighted_token_mass
        if row_balanced:
            objective_values = row_advantages
        else:
            objective_values = (
                (advantages * response_mask).sum(dim=-1)
                / token_counts.clamp_min(1.0)
            )
        objective_total = float(objective_mass.sum().item())
        metrics["el_tsbpo/policy_weight/weighted_objective_advantage_mean"] = (
            float((objective_values * objective_mass).sum().item() / objective_total)
            if objective_total > 0.0 else 0.0
        )
    return policy_weight, metrics


def compute_el_tsbpo_sampling_diagnostics(data: DataProto) -> Dict[str, float]:
    """Measure source imbalance and trajectory/state credit conflicts."""

    advantages = data.batch["advantages"].detach().float()
    response_mask = data.batch.get("response_mask")
    if response_mask is None:
        response_mask = compute_response_mask(data)
    response_mask = response_mask.detach().float()
    if "loss_mask" in data.batch and "responses" in data.batch:
        response_width = int(data.batch["responses"].shape[-1])
        response_mask = data.batch["loss_mask"][:, -response_width:].detach().float()
    sample_weight = data.batch.get(
        "el_tsbpo_sample_weight",
        torch.ones(len(data), device=advantages.device, dtype=advantages.dtype),
    ).detach().float().reshape(-1)
    active_rows = sample_weight > 0
    active_mask = response_mask * active_rows.to(response_mask.dtype)[:, None]
    branch_mask_np = _el_tsbpo_branch_mask(data)
    branch_mask = torch.as_tensor(
        branch_mask_np, device=advantages.device, dtype=torch.bool
    )
    lengths = active_mask.sum(dim=-1)
    active_rows = active_rows & (lengths > 0)

    def as_row_value(tensor: torch.Tensor) -> torch.Tensor:
        value = tensor.detach().float()
        return (value * active_mask).sum(dim=-1) / lengths.clamp_min(1.0)

    row_advantage = as_row_value(advantages)
    trajectory_advantages = data.batch.get("el_tsbpo_trajectory_advantages")
    raw_trajectory_advantages = data.batch.get(
        "el_tsbpo_raw_trajectory_advantages"
    )
    state_advantages = data.batch.get(
        "el_tsbpo_state_advantages",
        data.batch.get("el_tsbpo_sibling_advantages"),
    )
    component_rows = {"combined": row_advantage}
    if trajectory_advantages is not None:
        component_rows["trajectory"] = as_row_value(trajectory_advantages)
    if raw_trajectory_advantages is not None:
        component_rows["raw_trajectory"] = as_row_value(
            raw_trajectory_advantages
        )
    if state_advantages is not None:
        component_rows["state"] = as_row_value(state_advantages)

    policy_weight = data.batch.get(
        "el_tsbpo_policy_weight",
        torch.ones(len(data), device=advantages.device, dtype=advantages.dtype),
    ).detach().float().reshape(-1)
    weighted_rows = policy_weight * active_rows.to(policy_weight.dtype)
    weighted_row_total = float(weighted_rows.sum().item())
    active_row_count = float(active_rows.sum().item())
    metrics: Dict[str, float] = {
        "el_tsbpo/sampling/branch_row_fraction": float(
            (branch_mask & active_rows).sum().item() / max(1.0, active_row_count)
        ),
        "el_tsbpo/sampling/branch_token_fraction": float(
            (active_mask * branch_mask[:, None]).sum().item()
            / max(1.0, float(active_mask.sum().item()))
        ),
    }
    terminal_rewards = data.non_tensor_batch.get("el_tsbpo_terminal_reward")
    terminal_flags = data.non_tensor_batch.get("el_tsbpo_node_is_terminal")
    if terminal_rewards is not None and terminal_flags is not None:
        terminal_rewards = np.asarray(terminal_rewards, dtype=np.float32).reshape(-1)
        terminal_flags = np.asarray(terminal_flags, dtype=bool).reshape(-1)
        if len(terminal_rewards) == len(data) and len(terminal_flags) == len(data):
            # ``terminal_*`` metadata is NumPy-backed while ``active_rows``
            # above is a torch mask. Keep this diagnostic-only intersection in
            # NumPy; mixing the two mask types raises before actor update.
            active_rows_np = active_rows.detach().cpu().numpy().astype(bool)
            for source, source_mask in (
                ("bare", ~branch_mask_np.astype(bool)),
                ("branch", branch_mask_np.astype(bool)),
            ):
                source_terminals = terminal_flags & source_mask & active_rows_np
                values = terminal_rewards[source_terminals]
                prefix = f"el_tsbpo/sampling/{source}"
                metrics[f"{prefix}_terminal_rows"] = float(values.size)
                metrics[f"{prefix}_terminal_success_rate"] = float(
                    (values > 0.0).mean() if values.size else 0.0
                )
                metrics[f"{prefix}_terminal_reward_mean"] = float(
                    values.mean() if values.size else 0.0
                )
                metrics[f"{prefix}_terminal_reward_bad_scale"] = float(
                    np.logical_and(
                        np.abs(values) > 1.0e-5,
                        np.abs(values - 10.0) > 1.0e-5,
                    ).sum()
                    if values.size
                    else 0.0
                )
    source_component_means: dict[str, dict[str, float]] = {}

    for source, source_mask in (("bare", ~branch_mask), ("branch", branch_mask)):
        row_mask = source_mask & active_rows
        token_mask = active_mask * row_mask[:, None].to(active_mask.dtype)
        row_count = int(row_mask.sum().item())
        token_count = float(token_mask.sum().item())
        prefix = f"el_tsbpo/sampling/{source}"
        metrics[f"{prefix}_rows"] = float(row_count)
        metrics[f"{prefix}_tokens"] = token_count
        metrics[f"{prefix}_policy_row_mass_fraction"] = float(
            weighted_rows[row_mask].sum().item() / max(1e-12, weighted_row_total)
        ) if weighted_row_total > 0.0 else 0.0
        source_component_means[source] = {}
        if row_count == 0 or token_count <= 0.0:
            for suffix in (
                "response_length_mean",
                "advantage_token_mean",
                "advantage_abs_token_mean",
                "advantage_positive_token_fraction",
                "advantage_negative_token_fraction",
                "row_advantage_mean",
                "length_advantage_covariance",
                "length_advantage_correlation",
                "trajectory_state_sign_conflict_fraction",
            ):
                metrics[f"{prefix}_{suffix}"] = 0.0
            for component in component_rows:
                for statistic in (
                    "row_mean", "abs_row_mean", "positive_row_fraction",
                    "negative_row_fraction", "policy_contribution",
                ):
                    metrics[f"{prefix}_{component}_{statistic}"] = 0.0
                source_component_means[source][component] = 0.0
            continue

        source_advantages = advantages[token_mask.bool()]
        source_lengths = lengths[row_mask]
        source_row_advantages = row_advantage[row_mask]
        centered_length = source_lengths - source_lengths.mean()
        centered_advantage = source_row_advantages - source_row_advantages.mean()
        covariance = (centered_length * centered_advantage).mean()
        correlation_denominator = (
            centered_length.square().mean().sqrt()
            * centered_advantage.square().mean().sqrt()
        )
        correlation = (
            covariance / correlation_denominator
            if float(correlation_denominator.item()) > 0.0
            else covariance.new_zeros(())
        )
        metrics.update({
            f"{prefix}_response_length_mean": float(source_lengths.mean().item()),
            f"{prefix}_advantage_token_mean": float(source_advantages.mean().item()),
            f"{prefix}_advantage_abs_token_mean": float(source_advantages.abs().mean().item()),
            f"{prefix}_advantage_positive_token_fraction": float((source_advantages > 0).float().mean().item()),
            f"{prefix}_advantage_negative_token_fraction": float((source_advantages < 0).float().mean().item()),
            f"{prefix}_row_advantage_mean": float(source_row_advantages.mean().item()),
            f"{prefix}_length_advantage_covariance": float(covariance.item()),
            f"{prefix}_length_advantage_correlation": float(correlation.item()),
        })
        for component, row_values in component_rows.items():
            source_values = row_values[row_mask]
            component_mean = float(source_values.mean().item())
            source_component_means[source][component] = component_mean
            metrics.update({
                f"{prefix}_{component}_row_mean": component_mean,
                f"{prefix}_{component}_abs_row_mean": float(
                    source_values.abs().mean().item()
                ),
                f"{prefix}_{component}_positive_row_fraction": float(
                    (source_values > 0).float().mean().item()
                ),
                f"{prefix}_{component}_negative_row_fraction": float(
                    (source_values < 0).float().mean().item()
                ),
                f"{prefix}_{component}_policy_contribution": float(
                    (row_values[row_mask] * policy_weight[row_mask]).sum().item()
                    / max(1e-12, weighted_row_total)
                ) if weighted_row_total > 0.0 else 0.0,
            })
        if "trajectory" in component_rows and "state" in component_rows:
            trajectory_values = component_rows["trajectory"][row_mask]
            state_values = component_rows["state"][row_mask]
            metrics[f"{prefix}_trajectory_state_sign_conflict_fraction"] = float(
                ((trajectory_values * state_values) < 0).float().mean().item()
            )
        else:
            metrics[f"{prefix}_trajectory_state_sign_conflict_fraction"] = 0.0

    for component in component_rows:
        bare_mean = source_component_means.get("bare", {}).get(component, 0.0)
        branch_mean = source_component_means.get("branch", {}).get(component, 0.0)
        metrics[f"el_tsbpo/sampling/bare_branch_{component}_mean_gap"] = float(
            bare_mean - branch_mean
        )
        metrics[f"el_tsbpo/sampling/bare_branch_{component}_mean_sign_conflict"] = float(
            bare_mean * branch_mean < 0.0
        )
    return metrics


def _el_tsbpo_structure_id(prefix: str, value) -> str:
    """Return a stable opaque ID without persisting observations or prompts."""

    if value is None:
        return ""
    text = str(value)
    if not text:
        return ""
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return f"{prefix}_{digest[:16]}"


def dump_el_tsbpo_sampling_structure(
    data: DataProto,
    global_step: int,
    save_dir: str,
    *,
    advantage_mode: str,
    gigpo_enable_similarity: bool = False,
    gigpo_similarity_thresh: float = 0.95,
) -> str | None:
    """Persist the sampled forest and the exact credit terms used this step.

    This is deliberately a structural audit artifact: node/edge/state IDs are
    opaque hashes and no prompt, observation, action text, or token sequence is
    written.  Scalars are response-mask means, matching the per-turn values
    seen by the policy loss after broadcasting.
    """

    if not hasattr(data, "batch") or not hasattr(data, "non_tensor_batch"):
        return None
    tb = data.batch
    ntb = data.non_tensor_batch
    advantages = tb.get("advantages")
    if advantages is None:
        return None
    bsz = int(advantages.shape[0])
    response_mask = tb.get("response_mask")
    if response_mask is None:
        response_mask = compute_response_mask(data)
    response_mask = response_mask.detach().float()
    if "loss_mask" in tb and "responses" in tb:
        response_width = int(tb["responses"].shape[-1])
        response_mask = tb["loss_mask"][:, -response_width:].detach().float()
    lengths = response_mask.sum(dim=-1).clamp_min(1.0)

    def row_values(name: str, fallback=None) -> torch.Tensor:
        value = tb.get(name, fallback)
        if value is None:
            return torch.zeros(bsz, device=advantages.device, dtype=torch.float32)
        value = value.detach().float()
        if value.ndim == 1:
            return value.reshape(-1)
        return (value * response_mask).sum(dim=-1) / lengths

    raw_traj = row_values(
        "el_tsbpo_raw_trajectory_advantages_unweighted",
        tb.get("el_tsbpo_raw_trajectory_advantages"),
    )
    weighted_traj = row_values("el_tsbpo_trajectory_advantages")
    raw_step = row_values(
        "el_tsbpo_raw_sibling_advantages",
        tb.get("el_tsbpo_sibling_advantages"),
    )
    weighted_step = row_values("el_tsbpo_sibling_advantages")
    final_adv = row_values("advantages")
    tree_reward = row_values("el_tsbpo_tree_rewards")
    discounted_reward = row_values("el_tsbpo_discounted_tree_rewards")

    def row_tensor(name: str, default: float = 1.0) -> np.ndarray:
        value = tb.get(name)
        if value is None:
            return np.full(bsz, default, dtype=np.float32)
        return value.detach().float().reshape(-1).cpu().numpy()

    trajectory_coeff = row_tensor("el_tsbpo_trajectory_policy_weight")
    sibling_coeff = row_tensor("el_tsbpo_sibling_policy_weight")
    source_coeff = row_tensor("el_tsbpo_policy_weight")
    sample_weight = row_tensor("el_tsbpo_sample_weight")
    active = (sample_weight > 0.0) & (lengths.detach().cpu().numpy() > 0.0)

    def ntb_values(name: str, default):
        values = ntb.get(name)
        if values is None or len(values) != bsz:
            return [default for _ in range(bsz)]
        return list(values)

    task_values = ntb_values("uid", "")
    traj_values = ntb_values("traj_uid", "")
    parent_values = ntb_values("el_tsbpo_parent_node_id", "")
    node_values = ntb_values("el_tsbpo_node_id", "")
    state_values = ntb_values("anchor_obs", "")
    group_values = ntb_values("el_tsbpo_state_group_key", "")
    branch_values = ntb_values("el_tsbpo_branch_type", "factual")
    terminal_values = ntb_values("el_tsbpo_terminal_reward", 0.0)
    terminal_flags = ntb_values("el_tsbpo_node_is_terminal", False)

    # Reuse the exact GIGPO state aggregation for every active edge. The DPS
    # allocator group is only defined on selected anchors, so it is retained
    # separately below instead of being used as the actor state-group ID.
    gigpo_state_groups = ["" for _ in range(bsz)]
    active_rows = np.flatnonzero(active)
    if active_rows.size > 0 and all(
        isinstance(state_values[row], (str, bytes)) for row in active_rows
    ):
        gigpo_state_groups_active = core_gigpo.build_step_group(
            np.asarray([state_values[row] for row in active_rows], dtype=object),
            np.asarray([task_values[row] for row in active_rows], dtype=object),
            enable_similarity=bool(gigpo_enable_similarity),
            similarity_thresh=float(gigpo_similarity_thresh),
            summarize=False,
        )
        for position, row in enumerate(active_rows.tolist()):
            gigpo_state_groups[row] = str(gigpo_state_groups_active[position])

    # First collect the physical parent relation.  Following these pointers
    # groups branch continuations under the same sampled root tree even when
    # their own trajectory IDs differ from the factual rollout.
    parent_by_node: dict[str, str] = {}
    for row in range(bsz):
        node = str(node_values[row] or "")
        parent = str(parent_values[row] or "")
        if node:
            parent_by_node[node] = parent

    def root_for(node: str, task: str) -> str:
        current = node or f"task:{task}:root"
        seen: set[str] = set()
        while current in parent_by_node and current not in seen:
            seen.add(current)
            parent = parent_by_node.get(current, "")
            if not parent:
                break
            current = parent
        return current or f"task:{task}:root"

    trees: dict[str, dict] = {}
    node_records: dict[str, dict] = {}
    for row in range(bsz):
        if not bool(active[row]):
            continue
        task = str(task_values[row] or "")
        traj = str(traj_values[row] or "")
        parent = str(parent_values[row] or "")
        node = str(node_values[row] or f"row:{row}")
        root = root_for(node, task)
        tree_id = _el_tsbpo_structure_id("tree", root)
        state_id = _el_tsbpo_structure_id("state", f"{task}\0{state_values[row]}")
        group_key = str(gigpo_state_groups[row] or "")
        group_id = _el_tsbpo_structure_id("group", group_key) if group_key else ""
        allocator_group_key = str(group_values[row] or "")
        allocator_group_id = (
            _el_tsbpo_structure_id("allocator_group", allocator_group_key)
            if allocator_group_key
            else ""
        )
        node_id = _el_tsbpo_structure_id("node", node)
        parent_id = _el_tsbpo_structure_id("node", parent) if parent else ""
        if node_id not in node_records:
            node_records[node_id] = {
                "node_id": node_id,
                "parent_node_id": parent_id,
                "state_id": state_id,
                "terminal": bool(terminal_flags[row]),
            }
        tree = trees.setdefault(
            tree_id,
            {
                "tree_id": tree_id,
                "task_id": _el_tsbpo_structure_id("task", task),
                "root_node_id": _el_tsbpo_structure_id("node", root),
                "nodes": {},
                "edges": [],
            },
        )
        trajectory_raw = float(raw_traj[row].item())
        step_raw = float(raw_step[row].item())
        edge = {
            "parent_node_id": parent_id,
            "node_id": node_id,
            "trajectory_id": _el_tsbpo_structure_id("traj", traj),
            "branch_type": str(branch_values[row] or "factual"),
            "state_id": state_id,
            "state_group_id": group_id,
            "allocator_state_group_id": allocator_group_id,
            "reward": float(tree_reward[row].item()),
            "discounted_reward": float(discounted_reward[row].item()),
            "terminal_reward": float(terminal_values[row] or 0.0),
            "terminal": bool(terminal_flags[row]),
            "trajectory_advantage_raw": trajectory_raw,
            "trajectory_coefficient": float(trajectory_coeff[row]),
            "trajectory_advantage_weighted": float(weighted_traj[row].item()),
            "step_advantage_raw": step_raw,
            "step_coefficient": float(sibling_coeff[row]),
            "step_advantage_weighted": float(weighted_step[row].item()),
            "source_coefficient": float(source_coeff[row]),
            "final_advantage_before_source": float(final_adv[row].item()),
            "final_advantage_after_source": float(
                final_adv[row].item() * source_coeff[row]
            ),
            "token_count": int(round(float(lengths[row].item()))),
        }
        tree["edges"].append(edge)

    # Add all referenced parents as nodes. A synthetic root has no persisted
    # environment state; its state_id is intentionally empty rather than a
    # prompt-derived value.
    for tree in trees.values():
        for edge in tree["edges"]:
            node_id = edge["node_id"]
            if node_id in node_records:
                tree["nodes"][node_id] = node_records[node_id]
            parent_id = edge["parent_node_id"]
            if parent_id and parent_id not in tree["nodes"]:
                tree["nodes"][parent_id] = {
                    "node_id": parent_id,
                    "parent_node_id": "",
                    "state_id": "",
                    "terminal": False,
                    "synthetic": True,
                }
        tree["edges"].sort(key=lambda value: (value["parent_node_id"], value["node_id"]))

    payload = {
        "schema_version": 1,
        "step": int(global_step),
        "advantage_mode": str(advantage_mode),
        "summary": {
            "rows": int(sum(len(tree["edges"]) for tree in trees.values())),
            "trees": int(len(trees)),
            "nodes": int(sum(len(tree["nodes"]) for tree in trees.values())),
            "edges": int(sum(len(tree["edges"]) for tree in trees.values())),
            "bare_edges": int(sum(
                1 for tree in trees.values() for edge in tree["edges"]
                if edge["branch_type"] == "factual"
            )),
            "branch_edges": int(sum(
                1 for tree in trees.values() for edge in tree["edges"]
                if edge["branch_type"] != "factual"
            )),
            "source_coefficient_min": float(source_coeff[active].min()) if active.any() else 1.0,
            "source_coefficient_max": float(source_coeff[active].max()) if active.any() else 1.0,
        },
        "trees": trees,
    }
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"el_tsbpo_structure_step{int(global_step):04d}.json")
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, separators=(",", ":"))
    os.replace(tmp_path, path)
    return path


def _context_mismatch_masks(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    responses: torch.Tensor,
    context_token_rows,
    context_position_rows=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-row token and position mismatch masks."""

    batch_size = int(input_ids.shape[0])
    token_mask = np.zeros(batch_size, dtype=bool)
    position_mask = np.zeros(batch_size, dtype=bool)
    if context_token_rows is None:
        return token_mask, position_mask
    if len(context_token_rows) != batch_size:
        token_mask[:] = True
        return token_mask, position_mask

    response_width = int(responses.shape[-1])
    prompt_slice = slice(None, -response_width) if response_width else slice(None)
    for row_idx, expected in enumerate(context_token_rows):
        expected_values = [] if expected is None else expected
        expected_ids = [int(token) for token in expected_values]
        prompt_ids = input_ids[row_idx, prompt_slice]
        prompt_mask = attention_mask[row_idx, prompt_slice].bool()
        actual_ids = prompt_ids[prompt_mask].detach().cpu().tolist()
        token_mask[row_idx] = actual_ids != expected_ids
        if context_position_rows is not None and context_position_rows[row_idx] is not None:
            expected_positions = np.asarray(context_position_rows[row_idx], dtype=np.int64)
            prompt_positions = position_ids[row_idx, ..., prompt_slice]
            actual_positions = prompt_positions[..., prompt_mask].detach().cpu().numpy()
            position_mask[row_idx] = not np.array_equal(actual_positions, expected_positions)
    return token_mask, position_mask


def context_mismatch_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    responses: torch.Tensor,
    context_token_rows,
    context_position_rows=None,
) -> np.ndarray:
    """Return rows whose saved rollout context cannot be trusted exactly."""

    token_mask, position_mask = _context_mismatch_masks(
        input_ids,
        attention_mask,
        position_ids,
        responses,
        context_token_rows,
        context_position_rows,
    )
    return np.logical_or(token_mask, position_mask)


def count_context_mismatches(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    responses: torch.Tensor,
    context_token_rows,
    context_position_rows=None,
) -> dict[str, int]:
    """Compare saved rollout context with the final prompt prefix.

    Rollout engines may strip left padding and later concatenate responses.
    This helper compares only the prompt prefix that reaches actor and
    reference log-prob recomputation, including multimodal position IDs. It
    returns counts instead of raising so the driver can log and quarantine a
    mismatched batch.
    """

    token_mask, position_mask = _context_mismatch_masks(
        input_ids,
        attention_mask,
        position_ids,
        responses,
        context_token_rows,
        context_position_rows,
    )
    return {
        "token_mismatches": int(token_mask.sum()),
        "position_mismatches": int(position_mask.sum()),
        "mismatch_rows": int(np.logical_or(token_mask, position_mask).sum()),
    }


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, step_advantage_w=1.0, gigpo_mode="mean_norm", gigpo_enable_similarity=False, gigpo_similarity_thresh=0.95, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GiGPO:
        trajectory_success = data.non_tensor_batch.get('trajectory_success')
        if trajectory_success is not None:
            data.meta_info = dict(data.meta_info or {})
            data.meta_info.setdefault('gigpo_metrics', {})[
                'sampling/effective_ratio'
            ] = core_gigpo.compute_effective_ratio(
                    index=data.non_tensor_batch['uid'],
                    traj_index=data.non_tensor_batch['traj_uid'],
                    trajectory_success=trajectory_success,
            )
        advantages, returns = core_gigpo.compute_gigpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'], # for episode group reward computing
            step_rewards=data.batch['step_rewards'], # for step group reward computing
            response_mask=data.batch['response_mask'],
            anchor_obs=data.non_tensor_batch['anchor_obs'],
            index=data.non_tensor_batch['uid'],
            traj_index=data.non_tensor_batch['traj_uid'],
            step_advantage_w=step_advantage_w,
            mode=gigpo_mode,
            enable_similarity=gigpo_enable_similarity,
            similarity_thresh=gigpo_similarity_thresh,
            )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.EL_TSBPO:
        import numpy as _np

        bsz = data.batch.batch_size[0]
        el_tsbpo_advantage_mode = str(
            kwargs.get('el_tsbpo_advantage_mode', 'tree')
        ).lower()
        trajectory_success = data.non_tensor_batch.get('trajectory_success')
        terminal_reward = _el_tsbpo_terminal_returns(data)
        if terminal_reward is not None:
            # Persist the factual terminal value. All output costs stay local
            # to their producing turn.
            data.non_tensor_batch['el_tsbpo_terminal_reward'] = terminal_reward
        advantages, returns, components = core_el_tsbpo.compute_el_tsbpo_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'],
            traj_index=data.non_tensor_batch['traj_uid'],
            bare_rollout_id=data.non_tensor_batch.get('el_tsbpo_bare_rollout_id'),
            trajectory_success=trajectory_success,
            sample_weight=data.batch.get('el_tsbpo_sample_weight'),
            parent_node_id=data.non_tensor_batch.get('el_tsbpo_parent_node_id'),
            node_id=data.non_tensor_batch.get('el_tsbpo_node_id'),
            terminal_reward=(
                terminal_reward
                if terminal_reward is not None
                else data.non_tensor_batch.get('trajectory_success')
            ),
            node_is_terminal=data.non_tensor_batch.get('el_tsbpo_node_is_terminal'),
            trajectory_advantage_weight=kwargs.get(
                'el_tsbpo_trajectory_advantage_weight', 1.0
            ),
            gamma=gamma,
            epsilon=kwargs.get('el_tsbpo_epsilon', 1e-6),
            advantage_mode=el_tsbpo_advantage_mode,
            gigpo_step_rewards=data.batch.get('step_rewards'),
            gigpo_anchor_obs=data.non_tensor_batch.get('anchor_obs'),
            gigpo_action_text=data.non_tensor_batch.get('action_text'),
            gigpo_step_advantage_w=kwargs.get(
                'el_tsbpo_gigpo_step_advantage_w', 1.0
            ),
            gigpo_mode=kwargs.get('el_tsbpo_gigpo_mode', 'mean_norm'),
            gigpo_enable_similarity=kwargs.get(
                'el_tsbpo_gigpo_enable_similarity', False
            ),
            gigpo_similarity_thresh=kwargs.get(
                'el_tsbpo_gigpo_similarity_thresh', 0.95
            ),
            graph_gigpo_use_tree_discounted_step_return=kwargs.get(
                'el_tsbpo_graph_gigpo_use_tree_discounted_step_return', False
            ),
            graph_gigpo_segment_balance=kwargs.get(
                'el_tsbpo_graph_gigpo_segment_balance', False
            ),
            graph_gigpo_policy_loss_agg_mode=kwargs.get(
                'el_tsbpo_graph_gigpo_policy_loss_agg_mode',
                'seq-mean-token-sum',
            ),
            graph_gigpo_segment_weight_cap=kwargs.get(
                'el_tsbpo_graph_gigpo_segment_weight_cap', 6.0
            ),
            # This compatibility argument represents the total local edge
            # cost; backing up environment outcomes must exclude both costs.
            invalid_action_penalty=data.batch.get(
                'turn_output_penalty', data.batch.get('invalid_action_penalty')
            ),
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        # Detached diagnostics/label inputs; the actor consumes only the sum.
        data.batch['el_tsbpo_trajectory_advantages'] = components['trajectory']
        data.batch['el_tsbpo_raw_trajectory_advantages'] = components.get(
            'raw_trajectory', components['trajectory']
        )
        data.batch['el_tsbpo_raw_trajectory_advantages_unweighted'] = components.get(
            'raw_trajectory_unweighted',
            data.batch['el_tsbpo_raw_trajectory_advantages'],
        )
        data.batch['el_tsbpo_trajectory_policy_weight'] = components.get(
            'trajectory_policy_weight',
            torch.ones(bsz, device=advantages.device, dtype=advantages.dtype),
        )
        data.batch['el_tsbpo_trajectory_segment_group_id'] = components.get(
            'trajectory_segment_group_id',
            torch.full((bsz,), -1, device=advantages.device, dtype=torch.long),
        )
        data.batch['el_tsbpo_trajectory_segment_id'] = components.get(
            'trajectory_segment_id',
            torch.full((bsz,), -1, device=advantages.device, dtype=torch.long),
        )
        data.batch['el_tsbpo_trajectory_segment_turn_count'] = components.get(
            'trajectory_segment_turn_count',
            torch.ones(bsz, device=advantages.device, dtype=advantages.dtype),
        )
        data.batch['el_tsbpo_trajectory_segment_child_count'] = components.get(
            'trajectory_segment_child_count',
            torch.ones(bsz, device=advantages.device, dtype=advantages.dtype),
        )
        data.batch['el_tsbpo_sibling_advantages'] = components['sibling']
        data.batch['el_tsbpo_raw_sibling_advantages'] = components.get(
            'raw_sibling', components['sibling']
        )
        data.batch['el_tsbpo_sibling_policy_weight'] = components.get(
            'sibling_policy_weight',
            torch.ones(bsz, device=advantages.device, dtype=advantages.dtype),
        )
        data.batch['el_tsbpo_global_advantages'] = components.get(
            'global', torch.zeros_like(components['trajectory'])
        )
        data.batch['el_tsbpo_tree_prefix_advantages'] = components.get(
            'tree_prefix', components['trajectory']
        )
        data.batch['el_tsbpo_state_advantages'] = components.get(
            'sibling', torch.zeros_like(components['trajectory'])
        )
        data.batch['el_tsbpo_tree_rewards'] = components['tree_reward']
        data.batch['el_tsbpo_discounted_tree_rewards'] = components[
            'discounted_tree_reward'
        ]
        data.batch['el_tsbpo_rollout_weight'] = components['rollout_weight']
        # Keep every valid rollout in the complete batch. A homogeneous or
        # unary sibling group has zero local advantage, but it is not rejected
        # or removed from predictor/calibration data; the actor update later
        # selects only nonzero effective edges for actual PPO computation.
        data.batch['el_tsbpo_sample_weight'] = components['rollout_weight']
        data.batch['el_tsbpo_effective_edge_weight'] = components['effective_edge_weight']
        if kwargs.get('el_tsbpo_capture_turn_diagnostics', False) and 'trajectory_segment_return' in components:
            data.batch['el_tsbpo_trajectory_segment_returns'] = components['trajectory_segment_return']
        if 'sibling_effective_edge_weight' in components:
            data.batch['el_tsbpo_sibling_effective_edge_weight'] = components[
                'sibling_effective_edge_weight'
            ]
        tree_diagnostics = components.get('tree_diagnostics', {})
        if isinstance(tree_diagnostics, dict):
            data.meta_info.setdefault('el_tsbpo_metrics', {}).update({
                f'el_tsbpo/tree/{key}': float(value)
                for key, value in tree_diagnostics.items()
                if isinstance(value, (int, float, np.integer, np.floating))
            })
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


@contextmanager
def _persistent_rollout_session(worker_group, rollout_config):
    """Keep the rollout engine awake for one agent-environment loop.

    This lets vLLM reuse prefix KV across sibling/turn generation calls while
    ensuring the cache is reset when the loop ends, before actor weights can
    change. Worker backends without the session API retain their old behavior.
    """
    enabled = bool(rollout_config.get("enable_prefix_caching", True)) and not bool(
        rollout_config.get("free_cache_engine", True)
    )
    begin = getattr(worker_group, "begin_rollout_session", None)
    end = getattr(worker_group, "end_rollout_session", None)
    active = False
    if enabled and callable(begin) and callable(end):
        begin()
        active = True
    try:
        yield active
    finally:
        if active:
            end()


def _select_el_tsbpo_probe_rows(
    batch: DataProto,
    *,
    rollout_weight: torch.Tensor,
    expanded_edge_indices_observed: list[int],
    utility_indices: list[int],
    round_index: int,
    probe_task_count: int,
    probe_max_rows: int = 8,
    probe_max_tokens: int = 8192,
) -> dict[str, object]:
    """Select one shared actor-score probe from sampled rollout rows.

    This selection only affects predictor labels.  It never changes the
    actor-update mask and does not manufacture exploration samples.  Rows not
    already used by an expanded utility group are preferred, but when every
    trajectory was touched by an anchor the sampled utility rows remain a
    valid fallback.  An all-zero actor signal is reported as unavailable.
    """

    empty: dict[str, object] = {
        "tasks": (),
        "rows": (),
        "weights": (),
        "candidate_rows": 0,
        "selected_rows": 0,
        "selected_tokens": 0,
        "truncated": False,
        "nonzero_task_count": 0,
        "selected_task_count": 0,
        "reason": "unavailable",
    }
    task_ids = batch.non_tensor_batch.get(
        "uid", np.asarray([""] * len(batch), dtype=object)
    )
    advantages = batch.batch.get("advantages")
    response_mask = batch.batch.get("response_mask")
    if response_mask is None:
        response_mask = batch.batch.get("attention_mask")
        if (
            response_mask is not None
            and response_mask.ndim == 2
            and advantages is not None
            and response_mask.shape[-1] != advantages.shape[-1]
        ):
            response_mask = response_mask[:, -advantages.shape[-1] :]
    if advantages is None or response_mask is None:
        empty["reason"] = "missing_advantages"
        return empty
    probe_mask = batch.batch.get("loss_mask", response_mask)
    if probe_mask.ndim == 2 and probe_mask.shape[-1] != advantages.shape[-1]:
        probe_mask = probe_mask[:, -advantages.shape[-1] :]

    weighted_rows: dict[str, list[tuple[int, float]]] = defaultdict(list)
    utility_row_set = {int(row) for row in utility_indices}
    for row in range(len(batch)):
        try:
            active = float(rollout_weight[row].detach().item()) > 0.0
        except (AttributeError, IndexError, TypeError):
            active = float(rollout_weight[row]) > 0.0
        if not active:
            continue
        mask = probe_mask[row].detach().float()
        token_count = float(mask.sum().item())
        if token_count <= 0.0:
            continue
        # EL-TSBPO stores masked token advantages.  Multiplying by the mask
        # again also keeps this helper correct for direct unmasked callers.
        score = float(
            (advantages[row].detach().float() * mask).sum().item()
            / max(1.0, token_count)
        )
        if not np.isfinite(score) or abs(score) <= 1e-8:
            continue
        weighted_rows[str(task_ids[row])].append((row, score))

    candidate_rows = sum(len(rows) for rows in weighted_rows.values())
    empty["candidate_rows"] = int(candidate_rows)
    if not weighted_rows:
        empty["reason"] = "no_nonzero_advantage"
        return empty

    all_tasks = list(weighted_rows)
    selected_tasks = {
        str(task_ids[row])
        for row in expanded_edge_indices_observed
        if 0 <= int(row) < len(task_ids)
    }
    available_tasks = [task for task in all_tasks if task not in selected_tasks]
    task_pool = available_tasks or all_tasks

    task_stats: dict[str, tuple[float, float, int]] = {}
    for task in task_pool:
        values = np.asarray(
            [value for _, value in weighted_rows[task]], dtype=np.float64
        )
        task_stats[task] = (
            float(values.std(ddof=0)) if len(values) > 1 else 0.0,
            float(abs(values.mean())) if len(values) else 0.0,
            int(len(values)),
        )
    rotation = max(0, int(round_index)) % max(1, len(task_pool))
    tie_order = {
        task: (index - rotation) % max(1, len(task_pool))
        for index, task in enumerate(task_pool)
    }
    task_pool.sort(
        key=lambda task: (
            -task_stats[task][0],
            -task_stats[task][1],
            -task_stats[task][2],
            tie_order[task],
            task,
        )
    )
    requested = max(0, int(probe_task_count))
    if requested <= 0:
        empty["reason"] = "probe_task_count_zero"
        return empty
    selected = task_pool[: min(requested, len(task_pool))]

    rows_by_task: list[list[tuple[int, float]]] = []
    for task in selected:
        heldout = [
            item
            for item in weighted_rows[task]
            if item[0] not in utility_row_set
        ]
        rows_by_task.append(heldout or weighted_rows[task])
    rows_and_values: list[tuple[int, float]] = []
    # A task can contain every physical turn of all of its tree trajectories.
    # The probe is one shared gradient direction, so retaining every turn is
    # both unnecessary and unsafe for the actor's activation memory.  Allocate
    # the row budget deterministically across selected tasks, preserving at
    # least one representative row for each task whenever possible.
    max_rows = max(0, int(probe_max_rows))
    if max_rows > 0:
        task_count = max(1, len(rows_by_task))
        quotas = [
            max_rows // task_count + (1 if index < max_rows % task_count else 0)
            for index in range(task_count)
        ]

        def evenly_sample(
            items: list[tuple[int, float]], limit: int
        ) -> list[tuple[int, float]]:
            if limit <= 0 or not items:
                return []
            if len(items) <= limit:
                return list(items)
            if limit == 1:
                return [items[len(items) // 2]]
            positions = np.linspace(0, len(items) - 1, num=limit)
            indices = []
            seen = set()
            for position in positions:
                index = int(round(float(position)))
                if index not in seen:
                    indices.append(index)
                    seen.add(index)
            return [items[index] for index in indices]

        for task_rows, quota in zip(rows_by_task, quotas):
            rows_and_values.extend(evenly_sample(task_rows, quota))
    else:
        for task_rows in rows_by_task:
            rows_and_values.extend(task_rows)
    if not rows_and_values:
        empty["reason"] = "no_heldout_rows"
        return empty

    # A token cap is independent of the row cap because prompt lengths vary
    # substantially in ALFWorld.  Use the full attention-mask length as a
    # conservative activation-memory estimate and always retain the first
    # selected row even if it alone exceeds the configured cap.
    attention_mask = batch.batch.get("attention_mask")
    selected_before_token_cap = len(rows_and_values)
    if int(probe_max_tokens) > 0 and attention_mask is not None:
        token_limited_rows: list[tuple[int, float]] = []
        total_tokens = 0
        token_cap = int(probe_max_tokens)
        for item in rows_and_values:
            row = int(item[0])
            row_mask = attention_mask[row].detach()
            row_tokens = max(1, int(row_mask.to(torch.int64).sum().item()))
            if token_limited_rows and total_tokens + row_tokens > token_cap:
                continue
            token_limited_rows.append(item)
            total_tokens += row_tokens
        rows_and_values = token_limited_rows
    truncated = len(rows_and_values) < selected_before_token_cap
    if not rows_and_values:
        empty["reason"] = "probe_token_budget_zero"
        return empty

    selected_tokens = 0
    if attention_mask is not None:
        selected_tokens = sum(
            max(
                1,
                int(
                    attention_mask[int(row)]
                    .detach()
                    .to(torch.int64)
                    .sum()
                    .item()
                ),
            )
            for row, _ in rows_and_values
        )

    base = np.asarray([value for _, value in rows_and_values], dtype=np.float64)
    std = float(base.std(ddof=0)) if len(base) > 1 else 0.0
    if np.isfinite(std) and std > 1e-8:
        normalized = (base - float(base.mean())) / std
        if not np.any(
            np.isfinite(normalized) & (np.abs(normalized) > 1e-8)
        ):
            normalized = np.ones_like(base)
    else:
        # A one-row or constant-return probe still has a valid score
        # direction; do not turn it into an unavailable label.
        normalized = np.ones_like(base)
    rows: list[int] = []
    weights: list[float] = []
    for (row, _), weight in zip(rows_and_values, normalized):
        if np.isfinite(weight) and abs(float(weight)) > 1e-8:
            rows.append(int(row))
            weights.append(float(weight))
    if not rows:
        empty["reason"] = "normalized_signal_zero"
        return empty
    represented_tasks = tuple(
        dict.fromkeys(str(task_ids[row]) for row in rows)
    )
    return {
        "tasks": represented_tasks,
        "rows": tuple(rows),
        "weights": tuple(weights),
        "candidate_rows": int(candidate_rows),
        "selected_rows": int(len(rows)),
        "selected_tokens": int(selected_tokens),
        "truncated": bool(truncated or len(rows) < candidate_rows),
        "nonzero_task_count": int(len(weighted_rows)),
        "selected_task_count": int(len(represented_tasks)),
        "reason": "ok",
    }


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        traj_collector: TrajectoryCollector = None,
        envs=None,
        val_envs=None,
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.envs = envs
        self.val_envs = val_envs
        self.traj_collector = traj_collector

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()
        adv_estimator = config.algorithm.get("adv_estimator", "")
        if hasattr(adv_estimator, "value"):
            adv_estimator = adv_estimator.value
        if str(adv_estimator).lower() == AdvantageEstimator.EL_TSBPO.value:
            with open_dict(config.actor_rollout_ref.actor):
                policy_loss_cfg = config.actor_rollout_ref.actor.get("policy_loss")
                if policy_loss_cfg is None:
                    config.actor_rollout_ref.actor.policy_loss = {}
                    policy_loss_cfg = config.actor_rollout_ref.actor.policy_loss
                if policy_loss_cfg.get("loss_agg_mode") is None:
                    el_tsbpo_cfg = config.algorithm.get("el_tsbpo", {})
                    el_tsbpo_mode = str(
                        el_tsbpo_cfg.get("advantage_mode", "tree")
                    ).lower().replace("-", "_")
                    graph_gigpo_modes = {
                        "graph_gigpo",
                        "graph_gigpo_lite",
                        "state_action_gigpo",
                    }
                    gigpo_family = {
                        "gigpo",
                        "gigpo_style",
                        "tree_gigpo",
                        "tree_gigpo_s",
                        "tree_gigpo_simple",
                        "tree_gigpo_lite",
                        "graph_gigpo",
                        "graph_gigpo_lite",
                        "state_action_gigpo",
                    }
                    # A Graph-GiGPO action is a whole generated turn: its
                    # score sums token log-prob gradients. Do not divide by
                    # that sampled response's length. Ordinary GIGPO retains
                    # its existing token-mean baseline.
                    policy_loss_cfg["loss_agg_mode"] = (
                        "seq-mean-token-sum"
                        if el_tsbpo_mode in graph_gigpo_modes
                        else (
                            "token-mean"
                            if el_tsbpo_mode in gigpo_family
                            else "seq-mean-token-mean"
                        )
                    )
                else:
                    el_tsbpo_cfg = config.algorithm.get("el_tsbpo", {})
                    el_tsbpo_mode = str(
                        el_tsbpo_cfg.get("advantage_mode", "tree")
                    ).lower().replace("-", "_")
                    graph_gigpo_modes = {
                        "graph_gigpo",
                        "graph_gigpo_lite",
                        "state_action_gigpo",
                    }
                if (
                    el_tsbpo_mode in graph_gigpo_modes
                    and policy_loss_cfg.get("loss_agg_mode")
                    not in {"seq-mean-token-sum", "seq-mean-token-mean"}
                ):
                    raise ValueError(
                        "Configured Graph-GiGPO requires "
                        "actor.policy_loss.loss_agg_mode="
                        "'seq-mean-token-sum' (or explicit legacy 'seq-mean-token-mean')"
                    )
                if (
                    el_tsbpo_mode in graph_gigpo_modes
                    and bool(
                        el_tsbpo_cfg.get(
                            "graph_gigpo_source_balanced_policy", False
                        )
                    )
                ):
                    raise ValueError(
                        "Graph-GiGPO's unweighted mode is incompatible with "
                        "global bare/branch source policy weighting"
                    )
            print(
                "[EL-TSBPO] policy loss aggregation = "
                f"{config.actor_rollout_ref.actor.policy_loss.loss_agg_mode} "
                "(Graph-GiGPO tree policy weights are neutral; ordinary "
                "GIGPO keeps the stable token-mean baseline)",
                flush=True,
            )
        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GiGPO
        ]:
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == AdvantageEstimator.EL_TSBPO:
            # The independent Qwen3 predictor allocates rollout only. Actor
            # advantages are terminal-return statistics, so no PPO critic is used.
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def close_environments(self):
        """Release train/validation Ray actors owned by this trainer."""

        if self.traj_collector is not None:
            self.traj_collector.close_predictors()

        closed = set()
        for attribute in ("envs", "val_envs"):
            manager = getattr(self, attribute, None)
            if manager is None or id(manager) in closed:
                continue
            closed.add(id(manager))
            try:
                manager.close()
            except Exception as exc:
                print(f"Warning: failed to close {attribute}: {exc}")
            finally:
                setattr(self, attribute, None)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        estimator = config.algorithm.get("adv_estimator", "")
        estimator = estimator.value if hasattr(estimator, "value") else str(estimator)
        if estimator.lower() == AdvantageEstimator.EL_TSBPO.value:
            if config.algorithm.get("use_kl_in_reward", False):
                raise ValueError(
                    "EL-TSBPO uses the behavior rollout log-probability and "
                    "explicit terminal returns; algorithm.use_kl_in_reward is "
                    "unsupported because it requires a reference reward pass. "
                    "Use actor.use_kl_loss for optional policy KL instead."
                )
            tree_mode = str(
                config.algorithm.get("el_tsbpo", {}).get("tree_mode", "tree_rpo")
            ).lower().replace("-", "_")
            if tree_mode not in {
                "tree_rpo",
                "treerpo",
                "tree_grpo",
                "treegrpo",
            }:
                raise ValueError(
                    "EL-TSBPO tree_mode must be TreeRPO or the explicit "
                    "Tree-GRPO trajectory-plus-sibling variant; "
                    f"Received tree_mode={tree_mode!r}."
                )
            actor_strategy = str(
                config.actor_rollout_ref.actor.get("strategy", "fsdp")
            ).lower()
            if actor_strategy != "fsdp":
                raise ValueError(
                    "EL-TSBPO full-parameter gradient labels currently require "
                    "the FSDP actor worker (strategy=fsdp); Megatron and FSDP2 "
                    "backends must not silently train without predictor labels. "
                    f"Received actor.strategy={actor_strategy!r}."
                )
            gradient_objective = str(
                config.algorithm.get("el_tsbpo", {}).get(
                    "gradient_objective", "policy_only"
                )
            ).lower()
            if gradient_objective != "policy_only":
                raise ValueError(
                    "EL-TSBPO currently supports gradient_objective=policy_only; "
                    "training-mirror KL/clipping labels require a backend-specific "
                    "reference-gradient implementation."
                )
            policy_loss_cfg = config.actor_rollout_ref.actor.get("policy_loss") or {}
            policy_loss_mode = str(policy_loss_cfg.get("loss_mode", "vanilla")).lower()
            if policy_loss_mode != "vanilla":
                raise ValueError(
                    "EL-TSBPO policy-only gradient labels currently mirror the "
                    "vanilla PPO policy objective; unsupported actor.policy_loss "
                    f"loss_mode={policy_loss_mode!r}."
                )

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"
        policy_loss_cfg = config.actor_rollout_ref.actor.get("policy_loss") or {}
        policy_loss_agg_mode = policy_loss_cfg.get("loss_agg_mode")
        if policy_loss_agg_mode is not None:
            assert policy_loss_agg_mode in [
                "token-mean",
                "seq-mean-token-sum",
                "seq-mean-token-mean",
                "seq-mean-token-sum-norm",
            ], f"Invalid policy_loss.loss_agg_mode: {policy_loss_agg_mode}"
        if policy_loss_cfg.get("use_global_turn_normalization", False):
            if (policy_loss_agg_mode != "seq-mean-token-sum"
                    or config.actor_rollout_ref.actor.loss_agg_mode != "seq-mean-token-sum"
                    or policy_loss_cfg.get("loss_mode", "vanilla") != "vanilla"):
                raise ValueError("Global turn normalization requires vanilla PPO and token-sum PG/KL/entropy")

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            if config.algorithm.adv_estimator not in [
                AdvantageEstimator.EL_TSBPO,
            ]:
                assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
                assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        # Branch workers repeatedly bind TextWorld games during EL-TSBPO
        # rollout. Release that pool before allocating validation workers so
        # the two large actor sets never overlap in host memory. The pool is
        # recreated lazily on the next training rollout.
        if self.traj_collector is not None and hasattr(
            self.traj_collector, "release_branch_envs"
        ):
            if self.traj_collector.release_branch_envs():
                print("Released branch environments before validation")
        if self.val_envs is None:
            from agent_system.environments import make_envs

            _, self.val_envs = make_envs(
                self.config,
                create_train=False,
                create_val=True,
            )
            print("Created validation environments lazily")
        try:
            return self._validate_with_envs()
        finally:
            if bool(self.config.env.get("release_validation_after_use", False)):
                try:
                    self.val_envs.close()
                except Exception as exc:
                    print(f"Warning: failed to release validation environments: {exc}")
                finally:
                    self.val_envs = None
                    print("Released validation environments")

    def _validate_with_envs(self):
        reward_tensor_lst = []
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        success_rate_dict = {}

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "env_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # # pad to be divisible by dp_size
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # # unpad
            # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            ################ agent-environment loop ###############
            with _persistent_rollout_session(
                self.actor_rollout_wg,
                self.config.actor_rollout_ref.rollout,
            ):
                test_output_gen_batch = self.traj_collector.multi_turn_loop(
                    gen_batch=test_gen_batch,
                    actor_rollout_wg=self.actor_rollout_wg,
                    envs=self.val_envs,
                    is_train=False,
                )
            print('validation generation end')
            del test_batch
            test_batch = test_output_gen_batch
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            tool_calling_list.append(test_output_gen_batch.non_tensor_batch['tool_callings'])
            traj_uid_list.append(test_output_gen_batch.non_tensor_batch['traj_uid'])
            # success rate
            for k in test_batch.non_tensor_batch.keys():
                if 'success_rate' in k:
                    if k not in success_rate_dict:
                        success_rate_dict[k] = []
                    success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])
                    # all success_rate should be the same
                    for i in range(1, len(test_batch.non_tensor_batch[k])):
                        assert test_batch.non_tensor_batch[k][0] == test_batch.non_tensor_batch[k][i], f'not all success_rate are the same, 0: {test_batch.non_tensor_batch[k][0]}, {i}: {test_batch.non_tensor_batch[k][i]}'

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)
        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        # evaluate tool call based on data source
        # the values in tool_callings represent the tool call count for each trajectory; however, since the batch is expanded by step, we only need to take one value for each unique trajectories.
        data_source_tool_calling = {}
        unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
        unique_data_sources = data_sources[unique_idx]
        unique_tool_callings = tool_callings[unique_idx]

        for i in range(unique_tool_callings.shape[0]):
            data_source = unique_data_sources[i]
            if data_source not in data_source_tool_calling:
                data_source_tool_calling[data_source] = []
            data_source_tool_calling[data_source].append(unique_tool_callings[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/{data_source}/test_score'] = np.mean(rewards)

        for data_source, tool_calls in data_source_tool_calling.items():
            metric_dict[f'val/{data_source}/tool_call_count/mean'] = np.mean(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/max'] = np.max(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/min'] = np.min(tool_calls)

        for k, v in success_rate.items():
            metric_dict[f'val/{k}'] = v

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.EL_TSBPO:
            self.traj_collector.save_agent_dps_checkpoint(local_global_step_folder)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        if self.config.algorithm.adv_estimator == AdvantageEstimator.EL_TSBPO:
            self.traj_collector.load_agent_dps_checkpoint(global_step_folder, self.global_steps)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if self.config.algorithm.adv_estimator == AdvantageEstimator.EL_TSBPO:
                val_metrics = filter_el_tsbpo_quality_metrics(val_metrics)
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        turn_diagnostic_capture_count = 0
        turn_diagnostic_run_dir = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                diagnostic_cfg = self.config.algorithm.get('el_tsbpo', {}).get('turn_diagnostics', {})
                diagnostic_max_steps = int(diagnostic_cfg.get('max_steps', 5))
                capture_turn_diagnostics = bool(diagnostic_cfg.get('enabled', False)) and (
                    diagnostic_max_steps == 0 or turn_diagnostic_capture_count < diagnostic_max_steps
                )
                if capture_turn_diagnostics and build_pre_update_dump is None:
                    raise RuntimeError(
                        'algorithm.el_tsbpo.turn_diagnostics.enabled=true requires '
                        'verl/utils/turn_diagnostics.py, but that optional module is missing'
                    )
                if capture_turn_diagnostics and (
                    self.config.algorithm.adv_estimator != AdvantageEstimator.EL_TSBPO
                    or self.config.algorithm.el_tsbpo.advantage_mode != 'graph_gigpo'
                ):
                    raise ValueError('turn diagnostics currently requires EL-TSBPO graph_gigpo')
                turn_diagnostic_old_entropy = None
                turn_diagnostic_step_dir = None
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        # if not self.async_rollout_mode:
                        #     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        # else:
                        #     self.async_rollout_manager.wake_up()
                        #     gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        #     self.async_rollout_manager.sleep()

                        ################ agent-environment loop ###############
                        # Every natural/active sibling generated in this
                        # training iteration must carry one immutable policy
                        # identity.  Historical records from another actor
                        # update are not eligible for exact policy groups.
                        gen_batch.meta_info = dict(gen_batch.meta_info)
                        model_cfg = getattr(self.config.actor_rollout_ref, "model", {})
                        model_identity = str(getattr(model_cfg, "path", "unknown-model"))
                        gen_batch.meta_info["rollout_policy_version"] = (
                            f"{model_identity}|train_step_{self.global_steps}"
                        )
                        with _persistent_rollout_session(
                            self.actor_rollout_wg,
                            self.config.actor_rollout_ref.rollout,
                        ):
                            gen_batch_output = self.traj_collector.multi_turn_loop(
                                gen_batch=gen_batch,
                                actor_rollout_wg=self.actor_rollout_wg,
                                envs=self.envs,
                                is_train=True,
                            )
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # batch = batch.union(gen_batch_output)
                    del batch
                    batch = gen_batch_output

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.GiGPO:
                        step_rewards_tensor = core_gigpo.compute_step_discounted_returns(
                            batch=batch,
                            gamma=self.config.algorithm.gamma
                        )
                        batch.batch['step_rewards'] = step_rewards_tensor

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.EL_TSBPO:
                        el_tsbpo_advantage_mode = str(
                            self.config.algorithm.get('el_tsbpo', {})
                            .get('advantage_mode', 'tree')
                        ).lower().replace('-', '_')
                        if el_tsbpo_advantage_mode in {
                            'gigpo', 'gigpo_style', 'tree_gigpo', 'tree_gigpo_s',
                            'tree_gigpo_simple', 'tree_gigpo_lite',
                            'graph_gigpo', 'graph_gigpo_lite',
                            'state_action_gigpo'
                        }:
                            # GIGPO comparison, Tree-GiGPO, and Tree-GiGPO-Lite
                            # all consume the canonical discounted per-turn
                            # return. Allocation and predictor collection
                            # remain BU-specific.
                            step_rewards_tensor = core_gigpo.compute_step_discounted_returns(
                                batch=batch,
                                gamma=self.config.algorithm.gamma,
                            )
                            batch.batch['step_rewards'] = step_rewards_tensor

                    batch = adjust_batch(self.config, batch)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    is_el_tsbpo = (
                        self.config.algorithm.adv_estimator
                        == AdvantageEstimator.EL_TSBPO
                    )
                    # Ordinary GIGPO recomputes old log-probs with the actor
                    # backend on the exact tensors entering PPO. Use the same
                    # path for every estimator; rollout values are audit-only.
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        if capture_turn_diagnostics:
                            # This entropy is already computed by the normal
                            # old-log-prob pass; retain it only for this dump.
                            turn_diagnostic_old_entropy = entropys.detach()
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        if is_el_tsbpo:
                            # Entropy remains a per-token quality monitor even
                            # when the training objective averages by turn.
                            loss_agg_mode = "token-mean"
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if (
                            "rollout_log_probs" in batch.batch.keys()
                        ):
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            audit_mask = batch.batch.get("loss_mask", batch.batch["attention_mask"])
                            audit_mask = audit_mask[:, -response_length:]
                            clip_ratio = float(self.config.actor_rollout_ref.actor.get("clip_ratio", 0.2))
                            clip_ratio_low = self.config.actor_rollout_ref.actor.get("clip_ratio_low", None)
                            clip_ratio_high = self.config.actor_rollout_ref.actor.get("clip_ratio_high", None)
                            metrics.update(
                                compute_rollout_logprob_audit_metrics(
                                    rollout_log_probs=rollout_old_log_probs,
                                    actor_log_probs=actor_old_log_probs,
                                    response_mask=audit_mask,
                                    sample_weight=batch.batch.get("el_tsbpo_sample_weight"),
                                    clip_low=1.0 - float(
                                        clip_ratio if clip_ratio_low is None else clip_ratio_low
                                    ),
                                    clip_high=1.0 + float(
                                        clip_ratio if clip_ratio_high is None else clip_ratio_high
                                    ),
                                )
                            )
                            if is_el_tsbpo:
                                branch_rows = torch.as_tensor(
                                    _el_tsbpo_branch_mask(batch),
                                    device=audit_mask.device,
                                    dtype=torch.bool,
                                )
                                audit_sample_weight = batch.batch.get(
                                    "el_tsbpo_sample_weight",
                                    torch.ones(
                                        len(batch),
                                        device=audit_mask.device,
                                        dtype=torch.float32,
                                    ),
                                ).detach().float()
                                for source_name, source_rows in (
                                    ("bare", ~branch_rows),
                                    ("branch", branch_rows),
                                ):
                                    metrics.update(
                                        compute_rollout_logprob_audit_metrics(
                                            rollout_log_probs=rollout_old_log_probs,
                                            actor_log_probs=actor_old_log_probs,
                                            response_mask=audit_mask,
                                            sample_weight=(
                                                audit_sample_weight
                                                * source_rows.to(
                                                    audit_sample_weight.dtype
                                                )
                                            ),
                                            clip_low=1.0 - float(
                                                clip_ratio
                                                if clip_ratio_low is None
                                                else clip_ratio_low
                                            ),
                                            clip_high=1.0 + float(
                                                clip_ratio
                                                if clip_ratio_high is None
                                                else clip_ratio_high
                                            ),
                                            prefix=(
                                                "training/rollout_logprob_audit/"
                                                f"{source_name}"
                                            ),
                                        )
                                    )

                    if (
                        self.use_reference_policy
                        and self.config.algorithm.adv_estimator != AdvantageEstimator.EL_TSBPO
                    ):
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_invalid_action_penalty if available
                        actor_config = self.config.actor_rollout_ref.actor
                        use_invalid_penalty = actor_config.get('use_invalid_action_penalty', True)
                        use_truncation_penalty = actor_config.get('use_truncation_penalty', False)
                        if use_invalid_penalty or use_truncation_penalty:
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=(
                                    actor_config.invalid_action_penalty_coef if use_invalid_penalty else 0.0
                                ),
                                truncation_penalty_coef=(
                                    actor_config.get('truncation_penalty_coef', 0.0) if use_truncation_penalty else 0.0
                                ),
                            )
                            metrics.update(invalid_metrics)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        el_tsbpo_cfg = self.config.algorithm.get('el_tsbpo', {})
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity= self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                            el_tsbpo_epsilon=el_tsbpo_cfg.get('epsilon', 1e-6),
                            el_tsbpo_capture_turn_diagnostics=capture_turn_diagnostics,
                            el_tsbpo_tree_mode=el_tsbpo_cfg.get(
                                'tree_mode', 'tree_rpo'
                            ),
                            el_tsbpo_sibling_advantage_weight=el_tsbpo_cfg.get(
                                'sibling_advantage_weight', 1.0
                            ),
                            el_tsbpo_trajectory_advantage_weight=el_tsbpo_cfg.get(
                                'trajectory_advantage_weight', 1.0
                            ),
                            el_tsbpo_advantage_mode=el_tsbpo_cfg.get(
                                'advantage_mode', 'tree'
                            ),
                            el_tsbpo_gigpo_step_advantage_w=el_tsbpo_cfg.get(
                                'gigpo_step_advantage_w',
                                self.config.algorithm.gigpo.step_advantage_w,
                            ),
                            el_tsbpo_gigpo_mode=el_tsbpo_cfg.get(
                                'gigpo_mode', self.config.algorithm.gigpo.mode
                            ),
                            el_tsbpo_gigpo_enable_similarity=el_tsbpo_cfg.get(
                                'gigpo_enable_similarity',
                                self.config.algorithm.gigpo.enable_similarity,
                            ),
                            el_tsbpo_gigpo_similarity_thresh=el_tsbpo_cfg.get(
                                'gigpo_similarity_thresh',
                                self.config.algorithm.gigpo.similarity_thresh,
                            ),
                            el_tsbpo_graph_gigpo_use_tree_discounted_step_return=el_tsbpo_cfg.get(
                                'graph_gigpo_use_tree_discounted_step_return', False
                            ),
                            el_tsbpo_graph_gigpo_segment_balance=el_tsbpo_cfg.get(
                                'graph_gigpo_segment_balance', False
                            ),
                            el_tsbpo_graph_gigpo_policy_loss_agg_mode=(
                                (self.config.actor_rollout_ref.actor.get('policy_loss') or {}).get(
                                    'loss_agg_mode',
                                    self.config.actor_rollout_ref.actor.get(
                                        'loss_agg_mode', 'seq-mean-token-mean'
                                    ),
                                )
                            ),
                            el_tsbpo_graph_gigpo_segment_weight_cap=el_tsbpo_cfg.get(
                                'graph_gigpo_segment_weight_cap', 6.0
                            ),
                        )
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.EL_TSBPO:
                            metrics['el_tsbpo/reward/invalid_penalty_terminal_total'] = float(
                                batch.meta_info.get('el_tsbpo_invalid_penalty_total', 0.0)
                            )

                    if capture_turn_diagnostics:
                        # Immutable identity survives actor padding/rebalance/shuffle.
                        batch.batch['turn_diagnostic_row_id'] = torch.arange(
                            len(batch), device=batch.batch['advantages'].device, dtype=torch.long
                        )
                    actor_batch = batch
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.EL_TSBPO:
                        branch_mask = _el_tsbpo_branch_mask(batch)
                        batch.batch['el_tsbpo_is_branch'] = torch.as_tensor(
                            branch_mask,
                            device=batch.batch['advantages'].device,
                            dtype=torch.bool,
                        )
                        advantage_mode = str(
                            el_tsbpo_cfg.get('advantage_mode', 'tree')
                        ).lower().replace('-', '_')
                        source_balanced_policy = bool(
                            el_tsbpo_cfg.get('graph_gigpo_source_balanced_policy', False)
                        ) and advantage_mode in {
                            'graph_gigpo', 'graph_gigpo_lite', 'state_action_gigpo'
                        }
                        if source_balanced_policy:
                            actor_cfg = self.config.actor_rollout_ref.actor
                            policy_loss_cfg = actor_cfg.get('policy_loss') or {}
                            source_balance_loss_mode = policy_loss_cfg.get(
                                'loss_agg_mode'
                            )
                            if source_balance_loss_mode is None:
                                source_balance_loss_mode = actor_cfg.get(
                                    'loss_agg_mode', 'token-mean'
                                )
                            policy_weight, policy_weight_metrics = (
                                compute_el_tsbpo_source_policy_weights(
                                    batch,
                                    branch_share=float(
                                        el_tsbpo_cfg.get(
                                            'graph_gigpo_branch_policy_share', 0.5
                                        )
                                    ),
                                    loss_agg_mode=str(source_balance_loss_mode),
                                )
                            )
                            metrics.update(policy_weight_metrics)
                        else:
                            # ``el_tsbpo_sample_weight`` is already applied to
                            # the actor response mask. Keep the policy factor
                            # neutral here so disabling source balancing truly
                            # removes every extra bare/branch loss weight.
                            policy_weight = torch.ones_like(
                                batch.batch['el_tsbpo_sample_weight'],
                                dtype=torch.float32,
                            )
                            metrics['el_tsbpo/policy_weight/enabled'] = 0.0
                        batch.batch['el_tsbpo_policy_weight'] = policy_weight
                        metrics.update(compute_el_tsbpo_sampling_diagnostics(batch))
                        if bool(
                            el_tsbpo_cfg.get(
                                'sampling_structure_dump_enabled', False
                            )
                        ):
                            structure_dump_dir = str(
                                el_tsbpo_cfg.get(
                                    'sampling_structure_dump_dir',
                                    'el_tsbpo_sampling_diagnostics',
                                )
                            )
                            structure_dump_path = dump_el_tsbpo_sampling_structure(
                                batch,
                                self.global_steps,
                                (
                                    structure_dump_dir
                                    if os.path.isabs(structure_dump_dir)
                                    else os.path.join(os.getcwd(), structure_dump_dir)
                                ),
                                advantage_mode=advantage_mode,
                                gigpo_enable_similarity=bool(
                                    el_tsbpo_cfg.get(
                                        'gigpo_enable_similarity', False
                                    )
                                ),
                                gigpo_similarity_thresh=float(
                                    el_tsbpo_cfg.get(
                                        'gigpo_similarity_thresh', 0.95
                                    )
                                ),
                            )
                            if structure_dump_path:
                                metrics[
                                    'el_tsbpo/sampling/structure_dump_edges'
                                ] = float(
                                    sum(
                                        1
                                        for value in batch.non_tensor_batch.get(
                                            'el_tsbpo_node_id', []
                                        )
                                        if value
                                    )
                                )
                                metrics[
                                    'el_tsbpo/sampling/structure_dump_enabled'
                                ] = 1.0
                        agent_dps_mode = bool(
                            hasattr(self.traj_collector, 'is_agent_dps_enabled')
                            and self.traj_collector.is_agent_dps_enabled()
                        )
                        effective_weight = batch.batch['el_tsbpo_effective_edge_weight']
                        rollout_weight = batch.batch['el_tsbpo_rollout_weight']
                        # Predictor gradient sketches are only needed for
                        # selected anchor-local sibling evidence.  In
                        # Tree-GRPO mode actor credit may additionally cover
                        # every trajectory suffix; using that broader actor
                        # mask here would trigger a full backward for every
                        # unary row and defeat the detached label design.
                        utility_edge_weight = (
                            torch.zeros_like(effective_weight)
                            if agent_dps_mode
                            else batch.batch.get(
                                'el_tsbpo_sibling_effective_edge_weight',
                                effective_weight,
                            )
                        )
                        parent_nodes = batch.non_tensor_batch.get('el_tsbpo_parent_node_id')
                        memberships = batch.non_tensor_batch.get('el_tsbpo_group_memberships')
                        expanded_edge_indices_observed, expanded_anchors = expanded_edge_indices(
                            parent_nodes,
                            memberships,
                            # Rollout-valid edges identify which anchors were
                            # actually expanded this round.
                            rollout_weight,
                        )
                        utility_indices, _ = expanded_edge_indices(
                            parent_nodes,
                            memberships,
                            # Do not spend a full actor backward on a
                            # quarantined or zero-variance TreeRPO edge.
                            utility_edge_weight,
                        )
                        sketch_dim = max(1, int(el_tsbpo_cfg.get('sketch_dim', 128)))
                        all_sketches = torch.zeros(
                            len(batch), sketch_dim, dtype=torch.float32,
                            device=batch.batch['input_ids'].device,
                        )
                        batch.meta_info['el_tsbpo_gradient_labels_available'] = False
                        batch.meta_info['el_tsbpo_gradient_scope'] = 'all_trainable'
                        batch.meta_info['el_tsbpo_gradient_convention'] = 'ascent'
                        batch.meta_info['el_tsbpo_gradient_projection_id'] = (
                            f"countsketch:{int(el_tsbpo_cfg.get('sketch_seed', 17))}:"
                            f"{sketch_dim}"
                        )
                        actor_cfg = self.config.actor_rollout_ref.actor
                        policy_loss_cfg = actor_cfg.get('policy_loss') or {}
                        configured_gradient_loss_mode = policy_loss_cfg.get(
                            'loss_agg_mode'
                        )
                        if configured_gradient_loss_mode is None:
                            configured_gradient_loss_mode = actor_cfg.get(
                                'loss_agg_mode', 'token-mean'
                            )
                        gradient_loss_agg_mode = str(
                            configured_gradient_loss_mode
                        )
                        batch.meta_info[
                            'el_tsbpo_gradient_loss_agg_mode'
                        ] = gradient_loss_agg_mode
                        batch.meta_info['el_tsbpo_probe_gradient_sketch'] = None
                        batch.meta_info['el_tsbpo_expanded_anchor_ids'] = expanded_anchors
                        batch.meta_info['el_tsbpo_probe_tasks'] = ()
                        gradient_stats = {}
                        probe_valid = False
                        probe_selection = _select_el_tsbpo_probe_rows(
                            batch,
                            rollout_weight=rollout_weight,
                            expanded_edge_indices_observed=expanded_edge_indices_observed,
                            utility_indices=utility_indices,
                            round_index=self.global_steps,
                            probe_task_count=(
                                0
                                if agent_dps_mode
                                else int(el_tsbpo_cfg.get('probe_task_count', 1))
                            ),
                            probe_max_rows=int(
                                el_tsbpo_cfg.get('probe_max_rows', 8)
                            ),
                            probe_max_tokens=int(
                                el_tsbpo_cfg.get('probe_max_tokens', 8192)
                            ),
                        )
                        probe_tasks = tuple(
                            probe_selection.get('tasks', ())
                        )
                        probe_rows = list(probe_selection.get('rows', ()))
                        probe_base_weights = list(
                            probe_selection.get('weights', ())
                        )
                        batch.meta_info['el_tsbpo_probe_tasks'] = probe_tasks
                        metrics['el_tsbpo/utility/probe_candidate_rows'] = float(
                            probe_selection.get('candidate_rows', 0)
                        )
                        metrics['el_tsbpo/utility/probe_selected_rows'] = float(
                            probe_selection.get('selected_rows', 0)
                        )
                        metrics['el_tsbpo/utility/probe_selected_tokens'] = float(
                            probe_selection.get('selected_tokens', 0)
                        )
                        metrics['el_tsbpo/utility/probe_max_rows'] = float(
                            el_tsbpo_cfg.get('probe_max_rows', 8)
                        )
                        metrics['el_tsbpo/utility/probe_max_tokens'] = float(
                            el_tsbpo_cfg.get('probe_max_tokens', 8192)
                        )
                        metrics['el_tsbpo/utility/probe_rows_truncated'] = float(
                            bool(probe_selection.get('truncated', False))
                        )
                        metrics['el_tsbpo/utility/probe_nonzero_task_count'] = float(
                            probe_selection.get('nonzero_task_count', 0)
                        )
                        metrics['el_tsbpo/utility/probe_selected_task_count'] = float(
                            probe_selection.get('selected_task_count', 0)
                        )
                        metrics['el_tsbpo/utility/probe_unavailable'] = float(
                            probe_selection.get('reason') != 'ok'
                        )
                        reason_codes = {
                            'ok': 0.0,
                            'missing_advantages': 1.0,
                            'no_nonzero_advantage': 2.0,
                            'no_heldout_rows': 3.0,
                            'normalized_signal_zero': 4.0,
                            'probe_task_count_zero': 5.0,
                        }
                        metrics['el_tsbpo/utility/probe_selection_reason_code'] = (
                            reason_codes.get(
                                str(probe_selection.get('reason', 'unavailable')),
                                9.0,
                            )
                        )
                        score_sketch_count = 0
                        if utility_indices and probe_rows:
                            sketch_batch = batch.select_idxs(utility_indices)
                            # DP dispatch chunks contiguous rows. Repeat the
                            # selected sequence for every rank so each FSDP
                            # rank computes the same child-local backward and
                            # the collected first block preserves row order.
                            sketch_batch = sketch_batch.repeat(
                                repeat_times=max(1, int(self.actor_rollout_wg.world_size)),
                                interleave=False,
                            )
                            sketch_batch.meta_info['el_tsbpo_sketch_dim'] = sketch_dim
                            sketch_batch.meta_info['el_tsbpo_sketch_seed'] = int(
                                el_tsbpo_cfg.get('sketch_seed', 17)
                            )
                            sketch_batch.meta_info['el_tsbpo_gradient_mode'] = 'per_edge'
                            # Keep actor shards resident for the optional probe
                            # RPC and the immediately following actor update.
                            sketch_batch.meta_info['el_tsbpo_keep_model_on_gpu'] = True
                            with _timer('el_tsbpo_score_sketch', timing_raw):
                                if hasattr(
                                    self.actor_rollout_wg,
                                    'compute_el_tsbpo_score_sketch',
                                ):
                                    sketch_output = self.actor_rollout_wg.compute_el_tsbpo_score_sketch(
                                        sketch_batch
                                    )
                                    returned = sketch_output.batch[
                                        'el_tsbpo_score_sketches'
                                    ][: len(utility_indices)].to(all_sketches.device)
                                    all_sketches[utility_indices] = returned
                                    score_sketch_count = len(utility_indices)
                                    batch.meta_info['el_tsbpo_gradient_labels_available'] = True
                                    gradient_stats = dict(
                                        sketch_output.meta_info.get(
                                            'el_tsbpo_gradient_stats', {}
                                        )
                                    )
                                else:
                                    metrics[
                                        'el_tsbpo/utility/gradient_backend_unsupported'
                                    ] = 1.0
                        metrics['el_tsbpo/utility/n_expanded_anchor_edges'] = float(
                            len(expanded_edge_indices_observed)
                        )
                        metrics['el_tsbpo/utility/n_expanded_anchors'] = float(
                            len(expanded_anchors)
                        )
                        metrics['el_tsbpo/utility/gradient_trainable_numel'] = float(
                            gradient_stats.get('trainable_numel', 0.0)
                        )
                        metrics['el_tsbpo/utility/gradient_backward_count'] = float(
                            gradient_stats.get('backward_count', 0.0)
                        )
                        metrics['el_tsbpo/utility/gradient_sketch_norm_mean'] = float(
                            gradient_stats.get('sketch_norm_mean', 0.0)
                        )
                        metrics['el_tsbpo/utility/gradient_sketch_norm_max'] = float(
                            gradient_stats.get('sketch_norm_max', 0.0)
                        )
                        metrics['el_tsbpo/utility/gradient_objective_policy_only'] = 1.0

                        if probe_rows and batch.meta_info.get(
                            'el_tsbpo_gradient_labels_available', False
                        ):
                            probe_batch = batch.select_idxs(probe_rows)
                            probe_batch.batch[
                                'el_tsbpo_gradient_weight'
                            ] = torch.as_tensor(
                                probe_base_weights,
                                dtype=torch.float32,
                                device=probe_batch.batch['input_ids'].device,
                            )
                            probe_count = len(probe_rows)
                            probe_batch, probe_pad = pad_dataproto_to_divisor(
                                probe_batch, max(1, int(self.actor_rollout_wg.world_size))
                            )
                            if probe_pad:
                                probe_batch.batch['el_tsbpo_gradient_weight'][-probe_pad:] = 0.0
                            probe_batch.meta_info['el_tsbpo_sketch_dim'] = sketch_dim
                            probe_batch.meta_info['el_tsbpo_sketch_seed'] = int(
                                el_tsbpo_cfg.get('sketch_seed', 17)
                            )
                            probe_batch.meta_info['el_tsbpo_gradient_mode'] = 'probe'
                            probe_batch.meta_info[
                                'el_tsbpo_probe_micro_batch_size'
                            ] = max(
                                1,
                                int(
                                    el_tsbpo_cfg.get(
                                        'probe_micro_batch_size', 1
                                    )
                                ),
                            )
                            # The actor update follows immediately after the
                            # probe. Keep the same FSDP shards resident so the
                            # worker does not reload them between label RPCs.
                            probe_batch.meta_info['el_tsbpo_keep_model_on_gpu'] = True
                            probe_batch.meta_info['el_tsbpo_global_probe_count'] = probe_count
                            with _timer('el_tsbpo_probe_gradient', timing_raw):
                                probe_output = self.actor_rollout_wg.compute_el_tsbpo_score_sketch(
                                    probe_batch
                                )
                            probe_sketch = probe_output.batch[
                                'el_tsbpo_score_sketches'
                            ][0].detach().float().cpu()
                            probe_stats = probe_output.meta_info.get(
                                'el_tsbpo_gradient_stats', {}
                            )
                            probe_norm = float(probe_sketch.norm().item())
                            probe_valid = bool(
                                np.isfinite(probe_norm) and probe_norm > 1e-8
                            )
                            if probe_valid:
                                batch.meta_info['el_tsbpo_probe_gradient_sketch'] = (
                                    probe_sketch
                                )
                                batch.meta_info['el_tsbpo_probe_tasks'] = probe_tasks
                            else:
                                batch.meta_info[
                                    'el_tsbpo_gradient_labels_available'
                                ] = False
                            metrics['el_tsbpo/utility/probe_gradient_computed'] = float(
                                probe_valid
                            )
                            metrics['el_tsbpo/utility/probe_backward_count'] = float(
                                probe_stats.get('backward_count', 0.0)
                                if probe_valid else 0.0
                            )
                            metrics[
                                'el_tsbpo/utility/probe_micro_batch_count'
                            ] = float(
                                probe_stats.get('probe_micro_batch_count', 0.0)
                            )
                            metrics[
                                'el_tsbpo/utility/probe_micro_batch_size'
                            ] = float(
                                probe_stats.get('probe_micro_batch_size', 0.0)
                            )
                            metrics['el_tsbpo/utility/probe_sketch_norm_mean'] = float(
                                probe_stats.get('sketch_norm_mean', 0.0)
                            )
                            metrics['el_tsbpo/utility/probe_zero_signal'] = float(
                                not probe_valid
                            )
                            metrics['el_tsbpo/utility/n_probe_edges'] = float(
                                probe_count if probe_valid else 0
                            )
                            # The detached probe is used only to label the
                            # predictor. Do not remove any valid rollout from
                            # actor training based on probe membership.
                            metrics['el_tsbpo/utility/probe_actor_rows_masked'] = 0.0
                        else:
                            metrics['el_tsbpo/utility/probe_gradient_computed'] = 0.0
                            metrics['el_tsbpo/utility/probe_backward_count'] = 0.0
                            metrics[
                                'el_tsbpo/utility/probe_micro_batch_count'
                            ] = 0.0
                            metrics[
                                'el_tsbpo/utility/probe_micro_batch_size'
                            ] = 0.0
                            metrics['el_tsbpo/utility/probe_sketch_norm_mean'] = 0.0
                            metrics['el_tsbpo/utility/probe_zero_signal'] = 0.0
                            metrics['el_tsbpo/utility/n_probe_edges'] = 0.0
                            metrics['el_tsbpo/utility/probe_actor_rows_masked'] = 0.0
                        if not probe_valid:
                            score_sketch_count = 0
                        metrics['el_tsbpo/utility/n_gradient_label_edges'] = float(
                            score_sketch_count
                        )
                        batch.batch['el_tsbpo_score_sketches'] = all_sketches
                        metrics['el_tsbpo/utility/n_score_sketch_edges'] = float(
                            score_sketch_count
                        )
                        metrics['el_tsbpo/utility/gradient_sketch_discarded'] = float(
                            max(0, len(utility_indices) - score_sketch_count)
                        )
                        effective_indices = torch.nonzero(
                            effective_weight > 0, as_tuple=False
                        ).reshape(-1).detach().cpu().tolist()
                        # Graph-GiGPO keeps the canonical GIGPO episode signal
                        # on every active turn.  Do not drop zero local/tree
                        # rows from the PPO population: doing so changes the
                        # empirical rollout distribution and overweights long
                        # branch continuations.  The effective mask remains a
                        # diagnostic and predictor/allocator scope indicator.
                        actor_indices = torch.nonzero(
                            rollout_weight > 0, as_tuple=False
                        ).reshape(-1).detach().cpu().tolist()
                        n_rollout_edges = int((rollout_weight > 0).sum().item())
                        n_zero_signal_edges_skipped = max(
                            0, n_rollout_edges - len(actor_indices)
                        )
                        metrics.update({
                            'el_tsbpo/actor/n_rollout_edges': float(n_rollout_edges),
                            'el_tsbpo/actor/n_actor_edges': float(len(actor_indices)),
                            'el_tsbpo/actor/n_effective_edges': float(len(effective_indices)),
                            'el_tsbpo/actor/n_zero_signal_edges_skipped': float(
                                n_zero_signal_edges_skipped
                            ),
                            'el_tsbpo/actor/effective_edge_fraction': float(
                                len(effective_indices) / max(1, n_rollout_edges)
                            ),
                        })
                        if not actor_indices:
                            actor_batch = None
                        else:
                            actor_batch = batch.select_idxs(actor_indices)
                            world_size = max(1, int(self.actor_rollout_wg.world_size))
                            native_global_mini = max(
                                1,
                                int(
                                    self.config.actor_rollout_ref.actor.get(
                                        'ppo_mini_batch_size', world_size
                                    )
                                )
                                * int(self.config.actor_rollout_ref.rollout.n),
                            )
                            n_unpadded_actor_rows = len(actor_batch)
                            # Match native VERL/GIGPO: only pad enough for DP
                            # dispatch. The actor itself splits the realized
                            # bare+branch population with its configured PPO
                            # mini- and micro-batch sizes.
                            actor_batch, actor_pad_size = pad_dataproto_to_divisor(
                                actor_batch, world_size
                            )
                            if actor_pad_size:
                                actor_batch.batch['el_tsbpo_sample_weight'][
                                    -actor_pad_size:
                                ] = 0.0
                                actor_batch.batch['el_tsbpo_policy_weight'][
                                    -actor_pad_size:
                                ] = 0.0
                                actor_batch.batch['el_tsbpo_effective_edge_weight'][
                                    -actor_pad_size:
                                ] = 0.0
                            # Selection and padding invalidate the token balance
                            # computed on the full rollout batch. Rebalance the
                            # actual PPO population, then shuffle inside each
                            # rank so bare and branch phases cannot become
                            # separate optimizer steps.
                            if self.config.trainer.balance_batch:
                                self._balance_batch(
                                    actor_batch,
                                    metrics=metrics,
                                    logging_prefix='el_tsbpo_actor_seqlen',
                                )
                            shuffle_actor_rows = bool(
                                el_tsbpo_cfg.get('shuffle_actor_rows', True)
                            )
                            if shuffle_actor_rows and len(actor_batch) > 1:
                                rows_per_rank = len(actor_batch) // world_size
                                generator = torch.Generator(device='cpu')
                                generator.manual_seed(
                                    int(el_tsbpo_cfg.get('sketch_seed', 17))
                                    + 1_000_003 * int(self.global_steps)
                                )
                                rank_permutations = []
                                for rank in range(world_size):
                                    offset = rank * rows_per_rank
                                    rank_permutations.append(
                                        torch.randperm(
                                            rows_per_rank, generator=generator
                                        ) + offset
                                    )
                                actor_batch.reorder(torch.cat(rank_permutations))
                            active_source = actor_batch.batch[
                                'el_tsbpo_sample_weight'
                            ] > 0
                            source_flags = actor_batch.batch[
                                'el_tsbpo_is_branch'
                            ][active_source].bool()
                            source_transitions = (
                                (source_flags[1:] != source_flags[:-1]).float().mean()
                                if source_flags.numel() > 1
                                else torch.zeros(())
                            )
                            metrics.update({
                                'el_tsbpo/actor/rows_shuffled': float(
                                    shuffle_actor_rows
                                ),
                                'el_tsbpo/actor/source_transition_fraction': float(
                                    source_transitions.item()
                                ),
                                'el_tsbpo/actor/native_global_mini_batch_size': float(
                                    native_global_mini
                                ),
                                'el_tsbpo/actor/optimizer_steps_per_epoch': float(
                                    np.ceil(
                                        n_unpadded_actor_rows
                                        / native_global_mini
                                    )
                                ),
                                'el_tsbpo/actor/padded_actor_rows': float(
                                    actor_pad_size
                                ),
                                'el_tsbpo/actor/padded_actor_row_fraction': float(
                                    actor_pad_size / max(1, len(actor_batch))
                                ),
                            })
                            actor_batch.meta_info['global_token_num'] = torch.sum(
                                actor_batch.batch['attention_mask'], dim=-1
                            ).tolist()
                            if self.use_reference_policy:
                                with _timer("ref", timing_raw):
                                    if not self.ref_in_actor:
                                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                            actor_batch
                                        )
                                    else:
                                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(
                                            actor_batch
                                        )
                                    actor_batch = actor_batch.union(ref_log_prob)
                            metrics['el_tsbpo/actor/n_padded_actor_edges'] = float(
                                len(actor_batch)
                            )

                    if capture_turn_diagnostics:
                        with _timer('turn_diagnostics', timing_raw):
                            if turn_diagnostic_run_dir is None:
                                output_dir = diagnostic_cfg.get('output_dir') or os.path.join(
                                    self.config.trainer.default_local_dir, 'turn_diagnostics'
                                )
                                run_name = datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8]
                                turn_diagnostic_run_dir = os.path.join(os.path.expanduser(str(output_dir)), run_name)
                            eos_values = batch.meta_info.get('eos_token_id', self.tokenizer.eos_token_id)
                            if not isinstance(eos_values, (list, tuple, np.ndarray)):
                                eos_values = [eos_values]
                            generation_path = os.path.join(str(self.config.actor_rollout_ref.model.path), 'generation_config.json')
                            if os.path.isfile(generation_path):
                                with open(generation_path) as generation_file:
                                    generation_eos = json.load(generation_file).get('eos_token_id', [])
                                eos_values = list(eos_values) + (
                                    generation_eos if isinstance(generation_eos, list) else [generation_eos]
                                )
                            eos_ids = sorted({int(value) for value in eos_values if value is not None})
                            configuration = {}
                            for key in CONFIG_KEYS:
                                value = OmegaConf.select(self.config, key)
                                configuration[key] = (
                                    OmegaConf.to_container(value, resolve=True)
                                    if OmegaConf.is_config(value) else value
                                )
                            selected_mask, turn_diagnostic_step_dir, dump_counts = build_pre_update_dump(
                                batch, actor_batch, self.tokenizer, step=self.global_steps,
                                directory=turn_diagnostic_run_dir, configuration=configuration,
                                old_entropy=turn_diagnostic_old_entropy,
                                controls_per_group=int(diagnostic_cfg.get('controls_per_group', 1)),
                                random_controls=int(diagnostic_cfg.get('length_controls_per_band', 4)),
                                eos_ids=eos_ids,
                            )
                            if actor_batch is not None:
                                ids = actor_batch.batch['turn_diagnostic_row_id']
                                actor_batch.batch['turn_diagnostic_selected'] = torch.as_tensor(
                                    selected_mask, dtype=torch.bool, device=ids.device
                                )[ids]
                                error_code = (
                                    (np.asarray(batch.non_tensor_batch['generation_finish_reason']) == 'length').astype(np.int64)
                                    + 2*(~np.asarray(batch.non_tensor_batch['is_action_valid'], dtype=bool)).astype(np.int64)
                                )
                                actor_batch.batch['turn_diagnostic_error_code'] = torch.as_tensor(
                                    error_code, dtype=torch.long, device=ids.device
                                )[ids]
                                actor_batch.meta_info['turn_diagnostics'] = {
                                    'output_dir': str(turn_diagnostic_step_dir),
                                    'global_step': self.global_steps, 'eos_token_ids': eos_ids,
                                    'capture_eos_gradients': bool(diagnostic_cfg.get('capture_eos_gradients', True)),
                                }
                            turn_diagnostic_capture_count += 1
                            turn_diagnostic_old_entropy = None
                            print(f"[turn diagnostics] step={self.global_steps}, errors={dump_counts['errors']}, "
                                  f"full_records={dump_counts['full_records']}, path={turn_diagnostic_step_dir}")

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if (
                        self.config.trainer.critic_warmup <= self.global_steps
                        and actor_batch is not None
                    ):
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        actor_output = self.actor_rollout_wg.update_actor(actor_batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                        if turn_diagnostic_step_dir is not None:
                            write_json(os.path.join(turn_diagnostic_step_dir, 'update_complete.json'), {
                                'global_step': self.global_steps, 'actor_update_completed': True,
                                'actor_metrics': actor_output_metrics,
                                'gradient_scope': 'Per-turn output derivatives and optimizer-minibatch parameter norms are in actor_rank*.json.',
                            })

                    # Utility supervision is generated from completed rollout
                    # returns and pre-update log-probs, then fitted only after
                    # the actor step. It affects allocation in later rounds.
                    if (
                        self.config.algorithm.adv_estimator == AdvantageEstimator.EL_TSBPO
                        and not (
                            hasattr(self.traj_collector, 'is_agent_dps_enabled')
                            and self.traj_collector.is_agent_dps_enabled()
                        )
                    ):
                        with _timer("update_el_tsbpo_predictor", timing_raw):
                            metrics.update(self.traj_collector.update_el_tsbpo_predictor(batch))

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if self.config.algorithm.adv_estimator == AdvantageEstimator.EL_TSBPO:
                                val_metrics = filter_el_tsbpo_quality_metrics(val_metrics)
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_host_memory_metrics())
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                el_tsbpo_metrics = batch.meta_info.get('el_tsbpo_metrics', {})
                if el_tsbpo_metrics:
                    metrics.update(el_tsbpo_metrics)
                gigpo_metrics = batch.meta_info.get('gigpo_metrics', {})
                if gigpo_metrics:
                    metrics.update(gigpo_metrics)

                # TODO: make a canonical logger that supports various backend
                if self.config.algorithm.adv_estimator == AdvantageEstimator.EL_TSBPO:
                    metrics = filter_el_tsbpo_quality_metrics(metrics)
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
