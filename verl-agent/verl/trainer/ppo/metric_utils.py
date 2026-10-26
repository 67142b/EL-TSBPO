# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Metrics related to the PPO trainer.
"""

from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List

import numpy as np
import torch

from verl import DataProto
from verl.utils.import_utils import deprecated


_EL_TSBPO_PROBABILITY_METRICS = frozenset(
    f"agent_dps/probability/{stage}/{key}"
    for stage in ("root", "branch")
    for key in ("enabled", "groups", "tasks", "effective_task_rounds", "queried_tasks",
                "calibrated_fraction", "history_reliability", "history_disabled_probability",
                "legacy_groups_skipped", "semantic_reliability", "semantic_disabled_probability",
                "prior_strength", "prior_success", "parameter_ess", "groups_committed",
                "incomplete_groups", "predicted_mixed", "observed_mixed", "brier", "count_nll")
)

_EL_TSBPO_QUALITY_METRICS = frozenset(
    {
        *_EL_TSBPO_PROBABILITY_METRICS,
        "training/global_step",
        "training/epoch",
        "perf/time_per_step",
        "system/memory_used_gib",
        "system/memory_used_fraction",
        "system/shared_memory_gib",
        "actor/entropy_loss",
        "actor/pg_loss",
        "actor/ppo_kl",
        "actor/kl_loss",
        "actor/kl_loss_objective",
        "actor/kl_coef",
        "actor/pg_clipfrac",
        "actor/grad_norm",
        "actor/lr",
        "episode/reward/mean",
        "episode/length/mean",
        "episode/valid_action_ratio",
        "episode/action_format_valid_ratio",
        "episode/admissible_action_ratio",
        "episode/natural_valid_action_ratio",
        "episode/branch_valid_action_ratio",
        "episode/truncation_rate",
        "episode/natural_truncation_rate",
        "episode/branch_truncation_rate",
        "el_tsbpo/sampling/bare_trajectory_state_sign_conflict_fraction",
        "el_tsbpo/sampling/branch_trajectory_state_sign_conflict_fraction",
        "response_length/mean",
        "response_length/clip_ratio",
        "training/rollout_logprob_audit/nonfinite_tokens",
        "training/rollout_logprob_audit/abs_dlog_p99",
        "training/rollout_logprob_audit/ratio_outside_clipfrac",
        "el_tsbpo/sampling/branch_row_fraction",
        "el_tsbpo/sampling/bare_terminal_success_rate",
        "el_tsbpo/sampling/branch_terminal_success_rate",
        "el_tsbpo/sampling/bare_combined_abs_row_mean",
        "el_tsbpo/sampling/branch_combined_abs_row_mean",
        "el_tsbpo/rollout/n_quarantined_trajectories",
        "el_tsbpo/audit/checked_trajectories",
        "el_tsbpo/audit/fail_context",
        "el_tsbpo/audit/fail_decision",
        "el_tsbpo/audit/fail_policy_group",
        "el_tsbpo/audit/fail_environment",
        "el_tsbpo/audit/fail_replay_observation",
        "el_tsbpo/audit/fail_replay_admissible",
        "el_tsbpo/audit/fail_cardinality",
        "el_tsbpo/rollout/natural_generation_worker_calls",
        "el_tsbpo/rollout/natural_generation_mean_batch_rows",
        "el_tsbpo/rollout/branch_generation_worker_calls",
        "el_tsbpo/rollout/branch_generation_efficiency_mean_batch_rows",
        "el_tsbpo/rollout/branch_ready_mean_rows",
        "el_tsbpo/tree/graph_gigpo_fallback_fraction",
        "el_tsbpo/tree/graph_gigpo_subtree_abs_mean",
        "el_tsbpo/tree/graph_gigpo_local_abs_mean",
        "el_tsbpo/tree/graph_gigpo_trajectory_policy_weight_p95",
        "el_tsbpo/tree/graph_gigpo_trajectory_policy_weight_p50",
        "el_tsbpo/tree/graph_gigpo_trajectory_policy_weight_max",
        "el_tsbpo/tree/graph_gigpo_trajectory_policy_mass_ess",
        "el_tsbpo/tree/graph_gigpo_segment_cap_hit_fraction",
        "el_tsbpo/tree/graph_gigpo_sibling_child_mass_relative_gap_max",
        "el_tsbpo/tree/graph_gigpo_raw_sibling_child_mass_relative_gap_max",
        "agent_dps/outcome/observed_mixed_state_rate",
        "agent_dps/outcome/selected_factual_mixed_state_rate",
        "agent_dps/outcome/new_mixed_states",
        "agent_dps/outcome/mixed_state_rate_gain",
        "agent_dps/outcome/homogeneous_factual_opposite_branch_rate",
        "agent_dps/outcome/branch_minus_factual_success_rate",
        "agent_dps/outcome/branch_trajectories",
        "agent_dps/outcome/selected_states_with_branch_evidence",
        "agent_dps/outcome/branch_transitions_committed",
        "agent_dps/outcome/branch_first_edge_coverage",
        "agent_dps/outcome/transition_contrast_state_rate",
        "agent_dps/outcome/transition_contrast_branch_rate",
        "agent_dps/outcome/terminal_contrast_state_rate",
        "agent_dps/outcome/terminal_contrast_branch_rate",
        "agent_dps/outcome/same_first_edge_terminal_contrast_branch_rate",
        "agent_dps/outcome/created_contrast_groups",
        "agent_dps/outcome/created_contrast_states",
        "agent_dps/outcome/created_contrast_per_branch",
        "agent_dps/outcome/predicted_created_contrast_rate",
        "agent_dps/outcome/contrast_calibration_bias",
        "agent_dps/outcome/contrast_calibration_abs_error",
        "agent_dps/outcome/contrast_brier_score",
        "sampling/effective_ratio",
        "agent_dps/root/random_task_count",
        "agent_dps/root/dps_replay_task_count",
        "agent_dps/root/replay_target_slots",
        "agent_dps/root/replay_pool_size",
        "agent_dps/root/exact_history_hit_fraction",
        "agent_dps/root/natural_random_revisit_count",
        "agent_dps/root/fresh_pool_epoch",
        "agent_dps/root/fresh_pool_cursor",
        "agent_dps/root/fresh_pool_size",
        "agent_dps/root/fresh_pool_covered",
        "agent_dps/root/fresh_pool_fraction",
        "agent_dps/root/fresh_pool_excluded_skips",
        "agent_dps/root/history_seen_fraction",
        "agent_dps/root/p_std",
        "agent_dps/root/quota_std",
        "agent_dps/root/min_quota",
        "agent_dps/root/mean_evidence_mass",
        "agent_dps/root/floor_binding_fraction",
        "agent_dps/root/max_reallocation_gain",
        "agent_dps/root/task_selection_dps_fraction",
        "agent_dps/root/replay_horizon",
        "agent_dps/root/dynamic_curriculum_enabled",
        "agent_dps/root/curriculum_candidate_count",
        "agent_dps/root/curriculum_selected_count",
        "agent_dps/root/curriculum_exploration_fraction",
        "agent_dps/root/candidate_frontier_mean",
        "agent_dps/root/selected_frontier_mean",
        "agent_dps/root/frontier_selection_lift",
        "agent_dps/root/candidate_history_seen_fraction",
        "agent_dps/root/selected_history_seen_fraction",
        "agent_dps/allocator/candidate_top_m",
        "agent_dps/allocator/used_budget",
        "agent_dps/allocator/budget_utilization",
        "agent_dps/allocator/unspent_budget",
        "agent_dps/allocator/n_states",
        "agent_dps/allocator/n_selected_states",
        "agent_dps/allocator/task_union_enabled",
        "agent_dps/allocator/local_state_contrast_enabled",
        "agent_dps/allocator/task_union_beam_width",
        "agent_dps/allocator/n_eligible_states",
        "agent_dps/allocator/predicted_new_mixed_states",
        "agent_dps/allocator/predicted_new_mixed_tasks",
        "agent_dps/calibration/enabled",
        "agent_dps/calibration/groups",
        "agent_dps/calibration/groups_committed",
        "agent_dps/calibration/tasks",
        "agent_dps/calibration/effective_task_rounds",
        "agent_dps/calibration/queried_tasks",
        "agent_dps/calibration/calibrated_fraction",
        "agent_dps/calibration/prior_strength_mean",
        "agent_dps/calibration/prior_success_mean",
        "agent_dps/calibration/grid_boundary_fraction",
        "agent_dps/outcome/bare_mixed_task_ratio",
        "agent_dps/outcome/post_branch_mixed_task_ratio",
        "agent_dps/outcome/new_mixed_tasks",
        "agent_dps/outcome/new_mixed_states_in_bare_mixed_tasks",
        "agent_dps/outcome/task_activation_brier",
        "agent_dps/allocator/state_prior_hit_fraction",
        "agent_dps/allocator/semantic_prior_hit_fraction",
        "agent_dps/allocator/selected_semantic_prior_hit_fraction",
        "agent_dps/allocator/semantic_prior_mass_mean",
        "agent_dps/branch_embedding/encoded",
        "agent_dps/branch_embedding/cache_hits",
        "agent_dps/branch_embedding/cuda_calls",
        "agent_dps/branch_embedding/cpu_calls",
        "agent_dps/branch_embedding/fallback_calls",
        "agent_dps/branch_embedding/skipped",
        "agent_dps/branch_embedding/retry_suppressed",
        "agent_dps/branch_embedding/missing_goal",
        "agent_dps/branch_embedding/cuda_peak_mib",
        "agent_dps/branch_embedding/cuda_budget_mib",
        "agent_dps/branch_embedding/cuda_free_before_mib",
        "agent_dps/branch_embedding/cuda_oom_retries",
        "agent_dps/branch_embedding/host_raw_free_gib",
        "agent_dps/branch_embedding/host_reclaimable_gib",
        "agent_dps/branch_embedding/host_available_gib",
        "agent_dps/branch_embedding/host_memory_skips",
        "agent_dps/state_prior/states",
        "agent_dps/state_prior/buckets",
        "agent_dps/state_prior/clusters",
        "agent_dps/state_prior/bare_paths",
        "agent_dps/state_prior/branch_paths",
        "agent_dps/state_prior/bare_incomplete",
        "agent_dps/state_prior/branch_incomplete",
        "agent_dps/survival/bare_origins",
        "agent_dps/survival/branch_origins",
        "agent_dps/survival/bare_censored",
        "agent_dps/survival/branch_censored",
        "agent_dps/survival/contexts",
        "agent_dps/survival/particles",
        "agent_dps/survival/prior_strength",
        "agent_dps/survival/frailty_scale",
        "agent_dps/survival/bare_suffix_views",
        "agent_dps/survival/branch_suffix_views",
        "agent_dps/survival/suffix_paths",
        "agent_dps/survival/suffix_trees",
        "agent_dps/survival/suffix_views",
        "agent_dps/allocator/state_suffix_conditioned_fraction",
        "agent_dps/allocator/state_suffix_mass_mean",
        "agent_dps/allocator/bare_mixed_state_groups_considered",
        "agent_dps/allocator/selected_bare_mixed_state_groups",
        "agent_dps/allocator/bare_mixed_branch_budget_fraction",
        "agent_dps/allocator/mean_k",
        "agent_dps/allocator/min_k",
        "agent_dps/allocator/max_k",
        "agent_dps/allocator/prefilter_keep_fraction",
        "agent_dps/allocator/acquisition_prefilter_enabled",
        "agent_dps/allocator/candidate_acquisition_mean",
        "agent_dps/allocator/selected_acquisition_mean",
        "agent_dps/allocator/acquisition_selection_lift",
        "agent_dps/allocator/bayesian_transition_objective_enabled",
        "agent_dps/allocator/bayesian_terminal_objective_enabled",
        "agent_dps/allocator/posterior_optimal_knapsack",
        "agent_dps/allocator/predicted_created_contrast_sum",
        "agent_dps/allocator/predicted_created_contrast_rate",
        "agent_dps/allocator/candidate_contrast_probability_mean",
        "agent_dps/allocator/selected_contrast_probability_mean",
        "agent_dps/allocator/contrast_probability_selection_lift",
        "agent_dps/allocator/history_prior_mass_mean",
        "agent_dps/allocator/current_likelihood_mass_mean",
        "agent_dps/allocator/candidate_factual_edge_coverage",
        "agent_dps/allocator/selected_factual_edge_coverage",
        "agent_dps/allocator/selected_same_edge_opposite_probability",
        "agent_dps/allocator/candidate_transition_edge_count_mean",
        "agent_dps/allocator/selected_transition_edge_count_mean",
        "agent_dps/allocator/transition_edge_count_selection_lift",
        "agent_dps/allocator/same_round_factual_excluded",
        "agent_dps/allocator/current_factual_committed_before_selection",
        "agent_dps/allocator/current_factual_conditioned_once",
        "agent_dps/allocator/transition_evidence_written_once",
        "agent_dps/allocator/current_factual_mixed_states",
        "agent_dps/allocator/current_factual_homogeneous_states",
        "agent_dps/allocator/factual_observations_committed_before_selection",
        "agent_dps/allocator/factual_observations_committed_after_selection",
        "agent_dps/allocator/factual_transitions_committed",
        "agent_dps/allocator/root_bare_observations_committed",
        "agent_dps/allocator/candidate_current_factual_success_mean",
        "agent_dps/allocator/selected_current_factual_success_mean",
        "agent_dps/allocator/current_factual_selection_lift",
        "webshop/task_sampler/replacement_across_batches",
        "webshop/task_sampler/batch_natural_revisits",
        "webshop/task_sampler/goal_coverage",
        "webshop/task_sampler/query_family_coverage",
    }
)


def compute_host_memory_metrics() -> Dict[str, float]:
    """Report the memory accounting used by Ray and Linux shared memory."""
    from pathlib import Path
    from ray._private.utils import get_system_memory, get_used_memory

    used = get_used_memory()
    metrics = {
        "system/memory_used_gib": used / 2**30,
        "system/memory_used_fraction": used / max(1, get_system_memory()),
    }
    try:
        stats = dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
        metrics["system/shared_memory_gib"] = int(stats['shmem']) / 2**30
    except (OSError, ValueError, KeyError):
        pass
    return metrics


def filter_el_tsbpo_quality_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Select training-quality and timing signals for EL-TSBPO logging backends."""
    return {
        key: value
        for key, value in metrics.items()
        if key in _EL_TSBPO_QUALITY_METRICS
        or key.startswith(("timing_s/", "timing_per_token_ms/", "gpu_memory/"))
        or (
            key.startswith("el_tsbpo/rollout/")
            and key.endswith(("_seconds", "_ms_per_response_token", "_worker_call_ms_mean"))
        )
        or key.startswith("agent_dps/allocator/new_k_")
        or key.startswith("agent_dps/allocator/actual_k_")
        or key.startswith(("agent_dps/allocator/prefix_", "agent_dps/allocator/selected_quota_",
                           "agent_dps/root/quota_", "agent_dps/root/semantic_"))
        or key == "agent_dps/root/shared_mass_mean"
        or (key.startswith("episode/") and key.endswith("success_rate"))
        or (
            key.startswith("val/")
            and (key.endswith("success_rate") or key.endswith("/test_score"))
        )
    }


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.
    
    This is an internal helper function that extracts masks and lengths for prompts and responses.
    
    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        
    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)
    unique_traj_uid, unique_idx = np.unique(batch.non_tensor_batch['traj_uid'], return_index=True)

    # non_tensor_batch stores values as dtype=object arrays; convert to proper
    # numeric types so that .max()/.min()/.mean() return numpy scalars (which
    # have .item()) instead of Python floats (which do not).
    episode_rewards = np.asarray(batch.non_tensor_batch["episode_rewards"], dtype=np.float64)
    episode_lengths = np.asarray(batch.non_tensor_batch["episode_lengths"], dtype=np.int64)
    tool_callings = np.asarray(batch.non_tensor_batch["tool_callings"], dtype=np.float64)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
        # episode
        "episode/reward/mean": 
            episode_rewards[unique_idx].mean().item(),
        "episode/reward/max": 
            episode_rewards[unique_idx].max().item(),
        "episode/reward/min": 
            episode_rewards[unique_idx].min().item(),
        "episode/length/mean": 
            episode_lengths[unique_idx].mean().item(),
        "episode/length/max":
            episode_lengths[unique_idx].max().item(),
        "episode/length/min": 
            episode_lengths[unique_idx].min().item(),
        "episode/tool_call_count/mean": 
            tool_callings[unique_idx].mean().item(),
        # "episode/tool_call_count/max":
        #     tool_callings[unique_idx].max().item(),
        # "episode/tool_call_count/min":
        #     tool_callings[unique_idx].min().item(),
        **({f"episode/{k}": float(v[0]) for k, v in batch.non_tensor_batch.items() if "success_rate" in k}),
    }
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.
    
    This function calculates both raw timing metrics (in seconds) and per-token timing metrics 
    (in milliseconds) for various processing stages like generation, reference computation, 
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())},
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    """
    Computes throughput metrics for PPO training.
    
    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).
    
    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.
        
    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU
            
    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.
    
    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.
    
    Args:
        data_sources: List of data source identifiers for each sample.
        sample_inputs: List of input prompts corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }
        
        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)
        
    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_inputs = ["prompt1", "prompt1", "prompt2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_inputs, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed)
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val
