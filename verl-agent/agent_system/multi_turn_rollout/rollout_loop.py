# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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
from __future__ import annotations

import torch
import numpy as np
import os
import json
import sys
import time
import gc
import logging
import random
import math
import hashlib
import io
from contextlib import redirect_stdout
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any, Mapping, Sequence
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import (
    process_image,
    to_list_of_dict,
    torch_to_numpy,
    filter_group_data,
)
from agent_system.environments import EnvironmentManagerBase
from agent_system.environments.prompts.alfworld import (
    ALFWORLD_TEMPLATE,
    ALFWORLD_TEMPLATE_NO_HIS,
)
from agent_system.environments.prompts.webshop import (
    WEBSHOP_TEMPLATE,
    WEBSHOP_TEMPLATE_NO_HIS,
)
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from agent_system.decision_identity import (
    context_hash,
    decode_config_hash,
    decision_key,
    environment_state_hash,
    environment_snapshot_hash,
    policy_group_key,
)


def _pad_non_tensor(ref, size: int, key: str):
    if isinstance(ref, np.ndarray):
        if ref.ndim == 0:

            return np.full(size, ref.item(), dtype=ref.dtype)
        if ref.dtype == object:
            return np.full((size,) + ref.shape[1:], None, dtype=ref.dtype)
        if ref.dtype.kind in ("U", "S"):
            return np.full((size,) + ref.shape[1:], "", dtype=ref.dtype)
        return np.zeros((size,) + ref.shape[1:], dtype=ref.dtype)

    return np.array([None] * size, dtype=object)


def _pack_token_budget_batches(
    prompt_lengths: Sequence[int],
    *,
    token_budget: int,
    max_response_tokens: int,
    max_rows: int = 0,
) -> List[List[int]]:

    budget = max(1, int(token_budget))
    response_cost = max(1, int(max_response_tokens))
    row_limit = max(0, int(max_rows))
    batches: List[List[int]] = []
    current: List[int] = []
    current_tokens = 0
    for row_idx, prompt_len in enumerate(prompt_lengths):
        estimated = max(1, int(prompt_len)) + response_cost
        exceeds_budget = bool(current) and current_tokens + estimated > budget
        exceeds_rows = row_limit > 0 and bool(current) and len(current) >= row_limit
        if exceeds_budget or exceeds_rows:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(row_idx)
        current_tokens += estimated
    if current:
        batches.append(current)
    return batches


def _is_truthy_flag(value: Any) -> bool:

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _branch_generation_batches(
    prompt_lengths: Sequence[int],
    *,
    token_budget: int,
    max_response_tokens: int,
    max_rows: int = 0,
    synchronous_frontier: bool = False,
    local_token_packing: bool = True,
) -> List[List[int]]:

    if not prompt_lengths:
        return []
    if synchronous_frontier or not local_token_packing:
        return [list(range(len(prompt_lengths)))]
    return _pack_token_budget_batches(
        prompt_lengths,
        token_budget=token_budget,
        max_response_tokens=max_response_tokens,
        max_rows=max_rows,
    )


def _branch_frontier_target(
    pending_count: int,
    configured_batch_size: int,
    *,
    synchronous_frontier: bool = False,
) -> int:

    pending = max(0, int(pending_count))
    if pending == 0:
        return 0
    if synchronous_frontier:
        return pending
    return min(pending, max(1, int(configured_batch_size)))


def _agent_dps_branch_evidence(rows: Sequence[Mapping[str, Any]]):

    from agent_system.multi_turn_rollout.agent_dps import action_key, state_key

    anchor_observations = []
    descendant_observations = []
    transition_observations = []
    outcomes_by_anchor: dict[str, dict[str, float]] = defaultdict(dict)
    first_edges_by_anchor: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    rows_by_trajectory: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if bool(row.get("el_tsbpo_quarantined", False)):
            continue
        anchor_key = str(row.get("el_tsbpo_state_group_key", "") or "")
        trajectory = str(row.get("traj_uid", "") or "")
        if not anchor_key or not trajectory:
            continue
        outcome = float(row.get("trajectory_success", 0.0) or 0.0)
        anchor_observations.append((anchor_key, outcome, trajectory))
        outcomes_by_anchor[anchor_key][trajectory] = outcome
        rows_by_trajectory[trajectory].append(row)

        scope = str(row.get("uid", "") or "")
        observation = str(row.get("anchor_obs", "") or "")
        if scope and observation:
            key = state_key(scope, observation)
            if key != anchor_key:
                descendant_observations.append((key, outcome, trajectory))

    for trajectory, trajectory_rows in rows_by_trajectory.items():
        ordered = sorted(trajectory_rows, key=lambda row: int(row.get("turn_idx", 0)))
        for source in ordered:
            scope = str(source.get("uid", "") or "")
            action = str(source.get("action_text", "") or "")
            if not bool(source.get("is_action_format_valid", True)):
                action_status = "format_invalid"
            elif not bool(source.get("is_action_admissible", True)):
                action_status = "illegal"
            else:
                action_status = "legal"
            action = action_key(f"{action_status}:{action}")
            observation = str(source.get("anchor_obs", "") or "")
            next_observation = str(source.get("next_anchor_obs", "") or "")
            if bool(source.get("el_tsbpo_node_is_terminal", False)):
                next_observation = (
                    "__agent_dps_terminal_success__"
                    if float(source.get("trajectory_success", 0.0) or 0.0) >= 0.5
                    else "__agent_dps_terminal_failure__"
                )
            if not scope or not action or not observation or not next_observation:
                continue
            next_key = state_key(scope, next_observation)
            transition_observations.append(
                (
                    state_key(scope, observation),
                    action,
                    next_key,
                    float(source.get("trajectory_success", 0.0) or 0.0),
                    trajectory,
                )
            )
            anchor_key = str(source.get("el_tsbpo_state_group_key", "") or "")
            if anchor_key and trajectory not in first_edges_by_anchor[anchor_key]:
                first_edges_by_anchor[anchor_key][trajectory] = (
                    action,
                    next_key,
                )

    return (
        anchor_observations,
        descendant_observations,
        transition_observations,
        outcomes_by_anchor,
        first_edges_by_anchor,
    )


class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

        self._branch_envs = None
        self._el_tsbpo_step = 0
        self._el_tsbpo_state_group_records: List[Dict[str, Any]] = []
        self._el_tsbpo_state_group_by_anchor: Dict[str, Dict[str, Any]] = {}
        self._agent_dps_memory = None
        self._agent_dps_memory_dirty = False
        self._agent_dps_last_save_step = -1


        self._agent_dps_root_candidates = None
        self._agent_dps_root_candidate_step = -1
        self._agent_dps_root_replay_pool: Dict[str, Dict[str, Any]] = {}
        self._agent_dps_task_sampler_restored = False
        self._active_policy_version = "unknown"
        self._active_decode_config_hash = None


        self._el_tsbpo_generation_stats = {
            "calls": 0,
            "worker_calls": 0,
            "rows": 0,
            "batches": 0,
            "max_batch_rows": 0,
            "token_budget": 0,
            "prompt_tokens": 0,


            "response_tokens": 0,
            "padded_rows": 0,
            "wall_seconds": 0.0,
            "tokenize_seconds": 0.0,
            "batch_rows": 0,
            "batch_prompt_tokens": 0,
            "batch_planned_tokens": 0,
            "natural": self._new_generation_kind_stats(),
            "branch": self._new_generation_kind_stats(),
            "env_step_seconds": 0.0,
            "branch_env_wait_seconds": 0.0,
            "branch_env_submit_seconds": 0.0,
            "branch_initial_spawn_seconds": 0.0,
            "branch_refill_spawn_seconds": 0.0,
            "branch_ready_batches": 0,
            "branch_ready_rows": 0,
            "branch_ready_max_rows": 0,
            "branch_synchronous_frontier": 0,
            "branch_local_token_packing": 1,
            "branch_horizon_bucket_count": 0,
            "branch_horizon_bucket_rows": 0,
            "branch_horizon_bucket_max_rows": 0,
            "branch_refill_spawn_calls": 0,
            "branch_refill_branches": 0,
            "branch_active_frontier_batches": 0,
            "branch_active_frontier_rows": 0,
            "branch_active_frontier_max_rows": 0,
        }
        self._el_tsbpo_generation_stats["branch_synchronous_frontier"] = int(
            _is_truthy_flag(
                self._get_el_tsbpo_config().get("branch_synchronous_frontier", False)
            )
        )

    @staticmethod
    def _new_generation_kind_stats() -> Dict[str, Any]:

        return {
            "calls": 0,
            "worker_calls": 0,
            "rows": 0,
            "padded_rows": 0,
            "prompt_tokens": 0,
            "response_tokens": 0,
            "wall_seconds": 0.0,
            "batch_count": 0,
            "batch_rows": 0,
            "batch_prompt_tokens": 0,
            "batch_planned_tokens": 0,
        }

    def _rollout_policy_version(
        self, meta_info: Optional[Dict[str, Any]] = None
    ) -> str:

        meta_info = meta_info or {}
        value = meta_info.get("rollout_policy_version")
        if value is None:


            return "unknown"
        return str(value)

    def _decode_config_hash(self, meta_info: Optional[Dict[str, Any]] = None) -> str:

        meta_info = meta_info or {}
        actor_rollout_ref = getattr(self.config, "actor_rollout_ref", {})
        rollout_cfg = getattr(actor_rollout_ref, "rollout", None)
        if rollout_cfg is None and hasattr(actor_rollout_ref, "get"):
            rollout_cfg = actor_rollout_ref.get("rollout", {})
        if rollout_cfg is None:
            rollout_cfg = {}
        settings = {
            "temperature": meta_info.get(
                "temperature", rollout_cfg.get("temperature", None)
            ),
            "top_p": rollout_cfg.get("top_p", None),
            "top_k": rollout_cfg.get("top_k", None),
            "min_p": rollout_cfg.get("min_p", None),
            "response_length": rollout_cfg.get("response_length", None),
            "max_new_tokens": rollout_cfg.get("max_new_tokens", None),
            "n": rollout_cfg.get("n", 1),
            "best_of": rollout_cfg.get("best_of", None),
            "ignore_eos": rollout_cfg.get("ignore_eos", None),
            "presence_penalty": rollout_cfg.get("presence_penalty", None),
            "frequency_penalty": rollout_cfg.get("frequency_penalty", None),
            "repetition_penalty": rollout_cfg.get("repetition_penalty", None),
            "eos_token_id": meta_info.get("eos_token_id", None),
            "do_sample": meta_info.get("do_sample", True),
            "validate": meta_info.get("validate", False),
        }
        return decode_config_hash(settings)

    def _context_metadata(
        self, apply_chat_template_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:

        data_cfg = getattr(self.config, "data", {})
        return {
            "tokenizer": getattr(
                self.tokenizer, "name_or_path", self.tokenizer.__class__.__name__
            ),
            "tokenizer_class": self.tokenizer.__class__.__name__,
            "padding_side": getattr(self.tokenizer, "padding_side", None),
            "chat_template_kwargs": apply_chat_template_kwargs,
            "max_prompt_length": data_cfg.get("max_prompt_length", None),
            "truncation": data_cfg.get("truncation", None),
        }

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):

        raw_prompt = gen_batch.non_tensor_batch["raw_prompt"][item]
        data_source = gen_batch.non_tensor_batch["data_source"][item]
        apply_chat_template_kwargs = self.config.data.get(
            "apply_chat_template_kwargs", {}
        )


        obs_texts = obs.get("text", None)
        obs_images = obs.get("image", None)
        obs_anchors = obs.get("anchor", None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = (
            torch_to_numpy(obs_anchor, is_object=True)
            if isinstance(obs_anchor, torch.Tensor)
            else obs_anchor
        )


        obs_content = ""
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        chat = np.array(
            [
                {
                    "content": obs_content,
                    "role": "user",
                }
            ]
        )


        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs,
        )


        row_dict = {}


        if is_multi_modal:

            raw_prompt = prompt_with_chat_template.replace(
                "<image>", "<|vision_start|><|image_pad|><|vision_end|>"
            )
            row_dict["multi_modal_data"] = {"image": [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(
                row_dict["multi_modal_data"]["image"], return_tensors="pt"
            )
            image_grid_thw = image_inputs["image_grid_thw"]
            row_dict["multi_modal_inputs"] = {
                key: val for key, val in image_inputs.items()
            }
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while "<image>" in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        "<image>",
                        "<|vision_start|>"
                        + "<|placeholder|>"
                        * (image_grid_thw[index].prod() // merge_length)
                        + "<|vision_end|>",
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace(
                    "<|placeholder|>", self.processor.image_token
                )

        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.config.data.truncation,
        )

        if is_multi_modal:
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [
                torch.cat((text_position_ids, vision_position_ids), dim=0)
            ]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = (
                    raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
                )
            elif self.config.data.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}."
                )


        context_digest = context_hash(
            input_ids[0],
            attention_mask[0],
            position_ids[0],
            metadata=self._context_metadata(apply_chat_template_kwargs),
        )
        valid_prompt_tokens = (
            input_ids[0][attention_mask[0].bool()].detach().cpu().tolist()
        )
        row_dict.update(
            {
                "input_ids": input_ids[0],
                "attention_mask": attention_mask[0],
                "position_ids": position_ids[0],
                "raw_prompt_ids": raw_prompt_ids,
                "anchor_obs": _obs_anchor,


                "context_hash": context_digest,
                "context_token_ids": valid_prompt_tokens,
                "context_position_ids": position_ids[0][..., attention_mask[0].bool()]
                .detach()
                .cpu()
                .tolist(),
                "decision_context": prompt_with_chat_template,
                "policy_version": self._rollout_policy_version(gen_batch.meta_info),
                "decode_config_hash": self._decode_config_hash(gen_batch.meta_info),
                "decision_key": None,
                "policy_group_key": None,
                "env_key": None,
                "env_snapshot_hash": None,
                "env_snapshot_ref": None,
                "index": item,
                "data_source": data_source,
            }
        )

        if self.config.data.get("return_raw_chat", False):
            row_dict["raw_prompt"] = chat.tolist()

        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto,
        obs: Dict,
    ) -> DataProto:
        batch_size = len(gen_batch.batch["input_ids"])
        processed_samples = []


        for item in range(batch_size):

            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)


        batch = collate_fn(processed_samples)


        new_batch = DataProto.from_single_dict(
            data=batch, meta_info=gen_batch.meta_info
        )

        return new_batch

    @staticmethod
    def _select_observation_rows(
        obs: Dict[str, Any], indices: List[int]
    ) -> Dict[str, Any]:
        selected: Dict[str, Any] = {}
        for key, value in obs.items():
            if value is None:
                selected[key] = None
            elif isinstance(value, torch.Tensor):
                selected[key] = value[indices]
            elif isinstance(value, np.ndarray):
                selected[key] = value[indices]
            elif isinstance(value, list):
                selected[key] = [value[i] for i in indices]
            elif isinstance(value, tuple):
                selected[key] = tuple(value[i] for i in indices)
            else:

                selected[key] = value
        return selected

    def gather_rollout_data(
        self,
        total_batch_list: List[List[Dict]],
        episode_rewards: np.ndarray,
        episode_lengths: np.ndarray,
        success: Dict[str, np.ndarray],
        traj_uid: np.ndarray,
        tool_callings: np.ndarray,
    ) -> DataProto:
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)


        trajectory_success = np.asarray(
            success.get("success_rate", np.zeros(batch_size, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)

        effective_batch = []
        for bs in range(batch_size):

            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data["traj_uid"], (
                    "data is not from the same trajectory"
                )
                if data["active_masks"]:

                    data["episode_rewards"] = episode_rewards[bs]

                    data["episode_lengths"] = episode_lengths[bs]

                    data["tool_callings"] = tool_callings[bs]

                    for key, value in success_rate.items():
                        data[key] = value
                    data["trajectory_success"] = float(
                        trajectory_success[bs]
                        if bs < len(trajectory_success)
                        else float(episode_rewards[bs] > 0.0)
                    )

                    effective_batch.append(data)


        gen_batch_output = DataProto.from_single_dict(data=collate_fn(effective_batch))
        return gen_batch_output


    def _repeat_with_allocation(
        self, gen_batch: DataProto, alloc: np.ndarray
    ) -> DataProto:

        repeat_indices = np.concatenate(
            [np.full(int(alloc[i]), i, dtype=np.int64) for i in range(len(alloc))]
        )


        if gen_batch.batch is not None:
            repeated_tensors = {}
            for key, tensor in gen_batch.batch.items():
                idx_tensor = torch.as_tensor(
                    repeat_indices, dtype=torch.long, device=tensor.device
                )
                repeated_tensors[key] = tensor.index_select(0, idx_tensor)
            from tensordict import TensorDict

            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(len(repeat_indices),),
            )
        else:
            repeated_batch = None


        repeated_non_tensor_batch = {}
        for key, val in gen_batch.non_tensor_batch.items():
            repeated_non_tensor_batch[key] = val[repeat_indices]

        return type(gen_batch)(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=gen_batch.meta_info,
        )

    def _get_branch_envs(self):
        if self._branch_envs is not None:
            return self._branch_envs

        from omegaconf import OmegaConf

        try:
            resources_per_worker = OmegaConf.to_container(
                self.config.env.resources_per_worker, resolve=True
            )
        except Exception:
            resources_per_worker = {"num_cpus": 0.1}
        if resources_per_worker is None:
            resources_per_worker = {"num_cpus": 0.1}
        resources_per_worker.setdefault("num_gpus", 0.0)
        bu_cfg = self._get_el_tsbpo_config()
        try:
            branch_pool_size = max(
                1,
                int(bu_cfg.get("branch_worker_pool_size", 128)),
            )
        except (TypeError, ValueError):
            branch_pool_size = 128

        env_name = str(self.config.env.get("env_name", "")).lower()
        if "webshop" in env_name:
            from agent_system.environments.branch_envs import BranchWebshopEnvs

            webshop_root = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "environments",
                "env_package",
                "webshop",
                "webshop",
            )
            use_small = bool(self.config.env.webshop.get("use_small", True))
            self._branch_envs = BranchWebshopEnvs(
                env_kwargs={
                    "observation_mode": "text",
                    "num_products": 1000 if use_small else None,
                    "human_goals": bool(
                        self.config.env.webshop.get("human_goals", False)
                    ),
                    "_shared_goal_order": True,
                    "_goal_order_seed": int(self.config.env.get("seed", 0)),
                    "file_path": os.path.join(
                        webshop_root,
                        "data",
                        "items_shuffle_1000.json"
                        if use_small
                        else "items_shuffle.json",
                    ),
                    "attr_path": os.path.join(
                        webshop_root,
                        "data",
                        "items_ins_v2_1000.json" if use_small else "items_ins_v2.json",
                    ),
                },
                resources_per_worker=resources_per_worker,
                seed=self.config.env.get("seed", 0),
                max_workers=branch_pool_size,
                max_steps=int(self.config.env.max_steps),
            )
        else:
            from agent_system.environments.branch_envs import BranchAlfworldEnvs

            alf_config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "environments",
                "env_package",
                "alfworld",
                "configs",
                "config_tw.yaml",
            )
            self._branch_envs = BranchAlfworldEnvs(
                alf_config_path=alf_config_path,
                resources_per_worker=resources_per_worker,
                seed=self.config.env.get("seed", 0),
                max_workers=branch_pool_size,
            )
        return self._branch_envs

    @staticmethod
    def _extract_thought_and_action(response_text: str) -> Tuple[str, str]:
        think_start = response_text.find("<think>")
        think_end = response_text.find("</think>")
        thought = ""
        if think_start != -1 and think_end != -1:
            thought = response_text[think_start + len("<think>") : think_end].strip()

        action_start = response_text.find("<action>")
        action_end = response_text.find("</action>")
        if action_start != -1 and action_end != -1:
            action = (
                response_text[action_start + len("<action>") : action_end]
                .strip()
                .lower()
            )
        else:
            action = response_text.strip()[-30:].lower()
        return thought, action

    def _attach_el_tsbpo_action_token_mask(self, output: DataProto) -> None:

        responses = output.batch.get("responses")
        attention_mask = output.batch.get("attention_mask")
        if responses is None or attention_mask is None or responses.ndim != 2:
            raise ValueError(
                "EL-TSBPO Tree-GiGPO requires 2D responses and attention_mask"
            )
        response_length = responses.shape[-1]
        response_valid = attention_mask[:, -response_length:].to(dtype=torch.bool)
        action_mask = torch.zeros_like(responses, dtype=torch.bool)
        span_found = np.zeros(responses.shape[0], dtype=bool)

        open_ids = self.tokenizer.encode("<action>", add_special_tokens=False)
        close_ids = self.tokenizer.encode("</action>", add_special_tokens=False)
        if not open_ids or not close_ids:
            raise ValueError("tokenizer produced an empty EL-TSBPO action marker")

        def marker_matches(marker_ids: Sequence[int]) -> torch.Tensor:
            marker_length = len(marker_ids)
            if marker_length > response_length:
                return torch.zeros(
                    (responses.shape[0], 0),
                    dtype=torch.bool,
                    device=responses.device,
                )
            marker = torch.as_tensor(
                marker_ids, dtype=responses.dtype, device=responses.device
            )
            token_windows = responses.unfold(-1, marker_length, 1)
            valid_windows = response_valid.unfold(-1, marker_length, 1)
            return (token_windows == marker).all(dim=-1) & valid_windows.all(dim=-1)

        open_matches = marker_matches(open_ids)
        close_matches = marker_matches(close_ids)
        for row in range(responses.shape[0]):
            open_positions = torch.nonzero(open_matches[row], as_tuple=False).reshape(
                -1
            )
            if open_positions.numel() == 0:
                continue
            open_position = int(open_positions[0].item())
            action_start = open_position + len(open_ids)
            close_positions = torch.nonzero(close_matches[row], as_tuple=False).reshape(
                -1
            )
            valid_close_positions = close_positions[close_positions >= action_start]
            if valid_close_positions.numel() == 0:
                continue
            action_end = int(valid_close_positions[0].item())
            if action_end <= action_start:
                continue
            action_mask[row, action_start:action_end] = response_valid[
                row, action_start:action_end
            ]
            span_found[row] = True

        output.batch["el_tsbpo_action_token_mask"] = action_mask
        output.non_tensor_batch["el_tsbpo_action_span_found"] = span_found

    def _build_branch_prompt(
        self,
        task: str,
        memory_steps: List[Dict[str, str]],
        current_obs: str,
        admissible_commands: List[str],
        history_length: int = 5,
    ) -> str:
        is_webshop = "webshop" in str(self.config.env.get("env_name", "")).lower()
        if is_webshop:
            reformatted = "\n".join(f"'{action}'," for action in admissible_commands)
        else:
            reformatted = "\n ".join(
                f"'{action}'" for action in admissible_commands if action != "help"
            )

        if not memory_steps or history_length <= 0:
            if is_webshop:
                prompt = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=task,
                    current_observation=current_obs,
                    available_actions=reformatted,
                )
            else:
                prompt = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=current_obs,
                    admissible_actions=reformatted,
                )
        else:
            recent = memory_steps[-history_length:]
            start_step = len(memory_steps) - len(recent) + 1
            lines = []
            for j, step in enumerate(recent):
                step_num = start_step + j
                lines.append(
                    f"[Observation {step_num}: '{step.get('observation', '')}', "
                    f"Action {step_num}: '{step.get('action', '')}']"
                )
            action_history = "\n".join(lines)
            common = {
                "task_description": task,
                "step_count": len(memory_steps),
                "history_length": len(recent),
                "action_history": action_history,
                "current_step": len(memory_steps) + 1,
                "current_observation": current_obs,
            }
            if is_webshop:
                prompt = WEBSHOP_TEMPLATE.format(
                    **common,
                    available_actions=reformatted,
                )


                if len(prompt) > 13000:
                    prompt = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=task,
                        current_observation=current_obs,
                        available_actions=reformatted,
                    )
            else:
                prompt = ALFWORLD_TEMPLATE.format(
                    **common,
                    admissible_actions=reformatted,
                )

        return prompt

    def _project_branch_actions(self, responses, admissible_pools):

        is_webshop = "webshop" in str(self.config.env.get("env_name", "")).lower()
        if not is_webshop:
            from agent_system.environments.env_package.alfworld.projection import (
                alfworld_projection,
            )

            return alfworld_projection(
                list(responses), admissible_pools, return_details=True
            )

        from agent_system.environments.env_package.webshop.projection import (
            webshop_projection,
        )

        return webshop_projection(
            list(responses), admissible_pools, return_details=True
        )

    def _tokenize_prompts_for_generation(self, prompts: List[str]) -> DataProto:
        apply_chat_template_kwargs = self.config.data.get(
            "apply_chat_template_kwargs", {}
        )
        max_prompt_len = self.config.data.max_prompt_length
        truncation = self.config.data.truncation

        processed = []
        identity_metadata = self._context_metadata(apply_chat_template_kwargs)
        for prompt in prompts:
            chat = np.array([{"content": prompt, "role": "user"}])
            prompt_with_chat_template = self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
                **apply_chat_template_kwargs,
            )
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=self.tokenizer,
                max_length=max_prompt_len,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=truncation,
            )
            position_ids = compute_position_id_with_mask(attention_mask)
            raw_prompt_ids = self.tokenizer.encode(
                prompt_with_chat_template, add_special_tokens=False
            )
            if len(raw_prompt_ids) > max_prompt_len:
                if truncation == "left":
                    raw_prompt_ids = raw_prompt_ids[-max_prompt_len:]
                elif truncation == "right":
                    raw_prompt_ids = raw_prompt_ids[:max_prompt_len]

            processed.append(
                {
                    "input_ids": input_ids[0],
                    "attention_mask": attention_mask[0],
                    "position_ids": position_ids[0],
                    "raw_prompt_ids": np.array(raw_prompt_ids, dtype=object),
                    "context_hash": context_hash(
                        input_ids[0],
                        attention_mask[0],
                        position_ids[0],
                        metadata=identity_metadata,
                    ),
                    "context_token_ids": input_ids[0][attention_mask[0].bool()]
                    .detach()
                    .cpu()
                    .tolist(),
                    "context_position_ids": position_ids[0][
                        ..., attention_mask[0].bool()
                    ]
                    .detach()
                    .cpu()
                    .tolist(),
                    "decision_context": prompt_with_chat_template,
                    "policy_version": self._active_policy_version,
                    "decode_config_hash": self._active_decode_config_hash
                    or self._decode_config_hash({}),
                }
            )

        batch = collate_fn(processed)
        gen_input = DataProto.from_single_dict(data=batch)
        gen_input.meta_info["rollout_policy_version"] = self._active_policy_version
        return gen_input

    def _generate_branch_responses(
        self,
        actor_rollout_wg,
        prompts: List[str],
        temperature: Optional[float] = None,
    ) -> Tuple[Optional[DataProto], List[str]]:
        if not prompts:
            return None, []

        tokenize_started = time.perf_counter()
        gen_input = self._tokenize_prompts_for_generation(prompts)
        self._el_tsbpo_generation_stats["tokenize_seconds"] += (
            time.perf_counter() - tokenize_started
        )
        if temperature is not None:
            gen_input.meta_info["temperature"] = temperature


            decode_digest = self._decode_config_hash(gen_input.meta_info)
            gen_input.non_tensor_batch["decode_config_hash"] = np.asarray(
                [decode_digest] * len(prompts), dtype=object
            )

        cfg = self._get_el_tsbpo_config()
        synchronous_frontier = _is_truthy_flag(
            cfg.get("branch_synchronous_frontier", False)
        )
        local_token_packing = _is_truthy_flag(
            cfg.get("branch_local_token_packing", True)
        )
        rollout_cfg = getattr(
            getattr(self.config, "actor_rollout_ref", {}), "rollout", {}
        )
        token_budget = int(cfg.get("branch_token_budget", 0) or 0)
        if token_budget <= 0:
            token_budget = int(rollout_cfg.get("max_num_batched_tokens", 8192))
        max_rows = int(cfg.get("branch_max_generation_batch", 0) or 0)
        max_response = int(
            rollout_cfg.get(
                "response_length",
                getattr(self.config, "data", {}).get("max_response_length", 512),
            )
        )
        raw_prompt_rows = gen_input.non_tensor_batch.get("raw_prompt_ids")
        raw_prompt_rows = [] if raw_prompt_rows is None else raw_prompt_rows
        prompt_lengths = [len(row) if row is not None else 0 for row in raw_prompt_rows]


        batches = _branch_generation_batches(
            prompt_lengths,
            token_budget=token_budget,
            max_response_tokens=max_response,
            max_rows=max_rows,
            synchronous_frontier=synchronous_frontier,
            local_token_packing=local_token_packing,
        )
        if not batches:
            batches = [list(range(len(prompts)))]

        combined_chunks: List[DataProto] = []
        stats = self._el_tsbpo_generation_stats
        stats["branch_synchronous_frontier"] = int(synchronous_frontier)
        stats["branch_local_token_packing"] = int(local_token_packing)
        stats["calls"] += 1
        stats["batches"] += len(batches)
        stats["max_batch_rows"] = max(
            stats["max_batch_rows"], max((len(batch) for batch in batches), default=0)
        )
        stats["token_budget"] = token_budget

        for row_indices in batches:
            batch_input = gen_input.select_idxs(row_indices)
            batch_input_padded, pad_size = pad_dataproto_to_divisor(
                batch_input, actor_rollout_wg.world_size
            )
            batch_prompt_tokens = sum(prompt_lengths[i] for i in row_indices)
            batch_rows = len(row_indices)
            stats["batch_rows"] += batch_rows
            stats["batch_prompt_tokens"] += batch_prompt_tokens
            stats["batch_planned_tokens"] += batch_prompt_tokens + (
                max_response * batch_rows
            )
            stats["branch"]["batch_count"] += 1
            stats["branch"]["batch_rows"] += batch_rows
            stats["branch"]["batch_prompt_tokens"] += batch_prompt_tokens
            stats["branch"]["batch_planned_tokens"] += batch_prompt_tokens + (
                max_response * batch_rows
            )
            output_padded = self._generate_sequences_observed(
                actor_rollout_wg,
                batch_input_padded,
                kind="branch",
                logical_rows=batch_rows,
                prompt_tokens=batch_prompt_tokens,
                padded_rows=len(batch_input_padded.batch),
            )
            output = unpad_dataproto(output_padded, pad_size=pad_size)


            gen_keys = set(batch_input.batch.keys())
            out_keys = set(output.batch.keys())
            overlap_keys = list(gen_keys & out_keys)
            if overlap_keys:
                batch_input.pop(batch_keys=overlap_keys)
            combined_chunks.append(batch_input.union(output))

        combined = DataProto.concat(combined_chunks)
        text_actions = self.tokenizer.batch_decode(
            combined.batch["responses"], skip_special_tokens=True
        )
        return combined, text_actions

    def _generate_sequences_observed(
        self,
        actor_rollout_wg,
        prompts: DataProto,
        *,
        kind: str,
        logical_rows: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        padded_rows: Optional[int] = None,
    ) -> DataProto:

        started = time.perf_counter()
        output = actor_rollout_wg.generate_sequences(prompts)
        elapsed = time.perf_counter() - started

        stats = self._el_tsbpo_generation_stats
        kind_stats = stats.get(kind)
        if kind_stats is None:
            kind_stats = stats[kind] = self._new_generation_kind_stats()
        rows = int(logical_rows if logical_rows is not None else len(prompts.batch))
        physical_rows = int(
            padded_rows if padded_rows is not None else len(prompts.batch)
        )
        prompt_count = int(prompt_tokens or 0)
        response_count = 0
        if hasattr(output, "batch") and "responses" in output.batch:
            response_mask = output.batch.get("attention_mask")
            response_rows = min(rows, int(output.batch["responses"].shape[0]))
            if response_mask is not None:
                response_width = int(output.batch["responses"].shape[-1])
                response_count = int(
                    response_mask[:response_rows, -response_width:].sum().item()
                )
            else:
                response_count = int(output.batch["responses"][:response_rows].numel())

        stats["worker_calls"] += 1
        stats["rows"] += rows
        stats["prompt_tokens"] += prompt_count
        stats["wall_seconds"] += elapsed
        stats["response_tokens"] += response_count
        stats["padded_rows"] += max(0, physical_rows - rows)
        kind_stats["worker_calls"] += 1
        kind_stats["calls"] += 1
        kind_stats["rows"] += rows
        if kind == "natural":
            kind_stats["batch_count"] += 1
            kind_stats["batch_rows"] += rows
            kind_stats["batch_prompt_tokens"] += prompt_count
        kind_stats["prompt_tokens"] += prompt_count
        kind_stats["wall_seconds"] += elapsed
        kind_stats["response_tokens"] += response_count
        kind_stats["padded_rows"] += max(0, physical_rows - rows)
        return output

    def vanilla_multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
    ) -> DataProto:

        batch_size = len(gen_batch.batch)


        self._active_policy_version = self._rollout_policy_version(gen_batch.meta_info)
        self._active_decode_config_hash = self._decode_config_hash(gen_batch.meta_info)


        root_gamefiles = gen_batch.non_tensor_batch.pop("el_tsbpo_root_gamefile", None)
        if root_gamefiles is not None:
            obs, infos = envs.reset_allocated_roots(root_gamefiles)
        else:
            obs, infos = envs.reset(
                kwargs=gen_batch.non_tensor_batch.pop("env_kwargs", None)
            )

        lenght_obs = len(obs["text"]) if obs["text"] is not None else len(obs["image"])
        assert len(gen_batch.batch) == lenght_obs, (
            f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        )

        preset_uid = gen_batch.non_tensor_batch.get("el_tsbpo_root_uid")
        if preset_uid is not None and len(preset_uid) == batch_size:

            uid_batch = np.asarray(preset_uid, dtype=object)
        elif self.config.env.rollout.n > 0:
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else:
            uid = str(uuid.uuid4())
            uid_batch = np.array(
                [uid for _ in range(len(gen_batch.batch))], dtype=object
            )
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array(
            [str(uuid.uuid4()) for _ in range(batch_size)], dtype=object
        )


        executed_action_history: List[List[str]] = [[] for _ in range(batch_size)]
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        natural_model_rows_generated = 0
        natural_env_rows_stepped = 0
        natural_env_rounds = 0

        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)
            n_active = int(active_masks.sum())
            if n_active == 0:
                break


            active_indices = np.flatnonzero(active_masks).tolist()
            active_gen_batch = gen_batch.select_idxs(active_indices)
            active_obs = self._select_observation_rows(obs, active_indices)
            preprocess_started = time.perf_counter()
            batch = self.preprocess_batch(gen_batch=active_gen_batch, obs=active_obs)
            self._el_tsbpo_generation_stats["tokenize_seconds"] += (
                time.perf_counter() - preprocess_started
            )
            active_batch_size = len(active_indices)
            natural_model_rows_generated += active_batch_size
            natural_env_rows_stepped += active_batch_size
            natural_env_rounds += 1

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            prompt_tokens = int(batch.batch["attention_mask"].sum().item())
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info


            batch_input_padded, pad_size = pad_dataproto_to_divisor(
                batch_input, actor_rollout_wg.world_size
            )
            batch_output_padded = self._generate_sequences_observed(
                actor_rollout_wg,
                batch_input_padded,
                kind="natural",
                logical_rows=active_batch_size,
                prompt_tokens=prompt_tokens,
                padded_rows=len(batch_input_padded.batch),
            )
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch["uid"] = uid_batch[active_indices]
            batch.non_tensor_batch["traj_uid"] = traj_uid[active_indices]

            batch = batch.union(batch_output)

            active_text_actions = self.tokenizer.batch_decode(
                batch.batch["responses"], skip_special_tokens=True
            )


            text_actions = [""] * batch_size
            for local_idx, global_idx in enumerate(active_indices):
                text_actions[global_idx] = active_text_actions[local_idx]


            batch.non_tensor_batch["action_text"] = np.asarray(
                [
                    self._extract_thought_and_action(text)[1]
                    for text in active_text_actions
                ],
                dtype=object,
            )


            policy_version = self._rollout_policy_version(gen_batch.meta_info)
            decode_digest = self._decode_config_hash(gen_batch.meta_info)
            context_digests = batch.non_tensor_batch.get(
                "context_hash", np.asarray([""] * active_batch_size, dtype=object)
            )
            env_backend = getattr(envs, "envs", None)
            admissible_rows = getattr(env_backend, "get_admissible_commands", None)
            row = batch.non_tensor_batch
            row.setdefault(
                "policy_version",
                np.asarray([policy_version] * active_batch_size, dtype=object),
            )
            row.setdefault(
                "decode_config_hash",
                np.asarray([decode_digest] * active_batch_size, dtype=object),
            )
            if not isinstance(row["policy_version"], np.ndarray):
                row["policy_version"] = np.asarray(
                    [policy_version] * active_batch_size,
                    dtype=object,
                )
            if not isinstance(row["decode_config_hash"], np.ndarray):
                row["decode_config_hash"] = np.asarray(
                    [decode_digest] * active_batch_size,
                    dtype=object,
                )
            row.setdefault(
                "decision_key",
                np.asarray([None] * active_batch_size, dtype=object),
            )
            row.setdefault(
                "policy_group_key",
                np.asarray([None] * active_batch_size, dtype=object),
            )
            row.setdefault(
                "env_key",
                np.asarray([None] * active_batch_size, dtype=object),
            )
            row.setdefault(
                "env_snapshot_hash",
                np.asarray([None] * active_batch_size, dtype=object),
            )
            row.setdefault(
                "env_snapshot_ref",
                np.asarray([None] * active_batch_size, dtype=object),
            )
            if "decision_admissible_commands" not in row:
                admissible_values = np.empty(active_batch_size, dtype=object)
                for admissible_row in range(active_batch_size):
                    admissible_values[admissible_row] = []
                row["decision_admissible_commands"] = admissible_values
            row["dps_task_text"] = np.asarray([""] * active_batch_size, dtype=object)
            for local_idx, global_idx in enumerate(active_indices):
                gamefile = None
                if (
                    hasattr(envs, "gamefile")
                    and envs.gamefile is not None
                    and global_idx < len(envs.gamefile)
                ):
                    gamefile = envs.gamefile[global_idx]
                task_scope = ""
                if (
                    hasattr(envs, "tasks")
                    and envs.tasks is not None
                    and global_idx < len(envs.tasks)
                ):
                    task_scope = str(envs.tasks[global_idx])
                row["dps_task_text"][local_idx] = task_scope
                admissible_commands = (
                    list(admissible_rows[global_idx] or [])
                    if admissible_rows is not None and global_idx < len(admissible_rows)
                    else []
                )
                snapshot_ref = {
                    "gamefile": gamefile,
                    "action_history": list(executed_action_history[global_idx]),
                    "step_count": len(executed_action_history[global_idx]),
                    "seed": self.config.env.get("seed", None),


                    "anchor_obs": str(batch.non_tensor_batch["anchor_obs"][local_idx]),
                    "admissible_commands": admissible_commands,
                }
                snapshot_digest = environment_snapshot_hash(**snapshot_ref)
                state_digest = environment_state_hash(
                    task_scope=task_scope,
                    gamefile=gamefile,
                    anchor_obs=snapshot_ref["anchor_obs"],
                    admissible_commands=admissible_commands,
                    step_count=snapshot_ref["step_count"],
                )
                context_digest = str(context_digests[local_idx])
                decision_digest = decision_key(
                    context_digest,
                    snapshot_digest,
                    runtime_state={"task_scope": task_scope},
                )
                row["policy_version"][local_idx] = policy_version
                row["decode_config_hash"][local_idx] = decode_digest
                row["decision_key"][local_idx] = decision_digest
                row["policy_group_key"][local_idx] = (
                    policy_group_key(decision_digest, policy_version, decode_digest)
                    if policy_version != "unknown"
                    else None
                )


                row["env_key"][local_idx] = state_digest
                row["env_snapshot_hash"][local_idx] = snapshot_digest
                row["env_snapshot_ref"][local_idx] = snapshot_ref
                row["decision_admissible_commands"][local_idx] = list(
                    admissible_commands
                )


            active_text_actions = [text_actions[index] for index in active_indices]
            env_step_started = time.perf_counter()
            next_obs, rewards, dones, infos = envs.step_active(
                active_indices, active_text_actions
            )
            self._el_tsbpo_generation_stats["env_step_seconds"] += (
                time.perf_counter() - env_step_started
            )

            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:

                dones = dones.squeeze(1)

            rewards_np = torch_to_numpy(rewards)
            dones_np = torch_to_numpy(dones).astype(bool)
            active_infos = [infos[i] for i in active_indices]
            active_rewards = rewards_np[active_indices]
            active_dones = dones_np[active_indices]

            if infos and "is_action_valid" in infos[0]:
                batch.non_tensor_batch["is_action_valid"] = np.array(
                    [info["is_action_valid"] for info in active_infos], dtype=bool
                )
            else:
                batch.non_tensor_batch["is_action_valid"] = np.ones(
                    active_batch_size, dtype=bool
                )
            batch.non_tensor_batch["is_action_format_valid"] = np.array(
                [
                    info.get(
                        "is_action_format_valid", info.get("is_action_valid", True)
                    )
                    for info in active_infos
                ],
                dtype=bool,
            )
            batch.non_tensor_batch["is_action_admissible"] = np.array(
                [
                    info.get("is_action_admissible", info.get("is_action_valid", True))
                    for info in active_infos
                ],
                dtype=bool,
            )

            if infos and "tool_calling" in infos[0]:
                tool_callings[active_indices] += np.array(
                    [info["tool_calling"] for info in active_infos], dtype=np.float32
                )


            episode_rewards[active_indices] += active_rewards
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, (
                f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            )
            batch.non_tensor_batch["rewards"] = np.asarray(active_rewards, dtype=object)
            batch.non_tensor_batch["active_masks"] = np.ones(
                active_batch_size, dtype=bool
            )


            for local_idx, (global_idx, info) in enumerate(
                zip(active_indices, active_infos)
            ):
                executed = (
                    info.get("executed_action") if isinstance(info, dict) else None
                )
                if executed is None:
                    executed = self._extract_thought_and_action(
                        text_actions[global_idx]
                    )[1]


                batch.non_tensor_batch["action_text"][local_idx] = str(executed)
                executed_action_history[global_idx].append(str(executed))


            batch_list: list[dict] = to_list_of_dict(batch)
            for local_idx, global_idx in enumerate(active_indices):

                batch_list[local_idx]["turn_idx"] = len(total_batch_list[global_idx])
                next_anchor_rows = (
                    next_obs.get("anchor", [])
                    if isinstance(next_obs, Mapping)
                    else next_obs
                )
                next_anchor = (
                    next_anchor_rows[global_idx]
                    if global_idx < len(next_anchor_rows)
                    else ""
                )
                batch_list[local_idx]["next_anchor_obs"] = str(
                    next_anchor if next_anchor is not None else ""
                )
                batch_list[local_idx]["el_tsbpo_node_is_terminal"] = bool(
                    active_dones[local_idx]
                )
                batch_list[local_idx]["dps_env_done"] = bool(active_dones[local_idx])
                batch_list[local_idx]["dps_env_truncated"] = bool(
                    active_infos[local_idx].get("TimeLimit.truncated", False)
                    or active_infos[local_idx].get("truncated", False)
                )
                total_batch_list[global_idx].append(batch_list[local_idx])
                total_infos[global_idx].append(infos[global_idx])


            is_done[active_indices] = np.logical_or(
                is_done[active_indices], active_dones
            )


            obs = next_obs


            if is_done.all():
                break

        success: Dict[str, np.ndarray] = envs.success_evaluator(
            total_infos=total_infos,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
        )


        self._last_natural_model_rows_generated = int(natural_model_rows_generated)
        self._last_natural_model_rows_max = int(batch_size * self.config.env.max_steps)
        self._last_natural_env_rows_stepped = int(natural_env_rows_stepped)
        self._last_natural_env_rows_full = int(batch_size * natural_env_rounds)

        return (
            total_batch_list,
            episode_rewards,
            episode_lengths,
            success,
            traj_uid,
            tool_callings,
        )


    def _run_branch_rollouts(
        self,
        branch_info: List[Dict[str, Any]],
        branch_envs,
        actor_rollout_wg,
        all_nodes: Dict[str, Any],
        k_nat: int,
        max_steps: int,
        history_length: int,
        total_batch_list: List[List[Dict]],
        episode_rewards: np.ndarray,
        episode_lengths: np.ndarray,
        success: Dict[str, np.ndarray],
        tool_callings: np.ndarray,
    ) -> List[Dict]:
        """Run replayed natural continuations from selected states."""
        from agent_system.multi_turn_rollout.tree_node import TreeNode as _TreeNode

        n_branches = len(branch_info)
        branch_cfg = self._get_el_tsbpo_config()
        continuous_refill = _is_truthy_flag(
            branch_cfg.get("branch_continuous_refill", False)
        ) and all("spawn_spec" in item for item in branch_info)
        synchronous_frontier = _is_truthy_flag(
            branch_cfg.get("branch_synchronous_frontier", False)
        )
        horizon_bucket_size = max(
            1, int(branch_cfg.get("branch_horizon_bucket_size", 4))
        )

        def _horizon_bucket(index: int) -> int:
            node = branch_info[index]["branch_node"]
            remaining = max(
                0,
                int(max_steps) - (int(node.depth) + int(branch_step_count[index])),
            )
            return remaining // horizon_bucket_size

        def _order_by_horizon(indices: List[int]) -> List[int]:

            ordered = sorted(
                indices,
                key=lambda index: (
                    _horizon_bucket(index),
                    -max(
                        0,
                        int(max_steps)
                        - int(branch_info[index]["branch_node"].depth)
                        - int(branch_step_count[index]),
                    ),
                    int(index),
                ),
            )
            buckets = {_horizon_bucket(index) for index in ordered}
            stats = self._el_tsbpo_generation_stats
            stats["branch_horizon_bucket_count"] += len(buckets)
            stats["branch_horizon_bucket_rows"] += len(ordered)
            stats["branch_horizon_bucket_max_rows"] = max(
                stats["branch_horizon_bucket_max_rows"],
                max(
                    (
                        sum(_horizon_bucket(i) == bucket for i in ordered)
                        for bucket in buckets
                    ),
                    default=0,
                ),
            )
            return ordered


        branch_dones = (
            [False] * n_branches
            if continuous_refill
            else [bi["worker_idx"] < 0 for bi in branch_info]
        )
        branch_started = [False] * n_branches
        branch_memory: List[List[Dict[str, str]]] = []
        branch_action_history: List[List[str]] = []
        branch_responses: List[Optional[str]] = [None] * n_branches
        branch_actions: List[Optional[str]] = [None] * n_branches
        branch_action_valid: List[bool] = [False] * n_branches
        branch_action_format_valid: List[bool] = [False] * n_branches
        branch_action_admissible: List[bool] = [False] * n_branches
        branch_step_count = [0] * n_branches
        branch_total_reward = [0.0] * n_branches
        branch_won = [False] * n_branches
        branch_traj_uid = [str(uuid.uuid4()) for _ in range(n_branches)]
        branch_prev_node_ids: List[Optional[str]] = [None] * n_branches

        branch_prompt_obs: List[str] = [""] * n_branches

        pending_gen_data: List[Optional[Dict]] = [None] * n_branches

        for i, bi in enumerate(branch_info):
            bn = bi["branch_node"]
            branch_memory.append([dict(s) for s in bn.memory_steps])
            branch_action_history.append(list(bn.action_history))
            branch_prev_node_ids[i] = bn.node_id

            branch_prompt_obs[i] = str(
                bi.get("branch_anchor_obs")
                or (
                    bn.memory_steps[-1].get("observation", "")
                    if bn.memory_steps
                    else ""
                )
            )

        branch_current_admissibles: List[List[str]] = [[] for _ in range(n_branches)]
        branch_current_prompt_commands: List[List[str]] = [[] for _ in range(n_branches)]
        branch_replayed_observations: List[str] = [""] * n_branches


        pending_order = sorted(
            range(n_branches),
            key=lambda index: (
                -max(
                    0,
                    int(max_steps) - int(branch_info[index]["branch_node"].depth),
                ),
                int(index),
            ),
        )

        def _record_replay_state(indices: List[int]) -> None:
            if not indices:
                return
            worker_rows = [branch_info[index]["worker_idx"] for index in indices]
            admissible_rows = branch_envs.get_admissible_commands_batch(worker_rows)
            prompt_rows = (
                branch_envs.get_prompt_commands_batch(worker_rows)
                if hasattr(branch_envs, "get_prompt_commands_batch")
                else admissible_rows
            )
            if hasattr(branch_envs, "get_observations_batch"):
                observation_rows = branch_envs.get_observations_batch(worker_rows)
            else:


                observation_rows = [""] * len(worker_rows)
            for local_index, branch_index in enumerate(indices):
                branch_started[branch_index] = True
                branch_current_admissibles[branch_index] = list(
                    admissible_rows[local_index] or []
                )
                branch_current_prompt_commands[branch_index] = list(
                    prompt_rows[local_index] or []
                )
                branch_replayed_observations[branch_index] = str(
                    observation_rows[local_index]
                    if local_index < len(observation_rows)
                    else ""
                )

        def _fill_free_worker_slots(
            *, refill: bool, preferred_buckets: Optional[Sequence[int]] = None
        ) -> List[int]:

            started: List[int] = []
            while continuous_refill:
                free_count = max(
                    0, int(branch_envs.max_workers) - int(branch_envs.num_active)
                )
                if free_count <= 0:
                    break
                pending = [
                    index
                    for index in pending_order
                    if not branch_started[index] and not branch_dones[index]
                ]
                if not pending:
                    break
                preferred = set(int(bucket) for bucket in (preferred_buckets or ()))
                selected = [
                    index
                    for index in pending
                    if not preferred or _horizon_bucket(index) in preferred
                ][:free_count]
                if len(selected) < min(free_count, len(pending)):
                    selected.extend(index for index in pending if index not in selected)
                    selected = selected[:free_count]
                spawn_started = time.perf_counter()
                worker_indices = branch_envs.spawn_branches_parallel(
                    [branch_info[index]["spawn_spec"] for index in selected]
                )
                spawn_key = (
                    "branch_refill_spawn_seconds"
                    if refill
                    else "branch_initial_spawn_seconds"
                )
                stats = self._el_tsbpo_generation_stats
                stats[spawn_key] = stats.get(spawn_key, 0.0) + (
                    time.perf_counter() - spawn_started
                )
                successful: List[int] = []
                for branch_index, worker_index in zip(selected, worker_indices):
                    branch_info[branch_index]["worker_idx"] = int(worker_index)
                    if worker_index < 0:
                        branch_dones[branch_index] = True
                        continue
                    successful.append(branch_index)
                _record_replay_state(successful)
                started.extend(successful)
                stats = self._el_tsbpo_generation_stats
                stats["branch_refill_spawn_calls"] += int(refill)
                stats["branch_refill_branches"] += len(successful) if refill else 0
            return started

        if continuous_refill:
            alive_indices = _fill_free_worker_slots(refill=False)
        else:
            alive_indices = [i for i in range(n_branches) if not branch_dones[i]]
            _record_replay_state(alive_indices)
        alive_indices = _order_by_horizon(alive_indices)
        initial_admissibles = [
            branch_current_admissibles[index] for index in alive_indices
        ]
        initial_prompt_commands = [
            branch_current_prompt_commands[index] for index in alive_indices
        ]

        admissible_map = {i: idx for idx, i in enumerate(alive_indices)}
        nat_alive = alive_indices
        nat_prompts = []
        for i in nat_alive:
            bi = branch_info[i]
            bn = bi["branch_node"]
            current_obs = branch_prompt_obs[i]
            prompt_commands = (
                initial_prompt_commands[admissible_map[i]] if i in admissible_map else []
            )
            prompt = self._build_branch_prompt(
                task=bn.task,
                memory_steps=branch_memory[i],
                current_obs=current_obs,
                admissible_commands=prompt_commands,
                history_length=history_length,
            )
            nat_prompts.append(prompt)

        if nat_prompts:
            self._el_tsbpo_generation_stats["branch_active_frontier_rows"] += len(
                nat_prompts
            )
            self._el_tsbpo_generation_stats["branch_active_frontier_batches"] += 1
            self._el_tsbpo_generation_stats["branch_active_frontier_max_rows"] = max(
                self._el_tsbpo_generation_stats["branch_active_frontier_max_rows"],
                len(nat_prompts),
            )
            combined_gen, text_actions = self._generate_branch_responses(
                actor_rollout_wg, nat_prompts
            )
            nat_gen_data = (
                to_list_of_dict(combined_gen) if combined_gen is not None else []
            )
            nat_pools = [
                initial_admissibles[admissible_map[i]] if i in admissible_map else []
                for i in nat_alive
            ]
            projected_actions, projected_valids, validity = (
                self._project_branch_actions(text_actions, nat_pools)
            )
            for idx, i in enumerate(nat_alive):
                branch_responses[i] = text_actions[idx]
                branch_actions[i] = projected_actions[idx]
                branch_action_valid[i] = bool(projected_valids[idx])
                branch_action_format_valid[i] = bool(validity["format_valid"][idx])
                branch_action_admissible[i] = bool(validity["admissible_valid"][idx])
                if idx < len(nat_gen_data):
                    pending_gen_data[i] = nat_gen_data[idx]

        all_turn_data: List[Dict] = []


        initial_ready = [
            i
            for i in range(n_branches)
            if not branch_dones[i] and branch_actions[i] is not None
        ]
        submit_started = time.perf_counter()
        step_requests = (
            branch_envs.submit_step_batch(
                [branch_info[i]["worker_idx"] for i in initial_ready],
                [branch_actions[i] for i in initial_ready],
            )
            if hasattr(branch_envs, "submit_step_batch")
            else []
        )
        self._el_tsbpo_generation_stats["branch_env_submit_seconds"] += (
            time.perf_counter() - submit_started
        )
        if not step_requests and initial_ready:


            wait_started = time.perf_counter()
            step_requests = [
                {"branch_idx": i, "result": result}
                for i, result in zip(
                    initial_ready,
                    branch_envs.step_batch(
                        [branch_info[i]["worker_idx"] for i in initial_ready],
                        [branch_actions[i] for i in initial_ready],
                    ),
                )
            ]
            self._el_tsbpo_generation_stats["branch_env_wait_seconds"] += (
                time.perf_counter() - wait_started
            )
        request_branch_by_worker = {
            int(branch_info[i]["worker_idx"]): i for i in initial_ready
        }
        ready_batch_size = max(
            1,
            min(
                int(branch_envs.max_workers),
                int(
                    self._get_el_tsbpo_config().get(
                        "branch_ready_queue_batch_size", branch_envs.max_workers
                    )
                ),
            ),
        )
        ready_timeout_ms = max(
            0.0,
            float(self._get_el_tsbpo_config().get("branch_ready_queue_timeout_ms", 25)),
        )


        for step in range(max_steps * max(1, n_branches)):
            if not step_requests:
                break
            if (
                hasattr(branch_envs, "wait_step_requests")
                and step_requests
                and "future" in step_requests[0]
            ):
                wait_started = time.perf_counter()
                frontier_target = _branch_frontier_target(
                    len(step_requests),
                    ready_batch_size,
                    synchronous_frontier=synchronous_frontier,
                )
                completed, step_requests = branch_envs.wait_step_requests(
                    step_requests,
                    num_returns=frontier_target,
                    wait_all=synchronous_frontier,
                    wait_for_target=not synchronous_frontier,
                    drain_ready=not synchronous_frontier,
                    timeout=(ready_timeout_ms / 1000.0)
                    if not synchronous_frontier and ready_timeout_ms > 0
                    else None,
                )
                self._el_tsbpo_generation_stats["branch_env_wait_seconds"] += (
                    time.perf_counter() - wait_started
                )
                alive = [
                    request_branch_by_worker[int(item["worker_idx"])]
                    for item in completed
                ]
                results = [item["result"] for item in completed]
            else:
                completed = step_requests
                step_requests = []
                alive = [item["branch_idx"] for item in completed]
                results = [item["result"] for item in completed]
            self._el_tsbpo_generation_stats["branch_ready_batches"] += 1
            self._el_tsbpo_generation_stats["branch_ready_rows"] += len(alive)
            self._el_tsbpo_generation_stats["branch_ready_max_rows"] = max(
                self._el_tsbpo_generation_stats["branch_ready_max_rows"],
                len(alive),
            )
            if not alive:
                continue


            newly_done = []
            for idx_in_alive, i in enumerate(alive):
                obs_i, reward_i, done_i, info_i = results[idx_in_alive]
                bi = branch_info[i]
                bn = bi["branch_node"]


                pre_action_history = list(branch_action_history[i])
                branch_step_count[i] += 1
                step_num = bn.depth + branch_step_count[i]
                branch_total_reward[i] += reward_i


                thought_i, _ = (
                    self._extract_thought_and_action(branch_responses[i])
                    if branch_responses[i]
                    else ("", "")
                )
                sampled_action = branch_actions[i] if branch_actions[i] else "pass"


                action_taken = str(sampled_action)
                obs_str = obs_i if obs_i else ""
                if not isinstance(obs_str, str):
                    obs_str = str(obs_str)
                memory_step = {
                    "thought": thought_i,
                    "action": action_taken,


                    "observation": str(branch_prompt_obs[i] or ""),
                }
                branch_memory[i].append(memory_step)
                branch_action_history[i].append(action_taken)


                bt = "natural"
                parent_node_id = branch_prev_node_ids[i]
                node_id = (
                    f"{bi['branch_group_id']}:branch{bi['branch_idx']}:"
                    f"n{branch_step_count[i]}"
                )
                node = _TreeNode(
                    node_id=node_id,
                    parent_id=parent_node_id,
                    depth=step_num,
                    task=bn.task,
                    gamefile=bn.gamefile,
                    action_history=list(branch_action_history[i]),
                    memory_steps=[dict(s) for s in branch_memory[i]],
                    branch_type=bt,
                    branch_group_id=bi["branch_group_id"],
                )
                all_nodes[node_id] = node
                if parent_node_id in all_nodes:
                    if node_id not in all_nodes[parent_node_id].children:
                        all_nodes[parent_node_id].children.append(node_id)
                branch_prev_node_ids[i] = node_id


                gen_data = pending_gen_data[i]
                pending_gen_data[i] = None

                context_digest = (
                    str(gen_data.get("context_hash"))
                    if gen_data is not None and gen_data.get("context_hash") is not None
                    else ""
                )
                policy_version = (
                    str(gen_data.get("policy_version", self._active_policy_version))
                    if gen_data is not None
                    else self._active_policy_version
                )
                decode_digest = (
                    str(
                        gen_data.get(
                            "decode_config_hash", self._active_decode_config_hash or ""
                        )
                    )
                    if gen_data is not None
                    else (self._active_decode_config_hash or "")
                )
                identity_audit = branch_step_count[i] == 1
                snapshot_ref = {
                    "gamefile": bn.gamefile,
                    "action_history": pre_action_history,
                    "step_count": len(pre_action_history),
                    "seed": self.config.env.get("seed", None),
                    "anchor_obs": branch_prompt_obs[i],
                    "admissible_commands": list(branch_current_admissibles[i]),
                }
                snapshot_digest = environment_snapshot_hash(**snapshot_ref)
                decision_digest = (
                    decision_key(
                        context_digest,
                        snapshot_digest,
                        runtime_state={"task_scope": str(bn.task)},
                    )
                    if context_digest
                    else None
                )
                policy_group_digest = (
                    policy_group_key(decision_digest, policy_version, decode_digest)
                    if decision_digest and decode_digest and policy_version != "unknown"
                    else None
                )
                actual_env_key = environment_state_hash(
                    task_scope=str(bn.task),
                    gamefile=bn.gamefile,
                    anchor_obs=(
                        branch_replayed_observations[i]
                        if identity_audit and branch_replayed_observations[i]
                        else branch_prompt_obs[i]
                    ),
                    admissible_commands=branch_current_admissibles[i],
                    step_count=len(pre_action_history),
                )
                expected_context = str(bi.get("expected_context_hash", "") or "")
                expected_decision = str(bi.get("expected_decision_key", "") or "")
                expected_policy_group = str(
                    bi.get("expected_policy_group_key", "") or ""
                )
                expected_env_key = str(bi.get("expected_env_key", "") or "")
                context_matches_parent = (
                    bool(
                        expected_context
                        and context_digest
                        and expected_context == context_digest
                    )
                    if identity_audit and expected_context
                    else None
                )
                decision_matches_parent = (
                    bool(
                        expected_decision
                        and decision_digest
                        and expected_decision == decision_digest
                    )
                    if identity_audit and expected_decision
                    else None
                )
                policy_group_matches_parent = (
                    bool(
                        expected_policy_group
                        and policy_group_digest
                        and expected_policy_group == policy_group_digest
                    )
                    if identity_audit and expected_policy_group
                    else None
                )
                env_matches_parent = (
                    bool(expected_env_key and expected_env_key == actual_env_key)
                    if identity_audit and expected_env_key
                    else None
                )
                replay_observation_matches_parent = None
                replay_admissible_matches_parent = None
                if identity_audit:
                    expected_anchor_obs = str(bi.get("expected_anchor_obs", "") or "")
                    replay_observation_matches_parent = (
                        bool(
                            expected_anchor_obs
                            and branch_replayed_observations[i]
                            and expected_anchor_obs == branch_replayed_observations[i]
                        )
                        if expected_anchor_obs
                        else None
                    )
                    expected_admissible = list(
                        bi.get("expected_admissible_commands", []) or []
                    )
                    replay_admissible_matches_parent = (
                        sorted(map(str, expected_admissible))
                        == sorted(map(str, branch_current_admissibles[i]))
                        if expected_admissible
                        else None
                    )
                    if replay_observation_matches_parent is False:
                        env_matches_parent = False
                    if replay_admissible_matches_parent is False:
                        env_matches_parent = False


                env_idx = bi["env_idx"]
                parent_uid = (
                    total_batch_list[env_idx][0].get("uid", "")
                    if total_batch_list[env_idx]
                    else ""
                )
                is_first_turn = branch_step_count[i] == 1

                turn_data = {
                    "input_ids": gen_data["input_ids"]
                    if gen_data
                    else torch.zeros(1, dtype=torch.long),
                    "attention_mask": gen_data["attention_mask"]
                    if gen_data
                    else torch.ones(1, dtype=torch.long),
                    "position_ids": gen_data["position_ids"]
                    if gen_data
                    else torch.zeros(1, dtype=torch.long),
                    "responses": gen_data["responses"]
                    if gen_data
                    else torch.zeros(1, dtype=torch.long),
                    "generation_finish_reason": (
                        gen_data.get("generation_finish_reason") if gen_data else None
                    ),
                    "generation_response_length": (
                        gen_data.get("generation_response_length", 0) if gen_data else 0
                    ),

                    "rollout_log_probs": (
                        gen_data["rollout_log_probs"]
                        if gen_data is not None and "rollout_log_probs" in gen_data
                        else torch.zeros_like(
                            gen_data["responses"], dtype=torch.float32
                        )
                        if gen_data is not None and "responses" in gen_data
                        else torch.zeros(1, dtype=torch.float32)
                    ),
                    "raw_prompt_ids": gen_data["raw_prompt_ids"]
                    if gen_data
                    else np.array([], dtype=object),
                    "anchor_obs": branch_prompt_obs[i],
                    "next_anchor_obs": obs_str,
                    "dps_task_text": str(bn.task),
                    "agent_dps_prefix_history": [
                        {
                            "observation": str(s.get("observation", "")),
                            "action": str(s.get("action", "")),
                        }
                        for s in branch_memory[i][:-1][
                            -int(self.config.env.history_length) :
                        ]
                    ]
                    if int(self.config.env.history_length) > 0
                    else [],
                    "context_hash": context_digest,
                    "context_token_ids": gen_data.get("context_token_ids", [])
                    if gen_data
                    else [],
                    "context_position_ids": gen_data.get("context_position_ids", [])
                    if gen_data
                    else [],
                    "policy_version": policy_version,
                    "decode_config_hash": decode_digest,
                    "decision_key": decision_digest,
                    "policy_group_key": policy_group_digest,
                    "env_key": actual_env_key,
                    "env_snapshot_hash": snapshot_digest,
                    "env_snapshot_ref": snapshot_ref,
                    "context_matches_parent": context_matches_parent,
                    "decision_matches_parent": decision_matches_parent,
                    "policy_group_matches_parent": policy_group_matches_parent,
                    "env_matches_parent": env_matches_parent,
                    "replay_observation_matches_parent": replay_observation_matches_parent,
                    "replay_admissible_matches_parent": replay_admissible_matches_parent,
                    "action_text": action_taken,
                    "uid": parent_uid,
                    "traj_uid": branch_traj_uid[i],
                    "rewards": float(reward_i),
                    "active_masks": True,
                    "is_action_valid": bool(branch_action_valid[i]),
                    "is_action_format_valid": bool(branch_action_format_valid[i]),
                    "is_action_admissible": bool(branch_action_admissible[i]),
                    "data_source": "text",
                    "index": env_idx,
                    "episode_rewards": 0.0,
                    "episode_lengths": 0,
                    "tool_callings": 0.0,
                    "success_rate": 0.0,
                    "trajectory_success": 0.0,
                    "dps_env_done": bool(done_i),
                    "dps_env_truncated": bool(
                        info_i.get("TimeLimit.truncated", False)
                        or info_i.get("truncated", False)
                    )
                    if isinstance(info_i, dict)
                    else False,

                    "branch_group_id": bi["branch_group_id"] if is_first_turn else None,
                    "is_counterfactual": False,
                    "parent_step_idx": bi["parent_step_idx"] if is_first_turn else -1,
                    "parent_traj_uid": bi["parent_traj_uid"],


                    "turn_idx": step_num - 1,
                    "el_tsbpo_parent_node_id": parent_node_id,
                    "el_tsbpo_node_id": node_id,
                    "el_tsbpo_node_is_terminal": False,
                    "el_tsbpo_terminal_reward": 0.0,
                    "el_tsbpo_bare_rollout_id": bi["parent_traj_uid"],
                    "el_tsbpo_branch_type": "natural",
                    "el_tsbpo_snapshot_id": snapshot_digest,
                    "el_tsbpo_terminal_reason": "",
                    "el_tsbpo_anchor_node_id": bi.get("el_tsbpo_anchor_node_id", ""),
                    "el_tsbpo_group_memberships": [],
                    "el_tsbpo_allocation_provenance": "branch",


                    "el_tsbpo_quarantined": True,
                    "el_tsbpo_policy_version": policy_version,
                }
                all_turn_data.append(turn_data)

                if isinstance(info_i, dict):
                    branch_current_admissibles[i] = list(
                        info_i.get("admissible_commands", []) or []
                    )
                    branch_current_prompt_commands[i] = list(
                        info_i.get(
                            "prompt_commands", branch_current_admissibles[i]
                        ) or []
                    )


                branch_prompt_obs[i] = obs_str

                if done_i:
                    node.is_leaf = True
                    node.won = bool(info_i.get("won", False)) if info_i else False
                    node.reward = 10.0 if node.won else 0.0
                    turn_data["el_tsbpo_node_is_terminal"] = True


                    turn_data["el_tsbpo_terminal_reward"] = 10.0 if node.won else 0.0
                    turn_data["el_tsbpo_terminal_reason"] = (
                        "success"
                        if node.won
                        else "timeout"
                        if turn_data["dps_env_truncated"]
                        else "failure"
                    )
                    branch_dones[i] = True
                    branch_won[i] = node.won
                    newly_done.append(i)
                elif branch_step_count[i] >= max_steps - bn.depth:
                    node.is_leaf = True
                    node.won = False
                    node.reward = 0.0
                    turn_data["el_tsbpo_node_is_terminal"] = True
                    turn_data["el_tsbpo_terminal_reward"] = 0.0
                    turn_data["el_tsbpo_terminal_reason"] = "timeout"
                    branch_dones[i] = True
                    newly_done.append(i)


            for i in newly_done:
                for td in all_turn_data:
                    if td["traj_uid"] == branch_traj_uid[i]:


                        td["episode_rewards"] = 10.0 if branch_won[i] else 0.0
                        td["el_tsbpo_environment_return"] = branch_total_reward[i]
                        td["episode_lengths"] = branch_step_count[i]
                        td["success_rate"] = 1.0 if branch_won[i] else 0.0
                        td["trajectory_success"] = 1.0 if branch_won[i] else 0.0

            newly_started: List[int] = []
            if continuous_refill:


                for branch_index in newly_done:
                    worker_index = int(branch_info[branch_index]["worker_idx"])
                    if worker_index >= 0:
                        branch_envs.close_branch(worker_index)
                    branch_info[branch_index]["worker_idx"] = -1
                surviving_buckets = [
                    _horizon_bucket(index) for index in alive if not branch_dones[index]
                ]
                newly_started = _fill_free_worker_slots(
                    refill=True,
                    preferred_buckets=surviving_buckets,
                )


            still_alive = [i for i in alive if not branch_dones[i]]
            still_alive.extend(newly_started)
            still_alive = _order_by_horizon(still_alive)
            if not still_alive:
                continue

            self._el_tsbpo_generation_stats["branch_active_frontier_rows"] += len(
                still_alive
            )
            self._el_tsbpo_generation_stats["branch_active_frontier_batches"] += 1
            self._el_tsbpo_generation_stats["branch_active_frontier_max_rows"] = max(
                self._el_tsbpo_generation_stats["branch_active_frontier_max_rows"],
                len(still_alive),
            )


            admissibles = [branch_current_admissibles[i] for i in still_alive]
            prompt_commands = [branch_current_prompt_commands[i] for i in still_alive]

            next_prompts = []
            for j, i in enumerate(still_alive):
                bi = branch_info[i]
                bn = bi["branch_node"]
                cur_obs = branch_prompt_obs[i]
                prompt = self._build_branch_prompt(
                    task=bn.task,
                    memory_steps=branch_memory[i],
                    current_obs=cur_obs,
                    admissible_commands=(
                        prompt_commands[j] if j < len(prompt_commands) else []
                    ),
                    history_length=history_length,
                )
                next_prompts.append(prompt)


            combined_next, next_text_actions = self._generate_branch_responses(
                actor_rollout_wg, next_prompts
            )
            next_gen_data = (
                to_list_of_dict(combined_next) if combined_next is not None else []
            )
            projected_actions, projected_valids, validity = (
                self._project_branch_actions(next_text_actions, admissibles)
            )


            for j, i in enumerate(still_alive):
                branch_responses[i] = next_text_actions[j]
                branch_actions[i] = projected_actions[j]
                branch_action_valid[i] = bool(projected_valids[j])
                branch_action_format_valid[i] = bool(validity["format_valid"][j])
                branch_action_admissible[i] = bool(validity["admissible_valid"][j])
                if j < len(next_gen_data):
                    pending_gen_data[i] = next_gen_data[j]

            next_ready = [
                i
                for i in still_alive
                if not branch_dones[i] and branch_actions[i] is not None
            ]
            if next_ready:
                submit_started = time.perf_counter()
                new_requests = (
                    branch_envs.submit_step_batch(
                        [branch_info[i]["worker_idx"] for i in next_ready],
                        [branch_actions[i] for i in next_ready],
                    )
                    if hasattr(branch_envs, "submit_step_batch")
                    else []
                )
                self._el_tsbpo_generation_stats["branch_env_submit_seconds"] += (
                    time.perf_counter() - submit_started
                )
                for request, branch_idx in zip(new_requests, next_ready):
                    request_branch_by_worker[int(request["worker_idx"])] = branch_idx
                step_requests.extend(new_requests)


        for i in range(n_branches):
            if not branch_dones[i]:
                branch_rows = [
                    td for td in all_turn_data if td["traj_uid"] == branch_traj_uid[i]
                ]
                if branch_rows:
                    branch_rows[-1]["el_tsbpo_node_is_terminal"] = True
                    branch_rows[-1]["el_tsbpo_terminal_reward"] = 0.0
                    if not branch_rows[-1].get("el_tsbpo_terminal_reason"):
                        branch_rows[-1]["el_tsbpo_terminal_reason"] = "timeout"
                for td in all_turn_data:
                    if td["traj_uid"] == branch_traj_uid[i]:
                        td["episode_rewards"] = 0.0
                        td["el_tsbpo_environment_return"] = branch_total_reward[i]
                        td["episode_lengths"] = branch_step_count[i]
                        td["success_rate"] = 0.0
                        td["trajectory_success"] = 0.0


        if continuous_refill:
            for branch_index, branch in enumerate(branch_info):
                worker_index = int(branch.get("worker_idx", -1))
                if worker_index >= 0:
                    branch_envs.close_branch(worker_index)
                    branch["worker_idx"] = -1
        return all_turn_data


    @staticmethod
    def _concat_branch_turn_data(
        base: DataProto, branch_turn_data: List[Dict]
    ) -> DataProto:

        branch_batch = DataProto.from_single_dict(data=collate_fn(branch_turn_data))

        def normalize_row_objects(proto: DataProto) -> None:

            for key, value in list(proto.non_tensor_batch.items()):
                array = np.asarray(value, dtype=object)
                if array.ndim <= 1:
                    continue
                packed = np.empty(array.shape[0], dtype=object)
                for row in range(array.shape[0]):
                    packed[row] = array[row].tolist()
                proto.non_tensor_batch[key] = packed

        normalize_row_objects(base)
        normalize_row_objects(branch_batch)

        def normalize_memberships(proto: DataProto) -> None:
            value = proto.non_tensor_batch.get("el_tsbpo_group_memberships")
            if value is None:
                return
            array = np.asarray(value, dtype=object)
            rows: list[list[dict]] = []
            for raw in array.reshape(array.shape[0], -1):
                memberships: list[dict] = []
                for item in raw.tolist():
                    if isinstance(item, Mapping):
                        memberships.append(dict(item))
                    elif isinstance(item, (list, tuple)):
                        memberships.extend(
                            dict(entry) for entry in item if isinstance(entry, Mapping)
                        )
                rows.append(memberships)
            normalized = np.empty(len(rows), dtype=object)
            for row, memberships in enumerate(rows):
                normalized[row] = memberships
            proto.non_tensor_batch["el_tsbpo_group_memberships"] = normalized

        normalize_memberships(base)
        normalize_memberships(branch_batch)
        bare_bsz = len(base.batch)
        branch_bsz = len(branch_batch.batch)
        bare_tensor_keys = set(base.batch.keys())
        branch_tensor_keys = set(branch_batch.batch.keys())
        for key in bare_tensor_keys | branch_tensor_keys:
            if key in bare_tensor_keys and key not in branch_tensor_keys:
                ref = base.batch[key]
                branch_batch.batch[key] = torch.zeros(
                    (branch_bsz,) + ref.shape[1:], dtype=ref.dtype, device=ref.device
                )
            elif key not in bare_tensor_keys and key in branch_tensor_keys:
                ref = branch_batch.batch[key]
                base.batch[key] = torch.zeros(
                    (bare_bsz,) + ref.shape[1:], dtype=ref.dtype, device=ref.device
                )
        bare_nt_keys = set(base.non_tensor_batch.keys())
        branch_nt_keys = set(branch_batch.non_tensor_batch.keys())
        for key in bare_nt_keys | branch_nt_keys:
            if key in bare_nt_keys and key not in branch_nt_keys:
                branch_batch.non_tensor_batch[key] = _pad_non_tensor(
                    base.non_tensor_batch[key], branch_bsz, key
                )
            elif key not in bare_nt_keys and key in branch_nt_keys:
                base.non_tensor_batch[key] = _pad_non_tensor(
                    branch_batch.non_tensor_batch[key], bare_bsz, key
                )
        merged = DataProto.concat([base, branch_batch])
        for key in (
            "is_action_valid",
            "is_action_format_valid",
            "is_action_admissible",
            "active_masks",
        ):
            value = merged.non_tensor_batch.get(key)
            if isinstance(value, np.ndarray) and value.dtype == object:
                merged.non_tensor_batch[key] = value.astype(bool)
        return merged


    def _get_el_tsbpo_config(self):
        return self.config.algorithm.get("el_tsbpo", {})

    def is_agent_dps_enabled(self) -> bool:
        return self._get_el_tsbpo_config().get("allocation_backend") == "agent_dps"

    def _get_agent_dps_memory(self):
        if self._agent_dps_memory is not None:
            return self._agent_dps_memory
        from agent_system.multi_turn_rollout.agent_dps import AgentDPSStateMemory

        cfg = self._get_el_tsbpo_config()
        checkpoint_root = self.config.trainer.get("default_local_dir", "./checkpoints")
        path = os.path.join(
            str(checkpoint_root),
            str(cfg.get("state_memory_filename", "agent_dps_state.json")),
        )
        alpha = float(cfg.get("state_prior_alpha", 1.0))
        beta = float(cfg.get("state_prior_beta", 1.0))
        decay = float(cfg.get("state_memory_decay", 0.90))
        root_decay = float(cfg.get("agent_dps_root_decay", decay))
        max_states = int(cfg.get("state_memory_max_states", 100000))
        max_edges = int(cfg.get("agent_dps_graph_max_edges", 200000))
        horizon = max(0, int(cfg.get("state_memory_horizon", 32)))
        hmm_prior_alpha = float(cfg.get("agent_dps_hmm_prior_alpha", 1.0))
        hmm_transition_decay = float(cfg.get("agent_dps_hmm_transition_decay", 0.9))
        memory_mode = str(cfg.get("state_memory_mode", "fresh")).lower()
        resume_path = getattr(self, "_agent_dps_checkpoint_path", None)
        if resume_path:
            memory_mode = "resume"
        elif hasattr(self, "_agent_dps_checkpoint_path"):


            memory_mode = "fresh"
        if memory_mode in {"fresh", "new", "reset"}:


            self._agent_dps_memory = AgentDPSStateMemory(
                alpha=alpha,
                beta=beta,
                decay=decay,
                root_decay=root_decay,
                max_states=max_states,
                max_age_steps=horizon,
                hmm_prior_alpha=hmm_prior_alpha,
                hmm_transition_decay=hmm_transition_decay,
                max_edges=max_edges,
            )


            self._agent_dps_memory.save(path)
        elif memory_mode in {"resume", "load"}:
            self._agent_dps_memory = AgentDPSStateMemory.load_or_create(
                resume_path or path,
                alpha=alpha,
                beta=beta,
                decay=decay,
                root_decay=root_decay,
                max_states=max_states,
                max_age_steps=horizon,
                hmm_prior_alpha=hmm_prior_alpha,
                hmm_transition_decay=hmm_transition_decay,
                max_edges=max_edges,
            )
        else:
            raise ValueError(
                "el_tsbpo.state_memory_mode must be one of fresh or resume; "
                f"got {memory_mode!r}"
            )
        self._agent_dps_memory_path = path
        print(
            f"[EL-TSBPO-DPS] state_memory mode={memory_mode} "
            f"forgetting=soft branch_decay={decay} root_decay={root_decay} "
            f"states={len(self._agent_dps_memory.states)} "
            f"path={path}",
            flush=True,
        )
        return self._agent_dps_memory

    def _agent_dps_root_parameters(self, task_ref, key, step):
        memory = self._get_agent_dps_memory()
        cfg = self._get_el_tsbpo_config()
        if not bool(cfg.get("agent_dps_root_similarity_enabled", False)):
            alpha, beta = memory.posterior_parameters(key, step)
            shared = dict(neighbors=0.0, max_similarity=0.0, shared_mass=0.0)
        else:
            from agent_system.multi_turn_rollout.task_similarity import (
                TaskSimilarityPrior,
            )

            prior = getattr(self, "_agent_dps_task_prior", None)
            if prior is None:
                prior = TaskSimilarityPrior(
                    str(cfg["agent_dps_root_embedding_cache_dir"]),
                    threshold=float(
                        cfg.get("agent_dps_root_similarity_threshold", 0.95)
                    ),
                    top_k=int(cfg.get("agent_dps_root_similarity_top_k", 8)),
                    max_strength=float(
                        cfg.get("agent_dps_root_similarity_max_strength", 4.0)
                    ),
                )
                self._agent_dps_task_prior = prior
            alpha, beta, shared = prior.parameters(
                memory, str(task_ref), str(key), int(step)
            )
        if cfg.get("agent_dps_probability_calibration", False):
            cache = getattr(self, "_agent_dps_root_forecasts", {})
            cache_key = (int(step), str(key))
            if cache_key not in cache:
                direct = memory.evidence_counts(key, step)
                semantic = (
                    shared.get("semantic_success_mass", 0.0),
                    shared.get("semantic_failure_mass", 0.0),
                )
                model = self._get_agent_dps_probability_calibrator("root")
                forecast = model.forecast(str(task_ref), direct, semantic)
                context = dict(
                    task=str(task_ref),
                    group=str(key),
                    history_success=float(direct[0]),
                    history_failure=float(direct[1]),
                    current_success=0.0,
                    current_failure=0.0,
                    direct_success=float(direct[0]),
                    direct_failure=float(direct[1]),
                    semantic_success=float(semantic[0]),
                    semantic_failure=float(semantic[1]),
                )
                cache = {k: v for k, v in cache.items() if k[0] == int(step)}
                cache[cache_key] = forecast, context
                self._agent_dps_root_forecasts = cache
            forecast, _ = cache[cache_key]

            strength = max(
                1e-6,
                forecast.mean * (1 - forecast.mean) / max(forecast.variance, 1e-12) - 1,
            )
            alpha, beta = (
                max(1e-9, forecast.mean * strength),
                max(1e-9, (1 - forecast.mean) * strength),
            )
            shared = {**shared, "probability_calibrated": True, **forecast.info}
        return alpha, beta, shared

    def _get_agent_dps_probability_calibrator(self, stage):
        from agent_system.multi_turn_rollout.dps_probability_calibration import (
            ProbabilityCalibrator,
        )

        memory = self._get_agent_dps_memory()
        cache = getattr(self, "_agent_dps_probability_models", {})
        key = (stage, int(self._el_tsbpo_step))
        if key not in cache:
            cache = {k: v for k, v in cache.items() if k[1] == key[1]}
            cache[key] = ProbabilityCalibrator(
                memory.probability_calibration,
                key[1],
                stage,
                memory.root_decay if stage == "root" else memory.decay,
            )
            self._agent_dps_probability_models = cache
        return cache[key]

    def _get_agent_dps_branch_similarity(self):
        cfg = self._get_el_tsbpo_config().get("agent_dps_branch_similarity", {})
        if not cfg.get("enabled", False):
            return None
        prior = getattr(self, "_agent_dps_branch_similarity", None)
        if prior is None:
            from agent_system.multi_turn_rollout.branch_similarity import (
                BranchSimilarity,
            )

            options = dict(cfg)
            options.pop("enabled", None)
            model_path, cache_dir = options.pop("model_path"), options.pop("cache_dir")
            prior = BranchSimilarity(str(model_path), str(cache_dir), **options)
            self._agent_dps_branch_similarity = prior
        return prior

    def _agent_dps_root_frontier(self, key, step, alpha, beta):
        memory = self._get_agent_dps_memory()
        if self._get_el_tsbpo_config().get("agent_dps_probability_calibration", False):
            forecast, _ = self._agent_dps_root_forecasts[int(step), str(key)]
            return forecast.mixed_curve(int(self.config.env.rollout.n))[-1]
        if not bool(
            self._get_el_tsbpo_config().get("agent_dps_root_similarity_enabled", False)
        ):
            return float(memory.dps_belief(key, step)[1])
        from agent_system.multi_turn_rollout.task_similarity import (
            root_frontier_probability,
        )

        return root_frontier_probability(
            memory, key, step, alpha, beta, int(self.config.env.rollout.n)
        )

    def _write_agent_dps_allocation(self, stage, payload):
        cfg = self._get_el_tsbpo_config()
        directory = cfg.get("agent_dps_allocation_diagnostics_dir")
        if not directory:
            return
        os.makedirs(str(directory), exist_ok=True)
        path = os.path.join(
            str(directory), f"step_{int(self._el_tsbpo_step):06d}_{stage}.json"
        )
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(
                {"allocation_step": int(self._el_tsbpo_step), **payload},
                stream,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(temporary, path)

    def save_agent_dps_checkpoint(self, checkpoint_dir: str) -> None:
        if not self.is_agent_dps_enabled() or self._agent_dps_memory is None:
            return
        filename = str(
            self._get_el_tsbpo_config().get(
                "state_memory_filename", "agent_dps_state.json"
            )
        )
        self._agent_dps_memory.save(os.path.join(checkpoint_dir, filename))

    def load_agent_dps_checkpoint(self, checkpoint_dir: str, global_step: int) -> None:
        self._el_tsbpo_step = int(global_step)
        if not self.is_agent_dps_enabled():
            return
        filename = str(
            self._get_el_tsbpo_config().get(
                "state_memory_filename", "agent_dps_state.json"
            )
        )
        path = os.path.join(checkpoint_dir, filename)
        self._agent_dps_checkpoint_path = path if os.path.isfile(path) else None
        self._agent_dps_memory = None
        self._agent_dps_last_save_step = int(global_step) - 1
        if self._agent_dps_checkpoint_path:
            print(
                f"[EL-TSBPO-DPS] resume checkpoint step={global_step} path={path}",
                flush=True,
            )
        else:
            print(
                f"[EL-TSBPO-DPS] legacy checkpoint step={global_step} has no DPS archive; "
                "starting fresh DPS evidence at the restored training step",
                flush=True,
            )

    def _save_agent_dps_memory(self, *, force: bool = False) -> None:
        if self._agent_dps_memory is None or not self._agent_dps_memory.dirty:
            return
        cfg = self._get_el_tsbpo_config()
        interval = max(1, int(cfg.get("state_memory_save_interval", 1)))
        if (
            not force
            and self._el_tsbpo_step - self._agent_dps_last_save_step < interval
        ):
            return
        self._agent_dps_memory.prune(self._el_tsbpo_step)
        self._agent_dps_memory.save(self._agent_dps_memory_path)
        self._agent_dps_last_save_step = self._el_tsbpo_step
        self._agent_dps_memory_dirty = False

    def _collect_el_tsbpo_candidates(
        self,
        total_batch_list: List[List[Dict]],
        traj_uid: np.ndarray,
        envs: EnvironmentManagerBase,
        source_env_indices: Optional[List[int]] = None,
    ):

        from el_tsbpo.allocator import AnchorCandidate

        cfg = self._get_el_tsbpo_config()
        min_depth = max(0, int(cfg.get("min_branch_depth", 0)))
        max_depth = int(cfg.get("max_branch_depth", self.config.env.max_steps - 1))
        candidates: List[AnchorCandidate] = []
        for env_idx, steps in enumerate(total_batch_list):
            if not steps or env_idx >= len(traj_uid):
                continue
            source_env_idx = (
                int(source_env_indices[env_idx])
                if source_env_indices is not None and env_idx < len(source_env_indices)
                else env_idx
            )
            if (
                not hasattr(envs, "gamefile")
                or envs.gamefile is None
                or source_env_idx >= len(envs.gamefile)
            ):
                continue
            gamefile = str(envs.gamefile[source_env_idx] or "")
            if not gamefile:
                continue
            effective_length = len(steps)
            for position, row in enumerate(steps):
                if not bool(row.get("active_masks", True)):
                    effective_length = position
                    break
            actions: List[str] = []
            action_statuses: List[str] = []
            thoughts: List[str] = []
            for row in steps[:effective_length]:
                action = row.get("action_text")
                response = row.get("responses", "")
                response_text = (
                    self.tokenizer.decode(response, skip_special_tokens=True)
                    if isinstance(response, torch.Tensor)
                    else str(response or "")
                )
                thought, parsed_action = self._extract_thought_and_action(response_text)
                actions.append(str(action if action is not None else parsed_action))
                if not bool(row.get("is_action_format_valid", True)):
                    action_statuses.append("format_invalid")
                elif not bool(row.get("is_action_admissible", True)):
                    action_statuses.append("illegal")
                else:
                    action_statuses.append("legal")
                thoughts.append(thought)

            for step_idx, row in enumerate(steps[:effective_length]):
                if step_idx < min_depth or step_idx > max_depth:
                    continue
                context = str(row.get("decision_context", "") or "")
                if not context:
                    token_ids = row.get("context_token_ids", [])
                    context = self.tokenizer.decode(
                        token_ids, skip_special_tokens=False
                    )
                exact_key = str(
                    row.get("policy_group_key")
                    or row.get("decision_key")
                    or f"{traj_uid[env_idx]}:{step_idx}"
                )
                task_uid = str(row.get("uid", ""))


                root_node_id = f"el_tsbpo:task:{row.get('uid', '')}:root"
                anchor_node_id = (
                    root_node_id
                    if step_idx == 0
                    else f"el_tsbpo:{traj_uid[env_idx]}:n{step_idx}"
                )


                parent_node_id = (
                    ""
                    if step_idx == 0
                    else (
                        root_node_id
                        if step_idx == 1
                        else f"el_tsbpo:{traj_uid[env_idx]}:n{step_idx - 1}"
                    )
                )
                memory_steps: List[Dict[str, str]] = []
                for previous in range(step_idx):


                    observation = (
                        str(steps[previous].get("anchor_obs", "") or "")
                        if previous < effective_length
                        else ""
                    )
                    memory_steps.append(
                        {
                            "thought": thoughts[previous],
                            "action": actions[previous],
                            "observation": observation,
                        }
                    )
                task = (
                    str(envs.tasks[source_env_idx])
                    if hasattr(envs, "tasks") and source_env_idx < len(envs.tasks)
                    else ""
                )
                raw_admissible = row.get("decision_admissible_commands", [])
                if isinstance(raw_admissible, np.ndarray):
                    raw_admissible = raw_admissible.tolist()
                candidates.append(
                    AnchorCandidate(
                        anchor_id=(
                            f"root_{task_uid}_{traj_uid[env_idx]}"
                            if step_idx == 0
                            else f"env{env_idx}_s{step_idx}_{exact_key[:12]}"
                        ),
                        trajectory_id=str(traj_uid[env_idx]),
                        task_uid=task_uid,
                        env_idx=env_idx,
                        step_idx=step_idx,
                        context=context,
                        gamefile=gamefile,
                        action_history=list(actions[:step_idx]),
                        memory_steps=memory_steps,
                        anchor_obs=str(row.get("anchor_obs", "") or ""),
                        admissible_commands=list(raw_admissible or []),
                        action_status_history=list(action_statuses[:step_idx]),
                        task=task,
                        context_hash=str(row.get("context_hash", "") or ""),
                        decision_key=str(row.get("decision_key", "") or ""),
                        policy_group_key=str(row.get("policy_group_key", "") or ""),
                        env_key=str(row.get("env_key", "") or ""),
                        node_id=anchor_node_id,
                        parent_node_id=parent_node_id,
                        current_action=actions[step_idx],
                        current_action_status=action_statuses[step_idx],
                        next_anchor_obs=str(row.get("next_anchor_obs", "") or ""),
                        is_terminal=bool(
                            row.get("el_tsbpo_node_is_terminal", False)
                            or step_idx == effective_length - 1
                        ),
                    )
                )
        return candidates

    def _allocate_agent_dps_roots(
        self,
        root_obs,
        root_infos,
        root_budget: int,
        root_budgets=None,
        root_scopes=None,
    ):
        """Allocate factual rollouts using calibrated root probabilities."""

        from agent_system.multi_turn_rollout.agent_dps import (
            AgentDPSNode,
            action_key,
            allocate_agent_dps,
            beta_mixed_utility,
            root_curriculum_key,
            state_key,
        )

        memory = self._get_agent_dps_memory()
        cfg = self._get_el_tsbpo_config()
        step = self._el_tsbpo_step
        nodes = []
        root_visits = []
        root_evidence_mass = []
        root_hmm_partial = []
        root_acquisition_scores = []
        root_details = []
        root_min_quota = max(1, int(cfg.get("agent_dps_root_min_quota", 2)))
        for index, (observation, info) in enumerate(zip(root_obs, root_infos)):
            gamefile = str(info.get("extra.gamefile", "") or "")


            scope = gamefile
            if not scope and root_scopes is not None and index < len(root_scopes):
                scope = root_scopes[index]
            key = str(
                memory.root_state_keys.get(str(scope), "") or root_curriculum_key(scope)
            )
            memory.root_state_keys[str(scope)] = key
            probability, uncertainty, evidence_mass, visit_count = memory.posterior(
                key, step
            )
            posterior_alpha, posterior_beta = memory.posterior_parameters(key, step)
            hmm_belief = memory.dps_belief(key, step)
            shared = dict(neighbors=0.0, max_similarity=0.0, shared_mass=0.0)
            if bool(cfg.get("agent_dps_root_similarity_enabled", False)) or cfg.get(
                "agent_dps_probability_calibration", False
            ):
                posterior_alpha, posterior_beta, shared = (
                    self._agent_dps_root_parameters(scope, key, step)
                )
                total = posterior_alpha + posterior_beta
                probability = posterior_alpha / total
                uncertainty = (
                    posterior_alpha * posterior_beta / (total * total * (total + 1))
                )
                hmm_belief = (
                    0.0,
                    self._agent_dps_root_frontier(
                        key, step, posterior_alpha, posterior_beta
                    ),
                    0.0,
                )
            root_details.append(
                dict(
                    task_ref=str(scope),
                    state_key=key,
                    probability=float(probability),
                    alpha=float(posterior_alpha),
                    beta=float(posterior_beta),
                    own_evidence_mass=float(evidence_mass),
                    **shared,
                )
            )
            root_curve = ()
            if cfg.get("agent_dps_probability_calibration", False):
                forecast, context = self._agent_dps_root_forecasts[int(step), str(key)]
                root_curve = forecast.mixed_curve(int(root_budget))
                root_details[-1]["calibration_context"] = context
            acquisition_score = beta_mixed_utility(posterior_alpha, posterior_beta, 2)
            if root_curve:
                acquisition_score = root_curve[min(2, len(root_curve) - 1)]
            tie_payload = (
                f"{int(cfg.get('agent_dps_tie_break_seed', 17))}:{int(step)}:{key}"
            ).encode("utf-8")
            tie_breaker = int.from_bytes(
                hashlib.sha256(tie_payload).digest()[:8], "big"
            ) / float(1 << 64)
            root_evidence_mass.append(float(evidence_mass))
            root_visits.append(float(visit_count))
            root_hmm_partial.append(float(hmm_belief[1]))
            root_acquisition_scores.append(float(acquisition_score))
            node = AgentDPSNode(
                state_key=key,
                task_uid=str(index),
                anchor_id=f"agent_dps_root_{index}",
                trajectory_id="",
                probability=probability,
                uncertainty=uncertainty,
                posterior_alpha=float(posterior_alpha),
                posterior_beta=float(posterior_beta),
                hmm_partial_probability=float(hmm_belief[1]),
                acquisition_score=1.0,
                historical_evidence_count=float(evidence_mass),
                tie_breaker=tie_breaker,
                representative=None,
                max_k=(
                    int(root_budgets[index])
                    if root_budgets is not None and index < len(root_budgets)
                    else int(root_budget)
                ),
                root_utility_curve=root_curve,
            )
            nodes.append(node)
        allocations, allocator_metrics, _ = allocate_agent_dps(
            nodes,
            int(root_budget),
            initial_branch_size=root_min_quota,
            min_new_branch=root_min_quota,
            max_k=int(root_budget),
            root_baseline=root_min_quota,
            root_mode=True,
        )
        root_allocator_metrics = {
            key.replace("agent_dps/allocator/", "agent_dps/root/", 1): value
            for key, value in allocator_metrics.items()
        }
        quotas = np.zeros(len(root_obs), dtype=np.int64)
        by_anchor = {node.anchor_id: index for index, node in enumerate(nodes)}
        for allocation in allocations:
            index = by_anchor.get(str(allocation.anchor_id))
            if index is not None:
                quotas[index] = int(allocation.continuation_count)
        probabilities = np.asarray(
            [float(node.probability) for node in nodes], dtype=np.float64
        )


        additions = []
        removals = []

        def root_utility(node, k):
            return (
                node.root_utility_curve[k]
                if node.root_utility_curve
                else beta_mixed_utility(node.posterior_alpha, node.posterior_beta, k)
            )

        for index, node in enumerate(nodes):
            quota = int(quotas[index])
            utility = root_utility(node, quota)
            if quota < node.max_k:
                additions.append((index, root_utility(node, quota + 1) - utility))
            if quota > root_min_quota:
                removals.append((index, utility - root_utility(node, quota - 1)))
        reallocation_gain = max(
            [0.0]
            + [
                gain - loss
                for recipient, gain in additions
                for donor, loss in removals
                if recipient != donor
            ]
        )
        root_metrics = {
            **root_allocator_metrics,
            "el_tsbpo/root/predictor_error": 0.0,
            "el_tsbpo/root/cold_start": float(
                bool(nodes) and not any(value > 0.0 for value in root_evidence_mass)
            ),
            "agent_dps/root/history_seen_fraction": float(
                np.mean(np.asarray(root_evidence_mass) > 0.0)
                if root_evidence_mass
                else 0.0
            ),
            "el_tsbpo/root/predicted_success_mean": float(probabilities.mean())
            if len(probabilities)
            else 0.0,
            "el_tsbpo/root/predicted_success_min": float(probabilities.min())
            if len(probabilities)
            else 0.0,
            "el_tsbpo/root/predicted_success_max": float(probabilities.max())
            if len(probabilities)
            else 0.0,
            "el_tsbpo/root/active_predicted_success_mean": float(
                probabilities[quotas > 0].mean() if np.any(quotas > 0) else 0.0
            ),
            "el_tsbpo/root/skipped_predicted_success_mean": float(
                probabilities[quotas == 0].mean() if np.any(quotas == 0) else 0.0
            ),
            "agent_dps/root/n_states": float(len(nodes)),
            "agent_dps/root/min_quota": float(root_min_quota),
            "agent_dps/root/floor_binding_fraction": float(
                np.mean(quotas == root_min_quota) if len(quotas) else 0.0
            ),
            "agent_dps/root/max_reallocation_gain": float(reallocation_gain),
            "agent_dps/root/floor_tasks": float(
                np.count_nonzero(quotas >= root_min_quota)
            ),
            "agent_dps/root/n_selected_states": float(len(allocations)),
            "agent_dps/root/quota_std": float(np.std(quotas)) if len(quotas) else 0.0,
            "agent_dps/root/p_std": float(np.std(probabilities))
            if len(probabilities)
            else 0.0,
            "agent_dps/root/seen_fraction": float(
                np.mean(np.asarray(root_visits) > 0.0) if root_visits else 0.0
            ),
            "agent_dps/root/mean_visits": float(
                np.mean(root_visits) if root_visits else 0.0
            ),
            "agent_dps/root/mean_evidence_mass": float(
                np.mean(root_evidence_mass) if root_evidence_mass else 0.0
            ),
            "agent_dps/root/hmm_partial_mean": float(
                np.mean(root_hmm_partial) if root_hmm_partial else 0.0
            ),
            "agent_dps/root/hmm_partial_std": float(
                np.std(root_hmm_partial) if root_hmm_partial else 0.0
            ),
            "agent_dps/root/acquisition_mean": float(
                np.mean(root_acquisition_scores) if root_acquisition_scores else 0.0
            ),
            "agent_dps/root/acquisition_std": float(
                np.std(root_acquisition_scores) if root_acquisition_scores else 0.0
            ),
        }
        for quota in sorted(set(int(value) for value in quotas)):
            root_metrics[f"agent_dps/root/quota_{quota}_tasks"] = float(
                np.count_nonzero(quotas == quota)
            )
        root_metrics.update(
            {
                "agent_dps/root/semantic_hit_fraction": float(
                    np.mean([r["shared_mass"] > 0 for r in root_details])
                )
                if root_details
                else 0.0,
                "agent_dps/root/shared_mass_mean": float(
                    np.mean([r["shared_mass"] for r in root_details])
                )
                if root_details
                else 0.0,
                "agent_dps/root/quota_uniform": float(len(set(quotas.tolist())) <= 1),
                "agent_dps/root/quota_min": float(quotas.min()) if len(quotas) else 0.0,
                "agent_dps/root/quota_max": float(quotas.max()) if len(quotas) else 0.0,
            }
        )
        if cfg.get("agent_dps_probability_calibration", False):
            self._agent_dps_root_calibration_quotas = {
                row["task_ref"]: int(quota) for row, quota in zip(root_details, quotas)
            }
            model = self._get_agent_dps_probability_calibrator("root")
            root_metrics.update(
                {
                    f"agent_dps/probability/root/{k}": float(v)
                    for k, v in model.diagnostics().items()
                }
            )
        if cfg.get("agent_dps_allocation_diagnostics_dir"):
            for row, quota in zip(root_details, quotas):
                row["quota"] = int(quota)
                if cfg.get("agent_dps_probability_calibration", False):
                    forecast, _ = self._agent_dps_root_forecasts[
                        int(step), row["state_key"]
                    ]
                    row["predicted_mixed_probability"] = forecast.mixed_curve(
                        int(quota)
                    )[-1]
            self._write_agent_dps_allocation(
                "root",
                {
                    "budget": int(root_budget),
                    "roots": root_details,
                    "metrics": root_metrics,
                },
            )
        if (
            len(probabilities) > 1
            and np.std(probabilities) > 1e-12
            and np.std(quotas) > 1e-12
        ):
            root_metrics["agent_dps/root/p_quota_correlation"] = float(
                np.corrcoef(probabilities, quotas)[0, 1]
            )
        else:
            root_metrics["agent_dps/root/p_quota_correlation"] = 0.0
        return quotas, root_metrics, probabilities

    def _refresh_agent_dps_root_candidates(self, envs, n_tasks: int):

        from agent_system.multi_turn_rollout.agent_dps import (
            AgentDPSCurriculumCandidate,
            root_curriculum_key,
            select_agent_dps_curriculum,
            state_key,
        )

        memory = self._get_agent_dps_memory()
        step = self._el_tsbpo_step
        cfg = self._get_el_tsbpo_config()
        sampler_backend = envs.envs
        if not getattr(self, "_agent_dps_task_sampler_restored", False):
            loader = getattr(
                sampler_backend, "load_fresh_task_sampler_state_dict", None
            )
            sampler_state = dict(getattr(memory, "task_sampler_state", {}) or {})
            if callable(loader) and sampler_state:
                loader(sampler_state)
            self._agent_dps_task_sampler_restored = True
        candidate_ratio = max(
            1.0, float(cfg.get("agent_dps_root_candidate_ratio", 8.0))
        )
        candidate_count = max(
            int(n_tasks), int(math.ceil(float(n_tasks) * candidate_ratio))
        )
        reference_sampler = getattr(sampler_backend, "sample_root_task_refs", None)
        candidate_refs = []
        history_candidate_count = 0
        if callable(reference_sampler):


            historical_candidates = []
            for task_ref, key in memory.root_state_keys.items():
                key = str(key or "")
                if not key or key not in memory.states:
                    continue
                _, _, evidence_mass, _ = memory.posterior(key, step)
                if evidence_mass <= 0.0:
                    continue
                history_frontier = float(memory.dps_belief(key, step)[1])
                if bool(cfg.get("agent_dps_root_similarity_enabled", False)) or cfg.get(
                    "agent_dps_probability_calibration", False
                ):
                    pa, pb, _ = self._agent_dps_root_parameters(task_ref, key, step)
                    history_frontier = self._agent_dps_root_frontier(key, step, pa, pb)
                historical_candidates.append(
                    AgentDPSCurriculumCandidate(
                        task_ref=str(task_ref),
                        state_key=key,
                        frontier_probability=history_frontier,
                        evidence_mass=float(evidence_mass),
                    )
                )
            history_limit = min(
                len(historical_candidates), max(1, candidate_count // 2)
            )
            if len(historical_candidates) > history_limit:
                historical_candidates, _ = select_agent_dps_curriculum(
                    historical_candidates,
                    history_limit,
                    exploration_fraction=float(
                        cfg.get("agent_dps_root_exploration_fraction", 0.30)
                    ),
                    temperature=float(
                        cfg.get("agent_dps_root_curriculum_temperature", 0.25)
                    ),
                    seed=(
                        int(cfg.get("agent_dps_tie_break_seed", 17))
                        + 1_000_003 * int(step)
                        + 7919
                    ),
                )
            historical_refs = [item.task_ref for item in historical_candidates]
            history_candidate_count = len(historical_refs)
            fresh_count = candidate_count - history_candidate_count
            fresh_refs = list(reference_sampler(fresh_count, historical_refs))
            candidate_refs = historical_refs + [
                str(task_ref) for task_ref in fresh_refs
            ]
            if len(candidate_refs) != len(set(candidate_refs)):
                raise RuntimeError(
                    "Agent-DPS root candidate pool contains duplicate tasks"
                )
        if candidate_refs:
            if bool(cfg.get("agent_dps_root_similarity_enabled", False)) or cfg.get(
                "agent_dps_probability_calibration", False
            ):
                texts = sampler_backend.get_root_task_texts(candidate_refs)

                first = str(candidate_refs[0])
                self._agent_dps_root_parameters(first, root_curriculum_key(first), step)
                if bool(cfg.get("agent_dps_root_similarity_enabled", False)):
                    self._agent_dps_task_prior.validate_texts(texts)
            curriculum_candidates = []
            curriculum_details = []
            for task_ref in candidate_refs:
                task_ref = str(task_ref)
                key = str(
                    memory.root_state_keys.get(task_ref, "")
                    or root_curriculum_key(task_ref)
                )
                if key:
                    _, _, evidence_mass, _ = memory.posterior(key, step)
                    frontier = memory.dps_belief(key, step)[1]
                else:
                    evidence_mass = 0.0
                    frontier = 1.0 / 3.0
                if bool(cfg.get("agent_dps_root_similarity_enabled", False)) or cfg.get(
                    "agent_dps_probability_calibration", False
                ):
                    pa, pb, shared = self._agent_dps_root_parameters(
                        task_ref, key, step
                    )
                    frontier = self._agent_dps_root_frontier(key, step, pa, pb)
                    curriculum_details.append(
                        dict(
                            task_ref=task_ref,
                            alpha=pa,
                            beta=pb,
                            probability=pa / (pa + pb),
                            frontier=frontier,
                            **shared,
                        )
                    )
                curriculum_candidates.append(
                    AgentDPSCurriculumCandidate(
                        task_ref=task_ref,
                        state_key=key,
                        frontier_probability=float(frontier),
                        evidence_mass=float(evidence_mass),
                    )
                )
            selected, curriculum_metrics = select_agent_dps_curriculum(
                curriculum_candidates,
                int(n_tasks),
                exploration_fraction=float(
                    cfg.get("agent_dps_root_exploration_fraction", 0.30)
                ),
                temperature=float(
                    cfg.get("agent_dps_root_curriculum_temperature", 0.25)
                ),
                seed=(
                    int(cfg.get("agent_dps_tie_break_seed", 17)) + 1_000_003 * int(step)
                ),
            )
            selected_refs = [item.task_ref for item in selected]
            if cfg.get("agent_dps_allocation_diagnostics_dir"):
                self._write_agent_dps_allocation(
                    "root_candidates",
                    {"candidates": curriculum_details, "selected_refs": selected_refs},
                )
            loader = getattr(sampler_backend, "load_root_candidates", None)
            if not callable(loader):
                raise RuntimeError(
                    "Agent-DPS dynamic curriculum requires load_root_candidates"
                )
            sampled_obs, sampled_infos = loader(selected_refs)
        else:


            selected_refs = []
            curriculum_metrics = {
                "agent_dps/root/curriculum_candidate_count": float(n_tasks),
                "agent_dps/root/curriculum_selected_count": float(n_tasks),
                "agent_dps/root/curriculum_fallback": 1.0,
            }
            sampler = getattr(
                sampler_backend,
                "sample_fresh_root_candidates",
                sampler_backend.sample_root_candidates,
            )
            sampled_obs, sampled_infos = sampler(int(n_tasks))
            candidate_count = int(n_tasks)
        sampler_state_fn = getattr(
            sampler_backend, "fresh_task_sampler_state_dict", None
        )
        if callable(sampler_state_fn):
            memory.task_sampler_state = dict(sampler_state_fn())
            memory.dirty = True
        canonicalize_roots = getattr(envs, "canonicalize_root_candidates", None)
        if callable(canonicalize_roots):
            sampled_obs, sampled_infos = canonicalize_roots(
                sampled_obs, sampled_infos
            )
        if len(sampled_obs) != n_tasks:
            raise RuntimeError(
                "Agent-DPS curriculum root loader returned "
                f"{len(sampled_obs)} candidates, expected {n_tasks}"
            )

        selected_history = 0
        selected_keys = set()
        normalized_infos = []
        for index, (observation, raw_info) in enumerate(
            zip(sampled_obs, sampled_infos)
        ):
            info = dict(raw_info or {})
            task_ref = str(info.get("extra.gamefile", "") or "")
            if not task_ref and index < len(selected_refs):
                task_ref = str(selected_refs[index])
                info["extra.gamefile"] = task_ref
            old_key = str(memory.root_state_keys.get(task_ref, "") or "")
            if old_key and old_key in memory.states:
                selected_history += 1
            key = root_curriculum_key(task_ref)
            if key in selected_keys:
                raise RuntimeError(
                    f"Agent-DPS curriculum produced a duplicate root state: {task_ref}"
                )
            selected_keys.add(key)
            memory.root_state_keys[task_ref] = key
            normalized_infos.append(info)
        memory.dirty = True
        root_obs = list(sampled_obs)
        root_infos = normalized_infos
        self._agent_dps_root_candidates = (list(root_obs), list(root_infos))
        self._agent_dps_root_candidate_step = step
        replay_metrics = {
            **curriculum_metrics,
            "agent_dps/root/history_candidate_injected_count": float(
                history_candidate_count
            ),
            "agent_dps/root/history_candidate_injected_fraction": float(
                history_candidate_count / max(1, candidate_count)
            ),
            "agent_dps/root/random_task_count": float(
                max(0, int(n_tasks) - selected_history)
            ),
            "agent_dps/root/dps_replay_task_count": float(selected_history),
            "agent_dps/root/replay_target_slots": 0.0,
            "agent_dps/root/replay_pool_size": float(len(memory.root_state_keys)),
            "agent_dps/root/replay_horizon": 0.0,
            "agent_dps/root/exact_history_hit_fraction": float(
                selected_history / max(1, int(n_tasks))
            ),
            "agent_dps/root/natural_random_revisit_count": 0.0,
            "agent_dps/root/dynamic_curriculum_enabled": 1.0,
        }
        sampler_metrics_fn = getattr(
            sampler_backend, "fresh_task_sampler_metrics", None
        )
        if callable(sampler_metrics_fn):
            replay_metrics.update(sampler_metrics_fn())
        return (
            root_obs,
            root_infos,
            selected_history,
            max(0, int(n_tasks) - selected_history),
            candidate_count,
            replay_metrics,
        )

    def _prepare_el_tsbpo_root_batch(
        self, gen_batch: DataProto, envs=None
    ) -> DataProto:
        """Select tasks and root quotas before actor generation."""
        cfg = self._get_el_tsbpo_config()
        n_tasks = len(gen_batch.batch)
        if n_tasks == 0:
            return gen_batch
        if not self.is_agent_dps_enabled():
            raise ValueError("EL-TSBPO requires the AgentDPS allocator")
        if envs is None or not hasattr(envs.envs, "sample_root_candidates"):
            raise RuntimeError("Root allocation requires replayable task candidates")
        root_budget = int(cfg["root_budget"])
        if root_budget < 2 or root_budget > envs.envs.worker_capacity:
            raise ValueError("Root budget exceeds the available worker pool")
        root_min_quota = max(1, int(cfg.get("agent_dps_root_min_quota", 2)))
        if root_budget < root_min_quota * n_tasks:
            raise ValueError("Root budget cannot cover the minimum task quota")
        (root_obs, root_infos, retained, replaced, sampled, replay_metrics) = (
            self._refresh_agent_dps_root_candidates(envs, n_tasks)
        )
        if len(root_obs) != n_tasks:
            raise RuntimeError("Root sampler returned the wrong number of tasks")
        root_scopes = gen_batch.non_tensor_batch.get("uid")
        if root_scopes is None:
            root_scopes = gen_batch.non_tensor_batch.get("index")
        quotas, allocation_metrics, probabilities = self._allocate_agent_dps_roots(
            root_obs,
            root_infos,
            root_budget,
            root_budgets=cfg.get("root_budgets"),
            root_scopes=root_scopes,
        )
        gamefiles = [str(info.get("extra.gamefile", "")) for info in root_infos]
        if any(not gamefile for gamefile in gamefiles):
            raise ValueError("A selected task lacks a replayable reference")
        repeated = self._repeat_with_allocation(gen_batch, quotas)
        repeated.meta_info = dict(repeated.meta_info)
        sampler = getattr(envs.envs, "task_sampler_metrics", None)
        repeated.meta_info["el_tsbpo_root_metrics"] = {
            "el_tsbpo/root/budget": float(root_budget),
            "el_tsbpo/root/n_candidates": float(n_tasks),
            "el_tsbpo/root/n_active_tasks": float(np.count_nonzero(quotas)),
            "el_tsbpo/root/n_skipped_candidates": float(np.count_nonzero(quotas == 0)),
            "el_tsbpo/root/min_active_rollouts": float(quotas[quotas > 0].min()),
            "el_tsbpo/root/max_active_rollouts": float(quotas.max()),
            "el_tsbpo/root/mean_active_rollouts": float(
                quotas.sum() / max(1, np.count_nonzero(quotas))
            ),
            "el_tsbpo/root/predictor_error": 0.0,
            "el_tsbpo/root/cold_start": 0.0,
            "el_tsbpo/root/predicted_success_mean": float(np.mean(probabilities)),
            "el_tsbpo/root/predicted_success_min": float(np.min(probabilities)),
            "el_tsbpo/root/predicted_success_max": float(np.max(probabilities)),
            "el_tsbpo/root/active_predicted_success_mean": float(
                np.mean(probabilities[quotas > 0])
            ),
            "el_tsbpo/root/skipped_predicted_success_mean": float(
                np.mean(probabilities[quotas == 0]) if np.any(quotas == 0) else 0.0
            ),
            "agent_dps/root/candidate_cache_hit": float(retained > 0),
            "agent_dps/root/candidate_cache_refresh": float(retained == 0),
            "agent_dps/root/candidate_cache_retained": float(retained),
            "agent_dps/root/candidate_cache_replaced": float(replaced),
            "agent_dps/root/candidate_cache_sampled": float(sampled),
            "agent_dps/root/task_selection_delegated": 1.0,
            "agent_dps/root/task_selection_dps_fraction": 1.0,
            **replay_metrics,
            **allocation_metrics,
            **(dict(sampler()) if callable(sampler) else {}),
        }
        repeated.non_tensor_batch["el_tsbpo_root_gamefile"] = np.asarray(
            [
                gamefiles[index]
                for index, quota in enumerate(quotas)
                for _ in range(int(quota))
            ],
            dtype=object,
        )
        stable_uids = [
            gamefiles[index]
            for index, quota in enumerate(quotas)
            for _ in range(int(quota))
        ]
        repeated.non_tensor_batch["el_tsbpo_root_uid"] = np.asarray(
            stable_uids, dtype=object
        )
        print(f"[EL-TSBPO] root quotas={quotas.tolist()}", flush=True)
        return repeated

    def _allocate_agent_dps(
        self,
        candidates,
        task_budgets: Mapping[str, int] | None = None,
        factual_outcomes: Mapping[str, float] | None = None,
        factual_task_ids: Mapping[str, str] | None = None,
    ):
        """Allocate branch rollouts for local factual-state contrast."""
        from agent_system.multi_turn_rollout.agent_dps import (
            AgentDPSNode,
            action_key,
            allocate_agent_dps,
            root_curriculum_key,
            state_key,
        )

        cfg = self._get_el_tsbpo_config()
        memory = self._get_agent_dps_memory()
        allocation_started = time.perf_counter()
        from agent_system.multi_turn_rollout.agent_dps import TerminalContrastPosterior

        state_posterior = None
        state_prior_memory = None
        task_union_plans = []
        from agent_system.multi_turn_rollout.dps_survival import allocate_local_state_contrast
        from agent_system.multi_turn_rollout.dps_state_prior import (
            StatePriorMemory,
            StatePriorPosterior,
        )

        opening_size = max(4, int(cfg.get("agent_dps_initial_branch_size", 4)))
        try:
            branch_worker_capacity = max(
                1, int(cfg.get("branch_worker_pool_size", 128))
            )
        except (TypeError, ValueError):
            branch_worker_capacity = 128
        outcomes = factual_outcomes or {}
        task_by_trajectory = {
            str(candidate.trajectory_id): str(candidate.task_uid)
            for candidate in candidates
        }
        if factual_task_ids is not None:
            task_by_trajectory.update(
                {
                    str(trajectory): str(task_uid)
                    for trajectory, task_uid in factual_task_ids.items()
                }
            )
        bare_task_labels: dict[str, set[bool]] = defaultdict(set)
        bare_task_sizes: dict[str, int] = defaultdict(int)
        for trajectory, outcome in outcomes.items():
            task_uid = task_by_trajectory.get(str(trajectory))
            if task_uid is not None:
                bare_task_labels[task_uid].add(float(outcome) >= 0.5)
                bare_task_sizes[task_uid] += 1
        mixed_bare_tasks = {
            task_uid
            for task_uid, labels in bare_task_labels.items()
            if len(labels) == 2
        }
        self._agent_dps_bare_task_labels = dict(bare_task_labels)
        current_transitions: dict[str, list[tuple[str, str, float, str]]] = defaultdict(
            list
        )
        current_transition_by_anchor: dict[str, tuple[str, str]] = {}
        factual_transition_observations = []
        for source in candidates:
            trajectory = str(source.trajectory_id)
            if trajectory not in outcomes:
                continue
            action = action_key(f"{'True'}:{str(source.current_action)}")
            source_key = state_key(source.task_uid, source.anchor_obs)
            next_observation = str(source.next_anchor_obs)
            if bool(source.is_terminal):
                next_observation = (
                    "__agent_dps_terminal_success__"
                    if float(outcomes[trajectory]) >= 0.5
                    else "__agent_dps_terminal_failure__"
                )
            if not next_observation:
                continue
            next_key = state_key(source.task_uid, next_observation)
            outcome = float(outcomes[trajectory])
            current_transitions[source_key].append(
                (action, next_key, outcome, trajectory)
            )
            current_transition_by_anchor[str(source.anchor_id)] = (action, next_key)
            factual_transition_observations.append(
                (source_key, action, next_key, outcome, trajectory)
            )
        grouped: dict[str, list[Any]] = defaultdict(list)
        for candidate in candidates:
            key = state_key(candidate.task_uid, candidate.anchor_obs)
            grouped[key].append(candidate)
        state_prior_memory = getattr(self, "_agent_dps_pending_state_prior", None)
        if state_prior_memory is None:
            state_prior_memory = StatePriorMemory(memory.survival_memory)
            by_trajectory = {}
            for key, members in grouped.items():
                for member in members:
                    uid = str(member.trajectory_id)
                    if uid not in outcomes:
                        continue
                    row = by_trajectory.setdefault(
                        uid,
                        dict(
                            id=uid,
                            task=str(member.task_uid),
                            cluster=uid,
                            kind="success"
                            if float(outcomes[uid]) >= 0.5
                            else "failure",
                            terminal_observed=True,
                            suffixes=[],
                        ),
                    )
                    row["suffixes"].append(dict(state=key, depth=int(member.step_idx)))
            state_prior_memory.observe(
                by_trajectory.values(),
                self._el_tsbpo_step,
                int(self.config.env.max_steps),
                max_entries=memory.max_states,
                max_age_steps=memory.max_age_steps,
            )
        state_prior_memory.prune(
            self._el_tsbpo_step,
            max_entries=memory.max_states,
            max_age_steps=memory.max_age_steps,
        )
        semantic_prior = None
        if cfg.get("agent_dps_branch_similarity", {}).get("enabled", False):
            similarity = self._get_agent_dps_branch_similarity()
            similarity.register(
                state_prior_memory,
                (
                    (
                        state_key(c.task_uid, c.anchor_obs),
                        getattr(c, "task", ""),
                        c.anchor_obs,
                    )
                    for c in candidates
                ),
            )
            semantic_prior = similarity.prepare(
                state_prior_memory, self._el_tsbpo_step, memory.decay
            )
        state_posterior = StatePriorPosterior(
            state_prior_memory,
            self._el_tsbpo_step,
            int(self.config.env.max_steps),
            memory.decay,
            particles=int(cfg.get("agent_dps_branch_particles", 512)),
            seed=int(cfg.get("agent_dps_tie_break_seed", 17)) + self._el_tsbpo_step,
            semantic_prior=semantic_prior,
            probability_calibrator=self._get_agent_dps_probability_calibrator("branch"),
        )
        root_factual_observations = []
        for trajectory, outcome in outcomes.items():
            task_ref = task_by_trajectory.get(str(trajectory))
            if task_ref is None:
                continue
            root_key = str(
                memory.root_state_keys.get(task_ref, "")
                or root_curriculum_key(task_ref)
            )
            memory.root_state_keys[task_ref] = root_key
            root_factual_observations.append((root_key, outcome, str(trajectory)))
        nodes_by_task: dict[str, list[AgentDPSNode]] = defaultdict(list)
        records_by_key: dict[str, dict[str, Any]] = {}
        node_by_key: dict[str, AgentDPSNode] = {}
        for key, members in sorted(grouped.items()):
            members = sorted(
                members, key=lambda item: (int(item.step_idx), str(item.anchor_id))
            )
            representative = members[0]
            physical_factual_count = (
                bare_task_sizes[str(representative.task_uid)]
                if str(representative.node_id)
                == f"el_tsbpo:task:{representative.task_uid}:root"
                else 1
            )
            observed = []
            current_outcome_rows = []
            seen_trajectories = set()
            for member in members:
                trajectory = str(member.trajectory_id)
                if trajectory in seen_trajectories:
                    continue
                if trajectory not in outcomes:
                    continue
                seen_trajectories.add(trajectory)
                outcome = float(outcomes[trajectory])
                observed.append(outcome)
                current_outcome_rows.append((outcome, trajectory))
            _, _, evidence_mass, visit_count = memory.posterior(
                key, self._el_tsbpo_step
            )
            historical_alpha, historical_beta = memory.posterior_parameters(
                key, self._el_tsbpo_step
            )
            posterior_alpha = historical_alpha + sum(
                (value >= 0.5 for value in observed)
            )
            posterior_beta = historical_beta + sum((value < 0.5 for value in observed))
            posterior_total = max(posterior_alpha + posterior_beta, 1e-12)
            probability = posterior_alpha / posterior_total
            uncertainty = (
                posterior_alpha
                * posterior_beta
                / max(
                    posterior_total * posterior_total * (posterior_total + 1.0), 1e-12
                )
            )
            hmm_belief = memory.dps_belief(key, self._el_tsbpo_step)
            factual_action, factual_next_state = current_transition_by_anchor.get(
                str(representative.anchor_id), ("", "")
            )
            factual_outcome = float(
                outcomes.get(str(representative.trajectory_id), 0.0)
            )
            contrast = memory.terminal_contrast_posterior(
                key,
                self._el_tsbpo_step,
                factual_outcome=factual_outcome,
                current_outcomes=current_outcome_rows,
            )
            contrast_edge_count = len(
                {
                    (action, next_key)
                    for action, next_key, _, _ in current_transitions.get(key, ())
                }
            )
            same_edge_probability = 0.0
            factual_edge_available = bool(factual_action and factual_next_state)
            opposite_particles = ()
            state_prior_diagnostics = {}
            if state_posterior is not None:
                depth = int(representative.step_idx)
                ps = state_posterior.success_particles(
                    str(representative.task_uid), key, depth, state_key=key
                )
                qs = ps if factual_outcome < 0.5 else 1 - ps
                probability, uncertainty = (float(ps.mean()), float(ps.var()))
                opposite_particles = tuple((float(q) for q in qs))
                prior_diagnostics = state_posterior.state_diagnostics(
                    str(representative.task_uid), key, depth
                )
                mean_q, var_q = (float(qs.mean()), float(qs.var()))
                concentration = max(
                    0.001, mean_q * (1 - mean_q) / max(var_q, 1e-12) - 1
                )
                contrast = TerminalContrastPosterior(
                    max(1e-08, mean_q * concentration),
                    max(1e-08, (1 - mean_q) * concentration),
                    prior_diagnostics["history_mass"],
                    prior_diagnostics["current_mass"],
                )
                state_prior_diagnostics = dict(
                    depth=depth,
                    remaining_steps=int(self.config.env.max_steps) - depth,
                    success_probability=probability,
                    opposite_probability=mean_q,
                    **prior_diagnostics,
                )
            acquisition_score = (
                float(1 - np.mean((1 - np.asarray(opposite_particles)) ** opening_size))
                if opposite_particles
                else contrast.utility(opening_size)
            )
            tie_payload = f"{int(cfg.get('agent_dps_tie_break_seed', 17))}:{int(self._el_tsbpo_step)}:{key}".encode(
                "utf-8"
            )
            tie_breaker = int.from_bytes(
                hashlib.sha256(tie_payload).digest()[:8], "big"
            ) / float(1 << 64)
            node = AgentDPSNode(
                state_key=key,
                task_uid=str(representative.task_uid),
                anchor_id=str(representative.anchor_id),
                trajectory_id=str(representative.trajectory_id),
                probability=probability,
                uncertainty=uncertainty,
                posterior_alpha=float(posterior_alpha),
                posterior_beta=float(posterior_beta),
                hmm_partial_probability=float(hmm_belief[1]),
                graph_score=float(contrast.probability),
                graph_dispersion=float(contrast.variance),
                graph_action_count=int(contrast_edge_count),
                graph_history_mass=float(contrast.history_mass),
                acquisition_score=float(acquisition_score),
                contrast_alpha=float(contrast.alpha),
                contrast_beta=float(contrast.beta),
                contrast_history_mass=float(contrast.history_mass),
                contrast_current_mass=float(contrast.current_mass),
                contrast_edge_count=int(contrast_edge_count),
                contrast_same_edge_opposite_probability=float(same_edge_probability),
                factual_edge_available=bool(factual_edge_available),
                factual_action=str(factual_action),
                factual_next_state=str(factual_next_state),
                factual_outcome=float(factual_outcome),
                factual_outcomes=tuple(observed),
                existing_factual_count=physical_factual_count,
                historical_evidence_count=float(evidence_mass),
                tie_breaker=tie_breaker,
                candidate_count=len(members),
                representative=representative,
                max_k=0,
                contrast_objective="terminal_contrast",
                opposite_particles=opposite_particles,
            )
            node_by_key[key] = node
            nodes_by_task[str(representative.task_uid)].append(node)
            records_by_key[key] = {
                "state_group_key": key,
                "task_ref": str(representative.task_uid),
                "bare_task_mixed": str(representative.task_uid) in mixed_bare_tasks,
                "conditioned_probability": float(probability),
                "branch_objective": "terminal_contrast",
                "prediction_model": "conditional_calibrated_beta_mixture_task_union",
                "state_prior": state_prior_diagnostics,
                "candidate_count": len(members),
                "factual_evidence_count": len(observed),
                "factual_success_count": sum((value >= 0.5 for value in observed)),
                "factual_failure_count": sum((value < 0.5 for value in observed)),
                "historical_evidence_mass": float(evidence_mass),
                "historical_visit_count": int(visit_count),
                "hmm_partial_probability": float(hmm_belief[1]),
                "graph_score": float(contrast.probability),
                "graph_dispersion": float(contrast.variance),
                "graph_action_count": int(contrast_edge_count),
                "graph_history_mass": float(contrast.history_mass),
                "acquisition_score": float(acquisition_score),
                "contrast_alpha": float(contrast.alpha),
                "contrast_beta": float(contrast.beta),
                "contrast_probability": float(contrast.probability),
                "contrast_history_mass": float(contrast.history_mass),
                "contrast_current_mass": float(contrast.current_mass),
                "contrast_edge_count": int(contrast_edge_count),
                "contrast_same_edge_opposite_probability": float(same_edge_probability),
                "factual_edge_available": bool(factual_edge_available),
                "factual_action": str(factual_action),
                "factual_next_state": str(factual_next_state),
                "factual_outcome": float(factual_outcome),
                "physical_factual_count": physical_factual_count,
                "selected_count": 0,
                "allocated_k": 0,
                "actual_k": physical_factual_count,
                "selected_anchor_ids": [],
                "selected_k": [],
                "candidate_anchor_ids": [str(item.anchor_id) for item in members],
                "candidate_node_ids": [str(item.node_id) for item in members],
                "candidate_trajectory_ids": [
                    str(item.trajectory_id) for item in members
                ],
                "candidate_steps": [int(item.step_idx) for item in members],
            }
        configured = {
            str(task_uid): max(0, int(value))
            for task_uid, value in (task_budgets or {}).items()
        }
        allocations = []
        allocator_metrics: dict[str, float] = {}
        reject_homogeneous = bool(cfg.get("agent_dps_reject_homogeneous_groups", False))
        for task_uid in sorted(nodes_by_task):
            configured.setdefault(task_uid, 0)
        global_budget = int(sum(configured.values()))
        all_nodes = [
            node for task_nodes in nodes_by_task.values() for node in task_nodes
        ]
        if global_budget > 0 and all_nodes:
            allocations, allocator_metrics, task_union_plans = allocate_local_state_contrast(
                all_nodes,
                global_budget,
                opening=max(opening_size, int(cfg.get("agent_dps_min_new_branch", 4))),
                max_k=min(global_budget, branch_worker_capacity),
                already_mixed_tasks=mixed_bare_tasks,
            )
        else:
            allocator_metrics.update(
                {
                    "agent_dps/allocator/budget": float(global_budget),
                    "agent_dps/allocator/used_budget": 0.0,
                    "agent_dps/allocator/unspent_budget": float(global_budget),
                    "agent_dps/allocator/budget_utilization": 0.0,
                    "agent_dps/allocator/n_states": float(len(all_nodes)),
                    "agent_dps/allocator/n_selected_states": 0.0,
                }
            )
        factual_observations_committed = 0
        root_factual_observations_committed = memory.observe_unique(
            root_factual_observations, self._el_tsbpo_step
        )
        factual_transitions_committed = 0
        if state_prior_memory is not None:
            memory.survival_memory = state_prior_memory.dump()
            memory.dirty = True
            self._agent_dps_pending_state_prior = None
        allocator_metrics["agent_dps/allocator/global_budget"] = float(global_budget)
        allocator_metrics["agent_dps/allocator/n_tasks"] = float(len(nodes_by_task))
        allocator_metrics["agent_dps/allocator/task_budget_sum"] = float(global_budget)
        allocator_metrics["agent_dps/allocator/global_competition"] = 1.0
        allocator_metrics["agent_dps/allocator/factual_transitions_committed"] = float(
            factual_transitions_committed
        )
        allocator_metrics["agent_dps/allocator/root_bare_observations_committed"] = (
            float(root_factual_observations_committed)
        )
        allocation_by_key = {str(item.state_group_key): item for item in allocations}
        for key, record in records_by_key.items():
            allocation = allocation_by_key.get(key)
            if allocation is None:
                continue
            record["selected_count"] = 1
            record["allocated_k"] = int(allocation.continuation_count)
            record["actual_k"] = int(
                allocation.continuation_count + allocation.physical_factual_count
            )
            record["predicted_created_contrast"] = float(allocation.cumulative_utility)
            record["selected_anchor_ids"] = [allocation.anchor_id]
            record["selected_k"] = [int(allocation.continuation_count)]
        records = sorted(
            records_by_key.values(),
            key=lambda item: (
                -int(item["allocated_k"]),
                -int(item["candidate_count"]),
                str(item["state_group_key"]),
            ),
        )
        anchor_info = {}
        for record in records:
            for anchor_id in record["candidate_anchor_ids"]:
                anchor_info[str(anchor_id)] = {
                    "state_group_key": record["state_group_key"],
                    "candidate_count": int(record["candidate_count"]),
                    "selected_count": int(record["selected_count"]),
                    "allocated_k": int(record["allocated_k"]),
                    "actual_k": int(record.get("actual_k", 1)),
                    "factual_evidence_count": int(
                        record.get("factual_evidence_count", 0)
                    ),
                }
        selected_groups = sum((int(item["selected_count"]) > 0 for item in records))
        selected_multi = sum(
            (
                int(item["selected_count"]) > 0 and int(item["candidate_count"]) > 1
                for item in records
            )
        )
        total_multi = sum((int(item["candidate_count"]) > 1 for item in records))
        used = sum((int(item.continuation_count) for item in allocations))
        bare_mixed_records = [record for record in records if record["bare_task_mixed"]]
        selected_bare_mixed_records = [
            record for record in bare_mixed_records if record["allocated_k"] > 0
        ]
        task_allocated = defaultdict(int)
        for allocation in allocations:
            node = node_by_key.get(str(allocation.state_group_key))
            if node is not None:
                task_allocated[str(node.task_uid)] += int(allocation.continuation_count)
        for task_uid in nodes_by_task:
            task_allocated.setdefault(str(task_uid), 0)
        factual_mixed = sum(
            (
                int(record.get("factual_success_count", 0)) > 0
                and int(record.get("factual_failure_count", 0)) > 0
                for record in records
            )
        )
        factual_homogeneous = sum(
            (
                int(record.get("factual_evidence_count", 0)) > 0
                and (
                    int(record.get("factual_success_count", 0)) == 0
                    or int(record.get("factual_failure_count", 0)) == 0
                )
                for record in records
            )
        )
        all_probabilities = np.asarray(
            [float(node.probability) for node in node_by_key.values()], dtype=np.float64
        )
        selected_nodes = [
            node_by_key[str(allocation.state_group_key)]
            for allocation in allocations
            if str(allocation.state_group_key) in node_by_key
        ]
        selected_probabilities = np.asarray(
            [float(node.probability) for node in selected_nodes], dtype=np.float64
        )
        all_uncertainties = np.asarray(
            [float(max(0.0, node.uncertainty)) ** 0.5 for node in node_by_key.values()],
            dtype=np.float64,
        )
        selected_uncertainties = np.asarray(
            [float(max(0.0, node.uncertainty)) ** 0.5 for node in selected_nodes],
            dtype=np.float64,
        )
        all_hmm_partial = np.asarray(
            [float(node.hmm_partial_probability) for node in node_by_key.values()],
            dtype=np.float64,
        )
        selected_hmm_partial = np.asarray(
            [float(node.hmm_partial_probability) for node in selected_nodes],
            dtype=np.float64,
        )
        all_graph_scores = np.asarray(
            [float(node.graph_score) for node in node_by_key.values()], dtype=np.float64
        )
        selected_graph_scores = np.asarray(
            [float(node.graph_score) for node in selected_nodes], dtype=np.float64
        )
        all_acquisition_scores = np.asarray(
            [float(node.acquisition_score) for node in node_by_key.values()],
            dtype=np.float64,
        )
        selected_acquisition_scores = np.asarray(
            [float(node.acquisition_score) for node in selected_nodes], dtype=np.float64
        )
        all_contrast_probabilities = np.asarray(
            [
                float(node.contrast_alpha)
                / max(float(node.contrast_alpha + node.contrast_beta), 1e-12)
                for node in node_by_key.values()
            ],
            dtype=np.float64,
        )
        selected_contrast_probabilities = np.asarray(
            [
                float(node.contrast_alpha)
                / max(float(node.contrast_alpha + node.contrast_beta), 1e-12)
                for node in selected_nodes
            ],
            dtype=np.float64,
        )
        all_contrast_history_mass = np.asarray(
            [float(node.contrast_history_mass) for node in node_by_key.values()],
            dtype=np.float64,
        )
        all_contrast_current_mass = np.asarray(
            [float(node.contrast_current_mass) for node in node_by_key.values()],
            dtype=np.float64,
        )
        all_factual_edge_coverage = np.asarray(
            [float(node.factual_edge_available) for node in node_by_key.values()],
            dtype=np.float64,
        )
        selected_factual_edge_coverage = np.asarray(
            [float(node.factual_edge_available) for node in selected_nodes],
            dtype=np.float64,
        )
        selected_same_edge_overlap = np.asarray(
            [
                float(node.contrast_same_edge_opposite_probability)
                for node in selected_nodes
            ],
            dtype=np.float64,
        )
        graph_covered = np.asarray(
            [float(node.graph_action_count >= 2) for node in node_by_key.values()],
            dtype=np.float64,
        )
        selected_graph_covered = np.asarray(
            [float(node.graph_action_count >= 2) for node in selected_nodes],
            dtype=np.float64,
        )
        all_transition_edge_counts = np.asarray(
            [float(node.contrast_edge_count) for node in node_by_key.values()],
            dtype=np.float64,
        )
        selected_transition_edge_counts = np.asarray(
            [float(node.contrast_edge_count) for node in selected_nodes],
            dtype=np.float64,
        )
        selected_new_k = np.asarray(
            [int(allocation.continuation_count) for allocation in allocations],
            dtype=np.float64,
        )
        selected_actual_k = np.asarray(
            [
                int(allocation.continuation_count)
                + int(allocation.physical_factual_count)
                for allocation in allocations
            ],
            dtype=np.float64,
        )
        selected_branch_mean_variance = np.asarray(
            [
                float(node.probability)
                * (1.0 - float(node.probability))
                / max(1, int(allocation.continuation_count))
                for allocation, node in zip(allocations, selected_nodes)
            ],
            dtype=np.float64,
        )
        historical_seen = np.asarray(
            [
                float(node.historical_evidence_count > 0.0)
                for node in node_by_key.values()
            ],
            dtype=np.float64,
        )
        selected_historical_seen = np.asarray(
            [float(node.historical_evidence_count > 0.0) for node in selected_nodes],
            dtype=np.float64,
        )

        def factual_success_mean(keys):
            successes = 0
            observations = 0
            for state_group_key in keys:
                record = records_by_key.get(str(state_group_key), {})
                successes += int(record.get("factual_success_count", 0))
                observations += int(record.get("factual_evidence_count", 0))
            return float(successes / observations) if observations else 0.0

        candidate_factual_success = factual_success_mean(node_by_key)
        selected_factual_success = factual_success_mean(
            (allocation.state_group_key for allocation in allocations)
        )
        metrics = {
            "el_tsbpo/allocator/n_candidates": float(len(candidates)),
            "el_tsbpo/allocator/n_anchors": float(len(allocations)),
            "el_tsbpo/allocator/branch_budget": float(sum(configured.values())),
            "el_tsbpo/allocator/branches_allocated": float(used),
            "el_tsbpo/allocator/budget_utilization": float(
                used / max(1, sum(configured.values()))
            ),
            "el_tsbpo/allocator/external_cost": float(used),
            "el_tsbpo/allocator/predicted_score": float(
                sum((float(item.score) for item in allocations))
            ),
            "el_tsbpo/allocator/mean_continuations_per_anchor": float(
                used / max(1, len(allocations))
            ),
            "el_tsbpo/allocator/k_0_anchors": float(
                max(0, len(records) - len(allocations))
            ),
            "el_tsbpo/allocator/state_group/n_groups": float(len(records)),
            "el_tsbpo/allocator/state_group/n_multi_candidate_groups": float(
                total_multi
            ),
            "el_tsbpo/allocator/state_group/n_selected_groups": float(selected_groups),
            "el_tsbpo/allocator/state_group/n_selected_multi_groups": float(
                selected_multi
            ),
            "el_tsbpo/allocator/state_group/n_multi_candidate_anchors": float(
                sum(
                    (
                        int(item["candidate_count"])
                        for item in records
                        if int(item["candidate_count"]) > 1
                    )
                )
            ),
            "el_tsbpo/allocator/state_group/multi_anchor_selection_rate": float(
                selected_multi / max(1, total_multi)
            ),
            "agent_dps/allocator/reject_homogeneous_groups": float(reject_homogeneous),
            "agent_dps/allocator/local_state_contrast_enabled": 1.0,
            "agent_dps/allocator/single_anchor_k_cap": float(branch_worker_capacity),
            "agent_dps/allocator/bare_mixed_state_groups_considered": float(
                len(bare_mixed_records)
            ),
            "agent_dps/allocator/selected_bare_mixed_state_groups": float(
                len(selected_bare_mixed_records)
            ),
            "agent_dps/allocator/bare_mixed_branch_budget_fraction": float(
                sum(record["allocated_k"] for record in selected_bare_mixed_records)
                / max(1, used)
            ),
            "agent_dps/allocator/n_factual_mixed_states": float(factual_mixed),
            "agent_dps/allocator/n_factual_homogeneous_states": float(
                factual_homogeneous
            ),
            "agent_dps/allocator/same_round_factual_excluded": 0.0,
            "agent_dps/allocator/current_factual_committed_before_selection": 0.0,
            "agent_dps/allocator/current_factual_conditioned_once": 1.0,
            "agent_dps/allocator/transition_evidence_written_once": 1.0,
            "agent_dps/allocator/current_factual_mixed_states": float(factual_mixed),
            "agent_dps/allocator/current_factual_homogeneous_states": float(
                factual_homogeneous
            ),
            "agent_dps/allocator/factual_observations_committed_before_selection": 0.0,
            "agent_dps/allocator/factual_observations_committed_after_selection": float(
                factual_observations_committed
            ),
            "agent_dps/allocator/cold_start_random_tiebreak": float(
                bool(len(node_by_key)) and (not bool(historical_seen.any()))
            ),
            "agent_dps/allocator/candidate_history_seen_fraction": float(
                historical_seen.mean() if historical_seen.size else 0.0
            ),
            "agent_dps/allocator/selected_history_seen_fraction": float(
                selected_historical_seen.mean()
                if selected_historical_seen.size
                else 0.0
            ),
            "agent_dps/allocator/candidate_current_factual_success_mean": float(
                candidate_factual_success
            ),
            "agent_dps/allocator/selected_current_factual_success_mean": float(
                selected_factual_success
            ),
            "agent_dps/allocator/current_factual_selection_lift": float(
                selected_factual_success - candidate_factual_success
            ),
            "agent_dps/allocator/candidate_p_mean": float(
                all_probabilities.mean() if all_probabilities.size else 0.0
            ),
            "agent_dps/allocator/candidate_p_std": float(
                all_probabilities.std() if all_probabilities.size else 0.0
            ),
            "agent_dps/allocator/selected_p_mean": float(
                selected_probabilities.mean() if selected_probabilities.size else 0.0
            ),
            "agent_dps/allocator/selected_p_std": float(
                selected_probabilities.std() if selected_probabilities.size else 0.0
            ),
            "agent_dps/allocator/candidate_uncertainty_mean": float(
                all_uncertainties.mean() if all_uncertainties.size else 0.0
            ),
            "agent_dps/allocator/selected_uncertainty_mean": float(
                selected_uncertainties.mean() if selected_uncertainties.size else 0.0
            ),
            "agent_dps/allocator/selection_uncertainty_lift": float(
                selected_uncertainties.mean() - all_uncertainties.mean()
                if selected_uncertainties.size and all_uncertainties.size
                else 0.0
            ),
            "agent_dps/allocator/candidate_hmm_partial_mean": float(
                all_hmm_partial.mean() if all_hmm_partial.size else 0.0
            ),
            "agent_dps/allocator/selected_hmm_partial_mean": float(
                selected_hmm_partial.mean() if selected_hmm_partial.size else 0.0
            ),
            "agent_dps/allocator/hmm_partial_selection_lift": float(
                selected_hmm_partial.mean() - all_hmm_partial.mean()
                if selected_hmm_partial.size and all_hmm_partial.size
                else 0.0
            ),
            "agent_dps/allocator/candidate_graph_score_mean": float(
                all_graph_scores.mean() if all_graph_scores.size else 0.0
            ),
            "agent_dps/allocator/selected_graph_score_mean": float(
                selected_graph_scores.mean() if selected_graph_scores.size else 0.0
            ),
            "agent_dps/allocator/graph_score_selection_lift": float(
                selected_graph_scores.mean() - all_graph_scores.mean()
                if selected_graph_scores.size and all_graph_scores.size
                else 0.0
            ),
            "agent_dps/allocator/candidate_graph_coverage": float(
                graph_covered.mean() if graph_covered.size else 0.0
            ),
            "agent_dps/allocator/selected_graph_coverage": float(
                selected_graph_covered.mean() if selected_graph_covered.size else 0.0
            ),
            "agent_dps/allocator/candidate_acquisition_mean": float(
                all_acquisition_scores.mean() if all_acquisition_scores.size else 0.0
            ),
            "agent_dps/allocator/selected_acquisition_mean": float(
                selected_acquisition_scores.mean()
                if selected_acquisition_scores.size
                else 0.0
            ),
            "agent_dps/allocator/acquisition_selection_lift": float(
                selected_acquisition_scores.mean() - all_acquisition_scores.mean()
                if selected_acquisition_scores.size and all_acquisition_scores.size
                else 0.0
            ),
            "agent_dps/allocator/candidate_contrast_probability_mean": float(
                all_contrast_probabilities.mean()
                if all_contrast_probabilities.size
                else 0.0
            ),
            "agent_dps/allocator/selected_contrast_probability_mean": float(
                selected_contrast_probabilities.mean()
                if selected_contrast_probabilities.size
                else 0.0
            ),
            "agent_dps/allocator/contrast_probability_selection_lift": float(
                selected_contrast_probabilities.mean()
                - all_contrast_probabilities.mean()
                if selected_contrast_probabilities.size
                and all_contrast_probabilities.size
                else 0.0
            ),
            "agent_dps/allocator/history_prior_mass_mean": float(
                all_contrast_history_mass.mean()
                if all_contrast_history_mass.size
                else 0.0
            ),
            "agent_dps/allocator/current_likelihood_mass_mean": float(
                all_contrast_current_mass.mean()
                if all_contrast_current_mass.size
                else 0.0
            ),
            "agent_dps/allocator/candidate_factual_edge_coverage": float(
                all_factual_edge_coverage.mean()
                if all_factual_edge_coverage.size
                else 0.0
            ),
            "agent_dps/allocator/selected_factual_edge_coverage": float(
                selected_factual_edge_coverage.mean()
                if selected_factual_edge_coverage.size
                else 0.0
            ),
            "agent_dps/allocator/selected_same_edge_opposite_probability": float(
                selected_same_edge_overlap.mean()
                if selected_same_edge_overlap.size
                else 0.0
            ),
            "agent_dps/allocator/candidate_transition_edge_count_mean": float(
                all_transition_edge_counts.mean()
                if all_transition_edge_counts.size
                else 0.0
            ),
            "agent_dps/allocator/selected_transition_edge_count_mean": float(
                selected_transition_edge_counts.mean()
                if selected_transition_edge_counts.size
                else 0.0
            ),
            "agent_dps/allocator/transition_edge_count_selection_lift": float(
                selected_transition_edge_counts.mean()
                - all_transition_edge_counts.mean()
                if selected_transition_edge_counts.size
                and all_transition_edge_counts.size
                else 0.0
            ),
            "agent_dps/allocator/selected_task_count": float(
                sum((value > 0 for value in task_allocated.values()))
            ),
            "agent_dps/allocator/max_task_allocated": float(
                max(task_allocated.values(), default=0)
            ),
            "agent_dps/allocator/task_allocation_std": float(
                np.std(list(task_allocated.values())) if task_allocated else 0.0
            ),
            "agent_dps/allocator/mean_new_k": float(used / max(1, len(allocations))),
            "agent_dps/allocator/min_new_k": float(
                selected_new_k.min() if selected_new_k.size else 0.0
            ),
            "agent_dps/allocator/max_new_k": float(
                selected_new_k.max() if selected_new_k.size else 0.0
            ),
            "agent_dps/allocator/mean_actual_group_k": float(
                selected_actual_k.mean() if selected_actual_k.size else 0.0
            ),
            "agent_dps/allocator/min_actual_group_k": float(
                selected_actual_k.min() if selected_actual_k.size else 0.0
            ),
            "agent_dps/allocator/max_actual_group_k": float(
                selected_actual_k.max() if selected_actual_k.size else 0.0
            ),
            "agent_dps/allocator/singleton_new_branch_fraction": float(
                np.mean(selected_new_k == 1) if selected_new_k.size else 0.0
            ),
            "agent_dps/allocator/estimated_branch_mean_variance": float(
                selected_branch_mean_variance.mean()
                if selected_branch_mean_variance.size
                else 0.0
            ),
            **allocator_metrics,
            **memory.metrics(self._el_tsbpo_step),
        }
        metrics.pop("agent_dps/allocator/selected_same_edge_opposite_probability", None)
        for record in records:
            record.pop("contrast_same_edge_opposite_probability", None)
        metrics.update(
            {
                "agent_dps/allocator/selected_quota_uniform": float(
                    len(set((int(a.continuation_count) for a in allocations))) <= 1
                ),
                "agent_dps/allocator/selected_quota_std": float(
                    np.std([int(a.continuation_count) for a in allocations])
                )
                if allocations
                else 0.0,
            }
        )
        if state_posterior is not None:
            diag = state_posterior.diagnostics()
            metrics.update(
                {
                    f"agent_dps/state_prior/{key}": float(value)
                    for key, value in diag.items()
                    if isinstance(value, (int, float))
                }
            )
            metrics["agent_dps/allocator/state_prior_hit_fraction"] = (
                float(np.mean([r["state_prior"]["history_mass"] > 0 for r in records]))
                if records
                else 0.0
            )
            metrics["agent_dps/allocator/candidate_history_seen_fraction"] = metrics[
                "agent_dps/allocator/state_prior_hit_fraction"
            ]
            metrics["agent_dps/allocator/selected_history_seen_fraction"] = (
                float(
                    np.mean(
                        [
                            r["state_prior"]["history_mass"] > 0
                            for r in records
                            if r["selected_count"] > 0
                        ]
                    )
                )
                if allocations
                else 0.0
            )
            if state_posterior.probability_calibrator is not None:
                metrics.update(
                    {
                        f"agent_dps/probability/branch/{k}": float(v)
                        for k, v in state_posterior.probability_calibrator.diagnostics().items()
                    }
                )
            if state_posterior.semantic_prior is not None:
                metrics["agent_dps/allocator/semantic_prior_hit_fraction"] = (
                    float(
                        np.mean(
                            [r["state_prior"]["semantic_mass"] > 0 for r in records]
                        )
                    )
                    if records
                    else 0.0
                )
                metrics["agent_dps/allocator/semantic_prior_mass_mean"] = (
                    float(np.mean([r["state_prior"]["semantic_mass"] for r in records]))
                    if records
                    else 0.0
                )
                selected_semantic = [
                    r["state_prior"] for r in records if r["selected_count"] > 0
                ]
                metrics["agent_dps/allocator/selected_semantic_prior_hit_fraction"] = (
                    float(np.mean([d["semantic_mass"] > 0 for d in selected_semantic]))
                    if selected_semantic
                    else 0.0
                )
                metrics.update(
                    {
                        f"agent_dps/branch_embedding/{k}": float(v)
                        for k, v in self._get_agent_dps_branch_similarity().stats.items()
                    }
                )
                metrics["timing_s/agent_dps_branch_embedding"] = float(
                    self._get_agent_dps_branch_similarity().stats["encode_seconds"]
                )
        if cfg.get("agent_dps_allocation_diagnostics_dir"):
            self._write_agent_dps_allocation(
                "branch",
                {
                    "budget": global_budget,
                    "branch_utility": "local_state_contrast",
                    "bare_mixed_task_refs": sorted(mixed_bare_tasks),
                    "task_union_plans": task_union_plans,
                    "state_prior_model": state_posterior.diagnostics()
                    if state_posterior
                    else {},
                    "states": records,
                    "metrics": metrics,
                },
            )
        self._agent_dps_task_union_plans = task_union_plans
        metrics["timing_s/agent_dps_branch_allocation"] = (
            time.perf_counter() - allocation_started
        )
        for allocation in allocations:
            key = f"el_tsbpo/allocator/k_{int(allocation.continuation_count)}_anchors"
            metrics[key] = metrics.get(key, 0.0) + 1.0
            new_key = (
                f"agent_dps/allocator/new_k_{int(allocation.continuation_count)}_states"
            )
            actual_key = f"agent_dps/allocator/actual_k_{int(allocation.continuation_count) + int(allocation.physical_factual_count)}_states"
            metrics[new_key] = metrics.get(new_key, 0.0) + 1.0
            metrics[actual_key] = metrics.get(actual_key, 0.0) + 1.0
        self._el_tsbpo_state_group_records = records
        self._el_tsbpo_state_group_by_anchor = anchor_info
        if state_posterior is not None and state_posterior.calibrator is not None:
            metrics.update(
                {
                    f"agent_dps/calibration/{key}": float(value)
                    for key, value in state_posterior.calibrator.diagnostics().items()
                }
            )
        self._agent_dps_memory_dirty = True
        self._save_agent_dps_memory()
        return (allocations, metrics)

    def _agent_dps_observe_continuations(self, rows, *, branch=False):
        from agent_system.multi_turn_rollout.dps_survival import continuation_records
        from agent_system.multi_turn_rollout.dps_state_prior import StatePriorMemory

        memory = self._get_agent_dps_memory()
        horizon = int(self.config.env.max_steps)
        observations = continuation_records(rows, horizon, 0, branch=branch)
        prior = (
            None if branch else getattr(self, "_agent_dps_pending_state_prior", None)
        )
        if prior is None:
            prior = StatePriorMemory(memory.survival_memory)
        seen = prior.seen_ids if prior.seen_step == int(self._el_tsbpo_step) else set()
        new_observations = [
            record for record in observations if str(record["id"]) not in seen
        ]
        added = prior.observe(
            observations,
            self._el_tsbpo_step,
            horizon,
            max_entries=memory.max_states,
            max_age_steps=memory.max_age_steps,
        )
        probability_metrics = {}
        if not branch and self._get_el_tsbpo_config().get(
            "agent_dps_probability_calibration", False
        ):
            from agent_system.multi_turn_rollout.dps_probability_calibration import (
                observe_groups,
            )

            by_task = defaultdict(lambda: [0, 0])
            for record in observations:
                if record["terminal_observed"]:
                    by_task[str(record["task"])][record["kind"] != "success"] += 1
            contexts = {
                context["task"]: (forecast, context)
                for (step, _), (
                    forecast,
                    context,
                ) in self._agent_dps_root_forecasts.items()
                if step == int(self._el_tsbpo_step)
            }
            records, predicted, actual, nll = [], [], [], []
            skipped = 0
            for task, (success, failure) in sorted(by_task.items()):
                forecast, context = contexts[task]
                count = success + failure
                if count != self._agent_dps_root_calibration_quotas[task]:
                    skipped += 1
                    continue
                records.append({**context, "success": success, "failure": failure})
                predicted.append(forecast.mixed_curve(count)[-1])
                actual.append(float(success > 0 and failure > 0))
                nll.append(-forecast.count_log_probability(success, failure))
            committed = observe_groups(
                memory.probability_calibration,
                "root",
                records,
                self._el_tsbpo_step,
                max_entries=memory.max_states,
            )
            memory.dirty = True
            probability_metrics = {
                "agent_dps/probability/root/groups_committed": float(committed),
                "agent_dps/probability/root/incomplete_groups": float(skipped),
                "agent_dps/probability/root/predicted_mixed": float(np.mean(predicted))
                if predicted
                else 0.0,
                "agent_dps/probability/root/observed_mixed": float(np.mean(actual))
                if actual
                else 0.0,
                "agent_dps/probability/root/brier": float(
                    np.mean((np.asarray(predicted) - actual) ** 2)
                )
                if predicted
                else 0.0,
                "agent_dps/probability/root/count_nll": float(np.mean(nll))
                if nll
                else 0.0,
            }
            self._write_agent_dps_allocation(
                "root_outcomes",
                dict(
                    records=records,
                    predicted_mixed=predicted,
                    observed_mixed=actual,
                    metrics=probability_metrics,
                ),
            )
        similarity = None
        if (
            self._get_el_tsbpo_config()
            .get("agent_dps_branch_similarity", {})
            .get("enabled", False)
        ):
            from agent_system.multi_turn_rollout.agent_dps import state_key

            similarity = self._get_agent_dps_branch_similarity()
            if not branch:
                similarity.reset_stats()
            valid = {
                str(record["id"])
                for record in observations
                if record["terminal_observed"]
            }
            similarity.register(
                prior,
                (
                    (
                        state_key(
                            str(row.get("uid", "")), str(row.get("anchor_obs", ""))
                        ),
                        row.get("dps_task_text", ""),
                        row.get("anchor_obs", ""),
                    )
                    for row in rows
                    if str(row.get("traj_uid", "")) in valid
                    and bool(row.get("active_masks", True))
                ),
            )
        if branch:
            memory.survival_memory = prior.dump()
            memory.dirty = True
            self._agent_dps_memory_dirty = True
        else:
            self._agent_dps_pending_state_prior = prior
        kind = "branch" if branch else "bare"
        summary = dict(
            paths=added,
            incomplete=sum(not r["terminal_observed"] for r in new_observations),
            timeouts=sum(
                r["kind"] == "timeout" and r["terminal_observed"]
                for r in new_observations
            ),
            state_views=sum(
                len(r["suffixes"]) for r in new_observations if r["terminal_observed"]
            ),
        )

        metrics = {
            f"agent_dps/state_prior/{kind}_{key}": float(value)
            for key, value in summary.items()
        }
        metrics.update(probability_metrics)
        if similarity is not None:
            summary["embedding"] = dict(similarity.stats)
            metrics.update(
                {
                    f"agent_dps/branch_embedding/{k}": float(v)
                    for k, v in similarity.stats.items()
                }
            )
            metrics["timing_s/agent_dps_branch_embedding"] = float(
                similarity.stats["encode_seconds"]
            )
        self._write_agent_dps_allocation(f"state_prior_{kind}", summary)
        return metrics

    def _agent_dps_update_factual_memory(
        self,
        candidates,
        factual_outcomes: Mapping[str, float],
    ) -> dict[str, float]:
        from agent_system.multi_turn_rollout.agent_dps import root_curriculum_key

        memory = self._get_agent_dps_memory()
        root_observations = []
        for candidate in candidates:
            trajectory = str(candidate.trajectory_id)
            if trajectory not in factual_outcomes or int(candidate.step_idx) != 0:
                continue
            task_ref = str(candidate.task_uid)
            root_key = str(
                memory.root_state_keys.get(task_ref, "") or root_curriculum_key(task_ref)
            )
            memory.root_state_keys[task_ref] = root_key
            root_observations.append((root_key, factual_outcomes[trajectory], trajectory))
        root_count = memory.observe_unique(root_observations, self._el_tsbpo_step)
        pending = getattr(self, "_agent_dps_pending_state_prior", None)
        if pending is not None:
            memory.survival_memory = pending.dump()
            memory.dirty = True
            self._agent_dps_pending_state_prior = None
        self._agent_dps_memory_dirty = True
        self._save_agent_dps_memory()
        return {
            "agent_dps/state_memory/factual_observations": 0.0,
            "agent_dps/state_memory/root_bare_observations": float(root_count),
            "agent_dps/state_memory/factual_transitions": 0.0,
            **memory.metrics(self._el_tsbpo_step),
        }

    def el_tsbpo_multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
        is_train: bool = True,
    ) -> DataProto:
        """Collect factual rollouts, allocate branches, and assemble the tree."""

        from agent_system.multi_turn_rollout.tree_node import TreeNode


        self._el_tsbpo_generation_stats = {
            "calls": 0,
            "worker_calls": 0,
            "rows": 0,
            "batches": 0,
            "max_batch_rows": 0,
            "token_budget": 0,
            "prompt_tokens": 0,
            "response_tokens": 0,
            "padded_rows": 0,
            "wall_seconds": 0.0,
            "tokenize_seconds": 0.0,
            "batch_rows": 0,
            "batch_prompt_tokens": 0,
            "batch_planned_tokens": 0,
            "natural": self._new_generation_kind_stats(),
            "branch": self._new_generation_kind_stats(),
            "env_step_seconds": 0.0,
            "branch_env_wait_seconds": 0.0,
            "branch_env_submit_seconds": 0.0,
            "branch_initial_spawn_seconds": 0.0,
            "branch_refill_spawn_seconds": 0.0,
            "branch_ready_batches": 0,
            "branch_ready_rows": 0,
            "branch_ready_max_rows": 0,
            "branch_synchronous_frontier": 0,
            "branch_local_token_packing": 1,
            "branch_horizon_bucket_count": 0,
            "branch_horizon_bucket_rows": 0,
            "branch_horizon_bucket_max_rows": 0,
            "branch_refill_spawn_calls": 0,
            "branch_refill_branches": 0,
            "branch_active_frontier_batches": 0,
            "branch_active_frontier_rows": 0,
            "branch_active_frontier_max_rows": 0,
        }
        self._el_tsbpo_generation_stats["branch_synchronous_frontier"] = int(
            _is_truthy_flag(
                self._get_el_tsbpo_config().get("branch_synchronous_frontier", False)
            )
        )

        if is_train:
            gen_batch = self._prepare_el_tsbpo_root_batch(gen_batch, envs=envs)
        (
            total_batch_list,
            episode_rewards,
            episode_lengths,
            success,
            traj_uid,
            tool_callings,
        ) = self.vanilla_multi_turn_loop(gen_batch, actor_rollout_wg, envs)
        source_env_indices = None
        output = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            success=success,
            traj_uid=traj_uid,
            tool_callings=tool_callings,
        )
        bsz = len(output.batch)
        output.non_tensor_batch["branch_group_id"] = np.asarray(
            [None] * bsz, dtype=object
        )
        output.non_tensor_batch["is_counterfactual"] = np.asarray(
            [False] * bsz, dtype=bool
        )
        output.non_tensor_batch["parent_step_idx"] = np.asarray(
            [-1] * bsz, dtype=np.int64
        )
        output.non_tensor_batch["parent_traj_uid"] = np.asarray(
            [None] * bsz, dtype=object
        )
        output.non_tensor_batch["el_tsbpo_sibling_group_id"] = np.asarray(
            [None] * bsz, dtype=object
        )
        output.non_tensor_batch["el_tsbpo_is_extra"] = np.asarray(
            [False] * bsz, dtype=bool
        )
        output.non_tensor_batch["el_tsbpo_anchor_context"] = np.asarray(
            [""] * bsz, dtype=object
        )
        output.non_tensor_batch["el_tsbpo_allocated_k"] = np.zeros(bsz, dtype=np.int64)
        output.non_tensor_batch["el_tsbpo_rollout_seconds"] = np.zeros(
            bsz, dtype=np.float64
        )
        output.non_tensor_batch["el_tsbpo_parent_node_id"] = np.asarray(
            [""] * bsz, dtype=object
        )
        output.non_tensor_batch["el_tsbpo_node_id"] = np.asarray(
            [""] * bsz, dtype=object
        )
        output.non_tensor_batch["el_tsbpo_node_is_terminal"] = np.zeros(bsz, dtype=bool)
        output.non_tensor_batch["el_tsbpo_terminal_reward"] = np.zeros(
            bsz, dtype=np.float32
        )
        output.non_tensor_batch["el_tsbpo_bare_rollout_id"] = np.asarray(
            [""] * bsz, dtype=object
        )
        output.non_tensor_batch["el_tsbpo_branch_type"] = np.asarray(
            ["factual"] * bsz, dtype=object
        )
        output.non_tensor_batch["el_tsbpo_snapshot_id"] = np.asarray(
            [""] * bsz, dtype=object
        )
        output.non_tensor_batch["el_tsbpo_terminal_reason"] = np.asarray(
            [""] * bsz, dtype=object
        )
        output.non_tensor_batch["el_tsbpo_anchor_node_id"] = np.asarray(
            [""] * bsz, dtype=object
        )
        output.non_tensor_batch["el_tsbpo_state_group_key"] = np.asarray(
            [""] * bsz, dtype=object
        )
        output.non_tensor_batch["el_tsbpo_state_group_candidate_count"] = np.zeros(
            bsz, dtype=np.int64
        )
        output.non_tensor_batch["el_tsbpo_state_group_selected_count"] = np.zeros(
            bsz, dtype=np.int64
        )
        output.non_tensor_batch["el_tsbpo_state_group_allocated_k"] = np.zeros(
            bsz, dtype=np.int64
        )
        output.non_tensor_batch["el_tsbpo_state_group_actual_k"] = np.ones(
            bsz, dtype=np.int64
        )
        memberships = np.empty(bsz, dtype=object)
        for row in range(bsz):
            memberships[row] = []
        output.non_tensor_batch["el_tsbpo_group_memberships"] = memberships
        output.non_tensor_batch["el_tsbpo_allocation_provenance"] = np.asarray(
            ["factual"] * bsz, dtype=object
        )
        output.non_tensor_batch["el_tsbpo_quarantined"] = np.zeros(bsz, dtype=bool)
        output.non_tensor_batch["el_tsbpo_policy_version"] = np.asarray(
            [
                str(value)
                for value in output.non_tensor_batch.get(
                    "policy_version", np.asarray(["unknown"] * bsz, dtype=object)
                )
            ],
            dtype=object,
        )


        output_traj_ids = output.non_tensor_batch.get(
            "traj_uid", np.asarray([""] * bsz, dtype=object)
        )
        output_task_ids = output.non_tensor_batch.get(
            "uid", np.asarray([""] * bsz, dtype=object)
        )
        output_turn_ids = output.non_tensor_batch.get(
            "turn_idx", np.zeros(bsz, dtype=np.int64)
        )
        max_turn_by_traj: Dict[str, int] = defaultdict(int)
        for row in range(bsz):
            max_turn_by_traj[str(output_traj_ids[row])] = max(
                max_turn_by_traj[str(output_traj_ids[row])],
                int(output_turn_ids[row]),
            )
        for row in range(bsz):
            trajectory_id = str(output_traj_ids[row])
            task_id = str(output_task_ids[row])
            turn_idx = int(output_turn_ids[row])
            output.non_tensor_batch["el_tsbpo_parent_node_id"][row] = (
                f"el_tsbpo:task:{task_id}:root"
                if turn_idx == 0
                else f"el_tsbpo:{trajectory_id}:n{turn_idx}"
            )
            output.non_tensor_batch["el_tsbpo_node_id"][row] = (
                f"el_tsbpo:{trajectory_id}:n{turn_idx + 1}"
            )
            output.non_tensor_batch["el_tsbpo_node_is_terminal"][row] = (
                turn_idx >= max_turn_by_traj[trajectory_id]
            )
            if output.non_tensor_batch["el_tsbpo_node_is_terminal"][row]:


                output.non_tensor_batch["el_tsbpo_terminal_reward"][row] = float(
                    output.non_tensor_batch.get(
                        "episode_rewards",
                        np.zeros(bsz, dtype=np.float32),
                    )[row]
                )
            output.non_tensor_batch["el_tsbpo_bare_rollout_id"][row] = trajectory_id
            output.non_tensor_batch["el_tsbpo_branch_type"][row] = "factual"
            output.non_tensor_batch["el_tsbpo_snapshot_id"][row] = str(
                output.non_tensor_batch.get(
                    "env_snapshot_hash", np.asarray([""] * bsz, dtype=object)
                )[row]
                or ""
            )
            output.non_tensor_batch["el_tsbpo_terminal_reason"][row] = (
                "terminal"
                if output.non_tensor_batch["el_tsbpo_node_is_terminal"][row]
                else ""
            )
        metrics: Dict[str, float] = {
            "el_tsbpo/rollout/n_factual_trajectories": float(len(total_batch_list)),
            "el_tsbpo/rollout/n_extra_turns": 0.0,
            "el_tsbpo/rollout/n_exact_sibling_trajectories": 0.0,
            "el_tsbpo/rollout/n_quarantined_trajectories": 0.0,
            "el_tsbpo/rollout/actual_seconds": 0.0,


            "el_tsbpo/audit/checked_trajectories": 0.0,
            "el_tsbpo/audit/fail_context": 0.0,
            "el_tsbpo/audit/fail_decision": 0.0,
            "el_tsbpo/audit/fail_policy_group": 0.0,
            "el_tsbpo/audit/fail_environment": 0.0,
            "el_tsbpo/audit/fail_replay_observation": 0.0,
            "el_tsbpo/audit/fail_replay_admissible": 0.0,
            "el_tsbpo/audit/fail_cardinality": 0.0,
        }
        if self.is_agent_dps_enabled() and is_train:
            from gigpo.core_gigpo import compute_effective_ratio

            metrics["sampling/effective_ratio"] = compute_effective_ratio(
                index=output.non_tensor_batch["uid"],
                traj_index=output.non_tensor_batch["traj_uid"],
                trajectory_success=output.non_tensor_batch["trajectory_success"],
            )
            metrics["agent_dps/outcome/bare_mixed_task_ratio"] = metrics[
                "sampling/effective_ratio"
            ]
            if self._get_el_tsbpo_config().get("agent_dps_branch_task_union", False):
                metrics.update(
                    self._agent_dps_observe_continuations(
                        [row for steps in total_batch_list for row in steps]
                    )
                )
        if is_train:
            metrics.update(gen_batch.meta_info.get("el_tsbpo_root_metrics", {}))
            metrics.update(
                {
                    "el_tsbpo/root/generated": float(len(total_batch_list)),
                    "el_tsbpo/root/used": float(len(total_batch_list)),
                    "el_tsbpo/root/discarded": 0.0,
                    "el_tsbpo/root/post_selection": 0.0,
                    "el_tsbpo/root/preallocated": float(
                        int(self._get_el_tsbpo_config().get("root_budget", 0)) > 0
                    ),
                }
            )
        if (
            not is_train
            or int(self._get_el_tsbpo_config().get("branch_budget", 0)) <= 0
        ):
            if self.is_agent_dps_enabled() and is_train:
                factual_candidates = self._collect_el_tsbpo_candidates(
                    total_batch_list,
                    traj_uid,
                    envs,
                    source_env_indices=source_env_indices,
                )
                factual_success = success.get(
                    "success_rate", np.zeros(len(total_batch_list), dtype=bool)
                )
                factual_outcomes = {
                    str(traj_uid[index]): float(bool(factual_success[index]))
                    for index in range(min(len(traj_uid), len(factual_success)))
                }
                metrics.update(
                    self._agent_dps_update_factual_memory(
                        factual_candidates, factual_outcomes
                    )
                )
            output.batch["loss_mask"] = output.batch["attention_mask"].clone()
            advantage_mode = (
                str(self._get_el_tsbpo_config().get("advantage_mode", "tree"))
                .lower()
                .replace("-", "_")
            )
            if advantage_mode in {"tree_gigpo", "tree_gigpo_s", "tree_gigpo_simple"}:
                self._attach_el_tsbpo_action_token_mask(output)
            if is_train:
                self._el_tsbpo_step += 1
                output.meta_info["el_tsbpo_metrics"] = metrics
            return output

        candidates = self._collect_el_tsbpo_candidates(
            total_batch_list, traj_uid, envs, source_env_indices=source_env_indices
        )
        factual_counts: dict[str, int] = defaultdict(int)
        for steps in total_batch_list:
            if steps:
                factual_counts[str(steps[0].get("uid", ""))] += 1
        configured_branch_budget = max(
            0, int(self._get_el_tsbpo_config().get("branch_budget", 0))
        )
        target_local_budgets = {
            task_uid: count * 2 for task_uid, count in factual_counts.items()
        }
        if configured_branch_budget > 0:
            target_total = sum(target_local_budgets.values())
            if target_total > configured_branch_budget:
                scale = configured_branch_budget / max(1, target_total)
                target_local_budgets = {
                    task_uid: int(value * scale)
                    for task_uid, value in target_local_budgets.items()
                }
                remainder = configured_branch_budget - sum(
                    target_local_budgets.values()
                )
                for task_uid in sorted(target_local_budgets):
                    if remainder <= 0:
                        break
                    target_local_budgets[task_uid] += 1
                    remainder -= 1
        success_rate = success.get(
            "success_rate", np.zeros(len(total_batch_list), dtype=bool)
        )
        factual_outcomes = {
            str(traj_uid[index]): float(bool(success_rate[index]))
            for index in range(min(len(traj_uid), len(success_rate)))
        }
        factual_task_ids = {
            str(traj_uid[index]): str(steps[0].get("uid", ""))
            for index, steps in enumerate(total_batch_list)
            if steps and index < len(traj_uid)
        }
        allocations, allocation_metrics = self._allocate_agent_dps(
            candidates,
            task_budgets=target_local_budgets,
            factual_outcomes=factual_outcomes,
            factual_task_ids=factual_task_ids,
        )
        metrics.update(allocation_metrics)
        state_group_by_anchor = self._el_tsbpo_state_group_by_anchor
        repeated_state_records = [
            record
            for record in self._el_tsbpo_state_group_records
            if int(record.get("candidate_count", 0)) > 1
        ]
        for chunk_start in range(0, len(repeated_state_records), 64):
            chunk = repeated_state_records[chunk_start : chunk_start + 64]
            print(
                "[EL-TSBPO-STATE-GROUP] "
                f"step={self._el_tsbpo_step} "
                f"n_repeated={len(repeated_state_records)} "
                f"offset={chunk_start} "
                f"groups={chunk}",
                flush=True,
            )
        factual_trajectory_count = sum(1 for steps in total_batch_list if steps)
        metrics.update(
            {
                "el_tsbpo/allocator/n_factual_trajectories": float(
                    factual_trajectory_count
                ),
                "el_tsbpo/allocator/target_local_budget_total": float(
                    sum(target_local_budgets.values())
                ),
                "el_tsbpo/allocator/branches_per_factual_trajectory": float(
                    metrics.get("el_tsbpo/allocator/branches_allocated", 0.0)
                    / max(1, factual_trajectory_count)
                ),
            }
        )
        candidate_by_id = {candidate.anchor_id: candidate for candidate in candidates}
        cfg = self._get_el_tsbpo_config()
        branch_envs = self._get_branch_envs()
        history_length = int(self.config.env.get("history_length", 5))
        max_steps = int(self.config.env.max_steps)
        all_branch_rows: List[Dict] = []
        valid_groups: Dict[str, Tuple[Any, Any, float]] = {}


        max_anchors = max(
            1, int(cfg.get("max_anchors_per_batch", cfg.get("branch_budget", 128)))
        )
        worker_capacity = max(1, int(branch_envs.max_workers))
        continuous_refill = _is_truthy_flag(cfg.get("branch_continuous_refill", False))
        allocation_batches: List[List[Any]] = []
        current_batch: List[Any] = []
        current_continuations = 0
        for allocation in allocations:
            continuation_count = int(allocation.continuation_count)
            if continuation_count > worker_capacity:
                raise ValueError(
                    "one EL-TSBPO sibling group exceeds branch worker capacity: "
                    f"k={continuation_count}, capacity={worker_capacity}. "
                    "Increase algorithm.el_tsbpo.branch_worker_pool_size."
                )
            if continuous_refill:
                continue
            if current_batch and (
                len(current_batch) >= max_anchors
                or current_continuations + continuation_count > worker_capacity
            ):
                allocation_batches.append(current_batch)
                current_batch = []
                current_continuations = 0
            current_batch.append(allocation)
            current_continuations += continuation_count
        if continuous_refill:


            allocation_batches = [list(allocations)] if allocations else []
        elif current_batch:
            allocation_batches.append(current_batch)
        allocated_continuations = sum(
            int(item.continuation_count) for item in allocations
        )
        metrics["el_tsbpo/rollout/branch_submission_waves"] = float(
            len(allocation_batches)
        )
        metrics["el_tsbpo/rollout/branch_environment_cohorts"] = float(
            (allocated_continuations + worker_capacity - 1) // worker_capacity
            if allocated_continuations
            else 0
        )
        metrics["el_tsbpo/rollout/branch_continuous_refill"] = float(continuous_refill)
        metrics["el_tsbpo/rollout/branch_worker_capacity"] = float(worker_capacity)
        allocation_offset = 0
        for allocation_batch in allocation_batches:
            batch_offset = allocation_offset
            allocation_offset += len(allocation_batch)
            all_nodes: Dict[str, TreeNode] = {}
            spawn_specs: List[Dict[str, Any]] = []
            branch_info: List[Dict[str, Any]] = []
            group_metadata: Dict[str, Tuple[Any, Any]] = {}
            for offset, allocation in enumerate(allocation_batch):
                candidate = candidate_by_id.get(allocation.anchor_id)
                if candidate is None:
                    continue
                group_id = (
                    f"el_tsbpo_g{self._el_tsbpo_step}_"
                    f"{batch_offset + offset}_{candidate.anchor_id}"
                )
                node = TreeNode(


                    node_id=candidate.node_id,
                    parent_id=candidate.parent_node_id or None,
                    depth=candidate.step_idx,
                    task=(candidate.task),
                    gamefile=candidate.gamefile,
                    action_history=list(candidate.action_history),
                    memory_steps=[dict(step) for step in candidate.memory_steps],
                    branch_type="bare",
                    branch_group_id=group_id,
                )
                all_nodes[node.node_id] = node
                group_metadata[group_id] = (candidate, allocation)
                for branch_idx in range(allocation.continuation_count):
                    spawn_spec = {
                        "gamefile": candidate.gamefile,
                        "action_history": list(candidate.action_history),
                    }
                    spawn_specs.append(spawn_spec)
                    branch_info.append(
                        {
                            "env_idx": candidate.env_idx,
                            "t_star_idx": candidate.step_idx,
                            "branch_node": node,
                            "branch_idx": branch_idx,
                            "is_counterfactual": False,
                            "branch_group_id": group_id,
                            "parent_step_idx": candidate.step_idx,
                            "parent_traj_uid": candidate.trajectory_id,
                            "natural_action": "",
                            "branch_anchor_obs": candidate.anchor_obs,
                            "expected_context_hash": candidate.context_hash,
                            "expected_anchor_obs": candidate.anchor_obs,
                            "expected_decision_key": candidate.decision_key,
                            "expected_policy_group_key": candidate.policy_group_key,
                            "expected_env_key": candidate.env_key,
                            "expected_admissible_commands": list(
                                candidate.admissible_commands
                            ),
                            "el_tsbpo_anchor_node_id": candidate.node_id,
                            "el_tsbpo_anchor_parent_node_id": candidate.parent_node_id,
                            "el_tsbpo_policy_version": self._active_policy_version,


                            "spawn_spec": spawn_spec,
                        }
                    )
            if not spawn_specs:
                continue
            worker_indices: List[int] = []
            if continuous_refill:
                for row in branch_info:
                    row["worker_idx"] = -1
            else:
                worker_indices = branch_envs.spawn_branches_parallel(spawn_specs)
                for row, worker_idx in zip(branch_info, worker_indices):
                    row["worker_idx"] = worker_idx
            started = time.perf_counter()
            branch_rows = self._run_branch_rollouts(
                branch_info=branch_info,
                branch_envs=branch_envs,
                actor_rollout_wg=actor_rollout_wg,
                all_nodes=all_nodes,
                k_nat=max(
                    (item.continuation_count for item in allocation_batch), default=0
                ),
                max_steps=max_steps,
                history_length=history_length,
                total_batch_list=total_batch_list,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                success=success,
                tool_callings=tool_callings,
            )
            elapsed = time.perf_counter() - started
            metrics["el_tsbpo/rollout/actual_seconds"] += float(elapsed)
            allocated_in_batch = max(
                1, sum(item.continuation_count for item in allocation_batch)
            )
            rows_by_trajectory: Dict[str, List[Dict]] = defaultdict(list)
            for row in branch_rows:
                rows_by_trajectory[str(row.get("traj_uid", ""))].append(row)
            observed_by_group: Dict[str, int] = defaultdict(int)
            for trajectory_rows in rows_by_trajectory.values():
                if trajectory_rows:
                    group_id = str(trajectory_rows[0].get("branch_group_id", "") or "")
                    observed_by_group[group_id] += 1
            for trajectory_rows in rows_by_trajectory.values():
                first = trajectory_rows[0]
                group_id = str(first.get("branch_group_id", "") or "")
                metadata = group_metadata.get(group_id)
                if metadata is None:
                    continue
                candidate, allocation = metadata
                metrics["el_tsbpo/audit/checked_trajectories"] += 1.0


                exact = all(
                    first.get(key) is True
                    for key in (
                        "context_matches_parent",
                        "decision_matches_parent",
                        "env_matches_parent",
                    )
                )


                if candidate.policy_group_key:
                    exact = exact and (first.get("policy_group_matches_parent") is True)
                if candidate.anchor_obs:
                    exact = exact and (
                        first.get("replay_observation_matches_parent") is True
                    )
                if candidate.admissible_commands:
                    exact = exact and (
                        first.get("replay_admissible_matches_parent") is True
                    )


                exact = exact and (
                    observed_by_group[group_id] == int(allocation.continuation_count)
                )
                audit_requirements = {
                    "context": first.get("context_matches_parent") is True,
                    "decision": first.get("decision_matches_parent") is True,
                    "environment": first.get("env_matches_parent") is True,
                }
                if candidate.policy_group_key:
                    audit_requirements["policy_group"] = (
                        first.get("policy_group_matches_parent") is True
                    )
                if candidate.anchor_obs:
                    audit_requirements["replay_observation"] = (
                        first.get("replay_observation_matches_parent") is True
                    )
                if candidate.admissible_commands:
                    audit_requirements["replay_admissible"] = (
                        first.get("replay_admissible_matches_parent") is True
                    )
                audit_requirements["cardinality"] = observed_by_group[group_id] == int(
                    allocation.continuation_count
                )
                for requirement, passed in audit_requirements.items():
                    if not passed:
                        metrics[f"el_tsbpo/audit/fail_{requirement}"] += 1.0
                group_seconds = (
                    elapsed * allocation.continuation_count / allocated_in_batch
                )
                if not exact:
                    metrics["el_tsbpo/rollout/n_quarantined_trajectories"] += 1.0
                else:
                    valid_groups[group_id] = (candidate, allocation, group_seconds)
                    metrics["el_tsbpo/rollout/n_exact_sibling_trajectories"] += 1.0
                for row in trajectory_rows:
                    state_info = state_group_by_anchor.get(str(candidate.anchor_id), {})
                    row["el_tsbpo_sibling_group_id"] = group_id if exact else None
                    row["el_tsbpo_is_extra"] = True
                    row["el_tsbpo_anchor_context"] = candidate.context if exact else ""
                    row["el_tsbpo_allocated_k"] = allocation.continuation_count
                    row["el_tsbpo_rollout_seconds"] = group_seconds
                    row["el_tsbpo_anchor_node_id"] = candidate.node_id if exact else ""
                    row["el_tsbpo_state_group_key"] = state_info.get(
                        "state_group_key", ""
                    )
                    row["el_tsbpo_state_group_candidate_count"] = int(
                        state_info.get("candidate_count", 0)
                    )
                    row["el_tsbpo_state_group_selected_count"] = int(
                        state_info.get("selected_count", 0)
                    )
                    row["el_tsbpo_state_group_allocated_k"] = int(
                        state_info.get("allocated_k", 0)
                    )
                    row["el_tsbpo_state_group_actual_k"] = int(
                        state_info.get("actual_k", 1)
                    )
                    row["el_tsbpo_allocation_provenance"] = (
                        allocation.provenance if exact else "quarantined"
                    )
                    row["el_tsbpo_quarantined"] = not exact
                    row["el_tsbpo_group_memberships"] = (
                        [
                            {
                                "group_id": group_id,
                                "anchor_node_id": candidate.node_id,
                                "context": candidate.context,
                                "allocated_k": allocation.continuation_count,
                                "allocation_provenance": allocation.provenance,
                                "selection_probability": allocation.selection_probability,
                                "state_group_key": state_info.get(
                                    "state_group_key", ""
                                ),
                                "state_group_candidate_count": int(
                                    state_info.get("candidate_count", 0)
                                ),
                                "state_group_selected_count": int(
                                    state_info.get("selected_count", 0)
                                ),
                                "state_group_allocated_k": int(
                                    state_info.get("allocated_k", 0)
                                ),
                                "state_group_actual_k": int(
                                    state_info.get("actual_k", 1)
                                ),
                                "policy_version": row.get(
                                    "el_tsbpo_policy_version",
                                    self._active_policy_version,
                                ),
                            }
                        ]
                        if exact
                        else []
                    )


            for group_id, (candidate, allocation) in group_metadata.items():
                expected = int(allocation.continuation_count)
                observed = int(observed_by_group.get(group_id, 0))
                if observed < expected:
                    metrics["el_tsbpo/rollout/n_quarantined_trajectories"] += float(
                        expected - observed
                    )
                    metrics["el_tsbpo/audit/fail_cardinality"] += float(
                        expected - observed
                    )
            all_branch_rows.extend(branch_rows)
            for worker_idx in worker_indices:
                if worker_idx >= 0:
                    branch_envs.close_branch(worker_idx)


        sibling_ids = output.non_tensor_batch["el_tsbpo_sibling_group_id"]
        anchor_contexts = output.non_tensor_batch["el_tsbpo_anchor_context"]
        allocated_k = output.non_tensor_batch["el_tsbpo_allocated_k"]
        rollout_seconds = output.non_tensor_batch["el_tsbpo_rollout_seconds"]
        anchor_node_ids = output.non_tensor_batch["el_tsbpo_anchor_node_id"]
        state_group_keys = output.non_tensor_batch["el_tsbpo_state_group_key"]
        state_group_candidate_counts = output.non_tensor_batch[
            "el_tsbpo_state_group_candidate_count"
        ]
        state_group_selected_counts = output.non_tensor_batch[
            "el_tsbpo_state_group_selected_count"
        ]
        state_group_allocated_k = output.non_tensor_batch[
            "el_tsbpo_state_group_allocated_k"
        ]
        group_memberships = output.non_tensor_batch["el_tsbpo_group_memberships"]
        output_traj = output.non_tensor_batch["traj_uid"]
        output_task = output.non_tensor_batch["uid"]
        output_turn = output.non_tensor_batch.get("turn_idx", np.arange(bsz))
        for group_id, (candidate, allocation, elapsed) in valid_groups.items():
            state_info = state_group_by_anchor.get(str(candidate.anchor_id), {})
            root_anchor = str(candidate.node_id).endswith(":root")
            for row in range(bsz):
                if root_anchor:
                    if str(output_task[row]) != candidate.task_uid:
                        continue
                elif str(output_traj[row]) != candidate.trajectory_id:
                    continue


                if int(output_turn[row]) != candidate.step_idx:
                    continue
                sibling_ids[row] = group_id
                anchor_contexts[row] = candidate.context
                allocated_k[row] = allocation.continuation_count
                rollout_seconds[row] = elapsed
                anchor_node_ids[row] = candidate.node_id
                state_group_keys[row] = state_info.get("state_group_key", "")
                state_group_candidate_counts[row] = int(
                    state_info.get("candidate_count", 0)
                )
                state_group_selected_counts[row] = int(
                    state_info.get("selected_count", 0)
                )
                state_group_allocated_k[row] = int(state_info.get("allocated_k", 0))
                output.non_tensor_batch["el_tsbpo_state_group_actual_k"][row] = int(
                    state_info.get("actual_k", 1)
                )
                output.non_tensor_batch["el_tsbpo_allocation_provenance"][row] = (
                    allocation.provenance
                )
                memberships = (
                    list(group_memberships[row])
                    if isinstance(group_memberships[row], (list, tuple))
                    else []
                )
                memberships.append(
                    {
                        "group_id": group_id,
                        "anchor_node_id": candidate.node_id,
                        "context": candidate.context,
                        "allocated_k": allocation.continuation_count,
                        "allocation_provenance": allocation.provenance,
                        "selection_probability": allocation.selection_probability,
                        "state_group_key": state_info.get("state_group_key", ""),
                        "state_group_candidate_count": int(
                            state_info.get("candidate_count", 0)
                        ),
                        "state_group_selected_count": int(
                            state_info.get("selected_count", 0)
                        ),
                        "state_group_allocated_k": int(
                            state_info.get("allocated_k", 0)
                        ),
                        "state_group_actual_k": int(state_info.get("actual_k", 1)),
                        "policy_version": output.non_tensor_batch.get(
                            "el_tsbpo_policy_version",
                            np.asarray(["unknown"] * bsz, dtype=object),
                        )[row],
                    }
                )
                group_memberships[row] = memberships

        if self.is_agent_dps_enabled() and all_branch_rows:
            from agent_system.multi_turn_rollout.agent_dps import branch_contrast_counts

            memory = self._get_agent_dps_memory()
            metrics.update(
                self._agent_dps_observe_continuations(all_branch_rows, branch=True)
            )
            (
                branch_observations,
                descendant_observations,
                branch_transition_observations,
                branch_outcomes_by_state,
                branch_first_edges_by_state,
            ) = _agent_dps_branch_evidence(all_branch_rows)
            committed_anchor_observations = 0
            committed_descendant_observations = 0
            committed_branch_transitions = 0
            selected_records = [
                record
                for record in self._el_tsbpo_state_group_records
                if int(record.get("allocated_k", 0)) > 0
            ]
            observed_state_count = 0
            observed_mixed_count = 0
            homogeneous_factual_count = 0
            opposite_branch_count = 0
            opposite_branch_count_in_bare_mixed_tasks = 0
            transition_contrast_state_count = 0
            transition_contrast_branch_count = 0
            terminal_contrast_state_count = 0
            terminal_contrast_branch_count = 0
            same_first_edge_terminal_count = 0
            branch_first_edge_count = 0
            predicted_contrast = []
            transition_contrast_labels = []
            objective_contrast_labels = []
            factual_successes = 0
            factual_total = 0
            branch_values = []
            for record in selected_records:
                key = str(record.get("state_group_key", "") or "")
                state_branch_values = list(
                    branch_outcomes_by_state.get(key, {}).values()
                )
                if not state_branch_values:
                    continue
                observed_state_count += 1
                branch_values.extend(state_branch_values)
                factual_outcome = float(record.get("factual_outcome", 0.0))
                factual_edge = (
                    str(record.get("factual_action", "") or ""),
                    str(record.get("factual_next_state", "") or ""),
                )
                branch_edges = branch_first_edges_by_state.get(key, {})
                contrasts = branch_contrast_counts(
                    factual_outcome,
                    branch_outcomes_by_state.get(key, {}),
                    factual_edge=factual_edge,
                    branch_edges=branch_edges,
                )
                branch_first_edge_count += contrasts["edge_coverage"]
                transition_contrast_branch_count += contrasts["transition"]
                terminal_contrast_branch_count += contrasts["terminal"]
                same_first_edge_terminal_count += contrasts["same_first_edge"]
                transition_label = float(contrasts["transition"] > 0)
                terminal_label = float(contrasts["terminal"] > 0)
                record["observed_branch_count"] = len(state_branch_values)
                record["observed_branch_success_count"] = sum(
                    value >= 0.5 for value in state_branch_values
                )
                record["observed_terminal_contrast"] = bool(terminal_label)
                terminal_contrast_state_count += int(terminal_label)
                transition_contrast_state_count += int(transition_label)
                transition_contrast_labels.append(transition_label)
                objective_contrast_labels.append(terminal_label)
                predicted_contrast.append(
                    float(record.get("predicted_created_contrast", 0.0))
                )
                factual_success = int(record.get("factual_success_count", 0))
                factual_failure = int(record.get("factual_failure_count", 0))
                factual_successes += factual_success
                factual_total += factual_success + factual_failure
                combined_success = factual_success + sum(
                    value >= 0.5 for value in state_branch_values
                )
                combined_failure = factual_failure + sum(
                    value < 0.5 for value in state_branch_values
                )
                observed_mixed_count += int(
                    combined_success > 0 and combined_failure > 0
                )
                factual_is_homogeneous = factual_success + factual_failure > 0 and (
                    factual_success == 0 or factual_failure == 0
                )
                if factual_is_homogeneous:
                    homogeneous_factual_count += 1
                    created_local_contrast = int(
                        (
                            factual_success > 0
                            and any(value < 0.5 for value in state_branch_values)
                        )
                        or (
                            factual_failure > 0
                            and any(value >= 0.5 for value in state_branch_values)
                        )
                    )
                    opposite_branch_count += created_local_contrast
                    if record.get("bare_task_mixed", False):
                        opposite_branch_count_in_bare_mixed_tasks += created_local_contrast
            branch_success_rate = (
                float(sum(value >= 0.5 for value in branch_values) / len(branch_values))
                if branch_values
                else 0.0
            )
            factual_success_rate = (
                float(factual_successes / factual_total) if factual_total else 0.0
            )
            predicted_contrast_array = np.asarray(predicted_contrast, dtype=np.float64)
            transition_label_array = np.asarray(
                transition_contrast_labels, dtype=np.float64
            )
            objective_label_array = np.asarray(
                objective_contrast_labels, dtype=np.float64
            )
            observed_objective_contrast_rate = float(
                objective_label_array.mean() if objective_label_array.size else 0.0
            )
            created_contrast_groups = int(objective_label_array.sum())
            predicted_contrast_mean = float(
                predicted_contrast_array.mean()
                if predicted_contrast_array.size
                else 0.0
            )
            observed_transition_contrast_rate = float(
                transition_label_array.mean() if transition_label_array.size else 0.0
            )
            metrics.update(
                {
                    "agent_dps/outcome/branch_trajectories": float(len(branch_values)),
                    "agent_dps/outcome/anchor_observations_committed": float(
                        committed_anchor_observations
                    ),
                    "agent_dps/outcome/descendant_observations_committed": float(
                        committed_descendant_observations
                    ),
                    "agent_dps/outcome/descendant_states_observed": float(
                        len({key for key, _, _ in descendant_observations})
                    ),
                    "agent_dps/outcome/branch_transitions_committed": float(
                        committed_branch_transitions
                    ),
                    "agent_dps/outcome/branch_success_rate": branch_success_rate,
                    "agent_dps/outcome/selected_factual_success_rate": factual_success_rate,
                    "agent_dps/outcome/branch_minus_factual_success_rate": float(
                        branch_success_rate - factual_success_rate
                    ),
                    "agent_dps/outcome/selected_states_with_branch_evidence": float(
                        observed_state_count
                    ),
                    "agent_dps/outcome/observed_mixed_state_rate": float(
                        observed_mixed_count / max(1, observed_state_count)
                    ),
                    "agent_dps/outcome/selected_factual_mixed_state_rate": float(
                        (observed_mixed_count - opposite_branch_count)
                        / max(1, observed_state_count)
                    ),
                    "agent_dps/outcome/new_mixed_states": float(opposite_branch_count),
                    "agent_dps/outcome/new_mixed_states_in_bare_mixed_tasks": float(
                        opposite_branch_count_in_bare_mixed_tasks
                    ),
                    "agent_dps/outcome/mixed_state_rate_gain": float(
                        opposite_branch_count / max(1, observed_state_count)
                    ),
                    "agent_dps/outcome/homogeneous_factual_opposite_branch_rate": float(
                        opposite_branch_count / max(1, homogeneous_factual_count)
                    ),
                    "agent_dps/outcome/branch_first_edge_coverage": float(
                        branch_first_edge_count / max(1, len(branch_values))
                    ),
                    "agent_dps/outcome/transition_contrast_state_rate": float(
                        observed_transition_contrast_rate
                    ),
                    "agent_dps/outcome/transition_contrast_branch_rate": float(
                        transition_contrast_branch_count / max(1, len(branch_values))
                    ),
                    "agent_dps/outcome/terminal_contrast_state_rate": float(
                        terminal_contrast_state_count / max(1, observed_state_count)
                    ),
                    "agent_dps/outcome/terminal_contrast_branch_rate": float(
                        terminal_contrast_branch_count / max(1, len(branch_values))
                    ),
                    "agent_dps/outcome/same_first_edge_terminal_contrast_branch_rate": float(
                        same_first_edge_terminal_count / max(1, len(branch_values))
                    ),
                    "agent_dps/outcome/created_contrast_groups": float(
                        created_contrast_groups
                    ),

                    "agent_dps/outcome/created_contrast_states": float(
                        created_contrast_groups
                    ),
                    "agent_dps/outcome/created_contrast_per_branch": float(
                        created_contrast_groups / max(1, len(branch_values))
                    ),
                    "agent_dps/outcome/predicted_created_contrast_rate": float(
                        predicted_contrast_mean
                    ),
                    "agent_dps/outcome/contrast_calibration_bias": float(
                        observed_objective_contrast_rate - predicted_contrast_mean
                    ),
                    "agent_dps/outcome/contrast_calibration_abs_error": float(
                        abs(observed_objective_contrast_rate - predicted_contrast_mean)
                    ),
                    "agent_dps/outcome/contrast_brier_score": float(
                        np.mean((predicted_contrast_array - objective_label_array) ** 2)
                        if predicted_contrast_array.size
                        else 0.0
                    ),
                }
            )
            if self._get_el_tsbpo_config().get("agent_dps_probability_calibration", False):
                from agent_system.multi_turn_rollout.dps_probability_calibration import (
                    audited_branch_groups,
                    observe_groups,
                )

                calibration_records, incomplete = audited_branch_groups(
                    selected_records, all_branch_rows, int(self.config.env.max_steps)
                )
                nll = []
                model = self._get_agent_dps_probability_calibrator("branch")
                for record in calibration_records:
                    nll.append(
                        -model.forecast(
                            record["task"],
                            (record["history_success"], record["history_failure"]),
                            (record["semantic_success"], record["semantic_failure"]),
                            current=(
                                record["current_success"],
                                record["current_failure"],
                            ),
                        ).count_log_probability(record["success"], record["failure"])
                    )
                committed = observe_groups(
                    memory.probability_calibration,
                    "branch",
                    calibration_records,
                    self._el_tsbpo_step,
                    max_entries=memory.max_states,
                )
                metrics["agent_dps/probability/branch/groups_committed"] = float(
                    committed
                )
                metrics["agent_dps/probability/branch/incomplete_groups"] = float(
                    incomplete
                )
                metrics["agent_dps/probability/branch/count_nll"] = (
                    float(np.mean(nll)) if nll else 0.0
                )
                memory.dirty = True
            metrics.update(memory.metrics(self._el_tsbpo_step))
            task_results = []
            if self._get_el_tsbpo_config().get("agent_dps_branch_task_union", False):
                labels = {
                    t: set(v) for t, v in self._agent_dps_bare_task_labels.items()
                }
                initial_mixed = sum(len(v) == 2 for v in labels.values())
                for record in selected_records:
                    task = str(record["task_ref"])
                    labels.setdefault(task, set()).update(
                        value >= 0.5
                        for value in branch_outcomes_by_state.get(
                            record["state_group_key"], {}
                        ).values()
                    )
                final_mixed = sum(len(v) == 2 for v in labels.values())
                for plan in self._agent_dps_task_union_plans:
                    actual = int(
                        not plan.get("already_mixed_bare", False)
                        and len(labels.get(plan["task_ref"], ())) == 2
                    )
                    task_results.append(dict(**plan, observed_activation=bool(actual)))
                metrics.update(
                    {
                        "agent_dps/outcome/new_mixed_tasks": float(
                            final_mixed - initial_mixed
                        ),
                        "agent_dps/outcome/post_branch_mixed_task_ratio": float(
                            final_mixed / max(1, len(labels))
                        ),
                        "agent_dps/outcome/task_activation_brier": float(
                            np.mean(
                                [
                                    (
                                        p["predicted_activation"]
                                        - p["observed_activation"]
                                    )
                                    ** 2
                                    for p in task_results
                                ]
                            )
                        )
                        if task_results
                        else 0.0,
                    }
                )
            if self._get_el_tsbpo_config().get("agent_dps_allocation_diagnostics_dir"):
                self._write_agent_dps_allocation(
                    "branch_outcomes",
                    {
                        "selected_states": selected_records,
                        "task_outcomes": task_results,
                        "metrics": {
                            key: float(value)
                            for key, value in metrics.items()
                            if key.startswith(
                                ("agent_dps/outcome/", "agent_dps/probability/branch/")
                            )
                        },
                    },
                )
            self._agent_dps_memory_dirty = True
            self._save_agent_dps_memory()


        if all_branch_rows:
            output = self._concat_branch_turn_data(output, all_branch_rows)
        if self.is_agent_dps_enabled() and is_train:
            metrics["sampling/effective_ratio"] = compute_effective_ratio(
                index=output.non_tensor_batch["uid"],
                traj_index=output.non_tensor_batch["traj_uid"],
                trajectory_success=output.non_tensor_batch["trajectory_success"],
            )
        quarantined = output.non_tensor_batch.get(
            "el_tsbpo_quarantined", np.zeros(len(output.batch), dtype=bool)
        )
        output.batch["el_tsbpo_sample_weight"] = torch.as_tensor(
            np.logical_not(np.asarray(quarantined, dtype=bool)),
            dtype=torch.float32,
            device=output.batch["input_ids"].device,
        )
        if "loss_mask" not in output.batch:
            output.batch["loss_mask"] = output.batch["attention_mask"].clone()
        advantage_mode = (
            str(self._get_el_tsbpo_config().get("advantage_mode", "tree"))
            .lower()
            .replace("-", "_")
        )
        if advantage_mode in {
            "tree_gigpo",
            "tree_gigpo_s",
            "tree_gigpo_simple",
            "graph_gigpo",
            "graph_gigpo_lite",
            "state_action_gigpo",
        }:
            self._attach_el_tsbpo_action_token_mask(output)
        metrics["el_tsbpo/rollout/n_extra_turns"] = float(len(all_branch_rows))
        metrics["el_tsbpo/rollout/branch_worker_pool_size"] = float(
            branch_envs.num_total_slots
        )
        metrics["el_tsbpo/rollout/branch_worker_pool_alive"] = float(
            branch_envs.num_alive
        )
        metrics["el_tsbpo/rollout/branch_worker_pool_active"] = float(
            branch_envs.num_active
        )
        metrics["el_tsbpo/rollout/branch_workers_created"] = float(
            branch_envs.num_created_workers
        )
        metrics["el_tsbpo/rollout/branch_workers_reconfigured"] = float(
            branch_envs.num_reconfigured_workers
        )
        metrics["el_tsbpo/rollout/branch_same_gamefile_reuses"] = float(
            branch_envs.num_same_gamefile_reuses
        )
        generation_stats = self._el_tsbpo_generation_stats
        branch_stats = generation_stats["branch"]
        natural_stats = generation_stats["natural"]

        def _safe_rate(numerator: float, denominator: float) -> float:
            return float(numerator / denominator) if denominator > 0 else 0.0

        def _generation_metrics(
            prefix: str, values: Dict[str, Any]
        ) -> Dict[str, float]:
            rows = float(values["rows"])
            batches = float(values["batch_count"])
            wall = float(values["wall_seconds"])
            prompt_tokens = float(values["prompt_tokens"])
            response_tokens = float(values["response_tokens"])
            padded_rows = float(values["padded_rows"])
            physical_rows = rows + padded_rows
            return {
                f"el_tsbpo/rollout/{prefix}_worker_calls": float(
                    values["worker_calls"]
                ),
                f"el_tsbpo/rollout/{prefix}_rows": rows,
                f"el_tsbpo/rollout/{prefix}_prompt_tokens": prompt_tokens,
                f"el_tsbpo/rollout/{prefix}_response_tokens": response_tokens,
                f"el_tsbpo/rollout/{prefix}_wall_seconds": wall,
                f"el_tsbpo/rollout/{prefix}_rows_per_second": _safe_rate(rows, wall),
                f"el_tsbpo/rollout/{prefix}_response_tokens_per_second": _safe_rate(
                    response_tokens, wall
                ),
                f"el_tsbpo/rollout/{prefix}_prompt_tokens_per_second": _safe_rate(
                    prompt_tokens, wall
                ),
                f"el_tsbpo/rollout/{prefix}_ms_per_response_token": _safe_rate(
                    wall * 1000.0, response_tokens
                ),
                f"el_tsbpo/rollout/{prefix}_response_tokens_per_row": _safe_rate(
                    response_tokens, rows
                ),
                f"el_tsbpo/rollout/{prefix}_mean_batch_rows": _safe_rate(
                    float(values["batch_rows"]), batches
                ),
                f"el_tsbpo/rollout/{prefix}_mean_batch_prompt_tokens": _safe_rate(
                    float(values["batch_prompt_tokens"]), batches
                ),
                f"el_tsbpo/rollout/{prefix}_padding_fraction": _safe_rate(
                    padded_rows, physical_rows
                ),
                f"el_tsbpo/rollout/{prefix}_planned_tokens": float(
                    values["batch_planned_tokens"]
                ),
                f"el_tsbpo/rollout/{prefix}_worker_call_ms_mean": _safe_rate(
                    wall * 1000.0, float(values["worker_calls"])
                ),
            }

        metrics.update(
            {


                "el_tsbpo/rollout/branch_generation_calls": float(
                    generation_stats["calls"]
                ),
                "el_tsbpo/rollout/branch_generation_worker_calls": float(
                    branch_stats["worker_calls"]
                ),
                "el_tsbpo/rollout/branch_generation_batches": float(
                    generation_stats["batches"]
                ),
                "el_tsbpo/rollout/branch_generation_rows": float(branch_stats["rows"]),
                "el_tsbpo/rollout/branch_generation_max_batch_rows": float(
                    generation_stats["max_batch_rows"]
                ),
                "el_tsbpo/rollout/branch_generation_token_budget": float(
                    generation_stats["token_budget"]
                ),
                "el_tsbpo/rollout/branch_generation_local_token_packing_enabled": float(
                    generation_stats.get("branch_local_token_packing", 1)
                ),
                "el_tsbpo/rollout/branch_generation_prompt_tokens": float(
                    branch_stats["prompt_tokens"]
                ),
                "el_tsbpo/rollout/branch_generation_planned_tokens": float(
                    branch_stats["batch_planned_tokens"]
                ),
                "el_tsbpo/rollout/branch_generation_token_budget_utilization": _safe_rate(
                    float(branch_stats["batch_planned_tokens"]),
                    float(generation_stats["token_budget"])
                    * float(branch_stats["batch_count"])
                    if bool(generation_stats.get("branch_local_token_packing", 1))
                    else 0.0,
                ),
                "el_tsbpo/rollout/generation_worker_calls": float(
                    generation_stats["worker_calls"]
                ),
                "el_tsbpo/rollout/generation_rows": float(generation_stats["rows"]),
                "el_tsbpo/rollout/generation_prompt_tokens": float(
                    generation_stats["prompt_tokens"]
                ),
                "el_tsbpo/rollout/generation_response_tokens": float(
                    generation_stats["response_tokens"]
                ),
                "el_tsbpo/rollout/generation_wall_seconds": float(
                    generation_stats["wall_seconds"]
                ),
                "el_tsbpo/rollout/generation_tokenize_seconds": float(
                    generation_stats["tokenize_seconds"]
                ),
                "el_tsbpo/rollout/generation_rows_per_second": _safe_rate(
                    float(generation_stats["rows"]),
                    float(generation_stats["wall_seconds"]),
                ),
                "el_tsbpo/rollout/generation_response_tokens_per_second": _safe_rate(
                    float(generation_stats["response_tokens"]),
                    float(generation_stats["wall_seconds"]),
                ),
                "el_tsbpo/rollout/generation_effective_ms_per_response_token": _safe_rate(
                    float(generation_stats["wall_seconds"]) * 1000.0,
                    float(generation_stats["response_tokens"]),
                ),
                "el_tsbpo/rollout/generation_padding_fraction": _safe_rate(
                    float(generation_stats["padded_rows"]),
                    float(generation_stats["rows"] + generation_stats["padded_rows"]),
                ),
                "el_tsbpo/rollout/env_step_seconds": float(
                    generation_stats["env_step_seconds"]
                ),
                "el_tsbpo/rollout/branch_env_wait_seconds": float(
                    generation_stats["branch_env_wait_seconds"]
                ),
                "el_tsbpo/rollout/branch_env_submit_seconds": float(
                    generation_stats["branch_env_submit_seconds"]
                ),
                "el_tsbpo/rollout/branch_initial_spawn_seconds": float(
                    generation_stats.get("branch_initial_spawn_seconds", 0.0)
                ),
                "el_tsbpo/rollout/branch_refill_spawn_seconds": float(
                    generation_stats.get("branch_refill_spawn_seconds", 0.0)
                ),
                "el_tsbpo/rollout/branch_ready_batches": float(
                    generation_stats["branch_ready_batches"]
                ),
                "el_tsbpo/rollout/branch_ready_rows": float(
                    generation_stats["branch_ready_rows"]
                ),
                "el_tsbpo/rollout/branch_ready_max_rows": float(
                    generation_stats["branch_ready_max_rows"]
                ),
                "el_tsbpo/rollout/branch_ready_queue_target_rows": float(
                    max(
                        1,
                        int(
                            cfg.get(
                                "branch_ready_queue_batch_size",
                                generation_stats.get("branch_ready_max_rows", 0) or 1,
                            )
                        ),
                    )
                ),
                "el_tsbpo/rollout/branch_ready_queue_timeout_ms": float(
                    cfg.get("branch_ready_queue_timeout_ms", 0)
                ),
                "el_tsbpo/rollout/branch_horizon_bucket_size": float(
                    cfg.get("branch_horizon_bucket_size", 0)
                ),
                "el_tsbpo/rollout/branch_horizon_bucket_count": float(
                    generation_stats.get("branch_horizon_bucket_count", 0)
                ),
                "el_tsbpo/rollout/branch_horizon_bucket_rows": float(
                    generation_stats.get("branch_horizon_bucket_rows", 0)
                ),
                "el_tsbpo/rollout/branch_horizon_bucket_max_rows": float(
                    generation_stats.get("branch_horizon_bucket_max_rows", 0)
                ),
                "el_tsbpo/rollout/branch_refill_spawn_calls": float(
                    generation_stats.get("branch_refill_spawn_calls", 0)
                ),
                "el_tsbpo/rollout/branch_refill_branches": float(
                    generation_stats.get("branch_refill_branches", 0)
                ),
                "el_tsbpo/rollout/branch_active_frontier_batches": float(
                    generation_stats.get("branch_active_frontier_batches", 0)
                ),
                "el_tsbpo/rollout/branch_active_frontier_mean_rows": _safe_rate(
                    float(generation_stats.get("branch_active_frontier_rows", 0)),
                    float(generation_stats.get("branch_active_frontier_batches", 0)),
                ),
                "el_tsbpo/rollout/branch_active_frontier_max_rows": float(
                    generation_stats.get("branch_active_frontier_max_rows", 0)
                ),
                "el_tsbpo/rollout/branch_synchronous_frontier": float(
                    generation_stats.get("branch_synchronous_frontier", 0)
                ),
                "el_tsbpo/rollout/branch_ready_mean_rows": _safe_rate(
                    float(generation_stats["branch_ready_rows"]),
                    float(generation_stats["branch_ready_batches"]),
                ),
                "el_tsbpo/rollout/generation_plus_scheduler_seconds": float(
                    generation_stats["wall_seconds"]
                    + generation_stats["tokenize_seconds"]
                    + generation_stats["env_step_seconds"]
                    + generation_stats["branch_env_wait_seconds"]
                    + generation_stats["branch_env_submit_seconds"]
                ),
            }
        )
        metrics.update(_generation_metrics("natural_generation", natural_stats))
        metrics.update(
            _generation_metrics("branch_generation_efficiency", branch_stats)
        )


        del all_branch_rows, valid_groups, candidate_by_id, candidates, allocations
        gc.collect()
        self._el_tsbpo_step += 1
        output.meta_info["el_tsbpo_metrics"] = metrics
        return output

    def close_predictors(self) -> None:
        if self._agent_dps_memory is not None:
            self._save_agent_dps_memory(force=True)
        self.release_branch_envs()

    def release_branch_envs(self) -> bool:

        if self._branch_envs is None:
            return False
        try:
            self._branch_envs.close_all()
        except Exception as exc:
            print(f"Warning: failed to release branch environments: {exc}")
        finally:
            self._branch_envs = None
        return True

    def multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
        is_train: bool = True,
    ) -> DataProto:
        if str(self.config.algorithm.adv_estimator).lower() != "el_tsbpo":
            raise ValueError("EL-TSBPO requires the el_tsbpo estimator")
        return self.el_tsbpo_multi_turn_loop(
            gen_batch, actor_rollout_wg, envs, is_train
        )
