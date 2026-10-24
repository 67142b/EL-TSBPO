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
Single Process Actor
"""

import itertools
import time
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_gspo, global_valid_turn_count, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor
try:
    from verl.utils.turn_diagnostics import (
        collect_gradient_records,
        eos_statistics_from_logits,
        write_json,
    )
except ModuleNotFoundError as exc:
    if exc.name != "verl.utils.turn_diagnostics":
        raise
    collect_gradient_records = None
    eos_statistics_from_logits = None
    write_json = None

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False,
                             diagnostic_eos_token_ids=None) -> Tuple:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            When EOS diagnostics are requested, a third return holds detached
            (sum_EOS_probability, sum_EOS_p_log_p) or None for fused outputs.
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            diagnostic_eos_statistics = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )
                    if diagnostic_eos_token_ids is not None:
                        valid_eos_ids = [token_id for token_id in diagnostic_eos_token_ids
                                         if 0 <= token_id < logits_rmpad.shape[-1]]
                        if valid_eos_ids:
                            diagnostic_eos_statistics = torch.stack(eos_statistics_from_logits(
                                logits_rmpad, input_ids_rmpad_rolled, log_probs, valid_eos_ids
                            ), dim=-1)

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if diagnostic_eos_statistics is not None:
                        diagnostic_eos_statistics = gather_outpus_and_unpad(
                            diagnostic_eos_statistics, gather_dim=0, unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                if diagnostic_eos_statistics is not None:
                    diagnostic_eos_statistics = pad_input(
                        hidden_states=diagnostic_eos_statistics, indices=indices,
                        batch=batch_size, seqlen=seqlen,
                    )[:, -response_length - 1 : -1, :].unbind(dim=-1)

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if diagnostic_eos_token_ids is not None:
                        valid_eos_ids = [token_id for token_id in diagnostic_eos_token_ids
                                         if 0 <= token_id < logits.shape[-1]]
                        if valid_eos_ids:
                            diagnostic_eos_statistics = eos_statistics_from_logits(
                                logits, micro_batch["responses"], log_probs, valid_eos_ids
                            )
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            if diagnostic_eos_token_ids is not None:
                return entropy, log_probs, diagnostic_eos_statistics
            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        data_device = batch["input_ids"].device if batch is not None else "None"

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for i, micro_batch in enumerate(micro_batches):
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch.to(get_torch_device().current_device()), **micro_batch.non_tensor_batch}
            else:
                micro_batch = micro_batch.to(get_torch_device().current_device())
            try:
                with torch.no_grad():
                    entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            except Exception as e:
                mb_ids = micro_batch["input_ids"] if isinstance(micro_batch, dict) else micro_batch.get("input_ids")
                free_mem = torch.cuda.mem_get_info()[0] / 1024**3
                alloc_mem = torch.cuda.memory_allocated() / 1024**3
                print(f"[EL-TSBPO-ERROR] _forward_micro_batch failed at batch {i}: "
                      f"shape={mb_ids.shape}, GPU free={free_mem:.1f}GB, allocated={alloc_mem:.1f}GB, "
                      f"error={e}", flush=True)
                raise
            log_probs_lst.append(log_probs.cpu())
            if calculate_entropy:
                entropy_lst.append(entropy.cpu())
            del micro_batch

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    def _el_tsbpo_gradient_sketch(self, dimension: int, seed: int) -> tuple[torch.Tensor, int]:
        """CountSketch the current full trainable gradient without gathering it."""

        from el_tsbpo.gradient import fixed_count_sketch

        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if distributed else 0
        world_size = torch.distributed.get_world_size() if distributed else 1
        sharding_strategy = getattr(self.actor_module, "sharding_strategy", None)
        if sharding_strategy is not None and "HYBRID_SHARD" in str(
            sharding_strategy
        ).upper():
            raise NotImplementedError(
                "EL-TSBPO full-gradient sketch currently requires FSDP FULL_SHARD; "
                "HYBRID_SHARD needs mesh-local shard offsets"
            )
        device = get_torch_device().current_device()
        sketch = torch.zeros(dimension, dtype=torch.float32, device=device)
        global_offset = 0

        for _, parameter in self.actor_module.named_parameters():
            if not parameter.requires_grad:
                continue
            unsharded_size = getattr(parameter, "_unpadded_unsharded_size", None)
            full_numel = (
                int(unsharded_size.numel())
                if unsharded_size is not None
                else int(parameter.numel())
            )
            gradient = parameter.grad
            if gradient is not None:
                if hasattr(gradient, "to_local"):
                    raise NotImplementedError(
                        "EL-TSBPO full-gradient sketch does not yet support FSDP2 DTensor "
                        "gradients; use actor.strategy=fsdp"
                    )
                local_gradient = gradient.detach().reshape(-1)
                local_numel = int(local_gradient.numel())
                if local_numel == full_numel:
                    # Replicated parameters must enter the global sketch once.
                    local_start = 0
                    valid_numel = full_numel if rank == 0 else 0
                else:
                    if local_numel * world_size < full_numel:
                        raise RuntimeError(
                            "FSDP gradient shard is smaller than its declared global parameter"
                        )
                    local_start = rank * local_numel
                    valid_numel = max(0, min(local_numel, full_numel - local_start))
                if valid_numel:
                    sketch.add_(
                        fixed_count_sketch(
                            local_gradient[:valid_numel],
                            global_offset=global_offset + local_start,
                            dimension=dimension,
                            seed=seed,
                        )
                    )
            global_offset += full_numel

        if distributed:
            torch.distributed.all_reduce(sketch, op=torch.distributed.ReduceOp.SUM)
        if not torch.isfinite(sketch).all():
            raise FloatingPointError(
                "EL-TSBPO full-parameter gradient sketch contains non-finite values"
            )
        return sketch, global_offset

    def compute_el_tsbpo_gradient_sketch(
        self, data: DataProto
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Extract detached full-parameter policy-score sketches.

        ``per_edge`` expects every data-parallel rank to receive the same rows,
        allowing one synchronized FSDP backward per expanded child. ``probe``
        accepts disjoint DP shards and accumulates bounded micro-batch
        backwards into exactly one shared gradient sketch. No optimizer step or
        optimizer-state mutation occurs in either mode.
        """

        if self.ulysses_sequence_parallel_size > 1:
            raise NotImplementedError(
                "EL-TSBPO gradient sketches require ulysses_sequence_parallel_size=1"
            )
        if self.actor_optimizer is None:
            raise RuntimeError("EL-TSBPO gradient labels require the trainable actor")

        mode = str(data.meta_info.get("el_tsbpo_gradient_mode", "per_edge"))
        if mode not in {"per_edge", "probe"}:
            raise ValueError(f"invalid EL-TSBPO gradient mode: {mode!r}")
        dimension = max(1, int(data.meta_info.get("el_tsbpo_sketch_dim", 128)))
        seed = int(data.meta_info.get("el_tsbpo_sketch_seed", 17))
        temperature = max(float(data.meta_info["temperature"]), 1e-6)
        loss_agg_mode = str(
            data.meta_info.get("el_tsbpo_gradient_loss_agg_mode", "token-mean")
        )
        valid_loss_modes = {
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        }
        if loss_agg_mode not in valid_loss_modes:
            raise ValueError(
                "EL-TSBPO gradient labels received unsupported policy loss "
                f"aggregation mode {loss_agg_mode!r}"
            )
        mask_key = "loss_mask" if "loss_mask" in data.batch else "attention_mask"
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if mask_key not in select_keys:
            select_keys.append(mask_key)
        if mode == "probe":
            select_keys.append("el_tsbpo_gradient_weight")
        batch = data.select(batch_keys=select_keys).batch.to(
            get_torch_device().current_device()
        )
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        world_size = torch.distributed.get_world_size() if distributed else 1
        was_training = self.actor_module.training
        self.actor_module.train()
        outputs: list[torch.Tensor] = []
        trainable_numel = 0
        backward_count = 0
        probe_micro_batch_count = 0
        probe_micro_batch_size = 0

        def backward_and_extract(objective: torch.Tensor) -> torch.Tensor:
            nonlocal backward_count, trainable_numel
            if not torch.isfinite(objective).all():
                raise FloatingPointError(
                    "EL-TSBPO gradient-label objective contains non-finite values"
                )
            self.actor_module.zero_grad(set_to_none=True)
            objective.backward()
            result, trainable_numel = self._el_tsbpo_gradient_sketch(
                dimension, seed
            )
            self.actor_module.zero_grad(set_to_none=True)
            backward_count += 1
            return result.detach().cpu()

        def backward_accumulate(objective: torch.Tensor) -> None:
            """Accumulate one probe micro-batch into the shared gradient.

            The probe objective is additive across micro-batches.  Do not clear
            gradients here: clearing between chunks would make the returned
            sketch depend only on the final chunk.
            """

            nonlocal backward_count, probe_micro_batch_count
            if not torch.isfinite(objective).all():
                raise FloatingPointError(
                    "EL-TSBPO gradient-label objective contains non-finite values"
                )
            objective.backward()
            backward_count += 1
            probe_micro_batch_count += 1

        try:
            if mode == "per_edge":
                for row in range(len(batch)):
                    micro_batch = batch[row : row + 1]
                    _, log_probs = self._forward_micro_batch(
                        micro_batch,
                        temperature=temperature,
                        calculate_entropy=False,
                    )
                    response_length = micro_batch["responses"].shape[-1]
                    response_mask = micro_batch[mask_key][
                        :, -response_length:
                    ].to(log_probs.dtype)
                    token_sum = (log_probs * response_mask).sum()
                    token_count = response_mask.sum().clamp_min(1.0)
                    if loss_agg_mode in {
                        "token-mean",
                        "seq-mean-token-mean",
                    }:
                        objective = token_sum / token_count
                    elif loss_agg_mode == "seq-mean-token-sum":
                        objective = token_sum
                    else:
                        objective = token_sum / max(1, response_length)
                    outputs.append(backward_and_extract(objective))
            else:
                response_length = batch["responses"].shape[-1]
                full_response_mask = batch[mask_key][:, -response_length:].to(
                    torch.float32
                )
                raw_token_count = full_response_mask.sum(dim=-1)
                token_count = raw_token_count.clamp_min(1.0)
                weights = batch["el_tsbpo_gradient_weight"].to(torch.float32)
                valid_rows = (raw_token_count > 0) & (weights != 0)
                if loss_agg_mode == "token-mean":
                    denominator = token_count[valid_rows].sum().detach()
                    if distributed:
                        torch.distributed.all_reduce(
                            denominator, op=torch.distributed.ReduceOp.SUM
                        )
                    denominator = denominator.clamp_min(1.0)
                elif loss_agg_mode == "seq-mean-token-sum":
                    denominator = valid_rows.to(torch.float32).sum().detach()
                    if distributed:
                        torch.distributed.all_reduce(
                            denominator, op=torch.distributed.ReduceOp.SUM
                        )
                    denominator = denominator.clamp_min(1.0)
                elif loss_agg_mode == "seq-mean-token-mean":
                    denominator = valid_rows.to(torch.float32).sum().detach()
                    if distributed:
                        torch.distributed.all_reduce(
                            denominator, op=torch.distributed.ReduceOp.SUM
                        )
                    denominator = denominator.clamp_min(1.0)
                else:
                    denominator = torch.as_tensor(
                        max(1, response_length),
                        dtype=torch.float32,
                        device=batch["input_ids"].device,
                    )
                # ``probe`` used to run one forward/backward over every
                # selected row.  A selected task can contain hundreds of tree
                # turns, so that implementation made activation memory scale
                # with the whole task.  Accumulate the same additive objective
                # over bounded chunks instead.  The denominator is computed
                # globally once, preserving the original objective exactly.
                probe_micro_batch_size = max(
                    1,
                    int(
                        data.meta_info.get(
                            "el_tsbpo_probe_micro_batch_size",
                            data.meta_info.get("micro_batch_size", 1),
                        )
                    ),
                )
                self.actor_module.zero_grad(set_to_none=True)
                for micro_batch in batch.split(probe_micro_batch_size):
                    _, log_probs = self._forward_micro_batch(
                        micro_batch,
                        temperature=temperature,
                        calculate_entropy=False,
                    )
                    micro_response_mask = micro_batch[mask_key][
                        :, -response_length:
                    ].to(log_probs.dtype)
                    token_sums = (log_probs * micro_response_mask).sum(dim=-1)
                    micro_token_count = micro_response_mask.sum(dim=-1).clamp_min(
                        1.0
                    )
                    micro_row_scores = token_sums / micro_token_count
                    micro_weights = micro_batch[
                        "el_tsbpo_gradient_weight"
                    ].to(micro_row_scores.dtype)
                    if loss_agg_mode == "seq-mean-token-mean":
                        numerator = (micro_row_scores * micro_weights).sum()
                    else:
                        numerator = (token_sums * micro_weights).sum()
                    # FSDP averages gradients across DP ranks. Multiplying by
                    # world_size recovers one global numerator before the
                    # fixed denominator.
                    objective = numerator * float(world_size) / denominator
                    backward_accumulate(objective)
                    del (
                        objective,
                        numerator,
                        token_sums,
                        micro_token_count,
                        micro_row_scores,
                        micro_weights,
                        micro_response_mask,
                        log_probs,
                        micro_batch,
                    )
                result, trainable_numel = self._el_tsbpo_gradient_sketch(
                    dimension, seed
                )
                self.actor_module.zero_grad(set_to_none=True)
                outputs.append(result.detach().cpu())
        finally:
            self.actor_module.zero_grad(set_to_none=True)
            self.actor_module.train(was_training)

        sketches = torch.stack(outputs, dim=0) if outputs else torch.zeros(
            (0, dimension), dtype=torch.float32
        )
        if sketches.numel():
            sketch_norms = sketches.norm(dim=-1)
            sketch_norm_mean = float(sketch_norms.mean().item())
            sketch_norm_max = float(sketch_norms.max().item())
        else:
            sketch_norm_mean = 0.0
            sketch_norm_max = 0.0
        return sketches, {
            "backward_count": float(backward_count),
            "probe_micro_batch_count": float(
                probe_micro_batch_count if mode == "probe" else 0
            ),
            "probe_micro_batch_size": float(
                probe_micro_batch_size
                if mode == "probe"
                else 0
            ),
            "trainable_numel": float(trainable_numel),
            "sketch_norm_mean": sketch_norm_mean,
            "sketch_norm_max": sketch_norm_max,
        }

    def compute_el_tsbpo_score_sketch(
        self, data: DataProto
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compatibility entry point for the worker-group RPC."""

        return self.compute_el_tsbpo_gradient_sketch(data)

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)
        is_el_tsbpo_batch = "el_tsbpo_sample_weight" in data.batch
        policy_loss_cfg = self.config.get("policy_loss") or {}
        policy_loss_agg_mode = policy_loss_cfg.get("loss_agg_mode") or self.config.loss_agg_mode
        global_turn_normalization = bool(policy_loss_cfg.get("use_global_turn_normalization", False))
        if global_turn_normalization and (
            policy_loss_agg_mode != "seq-mean-token-sum"
            or self.config.loss_agg_mode != "seq-mean-token-sum"
            or policy_loss_cfg.get("loss_mode", "vanilla") != "vanilla"
        ):
            raise ValueError("Global turn normalization requires vanilla PPO and token-sum PG/KL/entropy")
        diagnostic_context = data.meta_info.get("turn_diagnostics")
        if diagnostic_context and collect_gradient_records is None:
            raise RuntimeError(
                "turn diagnostics were requested, but verl/utils/turn_diagnostics.py is missing"
            )
        diagnostic_keys = []
        if diagnostic_context:
            diagnostic_keys = [name for name in (
                "turn_diagnostic_row_id", "turn_diagnostic_selected",
                "turn_diagnostic_error_code",
                "el_tsbpo_trajectory_advantages", "el_tsbpo_sibling_advantages",
            ) if name in data.batch]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        select_keys.extend(diagnostic_keys)
        if multi_turn:
            select_keys.append("loss_mask")
        if "el_tsbpo_sample_weight" in data.batch:
            select_keys.append("el_tsbpo_sample_weight")
        if "el_tsbpo_policy_weight" in data.batch:
            select_keys.append("el_tsbpo_policy_weight")
        if "el_tsbpo_is_branch" in data.batch:
            select_keys.append("el_tsbpo_is_branch")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                normalization_kwargs = {}
                if is_el_tsbpo_batch or global_turn_normalization:
                    mini_tensor_batch = (
                        mini_batch.batch
                        if isinstance(mini_batch, DataProto)
                        else mini_batch
                    )
                    mini_responses = mini_tensor_batch["responses"]
                    mini_response_length = mini_responses.shape[-1]
                    mini_mask_key = (
                        "loss_mask"
                        if multi_turn and "loss_mask" in mini_tensor_batch
                        else "attention_mask"
                    )
                    mini_response_mask = mini_tensor_batch[mini_mask_key][
                        :, -mini_response_length:
                    ].to(mini_tensor_batch["advantages"].dtype)
                    if "el_tsbpo_sample_weight" in mini_tensor_batch:
                        mini_response_mask = (
                            mini_response_mask
                            * mini_tensor_batch["el_tsbpo_sample_weight"][:, None].to(
                                mini_response_mask.dtype
                            )
                        )
                    mini_token_count = mini_response_mask.sum(dim=-1)
                    mini_valid_rows = mini_token_count > 0
                    if global_turn_normalization:
                        normalization_kwargs["normalization_count"] = global_valid_turn_count(
                            mini_response_mask, device=get_torch_device().current_device(),
                        )
                    if torch.any(mini_valid_rows):
                        mini_row_advantage = (
                            (
                                mini_tensor_batch["advantages"]
                                * mini_response_mask
                            ).sum(dim=-1)
                            / mini_token_count.clamp_min(1.0)
                        )
                        mini_advantage_mean = mini_row_advantage[
                            mini_valid_rows
                        ].mean()
                        append_to_dict(metrics, {
                            "actor/el_tsbpo_optimizer_minibatch_advantage_mean": float(
                                mini_advantage_mean.item()
                            ),
                            "actor/el_tsbpo_optimizer_minibatch_advantage_mean_abs": float(
                                mini_advantage_mean.abs().item()
                            ),
                            "actor/el_tsbpo_optimizer_minibatch_advantage_abs_mean": float(
                                mini_row_advantage[mini_valid_rows].abs().mean().item()
                            ),
                        })
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                diagnostic_records = []
                diagnostic_summaries = []
                minibatch_pg_loss = minibatch_kl_loss = minibatch_kl_numerator = minibatch_kl_tokens = 0.0
                minibatch_source_pg_losses = {}
                minibatch_token_count = 0.0
                minibatch_token_metrics = {name: 0.0 for name in (
                    "actor/pg_clipfrac", "actor/ppo_kl", "actor/pg_clipfrac_lower",
                )}

                for microbatch_idx, data in enumerate(micro_batches):
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]
                    if global_turn_normalization:
                        response_mask = response_mask.to(data["advantages"].dtype)
                    if "el_tsbpo_sample_weight" in data:
                        sample_weight = data["el_tsbpo_sample_weight"].to(response_mask.dtype)
                        response_mask = response_mask * sample_weight.unsqueeze(-1)

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]
                    policy_advantages = advantages
                    policy_weight = None
                    if "el_tsbpo_policy_weight" in data:
                        policy_weight = data["el_tsbpo_policy_weight"].to(
                            advantages.dtype
                        )
                        policy_advantages = advantages * policy_weight.unsqueeze(-1)

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    policy_loss_cfg = self.config.get("policy_loss") or {}
                    policy_loss_agg_mode = policy_loss_cfg.get("loss_agg_mode")
                    if policy_loss_agg_mode is None:
                        policy_loss_agg_mode = loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    capture_this_microbatch = bool(diagnostic_context)
                    eos_statistics = None
                    if capture_this_microbatch and diagnostic_context.get("capture_eos_gradients", True):
                        entropy, log_prob, eos_statistics = self._forward_micro_batch(
                            micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy,
                            diagnostic_eos_token_ids=diagnostic_context.get("eos_token_ids", []),
                        )
                    else:
                        entropy, log_prob = self._forward_micro_batch(
                            micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy
                        )
                    
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    if loss_mode == "vanilla":
                        policy_loss_fn = compute_policy_loss
                    elif loss_mode == "gspo":
                        policy_loss_fn = compute_policy_loss_gspo
                    else:
                        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=policy_advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=policy_loss_agg_mode,
                        **normalization_kwargs,
                    )

                    source_pg_losses = {}
                    if "el_tsbpo_is_branch" in data:
                        is_branch = data["el_tsbpo_is_branch"].bool()
                        for source_name, source_rows in (
                            ("bare", ~is_branch),
                            ("branch", is_branch),
                        ):
                            source_mask = response_mask * source_rows[:, None].to(
                                response_mask.dtype
                            )
                            if float(source_mask.sum().item()) <= 0.0:
                                continue
                            source_pg_loss, _, _, _ = policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=policy_advantages,
                                response_mask=source_mask,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                                loss_agg_mode=policy_loss_agg_mode,
                                **normalization_kwargs,
                            )
                            source_pg_losses[
                                f"actor/el_tsbpo_{source_name}_pg_loss"
                            ] = source_pg_loss.detach().item()
                            source_pg_losses[
                                f"actor/el_tsbpo_{source_name}_advantage_abs_mean"
                            ] = (
                                (policy_advantages.abs() * source_mask).sum()
                                / source_mask.sum().clamp_min(1.0)
                            ).detach().item()
                            if policy_weight is not None:
                                source_pg_losses[
                                    f"actor/el_tsbpo_{source_name}_policy_weight_mean"
                                ] = policy_weight[source_rows].mean().detach().item()


                    entropy_loss = None
                    kl_loss = None
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask,
                                                loss_agg_mode=loss_agg_mode, **normalization_kwargs)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask,
                                           loss_agg_mode=loss_agg_mode, **normalization_kwargs)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        # Preserve the historical per-token KL monitor when
                        # EL-TSBPO trains KL on the same turn measure as PG.
                        if global_turn_normalization:
                            minibatch_kl_loss += kl_loss.detach().item()
                            minibatch_kl_numerator += (kld.detach() * response_mask).sum().item()
                            minibatch_kl_tokens += response_mask.sum().item()
                        elif is_el_tsbpo_batch:
                            metrics["actor/kl_loss"] = agg_loss(
                                loss_mat=kld.detach(), loss_mask=response_mask,
                                loss_agg_mode="token-mean",
                            ).item()
                            metrics["actor/kl_loss_objective"] = kl_loss.detach().item()
                        else:
                            metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if global_turn_normalization:
                        # Each microbatch already contributes its token sum /
                        # shared valid-turn count. Applying gradient_accumulation
                        # again would divide the same minibatch a second time.
                        loss_scale = 1.0
                        loss = policy_loss
                    elif self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        # Diagnostic fields must not alter an existing dict
                        # length based scale in multimodal compatibility paths.
                        scale_count = len(data)
                        if isinstance(data, dict):
                            scale_count -= sum(name in data for name in diagnostic_keys)
                        loss_scale = scale_count / self.config.ppo_mini_batch_size
                        loss = policy_loss * loss_scale
                    else:
                        loss_scale = 1.0 / self.gradient_accumulation
                        loss = policy_loss / self.gradient_accumulation
                    if capture_this_microbatch:
                        records, summary = collect_gradient_records(
                            data=data, log_prob=log_prob, response_mask=response_mask,
                            pg_loss=pg_loss, kl_loss=kl_loss, entropy=entropy,
                            entropy_loss=entropy_loss, eos_statistics=eos_statistics,
                            loss_scale=loss_scale, kl_coef=self.config.kl_loss_coef,
                            entropy_coef=entropy_coeff, epoch=epoch, minibatch=batch_idx,
                            microbatch=microbatch_idx, policy_loss_agg_mode=policy_loss_agg_mode,
                            loss_agg_mode=loss_agg_mode,
                            eos_ids=diagnostic_context.get("eos_token_ids", []),
                            include_summary=True,
                        )
                        diagnostic_records.extend(records)
                        for record in records:
                            record["optimizer_minibatch_valid_turns_per_rank"] = normalization_kwargs.get("normalization_count")
                        diagnostic_summaries.append(summary)
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    if global_turn_normalization:
                        # Each loss is a contribution to the shared minibatch,
                        # so monitoring must sum contributions, not average
                        # them a second time over microbatches.
                        minibatch_pg_loss += data.pop("actor/pg_loss")
                        micro_token_count = response_mask.sum().item()
                        minibatch_token_count += micro_token_count
                        for name in minibatch_token_metrics:
                            minibatch_token_metrics[name] += data.pop(name) * micro_token_count
                        for name in list(source_pg_losses):
                            if name.endswith("_pg_loss"):
                                minibatch_source_pg_losses[name] = (
                                    minibatch_source_pg_losses.get(name, 0.0) + source_pg_losses.pop(name)
                                )
                    data.update(source_pg_losses)
                    append_to_dict(metrics, data)

                if global_turn_normalization:
                    append_to_dict(metrics, {
                        "actor/pg_loss": minibatch_pg_loss,
                        **minibatch_source_pg_losses,
                        **{name: value / max(minibatch_token_count, 1.0)
                           for name, value in minibatch_token_metrics.items()},
                    })
                    if self.config.use_kl_loss:
                        append_to_dict(metrics, {
                            "actor/kl_loss_objective": minibatch_kl_loss,
                            "actor/kl_loss": minibatch_kl_numerator / max(minibatch_kl_tokens, 1.0),
                        })
                grad_norm = self._optimizer_step()
                if diagnostic_context:
                    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                    write_json(os.path.join(
                        diagnostic_context["output_dir"],
                        f"actor_rank{rank:03d}_epoch{epoch:03d}_minibatch{batch_idx:05d}.json",
                    ), {
                        "schema_version": 1, "global_step": diagnostic_context["global_step"],
                        "rank": rank, "ppo_epoch": epoch, "minibatch": batch_idx,
                        "temperature": float(temperature),
                        "global_turn_normalization": global_turn_normalization,
                        "optimizer_minibatch_valid_turns_per_rank": normalization_kwargs.get("normalization_count"),
                        "eos_token_ids": diagnostic_context.get("eos_token_ids", []),
                        "world_size": torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
                        "optimizer": {
                            "parameter_grad_norm_pre_clip": float(grad_norm.detach().item()),
                            "grad_clip_threshold": float(self.config.grad_clip),
                            "learning_rates": [group["lr"] for group in self.actor_optimizer.param_groups],
                            "optimizer_step_skipped_nonfinite_grad": not bool(torch.isfinite(grad_norm).item()),
                            "scope": "Whole optimizer minibatch, including normal turns and DP reduction; not a per-turn parameter gradient.",
                        },
                        "turn_gradients": diagnostic_records,
                        "all_turn_gradient_summaries": diagnostic_summaries,
                    })
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        if is_el_tsbpo_batch:
            metrics["actor/el_tsbpo_native_local_mini_batch_size"] = [
                float(self.config.ppo_mini_batch_size)
            ]
            metrics["actor/el_tsbpo_optimizer_steps_per_epoch"] = [
                float(
                    (len(batch) + self.config.ppo_mini_batch_size - 1)
                    // self.config.ppo_mini_batch_size
                )
            ]
        # Expose the effective policy aggregation so LACE runs can verify that
        # A_step/N_tokens is summed back to one turn contribution.  The actor
        # entropy/KL aggregation is controlled separately by actor.loss_agg_mode.
        metrics["actor/policy_loss_step_balanced"] = float(
            policy_loss_agg_mode == "seq-mean-token-sum"
        ) if "policy_loss_agg_mode" in locals() else 0.0
        metrics["actor/policy_loss_edge_balanced"] = float(
            policy_loss_agg_mode == "seq-mean-token-mean"
        ) if "policy_loss_agg_mode" in locals() else 0.0
        metrics["actor/policy_loss_global_turn_normalization"] = float(global_turn_normalization)
        self.actor_optimizer.zero_grad()
        return metrics
