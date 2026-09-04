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
import logging
import math
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.opd import (
    EMAUpdateState,
    ObjectiveMode,
    OPDConfig,
    PrivilegedReplay,
    TeacherType,
    categorical_suffix_mask,
    ddp_scaled_local_loss,
    latent_mask_from_topk_support,
    opd_loss_support_mask,
    parameter_squared_distance_sum_and_count,
    rms_from_squared_sum_and_count,
    teacher_gradient_isolation_violations,
    update_ema_once_,
)
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits, logprobs_from_logits_topk_dirichlet, logprobs_from_logits_topk_gumbel, logprobs_from_logits_topk_normal
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs, ulysses_pad_and_slice_inputs_3d
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _continuous_support_gradient_geometry(
    *,
    support_logits: torch.Tensor,
    stored_perturbed_logits: torch.Tensor,
    policy_log_density_sensitivity: torch.Tensor,
    opd_support_gradient: torch.Tensor | None,
    gumbel_temperature: float,
    policy_scale: float,
    opd_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return squared norms and dot product in fixed-support logit space.

    These are exact gradients of the released conditional-support Gumbel
    replay density and the OPD categorical KL, restricted to the rollout's
    stored top-five support.  The metric is cheap enough to compute every
    iteration and avoids allocating a second full-vocabulary backward graph.
    """

    if support_logits.shape != stored_perturbed_logits.shape:
        raise ValueError("support logits and stored Gumbel actions must align")
    if policy_log_density_sensitivity.shape != support_logits.shape[:-1]:
        raise ValueError("one policy sensitivity is required per latent action")
    if not math.isfinite(float(gumbel_temperature)) or float(gumbel_temperature) <= 0.0:
        raise ValueError("Gumbel temperature must be finite and positive")
    diagnostic_logits = support_logits.detach().float().requires_grad_(True)
    support_log_probs = (torch.softmax(diagnostic_logits, dim=-1) + 1e-6).log()
    reparameterized = (stored_perturbed_logits.float() - support_log_probs).clamp(-1.5, 3.0)
    support_mask = (support_log_probs > -3.0).float()
    log_density = (-reparameterized - (-reparameterized).exp())
    log_density = (log_density * support_mask).sum(-1) / support_mask.sum(-1).clamp_min(1.0)
    policy_objective = (
        log_density * policy_log_density_sensitivity.detach().float()
    ).sum()
    policy_gradient = torch.autograd.grad(policy_objective, diagnostic_logits)[0] * float(policy_scale)
    if opd_support_gradient is None:
        opd_gradient = torch.zeros_like(policy_gradient)
    else:
        if opd_support_gradient.shape != policy_gradient.shape:
            raise ValueError("OPD and policy support gradients must align")
        opd_gradient = opd_support_gradient.detach().float() * float(opd_scale)
    return (
        policy_gradient.square().sum(dtype=torch.float64),
        opd_gradient.square().sum(dtype=torch.float64),
        (policy_gradient * opd_gradient).sum(dtype=torch.float64),
    )


def _all_gather_variable_1d(value: torch.Tensor) -> torch.Tensor:
    """Gather a modest one-dimensional diagnostic vector across DP ranks."""

    if value.ndim != 1:
        raise ValueError("distributed diagnostic values must be rank one")
    value = value.contiguous()
    if not torch.distributed.is_initialized():
        return value
    world_size = torch.distributed.get_world_size()
    local_size = torch.tensor([value.numel()], dtype=torch.int64, device=value.device)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(sizes, local_size)
    maximum = max(int(size.item()) for size in sizes)
    padded = torch.zeros(maximum, dtype=value.dtype, device=value.device)
    padded[: value.numel()] = value
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, padded)
    return torch.cat(
        [rank_value[: int(size.item())] for rank_value, size in zip(gathered, sizes)]
    )


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        *,
        opd_config=None,
        opd_teacher_module: nn.Module | None = None,
        tokenizer=None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.opd_config = OPDConfig.from_mapping(opd_config or {"enabled": False})
        self.opd_teacher_module = opd_teacher_module
        self.opd_replay = None
        self.ema_state = EMAUpdateState()
        if self.opd_config.active:
            if tokenizer is None:
                raise RuntimeError("active OPD requires the actor tokenizer")
            if self.opd_config.teacher.type is TeacherType.CURRENT_ACTOR:
                if opd_teacher_module is not None:
                    raise RuntimeError("current_actor OPD must not construct a separate teacher")
                teacher_module = actor_module
            else:
                if opd_teacher_module is None:
                    raise RuntimeError("configured OPD teacher module was not constructed")
                teacher_module = opd_teacher_module
            self.opd_replay = PrivilegedReplay(teacher_module, tokenizer, self.opd_config)

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, add_noise_dirichlet=False,
                             add_noise_gumbel_softmax=True, compute_opd=False,
                             collect_gradient_info=False, continuous_replay=True):
        def safe_lookup_embeddings(fsdp_wrapped_module, input_ids, target_device=None, target_dtype=None):
            """FSDP兼容安全查表：无FSDP时等价于普通调用。"""
            embed = fsdp_wrapped_module.get_input_embeddings()

            # Only the root FSDP unit owns the embedding table.  Recursively
            # summoning the independently wrapped decoder layers here would
            # gather the complete model before every replay forward.
            ctx = FSDP.summon_full_params(
                fsdp_wrapped_module,
                recurse=False,
                writeback=False,
                with_grads=False,
            )
            with ctx:
                w = embed.weight
                _input_ids = input_ids.to(w.device)
                embs = embed(_input_ids)
            # 输出转回目标device和dtype，和你的缓冲区保持一致
            if target_dtype is not None and embs.dtype != target_dtype:
                embs = embs.to(dtype=target_dtype)
            if target_device is not None and embs.device != target_device:
                embs = embs.to(target_device)
            return embs

        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        if compute_opd and self.opd_replay is None:
            raise RuntimeError("OPD replay requested while the objective is inactive")
        if compute_opd and not continuous_replay:
            raise RuntimeError("OPD cannot run on categorical hard-token replay")
        if continuous_replay and not self.use_remove_padding:
            raise RuntimeError("continuous replay diagnostics require remove-padding replay")
        if compute_opd and self.use_ulysses_sp:
            raise RuntimeError("OPD currently requires Ulysses sequence parallel size 1")
        if compute_opd and float(temperature) != 1.0:
            raise RuntimeError("OPD production study requires released LM temperature 1.0")
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]],
                                                    dim=0)
        # print(f"use_remove_padding{self.use_remove_padding}") #True
        # print(f"use_ulysses_sp{self.use_ulysses_sp}") #False
        # print(f"use_fused_kernels{self.use_fused_kernels}") #False

        # print(f'type gumbels{micro_batch["rollout_topk_gumbels"].dtype}')
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            rollout_topk_ids = micro_batch.get("rollout_topk_ids") if continuous_replay else None
            rollout_topk_gumbels = micro_batch.get("rollout_topk_gumbels") if continuous_replay else None
            if continuous_replay and (rollout_topk_ids is None or rollout_topk_gumbels is None):
                raise RuntimeError("continuous replay requires stored top-k IDs and perturbations")
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            gumbel_temperature = (
                micro_batch["gumbel_temperature"][0].item() if continuous_replay else None
            )
            entropy = None
            opd_result = None
            gradient_info = None
            teacher_logits = None
            teacher_seconds = 0.0
            student_query_indices = None
            latent_mask = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            # print(input_ids.size(), rollout_topk_ids.size(), rollout_topk_gumbels.size())
            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
                if continuous_replay:
                    topk_ids_rmpad, _, *_ = unpad_input(rollout_topk_ids,
                                                        attention_mask)  # input_ids_rmpad (total_nnz, ...)
                    topk_gumbels_rmpad, _, *_ = unpad_input(rollout_topk_gumbels,
                                                            attention_mask)  # input_ids_rmpad (total_nnz, ...)

                # print(input_ids_rmpad.size(), topk_ids_rmpad.size(), topk_gumbels_rmpad.size())
                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                                          indices).transpose(0, 1).unsqueeze(
                        1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                # for compute the log_prob

                if continuous_replay:
                    topk_embs = safe_lookup_embeddings(
                        self.actor_module,
                        topk_ids_rmpad,
                        target_device=topk_gumbels_rmpad.device,
                        target_dtype=topk_gumbels_rmpad.dtype
                    )
                # print(f"topk_gumbels_rmpad{topk_gumbels_rmpad.tolist()}")
                # print(f"topk_ids_rmpad{topk_ids_rmpad.tolist()}")
                # print(f"position_ids_rmpad{position_ids_rmpad.tolist()}")

                    mask = (topk_ids_rmpad[:, 1:] == 0).all(dim=-1, keepdim=True)  # bool [B,1]
                    masked = topk_gumbels_rmpad.clone()
                    masked[:, 1:] = masked[:, 1:].masked_fill(mask, -torch.inf)
                    gumbel_y = torch.softmax(masked / gumbel_temperature, dim=-1).to(topk_gumbels_rmpad.dtype)
                    # print(gumbel_y)
                    topk_embs = torch.sum(gumbel_y.unsqueeze(-1) * topk_embs, dim=1, dtype=torch.bfloat16)
                if continuous_replay and (compute_opd or collect_gradient_info):
                    response_mask = attention_mask[:, -response_length:].bool()
                    response_support = rollout_topk_ids[:, -response_length:]
                    latent_mask = latent_mask_from_topk_support(response_mask, response_support)
                    if not bool(latent_mask.any(dim=1).all().item()):
                        raise RuntimeError(
                            "continuous replay fallback detected: every response must contain latent actions"
                        )
                    objective_mask = (
                        opd_loss_support_mask(
                            response_mask,
                            response_support,
                            loss_support=self.opd_config.loss_support,
                            responses=micro_batch["responses"],
                            close_tag_token_id=self.opd_replay.think_end_id,
                        )
                        if compute_opd
                        else latent_mask
                    )
                    answer_mask = (
                        categorical_suffix_mask(
                            response_mask,
                            response_support,
                            micro_batch["responses"],
                            self.opd_replay.think_end_id,
                        )
                        if compute_opd
                        else torch.zeros_like(latent_mask)
                    )
                    query_indices = []
                    flat_offset = 0
                    prompt_width = seqlen - response_length
                    for row in range(batch_size):
                        valid_prompt = int(attention_mask[row, :prompt_width].sum().item())
                        active = torch.nonzero(objective_mask[row], as_tuple=False).flatten()
                        query_indices.extend((flat_offset + valid_prompt + active - 1).tolist())
                        flat_offset += int(attention_mask[row].sum().item())
                    student_query_indices = torch.tensor(
                        query_indices,
                        dtype=torch.long,
                        device=topk_embs.device,
                    )

                if compute_opd:
                    if "extra_info" not in micro_batch:
                        raise RuntimeError("OPD replay is missing aligned privileged extra_info")
                    replay_dense = pad_input(
                        hidden_states=topk_embs.unsqueeze(-2),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    ).squeeze(-2)
                    # FSDP modules must execute collectives in identical order on
                    # every rank. Actor microbatches are fixed-size, so score one
                    # privileged row per rank before building the actor graph.
                    teacher_logits, teacher_seconds = self.opd_replay.teacher_logits(
                        response_embeddings=replay_dense[:, -response_length:].detach(),
                        response_mask=response_mask,
                        latent_mask=objective_mask,
                        extra_infos=micro_batch["extra_info"],
                    )
                # topk_embs = torch.bmm(gumbel_y.unsqueeze(1), topk_embs).squeeze(1)
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                if continuous_replay:
                    topk_gumbels_rmpad_rolled = torch.roll(topk_gumbels_rmpad, shifts=-1, dims=0)  # (total_nnz, k)
                    topk_ids_rmpad_rolled = torch.roll(topk_ids_rmpad, shifts=-1, dims=0)  # (total_nnz, k)
                # print(topk_gumbels_rmpad_rolled.tolist())
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
                    if continuous_replay:
                        topk_ids_rmpad, _, _ = ulysses_pad_and_slice_inputs_3d(
                            rollout_topk_ids,
                            position_ids_rmpad=None,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                        topk_gumbels_rmpad, _, _ = ulysses_pad_and_slice_inputs_3d(
                            rollout_topk_gumbels,
                            position_ids_rmpad=None,
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
                # print(input_ids_rmpad.size(), topk_embeds_rmpad.size(), topk_gumbels_rmpad_rolled.size())
                actor_inputs = (
                    {"inputs_embeds": topk_embs.unsqueeze(0).detach()}
                    if continuous_replay
                    else {"input_ids": input_ids_rmpad}
                )
                output = self.actor_module(
                    **actor_inputs,
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
                    if compute_opd:
                        opd_result = self.opd_replay.loss_from_teacher_logits(
                            student_logits=logits_rmpad,
                            student_query_indices=student_query_indices,
                            teacher_logits=teacher_logits,
                            teacher_seconds=teacher_seconds,
                            latent_mask=latent_mask,
                            objective_mask=objective_mask,
                            answer_mask=answer_mask,
                            advantages=micro_batch.get("advantages"),
                            latent_support_ids=response_support[latent_mask],
                            vocab_chunk_size=int(self.config.get("opd_vocab_chunk_size", 8192)),
                        )
                    if collect_gradient_info and continuous_replay:
                        latent_query_indices = []
                        flat_offset = 0
                        prompt_width = seqlen - response_length
                        for row in range(batch_size):
                            valid_prompt = int(attention_mask[row, :prompt_width].sum().item())
                            active = torch.nonzero(latent_mask[row], as_tuple=False).flatten()
                            latent_query_indices.extend((flat_offset + valid_prompt + active - 1).tolist())
                            flat_offset += int(attention_mask[row].sum().item())
                        latent_query_indices = torch.tensor(
                            latent_query_indices, dtype=torch.long, device=logits_rmpad.device
                        )
                        selected_logits = logits_rmpad.index_select(0, latent_query_indices)
                        latent_support_ids = response_support[latent_mask]
                        gradient_info = {
                            "latent_mask": latent_mask,
                            "support_logits": selected_logits.gather(-1, latent_support_ids).detach(),
                            "support_gumbels": micro_batch["rollout_topk_gumbels"][:, -response_length:][latent_mask].detach(),
                        }
                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    # log_probs = logprobs_from_logits(
                    #     logits=logits_rmpad,
                    #     labels=input_ids_rmpad_rolled,
                    #     inplace_backward=inplace_backward,
                    # )
                    if not continuous_replay:
                        log_probs = logprobs_from_logits(
                            logits=logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=inplace_backward,
                        )
                    elif add_noise_gumbel_softmax:
                        log_probs = logprobs_from_logits_topk_gumbel(
                            logits=logits_rmpad,
                            rollout_topk_ids=topk_ids_rmpad_rolled,
                            rollout_topk_gumbels=topk_gumbels_rmpad_rolled,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=inplace_backward,
                        )
                    elif add_noise_dirichlet:
                        log_probs = logprobs_from_logits_topk_dirichlet(
                            logits=logits_rmpad,
                            rollout_topk_ids=topk_ids_rmpad_rolled,
                            rollout_topk_gumbels=topk_gumbels_rmpad_rolled,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=inplace_backward,
                        )
                    else:
                        log_probs = logprobs_from_logits_topk_normal(
                            logits=logits_rmpad,
                            rollout_topk_ids=topk_ids_rmpad_rolled,
                            rollout_topk_gumbels=topk_gumbels_rmpad_rolled,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=inplace_backward,
                        )

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

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1: -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1: -1]  # (bsz, response_length)
                # print(log_probs)
            else:  # not using rmpad and no ulysses sp
                if compute_opd:
                    raise RuntimeError("OPD forbids categorical padded replay fallback")
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
                    log_probs = output.log_probs[:, -response_length - 1: -1]
                    entropy = output.entropy[:, -response_length - 1: -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1: -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs, opd_result, gradient_info

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        # print(self.actor_module.parameters())
        # print(grad_norm)
        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            self.actor_optimizer.zero_grad()
            raise FloatingPointError(
                f"rank {torch.distributed.get_rank()} has non-finite actor gradient norm {grad_norm}"
            )
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

        add_noise_dirichlet = data.meta_info['add_noise_dirichlet']
        add_noise_gumbel_softmax = data.meta_info['add_noise_gumbel_softmax']
        continuous_replay = bool(data.meta_info.get("continuous_replay", True))

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if continuous_replay:
            select_keys.extend(
                ["rollout_topk_ids", "rollout_topk_gumbels", "gumbel_temperature"]
            )
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

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
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, _, _ = self._forward_micro_batch(micro_batch, temperature=temperature,
                                                               calculate_entropy=calculate_entropy,
                                                               add_noise_dirichlet=add_noise_dirichlet,
                                                               add_noise_gumbel_softmax=add_noise_gumbel_softmax,
                                                               continuous_replay=continuous_replay)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

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

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        multi_turn = data.meta_info.get("multi_turn", False)
        add_noise_dirichlet = data.meta_info["add_noise_dirichlet"]
        add_noise_gumbel_softmax = data.meta_info["add_noise_gumbel_softmax"]
        continuous_replay = bool(data.meta_info.get("continuous_replay", True))
        opd_active = self.opd_config.active
        standalone = opd_active and self.opd_config.mode is ObjectiveMode.STANDALONE
        opd_beta = float(data.meta_info.get("opd_beta_effective", 0.0))
        rollout_iteration = int(data.meta_info.get("opd_rollout_iteration", -1))
        if opd_active:
            if not continuous_replay:
                raise RuntimeError("active OPD requires native continuous replay")
            if rollout_iteration < 0:
                raise RuntimeError("active OPD update is missing its zero-based rollout iteration")
            if not math.isfinite(opd_beta) or opd_beta < 0.0:
                raise RuntimeError(f"invalid effective OPD coefficient {opd_beta}")
            if int(self.config.ppo_epochs) != 1:
                raise RuntimeError("OPD production study requires exactly one PPO epoch")
            if self.config.use_dynamic_bsz:
                raise RuntimeError("OPD requires fixed microbatches for global slot weighting")
        if standalone and (opd_beta <= 0.0 or self.config.use_kl_loss):
            raise RuntimeError("standalone OPD requires a positive dose and forbids native reference KL")
        compute_opd = opd_active and opd_beta > 0.0

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if not standalone:
            select_keys.extend(["old_log_probs", "advantages"])
        if continuous_replay:
            select_keys.extend(
                ["rollout_topk_ids", "rollout_topk_gumbels", "gumbel_temperature"]
            )
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss and not standalone:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch
        if compute_opd and has_multi_modal_inputs:
            raise RuntimeError("OPD MATH production study does not support multimodal replay")
        if compute_opd:
            if "extra_info" not in data.non_tensor_batch:
                raise RuntimeError("OPD update is missing privileged example metadata")
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            dataloader = data.select(select_keys, ["extra_info"]).chunk(num_mini_batches)
        elif has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            dataloader = data.select(select_keys, ["multi_modal_inputs"]).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        optimizer_steps = 0
        local_pg_sum = 0.0
        local_native_kl_sum = 0.0
        local_base_total_sum = 0.0
        local_policy_tokens = 0
        local_opd_kl_sum = 0.0
        local_opd_latent_kl_sum = 0.0
        local_opd_answer_kl_sum = 0.0
        local_opd_denominator = 0
        local_opd_latent_slots = 0
        local_opd_answer_slots = 0
        local_opd_selected = 0
        local_teacher_entropy_sum = 0.0
        teacher_seconds = 0.0
        support_grpo_grad_sq = None
        support_opd_grad_sq = None
        support_grad_dot = None
        ratio_observations = []
        negative_log_ratio_observations = []
        clip_observations = []
        grad_norm_sum = 0.0
        grad_clip_count = 0.0

        for _epoch in range(self.config.ppo_epochs):
            for mini_batch in dataloader:
                if compute_opd:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size
                        // self.config.ppo_micro_batch_size_per_gpu
                    )
                    num_micro_batches = (
                        mini_batch.batch.batch_size[0]
                        // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.chunk(num_micro_batches)
                elif has_multi_modal_inputs:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size
                        // self.config.ppo_micro_batch_size_per_gpu
                    )
                    num_micro_batches = (
                        mini_batch.batch.batch_size[0]
                        // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = (
                        self.config.ppo_max_token_len_per_gpu
                        * self.ulysses_sequence_parallel_size
                    )
                    micro_batches, _ = rearrange_micro_batches(
                        batch=mini_batch, max_token_len=max_token_len
                    )
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size
                        // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(
                        self.config.ppo_micro_batch_size_per_gpu
                    )

                self.actor_optimizer.zero_grad()
                global_opd_denominator = None
                if compute_opd:
                    mini_tensors = mini_batch.batch
                    mini_response_length = mini_tensors["responses"].size(-1)
                    mini_response_mask = mini_tensors["attention_mask"][
                        :, -mini_response_length:
                    ].bool()
                    mini_support = mini_tensors["rollout_topk_ids"][
                        :, -mini_response_length:
                    ]
                    local_slots = opd_loss_support_mask(
                        mini_response_mask,
                        mini_support,
                        loss_support=self.opd_config.loss_support,
                        responses=mini_tensors["responses"],
                        close_tag_token_id=self.opd_replay.think_end_id,
                    ).sum()
                    global_opd_denominator = local_slots.to(dtype=torch.int64)
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(global_opd_denominator)
                    if int(global_opd_denominator.item()) <= 0:
                        raise RuntimeError("distributed OPD optimizer batch has no selected support slots")

                for micro_batch in micro_batches:
                    if isinstance(micro_batch, DataProto):
                        micro_data = {
                            **micro_batch.batch.to(get_torch_device().current_device()),
                            **micro_batch.non_tensor_batch,
                        }
                    else:
                        micro_data = micro_batch.to(get_torch_device().current_device())
                    responses = micro_data["responses"]
                    response_length = responses.size(1)
                    attention_mask = micro_data["attention_mask"]
                    response_mask = (
                        micro_data["loss_mask"][:, -response_length:]
                        if multi_turn
                        else attention_mask[:, -response_length:]
                    )
                    response_mask = response_mask.bool()
                    entropy_coeff = float(self.config.entropy_coeff)
                    loss_agg_mode = self.config.loss_agg_mode
                    collect_gradient_info = continuous_replay and (
                        compute_opd or not standalone
                    )
                    entropy, log_prob, opd_result, gradient_info = self._forward_micro_batch(
                        micro_batch=micro_data,
                        temperature=temperature,
                        calculate_entropy=(not standalone and entropy_coeff != 0.0),
                        add_noise_dirichlet=add_noise_dirichlet,
                        add_noise_gumbel_softmax=add_noise_gumbel_softmax,
                        compute_opd=compute_opd,
                        collect_gradient_info=collect_gradient_info,
                        continuous_replay=continuous_replay,
                    )

                    loss = None
                    pg_loss = None
                    if not standalone:
                        old_log_prob = micro_data["old_log_probs"]
                        advantages = micro_data["advantages"]
                        clip_ratio = self.config.clip_ratio
                        clip_ratio_low = (
                            self.config.clip_ratio_low
                            if self.config.clip_ratio_low is not None
                            else clip_ratio
                        )
                        clip_ratio_high = (
                            self.config.clip_ratio_high
                            if self.config.clip_ratio_high is not None
                            else clip_ratio
                        )
                        clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                        pg_loss, _pg_clipfrac, _ppo_kl, _pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )
                        active_ratio = torch.exp(log_prob - old_log_prob)[response_mask]
                        if active_ratio.numel() == 0 or not torch.isfinite(active_ratio).all():
                            raise FloatingPointError(
                                "PPO replay produced an empty or non-finite density ratio"
                            )
                        active_advantages = advantages[response_mask]
                        active_unclipped = -active_advantages * active_ratio
                        active_clipped = -active_advantages * torch.clamp(
                            active_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high
                        )
                        active_dual = torch.minimum(
                            -active_advantages * clip_ratio_c,
                            torch.maximum(active_unclipped, active_clipped),
                        )
                        active_pg = torch.where(
                            active_advantages < 0,
                            active_dual,
                            torch.maximum(active_unclipped, active_clipped),
                        )
                        local_pg_sum += float(active_pg.detach().sum().item())
                        local_policy_tokens += int(active_pg.numel())
                        local_base_total_sum += float(active_pg.detach().sum().item())
                        ratio_observations.append(active_ratio.detach().float())
                        negative_log_ratio_observations.append(
                            (old_log_prob - log_prob)[response_mask].detach().float()
                        )
                        clip_observations.append(
                            active_clipped.gt(active_unclipped).detach().float()
                        )

                        policy_loss = pg_loss
                        if entropy_coeff != 0.0:
                            entropy_loss = agg_loss(
                                loss_mat=entropy,
                                loss_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                            )
                            policy_loss = policy_loss - entropy_loss * entropy_coeff
                            local_base_total_sum -= float(
                                entropy[response_mask].detach().sum().item()
                            ) * entropy_coeff
                        if self.config.use_kl_loss:
                            kld = kl_penalty(
                                logprob=log_prob,
                                ref_logprob=micro_data["ref_log_prob"],
                                kl_penalty=self.config.kl_loss_type,
                            )
                            kl_loss = agg_loss(
                                loss_mat=kld,
                                loss_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                            )
                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            local_native_kl_sum += float(kld[response_mask].detach().sum().item())
                            local_base_total_sum += float(
                                kld[response_mask].detach().sum().item()
                            ) * float(self.config.kl_loss_coef)
                            metrics["actor/kl_coef"] = self.config.kl_loss_coef
                        if self.config.use_dynamic_bsz:
                            loss = policy_loss * (
                                len(micro_data) / self.config.ppo_mini_batch_size
                            )
                        else:
                            loss = policy_loss / self.gradient_accumulation

                    if compute_opd:
                        if opd_result is None:
                            raise RuntimeError("active nonzero OPD update returned no KL result")
                        world_size = (
                            torch.distributed.get_world_size()
                            if torch.distributed.is_initialized()
                            else 1
                        )
                        opd_loss = ddp_scaled_local_loss(
                            opd_result.kl_sum, global_opd_denominator, world_size
                        )
                        weighted_local_opd = opd_beta * opd_loss
                        loss = weighted_local_opd if loss is None else loss + weighted_local_opd
                        local_opd_kl_sum += float(opd_result.kl_sum.detach().item())
                        local_opd_latent_kl_sum += float(
                            opd_result.latent_kl_sum.detach().item()
                        )
                        local_opd_answer_kl_sum += float(
                            opd_result.answer_kl_sum.detach().item()
                        )
                        local_opd_denominator += int(opd_result.denominator_slots)
                        local_opd_latent_slots += int(opd_result.latent_slots)
                        local_opd_answer_slots += int(opd_result.answer_slots)
                        local_opd_selected += int(opd_result.selected_slots)
                        local_teacher_entropy_sum += opd_result.teacher_entropy_sum
                        teacher_seconds += opd_result.teacher_seconds
                    if loss is None:
                        raise RuntimeError("actor update constructed no optimization objective")

                    if collect_gradient_info:
                        if gradient_info is None:
                            raise RuntimeError(
                                "continuous support gradient inventory was not returned"
                            )
                        if standalone:
                            latent_policy_sensitivity = torch.zeros(
                                gradient_info["latent_mask"].sum(),
                                device=log_prob.device,
                                dtype=log_prob.dtype,
                            )
                        else:
                            policy_sensitivity = torch.autograd.grad(
                                pg_loss,
                                log_prob,
                                retain_graph=True,
                                create_graph=False,
                            )[0].detach()
                            latent_policy_sensitivity = policy_sensitivity[
                                gradient_info["latent_mask"]
                            ]
                        world_size = (
                            torch.distributed.get_world_size()
                            if torch.distributed.is_initialized()
                            else 1
                        )
                        opd_gradient = (
                            opd_result.opd_support_gradient
                            if opd_result is not None
                            else None
                        )
                        opd_gradient_scale = (
                            opd_beta * world_size / float(global_opd_denominator.item())
                            if compute_opd
                            else 0.0
                        )
                        micro_grpo_sq, micro_opd_sq, micro_dot = (
                            _continuous_support_gradient_geometry(
                                support_logits=gradient_info["support_logits"],
                                stored_perturbed_logits=gradient_info["support_gumbels"],
                                policy_log_density_sensitivity=latent_policy_sensitivity,
                                opd_support_gradient=opd_gradient,
                                gumbel_temperature=float(
                                    micro_data["gumbel_temperature"][0].item()
                                ),
                                policy_scale=(
                                    0.0 if standalone else 1.0 / self.gradient_accumulation
                                ),
                                opd_scale=opd_gradient_scale,
                            )
                        )
                        if support_grpo_grad_sq is None:
                            support_grpo_grad_sq = micro_grpo_sq
                            support_opd_grad_sq = micro_opd_sq
                            support_grad_dot = micro_dot
                        else:
                            support_grpo_grad_sq += micro_grpo_sq
                            support_opd_grad_sq += micro_opd_sq
                            support_grad_dot += micro_dot

                    loss.backward()

                grad_norm = self._optimizer_step()
                optimizer_steps += 1
                grad_norm_value = float(grad_norm.detach().item())
                grad_norm_sum += grad_norm_value
                grad_clip_count += float(grad_norm_value > float(self.config.grad_clip))

        if optimizer_steps <= 0:
            raise RuntimeError("actor update completed without an optimizer step")

        teacher_student_rms = 0.0
        if opd_active and self.opd_config.teacher.type is not TeacherType.CURRENT_ACTOR:
            isolation_counts = torch.tensor(
                teacher_gradient_isolation_violations(self.opd_teacher_module),
                dtype=torch.int64,
                device=get_torch_device().current_device(),
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    isolation_counts, op=torch.distributed.ReduceOp.MAX
                )
            if bool((isolation_counts != 0).any().item()):
                requires_grad_count, accumulated_grad_count = (
                    int(value) for value in isolation_counts.tolist()
                )
                raise RuntimeError(
                    "privileged OPD teacher received gradients during the student "
                    f"update: requires_grad_parameters={requires_grad_count}, "
                    f"accumulated_gradients={accumulated_grad_count}"
                )
            squared_sum, parameter_count = parameter_squared_distance_sum_and_count(
                self.opd_teacher_module, self.actor_module
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(squared_sum)
                torch.distributed.all_reduce(parameter_count)
            teacher_student_rms = float(
                rms_from_squared_sum_and_count(squared_sum, parameter_count).item()
            )

        # The EMA snapshot is fixed for both optimizer steps and advances only
        # after every step in the outer rollout iteration has succeeded.
        ema_updates_this_iteration = 0
        if opd_active and self.opd_config.teacher.type is TeacherType.EMA:
            update_ema_once_(
                teacher=self.opd_teacher_module,
                student=self.actor_module,
                decay=self.opd_config.teacher.ema_decay,
                rollout_iteration=rollout_iteration,
                state=self.ema_state,
            )
            ema_updates_this_iteration = 1

        device = get_torch_device().current_device()
        global_values = torch.tensor(
            [
                local_pg_sum,
                local_native_kl_sum,
                local_base_total_sum,
                float(local_policy_tokens),
                local_opd_kl_sum,
                local_opd_latent_kl_sum,
                local_opd_answer_kl_sum,
                float(local_opd_denominator),
                float(local_opd_latent_slots),
                float(local_opd_answer_slots),
                float(local_opd_selected),
                local_teacher_entropy_sum,
                grad_norm_sum,
                grad_clip_count,
                float(optimizer_steps),
            ],
            dtype=torch.float64,
            device=device,
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(global_values)
        (
            global_pg_sum,
            global_native_kl_sum,
            global_base_total_sum,
            global_policy_tokens,
            global_opd_kl_sum,
            global_opd_latent_kl_sum,
            global_opd_answer_kl_sum,
            global_opd_slots,
            global_latent_slots,
            global_answer_slots,
            global_selected_slots,
            global_entropy_sum,
            global_grad_norm_sum,
            global_grad_clip_count,
            global_optimizer_step_observations,
        ) = global_values
        opd_kl_mean = float(
            (global_opd_kl_sum / global_opd_slots.clamp_min(1.0)).item()
        )
        opd_latent_mean = float(
            (global_opd_latent_kl_sum / global_latent_slots.clamp_min(1.0)).item()
        )
        opd_answer_mean = float(
            (global_opd_answer_kl_sum / global_answer_slots.clamp_min(1.0)).item()
        )
        teacher_entropy_mean = float(
            (global_entropy_sum / global_selected_slots.clamp_min(1.0)).item()
        )
        weighted_opd = opd_beta * opd_kl_mean
        base_loss_mean = (
            0.0
            if standalone
            else float((global_base_total_sum / global_policy_tokens.clamp_min(1.0)).item())
        )
        metrics.update(
            {
                "loss/opd_kl_latent": opd_latent_mean,
                "loss/opd_kl_answer": opd_answer_mean,
                "loss/opd_kl_unweighted": opd_kl_mean,
                "loss/opd_weighted": weighted_opd,
                "loss/total": base_loss_mean + weighted_opd,
                "opd/teacher_student_param_rms": teacher_student_rms,
                "opd/ema_update_count": float(self.ema_state.update_count),
                "opd/ema_updates_this_iteration": float(ema_updates_this_iteration),
                "opd/active_slots": float(global_opd_slots.item()),
                "opd/selected_slots": float(global_selected_slots.item()),
                "opd/latent_slot_count": float(global_latent_slots.item()),
                "opd/answer_slot_count": float(global_answer_slots.item()),
                "opd/selected_slot_fraction": float(
                    (global_selected_slots / global_opd_slots.clamp_min(1.0)).item()
                ),
                "opd/teacher_entropy": teacher_entropy_mean,
                "perf/teacher_seconds": teacher_seconds,
                "trainer/optimizer_steps_this_iteration": float(optimizer_steps),
                "actor/opd_kl_latent": opd_latent_mean,
                "actor/opd_kl_answer": opd_answer_mean,
                "actor/opd_kl_unweighted": opd_kl_mean,
                "actor/opd_weighted": weighted_opd,
                "actor/total_loss": base_loss_mean + weighted_opd,
                "actor/opd_teacher_student_param_rms": teacher_student_rms,
                "actor/opd_ema_update_count": float(self.ema_state.update_count),
                "actor/opd_teacher_seconds": teacher_seconds,
                "actor/grad_norm": float(
                    (global_grad_norm_sum / global_optimizer_step_observations).item()
                ),
                "actor/gradient_clipfrac": float(
                    (global_grad_clip_count / global_optimizer_step_observations).item()
                ),
            }
        )

        grpo_support_norm = 0.0
        opd_support_norm = 0.0
        grpo_opd_cosine = 0.0
        if continuous_replay:
            if support_grpo_grad_sq is None:
                raise RuntimeError("continuous support gradient diagnostics were not accumulated")
            for value in (support_grpo_grad_sq, support_opd_grad_sq, support_grad_dot):
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(value)
            gradient_world_size = (
                torch.distributed.get_world_size()
                if torch.distributed.is_initialized()
                else 1
            )
            grpo_support_norm = (
                math.sqrt(max(float(support_grpo_grad_sq.item()), 0.0))
                / gradient_world_size
            )
            opd_support_norm = (
                math.sqrt(max(float(support_opd_grad_sq.item()), 0.0))
                / gradient_world_size
            )
            if grpo_support_norm > 0.0 and opd_support_norm > 0.0:
                grpo_opd_cosine = float(support_grad_dot.item()) / (
                    math.sqrt(float(support_grpo_grad_sq.item()))
                    * math.sqrt(float(support_opd_grad_sq.item()))
                )
                grpo_opd_cosine = max(-1.0, min(1.0, grpo_opd_cosine))
        metrics.update(
            {
                "actor/opd_grad_norm": opd_support_norm,
                "grad/opd_norm": opd_support_norm,
                "grad/fixed_support_opd_norm": opd_support_norm,
            }
        )

        if not standalone:
            global_ratios = _all_gather_variable_1d(torch.cat(ratio_observations))
            global_negative_log_ratios = _all_gather_variable_1d(
                torch.cat(negative_log_ratio_observations)
            )
            global_clip_indicators = _all_gather_variable_1d(
                torch.cat(clip_observations)
            )
            exact_ratio_mean = float(global_ratios.mean().item())
            exact_ratio_p95 = float(torch.quantile(global_ratios, 0.95).item())
            exact_approx_kl = float(global_negative_log_ratios.mean().item())
            exact_clip_fraction = float(global_clip_indicators.mean().item())
            exact_pg_mean = float(
                (global_pg_sum / global_policy_tokens.clamp_min(1.0)).item()
            )
            exact_native_kl_mean = float(
                (global_native_kl_sum / global_policy_tokens.clamp_min(1.0)).item()
            )
            metrics.update(
                {
                    "actor/pg_loss": exact_pg_mean,
                    "actor/kl_loss": exact_native_kl_mean,
                    "actor/ratio_mean": exact_ratio_mean,
                    "actor/ratio_p95": exact_ratio_p95,
                    "actor/pg_clipfrac": exact_clip_fraction,
                    "actor/ppo_kl": exact_approx_kl,
                    "actor/grpo_grad_norm": grpo_support_norm,
                    "actor/grpo_opd_grad_cosine": grpo_opd_cosine,
                    "loss/grpo": exact_pg_mean,
                    "loss/native_ref_kl": exact_native_kl_mean,
                    "grad/grpo_norm": grpo_support_norm,
                    "grad/grpo_opd_cosine": grpo_opd_cosine,
                    "grad/fixed_support_grpo_norm": grpo_support_norm,
                    "grad/fixed_support_grpo_opd_cosine": grpo_opd_cosine,
                }
            )

        self.actor_optimizer.zero_grad()
        return metrics
