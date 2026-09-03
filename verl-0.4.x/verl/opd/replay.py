"""Privileged teacher replay on released SofT-GRPO continuous actions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from .config import OPDConfig
from .losses import full_vocab_kl
from .prompts import render_privileged_prompt


@dataclass
class OPDReplayResult:
    """Unreduced OPD values for one actor microbatch."""

    kl_sum: torch.Tensor
    denominator_slots: int
    selected_slots: int
    teacher_entropy_sum: float
    teacher_seconds: float
    opd_support_gradient: torch.Tensor
    student_support_logits: torch.Tensor


def validate_reasoning_tokenizer(tokenizer: Any) -> tuple[int, int]:
    """Require the exact atomic delimiters used by SofT-GRPO's mode switch."""

    token_ids = []
    for text in ("<think>", "</think>"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"{text!r} must be one native tokenizer token; got {ids}")
        if tokenizer.decode(ids, skip_special_tokens=False) != text:
            raise RuntimeError(f"tokenizer does not round-trip atomic delimiter {text!r}")
        token_ids.append(int(ids[0]))
    if token_ids[0] == token_ids[1]:
        raise RuntimeError("thinking start and end delimiters share a token ID")
    return token_ids[0], token_ids[1]


def _model_embeddings(
    module: nn.Module,
    token_ids: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Look up embeddings without retaining a full FSDP parameter gathering."""

    if isinstance(module, FSDP):
        # Decoder blocks are independently auto-wrapped.  The embedding table
        # belongs to the root FSDP unit, so recursive summoning would needlessly
        # all-gather every decoder block once per replay row before the actual
        # teacher forward all-gathers them again.
        context = FSDP.summon_full_params(
            module,
            recurse=False,
            writeback=False,
            with_grads=False,
        )
    else:
        from contextlib import nullcontext

        context = nullcontext()
    with context:
        embedding = module.get_input_embeddings()
        values = embedding(token_ids.to(embedding.weight.device))
    return values.to(device=device, dtype=dtype)


def _extra_field(extra: Mapping[str, Any], name: str) -> str:
    value = extra.get(name)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"OPD example is missing nonempty extra_info[{name!r}]")
    return value


class PrivilegedReplay:
    """Score one detached actor trajectory under a privileged teacher.

    All response embeddings are reconstructed by the current actor before this
    class is called.  The same detached tensors are appended to the privileged
    prompt, so the student and teacher query exactly the same continuous path.
    """

    def __init__(self, teacher_module: nn.Module, tokenizer: Any, config: OPDConfig):
        if not config.active:
            raise ValueError("PrivilegedReplay requires an active OPD configuration")
        self.teacher_module = teacher_module
        self.tokenizer = tokenizer
        self.config = config
        self.think_start_id, self.think_end_id = validate_reasoning_tokenizer(tokenizer)

    def _prompt_ids(self, extra: Mapping[str, Any]) -> list[int]:
        content = render_privileged_prompt(
            original_user_content=_extra_field(extra, "opd_original_user_content"),
            gold_cot=_extra_field(extra, "opd_gold_cot"),
            gold_answer=extra.get("opd_gold_answer"),
            template=self.config.prompt_template,
        )
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
        )
        ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        if not ids:
            raise RuntimeError("privileged teacher prompt tokenized to an empty sequence")
        # DeepSeek-R1-Distill-Qwen's native generation template opens reasoning.
        # The detached response begins immediately after this prompt.
        if self.think_start_id not in ids:
            raise RuntimeError("privileged native chat template did not open <think>")
        return [int(token_id) for token_id in ids]

    @torch.no_grad()
    def teacher_logits(
        self,
        *,
        response_embeddings: torch.Tensor,
        response_mask: torch.Tensor,
        latent_mask: torch.Tensor,
        extra_infos: Sequence[Mapping[str, Any]],
    ) -> tuple[torch.Tensor, float]:
        """Return logits at every valid latent query, in row-major order."""

        if response_embeddings.ndim != 3:
            raise ValueError("response_embeddings must be [batch, response, hidden]")
        if response_mask.shape != response_embeddings.shape[:2]:
            raise ValueError("response mask does not match replay embeddings")
        if latent_mask.shape != response_mask.shape:
            raise ValueError("latent mask does not match response mask")
        if len(extra_infos) != response_embeddings.shape[0]:
            raise ValueError("privileged metadata is not aligned with the replay batch")

        started = time.perf_counter()
        active_logits = []
        # Actor microbatches have equal row counts on every FSDP rank.  Execute
        # one teacher forward per row even if a future trajectory gate selects
        # no slots, keeping collective call order rank-consistent.
        for row, raw_extra in enumerate(extra_infos):
            extra = dict(raw_extra)
            prompt_ids = self._prompt_ids(extra)
            prompt_tensor = torch.tensor(
                prompt_ids,
                dtype=torch.long,
                device=response_embeddings.device,
            )
            prompt_embeddings = _model_embeddings(
                self.teacher_module,
                prompt_tensor,
                device=response_embeddings.device,
                dtype=response_embeddings.dtype,
            )
            valid_response = response_embeddings[row][response_mask[row].bool()].detach()
            active = torch.nonzero(latent_mask[row], as_tuple=False).flatten()
            if active.numel() == 0:
                raise RuntimeError("every replay row must contain a continuous latent action")
            sequence = torch.cat((prompt_embeddings, valid_response), dim=0).unsqueeze(0).detach()
            query_indices = (len(prompt_ids) + active - 1).to(
                device=sequence.device,
                dtype=torch.long,
            )
            positions = torch.arange(sequence.shape[1], device=sequence.device).unsqueeze(0)
            output = self.teacher_module(
                inputs_embeds=sequence,
                attention_mask=torch.ones(sequence.shape[:2], dtype=torch.long, device=sequence.device),
                position_ids=positions,
                use_cache=False,
                logits_to_keep=query_indices,
            )
            logits = output.logits.squeeze(0)
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            if logits.shape[0] != active.numel():
                raise RuntimeError(
                    "teacher LM-head rows do not match latent queries: "
                    f"expected {active.numel()}, got {logits.shape[0]}"
                )
            active_logits.append(logits.detach())
        return torch.cat(active_logits, dim=0), time.perf_counter() - started

    def loss(
        self,
        *,
        student_logits: torch.Tensor,
        student_query_indices: torch.Tensor,
        response_embeddings: torch.Tensor,
        response_mask: torch.Tensor,
        latent_mask: torch.Tensor,
        extra_infos: Sequence[Mapping[str, Any]],
        advantages: torch.Tensor,
        latent_support_ids: torch.Tensor,
        vocab_chunk_size: int | None = 8192,
    ) -> OPDReplayResult:
        """Compute the local KL numerator with a global-denominator contract."""

        teacher_logits, teacher_seconds = self.teacher_logits(
            response_embeddings=response_embeddings,
            response_mask=response_mask,
            latent_mask=latent_mask,
            extra_infos=extra_infos,
        )
        return self.loss_from_teacher_logits(
            student_logits=student_logits,
            student_query_indices=student_query_indices,
            teacher_logits=teacher_logits,
            teacher_seconds=teacher_seconds,
            latent_mask=latent_mask,
            advantages=advantages,
            latent_support_ids=latent_support_ids,
            vocab_chunk_size=vocab_chunk_size,
        )

    def loss_from_teacher_logits(
        self,
        *,
        student_logits: torch.Tensor,
        student_query_indices: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_seconds: float,
        latent_mask: torch.Tensor,
        advantages: torch.Tensor,
        latent_support_ids: torch.Tensor,
        vocab_chunk_size: int | None = 8192,
    ) -> OPDReplayResult:
        """Finish KL scoring after a teacher forward performed before the actor."""

        selected_student = student_logits.index_select(0, student_query_indices)
        if selected_student.shape != teacher_logits.shape:
            raise RuntimeError(
                "student/teacher latent-query alignment differs: "
                f"student={tuple(selected_student.shape)}, teacher={tuple(teacher_logits.shape)}"
            )
        token_kl = full_vocab_kl(
            selected_student,
            teacher_logits,
            direction=self.config.kl_direction,
            temperature=self.config.temperature,
            vocab_chunk_size=vocab_chunk_size,
        )
        if latent_support_ids.ndim != 2 or latent_support_ids.shape[0] != token_kl.numel():
            raise ValueError("latent_support_ids must provide one support row per latent query")
        if latent_support_ids.device != selected_student.device:
            latent_support_ids = latent_support_ids.to(selected_student.device)
        student_logp = torch.log_softmax(
            selected_student.float() / self.config.temperature,
            dim=-1,
        )
        teacher_logp = torch.log_softmax(
            teacher_logits.float() / self.config.temperature,
            dim=-1,
        )
        support_student_logp = student_logp.gather(-1, latent_support_ids)
        support_teacher_logp = teacher_logp.gather(-1, latent_support_ids)
        if self.config.kl_direction.value == "teacher_to_student":
            opd_support_gradient = (
                support_student_logp.exp() - support_teacher_logp.exp()
            ) / self.config.temperature
        else:
            opd_support_gradient = support_student_logp.exp() * (
                support_student_logp - support_teacher_logp - token_kl.unsqueeze(-1)
            ) / self.config.temperature
        flat_gate = torch.ones_like(token_kl, dtype=torch.bool)
        if self.config.trajectory_gate.value == "positive_advantage":
            detached_advantages = advantages.detach()
            if detached_advantages.ndim == 1:
                if detached_advantages.shape != (latent_mask.shape[0],):
                    raise ValueError(
                        "trajectory advantages must have one value per replay row"
                    )
                row_advantages = detached_advantages
            elif detached_advantages.shape == latent_mask.shape:
                # VERL carries GRPO's sequence-level advantage at every valid
                # response position.  Reduce over the latent positions rather
                # than calling ``.item()`` on a response-length vector.  The
                # mean is also robust to future estimators whose token values
                # are not bit-identical within a trajectory.
                latent_float = latent_mask.to(
                    device=detached_advantages.device,
                    dtype=detached_advantages.dtype,
                )
                row_advantages = (
                    (detached_advantages * latent_float).sum(dim=-1)
                    / latent_float.sum(dim=-1).clamp_min(1)
                )
            else:
                raise ValueError(
                    "advantages must have shape [batch] or match latent_mask"
                )
            if not torch.isfinite(row_advantages).all():
                raise FloatingPointError("trajectory advantages must be finite")
            row_gate = (row_advantages > 0).to(device=token_kl.device)
            latent_counts = latent_mask.sum(dim=-1).to(
                device=token_kl.device,
                dtype=torch.long,
            )
            flat_gate = torch.repeat_interleave(row_gate, latent_counts)
            if flat_gate.shape != token_kl.shape:
                raise RuntimeError("trajectory gate did not align with latent queries")
        teacher_logp = torch.log_softmax(teacher_logits.float() / self.config.temperature, dim=-1)
        entropy = -(teacher_logp.exp() * teacher_logp).sum(dim=-1)
        return OPDReplayResult(
            kl_sum=token_kl.masked_fill(~flat_gate, 0.0).sum(),
            denominator_slots=int(latent_mask.sum().item()),
            selected_slots=int(flat_gate.sum().item()),
            teacher_entropy_sum=float(entropy.masked_fill(~flat_gate, 0.0).sum().item()),
            teacher_seconds=teacher_seconds,
            opd_support_gradient=opd_support_gradient.masked_fill(~flat_gate.unsqueeze(-1), 0.0).detach(),
            student_support_logits=selected_student.gather(-1, latent_support_ids).detach(),
        )
