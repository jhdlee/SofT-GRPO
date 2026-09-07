"""Numerically stable, memory-bounded full-vocabulary OPD losses."""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
from torch.autograd.function import once_differentiable

from .config import KLDirection


def _logit_chunk(logits, start, stop, row_indices=None):
    # Slice vocabulary first: selecting rows before this slice would allocate
    # positions x vocabulary logits and retain that copy through backward.
    chunk = logits[..., start:stop]
    return chunk if row_indices is None else chunk.index_select(0, row_indices)


def _chunked_logsumexp(logits: torch.Tensor, temperature: float, chunk_size: int, row_indices=None) -> torch.Tensor:
    normalizer = None
    for start in range(0, logits.shape[-1], chunk_size):
        chunk = _logit_chunk(logits, start, start + chunk_size, row_indices)
        chunk_normalizer = torch.logsumexp(chunk.float() / temperature, dim=-1)
        normalizer = chunk_normalizer if normalizer is None else torch.logaddexp(normalizer, chunk_normalizer)
    return normalizer


class _FullVocabularyKL(torch.autograd.Function):
    """Keep input logits and O(positions) statistics; recompute chunk gradients.

    Ordinary autograd through a vocabulary-chunked forward still saves FP32
    intermediates for *every* chunk until backward.  This first-order function
    saves references to the original logits and only three per-position FP32
    tensors, then evaluates the exact categorical KL derivative a chunk at a
    time.  Teacher logits and diagnostic outputs never receive gradients.
    """

    @staticmethod
    def forward(ctx, student_logits, teacher_logits, temperature, direction, chunk_size, student_row_indices):
        student_normalizer = _chunked_logsumexp(student_logits, temperature, chunk_size, student_row_indices)
        teacher_normalizer = _chunked_logsumexp(teacher_logits, temperature, chunk_size)
        result = torch.zeros_like(student_normalizer, dtype=torch.float32)
        teacher_entropy = torch.zeros_like(teacher_normalizer, dtype=torch.float32)
        for start in range(0, student_logits.shape[-1], chunk_size):
            stop = start + chunk_size
            student_logp = _logit_chunk(student_logits, start, stop, student_row_indices).float() / temperature - student_normalizer.unsqueeze(-1)
            teacher_logp = teacher_logits[..., start:stop].float() / temperature - teacher_normalizer.unsqueeze(-1)
            teacher_probs = teacher_logp.exp()
            if direction is KLDirection.TEACHER_TO_STUDENT:
                contribution = teacher_probs * (teacher_logp - student_logp)
            else:
                contribution = student_logp.exp() * (student_logp - teacher_logp)
            result.add_(contribution.sum(dim=-1))
            teacher_entropy.sub_((teacher_probs * teacher_logp).sum(dim=-1))
        ctx.temperature = temperature
        ctx.direction = direction
        ctx.chunk_size = chunk_size
        ctx.indexed = student_row_indices is not None
        ctx.save_for_backward(
            student_logits, teacher_logits, student_normalizer, teacher_normalizer, result,
            *((student_row_indices,) if ctx.indexed else ()),
        )
        ctx.mark_non_differentiable(student_normalizer, teacher_normalizer, teacher_entropy)
        ctx.set_materialize_grads(False)
        return result, student_normalizer, teacher_normalizer, teacher_entropy

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_kl, _grad_student_normalizer, _grad_teacher_normalizer, _grad_entropy):
        if grad_kl is None:
            return None, None, None, None, None, None
        student_logits, teacher_logits, student_normalizer, teacher_normalizer, token_kl = ctx.saved_tensors[:5]
        student_row_indices = ctx.saved_tensors[5] if ctx.indexed else None
        # One output-gradient tensor is unavoidable.  All temporary FP32
        # vocabulary tensors are limited to chunk_size columns and discarded
        # each iteration; none are retained as autograd saved activations.
        student_gradient = torch.zeros_like(student_logits) if ctx.indexed else torch.empty_like(student_logits)
        scale = grad_kl.float().unsqueeze(-1) / ctx.temperature
        for start in range(0, student_logits.shape[-1], ctx.chunk_size):
            stop = start + ctx.chunk_size
            student_logp = _logit_chunk(student_logits, start, stop, student_row_indices).float() / ctx.temperature - student_normalizer.unsqueeze(-1)
            teacher_logp = teacher_logits[..., start:stop].float() / ctx.temperature - teacher_normalizer.unsqueeze(-1)
            if ctx.direction is KLDirection.TEACHER_TO_STUDENT:
                gradient = student_logp.exp() - teacher_logp.exp()
            else:
                gradient = student_logp.exp() * (student_logp - teacher_logp - token_kl.unsqueeze(-1))
            if ctx.indexed:
                # Accumulate repeated query rows exactly as index_select's
                # backward does, but without a selected-rows x vocabulary
                # gradient followed by a second parent-sized scatter buffer.
                student_gradient[..., start:stop].index_add_(
                    0, student_row_indices, (gradient * scale).to(student_gradient.dtype)
                )
            else:
                student_gradient[..., start:stop] = gradient * scale
        return student_gradient, None, None, None, None, None


def full_vocab_kl_with_statistics(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    direction: Union[KLDirection, str] = KLDirection.TEACHER_TO_STUDENT,
    temperature: float = 1.0,
    vocab_chunk_size: Optional[int] = None,
    *,
    student_row_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return KL, detached student/teacher log normalizers and teacher entropy.

    All outputs use FP32 arithmetic.  Only KL is differentiable, with respect
    to the student logits only; higher-order derivatives are unsupported.
    The auxiliary per-position statistics let replay diagnostics gather the
    small action supports without materializing full-vocabulary distributions.
    With ``student_row_indices``, inputs are two-dimensional and teacher rows
    align with those student rows in index order. Repeated indices add their
    gradients. Selection is performed inside each vocabulary chunk, avoiding
    a full-vocabulary selected-student copy or its separate backward scatter.
    """

    if student_row_indices is None and student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student_logits and teacher_logits must have identical shapes; "
            f"got {tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
        )
    if student_logits.ndim < 1 or student_logits.shape[-1] < 1:
        raise ValueError("logits must have a nonempty vocabulary dimension")
    if not student_logits.is_floating_point() or not teacher_logits.is_floating_point():
        raise TypeError("student_logits and teacher_logits must be floating-point tensors")
    if student_logits.device != teacher_logits.device:
        raise ValueError("student_logits and teacher_logits must be on the same device")
    if student_row_indices is not None:
        if student_logits.ndim != 2 or teacher_logits.ndim != 2:
            raise ValueError("indexed KL requires two-dimensional student and teacher logits")
        if not isinstance(student_row_indices, torch.Tensor) or student_row_indices.ndim != 1:
            raise ValueError("student_row_indices must be a one-dimensional tensor")
        if student_row_indices.dtype != torch.long:
            raise TypeError("student_row_indices must use torch.long indices")
        if student_row_indices.device != student_logits.device:
            raise ValueError("student_row_indices must be on the logits device")
        if teacher_logits.shape != (student_row_indices.numel(), student_logits.shape[-1]):
            raise ValueError("teacher logits must align with selected student rows and vocabulary")
        if bool(((student_row_indices < 0) | (student_row_indices >= student_logits.shape[0])).any().item()):
            raise ValueError("student_row_indices contain an out-of-range row")

    temperature_value = float(temperature)
    if not math.isfinite(temperature_value) or temperature_value <= 0.0:
        raise ValueError("temperature must be finite and positive")
    try:
        kl_direction = KLDirection(direction)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown KL direction: {direction!r}") from exc

    vocabulary_size = student_logits.shape[-1]
    if vocab_chunk_size is None:
        chunk_size = vocabulary_size
    else:
        if isinstance(vocab_chunk_size, bool) or not isinstance(vocab_chunk_size, int) or vocab_chunk_size <= 0:
            raise ValueError("vocab_chunk_size must be a positive integer or None")
        chunk_size = min(vocab_chunk_size, vocabulary_size)

    return _FullVocabularyKL.apply(
        student_logits, teacher_logits.detach(), temperature_value, kl_direction, chunk_size, student_row_indices
    )


def full_vocab_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    direction: Union[KLDirection, str] = KLDirection.TEACHER_TO_STUDENT,
    temperature: float = 1.0,
    vocab_chunk_size: Optional[int] = None,
    *,
    student_row_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute exact per-position categorical KL in FP32.

    ``temperature`` applies to both distributions without a temperature-squared
    multiplier.  The teacher is detached.  ``vocab_chunk_size`` bounds FP32
    temporaries in both forward and first-order backward; saved activations
    contain the original input logits and O(positions) normalization/KL values,
    regardless of the number of vocabulary chunks.  ``None`` uses one chunk.
    Higher-order derivatives are explicitly unsupported.
    ``student_row_indices`` selects two-dimensional student rows inside the
    custom operation; teacher rows must already follow that index order.
    """

    return full_vocab_kl_with_statistics(
        student_logits, teacher_logits, direction, temperature, vocab_chunk_size,
        student_row_indices=student_row_indices,
    )[0]
