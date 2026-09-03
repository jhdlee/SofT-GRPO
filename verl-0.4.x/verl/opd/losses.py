"""Numerically stable full-vocabulary OPD losses."""

from __future__ import annotations

import math
from typing import Optional, Union

import torch

from .config import KLDirection


def full_vocab_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    direction: Union[KLDirection, str] = KLDirection.TEACHER_TO_STUDENT,
    temperature: float = 1.0,
    vocab_chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Compute exact per-position categorical KL in FP32.

    Args:
        student_logits: Tensor whose final dimension is the full vocabulary.
        teacher_logits: Tensor of the same shape.  It is detached internally.
        direction: Distribution order in the KL divergence.
        temperature: Temperature applied to both distributions.  The returned
            value is the KL itself and is not multiplied by ``temperature**2``.
        vocab_chunk_size: If set, compute normalizers and KL contributions in
            vocabulary chunks.  This remains an exact full-vocabulary KL while
            avoiding full-size FP32 log-probability and probability tensors.

    Returns:
        An FP32 tensor with shape ``student_logits.shape[:-1]``.  Gradients can
        reach only ``student_logits``.
    """

    if student_logits.shape != teacher_logits.shape:
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

    teacher_detached = teacher_logits.detach()

    def chunked_logsumexp(logits: torch.Tensor) -> torch.Tensor:
        normalizer = None
        for start in range(0, vocabulary_size, chunk_size):
            stop = min(start + chunk_size, vocabulary_size)
            chunk_normalizer = torch.logsumexp(logits[..., start:stop].float() / temperature_value, dim=-1)
            normalizer = chunk_normalizer if normalizer is None else torch.logaddexp(normalizer, chunk_normalizer)
        return normalizer

    student_normalizer = chunked_logsumexp(student_logits)
    teacher_normalizer = chunked_logsumexp(teacher_detached)
    result = torch.zeros_like(student_normalizer, dtype=torch.float32)

    for start in range(0, vocabulary_size, chunk_size):
        stop = min(start + chunk_size, vocabulary_size)
        student_log_probs = (
            student_logits[..., start:stop].float() / temperature_value - student_normalizer.unsqueeze(-1)
        )
        teacher_log_probs = (
            teacher_detached[..., start:stop].float() / temperature_value - teacher_normalizer.unsqueeze(-1)
        )
        if kl_direction is KLDirection.TEACHER_TO_STUDENT:
            contribution = teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)
        else:
            contribution = student_log_probs.exp() * (student_log_probs - teacher_log_probs)
        result = result + contribution.sum(dim=-1)
    return result
