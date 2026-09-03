"""Explicit EMA teacher update helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


def _validate_decay(decay: float) -> float:
    result = float(decay)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError("EMA decay must be finite and in [0, 1)")
    return result


@dataclass(frozen=True)
class EMAUpdateReport:
    parameter_tensors: int
    averaged_buffers: int
    copied_buffers: int


@dataclass
class EMAUpdateState:
    """Checkpointable guard against multiple EMA updates in one iteration."""

    update_count: int = 0
    last_rollout_iteration: int = -1

    def state_dict(self) -> dict[str, int]:
        return {
            "update_count": self.update_count,
            "last_rollout_iteration": self.last_rollout_iteration,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if set(state) != {"update_count", "last_rollout_iteration"}:
            raise ValueError("invalid EMA update state")
        update_count = state["update_count"]
        last_iteration = state["last_rollout_iteration"]
        if not isinstance(update_count, int) or update_count < 0:
            raise ValueError("EMA update_count must be a nonnegative integer")
        if not isinstance(last_iteration, int) or last_iteration < -1:
            raise ValueError("EMA last_rollout_iteration must be an integer >= -1")
        self.update_count = update_count
        self.last_rollout_iteration = last_iteration


def freeze_teacher_(teacher: nn.Module) -> nn.Module:
    """Put a teacher in evaluation mode and disable all autograd leaves."""

    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def _validate_matching_tensors(
    kind: str,
    teacher_tensors: dict[str, torch.Tensor],
    student_tensors: dict[str, torch.Tensor],
) -> None:
    if teacher_tensors.keys() != student_tensors.keys():
        missing = sorted(student_tensors.keys() - teacher_tensors.keys())
        extra = sorted(teacher_tensors.keys() - student_tensors.keys())
        raise ValueError(f"EMA {kind} names differ; missing={missing}, extra={extra}")
    for name, teacher_tensor in teacher_tensors.items():
        student_tensor = student_tensors[name]
        if teacher_tensor.shape != student_tensor.shape:
            raise ValueError(f"EMA {kind} shape differs for {name!r}")
        if teacher_tensor.dtype != student_tensor.dtype:
            raise ValueError(f"EMA {kind} dtype differs for {name!r}")
        if teacher_tensor.device != student_tensor.device:
            raise ValueError(f"EMA {kind} device differs for {name!r}")


@torch.no_grad()
def update_ema_module_(teacher: nn.Module, student: nn.Module, decay: float = 0.99) -> EMAUpdateReport:
    """Update corresponding teacher parameters and buffers in place.

    Floating-point and complex buffers receive EMA updates.  Integer and boolean
    buffers (for example counters) are copied exactly from the student.
    """

    decay_value = _validate_decay(decay)
    teacher_parameters = dict(teacher.named_parameters())
    student_parameters = dict(student.named_parameters())
    teacher_buffers = dict(teacher.named_buffers())
    student_buffers = dict(student.named_buffers())
    _validate_matching_tensors("parameter", teacher_parameters, student_parameters)
    _validate_matching_tensors("buffer", teacher_buffers, student_buffers)

    for name, teacher_parameter in teacher_parameters.items():
        if not (teacher_parameter.is_floating_point() or teacher_parameter.is_complex()):
            raise TypeError(f"EMA parameter {name!r} must be floating point or complex")
        # ``lerp_`` computes self + weight * (end - self), making an identical
        # teacher/student pair a bit-exact fixed point.
        teacher_parameter.lerp_(student_parameters[name].detach(), 1.0 - decay_value)

    averaged_buffers = 0
    copied_buffers = 0
    for name, teacher_buffer in teacher_buffers.items():
        student_buffer = student_buffers[name].detach()
        if teacher_buffer.is_floating_point() or teacher_buffer.is_complex():
            teacher_buffer.lerp_(student_buffer, 1.0 - decay_value)
            averaged_buffers += 1
        else:
            teacher_buffer.copy_(student_buffer)
            copied_buffers += 1

    return EMAUpdateReport(
        parameter_tensors=len(teacher_parameters),
        averaged_buffers=averaged_buffers,
        copied_buffers=copied_buffers,
    )


@torch.no_grad()
def parameter_squared_distance_sum_and_count(
    teacher: nn.Module,
    student: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local squared parameter distance and element count.

    For a sharded model, all-reduce both returned scalars with ``SUM`` and pass
    them to :func:`rms_from_squared_sum_and_count`.  Accumulation is FP64 while
    parameter differences are evaluated in FP32.
    """

    teacher_parameters = dict(teacher.named_parameters())
    student_parameters = dict(student.named_parameters())
    _validate_matching_tensors("parameter", teacher_parameters, student_parameters)
    if not teacher_parameters:
        raise ValueError("cannot compute parameter distance for modules without parameters")

    first_parameter = next(iter(teacher_parameters.values()))
    squared_sum = torch.zeros((), dtype=torch.float64, device=first_parameter.device)
    count = torch.zeros((), dtype=torch.int64, device=first_parameter.device)
    for name, teacher_parameter in teacher_parameters.items():
        delta = teacher_parameter.detach().float() - student_parameters[name].detach().float()
        squared_sum.add_(delta.square().sum(dtype=torch.float64))
        count.add_(teacher_parameter.numel())
    return squared_sum, count


def rms_from_squared_sum_and_count(squared_sum: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """Compute RMS after optional distributed summation of the input scalars."""

    if squared_sum.numel() != 1 or count.numel() != 1:
        raise ValueError("squared_sum and count must be scalar tensors")
    count_float = count.to(device=squared_sum.device, dtype=squared_sum.dtype)
    if count_float.item() <= 0:
        raise ValueError("parameter count must be positive")
    return torch.sqrt(squared_sum / count_float)


def update_ema_once_(
    teacher: nn.Module,
    student: nn.Module,
    decay: float,
    rollout_iteration: int,
    state: EMAUpdateState,
) -> EMAUpdateReport:
    """Apply one EMA update and record its completed rollout iteration."""

    if not isinstance(rollout_iteration, int) or rollout_iteration < 0:
        raise ValueError("rollout_iteration must be a nonnegative integer")
    if rollout_iteration <= state.last_rollout_iteration:
        raise RuntimeError(
            "EMA has already been updated for this or a later rollout iteration: "
            f"requested={rollout_iteration}, last={state.last_rollout_iteration}"
        )
    report = update_ema_module_(teacher, student, decay)
    state.update_count += 1
    state.last_rollout_iteration = rollout_iteration
    return report
