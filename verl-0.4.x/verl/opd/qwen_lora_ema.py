"""Dense effective-weight teacher operations for native Qwen LoRA actors.

FSDP actors and dense teachers have different parameter inventories. Each
operation gathers corresponding root/decoder units, processes canonical dense
parameters, and releases that unit before proceeding. No full-model gathered
copy or adapter-factor EMA is used. Callers must put both models on the same
execution device and call collectively after every actor update has completed.
"""
from __future__ import annotations

from contextlib import nullcontext
import math

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from .ema import EMAUpdateReport
from .qwen_lora import effective_weight_fp32, has_qwen_lora


def _canonical(name):
    return ".".join(part for part in name.split(".") if part != "_fsdp_wrapped_module")


def _fsdp_units(model):
    if not isinstance(model, FSDP):
        if any(isinstance(child, FSDP) for child in model.modules()):
            raise ValueError("native dense teacher helpers require a root FSDP wrapper")
        return {"": model}
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, FSDP):
            key = _canonical(name)
            if key in result:
                raise ValueError("native FSDP unit names collide")
            result[key] = module
    return result


def _summon(unit, *, writeback=False):
    return (FSDP.summon_full_params(unit, recurse=False, writeback=writeback, rank0_only=False, with_grads=False)
            if isinstance(unit, FSDP) else nullcontext())


def _owned_tensors(unit, *, buffers=False):
    """Visit this unit's tensors, excluding separately wrapped descendants."""
    core = unit._fsdp_wrapped_module if isinstance(unit, FSDP) else unit
    result, seen = {}, set()
    def visit(module, prefix):
        if isinstance(module, FSDP):
            return
        tensors = module._buffers if buffers else module._parameters
        for name, tensor in tensors.items():
            if tensor is not None and id(tensor) not in seen:
                if name == "_flat_param":
                    raise ValueError("native teacher operation requires unflattened full parameters")
                seen.add(id(tensor))
                result[prefix + name] = (tensor, module, name)
        for name, child in module.named_children():
            visit(child, prefix + name + ".")
    visit(core, "")
    return result


def _dense_pairs(teacher_unit, student_unit, *, buffers=False):
    teachers = _owned_tensors(teacher_unit, buffers=buffers)
    students = _owned_tensors(student_unit, buffers=buffers)
    if not buffers:
        students = {name: value for name, value in students.items()
                    if value[2] not in ("qwen_lora_A", "qwen_lora_B")}
    if teachers.keys() != students.keys():
        raise ValueError("dense teacher and effective actor tensor names differ")
    for name, (teacher, _, _) in teachers.items():
        student, module, leaf = students[name]
        if teacher.shape != student.shape or teacher.device != student.device or teacher.dtype != student.dtype:
            raise ValueError(f"dense teacher and actor shape/device/dtype differ: {name}")
        if not buffers and (teacher.dtype != torch.float32 or teacher.requires_grad or teacher.grad is not None):
            raise ValueError("dense EMA teacher parameters must be isolated FP32 masters")
        value = effective_weight_fp32(module) if not buffers and leaf == "weight" and hasattr(module, "qwen_lora_A") else student
        yield teacher, value.detach()


def _paired_units(teacher, student):
    if has_qwen_lora(teacher):
        raise ValueError("effective-weight EMA requires a dense teacher, not an adapter teacher")
    teachers, students = _fsdp_units(teacher), _fsdp_units(student)
    if teachers.keys() != students.keys():
        raise ValueError("dense teacher and actor FSDP root/decoder boundaries differ")
    return [(teachers[name], students[name]) for name in teachers]


@torch.no_grad()
def _update_dense_teacher(teacher, student, decay):
    parameters = averaged_buffers = copied_buffers = 0
    for teacher_unit, student_unit in _paired_units(teacher, student):
        with _summon(student_unit), _summon(teacher_unit, writeback=True):
            for target, source in _dense_pairs(teacher_unit, student_unit):
                if decay == 0.0:
                    target.copy_(source)
                else:
                    target.lerp_(source, 1.0 - decay)
                parameters += 1
            for target, source in _dense_pairs(teacher_unit, student_unit, buffers=True):
                if target.is_floating_point() or target.is_complex():
                    if decay == 0.0:
                        target.copy_(source)
                    else:
                        target.lerp_(source, 1.0 - decay)
                    averaged_buffers += 1
                else:
                    target.copy_(source)
                    copied_buffers += 1
    return EMAUpdateReport(parameters, averaged_buffers, copied_buffers)


@torch.no_grad()
def initialize_dense_teacher_(teacher, student):
    """Copy exact FP32 effective actor weights before the first rollout."""
    return _update_dense_teacher(teacher, student, 0.0)


@torch.no_grad()
def update_dense_ema_once_(teacher, student, decay, rollout_iteration, state):
    if isinstance(decay, bool) or not math.isfinite(float(decay)) or not 0 <= decay < 1:
        raise ValueError("dense EMA decay must be finite and in [0, 1)")
    if type(rollout_iteration) is not int or rollout_iteration < 0:
        raise ValueError("dense EMA rollout iteration must be a nonnegative integer")
    if rollout_iteration <= state.last_rollout_iteration:
        raise RuntimeError("dense EMA has already updated for this or a later rollout iteration")
    report = _update_dense_teacher(teacher, student, float(decay))
    state.update_count += 1
    state.last_rollout_iteration = rollout_iteration
    return report


@torch.no_grad()
def effective_parameter_squared_distance_sum_and_count(teacher, student):
    """Return a SUM-reducible effective-parameter distance and element count.

    FSDP summons produce full parameters on all ranks. Only global rank zero
    contributes to these scalars, so the caller's ordinary all-reduce counts
    each dense parameter once. All ranks still enter identical summons.
    """
    first = next(teacher.parameters())
    squared = torch.zeros((), device=first.device, dtype=torch.float64)
    count = torch.zeros((), device=first.device, dtype=torch.int64)
    contribute = not isinstance(teacher, FSDP) or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    for teacher_unit, student_unit in _paired_units(teacher, student):
        with _summon(student_unit), _summon(teacher_unit):
            for target, source in _dense_pairs(teacher_unit, student_unit):
                if contribute:
                    squared.add_((target - source).double().square().sum())
                    count.add_(target.numel())
    return squared, count
