"""Iteration-level coefficient schedules for OPD."""

from __future__ import annotations

import math
from typing import Union

from .config import OPDConfig, ScheduleType


def _validate_total_iterations(total_iterations: int) -> None:
    if isinstance(total_iterations, bool) or not isinstance(total_iterations, int):
        raise TypeError("total_iterations must be an integer")
    if total_iterations <= 0:
        raise ValueError("total_iterations must be positive")


def _validate_rollout_iteration(rollout_iteration: int) -> None:
    if isinstance(rollout_iteration, bool) or not isinstance(rollout_iteration, int):
        raise TypeError("rollout_iteration must be an integer")
    if rollout_iteration < 0:
        raise ValueError("rollout_iteration must be nonnegative")


def warmup_iterations(total_iterations: int, warmup_fraction: float = 0.10) -> int:
    """Return ``ceil(warmup_fraction * total_iterations)``.

    The configured fraction is resolved only after the trainer knows its final
    number of rollout iterations; there is deliberately no independently
    configurable warm-up step count.
    """

    _validate_total_iterations(total_iterations)
    fraction = float(warmup_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("warmup_fraction must be finite and in (0, 1]")
    return int(math.ceil(total_iterations * fraction))


def schedule_multiplier(
    rollout_iteration: int,
    total_iterations: int,
    schedule: Union[ScheduleType, str] = ScheduleType.WARMUP_CONSTANT,
    warmup_fraction: float = 0.10,
) -> float:
    """Return the OPD schedule multiplier for a zero-based iteration.

    ``warmup_decay`` linearly decays after warm-up and reaches zero at the
    boundary ``rollout_iteration == total_iterations``.  Normal training
    iterations are ``0 .. total_iterations - 1``.
    """

    _validate_rollout_iteration(rollout_iteration)
    _validate_total_iterations(total_iterations)
    try:
        schedule_type = ScheduleType(schedule)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown OPD schedule: {schedule!r}") from exc

    if schedule_type is ScheduleType.CONSTANT:
        return 1.0

    warmup = warmup_iterations(total_iterations, warmup_fraction)
    if rollout_iteration <= warmup:
        return min(1.0, rollout_iteration / warmup)

    if schedule_type is ScheduleType.WARMUP_CONSTANT:
        return 1.0

    decay_iterations = total_iterations - warmup
    if decay_iterations <= 0:
        return 1.0
    return max(0.0, (total_iterations - rollout_iteration) / decay_iterations)


def effective_beta(config: OPDConfig, rollout_iteration: int, total_iterations: int) -> float:
    """Return the effective auxiliary-loss coefficient.

    Inactive and zero-dose configurations return exactly zero.  Integration
    should use :attr:`OPDConfig.active` *before* any teacher work.
    """

    if not isinstance(config, OPDConfig):
        raise TypeError("config must be an OPDConfig")
    if not config.active:
        return 0.0
    return config.beta_base * schedule_multiplier(
        rollout_iteration=rollout_iteration,
        total_iterations=total_iterations,
        schedule=config.schedule,
        warmup_fraction=config.warmup_fraction,
    )
