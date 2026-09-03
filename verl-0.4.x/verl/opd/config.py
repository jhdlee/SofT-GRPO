"""Validated, dependency-free configuration objects for OPD."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, TypeVar


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class ScheduleType(_StringEnum):
    WARMUP_CONSTANT = "warmup_constant"
    WARMUP_DECAY = "warmup_decay"


class TeacherType(_StringEnum):
    EMA = "ema"
    CURRENT_ACTOR = "current_actor"
    FROZEN_INITIAL = "frozen_initial"


class KLDirection(_StringEnum):
    TEACHER_TO_STUDENT = "teacher_to_student"
    STUDENT_TO_TEACHER = "student_to_teacher"


class TrajectoryGate(_StringEnum):
    ALL = "all"
    POSITIVE_ADVANTAGE = "positive_advantage"


class PromptTemplate(_StringEnum):
    SDFT = "sdft"
    SDPG = "sdpg"


EnumT = TypeVar("EnumT", bound=_StringEnum)


def _as_enum(value: Any, enum_type: type[EnumT], field_name: str) -> EnumT:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{field_name} must be one of: {choices}; got {value!r}") from exc


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a finite float, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be a finite float; got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite; got {value!r}")
    return result


@dataclass(frozen=True)
class OPDTeacherConfig:
    """Configuration for the privileged teacher."""

    type: TeacherType = TeacherType.EMA
    ema_decay: float = 0.99

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", _as_enum(self.type, TeacherType, "teacher.type"))
        decay = _finite_float(self.ema_decay, "teacher.ema_decay")
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"teacher.ema_decay must be in [0, 1); got {decay}")
        object.__setattr__(self, "ema_decay", decay)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> OPDTeacherConfig:
        allowed = {"type", "ema_decay"}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown teacher configuration fields: {sorted(unknown)}")
        return cls(**dict(values))


@dataclass(frozen=True)
class OPDConfig:
    """The ``algorithm.opd`` configuration subtree.

    ``active`` is the integration guard.  Callers must check it before creating
    or evaluating a teacher, which makes a zero coefficient identical to
    disabling OPD with respect to forwards and RNG consumption.
    """

    enabled: bool = True
    beta_base: float = 0.001
    schedule: ScheduleType = ScheduleType.WARMUP_CONSTANT
    warmup_fraction: float = 0.10
    teacher: OPDTeacherConfig = OPDTeacherConfig()
    kl_direction: KLDirection = KLDirection.TEACHER_TO_STUDENT
    trajectory_gate: TrajectoryGate = TrajectoryGate.ALL
    prompt_template: PromptTemplate = PromptTemplate.SDFT
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError(f"enabled must be bool; got {self.enabled!r}")

        beta = _finite_float(self.beta_base, "beta_base")
        if beta < 0.0:
            raise ValueError(f"beta_base must be nonnegative; got {beta}")
        object.__setattr__(self, "beta_base", beta)

        warmup = _finite_float(self.warmup_fraction, "warmup_fraction")
        if not 0.0 < warmup <= 1.0:
            raise ValueError(f"warmup_fraction must be in (0, 1]; got {warmup}")
        object.__setattr__(self, "warmup_fraction", warmup)

        temperature = _finite_float(self.temperature, "temperature")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive; got {temperature}")
        object.__setattr__(self, "temperature", temperature)

        object.__setattr__(self, "schedule", _as_enum(self.schedule, ScheduleType, "schedule"))
        object.__setattr__(self, "kl_direction", _as_enum(self.kl_direction, KLDirection, "kl_direction"))
        object.__setattr__(
            self,
            "trajectory_gate",
            _as_enum(self.trajectory_gate, TrajectoryGate, "trajectory_gate"),
        )
        object.__setattr__(
            self,
            "prompt_template",
            _as_enum(self.prompt_template, PromptTemplate, "prompt_template"),
        )

        teacher = self.teacher
        if isinstance(teacher, Mapping):
            teacher = OPDTeacherConfig.from_mapping(teacher)
        if not isinstance(teacher, OPDTeacherConfig):
            raise TypeError("teacher must be OPDTeacherConfig or a mapping")
        object.__setattr__(self, "teacher", teacher)

    @property
    def active(self) -> bool:
        """Whether integration should construct and evaluate a teacher."""

        return self.enabled and self.beta_base > 0.0

    @property
    def uses_ema_teacher(self) -> bool:
        return self.active and self.teacher.type is TeacherType.EMA

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> OPDConfig:
        allowed = {
            "enabled",
            "beta_base",
            "schedule",
            "warmup_fraction",
            "teacher",
            "kl_direction",
            "trajectory_gate",
            "prompt_template",
            "temperature",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown OPD configuration fields: {sorted(unknown)}")
        return cls(**dict(values))
