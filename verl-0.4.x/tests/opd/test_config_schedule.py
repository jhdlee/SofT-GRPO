import math

import pytest

from verl.opd import (
    KLDirection,
    OPDConfig,
    ScheduleType,
    TeacherType,
    effective_beta,
    schedule_multiplier,
    warmup_iterations,
)


def test_primary_defaults_and_mapping_conversion():
    config = OPDConfig.from_mapping(
        {
            "teacher": {"type": "ema", "ema_decay": 0.99},
            "kl_direction": "teacher_to_student",
        }
    )

    assert config.active
    assert config.uses_ema_teacher
    assert config.teacher.type is TeacherType.EMA
    assert config.kl_direction is KLDirection.TEACHER_TO_STUDENT
    assert config.schedule is ScheduleType.WARMUP_CONSTANT
    assert config.warmup_fraction == 0.10


def test_configuration_rejects_unknown_and_invalid_values():
    with pytest.raises(ValueError, match="unknown OPD"):
        OPDConfig.from_mapping({"unexpected": 1})
    with pytest.raises(ValueError, match="nonnegative"):
        OPDConfig(beta_base=-1)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        OPDConfig(warmup_fraction=0)
    with pytest.raises(ValueError, match="kl_direction"):
        OPDConfig(kl_direction="sideways")
    with pytest.raises(ValueError, match="ema_decay"):
        OPDConfig(teacher={"ema_decay": 1.0})


def test_ten_percent_warmup_is_derived_from_total_iterations():
    assert warmup_iterations(327, 0.10) == 33
    config = OPDConfig(beta_base=0.001)

    assert effective_beta(config, 0, 327) == 0.0
    assert math.isclose(effective_beta(config, 16, 327), 0.001 * 16 / 33)
    assert effective_beta(config, 33, 327) == 0.001
    assert effective_beta(config, 326, 327) == 0.001


def test_schedule_is_iteration_based_and_warmup_decay_is_available():
    assert schedule_multiplier(33, 327, "warmup_decay", 0.10) == 1.0
    assert 0.0 < schedule_multiplier(326, 327, "warmup_decay", 0.10) < 0.01
    assert schedule_multiplier(327, 327, "warmup_decay", 0.10) == 0.0


@pytest.mark.parametrize("config", [OPDConfig(enabled=False), OPDConfig(beta_base=0.0)])
def test_disabled_or_zero_beta_is_an_exact_inactive_configuration(config):
    assert not config.active
    assert not config.uses_ema_teacher
    assert effective_beta(config, 10, 327) == 0.0
