import math

import pytest

from verl.opd import (
    KLDirection,
    LossSupport,
    ObjectiveMode,
    OPDConfig,
    PromptTemplate,
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
    assert config.prompt_template is PromptTemplate.SDPG
    assert config.mode is ObjectiveMode.AUXILIARY
    # Compatibility default: existing OPD runs remain latent-only unless the
    # new seven-arm registry opts into all-response supervision explicitly.
    assert config.loss_support is LossSupport.LATENT_ONLY
    assert config.schedule is ScheduleType.WARMUP_CONSTANT
    assert config.warmup_fraction == 0.10


def test_new_fields_preserve_historical_positional_field_order():
    config = OPDConfig(True, 0.2, "warmup_decay", 0.25)
    assert config.schedule is ScheduleType.WARMUP_DECAY
    assert config.warmup_fraction == 0.25
    assert config.mode is ObjectiveMode.AUXILIARY
    assert config.loss_support is LossSupport.LATENT_ONLY


def test_configuration_rejects_unknown_and_invalid_values():
    with pytest.raises(ValueError, match="unknown OPD"):
        OPDConfig.from_mapping({"unexpected": 1})
    with pytest.raises(ValueError, match="nonnegative"):
        OPDConfig(beta_base=-1)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        OPDConfig(warmup_fraction=0)
    with pytest.raises(ValueError, match="kl_direction"):
        OPDConfig(kl_direction="sideways")
    with pytest.raises(ValueError, match="mode"):
        OPDConfig(mode="hybridish")
    with pytest.raises(ValueError, match="loss_support"):
        OPDConfig(loss_support="answer_only")
    with pytest.raises(ValueError, match="ema_decay"):
        OPDConfig(teacher={"ema_decay": 1.0})


def test_ten_percent_warmup_is_derived_from_total_iterations():
    assert warmup_iterations(109, 0.10) == 11
    config = OPDConfig(beta_base=0.001)

    assert effective_beta(config, 0, 109) == 0.0
    assert math.isclose(effective_beta(config, 5, 109), 0.001 * 5 / 11)
    assert effective_beta(config, 11, 109) == 0.001
    assert effective_beta(config, 108, 109) == 0.001


def test_schedule_is_iteration_based_and_warmup_decay_is_available():
    assert schedule_multiplier(11, 109, "warmup_decay", 0.10) == 1.0
    assert 0.0 < schedule_multiplier(108, 109, "warmup_decay", 0.10) < 0.02
    assert schedule_multiplier(109, 109, "warmup_decay", 0.10) == 0.0


def test_constant_schedule_has_full_dose_from_first_iteration():
    config = OPDConfig(
        mode="standalone",
        loss_support="all_response",
        beta_base=1.0,
        schedule="constant",
    )

    assert config.mode is ObjectiveMode.STANDALONE
    assert config.loss_support is LossSupport.ALL_RESPONSE
    assert config.schedule is ScheduleType.CONSTANT
    assert schedule_multiplier(0, 109, config.schedule, config.warmup_fraction) == 1.0
    assert effective_beta(config, 0, 109) == 1.0
    assert effective_beta(config, 108, 109) == 1.0


@pytest.mark.parametrize("config", [OPDConfig(enabled=False), OPDConfig(beta_base=0.0)])
def test_disabled_or_zero_beta_is_an_exact_inactive_configuration(config):
    assert not config.active
    assert not config.uses_ema_teacher
    assert effective_beta(config, 10, 109) == 0.0
