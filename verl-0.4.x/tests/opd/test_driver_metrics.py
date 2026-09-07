from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from verl.opd import (
    ITERATION_METRICS,
    STANDALONE_INAPPLICABLE_METRICS,
    STANDALONE_ITERATION_METRICS,
    VALIDATION_METRICS,
    ObjectiveMode,
    OPDConfig,
    required_iteration_metrics,
    validate_resource_limits,
)
from verl.trainer.ppo.opd_driver import (
    RolloutIntegrityConfig,
    add_canonical_metric_aliases,
    compute_rollout_diagnostics,
    mask_invalid_native_boundary_scores,
    replay_integrity_mask,
    replay_ratio_abs_error_max,
    reward_and_group_metrics,
    schedule_meta_info,
    training_rollout_iteration,
    training_wandb_step,
    validate_full_dose_gradient_integrity,
    validate_iteration_metric_contract,
    validate_rollout_integrity,
    validate_validation_metric_contract,
    validation_metric_aliases,
    validation_rollout_iteration,
    validation_wandb_step,
)
from verl.utils.metric import reduce_metrics


def _valid_transition_diagnostics():
    responses = torch.tensor([[4, 99, 7, 8]])
    mask = torch.ones_like(responses)
    # Released SGLang records the emitted </think> action itself as the first
    # categorical position when it flips out of soft-thinking mode.
    topk_ids = torch.tensor([[[4, 5, 6], [99, 0, 0], [7, 0, 0], [8, 0, 0]]])
    topk_actions = torch.tensor([[[9.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    return compute_rollout_diagnostics(
        responses=responses,
        response_mask=mask,
        rollout_topk_ids=topk_ids,
        rollout_topk_gumbels=topk_actions,
        gumbel_temperature=0.1,
        close_tag_token_id=99,
        decode=lambda ids: r"</think>\boxed{1}" if ids == [99, 7, 8] else "",
    )


def _qwen_completion_rate_diagnostics():
    # Short trajectories reproduce the observed 469159 population counts,
    # without loading private rollout artifacts or allocating 8192-token rows.
    # Twelve unclosed soft caps, two unboxed hard caps, and fifty boxed answers.
    responses = torch.tensor([[4, 99, 7, 8, 0]] * 64)
    responses[:12] = torch.tensor([4, 5, 6, 7, 8])
    responses[12:14] = torch.tensor([4, 99, 17, 18, 19])
    supports = torch.stack((responses, torch.zeros_like(responses), torch.zeros_like(responses)), dim=-1)
    supports[:, 0, 1:] = torch.tensor([10, 11])
    supports[:12, :, 1:] = torch.tensor([10, 11])
    actions = torch.zeros_like(supports, dtype=torch.float32)
    actions[:, 0, 0] = 4
    actions[:12, :, 0] = 4
    return compute_rollout_diagnostics(
        responses=responses, response_mask=responses.ne(0),
        rollout_topk_ids=supports, rollout_topk_gumbels=actions,
        gumbel_temperature=0.1, close_tag_token_id=99,
        decode=lambda ids: r"</think>\boxed{1}" if 7 in ids else "",
    )


def test_disabled_completion_gate_accepts_observed_qwen_rates_without_changing_diagnostics():
    diagnostics = _qwen_completion_rate_diagnostics()
    before = deepcopy(diagnostics)
    assert diagnostics.metrics["latent/cap_rate"] == 14 / 64
    assert diagnostics.all_soft_rate == 12 / 64
    assert diagnostics.metrics["latent/close_tag_rate"] == 52 / 64
    assert diagnostics.metrics["latent/soft_to_hard_rate"] == 52 / 64
    assert diagnostics.categorical_boxed_answer_rate == 50 / 64
    assert sum(diagnostics.boundary_valid_mask) == 50
    assert diagnostics.metrics["replay/fallback_count"] == 0
    config = RolloutIntegrityConfig.from_mapping({
        "enabled": True, "gate_first_n_iterations": 1, "completion_gate_enabled": False,
    })
    assert config.max_replay_ratio_abs_error == 1e-4
    validate_rollout_integrity(diagnostics, replay_error=2.2649765014648438e-6,
                               config=config, rollout_iteration=0)
    assert diagnostics == before
    assert diagnostics.boundary_valid_mask[:14] == (False,) * 14


def test_legacy_completion_gate_still_rejects_observed_qwen_rates():
    config = RolloutIntegrityConfig(enabled=True, gate_first_n_iterations=1)
    assert config.completion_gate_enabled is True
    with pytest.raises(RuntimeError, match="cap rate=0.21875") as error:
        validate_rollout_integrity(_qwen_completion_rate_diagnostics(), replay_error=0,
                                   config=config, rollout_iteration=0)
    for field in ("all-soft rate=0.1875", "close-tag rate=0.8125",
                  "soft-to-hard rate=0.8125", "categorical boxed-answer rate=0.78125"):
        assert field in str(error.value)


@pytest.mark.parametrize("invalid", [0, 1, "false", "true", None, []])
def test_completion_gate_configuration_requires_an_actual_boolean(invalid):
    with pytest.raises(TypeError, match="completion_gate_enabled"):
        RolloutIntegrityConfig.from_mapping({"completion_gate_enabled": invalid})


@pytest.mark.parametrize("failure,match", [
    ("ratio", "ratio error"), ("nan_ratio", "ratio error"), ("infinite_ratio", "ratio error"),
    ("nonfinite_metric", "non-finite"), ("nonfinite_all_soft", "non-finite"),
    ("nonfinite_boxed", "non-finite"), ("fallback", "categorical replay fallback"),
    ("inactive", "replay is not active"),
])
def test_disabled_completion_gate_keeps_numerical_replay_and_metadata_checks(failure, match):
    diagnostics = _qwen_completion_rate_diagnostics()
    replay_error = 0.0
    if failure == "ratio": replay_error = 1.0001e-4
    elif failure == "nan_ratio": replay_error = float("nan")
    elif failure == "infinite_ratio": replay_error = float("inf")
    elif failure == "nonfinite_metric": diagnostics.metrics["latent/cap_rate"] = float("nan")
    elif failure == "nonfinite_all_soft": diagnostics = replace(diagnostics, all_soft_rate=float("inf"))
    elif failure == "nonfinite_boxed": diagnostics = replace(diagnostics, categorical_boxed_answer_rate=float("nan"))
    elif failure == "fallback": diagnostics.metrics["replay/fallback_count"] = 1.0
    elif failure == "inactive": diagnostics.metrics["integrity/continuous_replay_active"] = 0.0
    with pytest.raises(RuntimeError, match=match):
        validate_rollout_integrity(diagnostics, replay_error=replay_error,
            config=RolloutIntegrityConfig(enabled=True, completion_gate_enabled=False, gate_first_n_iterations=1),
            rollout_iteration=0)


def test_training_and_validation_use_distinct_exact_rollout_axes():
    assert [training_rollout_iteration(step) for step in range(1, 110)] == list(
        range(109)
    )
    completed = [0, 25, 50, 75, 100, 109]
    assert [validation_rollout_iteration(step) for step in completed] == completed

    # W&B's internal event steps remain strictly ordered even when validation
    # at completed count 25 is followed by training rollout index 25.
    assert validation_wandb_step(0) < training_wandb_step(1)
    for step in completed[1:]:
        assert training_wandb_step(step) < validation_wandb_step(step)
        if step < 109:
            assert validation_wandb_step(step) < training_wandb_step(step + 1)


def test_trainer_logs_validation_as_separate_completed_count_events():
    source = (
        Path(__file__).resolve().parents[2]
        / "verl"
        / "trainer"
        / "ppo"
        / "ray_trainer.py"
    ).read_text(encoding="utf-8")
    assert 'metric_dict["trainer/rollout_iteration"] = validation_rollout_iteration(' in source
    assert "logger.log(data=metrics, step=training_wandb_step(self.global_steps))" in source
    assert "step=validation_wandb_step(self.global_steps)" in source
    assert "metrics.update(val_metrics)" not in source

    scheduled_completed_counts = [
        step for step in range(1, 110) if step % 25 == 0 or step == 109
    ]
    assert [0, *scheduled_completed_counts] == [0, 25, 50, 75, 100, 109]


@pytest.mark.parametrize(
    "function,value,error",
    [
        (training_rollout_iteration, 0, ValueError),
        (training_wandb_step, 0, ValueError),
        (validation_rollout_iteration, -1, ValueError),
        (validation_wandb_step, True, TypeError),
    ],
)
def test_rollout_axis_helpers_fail_closed(function, value, error):
    with pytest.raises(error):
        function(value)


def test_rollout_diagnostics_detect_real_soft_to_hard_boundary():
    diagnostics = _valid_transition_diagnostics()
    assert diagnostics.metrics["latent/length_mean"] == 1.0
    assert diagnostics.metrics["latent/hard_answer_length_mean"] == 3.0
    assert diagnostics.metrics["latent/close_tag_rate"] == 1.0
    assert diagnostics.metrics["latent/soft_to_hard_rate"] == 1.0
    assert diagnostics.metrics["latent/soft_hard_agreement"] == 1.0
    assert diagnostics.metrics["replay/fallback_count"] == 0.0
    assert diagnostics.all_soft_rate == 0.0
    assert diagnostics.categorical_boxed_answer_rate == 1.0
    assert diagnostics.boundary_valid_mask == (True,)


def test_validation_masks_correct_hard_shadow_when_native_boundary_is_invalid():
    diagnostics = compute_rollout_diagnostics(
        responses=torch.tensor([[4, 5, 0, 0], [4, 99, 7, 8]]),
        response_mask=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]]),
        rollout_topk_ids=torch.tensor(
            [
                [[4, 6, 7], [5, 6, 7], [0, 0, 0], [0, 0, 0]],
                [[4, 5, 6], [99, 0, 0], [7, 0, 0], [8, 0, 0]],
            ]
        ),
        rollout_topk_gumbels=torch.tensor(
            [
                [[9.0, 1.0, 0.0], [9.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[9.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        ),
        gumbel_temperature=0.1,
        close_tag_token_id=99,
        decode=lambda ids: r"\boxed{1}" if ids else "",
    )
    assert diagnostics.boundary_valid_mask == (False, True)

    reward, extra = mask_invalid_native_boundary_scores(
        reward_tensor=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        reward_extra_info={
            "score": [1.0, 1.0],
            "released_reward": [1.0, 1.0],
            "math_verify": [1.0, 1.0],
            "diagnostic": [3.0, 4.0],
        },
        boundary_valid_mask=diagnostics.boundary_valid_mask,
    )
    assert reward.sum(dim=-1).tolist() == [0.0, 1.0]
    assert extra["score"] == [0.0, 1.0]
    assert extra["released_reward"] == [0.0, 1.0]
    assert extra["math_verify"] == [0.0, 1.0]
    assert extra["diagnostic"] == [3.0, 4.0]


def test_rollout_diagnostics_counts_observed_replay_fallbacks():
    # Position 0 is incorrectly categorical before </think>; position 2 is
    # incorrectly continuous afterward.  Both are observed metadata failures,
    # rather than a synthetic counter supplied by the test.
    diagnostics = compute_rollout_diagnostics(
        responses=torch.tensor([[4, 99, 7]]),
        response_mask=torch.ones(1, 3, dtype=torch.long),
        rollout_topk_ids=torch.tensor([[[4, 0, 0], [99, 0, 0], [7, 8, 9]]]),
        rollout_topk_gumbels=torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [9.0, 1.0, 0.0]]]
        ),
        gumbel_temperature=0.1,
        close_tag_token_id=99,
        decode=lambda ids: r"\boxed{1}",
    )

    assert diagnostics.metrics["replay/fallback_count"] == 2.0
    with pytest.raises(RuntimeError, match="categorical replay fallback"):
        validate_rollout_integrity(
            diagnostics,
            replay_error=0.0,
            config=RolloutIntegrityConfig(enabled=True),
            rollout_iteration=20,
        )


def test_rollout_diagnostics_counts_support_head_mismatch_as_fallback():
    diagnostics = compute_rollout_diagnostics(
        responses=torch.tensor([[4, 99, 7]]),
        response_mask=torch.ones(1, 3, dtype=torch.long),
        rollout_topk_ids=torch.tensor([[[5, 4, 6], [99, 0, 0], [7, 0, 0]]]),
        rollout_topk_gumbels=torch.tensor(
            [[[9.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        ),
        gumbel_temperature=0.1,
        close_tag_token_id=99,
        decode=lambda ids: r"\boxed{1}",
    )

    assert diagnostics.metrics["replay/fallback_count"] == 1.0


@pytest.mark.parametrize("completion_gate_enabled", [True, False])
def test_all_soft_cap_diagnostics_are_preserved_with_optional_completion_gate(completion_gate_enabled):
    responses = torch.tensor([[4, 5]])
    topk_ids = torch.tensor([[[4, 7], [5, 8]]])
    diagnostics = compute_rollout_diagnostics(
        responses=responses,
        response_mask=torch.ones_like(responses),
        rollout_topk_ids=topk_ids,
        rollout_topk_gumbels=torch.tensor([[[4.0, 0.0], [4.0, 0.0]]]),
        gumbel_temperature=0.1,
        close_tag_token_id=99,
        decode=lambda ids: "",
    )
    before = deepcopy(diagnostics)
    config = RolloutIntegrityConfig(enabled=True, gate_first_n_iterations=1,
                                    completion_gate_enabled=completion_gate_enabled)
    if completion_gate_enabled:
        with pytest.raises(RuntimeError, match="integrity gate failed"):
            validate_rollout_integrity(diagnostics, replay_error=0.0, config=config, rollout_iteration=0)
    else:
        # An all-soft prefix is replayable even when the policy has not emitted
        # a close tag. Its incompleteness remains visible, not fabricated away.
        validate_rollout_integrity(
            diagnostics, replay_error=1e-4, config=config, rollout_iteration=0,
        )
    assert diagnostics == before
    assert diagnostics.all_soft_rate == diagnostics.metrics["latent/cap_rate"] == 1.0
    assert diagnostics.metrics["latent/soft_to_hard_rate"] == diagnostics.categorical_boxed_answer_rate == 0.0
    assert diagnostics.boundary_valid_mask == (False,)


def test_integrity_gate_always_enforces_exact_replay():
    with pytest.raises(RuntimeError, match="ratio error"):
        validate_rollout_integrity(
            _valid_transition_diagnostics(),
            replay_error=1e-2,
            config=RolloutIntegrityConfig(enabled=True, gate_first_n_iterations=0),
            rollout_iteration=100,
        )


def _gradient_gate_config(**overrides):
    values = {
        "enabled": True,
        "full_dose_gradient_gate_enabled": True,
    }
    values.update(overrides)
    return RolloutIntegrityConfig.from_mapping(values)


@pytest.mark.parametrize("ratio", [1e-12, 0.000507638, 0.099, 0.1, 1.0, 10.0, 10.01, 1e6])
def test_full_dose_gradient_gate_keeps_ratio_diagnostic_with_clip_boundary(ratio):
    config = _gradient_gate_config()
    validate_full_dose_gradient_integrity(
        {"grad/grpo_norm": 2.0, "grad/opd_norm": 2.0 * ratio, "actor/gradient_clipfrac": 0.5},
        config,
        schedule_multiplier=1.0,
    )


def test_disabling_completion_gate_does_not_disable_full_dose_gradient_gate():
    with pytest.raises(RuntimeError, match="clip fraction"):
        validate_full_dose_gradient_integrity(
            {"grad/grpo_norm": 1.0, "grad/opd_norm": 100.0, "actor/gradient_clipfrac": 1.0},
            _gradient_gate_config(completion_gate_enabled=False), schedule_multiplier=1.0,
        )


@pytest.mark.parametrize(
    ("metrics", "message"),
    [
        (
            {
                "grad/grpo_norm": 1.0,
                "grad/opd_norm": 0.0,
                "actor/gradient_clipfrac": 0.0,
            },
            "positive OPD",
        ),
        (
            {
                "grad/grpo_norm": 0.0,
                "grad/opd_norm": 10.01,
                "actor/gradient_clipfrac": 0.0,
            },
            "positive GRPO",
        ),
        (
            {
                "grad/grpo_norm": 1.0,
                "grad/opd_norm": 1.0,
                "actor/gradient_clipfrac": 0.5001,
            },
            "clip fraction",
        ),
    ],
)
def test_full_dose_gradient_gate_rejects_zero_gradients_and_excessive_clipping(metrics, message):
    with pytest.raises(RuntimeError, match=message):
        validate_full_dose_gradient_integrity(
            metrics,
            _gradient_gate_config(),
            schedule_multiplier=1.0,
        )


def test_full_dose_gradient_gate_skips_warmup_and_disabled_stress_arm():
    invalid = {
        "grad/grpo_norm": 1.0,
        "grad/opd_norm": 100.0,
        "actor/gradient_clipfrac": 1.0,
    }
    validate_full_dose_gradient_integrity(
        invalid,
        _gradient_gate_config(),
        schedule_multiplier=0.999,
    )
    validate_full_dose_gradient_integrity(
        invalid,
        RolloutIntegrityConfig(enabled=True),
        schedule_multiplier=1.0,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_full_dose_gradient_clip_fraction": 1.01},
        {"full_dose_gradient_gate_enabled": 1},
    ],
)
def test_full_dose_gradient_gate_config_rejects_invalid_thresholds(overrides):
    with pytest.raises((TypeError, ValueError)):
        _gradient_gate_config(**overrides)


@pytest.mark.parametrize("legacy", [
    {"min_opd_grpo_support_gradient_ratio": 0.0},
    {"min_opd_grpo_support_gradient_ratio": 2.0, "max_opd_grpo_support_gradient_ratio": 1.0},
    {"min_opd_grpo_support_gradient_ratio": 1e100, "max_opd_grpo_support_gradient_ratio": 1e101},
])
def test_legacy_ratio_bounds_cannot_reenable_acceptance_range(legacy):
    validate_full_dose_gradient_integrity(
        {"grad/grpo_norm": 1.0, "grad/opd_norm": 1e-8, "actor/gradient_clipfrac": 0.0},
        _gradient_gate_config(**legacy), schedule_multiplier=1.0,
    )


@pytest.mark.parametrize("name", ["grad/grpo_norm", "grad/opd_norm", "actor/gradient_clipfrac"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_gradient_diagnostics_still_reject_nonfinite_or_negative_values(name, value):
    metrics = {"grad/grpo_norm": 1.0, "grad/opd_norm": 0.001, "actor/gradient_clipfrac": 0.0}
    metrics[name] = value
    with pytest.raises(RuntimeError):
        validate_full_dose_gradient_integrity(metrics, _gradient_gate_config(), schedule_multiplier=1.0)


def test_full_dose_gradient_gate_rejects_nonfinite_derived_ratio():
    with pytest.raises(RuntimeError, match="ratio must be finite"):
        validate_full_dose_gradient_integrity(
            {"grad/grpo_norm": 1e-300, "grad/opd_norm": 1e300, "actor/gradient_clipfrac": 0.0},
            _gradient_gate_config(), schedule_multiplier=1.0,
        )


@pytest.mark.parametrize("opd_norm,grpo_norm", [(1e-8, 1.0), (100.0, 1.0), (0.0, 1.0)])
def test_canonical_metrics_record_component_ratio_without_a_range_gate(opd_norm, grpo_norm):
    metrics = add_canonical_metric_aliases(
        {"actor/grpo_grad_norm": grpo_norm, "actor/opd_grad_norm": opd_norm},
        opd_config=OPDConfig(), rollout_iteration=11, total_iterations=109,
        optimizer_step=24, grad_clip=1.0, checkpoint_committed=False, resumed=False,
    )
    assert metrics["grad/opd_grpo_support_gradient_ratio"] == opd_norm / grpo_norm


def test_canonical_metrics_omit_undefined_component_ratio():
    metrics = add_canonical_metric_aliases(
        {"actor/grpo_grad_norm": 0.0, "actor/opd_grad_norm": 1.0},
        opd_config=OPDConfig(), rollout_iteration=11, total_iterations=109,
        optimizer_step=24, grad_clip=1.0, checkpoint_committed=False, resumed=False,
    )
    assert "grad/opd_grpo_support_gradient_ratio" not in metrics


def test_replay_ratio_metric_equals_zero_for_identical_log_densities():
    logp = torch.tensor([[-2.0, -0.5]])
    assert replay_ratio_abs_error_max(logp, logp.clone(), torch.ones_like(logp)) == 0.0


def test_native_soft_replay_integrity_excludes_rewritten_close_transition():
    responses = torch.tensor([[4, 99, 7, 8]])
    response_mask = torch.ones_like(responses)
    supports = torch.tensor(
        [[[4, 5, 6], [99, 0, 0], [7, 0, 0], [8, 0, 0]]]
    )

    comparable = replay_integrity_mask(
        response_mask=response_mask,
        continuous_replay=True,
        rollout_topk_ids=supports,
        responses=responses,
        close_tag_token_id=99,
    )

    assert comparable.tolist() == [[True, False, True, True]]
    # SGLang rewrites the close action's support and density as categorical
    # metadata after selecting it as a Gumbel action. Its exclusion must
    # prevent that non-replayable boundary from causing a false failure.
    rollout_logp = torch.tensor([[-2.0, -0.1, -0.5, -0.25]])
    actor_logp = torch.tensor([[-2.0, -9.0, -0.5, -0.25]])
    assert replay_ratio_abs_error_max(
        rollout_logp, actor_logp, comparable
    ) == 0.0


def test_categorical_replay_integrity_keeps_every_valid_action():
    response_mask = torch.tensor([[1, 1, 0]])
    comparable = replay_integrity_mask(
        response_mask=response_mask,
        continuous_replay=False,
    )
    assert comparable.tolist() == [[True, True, False]]


def test_reward_metrics_count_informative_groups_not_trajectories():
    metrics = reward_and_group_metrics(
        torch.tensor([1.0, 0.0, 1.0, 1.0]),
        torch.tensor([1.0, -1.0, 0.0, 0.0]),
        ["a", "a", "b", "b"],
    )
    assert metrics["train/reward_mean"] == 0.75
    assert metrics["train/correct_fraction"] == 0.75
    assert metrics["train/nonzero_advantage_group_fraction"] == 0.5


def test_schedule_metadata_uses_zero_based_iteration_and_resolved_warmup():
    metadata = schedule_meta_info(OPDConfig(), rollout_iteration=0, total_iterations=109)
    assert metadata["opd_warmup_iterations"] == 11
    assert metadata["opd_schedule_multiplier"] == 0.0
    assert metadata["opd_beta_effective"] == 0.0
    assert "warmup_iterations" not in metadata["opd_config"]


def test_canonical_aliases_never_emit_ambiguous_kl_weight():
    metrics = add_canonical_metric_aliases(
        {
            "actor/pg_loss": 2.0,
            "actor/kl_loss": 3.0,
            "actor/grad_norm": 0.5,
            "actor/gradient_clipfrac": 0.0,
        },
        opd_config=OPDConfig(enabled=False),
        rollout_iteration=0,
        total_iterations=327,
        optimizer_step=2,
        grad_clip=1.0,
        checkpoint_committed=False,
        resumed=False,
    )
    assert metrics["trainer/rollout_iteration"] == 0
    assert metrics["trainer/optimizer_step"] == 2
    assert metrics["opd/schedule_multiplier"] == 0.0
    assert metrics["opd/beta_effective"] == 0.0
    assert "opd/kl_weight" not in metrics


@pytest.mark.parametrize(
    ("clip_fraction", "expected"),
    [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)],
)
def test_grad_clipped_reports_fraction_of_clipped_optimizer_updates(
    clip_fraction,
    expected,
):
    metrics = add_canonical_metric_aliases(
        {
            # A mean below the clipping threshold must not hide a clipped step.
            "actor/grad_norm": 0.75,
            "actor/gradient_clipfrac": clip_fraction,
        },
        opd_config=OPDConfig(enabled=False),
        rollout_iteration=0,
        total_iterations=327,
        optimizer_step=2,
        grad_clip=1.0,
        checkpoint_committed=False,
        resumed=False,
    )

    assert metrics["grad/clipped"] == expected


@pytest.mark.parametrize("clip_fraction", [-0.1, 1.1, float("nan")])
def test_grad_clipped_rejects_invalid_actor_indicator(clip_fraction):
    with pytest.raises(ValueError, match="gradient_clipfrac"):
        add_canonical_metric_aliases(
            {
                "actor/grad_norm": 0.75,
                "actor/gradient_clipfrac": clip_fraction,
            },
            opd_config=OPDConfig(enabled=False),
            rollout_iteration=0,
            total_iterations=327,
            optimizer_step=2,
            grad_clip=1.0,
            checkpoint_committed=False,
            resumed=False,
        )


def test_inactive_opd_production_metrics_still_satisfy_iteration_contract():
    """The baseline arm publishes the same schema without doing teacher work."""

    upstream_metrics = {
        "train/reward_mean": 0.25,
        "train/reward_std": 0.1,
        "train/correct_fraction": 0.25,
        "train/nonzero_advantage_group_fraction": 0.5,
        "actor/pg_loss": -0.02,
        "actor/kl_loss": 0.03,
        "actor/grad_norm": 0.4,
        "actor/gradient_clipfrac": 0.0,
        "actor/grpo_grad_norm": 0.2,
        "actor/ratio_mean": 1.0,
        "actor/ratio_p95": 1.01,
        "actor/pg_clipfrac": 0.0,
        "actor/ppo_kl": 0.001,
        "perf/max_memory_allocated_gb": 11.0,
        "perf/cpu_memory_used_gb": 20.0,
        "perf/rollout_tokens_per_second": 100.0,
        "perf/train_tokens_per_second": 50.0,
        "perf/iteration_seconds": 5.0,
        "system/cpu_utilization": 25.0,
        "latent/length_mean": 8.0,
        "latent/length_p95": 12.0,
        "latent/hard_answer_length_mean": 4.0,
        "latent/close_tag_rate": 1.0,
        "latent/soft_to_hard_rate": 1.0,
        "latent/cap_rate": 0.0,
        "latent/mixture_entropy_mean": 0.1,
        "latent/top1_weight_mean": 0.9,
        "latent/soft_hard_agreement": 1.0,
        "replay/ratio_abs_error_max": 0.0,
        "replay/fallback_count": 0.0,
        "integrity/continuous_replay_active": 1.0,
    }
    metrics = add_canonical_metric_aliases(
        upstream_metrics,
        opd_config=OPDConfig(enabled=False),
        rollout_iteration=4,
        total_iterations=327,
        optimizer_step=10,
        grad_clip=1.0,
        checkpoint_committed=False,
        resumed=False,
    )

    validate_iteration_metric_contract(metrics)
    assert metrics["loss/opd_kl_unweighted"] == 0.0
    assert metrics["loss/opd_weighted"] == 0.0
    assert metrics["opd/teacher_student_param_rms"] == 0.0
    assert metrics["opd/ema_update_count"] == 0.0
    assert metrics["grad/opd_norm"] == 0.0
    assert metrics["perf/teacher_seconds"] == 0.0


def test_validation_aliases_supply_both_required_reward_interfaces():
    raw = {
        "val-core/math/reward/mean@1": 0.5,
        "val-aux/math/math_verify/mean@1": 0.75,
    }
    aliases = validation_metric_aliases(raw)
    payload = {
        "trainer/rollout_iteration": 0,
        **aliases,
        "val/response_length_mean": 20.0,
        "val/latent_length_mean": 12.0,
        "val/cap_rate": 0.0,
        "val/soft_to_hard_rate": 1.0,
    }

    validate_validation_metric_contract(payload)
    assert aliases == {
        "val/released_reward/mean_at_1": 0.5,
        "val/math_verify/mean_at_1": 0.75,
        "val/released_reward/pass_at_1": 0.5,
        "val/math_verify/pass_at_1": 0.75,
    }


def test_hbm_alias_uses_the_maximum_worker_rank():
    # Ray collects one worker value per rank.  VERL's reduction uses MAX for
    # metric names containing "max", after which the canonical alias must retain
    # that value rather than averaging it a second time.
    reduced = reduce_metrics(
        {"perf/max_memory_allocated_gb": [41.0, 68.5, 52.0, 47.0]}
    )
    metrics = add_canonical_metric_aliases(
        reduced,
        opd_config=OPDConfig(enabled=False),
        rollout_iteration=0,
        total_iterations=327,
        optimizer_step=2,
        grad_clip=1.0,
        checkpoint_committed=False,
        resumed=False,
    )

    assert reduced["perf/max_memory_allocated_gb"] == 68.5
    assert metrics["system/hbm_peak_gib"] == 68.5


@pytest.mark.parametrize("coefficient", [0.0, 0.001])
def test_reference_kl_option_changes_only_reference_part_of_total_metric(coefficient):
    metrics = add_canonical_metric_aliases(
        {"actor/pg_loss": 2.0, "actor/kl_loss": 3.0, "actor/opd_weighted": 0.25},
        opd_config=OPDConfig(), rollout_iteration=1, total_iterations=109,
        optimizer_step=4, grad_clip=1.0, checkpoint_committed=False, resumed=False,
        reference_kl_coef=coefficient,
    )
    assert metrics["loss/total"] == pytest.approx(2.25 + 3.0 * coefficient)
    assert metrics["loss/opd_weighted"] == 0.25


def test_per_rank_resource_gate_enforces_fixed_acceptance_limits():
    validate_resource_limits(hbm_peak_gib=71.999, host_ram_percent=89.999)
    with pytest.raises(RuntimeError, match="HBM gate failed"):
        validate_resource_limits(hbm_peak_gib=72.0, host_ram_percent=20.0)
    with pytest.raises(RuntimeError, match="host-RAM gate failed"):
        validate_resource_limits(hbm_peak_gib=20.0, host_ram_percent=90.0)


@pytest.mark.parametrize(
    ("required", "validator"),
    [
        (ITERATION_METRICS, validate_iteration_metric_contract),
        (VALIDATION_METRICS, validate_validation_metric_contract),
    ],
)
def test_study_metric_contract_is_complete_and_fails_closed(required, validator):
    payload = {name: 0.0 for name in required}
    if "opd/teacher_type" in payload:
        payload["opd/teacher_type"] = "ema"
        payload["algorithm/objective_mode"] = "auxiliary"
        payload["opd/loss_support"] = "latent_only"
    if "grad/diagnostic_space" in payload:
        payload["grad/diagnostic_space"] = "fixed_top5_action_logits"
    validator(payload)

    missing_name = sorted(required)[0]
    payload.pop(missing_name)
    with pytest.raises(ValueError, match="missing required W&B metrics"):
        validator(payload)


@pytest.mark.parametrize(
    ("required", "validator", "numeric_name"),
    [
        (ITERATION_METRICS, validate_iteration_metric_contract, "loss/total"),
        (VALIDATION_METRICS, validate_validation_metric_contract, "val/cap_rate"),
    ],
)
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), True, "1.0"])
def test_study_metric_contract_rejects_nonfinite_or_nonnumeric_values(
    required,
    validator,
    numeric_name,
    bad_value,
):
    payload = {name: 0.0 for name in required}
    if "opd/teacher_type" in payload:
        payload["opd/teacher_type"] = "ema"
        payload["algorithm/objective_mode"] = "auxiliary"
        payload["opd/loss_support"] = "latent_only"
    if "grad/diagnostic_space" in payload:
        payload["grad/diagnostic_space"] = "fixed_top5_action_logits"
    payload[numeric_name] = bad_value

    with pytest.raises((TypeError, ValueError), match="required W&B metric"):
        validator(payload)


@pytest.mark.parametrize(
    "name",
    [
        "opd/loss_support",
        "opd/teacher_type",
        "grad/diagnostic_space",
    ],
)
def test_study_metric_contract_requires_nonempty_string_fields(name):
    payload = {metric: 0.0 for metric in ITERATION_METRICS}
    payload["algorithm/objective_mode"] = "auxiliary"
    payload["opd/loss_support"] = "latent_only"
    payload["opd/teacher_type"] = "ema"
    payload["grad/diagnostic_space"] = "fixed_top5_action_logits"
    payload[name] = ""

    with pytest.raises(TypeError, match="nonempty string"):
        validate_iteration_metric_contract(payload)


def test_study_metric_contract_rejects_invalid_objective_mode():
    payload = {metric: 0.0 for metric in ITERATION_METRICS}
    payload["algorithm/objective_mode"] = ""
    payload["opd/loss_support"] = "latent_only"
    payload["opd/teacher_type"] = "ema"
    payload["grad/diagnostic_space"] = "fixed_top5_action_logits"

    with pytest.raises(ValueError, match="unknown OPD objective mode"):
        validate_iteration_metric_contract(payload)


def _complete_standalone_metric_payload():
    payload = {metric: 0.0 for metric in STANDALONE_ITERATION_METRICS}
    payload["algorithm/objective_mode"] = "standalone"
    payload["opd/loss_support"] = "all_response"
    payload["opd/teacher_type"] = "ema"
    payload["grad/diagnostic_space"] = "fixed_top5_action_logits"
    return payload


def test_standalone_required_metrics_omit_only_inapplicable_grpo_ppo_fields():
    assert required_iteration_metrics(ObjectiveMode.STANDALONE) is STANDALONE_ITERATION_METRICS
    assert STANDALONE_ITERATION_METRICS == (
        ITERATION_METRICS - STANDALONE_INAPPLICABLE_METRICS
    )
    assert {
        "loss/grpo",
        "loss/native_ref_kl",
        "ppo/ratio_mean",
        "ppo/ratio_p95",
        "ppo/clip_fraction",
        "ppo/approx_kl",
        "grad/grpo_norm",
        "grad/grpo_opd_cosine",
        "train/nonzero_advantage_group_fraction",
    } == STANDALONE_INAPPLICABLE_METRICS
    assert "replay/ratio_abs_error_max" in STANDALONE_ITERATION_METRICS
    assert "train/reward_mean" in STANDALONE_ITERATION_METRICS
    assert "loss/opd_kl_unweighted" in STANDALONE_ITERATION_METRICS


def test_standalone_metric_contract_accepts_exact_required_payload():
    validate_iteration_metric_contract(_complete_standalone_metric_payload())


@pytest.mark.parametrize("name", sorted(STANDALONE_INAPPLICABLE_METRICS))
def test_standalone_metric_contract_rejects_inapplicable_grpo_ppo_fields(name):
    payload = _complete_standalone_metric_payload()
    payload[name] = 0.0

    with pytest.raises(ValueError, match="must omit inapplicable PPO/GRPO metrics"):
        validate_iteration_metric_contract(payload)


def test_standalone_metric_contract_fails_closed_when_required_metric_is_missing():
    payload = _complete_standalone_metric_payload()
    payload.pop("replay/ratio_abs_error_max")

    with pytest.raises(ValueError, match="missing required W&B metrics"):
        validate_iteration_metric_contract(payload)
