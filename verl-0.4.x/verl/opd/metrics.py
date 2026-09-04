"""Stable W&B metric-name contract for the seed-11 OPD study."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Union

from .config import ObjectiveMode, OPDConfig, ScheduleType
from .schedule import effective_beta, schedule_multiplier, warmup_iterations

PRIMARY_STEP_METRIC = "trainer/rollout_iteration"
OPTIMIZER_STEP_METRIC = "trainer/optimizer_step"

ITERATION_METRICS = frozenset(
    {
        PRIMARY_STEP_METRIC,
        OPTIMIZER_STEP_METRIC,
        "train/reward_mean",
        "train/reward_std",
        "train/correct_fraction",
        "train/nonzero_advantage_group_fraction",
        "loss/grpo",
        "loss/native_ref_kl",
        "loss/opd_kl_latent",
        "loss/opd_kl_answer",
        "loss/opd_kl_unweighted",
        "loss/opd_weighted",
        "loss/total",
        "algorithm/objective_mode",
        "opd/beta_base",
        "opd/loss_support",
        "opd/warmup_fraction",
        "opd/warmup_iterations",
        "opd/schedule_multiplier",
        "opd/beta_effective",
        "opd/teacher_type",
        "opd/teacher_student_param_rms",
        "opd/ema_update_count",
        "opd/latent_slot_count",
        "opd/answer_slot_count",
        "opd/selected_slot_fraction",
        "ppo/ratio_mean",
        "ppo/ratio_p95",
        "ppo/clip_fraction",
        "ppo/approx_kl",
        "grad/total_norm",
        "grad/grpo_norm",
        "grad/opd_norm",
        "grad/grpo_opd_cosine",
        "grad/diagnostic_space",
        "grad/clipped",
        "latent/length_mean",
        "latent/length_p95",
        "latent/hard_answer_length_mean",
        "latent/close_tag_rate",
        "latent/soft_to_hard_rate",
        "latent/cap_rate",
        "latent/mixture_entropy_mean",
        "latent/top1_weight_mean",
        "latent/soft_hard_agreement",
        "replay/ratio_abs_error_max",
        "replay/fallback_count",
        "perf/rollout_tokens_per_second",
        "perf/train_tokens_per_second",
        "perf/iteration_seconds",
        "perf/teacher_seconds",
        "system/hbm_peak_gib",
        "system/host_ram_gib",
        "system/cpu_utilization",
        "integrity/continuous_replay_active",
        "integrity/checkpoint_committed",
        "integrity/resumed",
    }
)

# Standalone OPD has no grouped reward optimization, frozen-reference anchor,
# advantages, or PPO ratio.  These fields must be absent rather than populated
# with synthetic zeroes.  Keep ``ITERATION_METRICS`` as the auxiliary-mode
# compatibility contract used by existing callers.
STANDALONE_INAPPLICABLE_METRICS = frozenset(
    {
        "train/nonzero_advantage_group_fraction",
        "loss/grpo",
        "loss/native_ref_kl",
        "ppo/ratio_mean",
        "ppo/ratio_p95",
        "ppo/clip_fraction",
        "ppo/approx_kl",
        "grad/grpo_norm",
        "grad/grpo_opd_cosine",
    }
)
STANDALONE_ITERATION_METRICS = ITERATION_METRICS - STANDALONE_INAPPLICABLE_METRICS

VALIDATION_METRICS = frozenset(
    {
        PRIMARY_STEP_METRIC,
        "val/math_verify/mean_at_1",
        "val/math_verify/pass_at_1",
        "val/released_reward/mean_at_1",
        "val/released_reward/pass_at_1",
        "val/response_length_mean",
        "val/latent_length_mean",
        "val/cap_rate",
        "val/soft_to_hard_rate",
    }
)

FORBIDDEN_METRICS = frozenset({"opd/kl_weight"})
STRING_METRICS = frozenset(
    {
        "algorithm/objective_mode",
        "opd/loss_support",
        "opd/teacher_type",
        "grad/diagnostic_space",
    }
)


def required_iteration_metrics(
    mode: Union[ObjectiveMode, str],
) -> frozenset[str]:
    """Return the fail-closed W&B contract for an objective mode."""

    try:
        objective_mode = ObjectiveMode(mode)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown OPD objective mode: {mode!r}") from exc
    if objective_mode is ObjectiveMode.STANDALONE:
        return STANDALONE_ITERATION_METRICS
    return ITERATION_METRICS


def validate_resource_limits(
    *,
    hbm_peak_gib: float,
    host_ram_percent: float,
    hbm_limit_gib: float = 72.0,
    host_ram_limit_percent: float = 90.0,
) -> None:
    """Fail a study worker as soon as its local resource ceiling is crossed.

    This check intentionally operates on each worker's local CUDA high-water
    mark before Ray aggregates metrics.  The driver separately logs the maximum
    rank because VERL reduces the source ``perf/max_*`` metric with ``MAX``.
    """

    values = {
        "HBM peak": float(hbm_peak_gib),
        "host RAM percent": float(host_ram_percent),
        "HBM limit": float(hbm_limit_gib),
        "host RAM limit": float(host_ram_limit_percent),
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise RuntimeError(f"resource-integrity metrics must be finite: {values}")
    if values["HBM peak"] >= values["HBM limit"]:
        raise RuntimeError(
            "rollout-integrity HBM gate failed: "
            f"{values['HBM peak']:.3f} GiB >= {values['HBM limit']:.3f} GiB"
        )
    if values["host RAM percent"] >= values["host RAM limit"]:
        raise RuntimeError(
            "rollout-integrity host-RAM gate failed: "
            f"{values['host RAM percent']:.3f}% >= {values['host RAM limit']:.3f}%"
        )


def opd_schedule_metrics(config: OPDConfig, rollout_iteration: int, total_iterations: int) -> dict[str, Any]:
    """Build the unambiguous schedule portion of an iteration payload."""

    multiplier = 0.0
    if config.active:
        multiplier = schedule_multiplier(
            rollout_iteration,
            total_iterations,
            schedule=config.schedule,
            warmup_fraction=config.warmup_fraction,
        )
    resolved_warmup = 0
    if config.schedule is not ScheduleType.CONSTANT:
        resolved_warmup = warmup_iterations(total_iterations, config.warmup_fraction)
    return {
        "algorithm/objective_mode": config.mode.value,
        "opd/beta_base": config.beta_base,
        "opd/loss_support": config.loss_support.value,
        "opd/warmup_fraction": config.warmup_fraction,
        "opd/warmup_iterations": resolved_warmup,
        "opd/schedule_multiplier": multiplier,
        "opd/beta_effective": effective_beta(config, rollout_iteration, total_iterations),
        "opd/teacher_type": config.teacher.type.value,
    }


def validation_pass_at_1_aliases(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return pass@1 aliases for canonical one-sample validation means.

    With exactly one response per validation example, mean@1 and pass@1 are
    the same estimator.  This helper does not trigger another generation; it
    copies each available canonical mean and rejects contradictory preexisting
    aliases.
    """

    aliases: dict[str, Any] = {}
    for stem in ("val/math_verify", "val/released_reward"):
        mean_name = f"{stem}/mean_at_1"
        pass_name = f"{stem}/pass_at_1"
        if mean_name not in metrics:
            continue
        mean_value = metrics[mean_name]
        if pass_name in metrics and float(metrics[pass_name]) != float(mean_value):
            raise ValueError(
                f"validation pass@1 alias {pass_name!r} disagrees with {mean_name!r}"
            )
        aliases[pass_name] = mean_value
    return aliases


def validate_metric_payload(
    metrics: Mapping[str, Any],
    required: Iterable[str] = (),
) -> None:
    """Reject ambiguous names and report missing required contract fields.

    Additional upstream metrics are accepted so this contract can coexist with
    released verl logging.
    """

    names: set[str] = set(metrics)
    forbidden = names & FORBIDDEN_METRICS
    if forbidden:
        raise ValueError(f"forbidden ambiguous W&B metrics: {sorted(forbidden)}")
    missing = set(required) - names
    if missing:
        raise ValueError(f"missing required W&B metrics: {sorted(missing)}")

    for name in set(required):
        value = metrics[name]
        if name in STRING_METRICS:
            if not isinstance(value, str) or not value:
                raise TypeError(f"required W&B metric {name!r} must be a nonempty string")
            continue
        if isinstance(value, (bool, str, bytes)):
            raise TypeError(f"required W&B metric {name!r} must be a finite number")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(
                f"required W&B metric {name!r} must be a finite number"
            ) from exc
        if not math.isfinite(numeric_value):
            raise ValueError(f"required W&B metric {name!r} must be finite")
