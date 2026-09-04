"""Driver-side OPD scheduling, diagnostics, and metric aliases.

The model-side OPD implementation lives in :mod:`verl.opd` and the actor
workers.  This module deliberately contains only computations that can be made
from a completed rollout on the Ray driver.  Keeping these calculations here
makes the smoke/production integrity gate independent of worker log messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch

from verl.opd import (
    ITERATION_METRICS,
    VALIDATION_METRICS,
    ObjectiveMode,
    OPDConfig,
    opd_schedule_metrics,
    required_iteration_metrics,
    validate_metric_payload,
    validation_pass_at_1_aliases,
)


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def training_rollout_iteration(global_step: int) -> int:
    """Map VERL's one-based training step to the study's zero-based axis."""

    step = _nonnegative_int(global_step, "global_step")
    if step == 0:
        raise ValueError("a training event requires global_step >= 1")
    return step - 1


def validation_rollout_iteration(completed_rollouts: int) -> int:
    """Label validation by completed rollouts: 0, 25, ..., 109."""

    return _nonnegative_int(completed_rollouts, "completed_rollouts")


def training_wandb_step(global_step: int) -> int:
    """Reserve an ordered W&B event step for one training row."""

    step = _nonnegative_int(global_step, "global_step")
    if step == 0:
        raise ValueError("a training event requires global_step >= 1")
    return 2 * step


def validation_wandb_step(completed_rollouts: int) -> int:
    """Place validation after the corresponding training event in W&B."""

    return 2 * _nonnegative_int(completed_rollouts, "completed_rollouts") + 1


@dataclass(frozen=True)
class RolloutIntegrityConfig:
    """Fail-closed checks for SofT-GRPO's continuous replay contract."""

    enabled: bool = False
    gate_first_n_iterations: int = 0
    max_cap_rate: float = 0.05
    max_all_soft_rate: float = 0.05
    min_close_tag_rate: float = 0.95
    min_soft_to_hard_rate: float = 0.95
    min_categorical_boxed_answer_rate: float = 0.95
    max_replay_ratio_abs_error: float = 1e-4
    full_dose_gradient_gate_enabled: bool = False
    min_opd_grpo_support_gradient_ratio: float = 0.1
    max_opd_grpo_support_gradient_ratio: float = 10.0
    max_full_dose_gradient_clip_fraction: float = 0.5

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> RolloutIntegrityConfig:
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown rollout-integrity fields: {sorted(unknown)}")
        result = cls(**dict(values))
        if type(result.enabled) is not bool:
            raise TypeError("rollout_integrity.enabled must be bool")
        if type(result.full_dose_gradient_gate_enabled) is not bool:
            raise TypeError("full_dose_gradient_gate_enabled must be bool")
        if isinstance(result.gate_first_n_iterations, bool) or not isinstance(result.gate_first_n_iterations, int):
            raise TypeError("gate_first_n_iterations must be an integer")
        if result.gate_first_n_iterations < 0:
            raise ValueError("gate_first_n_iterations must be nonnegative")
        for name in (
            "max_cap_rate",
            "max_all_soft_rate",
            "min_close_tag_rate",
            "min_soft_to_hard_rate",
            "min_categorical_boxed_answer_rate",
        ):
            value = float(getattr(result, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not np.isfinite(result.max_replay_ratio_abs_error) or result.max_replay_ratio_abs_error < 0:
            raise ValueError("max_replay_ratio_abs_error must be finite and nonnegative")
        minimum_ratio = float(result.min_opd_grpo_support_gradient_ratio)
        maximum_ratio = float(result.max_opd_grpo_support_gradient_ratio)
        if not np.isfinite(minimum_ratio) or minimum_ratio <= 0.0:
            raise ValueError(
                "min_opd_grpo_support_gradient_ratio must be finite and positive"
            )
        if not np.isfinite(maximum_ratio) or maximum_ratio < minimum_ratio:
            raise ValueError(
                "max_opd_grpo_support_gradient_ratio must be finite and at least the minimum"
            )
        maximum_clip_fraction = float(result.max_full_dose_gradient_clip_fraction)
        if not np.isfinite(maximum_clip_fraction) or not 0.0 <= maximum_clip_fraction <= 1.0:
            raise ValueError(
                "max_full_dose_gradient_clip_fraction must be in [0, 1]"
            )
        return result


@dataclass(frozen=True)
class RolloutDiagnostics:
    """Public W&B values plus private values used by the integrity gate."""

    metrics: Mapping[str, float]
    all_soft_rate: float
    categorical_boxed_answer_rate: float
    boundary_valid_mask: tuple[bool, ...]


def _mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return values.float().mean().item()


def _p95(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return torch.quantile(values.float(), 0.95).item()


def compute_rollout_diagnostics(
    *,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_topk_ids: torch.Tensor,
    rollout_topk_gumbels: torch.Tensor,
    gumbel_temperature: float,
    close_tag_token_id: int,
    decode: Optional[Callable[[Sequence[int]], str]] = None,
) -> RolloutDiagnostics:
    """Compute latent-boundary diagnostics from released rollout metadata.

    ``rollout_topk_ids`` and ``rollout_topk_gumbels`` must cover response
    positions only.  Released SofT-GRPO encodes an ordinary categorical token
    as ``[token_id, 0, 0, ...]`` and a continuous action with its full fixed
    top-k support.  No inferred or categorical fallback is accepted here.
    """

    if responses.ndim != 2 or response_mask.shape != responses.shape:
        raise ValueError("responses and response_mask must be matching rank-2 tensors")
    if rollout_topk_ids.ndim != 3:
        raise ValueError("rollout_topk_ids must have shape [batch, response, support]")
    expected_support_shape = responses.shape + (rollout_topk_ids.shape[-1],)
    if rollout_topk_ids.shape != expected_support_shape:
        raise ValueError("rollout_topk_ids must have shape [batch, response, support]")
    if rollout_topk_gumbels.shape != rollout_topk_ids.shape:
        raise ValueError("rollout_topk_gumbels must match rollout_topk_ids")
    if rollout_topk_ids.shape[-1] < 2:
        raise ValueError("continuous replay requires support size >= 2")
    if not rollout_topk_gumbels.is_floating_point():
        raise TypeError("rollout_topk_gumbels must be floating point")
    temperature = float(gumbel_temperature)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("gumbel_temperature must be finite and positive")

    valid = response_mask.bool()
    categorical = (rollout_topk_ids[..., 1:] == 0).all(dim=-1)
    latent = valid & ~categorical
    hard = valid & categorical
    valid_lengths = valid.sum(dim=-1)
    latent_lengths = latent.sum(dim=-1)
    hard_lengths = hard.sum(dim=-1)

    # A trajectory transitions only if a categorical position occurs after a
    # valid continuous position; a hard prompt prefix is not part of this view.
    seen_latent = latent.to(torch.int64).cumsum(dim=-1) > 0
    transitioned = (hard & seen_latent).any(dim=-1)
    close_positions = (responses == int(close_tag_token_id)) & valid
    has_close = close_positions.any(dim=-1)
    seen_close = close_positions.to(torch.int64).cumsum(dim=-1) > 0

    # The released replay contract is continuous before ``</think>`` and
    # categorical from the close tag onward.  Count violations from the actual
    # returned action metadata instead of publishing a synthetic constant zero.
    # A support-head mismatch also means the stored action cannot replay the
    # emitted hard shadow.  Non-finite continuous actions are counted here and
    # will additionally fail the finite-metric check below.
    mode_mismatch = valid & categorical.ne(seen_close)
    support_head_mismatch = valid & rollout_topk_ids[..., 0].ne(responses)
    nonfinite_continuous = latent & ~torch.isfinite(rollout_topk_gumbels).all(dim=-1)
    replay_fallback = mode_mismatch | support_head_mismatch | nonfinite_continuous
    capped = valid.all(dim=-1)
    all_soft = (valid_lengths > 0) & (latent_lengths == valid_lengths)

    if latent.any():
        soft_weights = torch.softmax(rollout_topk_gumbels[latent].float() / temperature, dim=-1)
        mixture_entropy = -(soft_weights * soft_weights.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(dim=-1)
        top1_weight = soft_weights.max(dim=-1).values
        hard_shadow = rollout_topk_ids[latent].gather(
            dim=-1,
            index=soft_weights.argmax(dim=-1, keepdim=True),
        ).squeeze(-1)
        agreement = hard_shadow.eq(responses[latent])
    else:
        mixture_entropy = torch.empty(0)
        top1_weight = torch.empty(0)
        agreement = torch.empty(0, dtype=torch.bool)

    boxed_flags = []
    if decode is not None:
        response_cpu = responses.detach().cpu()
        hard_cpu = hard.detach().cpu()
        for row_ids, row_hard in zip(response_cpu, hard_cpu):
            hard_text = decode(row_ids[row_hard].tolist())
            boxed_flags.append(r"\boxed{" in hard_text)
    boxed_rate = float(np.mean(boxed_flags)) if boxed_flags else 0.0
    categorical_boxed = torch.tensor(
        boxed_flags,
        dtype=torch.bool,
        device=responses.device,
    ) if boxed_flags else torch.zeros(responses.shape[0], dtype=torch.bool, device=responses.device)
    boundary_valid = (
        has_close
        & transitioned
        & latent_lengths.gt(0)
        & hard_lengths.gt(1)
        & categorical_boxed
        & ~replay_fallback.any(dim=-1)
    )

    metrics = {
        "latent/length_mean": _mean(latent_lengths),
        "latent/length_p95": _p95(latent_lengths),
        "latent/hard_answer_length_mean": _mean(hard_lengths),
        "latent/close_tag_rate": _mean(has_close),
        "latent/soft_to_hard_rate": _mean(transitioned),
        "latent/cap_rate": _mean(capped),
        "latent/mixture_entropy_mean": _mean(mixture_entropy),
        "latent/top1_weight_mean": _mean(top1_weight),
        "latent/soft_hard_agreement": _mean(agreement),
        "integrity/continuous_replay_active": float(latent.any().item()),
        "replay/fallback_count": float(replay_fallback.sum().item()),
    }
    return RolloutDiagnostics(
        metrics=metrics,
        all_soft_rate=_mean(all_soft),
        categorical_boxed_answer_rate=boxed_rate,
        boundary_valid_mask=tuple(bool(value) for value in boundary_valid.detach().cpu().tolist()),
    )


def compute_categorical_rollout_diagnostics(
    *,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    close_tag_token_id: int,
    decode: Optional[Callable[[Sequence[int]], str]] = None,
) -> RolloutDiagnostics:
    """Return the common rollout fields for a genuinely categorical arm."""

    if responses.ndim != 2 or response_mask.shape != responses.shape:
        raise ValueError("responses and response_mask must be matching rank-2 tensors")
    valid = response_mask.bool()
    close = valid & responses.eq(int(close_tag_token_id))
    seen_close = close.to(torch.int64).cumsum(dim=-1) > 0
    after_close = torch.zeros_like(seen_close)
    if after_close.shape[-1] > 1:
        after_close[:, 1:] = seen_close[:, :-1]
    answer_lengths = (valid & after_close).sum(dim=-1)
    boxed_flags = []
    if decode is not None:
        for row_ids, row_answer in zip(responses.detach().cpu(), (valid & after_close).detach().cpu()):
            boxed_flags.append(r"\boxed{" in decode(row_ids[row_answer].tolist()))
    metrics = {
        "latent/length_mean": 0.0,
        "latent/length_p95": 0.0,
        "latent/hard_answer_length_mean": _mean(answer_lengths),
        "latent/close_tag_rate": _mean(close.any(dim=-1)),
        "latent/soft_to_hard_rate": 0.0,
        "latent/cap_rate": _mean(valid.all(dim=-1)),
        "latent/mixture_entropy_mean": 0.0,
        "latent/top1_weight_mean": 0.0,
        "latent/soft_hard_agreement": 0.0,
        "integrity/continuous_replay_active": 0.0,
        "replay/fallback_count": 0.0,
    }
    return RolloutDiagnostics(
        metrics=metrics,
        all_soft_rate=0.0,
        categorical_boxed_answer_rate=(
            float(np.mean(boxed_flags)) if boxed_flags else 0.0
        ),
        # Boundary gating applies only to native-soft trajectories.  A genuine
        # categorical arm is graded from its response text directly.
        boundary_valid_mask=tuple(True for _ in range(responses.shape[0])),
    )


_BOUNDARY_GRADED_EXTRA_INFO_KEYS = frozenset(
    {"score", "released_reward", "math_verify"}
)


def mask_invalid_native_boundary_scores(
    *,
    reward_tensor: torch.Tensor,
    reward_extra_info: Mapping[str, Sequence[Any]],
    boundary_valid_mask: Sequence[bool],
) -> tuple[torch.Tensor, dict[str, list[Any]]]:
    """Force invalid native-soft boundaries to receive zero validation credit.

    A decoded hard shadow is not an admissible native-soft completion unless
    the stored action metadata demonstrates a continuous prefix followed by a
    categorical ``</think>``/boxed-answer suffix.  Mask both the scalar reward
    tensor and the correctness fields used to select BEST checkpoints.  Other
    auxiliary metadata is copied unchanged.
    """

    if reward_tensor.ndim != 2:
        raise ValueError("reward_tensor must have shape [batch, response]")
    mask = torch.as_tensor(
        tuple(boundary_valid_mask),
        dtype=torch.bool,
        device=reward_tensor.device,
    )
    if mask.ndim != 1 or mask.shape[0] != reward_tensor.shape[0]:
        raise ValueError("boundary_valid_mask must contain one value per response")

    masked_reward = reward_tensor.clone()
    masked_reward[~mask] = 0.0
    mask_cpu = mask.detach().cpu().tolist()
    masked_extra: dict[str, list[Any]] = {}
    for key, raw_values in reward_extra_info.items():
        values = list(raw_values)
        if len(values) != reward_tensor.shape[0]:
            raise ValueError(
                f"reward_extra_info[{key!r}] must contain one value per response"
            )
        if key in _BOUNDARY_GRADED_EXTRA_INFO_KEYS:
            values = [value if valid else 0.0 for value, valid in zip(values, mask_cpu)]
        masked_extra[str(key)] = values
    return masked_reward, masked_extra


def replay_ratio_abs_error_max(
    rollout_log_probs: torch.Tensor,
    actor_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> float:
    """Return ``max |exp(log p_actor - log p_rollout) - 1|`` on valid slots."""

    if rollout_log_probs.shape != actor_log_probs.shape or response_mask.shape != actor_log_probs.shape:
        raise ValueError("replay log probabilities and mask must have identical shapes")
    valid = response_mask.bool()
    if not valid.any():
        raise ValueError("cannot validate replay without valid response positions")
    log_ratio = actor_log_probs[valid].float() - rollout_log_probs[valid].float()
    errors = (log_ratio.exp() - 1.0).abs()
    return errors.max().item()


def reward_and_group_metrics(
    sequence_scores: torch.Tensor,
    trajectory_advantages: torch.Tensor,
    group_ids: Sequence[Any],
) -> Mapping[str, float]:
    """Compute reward statistics and the fraction of informative GRPO groups."""

    if sequence_scores.ndim != 1 or trajectory_advantages.shape != sequence_scores.shape:
        raise ValueError("scores and trajectory advantages must be matching vectors")
    if len(group_ids) != sequence_scores.numel():
        raise ValueError("one group ID is required per trajectory")
    informative: dict[Any, bool] = {}
    for group_id, advantage in zip(group_ids, trajectory_advantages.detach().cpu().tolist()):
        informative[group_id] = informative.get(group_id, False) or abs(float(advantage)) > 1e-12
    informative_fraction = float(np.mean(list(informative.values()))) if informative else 0.0
    scores = sequence_scores.float()
    return {
        "train/reward_mean": scores.mean().item(),
        "train/reward_std": scores.std(unbiased=False).item(),
        "train/correct_fraction": (scores > 0.5).float().mean().item(),
        "train/nonzero_advantage_group_fraction": informative_fraction,
    }


def reward_only_metrics(sequence_scores: torch.Tensor) -> Mapping[str, float]:
    """Reward monitoring for standalone OPD without inventing GRPO fields."""

    if sequence_scores.ndim != 1 or sequence_scores.numel() == 0:
        raise ValueError("standalone reward scores must be a nonempty vector")
    scores = sequence_scores.float()
    if not torch.isfinite(scores).all():
        raise FloatingPointError("standalone reward scores must be finite")
    return {
        "train/reward_mean": scores.mean().item(),
        "train/reward_std": scores.std(unbiased=False).item(),
        "train/correct_fraction": (scores > 0.5).float().mean().item(),
    }


def schedule_meta_info(opd_config: OPDConfig, rollout_iteration: int, total_iterations: int) -> Mapping[str, Any]:
    """Worker metadata for one rollout iteration, without ambiguous aliases."""

    schedule = opd_schedule_metrics(opd_config, rollout_iteration, total_iterations)
    return {
        "opd_config": {
            "enabled": opd_config.enabled,
            "beta_base": opd_config.beta_base,
            "mode": str(opd_config.mode),
            "loss_support": str(opd_config.loss_support),
            "schedule": str(opd_config.schedule),
            "warmup_fraction": opd_config.warmup_fraction,
            "teacher": {
                "type": str(opd_config.teacher.type),
                "ema_decay": opd_config.teacher.ema_decay,
            },
            "kl_direction": str(opd_config.kl_direction),
            "trajectory_gate": str(opd_config.trajectory_gate),
            "prompt_template": str(opd_config.prompt_template),
            "temperature": opd_config.temperature,
        },
        "opd_rollout_iteration": rollout_iteration,
        "opd_total_iterations": total_iterations,
        "opd_warmup_iterations": schedule["opd/warmup_iterations"],
        "opd_schedule_multiplier": schedule["opd/schedule_multiplier"],
        "opd_beta_effective": schedule["opd/beta_effective"],
    }


def validate_rollout_integrity(
    diagnostics: RolloutDiagnostics,
    replay_error: float | None,
    config: RolloutIntegrityConfig,
    rollout_iteration: int,
) -> None:
    """Raise before an optimizer update when continuous replay is not genuine."""

    if not config.enabled:
        return
    values = diagnostics.metrics
    finite_values = {
        **{name: float(value) for name, value in values.items()},
        "all-soft rate": float(diagnostics.all_soft_rate),
        "categorical boxed-answer rate": float(diagnostics.categorical_boxed_answer_rate),
    }
    nonfinite = sorted(name for name, value in finite_values.items() if not np.isfinite(value))
    if nonfinite:
        raise RuntimeError(f"rollout integrity metrics are non-finite: {nonfinite}")
    if values["integrity/continuous_replay_active"] != 1.0:
        raise RuntimeError("continuous SofT-GRPO replay is not active")
    if values["replay/fallback_count"] != 0.0:
        raise RuntimeError("categorical replay fallback was detected")
    if replay_error is not None and (
        not np.isfinite(replay_error)
        or replay_error > config.max_replay_ratio_abs_error
    ):
        raise RuntimeError(
            f"rollout/replay ratio error {replay_error:.6g} exceeds "
            f"{config.max_replay_ratio_abs_error:.6g}"
        )
    if rollout_iteration >= config.gate_first_n_iterations:
        return

    checks = {
        "cap rate": (values["latent/cap_rate"], config.max_cap_rate, "max"),
        "all-soft rate": (diagnostics.all_soft_rate, config.max_all_soft_rate, "max"),
        "close-tag rate": (values["latent/close_tag_rate"], config.min_close_tag_rate, "min"),
        "soft-to-hard rate": (values["latent/soft_to_hard_rate"], config.min_soft_to_hard_rate, "min"),
        "categorical boxed-answer rate": (
            diagnostics.categorical_boxed_answer_rate,
            config.min_categorical_boxed_answer_rate,
            "min",
        ),
    }
    failures = []
    for name, (actual, threshold, kind) in checks.items():
        failed = actual > threshold if kind == "max" else actual < threshold
        if failed:
            comparator = "<=" if kind == "max" else ">="
            failures.append(f"{name}={actual:.6g} (required {comparator}{threshold:.6g})")
    if failures:
        raise RuntimeError("rollout integrity gate failed: " + "; ".join(failures))


def validate_categorical_rollout_integrity(
    replay_error: float,
    config: RolloutIntegrityConfig,
) -> None:
    """Fail closed on categorical actor/rollout drift without soft metadata."""

    if not config.enabled:
        return
    if not np.isfinite(replay_error) or replay_error > config.max_replay_ratio_abs_error:
        raise RuntimeError(
            f"categorical rollout/replay ratio error {replay_error:.6g} exceeds "
            f"{config.max_replay_ratio_abs_error:.6g}"
        )


def validate_full_dose_gradient_integrity(
    metrics: Mapping[str, Any],
    config: RolloutIntegrityConfig,
    *,
    schedule_multiplier: float,
) -> None:
    """Fail closed on pathological hybrid gradients at a full OPD dose.

    The fixed-support component gradients are available only after the actor
    update, so this gate deliberately runs after both optimizer steps have
    completed.  Warm-up iterations are not interpreted as failures: the gate
    becomes active only at the exact full-dose multiplier of one.
    """

    if not config.enabled or not config.full_dose_gradient_gate_enabled:
        return
    multiplier = float(schedule_multiplier)
    if not np.isfinite(multiplier) or not 0.0 <= multiplier <= 1.0:
        raise RuntimeError(
            "full-dose gradient gate received an invalid OPD schedule multiplier: "
            f"{multiplier!r}"
        )
    if multiplier < 1.0:
        return

    required = (
        "grad/grpo_norm",
        "grad/opd_norm",
        "actor/gradient_clipfrac",
    )
    missing = [name for name in required if name not in metrics]
    if missing:
        raise RuntimeError(
            "full-dose gradient gate is missing actor diagnostics: "
            f"{missing}"
        )
    grpo_norm = float(metrics["grad/grpo_norm"])
    opd_norm = float(metrics["grad/opd_norm"])
    clip_fraction = float(metrics["actor/gradient_clipfrac"])
    diagnostic_values = {
        "GRPO support-gradient norm": grpo_norm,
        "OPD support-gradient norm": opd_norm,
        "gradient clip fraction": clip_fraction,
    }
    nonfinite = sorted(
        name for name, value in diagnostic_values.items() if not np.isfinite(value)
    )
    if nonfinite:
        raise RuntimeError(
            "full-dose gradient diagnostics are non-finite: " f"{nonfinite}"
        )
    if grpo_norm <= 0.0:
        raise RuntimeError(
            "full-dose gradient gate requires a positive GRPO support-gradient norm"
        )
    if opd_norm < 0.0:
        raise RuntimeError(
            "full-dose gradient gate requires a nonnegative OPD support-gradient norm"
        )

    ratio = opd_norm / grpo_norm
    failures = []
    if not (
        config.min_opd_grpo_support_gradient_ratio
        <= ratio
        <= config.max_opd_grpo_support_gradient_ratio
    ):
        failures.append(
            "OPD/GRPO support-gradient ratio="
            f"{ratio:.6g} (required "
            f"[{config.min_opd_grpo_support_gradient_ratio:.6g}, "
            f"{config.max_opd_grpo_support_gradient_ratio:.6g}])"
        )
    if not 0.0 <= clip_fraction <= config.max_full_dose_gradient_clip_fraction:
        failures.append(
            f"gradient clip fraction={clip_fraction:.6g} (required "
            f"[0, {config.max_full_dose_gradient_clip_fraction:.6g}])"
        )
    if failures:
        raise RuntimeError(
            "full-dose gradient integrity gate failed: " + "; ".join(failures)
        )


def add_canonical_metric_aliases(
    metrics: Mapping[str, Any],
    *,
    opd_config: OPDConfig,
    rollout_iteration: int,
    total_iterations: int,
    optimizer_step: int,
    grad_clip: float,
    checkpoint_committed: bool,
    resumed: bool,
) -> dict[str, Any]:
    """Add stable study names while retaining every released verl metric."""

    result = dict(metrics)
    result.update(opd_schedule_metrics(opd_config, rollout_iteration, total_iterations))
    result["trainer/rollout_iteration"] = rollout_iteration
    result["trainer/optimizer_step"] = optimizer_step
    result["integrity/checkpoint_committed"] = float(checkpoint_committed)
    result["integrity/resumed"] = float(resumed)
    # Component-gradient geometry is measured exactly at the policy's stored
    # fixed-top-five action logits.  The separately logged total norm is the
    # optimizer-facing parameter-gradient norm.
    result["grad/diagnostic_space"] = (
        "fixed_top5_action_logits"
        if float(result.get("integrity/continuous_replay_active", 0.0)) == 1.0
        else "not_applicable_categorical"
    )

    aliases = {
        "loss/grpo": "actor/pg_loss",
        "loss/native_ref_kl": "actor/kl_loss",
        "loss/opd_kl_unweighted": "actor/opd_kl_unweighted",
        "loss/opd_kl_latent": "actor/opd_kl_latent",
        "loss/opd_kl_answer": "actor/opd_kl_answer",
        "loss/opd_weighted": "actor/opd_weighted",
        "loss/total": "actor/total_loss",
        "opd/teacher_student_param_rms": "actor/opd_teacher_student_param_rms",
        "opd/ema_update_count": "actor/opd_ema_update_count",
        "ppo/ratio_mean": "actor/ratio_mean",
        "ppo/ratio_p95": "actor/ratio_p95",
        "ppo/clip_fraction": "actor/pg_clipfrac",
        "ppo/approx_kl": "actor/ppo_kl",
        "grad/total_norm": "actor/grad_norm",
        "grad/grpo_norm": "actor/grpo_grad_norm",
        "grad/opd_norm": "actor/opd_grad_norm",
        "grad/grpo_opd_cosine": "actor/grpo_opd_grad_cosine",
        "perf/teacher_seconds": "actor/opd_teacher_seconds",
        "system/hbm_peak_gib": "perf/max_memory_allocated_gb",
        "system/host_ram_gib": "perf/cpu_memory_used_gb",
    }
    for canonical, released in aliases.items():
        if canonical not in result and released in result:
            result[canonical] = result[released]

    # The actor reports one exact clipping indicator per optimizer step and Ray
    # reduces those indicators to their mean across steps/ranks.  Any positive
    # fraction therefore means at least one update was clipped.  Do not infer
    # this from the mean gradient norm: a clipped and an unclipped update can
    # average below the threshold.
    if "actor/gradient_clipfrac" in result:
        gradient_clip_fraction = float(result["actor/gradient_clipfrac"])
        if not np.isfinite(gradient_clip_fraction) or not 0.0 <= gradient_clip_fraction <= 1.0:
            raise ValueError(
                "actor/gradient_clipfrac must be a finite fraction in [0, 1]"
            )
        result["grad/clipped"] = gradient_clip_fraction

    # Inactive OPD is a strict no-op.  Explicit zero values make baseline W&B
    # runs directly comparable without pretending that a teacher was evaluated.
    if not opd_config.active:
        result.setdefault("loss/opd_kl_unweighted", 0.0)
        result.setdefault("loss/opd_kl_latent", 0.0)
        result.setdefault("loss/opd_kl_answer", 0.0)
        result.setdefault("loss/opd_weighted", 0.0)
        result.setdefault("opd/teacher_student_param_rms", 0.0)
        result.setdefault("opd/ema_update_count", 0.0)
        result.setdefault("grad/opd_norm", 0.0)
        result.setdefault("grad/grpo_opd_cosine", 0.0)
        result.setdefault("perf/teacher_seconds", 0.0)
        result.setdefault("opd/latent_slot_count", 0.0)
        result.setdefault("opd/answer_slot_count", 0.0)
        result.setdefault("opd/selected_slot_fraction", 0.0)

    if "loss/total" not in result and "loss/grpo" in result:
        total = float(result["loss/grpo"])
        if "loss/native_ref_kl" in result:
            total += float(result["loss/native_ref_kl"]) * 0.001
        total += float(result.get("loss/opd_weighted", 0.0))
        result["loss/total"] = total

    validate_metric_payload(result)
    return result


def validate_iteration_metric_contract(metrics: Mapping[str, Any]) -> None:
    """Require every per-rollout W&B field promised by the OPD study.

    This is kept separate from :func:`add_canonical_metric_aliases` so the
    upstream-compatible helper remains useful to non-study configurations.
    The trainer invokes this fail-closed check whenever rollout-integrity mode
    is enabled by the seed-11 launchers.
    """

    mode = metrics.get("algorithm/objective_mode", ObjectiveMode.AUXILIARY.value)
    required = required_iteration_metrics(mode)
    validate_metric_payload(metrics, required=required)
    if ObjectiveMode(mode) is ObjectiveMode.STANDALONE:
        forbidden = set(ITERATION_METRICS) - set(required)
        present = sorted(forbidden & set(metrics))
        if present:
            raise ValueError(
                "standalone OPD must omit inapplicable PPO/GRPO metrics: "
                f"{present}"
            )


def validate_validation_metric_contract(metrics: Mapping[str, Any]) -> None:
    """Require every validation W&B field promised by the OPD study."""

    validate_metric_payload(metrics, required=VALIDATION_METRICS)


def validation_metric_aliases(metrics: Mapping[str, Any]) -> Mapping[str, float]:
    """Expose unqualified single-validation-set Mean@1 aliases."""

    released = []
    math_verify = []
    for name, value in metrics.items():
        if not name.startswith(("val-core/", "val-aux/")) or not name.endswith("/mean@1"):
            continue
        if "/reward/" in name:
            released.append(float(value))
        if any(marker in name for marker in ("/math_verify/", "/math-verify/", "/mathverify/")):
            math_verify.append(float(value))
    aliases = {}
    if released:
        aliases["val/released_reward/mean_at_1"] = float(np.mean(released))
    if math_verify:
        aliases["val/math_verify/mean_at_1"] = float(np.mean(math_verify))
    aliases.update(validation_pass_at_1_aliases({**metrics, **aliases}))
    return aliases
