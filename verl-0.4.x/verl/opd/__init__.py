"""Primitives for on-policy distillation (OPD) in SofT-GRPO.

The package is intentionally independent of the trainer and worker stacks.  This
keeps ``algorithm.opd.enabled=false`` and ``beta_base=0`` true no-op modes until
the integration layer explicitly opts into teacher construction and scoring.
"""

from .config import (
    KLDirection,
    LossSupport,
    ObjectiveMode,
    OPDConfig,
    OPDTeacherConfig,
    PromptTemplate,
    ScheduleType,
    TeacherType,
    TrajectoryGate,
)
from .ema import (
    EMAUpdateReport,
    EMAUpdateState,
    freeze_teacher_,
    parameter_squared_distance_sum_and_count,
    rms_from_squared_sum_and_count,
    update_ema_module_,
    update_ema_once_,
)
from .losses import full_vocab_kl
from .masks import (
    categorical_suffix_mask,
    ddp_scaled_local_loss,
    gated_latent_mask,
    latent_kl_sum_and_count,
    latent_mask_from_topk_support,
    opd_loss_support_mask,
    safe_mean_from_sum,
    trajectory_gate_mask,
)
from .metrics import (
    FORBIDDEN_METRICS,
    ITERATION_METRICS,
    OPTIMIZER_STEP_METRIC,
    PRIMARY_STEP_METRIC,
    STANDALONE_INAPPLICABLE_METRICS,
    STANDALONE_ITERATION_METRICS,
    VALIDATION_METRICS,
    opd_schedule_metrics,
    required_iteration_metrics,
    validate_metric_payload,
    validate_resource_limits,
    validation_pass_at_1_aliases,
)
from .prompts import render_privileged_prompt, render_sdft_prompt, render_sdpg_prompt
from .replay import OPDReplayResult, PrivilegedReplay, validate_reasoning_tokenizer
from .schedule import effective_beta, schedule_multiplier, warmup_iterations

__all__ = [
    "EMAUpdateReport",
    "EMAUpdateState",
    "FORBIDDEN_METRICS",
    "ITERATION_METRICS",
    "KLDirection",
    "LossSupport",
    "OPDConfig",
    "OPDTeacherConfig",
    "OPDReplayResult",
    "ObjectiveMode",
    "OPTIMIZER_STEP_METRIC",
    "PRIMARY_STEP_METRIC",
    "PromptTemplate",
    "PrivilegedReplay",
    "ScheduleType",
    "STANDALONE_INAPPLICABLE_METRICS",
    "STANDALONE_ITERATION_METRICS",
    "TeacherType",
    "TrajectoryGate",
    "VALIDATION_METRICS",
    "categorical_suffix_mask",
    "ddp_scaled_local_loss",
    "effective_beta",
    "freeze_teacher_",
    "full_vocab_kl",
    "gated_latent_mask",
    "latent_kl_sum_and_count",
    "latent_mask_from_topk_support",
    "opd_schedule_metrics",
    "opd_loss_support_mask",
    "parameter_squared_distance_sum_and_count",
    "render_privileged_prompt",
    "render_sdft_prompt",
    "render_sdpg_prompt",
    "required_iteration_metrics",
    "rms_from_squared_sum_and_count",
    "safe_mean_from_sum",
    "schedule_multiplier",
    "trajectory_gate_mask",
    "update_ema_module_",
    "update_ema_once_",
    "validate_metric_payload",
    "validation_pass_at_1_aliases",
    "validate_resource_limits",
    "validate_reasoning_tokenizer",
    "warmup_iterations",
]
