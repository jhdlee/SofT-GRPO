"""Opt-in production evidence around the ordinary RayPPOTrainer algorithm."""

from __future__ import annotations

import os
import math
from pathlib import Path
import time

from omegaconf import OmegaConf

from opd_tools.manifest import file_sha256
from opd_tools.training_capacity import CapacityRecorder, finite_snapshot


PRODUCTION_PROFILE = "qwen3-math-seven-arm-v1"
PRODUCTION_PROFILES = (PRODUCTION_PROFILE, "qwen3-math-seven-arm-lora-fa3-v1")
PRODUCTION_PHASES = ("production", "uninterrupted", "split", "resume", "full_dose", "zero_dose")


def remaining_runtime_estimate(measurement, *, gpus):
    """Project remaining critical-path work from this invocation's observations."""
    completed = int(measurement["completed_rollout_iterations"])
    remaining = max(0, 109 - completed)
    scheduled = sum(step > completed for step in (25, 50, 75, 100, 109))
    scheduled_checkpoints = scheduled + int(completed < 1 and measurement.get("configuration", {}).get(
        "trainer", {}).get("production_save_first_iteration") is True)
    active, zero_dose, saves = [], [], []
    for row in measurement.get("iterations", []):
        timing = row.get("timing_s", {})
        if "step" not in timing:
            continue
        values = [timing.get(key, 0.0) for key in ("step", "testing", "save_checkpoint", "gen", "old_log_prob", "ref", "update_actor")]
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values):
            continue
        step, validation, save, *tokens = values
        core = max(0.0, step - validation - save)
        token_seconds = min(core, sum(tokens))
        observation = {"core": core, "token": token_seconds, "fixed": max(0.0, core - token_seconds)}
        # Hard/soft GRPO have no OPD dose and use the active cost throughout.
        config = measurement.get("configuration", {}).get("algorithm", {}).get("opd", {})
        hybrid = bool(config.get("enabled")) and config.get("mode") == "auxiliary"
        target = zero_dose if hybrid and row["rollout_iteration"] == 0 else active
        target.append(observation)
        if save > 0:
            saves.append(save)
    validations = [row["timing_seconds"] for row in measurement.get("validations", [])
                   if isinstance(row.get("timing_seconds"), (int, float)) and math.isfinite(row["timing_seconds"]) and row["timing_seconds"] > 0]
    mean = lambda values: sum(values) / len(values) if values else None
    missing = []
    if remaining and not active:
        missing.append("active_iteration")
    if scheduled and not validations:
        missing.append("validation")
    if scheduled_checkpoints and not saves:
        missing.append("checkpoint")
    scenarios = {}
    for name, multiplier in (("central", 1.0), ("token_1p5x", 1.5), ("token_2x", 2.0)):
        iteration_cost = mean([row["fixed"] + multiplier * row["token"] for row in active])
        known = (remaining * (iteration_cost or 0) + scheduled * (mean(validations) or 0) * multiplier
                 + scheduled_checkpoints * (mean(saves) or 0))
        scenarios[name] = {"token_stage_multiplier": multiplier, "known_remaining_seconds": known,
                           "remaining_seconds": None if missing else known,
                           "remaining_gpu_hours": None if missing else known * gpus / 3600}
    return {
        "preliminary": True, "complete": not missing, "gpus": gpus,
        "remaining_rollout_iterations": remaining, "remaining_scheduled_validations": scheduled,
        "remaining_scheduled_checkpoints": scheduled_checkpoints, "missing_stage_measurements": missing,
        "active_iteration_samples": len(active), "zero_dose_iteration_samples": len(zero_dose),
        "active_core_mean_seconds": mean([row["core"] for row in active]),
        "zero_dose_core_mean_seconds": mean([row["core"] for row in zero_dose]),
        "validation_mean_seconds": mean(validations), "checkpoint_mean_seconds": mean(saves),
        "scenarios": scenarios,
        "scope": "Remaining work from this invocation's wall-clock observations; startup already incurred. Teacher time is included only inside update_actor. Token-stage and validation growth scenarios are planning assumptions, not confidence intervals; unknown overhead is incomplete.",
    }


class ProductionRecorder:
    """Instrument one invocation without replacing rollout/update/checkpoint code."""

    def __init__(self, trainer):
        self.trainer = trainer
        config = trainer.config
        if config.trainer.get("training_profile") not in PRODUCTION_PROFILES:
            raise ValueError("production mode requires the Qwen seven-arm profile")
        if config.trainer.get("training_benchmark_mode") is not None or config.trainer.get("training_capacity_mode", False):
            raise ValueError("production, benchmark and capacity modes are exclusive")
        if config.trainer.get("production_gradient_policy") != "diagnostic_clipping":
            raise ValueError("production requires diagnostic clipping frequency")
        if config.trainer.rollout_integrity.full_dose_gradient_gate_enabled or config.trainer.rollout_integrity.completion_gate_enabled:
            raise ValueError("production completion and full-dose frequency stop rules must be disabled")
        if not config.trainer.rollout_integrity.enabled or config.actor_rollout_ref.actor.grad_clip != 1.0:
            raise ValueError("production requires replay integrity and optimizer clip norm 1.0")
        phase = config.trainer.get("production_phase")
        if phase not in PRODUCTION_PHASES:
            raise ValueError("unknown production invocation phase")
        if config.trainer.training_profile == "qwen3-math-seven-arm-lora-fa3-v1":
            expected_resource_policy = {
                "mode": "physical_device_monitor_v1" if phase == "production" else "physical_device_v1",
                "max_device_used_fraction": 0.98,
                "sample_interval_seconds": 0.1,
            }
            for owner in (config.trainer, config.actor_rollout_ref):
                if OmegaConf.to_container(owner.get("resource_policy", OmegaConf.create({})), resolve=True) != expected_resource_policy:
                    raise ValueError("revised Qwen training requires the sealed physical-device resource policy on driver and workers")
        if trainer.total_rollout_iterations != 109 or trainer.optimizer_steps_per_rollout != 2:
            raise ValueError("production must retain the 109-iteration/two-update recipe")
        if len(trainer.train_dataset) != 6985 or len(trainer.val_dataset) != 512:
            raise ValueError("production requires the sealed 6985/512 data populations")
        if phase == "production":
            if (config.trainer.get("max_rollout_iterations_per_invocation") is not None
                    or not config.trainer.val_before_train
                    or config.trainer.test_freq != 25 or config.trainer.save_freq != 25
                    or config.trainer.get("production_save_first_iteration") is not True
                    or config.data.val_batch_size != 128):
                raise ValueError("production must preserve the full validation/checkpoint schedule")
        elif (config.trainer.val_before_train or config.trainer.test_freq > 0
              or config.trainer.get("production_save_first_iteration", False) is not False):
            raise ValueError("disposable production prologues cannot run validation")
        output = config.trainer.get("production_output")
        if not isinstance(output, str) or not Path(output).is_absolute():
            raise ValueError("production_output must be an absolute per-invocation file")
        if Path(output).exists() or Path(output).is_symlink():
            raise ValueError("production measurement must be a fresh file")
        self.original_fit = trainer.fit
        self.original_init_workers = trainer.init_workers
        self.original_save = trainer._save_checkpoint
        self.original_validate = trainer._validate
        self.started = time.monotonic()
        self.recorder = CapacityRecorder(output, {
            "schema_version": 1, "profile": PRODUCTION_PROFILE,
            "phase": phase, "arm_id": config.trainer.production_arm_id,
            "status": "initializing", "configuration": OmegaConf.to_container(config, resolve=True),
            "checkpoint_provenance": trainer.checkpoint_provenance,
            "wandb_run_id": os.environ.get("WANDB_RUN_ID"),
            "iterations": [], "validations": [], "checkpoints": [],
            "acceptance_policy": {
                "clipping_frequency": "diagnostic", "opd_grpo_ratio": "diagnostic",
                "completion_rate": "diagnostic", "optimizer_clip_norm": 1.0,
                "resource_policy": OmegaConf.to_container(
                    config.trainer.get("resource_policy", OmegaConf.create({})), resolve=True
                ),
            },
        })

    def stage(self, stage, iteration, timing, metrics, meta_info, *, update_state=None):
        self.recorder.enter(stage, iteration, timing, metrics, meta_info, update_state=update_state)

    def init_workers(self):
        self.recorder.enter("worker_initialization")
        try:
            return self.original_init_workers()
        except BaseException as error:
            self.recorder.fail(error)
            raise

    def failure(self, iteration, stage, error, timing, metrics, meta_info, diagnostics):
        self.recorder.enter(stage, iteration, timing, metrics, meta_info)
        self.recorder.fail(error, diagnostics=diagnostics)

    def iteration(self, iteration, timing, metrics, meta_info):
        record = finite_snapshot({
            "rollout_iteration": int(iteration), "timing_s": timing, "metrics": metrics,
            "trajectory_count": meta_info.get("capacity_rollout_trajectory_count"),
            "rollout_timing": meta_info.get("rollout_timing", {}),
            "actor_update_timing": meta_info.get("actor_update_timing", {}),
            "trainer_iteration_complete": True,
        })
        self.recorder.measurement["iterations"].append(record)
        self.recorder.measurement["completed_rollout_iterations"] = int(iteration) + 1
        if self.recorder.measurement["phase"] == "production":
            self.recorder.measurement["remaining_runtime_estimate"] = remaining_runtime_estimate(
                self.recorder.measurement,
                gpus=int(self.trainer.config.trainer.n_gpus_per_node) * int(self.trainer.config.trainer.nnodes),
            )
        self.recorder.persist()

    def save_checkpoint(self, *args, **kwargs):
        manifest = self.original_save(*args, **kwargs)
        path = Path(self.trainer.config.trainer.default_local_dir) / f"global_step_{manifest['global_step']}"
        # The normal save rehashes every payload before atomic publication.
        # The external controller independently verifies the published tree.
        checkpoint = {
            "authenticated": True, "payload_rehashed": True,
            "path": str(path), "manifest": manifest,
            "manifest_sha256": file_sha256(path / "checkpoint_manifest.json"),
        }
        self.recorder.measurement["checkpoint"] = checkpoint
        self.recorder.measurement["checkpoints"].append(checkpoint)
        self.recorder.persist()
        return manifest

    def validate(self):
        started = time.monotonic()
        self.recorder.enter("validation", *self.recorder.current)
        metrics = self.original_validate()
        self.recorder.measurement["validations"].append({
            "completed_rollout_iterations": self.trainer.global_steps,
            "example_count": len(self.trainer.val_dataset),
            "timing_seconds": time.monotonic() - started,
            "metrics": finite_snapshot(metrics),
        })
        self.recorder.persist()
        return metrics

    def fit(self):
        self.recorder.measurement.update(status="running", startup_seconds=max(0.0, time.time() - float(os.environ.get("OPD_PRODUCTION_STARTED", time.time()))))
        self.recorder.enter("training_initialization")
        succeeded = False
        try:
            result = self.original_fit()
            if "completed_rollout_iterations" not in self.recorder.measurement:
                self.recorder.measurement["completed_rollout_iterations"] = self.trainer.global_steps
            self.recorder.measurement["status"] = "complete"
            self.recorder.enter("complete")
            succeeded = True
            return result
        except BaseException as error:
            if self.recorder.measurement.get("status") != "failed":
                self.recorder.fail(error)
            raise
        finally:
            self.recorder.measurement["elapsed_seconds"] = time.monotonic() - self.started
            self.recorder.persist()
            import wandb

            if wandb.run is not None:
                self.recorder.measurement["wandb_online"] = not bool(wandb.run.settings._offline)
                wandb.finish(exit_code=0 if succeeded else 1)
                self.recorder.measurement["wandb_finished"] = True
                self.recorder.persist()


def attach_production_recorder(trainer):
    """Attach only to the explicitly selected production trainer instance."""
    recorder = ProductionRecorder(trainer)
    trainer._production_recorder = recorder
    trainer.record_capacity_stage = recorder.stage
    trainer.record_benchmark_failure = recorder.failure
    trainer.record_benchmark_iteration = recorder.iteration
    trainer._save_checkpoint = recorder.save_checkpoint
    trainer._validate = recorder.validate
    trainer.init_workers = recorder.init_workers
    trainer.fit = recorder.fit
