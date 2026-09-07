"""A disposable single-iteration G8 capacity check with durable failure stages."""

import os
from pathlib import Path
import time

from omegaconf import OmegaConf

from opd_tools.manifest import file_sha256
from opd_tools.training_capacity import CapacityRecorder, capacity_contract, finite_snapshot, validate_capacity_beta_base, validate_capacity_iteration
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _verify_checkpoint


class QwenTrainingCapacityTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.config
        self.capacity_beta_base = validate_capacity_beta_base(config.trainer.get("training_capacity_beta_base", 0.001))
        if config.trainer.get("training_benchmark_mode") is not None:
            raise ValueError("capacity and three-iteration benchmark modes are exclusive")
        if config.trainer.get("training_profile") != "qwen3-training-benchmark-v1":
            raise ValueError("capacity requires the sealed Qwen3 training profile")
        if self.total_rollout_iterations != 109 or self.optimizer_steps_per_rollout != 2:
            raise ValueError("capacity must preserve the 109-iteration, two-update recipe")
        if self.standalone_opd or config.actor_rollout_ref.rollout.n != 8:
            raise ValueError("capacity requires the primary G8 hybrid")
        if config.trainer.max_rollout_iterations_per_invocation != 1:
            raise ValueError("capacity must stop after one iteration")
        if config.algorithm.opd.schedule != "constant" or config.algorithm.opd.beta_base != self.capacity_beta_base:
            raise ValueError("capacity OPD dose must match the selected constant full beta")
        if not self.rollout_integrity_config.full_dose_gradient_gate_enabled or self.rollout_integrity_config.completion_gate_enabled:
            raise ValueError("capacity requires the full-dose gate and disables completion gating")
        if config.trainer.val_before_train or config.trainer.test_freq > 0:
            raise ValueError("capacity must not run validation")
        for key, expected in {
            "data.train_batch_size": 64, "data.max_response_length": 8192,
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 2,
            "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
            "actor_rollout_ref.model.qwen_replay_backend": "native_fa3_v1",
            "actor_rollout_ref.rollout.dispatch_mode": "bounded_async",
            "actor_rollout_ref.rollout.max_running_requests": 32,
            "actor_rollout_ref.rollout.async_queue_size": 64,
            "trainer.n_gpus_per_node": 2, "trainer.nnodes": 1,
            "trainer.resume_mode": "disable", "trainer.rollout_integrity.enabled": True,
            "algorithm.opd.enabled": True, "algorithm.opd.mode": "auxiliary",
            "algorithm.opd.loss_support": "all_response", "algorithm.opd.teacher.type": "ema",
            "algorithm.opd.teacher.ema_decay": 0.99, "algorithm.opd.trajectory_gate": "all",
            "actor_rollout_ref.opd.beta_base": self.capacity_beta_base,
            "actor_rollout_ref.opd.schedule": "constant",
        }.items():
            if OmegaConf.select(config, key) != expected:
                raise ValueError("capacity configuration differs at " + key)
        from verl.opd.qwen_replay_backend import qwen_replay_arithmetic_identity

        self.capacity = CapacityRecorder(config.trainer.training_capacity_output, {
            "schema_version": 1, "status": "initializing", "phase": "capacity",
            "variant": "bounded_async32", "contract": capacity_contract(self.capacity_beta_base),
            "configuration": OmegaConf.to_container(config, resolve=True),
            "checkpoint_provenance": self.checkpoint_provenance,
            "qwen_replay_arithmetic": qwen_replay_arithmetic_identity(),
            "wandb_run_id": os.environ.get("WANDB_RUN_ID"), "iterations": [], "rows": [],
        })

    def init_workers(self):
        self.capacity.enter("worker_initialization")
        try:
            return super().init_workers()
        except BaseException as error:
            self.capacity.fail(error)
            raise

    def record_capacity_stage(self, stage, iteration, timing, metrics, meta_info, *, update_state=None):
        self.capacity.enter(stage, iteration, timing, metrics, meta_info, update_state=update_state)

    def record_benchmark_failure(self, iteration, stage, error, timing, metrics, meta_info, diagnostics):
        # Existing replay guards call this hook. Preserve their richer partial
        # timer and diagnostics while keeping the capacity-specific update state.
        self.capacity.enter(stage, iteration, timing, metrics, meta_info)
        self.capacity.fail(error, diagnostics=diagnostics)

    def record_benchmark_iteration(self, iteration, timing, metrics, meta_info):
        record = finite_snapshot({
            "rollout_iteration": int(iteration), "timing_s": timing, "metrics": metrics,
            "trajectory_count": meta_info.get("capacity_rollout_trajectory_count"),
            "rollout_timing": meta_info.get("rollout_timing", {}),
            "actor_update_timing": meta_info.get("actor_update_timing", {}),
        })
        record["capacity_acceptance"] = validate_capacity_iteration(record, beta_base=self.capacity_beta_base)
        self.capacity.measurement["iterations"].append(record)
        self.capacity.measurement["checkpoint_seconds"] = timing.get("save_checkpoint")
        self.capacity.persist()

    def _save_checkpoint(self, *args, **kwargs):
        manifest = super()._save_checkpoint(*args, **kwargs)
        path = Path(self.config.trainer.default_local_dir) / "global_step_1"
        # The normal save already authenticates the unpublished tree. Recheck
        # the published name and every payload byte before calling it accepted.
        self.capacity.enter("checkpoint_authentication", *self.capacity.current)
        verified = _verify_checkpoint(str(path), expected_provenance=self.checkpoint_provenance)
        if verified != manifest:
            raise RuntimeError("capacity checkpoint changed during publication")
        if (manifest["global_step"], manifest["optimizer_step"], manifest["world_size"], manifest["total_rollout_iterations"]) != (1, 2, 2, 109):
            raise RuntimeError("capacity checkpoint cadence or topology differs")
        if manifest["reason"] != "invocation_limit" or manifest["completed_rollout_iteration"] != 0 or manifest["next_rollout_iteration"] != 1:
            raise RuntimeError("capacity checkpoint boundary differs")
        if any(manifest.get(key) is not None for key in ("selection_metric_name", "selection_metric_value", "selection_tiebreak_metric_name", "selection_tiebreak_metric_value")):
            raise RuntimeError("capacity checkpoint must not enter BEST selection")
        for rank in range(2):
            import json
            state = json.loads((path / "actor" / "opd_teacher" / f"ema_state_world_size_2_rank_{rank}.json").read_text())
            if state.get("last_rollout_iteration") != 0 or state.get("update_count") != 1:
                raise RuntimeError("capacity checkpoint teacher EMA cadence differs")
        self.capacity.measurement["checkpoint"] = {
            "authenticated": True, "payload_rehashed": True, "path": str(path),
            "manifest_sha256": file_sha256(path / "checkpoint_manifest.json"),
            "manifest": manifest,
        }
        self.capacity.persist()
        return manifest

    def fit(self):
        self.capacity.measurement.update(
            status="running", startup_seconds=max(0, time.time() - float(os.environ["OPD_CAPACITY_STARTED"])),
        )
        self.capacity.enter("training_initialization")
        succeeded = False
        try:
            super().fit()
            if len(self.capacity.measurement["iterations"]) != 1 or not self.capacity.measurement.get("checkpoint", {}).get("authenticated"):
                raise RuntimeError("capacity did not complete one accepted iteration and checkpoint")
            self.capacity.enter("complete", update_state="completed")
            self.capacity.measurement["status"] = "complete"
            succeeded = True
        except BaseException as error:
            # Preserve the richer replay failure already written by its hook.
            if self.capacity.measurement.get("status") != "failed":
                self.capacity.fail(error)
            raise
        finally:
            self.capacity.persist()
            import wandb
            if wandb.run is not None:
                self.capacity.measurement["wandb_online"] = not bool(wandb.run.settings._offline)
                wandb.finish(exit_code=0 if succeeded else 1)
                self.capacity.measurement["wandb_finished"] = True
                self.capacity.persist()
