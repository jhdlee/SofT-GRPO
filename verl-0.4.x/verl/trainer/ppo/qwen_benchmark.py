"""Bounded timing experiments using the actual live-policy training workers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.opd_driver import compute_rollout_diagnostics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.dataset.rl_dataset import collate_fn


def _write(path, value):
    from opd_tools.manifest import write_manifest_atomic

    write_manifest_atomic(Path(path), value, validator=None)


def _json_metrics(metrics):
    result = {}
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (str, bool, int)) or value is None:
            result[key] = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("non-finite benchmark metric: " + key)
            result[key] = value
    return result


class QwenTrainingBenchmarkTrainer(RayPPOTrainer):
    """A separate measurement mode; it never selects production checkpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.trainer.get("training_profile") != "qwen3-training-benchmark-v1":
            raise ValueError("training benchmark requires the sealed Qwen3 profile")
        self.measurement_path = Path(self.config.trainer.training_benchmark_output)
        self.selection = json.loads(
            Path(self.config.trainer.training_benchmark_selection).read_text()
        )
        self.measurement = {
            "schema_version": 1,
            "status": "initializing",
            "phase": self.config.trainer.training_benchmark_mode,
            "variant": self.config.trainer.training_benchmark_variant,
            "iterations": [],
            "rows": [],
            "configuration": OmegaConf.to_container(self.config, resolve=True),
            "checkpoint_provenance": self.checkpoint_provenance,
            "wandb_run_id": os.environ.get("WANDB_RUN_ID"),
        }
        self._persist()

    def _persist(self):
        _write(self.measurement_path, self.measurement)

    def record_benchmark_iteration(self, iteration, timing, metrics, meta_info):
        from opd_tools.training_benchmark import validate_pilot_metrics

        timings = dict(timing)
        ranks = meta_info.get("rollout_timing", {}).get("ranks", [])
        if ranks:
            timings["weight_sync"] = max(
                float(rank.get("weight_sync_seconds", 0.0)) for rank in ranks
            )
        record = {
            "rollout_iteration": int(iteration),
            "timing_s": timings,
            "metrics": _json_metrics(metrics),
            "rollout_timing": meta_info.get("rollout_timing", {}),
            "actor_update_timing": meta_info.get("actor_update_timing", {}),
        }
        self.measurement["iterations"].append(record)
        if "save_checkpoint" in timing:
            self.measurement["checkpoint_seconds"] = float(timing["save_checkpoint"])
        try:
            record["pilot_acceptance"] = validate_pilot_metrics(
                "standalone" if self.standalone_opd else "hybrid", int(iteration),
                record["metrics"], actor_update_timing=record["actor_update_timing"],
                expected_ranks=int(self.config.trainer.n_gpus_per_node),
            )
        except ValueError as error:
            record["pilot_acceptance"] = {"accepted": False, "error": str(error)}
            self._persist()
            raise
        self._persist()

    def _generation_batch(self, indices, *, warmup=False):
        batch = DataProto.from_single_dict(collate_fn([
            self.train_dataset[int(index)] for index in indices
        ]))
        keys = [key for key in ("raw_prompt_ids", "index") if key in batch.non_tensor_batch]
        generation = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=keys,
        )
        generation.meta_info.update({
            "rollout_seed": 11,
            "rollout_iteration": 0,
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        })
        if warmup:
            generation.meta_info["benchmark_max_new_tokens"] = 32
        return generation

    def _calibrate(self):
        from opd_tools.icl_resource_monitor import ResourceMonitor
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self._benchmark_logger = logger
        batch_indices = list(self.config.trainer.training_benchmark_batches)
        # All dispatch variants get one identical, excluded warmup.  The normal
        # sharding context clears KV state and resynchronizes unchanged weights.
        self.actor_rollout_wg.generate_sequences(self._generation_batch(
            self.selection["train_batches"][0][:8], warmup=True
        ))
        for batch_index in batch_indices:
            generation = self._generation_batch(self.selection["train_batches"][batch_index])
            group_size = int(self.config.actor_rollout_ref.rollout.n)
            prompt_ids = [list(map(int, ids)) for ids in generation.non_tensor_batch["raw_prompt_ids"]]
            expected_indices = np.repeat(generation.non_tensor_batch["index"], group_size)
            allocation_environment = dict(os.environ)
            allocation_environment["CUDA_VISIBLE_DEVICES"] = os.environ["OPD_BENCHMARK_GPU_ASSIGNMENT"]
            monitor = ResourceMonitor(
                interval_seconds=2.0,
                gpu_visibility_environment=allocation_environment,
            ).start()
            try:
                started = time.perf_counter()
                output = self.actor_rollout_wg.generate_sequences(generation)
                elapsed = time.perf_counter() - started
            finally:
                telemetry = monitor.stop().to_dict()
            if not np.array_equal(output.non_tensor_batch["index"], expected_indices):
                raise RuntimeError("calibration changed prompt/sample identity or ordering")
            response = output.batch["responses"]
            mask = output.batch["attention_mask"][:, -response.shape[-1]:].bool()
            supports = output.batch["rollout_topk_ids"][:, -response.shape[-1]:]
            perturbations = output.batch["rollout_topk_gumbels"][:, -response.shape[-1]:]
            diagnostics = compute_rollout_diagnostics(
                responses=response, response_mask=mask,
                rollout_topk_ids=supports, rollout_topk_gumbels=perturbations,
                gumbel_temperature=0.1, close_tag_token_id=self.close_tag_token_id,
                decode=lambda ids: self.tokenizer.decode(ids, skip_special_tokens=False),
            )
            fingerprints = []
            for row in range(response.shape[0]):
                valid = mask[row]
                fields = {
                    "tokens": response[row][valid],
                    "support": supports[row][valid],
                    "perturbations": perturbations[row][valid],
                    "log_probs": output.batch["rollout_log_probs"][row][valid],
                }
                fingerprints.append({
                    "example_id": self.selection["populations"]["train"][
                        self.selection["train_batches"][batch_index][row // group_size]
                    ]["example_id"],
                    "sample_index": row % group_size,
                    "prompt_ids_sha256": hashlib.sha256(json.dumps(
                        prompt_ids[row // group_size], separators=(",", ":")
                    ).encode()).hexdigest(),
                    "request_seed": int(output.batch["rollout_sampling_seed"][row].item()),
                    "token_count": int(valid.sum().item()),
                    **{name + "_sha256": hashlib.sha256(
                        tensor.detach().cpu().contiguous().numpy().tobytes()
                    ).hexdigest() for name, tensor in fields.items()},
                })
            expected = 64 * int(self.config.actor_rollout_ref.rollout.n)
            if len(fingerprints) != expected:
                raise RuntimeError("calibration did not return every prompt/sample")
            # The timing study requires intact metadata and a real boundary;
            # the unmodified training integrity gates run during the pilot.
            valid = (
                all(math.isfinite(float(value)) for value in diagnostics.metrics.values())
                and float(diagnostics.metrics.get("replay/fallback_count", 0)) == 0
                and any(diagnostics.boundary_valid_mask)
            )
            row = {
                "variant": self.config.trainer.training_benchmark_variant,
                "batch_index": int(batch_index), "wall_seconds": elapsed,
                "generated_tokens": int(mask.sum().item()), "valid": valid,
                "diagnostics": dict(diagnostics.metrics),
                "rollout_timing": output.meta_info.get("rollout_timing", {}),
                "resource_telemetry": telemetry,
                "requests": fingerprints,
            }
            self.measurement["rows"].append(row)
            self._persist()
            logger.log({
                "benchmark/batch_index": int(batch_index),
                "benchmark/wall_seconds": elapsed,
                "benchmark/tokens_per_second": row["generated_tokens"] / elapsed,
                "benchmark/valid": int(valid), **dict(diagnostics.metrics),
            }, step=int(batch_index))

    def _timing_validation(self):
        # Invoke the actual validation path without calling BEST selection or
        # writing a validation event at the full 512-example study identity.
        original_loader = self.val_dataloader
        self.val_dataloader = [collate_fn([
            self.val_dataset[int(index)] for index in self.selection["validation_indices"]
        ])]
        try:
            started = time.perf_counter()
            metrics = self._validate()
            self.measurement["validation_seconds"] = time.perf_counter() - started
            self.measurement["validation_example_count"] = 128
            self.measurement["timing_only_validation_metrics"] = _json_metrics(metrics)
        finally:
            self.val_dataloader = original_loader

    def fit(self):
        self.measurement["startup_seconds"] = max(
            0.0, time.time() - float(os.environ["OPD_BENCHMARK_PHASE_STARTED"])
        )
        self.measurement["status"] = "running"
        self._persist()
        succeeded = False
        try:
            if self.config.trainer.training_benchmark_mode == "calibration":
                self._calibrate()
            elif self.config.trainer.training_benchmark_mode == "pilot":
                if self.total_rollout_iterations != 109 or self.optimizer_steps_per_rollout != 2:
                    raise RuntimeError("pilot changed the production horizon or optimizer cadence")
                super().fit()
                if len(self.measurement["iterations"]) != 3:
                    raise RuntimeError("pilot did not complete three iterations")
                self._timing_validation()
            else:
                raise ValueError("unknown training benchmark phase")
            self.measurement["status"] = "complete"
            succeeded = True
        except BaseException as error:
            self.measurement["status"] = "failed"
            self.measurement["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            self._persist()
            import wandb

            if wandb.run is not None:
                self.measurement["wandb_online"] = not wandb.run.settings._offline
                wandb.finish(exit_code=0 if succeeded else 1)
                self.measurement["wandb_finished"] = True
                self._persist()
