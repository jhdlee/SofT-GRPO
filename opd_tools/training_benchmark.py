"""Run one bounded Qwen3 training calibration cell, using fresh trainer processes."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping

from .manifest import canonical_sha256, file_sha256, write_manifest_atomic


from .training_runtime import MODEL_ID, MODEL_REVISION, VARIANTS, estimate_runtime


VARIANT_CONFIGS = {name: (value["dispatch_mode"], value["max_running_requests"], value["async_queue_size"]) for name, value in VARIANTS.items()}


def source_identity(fork_root):
    """Authenticate committed source before a direct or Slurm invocation."""
    fork_root = Path(fork_root).resolve()
    parent = fork_root.parents[1]
    def git(root, *args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    result = {}
    for name, root in (("parent", parent), ("fork", fork_root)):
        commit = git(root, "rev-parse", "HEAD")
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise ValueError(f"{name} source is not a full Git commit")
        if git(root, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError(f"benchmark requires a clean committed {name} source snapshot")
        expected = os.environ.get("OPD_QTB_" + name.upper() + "_COMMIT")
        if expected is not None and expected != commit:
            raise ValueError(f"{name} source differs from submitted commit")
        result[name + "_commit"] = commit
    if git(parent, "rev-parse", "HEAD:3rdparty/SofT-GRPO") != result["fork_commit"]:
        raise ValueError("parent gitlink differs from the running benchmark fork")
    result["snapshot_verified"] = True
    return result


def write_json(path, value):
    write_manifest_atomic(Path(path), value, validator=None)


def compare_fingerprints(baseline, candidate):
    """Describe changed outputs without asserting scheduling equivalence."""
    left, right = baseline.get("requests", []), candidate.get("requests", [])
    if len(left) != len(right) or not left:
        raise ValueError("dispatch comparison has missing request identities")
    identity_fields = ("example_id", "sample_index", "prompt_ids_sha256", "request_seed")
    if any(any(field not in row for field in identity_fields) for row in [*left, *right]):
        raise ValueError("dispatch comparison has incomplete prompt/sample identities")
    left_identities = [tuple(row[field] for field in identity_fields) for row in left]
    right_identities = [tuple(row[field] for field in identity_fields) for row in right]
    if len(set(left_identities)) != len(left_identities):
        raise ValueError("dispatch comparison repeats a prompt/sample identity")
    if left_identities != right_identities:
        raise ValueError("dispatch changed prompt/sample seed identity or order")
    optional = {}
    for field in ("probabilities_sha256", "retained_mask_sha256", "raw_noise_sha256"):
        paired = [(a[field], b[field]) for a, b in zip(left, right) if field in a and field in b]
        optional[field] = {
            "paired_requests": len(paired),
            "different_requests": sum(a != b for a, b in paired) if paired else None,
            "baseline_missing": sum(field not in row for row in left),
            "candidate_missing": sum(field not in row for row in right),
        }
    return {
        "paired_requests": len(left),
        "matching_request_seeds": len(left),
        **{
            field + "_different_requests": sum(a[field] != b[field] for a, b in zip(left, right))
            for field in ("token_count", "tokens_sha256", "support_sha256", "perturbations_sha256", "log_probs_sha256")
        },
        "interpretation": "fixed-seed scheduling comparison; not a bitwise-equivalence acceptance gate",
        "optional_replay_metadata": optional,
    }


def validate_pilot_metrics(objective, iteration, metrics, *, actor_update_timing=None, expected_ranks=None):
    """Accept one of the three benchmark updates using existing actor evidence.

    Production replay, causal-mask, frozen-teacher, and finite-update guards
    remain authoritative. This adds benchmark cadence and nonzero OPD evidence;
    the full-dose gradient-ratio gate is intentionally outside this warmup pilot.
    """
    if objective not in ("standalone", "hybrid"):
        raise ValueError("pilot objective must be standalone or hybrid")
    if type(iteration) is not int or iteration not in range(3):
        raise ValueError("pilot iteration must be 0, 1, or 2")
    if not isinstance(metrics, Mapping):
        raise ValueError("pilot metrics must be a mapping")

    checked = {}

    def number(name, source=metrics, evidence=checked, label="pilot metric"):
        value = source.get(name)
        if isinstance(value, (bool, str, bytes)):
            raise ValueError(f"{label} {name} must be a finite number")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{label} {name} must be a finite number") from error
        if not math.isfinite(value):
            raise ValueError(f"{label} {name} must be finite")
        evidence[name] = value
        return value

    active = objective == "standalone" or iteration > 0
    # EMA follows the completed policy update, including hybrid iteration zero.
    # The effective OPD dose controls teacher KL and OPD gradients, not EMA.
    exact = {
        "integrity/continuous_replay_active": 1,
        "replay/fallback_count": 0,
        "trainer/rollout_iteration": iteration,
        "trainer/optimizer_steps_this_iteration": 2,
        "trainer/optimizer_step": 2 * (iteration + 1),
        "opd/ema_updates_this_iteration": 1,
        "opd/ema_update_count": iteration + 1,
    }
    for name, expected in exact.items():
        if number(name) != expected:
            raise ValueError(f"pilot metric {name} must equal {expected}")
    expected_beta = 1.0 if objective == "standalone" else 0.001 * iteration / 11
    if not math.isclose(number("opd/beta_effective"), expected_beta, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError(f"pilot metric opd/beta_effective must equal {expected_beta}")
    if not 0.0 <= number("replay/ratio_abs_error_max") <= 1e-4:
        raise ValueError("pilot replay/ratio_abs_error_max must be in [0, 1e-4]")
    if not 0.0 < number("latent/soft_to_hard_rate") <= 1.0:
        raise ValueError("pilot latent/soft_to_hard_rate must be in (0, 1]")
    for name in ("opd/latent_slot_count", "opd/answer_slot_count", "grad/opd_norm"):
        value = number(name)
        if active and value <= 0.0:
            raise ValueError(f"active pilot metric {name} must be positive")
        if not active and value != 0.0:
            raise ValueError(f"zero-dose pilot metric {name} must equal zero")
    if number("grad/total_norm") < 0.0:
        raise ValueError("pilot grad/total_norm must be nonnegative")
    result = {
        "accepted": True, "objective": objective, "rollout_iteration": iteration,
        "opd_active": active, "metrics": checked,
    }
    if actor_update_timing is not None or expected_ranks is not None:
        if type(expected_ranks) is not int or expected_ranks not in (1, 2):
            raise ValueError("pilot expected_ranks must be the authorized rank count 1 or 2")
        if not isinstance(actor_update_timing, Mapping):
            raise ValueError("pilot actor_update_timing must contain the rank inventory")
        ranks = actor_update_timing.get("ranks")
        if not isinstance(ranks, list) or len(ranks) != expected_ranks:
            raise ValueError("pilot rank inventory must contain exactly one row per expected rank")
        rank_ids = [row.get("rank") if isinstance(row, Mapping) else None for row in ranks]
        if any(type(rank) is not int for rank in rank_ids) or set(rank_ids) != set(range(expected_ranks)):
            raise ValueError("pilot rank inventory must contain each expected rank exactly once")
        result["ranks"] = []
        for row in sorted(ranks, key=lambda item: item["rank"]):
            evidence = {"rank": row["rank"]}
            label = f"pilot rank {row['rank']} metric"
            for name, expected in (
                ("optimizer_steps", 2),
                ("ema_updates_this_iteration", 1),
                ("ema_update_count", iteration + 1),
            ):
                if number(name, row, evidence, label) != expected:
                    raise ValueError(f"{label} {name} must equal {expected}")
            teacher_seconds = number("teacher_seconds", row, evidence, label)
            if (active and teacher_seconds <= 0.0) or (not active and teacher_seconds != 0.0):
                raise ValueError(f"{label} teacher_seconds must be positive when active and zero at zero dose")
            for name in ("policy_update_seconds", "worker_update_seconds"):
                if number(name, row, evidence, label) < 0.0:
                    raise ValueError(f"{label} {name} must be nonnegative")
            result["ranks"].append(evidence)
    return result


def validate_phase_measurement(measured, *, phase, variant, batches, overrides, source, assets, run_id):
    """Bind child output to the exact source, assets, request, and invocation."""
    from verl.opd.provenance import validate_checkpoint_provenance

    if not isinstance(measured, Mapping) or measured.get("phase") != phase or measured.get("variant") != variant:
        raise ValueError("phase measurement identity differs from invocation")
    if measured.get("wandb_run_id") != run_id:
        raise ValueError("phase W&B identity differs from invocation")
    configuration = measured.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("phase has no resolved configuration")
    expected = {}
    for override in overrides:
        key, text = override.lstrip("+").split("=", 1)
        if key.startswith("hydra."):
            continue  # Hydra's own invocation settings are absent from task config.
        try:
            expected[key] = json.loads(text)
        except json.JSONDecodeError:
            expected[key] = text
    for key, value in expected.items():
        observed = configuration
        for part in key.split("."):
            if not isinstance(observed, Mapping) or part not in observed:
                raise ValueError("phase configuration is missing " + key)
            observed = observed[part]
        if canonical_sha256(observed) != canonical_sha256(value):
            raise ValueError("phase configuration differs at " + key)
    provenance = validate_checkpoint_provenance(measured.get("checkpoint_provenance"))
    if provenance["source"]["commit"] != source["fork_commit"]:
        raise ValueError("phase source differs from the submitted benchmark fork")
    model = provenance["model"]
    if model["id"] != MODEL_ID or model["resolved_revision"] != MODEL_REVISION:
        raise ValueError("phase model differs from pinned Qwen3 model")
    if model["manifest_content_sha256"] != assets["model_manifest_content_sha256"]:
        raise ValueError("phase model asset authentication differs")
    manifests = provenance["data"]["manifests"]
    if len(manifests) != 1 or manifests[0]["manifest_content_sha256"] != assets["data_manifest_content_sha256"]:
        raise ValueError("phase train/validation asset authentication differs")
    if provenance["resolved_hydra_config"]["full_sha256"] != canonical_sha256(configuration):
        raise ValueError("phase resolved configuration does not match checkpoint provenance")
    rows = measured.get("rows", [])
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) or row.get("variant") != variant or row.get("batch_index") not in batches for row in rows):
        raise ValueError("phase returned unrequested dispatch/batch rows")
    if len({row["batch_index"] for row in rows}) != len(rows):
        raise ValueError("phase repeats a calibration batch")
    if measured.get("status") == "complete" and phase == "calibration" and [row["batch_index"] for row in rows] != batches:
        raise ValueError("complete phase is missing a requested calibration batch")


class CellRunner:
    def __init__(self, args):
        from .qwen_training import verify

        self.args = args
        self.assets = verify(args.root)
        self.fork_root = Path(__file__).resolve().parents[1]
        self.source = source_identity(self.fork_root)
        self.root = args.output_dir
        self.root.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.deadline = self.started + args.time_limit_seconds
        self.child = None
        self.cell = {
            "schema_version": 1, "profile": "qwen3-training-benchmark-v1",
            "objective": args.objective, "gpus": args.gpus, "status": "running",
            "source": self.source, "assets": self.assets,
            "configuration": {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "objective": args.objective, "gpus": args.gpus, "tensor_parallel_size": 1, "rollout_iterations": 109, "time_limit_seconds": args.time_limit_seconds, "variants": VARIANT_CONFIGS},
            "jobs": {"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "account": os.environ.get("SLURM_JOB_ACCOUNT")},
            "wandb_run_ids": [], "screen_rows": [], "confirm_rows": [], "iterations": [], "phases": [],
        }
        self._persist()

    def _persist(self):
        self.cell["elapsed_seconds"] = time.monotonic() - self.started
        write_json(self.root / "cell.json", self.cell)

    def _terminate(self):
        if self.child is not None:
            try:
                os.killpg(self.child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.child.pid, signal.SIGKILL)
                self.child.wait(timeout=15)
            self.child = None

    def phase(self, phase, variant, batches):
        from .qwen_training import profile_overrides

        remaining = self.deadline - time.monotonic() - 45
        if remaining < 60:
            raise TimeoutError("cell allocation has no remaining phase budget")
        label = f"{len(self.cell['phases']):02d}-{phase}-{variant}"
        phase_root = self.root / label
        phase_root.mkdir()
        output = phase_root / "measurement.json"
        dispatch, running, queue = VARIANT_CONFIGS[variant]
        overrides = profile_overrides(self.args.objective, self.args.gpus, self.args.root, phase_root / "training",
                                      replay_backend=getattr(self.args, "qwen_replay_backend", "disabled"))
        overrides += [
            f"actor_rollout_ref.rollout.dispatch_mode={dispatch}",
            f"actor_rollout_ref.rollout.max_running_requests={running}",
            f"actor_rollout_ref.rollout.async_queue_size={queue}",
            f"++trainer.training_benchmark_mode={phase}",
            f"++trainer.training_benchmark_variant={variant}",
            f"++trainer.training_benchmark_batches={json.dumps(batches, separators=(',', ':'))}",
            f"++trainer.training_benchmark_selection={self.args.root / 'selection.json'}",
            f"++trainer.training_benchmark_output={output}",
            "trainer.val_before_train=false", "trainer.test_freq=-1", "trainer.save_freq=-1",
            "trainer.log_val_generations=0", "trainer.resume_mode=disable",
            "trainer.max_rollout_iterations_per_invocation=3",
            "trainer.rollout_integrity.full_dose_gradient_gate_enabled=false",
            f"hydra.run.dir={phase_root / 'hydra'}", "hydra.job.chdir=false",
        ]
        env = dict(os.environ)
        run_id = "qtb-" + canonical_sha256({"source": self.source, "job": self.cell["jobs"], "objective": self.args.objective, "gpus": self.args.gpus, "phase": label})[:24]
        env.update({"WANDB_RUN_ID": run_id, "WANDB_NAME": label,
                    "WANDB_MODE": "online", "WANDB_RESUME": "never",
                    "OPD_BENCHMARK_PHASE_STARTED": str(time.time())})
        self.cell["wandb_run_ids"].append(run_id)
        phase_record = {"phase": phase, "variant": variant, "batches": batches, "output": str(output), "wandb_run_id": run_id, "status": "running"}
        self.cell["phases"].append(phase_record)
        self._persist()
        command = [sys.executable, "-m", "verl.trainer.main_ppo", *overrides]
        write_json(phase_root / "invocation.json", {"command": command, "source": self.source, "wandb_run_id": run_id})
        failure = None
        measured = None
        try:
            with (phase_root / "trainer.log").open("w") as log:
                self.child = subprocess.Popen(command, cwd=self.fork_root / "verl-0.4.x", env=env,
                                              stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                code = self.child.wait(timeout=remaining)
            if code:
                raise RuntimeError(f"{label} failed with exit code {code}; see {phase_root / 'trainer.log'}")
        except BaseException as error:
            failure = error
        finally:
            self._terminate()
            phase_record["status"] = "incomplete"
            phase_record["authenticated"] = False
            try:
                if not output.is_file() or output.is_symlink():
                    raise ValueError("phase did not publish a regular measurement file")
                measured = json.loads(output.read_text())
                phase_record.update({"measurement_status": measured.get("status"), "sha256": file_sha256(output), "measurement": measured})
                validate_phase_measurement(measured, phase=phase, variant=variant, batches=batches,
                                           overrides=overrides, source=self.source, assets=self.assets, run_id=run_id)
                phase_record["authenticated"] = True
                if failure is None and measured.get("status") == "complete" and measured.get("wandb_online") is True and measured.get("wandb_finished") is True:
                    phase_record["status"] = "complete"
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
                phase_record["authentication_error"] = f"{type(error).__name__}: {error}"
            if isinstance(measured, Mapping):
                if phase == "pilot":
                    for key in ("startup_seconds", "iterations", "validation_seconds", "validation_example_count", "checkpoint_seconds", "checkpoint_provenance"):
                        if key in measured:
                            self.cell[key] = measured[key]
                elif isinstance(measured.get("rows"), list):
                    target = "screen_rows" if batches == [0] else "confirm_rows"
                    self.cell[target].extend({**row, "valid": row.get("valid") is True and phase_record["status"] == "complete"} for row in measured["rows"] if isinstance(row, Mapping))
            self._persist()
        if failure is not None:
            raise failure
        if phase_record["status"] != "complete":
            raise RuntimeError(f"{label} did not complete with online W&B publication")
        return measured

    def run(self):
        from .icl_resource_monitor import ResourceMonitor
        from .training_runtime import select_dispatch

        monitor = ResourceMonitor(interval_seconds=2.0).start()
        def interrupted(signum, frame):
            raise TimeoutError(f"received signal {signum}; no additional allocation requested")
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            repair_variant = getattr(self.args, "repair_pilot", None)
            if repair_variant is not None:
                # Explicit repair validation is separate from the four-cell
                # dispatch study and cannot stand in for its runtime report.
                self.cell["role"] = "repair_validation"
                self.cell["requested_dispatch"] = repair_variant
                self._persist()
                self.phase("pilot", repair_variant, [])
                self.cell["status"] = "repair_validation_complete"
                return
            for variant in VARIANT_CONFIGS:
                self.phase("calibration", variant, [0])
            screening = select_dispatch(self.cell["screen_rows"], [])
            candidate = screening["candidate_variant"]
            for variant in dict.fromkeys(("legacy_batch", candidate)):
                self.phase("calibration", variant, [1])
            selection = select_dispatch(self.cell["screen_rows"], self.cell["confirm_rows"])
            self.cell["dispatch_selection"] = selection
            self.cell["configuration"]["selected"] = selection["configuration"]
            comparisons = []
            for rows in (self.cell["screen_rows"], self.cell["confirm_rows"]):
                baseline = next(row for row in rows if row["variant"] == "legacy_batch")
                for candidate_row in rows:
                    if candidate_row["variant"] != "legacy_batch":
                        comparisons.append({"variant": candidate_row["variant"], "batch_index": candidate_row["batch_index"],
                                            **compare_fingerprints(baseline, candidate_row)})
            self.cell["output_comparisons"] = comparisons
            self._persist()
            self.phase("pilot", selection["selected_variant"], [])
            self.cell["estimate"] = estimate_runtime(self.args.objective, self.args.gpus,
                self.cell.get("startup_seconds"), self.cell["iterations"],
                self.cell.get("validation_seconds"), self.cell.get("checkpoint_seconds"))
            if self.cell.get("validation_example_count") != 128:
                raise ValueError("pilot timing validation must contain exactly 128 examples")
            self.cell["status"] = "complete"
        except BaseException as error:
            self.cell["status"] = "incomplete"
            self.cell["error"] = f"{type(error).__name__}: {error}"
            self.cell["reason"] = self.cell["error"]
            raise
        finally:
            self._terminate()
            self.cell["resource_telemetry"] = monitor.stop().to_dict()
            self._persist()
            (self.root / "cell.json.sha256").write_text(file_sha256(self.root / "cell.json") + "  cell.json\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--objective", choices=("standalone", "hybrid"), required=True)
    parser.add_argument("--gpus", type=int, choices=(1, 2), required=True)
    parser.add_argument("--time-limit-seconds", type=int, default=7100)
    parser.add_argument("--repair-pilot", choices=tuple(VARIANT_CONFIGS), default=None,
                        help="Run only the strict three-iteration pilot; no dispatch-study estimate")
    parser.add_argument("--qwen-replay-backend", choices=("disabled", "native_fa3_v1"), default="disabled",
                        help="Versioned common inference/replay arithmetic; opt-in repair pilot only")
    args = parser.parse_args(argv)
    if not 60 <= args.time_limit_seconds <= 7200:
        parser.error("benchmark time limit must remain within two hours")
    if args.repair_pilot is not None and args.time_limit_seconds > 1800:
        parser.error("repair validation must be explicitly capped at thirty minutes")
    if args.qwen_replay_backend != "disabled" and args.repair_pilot is None:
        parser.error("native replay arithmetic currently requires --repair-pilot")
    CellRunner(args).run()


if __name__ == "__main__":
    main()
