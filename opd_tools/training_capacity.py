"""One bounded, full-dose Qwen3 G8 capacity check; never a training launch."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Mapping

from .manifest import canonical_sha256, file_sha256, write_manifest_atomic


CAPACITY_BETA_BASES = (0.001, 0.1)


def validate_capacity_beta_base(beta_base):
    """Only the default and explicitly requested high-dose capacity tests exist."""
    if type(beta_base) not in (int, float) or beta_base not in CAPACITY_BETA_BASES:
        raise ValueError("capacity beta_base must be exactly 0.001 or 0.1")
    return float(beta_base)


def capacity_contract(beta_base=0.001):
    """Dependency-light, source-bound receipt contract for this single test."""
    beta_base = validate_capacity_beta_base(beta_base)
    return {
        "schema_version": 1, "role": "qwen3_g8_capacity",
        "model_id": "Qwen/Qwen3-0.6B",
        "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "objective": "hybrid", "group_size": 8, "prompt_batch_size": 64,
        "micro_batch_size_per_gpu": 2, "gpus": 2, "tensor_parallel_size": 1,
        "qwen_replay_backend": "native_fa3_v1", "dispatch_mode": "bounded_async",
        "max_running_requests": 32, "async_queue_size": 64,
        "beta_base": beta_base, "schedule": "constant", "seed": 11,
        "response_token_cap": 8192, "total_rollout_iterations": 109,
        "invocation_iterations": 1, "optimizer_steps": 2, "ema_updates": 1,
        "full_dose_gradient_gate_enabled": True, "completion_gate_enabled": False,
        "opd_grpo_ratio_range_gate_enabled": False,
        "validation_enabled": False, "checkpoint_after_iteration": 1,
        "maximum_process_seconds": 1700, "automatic_retry": False,
    }


def _capacity_experiment_name(beta_base):
    return "qwen3_g8_full_dose_capacity" + ("_beta0p1" if beta_base == 0.1 else "")


def capacity_overrides(assets_root, run_root, *, beta_base=0.001):
    from .qwen_training import profile_overrides

    beta_base = validate_capacity_beta_base(beta_base)
    root = Path(run_root).resolve()
    return profile_overrides("hybrid", 2, assets_root, root / "training", replay_backend="native_fa3_v1") + [
        f"algorithm.opd.beta_base={beta_base}", "algorithm.opd.schedule=constant",
        "actor_rollout_ref.rollout.dispatch_mode=bounded_async",
        "actor_rollout_ref.rollout.max_running_requests=32",
        "actor_rollout_ref.rollout.async_queue_size=64",
        "++trainer.training_capacity_mode=true",
        f"++trainer.training_capacity_beta_base={beta_base}",
        "++trainer.training_capacity_output=" + str(root / "measurement.json"),
        "trainer.max_rollout_iterations_per_invocation=1",
        "trainer.rollout_integrity.full_dose_gradient_gate_enabled=true",
        "trainer.rollout_integrity.completion_gate_enabled=false",
        "trainer.val_before_train=false", "trainer.test_freq=-1", "trainer.save_freq=-1",
        "trainer.log_val_generations=0", "trainer.resume_mode=disable",
        "trainer.project_name=opd-qwen3-training-capacity",
        "trainer.experiment_name=" + _capacity_experiment_name(beta_base),
        "hydra.run.dir=" + str(root / "hydra"), "hydra.job.chdir=false",
    ]


def classify_failure(error, *, stage, log_tail=""):
    """Keep an underlying CUDA/Ray cause separate from the last driver stage."""
    message = (str(error) + "\n" + str(log_tail)).lower()
    if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
        category = "timeout"
    elif any(value in message for value in ("outofmemoryerror", "out of memory", "oom-kill", "oom_kill")):
        category = "oom"
    elif "full-dose gradient" in message or stage == "full_dose_gradient_gate":
        category = "gradient_gate"
    elif any(value in message for value in ("replay integrity", "replay acceptance", "replay ratio", "replay gate")) or stage == "old_log_prob" or stage.startswith("replay"):
        category = "replay"
    elif stage in ("checkpoint", "checkpoint_authentication"):
        category = "checkpoint"
    elif stage in ("source_authentication", "asset_authentication", "measurement_authentication"):
        category = "authentication"
    elif stage in ("startup", "worker_initialization"):
        category = "startup"
    else:
        category = "execution"
    return {"category": category, "stage": stage, "error_type": type(error).__name__, "error": str(error)[-8192:]}


def finite_snapshot(value):
    """Preserve every scalar metric, including keys after the first 128."""
    if isinstance(value, Mapping):
        return {str(key): finite_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_snapshot(item) for item in value]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, RuntimeError):
            return str(value)[:2048]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return str(value)[:2048]


class CapacityRecorder:
    """Persist progress before expensive RPCs, including partial failure evidence."""

    def __init__(self, path, measurement):
        self.path = Path(path)
        self.measurement = measurement
        self.started = time.monotonic()
        self.stage_started = self.started
        self.stage_started_unix_seconds = time.time()
        self.stage = "startup"
        self.update_state = "not_started"
        self.current = (None, {}, {}, {})
        self.measurement.setdefault("stage_events", [])
        self.persist()

    def persist(self):
        write_manifest_atomic(self.path, finite_snapshot(self.measurement), validator=None)

    def snapshot(self):
        iteration, timing, metrics, meta = self.current
        return finite_snapshot({
            "rollout_iteration": iteration, "stage": self.stage,
            "optimizer_update_state": self.update_state,
            "optimizer_updates_completed": {"not_started": False, "completed": True}.get(self.update_state),
            "stage_elapsed_seconds": time.monotonic() - self.stage_started,
            "stage_started_unix_seconds": self.stage_started_unix_seconds,
            "trajectory_count": meta.get("capacity_rollout_trajectory_count"),
            "timing_s": timing, "metrics": metrics,
            "rollout_timing": meta.get("rollout_timing", {}),
            "actor_update_timing": meta.get("actor_update_timing", {}),
        })

    def enter(self, stage, iteration=None, timing=None, metrics=None, meta_info=None, *, update_state=None):
        now = time.monotonic()
        events = self.measurement["stage_events"]
        if events:
            events[-1]["elapsed_seconds"] = now - self.stage_started
        self.stage, self.stage_started = stage, now
        self.stage_started_unix_seconds = time.time()
        if update_state is not None:
            self.update_state = update_state
        self.current = (iteration, timing if timing is not None else {}, metrics if metrics is not None else {}, meta_info if meta_info is not None else {})
        events.append({"stage": stage, "elapsed_from_start_seconds": now - self.started, "optimizer_update_state": self.update_state})
        self.measurement["progress"] = self.snapshot()
        self.persist()

    def fail(self, error, *, diagnostics=None):
        progress = self.snapshot()
        failure = {**progress, **classify_failure(error, stage=self.stage)}
        if diagnostics is not None:
            failure["diagnostics"] = finite_snapshot(diagnostics)
        self.measurement.update(status="failed", failure=failure, progress=progress)
        self.persist()


def validate_capacity_iteration(record, *, beta_base=0.001):
    """Accept exactly one full-dose iteration, independently of warmup pilots."""
    beta_base = validate_capacity_beta_base(beta_base)
    if not isinstance(record, Mapping) or record.get("rollout_iteration") != 0:
        raise ValueError("capacity requires rollout iteration zero")
    if type(record.get("trajectory_count")) is not int or record["trajectory_count"] != 512:
        raise ValueError("capacity requires 512 actual rollout trajectories")
    metrics = record.get("metrics", {})

    def number(name, source=metrics):
        value = source.get(name)
        if isinstance(value, (bool, str)) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("capacity requires finite metric " + name)
        return value

    for name, expected in {
        "integrity/continuous_replay_active": 1, "replay/fallback_count": 0,
        "trainer/rollout_iteration": 0, "trainer/optimizer_steps_this_iteration": 2,
        "trainer/optimizer_step": 2, "opd/ema_updates_this_iteration": 1,
        "opd/ema_update_count": 1,
    }.items():
        if number(name) != expected:
            raise ValueError(f"capacity {name} must equal {expected}")
    if not math.isclose(number("opd/beta_effective"), beta_base, rel_tol=1e-12, abs_tol=0):
        raise ValueError(f"capacity requires full beta {beta_base} at iteration zero")
    if not 0 <= number("replay/ratio_abs_error_max") <= 1e-4:
        raise ValueError("capacity replay tolerance exceeded")
    if not 0 < number("latent/soft_to_hard_rate") <= 1:
        raise ValueError("capacity requires a real soft-to-hard boundary")
    for name in ("opd/latent_slot_count", "opd/answer_slot_count", "grad/opd_norm", "grad/grpo_norm"):
        if number(name) <= 0:
            raise ValueError("capacity requires positive " + name)
    if number("grad/total_norm") < 0:
        raise ValueError("capacity total gradient norm must be nonnegative")
    # Relative component strength is a diagnostic, not an acceptance range.
    # The trainer still checks gradient integrity before publishing a checkpoint.
    ratio = number("grad/opd_norm") / number("grad/grpo_norm")
    if not math.isfinite(ratio) or not 0 <= number("actor/gradient_clipfrac") <= 0.5:
        raise ValueError("capacity full-dose gradient integrity gate failed")
    ranks = record.get("actor_update_timing", {}).get("ranks")
    if not isinstance(ranks, list) or len(ranks) != 2 or {row.get("rank") for row in ranks} != {0, 1}:
        raise ValueError("capacity requires exactly ranks zero and one")
    for row in ranks:
        if type(row.get("rank")) is not int:
            raise ValueError("capacity rank identity must be an integer")
        for name, expected in (("optimizer_steps", 2), ("ema_updates_this_iteration", 1), ("ema_update_count", 1)):
            if number(name, row) != expected:
                raise ValueError("capacity rank cadence differs at " + name)
        if number("teacher_seconds", row) <= 0:
            raise ValueError("capacity teacher timing must be positive")
        for name in ("policy_update_seconds", "worker_update_seconds", "max_memory_allocated_gib", "max_memory_reserved_gib"):
            if number(name, row) < 0:
                raise ValueError("capacity rank timing/memory must be nonnegative")
    return {"accepted": True, "opd_grpo_support_gradient_ratio": ratio,
            "opd_grpo_ratio_range_gate_enabled": False, "full_dose_gradient_gate": "passed"}


def _atomic_text(path, value):
    path = Path(path)
    handle, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def publish_report(root, report):
    root = Path(root)
    write_manifest_atomic(root / "capacity.json", finite_snapshot(report), validator=None)
    failure = report.get("failure", {})
    measurement = report.get("measurement", {})
    telemetry = report.get("resource_telemetry", {})
    progress = measurement.get("failure", measurement.get("progress", {}))
    iterations = measurement.get("iterations", [])
    record = iterations[0] if iterations else progress
    timings = record.get("timing_s", {})
    text = (
        "# Preliminary Qwen3 G8 capacity check\n\n"
        f"Status: **{report['status']}**. One full-dose hybrid iteration; 64 prompts × 8 samples, two H100s.\n\n"
        f"Job: `{report.get('job_id')}`. W&B: `{report.get('wandb_run_id')}`.\n\n"
        f"Elapsed: {report.get('elapsed_seconds', 0):.1f} seconds. "
        f"Accepted iterations: {len(measurement.get('iterations', []))}.\n\n"
        f"Failure category: {failure.get('category', 'none')}; stage: {failure.get('stage', 'none')}. "
        f"Optimizer update state: {measurement.get('progress', {}).get('optimizer_update_state', 'not_observed')}.\n\n"
        f"Checkpoint authenticated: {report.get('checkpoint_authenticated', False)}.\n\n"
        f"This checks G8 capacity at beta {report.get('contract', {}).get('beta_base', 'unverified')}. "
        "It is not a runtime estimate, next-update resume acceptance, or full training. "
        "No validation or automatic retry is performed. See capacity.json for configuration, stage timings, resource peaks, and source hashes.\n"
    )
    text += "\n| Measurement | Value |\n|---|---|\n"
    for name, value in (
        ("Startup seconds", measurement.get("startup_seconds")),
        ("Rollout seconds", timings.get("gen")),
        ("Replay seconds", timings.get("old_log_prob")),
        ("Reference seconds", timings.get("ref")),
        ("Actor update seconds (includes teacher)", timings.get("update_actor")),
        ("Checkpoint seconds", timings.get("save_checkpoint")),
        ("Failed stage seconds (at observation if killed)", failure.get("stage_elapsed_seconds")),
        ("Peak HBM GiB per GPU (sampled)", telemetry.get("peak_hbm_gib_per_gpu")),
        ("Peak host RAM GiB (node)", telemetry.get("peak_host_ram_gib")),
        ("Actor rank CUDA peaks", record.get("actor_update_timing", {}).get("ranks")),
    ):
        if name == "Actor rank CUDA peaks" and isinstance(value, list):
            value = [{key: row.get(key) for key in ("rank", "max_memory_allocated_gib", "max_memory_reserved_gib")} for row in value]
        text += f"| {name} | {json.dumps(value, sort_keys=True) if value is not None else 'incomplete'} |\n"
    source = report.get("source", {})
    text += f"\nParent source: `{source.get('parent_commit', 'unverified')}`. Fork source: `{source.get('fork_commit', 'unverified')}`.\n"
    _atomic_text(root / "REPORT.md", text)
    for name in ("capacity.json", "REPORT.md"):
        _atomic_text(root / (name + ".sha256"), file_sha256(root / name) + "  " + name + "\n")


class CapacityRunner:
    def __init__(self, args):
        self.args = args
        self.beta_base = validate_capacity_beta_base(getattr(args, "beta_base", 0.001))
        self.started = time.monotonic()
        self.deadline = self.started + args.time_limit_seconds
        self.root = args.run_root.resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.child = None
        self.report = {"schema_version": 1, "status": "running", "contract": capacity_contract(self.beta_base), "job_id": os.environ.get("SLURM_JOB_ID"), "submission_id": os.environ.get("OPD_QTB_SUBMISSION_ID"), "time_limit_seconds": args.time_limit_seconds, "checkpoint_authenticated": False, "measurement_authenticated": False}
        self.stage = "startup"

    def _terminate(self):
        if self.child is None:
            return
        # The process leader can exit while Ray/SGLang descendants survive.
        # Always kill the complete session, even after a successful wait().
        try:
            os.killpg(self.child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(self.child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.child.wait(timeout=5)
        self.child = None

    def run(self):
        from .icl_resource_monitor import ResourceMonitor
        from .qwen_training import verify
        from .training_benchmark import source_identity, validate_phase_measurement

        monitor = ResourceMonitor(interval_seconds=2)
        old_handlers = {}
        def interrupted(signum, frame):
            raise TimeoutError(f"capacity deadline/signal {signum}; no retry requested")
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            old_handlers[signum] = signal.signal(signum, interrupted)
        # Reserve 45 seconds for process-group termination, telemetry and report.
        signal.setitimer(signal.ITIMER_REAL, max(0.01, self.deadline - time.monotonic() - 45))
        error = None
        measurement_path = self.root / "measurement.json"
        try:
            monitor.start()
            self.stage = "source_authentication"
            fork = Path(__file__).resolve().parents[1]
            source = self.report["source"] = source_identity(fork)
            self.stage = "asset_authentication"
            assets = self.report["assets"] = verify(self.args.assets_root)
            overrides = capacity_overrides(self.args.assets_root, self.root, beta_base=self.beta_base)
            self.report["overrides"] = overrides
            run_id = os.environ.get("WANDB_RUN_ID") or "qcap-" + canonical_sha256({"source": source, "job": self.report["job_id"], "root": str(self.root), "contract": self.report["contract"]})[:24]
            self.report["wandb_run_id"] = run_id
            environment = dict(os.environ)
            environment.update(WANDB_RUN_ID=run_id, WANDB_NAME=_capacity_experiment_name(self.beta_base), WANDB_MODE="online", WANDB_RESUME="never", OPD_CAPACITY_STARTED=str(time.time()))
            command = [sys.executable, "-m", "verl.trainer.main_ppo", *overrides]
            write_manifest_atomic(self.root / "invocation.json", {"command": command, "source": source, "assets": assets, "wandb_run_id": run_id, "contract": capacity_contract(self.beta_base)}, validator=None)
            self.stage = "startup"
            publish_report(self.root, self.report)
            with (self.root / "trainer.log").open("w") as log:
                self.child = subprocess.Popen(command, cwd=fork / "verl-0.4.x", env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                code = self.child.wait(timeout=max(0.01, self.deadline - time.monotonic() - 45))
            if code:
                raise RuntimeError(f"capacity trainer exited with code {code}")
            self.stage = "measurement_authentication"
            measured = json.loads(measurement_path.read_text())
            self.report["measurement"] = measured
            validate_phase_measurement(measured, phase="capacity", variant="bounded_async32", batches=[], overrides=overrides, source=source, assets=assets, run_id=run_id)
            self.report["measurement_authenticated"] = True
            if measured.get("status") != "complete" or measured.get("wandb_online") is not True or measured.get("wandb_finished") is not True:
                raise ValueError("capacity trainer did not finish with online W&B publication")
            iterations = measured.get("iterations", [])
            if len(iterations) != 1:
                raise ValueError("capacity requires exactly one accepted iteration")
            validate_capacity_iteration(iterations[0], beta_base=self.beta_base)
            self.stage = "checkpoint_authentication"
            checkpoint = measured.get("checkpoint", {})
            if checkpoint.get("authenticated") is not True or checkpoint.get("payload_rehashed") is not True:
                raise ValueError("capacity has no authenticated checkpoint")
            manifest_path = Path(checkpoint["path"]) / "checkpoint_manifest.json"
            if manifest_path.parent != self.root / "training" / "global_step_1" or manifest_path.parent.is_symlink() or manifest_path.is_symlink() or file_sha256(manifest_path) != checkpoint["manifest_sha256"]:
                raise ValueError("capacity checkpoint manifest identity differs")
            if json.loads(manifest_path.read_text()) != checkpoint.get("manifest"):
                raise ValueError("capacity checkpoint manifest differs from authenticated measurement")
            self.report["checkpoint_authenticated"] = True
            self.report["status"] = "complete"
        except BaseException as failure:
            error = failure
            self.report["status"] = "failed"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            # A second Slurm warning must not interrupt cleanup and lose evidence.
            for signum in (signal.SIGTERM, signal.SIGINT):
                signal.signal(signum, signal.SIG_IGN)
            try:
                self._terminate()
            except BaseException as cleanup_error:
                self.report["cleanup_error"] = str(cleanup_error)
                error = error or cleanup_error
                self.report["status"] = "failed"
            try:
                self.report["resource_telemetry"] = monitor.stop().to_dict()
            except BaseException as monitor_error:
                self.report["resource_telemetry"] = {"monitor_error": str(monitor_error), "host_metrics_available": False, "gpu_metrics_available": False}
                error = error or monitor_error
                self.report["status"] = "failed"
            if measurement_path.is_file() and not measurement_path.is_symlink():
                try:
                    self.report["measurement"] = json.loads(measurement_path.read_text())
                    self.report["measurement_sha256"] = file_sha256(measurement_path)
                    if all(key in self.report for key in ("overrides", "source", "assets", "wandb_run_id")):
                        validate_phase_measurement(self.report["measurement"], phase="capacity", variant="bounded_async32", batches=[], overrides=self.report["overrides"], source=self.report["source"], assets=self.report["assets"], run_id=self.report["wandb_run_id"])
                        self.report["measurement_authenticated"] = True
                except (ValueError, OSError, TypeError, KeyError, RuntimeError) as measurement_error:
                    self.report["measurement_read_error"] = str(measurement_error)
                    self.report["measurement_authenticated"] = False
                    error = error or measurement_error
                    self.report["status"] = "failed"
            if error is not None:
                measured = self.report.get("measurement", {})
                progress = measured.get("failure", measured.get("progress", {}))
                stage = self.stage if self.stage.endswith("authentication") else progress.get("stage", self.stage)
                log_path = self.root / "trainer.log"
                tail = ""
                if log_path.is_file():
                    with log_path.open("rb") as stream:
                        stream.seek(max(0, log_path.stat().st_size - 131072))
                        tail = stream.read().decode(errors="replace")
                self.report["failure"] = classify_failure(error, stage=stage, log_tail=tail + str(measured.get("failure", "")))
                observed = time.time()
                self.report["failure"]["observed_unix_seconds"] = observed
                if isinstance(measured.get("failure"), Mapping):
                    self.report["failure"]["stage_elapsed_seconds"] = measured["failure"].get("stage_elapsed_seconds")
                    self.report["failure"]["stage_timing_scope"] = "trainer failure callback"
                elif isinstance(progress.get("stage_started_unix_seconds"), (int, float)):
                    self.report["failure"]["stage_elapsed_seconds"] = max(0, observed - progress["stage_started_unix_seconds"])
                    self.report["failure"]["stage_timing_scope"] = "parent observation minus last durable stage start; includes termination cleanup"
            self.report["elapsed_seconds"] = time.monotonic() - self.started
            self.report["artifact_sha256"] = {name: file_sha256(self.root / name) for name in ("invocation.json", "measurement.json", "trainer.log") if (self.root / name).is_file()}
            publish_report(self.root, self.report)
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
        return 0 if self.report["status"] == "complete" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--time-limit-seconds", type=int, default=1700)
    parser.add_argument("--beta-base", type=float, choices=CAPACITY_BETA_BASES, default=0.001)
    args = parser.parse_args(argv)
    if not 60 <= args.time_limit_seconds <= 1700:
        parser.error("capacity process budget must be between 60 and 1700 seconds")
    return CapacityRunner(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
