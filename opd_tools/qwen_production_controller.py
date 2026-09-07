"""Run bounded correctness admission, then authenticated Qwen3 production.

Only clean iteration-boundary continuations return 75. Failed rollouts,
prologues and uncertain submission outcomes never request a retry.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .manifest import canonical_sha256, file_sha256, validate_sealed_content, write_manifest_atomic
from .training_capacity import classify_failure, finite_snapshot


def write_json(path, value, *, seal=False):
    payload = finite_snapshot(value)
    if seal:
        payload = {k: v for k, v in payload.items() if k != "manifest_content_sha256"}
        payload["manifest_content_sha256"] = canonical_sha256(payload)
    write_manifest_atomic(Path(path), payload, validator=validate_sealed_content if seal else None)
    return payload


def read_json(path, *, sealed=False):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular JSON artifact: {path}")
    value = json.loads(path.read_text())
    if sealed:
        validate_sealed_content(value)
    return value


def number(mapping, name):
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"missing finite production evidence: {name}")
    return float(value)


def validate_iteration(record, *, arm_id, phase):
    """Require four-rank cadence and real update evidence, without dose range gates."""
    iteration = record.get("rollout_iteration")
    if type(iteration) is not int or iteration < 0:
        raise ValueError("invalid rollout iteration")
    effective_arm = "softgrpo_math_s11" if phase == "zero_dose" else arm_id
    hard = effective_arm == "hardgrpo_math_s11"
    standalone = effective_arm == "softopd_math_s11"
    hybrid = effective_arm.startswith("softgrpo_math_opd")
    enabled = standalone or hybrid
    ema = enabled and effective_arm != "softgrpo_math_opd_current_s11"
    beta_base = 0.1 if effective_arm.endswith("beta0p1_s11") else 1.0
    beta = beta_base if enabled and (standalone or phase == "full_dose") else beta_base * min(1.0, iteration / 11) if enabled else 0.0
    metrics = record.get("metrics", {})
    expected = {
        "trainer/rollout_iteration": iteration,
        "trainer/optimizer_steps_this_iteration": 2,
        "trainer/optimizer_step": 2 * (iteration + 1),
    }
    for key, value in expected.items():
        if number(metrics, key) != value:
            raise ValueError(f"incorrect production update cadence: {key}")
    if record.get("trajectory_count") != (64 if standalone else 512):
        raise ValueError("production rollout trajectory count differs")
    if number(metrics, "grad/total_norm") < 0:
        raise ValueError("invalid student gradient norm")
    if not 0 <= number(metrics, "actor/gradient_clipfrac") <= 1:
        raise ValueError("invalid clipping-frequency diagnostic")
    if not hard:
        if number(metrics, "integrity/continuous_replay_active") != 1 or number(metrics, "replay/fallback_count") != 0:
            raise ValueError("native continuous replay is required without fallback")
        if not 0 <= number(metrics, "replay/ratio_abs_error_max") <= 1e-4:
            raise ValueError("replay acceptance exceeds existing 1e-4 tolerance")
        if phase != "production" and not 0 < number(metrics, "latent/soft_to_hard_rate") <= 1:
            raise ValueError("prologue must observe a real soft-to-hard transition")
    if enabled:
        if not math.isclose(number(metrics, "opd/beta_effective"), beta, rel_tol=1e-12, abs_tol=0):
            raise ValueError("production OPD schedule differs")
        if number(metrics, "opd/ema_updates_this_iteration") != int(ema):
            raise ValueError("incorrect EMA cadence")
        if number(metrics, "opd/ema_update_count") != (iteration + 1 if ema else 0):
            raise ValueError("incorrect cumulative EMA count")
        if beta > 0:
            slots = number(metrics, "opd/latent_slot_count") + number(metrics, "opd/answer_slot_count")
            empty = number(metrics, "opd/selected_slots") == 0 and effective_arm == "softgrpo_math_opd_posadv_s11"
            if phase != "production" and not empty and (slots <= 0 or number(metrics, "grad/opd_norm") <= 0):
                raise ValueError("active OPD requires eligible positions and positive finite gradients")
            if phase == "full_dose" and not empty:
                if number(metrics, "opd/latent_slot_count") <= 0 or number(metrics, "opd/answer_slot_count") <= 0:
                    raise ValueError("full-dose prologue must exercise latent and answer OPD")
                if hybrid and number(metrics, "grad/grpo_norm") <= 0:
                    raise ValueError("full-dose prologue requires active GRPO gradients")
    ranks = record.get("actor_update_timing", {}).get("ranks")
    if not isinstance(ranks, list) or len(ranks) != 4 or {row.get("rank") for row in ranks} != set(range(4)):
        raise ValueError("production requires four observed training ranks")
    for row in ranks:
        if type(row.get("rank")) is not int or number(row, "optimizer_steps") != 2:
            raise ValueError("per-rank optimizer cadence differs")
        if number(row, "ema_updates_this_iteration") != int(ema):
            raise ValueError("per-rank EMA cadence differs")
        if number(row, "ema_update_count") != (iteration + 1 if ema else 0):
            raise ValueError("per-rank cumulative EMA count differs")
        for key in ("worker_update_seconds", "policy_update_seconds", "max_memory_allocated_gib", "max_memory_reserved_gib"):
            if number(row, key) < 0:
                raise ValueError("invalid rank timing or memory")
    return {"accepted": True, "effective_beta": beta, "ratio_range_gate": False,
            "clipping_frequency_gate": False, "completion_rate_gate": False}


def requires_semantic_checkpoint(manifest):
    return manifest.get("profile_id") == "qwen3-math-seven-arm-lora-fa3-v1"


def authenticate_checkpoint(root, step, *, arm_id, teacher=True, require_semantic=False):
    from verl.trainer.ppo.ray_trainer import _verify_checkpoint
    manifest = _verify_checkpoint(str(Path(root) / f"global_step_{step}"), require_semantic=require_semantic)
    for key, value in {"global_step": step, "optimizer_step": 2 * step, "world_size": 4,
                       "total_rollout_iterations": 109, "next_rollout_iteration": step}.items():
        if manifest.get(key) != value:
            raise ValueError(f"checkpoint production contract differs at {key}")
    ema = teacher and arm_id not in {"hardgrpo_math_s11", "softgrpo_math_s11", "softgrpo_math_opd_current_s11"}
    if ema:
        base = Path(root) / f"global_step_{step}" / "actor/opd_teacher"
        for rank in range(4):
            state = read_json(base / f"ema_state_world_size_4_rank_{rank}.json")
            if state.get("last_rollout_iteration") != step - 1 or state.get("update_count") != step:
                raise ValueError("checkpoint EMA rank cadence differs")
        if not manifest.get("opd_teacher_tree_sha256"):
            raise ValueError("checkpoint lacks required EMA teacher")
    elif manifest.get("opd_teacher_tree_sha256") is not None:
        raise ValueError("non-EMA arm unexpectedly saved an EMA teacher")
    return manifest


def compare_checkpoints(left, right, *, include_teacher, require_semantic=False):
    semantic = left.get("semantic_identity"), right.get("semantic_identity")
    if require_semantic or any(value is not None for value in semantic):
        if any(not isinstance(value, dict) or value.get("schema") != "qwen_semantic_v1" for value in semantic):
            raise ValueError("exact next-update comparison requires matching semantic schemas")
        fields = ["actor_model_optimizer_scheduler_sha256", "worker_rng_sha256", "driver_rng_sha256", "dataloader_sha256"]
        if include_teacher:
            fields.append("teacher_model_ema_sha256")
        values = {"rollout_trajectory_sha256": left.get("rollout_trajectory_sha256")}
        if not values["rollout_trajectory_sha256"] or values["rollout_trajectory_sha256"] != right.get("rollout_trajectory_sha256"):
            raise ValueError("exact next-update parity failed: rollout_trajectory_sha256 differs")
        for field in fields:
            a, b = semantic[0].get(field), semantic[1].get(field)
            # An absent teacher is valid for hard/soft/current-teacher arms;
            # every other component must provide a SHA-256 identity.
            if (field not in semantic[0] or field not in semantic[1] or a != b
                    or (a is None and field != "teacher_model_ema_sha256")
                    or (a is not None and (not isinstance(a, str) or len(a) != 64
                                          or any(c not in "0123456789abcdef" for c in a)))):
                raise ValueError(f"exact next-update parity failed: semantic {field} differs or is missing")
            values[field] = a
        return {"passed": True, "schema": "qwen_semantic_v1",
                "compared_fields": list(values), "values": values}
    fields = ["rollout_trajectory_sha256", "actor_model_optimizer_tree_sha256"]
    if include_teacher:
        fields.append("opd_teacher_tree_sha256")
    for field in fields:
        if left.get(field) != right.get(field):
            raise ValueError(f"exact next-update parity failed: {field} differs")
    return {"passed": True, "compared_fields": fields, "values": {key: left.get(key) for key in fields}}


def select_arm(manifest, arm):
    matches = [row for row in manifest["arms"] if row["arm_id"] == arm]
    if len(matches) != 1:
        raise ValueError("arm missing or duplicated in production manifest")
    return matches[0]


def verify_gpu_allocation(manifest_path, manifest, row, *, job_id, restart_count):
    """Bind four real GPU preflight results to this source and Slurm segment."""
    if not requires_semantic_checkpoint(manifest):
        return None
    if not isinstance(job_id, str) or not job_id.isdecimal() or type(restart_count) is not int or restart_count < 0:
        raise ValueError("GPU allocation requires a valid Slurm job/restart identity")
    relative = f"segments/allocation-{job_id}-{restart_count}.json"
    path = Path(row["run_root"]) / relative
    record = read_json(path)
    if not isinstance(record, dict) or type(record.get("restart_count")) is not int:
        raise ValueError("GPU allocation record is malformed")
    expected = {"status": "passed", "job_id": job_id, "restart_count": restart_count,
                "arm_id": row["arm_id"], "manifest_sha256": file_sha256(manifest_path)}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("GPU allocation does not match the current job, source manifest, and arm")
    runtime = record.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("GPU allocation runtime record is malformed")
    unified = runtime.get("unified")
    if not isinstance(unified, dict):
        raise ValueError("GPU allocation lacks the sealed unified runtime")
    validate_sealed_content(unified)
    if (unified.get("manifest_content_sha256") != manifest.get("runtime_manifest", {}).get("manifest_content_sha256")
            or unified.get("build_record", {}).get("source") != {
                "parent_commit": manifest["parent_commit"], "fork_commit": manifest["fork_commit"]}):
        raise ValueError("GPU allocation runtime differs from the source-bound study runtime")
    devices, acceptance = record.get("devices"), runtime.get("native_fa3_acceptance")
    if (not isinstance(devices, list) or len(devices) != 4 or not isinstance(acceptance, list)
            or len(acceptance) != 4 or any(not isinstance(item, dict) or type(item.get("device")) is not int for item in acceptance)
            or {item["device"] for item in acceptance} != set(range(4))):
        raise ValueError("GPU allocation requires native FA3 acceptance on four distinct devices")
    required = {"native_fa3_forward_backward": True, "fresh_process_exact_match": True, "native_kernel": "opd_fa3._C",
                "dtype": "bfloat16", "head_dimension": 128,
                "vllm_flash_attention_version": 3, "vllm_kernel": "vllm._vllm_fa3_C",
                "packed_causal_gradient_isolation": True, "long_packed_lengths": [8192, 8192]}
    for item in acceptance:
        device = devices[item["device"]]
        if (not isinstance(device, dict) or "H100" not in str(device.get("name"))
                or item.get("name") != device.get("name") or any(item.get(key) != value for key, value in required.items())
                or any(item.get(key) is not value for key, value in required.items() if isinstance(value, bool))):
            raise ValueError("GPU allocation native FA3 kernel, shape, or device acceptance differs")
        hashes = item.get("long_output_gradient_sha256")
        if not isinstance(hashes, list) or len(hashes) != 4 or any(
                not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in hashes):
            raise ValueError("GPU allocation lacks exact long-context output/gradient evidence")
        lora = item.get("native_lora", {})
        if not isinstance(lora, dict) or any(lora.get(key) != value for key, value in {
                "schema_version": 1, "status": "passed", "optimizer_steps": 2,
                "frozen_base_unchanged": True, "dense_export_weight_exact": True,
                "dense_export_projection_exact": True, "teacher_or_training_data_used": False}.items()):
            raise ValueError("GPU allocation lacks controlled native LoRA acceptance")
        norms = lora.get("adapter_gradient_norms")
        if (not isinstance(norms, list) or len(norms) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in norms)
                or number(lora.get("effective_update", {}), "changed_elements") <= 0):
            raise ValueError("GPU allocation lacks finite controlled LoRA gradients and an effective BF16 update")
    return {"path": relative, "sha256": file_sha256(path), "job_id": job_id,
            "restart_count": restart_count, "device_count": 4}


def verify_gpu_evidence(manifest_path, manifest, row, record, *, restart_count):
    if not requires_semantic_checkpoint(manifest):
        return
    allocation = verify_gpu_allocation(manifest_path, manifest, row, job_id=record.get("job_id"), restart_count=restart_count)
    if (record.get("gpu_allocation") != allocation
            or record.get("evidence_files", {}).get(allocation["path"]) != allocation["sha256"]):
        raise ValueError("GPU allocation acceptance is missing from sealed evidence")


def verify_admission(manifest_path, manifest, row):
    path = Path(row["run_root"]) / "admission.json"
    admission = read_json(path, sealed=True)
    if (admission.get("status") != "passed" or admission.get("arm_id") != row["arm_id"]
            or admission.get("submission_manifest_sha256") != file_sha256(manifest_path)
            or admission.get("parent_commit") != manifest["parent_commit"]
            or admission.get("fork_commit") != manifest["fork_commit"]):
        raise ValueError("missing source-bound exact-resume admission")
    # Check durable comparison evidence, not only a mutable boolean.
    for relative, digest in admission["evidence_files"].items():
        path = Path(row["run_root"]) / relative
        if path.is_symlink() or not path.is_file() or file_sha256(path) != digest:
            raise ValueError("admission evidence changed")
    if admission["resume_parity"].get("passed") is not True:
        raise ValueError("exact next-update resume was not accepted")
    if requires_semantic_checkpoint(manifest) and admission["resume_parity"].get("schema") != "qwen_semantic_v1":
        raise ValueError("this production profile requires semantic next-update admission")
    verify_gpu_evidence(manifest_path, manifest, row, admission, restart_count=0)
    return admission


def verify_continuation(manifest_path, manifest, row):
    admission = verify_admission(manifest_path, manifest, row)
    continuation = read_json(Path(row["run_root"]) / "continuation.json", sealed=True)
    if (continuation.get("arm_id") != row["arm_id"] or continuation.get("submission_manifest_sha256") != file_sha256(manifest_path)
            or continuation.get("admission_sha256") != admission["manifest_content_sha256"]):
        raise ValueError("continuation is not bound to accepted source and arm")
    step = continuation.get("global_step")
    if type(step) is not int or not 0 < step < 109:
        raise ValueError("invalid continuation boundary")
    verify_gpu_evidence(manifest_path, manifest, row, continuation, restart_count=continuation.get("restart_count"))
    checkpoint = authenticate_checkpoint(row["phases"]["production"]["run_dir"], step, arm_id=row["arm_id"],
                                         require_semantic=requires_semantic_checkpoint(manifest))
    path = Path(row["phases"]["production"]["run_dir"]) / f"global_step_{step}" / "checkpoint_manifest.json"
    if checkpoint.get("reason") != "requeue_signal" or file_sha256(path) != continuation.get("checkpoint_manifest_sha256"):
        raise ValueError("continuation does not identify a clean signal checkpoint")
    return continuation


class ProductionController:
    def __init__(self, args):
        from .qwen_production import verify_manifest, PRODUCTION_PROJECT
        self.args = args
        self.manifest = verify_manifest(args.manifest)
        self.row = select_arm(self.manifest, args.arm)
        self.root = Path(self.row["run_root"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.child = None
        self.phase = "initialization"
        self.started = time.monotonic()
        self.deadline = self.started + args.prologue_limit_seconds - max(0.0, time.time() - float(os.environ.get("OPD_PROLOGUE_STARTED_EPOCH", time.time())))
        self.restart = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
        self.report = {"schema_version": 1, "arm_id": args.arm, "status": "running",
                       "job_id": os.environ.get("SLURM_JOB_ID"), "restart_count": self.restart,
                       "submission_manifest_sha256": file_sha256(args.manifest),
                       "parent_commit": self.manifest["parent_commit"], "fork_commit": self.manifest["fork_commit"],
                       "wandb_run_id": self.row["wandb_run_id"],
                       "wandb_project": self.row.get("wandb_project", PRODUCTION_PROJECT), "phases": {}}
        self.report_path = self.root / f"segment-{self.restart}.json"
        if self.report_path.exists():
            raise ValueError("this allocation segment was already attempted; inspect instead of retrying")
        self.allocation_evidence = verify_gpu_allocation(
            args.manifest, self.manifest, self.row, job_id=self.report["job_id"], restart_count=self.restart)
        if self.allocation_evidence is not None:
            self.report["gpu_allocation"] = self.allocation_evidence

    def persist(self):
        self.report.update(current_phase=self.phase, elapsed_seconds=time.monotonic() - self.started,
                           observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(self.report_path, self.report)

    def terminate_child(self):
        if self.child is None:
            return
        child = self.child
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=5)
        self.child = None

    def invoke(self, phase, *, resume_from_path=None):
        from .icl_resource_monitor import ResourceMonitor
        from .qwen_production import phase_command, PRODUCTION_PROJECT
        self.phase = phase
        details = self.row["phases"][phase]
        directory, output = Path(details["directory"]), Path(details["output"])
        if phase == "production" and self.restart:
            output = output.with_name(f"measurement-{self.restart}.json")
        directory.mkdir(parents=True, exist_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        suffix = f"-{self.restart}" if phase == "production" else ""
        log_path = directory / f"trainer-{phase}{suffix}.log"
        if log_path.exists() or (output.exists() and phase != "production"):
            raise ValueError(f"phase already attempted: {phase}")
        run_id = self.row["wandb_run_id"] if phase == "production" else self.row["wandb_run_id"] + "-" + ("resume" if phase in {"split", "resume"} else phase)
        project = self.row.get("wandb_project", PRODUCTION_PROJECT) + ("" if phase == "production" else "-prologue")
        command = phase_command(self.manifest, self.args.arm, phase, resume_from_path=resume_from_path)
        if phase == "production" and self.restart:
            command.append("++trainer.production_output=" + json.dumps(str(output)))
        environment = dict(os.environ)
        environment.update(WANDB_RUN_ID=run_id, WANDB_MODE="online", WANDB_RESUME="allow" if resume_from_path else "never",
                           WANDB_NAME=self.args.arm if phase == "production" else self.args.arm + "-" + phase,
                           WANDB_PROJECT=project,
                           OPD_PRODUCTION_STARTED=str(time.time()))
        if phase == "production":
            command.append("trainer.requeue_signal_file=" + json.dumps(str(self.args.signal_file)))
        invocation = {"phase": phase, "command": command, "wandb_run_id": run_id, "wandb_project": project,
                      "submission_manifest_sha256": self.report["submission_manifest_sha256"]}
        invocation_path = directory / f"invocation-{phase}{suffix}.json"
        write_json(invocation_path, invocation)
        self.report["phases"][phase] = {"status": "running", "log": str(log_path), "output": str(output),
                                         "wandb_run_id": run_id, "wandb_project": project}
        self.persist()
        monitor = ResourceMonitor(interval_seconds=5)
        started = time.monotonic()
        monitor.start()
        try:
            with log_path.open("x") as log:
                self.child = subprocess.Popen(command, cwd=Path(self.manifest["source_root"]) / "3rdparty/SofT-GRPO/verl-0.4.x",
                                              env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                remaining = None if phase == "production" else max(0.01, self.deadline - time.monotonic() - 30)
                code = self.child.wait(timeout=remaining)
            if code:
                raise RuntimeError(f"{phase} trainer exited with code {code}")
            measured = read_json(output)
            if measured.get("phase") != phase or measured.get("arm_id") != self.args.arm:
                raise ValueError("phase measurement identity differs")
            if measured.get("status") != "complete":
                raise ValueError(f"{phase} did not publish completed measurement")
            if measured.get("wandb_run_id") != run_id or measured.get("wandb_online") is not True or measured.get("wandb_finished") is not True:
                raise ValueError("phase lacks finished online W&B identity")
            records = measured.get("iterations", [])
            if phase != "production":
                expected_indices = {"uninterrupted": [0, 1], "split": [0], "resume": [1], "full_dose": [0], "zero_dose": [0]}[phase]
                if [record.get("rollout_iteration") for record in records] != expected_indices:
                    raise ValueError("prologue invocation did not complete the expected iterations")
            for record in records:
                validate_iteration(record, arm_id=self.args.arm, phase=phase)
            self.report["phases"][phase].update(status="complete", measurement_sha256=file_sha256(output))
            return measured
        except BaseException as error:
            tail = ""
            if log_path.exists():
                with log_path.open("rb") as stream:
                    stream.seek(max(0, log_path.stat().st_size - 131072))
                    tail = stream.read().decode(errors="replace")
            self.report["failure"] = classify_failure(error, stage=phase, log_tail=tail)
            self.report["phases"][phase]["status"] = "failed"
            raise
        finally:
            cleanup_error = None
            try:
                self.terminate_child()
            except BaseException as error:
                cleanup_error = error
                self.report["cleanup_error"] = str(error)
            try:
                telemetry = monitor.stop().to_dict()
            except BaseException as error:
                telemetry = {"error": str(error), "gpu_metrics_available": False, "host_metrics_available": False}
            self.report["phases"][phase].update(wall_seconds=time.monotonic() - started, resource_telemetry=telemetry)
            self.persist()
            if cleanup_error is not None and "failure" not in self.report:
                raise cleanup_error

    def admission(self):
        if (self.root / "admission.json").exists():
            raise ValueError("initial allocation must not reuse an old prologue")
        allocation = verify_gpu_allocation(self.args.manifest, self.manifest, self.row,
                                           job_id=self.report["job_id"], restart_count=self.restart)
        self.invoke("uninterrupted")
        self.invoke("split")
        split_root = Path(self.row["phases"]["split"]["run_dir"])
        authenticate_checkpoint(split_root, 1, arm_id=self.args.arm, require_semantic=requires_semantic_checkpoint(self.manifest))
        self.invoke("resume", resume_from_path=split_root / "global_step_1")
        left = authenticate_checkpoint(self.row["phases"]["uninterrupted"]["run_dir"], 2, arm_id=self.args.arm, require_semantic=requires_semantic_checkpoint(self.manifest))
        right = authenticate_checkpoint(self.row["phases"]["resume"]["run_dir"], 2, arm_id=self.args.arm, require_semantic=requires_semantic_checkpoint(self.manifest))
        parity = compare_checkpoints(left, right, include_teacher=True, require_semantic=requires_semantic_checkpoint(self.manifest))
        result = {"schema_version": 1, "status": "passed", "arm_id": self.args.arm,
                  "submission_manifest_sha256": self.report["submission_manifest_sha256"],
                  "parent_commit": self.manifest["parent_commit"], "fork_commit": self.manifest["fork_commit"],
                  "job_id": self.report["job_id"], "resume_parity": parity}
        if self.args.arm.startswith("softgrpo_math_opd"):
            self.invoke("full_dose")
            authenticate_checkpoint(self.row["phases"]["full_dose"]["run_dir"], 1, arm_id=self.args.arm, require_semantic=requires_semantic_checkpoint(self.manifest))
            self.invoke("zero_dose")
            baseline = authenticate_checkpoint(self.row["phases"]["zero_dose"]["run_dir"], 1, arm_id="softgrpo_math_s11", require_semantic=requires_semantic_checkpoint(self.manifest))
            hybrid = authenticate_checkpoint(self.row["phases"]["uninterrupted"]["run_dir"], 1, arm_id=self.args.arm, require_semantic=requires_semantic_checkpoint(self.manifest))
            result["zero_dose_parity"] = compare_checkpoints(baseline, hybrid, include_teacher=False, require_semantic=requires_semantic_checkpoint(self.manifest))
        if time.monotonic() > self.deadline:
            raise TimeoutError("two-hour correctness prologue exceeded its allocation budget")
        result["wall_seconds"] = time.monotonic() - self.started
        result["evidence_files"] = {}
        for phase in self.report["phases"]:
            output = Path(self.row["phases"][phase]["output"])
            result["evidence_files"][str(output.relative_to(self.root))] = file_sha256(output)
        if allocation is not None:
            if verify_gpu_allocation(self.args.manifest, self.manifest, self.row,
                                     job_id=self.report["job_id"], restart_count=0) != allocation:
                raise ValueError("GPU allocation evidence changed during admission")
            result["gpu_allocation"] = allocation
            result["evidence_files"][allocation["path"]] = allocation["sha256"]
        write_json(self.root / "admission.json", result, seal=True)
        return result

    def run(self):
        old_handlers = {}
        def interrupted(signum, frame):
            if signum == signal.SIGALRM:
                raise TimeoutError("two-hour correctness prologue deadline reached")
            raise RuntimeError(f"production controller interrupted by signal {signum}")
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            old_handlers[signum] = signal.signal(signum, interrupted)
        self.persist()
        try:
            resume = None
            if self.restart:
                continuation = verify_continuation(self.args.manifest, self.manifest, self.row)
                resume = Path(self.row["phases"]["production"]["run_dir"]) / f"global_step_{continuation['global_step']}"
                if self.args.signal_file.exists():
                    self.args.signal_file.unlink()
            else:
                if self.args.signal_file.exists():
                    raise ValueError("initial allocation has a stale checkpoint signal")
                signal.setitimer(signal.ITIMER_REAL, max(0.01, self.deadline - time.monotonic() - 30))
                self.admission()
                signal.setitimer(signal.ITIMER_REAL, 0)
            measured = self.invoke("production", resume_from_path=resume)
            records = measured.get("iterations", [])
            if not records:
                raise ValueError("production returned without an accepted iteration")
            step = records[-1]["rollout_iteration"] + 1
            checkpoint = authenticate_checkpoint(self.row["phases"]["production"]["run_dir"], step, arm_id=self.args.arm, require_semantic=requires_semantic_checkpoint(self.manifest))
            if step == 109:
                self.report["status"] = "complete"
                return 0
            if checkpoint.get("reason") != "requeue_signal":
                raise ValueError("production stopped early without a clean time-limit signal")
            admission = verify_admission(self.args.manifest, self.manifest, self.row)
            path = Path(self.row["phases"]["production"]["run_dir"]) / f"global_step_{step}" / "checkpoint_manifest.json"
            continuation = {
                "schema_version": 1, "arm_id": self.args.arm, "global_step": step,
                "submission_manifest_sha256": self.report["submission_manifest_sha256"],
                "admission_sha256": admission["manifest_content_sha256"],
                "checkpoint_manifest_sha256": file_sha256(path), "job_id": self.report["job_id"],
                "restart_count": self.restart,
            }
            allocation = verify_gpu_allocation(self.args.manifest, self.manifest, self.row,
                                               job_id=self.report["job_id"], restart_count=self.restart)
            if allocation is not None:
                if allocation != self.allocation_evidence:
                    raise ValueError("GPU allocation evidence changed during production")
                continuation["gpu_allocation"] = allocation
                continuation["evidence_files"] = {allocation["path"]: allocation["sha256"]}
            write_json(self.root / "continuation.json", continuation, seal=True)
            self.report["status"] = "continuation_ready"
            return 75
        except BaseException as error:
            self.report["status"] = "failed"
            self.report.setdefault("failure", classify_failure(error, stage=self.phase))
            print(f"Qwen production stopped: {error}", file=sys.stderr)
            return 1
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            self.terminate_child()
            self.persist()
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify-continuation"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--prologue-limit-seconds", type=int, default=7200)
    parser.add_argument("--signal-file", type=Path)
    args = parser.parse_args(argv)
    if args.prologue_limit_seconds != 7200:
        parser.error("production correctness prologue is capped at exactly two hours")
    from .qwen_production import verify_manifest
    if args.command == "verify-continuation":
        manifest = verify_manifest(args.manifest)
        print(json.dumps(verify_continuation(args.manifest, manifest, select_arm(manifest, args.arm)), sort_keys=True))
        return 0
    if args.signal_file is None:
        parser.error("run requires --signal-file")
    return ProductionController(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
