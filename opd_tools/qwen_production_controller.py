"""Run bounded correctness admission, then authenticated Qwen3 production.

Clean iteration-boundary continuations return 75. Production failures may
requeue from a verified checkpoint with a bounded retry budget. Admission
failures and uncertain submission outcomes never request a retry.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from .manifest import canonical_sha256, file_sha256, validate_sealed_content, write_manifest_atomic
from .qwen_acceptance import validate_fsdp_probe, validate_measurement_physical_memory
from .qwen_site import site_from_manifest, validate_artifact_path, validate_scheduler_allocation
from .training_capacity import classify_failure, finite_snapshot

MAX_FAILURE_RETRIES_WITHOUT_PROGRESS = 3


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
    if hard:
        if number(metrics, "integrity/continuous_replay_active") != 0:
            raise ValueError("hard production requires categorical replay")
        if not 0 <= number(metrics, "replay/ratio_abs_error_max") <= 1e-4:
            raise ValueError("categorical replay acceptance exceeds existing 1e-4 tolerance")
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


def validate_measurement(measured, manifest, row, phase, arguments):
    """Independently validate complete phase output and its sealed memory policy."""
    suffix = "resume" if phase in {"split", "resume"} else phase
    run_id = row["wandb_run_id"] + ("" if phase == "production" else "-" + suffix)
    if measured.get("phase") != phase or measured.get("arm_id") != row["arm_id"]:
        raise ValueError("phase measurement identity differs")
    if measured.get("status") != "complete":
        raise ValueError(f"{phase} did not publish completed measurement")
    if (measured.get("wandb_run_id") != run_id or measured.get("wandb_online") is not True
            or measured.get("wandb_finished") is not True):
        raise ValueError("phase lacks finished online W&B identity")
    records = measured.get("iterations")
    if not isinstance(records, list) or not records or any(not isinstance(item, dict) for item in records):
        raise ValueError("phase returned without accepted iteration evidence")
    indices = [record.get("rollout_iteration") for record in records]
    if phase != "production":
        expected = {"uninterrupted": [0, 1], "split": [0], "resume": [1],
                    "full_dose": [0], "zero_dose": [0]}[phase]
        if indices != expected:
            raise ValueError("prologue invocation did not complete the expected iterations")
    elif (any(type(index) is not int or not 0 <= index < 109 for index in indices)
          or indices != list(range(indices[0], indices[0] + len(indices)))):
        raise ValueError("production iteration evidence is not a consecutive bounded segment")
    for record in records:
        validate_iteration(record, arm_id=row["arm_id"], phase=phase)
    validate_measurement_physical_memory(measured, arguments, world_size=4,
                                         required=requires_semantic_checkpoint(manifest))


def authenticate_checkpoint(root, step, *, arm_id, teacher=True, require_semantic=False,
                            expected_provenance=None):
    from verl.trainer.ppo.ray_trainer import _verify_checkpoint
    options = {"require_semantic": require_semantic}
    if expected_provenance is not None:
        options["expected_provenance"] = expected_provenance
    manifest = _verify_checkpoint(str(Path(root) / f"global_step_{step}"), **options)
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


def verify_hard_long_replay(manifest_path, manifest, row, record):
    """Recompute the diagnostic gates and bind their files to this allocation."""
    if row["arm_id"] != "hardgrpo_math_s11":
        return
    evidence = record.get("runtime", {}).get("hard_long_replay_acceptance")
    if not isinstance(evidence, dict):
        raise ValueError("GPU allocation lacks categorical long replay acceptance")
    expected = {"job_id": record["job_id"], "restart_count": record["restart_count"],
                "arm_id": row["arm_id"], "manifest_sha256": file_sha256(manifest_path),
                "parent_commit": manifest["parent_commit"], "fork_commit": manifest["fork_commit"]}
    if (evidence.get("binding") != expected
            or type(evidence.get("binding", {}).get("restart_count")) is not int):
        raise ValueError("categorical long replay belongs to a different source or allocation")
    relative = evidence.get("path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("invalid categorical long replay evidence path")
    root = Path(row["run_root"]).resolve()
    path = root / relative
    segment = root / "segments" / f"preflight-{record['job_id']}-{record['restart_count']}"
    if (path.is_symlink() or not path.resolve().is_relative_to(segment)
            or not path.is_file() or file_sha256(path) != evidence.get("sha256")):
        raise ValueError("categorical long replay evidence changed or escapes its allocation")
    log = path.with_name("probe.log")
    if log.is_symlink() or not log.is_file() or file_sha256(log) != evidence.get("log_sha256"):
        raise ValueError("categorical long replay log evidence changed")
    diagnostic = read_json(path)
    budget = evidence.get("time_limit_seconds")
    if (type(budget) is not int or not 60 <= budget <= 1170
            or type(diagnostic.get("time_limit_seconds")) is not int
            or diagnostic["time_limit_seconds"] != budget or diagnostic.get("job_id") != record["job_id"]):
        raise ValueError("categorical long replay report job or preflight budget differs")
    runtime = diagnostic.get("runtime", {})
    runtime_root = Path(row["environment_root"]).resolve()
    if (runtime.get("root") != str(runtime_root)
            or runtime.get("manifest_sha256") != file_sha256(runtime_root / "opd-runtime-manifest.json")
            or runtime.get("source") != {key: manifest[key] for key in ("parent_commit", "fork_commit")}
            or diagnostic.get("assets_manifest_sha256") != file_sha256(Path(manifest["assets_root"]) / "manifest.json")
            or runtime.get("verifier_sha256") != file_sha256(Path(manifest["source_root"]) / "scripts/qwen_runtime.py")):
        raise ValueError("categorical long replay runtime or assets differ from the sealed study")
    source = Path(manifest["source_root"]) / "scripts/qwen_vllm_long_replay_diagnostic.py"
    if evidence.get("probe_sha256") != file_sha256(source) or diagnostic.get("source_sha256") != evidence["probe_sha256"]:
        raise ValueError("categorical long replay probe source differs from the sealed study")
    spec = importlib.util.spec_from_file_location("sealed_production_long_replay", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.validate_report(path) != evidence.get("acceptance"):
        raise ValueError("categorical long replay acceptance differs from recomputed evidence")


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
    if "site" in manifest:
        site = site_from_manifest(manifest)
        if record.get("site") != site:
            raise ValueError("GPU allocation differs from the sealed site")
        validate_scheduler_allocation(record.get("scheduler"), site, account=row["account"])
    runtime = record.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("GPU allocation runtime record is malformed")
    validate_fsdp_probe(runtime.get("native_lora_fsdp_acceptance"), world_size=4)
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
    gpu = site_from_manifest(manifest)["gpu"]
    for item in acceptance:
        device = devices[item["device"]]
        if (not isinstance(device, dict) or gpu["name_contains"] not in str(device.get("name"))
                or item.get("name") != device.get("name") or any(item.get(key) != value for key, value in required.items())
                or any(item.get(key) is not value for key, value in required.items() if isinstance(value, bool))):
            raise ValueError("GPU allocation native FA3 kernel, shape, or device acceptance differs")
        if "site" in manifest and device.get("compute_capability") != gpu["compute_capability"]:
            raise ValueError("GPU allocation compute capability differs from the sealed site")
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
        if any(lora.get(key) is not expected for key, expected in {
                "frozen_base_unchanged": True, "dense_export_weight_exact": True,
                "dense_export_projection_exact": True, "teacher_or_training_data_used": False}.items()):
            raise ValueError("controlled LoRA acceptance booleans differ")
        norms = lora.get("adapter_gradient_norms")
        if (not isinstance(norms, list) or len(norms) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in norms)
                or number(lora.get("effective_update", {}), "changed_elements") <= 0):
            raise ValueError("GPU allocation lacks finite controlled LoRA gradients and an effective BF16 update")
    verify_hard_long_replay(manifest_path, manifest, row, record)
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
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("invalid admission evidence path")
        path = Path(row["run_root"]) / relative
        if (path.is_symlink() or not path.resolve().is_relative_to(Path(row["run_root"]).resolve())
                or not path.is_file() or file_sha256(path) != digest):
            raise ValueError("admission evidence changed")
    if admission["resume_parity"].get("passed") is not True:
        raise ValueError("exact next-update resume was not accepted")
    if requires_semantic_checkpoint(manifest) and admission["resume_parity"].get("schema") != "qwen_semantic_v1":
        raise ValueError("this production profile requires semantic next-update admission")
    verify_gpu_evidence(manifest_path, manifest, row, admission, restart_count=0)
    if requires_semantic_checkpoint(manifest):
        phases = ["uninterrupted", "split", "resume"]
        if row["arm_id"].startswith("softgrpo_math_opd"):
            phases.extend(["full_dose", "zero_dose"])
            if (admission.get("zero_dose_parity", {}).get("passed") is not True
                    or admission["zero_dose_parity"].get("schema") != "qwen_semantic_v1"):
                raise ValueError("this production profile requires semantic zero-dose admission")
        for phase in phases:
            output = Path(row["phases"][phase]["output"])
            relative = str(output.relative_to(Path(row["run_root"])))
            if admission["evidence_files"].get(relative) != file_sha256(output):
                raise ValueError("admission phase measurement is not authenticated")
            validate_measurement(read_json(output), manifest, row, phase, row["production_overrides"])
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
    if requires_semantic_checkpoint(manifest):
        output = Path(row["phases"]["production"]["output"])
        if continuation["restart_count"]:
            output = output.with_name(f"measurement-{continuation['restart_count']}.json")
        relative = str(output.relative_to(Path(row["run_root"])))
        digest = file_sha256(output)
        if (continuation.get("production_measurement") != {"path": relative, "sha256": digest}
                or continuation.get("evidence_files", {}).get(relative) != digest):
            raise ValueError("continuation production measurement is not authenticated")
        measured = read_json(output)
        validate_measurement(measured, manifest, row, "production", row["production_overrides"])
        if measured["iterations"][-1]["rollout_iteration"] + 1 != step:
            raise ValueError("continuation measurement differs from the checkpoint boundary")
    checkpoint = authenticate_checkpoint(row["phases"]["production"]["run_dir"], step, arm_id=row["arm_id"],
                                         require_semantic=requires_semantic_checkpoint(manifest))
    path = Path(row["phases"]["production"]["run_dir"]) / f"global_step_{step}" / "checkpoint_manifest.json"
    if checkpoint.get("reason") != "requeue_signal" or file_sha256(path) != continuation.get("checkpoint_manifest_sha256"):
        raise ValueError("continuation does not identify a clean signal checkpoint")
    return continuation


def _recovery_expected_provenance(manifest, row):
    """Recover the resolved production identity, independently of a clean exit.

    The recorder publishes its resolved configuration before the first update.
    Rebuild its provenance and check the sealed recipe overrides rather than
    trusting a checkpoint's own claim about which run produced it.
    """
    from .qwen_production import _values
    from verl.opd.provenance import build_checkpoint_provenance, assert_checkpoint_provenance_matches
    directory = Path(row["phases"]["production"]["directory"])
    paths = list(directory.glob("measurement*.json"))
    def order(path):
        suffix = path.stem.removeprefix("measurement-")
        return int(suffix) if suffix.isdecimal() else -1
    expected = _values(row["production_overrides"])
    for path in sorted(paths, key=order, reverse=True):
        measured = read_json(path)
        if measured.get("phase") != "production" or measured.get("arm_id") != row["arm_id"]:
            raise ValueError("recovery measurement belongs to another phase or arm")
        config = measured.get("configuration")
        if not isinstance(config, dict) or not measured.get("checkpoint_provenance"):
            continue
        for key, wanted in expected.items():
            if key.startswith("hydra.") or key in {"trainer.resume_mode", "trainer.resume_from_path",
                    "trainer.production_output", "trainer.requeue_signal_file"}:
                continue
            observed = config
            for component in key.split("."):
                if not isinstance(observed, dict) or component not in observed:
                    raise ValueError(f"recovery configuration is missing {key}")
                observed = observed[component]
            # The trainer fills optimizer horizons after constructing the
            # authenticated 109-batch dataloader.
            if key in {"actor_rollout_ref.actor.optim.total_training_steps", "critic.optim.total_training_steps"}:
                wanted = 109
            if observed != wanted or (isinstance(wanted, bool) and type(observed) is not bool):
                raise ValueError(f"recovery configuration differs from the sealed recipe: {key}")
        provenance = build_checkpoint_provenance(config)
        if provenance["source"]["commit"] != manifest["fork_commit"]:
            raise ValueError("recovery training source differs from the submission")
        assert_checkpoint_provenance_matches(measured["checkpoint_provenance"], provenance)
        return provenance
    raise ValueError("no resolved production provenance exists for checkpoint recovery")


def _find_recovery_checkpoint(root, expected_provenance, *, require_semantic):
    from verl.trainer.ppo.ray_trainer import _find_latest_committed_checkpoint
    return _find_latest_committed_checkpoint(str(root), expected_provenance=expected_provenance,
            return_manifest=True, require_semantic=require_semantic)


def select_recovery_checkpoint(manifest_path, manifest, row):
    """Select the newest intact production checkpoint; never use prologue state."""
    admission = verify_admission(manifest_path, manifest, row)
    root = Path(row["phases"]["production"]["run_dir"])
    if root != Path(row["run_root"]) / "production" / "training":
        raise ValueError("recovery checkpoint root is outside this arm's production directory")
    validate_artifact_path(root, site_from_manifest(manifest), label="recovery checkpoints")
    expected = _recovery_expected_provenance(manifest, row)
    selected = _find_recovery_checkpoint(root, expected,
                                         require_semantic=requires_semantic_checkpoint(manifest))
    if selected is None:
        raise ValueError("no valid production checkpoint exists; automatic recovery cannot start")
    path, checkpoint = selected
    step = checkpoint.get("global_step")
    if type(step) is not int or not 1 <= step <= 109 or Path(path) != root / f"global_step_{step}":
        raise ValueError("invalid production recovery boundary")
    checkpoint = authenticate_checkpoint(root, step, arm_id=row["arm_id"],
                    require_semantic=requires_semantic_checkpoint(manifest), expected_provenance=expected)
    return {"global_step": step, "resume_from_path": str(path),
            "checkpoint_manifest_sha256": file_sha256(Path(path) / "checkpoint_manifest.json"),
            "admission_sha256": admission["manifest_content_sha256"],
            "resume_provenance_sha256": checkpoint["resume_provenance_sha256"]}


def finalize_recovered_checkpoint(manifest, row, recovery):
    """Finish metadata publication and retention after a final-checkpoint crash.

    The current allocation owns the training directory, but needs no workers:
    every training update is already committed in the authenticated final state.
    """
    from .qwen_production import _values
    from verl.trainer.ppo.ray_trainer import _repair_checkpoint_history, _prune_committed_checkpoints

    root = Path(row["phases"]["production"]["run_dir"])
    if root != Path(row["run_root"]) / "production" / "training":
        raise ValueError("final recovery checkpoint root is outside this arm's production directory")
    validate_artifact_path(root, site_from_manifest(manifest), label="final recovery checkpoints")
    if (type(recovery.get("global_step")) is not int or recovery["global_step"] != 109
            or recovery.get("resume_from_path") != str(root / "global_step_109")):
        raise ValueError("final recovery requires the selected iteration-109 checkpoint")
    expected = _recovery_expected_provenance(manifest, row)
    checkpoint = authenticate_checkpoint(root, 109, arm_id=row["arm_id"],
                    require_semantic=requires_semantic_checkpoint(manifest), expected_provenance=expected)
    if (file_sha256(root / "global_step_109" / "checkpoint_manifest.json")
            != recovery.get("checkpoint_manifest_sha256")
            or checkpoint["resume_provenance_sha256"] != recovery.get("resume_provenance_sha256")):
        raise ValueError("final recovery checkpoint differs from the selected state")
    values = _values(row["production_overrides"])
    repaired = _repair_checkpoint_history(str(root), resumed_manifest=checkpoint,
                    best_mode=str(values.get("trainer.checkpoint_best_mode", "max")))
    pruned = _prune_committed_checkpoints(str(root),
                    keep_latest=int(values.get("trainer.checkpoint_keep_latest", 2)))
    return {"status": "complete", "global_step": 109,
            "history_removed": repaired, "retention_removed": pruned}


def _recovery_segment(manifest_path, manifest, row, *, job_id, restart_count):
    if not str(job_id).isdecimal() or type(restart_count) is not int or restart_count < 0:
        raise ValueError("invalid recovery allocation identity")
    path = Path(row["run_root"]) / f"segment-{restart_count}.json"
    report = read_json(path)
    expected = {"arm_id": row["arm_id"], "job_id": str(job_id), "restart_count": restart_count,
                "submission_manifest_sha256": file_sha256(manifest_path),
                "parent_commit": manifest["parent_commit"], "fork_commit": manifest["fork_commit"]}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("recovery segment belongs to another source, arm or allocation")
    return report


def prepare_recovery(manifest_path, manifest, row, *, job_id, restart_count, failure_exit_code=None):
    """Authorize at most three retries without a newer committed checkpoint.

    The per-segment record makes the shell's request and the next allocation's
    startup idempotent. A node failure can enter here without a shell request.
    """
    if failure_exit_code is not None and (type(failure_exit_code) is not int
            or failure_exit_code <= 0 or failure_exit_code in {75, 130, 143}):
        raise ValueError("successful completion, continuation and cancellation are not failure retries")
    report = _recovery_segment(manifest_path, manifest, row, job_id=job_id, restart_count=restart_count)
    if "production" not in report.get("phases", {}):
        raise ValueError("automatic recovery requires a production attempt after accepted admission")
    failure = report.get("failure", {})
    message = str(failure.get("error", "")).lower()
    if (report.get("cleanup_error") or report.get("persistence_error")
            or failure.get("category") == "authentication"
            or any(term in message for term in ("interrupted by signal 15", "interrupted by signal 2",
                                                "keyboardinterrupt", "cancelled", "canceled"))):
        raise ValueError("cancellation, authentication or cleanup failure cannot automatically retry")
    selected = select_recovery_checkpoint(manifest_path, manifest, row)
    identity = {"schema_version": 1, "arm_id": row["arm_id"], "job_id": str(job_id),
                "restart_count": restart_count, "submission_manifest_sha256": file_sha256(manifest_path),
                "parent_commit": manifest["parent_commit"], "fork_commit": manifest["fork_commit"], **selected}
    directory = Path(row["run_root"]) / "recovery"
    validate_artifact_path(directory, site_from_manifest(manifest), label="recovery records")
    path = directory / f"restart-{restart_count}.json"
    if path.exists():
        existing = read_json(path, sealed=True)
        if any(existing.get(key) != value for key, value in identity.items()):
            raise ValueError("existing recovery request differs from the selected checkpoint or allocation")
        if (type(existing.get("consecutive_failure_retries")) is not int
                or not 1 <= existing["consecutive_failure_retries"] <= MAX_FAILURE_RETRIES_WITHOUT_PROGRESS):
            raise ValueError("invalid existing recovery retry count")
        return existing
    count = 1
    previous_path = directory / f"restart-{restart_count - 1}.json"
    if previous_path.exists():
        previous = read_json(previous_path, sealed=True)
        for key in ("arm_id", "job_id", "submission_manifest_sha256", "parent_commit", "fork_commit"):
            if previous.get(key) != identity[key]:
                raise ValueError("previous recovery request has a different run identity")
        previous_step, previous_count = previous.get("global_step"), previous.get("consecutive_failure_retries")
        if type(previous_step) is not int or type(previous_count) is not int or not 1 <= previous_count <= MAX_FAILURE_RETRIES_WITHOUT_PROGRESS:
            raise ValueError("invalid previous recovery progress or retry count")
        # A rollback to an older checkpoint is not progress.
        if selected["global_step"] <= previous_step:
            count = previous_count + 1
    if selected["global_step"] == 109:
        count = 1  # A finished checkpoint needs no further training attempt.
    if count > MAX_FAILURE_RETRIES_WITHOUT_PROGRESS:
        raise ValueError("three failure retries without checkpoint progress have been exhausted")
    result = {**identity, "status": "complete" if selected["global_step"] == 109 else "recovery_ready",
              "consecutive_failure_retries": count, "max_failure_retries_without_progress": MAX_FAILURE_RETRIES_WITHOUT_PROGRESS,
              "failure_exit_code": failure_exit_code,
              "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    directory.mkdir(exist_ok=True)
    write_json(path, result, seal=True)
    return read_json(path, sealed=True)


def _process_identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return {"pid": pid, "process_group": int(stat[2]), "start_ticks": int(stat[19]),
            "hostname": socket.gethostname(), "state": stat[0]}


def _owned_group_survives(group, owner):
    """Identify surviving descendants even after their session leader exited."""
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    found = False
    token = ("OPD_QPROD_TRAINER_OWNER=" + owner).encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            process = _process_identity(int(entry.name))
            if process is None or process["process_group"] != group or process["state"] == "Z":
                continue
            if entry.stat().st_uid != os.getuid() or token not in (entry / "environ").read_bytes().split(b"\0"):
                raise ValueError("surviving trainer group contains a process with different ownership")
            found = True
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found


def cleanup_trainer(manifest_path, manifest, row, *, job_id, restart_count):
    """Stop only the recorded trainer session belonging to this allocation."""
    report = _recovery_segment(manifest_path, manifest, row, job_id=job_id, restart_count=restart_count)
    owned = report.get("child_process")
    if owned is None:
        if report.get("status") in {"complete", "continuation_ready", "failed"}:
            return {"status": "no_recorded_trainer"}
        raise ValueError("trainer ownership is missing after an interrupted controller")
    if (type(owned.get("pid")) is not int or owned["pid"] <= 1
            or owned.get("process_group") != owned["pid"] or owned.get("hostname") != socket.gethostname()):
        raise ValueError("recorded trainer session identity is invalid")
    pid = owned["pid"]
    current = _process_identity(pid)
    expected_owner = f"{file_sha256(manifest_path)}:{job_id}:{restart_count}"
    if current is None or current["start_ticks"] != owned.get("start_ticks"):
        if not _owned_group_survives(pid, expected_owner):
            return {"status": "recorded_trainer_exited"}
    else:
        if current["process_group"] != pid or Path(f"/proc/{pid}").stat().st_uid != os.getuid():
            raise ValueError("recorded trainer session ownership differs")
        environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        if current["state"] != "Z" and ("OPD_QPROD_TRAINER_OWNER=" + expected_owner).encode() not in environ:
            raise ValueError("recorded trainer belongs to another allocation")
        # A zombie leader no longer has environment bytes; its descendants
        # still have to prove allocation ownership before a group signal.
        if current["state"] == "Z" and not _owned_group_survives(pid, expected_owner):
            return {"status": "recorded_trainer_exited"}
    try:
        os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                return {"status": "trainer_stopped"}
            time.sleep(.1)
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return {"status": "trainer_stopped"}


class ProductionController:
    def __init__(self, args):
        from .qwen_production import verify_manifest, PRODUCTION_PROJECT
        self.args = args
        self.manifest = verify_manifest(args.manifest)
        self.row = select_arm(self.manifest, args.arm)
        self.root = Path(self.row["run_root"])
        validate_artifact_path(self.root, site_from_manifest(self.manifest), label="production run root")
        self.root.mkdir(parents=True, exist_ok=True)
        self.child = None
        self.phase = "initialization"
        self.started = time.monotonic()
        limit = self.manifest.get("prologue_limit_seconds", 7200)
        if type(limit) is not int or limit <= 0:
            raise ValueError("invalid authenticated production prologue budget")
        if getattr(args, "prologue_limit_seconds", None) not in (None, limit):
            raise ValueError("prologue budget differs from the authenticated manifest")
        self.deadline = self.started + limit - max(0.0, time.time() - float(os.environ.get("OPD_PROLOGUE_STARTED_EPOCH", time.time())))
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
        environment["OPD_QPROD_TRAINER_OWNER"] = (
            f"{self.report['submission_manifest_sha256']}:{self.report.get('job_id', '')}:{self.restart}")
        if phase == "production":
            command.append("trainer.requeue_signal_file=" + json.dumps(str(self.args.signal_file)))
        # Python -m prepends cwd to sys.path. Keep the revised study outside
        # the source VERL tree so it imports the authenticated installed wheel.
        working_directory = (directory if requires_semantic_checkpoint(self.manifest)
                             else Path(self.manifest["source_root"]) / "3rdparty/SofT-GRPO/verl-0.4.x")
        invocation = {"phase": phase, "command": command, "wandb_run_id": run_id, "wandb_project": project,
                      "working_directory": str(working_directory),
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
                self.child = subprocess.Popen(command, cwd=working_directory,
                                              env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                if isinstance(getattr(self.child, "pid", None), int):
                    self.report["child_process"] = _process_identity(self.child.pid)
                    self.persist()
                remaining = None if phase == "production" else max(0.01, self.deadline - time.monotonic() - 30)
                code = self.child.wait(timeout=remaining)
            if code:
                raise RuntimeError(f"{phase} trainer exited with code {code}")
            measured = read_json(output)
            validate_measurement(measured, self.manifest, self.row, phase, command[3:])
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
            raise TimeoutError("correctness prologue exceeded its authenticated allocation budget")
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
                raise TimeoutError("correctness prologue authenticated deadline reached")
            raise RuntimeError(f"production controller interrupted by signal {signum}")
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            old_handlers[signum] = signal.signal(signum, interrupted)
        self.persist()
        try:
            resume = None
            if self.restart:
                continuation_path = self.root / "continuation.json"
                candidate = None
                if continuation_path.exists():
                    try:
                        candidate = read_json(continuation_path, sealed=True)
                    except (ValueError, OSError) as error:
                        self.report["unusable_continuation"] = str(error)
                if (candidate is not None and candidate.get("restart_count") == self.restart - 1
                        and candidate.get("job_id") == self.report["job_id"]):
                    try:
                        continuation = verify_continuation(self.args.manifest, self.manifest, self.row)
                        resume = Path(self.row["phases"]["production"]["run_dir"]) / f"global_step_{continuation['global_step']}"
                        self.report["resume_reason"] = "clean_continuation"
                    except (ValueError, OSError, RuntimeError) as error:
                        self.report["unusable_continuation"] = str(error)
                if resume is None:
                    recovery = prepare_recovery(self.args.manifest, self.manifest, self.row,
                                    job_id=self.report["job_id"], restart_count=self.restart - 1)
                    self.report["recovery"] = recovery
                    self.report["resume_reason"] = "latest_valid_checkpoint"
                    if recovery["global_step"] == 109:
                        self.report["final_checkpoint_repair"] = finalize_recovered_checkpoint(
                            self.manifest, self.row, recovery)
                        self.report["status"] = "complete"
                        self.report["recovered_completed_checkpoint"] = True
                        return 0
                    resume = Path(recovery["resume_from_path"])
                self.report["resume_from_path"] = str(resume)
                self.persist()
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
                output = Path(self.report["phases"]["production"]["output"])
                relative, digest = str(output.relative_to(self.root)), file_sha256(output)
                continuation["production_measurement"] = {"path": relative, "sha256": digest}
                continuation["evidence_files"][relative] = digest
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
    parser.add_argument("command", choices=("run", "verify-continuation", "verify-recovery", "cleanup-trainer"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--prologue-limit-seconds", type=int)
    parser.add_argument("--signal-file", type=Path)
    parser.add_argument("--failure-exit-code", type=int)
    args = parser.parse_args(argv)
    from .qwen_production import verify_manifest
    if args.command != "run":
        manifest = verify_manifest(args.manifest)
        row = select_arm(manifest, args.arm)
        if args.command == "verify-continuation":
            result = verify_continuation(args.manifest, manifest, row)
        else:
            allocation = {"job_id": os.environ.get("SLURM_JOB_ID", ""),
                          "restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0"))}
            if args.command == "cleanup-trainer":
                result = cleanup_trainer(args.manifest, manifest, row, **allocation)
            else:
                result = prepare_recovery(args.manifest, manifest, row, **allocation,
                                          failure_exit_code=args.failure_exit_code)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.signal_file is None:
        parser.error("run requires --signal-file")
    return ProductionController(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
