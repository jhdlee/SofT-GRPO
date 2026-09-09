"""Sealed Qwen3 seven-arm production profiles, separate from historical studies.

This module prepares/verifies assets and configuration; it never submits or runs
training. The controller consumes the phase commands emitted here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from . import qwen_site, qwen_training, study
from .manifest import canonical_sha256, file_sha256, validate_sealed_content


PROFILE_ID = "qwen3-math-seven-arm-v1"
LORA_PROFILE_ID = "qwen3-math-seven-arm-lora-fa3-v1"
PROMPT_PROFILE_ID = qwen_training.PROFILE_ID
MODEL = dict(qwen_training.MODEL)
SEED = 11
TOTAL_ROLLOUT_ITERATIONS = 109
TOTAL_OPTIMIZER_STEPS = 218
WARMUP_ITERATIONS = 11
ARM_IDS = (
    "hardgrpo_math_s11", "softgrpo_math_s11", "softopd_math_s11",
    "softgrpo_math_opd_s11", "softgrpo_math_opd_posadv_s11",
    "softgrpo_math_opd_current_s11", "softgrpo_math_opd_beta0p1_s11",
)
PHASES = ("production", "uninterrupted", "split", "resume", "full_dose", "zero_dose")
ADMISSION_MODES = ("diagnostics", "direct_training")
RESOURCES = {"gpus": 4, "cpus": 56, "memory_gib": 768,
             "time_limit_seconds": 36 * 3600, "exclusive": False}
PROLOGUE_LIMIT_SECONDS = 7200
PRODUCTION_PROJECT = "opd-qwen3-math-seven-arm"
RUNTIME_PACKAGE_PINS = {
    "soft": {"torch": "2.6.0", "transformers": "4.51.1", "ray": "2.49.2",
             "sglang": "0.4.6.post1", "xgrammar": "0.1.17", "numpy": "2.3.3",
             "flash-attn": "2.7.3", "sgl-kernel": "0.1.1", "triton": "3.2.0"},
    "hard": {"torch": "2.6.0", "transformers": "4.51.1", "ray": "2.49.2",
             "vllm": "0.8.5", "xgrammar": "0.1.18", "xformers": "0.0.29.post2",
             "numba": "0.61.2", "numpy": "2.2.6"},
}
HARD_RUNTIME_PINS = RUNTIME_PACKAGE_PINS["hard"]
SHARED_RUNTIME_PINS = {
    "torch": "2.6.0", "transformers": "4.51.1", "peft": "0.17.1",
    "ray": "2.43.0", "sglang": "0.4.6.post1", "vllm": "0.8.5",
    "xgrammar": "0.1.18", "numpy": "2.2.6", "numba": "0.61.2",
    "xformers": "0.0.29.post2", "triton": "3.2.0",
    "opd-fa3": "0.1.0", "sgl-kernel": "0.1.1", "flash-attn": "2.7.3",
    "opentelemetry-api": "1.26.0", "opentelemetry-sdk": "1.26.0",
    "opentelemetry-semantic-conventions": "0.47b0",
    "opentelemetry-exporter-prometheus": "0.47b0",
}
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class TrainingOptions:
    """Explicit options for the revised study; None preserves the old recipe."""

    finetuning: str = "lora"
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_target_modules: tuple[str, ...] = LORA_TARGET_MODULES
    reference_kl: str = "recipe"

    def __post_init__(self):
        if self.finetuning not in ("full", "lora") or self.reference_kl not in ("recipe", "off"):
            raise ValueError("invalid finetuning or reference-KL mode")
        if any(type(value) is not int or value <= 0 for value in (self.lora_rank, self.lora_alpha)):
            raise ValueError("LoRA rank and alpha must be positive integers")
        if self.lora_rank > 256:
            raise ValueError("native LoRA rank must not exceed 256")
        targets = tuple(self.lora_target_modules)
        if not targets or len(set(targets)) != len(targets) or not set(targets) <= set(LORA_TARGET_MODULES):
            raise ValueError("LoRA targets must be unique supported attention/MLP projections")
        object.__setattr__(self, "lora_target_modules", targets)

    def as_manifest(self):
        result = asdict(self)
        result["lora_target_modules"] = list(self.lora_target_modules)
        return result


def _options(value):
    return value if value is None or isinstance(value, TrainingOptions) else TrainingOptions(**value)


def _profile(options):
    return LORA_PROFILE_ID if options is not None else PROFILE_ID


def admission_mode(manifest: Mapping[str, Any]) -> str:
    """Historical manifests require diagnostics; only an explicit policy skips them."""
    mode = manifest.get("admission_mode", "diagnostics")
    if mode not in ADMISSION_MODES:
        raise ValueError("invalid production admission mode")
    return mode


def resolve_arm(identifier: str) -> study.ArmSpec:
    if identifier not in ARM_IDS:
        raise ValueError(f"unknown Qwen production arm: {identifier}")
    spec = study.resolve_arm(identifier).spec
    if spec.opd_enabled and spec.opd_mode == "auxiliary" and "beta0p1" not in identifier:
        spec = replace(spec, beta_base=1.0)
    return spec


def arm_contract(identifier: str, training_options=None) -> dict[str, Any]:
    options = _options(training_options)
    spec = resolve_arm(identifier)
    result = spec.as_manifest()
    result.update(protocol=PROFILE_ID, model=dict(MODEL), prompt_profile=PROMPT_PROFILE_ID,
                  total_optimizer_steps=TOTAL_OPTIMIZER_STEPS, training_examples=6985,
                  validation_examples=512, prompt_batch_size=64, response_token_cap=8192,
                  micro_batch_size_per_gpu=2, tensor_parallel_size=1,
                  resources=dict(RESOURCES), validation_iterations=[0, 25, 50, 75, 100, 109],
                  checkpoint_iterations=[25, 50, 75, 100, 109],
                  qwen_replay_backend="native_fa3_v1" if spec.rollout_kind == "native_soft" else "disabled",
                  gradient_policy="diagnostic_clipping", completion_gate_enabled=False,
                  ratio_range_gate_enabled=False, clipping_frequency_gate_enabled=False)
    result["experiment_name"] = "qwen3_0p6b_" + identifier
    result["wandb_tags"] = [PROFILE_ID, *result["wandb_tags"]]
    if options is not None:
        result.update(protocol=LORA_PROFILE_ID, training_options=options.as_manifest(),
                      qwen_replay_backend="native_fa3_v2", checkpoint_semantics="qwen_semantic_v1",
                      forward_dtype="bfloat16", master_dtype="float32", reduce_dtype="float32",
                      teacher_weight_space="dense_effective", reference_kl_coefficient=(
                          0.001 if options.reference_kl == "recipe" and spec.opd_mode != "standalone" else 0.0))
        result["wandb_tags"] = [LORA_PROFILE_ID, options.finetuning, "reference-kl-" + options.reference_kl,
                               *result["wandb_tags"][1:]]
    return result


def _absolute(path: Path | str) -> Path:
    return Path(path).expanduser().absolute().resolve()


def _values(overrides: Sequence[str]) -> dict[str, Any]:
    result = {}
    for override in overrides:
        key, value = override.split("=", 1)
        try:
            result[key.lstrip("+")] = json.loads(value)
        except json.JSONDecodeError:
            result[key.lstrip("+")] = value
    return result


def phase_metadata(identifier: str, run_root: Path | str, phase: str) -> dict[str, Any]:
    spec = resolve_arm(identifier)
    if phase not in PHASES:
        raise ValueError(f"unknown production phase: {phase}")
    run_root = _absolute(run_root)
    folder = "resume" if phase in ("split", "resume") else phase
    directory = run_root / "production" if phase == "production" else run_root / "prologue" / folder
    filename = "measurement-" + phase + ".json" if phase in ("split", "resume") else "measurement.json"
    return {
        "directory": str(directory), "run_dir": str(directory / "training"),
        "output": str(directory / filename),
        "invocation_limit": None if phase == "production" else 2 if phase == "uninterrupted" else 1,
        "effective_arm_id": "softgrpo_math_s11" if phase == "zero_dose" else identifier,
        "warmup_iterations": 0 if phase in ("full_dose", "zero_dose") or spec.schedule == "constant" else WARMUP_ITERATIONS,
        "applicable": phase not in ("full_dose", "zero_dose") or (spec.opd_enabled and spec.opd_mode == "auxiliary"),
    }


def production_overrides(
    identifier: str, assets_root: Path | str, run_root: Path | str, *,
    phase: str = "production", resume_from_path: Path | str | None = None,
    source_root: Path | str | None = None,
    training_options=None, runtime_manifest=None, site=None,
) -> list[str]:
    """Emit unique Hydra overrides; invocation limits never shorten the horizon."""
    spec = resolve_arm(identifier)
    options = _options(training_options)
    if site is not None:
        site = qwen_site.validate_site(site)
        qwen_site.validate_artifact_path(assets_root, site, "assets root")
        qwen_site.validate_artifact_path(run_root, site, "run root")
        if source_root is not None:
            qwen_site.validate_artifact_path(source_root, site, "source snapshot")
        if resume_from_path is not None:
            qwen_site.validate_artifact_path(resume_from_path, site, "resume checkpoint")
        if runtime_manifest is not None:
            qwen_site.validate_artifact_path(runtime_manifest["path"], site, "runtime manifest")
    metadata = phase_metadata(identifier, run_root, phase)
    if site is not None:
        for key in ("directory", "run_dir", "output"):
            qwen_site.validate_artifact_path(metadata[key], site, key)
    if not metadata["applicable"]:
        raise ValueError(f"{phase} only applies to hybrid OPD arms")
    if phase == "resume" and resume_from_path is None:
        raise ValueError("resume phase requires an explicit committed checkpoint")
    if resume_from_path is not None and phase not in ("resume", "production"):
        raise ValueError("only resume or production phases may restore a checkpoint")
    backend = "native_fa3_v1" if spec.rollout_kind == "native_soft" else "disabled"
    # Reuse the shared sealed asset/prompt and optimizer recipe. The public
    # benchmark profile and its one/two-GPU restrictions remain unchanged.
    values = _values(qwen_training.profile_overrides(
        "standalone" if spec.opd_mode == "standalone" else "hybrid", 2,
        assets_root, metadata["run_dir"], replay_backend=backend,
    ))
    values.update(_values(study.hydra_overrides(identifier)))
    values["algorithm.opd.beta_base"] = spec.beta_base
    if phase == "zero_dose":
        values.update(_values(study.hydra_overrides("softgrpo_math_s11")))
    elif phase == "full_dose":
        values["algorithm.opd.schedule"] = "constant"
    values.update({
        "trainer.training_profile": PROFILE_ID,
        "trainer.project_name": PRODUCTION_PROJECT if phase == "production" else PRODUCTION_PROJECT + "-prologue",
        "trainer.experiment_name": "qwen3_0p6b_" + identifier,
        "trainer.n_gpus_per_node": RESOURCES["gpus"], "ray_init.num_cpus": RESOURCES["cpus"],
        "trainer.total_epochs": 1, "trainer.total_training_steps": None,
        "trainer.max_rollout_iterations_per_invocation": metadata["invocation_limit"],
        "trainer.val_before_train": phase == "production",
        "trainer.test_freq": 25 if phase == "production" else -1,
        "trainer.save_freq": 25 if phase == "production" else 1,
        "trainer.logger": ["console", "wandb"],
        "trainer.resume_mode": "resume_path" if resume_from_path is not None else "disable",
        "trainer.resume_from_path": str(_absolute(resume_from_path)) if resume_from_path is not None else None,
        "trainer.validation_seed": SEED, "trainer.validation_seed_iteration": 0,
        "trainer.rollout_integrity.gate_first_n_iterations": TOTAL_ROLLOUT_ITERATIONS,
        "trainer.rollout_integrity.full_dose_gradient_gate_enabled": False,
        "trainer.rollout_integrity.completion_gate_enabled": False,
        "actor_rollout_ref.model.qwen_replay_backend": backend,
        "actor_rollout_ref.rollout.require_retained_support": spec.rollout_kind == "native_soft",
        "actor_rollout_ref.rollout.dispatch_mode": "bounded_async" if spec.rollout_kind == "native_soft" else "legacy_batch",
        "actor_rollout_ref.rollout.max_running_requests": 32,
        "actor_rollout_ref.rollout.async_queue_size": 64,
        "actor_rollout_ref.rollout.production_engine_isolation": spec.rollout_kind == "native_soft",
        "trainer.production_mode": True, "trainer.production_phase": phase,
        "trainer.production_arm_id": identifier, "trainer.production_output": metadata["output"],
        "trainer.production_gradient_policy": "diagnostic_clipping",
    })
    if source_root is not None:
        values["custom_reward_function.path"] = str(_absolute(source_root) / "3rdparty/SofT-GRPO/opd_tools/reward.py")
    if site is not None and site["artifact_root"] is not None:
        values["hydra.run.dir"] = str(qwen_site.validate_artifact_path(
            Path(metadata["directory"]) / "hydra", site, "Hydra outputs"))
    extra_keys = set()
    if options is not None:
        reference = options.reference_kl == "recipe" and spec.opd_mode != "standalone"
        additions = {
            "trainer.training_profile": LORA_PROFILE_ID,
            "trainer.project_name": PRODUCTION_PROJECT + "-lora-fa3" + ("" if phase == "production" else "-prologue"),
            "trainer.checkpoint_semantics": "qwen_semantic_v1",
            # vLLM sleep unmaps HBM while retaining logical PyTorch allocations.
            # The revised study checks sampled physical update pressure against
            # device capacity; historical profiles retain their allocator gate.
            "trainer.resource_policy.mode": "physical_device_v1",
            "trainer.resource_policy.max_device_used_fraction": 0.98,
            "trainer.resource_policy.sample_interval_seconds": 0.1,
            "actor_rollout_ref.checkpoint_semantics": "qwen_semantic_v1",
            "actor_rollout_ref.training_seed": SEED,
            "actor_rollout_ref.model.qwen_replay_backend": "native_fa3_v2",
            "actor_rollout_ref.rollout.qwen_replay_backend": "native_fa3_v2",
            "actor_rollout_ref.model.lora_rank": options.lora_rank if options.finetuning == "lora" else 0,
            "actor_rollout_ref.model.lora_alpha": options.lora_alpha,
            "actor_rollout_ref.model.target_modules": list(options.lora_target_modules),
            "actor_rollout_ref.actor.use_kl_loss": reference,
            "actor_rollout_ref.actor.kl_loss_coef": 0.001 if reference else 0.0,
            "algorithm.use_kl_in_reward": False,
            "actor_rollout_ref.rollout.seed": SEED,
        }
        if runtime_manifest is not None:
            additions.update({"trainer.runtime_manifest_path": runtime_manifest["path"],
                              "trainer.runtime_manifest_sha256": runtime_manifest["sha256"]})
        if source_root is not None:
            additions["trainer.runtime_source_root"] = str(_absolute(source_root) / "3rdparty/SofT-GRPO")
        extra_keys.update(additions)
        values.update(additions)
    return [("++" if key in extra_keys or key.startswith("trainer.production_") or key == "actor_rollout_ref.rollout.production_engine_isolation" else "") + key + "="
            + json.dumps(value, separators=(",", ":")) for key, value in values.items()]


def _commit(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a full lowercase Git commit")
    return value


def _verify_source(source_root: Path, parent_commit: str, fork_commit: str) -> None:
    def git(root, *args):
        return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()
    fork = source_root / "3rdparty/SofT-GRPO"
    if git(source_root, "rev-parse", "HEAD") != parent_commit or git(fork, "rev-parse", "HEAD") != fork_commit:
        raise ValueError("production source checkout differs from the pinned commits")
    entry = git(source_root, "ls-tree", "HEAD", "3rdparty/SofT-GRPO").split()
    if len(entry) != 4 or entry[:3] != ["160000", "commit", fork_commit]:
        raise ValueError("production parent Gitlink differs from the fork commit")
    for root in (source_root, fork):
        if git(root, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError("production source checkout has uncommitted changes or untracked files")


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["manifest_content_sha256"] = canonical_sha256(result)
    return result


def _build_manifest(*, assets_root, study_root, source_root, parent_commit, fork_commit,
                    parent_gitlink, soft_env, hard_env, assets_manifest, training_options=None,
                    site=None, prologue_limit_seconds=None, admission_mode="diagnostics") -> dict[str, Any]:
    if admission_mode not in ADMISSION_MODES:
        raise ValueError("invalid production admission mode")
    options = _options(training_options)
    profile_id = _profile(options)
    for name, value in (("parent_commit", parent_commit), ("fork_commit", fork_commit), ("parent_gitlink", parent_gitlink)):
        _commit(value, name)
    if parent_gitlink != fork_commit:
        raise ValueError("parent Gitlink must match production fork")
    validate_sealed_content(assets_manifest)
    if assets_manifest.get("profile_id") != PROMPT_PROFILE_ID or assets_manifest.get("model") != MODEL:
        raise ValueError("production assets must use the authenticated pinned Qwen profile")
    assets_root, study_root, source_root, soft_env, hard_env = map(_absolute, (assets_root, study_root, source_root, soft_env, hard_env))
    if site is not None:
        site = qwen_site.validate_site(site)
        for label, path in (("assets root", assets_root), ("study root", study_root),
                            ("source snapshot", source_root), ("soft environment", soft_env),
                            ("hard environment", hard_env)):
            qwen_site.validate_artifact_path(path, site, label)
    effective_site = qwen_site.resolve_site() if site is None else site
    if prologue_limit_seconds is None:
        prologue_limit_seconds = effective_site["production_prologue_limit_seconds"]
    qwen_site.validate_prologue_limit(prologue_limit_seconds, effective_site)
    runtime_manifest = None
    if options is not None:
        if soft_env != hard_env:
            raise ValueError("the LoRA/FA3 study requires one shared runtime environment")
        runtime_path = soft_env / "opd-runtime-manifest.json"
        if runtime_path.is_symlink() or not runtime_path.is_file():
            raise ValueError("shared runtime requires an authenticated build manifest")
        runtime = json.loads(runtime_path.read_text())
        validate_sealed_content(runtime)
        if runtime.get("cpu_ray_preflight") is not True or runtime.get("schema_version") != 1:
            raise ValueError("shared runtime requires passed real Ray acceptance")
        if runtime.get("build_record", {}).get("source") != {"parent_commit": parent_commit, "fork_commit": fork_commit}:
            raise ValueError("shared runtime source differs from the production snapshot")
        runtime_manifest = {"path": str(runtime_path), "sha256": file_sha256(runtime_path),
                            "manifest_content_sha256": runtime["manifest_content_sha256"]}
    base = {
        "schema_version": 1, "profile_id": profile_id, "protocol": "qwen3-production-submission-v1",
        "source_root": str(source_root), "parent_commit": parent_commit,
        "fork_commit": fork_commit, "parent_gitlink": parent_gitlink,
        "assets_root": str(assets_root), "study_root": str(study_root),
        "assets_manifest_content_sha256": assets_manifest["manifest_content_sha256"],
        "asset_profile_id": PROMPT_PROFILE_ID, "model": dict(MODEL),
        "runtime_environments": {"soft": str(soft_env), "hard": str(hard_env)},
        "runtime_package_pins": {kind: dict(pins) for kind, pins in RUNTIME_PACKAGE_PINS.items()},
        "resources": dict(RESOURCES), "prologue_limit_seconds": prologue_limit_seconds,
        "base_config_sha256": file_sha256(source_root / "3rdparty/SofT-GRPO/verl-0.4.x/verl/trainer/config/ppo_trainer.yaml"),
        "arm_order": list(ARM_IDS), "arms": [],
    }
    if site is not None:
        base["site"] = site
    if admission_mode != "diagnostics":
        base["admission_mode"] = admission_mode
    if options is not None:
        base.update(training_options=options.as_manifest(), runtime_manifest=runtime_manifest,
                    runtime_package_pins={kind: dict(SHARED_RUNTIME_PINS) for kind in ("soft", "hard")})
    for identifier in ARM_IDS:
        spec = resolve_arm(identifier)
        run_root = study_root / "arms" / identifier
        environment = soft_env if spec.rollout_kind == "native_soft" else hard_env
        contract = arm_contract(identifier, options)
        if site is not None and site["scheduler"]["account"] is not None:
            contract["account"] = site["scheduler"]["account"]
        identity_inputs = [profile_id, identifier, parent_commit, fork_commit,
                           assets_manifest["manifest_content_sha256"], str(run_root), str(environment), contract]
        if site is not None:
            identity_inputs.append({"site": site, "prologue_limit_seconds": prologue_limit_seconds})
        if admission_mode != "diagnostics":
            identity_inputs.append({"admission_mode": admission_mode})
        identity = canonical_sha256(identity_inputs)
        phases = {phase: phase_metadata(identifier, run_root, phase) for phase in PHASES}
        production = production_overrides(identifier, assets_root, run_root, source_root=source_root,
                                          training_options=options, runtime_manifest=runtime_manifest, site=site)
        base["arms"].append({
            "arm_id": identifier, "account": contract["account"], "run_root": str(run_root),
            "environment_root": str(environment), "python_bin": str(environment / "bin/python"),
            "runtime_packages": dict(base["runtime_package_pins"]["soft" if spec.rollout_kind == "native_soft" else "hard"]),
            "wandb_run_id": "qprod-" + identity[:24], "wandb_project": PRODUCTION_PROJECT + ("-lora-fa3" if options else ""),
            "contract": contract, "contract_sha256": canonical_sha256(contract),
            "phases": phases, "production_overrides": production,
            "production_overrides_sha256": canonical_sha256(production),
        })
        if site is not None:
            base["arms"][-1]["wandb_entity"] = site["wandb_entity"]
    return _seal(base)


def build_manifest(*, assets_root, study_root, source_root, parent_commit, fork_commit,
                   soft_env, hard_env, parent_gitlink=None, training_options=None,
                   site=None, prologue_limit_seconds=None, admission_mode="diagnostics") -> dict[str, Any]:
    assets = qwen_training.verify(assets_root)
    _verify_source(_absolute(source_root), parent_commit, fork_commit)
    return _build_manifest(assets_root=assets_root, study_root=study_root, source_root=source_root,
                           parent_commit=parent_commit, fork_commit=fork_commit,
                           parent_gitlink=fork_commit if parent_gitlink is None else parent_gitlink,
                           soft_env=soft_env, hard_env=hard_env, assets_manifest=assets,
                           training_options=training_options, site=site, prologue_limit_seconds=prologue_limit_seconds,
                           admission_mode=admission_mode)


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_symlink():
        raise ValueError("refusing a symlink manifest destination")
    encoded = (json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise ValueError("refusing to overwrite a different production manifest")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".qwen-production-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
                raise ValueError("production manifest raced with a different writer")
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def materialize_manifest(**kwargs) -> dict[str, Any]:
    manifest = build_manifest(**kwargs)
    root = Path(manifest["study_root"])
    for arm in manifest["arms"]:
        _write_immutable(root / "arms" / arm["arm_id"] / "profile.json", _seal({
            "profile_id": manifest["profile_id"], "study_manifest_content_sha256": manifest["manifest_content_sha256"],
            "arm": arm,
        }))
    _write_immutable(root / "manifest.json", manifest)
    return manifest


def verify_manifest(path: Path | str, *, verify_assets: bool = True, verify_source: bool = True) -> dict[str, Any]:
    path = Path(path).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError("production manifest must be a regular file")
    manifest = json.loads(path.read_text())
    validate_sealed_content(manifest)
    if manifest.get("profile_id") not in (PROFILE_ID, LORA_PROFILE_ID):
        raise ValueError("not a Qwen seven-arm production manifest")
    if (manifest["profile_id"] == LORA_PROFILE_ID) != (manifest.get("training_options") is not None):
        raise ValueError("production profile and training options differ")
    if path.resolve() != Path(manifest["study_root"]) / "manifest.json":
        raise ValueError("production manifest is not in its sealed study root")
    assets = qwen_training.verify(manifest["assets_root"]) if verify_assets else qwen_training._read_manifest(Path(manifest["assets_root"]) / "manifest.json")
    if verify_source:
        _verify_source(Path(manifest["source_root"]), manifest["parent_commit"], manifest["fork_commit"])
    expected = _build_manifest(**{key: manifest[key] for key in ("assets_root", "study_root", "source_root", "parent_commit", "fork_commit", "parent_gitlink")},
                               soft_env=manifest["runtime_environments"]["soft"], hard_env=manifest["runtime_environments"]["hard"], assets_manifest=assets,
                               training_options=manifest.get("training_options"), site=manifest.get("site"),
                               prologue_limit_seconds=manifest["prologue_limit_seconds"],
                               admission_mode=admission_mode(manifest))
    if manifest != expected:
        raise ValueError("production manifest differs from the source/profile/asset contract")
    for arm in manifest["arms"]:
        path = Path(arm["run_root"]) / "profile.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("missing regular per-arm production profile")
        expected_arm = _seal({"profile_id": manifest["profile_id"],
                              "study_manifest_content_sha256": manifest["manifest_content_sha256"], "arm": arm})
        if json.loads(path.read_text()) != expected_arm:
            raise ValueError("per-arm production profile differs from study manifest")
    return manifest


def manifest_arm(manifest: Mapping[str, Any], identifier: str) -> dict[str, Any]:
    resolve_arm(identifier)
    rows = [row for row in manifest["arms"] if row["arm_id"] == identifier]
    if len(rows) != 1:
        raise ValueError("production manifest has no unique requested arm")
    return dict(rows[0])


def phase_command(manifest: Mapping[str, Any], identifier: str, phase: str, *, resume_from_path=None) -> list[str]:
    arm = manifest_arm(manifest, identifier)
    return [arm["python_bin"], "-m", "verl.trainer.main_ppo", *production_overrides(
        identifier, manifest["assets_root"], arm["run_root"], phase=phase, resume_from_path=resume_from_path,
        source_root=manifest["source_root"],
        training_options=manifest.get("training_options"), runtime_manifest=manifest.get("runtime_manifest"),
        site=manifest.get("site"),
    )]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-assets", "verify-assets"):
        sub = commands.add_parser(name)
        sub.add_argument("--assets-root", type=Path, required=True)
        if name == "prepare-assets":
            sub.add_argument("--cache-dir", type=Path, required=True)
    sub = commands.add_parser("materialize")
    for name in ("assets-root", "study-root", "source-root"):
        sub.add_argument("--" + name, type=Path, required=True)
    for name in ("soft-env", "hard-env", "shared-env"):
        sub.add_argument("--" + name, type=Path)
    sub.add_argument("--profile", choices=(PROFILE_ID, LORA_PROFILE_ID), default=PROFILE_ID)
    sub.add_argument("--finetuning", choices=("full", "lora"), default="lora")
    sub.add_argument("--lora-rank", type=int, default=32)
    sub.add_argument("--lora-alpha", type=int, default=64)
    sub.add_argument("--lora-target-modules", default=",".join(LORA_TARGET_MODULES))
    sub.add_argument("--reference-kl", choices=("recipe", "off"), default="recipe")
    for name in ("parent-commit", "fork-commit"):
        sub.add_argument("--" + name, required=True)
    sub.add_argument("--parent-gitlink")
    sub.add_argument("--site", choices=qwen_site.SITE_IDS,
                     help="explicit cluster contract; omission preserves historical Marlowe manifests")
    sub.add_argument("--artifact-root", type=Path)
    sub.add_argument("--wandb-entity", help="explicit W&B account/team; H200 default uses the authenticated user")
    sub.add_argument("--prologue-limit-seconds", type=int, choices=(7200, 10800),
                     help="seal a reviewed startup budget; H200 defaults to 10800 seconds")
    sub.add_argument("--admission-mode", choices=ADMISSION_MODES, default="diagnostics",
                     help="seal direct_training to skip diagnostic runs while retaining allocation/runtime checks")
    for name in ("verify", "overrides", "command"):
        sub = commands.add_parser(name)
        sub.add_argument("--manifest", type=Path, required=True)
        if name != "verify":
            sub.add_argument("--arm", choices=ARM_IDS, required=True)
            sub.add_argument("--phase", choices=PHASES, default="production")
            sub.add_argument("--resume-from-path", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare-assets":
        result = qwen_training.prepare(args.assets_root, args.cache_dir)
    elif args.command == "verify-assets":
        result = qwen_training.verify(args.assets_root)
    elif args.command == "materialize":
        kwargs = vars(args).copy()
        kwargs.pop("command")
        site_id = kwargs.pop("site")
        artifact_root, wandb_entity = kwargs.pop("artifact_root"), kwargs.pop("wandb_entity")
        if site_id is not None:
            kwargs["site"] = qwen_site.resolve_site(site_id, artifact_root, wandb_entity)
        elif artifact_root is not None or wandb_entity is not None:
            parser.error("--artifact-root and --wandb-entity require --site")
        revised = kwargs.pop("profile") == LORA_PROFILE_ID
        shared = kwargs.pop("shared_env")
        choices = {key: kwargs.pop(key) for key in ("finetuning", "lora_rank", "lora_alpha", "lora_target_modules", "reference_kl")}
        choices["lora_target_modules"] = tuple(choices["lora_target_modules"].split(","))
        if revised:
            if shared is None or kwargs["soft_env"] is not None or kwargs["hard_env"] is not None:
                parser.error("LoRA/FA3 requires --shared-env, without --soft-env/--hard-env")
            kwargs.update(soft_env=shared, hard_env=shared, training_options=TrainingOptions(**choices))
        elif shared is not None or kwargs["soft_env"] is None or kwargs["hard_env"] is None:
            parser.error("legacy production requires --soft-env and --hard-env")
        elif TrainingOptions(**choices) != TrainingOptions():
            parser.error("LoRA/FA3 training options require the revised production profile")
        result = materialize_manifest(**kwargs)
    else:
        manifest = verify_manifest(args.manifest)
        if args.command == "verify":
            result = manifest
        else:
            result = phase_command(manifest, args.arm, args.phase, resume_from_path=args.resume_from_path)
            if args.command == "overrides":
                result = result[3:]
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
