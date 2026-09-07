"""Sealed Qwen3 seven-arm production profiles, separate from historical studies.

This module prepares/verifies assets and configuration; it never submits or runs
training. The controller consumes the phase commands emitted here.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from . import qwen_training, study
from .manifest import canonical_sha256, file_sha256, validate_sealed_content


PROFILE_ID = "qwen3-math-seven-arm-v1"
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


def resolve_arm(identifier: str) -> study.ArmSpec:
    if identifier not in ARM_IDS:
        raise ValueError(f"unknown Qwen production arm: {identifier}")
    spec = study.resolve_arm(identifier).spec
    if spec.opd_enabled and spec.opd_mode == "auxiliary" and "beta0p1" not in identifier:
        spec = replace(spec, beta_base=1.0)
    return spec


def arm_contract(identifier: str) -> dict[str, Any]:
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
) -> list[str]:
    """Emit unique Hydra overrides; invocation limits never shorten the horizon."""
    spec = resolve_arm(identifier)
    metadata = phase_metadata(identifier, run_root, phase)
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
    return [("++" if key.startswith("trainer.production_") or key == "actor_rollout_ref.rollout.production_engine_isolation" else "") + key + "="
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
                    parent_gitlink, soft_env, hard_env, assets_manifest) -> dict[str, Any]:
    for name, value in (("parent_commit", parent_commit), ("fork_commit", fork_commit), ("parent_gitlink", parent_gitlink)):
        _commit(value, name)
    if parent_gitlink != fork_commit:
        raise ValueError("parent Gitlink must match production fork")
    validate_sealed_content(assets_manifest)
    if assets_manifest.get("profile_id") != PROMPT_PROFILE_ID or assets_manifest.get("model") != MODEL:
        raise ValueError("production assets must use the authenticated pinned Qwen profile")
    assets_root, study_root, source_root, soft_env, hard_env = map(_absolute, (assets_root, study_root, source_root, soft_env, hard_env))
    base = {
        "schema_version": 1, "profile_id": PROFILE_ID, "protocol": "qwen3-production-submission-v1",
        "source_root": str(source_root), "parent_commit": parent_commit,
        "fork_commit": fork_commit, "parent_gitlink": parent_gitlink,
        "assets_root": str(assets_root), "study_root": str(study_root),
        "assets_manifest_content_sha256": assets_manifest["manifest_content_sha256"],
        "asset_profile_id": PROMPT_PROFILE_ID, "model": dict(MODEL),
        "runtime_environments": {"soft": str(soft_env), "hard": str(hard_env)},
        "runtime_package_pins": {kind: dict(pins) for kind, pins in RUNTIME_PACKAGE_PINS.items()},
        "resources": dict(RESOURCES), "prologue_limit_seconds": PROLOGUE_LIMIT_SECONDS,
        "base_config_sha256": file_sha256(source_root / "3rdparty/SofT-GRPO/verl-0.4.x/verl/trainer/config/ppo_trainer.yaml"),
        "arm_order": list(ARM_IDS), "arms": [],
    }
    for identifier in ARM_IDS:
        spec = resolve_arm(identifier)
        run_root = study_root / "arms" / identifier
        environment = soft_env if spec.rollout_kind == "native_soft" else hard_env
        contract = arm_contract(identifier)
        identity = canonical_sha256([PROFILE_ID, identifier, parent_commit, fork_commit,
                                     assets_manifest["manifest_content_sha256"], str(run_root), str(environment), contract])
        phases = {phase: phase_metadata(identifier, run_root, phase) for phase in PHASES}
        production = production_overrides(identifier, assets_root, run_root, source_root=source_root)
        base["arms"].append({
            "arm_id": identifier, "account": spec.account, "run_root": str(run_root),
            "environment_root": str(environment), "python_bin": str(environment / "bin/python"),
            "runtime_packages": dict(RUNTIME_PACKAGE_PINS["soft" if spec.rollout_kind == "native_soft" else "hard"]),
            "wandb_run_id": "qprod-" + identity[:24], "wandb_project": PRODUCTION_PROJECT,
            "contract": contract, "contract_sha256": canonical_sha256(contract),
            "phases": phases, "production_overrides": production,
            "production_overrides_sha256": canonical_sha256(production),
        })
    return _seal(base)


def build_manifest(*, assets_root, study_root, source_root, parent_commit, fork_commit,
                   soft_env, hard_env, parent_gitlink=None) -> dict[str, Any]:
    assets = qwen_training.verify(assets_root)
    _verify_source(_absolute(source_root), parent_commit, fork_commit)
    return _build_manifest(assets_root=assets_root, study_root=study_root, source_root=source_root,
                           parent_commit=parent_commit, fork_commit=fork_commit,
                           parent_gitlink=fork_commit if parent_gitlink is None else parent_gitlink,
                           soft_env=soft_env, hard_env=hard_env, assets_manifest=assets)


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
            "profile_id": PROFILE_ID, "study_manifest_content_sha256": manifest["manifest_content_sha256"],
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
    if manifest.get("profile_id") != PROFILE_ID:
        raise ValueError("not a Qwen seven-arm production manifest")
    if path.resolve() != Path(manifest["study_root"]) / "manifest.json":
        raise ValueError("production manifest is not in its sealed study root")
    assets = qwen_training.verify(manifest["assets_root"]) if verify_assets else qwen_training._read_manifest(Path(manifest["assets_root"]) / "manifest.json")
    if verify_source:
        _verify_source(Path(manifest["source_root"]), manifest["parent_commit"], manifest["fork_commit"])
    expected = _build_manifest(**{key: manifest[key] for key in ("assets_root", "study_root", "source_root", "parent_commit", "fork_commit", "parent_gitlink")},
                               soft_env=manifest["runtime_environments"]["soft"], hard_env=manifest["runtime_environments"]["hard"], assets_manifest=assets)
    if manifest != expected:
        raise ValueError("production manifest differs from the source/profile/asset contract")
    for arm in manifest["arms"]:
        path = Path(arm["run_root"]) / "profile.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("missing regular per-arm production profile")
        expected_arm = _seal({"profile_id": PROFILE_ID,
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
    for name in ("assets-root", "study-root", "source-root", "soft-env", "hard-env"):
        sub.add_argument("--" + name, type=Path, required=True)
    for name in ("parent-commit", "fork-commit"):
        sub.add_argument("--" + name, required=True)
    sub.add_argument("--parent-gitlink")
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
        result = materialize_manifest(**{key: value for key, value in vars(args).items() if key != "command"})
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
