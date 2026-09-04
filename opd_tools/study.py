"""Authoritative arm registry for the seed-11 MATH production study.

The registry is deliberately dependency-free so login-node launchers can use
it without importing Torch, Ray, Hydra, or a rollout backend.  Every consumer
must resolve an arm here instead of maintaining a second shell/Python matrix.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple


STUDY_PROTOCOL = "opd-softgrpo-math-seven-arm-v1"
STUDY_SEED = 11
TOTAL_ROLLOUT_ITERATIONS = 109
WARMUP_FRACTION = 0.10
WARMUP_ITERATIONS = 11
NATIVE_REFERENCE_KL_COEFFICIENT = 0.001
SOFTGRPO_UPSTREAM_BASE_COMMIT = "8d3c61380b15c3400818da5ce41c62c293a1bfb4"
SUBMISSION_MANIFEST_SCHEMA_VERSION = 1

ACCOUNT_M000120 = "marlowe-m000120-pm06"
ACCOUNT_M000215 = "marlowe-m000215-pm06"
ALLOWED_ACCOUNTS = (ACCOUNT_M000120, ACCOUNT_M000215)


@dataclass(frozen=True)
class ArmSpec:
    """One immutable production row and its arm-specific runtime settings."""

    arm_id: str
    account: str
    rollout_kind: str
    objective_mode: str
    group_size: int
    opd_enabled: bool
    opd_mode: str
    loss_support: str
    beta_base: float
    schedule: str
    warmup_fraction: float
    teacher_type: str
    ema_decay: float
    trajectory_gate: str
    native_reference_kl: bool
    wandb_tags: Tuple[str, ...]

    @property
    def experiment_name(self) -> str:
        return self.arm_id

    @property
    def slurm_job_name(self) -> str:
        return "opd-" + self.arm_id.replace("_", "-")

    @property
    def inference_mode(self) -> str:
        return "hard_token" if self.rollout_kind == "categorical" else "native_soft"

    def as_manifest(self) -> Dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload.update(
            {
                "experiment_name": self.experiment_name,
                "inference_mode": self.inference_mode,
                "protocol": STUDY_PROTOCOL,
                "seed": STUDY_SEED,
                "total_rollout_iterations": TOTAL_ROLLOUT_ITERATIONS,
                "warmup_iterations": (
                    WARMUP_ITERATIONS if self.schedule != "constant" else 0
                ),
                "native_reference_kl_coefficient": (
                    NATIVE_REFERENCE_KL_COEFFICIENT
                    if self.native_reference_kl
                    else 0.0
                ),
            }
        )
        payload["wandb_tags"] = list(self.wandb_tags)
        return payload


def _arm(
    arm_id: str,
    account: str,
    *,
    rollout_kind: str = "native_soft",
    objective_mode: str = "grpo",
    group_size: int = 8,
    opd_enabled: bool = False,
    opd_mode: str = "auxiliary",
    loss_support: str = "latent_only",
    beta_base: float = 0.001,
    schedule: str = "warmup_constant",
    teacher_type: str = "ema",
    trajectory_gate: str = "all",
    native_reference_kl: bool = True,
    tags: Sequence[str] = (),
) -> ArmSpec:
    return ArmSpec(
        arm_id=arm_id,
        account=account,
        rollout_kind=rollout_kind,
        objective_mode=objective_mode,
        group_size=group_size,
        opd_enabled=opd_enabled,
        opd_mode=opd_mode,
        loss_support=loss_support,
        beta_base=beta_base,
        schedule=schedule,
        warmup_fraction=WARMUP_FRACTION,
        teacher_type=teacher_type,
        ema_decay=0.99,
        trajectory_gate=trajectory_gate,
        native_reference_kl=native_reference_kl,
        wandb_tags=tuple(tags),
    )


ARM_SPECS: Tuple[ArmSpec, ...] = (
    _arm(
        "hardgrpo_math_s11",
        ACCOUNT_M000120,
        rollout_kind="categorical",
        tags=("hard_grpo", "baseline"),
    ),
    _arm(
        "softgrpo_math_s11",
        ACCOUNT_M000215,
        tags=("soft_grpo", "baseline"),
    ),
    _arm(
        "softgrpo_math_opd_s11",
        ACCOUNT_M000120,
        objective_mode="auxiliary",
        opd_enabled=True,
        loss_support="all_response",
        tags=("soft_grpo", "opd", "primary", "ema", "all"),
    ),
    _arm(
        "softgrpo_math_opd_posadv_s11",
        ACCOUNT_M000215,
        objective_mode="auxiliary",
        opd_enabled=True,
        loss_support="all_response",
        trajectory_gate="positive_advantage",
        tags=("soft_grpo", "opd", "positive_advantage"),
    ),
    _arm(
        "softgrpo_math_opd_current_s11",
        ACCOUNT_M000120,
        objective_mode="auxiliary",
        opd_enabled=True,
        loss_support="all_response",
        teacher_type="current_actor",
        tags=("soft_grpo", "opd", "current_actor"),
    ),
    _arm(
        "softgrpo_math_opd_beta0p1_s11",
        ACCOUNT_M000215,
        objective_mode="auxiliary",
        opd_enabled=True,
        loss_support="all_response",
        beta_base=0.1,
        tags=("soft_grpo", "opd", "beta_0.1"),
    ),
    _arm(
        "softopd_math_s11",
        ACCOUNT_M000120,
        objective_mode="standalone",
        group_size=1,
        opd_enabled=True,
        opd_mode="standalone",
        loss_support="all_response",
        beta_base=1.0,
        schedule="constant",
        native_reference_kl=False,
        tags=("soft_opd", "standalone", "ema"),
    ),
)

ARM_IDS = tuple(spec.arm_id for spec in ARM_SPECS)
_ARM_BY_ID: Mapping[str, ArmSpec] = {spec.arm_id: spec for spec in ARM_SPECS}

@dataclass(frozen=True)
class ArmResolution:
    requested_id: str
    spec: ArmSpec


def validate_registry() -> None:
    if len(ARM_SPECS) != 7 or len(_ARM_BY_ID) != 7:
        raise RuntimeError("the production registry must contain exactly seven arms")
    if ARM_IDS != (
        "hardgrpo_math_s11",
        "softgrpo_math_s11",
        "softgrpo_math_opd_s11",
        "softgrpo_math_opd_posadv_s11",
        "softgrpo_math_opd_current_s11",
        "softgrpo_math_opd_beta0p1_s11",
        "softopd_math_s11",
    ):
        raise RuntimeError("production arm order or identity drifted")
    for spec in ARM_SPECS:
        if spec.account not in ALLOWED_ACCOUNTS:
            raise RuntimeError("arm uses an unapproved Slurm account")
        if spec.group_size not in (1, 8):
            raise RuntimeError("only G=1 and G=8 are registered")
        if spec.rollout_kind not in ("categorical", "native_soft"):
            raise RuntimeError("unknown rollout kind")
        if spec.opd_mode not in ("auxiliary", "standalone"):
            raise RuntimeError("unknown OPD mode")
        if spec.loss_support not in ("latent_only", "all_response"):
            raise RuntimeError("unknown OPD loss support")
        if spec.schedule not in ("constant", "warmup_constant", "warmup_decay"):
            raise RuntimeError("unknown OPD schedule")
        if spec.opd_enabled and spec.loss_support != "all_response":
            raise RuntimeError("new OPD production arms must use all_response")
        if spec.opd_mode == "standalone":
            if not spec.opd_enabled or spec.group_size != 1:
                raise RuntimeError("standalone OPD requires enabled OPD and G=1")
            if spec.native_reference_kl or spec.trajectory_gate != "all":
                raise RuntimeError("standalone OPD cannot use reference KL or gating")
            if spec.beta_base != 1.0 or spec.schedule != "constant":
                raise RuntimeError("standalone OPD must use constant beta 1.0")
        elif spec.group_size != 8:
            raise RuntimeError("every non-standalone arm must use G=8")
        if not spec.opd_enabled and spec.objective_mode != "grpo":
            raise RuntimeError("non-OPD arms must use the GRPO objective")


def resolve_arm(identifier: str) -> ArmResolution:
    validate_registry()
    if identifier in _ARM_BY_ID:
        return ArmResolution(identifier, _ARM_BY_ID[identifier])
    allowed = ", ".join(ARM_IDS)
    raise ValueError("unknown arm %r; expected one of: %s" % (identifier, allowed))


def hydra_overrides(identifier: str) -> Tuple[str, ...]:
    """Return the complete arm-specific Hydra override set."""

    resolution = resolve_arm(identifier)
    spec = resolution.spec
    enabled = str(spec.opd_enabled).lower()
    use_reference = str(spec.native_reference_kl).lower()
    common = (
        "algorithm.opd.enabled=" + enabled,
        "algorithm.opd.mode=" + spec.opd_mode,
        "algorithm.opd.loss_support=" + spec.loss_support,
        "algorithm.opd.beta_base=" + str(spec.beta_base),
        "algorithm.opd.schedule=" + spec.schedule,
        "algorithm.opd.warmup_fraction=" + str(spec.warmup_fraction),
        "algorithm.opd.teacher.type=" + spec.teacher_type,
        "algorithm.opd.teacher.ema_decay=" + str(spec.ema_decay),
        "algorithm.opd.kl_direction=teacher_to_student",
        "algorithm.opd.trajectory_gate=" + spec.trajectory_gate,
        "algorithm.opd.prompt_template=sdpg",
        "algorithm.opd.temperature=1.0",
        "actor_rollout_ref.actor.use_kl_loss=" + use_reference,
        "actor_rollout_ref.actor.kl_loss_coef="
        + (str(NATIVE_REFERENCE_KL_COEFFICIENT) if spec.native_reference_kl else "0.0"),
        "actor_rollout_ref.rollout.n=" + str(spec.group_size),
    )
    if spec.rollout_kind == "categorical":
        rollout = (
            "actor_rollout_ref.rollout.name=vllm",
            "actor_rollout_ref.rollout.top_p=1.0",
            "actor_rollout_ref.rollout.top_k=-1",
            "actor_rollout_ref.rollout.temperature=1.0",
            "actor_rollout_ref.rollout.enable_soft_thinking=false",
            "actor_rollout_ref.rollout.add_noise_gumbel_softmax=false",
            "actor_rollout_ref.rollout.noise_gumbel=false",
            "actor_rollout_ref.rollout.noise_on_logits=false",
            "actor_rollout_ref.rollout.val_kwargs.do_sample=true",
            "actor_rollout_ref.rollout.val_kwargs.temperature=0.6",
            "actor_rollout_ref.rollout.val_kwargs.top_p=0.95",
            "actor_rollout_ref.rollout.val_kwargs.top_k=30",
            "actor_rollout_ref.rollout.val_kwargs.n=1",
        )
    else:
        rollout = (
            "actor_rollout_ref.rollout.name=sglang",
            "actor_rollout_ref.rollout.top_p=0.95",
            "actor_rollout_ref.rollout.top_k=5",
            "actor_rollout_ref.rollout.temperature=1.0",
            "actor_rollout_ref.rollout.after_thinking_temperature=1.0",
            "actor_rollout_ref.rollout.after_thinking_top_p=0.95",
            "actor_rollout_ref.rollout.after_thinking_top_k=5",
            "actor_rollout_ref.rollout.gumbel_softmax_temperature=0.1",
            "actor_rollout_ref.rollout.enable_soft_thinking=true",
            "actor_rollout_ref.rollout.add_noise_gumbel_softmax=true",
            "actor_rollout_ref.rollout.noise_gumbel=true",
            "actor_rollout_ref.rollout.val_kwargs.do_sample=true",
            "actor_rollout_ref.rollout.val_kwargs.temperature=0.6",
            "actor_rollout_ref.rollout.val_kwargs.top_p=0.95",
            "actor_rollout_ref.rollout.val_kwargs.top_k=5",
            "actor_rollout_ref.rollout.val_kwargs.n=1",
        )
    return common + rollout


def arm_contract_sha256(identifier: str) -> str:
    resolution = resolve_arm(identifier)
    payload = resolution.spec.as_manifest()
    payload["requested_id"] = resolution.requested_id
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_submission_manifest(
    identifier: str,
    *,
    parent_commit: str,
    submodule_commit: str,
    parent_gitlink: str,
) -> Dict[str, Any]:
    """Build the immutable per-arm contract sealed before Slurm submission."""

    for label, commit in (
        ("parent_commit", parent_commit),
        ("submodule_commit", submodule_commit),
        ("parent_gitlink", parent_gitlink),
    ):
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            raise ValueError("%s must be a full lowercase Git commit" % label)
    if parent_gitlink != submodule_commit:
        raise ValueError("parent Gitlink does not match the checked-out SofT-GRPO commit")

    resolution = resolve_arm(identifier)
    payload: Dict[str, Any] = {
        "schema_version": SUBMISSION_MANIFEST_SCHEMA_VERSION,
        "study_protocol": STUDY_PROTOCOL,
        "arm": resolution.spec.as_manifest(),
        "arm_contract_sha256": arm_contract_sha256(identifier),
        "hydra_overrides": list(hydra_overrides(identifier)),
        "source": {
            "parent_commit": parent_commit,
            "softgrpo_commit": submodule_commit,
            "parent_gitlink": parent_gitlink,
            "softgrpo_upstream_base_commit": SOFTGRPO_UPSTREAM_BASE_COMMIT,
        },
    }
    payload["manifest_content_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()
    return payload


def verify_submission_manifest(
    manifest: Mapping[str, Any],
    *,
    identifier: str,
    parent_commit: str,
    submodule_commit: str,
) -> Dict[str, Any]:
    """Authenticate a materialized manifest against the live source checkout."""

    observed = dict(manifest)
    observed_hash = observed.pop("manifest_content_sha256", None)
    expected_hash = hashlib.sha256(
        json.dumps(observed, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()
    if observed_hash != expected_hash:
        raise ValueError("submission manifest content hash mismatch")
    expected = build_submission_manifest(
        identifier,
        parent_commit=parent_commit,
        submodule_commit=submodule_commit,
        parent_gitlink=submodule_commit,
    )
    if dict(manifest) != expected:
        raise ValueError("submission manifest does not match the arm and source checkout")
    return expected


def materialize_submission_manifest(path: Path, manifest: Mapping[str, Any]) -> str:
    """Create a manifest atomically, or authenticate an identical existing file."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(dict(manifest), sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    file_hash = hashlib.sha256(encoded).hexdigest()
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise ValueError("submission manifest destination is not a regular file")
        if destination.read_bytes() != encoded:
            raise ValueError("refusing to overwrite a different submission manifest")
        return file_hash
    temporary = destination.with_name(".%s.tmp-%d" % (destination.name, os.getpid()))
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or destination.read_bytes() != encoded:
                raise ValueError("refusing to overwrite a different submission manifest")
        directory = os.open(str(destination.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_hash


def stable_wandb_run_id(identifier: str, manifest_sha256: str) -> str:
    if len(manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in manifest_sha256
    ):
        raise ValueError("manifest_sha256 must be a lowercase SHA-256")
    resolution = resolve_arm(identifier)
    digest = hashlib.sha256(
        (arm_contract_sha256(identifier) + "\0" + manifest_sha256).encode("ascii")
    ).hexdigest()[:16]
    return "%s-%s" % (resolution.spec.arm_id, digest)


def _field(resolution: ArmResolution, name: str) -> str:
    spec = resolution.spec
    fields = {
        "account": spec.account,
        "arm_contract_sha256": arm_contract_sha256(spec.arm_id),
        "arm_id": spec.arm_id,
        "experiment_name": spec.experiment_name,
        "inference_mode": spec.inference_mode,
        "objective_mode": spec.objective_mode,
        "rollout_kind": spec.rollout_kind,
        "slurm_job_name": spec.slurm_job_name,
        "wandb_tags": ",".join(spec.wandb_tags),
    }
    try:
        return fields[name]
    except KeyError as exc:
        raise ValueError("unknown launcher field: %s" % name) from exc


def main(argv: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="print the seven canonical arm IDs")
    resolve_parser = subparsers.add_parser("resolve", help="resolve one arm")
    resolve_parser.add_argument("arm")
    resolve_parser.add_argument(
        "--field",
        choices=(
            "account",
            "arm_contract_sha256",
            "arm_id",
            "experiment_name",
            "inference_mode",
            "objective_mode",
            "rollout_kind",
            "slurm_job_name",
            "wandb_tags",
        ),
    )
    overrides_parser = subparsers.add_parser(
        "hydra-overrides", help="print one override per line"
    )
    overrides_parser.add_argument("arm")
    manifest_parser = subparsers.add_parser("manifest", help="print canonical JSON")
    manifest_parser.add_argument("arm")
    materialize_parser = subparsers.add_parser(
        "materialize-manifest", help="atomically write a source-bound arm manifest"
    )
    materialize_parser.add_argument("arm")
    materialize_parser.add_argument("--output", required=True, type=Path)
    materialize_parser.add_argument("--parent-commit", required=True)
    materialize_parser.add_argument("--submodule-commit", required=True)
    materialize_parser.add_argument("--parent-gitlink", required=True)
    verify_parser = subparsers.add_parser(
        "verify-manifest", help="authenticate a source-bound arm manifest"
    )
    verify_parser.add_argument("arm")
    verify_parser.add_argument("--path", required=True, type=Path)
    verify_parser.add_argument("--parent-commit", required=True)
    verify_parser.add_argument("--submodule-commit", required=True)
    run_id_parser = subparsers.add_parser("wandb-run-id")
    run_id_parser.add_argument("arm")
    run_id_parser.add_argument("manifest_sha256")
    args = parser.parse_args(tuple(argv) or None)

    if args.command == "list":
        print("\n".join(ARM_IDS))
    elif args.command == "resolve":
        resolution = resolve_arm(args.arm)
        if args.field:
            print(_field(resolution, args.field))
        else:
            print(json.dumps(resolution.spec.as_manifest(), sort_keys=True))
    elif args.command == "hydra-overrides":
        print("\n".join(hydra_overrides(args.arm)))
    elif args.command == "manifest":
        resolution = resolve_arm(args.arm)
        payload = resolution.spec.as_manifest()
        payload["requested_id"] = args.arm
        payload["arm_contract_sha256"] = arm_contract_sha256(args.arm)
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    elif args.command == "materialize-manifest":
        payload = build_submission_manifest(
            args.arm,
            parent_commit=args.parent_commit,
            submodule_commit=args.submodule_commit,
            parent_gitlink=args.parent_gitlink,
        )
        file_hash = materialize_submission_manifest(args.output, payload)
        print(file_hash)
    elif args.command == "verify-manifest":
        payload = json.loads(args.path.read_text(encoding="utf-8"))
        verify_submission_manifest(
            payload,
            identifier=args.arm,
            parent_commit=args.parent_commit,
            submodule_commit=args.submodule_commit,
        )
        print(hashlib.sha256(args.path.read_bytes()).hexdigest())
    elif args.command == "wandb-run-id":
        print(stable_wandb_run_id(args.arm, args.manifest_sha256))
    return 0


validate_registry()


if __name__ == "__main__":
    raise SystemExit(main())
