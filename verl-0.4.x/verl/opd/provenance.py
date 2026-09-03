"""Pragmatic, fail-closed provenance for exact PPO checkpoint resume.

The study already authenticates model and data assets before training.  This
module records those sealed-manifest identities together with the Git commit,
the resume-semantic Hydra configuration, and relevant installed package
versions.  It deliberately does not fingerprint source trees, bytecode, or an
entire environment directory.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

CHECKPOINT_PROVENANCE_SCHEMA_VERSION = 1
SOFTGRPO_UPSTREAM_BASE_COMMIT = "8d3c61380b15c3400818da5ce41c62c293a1bfb4"
RESUME_INVOCATION_ONLY_CONFIG_FIELDS = (
    "trainer.max_rollout_iterations_per_invocation",
    "trainer.requeue_signal_file",
    "trainer.resume_from_path",
    "trainer.resume_mode",
)
CORE_RUNTIME_PACKAGES = (
    "datasets",
    "flash-attn",
    "flashinfer-python",
    "math-verify",
    "numpy",
    "pyarrow",
    "ray",
    "sglang",
    "torch",
    "transformers",
    "verl",
)
INFORMATIONAL_PACKAGES = ("wandb",)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"checkpoint provenance field {field!r} is not a SHA-256")
    return value


def _json_copy(value: Any) -> Any:
    """Return a detached, JSON-canonical copy and reject non-finite values."""

    return json.loads(_canonical_json_bytes(value))


def _resolved_config_identity(resolved_config: Mapping[str, Any]) -> dict[str, object]:
    """Hash the resolved training configuration, excluding invocation controls.

    ``resume_mode`` must change from ``disable``/an explicit first invocation to
    ``auto``/``resume_path`` when exercising exact resume.  The smoke-only
    per-invocation stop count is similarly not training semantics.  These are
    the only exclusions; optimization, rollout, OPD, data, model, logging, and
    checkpoint cadence settings remain covered.
    """

    if not isinstance(resolved_config, Mapping):
        raise TypeError("resolved Hydra config must be a mapping")
    exact = _json_copy(dict(resolved_config))
    normalized = _json_copy(exact)
    trainer = normalized.get("trainer")
    if not isinstance(trainer, dict):
        raise RuntimeError("resolved Hydra config has no trainer mapping")
    for dotted_path in RESUME_INVOCATION_ONLY_CONFIG_FIELDS:
        _, field = dotted_path.split(".", 1)
        trainer.pop(field, None)
    return {
        "full_sha256": _canonical_sha256(exact),
        "resume_semantic_sha256": _canonical_sha256(normalized),
        "excluded_invocation_fields": list(RESUME_INVOCATION_ONLY_CONFIG_FIELDS),
    }


def _run_git(source_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _resolve_source_commit(source_root: Path) -> str:
    root = source_root.expanduser().resolve()
    commit = _run_git(root, "rev-parse", "--verify", "HEAD")
    if _GIT_COMMIT_RE.fullmatch(commit) is None:
        raise RuntimeError(f"source HEAD is not a full Git commit: {commit!r}")
    top_level = Path(_run_git(root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != root:
        raise RuntimeError(f"source root is not the Git top level: {root} != {top_level}")
    dirty = _run_git(root, "status", "--porcelain", "--untracked-files=normal")
    if dirty:
        raise RuntimeError(
            "checkpoint provenance requires a clean nested source checkout; "
            f"first change: {dirty.splitlines()[0]}"
        )
    return commit


def _read_sealed_manifest(path: Path, label: str) -> tuple[dict[str, Any], str]:
    manifest_path = path.expanduser().resolve()
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError(f"{label} manifest is not a regular file: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} manifest is unreadable: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise RuntimeError(f"{label} manifest must be a JSON object")
    payload = dict(manifest)
    observed_content_hash = payload.pop("manifest_content_sha256", None)
    _require_sha256(observed_content_hash, f"{label}.manifest_content_sha256")
    if observed_content_hash != _canonical_sha256(payload):
        raise RuntimeError(f"{label} manifest content hash mismatch")
    return manifest, _file_sha256(manifest_path)


def _model_identity(model_path: str) -> dict[str, object]:
    root = Path(model_path).expanduser().resolve()
    manifest, manifest_file_hash = _read_sealed_manifest(root / "manifest.json", "model")
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise RuntimeError("model manifest has no model identity")
    model_id = model.get("id")
    revision = model.get("resolved_revision")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError("model manifest has no model ID")
    if not isinstance(revision, str) or _GIT_COMMIT_RE.fullmatch(revision) is None:
        raise RuntimeError("model manifest resolved revision is not a full commit")
    return {
        "id": model_id,
        "resolved_revision": revision,
        "manifest_file_sha256": manifest_file_hash,
        "manifest_content_sha256": _require_sha256(
            manifest.get("manifest_content_sha256"), "model.manifest_content_sha256"
        ),
        "inventory_sha256": _require_sha256(
            manifest.get("inventory_sha256"), "model.inventory_sha256"
        ),
    }


def _as_local_paths(value: object, field: str) -> list[Path]:
    if isinstance(value, str):
        raw_paths: Sequence[object] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_paths = value
    else:
        raise RuntimeError(f"resolved Hydra config field {field!r} is not a path or list")
    paths = []
    for item in raw_paths:
        if not isinstance(item, str) or not item:
            raise RuntimeError(f"resolved Hydra config field {field!r} has an invalid path")
        path = Path(item).expanduser().resolve()
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"configured data file is not regular: {path}")
        paths.append(path)
    if not paths:
        raise RuntimeError(f"resolved Hydra config field {field!r} is empty")
    return paths


def _data_identity(train_files: object, val_files: object) -> dict[str, object]:
    configured = _as_local_paths(train_files, "data.train_files") + _as_local_paths(
        val_files, "data.val_files"
    )
    by_parent: dict[Path, set[str]] = {}
    for path in configured:
        by_parent.setdefault(path.parent, set()).add(path.name)
    manifests = []
    for parent, names in sorted(by_parent.items(), key=lambda item: str(item[0])):
        manifest, manifest_file_hash = _read_sealed_manifest(
            parent / "manifest.json", "data"
        )
        files = manifest.get("files")
        if not isinstance(files, dict) or not names.issubset(files):
            raise RuntimeError(
                "configured train/validation files are absent from the sealed data manifest"
            )
        manifests.append(
            {
                "configured_files": sorted(names),
                "manifest_file_sha256": manifest_file_hash,
                "manifest_content_sha256": _require_sha256(
                    manifest.get("manifest_content_sha256"),
                    "data.manifest_content_sha256",
                ),
            }
        )
    payload: dict[str, object] = {"manifests": manifests}
    return {**payload, "identity_sha256": _canonical_sha256(payload)}


def _environment_identity(
    *,
    package_versions: Optional[Mapping[str, Optional[str]]] = None,
    informational_versions: Optional[Mapping[str, Optional[str]]] = None,
    python_version: Optional[str] = None,
    python_implementation: Optional[str] = None,
    torch_cuda_version: Optional[str] = None,
    version_resolver: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, object]:
    def resolve_versions(
        names: Sequence[str], supplied: Optional[Mapping[str, Optional[str]]]
    ) -> dict[str, Optional[str]]:
        versions = {}
        for distribution in names:
            if supplied is not None:
                value = supplied.get(distribution)
            else:
                try:
                    value = version_resolver(distribution)
                except importlib.metadata.PackageNotFoundError:
                    value = None
            versions[distribution] = None if value is None else str(value)
        return versions

    core_packages = resolve_versions(CORE_RUNTIME_PACKAGES, package_versions)
    informational_packages = resolve_versions(
        INFORMATIONAL_PACKAGES, informational_versions
    )
    if torch_cuda_version is None:
        try:
            import torch

            torch_cuda_version = torch.version.cuda
        except ImportError:
            torch_cuda_version = None
    gated_payload: dict[str, object] = {
        "python": {
            "implementation": python_implementation or platform.python_implementation(),
            "version": python_version or platform.python_version(),
        },
        "core_packages": core_packages,
        "torch_cuda": torch_cuda_version,
    }
    return {
        **gated_payload,
        "informational_packages": informational_packages,
        "identity_sha256": _canonical_sha256(gated_payload),
    }


def build_checkpoint_provenance(
    resolved_config: Mapping[str, Any],
    *,
    source_root: Optional[Path] = None,
    source_commit: Optional[str] = None,
    environment_identity: Optional[Mapping[str, Any]] = None,
) -> dict[str, object]:
    """Build the identity sealed into every committed study checkpoint."""

    config = _json_copy(dict(resolved_config))
    actor_rollout_ref = config.get("actor_rollout_ref")
    data = config.get("data")
    if not isinstance(actor_rollout_ref, dict) or not isinstance(
        actor_rollout_ref.get("model"), dict
    ):
        raise RuntimeError("resolved Hydra config has no actor model mapping")
    if not isinstance(data, dict):
        raise RuntimeError("resolved Hydra config has no data mapping")
    model_path = actor_rollout_ref["model"].get("path")
    if not isinstance(model_path, str) or not model_path:
        raise RuntimeError("resolved Hydra config has no actor model path")

    if source_commit is None:
        if source_root is None:
            source_root = Path(__file__).resolve().parents[3]
        source_commit = _resolve_source_commit(source_root)
    if _GIT_COMMIT_RE.fullmatch(source_commit) is None:
        raise RuntimeError("nested source commit must be a full 40-character Git SHA")

    environment = (
        _environment_identity()
        if environment_identity is None
        else _json_copy(dict(environment_identity))
    )
    _validate_environment_identity(environment)
    payload: dict[str, object] = {
        "schema_version": CHECKPOINT_PROVENANCE_SCHEMA_VERSION,
        "source": {
            "commit": source_commit,
            "upstream_base_commit": SOFTGRPO_UPSTREAM_BASE_COMMIT,
        },
        "resolved_hydra_config": _resolved_config_identity(config),
        "model": _model_identity(model_path),
        "data": _data_identity(data.get("train_files"), data.get("val_files")),
        "environment": environment,
    }
    resume_payload = {
        "source": payload["source"],
        "resolved_hydra_config_sha256": payload["resolved_hydra_config"][
            "resume_semantic_sha256"
        ],
        "model": payload["model"],
        "data": payload["data"],
        "environment_sha256": payload["environment"]["identity_sha256"],
    }
    result = {
        **payload,
        "resume_identity_sha256": _canonical_sha256(resume_payload),
        "identity_sha256": _canonical_sha256(payload),
    }
    validate_checkpoint_provenance(result)
    return result


def _validate_environment_identity(environment: object) -> None:
    if not isinstance(environment, dict):
        raise RuntimeError("checkpoint provenance environment must be an object")
    if set(environment) != {
        "python",
        "core_packages",
        "torch_cuda",
        "informational_packages",
        "identity_sha256",
    }:
        raise RuntimeError("checkpoint provenance environment fields differ")
    python = environment["python"]
    core_packages = environment["core_packages"]
    informational_packages = environment["informational_packages"]
    if (
        not isinstance(python, dict)
        or not isinstance(python.get("implementation"), str)
        or not isinstance(python.get("version"), str)
        or not isinstance(core_packages, dict)
        or set(core_packages) != set(CORE_RUNTIME_PACKAGES)
        or any(
            version is not None and not isinstance(version, str)
            for version in core_packages.values()
        )
        or not isinstance(informational_packages, dict)
        or set(informational_packages) != set(INFORMATIONAL_PACKAGES)
        or any(
            version is not None and not isinstance(version, str)
            for version in informational_packages.values()
        )
        or (
            environment["torch_cuda"] is not None
            and not isinstance(environment["torch_cuda"], str)
        )
    ):
        raise RuntimeError("checkpoint provenance environment identity is malformed")
    gated_payload = {
        "python": python,
        "core_packages": core_packages,
        "torch_cuda": environment["torch_cuda"],
    }
    observed = _require_sha256(
        environment.get("identity_sha256"), "environment.identity_sha256"
    )
    if observed != _canonical_sha256(gated_payload):
        raise RuntimeError("checkpoint provenance environment hash mismatch")


def validate_checkpoint_provenance(provenance: object) -> dict[str, object]:
    """Validate all nested identities and the top-level provenance seal."""

    if not isinstance(provenance, dict):
        raise RuntimeError("checkpoint provenance must be an object")
    payload = {
        key: value
        for key, value in provenance.items()
        if key not in {"identity_sha256", "resume_identity_sha256"}
    }
    if set(payload) != {
        "schema_version",
        "source",
        "resolved_hydra_config",
        "model",
        "data",
        "environment",
    }:
        raise RuntimeError("checkpoint provenance fields differ")
    if payload["schema_version"] != CHECKPOINT_PROVENANCE_SCHEMA_VERSION:
        raise RuntimeError("unsupported checkpoint provenance schema")

    source = payload["source"]
    if (
        not isinstance(source, dict)
        or set(source) != {"commit", "upstream_base_commit"}
        or not isinstance(source["commit"], str)
        or _GIT_COMMIT_RE.fullmatch(source["commit"]) is None
        or source["upstream_base_commit"] != SOFTGRPO_UPSTREAM_BASE_COMMIT
    ):
        raise RuntimeError("checkpoint provenance source commit is invalid")

    config = payload["resolved_hydra_config"]
    if (
        not isinstance(config, dict)
        or set(config)
        != {
            "full_sha256",
            "resume_semantic_sha256",
            "excluded_invocation_fields",
        }
        or config.get("excluded_invocation_fields")
        != list(RESUME_INVOCATION_ONLY_CONFIG_FIELDS)
    ):
        raise RuntimeError("checkpoint provenance resolved config identity is invalid")
    _require_sha256(config.get("full_sha256"), "resolved_hydra_config.full_sha256")
    _require_sha256(
        config.get("resume_semantic_sha256"),
        "resolved_hydra_config.resume_semantic_sha256",
    )

    model = payload["model"]
    if (
        not isinstance(model, dict)
        or set(model)
        != {
            "id",
            "resolved_revision",
            "manifest_file_sha256",
            "manifest_content_sha256",
            "inventory_sha256",
        }
        or not isinstance(model.get("id"), str)
        or not model["id"]
        or not isinstance(model.get("resolved_revision"), str)
        or _GIT_COMMIT_RE.fullmatch(model["resolved_revision"]) is None
    ):
        raise RuntimeError("checkpoint provenance model identity is invalid")
    for field in (
        "manifest_file_sha256",
        "manifest_content_sha256",
        "inventory_sha256",
    ):
        _require_sha256(model.get(field), f"model.{field}")

    data = payload["data"]
    if not isinstance(data, dict) or set(data) != {"manifests", "identity_sha256"}:
        raise RuntimeError("checkpoint provenance data identity is invalid")
    manifests = data.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        raise RuntimeError("checkpoint provenance has no data manifest identity")
    for manifest in manifests:
        if (
            not isinstance(manifest, dict)
            or set(manifest)
            != {
                "configured_files",
                "manifest_file_sha256",
                "manifest_content_sha256",
            }
            or not isinstance(manifest.get("configured_files"), list)
            or not manifest["configured_files"]
            or any(not isinstance(name, str) or not name for name in manifest["configured_files"])
        ):
            raise RuntimeError("checkpoint provenance data manifest entry is invalid")
        _require_sha256(
            manifest.get("manifest_file_sha256"), "data.manifest_file_sha256"
        )
        _require_sha256(
            manifest.get("manifest_content_sha256"),
            "data.manifest_content_sha256",
        )
    data_payload = {"manifests": manifests}
    if _require_sha256(data.get("identity_sha256"), "data.identity_sha256") != _canonical_sha256(
        data_payload
    ):
        raise RuntimeError("checkpoint provenance data hash mismatch")

    _validate_environment_identity(payload["environment"])
    resume_payload = {
        "source": payload["source"],
        "resolved_hydra_config_sha256": payload["resolved_hydra_config"][
            "resume_semantic_sha256"
        ],
        "model": payload["model"],
        "data": payload["data"],
        "environment_sha256": payload["environment"]["identity_sha256"],
    }
    if _require_sha256(
        provenance.get("resume_identity_sha256"),
        "provenance.resume_identity_sha256",
    ) != _canonical_sha256(resume_payload):
        raise RuntimeError("checkpoint provenance resume identity hash mismatch")
    if _require_sha256(
        provenance.get("identity_sha256"), "provenance.identity_sha256"
    ) != _canonical_sha256(payload):
        raise RuntimeError("checkpoint provenance identity hash mismatch")
    return _json_copy(provenance)


def assert_checkpoint_provenance_matches(
    observed: object, expected: object
) -> None:
    """Reject resume with a precise component-level mismatch message."""

    observed_valid = validate_checkpoint_provenance(observed)
    expected_valid = validate_checkpoint_provenance(expected)
    for component in ("source", "model", "data"):
        if observed_valid[component] != expected_valid[component]:
            raise RuntimeError(f"checkpoint provenance mismatch: {component}")
    if (
        observed_valid["environment"]["identity_sha256"]
        != expected_valid["environment"]["identity_sha256"]
    ):
        raise RuntimeError("checkpoint provenance mismatch: environment")
    observed_config = observed_valid["resolved_hydra_config"]
    expected_config = expected_valid["resolved_hydra_config"]
    if (
        observed_config["resume_semantic_sha256"]
        != expected_config["resume_semantic_sha256"]
        or observed_config["excluded_invocation_fields"]
        != expected_config["excluded_invocation_fields"]
    ):
        raise RuntimeError("checkpoint provenance mismatch: resolved_hydra_config")
    if (
        observed_valid["resume_identity_sha256"]
        != expected_valid["resume_identity_sha256"]
    ):
        raise RuntimeError("checkpoint provenance mismatch: resume_identity_sha256")


__all__ = [
    "CHECKPOINT_PROVENANCE_SCHEMA_VERSION",
    "CORE_RUNTIME_PACKAGES",
    "INFORMATIONAL_PACKAGES",
    "RESUME_INVOCATION_ONLY_CONFIG_FIELDS",
    "SOFTGRPO_UPSTREAM_BASE_COMMIT",
    "assert_checkpoint_provenance_matches",
    "build_checkpoint_provenance",
    "validate_checkpoint_provenance",
]
