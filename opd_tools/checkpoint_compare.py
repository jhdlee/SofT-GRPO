"""Verify and compare committed OPD checkpoint states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CHECKPOINT_SCHEMA_VERSION = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verified_manifest(run_root: Path, step: int) -> dict[str, Any]:
    checkpoint = run_root / f"global_step_{step}"
    manifest_path = checkpoint / "checkpoint_manifest.json"
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise RuntimeError(f"missing committed checkpoint directory: {checkpoint}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported checkpoint schema in {manifest_path}")
    if int(manifest.get("global_step", -1)) != step:
        raise RuntimeError(f"checkpoint step mismatch in {manifest_path}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"invalid checkpoint inventory in {manifest_path}")
    for entry in files:
        relative = entry.get("path")
        if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
            raise RuntimeError(f"unsafe checkpoint inventory path: {relative!r}")
        payload = checkpoint / relative
        if not payload.is_file() or payload.is_symlink():
            raise RuntimeError(f"checkpoint payload is missing or unsafe: {payload}")
        if payload.stat().st_size != int(entry.get("size", -1)):
            raise RuntimeError(f"checkpoint payload size mismatch: {payload}")
        if _sha256(payload) != entry.get("sha256"):
            raise RuntimeError(f"checkpoint payload hash mismatch: {payload}")
    for field in (
        "actor_model_optimizer_tree_sha256",
        "rollout_trajectory_sha256",
    ):
        digest = manifest.get(field)
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f"checkpoint lacks a valid {field}: {manifest_path}")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError(f"checkpoint lacks provenance: {manifest_path}")
    provenance_payload = {
        key: value
        for key, value in provenance.items()
        if key not in {"identity_sha256", "resume_identity_sha256"}
    }
    provenance_sha256 = provenance.get("identity_sha256")
    resume_provenance_sha256 = provenance.get("resume_identity_sha256")
    try:
        resume_payload = {
            "source": provenance_payload["source"],
            "resolved_hydra_config_sha256": provenance_payload[
                "resolved_hydra_config"
            ]["resume_semantic_sha256"],
            "model": provenance_payload["model"],
            "data": provenance_payload["data"],
            "environment_sha256": provenance_payload["environment"][
                "identity_sha256"
            ],
        }
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"checkpoint provenance is malformed: {manifest_path}"
        ) from error
    if (
        not isinstance(provenance_sha256, str)
        or len(provenance_sha256) != 64
        or provenance_sha256 != _canonical_sha256(provenance_payload)
        or manifest.get("provenance_sha256") != provenance_sha256
        or not isinstance(resume_provenance_sha256, str)
        or len(resume_provenance_sha256) != 64
        or resume_provenance_sha256 != _canonical_sha256(resume_payload)
        or manifest.get("resume_provenance_sha256") != resume_provenance_sha256
    ):
        raise RuntimeError(f"checkpoint provenance hash mismatch: {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--label", default="checkpoint parity")
    args = parser.parse_args()
    if args.step < 1:
        parser.error("--step must be positive")
    left = verified_manifest(args.left.resolve(), args.step)
    right = verified_manifest(args.right.resolve(), args.step)
    fields = (
        "rollout_trajectory_sha256",
        "actor_model_optimizer_tree_sha256",
    )
    for field in fields:
        if left[field] != right[field]:
            raise SystemExit(
                f"{args.label} failed: {field} differs "
                f"({left[field]} != {right[field]})"
            )
    print(
        f"{args.label} passed at global_step_{args.step}: "
        f"trajectory={left[fields[0]]} actor={left[fields[1]]}"
    )


if __name__ == "__main__":
    main()
