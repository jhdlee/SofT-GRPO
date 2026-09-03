"""Atomically stage and authenticate the pinned local model snapshot.

Usage::

    python -m opd_tools.assets --output-dir /scratch/.../assets/model \
        --cache-dir /scratch/.../hf-cache
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from .constants import MODEL_ID, MODEL_REVISION
from .manifest import (
    canonical_sha256,
    file_sha256,
    validate_sealed_content,
    write_manifest_atomic,
)

MODEL_ASSET_PROTOCOL = "opd-local-model-snapshot-v1"


def _inventory(root: Path) -> Dict[str, Dict[str, Any]]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.name == "manifest.json":
            continue
        if path.is_symlink():
            raise ValueError("model snapshot must be self-contained, not symlinked: %s" % path)
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            files[relative] = {"size": path.stat().st_size, "sha256": file_sha256(path)}
    return files


def _validate_required_transformers_files(files: Mapping[str, Mapping[str, Any]]) -> None:
    required = {"config.json", "tokenizer_config.json"}
    missing = required - set(files)
    if missing:
        raise ValueError("model snapshot lacks required files: %s" % sorted(missing))
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError("model snapshot contains no Safetensors weights")


def verify_model_snapshot(output_dir: Path) -> Dict[str, Any]:
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise ValueError("local model snapshot is incomplete: %s" % root)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("local model manifest is unreadable") from error
    payload = dict(manifest)
    observed_content_hash = payload.pop("manifest_content_sha256", None)
    if (
        payload.get("protocol") != MODEL_ASSET_PROTOCOL
        or payload.get("model")
        != {
            "id": MODEL_ID,
            "requested_revision": MODEL_REVISION,
            "resolved_revision": MODEL_REVISION,
        }
        or observed_content_hash != canonical_sha256(payload)
        or payload.get("transformers_local_path") != str(root.resolve())
    ):
        raise ValueError("local model manifest identity/content differs")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("local model manifest has no file inventory")
    _validate_required_transformers_files(files)
    actual = _inventory(root)
    if set(actual) != set(files):
        raise ValueError("local model snapshot file inventory differs")
    for name, record in files.items():
        if actual[name] != record:
            raise ValueError("local model file failed authentication: %s" % name)
    if manifest.get("inventory_sha256") != canonical_sha256(files):
        raise ValueError("local model inventory hash differs")
    return manifest


def stage_model_snapshot(
    output_dir: Path,
    cache_dir: Optional[Path] = None,
    snapshot_download_fn: Optional[Callable[..., str]] = None,
    model_info_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Resolve the immutable commit, download it, hash it, and rename once."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        return verify_model_snapshot(destination)
    if snapshot_download_fn is None or model_info_fn is None:
        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as error:
            raise RuntimeError("model staging requires huggingface-hub") from error
        if snapshot_download_fn is None:
            snapshot_download_fn = snapshot_download
        if model_info_fn is None:
            model_info_fn = HfApi().model_info

    info = model_info_fn(repo_id=MODEL_ID, revision=MODEL_REVISION)
    resolved_revision = getattr(info, "sha", None)
    if resolved_revision != MODEL_REVISION:
        raise ValueError(
            "model revision resolved to %r instead of %s"
            % (resolved_revision, MODEL_REVISION)
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".%s." % destination.name, dir=str(destination.parent))
    )
    try:
        downloaded = Path(
            snapshot_download_fn(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                local_dir=str(temporary),
                cache_dir=None if cache_dir is None else str(Path(cache_dir)),
            )
        ).resolve()
        if downloaded != temporary.resolve():
            raise ValueError("snapshot downloader wrote outside the atomic staging directory")
        # huggingface-hub may add local-dir cache metadata. It is unnecessary
        # for offline Transformers loading and not part of the model snapshot.
        local_metadata = temporary / ".cache"
        if local_metadata.exists():
            shutil.rmtree(str(local_metadata))
        files = _inventory(temporary)
        _validate_required_transformers_files(files)
        config = json.loads((temporary / "config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not config.get("model_type"):
            raise ValueError("model config lacks model_type")
        manifest = {
            "protocol": MODEL_ASSET_PROTOCOL,
            "model": {
                "id": MODEL_ID,
                "requested_revision": MODEL_REVISION,
                "resolved_revision": resolved_revision,
            },
            "transformers_local_path": str(destination),
            "model_type": config["model_type"],
            "files": files,
            "inventory_sha256": canonical_sha256(files),
        }
        manifest["manifest_content_sha256"] = canonical_sha256(manifest)
        write_manifest_atomic(
            temporary / "manifest.json", manifest, validator=validate_sealed_content
        )

        # Verification before publication needs the final path in the manifest,
        # but otherwise authenticates the temporary tree byte-for-byte.
        payload = json.loads((temporary / "manifest.json").read_text(encoding="utf-8"))
        if payload != manifest or _inventory(temporary) != files:
            raise ValueError("temporary model snapshot changed before publication")
        os.replace(str(temporary), str(destination))
        directory_descriptor = os.open(str(destination.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return verify_model_snapshot(destination)
    except BaseException:
        shutil.rmtree(str(temporary), ignore_errors=True)
        raise


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    destination = args.output_dir.expanduser().resolve()
    manifest = stage_model_snapshot(destination, args.cache_dir)
    print(
        json.dumps(
            {
                "model_path": str(destination),
                "resolved_revision": manifest["model"]["resolved_revision"],
                "inventory_sha256": manifest["inventory_sha256"],
                "file_count": len(manifest["files"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
