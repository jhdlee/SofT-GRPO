"""Race-safe, pinned model staging for the ICL-only evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .icl import (
    LEGACY_STUDY_ID,
    MODEL_SOURCES,
    SOFTGRPO_MODEL_ID,
    SOFTGRPO_MODEL_REVISION,
    SOFTGRPO_MODEL_SUBFOLDER,
    STARTING_MODEL_ID,
    STARTING_MODEL_REVISION,
    get_study_profile,
    prepare_icl_dataset,
    verify_icl_dataset,
)
from .icl_runtime import canonical_json_bytes, sha256_file


ASSET_SCHEMA_VERSION = 1
ASSET_PROTOCOL = "opd-softgrpo-icl-assets-v1"

# Historical public constants remain byte-for-byte compatible with existing
# manifests and callers. New studies resolve their own isolated specifications
# through ``model_specs_for_study``.
MODEL_SPECS = {
    label: dict(spec) for label, spec in MODEL_SOURCES.items()
}


def model_specs_for_study(study: str | None = None) -> dict[str, dict[str, Any]]:
    profile = get_study_profile(study)
    return {
        label: dict(profile.model_sources[label])
        for label in profile.model_labels
    }


def _manifest_study(manifest: Mapping[str, Any], requested: str | None) -> str:
    declared = manifest.get("study_id", LEGACY_STUDY_ID)
    if not isinstance(declared, str):
        raise ValueError("ICL asset manifest study_id is invalid")
    get_study_profile(declared)
    if requested is not None and declared != requested:
        raise ValueError(
            "ICL asset manifest belongs to study %s, not %s"
            % (declared, requested)
        )
    return declared


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".%s." % path.name, suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_json_bytes(dict(value)))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def model_inventory(model_root: Path | str) -> dict[str, Any]:
    root = Path(model_root).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("model root must be a real directory: %s" % root)
    files = []
    for path in sorted(root.rglob("*")):
        if ".cache" in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            raise ValueError("staged model may not contain symlinks: %s" % path)
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    names = {entry["path"] for entry in files}
    if "config.json" not in names or not any(
        name.endswith((".safetensors", ".bin")) for name in names
    ):
        raise ValueError("staged checkpoint lacks config or model weights")
    if not any(name.startswith("tokenizer") for name in names):
        raise ValueError("staged checkpoint lacks tokenizer files")
    digest = hashlib.sha256(canonical_json_bytes(files)).hexdigest()
    return {"path": str(root), "tree_sha256": digest, "files": files}


def _stage_one(destination: Path, spec: Mapping[str, Any], cache_dir: Path) -> None:
    if destination.exists():
        marker = destination / ".opd_source.json"
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != dict(spec):
            raise ValueError("existing staged model has no matching pinned-source marker")
        model_inventory(destination)
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("model staging requires huggingface-hub") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".%s." % destination.name, dir=destination.parent)
    )
    download_root = temporary / "download"
    try:
        subfolder = spec["subfolder"]
        prefix = None if subfolder is None else str(subfolder).rstrip("/")
        patterns = None if prefix is None else [prefix + "/*", prefix + "/**/*"]
        snapshot_download(
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            cache_dir=str(cache_dir),
            local_dir=str(download_root),
            allow_patterns=patterns,
        )
        source = download_root if subfolder is None else download_root / str(subfolder)
        if not source.is_dir():
            raise RuntimeError("pinned Hugging Face subfolder was not materialized")
        local_cache = source / ".cache"
        if local_cache.exists():
            shutil.rmtree(local_cache)
        _atomic_write(source / ".opd_source.json", spec)
        model_inventory(source)
        os.replace(source, destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def verify_icl_assets(
    root: Path | str, *, study: str | None = None
) -> dict[str, Any]:
    asset_root = Path(root).expanduser().resolve()
    manifest_path = asset_root / "asset_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("ICL asset manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claimed = manifest.get("content_sha256")
    unsigned = dict(manifest)
    unsigned.pop("content_sha256", None)
    if claimed != hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest():
        raise ValueError("ICL asset manifest content hash changed")
    study_id = _manifest_study(manifest, study)
    profile = get_study_profile(study_id)
    model_specs = model_specs_for_study(study_id)
    if (
        manifest.get("schema_version") != ASSET_SCHEMA_VERSION
        or manifest.get("protocol") != profile.asset_protocol
        or manifest.get("model_specs") != model_specs
    ):
        raise ValueError("ICL asset protocol or pins changed")
    if study_id != LEGACY_STUDY_ID and manifest.get("study_id") != study_id:
        raise ValueError("non-legacy ICL assets must declare their study_id")
    verify_icl_dataset(asset_root / "data")
    observed = {
        label: model_inventory(asset_root / "models" / label)
        for label in model_specs
    }
    expected = manifest.get("models")
    if not isinstance(expected, Mapping) or set(expected) != set(model_specs):
        raise ValueError("ICL asset model inventory is missing")
    expected_paths = {
        "data": str(asset_root / "data"),
        **{
            label: str(asset_root / "models" / label)
            for label in model_specs
        },
    }
    if manifest.get("paths") != expected_paths:
        raise ValueError("ICL asset paths differ from the isolated root")
    models_root = asset_root / "models"
    staged_labels = {
        path.name
        for path in models_root.iterdir()
        if not path.name.startswith(".")
    }
    if staged_labels != set(model_specs):
        raise ValueError("ICL asset root mixes models from different studies")
    for label in model_specs:
        if observed[label]["tree_sha256"] != expected.get(label, {}).get("tree_sha256"):
            raise ValueError("%s model tree changed" % label)
    return manifest


def prepare_icl_assets(
    root: Path | str,
    cache_dir: Path | str,
    *,
    study: str | None = None,
) -> dict[str, Any]:
    """Stage one study's exact checkpoints and data under an isolated root."""

    try:
        from filelock import FileLock
    except ImportError as error:
        raise RuntimeError("race-safe preparation requires filelock") from error
    asset_root = Path(root).expanduser().resolve()
    cache = Path(cache_dir).expanduser().resolve()
    profile = get_study_profile(study)
    study_id = profile.study_id
    model_specs = model_specs_for_study(study_id)
    asset_root.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    with FileLock(str(asset_root / ".prepare.lock")):
        manifest_path = asset_root / "asset_manifest.json"
        if manifest_path.exists():
            return verify_icl_assets(asset_root, study=study_id)
        models_root = asset_root / "models"
        if models_root.exists():
            staged_labels = {
                path.name
                for path in models_root.iterdir()
                if not path.name.startswith(".")
            }
            unexpected = staged_labels - set(model_specs)
            if unexpected:
                raise ValueError(
                    "ICL asset root already contains models from another study: %s"
                    % sorted(unexpected)
                )
        for label, spec in model_specs.items():
            _stage_one(asset_root / "models" / label, spec, cache / "huggingface")
        data_manifest = prepare_icl_dataset(asset_root / "data", cache / "datasets")
        models = {
            label: model_inventory(asset_root / "models" / label)
            for label in model_specs
        }
        manifest = {
            "schema_version": ASSET_SCHEMA_VERSION,
            "protocol": profile.asset_protocol,
            "model_specs": model_specs,
            "models": models,
            "data_content_sha256": data_manifest["content_sha256"],
            "paths": {"data": str(asset_root / "data")},
        }
        if study_id != LEGACY_STUDY_ID:
            manifest["study_id"] = study_id
        manifest["paths"].update(
            {
                label: str(asset_root / "models" / label)
                for label in model_specs
            }
        )
        manifest["content_sha256"] = hashlib.sha256(
            canonical_json_bytes(manifest)
        ).hexdigest()
        _atomic_write(manifest_path, manifest)
        return verify_icl_assets(asset_root, study=study_id)
