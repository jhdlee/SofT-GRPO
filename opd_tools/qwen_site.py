"""Explicit cluster and artifact contracts for Qwen launch manifests.

The site dictionary is included in its caller's sealed manifest. Missing site
fields in historical manifests retain the original Marlowe behavior.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping


SITE_IDS = ("marlowe-h100", "mbzuai-h200")
_PROFILES = {
    "marlowe-h100": {
        "scheduler": {"partition": "batch", "account": None, "qos": "medium", "constraint": None},
        "gpu": {"name_contains": "H100", "compute_capability": [9, 0]},
        "modules": ["slurm/slurm/25.05.2", "gcc/13.1.0", "stockcuda/12.6.2"],
        "production_prologue_limit_seconds": 7200,
    },
    "mbzuai-h200": {
        "scheduler": {"partition": "main", "account": "k2m", "qos": "k2m", "constraint": "nvidia_h200"},
        "gpu": {"name_contains": "H200", "compute_capability": [9, 0]},
        "modules": [],
        "production_prologue_limit_seconds": 10800,
    },
}


def shared_storage_root() -> Path:
    """Resolve the user's shared-storage link, never create a home fallback."""
    link = Path.home() / "shrd"
    if not link.is_symlink() or not link.is_dir():
        raise ValueError("local H200 storage requires an existing ~/shrd directory symlink")
    root = link.resolve(strict=True)
    if root.is_relative_to(Path.home().resolve()):
        raise ValueError("~/shrd must resolve outside the home directory")
    return root


def resolve_site(site_id: str = "marlowe-h100", artifact_root=None, wandb_entity=None) -> dict[str, Any]:
    if site_id not in _PROFILES:
        raise ValueError(f"unknown Qwen cluster site: {site_id}")
    if site_id == "mbzuai-h200" and artifact_root is None:
        artifact_root = shared_storage_root() / "opd-latent-reasoning"
    result = {"schema_version": 1, "site_id": site_id, **copy.deepcopy(_PROFILES[site_id]),
              "artifact_root": str(Path(artifact_root).expanduser().resolve()) if artifact_root is not None else None,
              "wandb_entity": ("jhdlee" if site_id == "marlowe-h100" else "columbia-homies") if wandb_entity is None else wandb_entity}
    return validate_site(result)


def validate_site(site: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(site, Mapping) or site.get("site_id") not in _PROFILES:
        raise ValueError("invalid Qwen site contract")
    identifier = site["site_id"]
    profile = _PROFILES[identifier]
    required = {"schema_version", "site_id", "artifact_root", "wandb_entity", *profile}
    if set(site) != required or type(site["schema_version"]) is not int or site["schema_version"] != 1:
        raise ValueError("invalid Qwen site contract fields")
    if any(json.dumps(site[key], sort_keys=True) != json.dumps(value, sort_keys=True)
           for key, value in profile.items()):
        raise ValueError("Qwen site scheduler, hardware, modules, or default budget differ from the profile")
    entity = site["wandb_entity"]
    if entity is not None and (not isinstance(entity, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", entity)):
        raise ValueError("W&B entity must be an explicit account/team name or null")
    artifact = site["artifact_root"]
    if identifier == "mbzuai-h200" and not isinstance(artifact, str):
        raise ValueError("H200 site requires an artifact root under resolved ~/shrd")
    if artifact is not None:
        if not isinstance(artifact, str) or str(Path(artifact).expanduser().resolve()) != artifact:
            raise ValueError("site artifact root must be an absolute resolved path")
        if identifier == "mbzuai-h200":
            shared = shared_storage_root()
            root = Path(artifact)
            if root == shared or not root.is_relative_to(shared):
                raise ValueError("H200 artifact root must be a dedicated directory within resolved ~/shrd")
    return copy.deepcopy(dict(site))


def site_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return resolve_site() if "site" not in manifest else validate_site(manifest["site"])


def validate_artifact_path(path, site: Mapping[str, Any], label: str = "artifact") -> Path:
    site = validate_site(site)
    resolved = Path(path).expanduser().resolve()
    root = site["artifact_root"]
    if root is not None and not resolved.is_relative_to(Path(root)):
        raise ValueError(f"{label} must remain within the sealed artifact root: {root}")
    return resolved


def validate_artifact_tree(path, site: Mapping[str, Any], label: str = "artifact tree") -> Path:
    """Check existing directory aliases before launch, without reading assets.

    Following directory aliases catches an escape hidden inside an internal
    alias. Tracking resolved directories also handles W&B's ``latest-run``
    links and prevents directory cycles from causing an unbounded traversal.
    """
    site = validate_site(site)
    root = validate_artifact_path(path, site, label)
    if site["artifact_root"] is None:
        return root
    artifact_root = Path(site["artifact_root"])
    pending, visited = [root], set()
    while pending:
        directory = pending.pop()
        if directory in visited or not directory.is_dir():
            continue
        visited.add(directory)
        with os.scandir(directory) as entries:
            for entry in entries:
                candidate = Path(entry.path)
                if entry.is_symlink():
                    try:
                        candidate = candidate.resolve()
                    except RuntimeError as error:
                        raise ValueError(f"{label} contains an unresolved symlink cycle") from error
                    if not candidate.is_relative_to(artifact_root):
                        raise ValueError(f"{label} symlink escapes the sealed artifact root: {entry.path}")
                if candidate.is_dir():
                    pending.append(candidate)
    return root


def validate_prologue_limit(value: int, site: Mapping[str, Any]) -> int:
    site = validate_site(site)
    allowed = (7200, 10800) if site["site_id"] == "mbzuai-h200" else (7200,)
    if type(value) is not int or value not in allowed:
        raise ValueError("production prologue limit must be explicitly sealed as 7200 or 10800 seconds")
    return value


def validate_scheduler_allocation(fields: Mapping[str, Any], site: Mapping[str, Any], *, account=None) -> dict[str, Any]:
    """Check live ``scontrol show job -o`` fields against the sealed site.

    Legacy Marlowe profiles keep per-arm accounts, which must be provided by the
    caller. The explicit H200 profile has one account for every arm.
    """
    if not isinstance(fields, Mapping):
        raise ValueError("live Slurm allocation fields must be a mapping")
    scheduler = validate_site(site)["scheduler"]
    expected_account = scheduler["account"] or account
    if expected_account is None or (account is not None and account != expected_account):
        raise ValueError("allocation account differs from the sealed site/arm")
    for key, expected in (("Partition", scheduler["partition"]), ("QOS", scheduler["qos"]),
                          ("Account", expected_account)):
        if fields.get(key) != expected:
            raise ValueError(f"live Slurm {key} differs from the sealed site: expected {expected}")
    constraint = scheduler["constraint"]
    if constraint is not None:
        # For this fixed profile the job requests one concrete feature. Do not
        # accept OR expressions, available-node features, or substring matches.
        features = str(fields.get("Features", ""))
        if "|" in features or constraint not in re.split(r"[,\[\]&*()]", features):
            raise ValueError(f"live Slurm Features must require the sealed constraint: {constraint}")
    return dict(fields)
