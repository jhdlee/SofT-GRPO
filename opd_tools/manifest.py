"""Canonical hashing and manifest construction for prepared OPD data."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from .constants import (
    DATA_PROTOCOL,
    DATA_SCHEMA_VERSION,
    MATH_DATASET_CONFIG,
    MATH_DATASET_ID,
    MATH_DATASET_REVISION,
    MATH_SPLIT_SEED,
    MODEL_ID,
    MODEL_REVISION,
    SOFTGRPO_UPSTREAM_COMMIT,
)
from .data import MathCleaningReport
from .records import RecordBundle


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256(b"opd-ordered-records-v1\0")
    for record in records:
        encoded = canonical_json_bytes(record)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def build_math_data_manifest(
    splits: Mapping[str, Sequence[RecordBundle]],
    cleaning_report: MathCleaningReport,
    split_seed: int = MATH_SPLIT_SEED,
) -> Dict[str, Any]:
    if set(splits) != {"train", "validation"}:
        raise ValueError("MATH data manifest requires train and validation splits")
    split_payload = {}
    all_ids = set()
    for split in ("train", "validation"):
        bundles = splits[split]
        student = [bundle.student.to_dict() for bundle in bundles]
        teacher = [bundle.teacher.to_dict() for bundle in bundles]
        reward = [bundle.reward.to_dict() for bundle in bundles]
        student_ids = [record["example_id"] for record in student]
        if student_ids != [record["example_id"] for record in teacher] or student_ids != [
            record["example_id"] for record in reward
        ]:
            raise ValueError("record channels are not identically ordered")
        if len(student_ids) != len(set(student_ids)) or all_ids.intersection(student_ids):
            raise ValueError("record IDs overlap within or across splits")
        all_ids.update(student_ids)
        split_payload[split] = {
            "count": len(bundles),
            "ids_sha256": canonical_sha256(student_ids),
            "student_sha256": ordered_records_sha256(student),
            "teacher_sha256": ordered_records_sha256(teacher),
            "reward_sha256": ordered_records_sha256(reward),
        }
    manifest = {
        "schema_version": DATA_SCHEMA_VERSION,
        "protocol": DATA_PROTOCOL,
        "upstream_commit": SOFTGRPO_UPSTREAM_COMMIT,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "source": {
            "id": MATH_DATASET_ID,
            "config": MATH_DATASET_CONFIG,
            "revision": MATH_DATASET_REVISION,
        },
        "cleaning": cleaning_report.to_dict(),
        "split": {
            "method": "subject-level-largest-remainder-sha256-ranking-v1",
            "strata": ["type", "level"],
            "seed": split_seed,
            "splits": split_payload,
        },
    }
    manifest["manifest_content_sha256"] = canonical_sha256(manifest)
    return manifest


def validate_manifest_content(manifest: Mapping[str, Any]) -> None:
    payload = dict(manifest)
    observed_hash = payload.pop("manifest_content_sha256", None)
    if payload.get("schema_version") != DATA_SCHEMA_VERSION:
        raise ValueError("unsupported data manifest schema")
    if payload.get("protocol") != DATA_PROTOCOL:
        raise ValueError("unsupported data manifest protocol")
    if observed_hash != canonical_sha256(payload):
        raise ValueError("data manifest content hash mismatch")


def validate_sealed_content(manifest: Mapping[str, Any]) -> None:
    payload = dict(manifest)
    observed_hash = payload.pop("manifest_content_sha256", None)
    if observed_hash != canonical_sha256(payload):
        raise ValueError("manifest content hash mismatch")


def write_manifest_atomic(
    path: Path,
    manifest: Mapping[str, Any],
    validator: Optional[Callable[[Mapping[str, Any]], None]] = validate_manifest_content,
) -> None:
    """Write a validated manifest atomically on the destination filesystem."""

    if validator is not None:
        validator(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % destination.name,
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(manifest) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
