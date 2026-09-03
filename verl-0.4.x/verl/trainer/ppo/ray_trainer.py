# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import hashlib
import json
import os
import random
import re
import shutil
import stat
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Mapping, Optional, Sequence, Type

import numpy as np
import psutil
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.opd import OPDConfig
from verl.opd.provenance import (
    assert_checkpoint_provenance_matches,
    build_checkpoint_provenance,
    validate_checkpoint_provenance,
)
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.opd_driver import (
    RolloutIntegrityConfig,
    add_canonical_metric_aliases,
    compute_rollout_diagnostics,
    replay_ratio_abs_error_max,
    reward_and_group_metrics,
    schedule_meta_info,
    validate_iteration_metric_contract,
    validate_rollout_integrity,
    validate_validation_metric_contract,
    validation_metric_aliases,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager

WorkerType = Type[Worker]


_CHECKPOINT_MANIFEST = "checkpoint_manifest.json"
_CHECKPOINT_ROLLOUT_METADATA = "rollout_metadata.json"
_CHECKPOINT_TRACKER = "latest_checkpointed_iteration.txt"
_BEST_CHECKPOINT_TRACKER = "best_checkpointed_iteration.txt"
_CHECKPOINT_SCHEMA_VERSION = 2
_CHECKPOINT_ROLLOUT_SCHEMA_VERSION = 1
_CHECKPOINT_ROLLOUT_FIELDS = (
    "prompts",
    "responses",
    "response_mask",
    "attention_mask",
    "rollout_topk_ids",
    "rollout_topk_gumbels",
    "rollout_sampling_seed",
    "gumbel_temperature",
)
_COMMITTED_CHECKPOINT_RE = re.compile(r"^global_step_([0-9]+)$")


def _fsync_directory(path: str) -> None:
    """Persist directory-entry updates before reporting a checkpoint commit."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_text(path: str, value: str) -> None:
    """Write a small control file with replace-and-fsync semantics."""

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    temporary_path = os.path.join(parent, f".{os.path.basename(path)}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary_path, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(parent)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _checkpoint_step_from_name(path: str) -> int:
    match = _COMMITTED_CHECKPOINT_RE.fullmatch(os.path.basename(os.path.normpath(path)))
    if match is None:
        raise RuntimeError(f"not a committed checkpoint name: {path}")
    return int(match.group(1))


def _sha256_file(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_files(checkpoint_dir: str) -> list[tuple[str, str]]:
    """Return regular checkpoint files, rejecting links and special files."""

    checkpoint_dir = os.path.abspath(checkpoint_dir)
    files: list[tuple[str, str]] = []
    for current_root, directory_names, file_names in os.walk(checkpoint_dir, followlinks=False):
        for directory_name in directory_names:
            directory_path = os.path.join(current_root, directory_name)
            if os.path.islink(directory_path):
                raise RuntimeError(f"checkpoint directories may not be symlinks: {directory_path}")
        for file_name in file_names:
            absolute_path = os.path.join(current_root, file_name)
            relative_path = os.path.relpath(absolute_path, checkpoint_dir)
            if relative_path == _CHECKPOINT_MANIFEST:
                continue
            file_stat = os.lstat(absolute_path)
            if not stat.S_ISREG(file_stat.st_mode):
                raise RuntimeError(f"checkpoint entries must be regular files: {absolute_path}")
            files.append((relative_path, absolute_path))
    files.sort(key=lambda item: item[0])
    return files


def _checkpoint_inventory(checkpoint_dir: str, *, sync_files: bool) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for relative_path, absolute_path in _checkpoint_files(checkpoint_dir):
        if sync_files:
            with open(absolute_path, "rb") as handle:
                os.fsync(handle.fileno())
        inventory.append(
            {
                "path": relative_path,
                "size": os.path.getsize(absolute_path),
                "sha256": _sha256_file(absolute_path),
            }
        )
    return inventory


def _inventory_digest(entries: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        digest.update(str(entry["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _tensor_integrity_descriptor(tensor: torch.Tensor) -> dict[str, object]:
    """Hash one dense tensor canonically without serializing a second copy."""

    if not torch.is_tensor(tensor) or tensor.layout != torch.strided:
        raise TypeError("rollout integrity fields must be dense torch tensors")
    cpu_tensor = tensor.detach().to(device="cpu").contiguous()
    raw_bytes = cpu_tensor.view(torch.uint8).numpy()
    digest = hashlib.sha256()
    digest.update(memoryview(raw_bytes))
    return {
        "dtype": str(cpu_tensor.dtype),
        "shape": list(cpu_tensor.shape),
        "numel": cpu_tensor.numel(),
        "sha256": digest.hexdigest(),
    }


def _canonical_json_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity_string(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str)


def _build_rollout_integrity_record(
    batch_tensors: Mapping[str, torch.Tensor],
    *,
    group_ids: Sequence[object],
    example_identities: Sequence[object],
    rollout_iteration: int,
) -> dict[str, object]:
    """Build a deterministic digest of one completed continuous rollout.

    The large tensors are reduced to canonical raw-byte hashes only when a
    checkpoint is requested.  The resulting JSON stays compact while still
    distinguishing every action, mask, request-local seed, and trajectory.
    """

    missing = sorted(set(_CHECKPOINT_ROLLOUT_FIELDS) - set(batch_tensors))
    if missing:
        raise RuntimeError(f"cannot checkpoint rollout without fields: {missing}")
    responses = batch_tensors["responses"]
    if responses.ndim != 2 or responses.shape[0] < 1:
        raise RuntimeError("checkpoint rollout responses must be a nonempty rank-2 tensor")
    trajectory_count, response_length = responses.shape
    if len(group_ids) != trajectory_count or len(example_identities) != trajectory_count:
        raise RuntimeError("checkpoint rollout identities do not match trajectory count")

    response_mask = batch_tensors["response_mask"]
    if response_mask.shape != responses.shape:
        raise RuntimeError("checkpoint response mask does not match responses")
    sampling_seeds = batch_tensors["rollout_sampling_seed"]
    if sampling_seeds.shape != (trajectory_count,):
        raise RuntimeError("checkpoint sampling seeds do not match trajectory count")
    seed_values = sampling_seeds.detach().to(device="cpu", dtype=torch.int64).tolist()
    if any(seed < 0 for seed in seed_values):
        raise RuntimeError("checkpoint rollout contains nondeterministic sampling seeds")

    tensor_view = {
        "prompts": batch_tensors["prompts"],
        "responses": responses,
        "response_mask": response_mask,
        "attention_mask": batch_tensors["attention_mask"],
        "rollout_topk_ids": batch_tensors["rollout_topk_ids"][:, -response_length:],
        "rollout_topk_gumbels": batch_tensors["rollout_topk_gumbels"][:, -response_length:],
        "rollout_sampling_seed": sampling_seeds,
        "gumbel_temperature": batch_tensors["gumbel_temperature"],
    }
    for name, tensor in tensor_view.items():
        if tensor.shape[0] != trajectory_count:
            raise RuntimeError(f"checkpoint rollout field {name} has the wrong batch dimension")
    if tensor_view["rollout_topk_ids"].shape != tensor_view["rollout_topk_gumbels"].shape:
        raise RuntimeError("checkpoint top-k IDs and Gumbels have different shapes")
    if tensor_view["rollout_topk_ids"].ndim != 3:
        raise RuntimeError("checkpoint top-k replay fields must be rank-3")

    identities = [
        {
            "group_id": _identity_string(group_id),
            "example_identity": _identity_string(example_identity),
            "sampling_seed": int(seed),
        }
        for group_id, example_identity, seed in zip(group_ids, example_identities, seed_values)
    ]
    identity_keys = [
        (entry["group_id"], entry["example_identity"], entry["sampling_seed"])
        for entry in identities
    ]
    if len(set(identity_keys)) != trajectory_count:
        raise RuntimeError("checkpoint rollout trajectory identities are not unique")

    payload: dict[str, object] = {
        "schema_version": _CHECKPOINT_ROLLOUT_SCHEMA_VERSION,
        "rollout_iteration": int(rollout_iteration),
        "trajectory_count": trajectory_count,
        "response_length": response_length,
        "identities": identities,
        "fields": {
            name: _tensor_integrity_descriptor(tensor)
            for name, tensor in tensor_view.items()
        },
    }
    return {**payload, "trajectory_sha256": _canonical_json_digest(payload)}


def _verify_rollout_integrity_record(
    record: object,
    *,
    expected_rollout_iteration: int,
) -> dict[str, object]:
    if not isinstance(record, dict):
        raise RuntimeError("checkpoint rollout metadata must be an object")
    trajectory_digest = record.get("trajectory_sha256")
    if not isinstance(trajectory_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", trajectory_digest):
        raise RuntimeError("checkpoint rollout metadata has no valid trajectory digest")
    payload = {key: value for key, value in record.items() if key != "trajectory_sha256"}
    if payload.get("schema_version") != _CHECKPOINT_ROLLOUT_SCHEMA_VERSION:
        raise RuntimeError("unsupported checkpoint rollout metadata schema")
    if payload.get("rollout_iteration") != expected_rollout_iteration:
        raise RuntimeError("checkpoint rollout metadata has the wrong iteration")
    trajectory_count = payload.get("trajectory_count")
    response_length = payload.get("response_length")
    identities = payload.get("identities")
    fields = payload.get("fields")
    if not isinstance(trajectory_count, int) or trajectory_count < 1:
        raise RuntimeError("checkpoint rollout trajectory count is invalid")
    if not isinstance(response_length, int) or response_length < 1:
        raise RuntimeError("checkpoint rollout response length is invalid")
    if not isinstance(identities, list) or len(identities) != trajectory_count:
        raise RuntimeError("checkpoint rollout identity count is invalid")
    identity_keys = []
    for identity in identities:
        if not isinstance(identity, dict) or set(identity) != {
            "group_id",
            "example_identity",
            "sampling_seed",
        }:
            raise RuntimeError("checkpoint rollout identity is malformed")
        if not isinstance(identity["group_id"], str) or not isinstance(
            identity["example_identity"], str
        ):
            raise RuntimeError("checkpoint rollout deterministic identity is invalid")
        if not isinstance(identity["sampling_seed"], int) or identity["sampling_seed"] < 0:
            raise RuntimeError("checkpoint rollout sampling identity is invalid")
        identity_keys.append(
            (identity["group_id"], identity["example_identity"], identity["sampling_seed"])
        )
    if len(set(identity_keys)) != trajectory_count:
        raise RuntimeError("checkpoint rollout identities are not unique")
    if not isinstance(fields, dict) or set(fields) != set(_CHECKPOINT_ROLLOUT_FIELDS):
        raise RuntimeError("checkpoint rollout tensor inventory is invalid")
    for name, descriptor in fields.items():
        if not isinstance(descriptor, dict):
            raise RuntimeError(f"checkpoint rollout descriptor is invalid: {name}")
        shape = descriptor.get("shape")
        if not isinstance(shape, list) or not shape or shape[0] != trajectory_count:
            raise RuntimeError(f"checkpoint rollout tensor shape is invalid: {name}")
        if not isinstance(descriptor.get("dtype"), str):
            raise RuntimeError(f"checkpoint rollout tensor dtype is invalid: {name}")
        if not isinstance(descriptor.get("numel"), int) or descriptor["numel"] < 1:
            raise RuntimeError(f"checkpoint rollout tensor size is invalid: {name}")
        expected_numel = 1
        for dimension in shape:
            if not isinstance(dimension, int) or dimension < 1:
                raise RuntimeError(f"checkpoint rollout tensor shape is invalid: {name}")
            expected_numel *= dimension
        if descriptor["numel"] != expected_numel:
            raise RuntimeError(f"checkpoint rollout tensor element count is invalid: {name}")
        if not isinstance(descriptor.get("sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", descriptor["sha256"]
        ):
            raise RuntimeError(f"checkpoint rollout tensor digest is invalid: {name}")
    response_shape = fields["responses"]["shape"]
    if response_shape != [trajectory_count, response_length]:
        raise RuntimeError("checkpoint rollout response shape is inconsistent")
    if fields["response_mask"]["shape"] != response_shape:
        raise RuntimeError("checkpoint rollout response-mask shape is inconsistent")
    for name in ("rollout_topk_ids", "rollout_topk_gumbels"):
        shape = fields[name]["shape"]
        if len(shape) != 3 or shape[:2] != response_shape:
            raise RuntimeError(f"checkpoint rollout replay shape is inconsistent: {name}")
    if fields["rollout_topk_ids"]["shape"] != fields["rollout_topk_gumbels"]["shape"]:
        raise RuntimeError("checkpoint rollout replay support shapes disagree")
    if fields["rollout_sampling_seed"]["shape"] != [trajectory_count]:
        raise RuntimeError("checkpoint rollout sampling-seed shape is inconsistent")
    if _canonical_json_digest(payload) != trajectory_digest:
        raise RuntimeError("checkpoint rollout trajectory digest mismatch")
    return record


def _write_rollout_integrity_record(checkpoint_dir: str, record: dict[str, object]) -> None:
    _verify_rollout_integrity_record(
        record,
        expected_rollout_iteration=int(record["rollout_iteration"]),
    )
    _atomic_write_text(
        os.path.join(checkpoint_dir, _CHECKPOINT_ROLLOUT_METADATA),
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_checkpoint_manifest(
    checkpoint_dir: str,
    *,
    checkpoint_name: str,
    global_step: int,
    completed_rollout_iteration: int,
    next_rollout_iteration: int,
    optimizer_step: int,
    total_rollout_iterations: int,
    world_size: int,
    reason: str,
    provenance: Mapping[str, object],
    selection_metric_name: Optional[str] = None,
    selection_metric_value: Optional[float] = None,
) -> dict[str, object]:
    """Hash every payload file and make the manifest durable in the temp tree."""

    if _checkpoint_step_from_name(checkpoint_name) != global_step:
        raise RuntimeError("checkpoint name and global step disagree")
    inventory = _checkpoint_inventory(checkpoint_dir, sync_files=True)
    if not inventory:
        raise RuntimeError("refusing to commit an empty checkpoint")
    inventory_by_path = {entry["path"]: entry for entry in inventory}
    for required_path in ("data.pt", "driver_state.pt", _CHECKPOINT_ROLLOUT_METADATA):
        if required_path not in inventory_by_path:
            raise RuntimeError(f"checkpoint payload is missing {required_path}")
    rollout_metadata_path = os.path.join(checkpoint_dir, _CHECKPOINT_ROLLOUT_METADATA)
    with open(rollout_metadata_path, encoding="utf-8") as handle:
        rollout_record = _verify_rollout_integrity_record(
            json.load(handle),
            expected_rollout_iteration=completed_rollout_iteration,
        )
    actor_state_entries = [
        entry
        for entry in inventory
        if re.fullmatch(r"actor/(?:model|optim)_world_size_[0-9]+_rank_[0-9]+[.]pt", str(entry["path"]))
    ]
    if not actor_state_entries:
        raise RuntimeError("checkpoint payload has no actor model/optimizer shards")
    checkpoint_provenance = validate_checkpoint_provenance(dict(provenance))
    opd_teacher_entries = [
        entry
        for entry in inventory
        if str(entry["path"]).startswith("actor/opd_teacher/")
    ]
    manifest: dict[str, object] = {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_name": checkpoint_name,
        "global_step": int(global_step),
        "completed_rollout_iteration": int(completed_rollout_iteration),
        "next_rollout_iteration": int(next_rollout_iteration),
        "optimizer_step": int(optimizer_step),
        "total_rollout_iterations": int(total_rollout_iterations),
        "world_size": int(world_size),
        "reason": str(reason),
        "provenance": checkpoint_provenance,
        "provenance_sha256": checkpoint_provenance["identity_sha256"],
        "resume_provenance_sha256": checkpoint_provenance[
            "resume_identity_sha256"
        ],
        "selection_metric_name": selection_metric_name,
        "selection_metric_value": selection_metric_value,
        "dataloader_state_sha256": inventory_by_path["data.pt"]["sha256"],
        "driver_state_sha256": inventory_by_path["driver_state.pt"]["sha256"],
        "rollout_metadata_file_sha256": inventory_by_path[_CHECKPOINT_ROLLOUT_METADATA]["sha256"],
        "rollout_trajectory_sha256": rollout_record["trajectory_sha256"],
        "payload_tree_sha256": _inventory_digest(inventory),
        # This byte-canonical digest is intended for disabled-OPD versus
        # beta_base=0 smoke parity.  FSDP shard filenames are identical across
        # arms, so equal serialized model/optimizer state gives an equal digest.
        "actor_model_optimizer_tree_sha256": _inventory_digest(actor_state_entries),
        "files": inventory,
    }
    if opd_teacher_entries:
        # Includes every teacher model shard and the per-rank EMA counter.
        # This permits an exact direct-vs-resumed OPD comparison without
        # conflating it with invocation-only driver state.
        manifest["opd_teacher_tree_sha256"] = _inventory_digest(
            opd_teacher_entries
        )
    manifest_path = os.path.join(checkpoint_dir, _CHECKPOINT_MANIFEST)
    temporary_path = f"{manifest_path}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, manifest_path)
        _fsync_directory(checkpoint_dir)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return manifest


def _verify_checkpoint(
    checkpoint_dir: str,
    *,
    require_committed_name: bool = True,
    expected_provenance: Optional[Mapping[str, object]] = None,
) -> dict[str, object]:
    """Fail closed unless a checkpoint manifest and every payload hash agree."""

    checkpoint_dir = os.path.abspath(checkpoint_dir)
    manifest_path = os.path.join(checkpoint_dir, _CHECKPOINT_MANIFEST)
    if not os.path.isdir(checkpoint_dir) or os.path.islink(checkpoint_dir):
        raise RuntimeError(f"checkpoint is not a real directory: {checkpoint_dir}")
    if not os.path.isfile(manifest_path) or os.path.islink(manifest_path):
        raise RuntimeError(f"checkpoint is not committed: {checkpoint_dir}")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported checkpoint manifest schema: {manifest.get('schema_version')}")
    checkpoint_name = manifest.get("checkpoint_name")
    if not isinstance(checkpoint_name, str):
        raise RuntimeError("checkpoint manifest is missing checkpoint_name")
    manifest_step = _checkpoint_step_from_name(checkpoint_name)
    if manifest_step < 1:
        raise RuntimeError("committed checkpoints must follow a completed rollout")
    if manifest.get("global_step") != manifest_step:
        raise RuntimeError("checkpoint manifest step and name disagree")
    if manifest.get("completed_rollout_iteration") != manifest_step - 1:
        raise RuntimeError("checkpoint completed-rollout metadata is inconsistent")
    if manifest.get("next_rollout_iteration") != manifest_step:
        raise RuntimeError("checkpoint next-rollout metadata is inconsistent")
    if not isinstance(manifest.get("optimizer_step"), int) or manifest["optimizer_step"] < 1:
        raise RuntimeError("checkpoint optimizer-step metadata is invalid")
    if not isinstance(manifest.get("total_rollout_iterations"), int) or manifest["total_rollout_iterations"] < manifest_step:
        raise RuntimeError("checkpoint rollout horizon is invalid")
    if not isinstance(manifest.get("world_size"), int) or manifest["world_size"] < 1:
        raise RuntimeError("checkpoint world size is invalid")
    if require_committed_name and os.path.basename(checkpoint_dir) != checkpoint_name:
        raise RuntimeError("checkpoint directory and manifest name disagree")
    checkpoint_provenance = validate_checkpoint_provenance(manifest.get("provenance"))
    if manifest.get("provenance_sha256") != checkpoint_provenance["identity_sha256"]:
        raise RuntimeError("checkpoint manifest provenance digest is inconsistent")
    if (
        manifest.get("resume_provenance_sha256")
        != checkpoint_provenance["resume_identity_sha256"]
    ):
        raise RuntimeError("checkpoint manifest resume-provenance digest is inconsistent")
    if expected_provenance is not None:
        assert_checkpoint_provenance_matches(checkpoint_provenance, expected_provenance)

    expected_entries = manifest.get("files")
    if not isinstance(expected_entries, list) or not expected_entries:
        raise RuntimeError("checkpoint manifest has no file inventory")
    expected: dict[str, dict[str, object]] = {}
    for entry in expected_entries:
        if not isinstance(entry, dict):
            raise RuntimeError("checkpoint manifest contains a malformed file entry")
        relative_path = entry.get("path")
        if not isinstance(relative_path, str) or not relative_path:
            raise RuntimeError("checkpoint manifest contains an invalid relative path")
        normalized_path = os.path.normpath(relative_path)
        if os.path.isabs(relative_path) or normalized_path.startswith("..") or normalized_path != relative_path:
            raise RuntimeError(f"unsafe path in checkpoint manifest: {relative_path}")
        if relative_path in expected:
            raise RuntimeError(f"duplicate path in checkpoint manifest: {relative_path}")
        expected[relative_path] = entry

    actual_files = dict(_checkpoint_files(checkpoint_dir))
    if set(actual_files) != set(expected):
        missing = sorted(set(expected) - set(actual_files))
        extra = sorted(set(actual_files) - set(expected))
        raise RuntimeError(f"checkpoint inventory mismatch: missing={missing}, extra={extra}")
    for relative_path, entry in expected.items():
        absolute_path = actual_files[relative_path]
        actual_size = os.path.getsize(absolute_path)
        if entry.get("size") != actual_size:
            raise RuntimeError(f"checkpoint size mismatch: {relative_path}")
        if entry.get("sha256") != _sha256_file(absolute_path):
            raise RuntimeError(f"checkpoint hash mismatch: {relative_path}")
    if manifest.get("dataloader_state_sha256") != expected.get("data.pt", {}).get("sha256"):
        raise RuntimeError("checkpoint dataloader digest is inconsistent")
    if manifest.get("driver_state_sha256") != expected.get("driver_state.pt", {}).get("sha256"):
        raise RuntimeError("checkpoint driver-state digest is inconsistent")
    if manifest.get("rollout_metadata_file_sha256") != expected.get(
        _CHECKPOINT_ROLLOUT_METADATA, {}
    ).get("sha256"):
        raise RuntimeError("checkpoint rollout-metadata file digest is inconsistent")
    rollout_metadata_path = actual_files.get(_CHECKPOINT_ROLLOUT_METADATA)
    if rollout_metadata_path is None:
        raise RuntimeError("checkpoint rollout metadata is missing")
    with open(rollout_metadata_path, encoding="utf-8") as handle:
        rollout_record = _verify_rollout_integrity_record(
            json.load(handle),
            expected_rollout_iteration=manifest_step - 1,
        )
    if manifest.get("rollout_trajectory_sha256") != rollout_record["trajectory_sha256"]:
        raise RuntimeError("checkpoint rollout trajectory digest is inconsistent")
    expected_entries_sorted = [expected[path] for path in sorted(expected)]
    if manifest.get("payload_tree_sha256") != _inventory_digest(expected_entries_sorted):
        raise RuntimeError("checkpoint payload-tree digest is inconsistent")
    actor_state_entries = [
        entry
        for path, entry in expected.items()
        if re.fullmatch(r"actor/(?:model|optim)_world_size_[0-9]+_rank_[0-9]+[.]pt", path)
    ]
    if not actor_state_entries:
        raise RuntimeError("checkpoint manifest has no actor model/optimizer shards")
    if manifest.get("actor_model_optimizer_tree_sha256") != _inventory_digest(actor_state_entries):
        raise RuntimeError("checkpoint actor model/optimizer digest is inconsistent")
    opd_teacher_entries = [
        entry
        for path, entry in expected.items()
        if path.startswith("actor/opd_teacher/")
    ]
    teacher_digest = manifest.get("opd_teacher_tree_sha256")
    if opd_teacher_entries:
        if teacher_digest != _inventory_digest(opd_teacher_entries):
            raise RuntimeError("checkpoint OPD teacher digest is inconsistent")
    elif teacher_digest is not None:
        raise RuntimeError("checkpoint has an OPD teacher digest without teacher state")
    return manifest


def _find_latest_committed_checkpoint(
    checkpoint_root: str,
    *,
    expected_provenance: Optional[Mapping[str, object]] = None,
    return_manifest: bool = False,
):
    """Resolve the newest atomically committed checkpoint, including stale trackers."""

    checkpoint_root = os.path.abspath(checkpoint_root)
    if not os.path.isdir(checkpoint_root):
        return None
    committed_steps = []
    for name in os.listdir(checkpoint_root):
        match = _COMMITTED_CHECKPOINT_RE.fullmatch(name)
        if match is not None:
            committed_steps.append(int(match.group(1)))

    tracker_path = os.path.join(checkpoint_root, _CHECKPOINT_TRACKER)
    tracker_step = None
    if os.path.exists(tracker_path):
        if not os.path.isfile(tracker_path) or os.path.islink(tracker_path):
            raise RuntimeError(f"invalid checkpoint tracker: {tracker_path}")
        with open(tracker_path, encoding="utf-8") as handle:
            tracker_value = handle.read().strip()
        if not tracker_value.isdigit():
            raise RuntimeError(f"invalid checkpoint tracker contents: {tracker_value!r}")
        tracker_step = int(tracker_value)

    candidate_steps = committed_steps + ([] if tracker_step is None else [tracker_step])
    if not candidate_steps:
        return None
    latest_step = max(candidate_steps)
    checkpoint_dir = os.path.join(checkpoint_root, f"global_step_{latest_step}")
    manifest = _verify_checkpoint(
        checkpoint_dir, expected_provenance=expected_provenance
    )
    if manifest["global_step"] != latest_step:
        raise RuntimeError("latest committed checkpoint has inconsistent step metadata")
    if return_manifest:
        return checkpoint_dir, manifest
    return checkpoint_dir


def _verified_actor_state_digest(checkpoint_dir: str) -> str:
    """Return the authenticated actor model/optimizer digest for smoke parity."""

    manifest = _verify_checkpoint(checkpoint_dir)
    value = manifest.get("actor_model_optimizer_tree_sha256")
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RuntimeError("checkpoint has no valid actor state digest")
    return value


def _verified_rollout_trajectory_digest(checkpoint_dir: str) -> str:
    """Return the authenticated completed-trajectory digest for resume tests."""

    manifest = _verify_checkpoint(checkpoint_dir)
    value = manifest.get("rollout_trajectory_sha256")
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RuntimeError("checkpoint has no valid rollout trajectory digest")
    return value


def _read_step_tracker(path: str) -> Optional[int]:
    if not os.path.exists(path):
        return None
    if not os.path.isfile(path) or os.path.islink(path):
        raise RuntimeError(f"invalid checkpoint tracker: {path}")
    with open(path, encoding="utf-8") as handle:
        value = handle.read().strip()
    if not value.isdigit():
        raise RuntimeError(f"invalid checkpoint tracker contents: {value!r}")
    return int(value)


def _maybe_update_best_checkpoint(
    checkpoint_root: str,
    *,
    global_step: int,
    metric_name: Optional[str],
    metric_value: Optional[float],
    mode: str,
    verified_candidate_manifest: Optional[dict[str, object]] = None,
) -> bool:
    """Point BEST at a committed checkpoint when its validation metric wins."""

    if metric_name is None or metric_value is None:
        return False
    metric_value = float(metric_value)
    if not np.isfinite(metric_value):
        raise RuntimeError(f"best-checkpoint metric is non-finite: {metric_name}={metric_value}")
    if mode not in {"max", "min"}:
        raise ValueError("checkpoint_best_mode must be 'max' or 'min'")
    candidate_path = os.path.join(checkpoint_root, f"global_step_{global_step}")
    candidate_manifest = verified_candidate_manifest
    if candidate_manifest is None:
        candidate_manifest = _verify_checkpoint(candidate_path)
    elif not os.path.isdir(candidate_path):
        raise RuntimeError(f"BEST candidate was not published: {candidate_path}")
    if candidate_manifest.get("selection_metric_name") != metric_name:
        raise RuntimeError("candidate checkpoint does not contain the configured selection metric")
    if candidate_manifest.get("selection_metric_value") != metric_value:
        raise RuntimeError("candidate checkpoint selection metric disagrees with the caller")

    tracker_path = os.path.join(checkpoint_root, _BEST_CHECKPOINT_TRACKER)
    previous_step = _read_step_tracker(tracker_path)
    previous_value = None
    if previous_step is not None:
        previous_path = os.path.join(checkpoint_root, f"global_step_{previous_step}")
        previous_manifest_path = os.path.join(previous_path, _CHECKPOINT_MANIFEST)
        if not os.path.isfile(previous_manifest_path):
            raise RuntimeError(f"BEST points to an uncommitted checkpoint: {previous_path}")
        with open(previous_manifest_path, encoding="utf-8") as handle:
            previous_manifest = json.load(handle)
        if previous_manifest.get("selection_metric_name") != metric_name:
            raise RuntimeError("BEST checkpoint uses a different selection metric")
        previous_value = previous_manifest.get("selection_metric_value")
        if not isinstance(previous_value, (float, int)) or not np.isfinite(previous_value):
            raise RuntimeError("BEST checkpoint contains an invalid selection metric")

    is_better = previous_value is None
    if previous_value is not None:
        is_better = metric_value > previous_value if mode == "max" else metric_value < previous_value
    if is_better:
        _atomic_write_text(tracker_path, f"{global_step}\n")
    return is_better


def _prune_committed_checkpoints(checkpoint_root: str, keep_latest: int) -> list[str]:
    """Remove only older committed checkpoints, preserving latest and BEST."""

    if keep_latest < 1:
        raise ValueError("keep_latest must be at least one")
    candidates: list[tuple[int, str]] = []
    for name in os.listdir(checkpoint_root):
        match = _COMMITTED_CHECKPOINT_RE.fullmatch(name)
        path = os.path.join(checkpoint_root, name)
        if match is not None and os.path.isdir(path) and not os.path.islink(path):
            if os.path.isfile(os.path.join(path, _CHECKPOINT_MANIFEST)):
                candidates.append((int(match.group(1)), path))
    candidates.sort()
    protected_steps = {step for step, _ in candidates[-keep_latest:]}
    best_step = _read_step_tracker(os.path.join(checkpoint_root, _BEST_CHECKPOINT_TRACKER))
    if best_step is not None:
        best_path = os.path.join(checkpoint_root, f"global_step_{best_step}")
        if not os.path.isfile(os.path.join(best_path, _CHECKPOINT_MANIFEST)):
            raise RuntimeError(f"BEST points to an uncommitted checkpoint: {best_path}")
        protected_steps.add(best_step)
    removed: list[str] = []
    for step, path in candidates:
        if step in protected_steps:
            continue
        shutil.rmtree(path)
        removed.append(path)
    if removed:
        _fsync_directory(checkpoint_root)
    return removed


def _requeue_requested(signal_file: Optional[str]) -> bool:
    if signal_file is None or str(signal_file).strip() == "":
        return False
    signal_path = os.path.abspath(os.path.expanduser(str(signal_file)))
    if os.path.exists(signal_path) and not os.path.isfile(signal_path):
        raise RuntimeError(f"requeue signal path is not a regular file: {signal_path}")
    return os.path.isfile(signal_path)


def _consume_requeue_request(signal_file: str) -> None:
    signal_path = os.path.abspath(os.path.expanduser(str(signal_file)))
    try:
        os.unlink(signal_path)
    except FileNotFoundError:
        return
    _fsync_directory(os.path.dirname(signal_path))


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.OPO:
        advantages, returns = core_algos.compute_opo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.opd_config = OPDConfig.from_mapping(
            OmegaConf.to_container(config.algorithm.opd, resolve=True)
        )
        self.rollout_integrity_config = RolloutIntegrityConfig.from_mapping(
            OmegaConf.to_container(config.trainer.rollout_integrity, resolve=True)
        )
        close_tag_ids = tokenizer.encode("</think>", add_special_tokens=False)
        self.close_tag_token_id = close_tag_ids[0] if len(close_tag_ids) == 1 else None
        if self.rollout_integrity_config.enabled and self.close_tag_token_id is None:
            raise RuntimeError(
                "rollout integrity requires </think> to be one atomic token; "
                f"got {close_tag_ids}"
            )
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        resolved_config = OmegaConf.to_container(self.config, resolve=True)
        if not isinstance(resolved_config, Mapping):
            raise RuntimeError("resolved Hydra trainer config must be a mapping")
        self.checkpoint_provenance = build_checkpoint_provenance(resolved_config)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            assert n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0, f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            megatron_dp = n_gpus // (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size)
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size ({minimal_bsz})"

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        if int(config.trainer.get("checkpoint_keep_latest", 2)) < 1:
            raise ValueError("trainer.checkpoint_keep_latest must be at least one")
        if str(config.trainer.get("checkpoint_best_mode", "max")) not in {"max", "min"}:
            raise ValueError("trainer.checkpoint_best_mode must be 'max' or 'min'")
        if float(config.trainer.get("checkpoint_max_seconds", 720)) <= 0:
            raise ValueError("trainer.checkpoint_max_seconds must be positive")
        invocation_limit = config.trainer.get("max_rollout_iterations_per_invocation", None)
        if invocation_limit is not None and int(invocation_limit) < 1:
            raise ValueError("trainer.max_rollout_iterations_per_invocation must be positive or null")

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        # In this trainer one dataloader batch is one rollout iteration.  Keep
        # the old name for upstream compatibility, but make the study meaning
        # explicit for schedules and W&B.
        self.total_rollout_iterations = total_training_steps
        mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        if self.config.data.train_batch_size % mini_batch_size != 0:
            raise ValueError("train_batch_size must be divisible by ppo_mini_batch_size")
        self.optimizer_steps_per_rollout = (
            self.config.data.train_batch_size
            // mini_batch_size
            * self.config.actor_rollout_ref.actor.ppo_epochs
        )
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        rollout_validation_sums: dict[str, float] = defaultdict(float)
        rollout_validation_examples = 0
        validation_response_tokens = 0

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # repeat test batch
            validation_sample_count = int(self.config.actor_rollout_ref.rollout.val_kwargs.n)
            validation_example_count = len(test_batch.batch)
            test_batch = test_batch.repeat(repeat_times=validation_sample_count, interleave=True)
            test_batch.non_tensor_batch["rollout_sample_index"] = np.tile(
                np.arange(validation_sample_count, dtype=np.int64),
                validation_example_count,
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "index", "rollout_sample_index"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "rollout_seed": int(self.config.data.seed),
                "rollout_iteration": max(int(self.global_steps) - 1, 0),
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            if self.close_tag_token_id is not None and {
                "rollout_topk_ids",
                "rollout_topk_gumbels",
                "gumbel_temperature",
            }.issubset(test_output_gen_batch.batch.keys()):
                val_responses = test_output_gen_batch.batch["responses"]
                val_response_mask = test_output_gen_batch.batch["attention_mask"][:, -val_responses.shape[-1]:]
                val_topk_ids = test_output_gen_batch.batch["rollout_topk_ids"][:, -val_responses.shape[-1]:]
                val_topk_gumbels = test_output_gen_batch.batch["rollout_topk_gumbels"][:, -val_responses.shape[-1]:]
                val_diag = compute_rollout_diagnostics(
                    responses=val_responses,
                    response_mask=val_response_mask,
                    rollout_topk_ids=val_topk_ids,
                    rollout_topk_gumbels=val_topk_gumbels,
                    gumbel_temperature=float(test_output_gen_batch.batch["gumbel_temperature"][0].item()),
                    close_tag_token_id=self.close_tag_token_id,
                    decode=lambda ids: self.tokenizer.decode(ids, skip_special_tokens=False),
                )
                val_examples = val_responses.shape[0]
                rollout_validation_examples += val_examples
                validation_response_tokens += int(val_response_mask.sum().item())
                for name in (
                    "latent/length_mean",
                    "latent/cap_rate",
                    "latent/soft_to_hard_rate",
                ):
                    rollout_validation_sums[name] += float(val_diag.metrics[name]) * val_examples

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        metric_dict.update(validation_metric_aliases(metric_dict))
        metric_dict["trainer/rollout_iteration"] = max(self.global_steps - 1, 0)
        if rollout_validation_examples:
            metric_dict.update(
                {
                    "val/response_length_mean": validation_response_tokens / rollout_validation_examples,
                    "val/latent_length_mean": rollout_validation_sums["latent/length_mean"] / rollout_validation_examples,
                    "val/cap_rate": rollout_validation_sums["latent/cap_rate"] / rollout_validation_examples,
                    "val/soft_to_hard_rate": rollout_validation_sums["latent/soft_to_hard_rate"] / rollout_validation_examples,
                }
            )

        if self.rollout_integrity_config.enabled:
            validate_validation_metric_contract(metric_dict)

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(
        self,
        *,
        rollout_batch: DataProto,
        reason: str = "scheduled",
        selection_metric_name: Optional[str] = None,
        selection_metric_value: Optional[float] = None,
    ):
        """Save and publish one complete-rollout checkpoint atomically.

        Ray workers first write into an unpublished sibling directory.  Only
        after every worker RPC, driver state, and dataloader state completes do
        we fsync and hash the entire tree.  The final directory rename is the
        commit point; the latest-step tracker is replaced atomically afterward.
        """

        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("atomic OPD checkpoints currently require shared local storage")
        if self.config.trainer.del_local_ckpt_after_load:
            raise RuntimeError("atomic checkpoints cannot be deleted piecemeal during load")
        if (selection_metric_name is None) != (selection_metric_value is None):
            raise ValueError("checkpoint selection metric name and value must be provided together")
        if selection_metric_value is not None and not np.isfinite(float(selection_metric_value)):
            raise RuntimeError("checkpoint selection metric must be finite")
        best_mode = str(self.config.trainer.get("checkpoint_best_mode", "max"))
        if best_mode not in {"max", "min"}:
            raise ValueError("trainer.checkpoint_best_mode must be 'max' or 'min'")
        keep_latest = int(self.config.trainer.get("checkpoint_keep_latest", 2))
        if keep_latest < 1:
            raise ValueError("trainer.checkpoint_keep_latest must be at least one")

        for identity_key in ("uid", "index"):
            if identity_key not in rollout_batch.non_tensor_batch:
                raise RuntimeError(
                    f"cannot checkpoint without deterministic rollout identity {identity_key!r}"
                )
        completed_rollout_iteration = self.global_steps - 1
        rollout_record = _build_rollout_integrity_record(
            rollout_batch.batch,
            group_ids=rollout_batch.non_tensor_batch["uid"],
            example_identities=rollout_batch.non_tensor_batch["index"],
            rollout_iteration=completed_rollout_iteration,
        )

        checkpoint_root = os.path.abspath(self.config.trainer.default_local_dir)
        BaseCheckpointManager.local_mkdir(checkpoint_root)
        checkpoint_name = f"global_step_{self.global_steps}"
        final_checkpoint_dir = os.path.join(checkpoint_root, checkpoint_name)
        if os.path.lexists(final_checkpoint_dir):
            raise RuntimeError(f"refusing to overwrite existing checkpoint: {final_checkpoint_dir}")
        temporary_checkpoint_dir = os.path.join(
            checkpoint_root,
            f".{checkpoint_name}.incomplete.{uuid.uuid4().hex}",
        )
        os.mkdir(temporary_checkpoint_dir)
        print(f"temporary checkpoint folder: {temporary_checkpoint_dir}")

        next_rollout_iteration = self.global_steps
        optimizer_step = self.global_steps * self.optimizer_steps_per_rollout
        driver_state = {
            "global_step": self.global_steps,
            "completed_rollout_iteration": completed_rollout_iteration,
            "next_rollout_iteration": next_rollout_iteration,
            "optimizer_step": optimizer_step,
            "total_rollout_iterations": self.total_rollout_iterations,
            "world_size": self.actor_rollout_wg.world_size,
            "completed_rollout_trajectory_sha256": rollout_record["trajectory_sha256"],
            "checkpoint_resume_provenance_sha256": self.checkpoint_provenance[
                "resume_identity_sha256"
            ],
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
            },
        }

        try:
            _write_rollout_integrity_record(temporary_checkpoint_dir, rollout_record)
            actor_local_path = os.path.join(temporary_checkpoint_dir, "actor")
            # Rotation belongs to the driver because worker checkpoint managers
            # only know the temporary name that is renamed at commit time.
            self.actor_rollout_wg.save_checkpoint(
                actor_local_path,
                None,
                self.global_steps,
                max_ckpt_to_keep=None,
            )

            if self.use_critic:
                critic_local_path = os.path.join(temporary_checkpoint_dir, "critic")
                self.critic_wg.save_checkpoint(
                    critic_local_path,
                    None,
                    self.global_steps,
                    max_ckpt_to_keep=None,
                )

            torch.save(self.train_dataloader.state_dict(), os.path.join(temporary_checkpoint_dir, "data.pt"))
            torch.save(driver_state, os.path.join(temporary_checkpoint_dir, "driver_state.pt"))
            manifest = _write_checkpoint_manifest(
                temporary_checkpoint_dir,
                checkpoint_name=checkpoint_name,
                global_step=self.global_steps,
                completed_rollout_iteration=completed_rollout_iteration,
                next_rollout_iteration=next_rollout_iteration,
                optimizer_step=optimizer_step,
                total_rollout_iterations=self.total_rollout_iterations,
                world_size=self.actor_rollout_wg.world_size,
                reason=reason,
                provenance=self.checkpoint_provenance,
                selection_metric_name=selection_metric_name,
                selection_metric_value=selection_metric_value,
            )
            _verify_checkpoint(
                temporary_checkpoint_dir,
                require_committed_name=False,
                expected_provenance=self.checkpoint_provenance,
            )
            os.replace(temporary_checkpoint_dir, final_checkpoint_dir)
            _fsync_directory(checkpoint_root)
            _atomic_write_text(
                os.path.join(checkpoint_root, _CHECKPOINT_TRACKER),
                f"{self.global_steps}\n",
            )
            _maybe_update_best_checkpoint(
                checkpoint_root,
                global_step=self.global_steps,
                metric_name=selection_metric_name,
                metric_value=selection_metric_value,
                mode=best_mode,
                verified_candidate_manifest=manifest,
            )
            _prune_committed_checkpoints(checkpoint_root, keep_latest=keep_latest)
            print(f"Committed checkpoint: {final_checkpoint_dir}")
            return manifest
        except BaseException:
            if os.path.isdir(temporary_checkpoint_dir):
                shutil.rmtree(temporary_checkpoint_dir)
                _fsync_directory(checkpoint_root)
            raise

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        manifest = None
        global_step_folder = None
        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        elif self.config.trainer.resume_mode == "auto":
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            resolved_checkpoint = _find_latest_committed_checkpoint(
                checkpoint_folder,
                expected_provenance=self.checkpoint_provenance,
                return_manifest=True,
            )
            if resolved_checkpoint is None:
                print("Training from scratch")
                return 0
            global_step_folder, manifest = resolved_checkpoint
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
            assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)
            manifest = _verify_checkpoint(
                global_step_folder,
                expected_provenance=self.checkpoint_provenance,
            )
        else:
            raise ValueError(f"unsupported trainer.resume_mode: {self.config.trainer.resume_mode}")
        if manifest is None or global_step_folder is None:
            raise RuntimeError("checkpoint resolution returned no verified checkpoint")
        self.global_steps = int(manifest["global_step"])
        if int(manifest["world_size"]) != int(self.actor_rollout_wg.world_size):
            raise RuntimeError(
                "checkpoint world size does not match this run: "
                f"{manifest['world_size']} != {self.actor_rollout_wg.world_size}"
            )
        if int(manifest["total_rollout_iterations"]) != int(self.total_rollout_iterations):
            raise RuntimeError(
                "checkpoint rollout horizon does not match this run: "
                f"{manifest['total_rollout_iterations']} != {self.total_rollout_iterations}"
            )
        expected_optimizer_step = self.global_steps * self.optimizer_steps_per_rollout
        if int(manifest["optimizer_step"]) != expected_optimizer_step:
            raise RuntimeError(
                "checkpoint optimizer-step metadata does not match this run: "
                f"{manifest['optimizer_step']} != {expected_optimizer_step}"
            )
        if int(manifest["next_rollout_iteration"]) != self.global_steps:
            raise RuntimeError("checkpoint next-rollout metadata is inconsistent")

        print(f"Load from checkpoint folder: {global_step_folder}")
        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        driver_state_path = os.path.join(global_step_folder, "driver_state.pt")
        dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
        self.train_dataloader.load_state_dict(dataloader_state_dict)
        driver_state = torch.load(driver_state_path, weights_only=False)
        required_driver_state = {
            "global_step": self.global_steps,
            "completed_rollout_iteration": self.global_steps - 1,
            "next_rollout_iteration": self.global_steps,
            "optimizer_step": expected_optimizer_step,
            "total_rollout_iterations": self.total_rollout_iterations,
            "world_size": self.actor_rollout_wg.world_size,
            "completed_rollout_trajectory_sha256": manifest["rollout_trajectory_sha256"],
            "checkpoint_resume_provenance_sha256": self.checkpoint_provenance[
                "resume_identity_sha256"
            ],
        }
        for key, expected_value in required_driver_state.items():
            if driver_state.get(key) != expected_value:
                raise RuntimeError(
                    f"driver checkpoint field {key!r} disagrees: "
                    f"{driver_state.get(key)!r} != {expected_value!r}"
                )
        rng_state = driver_state["rng"]
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch_cpu"])
        _atomic_write_text(
            os.path.join(os.path.dirname(global_step_folder), _CHECKPOINT_TRACKER),
            f"{self.global_steps}\n",
        )

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        tracking_config = OmegaConf.to_container(self.config, resolve=True)
        if not isinstance(tracking_config, dict):
            raise RuntimeError("resolved tracking config must be a mapping")
        tracking_config["checkpoint_provenance"] = self.checkpoint_provenance
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=tracking_config,
        )
        if "wandb" in logger.logger:
            # W&B still receives its monotonically increasing internal step,
            # but all charts use rollout iteration as the explicit study axis.
            wandb = logger.logger["wandb"]
            wandb.define_metric("trainer/rollout_iteration")
            wandb.define_metric("*", step_metric="trainer/rollout_iteration", step_sync=True)

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        self._resumed = self.global_steps > 0

        # perform validation before training
        # currently, we only support validation using the reward_function.
        should_run_initial_validation = self.config.trainer.get("val_before_train", True) and (
            not self._resumed or self.config.trainer.get("val_only", False)
        )
        if self.val_reward_fn is not None and should_run_initial_validation:
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        iterations_this_invocation = 0
        invocation_limit = self.config.trainer.get("max_rollout_iterations_per_invocation", None)
        if invocation_limit is not None:
            invocation_limit = int(invocation_limit)
            if invocation_limit < 1:
                raise ValueError("trainer.max_rollout_iterations_per_invocation must be positive or null")

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                rollout_diagnostics = None
                replay_error = float("nan")
                checkpoint_committed = False
                val_metrics_this_iteration = None
                requeue_requested = False
                invocation_limit_reached = False
                rollout_iteration = self.global_steps - 1
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "index"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                gen_batch.meta_info.update(
                    {
                        "rollout_seed": int(self.config.data.seed),
                        "rollout_iteration": int(rollout_iteration),
                    }
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # Stable group IDs make a resumed iteration semantically
                    # comparable without relying on OS-entropy UUIDs.
                    batch.non_tensor_batch["uid"] = np.array(
                        [
                            f"rollout-{rollout_iteration:08d}-prompt-{prompt_index:06d}"
                            for prompt_index in range(len(batch.batch))
                        ],
                        dtype=object,
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            # print(f"rollout_probs {rollout_probs}, {rollout_probs.grad_fn}")
                            # print(f"actor_probs {actor_probs, {actor_probs.grad_fn}")
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                            replay_error = replay_ratio_abs_error_max(
                                rollout_log_probs=rollout_old_log_probs,
                                actor_log_probs=actor_old_log_probs,
                                response_mask=response_mask,
                            )
                            metrics["replay/ratio_abs_error_max"] = replay_error

                    replay_keys = {
                        "rollout_topk_ids",
                        "rollout_topk_gumbels",
                        "gumbel_temperature",
                    }
                    if replay_keys.issubset(batch.batch.keys()) and self.close_tag_token_id is not None:
                        responses = batch.batch["responses"]
                        response_length = responses.shape[-1]
                        rollout_diagnostics = compute_rollout_diagnostics(
                            responses=responses,
                            response_mask=batch.batch["response_mask"],
                            rollout_topk_ids=batch.batch["rollout_topk_ids"][:, -response_length:],
                            rollout_topk_gumbels=batch.batch["rollout_topk_gumbels"][:, -response_length:],
                            gumbel_temperature=float(batch.batch["gumbel_temperature"][0].item()),
                            close_tag_token_id=self.close_tag_token_id,
                            decode=lambda ids: self.tokenizer.decode(ids, skip_special_tokens=False),
                        )
                        metrics.update(rollout_diagnostics.metrics)
                    elif self.rollout_integrity_config.enabled:
                        missing = sorted(replay_keys - set(batch.batch.keys()))
                        raise RuntimeError(
                            "continuous rollout metadata is unavailable before actor update; "
                            f"missing={missing}, atomic_close_tag={self.close_tag_token_id is not None}"
                        )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                        )

                        response_mask = batch.batch["response_mask"].bool()
                        sequence_scores = batch.batch["token_level_scores"].sum(dim=-1)
                        trajectory_advantages = (
                            batch.batch["advantages"].masked_fill(~response_mask, 0.0).sum(dim=-1)
                            / response_mask.sum(dim=-1).clamp_min(1)
                        )
                        metrics.update(
                            reward_and_group_metrics(
                                sequence_scores=sequence_scores,
                                trajectory_advantages=trajectory_advantages,
                                group_ids=batch.non_tensor_batch["uid"],
                            )
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        worker_schedule = schedule_meta_info(
                            self.opd_config,
                            rollout_iteration=rollout_iteration,
                            total_iterations=self.total_rollout_iterations,
                        )
                        batch.meta_info.update(worker_schedule)
                        if self.rollout_integrity_config.enabled:
                            if rollout_diagnostics is None:
                                raise RuntimeError("rollout diagnostics were not constructed")
                            validate_rollout_integrity(
                                diagnostics=rollout_diagnostics,
                                replay_error=replay_error,
                                config=self.rollout_integrity_config,
                                rollout_iteration=rollout_iteration,
                            )
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            batch.meta_info['add_noise_dirichlet'] = self.config.actor_rollout_ref.rollout.add_noise_dirichlet
                            batch.meta_info['add_noise_gumbel_softmax'] = self.config.actor_rollout_ref.rollout.add_noise_gumbel_softmax
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            val_metrics_this_iteration = val_metrics
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    requeue_signal_file = self.config.trainer.get("requeue_signal_file", None)
                    requeue_requested = _requeue_requested(requeue_signal_file)
                    invocation_limit_reached = (
                        invocation_limit is not None
                        and iterations_this_invocation + 1 >= invocation_limit
                        and not is_last_step
                    )
                    save_freq = int(self.config.trainer.save_freq)
                    scheduled_checkpoint = save_freq > 0 and (
                        is_last_step or self.global_steps % save_freq == 0
                    )
                    must_stop = requeue_requested or invocation_limit_reached
                    if scheduled_checkpoint or must_stop:
                        if requeue_requested:
                            checkpoint_reason = "requeue_signal"
                        elif invocation_limit_reached:
                            checkpoint_reason = "invocation_limit"
                        elif is_last_step:
                            checkpoint_reason = "final"
                        else:
                            checkpoint_reason = "scheduled"
                        selection_metric_name = str(
                            self.config.trainer.get("checkpoint_best_metric", "val/math_verify/mean_at_1")
                        )
                        selection_metric_value = None
                        if val_metrics_this_iteration is not None:
                            candidate_metric = val_metrics_this_iteration.get(selection_metric_name)
                            if candidate_metric is not None:
                                selection_metric_value = float(candidate_metric)
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint(
                                rollout_batch=batch,
                                reason=checkpoint_reason,
                                selection_metric_name=(
                                    selection_metric_name if selection_metric_value is not None else None
                                ),
                                selection_metric_value=selection_metric_value,
                            )
                        if timing_raw["save_checkpoint"] > float(
                            self.config.trainer.get("checkpoint_max_seconds", 720)
                        ):
                            raise RuntimeError(
                                "checkpoint save plus authentication exceeded the 12-minute limit: "
                                f"{timing_raw['save_checkpoint']:.3f}s"
                            )
                        checkpoint_committed = True
                        # Hashing multi-gigabyte shards can overlap a late
                        # Slurm warning.  The just-committed checkpoint is
                        # already sufficient; consume the request and stop
                        # without redundantly writing the same step.
                        requeue_requested = requeue_requested or _requeue_requested(
                            requeue_signal_file
                        )
                        if requeue_requested:
                            _consume_requeue_request(requeue_signal_file)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                        "integrity/requeue_requested": float(requeue_requested),
                        "integrity/invocation_limit_reached": float(invocation_limit_reached),
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                response_tokens = int(batch.batch["response_mask"].sum().item())
                overall_tokens = int(sum(batch.meta_info["global_token_num"]))
                metrics.update(
                    {
                        "perf/rollout_tokens_per_second": response_tokens / max(timing_raw.get("gen", 0.0), 1e-12),
                        "perf/train_tokens_per_second": overall_tokens / max(timing_raw.get("update_actor", 0.0), 1e-12),
                        "perf/iteration_seconds": timing_raw["step"],
                        "system/cpu_utilization": psutil.cpu_percent(interval=None),
                    }
                )
                if "save_checkpoint" in timing_raw:
                    metrics["perf/checkpoint_seconds"] = timing_raw["save_checkpoint"]
                optimizer_step = (rollout_iteration + 1) * self.optimizer_steps_per_rollout
                metrics = add_canonical_metric_aliases(
                    metrics,
                    opd_config=self.opd_config,
                    rollout_iteration=rollout_iteration,
                    total_iterations=self.total_rollout_iterations,
                    optimizer_step=optimizer_step,
                    grad_clip=self.config.actor_rollout_ref.actor.grad_clip,
                    checkpoint_committed=checkpoint_committed,
                    resumed=self._resumed,
                )
                if self.rollout_integrity_config.enabled:
                    validate_iteration_metric_contract(metrics)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                iterations_this_invocation += 1
                if requeue_requested or invocation_limit_reached:
                    pprint(
                        "Stopping cleanly after committed rollout iteration "
                        f"{rollout_iteration} ({'requeue signal' if requeue_requested else 'invocation limit'})"
                    )
                    progress_bar.close()
                    return
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
