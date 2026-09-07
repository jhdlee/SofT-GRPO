"""Exact logical checkpoint identity, separate from archive authentication.

No pickle bytes, cached DeviceMesh hashes, storage addresses, or physical CUDA
indices enter this identity. Tensor contents and distributed layouts do.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

SEMANTIC_SCHEMA = "qwen_semantic_v1"


def collective_checkpoint_stage(label, operation, *, distributed=None):
    """Propagate local I/O/validation errors before a following rank barrier."""
    distributed = torch.distributed if distributed is None else distributed
    result, failure = None, None
    try:
        result = operation()
    except BaseException as error:
        failure = error
    message = None if failure is None else f"{type(failure).__name__}: {failure}"[:2000]
    errors = [message]
    if distributed.is_initialized():
        errors = [None] * distributed.get_world_size()
        distributed.all_gather_object(errors, message)
    if any(error is not None for error in errors):
        raise RuntimeError(f"collective checkpoint {label} failed: {errors}") from failure
    return result


def semantic_sha256(value):
    from torch.distributed.tensor import DTensor
    from torch.distributed._shard.sharded_tensor import ShardedTensor

    digest = hashlib.sha256()

    def emit(tag, payload=b""):
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        digest.update(tag.encode("ascii") + b":" + str(len(payload)).encode("ascii") + b":" + payload)

    def tensor(t):
        emit("dtype", str(t.dtype))
        visit(tuple(t.shape))
        visit(tuple(t.stride()))
        if t.layout != torch.strided or t.is_quantized or t.device.type == "meta":
            raise ValueError("semantic checkpoint requires materialized strided tensors")
        raw = t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        view = memoryview(raw).cast("B")
        emit("nbytes", str(len(view)))
        for begin in range(0, len(view), 4 * 1024 * 1024):
            digest.update(view[begin:begin + 4 * 1024 * 1024])

    def visit(x):
        # DTensor is a Tensor subclass: handle the layout before plain tensors.
        if isinstance(x, DTensor):
            emit("dtensor")
            visit(tuple(x.shape))
            visit(tuple(x.stride()))
            mesh = x.device_mesh
            visit({"device_type": mesh.device_type, "mesh": mesh.mesh,
                   "dim_names": mesh.mesh_dim_names,
                   "placements": tuple(str(p) for p in x.placements)})
            tensor(x.to_local())
        elif isinstance(x, ShardedTensor):
            emit("sharded_tensor")
            metadata = x.metadata()

            def shard_layout(shard):
                placement = shard.placement
                return {"offsets": tuple(shard.shard_offsets), "sizes": tuple(shard.shard_sizes),
                        "rank": placement.rank(), "worker": placement.worker_name(),
                        "device_type": placement.device().type}

            visit(tuple(metadata.size))
            visit(str(metadata.tensor_properties.dtype))
            visit(str(metadata.tensor_properties.layout))
            visit(sorted((shard_layout(s) for s in metadata.shards_metadata),
                         key=lambda s: (s["offsets"], s["sizes"])))
            local = sorted(x.local_shards(), key=lambda s: tuple(s.metadata.shard_offsets))
            visit(len(local))
            for shard in local:
                visit(shard_layout(shard.metadata))
                tensor(shard.tensor)
        elif isinstance(x, torch.Tensor):
            emit("tensor")
            tensor(x)
        elif isinstance(x, np.ndarray):
            if x.dtype.hasobject:
                raise ValueError("object arrays are not semantic checkpoint state")
            emit("ndarray", x.dtype.str)
            visit(tuple(x.shape))
            emit("data", np.ascontiguousarray(x).tobytes())
        elif isinstance(x, Mapping):
            emit("mapping", str(len(x)))
            # Sorting by typed key digests handles integer optimizer keys without
            # conflating them with strings or depending on insertion order.
            for key in sorted(x, key=semantic_sha256):
                visit(key)
                visit(x[key])
        elif isinstance(x, (list, tuple)):
            emit("tuple" if isinstance(x, tuple) else "list", str(len(x)))
            for item in x:
                visit(item)
        elif x is None:
            emit("none")
        elif isinstance(x, (bool, np.bool_)):
            emit("bool", str(bool(x)))
        elif isinstance(x, (int, np.integer)):
            emit("int", str(int(x)))
        elif isinstance(x, (float, np.floating)):
            if not math.isfinite(float(x)):
                raise ValueError("nonfinite scalar in semantic checkpoint state")
            emit("float", float(x).hex())
        elif isinstance(x, str):
            emit("string", x)
        elif isinstance(x, bytes):
            emit("bytes", x)
        elif isinstance(x, (torch.dtype, torch.device)):
            emit("torch_type", str(x))
        else:
            raise TypeError(f"unsupported semantic checkpoint state: {type(x).__name__}")

    visit(value)
    return digest.hexdigest()


def write_record(path, payload):
    """Write inside the driver's unpublished checkpoint; never replace a file."""
    path = Path(path)
    if "schema" in payload or "record_sha256" in payload:
        raise ValueError("semantic record fields are reserved")
    record = {"schema": SEMANTIC_SCHEMA, **payload}
    record["record_sha256"] = semantic_sha256(record)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        import os
        os.fsync(handle.fileno())
    return record


def read_record(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing or unsafe semantic state: {path}")
    record = json.loads(path.read_text())
    if not isinstance(record, dict):
        raise RuntimeError(f"malformed semantic state: {path}")
    payload = {k: v for k, v in record.items() if k != "record_sha256"}
    if record.get("schema") != SEMANTIC_SCHEMA or record.get("record_sha256") != semantic_sha256(payload):
        raise RuntimeError(f"semantic state digest mismatch: {path}")
    return record


def model_state_record(model, optimizer, scheduler, *, rank, world_size):
    return {"rank": rank, "world_size": world_size,
            "model_sha256": semantic_sha256(model),
            "optimizer_sha256": semantic_sha256(optimizer),
            "scheduler_sha256": semantic_sha256(scheduler)}


def verify_model_state(path, model, optimizer, scheduler, *, rank, world_size):
    record = read_record(path)
    expected = model_state_record(model, optimizer, scheduler, rank=rank, world_size=world_size)
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(f"loaded checkpoint semantic {key} mismatch")
    return record


def checkpoint_semantic_identity(checkpoint, *, world_size, require=False):
    """Combine authenticated per-rank records; old archives need no new fields.

    Payload file rehashing remains the caller's responsibility. FSDP verifies
    model records against deserialized tensors before applying loaded state.
    """
    checkpoint = Path(checkpoint)
    driver_path = checkpoint / "driver_semantic.json"
    if not driver_path.exists():
        if require or list(checkpoint.glob("actor/semantic_state_*.json")):
            raise RuntimeError("checkpoint lacks required semantic identity")
        return None
    driver = read_record(driver_path)
    for field in ("rng_sha256", "dataloader_sha256"):
        value = driver.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise RuntimeError(f"semantic driver is missing {field}")
    actors, rngs, teachers = [], [], []
    teacher = (checkpoint / "actor/opd_teacher").is_dir()
    for rank in range(world_size):
        suffix = f"world_size_{world_size}_rank_{rank}.json"
        actor = read_record(checkpoint / "actor" / f"semantic_state_{suffix}")
        rng = read_record(checkpoint / "actor" / f"worker_rng_{suffix}")
        for item in (actor, rng):
            if item.get("rank") != rank or item.get("world_size") != world_size:
                raise RuntimeError("semantic checkpoint rank/world size mismatch")
        actors.append(actor)
        rngs.append(rng)
        if teacher:
            state = read_record(checkpoint / "actor/opd_teacher" / f"semantic_state_{suffix}")
            if state.get("rank") != rank or state.get("world_size") != world_size:
                raise RuntimeError("semantic teacher rank/world size mismatch")
            ema_path = checkpoint / "actor/opd_teacher" / f"ema_state_{suffix}"
            if ema_path.is_symlink():
                raise RuntimeError("unsafe EMA state")
            teachers.append({"state": state, "ema": json.loads(ema_path.read_text())})
    return {"schema": SEMANTIC_SCHEMA,
            "actor_model_optimizer_scheduler_sha256": semantic_sha256(actors),
            "worker_rng_sha256": semantic_sha256(rngs),
            "driver_rng_sha256": driver["rng_sha256"],
            "dataloader_sha256": driver["dataloader_sha256"],
            "teacher_model_ema_sha256": semantic_sha256(teachers) if teacher else None}
