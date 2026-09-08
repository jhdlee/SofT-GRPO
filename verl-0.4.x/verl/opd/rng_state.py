"""Versioned training RNG ownership; rollout request seeds remain unchanged."""
from __future__ import annotations

import hashlib
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from verl.opd.checkpoint_semantics import collective_checkpoint_stage, read_record, semantic_sha256, write_record


def seed_training_rng(seed, rank=0, *, namespace="worker"):
    if type(seed) is not int or seed < 0 or type(rank) is not int or rank < 0:
        raise ValueError("training seed and rank must be nonnegative integers")
    value = int.from_bytes(hashlib.sha256(f"qwen_rng_v1:{namespace}:{seed}:{rank}".encode()).digest()[:8], "big")
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(value)
    return value


def _torch_state(tensor):
    return tensor.detach().cpu().numpy().tobytes().hex()


def capture_rng_state():
    numpy = np.random.get_state()
    python = random.getstate()
    return {"python": [python[0], list(python[1]), python[2]],
            "numpy": [numpy[0], numpy[1].tolist(), int(numpy[2]), int(numpy[3]), float(numpy[4])],
            "torch_cpu": _torch_state(torch.get_rng_state()),
            "torch_cuda": _torch_state(torch.cuda.get_rng_state()) if torch.cuda.is_available() else None}


def _decode_torch(value):
    return torch.from_numpy(np.frombuffer(bytes.fromhex(value), dtype=np.uint8).copy())


def restore_rng_state(state):
    python, numpy = state["python"], state["numpy"]
    random.setstate((python[0], tuple(python[1]), python[2]))
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), numpy[2], numpy[3], numpy[4]))
    torch.set_rng_state(_decode_torch(state["torch_cpu"]))
    if torch.cuda.is_available():
        if state["torch_cuda"] is None:
            raise RuntimeError("CUDA RNG missing from CUDA checkpoint")
        torch.cuda.set_rng_state(_decode_torch(state["torch_cuda"]))
    elif state["torch_cuda"] is not None:
        raise RuntimeError("cannot restore CUDA training RNG without CUDA")
    if semantic_sha256(capture_rng_state()) != semantic_sha256(state):
        raise RuntimeError("restored process RNG differs from checkpoint")


@contextmanager
def preserve_training_rng():
    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)


def resume_dataloader_iterator(dataloader):
    """Recreate a loaded iterator without consuming the restored driver RNG.

    Torch 2.6 draws a new int64 worker base seed in DataLoader iterator
    construction, even with zero workers. StatefulDataLoader then restores its
    saved iterator/worker state. An uninterrupted run does not recreate that
    iterator at the checkpoint boundary, so the extra initialization draw is
    outside the training stream. Do not protect ``next`` here: fetching the
    next batch may legitimately consume dataset/transform/sampler randomness.
    Call only for the first iterator after loading a checkpoint.
    """
    with preserve_training_rng():
        return iter(dataloader)


def save_worker_rng(local_actor_path, rank, world_size, rollout_manager=None):
    return collective_checkpoint_stage("worker RNG publication", lambda: _save_worker_rng(
        local_actor_path, rank, world_size, rollout_manager))


def _save_worker_rng(local_actor_path, rank, world_size, rollout_manager=None):
    generation = getattr(rollout_manager, "gen_random_states", None)
    state = {"process": capture_rng_state(),
             "rollout_cuda": _torch_state(generation) if generation is not None else None}
    return write_record(Path(local_actor_path) / f"worker_rng_world_size_{world_size}_rank_{rank}.json",
                        {"rank": rank, "world_size": world_size, "state": state})


def restore_worker_rng(local_actor_path, rank, world_size, rollout_manager=None):
    return collective_checkpoint_stage("worker RNG restore", lambda: _restore_worker_rng(
        local_actor_path, rank, world_size, rollout_manager))


def _restore_worker_rng(local_actor_path, rank, world_size, rollout_manager=None):
    record = read_record(Path(local_actor_path) / f"worker_rng_world_size_{world_size}_rank_{rank}.json")
    if record.get("rank") != rank or record.get("world_size") != world_size:
        raise RuntimeError("worker RNG rank/world size differs")
    state = record["state"]
    generation = state["rollout_cuda"]
    if generation is not None:
        if rollout_manager is None or not hasattr(rollout_manager, "gen_random_states"):
            raise RuntimeError("checkpoint rollout RNG has no owning sharding manager")
        rollout_manager.gen_random_states = _decode_torch(generation)
    elif getattr(rollout_manager, "gen_random_states", None) is not None:
        raise RuntimeError("checkpoint is missing rollout generation RNG")
    restore_rng_state(state["process"])
    return record["record_sha256"]
