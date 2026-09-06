"""Versioned, opt-in arithmetic contract shared by Qwen replay workers."""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path

QWEN_REPLAY_BACKENDS = ("disabled", "native_fa3_v1")
QWEN_REPLAY_RUNTIME = {"torch": "2.6.0", "transformers": "4.51.1", "sgl-kernel": "0.1.1", "flash-attn": "2.7.3", "triton": "3.2.0"}


def validate_qwen_replay_runtime(version=importlib.metadata.version):
    observed = {name: version(name) for name in QWEN_REPLAY_RUNTIME}
    if any(observed[name].split("+")[0] != expected for name, expected in QWEN_REPLAY_RUNTIME.items()):
        raise RuntimeError(f"native_fa3_v1 requires the calibrated kernel runtime: {observed}")
    return observed


def validate_qwen_replay_backend(value):
    if value not in QWEN_REPLAY_BACKENDS:
        raise ValueError(f"qwen_replay_backend must be one of {QWEN_REPLAY_BACKENDS}")
    return value


def validate_qwen_replay_worker(config, *, use_remove_padding, use_fused_kernels,
                                enable_gradient_checkpointing, enable_activation_offload,
                                use_liger, sequence_parallel_size):
    mode = validate_qwen_replay_backend(config.model.get("qwen_replay_backend", "disabled"))
    rollout_mode = validate_qwen_replay_backend(config.rollout.get("qwen_replay_backend", "disabled"))
    if rollout_mode != mode:
        raise ValueError("actor and rollout qwen_replay_backend must match")
    if mode == "disabled":
        return mode
    if config.actor.strategy != "fsdp" or sequence_parallel_size != 1:
        raise ValueError("native_fa3_v1 requires FSDP1 with sequence parallel size one")
    if not use_remove_padding or use_fused_kernels or enable_gradient_checkpointing or enable_activation_offload or use_liger or config.model.get("lora_rank", 0):
        raise ValueError("native_fa3_v1 requires packed replay without fused kernels, checkpointing, activation offload, Liger, or LoRA")
    if config.rollout.name != "sglang" or config.rollout.tensor_model_parallel_size != 1:
        raise ValueError("native_fa3_v1 requires TP1 SGLang rollouts")
    return mode


def qwen_replay_arithmetic_identity():
    """Hash only implementation source, without exporting parameters or inputs."""
    directory = Path(__file__).resolve().parent
    names = ("qwen_replay_backend.py", "qwen_native_arithmetic.py", "batch_invariant_linear.py", "native_fa3_attention.py")
    return {"recipe": "native_fa3_v1", "attention_backend": "fa3", "attention_num_splits": 1,
            "projection_tile": [32, 64, 32], "forward_dtype": "bfloat16", "rope_cache_dtype": "float32",
            "required_runtime_versions": dict(QWEN_REPLAY_RUNTIME),
            "implementation_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}}
