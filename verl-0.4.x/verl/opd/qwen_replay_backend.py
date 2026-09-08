"""Versioned, opt-in arithmetic contract shared by Qwen replay workers."""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path

QWEN_REPLAY_BACKENDS = ("disabled", "native_fa3_v1", "native_fa3_v2")
QWEN_REPLAY_RUNTIME = {"torch": "2.6.0", "transformers": "4.51.1", "sgl-kernel": "0.1.1", "flash-attn": "2.7.3", "triton": "3.2.0"}
QWEN_REPLAY_RUNTIME_V2 = {**QWEN_REPLAY_RUNTIME, "opd-fa3": "0.1.0"}


def validate_qwen_replay_runtime(version=importlib.metadata.version, *, backend="native_fa3_v1"):
    if backend not in ("native_fa3_v1", "native_fa3_v2"):
        raise ValueError("runtime validation requires a native Qwen backend")
    required = QWEN_REPLAY_RUNTIME if backend == "native_fa3_v1" else QWEN_REPLAY_RUNTIME_V2
    observed = {name: version(name) for name in required}
    if any(observed[name].split("+")[0] != expected for name, expected in required.items()):
        raise RuntimeError(f"{backend} requires the calibrated kernel runtime: {observed}")
    if backend == "native_fa3_v2":
        from opd_fa3 import validate_build
        validate_build()
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
        raise ValueError(f"{mode} requires FSDP1 with sequence parallel size one")
    if not use_remove_padding or use_fused_kernels or enable_gradient_checkpointing or enable_activation_offload or use_liger or (mode == "native_fa3_v1" and config.model.get("lora_rank", 0)):
        raise ValueError(f"{mode} requires packed replay without unsupported fused kernels, checkpointing, activation offload, Liger, or legacy LoRA")
    if mode == "native_fa3_v1":
        if config.rollout.name != "sglang" or config.rollout.tensor_model_parallel_size != 1:
            raise ValueError("native_fa3_v1 requires TP1 SGLang rollouts")
    else:
        if config.rollout.name not in ("sglang", "vllm") or config.rollout.tensor_model_parallel_size != 1:
            raise ValueError("native_fa3_v2 requires TP1 SGLang or categorical vLLM rollouts")
        if config.rollout.name == "vllm" and config.rollout.get("enable_soft_thinking", True):
            raise ValueError("native_fa3_v2 vLLM rollouts must use categorical actions")
        rank = config.model.get("lora_rank", 0)
        if type(rank) is not int or not 0 <= rank <= 256:
            raise ValueError("native_fa3_v2 LoRA rank must be an integer in [0, 256]")
        if rank:
            from verl.opd.qwen_lora import QWEN_LORA_MERGE_RULE, validate_qwen_lora_config
            validate_qwen_lora_config({"rank": rank, "alpha": config.model.get("lora_alpha", 64),
                                      "target_modules": list(config.model.get("target_modules", [])),
                                      "dropout": config.model.get("lora_dropout", 0.0), "bias": "none",
                                      "seed": config.get("training_seed", 11), "merge_rule": QWEN_LORA_MERGE_RULE})
    return mode


def qwen_replay_arithmetic_identity(*, backend="native_fa3_v1"):
    """Hash only implementation source, without exporting parameters or inputs."""
    directory = Path(__file__).resolve().parent
    names = ("qwen_replay_backend.py", "qwen_native_arithmetic.py", "batch_invariant_linear.py", "native_fa3_attention.py")
    if backend not in ("native_fa3_v1", "native_fa3_v2"):
        raise ValueError("arithmetic identity requires a native Qwen backend")
    if backend == "native_fa3_v2":
        names += ("qwen_lora.py", "qwen_lora_ema.py", "qwen_vllm_arithmetic.py", "qwen_vllm_attention.py")
        return {"recipe": backend, "attention_backend": "opd_fa3", "attention_backward": "native_fa3",
                "attention_num_splits": 1, "projection_tile": [32, 64, 32],
                "forward_dtype": "bfloat16", "master_dtype": "float32", "rope_cache_dtype": "float32",
                "adapter_merge": "fp32_fixed_tile_ba_scale_add_then_bf16_v1", "teacher_ema_space": "dense_effective_fp32",
                "categorical_rollout_arithmetic": "qwen3_vllm_native_attention_v2",
                "required_runtime_versions": dict(QWEN_REPLAY_RUNTIME_V2),
                "implementation_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}}
    return {"recipe": "native_fa3_v1", "attention_backend": "fa3", "attention_num_splits": 1,
            "projection_tile": [32, 64, 32], "forward_dtype": "bfloat16", "rope_cache_dtype": "float32",
            "required_runtime_versions": dict(QWEN_REPLAY_RUNTIME),
            "implementation_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}}
