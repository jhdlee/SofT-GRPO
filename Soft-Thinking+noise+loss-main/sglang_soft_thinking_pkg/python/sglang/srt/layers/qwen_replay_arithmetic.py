"""Explicit Qwen3 replay arithmetic; disabled native launches stay unchanged.

The versioned backend installs fixed-tile linear operations on one loaded model
and one-split FA3 calls on one attention backend. It never replaces parameters,
changes sampling/density rules, or patches a process-global class or function.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import inspect
import json
import logging
from pathlib import Path
import types

logger = logging.getLogger(__name__)
BACKENDS = ("disabled", "native_fa3_v1", "native_fa3_v2")


def validate_qwen_replay_server_args(args):
    mode = getattr(args, "opd_qwen_replay_backend", "disabled")
    if mode not in BACKENDS:
        raise ValueError(f"unsupported opd_qwen_replay_backend: {mode!r}")
    if mode == "disabled":
        return False
    required = {
        "tp_size": 1, "dp_size": 1, "ep_size": 1, "device": "cuda",
        "attention_backend": "fa3", "disable_cuda_graph": True,
        "disable_overlap_schedule": True, "disable_radix_cache": True,
    }
    for field, expected in required.items():
        observed = getattr(args, field, None)
        if observed != expected or (isinstance(expected, bool) and observed is not expected):
            raise ValueError(f"native_fa3_v1 requires {field}={expected!r}; got {observed!r}")
    if getattr(args, "dtype", None) not in ("bfloat16", "bf16"):
        raise ValueError("native_fa3_v1 requires explicit BF16 model dtype")
    if getattr(args, "kv_cache_dtype", "auto") not in ("auto", "bfloat16", "bf16"):
        raise ValueError("native_fa3_v1 requires BF16 KV cache")
    for field in ("quantization", "quantization_param_path", "torchao_config", "lora_paths",
                  "speculative_algorithm", "speculative_draft_model_path"):
        if getattr(args, field, None):
            raise ValueError(f"native_fa3_v1 does not support {field}")
    if getattr(args, "enable_dp_attention", False) or getattr(args, "enable_ep_moe", False):
        raise ValueError("native_fa3_v1 does not support DP attention or MoE")
    return True


def fa3_replay_callables(args, varlen, kvcache):
    """Keep original callable identity/defaults unless this backend is selected."""
    if not validate_qwen_replay_server_args(args):
        return varlen, kvcache
    if args.opd_qwen_replay_backend == "native_fa3_v2":
        from opd_fa3 import flash_attn_varlen_func, flash_attn_with_kvcache, validate_build

        validate_build()
        varlen, kvcache = flash_attn_varlen_func, flash_attn_with_kvcache
    return functools.partial(varlen, num_splits=1), functools.partial(kvcache, num_splits=1)


def _validate_loaded_model(model, config):
    import torch

    if (getattr(config, "model_type", None) != "qwen3"
            or type(model).__name__ != "Qwen3ForCausalLM"
            or getattr(config, "rope_scaling", None)):
        raise ValueError("native_fa3_v1 supports unscaled dense Qwen3ForCausalLM only")
    # Version 1 is calibrated on the sealed Qwen3-0.6B architecture only.
    for name, expected in {"hidden_size": 1024, "intermediate_size": 3072,
                           "num_hidden_layers": 28, "num_attention_heads": 16,
                           "num_key_value_heads": 8, "head_dim": 128,
                           "vocab_size": 151936}.items():
        if getattr(config, name, None) != expected:
            raise ValueError(f"native_fa3_v1 requires Qwen3-0.6B {name}={expected}")
    if getattr(config, "quantization_config", None):
        raise ValueError("native_fa3_v1 does not support quantized model configurations")
    parameters = list(model.parameters())
    if not parameters or any(p.device.type != "cuda" or p.dtype != torch.bfloat16 for p in parameters):
        raise ValueError("native_fa3_v1 requires loaded BF16 CUDA parameters")
    processor = model.logits_processor
    if (processor.do_tensor_parallel_all_gather or processor.do_tensor_parallel_all_gather_dp_attn
            or processor.final_logit_softcapping or processor.logit_scale is not None):
        raise ValueError("native_fa3_v1 requires an unscaled TP1 Qwen3 logits head")


def _install_linear(model, linear, unquantized_type):
    modules = [module for module in model.modules()
               if isinstance(getattr(module, "quant_method", None), unquantized_type)]
    expected = len(model.model.layers) * 4
    if len(modules) != expected:
        raise ValueError(f"native_fa3_v1 expected {expected} unquantized projections, got {len(modules)}")
    # Validate all model structure before the first mutation.
    for layer in model.model.layers:
        for module in (layer.self_attn.qkv_proj, layer.self_attn.o_proj,
                       layer.mlp.gate_up_proj, layer.mlp.down_proj):
            if module not in modules:
                raise ValueError("native_fa3_v1 found an unsupported projection")
    for module in modules:
        method = copy.copy(module.quant_method)
        def apply(method_self, layer, value, bias=None):
            return linear(value, layer.weight, bias)
        method.apply = types.MethodType(apply, method)
        module.quant_method = method
    def get_logits(processor, hidden_states, lm_head, logits_metadata, embedding_bias=None):
        if embedding_bias is not None:
            raise ValueError("native_fa3_v1 does not support an embedding bias")
        logits = linear(hidden_states.to(lm_head.weight.dtype), lm_head.weight)
        return logits[:, :processor.config.vocab_size].float()
    model.logits_processor._get_logits = types.MethodType(get_logits, model.logits_processor)
    return len(modules)


def install_qwen_replay_backend(model, config, args):
    if not validate_qwen_replay_server_args(args):
        return {"backend": "disabled"}
    if getattr(model, "_opd_qwen_replay_backend", None) is not None:
        raise ValueError("Qwen replay arithmetic is already installed on this model")
    _validate_loaded_model(model, config)
    # Optional VERL dependency is imported only after explicit opt-in and guards.
    from verl.opd.batch_invariant_linear import batch_invariant_linear
    from verl.opd.qwen_replay_backend import validate_qwen_replay_runtime
    from sglang.srt.layers.linear import UnquantizedLinearMethod

    mode = args.opd_qwen_replay_backend
    runtime = validate_qwen_replay_runtime(backend=mode) if mode == "native_fa3_v2" else validate_qwen_replay_runtime()
    count = _install_linear(model, batch_invariant_linear, UnquantizedLinearMethod)
    model._opd_qwen_replay_backend = mode
    source_paths = {"native_integration": Path(__file__),
                    "linear": Path(inspect.getfile(batch_invariant_linear)),
                    "attention_backend": Path(__file__).parent / "attention" / "flashattention_backend.py"}
    result = {
        "backend": mode, "projection_policy": "fixed_tile_triton_32_64_32",
        "attention_backend": "fa3", "num_splits": 1, "tp_size": 1,
        "dtype": "bfloat16", "device": "cuda", "model_type": "qwen3",
        "model_architecture": "Qwen3-0.6B", "projection_module_count": count,
        "disable_cuda_graph": True, "disable_overlap_schedule": True,
        "disable_radix_cache": True,
        "runtime_versions": runtime,
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_paths.items()},
    }
    logger.info("OPD Qwen replay arithmetic: %s", json.dumps(result, sort_keys=True))
    return result
