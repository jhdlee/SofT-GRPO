"""Opt-in Qwen3 vLLM arithmetic matching native actor replay.

This installs the complete nonattention intervention measured by probe 472028.
vLLM still owns categorical sampling, paged KV attention, and request scheduling.
The caller must hold the frozen-batch lifecycle guard with an idle, awake engine.
All projection lookups use the current inference Parameters after weight loads.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
from pathlib import Path
import types

import torch

RECIPE = "qwen3_vllm_native_nonattention_v1"


def _runtime_check(device, *, backend, tensor_parallel_size, engine_config):
    if backend != "native_fa3_v2" or type(tensor_parallel_size) is not int or tensor_parallel_size != 1:
        raise ValueError("native vLLM arithmetic requires native_fa3_v2 and TP1")
    if importlib.metadata.version("vllm") != "0.8.5":
        raise ValueError("native vLLM arithmetic requires installed vLLM 0.8.5")
    if (device.type != "cuda" or torch.version.hip is not None
            or torch.cuda.get_device_capability(device) != (9, 0)):
        raise ValueError("native vLLM arithmetic requires an NVIDIA Hopper device")
    if (engine_config.model_config.enforce_eager is not True
            or int(engine_config.compilation_config.level) != 0):
        raise ValueError("native vLLM arithmetic requires eager execution without compilation")
    if (engine_config.parallel_config.tensor_parallel_size != 1
            or engine_config.parallel_config.pipeline_parallel_size != 1):
        raise ValueError("native vLLM arithmetic requires actual TP1/PP1 execution")
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("native vLLM arithmetic requires TF32 disabled")


def _validate_model(model, actor_config):
    config = model.config
    if (type(model).__name__ != "Qwen3ForCausalLM" or config.model_type != "qwen3"
            or actor_config.model_type != "qwen3"):
        raise ValueError("native vLLM arithmetic requires Qwen3")
    for name in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                 "num_key_value_heads", "head_dim", "max_position_embeddings", "rope_theta",
                 "vocab_size", "rms_norm_eps", "tie_word_embeddings"):
        if getattr(config, name) != getattr(actor_config, name):
            raise ValueError("native vLLM actor/model configuration differs at " + name)
    if (config.head_dim != 128 or config.rope_scaling is not None or config.attention_bias
            or config.hidden_act != "silu" or config.sliding_window is not None):
        raise ValueError("native vLLM arithmetic requires unscaled full-context Qwen3 with head dimension 128")
    if model.quant_config is not None or model.lora_config is not None:
        raise ValueError("native vLLM arithmetic requires dense unquantized inference weights")
    if model.logits_processor.scale != 1.0 or model.logits_processor.soft_cap is not None:
        raise ValueError("native vLLM arithmetic requires unscaled full-vocabulary logits")
    if model.logits_processor.org_vocab_size != config.vocab_size:
        raise ValueError("native vLLM vocabulary differs from actor")
    if config.tie_word_embeddings and model.lm_head is not model.model.embed_tokens:
        raise ValueError("native vLLM tied head/embedding ownership differs")
    layers = model.model.layers
    if len(layers) != config.num_hidden_layers:
        raise ValueError("native vLLM layer inventory differs")
    parameters = dict(model.named_parameters())
    if not parameters:
        raise ValueError("native vLLM inference weight inventory is empty")
    device = next(iter(parameters.values())).device
    expected = {"model.embed_tokens.weight": (config.vocab_size, config.hidden_size),
                "model.norm.weight": (config.hidden_size,)}
    if not config.tie_word_embeddings:
        expected["lm_head.weight"] = (config.vocab_size, config.hidden_size)
    q = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    for index, layer in enumerate(layers):
        attention = layer.self_attn
        if (attention.head_dim != 128 or attention.num_heads != config.num_attention_heads
                or attention.num_kv_heads != config.num_key_value_heads
                or getattr(attention.attn.impl, "vllm_flash_attn_version", None) != 3
                or type(attention.attn.impl).__name__ != "FlashAttentionImpl"):
            raise ValueError("native vLLM arithmetic requires unmodified TP1 FA3 attention")
        shapes = {"self_attn.qkv_proj.weight": (q + 2 * kv, config.hidden_size),
                  "self_attn.o_proj.weight": (config.hidden_size, q),
                  "self_attn.q_norm.weight": (config.head_dim,), "self_attn.k_norm.weight": (config.head_dim,),
                  "mlp.gate_up_proj.weight": (2 * config.intermediate_size, config.hidden_size),
                  "mlp.down_proj.weight": (config.hidden_size, config.intermediate_size),
                  "input_layernorm.weight": (config.hidden_size,),
                  "post_attention_layernorm.weight": (config.hidden_size,)}
        expected.update({f"model.layers.{index}.{name}": shape for name, shape in shapes.items()})
        for module in (attention.qkv_proj, attention.o_proj, layer.mlp.gate_up_proj, layer.mlp.down_proj):
            if type(module.quant_method).__name__ != "UnquantizedLinearMethod" or module.bias is not None:
                raise ValueError("native vLLM arithmetic requires unquantized bias-free projections")
        for module in (attention.q_norm, attention.k_norm, layer.input_layernorm, layer.post_attention_layernorm):
            if module.variance_epsilon != config.rms_norm_eps or module.variance_size_override is not None:
                raise ValueError("native vLLM normalization configuration differs")
    if model.model.norm.variance_epsilon != config.rms_norm_eps or model.model.norm.variance_size_override is not None:
        raise ValueError("native vLLM final normalization configuration differs")
    if set(parameters) != set(expected):
        raise ValueError("native vLLM fused inference weight inventory differs")
    for name, value in parameters.items():
        if (tuple(value.shape) != expected[name] or value.dtype != torch.bfloat16
                or value.device != device or not value.is_contiguous()):
            raise ValueError("native vLLM inference weight shape/dtype/device differs at " + name)
    return device, len(parameters)


def verify_qwen_vllm_weights(model, weights):
    """Compare every loaded fused/tied tensor to this actor version's export.

    Materialization is already complete: this function performs no collectives
    and holds at most one fused projection copy, not another model-sized copy.
    """
    remaining = set(weights)
    checked = 0
    for name, actual in model.named_parameters():
        if name.endswith(".self_attn.qkv_proj.weight"):
            names = [name.replace("qkv_proj", part) for part in ("q_proj", "k_proj", "v_proj")]
        elif name.endswith(".mlp.gate_up_proj.weight"):
            names = [name.replace("gate_up_proj", part) for part in ("gate_proj", "up_proj")]
        else:
            names = [name]
        if not set(names) <= remaining:
            raise ValueError("native vLLM exported weight inventory differs at " + name)
        values = [weights[key] for key in names]
        if any(not isinstance(value, torch.Tensor) or hasattr(value, "full_tensor")
               or value.dtype != torch.bfloat16 for value in values):
            raise ValueError("native vLLM weight verification requires materialized BF16 exports")
        reference = values[0] if len(values) == 1 else torch.cat(values, dim=0)
        if actual.shape != reference.shape or not torch.equal(actual, reference.to(actual.device)):
            raise ValueError("native vLLM loaded weight differs from current actor export at " + name)
        remaining.difference_update(names)
        checked += 1
    if model.config.tie_word_embeddings:
        if remaining != {"lm_head.weight"} or not torch.equal(weights["lm_head.weight"], weights["model.embed_tokens.weight"]):
            raise ValueError("native vLLM tied actor export differs")
        remaining.remove("lm_head.weight")
    if remaining:
        raise ValueError("native vLLM unconsumed actor export weights: " + str(sorted(remaining)))
    return {"loaded_weights_exact": True, "inference_weight_count": checked,
            "tied_embedding_head": bool(model.config.tie_word_embeddings)}


def install_qwen_vllm_replay_arithmetic(model, actor_config, *, backend, tensor_parallel_size, engine_config):
    """Install once, then verify the same hooks and current weights on each entry."""
    device, parameter_count = _validate_model(model, actor_config)
    _runtime_check(device, backend=backend, tensor_parallel_size=tensor_parallel_size, engine_config=engine_config)
    identity = {"recipe": RECIPE, "backend": backend, "vllm_version": "0.8.5",
                "attention": "stock_vllm_fa3", "projection_tile": [32, 64, 32],
                "rope_cache_dtype": "float32", "inference_weight_count": parameter_count,
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    previous = getattr(model, "_opd_qwen_vllm_arithmetic", None)
    if previous is not None:
        if previous != identity:
            raise ValueError("native vLLM arithmetic identity changed")
        for instance, name, value in model._opd_qwen_vllm_patch_inventory:
            if getattr(instance, name) is not value:
                raise ValueError("native vLLM arithmetic hook changed at " + name)
        cache = model._opd_qwen_vllm_rope_cache
        if cache.device != device or cache.dtype != torch.float32:
            raise ValueError("native vLLM RoPE cache device/dtype changed")
        return dict(identity)

    from .batch_invariant_linear import batch_invariant_linear
    from .qwen_native_arithmetic import _build_native_rope_cache, native_rms_norm, native_rope, native_silu_mul

    cache = _build_native_rope_cache(actor_config, device)
    changes = []
    installed = []

    def patch(instance, name, value):
        changes.append((instance, name, name in instance.__dict__, instance.__dict__.get(name)))
        setattr(instance, name, value)
        installed.append((instance, name, value))

    def normalize(module, value, residual=None):
        return native_rms_norm(value, module.weight, module.variance_epsilon, residual)

    def activation(module, value):
        return native_silu_mul(value)

    def apply(method, owner, value, bias=None):
        return batch_invariant_linear(value, owner.weight, bias)

    def rotate(module, positions, query, key, offsets=None):
        if offsets is not None:
            raise ValueError("native vLLM arithmetic disallows rotary offsets")
        qshape, kshape = query.shape, key.shape
        q, k = native_rope(query.reshape(query.shape[0], -1, 128),
                           key.reshape(key.shape[0], -1, 128), positions, cache)
        return q.reshape(qshape), k.reshape(kshape)

    def logits(processor, hidden, head, embedding_bias=None):
        if embedding_bias is not None:
            raise ValueError("native vLLM arithmetic disallows embedding bias")
        return batch_invariant_linear(hidden.to(head.weight.dtype), head.weight)[..., :processor.org_vocab_size]

    try:
        rotaries = set()
        for layer in model.model.layers:
            attention = layer.self_attn
            # vLLM may share the cached rotary object across decoder layers.
            if id(attention.rotary_emb) not in rotaries:
                patch(attention.rotary_emb, "forward", types.MethodType(rotate, attention.rotary_emb))
                rotaries.add(id(attention.rotary_emb))
            for module in (attention.q_norm, attention.k_norm):
                patch(module, "forward_native", types.MethodType(normalize, module))
            for module in (attention.qkv_proj, attention.o_proj, layer.mlp.gate_up_proj, layer.mlp.down_proj):
                method = copy.copy(module.quant_method)
                method.apply = types.MethodType(apply, method)
                patch(module, "quant_method", method)
                installed.append((method, "apply", method.apply))
            for module in (layer.input_layernorm, layer.post_attention_layernorm):
                patch(module, "forward", types.MethodType(normalize, module))
            patch(layer.mlp.act_fn, "forward", types.MethodType(activation, layer.mlp.act_fn))
        patch(model.model.norm, "forward", types.MethodType(normalize, model.model.norm))
        patch(model.logits_processor, "_get_logits", types.MethodType(logits, model.logits_processor))
    except BaseException:
        for instance, name, existed, value in reversed(changes):
            if existed:
                setattr(instance, name, value)
            else:
                delattr(instance, name)
        raise
    model._opd_qwen_vllm_rope_cache = cache
    model._opd_qwen_vllm_patch_inventory = installed
    model._opd_qwen_vllm_arithmetic = identity
    return dict(identity)
