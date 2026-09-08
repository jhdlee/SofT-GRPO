"""Explicit Qwen3 replay arithmetic and the compatible diagnostic installer.

CUDA primitives use native SGL kernels with mathematical backwards. Production
replay selects fixed-tile projections and FA3 attention with one split; the
probe installer retains configurable projections and attention for comparisons.
The worker must enforce the supported FSDP1 configuration before installation.
"""

from __future__ import annotations

import math
import types

import torch
import torch.nn.functional as F


def _accumulate(value):
    return value.double() if value.dtype == torch.float64 else value.float()


class _NativeRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight, residual, epsilon):
        total = _accumulate(value)
        if residual is not None:
            total = total + _accumulate(residual)
        inverse = torch.rsqrt(total.square().mean(-1, keepdim=True) + epsilon)
        ctx.save_for_backward(total, inverse, weight)
        ctx.has_residual = residual is not None
        ctx.input_dtype = value.dtype
        if value.is_cuda and value.dtype in (torch.float16, torch.bfloat16):
            from sgl_kernel import fused_add_rmsnorm, rmsnorm

            if residual is None:
                output = rmsnorm(value.contiguous(), weight.contiguous(), epsilon)
                next_residual = value.new_empty(0)
            else:
                # Native kernels mutate both arguments. Never mutate graph inputs.
                output = value.contiguous().clone()
                next_residual = residual.contiguous().clone()
                fused_add_rmsnorm(output, next_residual, weight.contiguous(), epsilon)
        else:
            output = (total * inverse * _accumulate(weight)).to(value.dtype)
            next_residual = total.to(value.dtype) if residual is not None else value.new_empty(0)
        return output, next_residual

    @staticmethod
    def backward(ctx, grad_output, grad_residual):
        total, inverse, weight = ctx.saved_tensors
        grad = torch.zeros_like(total) if grad_output is None else _accumulate(grad_output)
        weighted = grad * _accumulate(weight)
        dx = inverse * (weighted - total * inverse.square() * (weighted * total).mean(-1, keepdim=True))
        dw = (grad * total * inverse).sum(tuple(range(total.ndim - 1))).to(weight.dtype)
        if ctx.has_residual and grad_residual is not None:
            dx = dx + _accumulate(grad_residual)
        dx = dx.to(ctx.input_dtype)
        return dx, dw, dx if ctx.has_residual else None, None


def native_rms_norm(value, weight, epsilon=1e-6, residual=None):
    """Native forward ordering with gradients for input, residual, and weight."""
    if value.ndim < 1 or value.shape[-1] == 0 or weight.shape != (value.shape[-1],):
        raise ValueError("native RMSNorm requires nonempty features and a matching one-dimensional weight")
    tensors = (value, weight) if residual is None else (value, weight, residual)
    if any(t.dtype != value.dtype or t.device != value.device for t in tensors):
        # The native kernel reinterprets all pointers using the input dtype.
        raise ValueError("native RMSNorm input, weight, and residual must have matching dtype and device")
    if value.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("native RMSNorm requires real floating point tensors")
    if residual is not None and residual.shape != value.shape:
        raise ValueError("native RMSNorm residual shape must match the input")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0:
        raise ValueError("native RMSNorm epsilon must be finite and positive")
    shape = value.shape
    output, carry = _NativeRMSNorm.apply(
        value.reshape(-1, shape[-1]), weight,
        None if residual is None else residual.reshape(-1, shape[-1]), float(epsilon),
    )
    output = output.reshape(shape)
    return output if residual is None else (output, carry.reshape(shape))


class _NativeSiluMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_up):
        ctx.save_for_backward(gate_up)
        gate, up = _accumulate(gate_up).chunk(2, -1)
        if gate_up.is_cuda and gate_up.dtype in (torch.float16, torch.bfloat16):
            from sgl_kernel import silu_and_mul

            output = torch.empty_like(gate_up[..., :gate.shape[-1]])
            silu_and_mul(gate_up.contiguous(), output)
            return output
        return (F.silu(gate) * up).to(gate_up.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (gate_up,) = ctx.saved_tensors
        gate, up = _accumulate(gate_up).chunk(2, -1)
        sigmoid = gate.sigmoid()
        grad = _accumulate(grad_output)
        dgate = grad * up * sigmoid * (1 + gate * (1 - sigmoid))
        dup = grad * gate * sigmoid
        return torch.cat((dgate, dup), -1).to(gate_up.dtype)


def native_silu_mul(gate_up):
    if gate_up.shape[-1] % 2:
        raise ValueError("packed gate/up width must be even")
    return _NativeSiluMul.apply(gate_up)


def _rotate(value, positions, cache, inverse=False):
    cosine, sine = cache.index_select(0, positions.reshape(-1)).chunk(2, -1)
    cosine, sine = cosine.unsqueeze(1), sine.unsqueeze(1)
    if inverse:
        sine = -sine
    first, second = _accumulate(value).chunk(2, -1)
    return torch.cat((first * cosine - second * sine, second * cosine + first * sine), -1).to(value.dtype)


class _NativeRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, positions, cache):
        ctx.save_for_backward(positions, cache)
        if query.is_cuda and query.dtype in (torch.float16, torch.bfloat16):
            from sgl_kernel import apply_rope_with_cos_sin_cache_inplace

            q, k = query.contiguous().clone(), key.contiguous().clone()
            apply_rope_with_cos_sin_cache_inplace(
                positions=positions.reshape(-1), query=q.flatten(1), key=k.flatten(1),
                head_size=q.shape[-1], cos_sin_cache=cache, is_neox=True,
            )
            return q, k
        return _rotate(query, positions, cache), _rotate(key, positions, cache)

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        positions, cache = ctx.saved_tensors
        return (
            None if grad_q is None else _rotate(grad_q, positions, cache, inverse=True),
            None if grad_k is None else _rotate(grad_k, positions, cache, inverse=True),
            None, None,
        )


def native_rope(query, key, positions, cache, *, _positions_validated=False):
    """NeoX RoPE on [tokens, heads, head_dim], with a fixed FP32 cache."""
    if (query.ndim != 3 or key.ndim != 3 or query.shape[0] != key.shape[0]
            or query.shape[-1] != key.shape[-1] or query.shape[-1] == 0 or query.shape[-1] % 2):
        raise ValueError("RoPE requires aligned token/head tensors")
    if positions.numel() != query.shape[0] or cache.requires_grad:
        raise ValueError("RoPE requires one position per token and a fixed cache")
    if query.dtype != key.dtype or any(t.device != query.device for t in (key, positions, cache)):
        raise ValueError("RoPE requires matching Q/K dtype and matching tensor devices")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("RoPE requires real floating point Q/K tensors")
    if positions.dtype != torch.int64:
        raise ValueError("RoPE positions must be int64")
    if cache.ndim != 2 or cache.shape[1] != query.shape[-1]:
        raise ValueError("RoPE cache must have shape [max_positions, head_dim]")
    if cache.dtype not in (torch.float32, torch.float64) or (query.is_cuda and cache.dtype != torch.float32):
        raise ValueError("RoPE requires an FP32 CUDA cache (CPU double references may use FP64)")
    if not _positions_validated:
        _validate_rope_positions(positions, cache.shape[0])
    return _NativeRope.apply(query, key, positions, cache)


def _validate_rope_positions(positions, cache_length):
    if bool(((positions < 0) | (positions >= cache_length)).any()):
        raise ValueError("RoPE position is outside the initialized cache")


def _build_native_rope_cache(config, device):
    # Native SGLang performs these operations on its model's CUDA device.
    # Building this on CPU and transferring afterwards changes the arithmetic.
    with torch.autocast(device_type=device.type, enabled=False):
        head_dim = config.head_dim
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        positions = torch.arange(config.max_position_embeddings, dtype=torch.float32, device=device)
        angles = torch.einsum("i,j->ij", positions, inv_freq)
        return torch.cat((angles.cos(), angles.sin()), -1)


def _replay_cache_device(cache_device):
    device = torch.device(cache_device)
    if device.type != "cuda" or not torch.cuda.is_available() or torch.version.hip is not None:
        raise ValueError("Qwen replay arithmetic requires an actual NVIDIA CUDA cache device")
    return torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)


def _packed_linear(value, modules, linear=F.linear, *, fp32_masters=False):
    if fp32_masters:
        from verl.opd.qwen_lora import effective_projection_weight, native_base_parameter
        weight = torch.cat([effective_projection_weight(module, dtype=value.dtype) for module in modules], 0)
        biases = [None if module.bias is None else native_base_parameter(module, "bias") for module in modules]
    else:
        weight = torch.cat([module.weight for module in modules], 0)
        biases = [module.bias for module in modules]
    if any(bias is None for bias in biases) and not all(bias is None for bias in biases):
        raise ValueError("mixed packed-projection bias settings are unsupported")
    bias = None if biases[0] is None else torch.cat(biases, 0)
    if bias is not None and fp32_masters:
        bias = bias.to(value.dtype)
    return linear(value, weight, bias)


def install_probe_candidate(model, *, emit=None, linear=F.linear, attention=None):
    """Patch one HF model instance for a controlled native-arithmetic comparison.

    No global class patch, parameter replacement, checkpoint renaming, density
    change, or configuration default changes. Cached generation and checkpointed
    training are rejected: this is a forward-parity experiment until GPU evidence
    justifies integrating a complete training implementation.
    """
    return _install_native_arithmetic(model, emit=emit, linear=linear, attention=attention)


def install_qwen_replay_arithmetic(model, *, cache_device, emit=None, backend="native_fa3_v1"):
    """Install replay arithmetic before FSDP wrapping without replacing weights.

    ``cache_device`` must be the rank's actual CUDA execution device even when
    model parameters are still FP32 on CPU. Version one retains BF16 FSDP
    forward parameters; version two gathers FP32 masters and casts explicitly
    for BF16 arithmetic. Both require FSDP1, FP32 buffers, TP/SP=1, and disabled
    gradient checkpointing. Install actor and teacher before checkpoint loading.
    """
    from verl.opd.batch_invariant_linear import batch_invariant_linear
    from verl.opd.native_fa3_attention import native_fa3_attention

    if backend not in ("native_fa3_v1", "native_fa3_v2"):
        raise ValueError("Qwen replay installer requires native_fa3_v1 or native_fa3_v2")
    if backend == "native_fa3_v2":
        from verl.opd.native_fa3_attention import native_fa3_attention_v2
        native_fa3_attention = native_fa3_attention_v2
        if model.config.head_dim != 128:
            raise ValueError("native_fa3_v2 requires head dimension 128")

    return _install_native_arithmetic(
        model, emit=emit, linear=batch_invariant_linear, attention=native_fa3_attention,
        cache_device=cache_device, production=True, fp32_masters=backend == "native_fa3_v2",
    )


def _install_native_arithmetic(model, *, emit=None, linear=F.linear, attention=None,
                               cache_device=None, production=False, fp32_masters=False):
    if model.config.model_type != "qwen3" or getattr(model.config, "rope_scaling", None):
        raise ValueError("candidate supports unscaled dense Qwen3 only")
    if getattr(model, "_opd_native_arithmetic_candidate", False):
        raise ValueError("candidate already installed")
    if getattr(model, "_opd_qwen_lora_config", None) is not None and not fp32_masters:
        raise ValueError("native LoRA requires the v2 FP32-master arithmetic")
    if fp32_masters and any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("native_fa3_v2 requires FP32 model and adapter masters")
    core = model.model
    config = model.config
    if getattr(core, "gradient_checkpointing", False):
        raise ValueError("native arithmetic does not support gradient checkpointing")
    head_dim = config.head_dim
    if (type(head_dim) is not int or head_dim <= 0 or head_dim % 2
            or type(config.max_position_embeddings) is not int or config.max_position_embeddings <= 0
            or not math.isfinite(float(config.rope_theta)) or float(config.rope_theta) <= 0):
        raise ValueError("native arithmetic requires a valid fixed RoPE configuration")
    if hasattr(core, "_opd_native_rope_cache"):
        raise ValueError("native arithmetic cache already exists")
    if linear is not F.linear and (
            not hasattr(model, "lm_head") or not hasattr(model.lm_head, "weight") or not hasattr(model.lm_head, "bias")):
        raise ValueError("native arithmetic requires a compatible Qwen3 LM head")
    if production:
        if head_dim not in (64, 128, 256) or getattr(config, "hidden_act", "silu") != "silu":
            raise ValueError("Qwen replay requires SiLU and a native supported head dimension")
        if getattr(config, "attention_dropout", 0.0) != 0.0:
            raise ValueError("Qwen replay requires zero attention dropout")
        for layer in core.layers:
            if layer.self_attn.sliding_window not in (None, -1):
                raise ValueError("Qwen replay does not support sliding-window attention")
        device = _replay_cache_device(cache_device)
    else:
        device = core.embed_tokens.weight.device
    cache = _build_native_rope_cache(config, device)
    # All rejecting validation and cache allocation precedes instance mutation.
    core.register_buffer("_opd_native_rope_cache", cache, persistent=False)
    model._opd_native_arithmetic_candidate = True
    if production:
        model._opd_qwen_replay_arithmetic = True

    def record(name, tensor):
        if emit is not None:
            emit(name, tensor.detach().reshape(-1, tensor.shape[-1]))

    def norm_weight(module, dtype):
        if fp32_masters:
            from verl.opd.qwen_lora import native_base_parameter
            return native_base_parameter(module).to(dtype)
        return module.weight

    def projection(value, module):
        if fp32_masters:
            from verl.opd.qwen_lora import effective_projection_weight, native_base_parameter
            weight = effective_projection_weight(module, dtype=value.dtype)
            bias = None if module.bias is None else native_base_parameter(module, "bias").to(value.dtype)
            return linear(value, weight, bias)
        return linear(value, module.weight, module.bias)

    def layer_forward(layer, hidden_states, *, residual=None, attention_mask=None, position_ids=None, rope_cache=None,
                      _opd_rope_positions_validated=False, **kwargs):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

        prefix = f"layer.{layer.self_attn.layer_idx}."
        if residual is None:
            residual = hidden_states
            normalized = native_rms_norm(hidden_states, norm_weight(layer.input_layernorm, hidden_states.dtype), layer.input_layernorm.variance_epsilon)
        else:
            normalized, residual = native_rms_norm(hidden_states, norm_weight(layer.input_layernorm, hidden_states.dtype), layer.input_layernorm.variance_epsilon, residual)
        record(prefix + "norm_in", normalized)
        attn = layer.self_attn
        qkv = _packed_linear(normalized, (attn.q_proj, attn.k_proj, attn.v_proj), linear, fp32_masters=fp32_masters)
        record(prefix + "qkv", qkv)
        shape = normalized.shape[:-1]
        query, key, value = qkv.split((attn.q_proj.out_features, attn.k_proj.out_features, attn.v_proj.out_features), -1)
        query = native_rms_norm(query.reshape(-1, head_dim), norm_weight(attn.q_norm, query.dtype), attn.q_norm.variance_epsilon).reshape(-1, config.num_attention_heads, head_dim)
        key = native_rms_norm(key.reshape(-1, head_dim), norm_weight(attn.k_norm, key.dtype), attn.k_norm.variance_epsilon).reshape(-1, config.num_key_value_heads, head_dim)
        record(prefix + "q_norm", query.flatten(1)); record(prefix + "k_norm", key.flatten(1))
        query, key = native_rope(query, key, position_ids, rope_cache,
                                _positions_validated=_opd_rope_positions_validated)
        record(prefix + "q_rope", query.flatten(1)); record(prefix + "k_rope", key.flatten(1))
        query = query.reshape(*shape, -1, head_dim).transpose(1, 2)
        key = key.reshape(*shape, -1, head_dim).transpose(1, 2)
        value = value.reshape(*shape, -1, head_dim).transpose(1, 2)
        interface = attention or (eager_attention_forward if config._attn_implementation == "eager" else ALL_ATTENTION_FUNCTIONS[config._attn_implementation])
        attended, _ = interface(attn, query, key, value, attention_mask, dropout=attn.attention_dropout if layer.training else 0.0,
                                scaling=attn.scaling, sliding_window=attn.sliding_window, position_ids=position_ids, **kwargs)
        attended = attended.reshape(*shape, -1).contiguous()
        record(prefix + "attention", attended)
        projected = projection(attended, attn.o_proj)
        record(prefix + "attention_projected", projected)
        normalized, residual = native_rms_norm(projected, norm_weight(layer.post_attention_layernorm, projected.dtype), layer.post_attention_layernorm.variance_epsilon, residual)
        record(prefix + "norm_post", normalized)
        gate_up = _packed_linear(normalized, (layer.mlp.gate_proj, layer.mlp.up_proj), linear, fp32_masters=fp32_masters)
        record(prefix + "gate_up", gate_up)
        hidden_states = projection(native_silu_mul(gate_up.reshape(-1, gate_up.shape[-1])).reshape(*shape, -1),
                                   layer.mlp.down_proj)
        record(prefix + "mlp", hidden_states)
        if emit is not None:
            record(prefix + "block_total", (hidden_states.float() + residual.float()).to(hidden_states.dtype))
        return hidden_states, residual

    for layer in core.layers:
        layer.forward = types.MethodType(layer_forward, layer)

    def model_forward(core_self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                      inputs_embeds=None, use_cache=None, output_attentions=False, output_hidden_states=False,
                      cache_position=None, return_dict=None, opd_cu_seqlens=None, opd_max_seqlen=None, **kwargs):
        from transformers.modeling_outputs import BaseModelOutputWithPast

        if use_cache or past_key_values is not None or output_attentions or output_hidden_states or core_self.gradient_checkpointing:
            raise ValueError("parity candidate supports uncached replay without attention/hidden-state returns or gradient checkpointing")
        if return_dict is False:
            raise ValueError("native arithmetic requires return_dict=True")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        hidden_states = core_self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if fp32_masters:
            hidden_states = hidden_states.to(torch.bfloat16)
        if hidden_states.ndim != 3 or (production and hidden_states.shape[0] != 1):
            raise ValueError("Qwen replay requires hidden states [1, total_tokens, hidden_dim]")
        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0).expand(hidden_states.shape[0], -1)
        elif position_ids.shape[0] == 1 and hidden_states.shape[0] != 1:
            position_ids = position_ids.expand(hidden_states.shape[0], -1)
        if production:
            from verl.opd.native_fa3_attention import prepare_fa3_attention_layout

            layout = prepare_fa3_attention_layout(
                position_ids, total_tokens=hidden_states.shape[1], device=hidden_states.device,
                attention_mask=attention_mask, opd_cu_seqlens=opd_cu_seqlens, opd_max_seqlen=opd_max_seqlen,
            )
            kwargs["opd_attention_layout"] = layout
            mask = None
        else:
            if opd_cu_seqlens is not None or opd_max_seqlen is not None:
                raise ValueError("explicit packed replay layout requires the production installer")
            # HF 4.51 eager/SDPA does not separate rows at packed position resets.
            resets = position_ids[:, 1:] <= position_ids[:, :-1]
            if attention_mask is not None and attention_mask.ndim == 2:
                resets = resets & attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
            if config._attn_implementation != "flash_attention_2" and bool(resets.any()):
                raise ValueError("packed position resets require flash_attention_2 in this diagnostic candidate")
            if cache_position is None:
                cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
            mask = core_self._update_causal_mask(attention_mask, hidden_states, cache_position, None, False)
        record("embedding", hidden_states)
        residual = None
        if core_self._opd_native_rope_cache.dtype != torch.float32:
            raise ValueError("native arithmetic requires the original FP32 RoPE cache; do not cast an installed candidate")
        if core_self._opd_native_rope_cache.device != hidden_states.device:
            raise ValueError("native arithmetic RoPE cache must be initialized on the execution device")
        if position_ids.dtype != torch.int64 or position_ids.device != hidden_states.device:
            raise ValueError("native arithmetic positions must be int64 on the execution device")
        _validate_rope_positions(position_ids, core_self._opd_native_rope_cache.shape[0])
        for layer in core_self.layers:
            hidden_states, residual = layer(hidden_states, residual=residual, attention_mask=mask, position_ids=position_ids,
                                            rope_cache=core_self._opd_native_rope_cache,
                                            _opd_rope_positions_validated=True, **kwargs)
        hidden_states, _ = native_rms_norm(hidden_states, norm_weight(core_self.norm, hidden_states.dtype), core_self.norm.variance_epsilon, residual)
        record("final_norm", hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)

    core.forward = types.MethodType(model_forward, core)
    if fp32_masters and getattr(model, "_opd_qwen_lora_config", None) is not None:
        # External continuous/teacher replay also calls get_input_embeddings()
        # directly. Keep every lookup on the checked frozen base, including
        # FSDP's temporary views, without changing FP32 lookup/BF16 cast order.
        def embedding_forward(embedding, input_ids):
            from verl.opd.qwen_lora import native_base_parameter
            return F.embedding(input_ids, native_base_parameter(embedding), embedding.padding_idx,
                               embedding.max_norm, embedding.norm_type, embedding.scale_grad_by_freq,
                               embedding.sparse)
        core.embed_tokens.forward = types.MethodType(embedding_forward, core.embed_tokens)
    if linear is not F.linear or fp32_masters:
        def head_forward(head, hidden_states):
            return projection(hidden_states, head)
        model.lm_head.forward = types.MethodType(head_forward, model.lm_head)
    return model
