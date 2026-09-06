"""Opt-in Qwen3 forward-parity candidate for the diagnostic probe.

This module is deliberately not installed by a trainer or profile. CUDA forward
primitives use the same SGL kernels as native inference, with explicit backwards
instead of transplanting inference's detached weights/in-place operations into
the student. Attention still uses the configured HF/VERL backend. Thus matching
these primitives does not assert full native cached-decode/replay equivalence.
"""

from __future__ import annotations

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


def native_rope(query, key, positions, cache):
    """NeoX RoPE on [tokens, heads, head_dim], with a fixed FP32 cache."""
    if query.ndim != 3 or key.ndim != 3 or query.shape[0] != key.shape[0]:
        raise ValueError("RoPE requires aligned token/head tensors")
    if positions.numel() != query.shape[0] or cache.requires_grad:
        raise ValueError("RoPE requires one position per token and a fixed cache")
    return _NativeRope.apply(query, key, positions, cache)


def _packed_linear(value, modules):
    weight = torch.cat([module.weight for module in modules], 0)
    biases = [module.bias for module in modules]
    if any(bias is None for bias in biases) and not all(bias is None for bias in biases):
        raise ValueError("mixed packed-projection bias settings are unsupported")
    bias = None if biases[0] is None else torch.cat(biases, 0)
    return F.linear(value, weight, bias)


def install_probe_candidate(model, *, emit=None):
    """Patch one HF model instance for a controlled native-arithmetic comparison.

    No global class patch, parameter replacement, checkpoint renaming, density
    change, or configuration default changes. Cached generation and checkpointed
    training are rejected: this is a forward-parity experiment until GPU evidence
    justifies integrating a complete training implementation.
    """
    if model.config.model_type != "qwen3" or getattr(model.config, "rope_scaling", None):
        raise ValueError("candidate supports unscaled dense Qwen3 only")
    if getattr(model, "_opd_native_arithmetic_candidate", False):
        raise ValueError("candidate already installed")
    model._opd_native_arithmetic_candidate = True
    core = model.model
    config = model.config
    device = core.embed_tokens.weight.device
    head_dim = config.head_dim
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    positions = torch.arange(config.max_position_embeddings, dtype=torch.float32, device=device)
    angles = torch.einsum("i,j->ij", positions, inv_freq)
    core.register_buffer("_opd_native_rope_cache", torch.cat((angles.cos(), angles.sin()), -1), persistent=False)

    def record(name, tensor):
        if emit is not None:
            emit(name, tensor.detach().reshape(-1, tensor.shape[-1]))

    def layer_forward(layer, hidden_states, *, residual=None, attention_mask=None, position_ids=None, **kwargs):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

        prefix = f"layer.{layer.self_attn.layer_idx}."
        if residual is None:
            residual = hidden_states
            normalized = native_rms_norm(hidden_states, layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon)
        else:
            normalized, residual = native_rms_norm(hidden_states, layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon, residual)
        record(prefix + "norm_in", normalized)
        attn = layer.self_attn
        qkv = _packed_linear(normalized, (attn.q_proj, attn.k_proj, attn.v_proj))
        record(prefix + "qkv", qkv)
        shape = normalized.shape[:-1]
        query, key, value = qkv.split((attn.q_proj.out_features, attn.k_proj.out_features, attn.v_proj.out_features), -1)
        query = native_rms_norm(query.reshape(-1, head_dim), attn.q_norm.weight, attn.q_norm.variance_epsilon).reshape(-1, config.num_attention_heads, head_dim)
        key = native_rms_norm(key.reshape(-1, head_dim), attn.k_norm.weight, attn.k_norm.variance_epsilon).reshape(-1, config.num_key_value_heads, head_dim)
        record(prefix + "q_norm", query.flatten(1)); record(prefix + "k_norm", key.flatten(1))
        query, key = native_rope(query, key, position_ids, core._opd_native_rope_cache)
        record(prefix + "q_rope", query.flatten(1)); record(prefix + "k_rope", key.flatten(1))
        query = query.reshape(*shape, -1, head_dim).transpose(1, 2)
        key = key.reshape(*shape, -1, head_dim).transpose(1, 2)
        value = value.reshape(*shape, -1, head_dim).transpose(1, 2)
        interface = eager_attention_forward if config._attn_implementation == "eager" else ALL_ATTENTION_FUNCTIONS[config._attn_implementation]
        attended, _ = interface(attn, query, key, value, attention_mask, dropout=attn.attention_dropout if layer.training else 0.0,
                                scaling=attn.scaling, sliding_window=attn.sliding_window, position_ids=position_ids, **kwargs)
        attended = attended.reshape(*shape, -1).contiguous()
        record(prefix + "attention", attended)
        projected = attn.o_proj(attended)
        record(prefix + "attention_projected", projected)
        normalized, residual = native_rms_norm(projected, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.variance_epsilon, residual)
        record(prefix + "norm_post", normalized)
        gate_up = _packed_linear(normalized, (layer.mlp.gate_proj, layer.mlp.up_proj))
        record(prefix + "gate_up", gate_up)
        hidden_states = layer.mlp.down_proj(native_silu_mul(gate_up.reshape(-1, gate_up.shape[-1])).reshape(*shape, -1))
        record(prefix + "mlp", hidden_states)
        record(prefix + "block_total", (hidden_states.float() + residual.float()).to(hidden_states.dtype))
        return hidden_states, residual

    for layer in core.layers:
        layer.forward = types.MethodType(layer_forward, layer)

    def model_forward(core_self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                      inputs_embeds=None, use_cache=None, output_attentions=False, output_hidden_states=False,
                      cache_position=None, **kwargs):
        from transformers.modeling_outputs import BaseModelOutputWithPast

        if use_cache or past_key_values is not None or output_attentions or output_hidden_states or core_self.gradient_checkpointing:
            raise ValueError("parity candidate supports uncached replay without attention/hidden-state returns or gradient checkpointing")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        hidden_states = core_self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0).expand(hidden_states.shape[0], -1)
        elif position_ids.shape[0] == 1 and hidden_states.shape[0] != 1:
            position_ids = position_ids.expand(hidden_states.shape[0], -1)
        # HF 4.51's ordinary eager/SDPA causal mask does not use position
        # resets to separate packed rows. Only its FA2 varlen path does.
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
        for layer in core_self.layers:
            hidden_states, residual = layer(hidden_states, residual=residual, attention_mask=mask, position_ids=position_ids, **kwargs)
        hidden_states, _ = native_rms_norm(hidden_states, core_self.norm.weight, core_self.norm.variance_epsilon, residual)
        record("final_norm", hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)

    core.forward = types.MethodType(model_forward, core)
    return model
