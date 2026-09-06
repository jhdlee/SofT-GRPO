"""Diagnostic-only FA3 forward with an existing FA2 mathematical backward.

Forward uses sgl-kernel 0.1.1's ``flash_attn_varlen_func(num_splits=1)``.
That build exposes only forward (its CMake defines DISABLE_BACKWARD). Backward
uses flash-attn 2.7.3's ``_flash_attn_varlen_backward`` with the *actual* saved
FA3 output and natural-log LSE, both in FA2's unpadded varlen layout. There is
no second attention forward and no detached substitute output.

Both kernels implement the same causal softmax attention. Their floating-point
rounding differs: this is a mathematical attention gradient, not a claim to
differentiate FA3's discrete rounding exactly or to establish training parity.
No trainer installs this bridge. Its supported diagnostic is one unpacked,
unpadded self-attention row with GQA, no dropout/window/cache or custom mask.

Source contracts: vendored sgl-kernel/python/sgl_kernel/flash_attn.py:209;
https://github.com/Dao-AILab/flash-attention/blob/v2.7.3/flash_attn/flash_attn_interface.py
"""

from __future__ import annotations

import math

import torch
from torch.autograd.function import once_differentiable


def _fa3_forward(query, key, value, cumulative, length, scale):
    from sgl_kernel.flash_attn import flash_attn_varlen_func

    output, lse, *_ = flash_attn_varlen_func(
        query, key, value, cumulative, cumulative, length, length,
        softmax_scale=scale, causal=True, num_splits=1,
        return_softmax_lse=True,
    )
    return output, lse


def _fa2_backward(gradient, query, key, value, output, lse, cumulative, length, scale):
    from flash_attn.flash_attn_interface import _flash_attn_varlen_backward

    dq, dk, dv = (torch.empty_like(tensor) for tensor in (query, key, value))
    # Pinned 2.7.3 writes the supplied dq/dk/dv buffers and returns softmax_d.
    # FA3 varlen LSE is float32 [query_heads, total_query_tokens], already the
    # unpadded natural-log format expected here: never transpose or recompute it.
    _flash_attn_varlen_backward(
        gradient.contiguous(), query, key, value, output, lse,
        dq, dk, dv, cumulative, cumulative, length, length,
        dropout_p=0.0, softmax_scale=scale, causal=True,
        window_size_left=-1, window_size_right=-1, softcap=0.0,
        alibi_slopes=None, deterministic=True, rng_state=None,
    )
    return dq, dk, dv


class _NativeFA3Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, scale):
        length = query.shape[0]
        cumulative = torch.tensor([0, length], dtype=torch.int32, device=query.device)
        output, lse = _fa3_forward(query, key, value, cumulative, length, scale)
        if output.shape != query.shape or output.dtype != query.dtype or output.device != query.device:
            raise RuntimeError("FA3 returned an incompatible attention output")
        if lse.shape != (query.shape[1], length) or lse.dtype != torch.float32 or lse.device != query.device:
            raise RuntimeError("FA3 LSE must be float32 [query_heads, total_query_tokens]")
        ctx.save_for_backward(query, key, value, output, lse.contiguous(), cumulative)
        ctx.length, ctx.scale = length, scale
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        query, key, value, output, lse, cumulative = ctx.saved_tensors
        dq, dk, dv = _fa2_backward(
            gradient, query, key, value, output, lse, cumulative, ctx.length, ctx.scale,
        )
        return dq, dk, dv, None


def _dense_attention(query, key, value, scale):
    """CPU mathematical reference; CUDA never silently falls back here."""
    dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    groups = query.shape[1] // key.shape[1]
    q = query.to(dtype)
    k = key.to(dtype).repeat_interleave(groups, dim=1)
    v = value.to(dtype).repeat_interleave(groups, dim=1)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    causal = torch.ones(query.shape[-2], query.shape[-2], dtype=torch.bool, device=query.device).tril()
    scores = scores.masked_fill(~causal, -torch.inf)
    return torch.matmul(scores.softmax(-1), v).transpose(1, 2).to(query.dtype)


def native_fa3_attention(
    module, query, key, value, attention_mask=None, *, dropout=0.0, scaling=None,
    sliding_window=None, position_ids=None, is_causal=True, **kwargs,
):
    """HF attention interface: [1, heads, tokens, dim] -> [1, tokens, heads, dim].

    This intentionally rejects packed rows rather than allowing their causal
    histories to mix. Parameters and input tensors retain their graph identities.
    """
    if any(tensor.ndim != 4 for tensor in (query, key, value)):
        raise ValueError("diagnostic FA3 requires [batch, heads, tokens, dim] tensors")
    if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
        raise ValueError("diagnostic FA3 supports one unpacked row only")
    if query.shape[-2] == 0 or key.shape != value.shape or query.shape[-2:] != key.shape[-2:]:
        raise ValueError("diagnostic FA3 requires nonempty self-attention with matching head dimensions")
    if key.shape[1] == 0 or query.shape[1] == 0 or query.shape[1] % key.shape[1]:
        raise ValueError("diagnostic FA3 requires an integral GQA head ratio")
    if any(tensor.device != query.device or tensor.dtype != query.dtype for tensor in (key, value)):
        raise ValueError("diagnostic FA3 requires matching Q/K/V device and dtype")
    if not query.is_floating_point():
        raise ValueError("diagnostic FA3 requires floating point Q/K/V")
    if attention_mask is not None or dropout != 0.0 or sliding_window not in (None, -1) or not is_causal:
        raise ValueError("diagnostic FA3 does not support masks, padding, dropout, or sliding windows")
    for name, option in kwargs.items():
        if option is not None and option is not False:
            raise ValueError(f"unsupported diagnostic FA3 attention option: {name}")
    if position_ids is not None:
        if position_ids.shape != (1, query.shape[-2]) or not bool((position_ids[:, 1:] - position_ids[:, :-1] == 1).all()):
            raise ValueError("diagnostic FA3 rejects packed position resets or position gaps")
    scale = query.shape[-1] ** -0.5 if scaling is None else float(scaling)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("diagnostic FA3 requires a finite positive scale")
    if query.is_cuda:
        if query.dtype not in (torch.float16, torch.bfloat16) or query.shape[-1] % 8 or query.shape[-1] > 256:
            raise ValueError("diagnostic FA3 CUDA requires FP16/BF16 and a head dimension divisible by 8, at most 256")
        q, k, v = (tensor.transpose(1, 2).squeeze(0).contiguous() for tensor in (query, key, value))
        output = _NativeFA3Attention.apply(q, k, v, scale).unsqueeze(0)
    else:
        output = _dense_attention(query, key, value, scale)
    return output, None
