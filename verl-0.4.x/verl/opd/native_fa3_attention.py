"""Opt-in FA3 forward with an existing FA2 mathematical backward.

Forward uses sgl-kernel 0.1.1's ``flash_attn_varlen_func(num_splits=1)``.
That build exposes only forward (its CMake defines DISABLE_BACKWARD). Backward
uses flash-attn 2.7.3's ``_flash_attn_varlen_backward`` with the *actual* saved
FA3 output and natural-log LSE, both in FA2's unpadded varlen layout. There is
no second attention forward and no detached substitute output.

Both kernels implement the same causal softmax attention. Their floating-point
rounding differs: this is a mathematical attention gradient, not a claim to
differentiate FA3's discrete rounding exactly or to establish training parity.
Packed self-attention uses explicit, validated sequence boundaries. There is
no dropout/window/cache/custom mask support, and no automatic installation.

Source contracts: vendored sgl-kernel/python/sgl_kernel/flash_attn.py:209;
https://github.com/Dao-AILab/flash-attention/blob/v2.7.3/flash_attn/flash_attn_interface.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral

import torch
from torch.autograd.function import once_differentiable


def _version(tensor):
    return None if tensor is None or tensor.is_inference() else tensor._version


@dataclass(frozen=True, eq=False)
class FA3AttentionLayout:
    """Validated metadata for one model forward; never cache across batches.

    Construct with prepare_fa3_attention_layout. Tensor identity/version checks
    let each layer reuse validation without reading CUDA tensor contents.
    """

    cu_seqlens: torch.Tensor
    max_seqlen: int
    total_tokens: int
    segments: tuple[tuple[int, int], ...]
    position_ids: torch.Tensor | None = field(repr=False)
    attention_mask: torch.Tensor | None = field(repr=False)
    _versions: tuple = field(repr=False)


def prepare_fa3_attention_layout(
    position_ids, *, total_tokens, device, attention_mask=None,
    opd_cu_seqlens=None, opd_max_seqlen=None,
):
    """Validate once per model forward, including any CUDA-to-host reads.

    Explicit cumulative lengths describe [1,total_tokens] packed rows. Every
    segment must start at position zero and increment by one. Without metadata,
    only one contiguous row is admitted (a nonzero initial position is allowed).
    A supplied mask must be [1,total_tokens] and entirely one; padding must have
    been removed upstream. The cumulative-length buffer is privately copied.
    """
    if isinstance(total_tokens, bool) or not isinstance(total_tokens, Integral) or not 0 < total_tokens < 2**31:
        raise ValueError("FA3 requires a positive int32-sized token count")
    total_tokens = int(total_tokens)
    device = torch.device(device)
    explicit = opd_cu_seqlens is not None
    if explicit != (opd_max_seqlen is not None):
        raise ValueError("FA3 packed cumulative lengths and maximum length must be supplied together")
    if attention_mask is not None:
        if (not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != (1, total_tokens)
                or attention_mask.device != device or not bool((attention_mask.detach().cpu() == 1).all())):
            raise ValueError("FA3 supports only an all-ones [1,total_tokens] mask; remove padding first")
    if explicit:
        if (not isinstance(opd_cu_seqlens, torch.Tensor) or opd_cu_seqlens.dtype != torch.int32
                or opd_cu_seqlens.device != device or opd_cu_seqlens.ndim != 1
                or not opd_cu_seqlens.is_contiguous() or opd_cu_seqlens.numel() < 2):
            raise ValueError("FA3 cumulative lengths must be contiguous int32 on the Q/K/V device")
        if isinstance(opd_max_seqlen, bool) or not isinstance(opd_max_seqlen, Integral):
            raise ValueError("FA3 maximum sequence length must be an integer")
        boundaries = tuple(opd_cu_seqlens.detach().cpu().tolist())
        if boundaries[0] != 0 or boundaries[-1] != total_tokens or any(b >= e for b, e in zip(boundaries, boundaries[1:])):
            raise ValueError("FA3 boundaries must start at zero, end at total_tokens, and contain nonempty rows")
        maximum = max(e - b for b, e in zip(boundaries, boundaries[1:]))
        if int(opd_max_seqlen) != maximum:
            raise ValueError("FA3 maximum sequence length does not match cumulative lengths")
        if position_ids is None:
            raise ValueError("FA3 packed rows require position IDs to validate reset alignment")
    else:
        boundaries, maximum = (0, total_tokens), total_tokens
    if position_ids is not None:
        if (not isinstance(position_ids, torch.Tensor) or position_ids.shape != (1, total_tokens)
                or position_ids.device != device or position_ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError("FA3 position IDs must be integer [1,total_tokens] on the Q/K/V device")
        positions = position_ids.detach().cpu()[0]
        for begin, end in zip(boundaries, boundaries[1:]):
            segment = positions[begin:end]
            if (int(segment[0]) < 0 or (explicit and int(segment[0]) != 0)
                    or not bool((segment[1:] - segment[:-1] == 1).all())):
                raise ValueError("FA3 rejects packed position resets or gaps inconsistent with cumulative lengths")
    cumulative = (opd_cu_seqlens.detach().clone() if explicit
                  else torch.tensor(boundaries, dtype=torch.int32, device=device))
    return FA3AttentionLayout(
        cumulative, maximum, total_tokens, tuple(zip(boundaries, boundaries[1:])),
        position_ids, attention_mask, tuple(_version(t) for t in (cumulative, position_ids, attention_mask)),
    )


def _check_prepared_layout(layout, query, position_ids, attention_mask):
    if not isinstance(layout, FA3AttentionLayout):
        raise ValueError("FA3 requires a layout from prepare_fa3_attention_layout")
    if layout.total_tokens != query.shape[-2] or layout.cu_seqlens.device != query.device:
        raise ValueError("FA3 layout does not match the input token count or device")
    if position_ids is not None and position_ids is not layout.position_ids:
        raise ValueError("FA3 layout belongs to different position IDs; prepare it once for this forward")
    if attention_mask is not None and attention_mask is not layout.attention_mask:
        raise ValueError("FA3 layout belongs to a different attention mask")
    if tuple(_version(t) for t in (layout.cu_seqlens, layout.position_ids, layout.attention_mask)) != layout._versions:
        raise ValueError("FA3 validated layout metadata was modified in place")


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
    def forward(ctx, query, key, value, scale, layout=None):
        length = query.shape[0] if layout is None else layout.max_seqlen
        cumulative = (torch.tensor([0, length], dtype=torch.int32, device=query.device)
                      if layout is None else layout.cu_seqlens)
        output, lse = _fa3_forward(query, key, value, cumulative, length, scale)
        if output.shape != query.shape or output.dtype != query.dtype or output.device != query.device:
            raise RuntimeError("FA3 returned an incompatible attention output")
        if lse.shape != (query.shape[1], query.shape[0]) or lse.dtype != torch.float32 or lse.device != query.device:
            raise RuntimeError("FA3 LSE must be float32 [query_heads, total_query_tokens]")
        ctx.save_for_backward(query, key, value, output, lse.contiguous(), cumulative)
        ctx.length, ctx.scale = length, scale
        ctx.has_layout_argument = len(ctx.needs_input_grad) == 5
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        query, key, value, output, lse, cumulative = ctx.saved_tensors
        dq, dk, dv = _fa2_backward(
            gradient, query, key, value, output, lse, cumulative, ctx.length, ctx.scale,
        )
        gradients = (dq, dk, dv, None)
        return gradients + (None,) if ctx.has_layout_argument else gradients


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


def _native_attention(
    module, query, key, value, attention_mask=None, *, dropout=0.0, scaling=None,
    sliding_window=None, position_ids=None, is_causal=True,
    opd_cu_seqlens=None, opd_max_seqlen=None, opd_attention_layout=None,
    _autograd_function=_NativeFA3Attention, **kwargs,
):
    """HF attention interface: [1, heads, tokens, dim] -> [1, tokens, heads, dim].

    Packed rows require explicit boundaries. Pass a layout prepared once in the
    model forward to avoid reading CUDA metadata in each attention layer.
    """
    if any(tensor.ndim != 4 for tensor in (query, key, value)):
        raise ValueError("diagnostic FA3 requires [batch, heads, tokens, dim] tensors")
    if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
        raise ValueError("FA3 requires one unpacked row or explicitly packed [1,total_tokens] rows")
    if query.shape[-2] == 0 or key.shape != value.shape or query.shape[-2:] != key.shape[-2:]:
        raise ValueError("diagnostic FA3 requires nonempty self-attention with matching head dimensions")
    if key.shape[1] == 0 or query.shape[1] == 0 or query.shape[1] % key.shape[1]:
        raise ValueError("diagnostic FA3 requires an integral GQA head ratio")
    if any(tensor.device != query.device or tensor.dtype != query.dtype for tensor in (key, value)):
        raise ValueError("diagnostic FA3 requires matching Q/K/V device and dtype")
    if not query.is_floating_point():
        raise ValueError("diagnostic FA3 requires floating point Q/K/V")
    if dropout != 0.0 or sliding_window not in (None, -1) or not is_causal:
        raise ValueError("FA3 does not support dropout, noncausal attention, or sliding windows")
    for name, option in kwargs.items():
        if option is not None and option is not False:
            raise ValueError(f"unsupported diagnostic FA3 attention option: {name}")
    if opd_attention_layout is None:
        layout = prepare_fa3_attention_layout(
            position_ids, total_tokens=query.shape[-2], device=query.device, attention_mask=attention_mask,
            opd_cu_seqlens=opd_cu_seqlens, opd_max_seqlen=opd_max_seqlen,
        )
    else:
        if opd_cu_seqlens is not None or opd_max_seqlen is not None:
            raise ValueError("supply a prepared FA3 layout or raw cumulative lengths, not both")
        layout = opd_attention_layout
        _check_prepared_layout(layout, query, position_ids, attention_mask)
    scale = query.shape[-1] ** -0.5 if scaling is None else float(scaling)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("diagnostic FA3 requires a finite positive scale")
    if query.is_cuda:
        if query.dtype not in (torch.float16, torch.bfloat16) or query.shape[-1] % 8 or query.shape[-1] > 256:
            raise ValueError("diagnostic FA3 CUDA requires FP16/BF16 and a head dimension divisible by 8, at most 256")
        q, k, v = (tensor.transpose(1, 2).squeeze(0).contiguous() for tensor in (query, key, value))
        output = _autograd_function.apply(q, k, v, scale, layout).unsqueeze(0)
    else:
        output = torch.cat([
            _dense_attention(query[:, :, begin:end], key[:, :, begin:end], value[:, :, begin:end], scale)
            for begin, end in layout.segments
        ], dim=1)
    return output, None


def _fa3_v2_forward(query, key, value, cumulative, length, scale):
    from opd_fa3 import flash_attn_varlen_func

    output, lse, *_ = flash_attn_varlen_func(
        query, key, value, cumulative, cumulative, length, length,
        softmax_scale=scale, causal=True, num_splits=1, return_softmax_lse=True,
    )
    return output, lse


def _fa3_v2_backward(gradient, query, key, value, output, lse, cumulative, length, scale):
    from opd_fa3 import flash_attn_varlen_backward

    return flash_attn_varlen_backward(
        gradient.contiguous(), query, key, value, output, lse,
        cumulative, cumulative, length, length,
        softmax_scale=scale, causal=True, deterministic=True,
    )


class _NativeFA3AttentionV2(torch.autograd.Function):
    """Pinned native FA3 in both directions; never dispatches to FA2."""

    @staticmethod
    def forward(ctx, query, key, value, scale, layout):
        output, lse = _fa3_v2_forward(
            query, key, value, layout.cu_seqlens, layout.max_seqlen, scale,
        )
        if output.shape != query.shape or output.dtype != query.dtype or output.device != query.device:
            raise RuntimeError("FA3 v2 returned an incompatible attention output")
        if lse.shape != (query.shape[1], query.shape[0]) or lse.dtype != torch.float32 or lse.device != query.device:
            raise RuntimeError("FA3 v2 LSE must be float32 [query_heads, total_query_tokens]")
        ctx.save_for_backward(query, key, value, output, lse.contiguous(), layout.cu_seqlens)
        ctx.length, ctx.scale = layout.max_seqlen, scale
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        query, key, value, output, lse, cumulative = ctx.saved_tensors
        dq, dk, dv = _fa3_v2_backward(
            gradient, query, key, value, output, lse, cumulative, ctx.length, ctx.scale,
        )
        for name, actual, expected in (("dq", dq, query), ("dk", dk, key), ("dv", dv, value)):
            if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.device != expected.device:
                raise RuntimeError(f"FA3 v2 returned incompatible {name}")
        return dq, dk, dv, None, None


def native_fa3_attention(
    module, query, key, value, attention_mask=None, *, dropout=0.0, scaling=None,
    sliding_window=None, position_ids=None, is_causal=True,
    opd_cu_seqlens=None, opd_max_seqlen=None, opd_attention_layout=None, **kwargs,
):
    """Existing v1: sgl-kernel FA3 forward and flash-attn FA2 backward."""
    if "_autograd_function" in kwargs:
        raise ValueError("unsupported diagnostic FA3 attention option: _autograd_function")
    return _native_attention(
        module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
        sliding_window=sliding_window, position_ids=position_ids, is_causal=is_causal,
        opd_cu_seqlens=opd_cu_seqlens, opd_max_seqlen=opd_max_seqlen,
        opd_attention_layout=opd_attention_layout, **kwargs,
    )


def native_fa3_attention_v2(
    module, query, key, value, attention_mask=None, *, dropout=0.0, scaling=None,
    sliding_window=None, position_ids=None, is_causal=True,
    opd_cu_seqlens=None, opd_max_seqlen=None, opd_attention_layout=None, **kwargs,
):
    """Qwen3/H100 BF16 FA3 forward and native deterministic FA3 backward.

    CPU is a mathematical reference only. CUDA requires the separately built,
    source-pinned opd-fa3 extension; there is no older-kernel fallback.
    """
    if "_autograd_function" in kwargs:
        raise ValueError("unsupported diagnostic FA3 attention option: _autograd_function")
    if query.is_cuda:
        if query.dtype != torch.bfloat16 or query.shape[-1] != 128:
            raise ValueError("native_fa3_v2 requires BF16 with head dimension 128")
        if torch.cuda.get_device_capability(query.device) != (9, 0):
            raise ValueError("native_fa3_v2 requires Hopper SM90")
    return _native_attention(
        module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
        sliding_window=sliding_window, position_ids=position_ids, is_causal=is_causal,
        opd_cu_seqlens=opd_cu_seqlens, opd_max_seqlen=opd_max_seqlen,
        opd_attention_layout=opd_attention_layout, _autograd_function=_NativeFA3AttentionV2, **kwargs,
    )
