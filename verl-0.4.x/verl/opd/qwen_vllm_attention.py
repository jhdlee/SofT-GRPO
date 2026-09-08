"""Opt-in native FA3 attention for the guarded Qwen3 vLLM replay path.

The owner validates Qwen3/TP1/eager execution and holds an idle frozen-batch
guard before installation. vLLM still owns page allocation and KV writes. Its
scheduler metadata is never passed across the distinct native extension ABI.
Numerical acceptance remains the ordinary, unchanged rollout/replay check.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import importlib.metadata
import math
from numbers import Integral
from pathlib import Path
import types

import torch

RECIPE = "qwen3_vllm_native_fa3_attention_v1"
_STATE = "_opd_qwen_vllm_attention_state"


def _implementations(target):
    if hasattr(target, "model") and hasattr(target.model, "layers"):
        return [layer.self_attn.attn.impl for layer in target.model.layers]
    return [target]


def _validate_impl(impl):
    if (importlib.metadata.version("vllm") != "0.8.5"
            or type(impl).__name__ != "FlashAttentionImpl"
            or type(impl).__module__ != "vllm.v1.attention.backends.flash_attn"
            or impl.vllm_flash_attn_version != 3):
        raise ValueError("native vLLM attention requires the pinned vLLM 0.8.5 V1 FA3 implementation")
    if (impl.head_size != 128 or impl.num_kv_heads <= 0
            or impl.num_heads % impl.num_kv_heads != 0
            or impl.alibi_slopes is not None or tuple(impl.sliding_window) != (-1, -1)
            or impl.logits_soft_cap != 0 or impl.use_irope
            or impl.kv_cache_dtype not in ("auto", "bfloat16")
            or not math.isfinite(impl.scale) or impl.scale <= 0):
        raise ValueError("native vLLM attention requires unquantized full-context head128 GQA")


def _native_forward():
    from opd_fa3 import flash_attn_with_kvcache
    return flash_attn_with_kvcache


@lru_cache(maxsize=1)
def _native_identity():
    from opd_fa3 import _C
    path = Path(_C.__file__).resolve()
    return {"extension_path": str(path),
            "extension_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _validate_device(device):
    if (device.type != "cuda" or torch.version.hip is not None
            or torch.cuda.get_device_capability(device) != (9, 0)):
        raise ValueError("native vLLM attention requires NVIDIA Hopper")


def _new_telemetry(limit):
    return {"calls": 0, "profiling_calls": 0, "cascade_calls": 0,
            "max_query_tokens": 0, "max_query_length": 0, "max_key_length": 0,
            "sample_limit": limit, "samples": []}


def _observe(state, metadata):
    record = state["telemetry"]
    if metadata is None:
        record["profiling_calls"] += 1
        return
    record["calls"] += 1
    call = record["calls"]
    cascade = bool(metadata.use_cascade)
    previous_max = record["max_key_length"]
    record["cascade_calls"] += int(cascade)
    record["max_query_tokens"] = max(record["max_query_tokens"], int(metadata.num_actual_tokens))
    record["max_query_length"] = max(record["max_query_length"], int(metadata.max_query_len))
    record["max_key_length"] = max(previous_max, int(metadata.max_seq_len))
    # Copy small metadata only for bounded samples, including long decode. The
    # integer maxima/counts above cover every call without CUDA synchronization.
    sample = (call <= 2 or call & (call - 1) == 0
              or int(metadata.max_seq_len) // 1024 > previous_max // 1024
              or cascade and record["cascade_calls"] == 1)
    if not sample or len(record["samples"]) >= record["sample_limit"]:
        return
    offsets = metadata.query_start_loc.detach().cpu().tolist()
    lengths = metadata.seq_lens.detach().cpu().tolist()
    row = {"call": call, "mode": state["mode"], "use_cascade": cascade,
           "common_prefix_len": int(metadata.common_prefix_len),
           "query_tokens": int(metadata.num_actual_tokens),
           "query_lengths": [b - a for a, b in zip(offsets, offsets[1:])],
           "key_lengths": lengths, "max_query_length": int(metadata.max_query_len),
           "max_key_length": int(metadata.max_seq_len)}
    for name in ("scheduler_metadata", "prefix_scheduler_metadata"):
        tensor = getattr(metadata, name, None)
        raw = None if tensor is None else tensor.detach().cpu().flatten().tolist()
        row[name] = raw
        # The pinned vLLM extension uses one semaphore followed by B split
        # counts. Preserve the raw values too; native FA3 uses a different ABI.
        batch = 1 if name.startswith("prefix") else len(lengths)
        row[name + "_splits"] = raw[1:] if raw is not None and len(raw) == batch + 1 else None
    record["samples"].append(row)


def _validate_call(impl, query, key, value, cache, metadata, output):
    if output is None:
        raise ValueError("native vLLM attention requires the caller output buffer")
    if metadata is None:
        return
    if (metadata.use_cascade or metadata.common_prefix_len != 0
            or getattr(metadata, "local_attn_metadata", None) is not None):
        raise ValueError("native vLLM attention requires cascade and local attention disabled")
    if (query.ndim != 3 or tuple(query.shape[1:]) != (impl.num_heads, 128)
            or key.ndim != 3 or value.shape != key.shape
            or tuple(key.shape[1:]) != (impl.num_kv_heads, 128)
            or key.shape[0] != query.shape[0]
            or output.shape != query.shape
            or cache.ndim != 5 or cache.shape[0] != 2
            or tuple(cache.shape[3:]) != (impl.num_kv_heads, 128)
            or cache.shape[2] % 16 != 0):
        raise ValueError("native vLLM attention QKV/output/paged-cache shape mismatch")
    for tensor in (query, key, value, cache, output):
        if (tensor.dtype != torch.bfloat16 or tensor.device != query.device
                or tensor.stride(-1) != 1):
            raise ValueError("native vLLM attention requires matching BF16 tensors with contiguous head dimension")
    n = metadata.num_actual_tokens
    batch = metadata.seq_lens.numel()
    if (not isinstance(n, Integral) or isinstance(n, bool) or not 0 <= n <= query.shape[0]
            or metadata.seq_lens.ndim != 1
            or metadata.query_start_loc.shape != (batch + 1,)
            or metadata.block_table.ndim != 2 or metadata.block_table.shape[0] != batch
            or metadata.slot_mapping.shape != (n,)
            or not isinstance(metadata.max_query_len, Integral) or metadata.max_query_len < 0
            or not isinstance(metadata.max_seq_len, Integral) or metadata.max_seq_len < metadata.max_query_len):
        raise ValueError("native vLLM attention packed sequence metadata mismatch")
    for tensor in (metadata.query_start_loc, metadata.seq_lens, metadata.block_table):
        if tensor.dtype != torch.int32 or tensor.device != query.device or tensor.stride(-1) != 1:
            raise ValueError("native vLLM attention requires device-local int32 packed metadata")
    if metadata.slot_mapping.dtype != torch.int64 or metadata.slot_mapping.device != query.device:
        raise ValueError("native vLLM attention requires device-local int64 cache slots")


def _forward(impl, layer, query, key, value, kv_cache, attn_metadata, output=None):
    state = getattr(impl, _STATE)
    if state["mode"] == "stock_vllm_fa3":
        _observe(state, attn_metadata)
        return state["original"](layer, query, key, value, kv_cache, attn_metadata, output)
    _validate_call(impl, query, key, value, kv_cache, attn_metadata, output)
    _observe(state, attn_metadata)
    if attn_metadata is None or attn_metadata.num_actual_tokens == 0:
        return output
    if state["device"] is None:
        _validate_device(query.device)
        state["device"] = query.device
    elif state["device"] != query.device:
        raise ValueError("native vLLM attention device changed after installation")
    key_cache, value_cache = kv_cache.unbind(0)
    # Keep the pinned vLLM cache operation and full padded inputs exactly. The
    # unpadded slot_mapping bounds writes; the native call must not append again.
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key, value, key_cache, value_cache, attn_metadata.slot_mapping,
        impl.kv_cache_dtype, layer._k_scale, layer._v_scale)
    n = int(attn_metadata.num_actual_tokens)
    result = state["native"](
        q=query[:n], k_cache=key_cache, v_cache=value_cache,
        cache_seqlens=attn_metadata.seq_lens,
        cu_seqlens_q=attn_metadata.query_start_loc,
        max_seqlen_q=int(attn_metadata.max_query_len),
        page_table=attn_metadata.block_table, softmax_scale=impl.scale,
        causal=True, window_size=(-1, -1), softcap=0.,
        scheduler_metadata=None, num_splits=1)
    if not isinstance(result, torch.Tensor) or result.shape != output[:n].shape or result.dtype != output.dtype or result.device != output.device:
        raise ValueError("native vLLM attention returned an invalid output")
    output[:n].copy_(result)
    return output


def _install(target, *, mode, telemetry_limit, backend):
    if backend != "native_fa3_v2" or type(telemetry_limit) is not int or not 0 <= telemetry_limit <= 64:
        raise ValueError("native vLLM attention requires native_fa3_v2 and a bounded telemetry limit")
    implementations = _implementations(target)
    if not implementations or len({id(impl) for impl in implementations}) != len(implementations):
        raise ValueError("native vLLM attention implementation inventory is empty or shared")
    identity = {"recipe": RECIPE, "backend": backend, "attention": mode,
                "vllm_version": "0.8.5",
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    native = None
    if mode == "opd_fa3_kvcache_single_split":
        native = _native_forward()
        identity.update(_native_identity(), num_splits=1, scheduler_metadata=None, cascade=False)
    for impl in implementations:
        _validate_impl(impl)
        previous = getattr(impl, _STATE, None)
        if previous is not None:
            if impl.forward is not previous["hook"]:
                raise ValueError("native vLLM attention hook changed after installation")
            if previous["mode"] == mode and previous["identity"] != identity:
                raise ValueError("native vLLM attention identity changed after installation")
            if previous["mode"] != mode and mode == "stock_vllm_fa3":
                raise ValueError("native vLLM attention cannot silently revert to stock attention")
    changes = []
    try:
        for impl in implementations:
            previous = getattr(impl, _STATE, None)
            if previous is not None and previous["mode"] == mode:
                continue
            changes.append((impl, dict(impl.__dict__)))
            state = {"mode": mode, "identity": copy.deepcopy(identity), "native": native,
                     "original": impl.forward if previous is None else previous["original"],
                     "telemetry": _new_telemetry(telemetry_limit if impl is implementations[0] else 0),
                     "device": None}
            hook = types.MethodType(_forward, impl)
            state["hook"] = hook
            setattr(impl, _STATE, state)
            impl.forward = hook
    except BaseException:
        for impl, before in reversed(changes):
            impl.__dict__.clear()
            impl.__dict__.update(before)
        raise
    return copy.deepcopy(identity)


def install_qwen_vllm_attention(target, *, backend="native_fa3_v2", telemetry_limit=32):
    """Install on one impl (training) or all model impls (diagnostic).

    Engine construction must disable cascade attention. Errors after cache writes
    propagate to the owner's collective failure/engine-poison guard; no fallback.
    """
    return _install(target, mode="opd_fa3_kvcache_single_split", backend=backend,
                    telemetry_limit=telemetry_limit)


def install_qwen_vllm_attention_telemetry(target, *, telemetry_limit=32):
    """Observe the unmodified stock call, for a separate diagnostic baseline."""
    return _install(target, mode="stock_vllm_fa3", backend="native_fa3_v2",
                    telemetry_limit=telemetry_limit)


def qwen_vllm_attention_telemetry(target, *, reset=False):
    """Copy per-layer measurements without exposing live state to the caller."""
    rows = []
    for index, impl in enumerate(_implementations(target)):
        state = getattr(impl, _STATE, None)
        if state is None or impl.forward is not state["hook"]:
            raise ValueError("native vLLM attention telemetry hook is absent or changed")
        rows.append({"layer": index, "identity": copy.deepcopy(state["identity"]),
                     **copy.deepcopy(state["telemetry"])})
        if reset:
            state["telemetry"] = _new_telemetry(state["telemetry"]["sample_limit"])
    return rows
