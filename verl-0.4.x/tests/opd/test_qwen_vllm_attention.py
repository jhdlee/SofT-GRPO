"""Paged KV ownership and instance-hook tests; numerical GPU admission is separate."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.opd import qwen_vllm_attention as attention


class FlashAttentionImpl:
    __module__ = "vllm.v1.attention.backends.flash_attn"

    def __init__(self):
        self.vllm_flash_attn_version = 3
        self.head_size = 128
        self.num_heads, self.num_kv_heads = 2, 1
        self.scale = 128 ** -.5
        self.alibi_slopes = None
        self.sliding_window = (-1, -1)
        self.logits_soft_cap = 0
        self.use_irope = False
        self.kv_cache_dtype = "auto"
        self.stock_calls = []

    def forward(self, layer, query, key, value, kv_cache, metadata, output=None):
        self.stock_calls.append(metadata)
        return output


def inputs():
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(4, 2, 128, generator=generator).bfloat16()
    key = torch.randn(4, 1, 128, generator=generator).bfloat16()
    value = torch.randn(4, 1, 128, generator=generator).bfloat16()
    cache = torch.randn(2, 2, 16, 1, 128, generator=generator).bfloat16()
    output = torch.full_like(query, 999)
    metadata = SimpleNamespace(num_actual_tokens=3, max_query_len=2,
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32), max_seq_len=4,
        seq_lens=torch.tensor([4, 3], dtype=torch.int32),
        block_table=torch.tensor([[0], [1]], dtype=torch.int32),
        slot_mapping=torch.tensor([2, 3, 18], dtype=torch.int64),
        use_cascade=False, common_prefix_len=0, local_attn_metadata=None,
        scheduler_metadata=torch.tensor([0, 2, 4], dtype=torch.int32),
        prefix_scheduler_metadata=None)
    layer = SimpleNamespace(_k_scale=torch.tensor(1.), _v_scale=torch.tensor(1.))
    return layer, query, key, value, cache, metadata, output


@pytest.fixture
def cpu_native(monkeypatch):
    calls = []
    monkeypatch.setattr(attention.importlib.metadata, "version", lambda _: "0.8.5")
    monkeypatch.setattr(attention, "_validate_device", lambda _: None)
    monkeypatch.setattr(attention, "_native_identity", lambda: {
        "extension_path": "/sealed/opd_fa3/_C.so", "extension_sha256": "a" * 64})

    def write_cache(key, value, k_cache, v_cache, slots, dtype, k_scale, v_scale):
        calls.append(("write", slots.clone(), dtype))
        for row, slot in enumerate(slots.tolist()):
            if slot >= 0:
                block, offset = divmod(slot, k_cache.shape[1])
                k_cache[block, offset].copy_(key[row])
                v_cache[block, offset].copy_(value[row])

    def native(**kwargs):
        calls.append(("native", kwargs))
        return torch.full_like(kwargs["q"], 17)

    monkeypatch.setattr(torch.ops._C_cache_ops, "reshape_and_cache_flash", write_cache, raising=False)
    monkeypatch.setattr(attention, "_native_forward", lambda: native)
    return calls


def test_ragged_paged_call_preserves_stock_writes_and_output_buffer(cpu_native):
    impl = FlashAttentionImpl()
    identity = attention.install_qwen_vllm_attention(impl)
    args = inputs(); before = args[4].clone()
    actual = impl.forward(*args)
    assert actual is args[-1]
    assert torch.equal(actual[:3], torch.full_like(actual[:3], 17))
    assert torch.equal(actual[3], torch.full_like(actual[3], 999))
    assert [call[0] for call in cpu_native] == ["write", "native"]
    call = cpu_native[-1][1]
    assert call["q"].shape == (3, 2, 128)
    assert call["cu_seqlens_q"] is args[5].query_start_loc
    assert call["cache_seqlens"] is args[5].seq_lens
    assert call["page_table"] is args[5].block_table
    assert call["scheduler_metadata"] is None and call["num_splits"] == 1
    assert call["causal"] is True and call["window_size"] == (-1, -1)
    assert call["max_seqlen_q"] == 2 and call["softmax_scale"] == impl.scale
    assert "k" not in call and "v" not in call  # Cache must not append twice.
    assert torch.equal(args[4][0, 0, 2:4], args[2][:2])
    assert torch.equal(args[4][1, 1, 2], args[3][2])
    assert torch.equal(args[4][:, 0, 4:], before[:, 0, 4:])
    assert identity["num_splits"] == 1 and identity["cascade"] is False


def test_paged_reference_observes_updated_cache_and_bottom_right_causality(cpu_native, monkeypatch):
    args = inputs()
    before = args[4].clone()

    def dense_paged(**kwargs):
        q = kwargs["q"].double()
        out = torch.empty_like(q)
        offsets = kwargs["cu_seqlens_q"].tolist()
        for sequence, (begin, end) in enumerate(zip(offsets, offsets[1:])):
            length = int(kwargs["cache_seqlens"][sequence])
            pages = kwargs["page_table"][sequence].long()
            k = kwargs["k_cache"][pages].flatten(0, 1)[:length].double().repeat_interleave(2, 1)
            v = kwargs["v_cache"][pages].flatten(0, 1)[:length].double().repeat_interleave(2, 1)
            for row in range(begin, end):
                visible = length - (end - row - 1)
                scores = torch.einsum("hd,nhd->hn", q[row], k[:visible]) * kwargs["softmax_scale"]
                out[row] = torch.einsum("hn,nhd->hd", scores.softmax(-1), v[:visible])
        return out.bfloat16()

    monkeypatch.setattr(attention, "_native_forward", lambda: dense_paged)
    impl = FlashAttentionImpl(); attention.install_qwen_vllm_attention(impl)
    actual = impl.forward(*args)
    # Independent first-query reference includes old positions 0/1 and new2,
    # excluding the future key3 even though it was already written to the cache.
    keys = torch.cat([before[0, 0, :2], args[2][:1]], dim=0).double().repeat_interleave(2, 1)
    values = torch.cat([before[1, 0, :2], args[3][:1]], dim=0).double().repeat_interleave(2, 1)
    scores = torch.einsum("hd,nhd->hn", args[1][0].double(), keys) * impl.scale
    expected = torch.einsum("hn,nhd->hd", scores.softmax(-1), values).bfloat16()
    assert torch.equal(actual[0], expected)


def test_accepts_real_builder_numpy_integer_lengths(cpu_native):
    impl = FlashAttentionImpl(); attention.install_qwen_vllm_attention(impl)
    args = inputs()
    args[5].num_actual_tokens = np.int64(3)
    args[5].max_query_len = np.int64(2)
    args[5].max_seq_len = np.int32(4)
    impl.forward(*args)
    assert type(cpu_native[-1][1]["max_seqlen_q"]) is int


@pytest.mark.parametrize("field,value", [
    ("vllm_flash_attn_version", 2), ("head_size", 64), ("num_kv_heads", 3),
    ("alibi_slopes", [1.]), ("sliding_window", (127, 0)), ("logits_soft_cap", 1.),
    ("use_irope", True), ("kv_cache_dtype", "fp8"), ("scale", float("nan")),
])
def test_unsupported_impl_is_rejected_without_changing_forward(cpu_native, field, value):
    impl = FlashAttentionImpl(); original = impl.forward
    setattr(impl, field, value)
    with pytest.raises(ValueError): attention.install_qwen_vllm_attention(impl)
    assert impl.forward == original and not hasattr(impl, attention._STATE)


@pytest.mark.parametrize("fault", ["version", "class_module", "backend", "unbounded_telemetry"])
def test_pinned_runtime_and_opt_in_restrictions(cpu_native, monkeypatch, fault):
    impl = FlashAttentionImpl(); kwargs = {}
    if fault == "version": monkeypatch.setattr(attention.importlib.metadata, "version", lambda _: "0.8.6")
    if fault == "class_module": monkeypatch.setattr(FlashAttentionImpl, "__module__", "other")
    if fault == "backend": kwargs["backend"] = "native_fa3_v1"
    if fault == "unbounded_telemetry": kwargs["telemetry_limit"] = 10000
    with pytest.raises(ValueError): attention.install_qwen_vllm_attention(impl, **kwargs)
    assert not hasattr(impl, attention._STATE)


@pytest.mark.parametrize("fault", ["cascade", "prefix", "local", "slots", "packed", "table", "dtype", "output", "cache", "n"])
def test_bad_call_fails_before_kv_mutation(cpu_native, fault):
    impl = FlashAttentionImpl(); attention.install_qwen_vllm_attention(impl)
    args = list(inputs()); metadata = args[5]
    if fault == "cascade": metadata.use_cascade = True
    if fault == "prefix": metadata.common_prefix_len = 256
    if fault == "local": metadata.local_attn_metadata = object()
    if fault == "slots": metadata.slot_mapping = metadata.slot_mapping.int()
    if fault == "packed": metadata.query_start_loc = metadata.query_start_loc[:2]
    if fault == "table": metadata.block_table = metadata.block_table.long()
    if fault == "dtype": args[1] = args[1].float()
    if fault == "output": args[-1] = None
    if fault == "cache": args[4] = args[4][:, :, :15]
    if fault == "n": metadata.num_actual_tokens = 5
    with pytest.raises(ValueError): impl.forward(*args)
    assert cpu_native == []


def test_profiling_and_zero_tokens_do_not_write_or_launch(cpu_native):
    impl = FlashAttentionImpl(); attention.install_qwen_vllm_attention(impl)
    args = list(inputs()); metadata = args[5]; args[5] = None
    assert impl.forward(*args) is args[-1]
    metadata.num_actual_tokens = 0; metadata.slot_mapping = torch.empty(0, dtype=torch.int64)
    args[5] = metadata
    assert impl.forward(*args) is args[-1]
    assert cpu_native == []


def test_native_failure_propagates_without_stock_fallback(cpu_native, monkeypatch):
    def fail(**kwargs): raise RuntimeError("native engine failure")
    monkeypatch.setattr(attention, "_native_forward", lambda: fail)
    impl = FlashAttentionImpl(); attention.install_qwen_vllm_attention(impl)
    args = inputs()
    with pytest.raises(RuntimeError, match="native engine failure"): impl.forward(*args)
    assert len(cpu_native) == 1 and cpu_native[0][0] == "write"
    assert impl.stock_calls == []
    assert torch.equal(args[-1], torch.full_like(args[-1], 999))


def test_baseline_telemetry_transition_reset_and_idempotence(cpu_native):
    impl = FlashAttentionImpl(); args = inputs()
    attention.install_qwen_vllm_attention_telemetry(impl)
    impl.forward(*args)
    baseline = attention.qwen_vllm_attention_telemetry(impl, reset=True)[0]
    assert baseline["samples"][0]["scheduler_metadata_splits"] == [2, 4]
    assert baseline["samples"][0]["query_lengths"] == [2, 1]
    assert baseline["samples"][0]["key_lengths"] == [4, 3]
    assert attention.qwen_vllm_attention_telemetry(impl)[0]["calls"] == 0
    identity = attention.install_qwen_vllm_attention(impl)
    installed = impl.forward
    assert attention.install_qwen_vllm_attention(impl) == identity
    assert impl.forward is installed
    identity["num_splits"] = 999
    impl.forward(*args)
    candidate = attention.qwen_vllm_attention_telemetry(impl)[0]
    assert candidate["identity"]["num_splits"] == 1
    assert candidate["calls"] == 1 and len(impl.stock_calls) == 1
    with pytest.raises(ValueError, match="revert"): attention.install_qwen_vllm_attention_telemetry(impl)


def test_telemetry_is_bounded_but_tracks_long_decode_and_cascade(cpu_native):
    impl = FlashAttentionImpl(); args = inputs(); metadata = args[5]
    attention.install_qwen_vllm_attention_telemetry(impl, telemetry_limit=8)
    for length in range(1, 9000):
        metadata.max_seq_len = length
        metadata.use_cascade = length == 8
        impl.forward(*args)
    result = attention.qwen_vllm_attention_telemetry(impl)[0]
    assert result["calls"] == 8999 and result["max_key_length"] == 8999
    assert result["cascade_calls"] == 1 and len(result["samples"]) == 8
    result["samples"].clear()
    assert len(attention.qwen_vllm_attention_telemetry(impl)[0]["samples"]) == 8


def test_model_inventory_validated_before_any_hook_and_tamper_detected(cpu_native):
    impls = [FlashAttentionImpl(), FlashAttentionImpl()]
    model = SimpleNamespace(model=SimpleNamespace(layers=[
        SimpleNamespace(self_attn=SimpleNamespace(attn=SimpleNamespace(impl=impl))) for impl in impls]))
    impls[1].use_irope = True
    with pytest.raises(ValueError): attention.install_qwen_vllm_attention(model)
    assert all(not hasattr(impl, attention._STATE) for impl in impls)
    impls[1].use_irope = False
    attention.install_qwen_vllm_attention(model)
    assert len(attention.qwen_vllm_attention_telemetry(model)) == 2
    impls[0].forward = lambda *args: None
    with pytest.raises(ValueError, match="hook changed"): attention.install_qwen_vllm_attention(model)


def test_late_install_failure_restores_preexisting_stock_telemetry(cpu_native, monkeypatch):
    impls = [FlashAttentionImpl(), FlashAttentionImpl()]
    model = SimpleNamespace(model=SimpleNamespace(layers=[
        SimpleNamespace(self_attn=SimpleNamespace(attn=SimpleNamespace(impl=impl))) for impl in impls]))
    attention.install_qwen_vllm_attention_telemetry(model)
    original = [impl.forward for impl in impls]
    method_type = attention.types.MethodType
    def fail_second(function, instance):
        if instance is impls[1]: raise RuntimeError("injected installation failure")
        return method_type(function, instance)
    monkeypatch.setattr(attention.types, "MethodType", fail_second)
    with pytest.raises(RuntimeError, match="injected"): attention.install_qwen_vllm_attention(model)
    assert all(impl.forward is before for impl, before in zip(impls, original))
    assert all(row["identity"]["attention"] == "stock_vllm_fa3"
               for row in attention.qwen_vllm_attention_telemetry(model))
