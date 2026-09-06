from contextlib import nullcontext
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from opd_tools import qwen_replay_parity_probe as probe


def test_difference_preserves_nonfinite_and_shape_evidence():
    result = probe.difference(torch.tensor([1., float("nan"), 3.]), torch.tensor([2., 0., 3.]))
    assert result["nonfinite_elements"] == 1
    assert result["max_abs"] == 1 and result["exact_elements"] == 1
    assert not probe.difference(torch.ones(2), torch.ones(3))["shape_matches"]


def test_trace_clones_inplace_native_outputs_and_joins_cached_steps():
    trace = probe.Trace()
    trace.tokens = 2
    first = torch.arange(8.).reshape(2, 4)
    trace.emit("layer.0.norm_in", first)
    first.zero_()  # Native fused kernels may mutate the next residual input.
    trace.tokens = 1
    trace.emit("layer.0.norm_in", torch.full((1, 4), 8.))
    result = trace.finish()["layer.0.norm_in"]
    assert result.shape == (3, 4)
    assert torch.equal(result[:2], torch.arange(8.).reshape(2, 4))
    assert torch.equal(result[-1], torch.full((4,), 8.))


def test_final_logits_report_is_bounded_and_support_is_fixed_to_native():
    native = torch.arange(32.).reshape(1, 32)
    other = native.clone()
    other[0, 31] -= 10
    result = probe.logits_difference(native, other)
    assert len(result["worst"]) == 8
    assert {row["token_id"] for row in result["native_top5_support"]} == set(range(27, 32))
    assert result["worst"][0]["token_id"] == 31
    assert result["max_abs"] == 10


def test_synthetic_inputs_are_deterministic_and_mix_only_eight_known_positions():
    tokenizer = SimpleNamespace(vocab_size=1000, encode=lambda *args, **kwargs: list(range(400)))
    ids, support, probs = probe.synthetic_sequence(tokenizer, 64)
    again = probe.synthetic_sequence(tokenizer, 64)
    assert all(torch.equal(a, b) for a, b in zip((ids, support, probs), again))
    assert ((probs > 0).sum(-1) == 5).nonzero().flatten().tolist() == list(range(55, 63))
    assert torch.equal(probs.sum(-1), torch.ones(64))
    assert support[:, 0].tolist() == ids.tolist()
    assert torch.all((support >= 0) & (support < tokenizer.vocab_size))
    with pytest.raises(ValueError):
        probe.synthetic_sequence(tokenizer, 8192)


@pytest.mark.parametrize("soft", [False, True])
def test_native_cached_forward_forces_the_same_inputs_without_sampling(monkeypatch, soft):
    """Exercise real probe control flow; fake only native/GPU execution APIs."""
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(probe, "native_hooks", lambda *args: nullcontext())
    class Batch:
        def __init__(self, ids):
            self.input_ids = ids
            self.reqs = [SimpleNamespace()]
        def prepare_for_decode(self):
            self.input_ids = self.output_ids
        def get_model_worker_batch(self):
            return self
    monkeypatch.setattr(probe, "make_native_batch", lambda runner, ids: Batch(ids))
    fake_forward = SimpleNamespace(init_new=lambda batch, runner: batch)
    monkeypatch.setitem(sys.modules, "sglang.srt.model_executor.forward_batch_info", SimpleNamespace(ForwardBatch=fake_forward))
    seen = []
    def forward(batch):
        seen.append((batch.input_ids.clone(), None if batch.topk_indices is None else batch.topk_indices.clone()))
        return SimpleNamespace(next_token_logits=torch.zeros(1, 32))
    runner = SimpleNamespace(model=None, forward=forward)
    ids = torch.arange(16)
    support = torch.arange(80).reshape(16, 5) if soft else None
    probs = torch.full((16, 5), .2) if soft else None
    result = probe.native_forward(runner, ids, support, probs, cached=True)
    assert [x[0].tolist() for x in seen] == [list(range(12)), [12], [13], [14], [15]]
    assert result["logits"].shape == (5, 32)
    if soft:
        assert torch.equal(torch.cat([x[1] for x in seen]), support)
    else:
        assert all(x[1] is None for x in seen)


def test_packed_projection_controls_require_identical_pinned_weights():
    def linear(outputs):
        return torch.nn.Linear(3, outputs, bias=False)
    q, k, v, gate, up = [linear(x) for x in (4, 2, 2, 6, 6)]
    hlayer = SimpleNamespace(self_attn=SimpleNamespace(q_proj=q, k_proj=k, v_proj=v), mlp=SimpleNamespace(gate_proj=gate, up_proj=up))
    qkv, gu = linear(8), linear(12)
    with torch.no_grad():
        qkv.weight.copy_(torch.cat([q.weight, k.weight, v.weight]))
        gu.weight.copy_(torch.cat([gate.weight, up.weight]))
    nlayer = SimpleNamespace(self_attn=SimpleNamespace(qkv_proj=qkv), mlp=SimpleNamespace(gate_up_proj=gu))
    native = SimpleNamespace(model=SimpleNamespace(layers=[nlayer]))
    hf = SimpleNamespace(model=SimpleNamespace(layers=[hlayer]))
    x = torch.randn(4, 3)
    trace = {"layer.0.norm_in": x, "layer.0.norm_post": x, "layer.0.qkv": qkv(x), "layer.0.gate_up": gu(x)}
    rows = probe.packed_projection_controls(native, hf, trace)
    assert all(row["native_observed_vs_packed_same_input"]["max_abs"] == 0 for row in rows)
    with torch.no_grad():
        qkv.weight[0, 0] += 1
    with pytest.raises(RuntimeError, match="pinned packed weights differ"):
        probe.packed_projection_controls(native, hf, trace)


def test_native_batch_initializes_request_soft_control_before_tensorization(monkeypatch):
    cleared = []
    class Params:
        def __init__(self, **kwargs):
            self.soft_thinking_mode = None
        def normalize(self, tokenizer):
            self.normalized = True
    def request(**kwargs):
        params = kwargs["sampling_params"]
        assert params.normalized
        assert kwargs["enable_soft_thinking"] is True and kwargs["max_topk"] == 5
        params.soft_thinking_mode = True  # Actual Req invokes post_init_soft_thinking_mode.
        return SimpleNamespace(**kwargs)
    def init_new(**kwargs):
        req = kwargs["reqs"][0]
        assert torch.tensor([req.sampling_params.soft_thinking_mode], dtype=torch.bool).item()
        return SimpleNamespace(prepare_for_extend=lambda: None)
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.schedule_batch", SimpleNamespace(Req=request, ScheduleBatch=SimpleNamespace(init_new=init_new)))
    monkeypatch.setitem(sys.modules, "sglang.srt.sampling.sampling_params", SimpleNamespace(SamplingParams=Params))
    monkeypatch.setitem(sys.modules, "sglang.srt.speculative.spec_info", SimpleNamespace(SpeculativeAlgorithm=SimpleNamespace(NONE=None)))
    runner = SimpleNamespace(model_config=None, req_to_token_pool=SimpleNamespace(clear=lambda: cleared.append("request")),
                             token_to_kv_pool_allocator=SimpleNamespace(clear=lambda: cleared.append("kv")))
    probe.make_native_batch(runner, torch.arange(16))
    assert cleared == ["request", "kv"]


def test_probe_failure_keeps_completed_forward_evidence_and_previous_files(tmp_path, monkeypatch):
    output = tmp_path / "probe.json"
    def failed(args, **kwargs):
        args.output.write_text(json.dumps({"cases": [{"input": "hard_tokens", "measured": True}]}))
        raise RuntimeError("candidate backward failed")
    monkeypatch.setattr(probe, "run", failed)
    argv = ["--assets", str(tmp_path), "--output", str(output)]
    with pytest.raises(RuntimeError, match="candidate backward failed"):
        probe.main(argv)
    result = json.loads(output.read_text())
    assert result["status"] == "failed" and len(result["cases"]) == 1
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        probe.main(argv)
    assert output.read_bytes() == before


def test_stage_differences_separate_shared_prefix_from_decode_rows():
    left = {"layer.0.qkv": torch.zeros(16, 8), "logits": torch.zeros(1, 32)}
    right = {key: value.clone() for key, value in left.items()}
    right["layer.0.qkv"][3, :2] = 1
    right["layer.0.qkv"][13, 0] = 1
    regions = probe.compare_traces(left, right)["stages"]["layer.0.qkv"]["differing_rows"]
    assert regions == {"prefix_elements": 2, "decode_elements_by_row": [0, 1, 0, 0], "first_row_indices": [3, 13]}


def test_projection_shape_control_holds_exact_rows_and_weights_fixed():
    value = torch.arange(48.).reshape(16, 3)
    weight = torch.arange(15.).reshape(5, 3)
    calls = []
    def linear(x, w):
        assert w is weight
        calls.append(x.clone())
        return torch.nn.functional.linear(x, w)
    result = probe.projection_shape_controls(value, weight, linear)
    assert all(row["max_abs"] == 0 for row in result.values())
    assert [len(x) for x in calls] == [16, 12, 1, 1, 1, 1, 16]
    assert torch.equal(torch.cat(calls[1:6]), value)


def test_native_projection_replacement_is_instance_scoped_and_head_uses_it(monkeypatch):
    class Unquantized:
        def apply(self, layer, value, bias=None):
            return torch.nn.functional.linear(value, layer.weight, bias)
    class Native(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4, 3))
            self.quant_method = Unquantized()
            self.logits_processor = SimpleNamespace(
                do_tensor_parallel_all_gather=False, do_tensor_parallel_all_gather_dp_attn=False,
                final_logit_softcapping=None, logit_scale=None, config=SimpleNamespace(vocab_size=4))
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.linear", SimpleNamespace(UnquantizedLinearMethod=Unquantized))
    model, untouched = Native(), Native()
    original = model.quant_method
    parameters = list(model.parameters())
    calls = []
    def linear(value, weight, bias=None):
        calls.append(weight)
        return torch.nn.functional.linear(value, weight, bias)
    assert probe.install_native_probe_linear(model, linear) == 1
    assert model.quant_method is not original
    x = torch.ones(2, 3)
    assert torch.equal(model.quant_method.apply(model, x), original.apply(model, x))
    assert torch.equal(model.logits_processor._get_logits(x, model, None), original.apply(model, x))
    assert len(calls) == 2 and all(w is parameters[0] for w in calls)
    assert "apply" not in untouched.quant_method.__dict__
    model.logits_processor.do_tensor_parallel_all_gather = True
    with pytest.raises(ValueError, match="TP1"):
        probe.install_native_probe_linear(model, linear)


def test_native_fa3_probe_fixes_both_prefill_and_decode_split_counts(monkeypatch):
    calls = []
    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "actual-kernel-result"
    backend = SimpleNamespace(flash_attn_varlen_func=original, flash_attn_with_kvcache=original)
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.attention", SimpleNamespace(flashattention_backend=backend))
    probe.fix_native_fa3_split_count()
    for name in ("flash_attn_varlen_func", "flash_attn_with_kvcache"):
        assert getattr(backend, name)("query", num_splits=0, causal=True) == "actual-kernel-result"
    assert calls == [(('query',), {"num_splits": 1, "causal": True})] * 2
