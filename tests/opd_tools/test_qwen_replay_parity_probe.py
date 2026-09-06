from contextlib import contextmanager, nullcontext
import importlib
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


def test_selective_trace_captures_global_last_four_and_keeps_one_logit_row_per_forward():
    trace = probe.Trace(total_tokens=8192, capture_last=4)
    trace.tokens = 8188
    trace.emit("embedding", torch.ones(8188, 8))
    trace.emit("logits", torch.ones(8188, 16))
    assert "embedding" not in trace.values
    assert trace.values["logits"][0].shape == (1, 16)
    for position in range(8188, 8192):
        trace.start, trace.tokens = position, 1
        value = torch.full((1, 8), float(position))
        trace.emit("embedding", value)
        value.zero_()
        trace.emit("logits", torch.ones(1, 16))
    result = trace.finish()
    assert result["embedding"][:, 0].tolist() == list(range(8188, 8192))
    assert result["logits"].shape == (5, 16)
    full = probe.Trace(total_tokens=8192, capture_last=4)
    full.tokens = 8192
    full.emit("embedding", torch.arange(8192.)[:, None].expand(-1, 8))
    assert torch.equal(full.finish()["embedding"], result["embedding"])
    assert sum(tensor.numel() for values in trace.values.values() for tensor in values) == 4 * 8 + 5 * 16


def test_long_synthetic_sequence_keeps_original_short_prefix_and_bounded_soft_positions():
    seen = []
    def encode(text, **kwargs):
        seen.append(text)
        return [ord(character) % 32 for character in text]
    tokenizer = SimpleNamespace(vocab_size=32, encode=encode)
    short, _, _ = probe.synthetic_sequence(tokenizer, 64)
    long, support, probabilities = probe.synthetic_sequence(tokenizer, 8192)
    assert torch.equal(long[:64], short)
    assert len(long) == 8192 and len(seen) > 2
    assert ((probabilities > 0).sum(-1) == 5).nonzero().flatten().tolist() == list(range(8183, 8191))
    assert support.shape == (8192, 5)
    assert torch.equal(probabilities.sum(-1), torch.ones(8192))


def test_actual_cached_control_flow_excludes_prefix_activations(monkeypatch):
    state = {}
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    @contextmanager
    def hooks(model, trace):
        state["trace"] = trace
        yield
    monkeypatch.setattr(probe, "native_hooks", hooks)
    class Batch:
        def __init__(self, ids):
            self.input_ids, self.reqs = ids, [SimpleNamespace()]
        def prepare_for_decode(self):
            self.input_ids = self.output_ids
        def get_model_worker_batch(self):
            return self
    monkeypatch.setattr(probe, "make_native_batch", lambda runner, ids: Batch(ids))
    monkeypatch.setitem(sys.modules, "sglang.srt.model_executor.forward_batch_info",
                        SimpleNamespace(ForwardBatch=SimpleNamespace(init_new=lambda batch, runner: batch)))
    def forward(batch):
        state["trace"].emit("embedding", batch.input_ids[:, None].expand(-1, 8).float())
        return SimpleNamespace(next_token_logits=torch.zeros(len(batch.input_ids), 16))
    result = probe.native_forward(SimpleNamespace(model=None, forward=forward), torch.arange(8192),
                                  None, None, cached=True, capture_last=4)
    assert result["embedding"][:, 0].tolist() == list(range(8188, 8192))
    assert result["logits"].shape == (5, 16)


def test_selective_hf_forward_requests_only_the_final_logit_row(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    original_arange = torch.arange
    monkeypatch.setattr(torch, "arange", lambda *args, **kwargs: original_arange(*args, **{k: v for k, v in kwargs.items() if k != "device"}))
    state, calls = {}, []
    class Model:
        def train(self):
            pass
        def __call__(self, **kwargs):
            calls.append(kwargs)
            state["trace"].emit("embedding", torch.arange(16.)[:, None])
            return SimpleNamespace(logits=torch.zeros(1, 1, 16))
    result = probe.hf_forward(Model(), torch.arange(16), candidate_emit_state=state, capture_last=4)
    assert calls[0]["logits_to_keep"] == 1 and calls[0]["use_cache"] is False
    assert result["embedding"][:, 0].tolist() == [12, 13, 14, 15]
    assert result["logits"].shape == (1, 16)


def _exact_production_result(layers=1, vocab=16):
    native = {stage: torch.zeros(4, 8) for stage in probe.expected_stage_names(layers)}
    native["logits"] = torch.zeros(1, vocab)
    candidate = {stage: value for stage, value in native.items() if not stage.endswith(".block_input")}
    result = {"model_config": {"num_hidden_layers": layers, "vocab_size": vocab}, "capture": {"last_tokens": 4}, "cases": []}
    for input_kind in ("hard_tokens", "soft_mixtures"):
        result["cases"].append({"input": input_kind,
            "native_prefill_vs_cached": probe.compare_traces(native, native),
            "native_prefill_vs_candidate": probe.compare_traces(native, candidate),
            "native_cached_vs_candidate": probe.compare_traces(native, candidate)})
    return result


@pytest.mark.parametrize("corruption", ["drift", "nonfinite", "missing_both", "missing_comparison", "support", "wrong_rows", "missing_case"])
def test_production_exact_gate_blocks_incomplete_or_nonexact_evidence(corruption):
    result = _exact_production_result()
    assert probe.production_exact_issues(result) == []
    comparison = result["cases"][0]["native_prefill_vs_candidate"]
    if corruption == "drift":
        comparison["stages"]["layer.0.qkv"]["max_abs"] = 0.0001
    elif corruption == "nonfinite":
        comparison["final_logits"]["nonfinite_elements"] = 1
    elif corruption == "missing_both":
        comparison["stages"].pop("layer.0.qkv")
    elif corruption == "missing_comparison":
        comparison["missing_comparison"].append("layer.0.qkv")
    elif corruption == "support":
        comparison["final_logits"].pop("native_top5_support")
    elif corruption == "wrong_rows":
        comparison["stages"]["embedding"]["shape"][0] = 8192
    else:
        result["cases"].pop()
    assert probe.production_exact_issues(result)


def test_production_capacity_and_idle_gpu_accounting_are_explicit():
    options = probe.production_server_options("/sealed/model", 8192)
    assert options["opd_qwen_replay_backend"] == "native_fa3_v1" and options["attention_backend"] == "fa3"
    assert options["context_length"] >= 8224 and options["max_total_tokens"] >= 8448
    assert options["chunked_prefill_size"] >= 8192 and options["max_prefill_tokens"] >= 8192
    assert all(options[key] is True for key in ("disable_cuda_graph", "disable_overlap_schedule", "disable_radix_cache"))
    devices = probe.production_device_inventory(["NVIDIA H100", "NVIDIA H100"], allow_idle_second_gpu=True)
    assert devices["visible_count"] == 2 and devices["idle_device_indices"] == [1]
    assert devices["active_device_index"] == 0 and devices["allocated_devices"][1]["used_by_probe"] is False
    for names, allow in ((["H100", "H100"], False), (["H100"], True), (["H100", "A100"], True)):
        with pytest.raises(RuntimeError):
            probe.production_device_inventory(names, allow_idle_second_gpu=allow)


def test_production_cli_infers_installers_and_exact_bounded_capture(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(probe, "run", lambda args, **kwargs: seen.append(args))
    assert probe.main(["--assets", str(tmp_path), "--output", str(tmp_path / "result.json"),
                       "--production-arithmetic", "--length", "8192", "--capture-last", "4",
                       "--require-exact", "--allow-idle-second-gpu"]) == 0
    args = seen[0]
    assert args.candidate is None and args.attention_backend == "fa3"
    assert args.capture_last == 4 and args.length == 8192 and args.require_exact and args.allow_idle_second_gpu
    assert not args.backward_sanity and not args.batch_invariant_linear and not args.candidate_attention_fa3


@pytest.mark.parametrize("flags", [["--length", "8192"], ["--require-exact"], ["--allow-idle-second-gpu"],
    ["--production-arithmetic", "--capture-last", "8"],
    ["--production-arithmetic", "--candidate", "untrusted:installer"],
    ["--production-arithmetic", "--backward-sanity"]])
def test_production_cli_rejects_unbounded_or_mixed_modes(tmp_path, flags):
    with pytest.raises(SystemExit):
        probe.main(["--assets", str(tmp_path), "--output", str(tmp_path / "result.json"), *flags])


@pytest.mark.parametrize("drift", [False, True])
def test_production_run_uses_guarded_installers_only_and_persists_exact_gate(tmp_path, monkeypatch, drift):
    """Exercise actual orchestration; only model loading/GPU forwards are fake."""
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "NVIDIA H100")
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: calls.append(("device", index)))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda index: 1234)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(probe.importlib.metadata, "version", lambda name: "test-pinned")
    table = torch.nn.Embedding(16, 8)
    candidate = SimpleNamespace(config=SimpleNamespace(vocab_size=16, num_hidden_layers=28),
                                model=SimpleNamespace(embed_tokens=table))
    candidate.cuda = lambda: candidate
    runner = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(embed_tokens=table)),
        model_config=SimpleNamespace(), attn_backend=SimpleNamespace(),
        opd_qwen_replay_provenance={"backend": "native_fa3_v1", "attention_backend": "fa3", "num_splits": 1,
                                   "projection_policy": "fixed_tile_triton_32_64_32", "projection_module_count": 112})
    tokenizer = SimpleNamespace(vocab_size=16, encode=lambda *args, **kwargs: list(range(16)))

    def load_native(server, ports, rank):
        assert server.opd_qwen_replay_backend == "native_fa3_v1" and server.attention_backend == "fa3"
        assert rank == 0
        calls.append(("load_native",))
        return runner, tokenizer

    def load_hf(*args, **kwargs):
        calls.append(("load_hf",))
        return candidate

    monkeypatch.setitem(sys.modules, "sglang.bench_one_batch", SimpleNamespace(load_model=load_native))
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.engine", SimpleNamespace(_set_envs_and_config=lambda server: None))
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", SimpleNamespace(
        ServerArgs=lambda **kwargs: SimpleNamespace(**kwargs), PortArgs=SimpleNamespace(init_new=lambda server: None)))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_hf)))
    monkeypatch.setitem(sys.modules, "verl.models.transformers.monkey_patch", SimpleNamespace(apply_monkey_patch=lambda *args, **kwargs: None))
    arithmetic = importlib.import_module("verl.opd.qwen_native_arithmetic")

    def install(model, *, cache_device, emit):
        assert model is candidate and cache_device == torch.device("cuda:0") and callable(emit)
        model._opd_qwen_replay_arithmetic = True
        calls.append(("install_production",))
        return model

    monkeypatch.setattr(arithmetic, "install_qwen_replay_arithmetic", install)
    def forbidden(*args, **kwargs):
        raise AssertionError("production run entered a legacy patch or diagnostic control")
    for name in ("install_native_probe_linear", "fix_native_fa3_split_count", "packed_projection_controls", "projection_shape_controls"):
        monkeypatch.setattr(probe, name, forbidden)

    def native(model, ids, support, probabilities, *, cached=False, capture_last=None):
        assert model is runner and capture_last == 4
        calls.append(("native_forward", cached, support is None))
        trace = {name: torch.zeros(4, 8) for name in probe.expected_stage_names(28)}
        trace["logits"] = torch.zeros(5 if cached else 1, 16)
        return trace

    def hf(model, ids, embedding, *, candidate_emit_state, capture_last):
        assert model is candidate and capture_last == 4
        calls.append(("candidate_forward", embedding is None))
        trace = {name: torch.zeros(4, 8) for name in probe.expected_stage_names(28, candidate=True)}
        trace["logits"] = torch.zeros(1, 16)
        if drift:
            trace["logits"][0, 0] = 0.001
        return trace

    monkeypatch.setattr(probe, "native_forward", native)
    monkeypatch.setattr(probe, "hf_forward", hf)
    args = SimpleNamespace(assets=tmp_path, length=16, capture_last=4, require_exact=True,
                           allow_idle_second_gpu=True, output=tmp_path / "production.json")
    if drift:
        with pytest.raises(RuntimeError, match="parity gate failed"):
            probe._run_production(args, {}, {"manifest_content_sha256": "sealed", "model": {}})
    else:
        probe._run_production(args, {}, {"manifest_content_sha256": "sealed", "model": {}})
    result = json.loads(args.output.read_text())
    assert result["exact_gate"]["passed"] is (not drift)
    assert result["status"] == ("failed" if drift else "diagnostic_complete")
    assert len(result["cases"]) == 2
    assert calls.count(("load_hf",)) == 1 and calls.count(("install_production",)) == 1
    assert [call for call in calls if call[0] == "device"] == [("device", 0)]
    assert [call for call in calls if call[0] == "candidate_forward"] == [("candidate_forward", True), ("candidate_forward", False)]
    assert result["capture"]["cached_prefix_activation_rows_captured"] == 0
    assert result["capture"]["all_sequence_positions_captured"] is False
    assert result["device"]["visible_count"] == 2 and result["device"]["idle_device_indices"] == [1]
