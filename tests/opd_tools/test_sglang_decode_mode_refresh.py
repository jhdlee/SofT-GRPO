"""Execute native request/decode/sampler methods across the think boundary."""

from types import MethodType, SimpleNamespace

import pytest
import torch

from test_sglang_retained_metadata import SRT, load_nodes


def noop(*args, **kwargs):
    pass


def request(seed):
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(soft_thinking_mode=None,
                                        think_end_str_id=4, early_stopping_entropy_threshold=0),
        output_ids=[1], low_entropy_steps=0, seed=seed,
        return_logprob=False, stream=False, grammar=None, finished=lambda: False,
    )
    for field in ("prob", "gumbel", "gumbel_noise", "retained_mask", "idx"):
        setattr(req, f"output_topk_{field}_list_tmp", [])
    initialize = load_nodes(SRT / "sampling/sampling_params.py", ["post_init_soft_thinking_mode"],
                            {"torch": torch}, class_name="SamplingParams")["post_init_soft_thinking_mode"]
    initialize(req.sampling_params)
    assert req.sampling_params.soft_thinking_mode is True
    return req


def close_request(req):
    update = load_nodes(SRT / "managers/schedule_batch.py", ["update_topk_info"],
                        {"torch": torch}, class_name="Req")["update_topk_info"]
    req.output_ids[-1] = 4
    output = SimpleNamespace(
        topk_gumbels=torch.tensor([[0.2, 0.1, -4.0, -5.0, -6.0]]),
        topk_gumbel_noise=torch.ones(1, 5),
        topk_retained_mask=torch.ones(1, 5, dtype=torch.bool),
        topk_probs=torch.tensor([[0.8, 0.1, 0.06, 0.03, 0.01]]),
        topk_indices=torch.tensor([[4, 0, 1, 2, 3]]), entropy=torch.tensor([0.5]),
    )
    update(req, output, 0)
    assert req.sampling_params.soft_thinking_mode is False


def batch(reqs, *, enabled=True):
    n = len(reqs)
    info = SimpleNamespace(
        enable_soft_thinking=enabled, device="cpu", has_custom_logit_processor=False,
        is_all_greedy=False, is_all_no_noise=False, need_min_p_sampling=False,
        need_after_thinking_min_p_sampling=False, noise_gumbel=True, noise_on_logits=True,
        max_topk=5, grammars=None,
        penalizer_orchestrator=SimpleNamespace(is_required=False, filter=noop, merge=noop),
        temperatures=torch.ones(n, 1), top_ps=torch.full((n,), 0.95),
        top_ks=torch.full((n,), 5), min_ps=torch.zeros(n),
        random_seeds=torch.tensor([r.seed for r in reqs]),
        random_counters=torch.full((n,), 3, dtype=torch.int64),
        deterministic_random_mask=torch.ones(n, dtype=torch.bool),
        soft_thinking_modes=torch.ones(n, dtype=torch.bool),
        after_thinking_temperatures=torch.ones(n, 1), after_thinking_top_ps=torch.full((n,), 0.8),
        after_thinking_top_ks=torch.full((n,), 3), after_thinking_min_ps=torch.zeros(n),
        dirichlet_alphas=torch.zeros(n), early_stopping_entropy_threshold=torch.zeros(n),
        early_stopping_length_threshold=torch.zeros(n),
        gumbel_softmax_temperatures=torch.full((n, 1), 0.1), noise_factor=torch.ones(n),
    )
    info_methods = load_nodes(SRT / "sampling/sampling_batch_info.py", ["filter_batch", "merge_batch"],
                              {"torch": torch}, class_name="SamplingBatchInfo")
    for name in ("filter_batch", "merge_batch"):
        setattr(info, name, MethodType(info_methods[name], info))
    b = SimpleNamespace(
        reqs=list(reqs), sampling_info=info, enable_soft_thinking=enabled, enable_overlap=False,
        device="cpu", spec_algorithm=SimpleNamespace(is_eagle=lambda: False), spec_info=None,
        model_config=SimpleNamespace(is_encoder_decoder=False),
        req_pool_indices=torch.arange(n), seq_lens=torch.full((n,), 4, dtype=torch.int64),
        seq_lens_sum=n * 4, output_ids=torch.tensor([r.output_ids[-1] for r in reqs]),
        return_logprob=False, return_hidden_states=False, has_stream=False, has_grammar=False,
        req_to_token_pool=SimpleNamespace(write=noop),
        token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
        alloc_token_slots=lambda count: torch.arange(count),
    )
    methods = load_nodes(SRT / "managers/schedule_batch.py",
                         ["prepare_for_decode", "filter_batch", "merge_batch"],
                         {"torch": torch, "ForwardMode": SimpleNamespace(DECODE="decode")},
                         class_name="ScheduleBatch")
    for name in ("prepare_for_decode", "filter_batch", "merge_batch"):
        setattr(b, name, MethodType(methods[name], b))
    return b


def sample(info):
    calls = {}
    n = info.temperatures.shape[0]
    filtered = torch.tensor([[0.55, 0.2, 0.15, 0.07, 0.03, 0.0, 0.0]]).repeat(n, 1)

    def top_k(probs, ks):
        calls["top_ks"] = ks.clone()
        return probs

    def top_p(probs, ps):
        calls["top_ps"] = ps.clone()
        return filtered.clone()

    def categorical(probs, sampling_info):
        calls["categorical_probs"] = probs.clone()
        calls["categorical_seeds"] = sampling_info.random_seeds.clone()
        return torch.full((n, 1), 2, dtype=torch.long)

    namespace = {
        "torch": torch, "global_server_args_dict": {"sampling_backend": "flashinfer"},
        "SYNC_TOKEN_IDS_ACROSS_TP": False, "top_k_renorm_prob": top_k, "top_p_renorm_prob": top_p,
        "_request_local_gumbel": lambda logits, _: torch.tensor([[1.5, 0.4, -0.3, 0.2, -0.8]]).repeat(n, 1),
        "_request_local_categorical": categorical,
    }
    forward = load_nodes(SRT / "layers/sampler.py", ["forward"], namespace, class_name="Sampler")["forward"]
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0]]).repeat(n, 1)
    output = SimpleNamespace(next_token_logits=logits.clone())
    tokens = forward(SimpleNamespace(use_nan_detection=False), output, info, True, [], [],
                     enable_soft_thinking=True, add_noise_gumbel_softmax=True)
    return tokens, output, logits, calls


@pytest.mark.parametrize("layout", ["normal", "filtered", "merged", "filtered_merged"])
def test_actual_decode_refreshes_modes_after_close_in_current_request_order(layout):
    reqs = [request(11), request(22), request(33)]
    b = batch(reqs)
    close_request(reqs[2])
    assert b.sampling_info.soft_thinking_modes.tolist() == [True, True, True]
    if "filtered" in layout:
        b.filter_batch(keep_indices=[2, 0])
    if "merged" in layout:
        added = [request(44), request(55)]
        other = batch(added)
        close_request(added[1])
        b.merge_batch(other)
    expected_modes = [bool(r.sampling_params.soft_thinking_mode) for r in b.reqs]
    expected_seeds = [r.seed for r in b.reqs]
    counters = b.sampling_info.random_counters.clone()
    b.prepare_for_decode()
    assert b.sampling_info.soft_thinking_modes.dtype == torch.bool
    assert b.sampling_info.soft_thinking_modes.tolist() == expected_modes
    assert b.sampling_info.random_seeds.tolist() == expected_seeds
    assert torch.equal(b.sampling_info.random_counters, counters)
    tokens, output, logits, calls = sample(b.sampling_info)
    hard = ~torch.tensor(expected_modes)
    assert calls["categorical_seeds"].tolist() == expected_seeds
    assert calls["top_ks"].tolist() == [5 if mode else 3 for mode in expected_modes]
    assert torch.equal(calls["top_ps"], torch.tensor([0.95 if mode else 0.8 for mode in expected_modes]))
    assert tokens[hard].tolist() == [2] * int(hard.sum())
    assert torch.equal(output.topk_indices[hard], torch.tensor([[2, 0, 0, 0, 0]]).repeat(int(hard.sum()), 1))
    assert torch.equal(output.topk_probs[hard], torch.tensor([[1., 0, 0, 0, 0]]).repeat(int(hard.sum()), 1))
    assert output.topk_retained_mask[hard].tolist() == [[True, False, False, False, False]] * int(hard.sum())
    assert not output.topk_gumbel_noise[hard].any()
    # Preserve the released categorical score: full-vocabulary probability,
    # even though the actual categorical draw uses the truncated distribution.
    expected_logp = logits.softmax(-1).log().gather(-1, tokens.long()[:, None]).squeeze(-1)
    assert torch.equal(output.next_token_logprobs, expected_logp)
    assert torch.equal(b.sampling_info.random_counters, counters + 1)


def test_refresh_preserves_every_preclose_action_and_density_with_same_request_draws():
    original = batch([request(11), request(22)])
    refreshed = batch([request(11), request(22)])
    refreshed.prepare_for_decode()
    old_tokens, old_output, _, old_calls = sample(original.sampling_info)
    new_tokens, new_output, _, new_calls = sample(refreshed.sampling_info)
    assert torch.equal(old_tokens, new_tokens)
    assert "categorical_probs" not in old_calls and "categorical_probs" not in new_calls
    for name in ("topk_probs", "topk_indices", "topk_gumbels", "topk_gumbel_noise",
                 "topk_retained_mask", "next_token_logprobs", "next_token_gumbel_logprobs"):
        assert torch.equal(getattr(old_output, name), getattr(new_output, name))


def test_nonsoft_decode_does_not_inspect_or_replace_mode_flags():
    b = batch([request(11)], enabled=False)
    b.reqs[0].sampling_params = None
    sentinel = b.sampling_info.soft_thinking_modes
    b.prepare_for_decode()
    assert b.sampling_info.soft_thinking_modes is sentinel


def test_successive_decodes_observe_a_later_close_without_resetting_stream_position():
    req = request(11)
    b = batch([req])
    b.prepare_for_decode()
    sample(b.sampling_info)
    assert b.sampling_info.random_counters.tolist() == [4]
    close_request(req)
    b.output_ids = torch.tensor([4])
    b.prepare_for_decode()
    _, output, _, calls = sample(b.sampling_info)
    assert b.sampling_info.random_counters.tolist() == [5]
    assert "categorical_probs" in calls
    assert not output.topk_gumbel_noise.any()


@pytest.mark.parametrize("stop,expected_stops", [(None, []), ("STOP", ["STOP"]), (["END", "STOP"], ["END", "STOP"])])
def test_actual_sampling_params_normalize_and_mode_initialization_need_no_tensor_allocation(stop, expected_stops):
    class NoTensorAllocation:
        def __getattr__(self, name):
            raise AssertionError(f"request mode initialization must not access torch.{name}")

    namespace = {"torch": NoTensorAllocation(), "_SAMPLING_EPS": 1e-6, "MAX_SEED": (1 << 63) - 1}
    params_class = load_nodes(SRT / "sampling/sampling_params.py", ["SamplingParams"], namespace)["SamplingParams"]
    params = params_class(stop=stop, top_k=5, seed=11)
    params.verify()
    params.normalize(tokenizer=None)
    assert params.soft_thinking_mode is None
    assert params.stop_strs == expected_stops
    params.post_init_soft_thinking_mode()
    assert params.soft_thinking_mode is True
    assert type(params.soft_thinking_mode) is bool
    assert params.seed == 11 and params.top_k == 5
