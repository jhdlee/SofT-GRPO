"""Execute native scheduler output paths after an entropy-forced close."""

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch.nn.utils.rnn import pad_sequence

from test_sglang_retained_metadata import SRT, load_nodes


@pytest.mark.parametrize("path", ["prefill", "decode"])
@pytest.mark.parametrize("case", ["forced_close", "natural_close", "latent", "answer", "eos", "nonsoft_engine"])
def test_native_logged_ids_follow_emitted_action_without_changing_density_or_finish(path, case):
    req_source = SRT / "managers/schedule_batch.py"
    namespace = load_nodes(req_source, ["BaseFinishReason", "FINISH_MATCHED_TOKEN"], {"torch": torch})
    methods = ["finished", "check_finished", "update_topk_info", "get_output_topk_prob_list",
               "get_output_topk_idx_list", "get_output_topk_gumbel_list",
               "get_output_topk_gumbel_noise_list", "get_output_topk_retained_mask_list"]
    load_nodes(req_source, methods, namespace, class_name="Req")
    sampled_id = 9 if case == "natural_close" else 11 if case == "eos" else 3
    output = SimpleNamespace(
        next_token_logprobs=torch.tensor([-0.25]), next_token_gumbel_logprobs=torch.tensor([-1.75]),
        input_token_logprobs=None, hidden_states=None,
        topk_probs=torch.tensor([[0.9, 0.05, 0.03, 0.01, 0.01]]),
        topk_indices=torch.tensor([[sampled_id, 7, 2, 8, 5]]),
        topk_gumbels=torch.tensor([[0.2, 0.1, -4.0, -5.0, -6.0]]),
        topk_gumbel_noise=torch.tensor([[0.3, -0.1, 0.5, 1.0, -0.5]]),
        topk_retained_mask=torch.tensor([[True, True, True, False, False]]), entropy=torch.tensor([0.0]),
    )
    original_perturbations = output.topk_gumbels.clone()
    original_noise = output.topk_gumbel_noise.clone()
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(
            soft_thinking_mode=case in ("forced_close", "natural_close", "latent"),
            think_end_str_id=9, early_stopping_entropy_threshold=0.1 if case == "forced_close" else 0,
            early_stopping_length_threshold=1, max_new_tokens=20, ignore_eos=False,
            stop_token_ids=[], stop_strs=[],
        ),
        output_ids=[], output_token_logprobs_idx=[], output_token_logprobs_val=[],
        low_entropy_steps=0, to_abort=False, finished_reason=None, eos_token_ids={11}, tokenizer=None,
        is_retracted=False, is_chunked=0, return_logprob=True, return_hidden_states=False,
        top_logprobs_num=0, token_ids_logprob=None, grammar=None,
    )
    for field in ("prob", "idx", "gumbel", "gumbel_noise", "retained_mask"):
        setattr(req, f"output_topk_{field}_list", [])
        setattr(req, f"output_topk_{field}_list_tmp", [])
    for name in methods:
        setattr(req, name, MethodType(namespace[name], req))

    scheduler_methods = ["process_batch_result_prefill", "process_batch_result_decode", "add_logprob_return_values"]
    load_nodes(SRT / "managers/scheduler_output_processor_mixin.py", scheduler_methods, namespace,
               class_name="SchedulerOutputProcessorMixin")
    no_op = lambda *args, **kwargs: None
    streamed = []
    scheduler = SimpleNamespace(
        is_generation=True, enable_overlap=False, is_mixed_chunk=False,
        enable_soft_thinking=case != "nonsoft_engine", num_generated_tokens=0,
        forward_ct_decode=0, attn_tp_rank=1,
        tree_cache=SimpleNamespace(cache_finished_req=no_op, cache_unfinished_req=no_op),
        token_to_kv_pool_allocator=SimpleNamespace(free_group_begin=no_op, free_group_end=no_op),
        stream_output=lambda *args: streamed.append(list(req.output_ids)),
        add_input_logprob_return_values=no_op,
    )
    for name in scheduler_methods:
        setattr(scheduler, name, MethodType(namespace[name], scheduler))
    batch = SimpleNamespace(reqs=[req], return_logprob=True, decoding_reqs=[], next_batch_sampling_info=None,
                            spec_algorithm=SimpleNamespace(is_none=lambda: True))
    result = SimpleNamespace(logits_output=output, next_token_ids=torch.tensor([sampled_id]), bid=None,
                             extend_input_len_per_req=[0], extend_logprob_start_len_per_req=[0])

    getattr(scheduler, f"process_batch_result_{path}")(batch, result)

    emitted_id = 9 if case == "forced_close" else sampled_id
    assert req.output_ids == req.output_token_logprobs_idx == [emitted_id]
    assert streamed == [[emitted_id]]
    assert result.next_token_ids.tolist() == [sampled_id]
    assert req.finished() is (case == "eos")
    if case == "eos":
        assert req.finished_reason.to_json() == {"type": "stop", "matched": 11}
    # Mode switches retain the released density selection. Boundary scoring
    # remains excluded downstream; this fix changes only the exported ID label.
    expected_density = -1.75 if case == "latent" else -0.25
    assert float(req.output_token_logprobs_val[0]) == expected_density
    assert torch.equal(output.topk_gumbels, original_perturbations)
    assert torch.equal(output.topk_gumbel_noise, original_noise)

    if case != "nonsoft_engine":
        support = req.get_output_topk_idx_list()
        assert support[0][0] == emitted_id
        metadata = {
            "output_token_logprobs": [(float(req.output_token_logprobs_val[0]), emitted_id, None)],
            "output_topk_idx_list": support,
            "output_topk_gumbel_list": req.get_output_topk_gumbel_list(),
            "output_topk_gumbel_noise_list": req.get_output_topk_gumbel_noise_list(),
            "output_topk_retained_mask_list": req.get_output_topk_retained_mask_list(),
            "output_topk_prob_list": req.get_output_topk_prob_list(),
        }
        adapter_path = SRT.parents[4] / "verl-0.4.x/verl/workers/rollout/sglang_rollout/sglang_rollout.py"
        adapter = load_nodes(adapter_path, ["_post_process_outputs"], {"torch": torch, "pad_sequence": pad_sequence})
        response = adapter["_post_process_outputs"](
            SimpleNamespace(pad_token_id=0), [{"output_ids": req.output_ids, "meta_info": metadata}],
            require_retained_support=True,
        )
        assert response[0].tolist() == [[emitted_id]]
