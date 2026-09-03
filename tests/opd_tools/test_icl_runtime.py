import ast
import json
import tempfile
from pathlib import Path
import numpy as np
import pytest

from opd_tools.icl import request_seed
from opd_tools.icl_runtime import (
    AtomicChunkStore,
    SamplingSettings,
    boundary_gate,
    canonical_json_bytes,
    generation_chunk_metrics,
    parse_sglang_completion,
    required_context_length,
    source_provenance,
    stable_request_seed,
    validate_generation_cell,
)


class _ReleasedTextTokenizer:
    def __init__(self):
        self.add_special_tokens_values = []

    def encode(self, text, add_special_tokens=True):
        self.add_special_tokens_values.append(add_special_tokens)
        return [1, 1, 2] if add_special_tokens else [1, 2]


def _softmax(values, temperature=0.1):
    values = np.asarray(values, dtype=np.float64) / temperature
    values -= values.max(axis=-1, keepdims=True)
    result = np.exp(values)
    return (result / result.sum(axis=-1, keepdims=True)).tolist()


def test_context_length_matches_upstream_text_api_tokenizer_defaults():
    tokenizer = _ReleasedTextTokenizer()
    settings = SamplingSettings(max_new_tokens=7)
    assert required_context_length(tokenizer, ["rendered prompt"], settings) == 10
    assert tokenizer.add_special_tokens_values == [True]


def _output(
    *, supports=None, logits=None, noise=None, token_ids=None, text=None, finish="stop"
):
    supports = supports or [
        [10, 11, 12, 13, 14],
        [20, 21, 22, 23, 24],
        [2, 0, 0, 0, 0],
        [30, 0, 0, 0, 0],
    ]
    logits = logits or [
        [3.0, 1.0, 0.5, -0.5, -1.0],
        [2.5, 1.5, 0.0, -0.5, -1.5],
        [0.0] * 5,
        [0.0] * 5,
    ]
    token_ids = token_ids or [10, 20, 2, 30]
    default_noise = [
        [1.0, 0.5, -0.2, 0.0, 0.8],
        [0.3, 0.1, -0.4, 0.7, -0.6],
        [0.0] * 5,
        [0.0] * 5,
    ]
    noise = noise or default_noise[: len(supports)]
    probabilities = [
        ([1.0, 0.0, 0.0, 0.0, 0.0] if all(value == 0 for value in support[1:]) else _softmax([logit])[0])
        for support, logit in zip(supports, logits, strict=True)
    ]
    return {
        "text": text or "reasoning</think>\nThe final answer is: \\boxed{1}",
        "meta_info": {
            "finish_reason": {"type": finish},
            "output_token_logprobs": [(0.0, value) for value in token_ids],
            "output_topk_idx_list": supports,
            "output_topk_gumbel_list": logits,
            "output_topk_gumbel_noise_list": noise,
            "output_topk_prob_list": probabilities,
        },
    }


def _parse(output=None, **overrides):
    values = {
        "output": output or _output(),
        "model_label": "starting",
        "mode": "native_soft",
        "benchmark": "math500",
        "condition": "no_demo",
        "example_id": "math500-0",
        "sample_index": 0,
        "replay_row": 0,
        "settings": SamplingSettings(),
        "think_end_id": 2,
    }
    values.update(overrides)
    return parse_sglang_completion(**values)


def test_runtime_delegates_to_cpu_request_seed_contract():
    expected = request_seed("math500", "math500-0", 0)
    assert stable_request_seed(
        benchmark="math500", example_id="math500-0", sample_index=0
    ) == expected


def test_runtime_canonical_json_rejects_nonfinite_numbers():
    with pytest.raises(ValueError, match="Out of range float"):
        canonical_json_bytes({"undefined": float("nan")})


def test_source_provenance_seals_parent_fork_sampler_and_icl_files(monkeypatch):
    monkeypatch.delenv("OPD_EXPECTED_SUBMODULE_COMMIT", raising=False)
    monkeypatch.delenv("OPD_SUBMODULE_COMMIT", raising=False)
    monkeypatch.delenv("OPD_PARENT_COMMIT", raising=False)
    provenance = source_provenance()
    assert len(provenance["parent_commit"]) == 40
    assert len(provenance["fork_commit"]) == 40
    assert provenance["upstream_softgrpo_commit"] == (
        "8d3c61380b15c3400818da5ce41c62c293a1bfb4"
    )
    assert provenance["sampler_files"]
    sampler_paths = {entry["path"] for entry in provenance["sampler_files"]}
    assert {
        "Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python/sglang/srt/layers/sampler.py",
        "Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python/sglang/srt/layers/logits_processor.py",
        "Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python/sglang/srt/managers/schedule_batch.py",
        "Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python/sglang/srt/managers/scheduler_output_processor_mixin.py",
        "Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python/sglang/srt/managers/io_struct.py",
        "Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python/sglang/srt/managers/detokenizer_manager.py",
        "Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python/sglang/srt/managers/multi_tokenizer_mixin.py",
        "Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python/sglang/srt/managers/tokenizer_manager.py",
    }.issubset(sampler_paths)
    implementation_paths = {
        entry["path"] for entry in provenance["implementation_files"]
    }
    assert {
        "opd_tools/icl.py",
        "opd_tools/icl_assets.py",
        "opd_tools/icl_runtime.py",
        "opd_tools/icl_replay.py",
        "opd_tools/icl_eval.py",
        "opd_tools/graders.py",
        "opd_tools/records.py",
    }.issubset(implementation_paths)

    monkeypatch.setenv("OPD_SUBMODULE_COMMIT", "0" * 40)
    with pytest.raises(RuntimeError, match="submodule commit"):
        source_provenance()


def test_multi_tokenizer_plumbs_all_compact_soft_metadata_fields():
    repository = Path(__file__).resolve().parents[2]
    path = (
        repository
        / "Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python/sglang/srt/managers/multi_tokenizer_mixin.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    required = {
        "output_topk_gumbel_list",
        "output_topk_gumbel_noise_list",
        "output_topk_probs_list",
        "output_topk_indices_list",
    }
    constructors = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"BatchTokenIDOut", "BatchStrOut"}
    ]
    assert len(constructors) == 2
    for constructor in constructors:
        assert required.issubset(
            {keyword.arg for keyword in constructor.keywords if keyword.arg}
        )


def test_locked_sampling_params_match_released_training_path():
    settings = SamplingSettings()
    params = settings.request_params(17)
    assert (params["top_p"], params["top_k"], params["temperature"]) == (
        0.95,
        5,
        1.0,
    )
    assert params["gumbel_softmax_temperature"] == 0.1
    assert params["max_new_tokens"] == 8192
    assert params["noise_gumbel"] and params["noise_on_logits"]
    assert not params["noise_gaussian"] and not params["noise_on_inputs"]


def test_matrix_guard_rejects_unregistered_hard_token_cells():
    validate_generation_cell("starting", "hard_token", "sdft_matched")
    with pytest.raises(ValueError, match="post-trained"):
        validate_generation_cell("softgrpo", "hard_token", "sdft_matched")
    with pytest.raises(ValueError, match="mechanism"):
        validate_generation_cell("starting", "hard_token", "sdft_answer_only")


def test_parse_native_soft_retains_sufficient_metadata_and_boundary():
    record, trajectory = _parse()
    assert trajectory.response_token_ids == (10, 20, 2, 30)
    assert trajectory.latent_support_ids == (
        (10, 11, 12, 13, 14),
        (20, 21, 22, 23, 24),
    )
    assert len(trajectory.latent_perturbed_logits) == 2
    assert trajectory.latent_gumbel_noise[0] == (1.0, 0.5, -0.2, 0.0, 0.8)
    assert record.latent_token_count == 2
    assert record.hard_token_count == 2
    assert record.soft_to_hard and record.close_tag and record.boxed_answer
    assert record.boundary_valid and not record.all_soft
    assert record.stored_mixture_reconstruction_abs_error_max < 1e-10


def test_parse_rejects_continuous_reentry_and_probability_drift():
    supports = [
        [10, 11, 12, 13, 14],
        [2, 0, 0, 0, 0],
        [20, 21, 22, 23, 24],
    ]
    logits = [[3, 2, 1, 0, -1], [0] * 5, [3, 2, 1, 0, -1]]
    with pytest.raises(RuntimeError, match="re-entered"):
        _parse(output=_output(supports=supports, logits=logits, token_ids=[10, 2, 20]))

    output = _output()
    output["meta_info"]["output_topk_prob_list"][0] = [0.2] * 5
    with pytest.raises(RuntimeError, match="do not reconstruct"):
        _parse(output=output)

    wrong_noise = _output()
    wrong_noise["meta_info"]["output_topk_gumbel_noise_list"] = [[0.0] * 5]
    with pytest.raises(ValueError, match="metadata shape"):
        _parse(output=wrong_noise)


@pytest.mark.parametrize(
    "text",
    (
        "reasoning</think>\\boxed{",
        "reasoning</think>\\boxed{}",
        "reasoning</think>\\boxed   {   }",
    ),
)
def test_boundary_rejects_truncated_or_empty_final_box(text):
    record, _ = _parse(output=_output(text=text))
    assert not record.boxed_answer
    assert not record.boundary_valid


def test_boundary_accepts_balanced_nonempty_nested_final_box():
    record, _ = _parse(
        output=_output(text="reasoning</think>\\boxed {\\frac{1}{2}}")
    )
    assert record.boxed_answer and record.boundary_valid


def test_boundary_requires_box_in_categorical_suffix():
    record, _ = _parse(
        output=_output(text="reasoning \\boxed{1}</think> trailing prose")
    )
    assert record.boxed_answer
    assert record.soft_to_hard
    assert not record.boundary_valid


def test_parse_rejects_categorical_sentinel_drift():
    wrong_head = _output()
    wrong_head["meta_info"]["output_topk_idx_list"][2][0] = 99
    with pytest.raises(RuntimeError, match="emitted token"):
        _parse(output=wrong_head)

    wrong_probability = _output()
    wrong_probability["meta_info"]["output_topk_prob_list"][2] = [
        0.9,
        0.1,
        0.0,
        0.0,
        0.0,
    ]
    with pytest.raises(RuntimeError, match="not one-hot"):
        _parse(output=wrong_probability)


def test_no_fallback_gate_marks_all_soft_and_caps_invalid():
    supports = [[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]]
    logits = [[3, 2, 1, 0, -1], [3, 2, 1, 0, -1]]
    output = _output(
        supports=supports,
        logits=logits,
        token_ids=[10, 20],
        text="unfinished",
        finish="length",
    )
    record, _ = _parse(output=output)
    assert record.all_soft and record.capped
    assert not record.boundary_valid
    gate = boundary_gate([record])
    assert not gate["valid"] and gate["failure_rate"] == 1.0


def test_cell_gate_separates_malformed_answer_from_trajectory_failure():
    record, _ = _parse(
        output=_output(text="reasoning</think>categorical prose without a box")
    )
    assert record.soft_to_hard and not record.capped and not record.all_soft
    assert not record.boundary_valid
    gate = boundary_gate([record])
    assert gate["valid"]
    assert gate["failure_rate"] == 0.0
    assert gate["demonstrated_boundary_count"] == 0


def test_native_categorical_only_output_is_boundary_invalid_zero_latent_source():
    output = _output(
        supports=[[2, 0, 0, 0, 0], [30, 0, 0, 0, 0]],
        logits=[[0.0] * 5, [0.0] * 5],
        noise=[[0.0] * 5, [0.0] * 5],
        token_ids=[2, 30],
        text="</think>\\boxed{1}",
    )
    record, trajectory = _parse(output=output)
    assert record.latent_token_count == 0
    assert record.hard_token_count == 2
    assert not record.boundary_valid
    assert trajectory.latent_support_ids == ()


def test_hard_token_parser_does_not_accept_soft_diagnostics():
    output = {
        "text": "work</think>\\boxed{1}",
        "meta_info": {
            "finish_reason": {"type": "stop"},
            "output_token_logprobs": [(0.0, 5), (0.0, 6)],
        },
    }
    record, trajectory = _parse(
        output=output, mode="hard_token", condition="sdpg_matched"
    )
    assert record.hard_token_count == 2
    assert record.latent_token_count == 0
    assert trajectory.latent_support_ids == ()


def test_atomic_chunk_commit_resume_and_authentication():
    record, trajectory = _parse()
    identity = {"cell": "starting/native_soft/math500/no_demo", "chunk": 0}
    with tempfile.TemporaryDirectory() as directory:
        store = AtomicChunkStore(directory)
        first = store.commit("cell/chunk_0000", [record], [trajectory], identity=identity)
        second = store.commit("cell/chunk_0000", [record], [trajectory], identity=identity)
        assert first == second
        records, arrays = store.load("cell/chunk_0000")
        assert records == [record]
        assert arrays["latent_support_ids"].shape == (2, 5)
        assert arrays["latent_gumbel_noise"].shape == (2, 5)

        records_path, _, _ = store.paths("cell/chunk_0000")
        with records_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record.to_dict()) + "\n")
        with pytest.raises(RuntimeError, match="authentication"):
            store.verify("cell/chunk_0000")


def test_generation_metric_names_expose_boundary_not_fallback():
    record, _ = _parse()
    metrics = generation_chunk_metrics([record], elapsed_seconds=2.0)
    assert metrics["generation/capped_or_all_soft_rate"] == 0.0
    assert metrics["generation/boundary_valid_rate"] == 1.0
    assert metrics["generation/all_soft_rate"] == 0.0
    assert (
        metrics["generation/stored_mixture_reconstruction_abs_error_max"]
        < 1e-10
    )
