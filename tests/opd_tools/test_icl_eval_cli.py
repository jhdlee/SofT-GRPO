import argparse
import hashlib
import json
from dataclasses import replace

import pytest

import opd_tools.icl_eval as cli
from opd_tools.icl import (
    CORE_CONDITIONS,
    QWEN3_STUDY_ID,
    ICLEvaluationExample,
    ICLMatrixCell,
    request_seed,
)
from opd_tools.icl_eval import (
    _finish_generation_resources,
    _generation_chunk_identity,
    _grade,
    _primary_grade,
    _run_symmetric_throughput_warmup,
    _cell_metric_rows,
    _comparison_rows,
    _render,
    _throughput_warmup_contract,
    _validate_completion_records,
    _validate_generation_binding,
    _validate_replay_records,
    build_parser,
)
from opd_tools.icl_replay import ReplayRecord
from opd_tools.icl_runtime import (
    UPSTREAM_TEXT_PROMPT_TOKENIZATION,
    AtomicChunkStore,
    CompletionRecord,
    SamplingSettings,
    TrajectoryMetadata,
    canonical_json_bytes,
)


class _Tokenizer:
    def __init__(self, suffix="<think>\n"):
        self.suffix = suffix
        self.calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.calls.append((messages, tokenize, add_generation_prompt))
        return "native-chat-prefix" + self.suffix

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [1, 2]


class _Qwen3Tokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "<|im_start|>user\nQuestion<|im_end|>\n<|im_start|>assistant\n<think>\n"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [1, 2]


def test_generation_cleanup_flushes_wandb_before_sglang_shutdown():
    events = []

    class _Run:
        summary = {}

        def finish(self):
            events.append("wandb")

    class _Engine:
        def shutdown(self):
            events.append("sglang")

    _finish_generation_resources(
        wandb_run=_Run(), engine=_Engine(), succeeded=True
    )
    assert events == ["wandb", "sglang"]


def _generation_inputs():
    assets = {
        "content_sha256": "asset-manifest-sha",
        "models": {"starting": {"tree_sha256": "starting-tree-sha"}},
    }
    data_manifest = {"content_sha256": "data-manifest-sha"}
    manifest = {
        "asset_manifest_sha256": assets["content_sha256"],
        "data_manifest_sha256": data_manifest["content_sha256"],
        "model_label": "starting",
        "model_tree_sha256": assets["models"]["starting"]["tree_sha256"],
        "sampling": SamplingSettings().__dict__,
        "prompt_tokenization": UPSTREAM_TEXT_PROMPT_TOKENIZATION,
    }
    return manifest, assets, data_manifest


def _completion(
    example_id="math500-0",
    *,
    replay_row=0,
    condition="no_demo",
    inference_mode="hard_token",
    sample_index=0,
):
    native = inference_mode == "native_soft"
    return CompletionRecord(
        model_label="starting",
        inference_mode=inference_mode,
        benchmark="math500",
        condition=condition,
        example_id=example_id,
        sample_index=sample_index,
        request_seed=request_seed("math500", example_id, sample_index),
        response=(
            "reasoning</think>\\boxed{42}"
            if native
            else "The final answer is \\boxed{42}"
        ),
        response_token_count=4 if native else 2,
        finish_reason="stop",
        capped=False,
        latent_token_count=2 if native else 0,
        hard_token_count=2,
        close_tag=native,
        soft_to_hard=native,
        all_soft=False,
        boxed_answer=True,
        boundary_valid=native,
        mixture_entropy_mean=0.2 if native else None,
        top1_weight_mean=0.9 if native else None,
        soft_hard_agreement=1.0 if native else None,
        stored_mixture_reconstruction_abs_error_max=0.0 if native else None,
        replay_row=replay_row,
    )


def _replay_record(source, condition, *, excluded=False):
    if excluded:
        return ReplayRecord(
            model_label=source.model_label,
            benchmark=source.benchmark,
            example_id=source.example_id,
            sample_index=source.sample_index,
            prompted_condition=condition,
            request_seed=source.request_seed,
            latent_token_count=0,
            replay_exclusion_reason="zero_latent_slots",
            forward_kl_mean=None,
            forward_kl_sum=None,
            reverse_kl_mean=None,
            reverse_kl_sum=None,
            prompted_entropy_mean=None,
            prompted_top1_probability_mean=None,
            sglang_hf_active_support_exact_slots=0,
            sglang_hf_centered_logprob_value_count=0,
            sglang_hf_centered_logprob_abs_error_sum=None,
            sglang_hf_centered_logprob_abs_error_max=None,
            elapsed_seconds=0.0,
        )
    return ReplayRecord(
        model_label=source.model_label,
        benchmark=source.benchmark,
        example_id=source.example_id,
        sample_index=source.sample_index,
        prompted_condition=condition,
        request_seed=source.request_seed,
        latent_token_count=source.latent_token_count,
        replay_exclusion_reason=None,
        forward_kl_mean=0.1,
        forward_kl_sum=0.2,
        reverse_kl_mean=0.1,
        reverse_kl_sum=0.2,
        prompted_entropy_mean=0.3,
        prompted_top1_probability_mean=0.8,
        sglang_hf_active_support_exact_slots=source.latent_token_count,
        sglang_hf_centered_logprob_value_count=source.latent_token_count * 5,
        sglang_hf_centered_logprob_abs_error_sum=0.0,
        sglang_hf_centered_logprob_abs_error_max=0.0,
        elapsed_seconds=0.1,
    )


def test_render_uses_native_template_once_without_added_specials():
    tokenizer = _Tokenizer()
    rendered = _render(tokenizer, "Question")
    assert rendered.endswith("<think>\n")
    assert tokenizer.calls == [
        ([{"role": "user", "content": "Question"}], False, True)
    ]
    with pytest.raises(RuntimeError, match="native assistant"):
        _render(_Tokenizer(suffix="assistant>"), "Question")


def test_qwen3_render_enables_native_thinking_and_seals_one_fixed_opener():
    tokenizer = _Qwen3Tokenizer()
    rendered = _render(tokenizer, "Question", study=QWEN3_STUDY_ID)
    assert rendered.endswith("<|im_start|>assistant\n<think>\n")
    assert rendered.count("<think>\n") == 1
    assert tokenizer.calls == [
        (
            [{"role": "user", "content": "Question"}],
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": True,
            },
        )
    ]


def test_throughput_warmup_is_dedicated_and_symmetric_per_replica():
    tokenizer = _Qwen3Tokenizer()

    class Engine:
        def __init__(self):
            self.calls = []

        def warmup(self, prompt, seed, *, max_new_tokens):
            self.calls.append((prompt, seed, max_new_tokens))

    engine = Engine()
    _run_symmetric_throughput_warmup(
        engine,
        tokenizer,
        study=QWEN3_STUDY_ID,
        data_parallel_size=2,
    )
    assert len(engine.calls) == 2
    assert engine.calls[0] == engine.calls[1]
    assert engine.calls[0][2] == 32
    assert tokenizer.calls[0][0][0]["content"].startswith("Warm-up request only")
    dp1 = _throughput_warmup_contract(1)
    dp2 = _throughput_warmup_contract(2)
    assert dp1["request_count"] == 1
    assert dp2["request_count"] == 2
    assert {key: value for key, value in dp1.items() if key != "request_count"} == {
        key: value for key, value in dp2.items() if key != "request_count"
    }


def _qwen_states(*, capped_math=False):
    states = {}
    value = {0: True}
    for model in ("qwen3_0p6b", "qwen3_1p7b"):
        for mode in ("native_soft", "hard_token"):
            for benchmark in ("math500", "aime2024"):
                outcomes = {
                    grader: {benchmark + "-0": dict(value)}
                    for grader in ("math_verify", "released_last_boxed")
                }
                for condition in CORE_CONDITIONS:
                    states[(model, mode, benchmark, condition)] = {
                        "sample_count": 1,
                        "outcomes": outcomes,
                        "count": 1,
                        "capped_or_all_soft": int(
                            capped_math
                            and mode == "native_soft"
                            and benchmark == "math500"
                        ),
                    }
    return states


def test_qwen3_pass1_rows_have_no_pass8_fields():
    rows = _cell_metric_rows(
        _qwen_states(), smoke=False, study=QWEN3_STUDY_ID
    )
    assert rows
    assert all(row["samples_per_example"] == 1 for row in rows)
    assert all("pass_at_8" not in row for row in rows)
    assert all("pass_at_8_ci_low" not in row for row in rows)
    flattened = cli._flatten_wandb(rows, [], [], [])
    assert not any("pass_at_8" in key for key in flattened)


def test_qwen3_comparisons_have_registered_directions_and_math_only_gate():
    rows = _comparison_rows(
        _qwen_states(capped_math=True),
        smoke=False,
        study=QWEN3_STUDY_ID,
    )
    assert rows
    assert {row["estimand"] for row in rows} == {"pass_at_1"}
    assert "sdpg_matched_minus_sdft_matched" in {
        row["comparison"] for row in rows
    }
    assert "soft_thinking_minus_discrete_token_cot" in {
        row["comparison"] for row in rows
    }
    assert "qwen3_1p7b_minus_qwen3_0p6b" in {
        row["comparison"] for row in rows
    }
    math_soft = [
        row
        for row in rows
        if row["benchmark"] == "math500"
        and row["treatment_inference_mode"] == "native_soft"
    ]
    assert math_soft and all(
        row["treatment_boundary_gate_applied"]
        and row["treatment_boundary_gate_valid"] is False
        for row in math_soft
    )
    aime = [row for row in rows if row["benchmark"] == "aime2024"]
    assert aime and all(
        not row["treatment_boundary_gate_applied"]
        and row["treatment_boundary_gate_valid"] is None
        and row["comparison_boundary_gate_valid"] is None
        for row in aime
    )


def test_aime_graders_compare_integer_value_not_prompt_padding(monkeypatch):
    response = "The final answer is \\boxed{1}."
    assert _grade(
        response,
        "001",
        "released_last_boxed",
        "aime2024",
    )
    assert _grade(
        "The final answer is \\boxed{001}.",
        "001",
        "released_last_boxed",
        "aime2024",
    )
    assert not _grade(
        "The final answer is \\boxed{002}.",
        "001",
        "released_last_boxed",
        "aime2024",
    )
    observed = []

    def fake_math_verify(prediction, gold):
        observed.append((prediction, gold))
        return type("Grade", (), {"correct": True})()

    monkeypatch.setattr(cli, "math_verify_full_response_grade", fake_math_verify)
    assert _grade(
        response,
        "001",
        "math_verify",
        "aime2024",
    )
    assert observed == [(response, "1")]


def test_primary_native_soft_grade_rejects_invalid_boundary():
    class Record:
        response = "The final answer is \\boxed{1}."
        inference_mode = "native_soft"
        boundary_valid = False

    assert _grade(Record.response, "1", "released_last_boxed", "math500")
    assert not _primary_grade(Record(), "1", "released_last_boxed", "math500")
    Record.boundary_valid = True
    assert _primary_grade(Record(), "1", "released_last_boxed", "math500")


def test_generation_binding_accepts_only_the_authenticated_asset_data_and_model():
    manifest, assets, data_manifest = _generation_inputs()
    _validate_generation_binding(
        manifest,
        assets=assets,
        data_manifest=data_manifest,
        model_label="starting",
    )


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (
        ("asset_manifest_sha256", "wrong-asset"),
        ("data_manifest_sha256", "wrong-data"),
        ("model_label", "softgrpo"),
        ("model_tree_sha256", "wrong-model-tree"),
    ),
)
def test_generation_binding_rejects_wrong_authenticated_hashes_and_model(
    field, wrong_value
):
    manifest, assets, data_manifest = _generation_inputs()
    changed = dict(manifest)
    changed[field] = wrong_value
    with pytest.raises(ValueError, match=field):
        _validate_generation_binding(
            changed,
            assets=assets,
            data_manifest=data_manifest,
            model_label="starting",
        )


def test_generation_chunk_identity_seals_manifest_and_ordered_examples():
    manifest, _, _ = _generation_inputs()
    identity = _generation_chunk_identity(
        manifest,
        model_label="starting",
        inference_mode="native_soft",
        benchmark="math500",
        condition="no_demo",
        sample_index=3,
        chunk_index=4,
        example_ids=("math500-7", "math500-9"),
    )
    assert identity == {
        "generation_manifest_sha256": hashlib.sha256(
            canonical_json_bytes(manifest)
        ).hexdigest(),
        "model_label": "starting",
        "mode": "native_soft",
        "benchmark": "math500",
        "condition": "no_demo",
        "sample_index": 3,
        "chunk_index": 4,
        "example_ids": ["math500-7", "math500-9"],
    }
    changed = dict(manifest, model_tree_sha256="different")
    changed_identity = _generation_chunk_identity(
        changed,
        model_label="starting",
        inference_mode="native_soft",
        benchmark="math500",
        condition="no_demo",
        sample_index=3,
        chunk_index=4,
        example_ids=("math500-9", "math500-7"),
    )
    assert changed_identity["generation_manifest_sha256"] != identity[
        "generation_manifest_sha256"
    ]
    assert changed_identity["example_ids"] != identity["example_ids"]


def test_completion_record_identity_accepts_exact_ordered_chunk():
    records = [_completion("math500-0", replay_row=0), _completion("math500-1", replay_row=1)]
    _validate_completion_records(
        records,
        model_label="starting",
        inference_mode="hard_token",
        benchmark="math500",
        condition="no_demo",
        sample_index=0,
        example_ids=("math500-0", "math500-1"),
    )


@pytest.mark.parametrize(
    ("override", "error_field"),
    (
        ({"model_label": "softgrpo"}, "model_label"),
        ({"inference_mode": "native_soft"}, "inference_mode"),
        ({"benchmark": "gsm8k_test"}, "benchmark"),
        ({"condition": "sdft_matched"}, "condition"),
        ({"sample_index": 1}, "sample_index"),
        ({"example_ids": ("math500-wrong",)}, "example_id"),
    ),
)
def test_completion_record_identity_rejects_wrong_cell_sample_or_id(
    override, error_field
):
    values = {
        "model_label": "starting",
        "inference_mode": "hard_token",
        "benchmark": "math500",
        "condition": "no_demo",
        "sample_index": 0,
        "example_ids": ("math500-0",),
    }
    values.update(override)
    with pytest.raises(RuntimeError, match=error_field):
        _validate_completion_records([_completion()], **values)


def test_completion_record_identity_rejects_wrong_replay_row_and_order():
    with pytest.raises(RuntimeError, match="replay_row"):
        _validate_completion_records(
            [_completion(replay_row=7)],
            model_label="starting",
            inference_mode="hard_token",
            benchmark="math500",
            condition="no_demo",
            sample_index=0,
            example_ids=("math500-0",),
        )
    records = [_completion("math500-1", replay_row=0), _completion("math500-0", replay_row=1)]
    with pytest.raises(RuntimeError, match="example_id"):
        _validate_completion_records(
            records,
            model_label="starting",
            inference_mode="hard_token",
            benchmark="math500",
            condition="no_demo",
            sample_index=0,
            example_ids=("math500-0", "math500-1"),
        )


def test_replay_identity_accepts_exact_source_major_condition_order():
    sources = [
        _completion("math500-0", inference_mode="native_soft", replay_row=0),
        _completion("math500-1", inference_mode="native_soft", replay_row=1),
    ]
    records = [
        _replay_record(source, condition)
        for source in sources
        for condition in CORE_CONDITIONS
    ]
    _validate_replay_records(
        records,
        source_records=sources,
        model_label="starting",
        benchmark="math500",
        sample_index=0,
    )


def test_replay_identity_rejects_order_drift_and_wrong_source_identity():
    source = _completion(inference_mode="native_soft")
    records = [_replay_record(source, condition) for condition in CORE_CONDITIONS]
    records[1], records[2] = records[2], records[1]
    with pytest.raises(RuntimeError, match="prompted_condition"):
        _validate_replay_records(
            records,
            source_records=[source],
            model_label="starting",
            benchmark="math500",
            sample_index=0,
        )

    records = [_replay_record(source, condition) for condition in CORE_CONDITIONS]
    with pytest.raises(RuntimeError, match="model_label"):
        _validate_replay_records(
            records,
            source_records=[source],
            model_label="softgrpo",
            benchmark="math500",
            sample_index=0,
        )


def test_replay_identity_requires_zero_slot_exclusion_to_match_source():
    zero_source = replace(
        _completion(),
        inference_mode="native_soft",
        response_token_count=2,
        latent_token_count=0,
        hard_token_count=2,
        mixture_entropy_mean=None,
        top1_weight_mean=None,
        soft_hard_agreement=None,
        stored_mixture_reconstruction_abs_error_max=None,
    )
    excluded = [
        _replay_record(zero_source, condition, excluded=True)
        for condition in CORE_CONDITIONS
    ]
    _validate_replay_records(
        excluded,
        source_records=[zero_source],
        model_label="starting",
        benchmark="math500",
        sample_index=0,
    )
    object.__setattr__(excluded[0], "replay_exclusion_reason", None)
    with pytest.raises(RuntimeError, match="replay exclusion"):
        _validate_replay_records(
            excluded,
            source_records=[zero_source],
            model_label="starting",
            benchmark="math500",
            sample_index=0,
        )


def test_aggregate_excludes_invalid_soft_boundary_from_overlap_but_counts_copy(
    tmp_path, monkeypatch
):
    root = tmp_path / "assets"
    generation_root = tmp_path / "generation"
    report_root = tmp_path / "reports"
    example = ICLEvaluationExample(
        example_id="math500-0",
        benchmark="math500",
        source_index=0,
        question="What is 40 + 2?",
        gold_cot="Add forty and two.",
        gold_answer="42",
    )
    cell = ICLMatrixCell(
        model_label="starting",
        inference_mode="native_soft",
        condition="sdft_matched",
        benchmark="math500",
        subset="full",
        example_count=1,
        sample_count=1,
    )
    assets = {
        "content_sha256": "asset-manifest-sha",
        "models": {"starting": {"tree_sha256": "starting-tree-sha"}},
    }
    data_manifest = {"content_sha256": "data-manifest-sha"}
    provenance = {"sealed": True}
    manifest, _, _ = _generation_inputs()
    manifest.update(
        {
            "mode": "native_soft",
            "chunk_size": 1,
            "source_provenance": provenance,
        }
    )
    record = CompletionRecord(
        model_label="starting",
        inference_mode="native_soft",
        benchmark="math500",
        condition="sdft_matched",
        example_id=example.example_id,
        sample_index=0,
        request_seed=request_seed("math500", example.example_id, 0),
        response="Add forty and two. The final answer is \\boxed{42}",
        response_token_count=2,
        finish_reason="length",
        capped=True,
        latent_token_count=2,
        hard_token_count=0,
        close_tag=False,
        soft_to_hard=False,
        all_soft=True,
        boxed_answer=True,
        boundary_valid=False,
        mixture_entropy_mean=0.2,
        top1_weight_mean=0.9,
        soft_hard_agreement=1.0,
        stored_mixture_reconstruction_abs_error_max=0.0,
        replay_row=0,
    )
    trajectory = TrajectoryMetadata(
        response_token_ids=(10, 11),
        latent_support_ids=((10, 12, 13, 14, 15), (11, 12, 13, 14, 15)),
        latent_perturbed_logits=((3.0, 2.0, 1.0, 0.0, -1.0),) * 2,
        latent_gumbel_noise=((0.1, 0.2, 0.3, 0.4, 0.5),) * 2,
    )
    chunk_key = "starting/native_soft/math500/sdft_matched/sample_00/chunk_00000"
    identity = _generation_chunk_identity(
        manifest,
        model_label="starting",
        inference_mode="native_soft",
        benchmark="math500",
        condition="sdft_matched",
        sample_index=0,
        chunk_index=0,
        example_ids=(example.example_id,),
    )
    AtomicChunkStore(generation_root).commit(
        chunk_key, [record], [trajectory], identity=identity
    )
    completion_path = generation_root / "starting" / "native_soft" / "completion.json"
    completion_path.parent.mkdir(parents=True, exist_ok=True)
    completion_path.write_text("{}", encoding="utf-8")
    class _WandbRun:
        def __init__(self):
            self.summary = {}

        def log(self, *_args, **_kwargs):
            pass

        def log_artifact(self, *_args, **_kwargs):
            pass

        def finish(self):
            pass

    monkeypatch.setattr(cli, "verify_icl_assets", lambda _root: assets)
    monkeypatch.setattr(cli, "source_provenance", lambda: provenance)
    monkeypatch.setattr(
        cli,
        "load_icl_dataset",
        lambda _root: (
            [example],
            {"math500": {example.example_id: example.example_id}},
            {"math500": []},
            data_manifest,
        ),
    )
    monkeypatch.setattr(
        cli,
        "_generation_inventory",
        lambda _root, smoke: ({("starting", "native_soft"): manifest}, [cell]),
    )
    monkeypatch.setattr(cli, "_primary_grade", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(cli, "_cell_metric_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cli, "_comparison_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cli, "init_online_wandb", lambda **_kwargs: _WandbRun())
    cli.run_aggregate(
        argparse.Namespace(
            root=str(root),
            generation_dir=str(generation_root),
            output_dir=str(report_root),
            smoke=True,
        )
    )
    report = json.loads((report_root / "report.json").read_text(encoding="utf-8"))
    diagnostic = report["diagnostics"][0]
    assert diagnostic["demonstrated_answer_copy_count"] == 1
    assert diagnostic["demonstrated_answer_copy_rate"] == 1.0
    assert diagnostic["rationale_overlap_valid_count"] == 0
    assert diagnostic["rationale_overlap_invalid_boundary_excluded_count"] == 1
    assert diagnostic["rationale_overlap_f1_mean"] is None


def test_mechanism_cells_are_not_registerable_in_the_reduced_study():
    with pytest.raises(ValueError, match="unregistered ICL prompt condition"):
        ICLMatrixCell(
            model_label="starting",
            inference_mode="native_soft",
            condition="sdft_answer_only",
            benchmark="math500",
            subset="mechanism",
            example_count=128,
            sample_count=8,
        )


def test_public_cli_exposes_all_four_commands_and_output_overrides():
    parser = build_parser()
    prepare = parser.parse_args(["prepare", "--root", "/scratch/root", "--cache-dir", "/scratch/cache"])
    assert prepare.command == "prepare"
    generate = parser.parse_args(
        [
            "generate",
            "--root",
            "/scratch/root",
            "--model",
            "starting",
            "--mode",
            "native_soft",
            "--output-dir",
            "/scratch/smoke/generation",
            "--smoke",
        ]
    )
    assert generate.smoke and generate.output_dir.endswith("generation")
    assert generate.tensor_parallel_size == 1
    assert generate.data_parallel_size == 1
    assert generate.chunk_size == 64
    assert generate.queue_size is None
    replay = parser.parse_args(
        [
            "replay",
            "--root",
            "/scratch/root",
            "--model",
            "softgrpo",
            "--generation-dir",
            "/scratch/smoke/generation",
            "--output-dir",
            "/scratch/smoke/replay",
        ]
    )
    assert replay.generation_dir.endswith("generation")
    aggregate = parser.parse_args(
        [
            "aggregate",
            "--root",
            "/scratch/root",
            "--generation-dir",
            "/scratch/smoke/generation",
            "--output-dir",
            "/scratch/smoke/reports",
            "--smoke",
        ]
    )
    assert aggregate.smoke and aggregate.output_dir.endswith("reports")


def test_qwen3_cli_selects_both_models_and_modes_and_disables_replay(monkeypatch):
    parser = build_parser()
    for model in ("qwen3_0p6b", "qwen3_1p7b"):
        for mode in ("native_soft", "hard_token"):
            args = parser.parse_args(
                [
                    "generate",
                    "--study",
                    QWEN3_STUDY_ID,
                    "--root",
                    "/scratch/root",
                    "--model",
                    model,
                    "--mode",
                    mode,
                    "--smoke",
                ]
            )
            cells = cli._selected_cells(args)
            assert len(cells) == 6
            assert {cell.sample_count for cell in cells} == {1}
            assert {cell.example_count for cell in cells} == {16}

    monkeypatch.delenv("SLURM_GPUS_ON_NODE", raising=False)
    with pytest.raises(ValueError, match="does not schedule latent replay"):
        cli.main(
            [
                "replay",
                "--study",
                QWEN3_STUDY_ID,
                "--root",
                "/scratch/root",
                "--model",
                "qwen3_0p6b",
            ]
        )


def test_generate_cli_binds_tp_times_dp_to_slurm_allocation(monkeypatch):
    observed = []
    monkeypatch.setattr(cli, "run_generate", lambda args: observed.append(args))
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "8")
    cli.main(
        [
            "generate",
            "--root",
            "/scratch/root",
            "--model",
            "starting",
            "--mode",
            "native_soft",
            "--tensor-parallel-size",
            "1",
            "--data-parallel-size",
            "8",
        ]
    )
    assert len(observed) == 1
    assert observed[0].tensor_parallel_size == 1
    assert observed[0].data_parallel_size == 8

    with pytest.raises(ValueError, match="queue"):
        cli.main(
            [
                "generate",
                "--root",
                "/scratch/root",
                "--model",
                "starting",
                "--mode",
                "native_soft",
                "--tensor-parallel-size",
                "1",
                "--data-parallel-size",
                "8",
                "--queue-size",
                "0",
            ]
        )

    with pytest.raises(RuntimeError, match="SLURM_GPUS_ON_NODE"):
        cli.main(
            [
                "generate",
                "--root",
                "/scratch/root",
                "--model",
                "starting",
                "--mode",
                "native_soft",
                "--tensor-parallel-size",
                "1",
                "--data-parallel-size",
                "4",
            ]
        )


def test_qwen3_throughput_benchmark_cli_is_distinct_and_step_scoped(monkeypatch):
    observed = []
    monkeypatch.setattr(cli, "run_generate", lambda args: observed.append(args))
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "2")
    monkeypatch.setenv("OPD_EXPECTED_VISIBLE_GPUS", "1")
    cli.main(
        [
            "generate",
            "--study",
            "qwen3_icl_pass1_v1",
            "--root",
            "/scratch/root",
            "--model",
            "qwen3_0p6b",
            "--mode",
            "native_soft",
            "--data-parallel-size",
            "1",
            "--max-running-requests",
            "16",
            "--queue-size",
            "32",
            "--throughput-benchmark",
        ]
    )
    assert len(observed) == 1
    assert observed[0].throughput_benchmark is True
    assert observed[0].smoke is False

    with pytest.raises(ValueError, match="mutually exclusive"):
        cli.main(
            [
                "generate",
                "--study",
                "qwen3_icl_pass1_v1",
                "--root",
                "/scratch/root",
                "--model",
                "qwen3_0p6b",
                "--mode",
                "native_soft",
                "--smoke",
                "--throughput-benchmark",
            ]
        )
    with pytest.raises(ValueError, match="all registered"):
        cli.main(
            [
                "generate",
                "--study",
                "qwen3_icl_pass1_v1",
                "--root",
                "/scratch/root",
                "--model",
                "qwen3_0p6b",
                "--mode",
                "native_soft",
                "--benchmarks",
                "math500",
                "--throughput-benchmark",
            ]
        )


def test_benchmark_report_cli_routes_required_evidence(monkeypatch):
    observed = []
    monkeypatch.setattr(cli, "run_benchmark_report", lambda args: observed.append(args))
    cli.main(
        [
            "benchmark-report",
            "--study",
            "qwen3_icl_pass1_v1",
            "--root",
            "/scratch/assets",
            "--dp1-generation-dir",
            "/scratch/dp1",
            "--dp2-generation-dir",
            "/scratch/dp2",
            "--output-dir",
            "/scratch/report",
        ]
    )
    assert len(observed) == 1
    assert observed[0].dp1_generation_dir == "/scratch/dp1"
    assert observed[0].dp2_generation_dir == "/scratch/dp2"

def test_generate_cli_rejects_removed_benchmark_and_shuffled_condition():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "generate",
                "--root",
                "/scratch/root",
                "--model",
                "starting",
                "--mode",
                "native_soft",
                "--benchmarks",
                "gsm8k_test",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "generate",
                "--root",
                "/scratch/root",
                "--model",
                "starting",
                "--mode",
                "native_soft",
                "--conditions",
                "sdft_shuffled",
            ]
        )


def test_difference_in_differences_keeps_pass_estimator_identity(monkeypatch):
    bootstrap = {
        "difference": 0.0,
        "ci_low": -0.1,
        "ci_high": 0.1,
        "confidence": 0.95,
        "resamples": 10_000,
        "bootstrap_seed": 11,
        "example_count": 1,
    }

    monkeypatch.setattr(
        cli,
        "paired_bootstrap_difference",
        lambda *_args, **_kwargs: dict(bootstrap),
    )
    monkeypatch.setattr(
        cli,
        "paired_bootstrap_difference_in_differences",
        lambda *_args, **_kwargs: {
            **bootstrap,
            "estimand": (
                "(post_treatment-post_control)-"
                "(start_treatment-start_control)"
            ),
        },
    )

    states = {}
    values = {index: index % 2 == 0 for index in range(8)}
    for model, mode in (
        ("starting", "native_soft"),
        ("starting", "hard_token"),
        ("softgrpo", "native_soft"),
    ):
        for benchmark in ("math500", "aime2024"):
            outcomes = {
                grader: {benchmark + "-0": dict(values)}
                for grader in ("math_verify", "released_last_boxed")
            }
            for condition in CORE_CONDITIONS:
                states[(model, mode, benchmark, condition)] = {
                    "sample_count": 8,
                    "outcomes": outcomes,
                    "count": 8,
                    "capped_or_all_soft": 0,
                }

    rows = cli._comparison_rows(states, smoke=False)
    did = [row for row in rows if row["model_label"] == "difference_in_differences"]
    assert len(did) == 16
    assert {row["estimand"] for row in did} == {"pass_at_1", "pass_at_8"}
    assert {row["contrast_definition"] for row in did} == {
        "(post_treatment-post_control)-(start_treatment-start_control)"
    }
    assert all(row["comparison_boundary_gate_valid"] for row in did)
    flattened = cli._flatten_wandb([], [], did, [])
    assert sum(key.endswith("/difference") for key in flattened) == len(did)
    assert sum(
        key.endswith("/comparison_boundary_gate_valid") for key in flattened
    ) == len(did)


def test_from_getgo_assessment_reports_both_boundary_gates_without_omnibus_rule():
    metrics = []
    diagnostics = []
    comparisons = []
    for benchmark in ("math500", "aime2024"):
        diagnostics.append(
            {
                "model_label": "starting",
                "inference_mode": "native_soft",
                "benchmark": benchmark,
                "condition": "no_demo",
                "boundary_gate_valid": benchmark == "aime2024",
            }
        )
        for family in ("sdft", "sdpg"):
            condition = family + "_matched"
            metrics.append(
                {
                    "model_label": "starting",
                    "inference_mode": "native_soft",
                    "benchmark": benchmark,
                    "condition": condition,
                    "grader": "math_verify",
                    "pass_at_1": 0.5,
                }
            )
            diagnostics.append(
                {
                    "model_label": "starting",
                    "inference_mode": "native_soft",
                    "benchmark": benchmark,
                    "condition": condition,
                    "boundary_gate_valid": True,
                }
            )
            comparisons.append(
                {
                    "model_label": "starting",
                    "inference_mode": "native_soft",
                    "benchmark": benchmark,
                    "comparison": family + "_matched_minus_no_demo",
                    "grader": "math_verify",
                    "estimand": "pass_at_1",
                    "ci_low": 0.01,
                }
            )

    rows = cli._from_getgo_assessment(metrics, diagnostics, comparisons)
    math_rows = [row for row in rows if row["benchmark"] == "math500"]
    aime_rows = [row for row in rows if row["benchmark"] == "aime2024"]
    assert all(row["matched_boundary_gate_valid"] for row in rows)
    assert all(not row["no_demo_boundary_gate_valid"] for row in math_rows)
    assert all(row["no_demo_boundary_gate_valid"] for row in aime_rows)
    assert all(row["positive_paired_95ci"] for row in rows)
    assert all("overall_criterion" not in row for row in rows)
