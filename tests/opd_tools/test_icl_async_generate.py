import argparse
import json
import sys
import types

import pytest

import opd_tools.icl_eval as cli
from opd_tools.icl import (
    ICLEvaluationExample,
    ICLMatrixCell,
    QWEN3_STUDY_ID,
)
from opd_tools.icl_runtime import AtomicChunkStore


MODEL_LABEL = "qwen3_0p6b"
MODE = "hard_token"
MATH_KEY = (
    f"{MODEL_LABEL}/{MODE}/math500/no_demo/sample_00/chunk_00000"
)
AIME_KEY = (
    f"{MODEL_LABEL}/{MODE}/aime2024/sdpg_matched/sample_00/chunk_00000"
)


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": True,
        }
        user_content = messages[0]["content"]
        return (
            "<|im_start|>user\n"
            + user_content
            + "<|im_end|>\n<|im_start|>assistant\n"
        )

    def encode(self, text, add_special_tokens=True):
        if text == "<think>":
            return [151667]
        if text == "</think>":
            return [151668]
        return list(range(max(1, len(text) // 32)))

    def decode(self, values, skip_special_tokens=False):
        assert not skip_special_tokens
        return {151667: "<think>", 151668: "</think>"}[values[0]]


class _AutoTokenizer:
    @classmethod
    def from_pretrained(cls, *_args, **kwargs):
        assert kwargs == {"local_files_only": True}
        return _Tokenizer()


class _AutoConfig:
    @classmethod
    def from_pretrained(cls, *_args, **kwargs):
        assert kwargs == {"local_files_only": True}
        return types.SimpleNamespace(
            max_position_embeddings=40960,
            num_key_value_heads=8,
        )


class _WandbRun:
    def __init__(self):
        self.summary = {}
        self.logged = []
        self.artifacts = []
        self.finished = False

    def log(self, values, *, step):
        self.logged.append((step, dict(values)))

    def log_artifact(self, path, **kwargs):
        self.artifacts.append((path, kwargs))

    def finish(self):
        self.finished = True


def _examples():
    return [
        ICLEvaluationExample(
            example_id="math500-0",
            benchmark="math500",
            source_index=0,
            question="MATH UNIQUE ZERO?",
            gold_cot="Reasoning zero.",
            gold_answer="10",
            subject="algebra",
            difficulty="1",
        ),
        ICLEvaluationExample(
            example_id="math500-1",
            benchmark="math500",
            source_index=1,
            question="MATH UNIQUE ONE?",
            gold_cot="Reasoning one.",
            gold_answer="11",
            subject="geometry",
            difficulty="2",
        ),
        ICLEvaluationExample(
            example_id="aime2024-0",
            benchmark="aime2024",
            source_index=0,
            question="AIME UNIQUE ZERO?",
            gold_cot="AIME reasoning zero.",
            gold_answer="012",
        ),
        ICLEvaluationExample(
            example_id="aime2024-1",
            benchmark="aime2024",
            source_index=1,
            question="AIME UNIQUE ONE?",
            gold_cot="AIME reasoning one.",
            gold_answer="013",
        ),
    ]


def _cells():
    return [
        ICLMatrixCell(
            model_label=MODEL_LABEL,
            inference_mode=MODE,
            condition="no_demo",
            benchmark="math500",
            subset="full",
            example_count=2,
            sample_count=1,
        ),
        ICLMatrixCell(
            model_label=MODEL_LABEL,
            inference_mode=MODE,
            condition="sdpg_matched",
            benchmark="aime2024",
            subset="full",
            example_count=2,
            sample_count=1,
        ),
    ]


def _args(root, output_root):
    return argparse.Namespace(
        study=QWEN3_STUDY_ID,
        root=str(root),
        output_dir=str(output_root),
        model=MODEL_LABEL,
        mode=MODE,
        tensor_parallel_size=1,
        data_parallel_size=2,
        chunk_size=2,
        max_running_requests=3,
        queue_size=7,
        gpu_memory_utilization=0.8,
        smoke=False,
        benchmarks=["all"],
        conditions=["all"],
    )


def _output(prompt, index):
    return {
        "text": f"completion[{index}]::{prompt}::\\boxed{{{index + 1}}}",
        "meta_info": {
            "finish_reason": {"type": "stop"},
            "output_token_logprobs": [
                (0.0, 100 + index),
                (0.0, 200 + index),
            ],
        },
    }


def _install_common_fakes(monkeypatch, root):
    output_root = root / "generation"
    examples = _examples()
    assets = {
        "content_sha256": "asset-sha",
        "model_specs": {
            MODEL_LABEL: {
                "id": "Qwen/Qwen3-0.6B",
                "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            }
        },
        "models": {
            MODEL_LABEL: {
                "tree_sha256": "model-tree-sha",
                "files": [
                    {
                        "path": "tokenizer_config.json",
                        "size": 1,
                        "sha256": "tokenizer-sha",
                    }
                ],
            }
        },
    }
    data_manifest = {"content_sha256": "data-sha"}
    shuffled = {
        benchmark: {example.example_id: example.example_id for example in examples}
        for benchmark in ("math500", "aime2024")
    }
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=_AutoTokenizer,
            AutoConfig=_AutoConfig,
        ),
    )
    monkeypatch.setattr(
        cli,
        "_verify_assets_for_study",
        lambda _root, _study: assets,
    )
    monkeypatch.setattr(
        cli,
        "load_icl_dataset",
        lambda _root: (
            examples,
            shuffled,
            {"math500": [], "aime2024": []},
            data_manifest,
        ),
    )
    monkeypatch.setattr(cli, "_selected_cells", lambda _args: _cells())
    monkeypatch.setattr(cli, "source_provenance", lambda: {"sealed": True})
    runs = []

    def initialize_wandb(**_kwargs):
        run = _WandbRun()
        runs.append(run)
        return run

    monkeypatch.setattr(cli, "init_online_wandb", initialize_wandb)
    return output_root, runs


def _manifest_path(store, key):
    return store.paths(key)[2]


def test_run_generate_maps_out_of_order_cells_and_resumes_without_generation(
    tmp_path, monkeypatch
):
    output_root, runs = _install_common_fakes(monkeypatch, tmp_path)
    store = AtomicChunkStore(output_root)
    instances = []

    class _OutOfOrderEngine:
        def __init__(self, **_kwargs):
            self.generate_calls = 0
            self.shutdown_count = 0
            instances.append(self)

        def generate_as_completed(self, prompts, seeds, *, queue_size):
            self.generate_calls += 1
            assert queue_size == 7
            assert len(prompts) == len(seeds) == 4
            order = (0, 2, 3, 1)
            for ordinal, index in enumerate(order):
                if ordinal == 1:
                    assert not _manifest_path(store, MATH_KEY).exists()
                elif ordinal == 2:
                    assert not _manifest_path(store, AIME_KEY).exists()
                elif ordinal == 3:
                    assert _manifest_path(store, AIME_KEY).is_file()
                    assert not _manifest_path(store, MATH_KEY).exists()
                yield (
                    index,
                    _output(prompts[index], index),
                    100.0 + index,
                    200.0 + ordinal,
                )
            assert _manifest_path(store, MATH_KEY).is_file()
            assert _manifest_path(store, AIME_KEY).is_file()

        def shutdown(self):
            self.shutdown_count += 1

    monkeypatch.setattr(cli, "ReleasedSofTGRPOEngine", _OutOfOrderEngine)
    args = _args(tmp_path, output_root)
    cli.run_generate(args)

    assert len(instances) == 1
    assert instances[0].generate_calls == 1
    assert instances[0].shutdown_count == 1
    assert runs[0].finished

    generation_manifest = json.loads(
        (output_root / MODEL_LABEL / MODE / "generation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert generation_manifest["request_queue_size"] == 7
    assert generation_manifest["max_running_requests"] == 3
    assert generation_manifest["parallelism"]["data_parallel_size"] == 2

    math_records, _ = store.load(MATH_KEY)
    aime_records, _ = store.load(AIME_KEY)
    assert [record.example_id for record in math_records] == [
        "math500-0",
        "math500-1",
    ]
    assert [record.example_id for record in aime_records] == [
        "aime2024-0",
        "aime2024-1",
    ]
    assert "MATH UNIQUE ZERO?" in math_records[0].response
    assert "MATH UNIQUE ONE?" in math_records[1].response
    assert "AIME UNIQUE ZERO?" in aime_records[0].response
    assert "AIME UNIQUE ONE?" in aime_records[1].response
    assert [record.replay_row for record in math_records] == [0, 1]
    assert [record.replay_row for record in aime_records] == [0, 1]

    throughput_path = output_root / MODEL_LABEL / MODE / "throughput.json"
    throughput = json.loads(throughput_path.read_text(encoding="utf-8"))
    assert throughput["protocol"] == "opd-icl-async-throughput-v1"
    assert throughput["queue_size"] == 7
    assert throughput["request_count"] == 4
    assert throughput["response_tokens"] == 8
    assert throughput["eligible_for_allocation"] is True
    assert [row["request_index"] for row in throughput["rows"]] == [0, 1, 2, 3]
    assert [row["example_id"] for row in throughput["rows"]] == [
        "math500-0",
        "math500-1",
        "aime2024-0",
        "aime2024-1",
    ]
    assert throughput["cleanup"]["engine_shutdown_seconds"] >= 0.0
    completion_path = output_root / MODEL_LABEL / MODE / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert completion["chunks_committed"] == 2
    assert completion["throughput_sha256"]
    completion_bytes = completion_path.read_bytes()
    throughput_bytes = throughput_path.read_bytes()

    chunk_hashes = {
        key: _manifest_path(store, key).read_bytes()
        for key in (MATH_KEY, AIME_KEY)
    }
    cli.run_generate(args)
    assert len(instances) == 2
    assert instances[1].generate_calls == 0
    assert instances[1].shutdown_count == 1
    assert runs[1].finished
    assert {
        key: _manifest_path(store, key).read_bytes()
        for key in (MATH_KEY, AIME_KEY)
    } == chunk_hashes
    assert completion_path.read_bytes() == completion_bytes
    assert throughput_path.read_bytes() == throughput_bytes

    changed_queue = argparse.Namespace(**vars(args))
    changed_queue.queue_size = 8
    with pytest.raises(RuntimeError, match="resume manifest"):
        cli.run_generate(changed_queue)
    assert len(instances) == 2


def test_run_generate_failure_cancels_and_publishes_only_complete_chunks(
    tmp_path, monkeypatch
):
    output_root, runs = _install_common_fakes(monkeypatch, tmp_path)
    store = AtomicChunkStore(output_root)
    instances = []

    class _FailingEngine:
        def __init__(self, **_kwargs):
            self.cancelled = False
            self.shutdown_count = 0
            instances.append(self)

        def generate_as_completed(self, prompts, seeds, *, queue_size):
            assert queue_size == 7
            assert len(prompts) == len(seeds) == 4
            try:
                for ordinal, index in enumerate((2, 3, 0)):
                    yield (
                        index,
                        _output(prompts[index], index),
                        100.0 + index,
                        200.0 + ordinal,
                    )
                raise RuntimeError("synthetic async failure")
            finally:
                self.cancelled = True

        def shutdown(self):
            self.shutdown_count += 1

    monkeypatch.setattr(cli, "ReleasedSofTGRPOEngine", _FailingEngine)
    with pytest.raises(RuntimeError, match="synthetic async failure"):
        cli.run_generate(_args(tmp_path, output_root))

    assert len(instances) == 1
    assert instances[0].cancelled
    assert instances[0].shutdown_count == 1
    assert runs[0].finished
    assert runs[0].summary["generation/completed"] is False
    assert _manifest_path(store, AIME_KEY).is_file()
    assert not _manifest_path(store, MATH_KEY).exists()
    assert not (output_root / MODEL_LABEL / MODE / "completion.json").exists()
    assert not (output_root / MODEL_LABEL / MODE / "throughput.json").exists()

    aime_manifest = _manifest_path(store, AIME_KEY).read_bytes()

    class _ResumeEngine:
        def __init__(self, **_kwargs):
            self.generate_calls = 0
            self.shutdown_count = 0

        def generate_as_completed(self, prompts, seeds, *, queue_size):
            self.generate_calls += 1
            assert queue_size == 7
            assert len(prompts) == len(seeds) == 2
            for index in (1, 0):
                yield index, _output(prompts[index], index), 300.0 + index, 400.0 + index

        def shutdown(self):
            self.shutdown_count += 1

    monkeypatch.setattr(cli, "ReleasedSofTGRPOEngine", _ResumeEngine)
    cli.run_generate(_args(tmp_path, output_root))
    assert _manifest_path(store, AIME_KEY).read_bytes() == aime_manifest
    assert _manifest_path(store, MATH_KEY).is_file()
    throughput = json.loads(
        (output_root / MODEL_LABEL / MODE / "throughput.json").read_text(
            encoding="utf-8"
        )
    )
    assert throughput["request_count"] == 4
    assert throughput["resumed"] is True
    assert throughput["eligible_for_allocation"] is True
    assert throughput["queue_occupancy"]["timing_session_count"] == 2
    assert len(
        {
            (row["condition"], row["benchmark"], row["example_id"])
            for row in throughput["rows"]
        }
    ) == 4
    assert (output_root / MODEL_LABEL / MODE / "completion.json").is_file()


def test_resource_monitor_stop_failure_still_finishes_wandb_and_engine(
    tmp_path, monkeypatch
):
    output_root, runs = _install_common_fakes(monkeypatch, tmp_path)
    instances = []

    class _Engine:
        def __init__(self, **_kwargs):
            self.shutdown_count = 0
            instances.append(self)

        def generate_as_completed(self, prompts, seeds, *, queue_size):
            assert len(prompts) == len(seeds) == 4
            for index in range(4):
                yield index, _output(prompts[index], index), 100.0 + index, 200.0 + index

        def shutdown(self):
            self.shutdown_count += 1

    class _FailingMonitor:
        def start(self):
            return self

        def stop(self):
            raise RuntimeError("synthetic telemetry stop failure")

    monkeypatch.setattr(cli, "ReleasedSofTGRPOEngine", _Engine)
    monkeypatch.setattr(
        cli,
        "ResourceMonitor",
        lambda **_kwargs: _FailingMonitor(),
    )

    with pytest.raises(RuntimeError, match="resource telemetry failed during cleanup"):
        cli.run_generate(_args(tmp_path, output_root))

    assert instances[0].shutdown_count == 1
    assert runs[0].finished
    assert "synthetic telemetry stop failure" in runs[0].summary[
        "system/resource_monitor_failure"
    ]
    assert not (output_root / MODEL_LABEL / MODE / "completion.json").exists()


def test_resource_monitor_failure_does_not_mask_generation_failure(
    tmp_path, monkeypatch
):
    output_root, runs = _install_common_fakes(monkeypatch, tmp_path)
    instances = []

    class _Engine:
        def __init__(self, **_kwargs):
            self.shutdown_count = 0
            instances.append(self)

        def generate_as_completed(self, prompts, seeds, *, queue_size):
            if False:  # pragma: no cover - make this a generator
                yield None
            raise ValueError("synthetic generation failure")

        def shutdown(self):
            self.shutdown_count += 1

    class _FailingMonitor:
        def start(self):
            return self

        def stop(self):
            raise RuntimeError("synthetic telemetry stop failure")

    monkeypatch.setattr(cli, "ReleasedSofTGRPOEngine", _Engine)
    monkeypatch.setattr(
        cli,
        "ResourceMonitor",
        lambda **_kwargs: _FailingMonitor(),
    )

    with pytest.raises(ValueError, match="synthetic generation failure") as captured:
        cli.run_generate(_args(tmp_path, output_root))

    assert instances[0].shutdown_count == 1
    assert runs[0].finished
    assert runs[0].summary["generation/completed"] is False
    assert any(
        "synthetic telemetry stop failure" in note
        for note in getattr(captured.value, "__notes__", ())
    )
