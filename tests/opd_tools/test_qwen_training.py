"""Isolated Qwen3 preparation, deterministic workload, and recipe contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from opd_tools import qwen_training as training
from opd_tools.records import MathExample


class Tokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, enable_thinking):
        assert add_generation_prompt is True and tokenize is False and enable_thinking is True
        return "<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text in ("<think>", "</think>"):
            return [151667 if text == "<think>" else 151668]
        return [ord(char) for char in text] + [151667]

    def decode(self, ids, skip_special_tokens=False):
        return {151667: "<think>", 151668: "</think>"}[ids[0]]


def _examples(split, count):
    return [MathExample(
        example_id=f"{split}-{index}", source_index=index,
        question=f"Question {index}: " + "x " * (index % 12),
        gold_solution=f"Compute carefully. \\boxed{{{index}}}",
        gold_cot="Compute carefully.", gold_answer=str(index),
        subject=f"subject-{index % 3}", level=f"Level {index % 5 + 1}", split=split,
    ) for index in range(count)]


@pytest.fixture
def staged_assets(monkeypatch, tmp_path):
    import huggingface_hub
    import transformers

    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        model = Path(kwargs["local_dir"])
        model.mkdir(parents=True)
        (model / "config.json").write_text(json.dumps({"model_type": "qwen3", "max_position_embeddings": 40960}))
        (model / "tokenizer_config.json").write_text("{}")
        (model / "tokenizer.json").write_text("{}")
        (model / "model.safetensors").write_bytes(b"isolated mock weights")
        (model / ".cache").mkdir()
        return str(model)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", lambda self, **kwargs: SimpleNamespace(sha=training.MODEL_REVISION))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    splits = {"train": _examples("train", 160), "validation": _examples("validation", 128)}
    monkeypatch.setattr(training, "MATH_TRAIN_SIZE", 160)
    monkeypatch.setattr(training, "MATH_VALIDATION_SIZE", 128)
    monkeypatch.setattr(training, "load_pinned_math_train", lambda cache: [])
    monkeypatch.setattr(training, "prepare_math_example_splits", lambda rows: (splits, SimpleNamespace(to_dict=lambda: {"mock": True})))
    monkeypatch.setattr(training, "ordered_example_ids_sha256", lambda ids: training.MATH_VALIDATION_IDS_SHA256)
    return tmp_path / "training", tmp_path / "cache", calls, splits


def test_profile_has_exact_model_and_existing_math_split():
    assert training.PROFILE_ID == "qwen3-training-benchmark-v1"
    assert training.MODEL == {"id": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca"}
    assert (training.MATH_TRAIN_SIZE, training.MATH_VALIDATION_SIZE) == (6985, 512)


def test_prepare_publishes_only_training_assets_and_authenticates_resume(staged_assets):
    root, cache, calls, _ = staged_assets
    result = training.prepare(root, cache)
    assert result["model_dir"] == str(root / "model")
    assert result["data_dir"] == str(root / "data")
    assert set(path.name for path in (root / "data").iterdir()) == {*training.DATA_FILES.values(), "manifest.json"}
    assert len(calls) == 1
    assert calls[0]["repo_id"] == training.MODEL_ID
    assert calls[0]["revision"] == training.MODEL_REVISION
    assert not (root / "model/.cache").exists()
    assert training.prepare(root, cache) == result
    assert len(calls) == 1
    model_manifest = json.loads((root / "model/manifest.json").read_text())
    assert model_manifest["model"]["resolved_revision"] == training.MODEL_REVISION
    assert model_manifest["manifest_content_sha256"]
    (root / "model/model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="model inventory"):
        training.verify(root)


def test_selection_is_deterministic_disjoint_and_recomputed_on_verify(staged_assets):
    root, cache, _, _ = staged_assets
    training.prepare(root, cache)
    selection = json.loads((root / "selection.json").read_text())
    first, second = selection["train_batches"]
    assert len(first) == len(second) == len(set(first)) == len(set(second)) == 64
    assert not set(first) & set(second)
    assert len(set(selection["validation_indices"])) == 128
    assert selection == training._seal(training.build_selection(selection["populations"]))
    for split in selection["populations"].values():
        assert {row["prompt_length_quartile"] for row in split} == {0, 1, 2, 3}
    # Even a self-consistently resealed selection must obey the selector.
    selection["train_batches"][0][0], selection["train_batches"][0][1] = first[1], first[0]
    selection.pop("manifest_content_sha256")
    selection = training._write_manifest(root / "selection.json", selection)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest.pop("manifest_content_sha256")
    manifest["selection_content_sha256"] = selection["manifest_content_sha256"]
    training._write_manifest(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="deterministic contract"):
        training.verify(root)


def test_prepare_rejects_foreign_roots_without_downloading(staged_assets):
    root, cache, calls, _ = staged_assets
    root.mkdir()
    (root / "completed_icl_report.json").write_text("historical result")
    with pytest.raises(ValueError, match="manifest"):
        training.prepare(root, cache)
    assert not calls
    assert (root / "completed_icl_report.json").read_text() == "historical result"


def test_failed_preflight_never_publishes_a_partial_root(staged_assets, monkeypatch):
    root, cache, _, _ = staged_assets
    monkeypatch.setattr(training, "MAX_PROMPT_TOKENS", 1)
    with pytest.raises(ValueError, match="student prompt exceeds"):
        training.prepare(root, cache)
    assert not root.exists()
    assert not list(root.parent.glob("." + root.name + ".*/model"))


def test_preflight_counts_every_student_and_teacher_prompt_and_rejects_wrong_markers():
    splits = {"train": _examples("train", 2), "validation": _examples("validation", 1)}
    report, population = training.preflight_prompts(Tokenizer(), splits)
    assert report["counts"] == {"train": 2, "validation": 1}
    assert report["maxima"]["teacher_prompt_tokens"] > report["maxima"]["student_prompt_tokens"]
    assert [row["index"] for row in population["train"]] == [0, 1]
    tokenizer = Tokenizer()
    tokenizer.decode = lambda *args, **kwargs: "wrong marker"
    with pytest.raises(RuntimeError, match="round-trip"):
        training.preflight_prompts(tokenizer, splits)


@pytest.mark.parametrize("objective,group,beta,mode", [("standalone", 1, 1.0, "standalone"), ("hybrid", 8, 0.001, "auxiliary")])
def test_profile_overrides_preserve_recipe_horizon_and_use_isolated_qwen_identity(tmp_path, objective, group, beta, mode):
    overrides = training.profile_overrides(objective, 2, tmp_path / "assets", tmp_path / "run")
    values = {}
    for override in overrides:
        key, value = override.split("=", 1)
        try:
            values[key] = json.loads(value)
        except json.JSONDecodeError:
            values[key] = value
    assert values["algorithm.opd.mode"] == mode
    assert values["algorithm.opd.beta_base"] == beta
    assert values["actor_rollout_ref.rollout.n"] == group
    assert values["trainer.total_training_steps"] is None
    assert values["trainer.total_epochs"] == 1
    assert values["trainer.max_rollout_iterations_per_invocation"] == 3
    assert values["data.train_batch_size"] == 64
    assert values["actor_rollout_ref.rollout.engine_context_length"] == 12000
    assert values["actor_rollout_ref.rollout.tensor_model_parallel_size"] == 1
    assert values["algorithm.opd.prompt_profile"] == values["data.prompt_profile"] == training.PROFILE_ID
    assert values["trainer.training_profile"] == training.PROFILE_ID
    assert "qwen3_0p6b" in values["trainer.experiment_name"]


@pytest.mark.parametrize("objective", ["standalone", "hybrid"])
@pytest.mark.parametrize("gpus", [1, 2])
@pytest.mark.parametrize("phase", ["calibration", "pilot"])
@pytest.mark.parametrize("replay_backend", ["disabled", "native_fa3_v1"])
def test_full_profiles_compose_with_real_hydra_and_phase_overrides(tmp_path, objective, gpus, phase, replay_backend):
    hydra = pytest.importorskip("hydra")
    from omegaconf import OmegaConf
    from verl.opd.config import OPDConfig

    config_dir = Path(training.__file__).resolve().parents[1] / "verl-0.4.x/verl/trainer/config"
    overrides = training.profile_overrides(objective, gpus, tmp_path / "assets", tmp_path / "run", replay_backend=replay_backend)
    # The child-phase overrides applied by CellRunner must replace the common
    # settings without shortening the production optimization horizon.
    overrides += [
        "actor_rollout_ref.rollout.dispatch_mode=bounded_async",
        "actor_rollout_ref.rollout.max_running_requests=16",
        "actor_rollout_ref.rollout.async_queue_size=32",
        f"++trainer.training_benchmark_mode={phase}",
        "++trainer.training_benchmark_variant=bounded_async16",
        "++trainer.training_benchmark_batches=" + ("[0]" if phase == "calibration" else "[]"),
        f"++trainer.training_benchmark_selection={tmp_path / 'assets/selection.json'}",
        f"++trainer.training_benchmark_output={tmp_path / 'measurement.json'}",
        "trainer.val_before_train=false", "trainer.test_freq=-1", "trainer.save_freq=-1",
        "trainer.log_val_generations=0", "trainer.resume_mode=disable",
        "trainer.max_rollout_iterations_per_invocation=3",
        "trainer.rollout_integrity.full_dose_gradient_gate_enabled=false",
    ]
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = hydra.compose(config_name="ppo_trainer", overrides=overrides)
    opd = OPDConfig.from_mapping(OmegaConf.to_container(config.algorithm.opd, resolve=True))
    assert opd.prompt_profile == config.data.prompt_profile == config.trainer.training_profile == training.PROFILE_ID
    assert str(opd.mode) == ("standalone" if objective == "standalone" else "auxiliary")
    assert config.actor_rollout_ref.rollout.n == (1 if objective == "standalone" else 8)
    assert config.actor_rollout_ref.rollout.engine_context_length == 12000
    assert config.actor_rollout_ref.model.qwen_replay_backend == replay_backend
    assert config.actor_rollout_ref.rollout.qwen_replay_backend == replay_backend
    assert config.trainer.total_training_steps is None
    assert config.trainer.total_epochs == 1
    assert config.trainer.max_rollout_iterations_per_invocation == 3
    assert config.trainer.training_benchmark_mode == phase
    assert config.trainer.n_gpus_per_node == gpus
