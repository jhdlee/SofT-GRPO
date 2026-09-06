"""Isolated, pinned Qwen3 training-benchmark assets and launch configuration."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .assets import _inventory, _validate_required_transformers_files
from .constants import (
    MATH_DATASET_CONFIG, MATH_DATASET_ID, MATH_DATASET_REVISION,
    MATH_SPLIT_SEED, MATH_TRAIN_SIZE, MATH_VALIDATION_SIZE,
    MATH_VALIDATION_IDS_SHA256,
)
from .data import load_pinned_math_train, ordered_example_ids_sha256, prepare_math_example_splits
from .manifest import canonical_sha256, file_sha256, ordered_records_sha256, validate_sealed_content, write_manifest_atomic
from .prepare import _write_parquet
from .records import MathExample, build_verl_training_row, render_sdpg_teacher_user_content, render_student_user_content
from .study import hydra_overrides


PROFILE_ID = "qwen3-training-benchmark-v1"
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL = {"id": MODEL_ID, "revision": MODEL_REVISION}
MAX_PROMPT_TOKENS = 2048
MAX_RESPONSE_TOKENS = 8192
MODEL_CONTEXT_LENGTH = 40960
ENGINE_CONTEXT_LENGTH = 12000
SELECTION_SEED = 11
DATA_FILES = {
    "train": "math_lighteval_train.parquet",
    "validation": "math_lighteval_validation.parquet",
}


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["manifest_content_sha256"] = canonical_sha256(result)
    return result


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    result = _seal(payload)
    write_manifest_atomic(path, result, validator=validate_sealed_content)
    return result


def _read_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing regular sealed manifest: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    validate_sealed_content(result)
    if result.get("profile_id") != PROFILE_ID:
        raise ValueError("asset belongs to a different training profile")
    return result


def _rank(namespace: str, *parts: Any) -> str:
    return canonical_sha256([PROFILE_ID, SELECTION_SEED, namespace, *parts])


def _select_stratified(
    population: Sequence[Mapping[str, Any]], count: int, namespace: str,
    excluded: Sequence[int] = (),
) -> list[int]:
    """Largest-remainder subject/level/length allocations with stable tie breaks."""

    excluded_set = set(excluded)
    strata = defaultdict(list)
    for row in population:
        if row["index"] not in excluded_set:
            key = (row["subject"], row["level"], row["prompt_length_quartile"])
            strata[key].append(row)
    total = sum(map(len, strata.values()))
    if count <= 0 or total < count:
        raise ValueError("stratified selection requires enough distinct examples")
    allocations = {key: count * len(rows) // total for key, rows in strata.items()}
    remainder = count - sum(allocations.values())
    order = sorted(strata, key=lambda key: (-(count * len(strata[key]) % total), _rank(namespace, key)))
    for key in order[:remainder]:
        allocations[key] += 1
    selected = []
    for key, rows in strata.items():
        ranked = sorted(rows, key=lambda row: _rank(namespace, key, row["example_id"]))
        selected.extend(int(row["index"]) for row in ranked[:allocations[key]])
    return sorted(selected, key=lambda index: _rank(namespace, "output-order", index))


def build_selection(populations: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Select two disjoint 64-prompt training batches and 128 validation rows."""

    catalog = {}
    for split in ("train", "validation"):
        rows = [dict(row) for row in populations[split]]
        if [row.get("index") for row in rows] != list(range(len(rows))):
            raise ValueError("selection population indices must follow dataset row order")
        if len({row["example_id"] for row in rows}) != len(rows):
            raise ValueError("selection population contains duplicate identities")
        ranked = sorted(rows, key=lambda row: (row["prompt_tokens"], _rank("length", row["example_id"])))
        quartiles = {row["index"]: min(3, 4 * rank // len(rows)) for rank, row in enumerate(ranked)}
        for row in rows:
            row["prompt_length_quartile"] = quartiles[row["index"]]
        catalog[split] = rows
    if {row["example_id"] for row in catalog["train"]} & {row["example_id"] for row in catalog["validation"]}:
        raise ValueError("training and validation populations overlap")
    first = _select_stratified(catalog["train"], 64, "train-0")
    second = _select_stratified(catalog["train"], 64, "train-1", first)
    validation = _select_stratified(catalog["validation"], 128, "validation")
    return {
        "profile_id": PROFILE_ID,
        "protocol": "qwen3-training-stratified-selection-v1",
        "seed": SELECTION_SEED,
        "strata": ["subject", "level", "rendered_student_prompt_length_quartile"],
        "populations": catalog,
        "population_sha256": canonical_sha256(catalog),
        "train_batches": [first, second],
        "validation_indices": validation,
    }


def preflight_prompts(tokenizer: Any, splits: Mapping[str, Sequence[MathExample]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check every student and privileged SDPG context before publishing assets."""

    from verl.opd.chat import render_training_prompt, validate_training_reasoning_tokens

    token_ids = validate_training_reasoning_tokens(tokenizer, PROFILE_ID)
    maxima = {"student_prompt_tokens": 0, "teacher_prompt_tokens": 0}
    populations = {}
    prompt_hashes = {}
    for split, examples in splits.items():
        population = []
        rendered_hashes = []
        for index, example in enumerate(examples):
            original = render_student_user_content(example.question)
            privileged = render_sdpg_teacher_user_content(original, example.gold_cot, example.gold_answer)
            counts = []
            row_hashes = []
            for content in (original, privileged):
                rendered = render_training_prompt(tokenizer, [{"role": "user", "content": content}], PROFILE_ID)
                ids = tokenizer.encode(rendered, add_special_tokens=False)
                if not ids:
                    raise ValueError("training prompt tokenized to empty")
                counts.append(len(ids))
                row_hashes.append(canonical_sha256(ids))
            if counts[0] > MAX_PROMPT_TOKENS:
                raise ValueError(f"student prompt exceeds {MAX_PROMPT_TOKENS}: {example.example_id}")
            if counts[0] + MAX_RESPONSE_TOKENS + 1 > ENGINE_CONTEXT_LENGTH:
                raise ValueError("student prompt and completion exceed the rollout engine context")
            if counts[1] + MAX_RESPONSE_TOKENS + 1 > MODEL_CONTEXT_LENGTH:
                raise ValueError(f"privileged replay exceeds model context: {example.example_id}")
            maxima["student_prompt_tokens"] = max(maxima["student_prompt_tokens"], counts[0])
            maxima["teacher_prompt_tokens"] = max(maxima["teacher_prompt_tokens"], counts[1])
            rendered_hashes.append([example.example_id, *row_hashes])
            population.append({"index": index, "example_id": example.example_id, "subject": example.subject, "level": example.level, "prompt_tokens": counts[0]})
        populations[split] = population
        prompt_hashes[split] = canonical_sha256(rendered_hashes)
    return {
        "profile_id": PROFILE_ID,
        "reasoning_token_ids": list(token_ids),
        "fixed_think_opener": "<think>\n",
        "teacher_template": "sdpg",
        "teacher_content": "existing-training-sdpg-preserve-whitespace-v1",
        "tokenization": "encode-add-special-tokens-false-v1",
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_response_tokens": MAX_RESPONSE_TOKENS,
        "model_context_length": MODEL_CONTEXT_LENGTH,
        "engine_context_length": ENGINE_CONTEXT_LENGTH,
        "counts": {name: len(rows) for name, rows in splits.items()},
        "maxima": maxima,
        "rendered_prompt_ids_sha256": prompt_hashes,
    }, populations


def _stage_model(destination: Path, final_destination: Path, cache_dir: Path) -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download

    if HfApi().model_info(repo_id=MODEL_ID, revision=MODEL_REVISION).sha != MODEL_REVISION:
        raise ValueError("Qwen3 model revision did not resolve to the pinned commit")
    downloaded = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=str(destination), cache_dir=str(cache_dir))).resolve()
    if downloaded != destination.resolve():
        raise ValueError("model downloader wrote outside the staging directory")
    if (destination / ".cache").exists():
        shutil.rmtree(destination / ".cache")
    files = _inventory(destination)
    _validate_required_transformers_files(files)
    config = json.loads((destination / "config.json").read_text())
    if config.get("model_type") != "qwen3" or config.get("max_position_embeddings") != MODEL_CONTEXT_LENGTH:
        raise ValueError("Qwen3 model architecture/context differs from the pinned profile")
    return _write_manifest(destination / "manifest.json", {
        "profile_id": PROFILE_ID, "protocol": "qwen3-training-model-v1",
        "model": {"id": MODEL_ID, "requested_revision": MODEL_REVISION, "resolved_revision": MODEL_REVISION},
        "transformers_local_path": str(final_destination), "model_type": "qwen3",
        "files": files, "inventory_sha256": canonical_sha256(files),
    })


def prepare(root: Path | str, cache_dir: Path | str) -> dict[str, Any]:
    """Publish a new self-contained training root; never adopt legacy/ICL assets."""

    destination = Path(root).expanduser().absolute()
    if destination.is_symlink():
        raise ValueError("training asset root must not be a symlink")
    destination = destination.resolve()
    cache = Path(cache_dir).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / ("." + destination.name + ".prepare.lock")).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            return verify(destination)
        temporary = Path(tempfile.mkdtemp(prefix="." + destination.name + ".", dir=destination.parent))
        try:
            model_manifest = _stage_model(temporary / "model", destination / "model", cache)
            splits, cleaning = prepare_math_example_splits(load_pinned_math_train(cache))
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(temporary / "model"), local_files_only=True)
            preflight, populations = preflight_prompts(tokenizer, splits)
            data_dir = temporary / "data"
            data_dir.mkdir()
            files = {}
            for split, examples in splits.items():
                rows = [build_verl_training_row(example) for example in examples]
                for row in rows:
                    row["extra_info"]["training_profile"] = PROFILE_ID
                path = data_dir / DATA_FILES[split]
                _write_parquet(path, rows)
                files[path.name] = {"size": path.stat().st_size, "sha256": file_sha256(path), "row_count": len(rows), "logical_rows_sha256": ordered_records_sha256(rows)}
            data_manifest = _write_manifest(data_dir / "manifest.json", {
                "profile_id": PROFILE_ID, "protocol": "qwen3-training-data-v1", "model": dict(MODEL),
                "source": {"id": MATH_DATASET_ID, "config": MATH_DATASET_CONFIG, "revision": MATH_DATASET_REVISION},
                "cleaning": cleaning.to_dict(),
                "split": {"seed": MATH_SPLIT_SEED, "counts": {name: len(rows) for name, rows in splits.items()}, "validation_ids_sha256": ordered_example_ids_sha256([row.example_id for row in splits["validation"]])},
                "files": files,
            })
            selection = _write_manifest(temporary / "selection.json", build_selection(populations))
            preflight_manifest = _write_manifest(temporary / "preflight.json", preflight)
            _write_manifest(temporary / "manifest.json", {
                "profile_id": PROFILE_ID, "protocol": "qwen3-training-assets-v1", "model": dict(MODEL),
                "model_dir": str(destination / "model"), "data_dir": str(destination / "data"),
                "selection_path": str(destination / "selection.json"),
                "model_manifest_content_sha256": model_manifest["manifest_content_sha256"],
                "data_manifest_content_sha256": data_manifest["manifest_content_sha256"],
                "selection_content_sha256": selection["manifest_content_sha256"],
                "preflight_content_sha256": preflight_manifest["manifest_content_sha256"],
            })
            os.replace(temporary, destination)
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return verify(destination)


def verify(root: Path | str) -> dict[str, Any]:
    root = Path(root).expanduser().absolute()
    if root.is_symlink():
        raise ValueError("training asset root must not be a symlink")
    root = root.resolve()
    manifest = _read_manifest(root / "manifest.json")
    expected_paths = {"model_dir": str(root / "model"), "data_dir": str(root / "data"), "selection_path": str(root / "selection.json")}
    if manifest.get("protocol") != "qwen3-training-assets-v1" or manifest.get("model") != MODEL or any(manifest.get(key) != value for key, value in expected_paths.items()):
        raise ValueError("Qwen3 training asset identity differs")
    if {path.name for path in root.iterdir()} != {"model", "data", "selection.json", "preflight.json", "manifest.json"}:
        raise ValueError("training asset root contains unexpected entries")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("training assets must be self-contained regular files")
    model = _read_manifest(root / "model/manifest.json")
    data = _read_manifest(root / "data/manifest.json")
    selection = _read_manifest(root / "selection.json")
    preflight = _read_manifest(root / "preflight.json")
    for name, value in (("model_manifest", model), ("data_manifest", data), ("selection", selection), ("preflight", preflight)):
        if manifest.get(name + "_content_sha256") != value["manifest_content_sha256"]:
            raise ValueError(f"training {name} differs from the root seal")
    if model.get("model") != {"id": MODEL_ID, "requested_revision": MODEL_REVISION, "resolved_revision": MODEL_REVISION} or model.get("transformers_local_path") != str(root / "model"):
        raise ValueError("training model pin differs")
    observed_model = _inventory(root / "model")
    _validate_required_transformers_files(observed_model)
    if model.get("files") != observed_model or model.get("inventory_sha256") != canonical_sha256(observed_model):
        raise ValueError("training model inventory failed authentication")
    expected_counts = {"train": MATH_TRAIN_SIZE, "validation": MATH_VALIDATION_SIZE}
    if data.get("model") != MODEL or data.get("source") != {"id": MATH_DATASET_ID, "config": MATH_DATASET_CONFIG, "revision": MATH_DATASET_REVISION} or data.get("split") != {"seed": MATH_SPLIT_SEED, "counts": expected_counts, "validation_ids_sha256": MATH_VALIDATION_IDS_SHA256}:
        raise ValueError("training dataset differs from the pinned MATH split")
    if set(data.get("files", {})) != set(DATA_FILES.values()) or {path.name for path in (root / "data").iterdir()} != set(DATA_FILES.values()) | {"manifest.json"}:
        raise ValueError("training data inventory differs")
    for split, filename in DATA_FILES.items():
        path = root / "data" / filename
        entry = data["files"][filename]
        if entry.get("row_count") != expected_counts[split] or entry.get("size") != path.stat().st_size or entry.get("sha256") != file_sha256(path):
            raise ValueError(f"training data file failed authentication: {filename}")
    if preflight.get("counts") != expected_counts or preflight.get("reasoning_token_ids") != [151667, 151668] or preflight.get("max_response_tokens") != MAX_RESPONSE_TOKENS:
        raise ValueError("training preflight contract differs")
    populations = selection.get("populations", {})
    if set(populations) != set(expected_counts) or any(len(populations[name]) != count for name, count in expected_counts.items()):
        raise ValueError("training selection population differs")
    if selection != _seal(build_selection(populations)):
        raise ValueError("training selection does not match its deterministic contract")
    return manifest


def profile_overrides(objective: str, gpus: int, root: Path | str, run_dir: Path | str, *, replay_backend="disabled") -> list[str]:
    """Return the complete common recipe and a distinct Qwen3 objective profile."""

    if objective not in ("standalone", "hybrid"):
        raise ValueError("Qwen3 objective must be standalone or hybrid")
    if type(gpus) is not int or gpus not in (1, 2):
        raise ValueError("Qwen3 training benchmark supports only the authorized 1 or 2 GPUs")
    if replay_backend not in ("disabled", "native_fa3_v1"):
        raise ValueError("unsupported Qwen3 replay arithmetic backend")
    root, run_dir = Path(root).expanduser().resolve(), Path(run_dir).expanduser().resolve()
    values = {
        "algorithm.adv_estimator": "grpo", "algorithm.norm_adv_by_std_in_grpo": True, "algorithm.use_kl_in_reward": False,
        "data.train_files": str(root / "data" / DATA_FILES["train"]), "data.val_files": str(root / "data" / DATA_FILES["validation"]),
        "data.train_batch_size": 64, "data.val_batch_size": 128, "data.max_prompt_length": MAX_PROMPT_TOKENS, "data.max_response_length": MAX_RESPONSE_TOKENS,
        "data.filter_overlong_prompts": True, "data.truncation": "error", "data.seed": 11, "data.prompt_profile": PROFILE_ID, "data.dataloader_num_workers": 0,
        "actor_rollout_ref.model.path": str(root / "model"), "actor_rollout_ref.model.use_remove_padding": True, "actor_rollout_ref.model.enable_gradient_checkpointing": False,
        "actor_rollout_ref.actor.optim.lr": 1e-6, "actor_rollout_ref.actor.optim.weight_decay": 0.01, "actor_rollout_ref.actor.optim.warmup_style": "constant", "actor_rollout_ref.actor.optim.lr_warmup_steps": 0,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 32, "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 2, "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.clip_ratio": 0.2, "actor_rollout_ref.actor.clip_ratio_low": 0.2, "actor_rollout_ref.actor.clip_ratio_high": 0.2, "actor_rollout_ref.actor.grad_clip": 1.0,
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl", "actor_rollout_ref.actor.entropy_coeff": 0, "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 30720,
        "actor_rollout_ref.actor.fsdp_config.param_offload": True, "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True, "actor_rollout_ref.actor.checkpoint.contents": ["model", "optimizer", "extra", "hf_model"],
        "actor_rollout_ref.rollout.mode": "sync", "actor_rollout_ref.rollout.deterministic_sampling": True, "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 2,
        "actor_rollout_ref.rollout.max_model_len": ENGINE_CONTEXT_LENGTH, "actor_rollout_ref.rollout.engine_context_length": ENGINE_CONTEXT_LENGTH, "actor_rollout_ref.rollout.max_num_batched_tokens": ENGINE_CONTEXT_LENGTH,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1, "actor_rollout_ref.rollout.add_noise_dirichlet": False, "actor_rollout_ref.rollout.noise_gaussian": False,
        "actor_rollout_ref.rollout.noise_on_logits": True, "actor_rollout_ref.rollout.noise_on_inputs": False, "actor_rollout_ref.rollout.noise_factor": 1.0, "actor_rollout_ref.rollout.gpu_memory_utilization": 0.6,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 2, "actor_rollout_ref.ref.fsdp_config.param_offload": True, "actor_rollout_ref.ref.strategy": "fsdp2",
        "custom_reward_function.path": str(Path(__file__).resolve().parent / "reward.py"), "custom_reward_function.name": "compute_score",
        "trainer.logger": ["console", "wandb"], "trainer.project_name": "opd-qwen3-training-benchmark", "trainer.experiment_name": f"qwen3_0p6b_{objective}_gpus{gpus}_seed11", "trainer.training_profile": PROFILE_ID,
        "trainer.val_before_train": True, "trainer.total_epochs": 1, "trainer.total_training_steps": None, "trainer.max_rollout_iterations_per_invocation": 3,
        "trainer.n_gpus_per_node": gpus, "trainer.nnodes": 1, "trainer.default_local_dir": str(run_dir), "trainer.save_freq": 25, "trainer.test_freq": 25, "trainer.checkpoint_keep_latest": 2,
        "trainer.rollout_integrity.enabled": True, "trainer.rollout_integrity.gate_first_n_iterations": 1,
        "algorithm.opd.prompt_profile": PROFILE_ID,
        "actor_rollout_ref.rollout.require_retained_support": True,
        "ray_init.num_cpus": 16 * gpus,
    }
    result = [key + "=" + json.dumps(value, separators=(",", ":")) for key, value in values.items()]
    if replay_backend != "disabled":
        result.append("actor_rollout_ref.model.qwen_replay_backend=" + replay_backend)
    arm = "softopd_math_s11" if objective == "standalone" else "softgrpo_math_opd_s11"
    return result + list(hydra_overrides(arm))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare" and args.cache_dir is None:
        parser.error("prepare requires --cache-dir")
    result = prepare(args.root, args.cache_dir) if args.command == "prepare" else verify(args.root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
