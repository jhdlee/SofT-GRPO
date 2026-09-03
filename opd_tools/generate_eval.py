"""Generate exact native-soft and hard-token evaluation completions.

The CUDA path intentionally calls the SGLang engine shipped by the pinned
SofT-GRPO checkout.  It does not implement another sampler.  A shard is one
benchmark/generation-seed pair, so a preempted evaluation resumes without
regenerating committed samples.

Example::

    python -m opd_tools.generate_eval \
      --model-label initial --model-path /scratch/.../assets/model \
      --data-dir /scratch/.../data --output-dir /scratch/.../evaluation \
      --mode native_soft --num-gpus 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .evaluation import (
    BENCHMARKS,
    COMMON_GENERATION_SEEDS,
    EVALUATION_PROTOCOL,
    EVALUATION_SCHEMA_VERSION,
    EXPECTED_EXAMPLE_COUNTS,
    HARD_TOKEN_GENERATION_SEEDS,
    INFERENCE_MODES,
    MODEL_LABELS,
    GenerationRecord,
    evaluation_request_seed,
)
from .constants import SOFTGRPO_UPSTREAM_COMMIT
from .manifest import file_sha256


DATA_FILES = {
    "math500": "math500_test.parquet",
    "aime2024": "aime2024_test.parquet",
    "aime2025": "aime2025_test.parquet",
    "amc23": "amc23_test.parquet",
    "gsm8k_test": "gsm8k_test.parquet",
}


EVALUATION_SAMPLING_PROTOCOLS: Dict[str, Dict[str, Any]] = {
    # Defaults used by the released SofT-GRPO benchmark launchers.  Training
    # uses top-k 5 / tau 0.1; evaluation deliberately retains the published
    # top-k 30 / tau 0.5 sampling protocol and records it in every manifest.
    "released_anchor": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 30,
        "gumbel_softmax_temperature": 0.5,
        "max_new_tokens": 32_768,
    },
    # Useful as a diagnostic with exactly the rollout distribution used for
    # policy optimization.  It is never mislabeled as the released anchor.
    "training_matched": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 5,
        "gumbel_softmax_temperature": 0.1,
        "max_new_tokens": 8_192,
    },
}

GENERATION_IMPLEMENTATION = (
    "Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python/sglang"
)


def expected_sampling_source(mode: str, sampling_protocol: str) -> str:
    """Return the exact upstream launcher represented by an evaluation mode."""

    if mode not in INFERENCE_MODES:
        raise ValueError("unsupported inference mode: %r" % mode)
    if sampling_protocol not in EVALUATION_SAMPLING_PROTOCOLS:
        raise ValueError("unsupported evaluation sampling protocol: %r" % sampling_protocol)
    if sampling_protocol == "training_matched":
        return "seed-11 training configuration"
    filename = (
        "run_sample_gumbel_raw.sh"
        if mode == "native_soft"
        else "run_sample_discrete-token_raw.sh"
    )
    return "Soft-Thinking+noise+loss-main/%s" % filename


def expected_engine_mode(mode: str) -> Dict[str, bool]:
    """Manifest the flags that distinguish native-soft and hard sampling."""

    if mode not in INFERENCE_MODES:
        raise ValueError("unsupported inference mode: %r" % mode)
    native_soft = mode == "native_soft"
    return {
        "enable_soft_thinking": native_soft,
        "add_noise_gumbel_softmax": native_soft,
    }


_ATOMIC_TEMP_RE = re.compile(
    r"^[.](?:seed_[0-9]+[.](?:jsonl|manifest[.]json)|generation_manifest[.]json|"
    r"completion[.]json)[.][A-Za-z0-9_-]+[.]tmp$"
)


def cleanup_stale_atomic_files(mode_root: Path) -> list[Path]:
    """Remove only incomplete files produced by this module's atomic writer."""

    removed = []
    if not mode_root.exists():
        return removed
    if not mode_root.is_dir() or mode_root.is_symlink():
        raise ValueError("evaluation mode root must be a real directory")
    for path in sorted(mode_root.rglob(".*.tmp")):
        if (
            not path.is_file()
            or path.is_symlink()
            or not _ATOMIC_TEMP_RE.fullmatch(path.name)
        ):
            raise ValueError("unexpected temporary evaluation path: %s" % path)
        path.unlink()
        removed.append(path)
    return removed


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _tree_fingerprint(root: Path) -> Dict[str, Any]:
    """Authenticate an exported HF model without relying on its path name."""

    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("model path must be a real directory: %s" % root)
    inventory = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("model export may not contain symlinks: %s" % path)
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        inventory.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    if not inventory or not any(
        row["path"].endswith((".safetensors", ".bin")) for row in inventory
    ):
        raise ValueError("model export contains no model weights: %s" % root)
    digest = hashlib.sha256(_canonical_json(inventory)).hexdigest()
    return {"path": str(root), "tree_sha256": digest, "files": inventory}


def _finish_reason(value: Any) -> str:
    if isinstance(value, Mapping):
        candidate = value.get("type", value.get("matched", value.get("reason")))
        return str(candidate if candidate is not None else value)
    candidate = getattr(value, "type", None)
    return str(candidate if candidate is not None else value)


def _softmax(vector: np.ndarray) -> np.ndarray:
    shifted = vector.astype(np.float64) - float(np.max(vector))
    numerator = np.exp(shifted)
    return numerator / numerator.sum()


def native_soft_diagnostics(
    *,
    response_token_ids: Sequence[int],
    topk_ids: Sequence[Sequence[int]],
    perturbed_logits: Sequence[Sequence[float]],
    gumbel_temperature: float,
    response_text: str,
) -> Dict[str, Any]:
    """Decode the released support sentinel into per-response diagnostics."""

    token_ids = np.asarray(response_token_ids, dtype=np.int64)
    if token_ids.size == 0:
        if topk_ids is None or perturbed_logits is None:
            raise ValueError("empty response is missing native-soft metadata")
        if len(topk_ids) != 0 or len(perturbed_logits) != 0:
            raise ValueError("empty response has non-empty native-soft metadata")
        return {
            "latent_token_count": 0,
            "hard_token_count": 0,
            "close_tag": False,
            "soft_to_hard": False,
            "all_soft": False,
            "mixture_entropy_mean": None,
            "top1_weight_mean": None,
            "soft_hard_agreement": None,
        }
    supports = np.asarray(topk_ids, dtype=np.int64)
    logits = np.asarray(perturbed_logits, dtype=np.float64)
    if token_ids.ndim != 1 or supports.ndim != 2 or logits.shape != supports.shape:
        raise ValueError("malformed native-soft rollout metadata")
    if supports.shape[0] != token_ids.shape[0] or supports.shape[1] < 2:
        raise ValueError("native-soft metadata does not align with output tokens")
    if not math.isfinite(gumbel_temperature) or gumbel_temperature <= 0:
        raise ValueError("gumbel temperature must be finite and positive")

    categorical = np.all(supports[:, 1:] == 0, axis=1)
    latent = ~categorical
    seen_latent = np.cumsum(latent.astype(np.int64)) > 0
    transitioned = bool(np.any(categorical & seen_latent))
    entropies = []
    top1_weights = []
    agreements = []
    for index in np.flatnonzero(latent):
        weights = _softmax(logits[index] / gumbel_temperature)
        entropies.append(float(-(weights * np.log(np.maximum(weights, 1e-300))).sum()))
        top1_weights.append(float(weights.max()))
        hard_shadow = int(supports[index, int(np.argmax(weights))])
        agreements.append(float(hard_shadow == int(token_ids[index])))
    return {
        "latent_token_count": int(latent.sum()),
        "hard_token_count": int(categorical.sum()),
        "close_tag": "</think>" in response_text,
        "soft_to_hard": transitioned,
        "all_soft": bool(token_ids.size > 0 and latent.all()),
        "mixture_entropy_mean": (float(np.mean(entropies)) if entropies else None),
        "top1_weight_mean": (float(np.mean(top1_weights)) if top1_weights else None),
        "soft_hard_agreement": (float(np.mean(agreements)) if agreements else None),
    }


def _extract_output_token_ids(meta_info: Mapping[str, Any]) -> list[int]:
    values = meta_info.get("output_token_logprobs")
    if not isinstance(values, list):
        raise ValueError("SGLang did not return output-token log probabilities")
    token_ids = []
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            raise ValueError("malformed SGLang output_token_logprobs entry")
        token_ids.append(int(value[1]))
    return token_ids


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "as_py"):
        converted = value.as_py()
        if isinstance(converted, Mapping):
            return converted
    raise ValueError("%s must be a mapping" % name)


def _chat(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError("prompt must be a list of chat messages")
    result = []
    for message in value:
        mapping = _mapping(message, "chat message")
        role, content = mapping.get("role"), mapping.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("chat messages require string role/content")
        result.append({"role": role, "content": content})
    return result


def _load_benchmark(path: Path, benchmark: str) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("evaluation generation requires pandas/pyarrow") from error
    frame = pd.read_parquet(path)
    result = []
    for row in frame.to_dict(orient="records"):
        if row.get("data_source") != benchmark:
            raise ValueError("evaluation data_source disagrees with benchmark")
        extra = _mapping(row.get("extra_info"), "extra_info")
        reward = _mapping(row.get("reward_model"), "reward_model")
        example_id = extra.get("example_id")
        gold = reward.get("ground_truth")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("evaluation row has no example_id")
        if not isinstance(gold, str) or not gold.strip():
            raise ValueError("evaluation row has no ground truth")
        result.append(
            {
                "example_id": example_id,
                "gold_answer": gold,
                "chat": _chat(row.get("prompt")),
            }
        )
    if len({row["example_id"] for row in result}) != len(result):
        raise ValueError("evaluation example IDs are not unique")
    if len(result) != EXPECTED_EXAMPLE_COUNTS[benchmark]:
        raise ValueError(
            "%s has %d rows, expected %d"
            % (benchmark, len(result), EXPECTED_EXAMPLE_COUNTS[benchmark])
        )
    return result


def _sampling_params(
    protocol: Mapping[str, Any],
    generation_seed: int,
    benchmark: str,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        seed = evaluation_request_seed(generation_seed, benchmark, row["example_id"])
        result.append(
            {
                "temperature": float(protocol["temperature"]),
                "top_p": float(protocol["top_p"]),
                "top_k": int(protocol["top_k"]),
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "after_thinking_temperature": float(protocol["temperature"]),
                "after_thinking_top_p": float(protocol["top_p"]),
                "after_thinking_top_k": int(protocol["top_k"]),
                "after_thinking_min_p": 0.0,
                "n": 1,
                "max_new_tokens": int(protocol["max_new_tokens"]),
                "think_end_str": "</think>",
                "gumbel_softmax_temperature": float(
                    protocol["gumbel_softmax_temperature"]
                ),
                "early_stopping_entropy_threshold": 0.0,
                "early_stopping_length_threshold": 256,
                "noise_factor": 1.0,
                "noise_gaussian": False,
                "noise_gumbel": True,
                "noise_on_logits": True,
                "noise_on_inputs": False,
                "seed": seed,
            }
        )
    return result


def _shard_paths(
    output_dir: Path,
    model_label: str,
    mode: str,
    benchmark: str,
    generation_seed: int,
) -> tuple[Path, Path]:
    directory = output_dir / "raw" / model_label / mode / benchmark
    data = directory / ("seed_%d.jsonl" % generation_seed)
    return data, data.with_suffix(".manifest.json")


def _verify_shard(data_path: Path, manifest_path: Path) -> Dict[str, Any]:
    if not data_path.is_file() or not manifest_path.is_file():
        raise ValueError("generation shard is only partially committed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != EVALUATION_SCHEMA_VERSION
        or manifest.get("sha256") != file_sha256(data_path)
        or manifest.get("size") != data_path.stat().st_size
    ):
        raise ValueError("generation shard authentication failed: %s" % data_path)
    observed = 0
    with data_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            GenerationRecord.from_mapping(json.loads(line))
            observed += 1
    if observed != manifest.get("row_count"):
        raise ValueError("generation shard row count changed")
    return manifest


def _write_shard(path: Path, records: Sequence[GenerationRecord]) -> Dict[str, Any]:
    payload = b"".join(_canonical_json(record.to_dict()) for record in records)
    _atomic_write(path, payload)
    manifest = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "protocol": EVALUATION_PROTOCOL,
        "size": path.stat().st_size,
        "sha256": file_sha256(path),
        "row_count": len(records),
    }
    manifest_path = path.with_suffix(".manifest.json")
    _atomic_write(manifest_path, _canonical_json(manifest))
    return _verify_shard(path, manifest_path)


def _resume_shard(
    data_path: Path,
    manifest_path: Path,
    *,
    model_label: str,
    mode: str,
    benchmark: str,
    sample_index: int,
    generation_seed: int,
    example_ids: Sequence[str],
) -> Dict[str, Any] | None:
    """Verify a commit or safely adopt data atomically published before SIGTERM."""

    if not data_path.exists() and not manifest_path.exists():
        return None
    if manifest_path.exists() and not data_path.exists():
        raise ValueError("generation shard sidecar exists without data")
    if data_path.is_symlink() or not data_path.is_file():
        raise ValueError("generation shard data must be a regular file")
    if manifest_path.exists():
        return _verify_shard(data_path, manifest_path)

    records = []
    with data_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            records.append(GenerationRecord.from_mapping(json.loads(line)))
    if len(records) != len(example_ids):
        raise ValueError("orphan generation shard has the wrong row count")
    for expected_id, record in zip(example_ids, records):
        expected = {
            "model_label": model_label,
            "inference_mode": mode,
            "benchmark": benchmark,
            "sample_index": sample_index,
            "generation_seed": generation_seed,
            "example_id": expected_id,
        }
        if any(getattr(record, name) != value for name, value in expected.items()):
            raise ValueError("orphan generation shard has the wrong row identity")
    manifest = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "protocol": EVALUATION_PROTOCOL,
        "size": data_path.stat().st_size,
        "sha256": file_sha256(data_path),
        "row_count": len(records),
    }
    _atomic_write(manifest_path, _canonical_json(manifest))
    return _verify_shard(data_path, manifest_path)


def _stable_wandb_id(config: Mapping[str, Any]) -> str:
    """Bind a resumable W&B run to the complete generation contract."""

    if "wandb_run_id" in config:
        raise ValueError("W&B identity input may not contain its own run ID")
    digest = hashlib.sha256(_canonical_json(dict(config))).hexdigest()[:16]
    return "eval-%s-%s-%s" % (
        config["model_label"],
        str(config["mode"]).replace("_", "-"),
        digest,
    )


def _init_wandb(config: Mapping[str, Any]):
    if os.environ.get("WANDB_MODE") != "online":
        raise RuntimeError("evaluation requires WANDB_MODE=online")
    if os.environ.get("WANDB_RESUME", "allow") != "allow":
        raise RuntimeError("evaluation requires WANDB_RESUME=allow")
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("evaluation requires the wandb package") from error
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "opd-softgrpo-math"),
        group=os.environ.get("WANDB_GROUP", "seed-11"),
        id=config["wandb_run_id"],
        resume="allow",
        job_type="evaluation-generation",
        tags=["evaluation", config["model_label"], config["mode"]],
        config=dict(config),
    )


def _render_prompts(tokenizer: Any, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    prompts = [
        tokenizer.apply_chat_template(
            row["chat"], tokenize=False, add_generation_prompt=True
        )
        for row in rows
    ]
    bad = [
        index
        for index, prompt in enumerate(prompts)
        if not prompt.endswith("<think>\n")
    ]
    if bad:
        raise RuntimeError(
            "native model template must open <think> for generation; bad rows: %s"
            % bad[:5]
        )
    return prompts


def resolve_parallelism(
    *,
    legacy_num_gpus: int | None,
    tensor_parallel_size: int,
    data_parallel_size: int,
) -> tuple[int, int, int]:
    """Resolve legacy TP-only and explicit TP/DP execution arguments."""

    for name, value in (
        ("tensor_parallel_size", tensor_parallel_size),
        ("data_parallel_size", data_parallel_size),
    ):
        if isinstance(value, bool) or value <= 0:
            raise ValueError("%s must be a positive integer" % name)
    if legacy_num_gpus is not None:
        if isinstance(legacy_num_gpus, bool) or legacy_num_gpus <= 0:
            raise ValueError("num_gpus must be a positive integer")
        if tensor_parallel_size != 1 or data_parallel_size != 1:
            raise ValueError(
                "--num-gpus is legacy TP-only syntax and cannot be combined "
                "with explicit TP/DP"
            )
        tensor_parallel_size = legacy_num_gpus
    return (
        tensor_parallel_size,
        data_parallel_size,
        tensor_parallel_size * data_parallel_size,
    )


def required_context_length(
    tokenizer: Any, rendered_prompts: Sequence[str], max_new_tokens: int
) -> int:
    """Reserve the guard position required by the bundled SGLang validator."""

    if not rendered_prompts or max_new_tokens <= 0:
        raise ValueError("context sizing requires prompts and a positive token cap")
    maximum_prompt = max(len(tokenizer.encode(prompt)) for prompt in rendered_prompts)
    return maximum_prompt + int(max_new_tokens) + 1


def run(args: argparse.Namespace) -> None:
    model_root = Path(args.model_path).expanduser().resolve()
    data_root = Path(args.data_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    if args.model_label not in MODEL_LABELS or args.mode not in INFERENCE_MODES:
        raise ValueError("invalid model label or inference mode")
    benchmarks = list(BENCHMARKS) if args.benchmarks == ["all"] else args.benchmarks
    if not benchmarks or len(set(benchmarks)) != len(benchmarks):
        raise ValueError("benchmarks must be a unique non-empty list")
    if any(name not in BENCHMARKS for name in benchmarks):
        raise ValueError("unsupported benchmark selection")
    protocol = EVALUATION_SAMPLING_PROTOCOLS[args.sampling_protocol]
    seeds = (
        COMMON_GENERATION_SEEDS
        if args.mode == "native_soft"
        else HARD_TOKEN_GENERATION_SEEDS
    )

    tensor_parallel_size, data_parallel_size, world_size = resolve_parallelism(
        legacy_num_gpus=args.num_gpus,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
    )

    allocated = os.environ.get("SLURM_GPUS_ON_NODE")
    if allocated is not None and int(allocated) != world_size:
        raise RuntimeError(
            "TP*DP disagrees with SLURM_GPUS_ON_NODE (%d != %s)"
            % (world_size, allocated)
        )
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ.get("SLURM_JOB_ID"):
        # Slurm itself sets this for job steps.  The launcher never does; retain
        # the variable while making its provenance visible in the manifest.
        cuda_visibility_source = "slurm"
    else:
        cuda_visibility_source = "unset"

    model = _tree_fingerprint(model_root)
    from .prepare import verify_materialized_data

    data_manifest = verify_materialized_data(data_root)
    try:
        import sglang as sgl
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "generation requires the pinned SofT-GRPO SGLang environment"
        ) from error
    expected_sglang = (
        Path(__file__).resolve().parents[1]
        / "Soft-Thinking+noise+loss-main"
        / "sglang_soft_thinking_pkg"
        / "python"
        / "sglang"
    ).resolve()
    observed_sglang = Path(sgl.__file__).resolve()
    if not observed_sglang.is_relative_to(expected_sglang):
        raise RuntimeError(
            "generation imported a non-upstream SGLang package: %s"
            % observed_sglang
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_root), local_files_only=True)
    for tag in ("<think>", "</think>"):
        ids = tokenizer.encode(tag, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError("%s must be one atomic tokenizer ID" % tag)
    datasets = {
        benchmark: _load_benchmark(data_root / DATA_FILES[benchmark], benchmark)
        for benchmark in benchmarks
    }
    rendered = {
        benchmark: _render_prompts(tokenizer, rows)
        for benchmark, rows in datasets.items()
    }
    rendered_prompts = [
        prompt for benchmark in benchmarks for prompt in rendered[benchmark]
    ]
    context_length = required_context_length(
        tokenizer, rendered_prompts, int(protocol["max_new_tokens"])
    )
    model_config = AutoConfig.from_pretrained(str(model_root), local_files_only=True)
    key_value_heads = int(
        getattr(model_config, "num_key_value_heads", 0)
        or getattr(model_config, "num_attention_heads", 0)
        or 0
    )
    if key_value_heads and key_value_heads % tensor_parallel_size:
        raise RuntimeError(
            "tensor parallel size %d does not divide %d key/value heads"
            % (tensor_parallel_size, key_value_heads)
        )
    maximum_context = int(getattr(model_config, "max_position_embeddings", 0) or 0)
    if maximum_context and context_length > maximum_context:
        raise RuntimeError(
            "generation needs context length %d but the model supports %d"
            % (context_length, maximum_context)
        )
    config = {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "softgrpo_upstream_commit": SOFTGRPO_UPSTREAM_COMMIT,
        "parent_commit": os.environ.get("OPD_PARENT_COMMIT"),
        "fork_commit": os.environ.get("OPD_SUBMODULE_COMMIT"),
        "generation_implementation": GENERATION_IMPLEMENTATION,
        "sampling_source": expected_sampling_source(
            args.mode, args.sampling_protocol
        ),
        "engine_mode": expected_engine_mode(args.mode),
        "model_label": args.model_label,
        "model": model,
        "mode": args.mode,
        "benchmarks": benchmarks,
        "generation_seeds": list(seeds),
        "sampling_protocol": args.sampling_protocol,
        "sampling": dict(protocol),
        "parallelism": {
            "tensor_parallel_size": tensor_parallel_size,
            "data_parallel_size": data_parallel_size,
            "world_size": world_size,
            "load_balance_method": "round_robin",
        },
        "batch_size": args.batch_size,
        "max_running_requests": args.max_running_requests,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "context_length": context_length,
        "data_manifest_content_sha256": data_manifest.get("manifest_content_sha256"),
        "cuda_visible_devices_source": cuda_visibility_source,
    }
    config["wandb_run_id"] = _stable_wandb_id(config)
    manifest_path = (
        output_root / "raw" / args.model_label / args.mode / "generation_manifest.json"
    )
    cleanup_stale_atomic_files(manifest_path.parent)
    if manifest_path.exists():
        observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if observed != config:
            raise RuntimeError(
                "existing generation manifest differs from this invocation"
            )
    else:
        _atomic_write(manifest_path, _canonical_json(config))

    wandb_run = _init_wandb(config)
    engine = None
    started = time.monotonic()
    completed = 0
    committed_shards: list[Dict[str, Any]] = []
    succeeded = False
    try:
        engine = sgl.Engine(
            model_path=str(model_root),
            tp_size=tensor_parallel_size,
            dp_size=data_parallel_size,
            load_balance_method="round_robin",
            trust_remote_code=True,
            random_seed=11,
            context_length=context_length,
            max_running_requests=args.max_running_requests,
            mem_fraction_static=args.gpu_memory_utilization,
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
            enable_soft_thinking=args.mode == "native_soft",
            add_noise_gumbel_softmax=args.mode == "native_soft",
            max_topk=int(protocol["top_k"]),
            sampling_backend="flashinfer",
        )
        observed_topology = engine.server_args
        if (
            int(observed_topology.tp_size) != tensor_parallel_size
            or int(observed_topology.dp_size) != data_parallel_size
            or observed_topology.load_balance_method != "round_robin"
        ):
            raise RuntimeError("SGLang did not preserve the requested TP/DP topology")

        for benchmark in benchmarks:
            rows = datasets[benchmark]
            prompts = rendered[benchmark]
            for sample_index, generation_seed in enumerate(seeds):
                data_path, sidecar_path = _shard_paths(
                    output_root,
                    args.model_label,
                    args.mode,
                    benchmark,
                    generation_seed,
                )
                shard = _resume_shard(
                    data_path,
                    sidecar_path,
                    model_label=args.model_label,
                    mode=args.mode,
                    benchmark=benchmark,
                    sample_index=sample_index,
                    generation_seed=generation_seed,
                    example_ids=[row["example_id"] for row in rows],
                )
                if shard is not None:
                    if shard["row_count"] != len(rows):
                        raise RuntimeError(
                            "committed generation shard has wrong row count"
                        )
                    completed += 1
                    committed_shards.append(
                        {
                            "path": data_path.relative_to(output_root).as_posix(),
                            "size": shard["size"],
                            "sha256": shard["sha256"],
                            "row_count": shard["row_count"],
                        }
                    )
                    continue

                records: list[GenerationRecord] = []
                token_total = 0
                cap_total = 0
                shard_started = time.monotonic()
                for start in range(0, len(rows), args.batch_size):
                    stop = min(len(rows), start + args.batch_size)
                    batch_rows = rows[start:stop]
                    params = _sampling_params(
                        protocol, generation_seed, benchmark, batch_rows
                    )
                    outputs = engine.generate(
                        prompts[start:stop],
                        sampling_params=params,
                        return_logprob=True,
                    )
                    if isinstance(outputs, Mapping):
                        outputs = [outputs]
                    if len(outputs) != len(batch_rows):
                        raise RuntimeError("SGLang output batch changed cardinality")
                    for row, output in zip(batch_rows, outputs):
                        meta = _mapping(output.get("meta_info"), "SGLang meta_info")
                        response = output.get("text")
                        if not isinstance(response, str):
                            raise ValueError("SGLang response text must be a string")
                        response_ids = _extract_output_token_ids(meta)
                        finish = _finish_reason(meta.get("finish_reason"))
                        capped = finish.lower() == "length"
                        if args.mode == "native_soft":
                            diagnostics = native_soft_diagnostics(
                                response_token_ids=response_ids,
                                topk_ids=meta.get("output_topk_idx_list"),
                                perturbed_logits=meta.get("output_topk_gumbel_list"),
                                gumbel_temperature=float(
                                    protocol["gumbel_softmax_temperature"]
                                ),
                                response_text=response,
                            )
                        else:
                            diagnostics = {
                                "latent_token_count": 0,
                                "hard_token_count": len(response_ids),
                                "close_tag": "</think>" in response,
                                "soft_to_hard": False,
                                "all_soft": False,
                                "mixture_entropy_mean": None,
                                "top1_weight_mean": None,
                                "soft_hard_agreement": None,
                            }
                        request_seed = evaluation_request_seed(
                            generation_seed, benchmark, row["example_id"]
                        )
                        record = GenerationRecord(
                            model_label=args.model_label,
                            benchmark=benchmark,
                            example_id=row["example_id"],
                            inference_mode=args.mode,
                            sample_index=sample_index,
                            generation_seed=generation_seed,
                            request_seed=request_seed,
                            response=response,
                            response_token_count=len(response_ids),
                            finish_reason=finish,
                            capped=capped,
                            gold_answer=row["gold_answer"],
                            **diagnostics,
                        )
                        records.append(record)
                        token_total += len(response_ids)
                        cap_total += int(capped)
                shard = _write_shard(data_path, records)
                completed += 1
                committed_shards.append(
                    {
                        "path": data_path.relative_to(output_root).as_posix(),
                        "size": shard["size"],
                        "sha256": shard["sha256"],
                        "row_count": shard["row_count"],
                    }
                )
                elapsed = time.monotonic() - shard_started
                wandb_run.log(
                    {
                        "evaluation/shards_committed": completed,
                        "evaluation/current_sample_index": sample_index,
                        "evaluation/current_generation_seed": generation_seed,
                        "evaluation/response_tokens": token_total,
                        "evaluation/tokens_per_second": token_total
                        / max(elapsed, 1e-9),
                        "evaluation/cap_rate": cap_total / max(len(records), 1),
                    },
                    step=completed,
                )
                wandb_run.summary["evaluation/latest_shard_sha256"] = shard["sha256"]
                wandb_run.summary["evaluation/latest_shard_path"] = str(data_path)
        wandb_run.summary["evaluation/completed"] = True
        wandb_run.summary["evaluation/elapsed_seconds"] = time.monotonic() - started
        wandb_run.summary["evaluation/output_root"] = str(output_root)
        completion_path = manifest_path.parent / "completion.json"
        completion = {
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "generation_manifest_sha256": file_sha256(manifest_path),
            "model_label": args.model_label,
            "mode": args.mode,
            "benchmarks": benchmarks,
            "sampling_protocol": args.sampling_protocol,
            "shards_committed": completed,
            "expected_shards": len(benchmarks) * len(seeds),
            "rows_committed": sum(row["row_count"] for row in committed_shards),
            "shards": committed_shards,
        }
        _atomic_write(completion_path, _canonical_json(completion))
        wandb_run.log_artifact(
            str(manifest_path),
            name="%s-%s-generation-manifest" % (args.model_label, args.mode),
            type="evaluation-generation-manifest",
        )
        succeeded = True
    finally:
        try:
            if not succeeded:
                wandb_run.summary["evaluation/completed"] = False
            # The bundled SGLang shutdown terminates child processes, including
            # W&B's service. Flush W&B before asking the engine to shut down.
            wandb_run.finish()
        finally:
            if engine is not None:
                engine.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-label", required=True, choices=MODEL_LABELS)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", required=True, choices=INFERENCE_MODES)
    parser.add_argument(
        "--benchmarks", nargs="+", default=["all"], choices=("all",) + BENCHMARKS
    )
    parser.add_argument(
        "--sampling-protocol",
        choices=tuple(EVALUATION_SAMPLING_PROTOCOLS),
        default="released_anchor",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Legacy TP-only GPU count; prefer explicit TP/DP arguments.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-running-requests", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    resolve_parallelism(
        legacy_num_gpus=args.num_gpus,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
    )
    if args.batch_size <= 0 or args.max_running_requests <= 0:
        raise ValueError("batch/request counts must be positive")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("gpu-memory-utilization must be in (0, 1)")
    run(args)


if __name__ == "__main__":
    main()
