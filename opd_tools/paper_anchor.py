"""Aggregate the released base-model SofT-GRPO MATH-500 paper anchor.

This command intentionally accepts exactly one evaluation cell:

* ``model_label=initial``
* ``mode=native_soft``
* ``benchmark=math500``
* ``sampling_protocol=released_anchor``
* 32 generation seeds (11 through 42), with all 500 examples in every shard

The strict, standalone contract prevents a partial ICL matrix or a
training-matched sampler from being reported as the paper's ``Mean@32``
baseline.  Both the repository's Math-Verify metric and the exact rule judge
shipped by upstream are reported.  Raw generations remain in scratch.  The
compact authenticated reports use an ``upstream_rule`` sibling prefix so a
post-hoc pass cannot overwrite the reports produced by the original job.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import importlib.util
import io
import json
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np

from .assets import verify_model_snapshot
from .constants import (
    MATH_VERIFY_VERSION,
    MODEL_ID,
    MODEL_REVISION,
    SOFTGRPO_UPSTREAM_COMMIT,
)
from .evaluation import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    COMMON_GENERATION_SEEDS,
    EVALUATION_PROTOCOL,
    EVALUATION_SCHEMA_VERSION,
    GenerationRecord,
    example_level_metric,
)
from .generate_eval import (
    EVALUATION_SAMPLING_PROTOCOLS,
    GENERATION_IMPLEMENTATION,
    _atomic_write,
    _canonical_json,
    _stable_wandb_id,
    _tree_fingerprint,
    _verify_shard,
    expected_engine_mode,
    expected_sampling_source,
)
from .graders import math_verify_full_response_grade
from .manifest import file_sha256


PAPER_ANCHOR_SCHEMA_VERSION = 2
PAPER_ANCHOR_PROTOCOL = "softgrpo-paper-base-math500-anchor-v1"
PAPER_ANCHOR_REPORT_VARIANT = "upstream-rule-judge-v1"
PAPER_ANCHOR_REPORT_PREFIX = "paper_anchor_upstream_rule"
PAPER_ANCHOR_IMPLEMENTATION_PATH = "opd_tools/paper_anchor.py"
PAPER_ANCHOR_MODEL_LABEL = "initial"
PAPER_ANCHOR_MODE = "native_soft"
PAPER_ANCHOR_BENCHMARK = "math500"
PAPER_ANCHOR_SAMPLING_PROTOCOL = "released_anchor"
PAPER_ANCHOR_EXAMPLE_COUNT = 500
PAPER_ANCHOR_SAMPLE_COUNT = 32
PAPER_ANCHOR_GROUP = "paper-anchor-base"
PAPER_ANCHOR_PARALLELISM = {
    "tensor_parallel_size": 1,
    "data_parallel_size": 8,
    "world_size": 8,
    "load_balance_method": "round_robin",
}

UPSTREAM_MATH500_GRADER_PATH = (
    "Soft-Thinking+noise+loss-main/matheval.py"
)
UPSTREAM_MATH500_GRADER_SHA256 = (
    "43845ffe8520029e93e5f86967bee419203b755253314bf6065ff093e18ff5d1"
)
UPSTREAM_MATH500_GRADER_API = "MATH500Evaluator.rule_judge"
UPSTREAM_GRADER_DEPENDENCY_VERSIONS = {
    "math-verify": MATH_VERIFY_VERSION,
    "latex2sympy2-extended": "1.10.2",
}
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _required_commit_from_environment(
    environ: Mapping[str, str], *, preferred: str, legacy: str
) -> str:
    declared = {
        name: environ[name]
        for name in (preferred, legacy)
        if environ.get(name)
    }
    if not declared:
        raise RuntimeError(
            "%s (or legacy %s) is required" % (preferred, legacy)
        )
    values = set(declared.values())
    if len(values) != 1:
        raise RuntimeError("%s and %s disagree" % (preferred, legacy))
    value = next(iter(values))
    if _GIT_COMMIT_RE.fullmatch(value) is None:
        raise RuntimeError("%s must be a full lowercase Git commit" % preferred)
    return value


def required_regrader_source_identity(
    environ: Mapping[str, str] | None = None,
) -> Dict[str, str]:
    """Bind each post-hoc report to its exact parent and regrader source."""

    values = os.environ if environ is None else environ
    implementation_path = Path(__file__).resolve()
    if not implementation_path.is_file() or implementation_path.is_symlink():
        raise RuntimeError("paper-anchor implementation is missing or a symlink")
    return {
        "parent_commit": _required_commit_from_environment(
            values,
            preferred="OPD_PAPER_REGRADE_PARENT_COMMIT",
            legacy="OPD_PARENT_COMMIT",
        ),
        "submodule_commit": _required_commit_from_environment(
            values,
            preferred="OPD_PAPER_REGRADE_SUBMODULE_COMMIT",
            legacy="OPD_SUBMODULE_COMMIT",
        ),
        "implementation_path": PAPER_ANCHOR_IMPLEMENTATION_PATH,
        "implementation_sha256": file_sha256(implementation_path),
    }


def _upstream_math500_grader_source() -> Path:
    return Path(__file__).resolve().parents[1] / UPSTREAM_MATH500_GRADER_PATH


def upstream_grader_dependency_versions() -> Dict[str, str]:
    """Require the two parsing libraries used by upstream at their locked versions."""

    observed: Dict[str, str] = {}
    for distribution, expected in UPSTREAM_GRADER_DEPENDENCY_VERSIONS.items():
        try:
            value = distribution_version(distribution)
        except PackageNotFoundError as error:
            raise RuntimeError(
                "released MATH-500 grader requires %s==%s"
                % (distribution, expected)
            ) from error
        if value != expected:
            raise RuntimeError(
                "released MATH-500 grader requires %s==%s, found %s"
                % (distribution, expected, value)
            )
        observed[distribution] = value
    return observed


def upstream_math500_grader_provenance() -> Dict[str, Any]:
    """Authenticate the exact released MATH-500 rule grader source."""

    path = _upstream_math500_grader_source()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("released MATH-500 grader source is missing or a symlink")
    observed = file_sha256(path)
    if observed != UPSTREAM_MATH500_GRADER_SHA256:
        raise RuntimeError("released MATH-500 grader source hash differs from upstream")
    return {
        "path": UPSTREAM_MATH500_GRADER_PATH,
        "sha256": observed,
        "api": UPSTREAM_MATH500_GRADER_API,
        "upstream_commit": SOFTGRPO_UPSTREAM_COMMIT,
        "dependencies": upstream_grader_dependency_versions(),
    }


@functools.lru_cache(maxsize=1)
def _upstream_math500_evaluator() -> Any:
    """Load the released evaluator lazily from its authenticated source file."""

    provenance = upstream_math500_grader_provenance()
    path = _upstream_math500_grader_source()
    module_name = "_opd_authenticated_upstream_matheval"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load released MATH-500 grader source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluator_map = getattr(module, "evaluator_map", None)
    evaluator = evaluator_map.get("math500") if isinstance(evaluator_map, dict) else None
    if evaluator is None or not callable(getattr(evaluator, "rule_judge", None)):
        raise RuntimeError(
            "authenticated source does not expose %s" % provenance["api"]
        )
    return evaluator


def upstream_math500_rule_judge(response: str, gold_answer: str) -> tuple[bool, str]:
    """Run the paper repository's rule judge with its original soft-run API.

    The released soft-thinking evaluator passes ``finish_generation=False`` for
    every native-soft response.  ``MATH500Evaluator.rule_judge`` currently
    ignores that argument, but preserving it here makes the post-hoc metric an
    exact call through the released implementation rather than a reimplementation.
    """

    result = _upstream_math500_evaluator().rule_judge(
        response, gold_answer, False
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("released MATH-500 rule_judge must return a pair")
    correct, extracted = result
    if not isinstance(correct, (bool, np.bool_)):
        raise TypeError("released MATH-500 rule_judge correctness must be Boolean")
    return bool(correct), str(extracted)


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("invalid JSON at %s:%d" % (path, line_number)) from error
            if not isinstance(value, dict):
                raise ValueError("generation JSONL rows must be objects")
            yield value


def _chunks(
    values: Iterable[Mapping[str, Any]], size: int
) -> Iterator[list[Mapping[str, Any]]]:
    chunk: list[Mapping[str, Any]] = []
    for value in values:
        chunk.append(value)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def expected_shards(root: Path) -> list[tuple[Path, Path, int]]:
    directory = (
        root
        / "raw"
        / PAPER_ANCHOR_MODEL_LABEL
        / PAPER_ANCHOR_MODE
        / PAPER_ANCHOR_BENCHMARK
    )
    return [
        (
            directory / ("seed_%d.jsonl" % seed),
            directory / ("seed_%d.manifest.json" % seed),
            seed,
        )
        for seed in COMMON_GENERATION_SEEDS
    ]


def _generation_manifest_path(root: Path) -> Path:
    return (
        root
        / "raw"
        / PAPER_ANCHOR_MODEL_LABEL
        / PAPER_ANCHOR_MODE
        / "generation_manifest.json"
    )


def _completion_path(root: Path) -> Path:
    return (
        root
        / "raw"
        / PAPER_ANCHOR_MODEL_LABEL
        / PAPER_ANCHOR_MODE
        / "completion.json"
    )


def _require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("%s must be a lowercase SHA-256 digest" % field)
    return value


def _authenticate_starting_model(model: Mapping[str, Any]) -> Dict[str, Any]:
    """Prove that ``initial`` is the pinned starting checkpoint, not a label."""

    model_path = model.get("path")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("generation manifest has no model path")
    asset_manifest = verify_model_snapshot(Path(model_path))
    if asset_manifest.get("model") != {
        "id": MODEL_ID,
        "requested_revision": MODEL_REVISION,
        "resolved_revision": MODEL_REVISION,
    }:
        raise ValueError("paper anchor must use the pinned starting checkpoint")
    observed = _tree_fingerprint(Path(model_path))
    if dict(model) != observed:
        raise ValueError("generation model fingerprint no longer matches its snapshot")
    return {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "inventory_sha256": asset_manifest["inventory_sha256"],
        "tree_sha256": observed["tree_sha256"],
    }


def _validate_generation_manifest(value: Mapping[str, Any]) -> Dict[str, Any]:
    expected = {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "softgrpo_upstream_commit": SOFTGRPO_UPSTREAM_COMMIT,
        "generation_implementation": GENERATION_IMPLEMENTATION,
        "sampling_source": expected_sampling_source(
            PAPER_ANCHOR_MODE, PAPER_ANCHOR_SAMPLING_PROTOCOL
        ),
        "engine_mode": expected_engine_mode(PAPER_ANCHOR_MODE),
        "model_label": PAPER_ANCHOR_MODEL_LABEL,
        "mode": PAPER_ANCHOR_MODE,
        "benchmarks": [PAPER_ANCHOR_BENCHMARK],
        "generation_seeds": list(COMMON_GENERATION_SEEDS),
        "sampling_protocol": PAPER_ANCHOR_SAMPLING_PROTOCOL,
        "sampling": EVALUATION_SAMPLING_PROTOCOLS[PAPER_ANCHOR_SAMPLING_PROTOCOL],
        "parallelism": PAPER_ANCHOR_PARALLELISM,
    }
    for field, required in expected.items():
        if value.get(field) != required:
            raise ValueError("generation manifest has the wrong %s" % field)
    if len(COMMON_GENERATION_SEEDS) != PAPER_ANCHOR_SAMPLE_COUNT:
        raise AssertionError("paper anchor requires exactly 32 generation seeds")
    data_hash = _require_sha256(
        value.get("data_manifest_content_sha256"),
        "data_manifest_content_sha256",
    )
    for field in ("parent_commit", "fork_commit"):
        commit = value.get(field)
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise ValueError("generation manifest %s must be a full Git SHA" % field)
    context_length = value.get("context_length")
    if (
        isinstance(context_length, bool)
        or not isinstance(context_length, int)
        or context_length <= int(value["sampling"]["max_new_tokens"])
    ):
        raise ValueError("generation manifest has an invalid context length")
    model = value.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("generation manifest has no model fingerprint")
    model_identity = _authenticate_starting_model(model)
    wandb_identity = dict(value)
    wandb_identity.pop("wandb_run_id", None)
    expected_wandb_id = _stable_wandb_id(wandb_identity)
    if value.get("wandb_run_id") != expected_wandb_id:
        raise ValueError("generation manifest has the wrong stable W&B run ID")
    return {"data_manifest_content_sha256": data_hash, "model": model_identity}


def authenticate_input(input_dir: Path) -> Dict[str, Any]:
    """Authenticate the complete, base-only released-anchor shard inventory."""

    root = input_dir.expanduser().resolve()
    raw = root / "raw"
    manifest_path = _generation_manifest_path(root)
    completion_path = _completion_path(root)
    shards = expected_shards(root)
    expected_paths = {manifest_path, completion_path}
    for data_path, sidecar_path, _ in shards:
        expected_paths.update((data_path, sidecar_path))
    observed_paths = {
        path for path in raw.rglob("*") if path.is_file() or path.is_symlink()
    } if raw.is_dir() else set()
    if observed_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - observed_paths)
        extra = sorted(str(path) for path in observed_paths - expected_paths)
        raise ValueError(
            "paper-anchor inventory differs: missing=%s extra=%s"
            % (missing[:5], extra[:5])
        )
    if any(path.is_symlink() for path in observed_paths):
        raise ValueError("paper-anchor inputs may not be symlinks")

    try:
        generation_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("generation manifest is unreadable") from error
    if not isinstance(generation_manifest, dict):
        raise ValueError("generation manifest must be an object")
    identity = _validate_generation_manifest(generation_manifest)

    expected_example_ids: set[str] | None = None
    expected_gold: Dict[str, str] = {}
    shard_inventory = []
    for sample_index, (data_path, sidecar_path, seed) in enumerate(shards):
        sidecar = _verify_shard(data_path, sidecar_path)
        if set(sidecar) != {"schema_version", "protocol", "size", "sha256", "row_count"}:
            raise ValueError("generation shard sidecar has an unexpected schema")
        if sidecar.get("protocol") != EVALUATION_PROTOCOL:
            raise ValueError("generation shard has the wrong protocol")
        if sidecar.get("row_count") != PAPER_ANCHOR_EXAMPLE_COUNT:
            raise ValueError("generation shard must contain exactly 500 MATH examples")

        ids: set[str] = set()
        for value in _read_jsonl(data_path):
            record = GenerationRecord.from_mapping(value)
            if (
                record.model_label != PAPER_ANCHOR_MODEL_LABEL
                or record.inference_mode != PAPER_ANCHOR_MODE
                or record.benchmark != PAPER_ANCHOR_BENCHMARK
                or record.sample_index != sample_index
                or record.generation_seed != seed
            ):
                raise ValueError("generation row differs from its paper-anchor shard")
            if record.example_id in ids:
                raise ValueError("generation shard contains a duplicate example")
            ids.add(record.example_id)
            previous = expected_gold.setdefault(record.example_id, record.gold_answer)
            if previous != record.gold_answer:
                raise ValueError("gold answer differs across generation seeds")
        if expected_example_ids is None:
            expected_example_ids = ids
        elif ids != expected_example_ids:
            raise ValueError("generation shards do not contain identical MATH examples")
        shard_inventory.append(
            {
                "path": data_path.relative_to(root).as_posix(),
                "manifest_path": sidecar_path.relative_to(root).as_posix(),
                "generation_seed": seed,
                "row_count": sidecar["row_count"],
                "size": sidecar["size"],
                "sha256": sidecar["sha256"],
                "manifest_sha256": file_sha256(sidecar_path),
            }
        )
    if expected_example_ids is None or len(expected_example_ids) != PAPER_ANCHOR_EXAMPLE_COUNT:
        raise ValueError("paper anchor does not contain exactly 500 distinct examples")

    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("generation completion record is unreadable") from error
    expected_completion = {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "generation_manifest_sha256": file_sha256(manifest_path),
        "model_label": PAPER_ANCHOR_MODEL_LABEL,
        "mode": PAPER_ANCHOR_MODE,
        "benchmarks": [PAPER_ANCHOR_BENCHMARK],
        "sampling_protocol": PAPER_ANCHOR_SAMPLING_PROTOCOL,
        "shards_committed": PAPER_ANCHOR_SAMPLE_COUNT,
        "expected_shards": PAPER_ANCHOR_SAMPLE_COUNT,
        "rows_committed": PAPER_ANCHOR_EXAMPLE_COUNT * PAPER_ANCHOR_SAMPLE_COUNT,
        "shards": [
            {
                "path": row["path"],
                "size": row["size"],
                "sha256": row["sha256"],
                "row_count": row["row_count"],
            }
            for row in shard_inventory
        ],
    }
    if completion != expected_completion:
        raise ValueError("generation completion record differs from exact shard inventory")

    return {
        "protocol": PAPER_ANCHOR_PROTOCOL,
        "input_dir": str(root),
        "generation_manifest": {
            "path": manifest_path.relative_to(root).as_posix(),
            "sha256": file_sha256(manifest_path),
        },
        "generation_completion": {
            "path": completion_path.relative_to(root).as_posix(),
            "sha256": file_sha256(completion_path),
        },
        "model": identity["model"],
        "data_manifest_content_sha256": identity["data_manifest_content_sha256"],
        "sampling": dict(
            EVALUATION_SAMPLING_PROTOCOLS[PAPER_ANCHOR_SAMPLING_PROTOCOL]
        ),
        "generation_seeds": list(COMMON_GENERATION_SEEDS),
        "example_count": PAPER_ANCHOR_EXAMPLE_COUNT,
        "sample_count_per_example": PAPER_ANCHOR_SAMPLE_COUNT,
        "shards": shard_inventory,
    }


def _score_record(value: Mapping[str, Any]) -> Dict[str, Any]:
    record = GenerationRecord.from_mapping(value)
    math_verify_correct = math_verify_full_response_grade(
        record.response, record.gold_answer
    ).correct
    upstream_rule_correct, upstream_extracted_answer = upstream_math500_rule_judge(
        record.response, record.gold_answer
    )
    return {
        "example_id": record.example_id,
        "generation_seed": record.generation_seed,
        # Retain ``correct`` as the historical Math-Verify field so callers
        # consuming an in-memory score table do not silently change meaning.
        "correct": bool(math_verify_correct),
        "upstream_rule_correct": bool(upstream_rule_correct),
        "upstream_extracted_answer": upstream_extracted_answer,
        "response_length": float(record.response_token_count),
        "latent_length": float(record.latent_token_count),
        "hard_answer_length": float(record.hard_token_count),
        "capped": float(record.capped),
        "close_tag": float(record.close_tag),
        "soft_to_hard": float(record.soft_to_hard),
        "all_soft": float(record.all_soft),
        "mixture_entropy": record.mixture_entropy_mean,
        "top1_weight": record.top1_weight_mean,
        "soft_hard_agreement": record.soft_hard_agreement,
    }


def _score_chunk(values: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    return [_score_record(value) for value in values]


def score_input(
    input_manifest: Mapping[str, Any], *, workers: int, chunk_size: int
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    if workers <= 0 or chunk_size <= 0:
        raise ValueError("workers and chunk_size must be positive")
    root = Path(str(input_manifest["input_dir"]))
    grouped: MutableMapping[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    paths = [root / str(shard["path"]) for shard in input_manifest["shards"]]

    def add(row: Mapping[str, Any]) -> None:
        example_id = str(row["example_id"])
        seed = int(row["generation_seed"])
        if seed in grouped[example_id]:
            raise ValueError("duplicate scored example/seed pair")
        grouped[example_id][seed] = dict(row)

    if workers == 1:
        for path in paths:
            for chunk in _chunks(_read_jsonl(path), chunk_size):
                for row in _score_chunk(chunk):
                    add(row)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for path in paths:
                chunks = _chunks(_read_jsonl(path), chunk_size)
                for scored in executor.map(_score_chunk, chunks, chunksize=1):
                    for row in scored:
                        add(row)

    expected_seeds = set(COMMON_GENERATION_SEEDS)
    if len(grouped) != PAPER_ANCHOR_EXAMPLE_COUNT:
        raise ValueError("scored paper anchor does not have exactly 500 examples")
    if any(set(rows) != expected_seeds for rows in grouped.values()):
        raise ValueError("scored examples do not have the exact 32-seed inventory")
    return dict(grouped)


def bootstrap_mean(
    values: Mapping[str, float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    chunk_size: int = 256,
) -> Dict[str, float | int]:
    """Example-level nonparametric percentile interval for a population mean."""

    if resamples != BOOTSTRAP_RESAMPLES:
        raise ValueError("the paper anchor requires exactly 10,000 resamples")
    if seed != BOOTSTRAP_SEED:
        raise ValueError("the paper anchor requires bootstrap seed 11")
    if chunk_size <= 0 or not values:
        raise ValueError("bootstrap values must be non-empty and chunk size positive")
    ordered = np.asarray([float(values[key]) for key in sorted(values)], dtype=np.float64)
    if not np.isfinite(ordered).all():
        raise ValueError("bootstrap values must be finite")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, chunk_size):
        stop = min(resamples, start + chunk_size)
        indices = rng.integers(0, len(ordered), size=(stop - start, len(ordered)))
        means[start:stop] = ordered[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975], method="linear")
    return {
        "estimate": float(ordered.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": 0.95,
        "resamples": resamples,
        "bootstrap_seed": seed,
        "example_count": len(ordered),
    }


def _per_example_diagnostic(
    grouped: Mapping[str, Mapping[int, Mapping[str, Any]]], field: str
) -> tuple[Dict[str, float], int]:
    result: Dict[str, float] = {}
    observed_sequences = 0
    for example_id, by_seed in grouped.items():
        values = [by_seed[seed][field] for seed in COMMON_GENERATION_SEEDS]
        available = [float(value) for value in values if value is not None]
        observed_sequences += len(available)
        if available:
            result[example_id] = float(np.mean(available))
    return result, observed_sequences


def _aggregate_binary_outcomes(
    outcomes: Mapping[str, Sequence[int]],
    *,
    category: str,
    wandb_namespace: str,
    total_samples: int,
) -> tuple[Dict[str, Dict[str, Any]], list[Dict[str, Any]], Dict[str, float]]:
    metrics: Dict[str, Dict[str, Any]] = {}
    metric_rows: list[Dict[str, Any]] = []
    wandb_metrics: Dict[str, float] = {}
    for name in ("mean_at_32", "pass_at_8", "pass_at_16", "pass_at_32"):
        result = bootstrap_mean(example_level_metric(outcomes, name))
        estimator = (
            "mean correctness over 32 samples per example; estimates pass@1"
            if name == "mean_at_32"
            else "unbiased pass@k from 32 samples, averaged over examples"
        )
        result["estimator"] = estimator
        metrics[name] = result
        metric_rows.append(
            {
                "category": category,
                "metric": name,
                **result,
                "observed_sequence_count": total_samples,
            }
        )
        wandb_metrics["%s/%s" % (wandb_namespace, name)] = float(
            result["estimate"]
        )

    # Make the paper's naming explicit: these are aliases of Mean@32, not a
    # separate generation experiment or an accuracy from one privileged draw.
    metrics["pass_at_1"] = {
        **metrics["mean_at_32"],
        "alias_of": "mean_at_32",
    }
    metrics["sample_accuracy"] = {
        **metrics["mean_at_32"],
        "alias_of": "mean_at_32",
    }
    for alias in ("pass_at_1", "sample_accuracy"):
        wandb_metrics["%s/%s" % (wandb_namespace, alias)] = float(
            metrics[alias]["estimate"]
        )
        metric_rows.append(
            {
                "category": category,
                "metric": alias,
                **metrics[alias],
                "observed_sequence_count": total_samples,
            }
        )
    return metrics, metric_rows, wandb_metrics


def aggregate(
    grouped: Mapping[str, Mapping[int, Mapping[str, Any]]]
) -> tuple[Dict[str, Any], list[Dict[str, Any]], Dict[str, float]]:
    if len(grouped) != PAPER_ANCHOR_EXAMPLE_COUNT:
        raise ValueError("paper-anchor aggregate requires exactly 500 examples")
    expected_seeds = set(COMMON_GENERATION_SEEDS)
    if any(set(by_seed) != expected_seeds for by_seed in grouped.values()):
        raise ValueError("paper-anchor aggregate requires the exact 32-seed inventory")
    math_verify_outcomes = {
        example_id: tuple(int(by_seed[seed]["correct"]) for seed in COMMON_GENERATION_SEEDS)
        for example_id, by_seed in grouped.items()
    }
    upstream_rule_outcomes = {
        example_id: tuple(
            int(by_seed[seed]["upstream_rule_correct"])
            for seed in COMMON_GENERATION_SEEDS
        )
        for example_id, by_seed in grouped.items()
    }
    total_samples = PAPER_ANCHOR_EXAMPLE_COUNT * PAPER_ANCHOR_SAMPLE_COUNT
    math_verify_total_correct = sum(sum(values) for values in math_verify_outcomes.values())
    upstream_rule_total_correct = sum(
        sum(values) for values in upstream_rule_outcomes.values()
    )
    metrics, metric_rows, wandb_metrics = _aggregate_binary_outcomes(
        math_verify_outcomes,
        category="math_verify",
        wandb_namespace="paper_anchor/math500/math_verify",
        total_samples=total_samples,
    )
    upstream_metrics, upstream_rows, upstream_wandb = _aggregate_binary_outcomes(
        upstream_rule_outcomes,
        category="upstream_rule_judge",
        wandb_namespace="paper_anchor/math500/upstream_rule_judge",
        total_samples=total_samples,
    )
    metric_rows.extend(upstream_rows)
    wandb_metrics.update(upstream_wandb)

    diagnostic_fields = (
        "response_length",
        "latent_length",
        "hard_answer_length",
        "capped",
        "close_tag",
        "soft_to_hard",
        "all_soft",
        "mixture_entropy",
        "top1_weight",
        "soft_hard_agreement",
    )
    diagnostics: Dict[str, Any] = {}
    for field in diagnostic_fields:
        per_example, observed_sequences = _per_example_diagnostic(grouped, field)
        result = bootstrap_mean(per_example)
        result["observed_sequence_count"] = observed_sequences
        result["conditioning"] = (
            "responses with latent-mixture diagnostics"
            if field in {"mixture_entropy", "top1_weight", "soft_hard_agreement"}
            else "all responses"
        )
        diagnostics[field + "_mean"] = result
        metric_rows.append(
            {
                "category": "native_soft_diagnostic",
                "metric": field + "_mean",
                **result,
                "estimator": "mean of per-example means",
            }
        )
        wandb_metrics["paper_anchor/math500/native_soft/%s_mean" % field] = float(
            result["estimate"]
        )

    response_lengths = np.asarray(
        [
            float(row["response_length"])
            for by_seed in grouped.values()
            for row in by_seed.values()
        ],
        dtype=np.float64,
    )
    latent_lengths = np.asarray(
        [
            float(row["latent_length"])
            for by_seed in grouped.values()
            for row in by_seed.values()
        ],
        dtype=np.float64,
    )
    diagnostics["response_length_p95"] = float(
        np.quantile(response_lengths, 0.95, method="linear")
    )
    diagnostics["latent_length_p95"] = float(
        np.quantile(latent_lengths, 0.95, method="linear")
    )
    diagnostics["response_length_max"] = int(response_lengths.max())
    diagnostics["latent_length_max"] = int(latent_lengths.max())
    for name in (
        "response_length_p95",
        "latent_length_p95",
        "response_length_max",
        "latent_length_max",
    ):
        wandb_metrics["paper_anchor/math500/native_soft/%s" % name] = float(
            diagnostics[name]
        )

    invalid_boundary_count = sum(
        int(bool(row["capped"]) or bool(row["all_soft"]))
        for by_seed in grouped.values()
        for row in by_seed.values()
    )
    invalid_boundary_rate = invalid_boundary_count / total_samples
    boundary_gate = {
        "valid": bool(invalid_boundary_rate <= 0.05),
        "criterion": "at most 5% of responses are capped or all-soft",
        "invalid_response_count": invalid_boundary_count,
        "response_count": total_samples,
        "invalid_rate": invalid_boundary_rate,
    }
    wandb_metrics["paper_anchor/math500/native_soft/boundary_gate_valid"] = float(
        boundary_gate["valid"]
    )
    wandb_metrics["paper_anchor/math500/native_soft/invalid_boundary_rate"] = float(
        invalid_boundary_rate
    )

    upstream_invalid_as_incorrect = {
        example_id: tuple(
            int(by_seed[seed]["upstream_rule_correct"])
            * int(not (bool(by_seed[seed]["capped"]) or bool(by_seed[seed]["all_soft"])))
            for seed in COMMON_GENERATION_SEEDS
        )
        for example_id, by_seed in grouped.items()
    }
    invalid_as_incorrect = bootstrap_mean(
        example_level_metric(upstream_invalid_as_incorrect, "mean_at_32")
    )
    invalid_as_incorrect["estimator"] = (
        "mean correctness over all 32 samples after assigning zero to capped/all-soft "
        "responses"
    )
    valid_boundary_only: Dict[str, float] = {}
    valid_boundary_sequence_count = 0
    upstream_correct_invalid_count = 0
    for example_id, by_seed in grouped.items():
        valid_scores = []
        for seed in COMMON_GENERATION_SEEDS:
            row = by_seed[seed]
            valid = not (bool(row["capped"]) or bool(row["all_soft"]))
            correct = int(row["upstream_rule_correct"])
            if valid:
                valid_scores.append(correct)
            elif correct:
                upstream_correct_invalid_count += 1
        valid_boundary_sequence_count += len(valid_scores)
        if valid_scores:
            valid_boundary_only[example_id] = float(np.mean(valid_scores))
    if not valid_boundary_only:
        raise ValueError("paper anchor has no valid-boundary responses")
    valid_only = bootstrap_mean(valid_boundary_only)
    valid_only["estimator"] = (
        "mean of per-example correctness rates conditional on a valid soft-to-hard "
        "boundary; examples with no valid response are excluded"
    )
    valid_only["included_example_count"] = len(valid_boundary_only)
    valid_only["observed_sequence_count"] = valid_boundary_sequence_count
    upstream_boundary_sensitivity = {
        "invalid_as_incorrect_pass_at_1": invalid_as_incorrect,
        "valid_boundary_only_sample_accuracy": valid_only,
        "correct_invalid_response_count": upstream_correct_invalid_count,
    }
    for name, result in (
        ("invalid_as_incorrect_pass_at_1", invalid_as_incorrect),
        ("valid_boundary_only_sample_accuracy", valid_only),
    ):
        metric_rows.append(
            {
                "category": "upstream_rule_judge_boundary",
                "metric": name,
                **result,
                "observed_sequence_count": result.get(
                    "observed_sequence_count", total_samples
                ),
            }
        )
        wandb_metrics[
            "paper_anchor/math500/upstream_rule_judge/%s" % name
        ] = float(result["estimate"])
    wandb_metrics[
        "paper_anchor/math500/upstream_rule_judge/correct_invalid_response_count"
    ] = float(upstream_correct_invalid_count)

    summary = {
        "schema_version": PAPER_ANCHOR_SCHEMA_VERSION,
        "protocol": PAPER_ANCHOR_PROTOCOL,
        "report_variant": PAPER_ANCHOR_REPORT_VARIANT,
        "model": {
            "label": PAPER_ANCHOR_MODEL_LABEL,
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
        },
        "benchmark": PAPER_ANCHOR_BENCHMARK,
        "inference_mode": PAPER_ANCHOR_MODE,
        "sampling_protocol": PAPER_ANCHOR_SAMPLING_PROTOCOL,
        "generation_seeds": list(COMMON_GENERATION_SEEDS),
        "example_count": PAPER_ANCHOR_EXAMPLE_COUNT,
        "samples_per_example": PAPER_ANCHOR_SAMPLE_COUNT,
        "total_sample_count": total_samples,
        "math_verify": {
            "version": MATH_VERIFY_VERSION,
            "correct_sample_count": math_verify_total_correct,
            "metrics": metrics,
        },
        "upstream_rule_judge": {
            "provenance": upstream_math500_grader_provenance(),
            "correct_sample_count": upstream_rule_total_correct,
            "metrics": upstream_metrics,
            "boundary_sensitivity": upstream_boundary_sensitivity,
        },
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "unit": "MATH-500 example",
            "interval": "nonparametric percentile 95%",
            "quantile_method": "linear",
        },
        "native_soft_diagnostics": diagnostics,
        "boundary_gate": boundary_gate,
    }
    if not wandb_metrics or not all(math.isfinite(value) for value in wandb_metrics.values()):
        raise ValueError("paper-anchor W&B metrics must be finite and non-empty")
    return summary, metric_rows, wandb_metrics


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("cannot write an empty paper-anchor CSV")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def stable_wandb_run_id(
    input_manifest: Mapping[str, Any],
    regrader_source: Mapping[str, str],
    upstream_grader: Mapping[str, Any],
) -> str:
    identity = {
        "protocol": PAPER_ANCHOR_PROTOCOL,
        "report_variant": PAPER_ANCHOR_REPORT_VARIANT,
        "model_tree_sha256": input_manifest["model"]["tree_sha256"],
        "data_manifest_content_sha256": input_manifest[
            "data_manifest_content_sha256"
        ],
        "generation_manifest_sha256": input_manifest["generation_manifest"][
            "sha256"
        ],
        "shard_sha256": [row["sha256"] for row in input_manifest["shards"]],
        "upstream_grader": dict(upstream_grader),
        "regrader_source": dict(regrader_source),
    }
    digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
    return "paper-anchor-base-upstream-rule-" + digest[:16]


def _publish_wandb(
    *,
    report_manifest: Mapping[str, Any],
    metrics: Mapping[str, float],
    output_paths: Sequence[Path],
) -> str:
    if os.environ.get("WANDB_MODE") != "online":
        raise RuntimeError("paper-anchor publication requires WANDB_MODE=online")
    if os.environ.get("WANDB_RESUME", "allow") != "allow":
        raise RuntimeError("paper-anchor publication requires WANDB_RESUME=allow")
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("paper-anchor publication requires wandb") from error
    run_id = report_manifest.get("wandb_run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("report manifest has no stable W&B run ID")
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "opd-softgrpo-icl"),
        group=os.environ.get("WANDB_GROUP", PAPER_ANCHOR_GROUP),
        id=run_id,
        resume="allow",
        job_type="paper-anchor-evaluation",
        tags=[
            "paper-anchor",
            "base",
            "math500",
            "native-soft",
            "upstream-rule-judge",
        ],
        config=dict(report_manifest),
    )
    run.log(dict(metrics))
    for key, value in metrics.items():
        run.summary[key] = value
    run.summary["paper_anchor/completed"] = True
    run.summary["paper_anchor/bootstrap_resamples"] = BOOTSTRAP_RESAMPLES
    artifact = wandb.Artifact(
        name=run_id + "-report",
        type="evaluation-report",
        metadata={
            "protocol": PAPER_ANCHOR_PROTOCOL,
            "report_variant": PAPER_ANCHOR_REPORT_VARIANT,
        },
    )
    for path in output_paths:
        artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)
    run.finish()
    return run_id


def write_report(
    *,
    output_dir: Path,
    input_manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    metric_rows: Sequence[Mapping[str, Any]],
    wandb_metrics: Mapping[str, float],
) -> Dict[str, Any]:
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    regrader_source = required_regrader_source_identity()
    upstream_grader = upstream_math500_grader_provenance()
    summary_path = destination / (PAPER_ANCHOR_REPORT_PREFIX + "_summary.json")
    metrics_path = destination / (PAPER_ANCHOR_REPORT_PREFIX + "_metrics.csv")
    summary_with_provenance = {
        **dict(summary),
        "regrader_source": regrader_source,
    }
    _atomic_write(summary_path, _canonical_json(summary_with_provenance))
    _atomic_write(metrics_path, _csv_bytes(metric_rows))
    report_manifest = {
        "schema_version": PAPER_ANCHOR_SCHEMA_VERSION,
        "protocol": PAPER_ANCHOR_PROTOCOL,
        "report_variant": PAPER_ANCHOR_REPORT_VARIANT,
        "input": dict(input_manifest),
        "regrader_source": regrader_source,
        "upstream_rule_judge": upstream_grader,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "files": {
            path.name: {"size": path.stat().st_size, "sha256": file_sha256(path)}
            for path in (summary_path, metrics_path)
        },
        "wandb_run_id": stable_wandb_run_id(
            input_manifest, regrader_source, upstream_grader
        ),
    }
    manifest_path = destination / (PAPER_ANCHOR_REPORT_PREFIX + "_manifest.json")
    _atomic_write(manifest_path, _canonical_json(report_manifest))
    observed_run_id = _publish_wandb(
        report_manifest=report_manifest,
        metrics=wandb_metrics,
        output_paths=(summary_path, metrics_path, manifest_path),
    )
    if observed_run_id != report_manifest["wandb_run_id"]:
        raise AssertionError("published W&B identity differs from report manifest")
    return report_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--chunk-size", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    # Fail before the expensive 16,000-response scoring pass unless the exact
    # regrader checkout and parser environment can be sealed into the result.
    required_regrader_source_identity()
    upstream_math500_grader_provenance()
    input_manifest = authenticate_input(args.input_dir)
    grouped = score_input(
        input_manifest, workers=args.workers, chunk_size=args.chunk_size
    )
    summary, metric_rows, wandb_metrics = aggregate(grouped)
    report_manifest = write_report(
        output_dir=args.output_dir,
        input_manifest=input_manifest,
        summary=summary,
        metric_rows=metric_rows,
        wandb_metrics=wandb_metrics,
    )
    print(json.dumps(report_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
