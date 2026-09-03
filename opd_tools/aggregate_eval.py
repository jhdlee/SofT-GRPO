"""Grade, aggregate, bootstrap, and publish the final seed-11 evaluation.

The command is intentionally strict: it refuses to report Mean@32 or pass@k
unless initial, baseline, and OPD have exactly the same examples and common
generation seeds.  Raw completions remain in scratch; only compact JSON/CSV
reports and their authenticated manifest are uploaded to W&B.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np

from .evaluation import (
    BENCHMARKS,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    COMMON_GENERATION_SEEDS,
    EVALUATION_PROTOCOL,
    EXPECTED_EXAMPLE_COUNTS,
    HARD_TOKEN_GENERATION_SEEDS,
    INFERENCE_MODES,
    MODEL_LABELS,
    GenerationRecord,
    correctness_by_example,
    example_level_metric,
    graders_for_benchmark,
    paired_bootstrap_difference,
)
from .generate_eval import (
    EVALUATION_SAMPLING_PROTOCOLS,
    GENERATION_IMPLEMENTATION,
    _atomic_write,
    _canonical_json,
    _verify_shard,
    expected_engine_mode,
    expected_sampling_source,
)
from .graders import grade_gsm8k_interfaces, math_verify_full_response_grade
from .manifest import file_sha256


REPORT_SCHEMA_VERSION = 1
REPORT_PROTOCOL = "opd-softgrpo-seed11-report-v1"


def _score_record(value: Mapping[str, Any]) -> Dict[str, Any]:
    record = GenerationRecord.from_mapping(value)
    if record.benchmark == "gsm8k_test":
        interfaces = grade_gsm8k_interfaces(record.response, record.gold_answer)
        scores = {name: bool(result["correct"]) for name, result in interfaces.items()}
    else:
        scores = {
            "math_verify": bool(
                math_verify_full_response_grade(
                    record.response, record.gold_answer
                ).correct
            )
        }
    if set(scores) != set(graders_for_benchmark(record.benchmark)):
        raise AssertionError("grader inventory differs from the benchmark contract")
    return {
        "model_label": record.model_label,
        "benchmark": record.benchmark,
        "example_id": record.example_id,
        "inference_mode": record.inference_mode,
        "sample_index": record.sample_index,
        "generation_seed": record.generation_seed,
        "gold_answer": record.gold_answer,
        "scores": scores,
        "response_token_count": record.response_token_count,
        "capped": record.capped,
        "latent_token_count": record.latent_token_count,
        "hard_token_count": record.hard_token_count,
        "close_tag": record.close_tag,
        "soft_to_hard": record.soft_to_hard,
        "all_soft": record.all_soft,
        "mixture_entropy_mean": record.mixture_entropy_mean,
        "top1_weight_mean": record.top1_weight_mean,
        "soft_hard_agreement": record.soft_hard_agreement,
    }


def _score_chunk(values: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    return [_score_record(value) for value in values]


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "invalid JSON at %s:%d" % (path, line_number)
                ) from error
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


def expected_shards(input_dir: Path) -> list[tuple[Path, Path]]:
    result = []
    for model in MODEL_LABELS:
        for mode in INFERENCE_MODES:
            seeds = (
                COMMON_GENERATION_SEEDS
                if mode == "native_soft"
                else HARD_TOKEN_GENERATION_SEEDS
            )
            for benchmark in BENCHMARKS:
                directory = input_dir / "raw" / model / mode / benchmark
                for seed in seeds:
                    data = directory / ("seed_%d.jsonl" % seed)
                    result.append((data, data.with_suffix(".manifest.json")))
    return result


def authenticate_input(input_dir: Path) -> Dict[str, Any]:
    """Authenticate the exact shard inventory and generation configurations."""

    input_dir = input_dir.expanduser().resolve()
    expected = expected_shards(input_dir)
    expected_paths = {path.resolve() for pair in expected for path in pair}
    observed_paths = {
        path.resolve()
        for path in (input_dir / "raw").rglob("*")
        if path.is_file() and path.name != "generation_manifest.json"
    }
    if observed_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - observed_paths)
        extra = sorted(str(path) for path in observed_paths - expected_paths)
        raise ValueError(
            "evaluation shard inventory differs: missing=%s extra=%s"
            % (missing[:5], extra[:5])
        )

    shard_inventory = []
    for data_path, manifest_path in expected:
        shard = _verify_shard(data_path, manifest_path)
        benchmark = data_path.parent.name
        if shard["row_count"] != EXPECTED_EXAMPLE_COUNTS[benchmark]:
            raise ValueError("generation shard has the wrong benchmark row count")
        shard_inventory.append(
            {
                "path": data_path.relative_to(input_dir).as_posix(),
                "size": shard["size"],
                "sha256": shard["sha256"],
                "row_count": shard["row_count"],
            }
        )

    generation_manifests = []
    for model in MODEL_LABELS:
        for mode in INFERENCE_MODES:
            path = input_dir / "raw" / model / mode / "generation_manifest.json"
            if not path.is_file():
                raise ValueError("missing generation manifest: %s" % path)
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                value.get("evaluation_protocol") != EVALUATION_PROTOCOL
                or value.get("model_label") != model
                or value.get("mode") != mode
                or value.get("benchmarks") != list(BENCHMARKS)
            ):
                raise ValueError("generation manifest differs from the report contract")
            generation_manifests.append(
                {
                    "path": path.relative_to(input_dir).as_posix(),
                    "sha256": file_sha256(path),
                    "content": value,
                }
            )
    sampling_protocols = {
        item["content"].get("sampling_protocol") for item in generation_manifests
    }
    if len(sampling_protocols) != 1:
        raise ValueError("models/modes used different evaluation sampling protocols")
    data_hashes = {
        item["content"].get("data_manifest_content_sha256")
        for item in generation_manifests
    }
    if len(data_hashes) != 1 or None in data_hashes:
        raise ValueError("generation jobs did not use one authenticated dataset")
    by_model: Dict[str, set[str]] = defaultdict(set)
    for item in generation_manifests:
        content = item["content"]
        sampling_protocol = content.get("sampling_protocol")
        expected_seeds = (
            list(COMMON_GENERATION_SEEDS)
            if content["mode"] == "native_soft"
            else list(HARD_TOKEN_GENERATION_SEEDS)
        )
        if content.get("generation_seeds") != expected_seeds:
            raise ValueError("generation manifest has the wrong common seed set")
        if (
            sampling_protocol not in EVALUATION_SAMPLING_PROTOCOLS
            or content.get("sampling")
            != EVALUATION_SAMPLING_PROTOCOLS[sampling_protocol]
            or content.get("sampling_source")
            != expected_sampling_source(content["mode"], sampling_protocol)
            or content.get("generation_implementation")
            != GENERATION_IMPLEMENTATION
            or content.get("engine_mode") != expected_engine_mode(content["mode"])
        ):
            raise ValueError("generation manifest sampler provenance differs")
        model = content.get("model")
        if not isinstance(model, Mapping) or not isinstance(
            model.get("tree_sha256"), str
        ):
            raise ValueError("generation manifest has no model fingerprint")
        by_model[content["model_label"]].add(model["tree_sha256"])
    if any(len(hashes) != 1 for hashes in by_model.values()):
        raise ValueError("soft and hard modes did not use the same model export")
    return {
        "input_dir": str(input_dir),
        "sampling_protocol": sampling_protocols.pop(),
        "data_manifest_content_sha256": data_hashes.pop(),
        "generation_manifests": generation_manifests,
        "shards": shard_inventory,
    }


class _Accumulator:
    def __init__(self) -> None:
        self.keys: set[tuple[str, str, str, str, int]] = set()
        self.outcomes: MutableMapping[
            tuple[str, str, str, str], MutableMapping[str, Dict[int, bool]]
        ] = defaultdict(lambda: defaultdict(dict))
        self.behavior: MutableMapping[tuple[str, str, str], Dict[str, list[float]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        self.gold: Dict[tuple[str, str], str] = {}

    def add(self, row: Mapping[str, Any]) -> None:
        model = row["model_label"]
        benchmark = row["benchmark"]
        example_id = row["example_id"]
        mode = row["inference_mode"]
        seed = row["generation_seed"]
        key = (model, benchmark, example_id, mode, seed)
        if key in self.keys:
            raise ValueError("duplicate scored row: %r" % (key,))
        self.keys.add(key)
        gold_key = (benchmark, example_id)
        previous = self.gold.setdefault(gold_key, row["gold_answer"])
        if previous != row["gold_answer"]:
            raise ValueError("gold answer differs across paired rows")
        expected_graders = set(graders_for_benchmark(benchmark))
        if set(row["scores"]) != expected_graders:
            raise ValueError("scored row has the wrong graders")
        for grader, score in row["scores"].items():
            by_seed = self.outcomes[(model, benchmark, mode, grader)][example_id]
            if seed in by_seed:
                raise ValueError("duplicate grader example/seed pair")
            by_seed[seed] = bool(score)

        group = self.behavior[(model, benchmark, mode)]
        group["response_length"].append(float(row["response_token_count"]))
        group["cap"].append(float(row["capped"]))
        group["latent_length"].append(float(row["latent_token_count"]))
        group["hard_length"].append(float(row["hard_token_count"]))
        group["close_tag"].append(float(row["close_tag"]))
        group["soft_to_hard"].append(float(row["soft_to_hard"]))
        group["all_soft"].append(float(row["all_soft"]))
        for source, destination in (
            ("mixture_entropy_mean", "mixture_entropy"),
            ("top1_weight_mean", "top1_weight"),
            ("soft_hard_agreement", "soft_hard_agreement"),
        ):
            value = row[source]
            if value is not None:
                group[destination].append(float(value))

    def validate(self) -> None:
        expected_groups = {
            (model, benchmark, mode, grader)
            for model in MODEL_LABELS
            for benchmark in BENCHMARKS
            for mode in INFERENCE_MODES
            for grader in graders_for_benchmark(benchmark)
        }
        if set(self.outcomes) != expected_groups:
            raise ValueError("scored evaluation matrix is incomplete")
        example_sets: Dict[str, set[str]] = {}
        for (model, benchmark, mode, grader), examples in self.outcomes.items():
            del grader
            if len(examples) != EXPECTED_EXAMPLE_COUNTS[benchmark]:
                raise ValueError("scored benchmark has the wrong number of examples")
            expected_seeds = set(
                COMMON_GENERATION_SEEDS
                if mode == "native_soft"
                else HARD_TOKEN_GENERATION_SEEDS
            )
            if any(set(values) != expected_seeds for values in examples.values()):
                raise ValueError("scored example is missing common seeds")
            reference = example_sets.setdefault(benchmark, set(examples))
            if reference != set(examples):
                raise ValueError("scored rows are not paired by example")


def score_input(
    input_manifest: Mapping[str, Any], *, workers: int, chunk_size: int
) -> _Accumulator:
    if workers <= 0 or chunk_size <= 0:
        raise ValueError("workers and chunk_size must be positive")
    root = Path(input_manifest["input_dir"])
    accumulator = _Accumulator()
    paths = [root / row["path"] for row in input_manifest["shards"]]
    if workers == 1:
        for path in paths:
            for chunk in _chunks(_read_jsonl(path), chunk_size):
                for row in _score_chunk(chunk):
                    accumulator.add(row)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Map one authenticated shard at a time.  This bounds queued full
            # responses while keeping CPU parsing parallel.
            for path in paths:
                chunks = _chunks(_read_jsonl(path), chunk_size)
                for scored_chunk in executor.map(_score_chunk, chunks, chunksize=1):
                    for row in scored_chunk:
                        accumulator.add(row)
    accumulator.validate()
    return accumulator


def _ordered_outcomes(
    accumulator: _Accumulator,
    model: str,
    benchmark: str,
    mode: str,
    grader: str,
) -> Dict[str, tuple[int, ...]]:
    expected_seeds = (
        COMMON_GENERATION_SEEDS
        if mode == "native_soft"
        else HARD_TOKEN_GENERATION_SEEDS
    )
    rows = []
    for example_id, by_seed in accumulator.outcomes[
        (model, benchmark, mode, grader)
    ].items():
        for seed, score in by_seed.items():
            rows.append(
                {
                    "example_id": example_id,
                    "generation_seed": seed,
                    "scores": {grader: bool(score)},
                }
            )
    return correctness_by_example(rows, grader=grader, expected_seeds=expected_seeds)


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def aggregate(
    accumulator: _Accumulator,
) -> tuple[Dict[str, Any], list[Dict[str, Any]], list[Dict[str, Any]]]:
    summary: Dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_protocol": REPORT_PROTOCOL,
        "study_seed": 11,
        "exploratory_single_seed": True,
        "generation_seeds": list(COMMON_GENERATION_SEEDS),
        "hard_token_generation_seeds": list(HARD_TOKEN_GENERATION_SEEDS),
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "unit": "paired example",
            "interval": "percentile 95%",
            "quantile_method": "linear",
        },
        "models": {},
        "comparisons": [],
    }
    metric_rows: list[Dict[str, Any]] = []
    example_metrics: Dict[tuple[str, str, str, str, str], Dict[str, float]] = {}

    for model in MODEL_LABELS:
        model_summary: Dict[str, Any] = {"benchmarks": {}}
        summary["models"][model] = model_summary
        for benchmark in BENCHMARKS:
            benchmark_summary: Dict[str, Any] = {}
            model_summary["benchmarks"][benchmark] = benchmark_summary
            for mode in INFERENCE_MODES:
                behavior = accumulator.behavior[(model, benchmark, mode)]
                behavior_summary = {
                    "response_length_mean": _mean(behavior["response_length"]),
                    "cap_rate": _mean(behavior["cap"]),
                    "latent_length_mean": _mean(behavior["latent_length"]),
                    "hard_answer_length_mean": _mean(behavior["hard_length"]),
                    "close_tag_rate": _mean(behavior["close_tag"]),
                    "soft_to_hard_rate": _mean(behavior["soft_to_hard"]),
                    "all_soft_rate": _mean(behavior["all_soft"]),
                    "mixture_entropy_mean": _mean(behavior["mixture_entropy"]),
                    "top1_weight_mean": _mean(behavior["top1_weight"]),
                    "soft_hard_agreement": _mean(behavior["soft_hard_agreement"]),
                }
                mode_summary: Dict[str, Any] = {
                    **behavior_summary,
                    "graders": {},
                }
                benchmark_summary[mode] = mode_summary
                for grader in graders_for_benchmark(benchmark):
                    outcomes = _ordered_outcomes(
                        accumulator, model, benchmark, mode, grader
                    )
                    if mode == "native_soft":
                        metric_names = (
                            "mean_at_32",
                            "pass_at_8",
                            "pass_at_16",
                            "pass_at_32",
                        )
                    else:
                        metric_names = ("accuracy",)
                    values: Dict[str, float] = {}
                    for metric in metric_names:
                        per_example = example_level_metric(outcomes, metric)
                        example_metrics[(model, benchmark, mode, grader, metric)] = (
                            per_example
                        )
                        values[metric] = float(np.mean(list(per_example.values())))
                    mode_summary["graders"][grader] = values
                    metric_rows.append(
                        {
                            "model": model,
                            "benchmark": benchmark,
                            "inference_mode": mode,
                            "grader": grader,
                            "example_count": len(outcomes),
                            "sample_count": len(next(iter(outcomes.values()))),
                            **{
                                name: values.get(name)
                                for name in (
                                    "mean_at_32",
                                    "pass_at_8",
                                    "pass_at_16",
                                    "pass_at_32",
                                    "accuracy",
                                )
                            },
                            **behavior_summary,
                        }
                    )

    comparison_rows: list[Dict[str, Any]] = []
    comparisons = (
        ("baseline", "initial"),
        ("opd", "initial"),
        ("opd", "baseline"),
    )
    for treatment, control in comparisons:
        for benchmark in BENCHMARKS:
            for mode in INFERENCE_MODES:
                for grader in graders_for_benchmark(benchmark):
                    metric_names = (
                        ("mean_at_32", "pass_at_8", "pass_at_16", "pass_at_32")
                        if mode == "native_soft"
                        else ("accuracy",)
                    )
                    for metric in metric_names:
                        result = paired_bootstrap_difference(
                            example_metrics[
                                (treatment, benchmark, mode, grader, metric)
                            ],
                            example_metrics[(control, benchmark, mode, grader, metric)],
                        )
                        row = {
                            "treatment": treatment,
                            "control": control,
                            "benchmark": benchmark,
                            "inference_mode": mode,
                            "grader": grader,
                            "metric": metric,
                            **result,
                        }
                        comparison_rows.append(row)
                        summary["comparisons"].append(row)
    return summary, metric_rows, comparison_rows


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError("CSV rows do not have one stable schema")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _flatten_wandb_metrics(
    metric_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    behavior_groups: set[tuple[str, str, str]] = set()
    for row in metric_rows:
        behavior_key = (row["benchmark"], row["model"], row["inference_mode"])
        if behavior_key not in behavior_groups:
            behavior_groups.add(behavior_key)
            behavior_prefix = "eval/%s/%s/%s" % behavior_key
            for name in (
                "response_length_mean",
                "cap_rate",
                "latent_length_mean",
                "hard_answer_length_mean",
                "close_tag_rate",
                "soft_to_hard_rate",
                "all_soft_rate",
                "mixture_entropy_mean",
                "top1_weight_mean",
                "soft_hard_agreement",
            ):
                value = row.get(name)
                if value is not None:
                    result[behavior_prefix + "/" + name] = float(value)
        prefix = "eval/%s/%s/%s/%s" % (
            row["benchmark"],
            row["model"],
            row["inference_mode"],
            row["grader"],
        )
        for name in (
            "mean_at_32",
            "pass_at_8",
            "pass_at_16",
            "pass_at_32",
            "accuracy",
        ):
            value = row.get(name)
            if value is not None:
                result[prefix + "/" + name] = float(value)
    for row in comparison_rows:
        prefix = "eval_difference/%s/%s_minus_%s/%s/%s/%s" % (
            row["benchmark"],
            row["treatment"],
            row["control"],
            row["inference_mode"],
            row["grader"],
            row["metric"],
        )
        result[prefix] = float(row["difference"])
        result[prefix + "_ci_low"] = float(row["ci_low"])
        result[prefix + "_ci_high"] = float(row["ci_high"])
    if not result or not all(math.isfinite(value) for value in result.values()):
        raise ValueError("W&B metrics must be non-empty and finite")
    return result


def _publish_wandb(
    *,
    report_manifest: Mapping[str, Any],
    metric_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    output_paths: Sequence[Path],
) -> str:
    if os.environ.get("WANDB_MODE") != "online":
        raise RuntimeError("final evaluation publication requires WANDB_MODE=online")
    if os.environ.get("WANDB_RESUME", "allow") != "allow":
        raise RuntimeError("final evaluation publication requires WANDB_RESUME=allow")
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("final evaluation publication requires wandb") from error
    identity = hashlib.sha256(_canonical_json(report_manifest)).hexdigest()
    run_id = report_manifest.get("wandb_run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("report manifest must contain its stable W&B run ID")
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "opd-softgrpo-math"),
        group="seed-11",
        id=run_id,
        resume="allow",
        job_type="evaluation-summary",
        tags=["evaluation", "seed-11", "paired-bootstrap"],
        config=dict(report_manifest),
    )
    metrics = _flatten_wandb_metrics(metric_rows, comparison_rows)
    run.log(metrics)
    for key, value in metrics.items():
        run.summary[key] = value
    run.summary["evaluation/bootstrap_resamples"] = BOOTSTRAP_RESAMPLES
    run.summary["evaluation/bootstrap_seed"] = BOOTSTRAP_SEED
    artifact = wandb.Artifact(
        name="opd-softgrpo-math-seed11-evaluation",
        type="evaluation-report",
        metadata={"report_manifest_sha256": identity},
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
    comparison_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    metrics_path = output_dir / "benchmark_metrics.csv"
    comparisons_path = output_dir / "paired_differences.csv"
    _atomic_write(summary_path, _canonical_json(summary))
    _atomic_write(metrics_path, _csv_bytes(metric_rows))
    _atomic_write(comparisons_path, _csv_bytes(comparison_rows))
    report_manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_protocol": REPORT_PROTOCOL,
        "study_seed": 11,
        "exploratory_single_seed": True,
        "input": input_manifest,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "files": {
            path.name: {
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in (summary_path, metrics_path, comparisons_path)
        },
    }
    stable_identity = hashlib.sha256(_canonical_json(report_manifest)).hexdigest()
    report_manifest["wandb_run_id"] = "eval-summary-seed11-" + stable_identity[:16]
    manifest_path = output_dir / "report_manifest.json"
    _atomic_write(manifest_path, _canonical_json(report_manifest))
    run_id = _publish_wandb(
        report_manifest=report_manifest,
        metric_rows=metric_rows,
        comparison_rows=comparison_rows,
        output_paths=(summary_path, metrics_path, comparisons_path, manifest_path),
    )
    if run_id != report_manifest["wandb_run_id"]:
        raise AssertionError("published W&B run ID differs from report manifest")
    return report_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--chunk-size", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_manifest = authenticate_input(Path(args.input_dir))
    accumulator = score_input(
        input_manifest, workers=args.workers, chunk_size=args.chunk_size
    )
    summary, metric_rows, comparison_rows = aggregate(accumulator)
    summary["sampling_protocol"] = input_manifest["sampling_protocol"]
    manifest = write_report(
        output_dir=Path(args.output_dir),
        input_manifest=input_manifest,
        summary=summary,
        metric_rows=metric_rows,
        comparison_rows=comparison_rows,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
