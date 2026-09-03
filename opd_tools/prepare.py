"""Atomically materialize all pinned training and evaluation Parquet files.

Usage::

    python -m opd_tools.prepare --output-dir /scratch/.../data \
        --cache-dir /scratch/.../hf-cache
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .constants import (
    GSM8K_DATASET_CONFIG,
    GSM8K_DATASET_ID,
    GSM8K_DATASET_REVISION,
    MATH500_DATASET_CONFIG,
    MATH500_DATASET_ID,
    MATH500_DATASET_REVISION,
    MATH_DATASET_CONFIG,
    MATH_DATASET_ID,
    MATH_DATASET_REVISION,
    MATH_CLEAN_SIZE,
    MATH_DUPLICATE_DROP_INDICES,
    MATH_DUPLICATE_KEEP_BY_DROP,
    MATH_EMPTY_ANSWER_INDICES,
    MATH_RELEASED_EXTRACTOR_DISAGREEMENTS,
    MATH_SOURCE_TRAIN_SIZE,
    MATH_SPLIT_SEED,
    MATH_TRAIN_SIZE,
    MATH_VALIDATION_SIZE,
    MATH_VALIDATION_IDS_SHA256,
    MATERIALIZATION_PROTOCOL,
    RELEASED_EVAL_COUNTS,
    RELEASED_EVAL_FILE_SHA256,
    SOFTGRPO_UPSTREAM_COMMIT,
    MODEL_ID,
    MODEL_REVISION,
)
from .data import (
    build_gsm8k_evaluation_records,
    build_math500_evaluation_records,
    build_released_evaluation_records,
    canonicalize_math_problem,
    evaluation_record_to_verl_row,
    load_pinned_gsm8k_test,
    load_pinned_math500,
    load_pinned_math_train,
    prepare_math_example_splits,
    ordered_example_ids_sha256,
)
from .graders import gsm8k_grader_manifest
from .manifest import (
    build_math_data_manifest,
    canonical_sha256,
    file_sha256,
    ordered_records_sha256,
    validate_manifest_content,
    write_manifest_atomic,
)
from .records import build_record_bundle, build_verl_training_row

OUTPUT_FILENAMES = (
    "math_lighteval_train.parquet",
    "math_lighteval_validation.parquet",
    "math500_test.parquet",
    "gsm8k_test.parquet",
    "aime2024_test.parquet",
    "aime2025_test.parquet",
    "amc23_test.parquet",
)


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        from datasets import Dataset
    except ImportError as error:
        raise RuntimeError("Parquet materialization requires the datasets package") from error
    Dataset.from_list(list(rows)).to_parquet(str(path))
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError("Parquet writer did not produce %s" % path)


def _released_dataset_directory() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "Soft-Thinking+noise+loss-main"
        / "datasets"
    )


def load_released_eval_rows() -> Dict[str, Sequence[Mapping[str, Any]]]:
    root = _released_dataset_directory()
    result = {}
    for benchmark in sorted(RELEASED_EVAL_COUNTS):
        path = root / (benchmark + ".json")
        if not path.is_file():
            raise FileNotFoundError("released evaluation asset is missing: %s" % path)
        observed_hash = file_sha256(path)
        if observed_hash != RELEASED_EVAL_FILE_SHA256[benchmark]:
            raise ValueError("released evaluation asset hash changed: %s" % path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("released evaluation asset is not a JSON list: %s" % path)
        if len(payload) != RELEASED_EVAL_COUNTS[benchmark]:
            raise ValueError("released evaluation asset count changed: %s" % path)
        result[benchmark] = payload
    return result


def _file_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "size": path.stat().st_size,
        "sha256": file_sha256(path),
        "row_count": len(rows),
        "logical_rows_sha256": ordered_records_sha256(rows),
    }


def verify_materialized_data(
    output_dir: Path, enforce_pinned_contract: bool = True
) -> Dict[str, Any]:
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise ValueError("materialized data directory is incomplete: %s" % root)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("materialized data manifest is unreadable") from error
    validate_manifest_content(manifest)
    if manifest.get("materialization_protocol") != MATERIALIZATION_PROTOCOL:
        raise ValueError("materialized data protocol differs")
    if enforce_pinned_contract:
        expected_source = {
            "id": MATH_DATASET_ID,
            "config": MATH_DATASET_CONFIG,
            "revision": MATH_DATASET_REVISION,
        }
        expected_evaluation_sources = {
            "math500": {
                "id": MATH500_DATASET_ID,
                "config": MATH500_DATASET_CONFIG,
                "revision": MATH500_DATASET_REVISION,
            },
            "gsm8k_test": {
                "id": GSM8K_DATASET_ID,
                "config": GSM8K_DATASET_CONFIG,
                "revision": GSM8K_DATASET_REVISION,
            },
            "released_assets": {
                "softgrpo_commit": SOFTGRPO_UPSTREAM_COMMIT,
                "sha256": dict(RELEASED_EVAL_FILE_SHA256),
            },
        }
        split = manifest.get("split", {})
        split_rows = split.get("splits", {}) if isinstance(split, dict) else {}
        observed_counts = {
            name: split_rows.get(name, {}).get("count")
            for name in ("train", "validation")
        }
        expected_file_counts = {
            "math_lighteval_train.parquet": MATH_TRAIN_SIZE,
            "math_lighteval_validation.parquet": MATH_VALIDATION_SIZE,
            "math500_test.parquet": 500,
            "gsm8k_test.parquet": 1_319,
            "aime2024_test.parquet": RELEASED_EVAL_COUNTS["aime2024"],
            "aime2025_test.parquet": RELEASED_EVAL_COUNTS["aime2025"],
            "amc23_test.parquet": RELEASED_EVAL_COUNTS["amc23"],
        }
        observed_file_counts = {
            name: record.get("row_count") if isinstance(record, dict) else None
            for name, record in manifest.get("files", {}).items()
        }
        cleaning = manifest.get("cleaning", {})
        expected_disagreements = [
            {
                "source_index": index,
                "balanced_final_box": values[0],
                "released_preprocessor": values[1],
            }
            for index, values in sorted(MATH_RELEASED_EXTRACTOR_DISAGREEMENTS.items())
        ]
        if (
            manifest.get("pinned_contract_enforced") is not True
            or manifest.get("upstream_commit") != SOFTGRPO_UPSTREAM_COMMIT
            or manifest.get("model") != {"id": MODEL_ID, "revision": MODEL_REVISION}
            or manifest.get("source") != expected_source
            or manifest.get("evaluation_sources") != expected_evaluation_sources
            or manifest.get("gsm8k_grading") != gsm8k_grader_manifest()
            or split.get("seed") != MATH_SPLIT_SEED
            or split.get("validation_ids_sha256") != MATH_VALIDATION_IDS_SHA256
            or observed_counts
            != {"train": MATH_TRAIN_SIZE, "validation": MATH_VALIDATION_SIZE}
            or split.get("method")
            != "subject-level-largest-remainder-sha256-ranking-v1"
            or split.get("strata") != ["type", "level"]
            or observed_file_counts != expected_file_counts
            or not isinstance(cleaning, dict)
            or cleaning.get("source_count") != MATH_SOURCE_TRAIN_SIZE
            or cleaning.get("clean_count") != MATH_CLEAN_SIZE
            or cleaning.get("empty_answer_source_indices")
            != list(MATH_EMPTY_ANSWER_INDICES)
            or cleaning.get("duplicate_drop_source_indices")
            != list(MATH_DUPLICATE_DROP_INDICES)
            or cleaning.get("duplicate_keep_by_drop")
            != {str(key): value for key, value in MATH_DUPLICATE_KEEP_BY_DROP.items()}
            or cleaning.get("answer_extraction") != "balanced-final-box-v1"
            or cleaning.get("released_extractor_disagreements")
            != expected_disagreements
            or manifest.get("overlap_checks", {}).get("math_train_vs_math500_count")
            != 0
        ):
            raise ValueError("materialized data differs from current pinned contract")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(OUTPUT_FILENAMES):
        raise ValueError("materialized data file inventory differs")
    actual = {path.name for path in root.iterdir() if path.is_file()}
    if actual != set(OUTPUT_FILENAMES) | {"manifest.json"}:
        raise ValueError("materialized data directory has unexpected files")
    for name, record in files.items():
        path = root / name
        if (
            not isinstance(record, dict)
            or not path.is_file()
            or path.stat().st_size != record.get("size")
            or file_sha256(path) != record.get("sha256")
        ):
            raise ValueError("materialized Parquet failed authentication: %s" % name)
    return manifest


def materialize_from_rows(
    output_dir: Path,
    math_rows: Sequence[Mapping[str, Any]],
    math500_rows: Sequence[Mapping[str, Any]],
    gsm8k_rows: Sequence[Mapping[str, Any]],
    released_eval_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    validation_size: int = MATH_VALIDATION_SIZE,
    split_seed: int = MATH_SPLIT_SEED,
    enforce_pinned_contract: bool = True,
) -> Dict[str, Any]:
    """Build a complete sibling directory and promote it with one rename."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        return verify_materialized_data(destination, enforce_pinned_contract)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".%s." % destination.name, dir=str(destination.parent))
    )
    try:
        example_splits, cleaning_report = prepare_math_example_splits(
            math_rows,
            validation_size=validation_size,
            seed=split_seed,
            enforce_pinned_contract=enforce_pinned_contract,
        )
        bundle_splits = {
            split: [build_record_bundle(example) for example in examples]
            for split, examples in example_splits.items()
        }
        parquet_rows: Dict[str, Sequence[Mapping[str, Any]]] = {
            "math_lighteval_train.parquet": [
                build_verl_training_row(example) for example in example_splits["train"]
            ],
            "math_lighteval_validation.parquet": [
                build_verl_training_row(example)
                for example in example_splits["validation"]
            ],
        }
        evaluation_records = {
            "math500_test.parquet": build_math500_evaluation_records(
                math500_rows, enforce_pinned_contract
            ),
            "gsm8k_test.parquet": build_gsm8k_evaluation_records(
                gsm8k_rows, enforce_pinned_contract
            ),
        }
        if set(released_eval_rows) != set(RELEASED_EVAL_COUNTS):
            raise ValueError("released evaluation benchmark inventory differs")
        for benchmark in sorted(RELEASED_EVAL_COUNTS):
            records = build_released_evaluation_records(
                benchmark,
                released_eval_rows[benchmark],
                RELEASED_EVAL_COUNTS[benchmark] if enforce_pinned_contract else None,
            )
            evaluation_records[benchmark + "_test.parquet"] = records

        training_problem_keys = {
            canonicalize_math_problem(example.question)
            for examples in example_splits.values()
            for example in examples
        }
        math500_problem_keys = {
            canonicalize_math_problem(record.question)
            for record in evaluation_records["math500_test.parquet"]
        }
        overlap = sorted(training_problem_keys & math500_problem_keys)
        if overlap:
            raise ValueError(
                "cleaned MATH training data overlaps MATH-500 on %d problems"
                % len(overlap)
            )
        for filename, records in evaluation_records.items():
            parquet_rows[filename] = [
                evaluation_record_to_verl_row(record, index)
                for index, record in enumerate(records)
            ]
        if set(parquet_rows) != set(OUTPUT_FILENAMES):
            raise AssertionError("materializer did not construct the exact output inventory")

        for filename in OUTPUT_FILENAMES:
            _write_parquet(temporary / filename, parquet_rows[filename])

        manifest = build_math_data_manifest(
            bundle_splits, cleaning_report, split_seed=split_seed
        )
        manifest.pop("manifest_content_sha256")
        manifest.update(
            {
                "materialization_protocol": MATERIALIZATION_PROTOCOL,
                "pinned_contract_enforced": bool(enforce_pinned_contract),
                "evaluation_sources": {
                    "math500": {
                        "id": MATH500_DATASET_ID,
                        "config": MATH500_DATASET_CONFIG,
                        "revision": MATH500_DATASET_REVISION,
                    },
                    "gsm8k_test": {
                        "id": GSM8K_DATASET_ID,
                        "config": GSM8K_DATASET_CONFIG,
                        "revision": GSM8K_DATASET_REVISION,
                    },
                    "released_assets": {
                        "softgrpo_commit": SOFTGRPO_UPSTREAM_COMMIT,
                        "sha256": dict(RELEASED_EVAL_FILE_SHA256),
                    },
                },
                "gsm8k_grading": gsm8k_grader_manifest(),
                "overlap_checks": {
                    "problem_canonicalization": "whitespace-collapse-v1",
                    "math_train_vs_math500_count": 0,
                    "math_train_vs_math500_overlap_sha256": canonical_sha256(overlap),
                },
                "files": {
                    filename: _file_manifest(temporary / filename, parquet_rows[filename])
                    for filename in OUTPUT_FILENAMES
                },
            }
        )
        manifest["manifest_content_sha256"] = canonical_sha256(manifest)
        manifest["split"]["validation_ids_sha256"] = ordered_example_ids_sha256(
            [example.example_id for example in example_splits["validation"]]
        )
        # The split hash was added after the initial seal; reseal the complete
        # published payload.
        manifest.pop("manifest_content_sha256")
        manifest["manifest_content_sha256"] = canonical_sha256(manifest)
        write_manifest_atomic(temporary / "manifest.json", manifest)
        verify_materialized_data(temporary, enforce_pinned_contract)
        os.replace(str(temporary), str(destination))
        directory_descriptor = os.open(str(destination.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return verify_materialized_data(destination, enforce_pinned_contract)
    except BaseException:
        shutil.rmtree(str(temporary), ignore_errors=True)
        raise


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    destination = args.output_dir.expanduser().resolve()
    if destination.exists():
        manifest = verify_materialized_data(destination)
    else:
        cache = None if args.cache_dir is None else args.cache_dir.expanduser().resolve()
        manifest = materialize_from_rows(
            destination,
            load_pinned_math_train(cache),
            load_pinned_math500(cache),
            load_pinned_gsm8k_test(cache),
            load_released_eval_rows(),
        )
    print(
        json.dumps(
            {
                "output_dir": str(destination),
                "protocol": manifest["materialization_protocol"],
                "manifest_sha256": file_sha256(destination / "manifest.json"),
                "files": {
                    name: record["row_count"] for name, record in manifest["files"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
