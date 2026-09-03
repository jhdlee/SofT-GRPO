"""Read-only CPU preflight for sealed OPD model and data assets.

Usage::

    python -m opd_tools.preflight --data-dir /scratch/.../data \
        --model-dir /scratch/.../assets/model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .assets import verify_model_snapshot
from .constants import MATH_TRAIN_SIZE, MATH_VALIDATION_SIZE, STUDENT_PROMPT_SUFFIX
from .data import canonicalize_math_problem
from .prepare import verify_materialized_data
from .records import student_generation_payload

ParquetReader = Callable[[Path], List[Mapping[str, Any]]]
TokenizerLoader = Callable[[Path], Any]


def _read_parquet(path: Path) -> List[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("asset preflight requires pyarrow") from error
    return parquet.read_table(str(path)).to_pylist()


def _load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("asset preflight requires transformers") from error
    return AutoTokenizer.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=False,
    )


def _question_from_prompt(row: Mapping[str, Any]) -> str:
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or len(prompt) != 1:
        raise ValueError("materialized row must contain one prompt message")
    message = prompt[0]
    if not isinstance(message, Mapping) or message.get("role") != "user":
        raise ValueError("materialized prompt must contain one user message")
    content = message.get("content")
    if not isinstance(content, str) or not content.endswith(STUDENT_PROMPT_SUFFIX):
        raise ValueError("materialized prompt does not use the locked student suffix")
    return content[: -len(STUDENT_PROMPT_SUFFIX)]


def _validate_tokenizer(tokenizer: Any) -> Dict[str, Any]:
    token_ids = {}
    for tag in ("<think>", "</think>"):
        ids = tokenizer.encode(tag, add_special_tokens=False)
        if not isinstance(ids, list) or len(ids) != 1:
            raise ValueError("tokenizer does not encode %s atomically: %r" % (tag, ids))
        if tokenizer.decode(ids, skip_special_tokens=False) != tag:
            raise ValueError("tokenizer does not round-trip %s" % tag)
        token_ids[tag] = int(ids[0])
    if token_ids["<think>"] == token_ids["</think>"]:
        raise ValueError("thinking delimiters share a token ID")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Preflight probe."}],
        add_generation_prompt=True,
        tokenize=False,
    )
    if not isinstance(rendered, str) or not rendered.rstrip().endswith("<think>"):
        raise ValueError("native generation template does not open <think>")
    return {"token_ids": token_ids, "generation_template_opens_think": True}


def run_preflight(
    data_dir: Path,
    model_dir: Path,
    parquet_reader: Optional[ParquetReader] = None,
    tokenizer_loader: Optional[TokenizerLoader] = None,
    enforce_pinned_contract: bool = True,
) -> Dict[str, Any]:
    """Authenticate assets and inspect every materialized training prompt."""

    data_root = Path(data_dir).expanduser().resolve()
    model_root = Path(model_dir).expanduser().resolve()
    data_manifest = verify_materialized_data(data_root, enforce_pinned_contract)
    model_manifest = verify_model_snapshot(model_root)
    read_rows = _read_parquet if parquet_reader is None else parquet_reader
    load_tokenizer = _load_tokenizer if tokenizer_loader is None else tokenizer_loader

    train_rows = read_rows(data_root / "math_lighteval_train.parquet")
    validation_rows = read_rows(data_root / "math_lighteval_validation.parquet")
    math500_rows = read_rows(data_root / "math500_test.parquet")
    expected_counts = (
        (MATH_TRAIN_SIZE, MATH_VALIDATION_SIZE)
        if enforce_pinned_contract
        else (
            data_manifest["files"]["math_lighteval_train.parquet"]["row_count"],
            data_manifest["files"]["math_lighteval_validation.parquet"]["row_count"],
        )
    )
    if (len(train_rows), len(validation_rows)) != expected_counts:
        raise ValueError("Parquet train/validation counts differ from the sealed contract")

    all_ids = set()
    for expected_split, rows in (("train", train_rows), ("validation", validation_rows)):
        for row in rows:
            generation = student_generation_payload(row)
            if generation["split"] != expected_split:
                raise ValueError("student record split differs from its Parquet file")
            example_id = generation["example_id"]
            if example_id in all_ids:
                raise ValueError("train/validation example IDs overlap")
            all_ids.add(example_id)

    train_problem_keys = {
        canonicalize_math_problem(_question_from_prompt(row))
        for row in train_rows + validation_rows
    }
    math500_problem_keys = {
        canonicalize_math_problem(_question_from_prompt(row)) for row in math500_rows
    }
    overlap = train_problem_keys & math500_problem_keys
    if overlap:
        raise ValueError("materialized MATH train/validation overlaps MATH-500")

    tokenizer = load_tokenizer(model_root)
    tokenizer_result = _validate_tokenizer(tokenizer)
    maximum_tokens = -1
    maximum_id = None
    for row in train_rows:
        prompt = row["prompt"]
        token_ids = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
        )
        length = len(token_ids)
        if length > maximum_tokens:
            maximum_tokens = length
            maximum_id = row["extra_info"]["example_id"]
    if maximum_tokens > 1_024:
        raise ValueError(
            "rendered train prompt exceeds 1024 tokens: %s has %d"
            % (maximum_id, maximum_tokens)
        )

    return {
        "status": "ok",
        "data_dir": str(data_root),
        "model_dir": str(model_root),
        "data_manifest_sha256": data_manifest["manifest_content_sha256"],
        "model_inventory_sha256": model_manifest["inventory_sha256"],
        "counts": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "math500": len(math500_rows),
        },
        "student_payloads_checked": len(train_rows) + len(validation_rows),
        "math_train_math500_overlap": 0,
        "max_rendered_train_prompt_tokens": maximum_tokens,
        "max_rendered_train_prompt_example_id": maximum_id,
        "tokenizer": tokenizer_result,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    print(json.dumps(run_preflight(args.data_dir, args.model_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
