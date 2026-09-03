"""Pinned, deterministic data preparation for MATH and evaluation sets."""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .constants import (
    GSM8K_DATASET_CONFIG,
    GSM8K_DATASET_ID,
    GSM8K_DATASET_REVISION,
    GSM8K_TEST_SIZE,
    MATH500_DATASET_CONFIG,
    MATH500_DATASET_ID,
    MATH500_DATASET_REVISION,
    MATH500_TEST_SIZE,
    MATH_CLEAN_SIZE,
    MATH_DATASET_CONFIG,
    MATH_DATASET_ID,
    MATH_DATASET_REVISION,
    MATH_DUPLICATE_DROP_INDICES,
    MATH_DUPLICATE_KEEP_BY_DROP,
    MATH_EMPTY_ANSWER_INDICES,
    MATH_RELEASED_EXTRACTOR_DISAGREEMENT_INDICES,
    MATH_RELEASED_EXTRACTOR_DISAGREEMENTS,
    MATH_SOURCE_TRAIN_SIZE,
    MATH_SPLIT_SEED,
    MATH_TRAIN_SIZE,
    MATH_VALIDATION_SIZE,
    MATH_VALIDATION_IDS_SHA256,
)
from .records import (
    EvaluationRecord,
    MathExample,
    RecordBundle,
    build_record_bundle,
    build_verl_training_row,
    render_student_user_content,
)

_MATH_FIELDS = frozenset({"problem", "level", "solution", "type"})
_MATH500_FIELDS = frozenset(
    {"problem", "solution", "answer", "subject", "level", "unique_id"}
)


@dataclass(frozen=True)
class MathCleaningReport:
    source_count: int
    clean_count: int
    empty_answer_source_indices: Tuple[int, ...]
    duplicate_drop_source_indices: Tuple[int, ...]
    duplicate_keep_by_drop: Tuple[Tuple[int, int], ...]
    released_extractor_disagreement_source_indices: Tuple[int, ...]
    released_extractor_disagreements: Tuple[Tuple[int, str, str], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_count": self.source_count,
            "clean_count": self.clean_count,
            "empty_answer_source_indices": list(self.empty_answer_source_indices),
            "duplicate_drop_source_indices": list(self.duplicate_drop_source_indices),
            "duplicate_keep_by_drop": {
                str(dropped): kept for dropped, kept in self.duplicate_keep_by_drop
            },
            "answer_extraction": "balanced-final-box-v1",
            "released_extractor_disagreement_source_indices": list(
                self.released_extractor_disagreement_source_indices
            ),
            "released_extractor_disagreements": [
                {
                    "source_index": source_index,
                    "balanced_final_box": balanced,
                    "released_preprocessor": released,
                }
                for source_index, balanced, released in self.released_extractor_disagreements
            ],
        }


def _require_string(row: Mapping[str, Any], key: str, source_index: int) -> str:
    try:
        value = row[key]
    except KeyError as error:
        raise ValueError("row %d is missing %r" % (source_index, key)) from error
    if not isinstance(value, str) or not value.strip():
        raise ValueError("row %d has an invalid %s" % (source_index, key))
    return value


def canonicalize_math_problem(question: str) -> str:
    """Canonicalize only whitespace for known duplicate-problem detection."""

    return re.sub(r"\s+", " ", question).strip()


def extract_last_boxed_answer(solution: str) -> Optional[str]:
    """Extract the final balanced ``\\boxed`` payload; preserve its inner text."""

    if not isinstance(solution, str):
        raise TypeError("solution must be a string")
    index = solution.rfind("\\boxed")
    if index < 0:
        index = solution.rfind("\\fbox")
        if index < 0:
            return None
    cursor = index + len("\\boxed")
    if solution.startswith("\\fbox", index):
        cursor = index + len("\\fbox")
    while cursor < len(solution) and solution[cursor].isspace():
        cursor += 1
    if cursor >= len(solution):
        return None
    if solution[cursor] != "{":
        # Preserve the released ``\\boxed answer`` convention up to math/end
        # punctuation. MATH-lighteval normally uses balanced braces.
        tail = solution[cursor:].split("$", 1)[0].strip()
        return tail.rstrip(".").strip()
    start = cursor + 1
    depth = 1
    cursor = start
    while cursor < len(solution):
        char = solution[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return solution[start:cursor]
        cursor += 1
    return None


def gold_cot_without_final_box(solution: str) -> str:
    """Remove the final boxed expression while retaining all other rationale."""

    index = solution.rfind("\\boxed")
    if index < 0:
        index = solution.rfind("\\fbox")
        if index < 0:
            raise ValueError("solution has no final boxed expression")
    cursor = index + (len("\\boxed") if solution.startswith("\\boxed", index) else len("\\fbox"))
    while cursor < len(solution) and solution[cursor].isspace():
        cursor += 1
    if cursor >= len(solution):
        raise ValueError("solution has an incomplete final boxed expression")
    if solution[cursor] != "{":
        end = solution.find("$", cursor)
        if end < 0:
            end = len(solution)
    else:
        depth = 1
        end = cursor + 1
        while end < len(solution) and depth:
            if solution[end] == "{":
                depth += 1
            elif solution[end] == "}":
                depth -= 1
            end += 1
        if depth:
            raise ValueError("solution has an unbalanced final boxed expression")
    prefix = solution[:index]
    suffix = solution[end:]
    # A boxed expression is commonly the sole content of ``$...$``. Remove
    # that now-empty delimiter pair and repair only whitespace before terminal
    # punctuation; do not rewrite any earlier rationale text.
    if prefix.endswith("$") and suffix.startswith("$"):
        prefix = prefix[:-1]
        suffix = suffix[1:]
    rationale = (prefix + suffix).strip()
    rationale = re.sub(r"[ \t]+([.,;:])", r"\1", rationale)
    if not rationale:
        raise ValueError("removing the final boxed answer left an empty rationale")
    return rationale


def _extract_with_released_preprocessor(solution: str) -> Optional[str]:
    """Expose the released global boxed-space quirk for provenance only."""

    if "\\boxed " in solution:
        boxed = "\\boxed " + solution.split("\\boxed ")[-1].split("$")[0]
        return boxed[len("\\boxed ") :]
    boxed = extract_last_boxed_answer(solution)
    return boxed


def clean_math_lighteval(
    rows: Sequence[Mapping[str, Any]],
    enforce_pinned_contract: bool = True,
) -> Tuple[List[MathExample], MathCleaningReport]:
    """Remove the two empty boxed answers and the duplicate problem.

    At the pinned revision, source rows 5341 and 5343 contain non-empty
    rationales ending in ``\\boxed{}``; they are not empty solution strings.
    Rows 925 and 959 have the same problem after whitespace canonicalization
    but independently worded rationales. The earlier row is retained.
    """

    if enforce_pinned_contract and len(rows) != MATH_SOURCE_TRAIN_SIZE:
        raise ValueError(
            "pinned MATH train must contain %d rows, found %d"
            % (MATH_SOURCE_TRAIN_SIZE, len(rows))
        )

    cleaned: List[MathExample] = []
    empty_answers: List[int] = []
    duplicate_drops: List[int] = []
    duplicate_keep: List[Tuple[int, int]] = []
    extractor_disagreements: List[int] = []
    extractor_disagreement_values: List[Tuple[int, str, str]] = []
    first_by_question: Dict[str, int] = {}

    for source_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("row %d is not a mapping" % source_index)
        missing = _MATH_FIELDS - set(row)
        if missing:
            raise ValueError("row %d is missing fields %s" % (source_index, sorted(missing)))
        problem = _require_string(row, "problem", source_index)
        level = _require_string(row, "level", source_index)
        solution = _require_string(row, "solution", source_index)
        subject = _require_string(row, "type", source_index)
        answer = extract_last_boxed_answer(solution)
        if answer is None:
            raise ValueError("row %d has no balanced final boxed answer" % source_index)
        if not answer.strip():
            empty_answers.append(source_index)
            continue
        released_answer = _extract_with_released_preprocessor(solution)
        if released_answer != answer:
            extractor_disagreements.append(source_index)
            extractor_disagreement_values.append(
                (source_index, answer, "" if released_answer is None else released_answer)
            )

        duplicate_key = canonicalize_math_problem(problem)
        if duplicate_key in first_by_question:
            kept_index = first_by_question[duplicate_key]
            duplicate_drops.append(source_index)
            duplicate_keep.append((source_index, kept_index))
            continue
        first_by_question[duplicate_key] = source_index
        cleaned.append(
            MathExample(
                example_id="math-train-%06d" % source_index,
                source_index=source_index,
                question=problem,
                gold_solution=solution,
                gold_cot=gold_cot_without_final_box(solution),
                gold_answer=answer,
                subject=subject,
                level=level,
            )
        )

    report = MathCleaningReport(
        source_count=len(rows),
        clean_count=len(cleaned),
        empty_answer_source_indices=tuple(empty_answers),
        duplicate_drop_source_indices=tuple(duplicate_drops),
        duplicate_keep_by_drop=tuple(duplicate_keep),
        released_extractor_disagreement_source_indices=tuple(extractor_disagreements),
        released_extractor_disagreements=tuple(extractor_disagreement_values),
    )
    if enforce_pinned_contract:
        expected_pairs = tuple(sorted(MATH_DUPLICATE_KEEP_BY_DROP.items()))
        if tuple(empty_answers) != MATH_EMPTY_ANSWER_INDICES:
            raise ValueError("pinned empty-answer rows changed: %r" % (empty_answers,))
        if tuple(duplicate_drops) != MATH_DUPLICATE_DROP_INDICES:
            raise ValueError("pinned duplicate rows changed: %r" % (duplicate_drops,))
        if tuple(duplicate_keep) != expected_pairs:
            raise ValueError("pinned duplicate pairing changed: %r" % (duplicate_keep,))
        if tuple(extractor_disagreements) != MATH_RELEASED_EXTRACTOR_DISAGREEMENT_INDICES:
            raise ValueError(
                "pinned released-extractor disagreements changed: %r"
                % (extractor_disagreements,)
            )
        expected_disagreements = tuple(
            (index, values[0], values[1])
            for index, values in sorted(MATH_RELEASED_EXTRACTOR_DISAGREEMENTS.items())
        )
        if tuple(extractor_disagreement_values) != expected_disagreements:
            raise ValueError(
                "pinned released-extractor values changed: %r"
                % (extractor_disagreement_values,)
            )
        if len(cleaned) != MATH_CLEAN_SIZE:
            raise ValueError("pinned cleaned MATH size changed: %d" % len(cleaned))
    return cleaned, report


def _stratum_rank(
    seed: int, stratum: Tuple[str, str], example_id: str
) -> Tuple[bytes, str]:
    """Cross-runtime deterministic rank replacing PRNG-dependent shuffling."""

    digest = hashlib.sha256()
    digest.update(b"opd-math-stratum-rank-v1\0")
    digest.update(str(seed).encode("ascii"))
    digest.update(b"\0")
    for value in stratum:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    digest.update(example_id.encode("utf-8"))
    return digest.digest(), example_id


def _validation_allocations(
    strata: Mapping[Tuple[str, str], Sequence[MathExample]], validation_size: int
) -> Dict[Tuple[str, str], int]:
    total = sum(len(examples) for examples in strata.values())
    if validation_size <= 0 or validation_size >= total:
        raise ValueError("validation_size must leave non-empty train and validation sets")
    exact = {
        stratum: len(examples) * validation_size / total
        for stratum, examples in strata.items()
    }
    allocation = {
        stratum: min(int(math.floor(count)), max(0, len(strata[stratum]) - 1))
        for stratum, count in exact.items()
    }
    remaining = validation_size - sum(allocation.values())
    order = sorted(
        strata,
        key=lambda stratum: (-(exact[stratum] - math.floor(exact[stratum])), stratum),
    )
    while remaining:
        progressed = False
        for stratum in order:
            if allocation[stratum] < len(strata[stratum]) - 1:
                allocation[stratum] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise ValueError("strata cannot supply the requested validation size")
    return allocation


def ordered_example_ids_sha256(example_ids: Sequence[str]) -> str:
    digest = hashlib.sha256(b"opd-math-validation-ids-v1\0")
    for example_id in example_ids:
        encoded = example_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def stratified_math_split(
    examples: Sequence[MathExample],
    validation_size: int = MATH_VALIDATION_SIZE,
    seed: int = MATH_SPLIT_SEED,
) -> Dict[str, List[MathExample]]:
    """Split proportionally over ``(subject, level)`` with stable tie-breaking."""

    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError("cleaned MATH example IDs must be unique")
    strata: Dict[Tuple[str, str], List[MathExample]] = defaultdict(list)
    for example in examples:
        strata[(example.subject, example.level)].append(example)
    allocation = _validation_allocations(strata, validation_size)

    validation_ids = set()
    for stratum in sorted(strata):
        candidates = sorted(
            strata[stratum],
            key=lambda example: _stratum_rank(seed, stratum, example.example_id),
        )
        validation_ids.update(
            example.example_id for example in candidates[: allocation[stratum]]
        )
    train = [
        replace(example, split="train")
        for example in examples
        if example.example_id not in validation_ids
    ]
    validation = [
        replace(example, split="validation")
        for example in examples
        if example.example_id in validation_ids
    ]
    if len(validation) != validation_size or len(train) + len(validation) != len(examples):
        raise AssertionError("stratified split lost or duplicated examples")
    return {"train": train, "validation": validation}


def prepare_math_example_splits(
    rows: Sequence[Mapping[str, Any]],
    validation_size: int = MATH_VALIDATION_SIZE,
    seed: int = MATH_SPLIT_SEED,
    enforce_pinned_contract: bool = True,
) -> Tuple[Dict[str, List[MathExample]], MathCleaningReport]:
    cleaned, report = clean_math_lighteval(rows, enforce_pinned_contract)
    split_examples = stratified_math_split(cleaned, validation_size, seed)
    if enforce_pinned_contract:
        observed = {name: len(values) for name, values in split_examples.items()}
        expected = {"train": MATH_TRAIN_SIZE, "validation": MATH_VALIDATION_SIZE}
        if observed != expected or seed != MATH_SPLIT_SEED:
            raise ValueError("pinned split contract changed: %r" % observed)
        validation_hash = ordered_example_ids_sha256(
            [example.example_id for example in split_examples["validation"]]
        )
        if validation_hash != MATH_VALIDATION_IDS_SHA256:
            raise ValueError("pinned validation membership changed: %s" % validation_hash)
    return split_examples, report


def prepare_math_training_splits(
    rows: Sequence[Mapping[str, Any]],
    validation_size: int = MATH_VALIDATION_SIZE,
    seed: int = MATH_SPLIT_SEED,
    enforce_pinned_contract: bool = True,
) -> Tuple[Dict[str, List[RecordBundle]], MathCleaningReport]:
    split_examples, report = prepare_math_example_splits(
        rows,
        validation_size=validation_size,
        seed=seed,
        enforce_pinned_contract=enforce_pinned_contract,
    )
    return (
        {
            split: [build_record_bundle(example) for example in examples]
            for split, examples in split_examples.items()
        },
        report,
    )


def prepare_verl_math_splits(
    rows: Sequence[Mapping[str, Any]],
    validation_size: int = MATH_VALIDATION_SIZE,
    seed: int = MATH_SPLIT_SEED,
    enforce_pinned_contract: bool = True,
) -> Tuple[Dict[str, List[Dict[str, Any]]], MathCleaningReport]:
    split_examples, report = prepare_math_example_splits(
        rows,
        validation_size=validation_size,
        seed=seed,
        enforce_pinned_contract=enforce_pinned_contract,
    )
    return (
        {
            split: [build_verl_training_row(example) for example in examples]
            for split, examples in split_examples.items()
        },
        report,
    )


def build_math500_evaluation_records(
    rows: Sequence[Mapping[str, Any]], enforce_pinned_contract: bool = True
) -> List[EvaluationRecord]:
    if enforce_pinned_contract and len(rows) != MATH500_TEST_SIZE:
        raise ValueError("pinned MATH-500 must contain 500 rows")
    records = []
    for index, row in enumerate(rows):
        missing = _MATH500_FIELDS - set(row)
        if missing:
            raise ValueError("MATH-500 row %d is missing %s" % (index, sorted(missing)))
        records.append(
            EvaluationRecord(
                example_id=_require_string(row, "unique_id", index),
                benchmark="math500",
                question=_require_string(row, "problem", index),
                gold_answer=_require_string(row, "answer", index),
                gold_solution=_require_string(row, "solution", index),
            )
        )
    if len({record.example_id for record in records}) != len(records):
        raise ValueError("MATH-500 unique_id values are not unique")
    return records


def parse_gsm8k_gold(answer: str) -> Tuple[str, str]:
    if not isinstance(answer, str) or "####" not in answer:
        raise ValueError("GSM8K answer must contain the #### delimiter")
    rationale, final = answer.rsplit("####", 1)
    normalized = final.strip().replace(",", "")
    if not normalized:
        raise ValueError("GSM8K final answer is empty")
    return rationale, normalized


def build_gsm8k_evaluation_records(
    rows: Sequence[Mapping[str, Any]], enforce_pinned_contract: bool = True
) -> List[EvaluationRecord]:
    if enforce_pinned_contract and len(rows) != GSM8K_TEST_SIZE:
        raise ValueError("pinned GSM8K test must contain 1319 rows")
    records = []
    for index, row in enumerate(rows):
        question = _require_string(row, "question", index)
        raw_answer = _require_string(row, "answer", index)
        rationale, answer = parse_gsm8k_gold(raw_answer)
        records.append(
            EvaluationRecord(
                example_id="gsm8k-test-%06d" % index,
                benchmark="gsm8k_test",
                question=question,
                gold_answer=answer,
                gold_solution=rationale,
            )
        )
    return records


def build_released_evaluation_records(
    benchmark: str,
    rows: Sequence[Mapping[str, Any]],
    expected_count: Optional[int] = None,
) -> List[EvaluationRecord]:
    """Convert the JSON schema bundled with released SofT-GRPO."""

    if expected_count is not None and len(rows) != expected_count:
        raise ValueError("%s must contain %d rows" % (benchmark, expected_count))
    records = []
    for index, row in enumerate(rows):
        try:
            prompt = row["prompt"]
            answer = row["final_answer"]
        except KeyError as error:
            raise ValueError("%s row %d lacks %s" % (benchmark, index, error.args[0])) from error
        if not isinstance(prompt, list) or len(prompt) != 1:
            raise ValueError("%s row %d must have one prompt" % (benchmark, index))
        message = prompt[0]
        if not isinstance(message, Mapping) or set(message) != {"from", "value"}:
            raise ValueError("%s row %d has an invalid prompt schema" % (benchmark, index))
        if message["from"] != "user":
            raise ValueError("%s row %d is not a user prompt" % (benchmark, index))
        records.append(
            EvaluationRecord(
                example_id="%s-test-%04d" % (benchmark, index),
                benchmark=benchmark,
                question=_require_string(message, "value", index),
                gold_answer=str(answer).strip(),
            )
        )
    if any(not record.gold_answer for record in records):
        raise ValueError("%s contains an empty final answer" % benchmark)
    return records


def evaluation_record_to_verl_row(record: EvaluationRecord, index: int) -> Dict[str, Any]:
    return {
        "data_source": record.benchmark,
        "prompt": [
            {"role": "user", "content": render_student_user_content(record.question)}
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": record.gold_answer},
        "extra_info": {
            "index": index,
            "split": "test",
            "example_id": record.example_id,
        },
    }


def load_pinned_math_train(cache_dir: Optional[Path] = None) -> List[Mapping[str, Any]]:
    """Network-facing loader; imports Hugging Face Datasets lazily."""

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("loading pinned datasets requires the datasets package") from error
    dataset = load_dataset(
        MATH_DATASET_ID,
        MATH_DATASET_CONFIG,
        revision=MATH_DATASET_REVISION,
        split="train",
        cache_dir=None if cache_dir is None else str(cache_dir),
    )
    return list(dataset)


def load_pinned_math500(cache_dir: Optional[Path] = None) -> List[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("loading pinned datasets requires the datasets package") from error
    dataset = load_dataset(
        MATH500_DATASET_ID,
        MATH500_DATASET_CONFIG,
        revision=MATH500_DATASET_REVISION,
        split="test",
        cache_dir=None if cache_dir is None else str(cache_dir),
    )
    return list(dataset)


def load_pinned_gsm8k_test(cache_dir: Optional[Path] = None) -> List[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("loading pinned datasets requires the datasets package") from error
    dataset = load_dataset(
        GSM8K_DATASET_ID,
        GSM8K_DATASET_CONFIG,
        revision=GSM8K_DATASET_REVISION,
        split="test",
        cache_dir=None if cache_dir is None else str(cache_dir),
    )
    return list(dataset)
