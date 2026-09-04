"""CPU-only contracts for the native-soft in-context-learning evaluation.

This module owns the immutable benchmark preparation, prompt conditions,
evaluation matrix, common-random-number seeds, and statistical estimands.  It
deliberately imports no Torch, Ray, SGLang, Transformers, or W&B code so the
contract can be validated before a GPU job is submitted.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .constants import (
    GSM8K_DATASET_CONFIG,
    GSM8K_DATASET_ID,
    GSM8K_DATASET_REVISION,
    GSM8K_TEST_SIZE,
    MATH500_DATASET_CONFIG,
    MATH500_DATASET_ID,
    MATH500_DATASET_REVISION,
    MATH500_TEST_SIZE,
    RELEASED_EVAL_FILE_SHA256,
    STUDENT_PROMPT_SUFFIX,
)
from .data import extract_last_boxed_answer, gold_cot_without_final_box, parse_gsm8k_gold
from .graders import normalize_released_math_answer
from .manifest import file_sha256


ICL_SCHEMA_VERSION = 1
ICL_PROTOCOL = "opd-softgrpo-native-soft-icl-v1"
ICL_MATERIALIZATION_PROTOCOL = "opd-softgrpo-native-soft-icl-data-v1"

STARTING_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
STARTING_MODEL_REVISION = "c46dac620b4e4f12c5662a2133376a2823458d0e"
SOFTGRPO_MODEL_ID = "zz1358m/SofT-GRPO-master"
SOFTGRPO_MODEL_REVISION = "f3a0db41614abdda549e24896f1a9c131e95823f"
SOFTGRPO_MODEL_SUBFOLDER = "saved_weight/Deepeeek-Qwen-1.5B+SofT-GRPO"
MODEL_SOURCES = {
    "starting": {
        "repo_id": STARTING_MODEL_ID,
        "revision": STARTING_MODEL_REVISION,
        "subfolder": None,
    },
    "softgrpo": {
        "repo_id": SOFTGRPO_MODEL_ID,
        "revision": SOFTGRPO_MODEL_REVISION,
        "subfolder": SOFTGRPO_MODEL_SUBFOLDER,
    },
}

AIME2024_DATASET_ID = "HuggingFaceH4/aime_2024"
AIME2024_DATASET_CONFIG = "default"
AIME2024_DATASET_REVISION = "e6cf0cd64082ada1c025717826bd40e155b1ec81"
AIME2024_SPLIT = "train"
AIME2024_SIZE = 30
AIME2024_JOIN_MIN_SIMILARITY = 0.60
AIME2024_RELEASED_CORRECTION = {
    "released_index": 22,
    "expected_raw_answer": "480",
    "corrected_answer": "197",
    "canonical_question_sha256": (
        "671d8ed807928f33cbe44e844c98989195b0b30fc3c34c00644e4586a09d4263"
    ),
    "provenance_url": "https://live.poshenloh.com/past-contests/aime/2024I/problem/8",
    "reason": "released answer duplicates row 23; official 2024 AIME I problem 8 answer is 197",
}

MODEL_LABELS = ("starting", "softgrpo")
# ``BENCHMARKS`` and ``PROMPT_CONDITIONS`` describe the immutable materialized
# data artifact.  The smaller study registries below deliberately leave the
# old donor/control material available for provenance-compatible reuse without
# scheduling those cells in this evaluation.
BENCHMARKS = ("math500", "gsm8k_test", "aime2024")
STUDY_BENCHMARKS = ("math500", "aime2024")
INFERENCE_MODES = ("native_soft", "hard_token")
SUPPORTED_CORE_CONDITIONS = (
    "no_demo",
    "sdft_matched",
    "sdft_shuffled",
    "sdpg_matched",
    "sdpg_shuffled",
)
CORE_CONDITIONS = (
    "no_demo",
    "sdft_matched",
    "sdpg_matched",
)
MECHANISM_CONDITIONS = (
    "sdft_answer_only",
    "sdft_rationale_only",
    "sdpg_answer_only",
    "sdpg_rationale_only",
)
PROMPT_CONDITIONS = SUPPORTED_CORE_CONDITIONS + MECHANISM_CONDITIONS

DATA_SELECTION_SEED = 42
EXPERIMENT_SEED = 11
COMMON_SAMPLE_SEEDS = tuple(range(EXPERIMENT_SEED, EXPERIMENT_SEED + 8))
PRODUCTION_SAMPLE_COUNT = 8
SMOKE_SAMPLE_COUNT = 2
SMOKE_EXAMPLE_COUNT = 16
GSM8K_EVALUATION_SIZE = 512
MECHANISM_EXAMPLE_COUNT = 128
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 11

EXPECTED_EXAMPLE_COUNTS = {
    "math500": MATH500_TEST_SIZE,
    "gsm8k_test": GSM8K_EVALUATION_SIZE,
    "aime2024": AIME2024_SIZE,
}
EXPECTED_MECHANISM_COUNTS = {
    benchmark: min(MECHANISM_EXAMPLE_COUNT, count)
    for benchmark, count in EXPECTED_EXAMPLE_COUNTS.items()
}
PINNED_ORDERED_EXAMPLE_IDS_SHA256 = {
    "math500": "0fddf4d44cde1156456d14fedb12ae331b3ee7372ea5f6e9599499564913079b",
    "gsm8k_test": "f1b76c0b3db3b0f447700983e78d2e6f847aec0eb3abffe04b93287fd79dd3f6",
    "aime2024": "37e8b77e5ffb5ab05f0da3c8eb4af3e4ee7b5ad2064d4580c2bcca9169072a70",
}
PINNED_MECHANISM_IDS_SHA256 = {
    "math500": "1d87f94055d64c98c49e51d44115aed0e881f1e664df75e76f7ae041b5ce07c2",
    "gsm8k_test": "4477c1e85ceccb8d92c4a8b60e709d0a2daac76595c202c0e8aa1e8007a1cc81",
    "aime2024": "d4f014ed3832e57fa9f80d0a856cb55de3f7bdb624d45c4c0cd9707eb0ecb82d",
}
PINNED_DATA_FILE_SHA256 = {
    "examples.jsonl": "79c43b8c43d1eb954092e00e777e776ce4a5b76bd049ca4ea5ed7736bbef9ef4",
    "shuffled_pairs.json": "73158fca1bd569de04589a8a213f5108adbde388f7c7d0b5cb57b6210fd21bad",
    "mechanism_subset_ids.json": "58163576d553ed279d70f7460a51442ef03da5c0bd00bf66f3fc625e24168b40",
}

_DATA_FILENAMES = (
    "examples.jsonl",
    "shuffled_pairs.json",
    "mechanism_subset_ids.json",
)
_AIME_REQUIRED_FIELDS = frozenset({"id", "problem", "solution", "answer", "url", "year"})


def _require_text(value: Any, name: str, *, allow_surrounding_space: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty string" % name)
    if not allow_surrounding_space and value != value.strip():
        raise ValueError("%s must not have surrounding whitespace" % name)
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _stable_digest(namespace: str, *parts: Any) -> bytes:
    payload = [ICL_PROTOCOL, namespace]
    payload.extend(parts)
    return hashlib.sha256(_canonical_json(payload)).digest()


def _ordered_ids_sha256(namespace: str, values: Sequence[str]) -> str:
    digest = hashlib.sha256((ICL_PROTOCOL + "\0" + namespace + "\0").encode("utf-8"))
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def request_seed(benchmark: str, example_id: str, sample_index: int) -> int:
    """Derive a request seed shared across models and prompt conditions."""

    if benchmark not in BENCHMARKS:
        raise ValueError("unsupported benchmark: %r" % benchmark)
    _require_text(example_id, "example_id")
    if type(sample_index) is not int or not 0 <= sample_index < PRODUCTION_SAMPLE_COUNT:
        raise ValueError("sample_index must be in [0, 8)")
    value = int.from_bytes(
        _stable_digest("request-seed", EXPERIMENT_SEED, benchmark, example_id, sample_index)[:8],
        "big",
    )
    return value & ((1 << 63) - 1)


def model_source(model_label: str) -> Dict[str, Optional[str]]:
    """Return a defensive copy of the one pinned source for a model arm."""

    if model_label not in MODEL_SOURCES:
        raise ValueError("unsupported ICL model label: %r" % model_label)
    return dict(MODEL_SOURCES[model_label])


@dataclass(frozen=True)
class ICLEvaluationExample:
    """A privileged evaluation record; generation receives only its prompt view."""

    example_id: str
    benchmark: str
    source_index: int
    question: str
    gold_cot: str
    gold_answer: str
    subject: str = ""
    difficulty: str = ""
    schema_version: int = ICL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ICL_SCHEMA_VERSION:
            raise ValueError("unsupported ICL example schema")
        _require_text(self.example_id, "example_id")
        if self.benchmark not in BENCHMARKS:
            raise ValueError("unsupported benchmark: %r" % self.benchmark)
        if type(self.source_index) is not int or self.source_index < 0:
            raise ValueError("source_index must be a nonnegative integer")
        _require_text(self.question, "question", allow_surrounding_space=True)
        _require_text(self.gold_cot, "gold_cot", allow_surrounding_space=True)
        _require_text(self.gold_answer, "gold_answer")
        if not isinstance(self.subject, str) or not isinstance(self.difficulty, str):
            raise TypeError("subject and difficulty must be strings")
        if self.benchmark == "aime2024" and not re.fullmatch(r"\d{3}", self.gold_answer):
            raise ValueError("AIME prompt answers must be canonical three-digit strings")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ICLEvaluationExample":
        if not isinstance(value, Mapping):
            raise TypeError("ICL example must be a mapping")
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        if set(value) != expected:
            raise ValueError(
                "ICL example fields differ: missing=%s unknown=%s"
                % (sorted(expected - set(value)), sorted(set(value) - expected))
            )
        return cls(**dict(value))


@dataclass(frozen=True)
class ICLPrompt:
    """The exact gold-bearing or gold-free user payload sent to generation."""

    example_id: str
    benchmark: str
    condition: str
    user_content: str
    demonstration_example_id: Optional[str]

    def __post_init__(self) -> None:
        _require_text(self.example_id, "example_id")
        if self.benchmark not in BENCHMARKS:
            raise ValueError("unsupported benchmark")
        if self.condition not in PROMPT_CONDITIONS:
            raise ValueError("unsupported prompt condition")
        _require_text(self.user_content, "user_content", allow_surrounding_space=True)
        if self.condition == "no_demo":
            if self.demonstration_example_id is not None:
                raise ValueError("no-demo prompts cannot identify a demonstration")
        else:
            _require_text(self.demonstration_example_id, "demonstration_example_id")

    def generation_payload(self) -> Dict[str, Any]:
        """Return the only fields permitted to cross the generation boundary."""

        return {
            "example_id": self.example_id,
            "benchmark": self.benchmark,
            "condition": self.condition,
            "prompt": [{"role": "user", "content": self.user_content}],
        }


@dataclass(frozen=True)
class ICLMatrixCell:
    model_label: str
    inference_mode: str
    condition: str
    benchmark: str
    subset: str
    example_count: int
    sample_count: int

    def __post_init__(self) -> None:
        validate_matrix_cell(
            self.model_label,
            self.inference_mode,
            self.condition,
            self.benchmark,
        )
        if self.subset != "full":
            raise ValueError("registered conditions use the full evaluation subset")
        if type(self.example_count) is not int or self.example_count <= 0:
            raise ValueError("example_count must be positive")
        if type(self.sample_count) is not int or not 1 <= self.sample_count <= 8:
            raise ValueError("sample_count must be in [1, 8]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_matrix_cell(
    model_label: str,
    inference_mode: str,
    condition: str,
    benchmark: str,
) -> None:
    if model_label not in MODEL_LABELS:
        raise ValueError("unsupported ICL model label: %r" % model_label)
    if inference_mode not in INFERENCE_MODES:
        raise ValueError("unsupported ICL inference mode: %r" % inference_mode)
    if condition not in CORE_CONDITIONS:
        raise ValueError("unregistered ICL prompt condition: %r" % condition)
    if benchmark not in STUDY_BENCHMARKS:
        raise ValueError("unregistered ICL benchmark: %r" % benchmark)
    if model_label == "softgrpo" and inference_mode == "hard_token":
        raise ValueError("the post-trained checkpoint has no hard-token evaluation arm")


def build_icl_matrix(*, smoke: bool = False) -> Tuple[ICLMatrixCell, ...]:
    """Construct the complete allowed production or smoke matrix."""

    sample_count = SMOKE_SAMPLE_COUNT if smoke else PRODUCTION_SAMPLE_COUNT
    cells = []
    for model_label in MODEL_LABELS:
        for inference_mode in INFERENCE_MODES:
            if model_label == "softgrpo" and inference_mode == "hard_token":
                continue
            conditions = CORE_CONDITIONS
            for condition in conditions:
                for benchmark in STUDY_BENCHMARKS:
                    subset = "full"
                    expected = (
                        EXPECTED_EXAMPLE_COUNTS[benchmark]
                        if subset == "full"
                        else EXPECTED_MECHANISM_COUNTS[benchmark]
                    )
                    count = min(SMOKE_EXAMPLE_COUNT, expected) if smoke else expected
                    cells.append(
                        ICLMatrixCell(
                            model_label=model_label,
                            inference_mode=inference_mode,
                            condition=condition,
                            benchmark=benchmark,
                            subset=subset,
                            example_count=count,
                            sample_count=sample_count,
                        )
                    )
    return tuple(cells)


def _canonical_aime_answer(value: Any) -> str:
    text = str(value).strip().replace(",", "")
    if not re.fullmatch(r"\d{1,3}", text):
        raise ValueError("AIME answer must be an integer from 000 through 999")
    integer = int(text)
    if not 0 <= integer <= 999:
        raise ValueError("AIME answer is outside [000, 999]")
    return "%03d" % integer


def normalize_answer_key(value: str, benchmark: str) -> str:
    """Normalize answers for donor exclusion and diagnostic comparison."""

    text = _require_text(value, "answer")
    if benchmark == "aime2024":
        return str(int(_canonical_aime_answer(text)))
    return normalize_released_math_answer(text).casefold()


_ASY_BLOCK_RE = re.compile(
    r"(?:\[asy\].*?\[/asy\]|<asy>.*?</asy>)", re.IGNORECASE | re.DOTALL
)


def canonicalize_problem_for_join(value: str) -> str:
    """Normalize harmless Unicode, whitespace, and LaTeX presentation changes."""

    text = unicodedata.normalize("NFKC", _require_text(value, "problem", allow_surrounding_space=True))
    text = _ASY_BLOCK_RE.sub("", text)
    text = text.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\cdots", "\\dots")
    text = text.replace("×", "\\times").replace("−", "-")
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\\(?:,|!|;|:|quad|qquad)\s*", "", text)
    text = re.sub(r"(?:\$|\\\[|\\\]|\\\(|\\\))", "", text)
    return re.sub(r"\s+", "", text).casefold()


def _require_mapping(row: Any, index: int, source: str) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError("%s row %d is not a mapping" % (source, index))
    return row


def build_math500_icl_examples(
    rows: Sequence[Mapping[str, Any]], *, enforce_pinned_contract: bool = True
) -> Tuple[ICLEvaluationExample, ...]:
    if enforce_pinned_contract and len(rows) != MATH500_TEST_SIZE:
        raise ValueError("pinned MATH-500 must contain 500 rows")
    examples = []
    seen_ids = set()
    for index, raw in enumerate(rows):
        row = _require_mapping(raw, index, "MATH-500")
        required = {"problem", "solution", "answer", "subject", "level", "unique_id"}
        if not required.issubset(row):
            raise ValueError("MATH-500 row %d is missing required fields" % index)
        example_id = _require_text(str(row["unique_id"]), "unique_id")
        if example_id in seen_ids:
            raise ValueError("MATH-500 unique_id values are not unique")
        seen_ids.add(example_id)
        solution = _require_text(row["solution"], "solution", allow_surrounding_space=True)
        answer = _require_text(str(row["answer"]), "answer")
        boxed = extract_last_boxed_answer(solution)
        if boxed is None or normalize_answer_key(boxed, "math500") != normalize_answer_key(answer, "math500"):
            raise ValueError("MATH-500 solution does not resolve to its recorded answer")
        examples.append(
            ICLEvaluationExample(
                example_id=example_id,
                benchmark="math500",
                source_index=index,
                question=_require_text(row["problem"], "problem", allow_surrounding_space=True),
                gold_cot=gold_cot_without_final_box(solution),
                gold_answer=answer,
                subject=str(row["subject"]).strip(),
                difficulty=str(row["level"]).strip(),
            )
        )
    return tuple(examples)


def _stable_subset(
    examples: Sequence[ICLEvaluationExample],
    size: int,
    *,
    seed: int,
    namespace: str,
) -> Tuple[ICLEvaluationExample, ...]:
    if type(size) is not int or not 0 < size <= len(examples):
        raise ValueError("subset size must be in (0, len(examples)]")
    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError("subset source example IDs must be unique")
    ranked = sorted(
        examples,
        key=lambda example: (
            _stable_digest("subset", namespace, seed, example.example_id),
            example.example_id,
        ),
    )[:size]
    selected = {example.example_id for example in ranked}
    return tuple(example for example in examples if example.example_id in selected)


def build_gsm8k_icl_examples(
    rows: Sequence[Mapping[str, Any]], *, enforce_pinned_contract: bool = True
) -> Tuple[ICLEvaluationExample, ...]:
    if enforce_pinned_contract and len(rows) != GSM8K_TEST_SIZE:
        raise ValueError("pinned GSM8K test must contain 1319 rows")
    examples = []
    for index, raw in enumerate(rows):
        row = _require_mapping(raw, index, "GSM8K")
        if not {"question", "answer"}.issubset(row):
            raise ValueError("GSM8K row %d is missing required fields" % index)
        rationale, answer = parse_gsm8k_gold(row["answer"])
        examples.append(
            ICLEvaluationExample(
                example_id="gsm8k-test-%06d" % index,
                benchmark="gsm8k_test",
                source_index=index,
                question=_require_text(row["question"], "question", allow_surrounding_space=True),
                gold_cot=_require_text(rationale, "gold_cot", allow_surrounding_space=True),
                gold_answer=answer,
            )
        )
    target_size = GSM8K_EVALUATION_SIZE if enforce_pinned_contract else min(
        GSM8K_EVALUATION_SIZE, len(examples)
    )
    return _stable_subset(
        examples,
        target_size,
        seed=DATA_SELECTION_SEED,
        namespace="gsm8k-test-evaluation",
    )


def _question_sha256(question: str) -> str:
    return hashlib.sha256(canonicalize_problem_for_join(question).encode("utf-8")).hexdigest()


def _released_aime_rows(
    rows: Sequence[Mapping[str, Any]], *, enforce_pinned_contract: bool
) -> Tuple[Tuple[str, str, Optional[Dict[str, Any]]], ...]:
    result = []
    for index, raw in enumerate(rows):
        row = _require_mapping(raw, index, "released AIME 2024")
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or len(prompt) != 1 or not isinstance(prompt[0], Mapping):
            raise ValueError("released AIME row %d has an invalid prompt" % index)
        message = prompt[0]
        if message.get("from") != "user" or not isinstance(message.get("value"), str):
            raise ValueError("released AIME row %d must contain one user message" % index)
        question = message["value"]
        raw_answer = _canonical_aime_answer(row.get("final_answer"))
        correction = None
        correction_index = int(AIME2024_RELEASED_CORRECTION["released_index"])
        if index == correction_index:
            observed_digest = _question_sha256(question)
            expected_digest = AIME2024_RELEASED_CORRECTION["canonical_question_sha256"]
            expected_answer = _canonical_aime_answer(
                AIME2024_RELEASED_CORRECTION["expected_raw_answer"]
            )
            if observed_digest == expected_digest and raw_answer == expected_answer:
                corrected_answer = _canonical_aime_answer(
                    AIME2024_RELEASED_CORRECTION["corrected_answer"]
                )
                correction = {
                    **AIME2024_RELEASED_CORRECTION,
                    "expected_raw_answer": expected_answer,
                    "corrected_answer": corrected_answer,
                }
                raw_answer = corrected_answer
            elif enforce_pinned_contract:
                raise ValueError(
                    "pinned released AIME correction guard differs at row %d" % index
                )
        result.append((question, raw_answer, correction))
    return tuple(result)


def _aime_similarity(left: str, right: str) -> float:
    return float(
        SequenceMatcher(
            None,
            canonicalize_problem_for_join(left),
            canonicalize_problem_for_join(right),
            autojunk=False,
        ).ratio()
    )


_AIME_URL_RE = re.compile(
    r"/2024_AIME_(I|II)_Problems/Problem_([1-9]|1[0-5])/?$", re.IGNORECASE
)


def _aime_semantics_from_url(value: Any) -> Tuple[str, int]:
    url = _require_text(value, "AIME source URL")
    match = _AIME_URL_RE.search(url)
    if match is None:
        raise ValueError("AIME source URL does not identify a 2024 contest/problem")
    return match.group(1).upper(), int(match.group(2))


def _expected_released_aime_semantics(index: int) -> Tuple[str, int]:
    if not 0 <= index < AIME2024_SIZE:
        raise ValueError("released AIME index is outside the 30-problem contract")
    if index < 15:
        return "II", index + 1
    return "I", index - 14


def _aime_payload_answer(payload: str) -> str:
    text = payload.strip()
    text = re.sub(r"\\(?:textbf|mathbf|mathrm|text)\s*", "", text)
    text = text.replace("{", "").replace("}", "")
    text = text.strip().strip("$").strip()
    text = text.strip("().,;:").strip()
    if not re.fullmatch(r"\d{1,3}", text):
        raise ValueError("AIME terminal box payload is not a plain integer")
    return _canonical_aime_answer(text)


def _balanced_box_candidates(solution: str) -> Tuple[Dict[str, Any], ...]:
    candidates = []
    for match in re.finditer(r"\\(boxed|fbox|framebox)\s*\{", solution):
        opening = solution.find("{", match.start())
        depth = 1
        cursor = opening + 1
        while cursor < len(solution) and depth:
            if solution[cursor] == "{":
                depth += 1
            elif solution[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth:
            raise ValueError("AIME solution has an unbalanced box expression")
        candidates.append(
            {
                "command": match.group(1),
                "start": match.start(),
                "end": cursor,
                "payload": solution[opening + 1 : cursor - 1],
            }
        )
    return tuple(candidates)


def _remove_solution_span(solution: str, start: int, end: int) -> str:
    prefix = solution[:start]
    suffix = solution[end:]
    if prefix.endswith("$") and suffix.startswith("$"):
        prefix = prefix[:-1]
        suffix = suffix[1:]
    rationale = (prefix + suffix).strip()
    rationale = re.sub(r"[ \t]+([.,;:])", r"\1", rationale)
    if not rationale:
        raise ValueError("removing the AIME answer left an empty rationale")
    return rationale


def _aime_rationale_and_extraction(
    solution: str, canonical_answer: str
) -> Tuple[str, Dict[str, Any]]:
    """Resolve a verified terminal answer and strip its containing expression."""

    matching_boxes = []
    for candidate in _balanced_box_candidates(solution):
        try:
            candidate_answer = _aime_payload_answer(candidate["payload"])
        except ValueError:
            continue
        if candidate_answer == canonical_answer:
            matching_boxes.append(candidate)
    if matching_boxes:
        selected = matching_boxes[-1]
        rationale = _remove_solution_span(solution, selected["start"], selected["end"])
        return rationale, {
            "mode": selected["command"],
            "payload": selected["payload"],
            "payload_sha256": hashlib.sha256(
                selected["payload"].encode("utf-8")
            ).hexdigest(),
            "resolved_answer": canonical_answer,
        }

    # Fail-closed terminal fallback: only the final TeX math span may qualify,
    # and its last standalone integer must be the recorded answer.
    math_spans = list(re.finditer(r"\$([^$]+)\$", solution, re.DOTALL))
    if not math_spans:
        raise ValueError("AIME solution has no verified box or terminal math expression")
    selected_span = math_spans[-1]
    integers = re.findall(r"(?<![A-Za-z0-9])\d{1,4}(?![A-Za-z0-9])", selected_span.group(1))
    if not integers or _canonical_aime_answer(integers[-1]) != canonical_answer:
        raise ValueError("AIME terminal math expression does not resolve to its answer")
    rationale = _remove_solution_span(solution, selected_span.start(), selected_span.end())
    return rationale, {
        "mode": "terminal_math_expression",
        "payload": integers[-1],
        "payload_sha256": hashlib.sha256(
            selected_span.group(0).encode("utf-8")
        ).hexdigest(),
        "resolved_answer": canonical_answer,
    }


def _join_aime2024_with_report(
    source_rows: Sequence[Mapping[str, Any]],
    released_rows: Sequence[Mapping[str, Any]],
    *,
    enforce_pinned_contract: bool,
) -> Tuple[Tuple[ICLEvaluationExample, ...], Dict[str, Any]]:
    """Internal answer-constrained join returning its complete audit trail."""

    if enforce_pinned_contract and (
        len(source_rows) != AIME2024_SIZE or len(released_rows) != AIME2024_SIZE
    ):
        raise ValueError("both pinned AIME 2024 sources must contain 30 rows")
    parsed_source: Dict[Tuple[str, int], Tuple[int, Mapping[str, Any]]] = {}
    for index, raw in enumerate(source_rows):
        row = _require_mapping(raw, index, "HuggingFaceH4 AIME 2024")
        if enforce_pinned_contract and not _AIME_REQUIRED_FIELDS.issubset(row):
            raise ValueError("HuggingFaceH4 AIME row %d is missing pinned fields" % index)
        for field in ("problem", "solution", "answer"):
            if field not in row:
                raise ValueError("HuggingFaceH4 AIME row %d is missing %s" % (index, field))
        _require_text(row["problem"], "problem", allow_surrounding_space=True)
        if "url" not in row:
            raise ValueError("HuggingFaceH4 AIME row %d is missing url" % index)
        semantics = _aime_semantics_from_url(row["url"])
        if semantics in parsed_source:
            raise ValueError("HuggingFaceH4 AIME contest/problem semantics are not unique")
        parsed_source[semantics] = (index, row)
    parsed_released = _released_aime_rows(
        released_rows, enforce_pinned_contract=enforce_pinned_contract
    )

    if enforce_pinned_contract:
        expected_semantics = {
            _expected_released_aime_semantics(index) for index in range(AIME2024_SIZE)
        }
        if set(parsed_source) != expected_semantics:
            raise ValueError("HuggingFaceH4 AIME contest/problem inventory differs")

    examples = []
    join_rows = []
    corrections = []
    for released_index, (released_question, released_answer, correction) in enumerate(
        parsed_released
    ):
        expected_contest, expected_problem = _expected_released_aime_semantics(
            released_index
        )
        semantics = (expected_contest, expected_problem)
        if semantics not in parsed_source:
            raise ValueError(
                "AIME join lacks %s problem %d" % (expected_contest, expected_problem)
            )
        source_index, source = parsed_source[semantics]
        similarity = _aime_similarity(released_question, source["problem"])
        if similarity < AIME2024_JOIN_MIN_SIMILARITY:
            raise ValueError(
                "AIME text audit similarity %.6f is below %.2f at released row %d"
                % (similarity, AIME2024_JOIN_MIN_SIMILARITY, released_index)
            )
        source_answer = _canonical_aime_answer(source["answer"])
        if source_answer != released_answer:
            raise ValueError(
                "AIME answer differs at %s problem %d" % semantics
            )
        solution = _require_text(source["solution"], "solution", allow_surrounding_space=True)
        rationale, extraction = _aime_rationale_and_extraction(solution, source_answer)
        examples.append(
            ICLEvaluationExample(
                example_id="aime2024-test-%04d" % released_index,
                benchmark="aime2024",
                source_index=source_index,
                question=released_question,
                gold_cot=rationale,
                gold_answer=released_answer,
                subject="competition_math",
                difficulty="aime2024",
            )
        )
        exact_normalized_match = canonicalize_problem_for_join(
            released_question
        ) == canonicalize_problem_for_join(source["problem"])
        join_rows.append(
            {
                "released_index": released_index,
                "source_index": source_index,
                "source_id": source.get("id"),
                "expected_contest": expected_contest,
                "expected_problem": expected_problem,
                "answer": released_answer,
                "released_question_sha256": _question_sha256(released_question),
                "source_question_sha256": _question_sha256(source["problem"]),
                "exact_normalized_match": exact_normalized_match,
                "similarity": round(similarity, 12),
                "answer_extraction": extraction,
            }
        )
        if correction is not None:
            corrections.append(correction)
    if len(examples) != len(parsed_source) or len(examples) != len(released_rows):
        raise AssertionError("AIME join lost records")
    report = {
        "method": "contest-problem-index-with-text-audit-v1",
        "minimum_similarity": AIME2024_JOIN_MIN_SIMILARITY,
        "exact_normalized_match_count": sum(
            int(row["exact_normalized_match"]) for row in join_rows
        ),
        "corrections": corrections,
        "rows": join_rows,
    }
    return tuple(examples), report


def join_aime2024_icl_examples(
    source_rows: Sequence[Mapping[str, Any]],
    released_rows: Sequence[Mapping[str, Any]],
    *,
    enforce_pinned_contract: bool = True,
) -> Tuple[ICLEvaluationExample, ...]:
    """Bijectively attach worked H4 solutions to the released SofT-GRPO set."""

    examples, _ = _join_aime2024_with_report(
        source_rows,
        released_rows,
        enforce_pinned_contract=enforce_pinned_contract,
    )
    return examples


def join_aime2024_with_report(
    source_rows: Sequence[Mapping[str, Any]],
    released_rows: Sequence[Mapping[str, Any]],
    *,
    enforce_pinned_contract: bool = True,
) -> Tuple[Tuple[ICLEvaluationExample, ...], Dict[str, Any]]:
    return _join_aime2024_with_report(
        source_rows,
        released_rows,
        enforce_pinned_contract=enforce_pinned_contract,
    )


def build_icl_examples(
    math500_rows: Sequence[Mapping[str, Any]],
    gsm8k_rows: Sequence[Mapping[str, Any]],
    aime2024_rows: Sequence[Mapping[str, Any]],
    released_aime2024_rows: Sequence[Mapping[str, Any]],
    *,
    enforce_pinned_contract: bool = True,
) -> Tuple[ICLEvaluationExample, ...]:
    result, _ = build_icl_examples_with_report(
        math500_rows,
        gsm8k_rows,
        aime2024_rows,
        released_aime2024_rows,
        enforce_pinned_contract=enforce_pinned_contract,
    )
    return result


def build_icl_examples_with_report(
    math500_rows: Sequence[Mapping[str, Any]],
    gsm8k_rows: Sequence[Mapping[str, Any]],
    aime2024_rows: Sequence[Mapping[str, Any]],
    released_aime2024_rows: Sequence[Mapping[str, Any]],
    *,
    enforce_pinned_contract: bool = True,
) -> Tuple[Tuple[ICLEvaluationExample, ...], Dict[str, Any]]:
    aime, aime_report = join_aime2024_with_report(
        aime2024_rows,
        released_aime2024_rows,
        enforce_pinned_contract=enforce_pinned_contract,
    )
    result = (
        build_math500_icl_examples(
            math500_rows, enforce_pinned_contract=enforce_pinned_contract
        )
        + build_gsm8k_icl_examples(
            gsm8k_rows, enforce_pinned_contract=enforce_pinned_contract
        )
        + aime
    )
    keys = {(example.benchmark, example.example_id) for example in result}
    if len(keys) != len(result):
        raise ValueError("ICL benchmark/example IDs are not globally unique")
    return result, {"aime2024_join": aime_report}


def _similarity_key(target: ICLEvaluationExample, donor: ICLEvaluationExample) -> Tuple[int, ...]:
    return (
        int(bool(target.subject and donor.subject and target.subject != donor.subject)),
        int(
            bool(
                target.difficulty
                and donor.difficulty
                and target.difficulty != donor.difficulty
            )
        ),
        abs(len(target.question) - len(donor.question)),
        abs(len(target.gold_cot) - len(donor.gold_cot)),
    )


def build_shuffled_donor_map(
    examples: Sequence[ICLEvaluationExample], *, seed: int = DATA_SELECTION_SEED
) -> Dict[str, str]:
    """Choose a deterministic one-to-one, answer-mismatched cyclic derangement.

    Examples are ordered by metadata and length.  Among every valid cyclic
    rotation, the lexicographically minimum aggregate mismatch/length cost is
    selected, with a SHA256 tie breaker.  This is deterministic across Python
    and NumPy versions and keeps demonstrations as similar as the constrained
    rotation family permits.
    """

    if len(examples) < 2:
        raise ValueError("a shuffled control needs at least two examples")
    benchmarks = {example.benchmark for example in examples}
    if len(benchmarks) != 1:
        raise ValueError("shuffled donors must stay within one benchmark")
    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError("shuffled example IDs must be unique")
    ordered = sorted(
        examples,
        key=lambda example: (
            example.subject,
            example.difficulty,
            len(example.question),
            len(example.gold_cot),
            _stable_digest("shuffle-order", seed, example.example_id),
        ),
    )
    n = len(ordered)
    candidates = []
    for offset in range(1, n):
        pairs = [(ordered[index], ordered[(index + offset) % n]) for index in range(n)]
        if any(
            target.example_id == donor.example_id
            or normalize_answer_key(target.gold_answer, target.benchmark)
            == normalize_answer_key(donor.gold_answer, donor.benchmark)
            for target, donor in pairs
        ):
            continue
        costs = [_similarity_key(target, donor) for target, donor in pairs]
        aggregate = tuple(sum(cost[index] for cost in costs) for index in range(4))
        candidates.append(
            (
                aggregate,
                _stable_digest("shuffle-offset", seed, offset),
                offset,
                pairs,
            )
        )
    if not candidates:
        raise ValueError("no answer-mismatched cyclic derangement exists")
    _, _, _, best_pairs = min(candidates, key=lambda value: value[:3])
    result = {target.example_id: donor.example_id for target, donor in best_pairs}
    if set(result) != {example.example_id for example in examples} or len(set(result.values())) != n:
        raise AssertionError("shuffled donor map is not a bijection")
    return result


def mechanism_subset_ids(
    examples: Sequence[ICLEvaluationExample], *, seed: int = DATA_SELECTION_SEED
) -> Tuple[str, ...]:
    benchmarks = {example.benchmark for example in examples}
    if len(benchmarks) != 1:
        raise ValueError("mechanism subset input must contain one benchmark")
    size = min(MECHANISM_EXAMPLE_COUNT, len(examples))
    selected = _stable_subset(
        examples,
        size,
        seed=seed,
        namespace="%s-mechanism" % next(iter(benchmarks)),
    )
    return tuple(example.example_id for example in selected)


def smoke_subset_ids(
    examples: Sequence[ICLEvaluationExample], *, seed: int = DATA_SELECTION_SEED
) -> Tuple[str, ...]:
    benchmarks = {example.benchmark for example in examples}
    if len(benchmarks) != 1:
        raise ValueError("smoke subset input must contain one benchmark")
    selected = _stable_subset(
        examples,
        min(SMOKE_EXAMPLE_COUNT, len(examples)),
        seed=seed,
        namespace="%s-smoke" % next(iter(benchmarks)),
    )
    return tuple(example.example_id for example in selected)


def _student_user_content(question: str) -> str:
    return _require_text(question, "question", allow_surrounding_space=True) + STUDENT_PROMPT_SUFFIX


def _answer_surface_forms(answer: str, benchmark: str) -> Tuple[str, ...]:
    raw = _require_text(answer, "answer")
    values = {raw, raw.strip("$"), normalize_answer_key(raw, benchmark)}
    if benchmark == "aime2024":
        values.add(_canonical_aime_answer(raw))
    fraction = re.fullmatch(r"\\(?:d?frac|tfrac)\{([^{}]+)\}\{([^{}]+)\}", raw)
    if fraction:
        values.add("%s/%s" % fraction.groups())
    return tuple(sorted((value for value in values if value), key=lambda item: (-len(item), item)))


def _surface_pattern(surface: str) -> re.Pattern[str]:
    pieces = re.split(r"\s+", surface.strip())
    escaped = r"\s*".join(re.escape(piece) for piece in pieces)
    left = r"(?<![A-Za-z0-9])" if surface[0].isalnum() else ""
    right = r"(?![A-Za-z0-9])" if surface[-1].isalnum() else ""
    return re.compile(left + escaped + right, re.IGNORECASE)


def mask_answer_occurrences(rationale: str, answer: str, benchmark: str) -> str:
    """Mask every occurrence in the study's declared answer-normalization family."""

    masked = _require_text(rationale, "gold_cot", allow_surrounding_space=True)
    patterns = [_surface_pattern(surface) for surface in _answer_surface_forms(answer, benchmark)]
    for pattern in patterns:
        masked = pattern.sub("[MASKED]", masked)
    if any(pattern.search(masked) for pattern in patterns):
        raise AssertionError("normalized answer remains after rationale masking")
    if not masked.strip():
        raise ValueError("answer masking left an empty rationale")
    return masked


def _render_sdft(original: str, rationale: str, answer: Optional[str]) -> str:
    response = (
        original
        + "\n\nThis is an example for a response to the question:\n"
        + rationale
    )
    if answer is not None:
        response += "\nThe final answer is: \\boxed{" + answer + "}"
    return response + "\n\nNow answer with a response of your own, including the thinking process."


def _render_sdpg(original: str, rationale: str, answer: Optional[str]) -> str:
    if answer is None:
        hint = "[Hint] A common way to solve this is:\n" + rationale
        instruction = (
            "[Instruction] If possible, derive the answer using an alternative, equally "
            "rigorous mathematical approach to the one provided above. Otherwise, improve "
            "the given reasoning by making it clearer, more complete, and logically sound. "
            "Do NOT state that you were given a reference."
        )
    else:
        hint = (
            "[Hint] The correct answer is "
            + answer
            + ". A common way to solve this is:\n"
            + rationale
        )
        instruction = (
            "[Instruction] If possible, derive the answer "
            + answer
            + " using an alternative, equally rigorous mathematical approach to the one "
            "provided above. Otherwise, improve the given reasoning by making it clearer, "
            "more complete, and logically sound. Do NOT state that you were given the answer "
            "or reference."
        )
    return original + "\n\n" + hint + "\n\n" + instruction


def render_icl_prompt(
    target: ICLEvaluationExample,
    condition: str,
    *,
    shuffled_donor: Optional[ICLEvaluationExample] = None,
) -> ICLPrompt:
    """Render one exact condition while enforcing donor and answer isolation."""

    if condition not in PROMPT_CONDITIONS:
        raise ValueError("unsupported prompt condition: %r" % condition)
    original = _student_user_content(target.question)
    if condition == "no_demo":
        return ICLPrompt(target.example_id, target.benchmark, condition, original, None)

    if condition.endswith("_shuffled"):
        if shuffled_donor is None:
            raise ValueError("shuffled prompt conditions require a donor")
        if shuffled_donor.benchmark != target.benchmark:
            raise ValueError("shuffled donor must come from the same benchmark")
        if shuffled_donor.example_id == target.example_id:
            raise ValueError("shuffled donor must be a different example")
        if normalize_answer_key(shuffled_donor.gold_answer, target.benchmark) == normalize_answer_key(
            target.gold_answer, target.benchmark
        ):
            raise ValueError("shuffled donor must have a different normalized answer")
        demonstration = shuffled_donor
    else:
        if shuffled_donor is not None:
            raise ValueError("only shuffled conditions accept an explicit donor")
        demonstration = target

    family = condition.split("_", 1)[0]
    if condition.endswith("_answer_only"):
        rationale = "[Reasoning omitted.]"
        answer: Optional[str] = demonstration.gold_answer
    elif condition.endswith("_rationale_only"):
        rationale = mask_answer_occurrences(
            demonstration.gold_cot,
            demonstration.gold_answer,
            demonstration.benchmark,
        )
        answer = None
    else:
        rationale = demonstration.gold_cot
        answer = demonstration.gold_answer
    rendered = (
        _render_sdft(original, rationale, answer)
        if family == "sdft"
        else _render_sdpg(original, rationale, answer)
    )
    return ICLPrompt(
        target.example_id,
        target.benchmark,
        condition,
        rendered,
        demonstration.example_id,
    )


def materialize_prompts(
    examples: Sequence[ICLEvaluationExample],
    shuffled_pairs: Mapping[str, str],
    condition: str,
    *,
    selected_ids: Optional[Iterable[str]] = None,
) -> Tuple[ICLPrompt, ...]:
    by_id = {example.example_id: example for example in examples}
    if len(by_id) != len(examples):
        raise ValueError("example IDs must be unique")
    selected = set(by_id) if selected_ids is None else set(selected_ids)
    if not selected.issubset(by_id):
        raise ValueError("selected_ids contains an unknown example")
    result = []
    for target in examples:
        if target.example_id not in selected:
            continue
        donor = None
        if condition.endswith("_shuffled"):
            donor_id = shuffled_pairs.get(target.example_id)
            if donor_id not in by_id:
                raise ValueError("shuffled pair map is incomplete")
            donor = by_id[donor_id]
        result.append(render_icl_prompt(target, condition, shuffled_donor=donor))
    return tuple(result)


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    temporary = path.with_name(".%s.%s.tmp" % (path.name, os.getpid()))
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _jsonl_bytes(examples: Sequence[ICLEvaluationExample]) -> bytes:
    return b"".join(_canonical_json(example.to_dict()) + b"\n" for example in examples)


def _artifact_record(path: Path) -> Dict[str, Any]:
    return {"size": path.stat().st_size, "sha256": file_sha256(path)}


def _group_examples(
    examples: Sequence[ICLEvaluationExample],
) -> Dict[str, Tuple[ICLEvaluationExample, ...]]:
    return {
        benchmark: tuple(example for example in examples if example.benchmark == benchmark)
        for benchmark in BENCHMARKS
    }


def _materialized_prompt_contract(
    grouped: Mapping[str, Sequence[ICLEvaluationExample]],
    shuffled: Mapping[str, Mapping[str, str]],
    mechanism: Mapping[str, Sequence[str]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Render every registered prompt before publishing the data artifact.

    Besides pinning the prompt payload hashes, this makes rationale-only
    answer masking fail during CPU preparation rather than after a GPU has
    already been allocated.
    """

    result: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for benchmark in BENCHMARKS:
        benchmark_rows: Dict[str, Dict[str, Any]] = {}
        for condition in PROMPT_CONDITIONS:
            selected_ids = (
                None
                if condition in SUPPORTED_CORE_CONDITIONS
                else mechanism[benchmark]
            )
            prompts = materialize_prompts(
                grouped[benchmark],
                shuffled[benchmark],
                condition,
                selected_ids=selected_ids,
            )
            expected = (
                len(grouped[benchmark])
                if condition in SUPPORTED_CORE_CONDITIONS
                else len(mechanism[benchmark])
            )
            if len(prompts) != expected:
                raise RuntimeError("materialized prompt inventory is incomplete")
            payload = [prompt.generation_payload() for prompt in prompts]
            benchmark_rows[condition] = {
                "count": len(payload),
                "sha256": hashlib.sha256(_canonical_json(payload)).hexdigest(),
            }
        result[benchmark] = benchmark_rows
    return result


def materialize_icl_dataset_from_rows(
    output_dir: Path,
    math500_rows: Sequence[Mapping[str, Any]],
    gsm8k_rows: Sequence[Mapping[str, Any]],
    aime2024_rows: Sequence[Mapping[str, Any]],
    released_aime2024_rows: Sequence[Mapping[str, Any]],
    *,
    enforce_pinned_contract: bool = True,
) -> Dict[str, Any]:
    """Build and atomically publish the complete CPU-side ICL data artifact."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        return verify_icl_dataset(destination, enforce_pinned_contract=enforce_pinned_contract)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".%s." % destination.name, dir=destination.parent))
    try:
        examples, preparation_report = build_icl_examples_with_report(
            math500_rows,
            gsm8k_rows,
            aime2024_rows,
            released_aime2024_rows,
            enforce_pinned_contract=enforce_pinned_contract,
        )
        grouped = _group_examples(examples)
        shuffled = {
            benchmark: build_shuffled_donor_map(grouped[benchmark])
            for benchmark in BENCHMARKS
        }
        mechanism = {
            benchmark: list(mechanism_subset_ids(grouped[benchmark]))
            for benchmark in BENCHMARKS
        }
        prompt_contract = _materialized_prompt_contract(
            grouped, shuffled, mechanism
        )
        examples_path = temporary / "examples.jsonl"
        pairs_path = temporary / "shuffled_pairs.json"
        subset_path = temporary / "mechanism_subset_ids.json"
        _write_bytes_atomic(examples_path, _jsonl_bytes(examples))
        _write_bytes_atomic(pairs_path, _canonical_json(shuffled) + b"\n")
        _write_bytes_atomic(subset_path, _canonical_json(mechanism) + b"\n")
        manifest = {
            "schema_version": ICL_SCHEMA_VERSION,
            "protocol": ICL_MATERIALIZATION_PROTOCOL,
            "sources": {
                "math500": {
                    "id": MATH500_DATASET_ID,
                    "config": MATH500_DATASET_CONFIG,
                    "revision": MATH500_DATASET_REVISION,
                },
                "gsm8k": {
                    "id": GSM8K_DATASET_ID,
                    "config": GSM8K_DATASET_CONFIG,
                    "revision": GSM8K_DATASET_REVISION,
                },
                "aime2024_solutions": {
                    "id": AIME2024_DATASET_ID,
                    "config": AIME2024_DATASET_CONFIG,
                    "revision": AIME2024_DATASET_REVISION,
                },
                "aime2024_released": {
                    "sha256": RELEASED_EVAL_FILE_SHA256["aime2024"],
                },
            },
            "selection": {
                "seed": DATA_SELECTION_SEED,
                "gsm8k_method": "sha256-rank-512-v1",
                "mechanism_method": "sha256-rank-min-128-v1",
                "shuffle_method": "metadata-length-cyclic-derangement-v1",
            },
            "counts": {benchmark: len(grouped[benchmark]) for benchmark in BENCHMARKS},
            "ordered_example_ids_sha256": {
                benchmark: _ordered_ids_sha256(
                    benchmark, [example.example_id for example in grouped[benchmark]]
                )
                for benchmark in BENCHMARKS
            },
            "mechanism_ids_sha256": {
                benchmark: _ordered_ids_sha256(benchmark + "-mechanism", mechanism[benchmark])
                for benchmark in BENCHMARKS
            },
            "materialized_prompts": prompt_contract,
            "preparation": preparation_report,
            "files": {
                name: _artifact_record(temporary / name) for name in _DATA_FILENAMES
            },
            "pinned_contract_enforced": bool(enforce_pinned_contract),
        }
        if enforce_pinned_contract and (
            manifest["ordered_example_ids_sha256"]
            != PINNED_ORDERED_EXAMPLE_IDS_SHA256
            or manifest["mechanism_ids_sha256"] != PINNED_MECHANISM_IDS_SHA256
            or {
                name: record["sha256"] for name, record in manifest["files"].items()
            }
            != PINNED_DATA_FILE_SHA256
        ):
            raise ValueError("pinned ICL materialization hashes changed")
        manifest["content_sha256"] = hashlib.sha256(_canonical_json(manifest)).hexdigest()
        _write_bytes_atomic(temporary / "manifest.json", _canonical_json(manifest) + b"\n")
        os.replace(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _released_aime_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "Soft-Thinking+noise+loss-main"
        / "datasets"
        / "aime2024.json"
    )


def prepare_icl_dataset(output_dir: Path, cache_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load only pinned sources, then materialize an authenticated artifact."""

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("ICL data preparation requires the datasets package") from error
    cache = None if cache_dir is None else str(Path(cache_dir).expanduser().resolve())
    math500 = list(
        load_dataset(
            MATH500_DATASET_ID,
            MATH500_DATASET_CONFIG,
            revision=MATH500_DATASET_REVISION,
            split="test",
            cache_dir=cache,
        )
    )
    gsm8k = list(
        load_dataset(
            GSM8K_DATASET_ID,
            GSM8K_DATASET_CONFIG,
            revision=GSM8K_DATASET_REVISION,
            split="test",
            cache_dir=cache,
        )
    )
    aime = list(
        load_dataset(
            AIME2024_DATASET_ID,
            AIME2024_DATASET_CONFIG,
            revision=AIME2024_DATASET_REVISION,
            split=AIME2024_SPLIT,
            cache_dir=cache,
        )
    )
    released_path = _released_aime_path()
    if file_sha256(released_path) != RELEASED_EVAL_FILE_SHA256["aime2024"]:
        raise ValueError("released AIME 2024 asset hash changed")
    released = json.loads(released_path.read_text(encoding="utf-8"))
    return materialize_icl_dataset_from_rows(output_dir, math500, gsm8k, aime, released)


def _read_examples(path: Path) -> Tuple[ICLEvaluationExample, ...]:
    examples = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("invalid examples JSONL at line %d" % line_number) from error
            examples.append(ICLEvaluationExample.from_mapping(value))
    return tuple(examples)


def verify_icl_dataset(
    output_dir: Path, *, enforce_pinned_contract: bool = True
) -> Dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise ValueError("ICL materialization is incomplete")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("ICL manifest is unreadable") from error
    claimed = manifest.get("content_sha256")
    unsigned = dict(manifest)
    unsigned.pop("content_sha256", None)
    if claimed != hashlib.sha256(_canonical_json(unsigned)).hexdigest():
        raise ValueError("ICL manifest content hash differs")
    if manifest.get("schema_version") != ICL_SCHEMA_VERSION or manifest.get(
        "protocol"
    ) != ICL_MATERIALIZATION_PROTOCOL:
        raise ValueError("ICL materialization protocol differs")
    expected_files = set(_DATA_FILENAMES) | {"manifest.json"}
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ValueError("ICL materialization file inventory differs")
    file_records = manifest.get("files")
    if not isinstance(file_records, Mapping) or set(file_records) != set(_DATA_FILENAMES):
        raise ValueError("ICL file manifest differs")
    for name in _DATA_FILENAMES:
        record = file_records[name]
        path = root / name
        if (
            not isinstance(record, Mapping)
            or path.stat().st_size != record.get("size")
            or file_sha256(path) != record.get("sha256")
        ):
            raise ValueError("ICL data file failed authentication: %s" % name)
    if enforce_pinned_contract:
        expected_sources = {
            "math500": {
                "id": MATH500_DATASET_ID,
                "config": MATH500_DATASET_CONFIG,
                "revision": MATH500_DATASET_REVISION,
            },
            "gsm8k": {
                "id": GSM8K_DATASET_ID,
                "config": GSM8K_DATASET_CONFIG,
                "revision": GSM8K_DATASET_REVISION,
            },
            "aime2024_solutions": {
                "id": AIME2024_DATASET_ID,
                "config": AIME2024_DATASET_CONFIG,
                "revision": AIME2024_DATASET_REVISION,
            },
            "aime2024_released": {"sha256": RELEASED_EVAL_FILE_SHA256["aime2024"]},
        }
        if (
            manifest.get("pinned_contract_enforced") is not True
            or manifest.get("sources") != expected_sources
            or manifest.get("counts") != EXPECTED_EXAMPLE_COUNTS
            or manifest.get("ordered_example_ids_sha256")
            != PINNED_ORDERED_EXAMPLE_IDS_SHA256
            or manifest.get("mechanism_ids_sha256") != PINNED_MECHANISM_IDS_SHA256
            or {
                name: record.get("sha256")
                for name, record in manifest.get("files", {}).items()
                if isinstance(record, Mapping)
            }
            != PINNED_DATA_FILE_SHA256
        ):
            raise ValueError("ICL data differs from the pinned contract")

    examples = _read_examples(root / "examples.jsonl")
    grouped = _group_examples(examples)
    observed_counts = {benchmark: len(grouped[benchmark]) for benchmark in BENCHMARKS}
    if observed_counts != manifest.get("counts"):
        raise ValueError("ICL example counts differ from the manifest")
    preparation = manifest.get("preparation")
    join_report = preparation.get("aime2024_join") if isinstance(preparation, Mapping) else None
    if not isinstance(join_report, Mapping):
        raise ValueError("ICL manifest lacks the AIME join audit")
    join_rows = join_report.get("rows")
    corrections = join_report.get("corrections")
    if (
        join_report.get("method") != "contest-problem-index-with-text-audit-v1"
        or join_report.get("minimum_similarity") != AIME2024_JOIN_MIN_SIMILARITY
        or not isinstance(join_rows, list)
        or not isinstance(corrections, list)
    ):
        raise ValueError("ICL AIME join audit contract differs")
    aime_examples = grouped["aime2024"]
    if len(join_rows) != len(aime_examples):
        raise ValueError("ICL AIME join audit row count differs")
    if {row.get("released_index") for row in join_rows if isinstance(row, Mapping)} != set(
        range(len(aime_examples))
    ):
        raise ValueError("ICL AIME join audit indices differ")
    if len({row.get("source_index") for row in join_rows}) != len(join_rows):
        raise ValueError("ICL AIME join audit reused a source row")
    for released_index, (row, example) in enumerate(zip(join_rows, aime_examples)):
        extraction = row.get("answer_extraction") if isinstance(row, Mapping) else None
        expected_contest, expected_problem = _expected_released_aime_semantics(
            released_index
        )
        if (
            not isinstance(row, Mapping)
            or row.get("released_question_sha256") != _question_sha256(example.question)
            or row.get("answer") != example.gold_answer
            or row.get("expected_contest") != expected_contest
            or row.get("expected_problem") != expected_problem
            or not isinstance(row.get("similarity"), (int, float))
            or float(row["similarity"]) < AIME2024_JOIN_MIN_SIMILARITY
            or not isinstance(extraction, Mapping)
            or extraction.get("mode")
            not in {"boxed", "fbox", "framebox", "terminal_math_expression"}
            or extraction.get("resolved_answer") != example.gold_answer
            or not isinstance(extraction.get("payload"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(extraction.get("payload_sha256")))
        ):
            raise ValueError("ICL AIME join audit row differs from examples")
    if enforce_pinned_contract:
        expected_correction = {
            **AIME2024_RELEASED_CORRECTION,
            "expected_raw_answer": _canonical_aime_answer(
                AIME2024_RELEASED_CORRECTION["expected_raw_answer"]
            ),
            "corrected_answer": _canonical_aime_answer(
                AIME2024_RELEASED_CORRECTION["corrected_answer"]
            ),
        }
        if corrections != [expected_correction]:
            raise ValueError("ICL pinned AIME correction audit differs")
    shuffled = json.loads((root / "shuffled_pairs.json").read_text(encoding="utf-8"))
    mechanism = json.loads((root / "mechanism_subset_ids.json").read_text(encoding="utf-8"))
    if set(shuffled) != set(BENCHMARKS) or set(mechanism) != set(BENCHMARKS):
        raise ValueError("ICL benchmark maps are incomplete")
    for benchmark in BENCHMARKS:
        ids = [example.example_id for example in grouped[benchmark]]
        if manifest["ordered_example_ids_sha256"].get(benchmark) != _ordered_ids_sha256(
            benchmark, ids
        ):
            raise ValueError("ICL ordered example IDs differ")
        if shuffled[benchmark] != build_shuffled_donor_map(grouped[benchmark]):
            raise ValueError("ICL shuffled donor map differs")
        if tuple(mechanism[benchmark]) != mechanism_subset_ids(grouped[benchmark]):
            raise ValueError("ICL mechanism subset differs")
        if manifest["mechanism_ids_sha256"].get(benchmark) != _ordered_ids_sha256(
            benchmark + "-mechanism", mechanism[benchmark]
        ):
            raise ValueError("ICL mechanism subset hash differs")
    if manifest.get("materialized_prompts") != _materialized_prompt_contract(
        grouped, shuffled, mechanism
    ):
        raise ValueError("ICL materialized prompt contract differs")
    return manifest


def load_icl_dataset(
    output_dir: Path,
) -> Tuple[
    Tuple[ICLEvaluationExample, ...],
    Dict[str, Dict[str, str]],
    Dict[str, Tuple[str, ...]],
    Dict[str, Any],
]:
    manifest = verify_icl_dataset(output_dir)
    root = Path(output_dir).expanduser().resolve()
    examples = _read_examples(root / "examples.jsonl")
    pairs = json.loads((root / "shuffled_pairs.json").read_text(encoding="utf-8"))
    raw_subsets = json.loads(
        (root / "mechanism_subset_ids.json").read_text(encoding="utf-8")
    )
    subsets = {benchmark: tuple(values) for benchmark, values in raw_subsets.items()}
    return examples, pairs, subsets, manifest


def pass_at_k(n: int, correct: int, k: int) -> float:
    """Canonical unbiased pass@k estimator."""

    if type(n) is not int or type(correct) is not int or type(k) is not int:
        raise TypeError("n, correct, and k must be integers")
    if n <= 0 or not 0 <= correct <= n or not 1 <= k <= n:
        raise ValueError("require n > 0, 0 <= correct <= n, and 1 <= k <= n")
    if n - correct < k:
        return 1.0
    return 1.0 - math.comb(n - correct, k) / math.comb(n, k)


def pass_metrics_by_example(
    outcomes: Mapping[str, Sequence[bool]], *, expected_samples: int = PRODUCTION_SAMPLE_COUNT
) -> Dict[str, Dict[str, float]]:
    """Return explicit pass@1/pass@8 estimands for each example."""

    if expected_samples != PRODUCTION_SAMPLE_COUNT:
        raise ValueError("production pass metrics require exactly eight samples")
    if not outcomes:
        raise ValueError("outcomes cannot be empty")
    result: Dict[str, Dict[str, float]] = {}
    for example_id, values in outcomes.items():
        vector = tuple(values)
        if len(vector) != expected_samples or any(type(value) is not bool for value in vector):
            raise ValueError("each example must have exactly eight boolean outcomes")
        correct = sum(vector)
        result[example_id] = {
            "pass_at_1": pass_at_k(expected_samples, correct, 1),
            "pass_at_8": pass_at_k(expected_samples, correct, 8),
        }
    return result


def summarize_pass_metrics(
    outcomes: Mapping[str, Sequence[bool]],
) -> Dict[str, float | int]:
    per_example = pass_metrics_by_example(outcomes)
    return {
        "pass_at_1": float(np.mean([value["pass_at_1"] for value in per_example.values()])),
        "pass_at_8": float(np.mean([value["pass_at_8"] for value in per_example.values()])),
        "example_count": len(per_example),
        "samples_per_example": PRODUCTION_SAMPLE_COUNT,
    }


def paired_bootstrap_difference(
    treatment: Mapping[str, float],
    control: Mapping[str, float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    chunk_size: int = 256,
) -> Dict[str, float | int]:
    if resamples != BOOTSTRAP_RESAMPLES:
        raise ValueError("the ICL study requires exactly 10,000 bootstrap resamples")
    if seed != BOOTSTRAP_SEED:
        raise ValueError("the ICL study requires bootstrap seed 11")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not treatment or set(treatment) != set(control):
        raise ValueError("paired bootstrap requires identical non-empty example IDs")
    ids = sorted(treatment)
    differences = np.asarray(
        [float(treatment[key]) - float(control[key]) for key in ids], dtype=np.float64
    )
    if not np.isfinite(differences).all():
        raise ValueError("paired values must be finite")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        indices = rng.integers(0, len(ids), size=(stop - start, len(ids)))
        means[start:stop] = differences[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "difference": float(differences.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": 0.95,
        "resamples": resamples,
        "bootstrap_seed": seed,
        "example_count": len(ids),
    }


def paired_bootstrap_difference_in_differences(
    post_treatment: Mapping[str, float],
    post_control: Mapping[str, float],
    start_treatment: Mapping[str, float],
    start_control: Mapping[str, float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, float | int]:
    ids = set(post_treatment)
    if not ids or any(
        set(values) != ids
        for values in (post_control, start_treatment, start_control)
    ):
        raise ValueError("difference-in-differences requires identical example IDs")
    contrast = {
        key: (float(post_treatment[key]) - float(post_control[key]))
        - (float(start_treatment[key]) - float(start_control[key]))
        for key in ids
    }
    zeros = {key: 0.0 for key in ids}
    result = paired_bootstrap_difference(
        contrast, zeros, resamples=resamples, seed=seed
    )
    result["estimand"] = "(post_treatment-post_control)-(start_treatment-start_control)"
    return result


def rescue_harm_rates(
    treatment: Mapping[str, Sequence[bool]],
    control: Mapping[str, Sequence[bool]],
) -> Dict[str, float | int | None]:
    if not treatment or set(treatment) != set(control):
        raise ValueError("rescue/harm requires identical non-empty example IDs")
    rescue = harm = control_wrong = control_right = pairs = 0
    for example_id in sorted(treatment):
        treated = tuple(treatment[example_id])
        baseline = tuple(control[example_id])
        if len(treated) != len(baseline) or not treated:
            raise ValueError("rescue/harm requires paired non-empty samples")
        if any(type(value) is not bool for value in treated + baseline):
            raise TypeError("rescue/harm outcomes must be bool")
        for treatment_value, control_value in zip(treated, baseline):
            pairs += 1
            if control_value:
                control_right += 1
                harm += int(not treatment_value)
            else:
                control_wrong += 1
                rescue += int(treatment_value)
    return {
        "rescue_rate": None if control_wrong == 0 else rescue / control_wrong,
        "harm_rate": None if control_right == 0 else harm / control_right,
        "rescued": rescue,
        "harmed": harm,
        "control_incorrect": control_wrong,
        "control_correct": control_right,
        "paired_samples": pairs,
    }


_OVERLAP_TOKEN_RE = re.compile(r"\\[A-Za-z]+|[A-Za-z]+|\d+(?:\.\d+)?|[^\sA-Za-z0-9]")


def rationale_token_overlap_f1(generated: str, demonstration: str) -> float:
    """Lower-cased LaTeX-aware token-multiset F1 for copy diagnostics."""

    left = Counter(_OVERLAP_TOKEN_RE.findall(generated.casefold()))
    right = Counter(_OVERLAP_TOKEN_RE.findall(demonstration.casefold()))
    if not left or not right:
        return 0.0
    overlap = sum((left & right).values())
    precision = overlap / sum(left.values())
    recall = overlap / sum(right.values())
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def normalized_answer_copy(response: str, demonstrated_answer: str, benchmark: str) -> bool:
    """Whether the response's final boxed value equals the demonstrated answer."""

    boxed = extract_last_boxed_answer(response)
    if boxed is None:
        return False
    try:
        generated = normalize_answer_key(boxed, benchmark)
    except ValueError:
        # This is a copy diagnostic, not a grading precondition. A malformed
        # generated AIME box is simply not a copy of the valid demonstration.
        return False
    return generated == normalize_answer_key(demonstrated_answer, benchmark)


__all__ = [
    "AIME2024_DATASET_CONFIG",
    "AIME2024_DATASET_ID",
    "AIME2024_DATASET_REVISION",
    "AIME2024_RELEASED_CORRECTION",
    "BENCHMARKS",
    "BOOTSTRAP_RESAMPLES",
    "COMMON_SAMPLE_SEEDS",
    "CORE_CONDITIONS",
    "EXPECTED_EXAMPLE_COUNTS",
    "EXPECTED_MECHANISM_COUNTS",
    "ICLEvaluationExample",
    "ICLMatrixCell",
    "ICLPrompt",
    "MECHANISM_CONDITIONS",
    "MODEL_SOURCES",
    "MODEL_LABELS",
    "PROMPT_CONDITIONS",
    "PINNED_DATA_FILE_SHA256",
    "PINNED_MECHANISM_IDS_SHA256",
    "PINNED_ORDERED_EXAMPLE_IDS_SHA256",
    "SOFTGRPO_MODEL_ID",
    "SOFTGRPO_MODEL_REVISION",
    "SOFTGRPO_MODEL_SUBFOLDER",
    "STARTING_MODEL_ID",
    "STARTING_MODEL_REVISION",
    "STUDY_BENCHMARKS",
    "SUPPORTED_CORE_CONDITIONS",
    "build_gsm8k_icl_examples",
    "build_icl_examples",
    "build_icl_examples_with_report",
    "build_icl_matrix",
    "build_math500_icl_examples",
    "build_shuffled_donor_map",
    "canonicalize_problem_for_join",
    "join_aime2024_icl_examples",
    "join_aime2024_with_report",
    "load_icl_dataset",
    "mask_answer_occurrences",
    "materialize_icl_dataset_from_rows",
    "materialize_prompts",
    "mechanism_subset_ids",
    "model_source",
    "normalize_answer_key",
    "normalized_answer_copy",
    "paired_bootstrap_difference",
    "paired_bootstrap_difference_in_differences",
    "pass_at_k",
    "pass_metrics_by_example",
    "prepare_icl_dataset",
    "rationale_token_overlap_f1",
    "render_icl_prompt",
    "request_seed",
    "rescue_harm_rates",
    "summarize_pass_metrics",
    "smoke_subset_ids",
    "validate_matrix_cell",
    "verify_icl_dataset",
]
