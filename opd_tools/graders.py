"""Independent GSM8K grading interfaces used in the evaluation report."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple

from .constants import (
    GSM8K_GRADER_PROTOCOL,
    LM_EVAL_HARNESS_COMMIT,
    MATH_VERIFY_VERSION,
    SOFTGRPO_UPSTREAM_COMMIT,
)


@dataclass(frozen=True)
class Grade:
    correct: bool
    extracted_answer: Optional[str]
    normalized_prediction: Optional[str]
    normalized_gold: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _last_boxed_only_string(value: str) -> Optional[str]:
    index = value.rfind("\\boxed")
    if "\\boxed " in value:
        return "\\boxed " + value.split("\\boxed ")[-1].split("$")[0]
    if index < 0:
        index = value.rfind("\\fbox")
        if index < 0:
            return None
    cursor = index
    depth = 0
    while cursor < len(value):
        if value[cursor] == "{":
            depth += 1
        elif value[cursor] == "}":
            depth -= 1
            if depth == 0:
                return value[index : cursor + 1]
        cursor += 1
    return None


def _remove_boxed(value: str) -> str:
    if "\\boxed " in value:
        if not value.startswith("\\boxed "):
            raise ValueError("invalid boxed-space expression")
        return value[len("\\boxed ") :]
    if not value.startswith("\\boxed{") or not value.endswith("}"):
        raise ValueError("released scorer accepts only boxed expressions")
    return value[len("\\boxed{") : -1]


def _fix_fracs(value: str) -> str:
    pieces = value.split("\\frac")
    result = pieces[0]
    for piece in pieces[1:]:
        result += "\\frac"
        if not piece:
            return value
        if piece[0] == "{":
            result += piece
        elif len(piece) < 2:
            return value
        elif piece[1] != "{":
            result += "{%s}{%s}%s" % (piece[0], piece[1], piece[2:])
        else:
            result += "{%s}%s" % (piece[0], piece[1:])
    return result


def _fix_sqrt(value: str) -> str:
    pieces = value.split("\\sqrt")
    result = pieces[0]
    for piece in pieces[1:]:
        if not piece:
            return value
        result += "\\sqrt" + (piece if piece[0] == "{" else "{%s}%s" % (piece[0], piece[1:]))
    return result


def _fix_slash_fraction(value: str) -> str:
    pieces = value.split("/")
    if len(pieces) != 2:
        return value
    try:
        numerator, denominator = int(pieces[0]), int(pieces[1])
    except ValueError:
        return value
    if value != "%d/%d" % (numerator, denominator):
        return value
    return "\\frac{%d}{%d}" % (numerator, denominator)


def normalize_released_math_answer(value: str) -> str:
    """Mirror ``verl.utils.reward_score.math.strip_string``."""

    result = value.replace("\n", "").replace("\\!", "")
    result = result.replace("\\\\", "\\")
    result = result.replace("tfrac", "frac").replace("dfrac", "frac")
    result = result.replace("\\left", "").replace("\\right", "")
    result = result.replace("^{\\circ}", "").replace("^\\circ", "")
    result = result.replace("\\$", "")
    if "\\text{ " in result:
        pieces = result.split("\\text{ ")
        if len(pieces) != 2:
            raise ValueError("released unit normalization cannot handle multiple suffixes")
        result = pieces[0]
    result = result.replace("\\%", "")
    result = result.replace(" .", " 0.").replace("{.", "{0.")
    if not result:
        return result
    if result[0] == ".":
        result = "0" + result
    pieces = result.split("=")
    if len(pieces) == 2 and len(pieces[0]) <= 2:
        result = pieces[1]
    result = _fix_sqrt(result).replace(" ", "")
    result = _fix_fracs(result)
    if result == "0.5":
        result = "\\frac{1}{2}"
    return _fix_slash_fraction(result)


def released_last_boxed_grade(prediction: str, gold: str) -> Grade:
    """Released SofT-GRPO/VERL last-boxed string-equivalence reward."""

    boxed = _last_boxed_only_string(prediction)
    if boxed is None:
        # The released scorer returns zero before attempting to normalize gold.
        return Grade(False, None, None, gold)
    try:
        extracted = _remove_boxed(boxed)
    except Exception:
        return Grade(False, None, None, gold)
    try:
        normalized_prediction = normalize_released_math_answer(extracted)
        normalized_gold = normalize_released_math_answer(gold)
    except Exception:
        # Released ``is_equiv`` falls back to raw equality if normalization
        # fails; preserve that unusual behavior exactly.
        return Grade(extracted == gold, extracted, None, gold)
    return Grade(
        normalized_prediction == normalized_gold,
        extracted,
        normalized_prediction,
        normalized_gold,
    )


_LM_EVAL_FLEXIBLE_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def extract_lm_eval_flexible_number(prediction: str) -> Optional[str]:
    """Mirror lm-eval's regex filter with ``group_select=-1``."""

    matches = _LM_EVAL_FLEXIBLE_RE.findall(prediction)
    if not matches:
        return None
    groups = [group for group in matches[-1] if group]
    return groups[0].strip() if groups else None


def normalize_lm_eval_gsm8k(value: str) -> str:
    """Apply the exact-match ignores from lm-eval's ``gsm8k-cot.yaml``."""

    result = value
    for pattern in (r",", r"\$", r"(?s).*#### ", r"\.$"):
        result = re.sub(pattern, "", result)
    return result.lower()


def lm_eval_flexible_last_number_grade(prediction: str, gold: str) -> Grade:
    extracted = extract_lm_eval_flexible_number(prediction)
    normalized_gold = normalize_lm_eval_gsm8k(gold)
    normalized_prediction = (
        None if extracted is None else normalize_lm_eval_gsm8k(extracted)
    )
    return Grade(
        normalized_prediction is not None and normalized_prediction == normalized_gold,
        extracted,
        normalized_prediction,
        normalized_gold,
    )


MathVerifyScorer = Callable[[str, str], bool]


def gsm8k_grader_manifest() -> Dict[str, Any]:
    """Describe the three independent grading implementations and sources."""

    return {
        "protocol": GSM8K_GRADER_PROTOCOL,
        "graders": {
            "released_last_boxed": {
                "source": "verl/utils/reward_score/math.py",
                "softgrpo_commit": SOFTGRPO_UPSTREAM_COMMIT,
            },
            "lm_eval_flexible_last_number": {
                "source": "lm_eval/tasks/gsm8k/gsm8k-cot.yaml",
                "lm_eval_harness_commit": LM_EVAL_HARNESS_COMMIT,
                "regex": _LM_EVAL_FLEXIBLE_RE.pattern,
                "group_select": -1,
            },
            "math_verify": {
                "source": "math-verify",
                "version": MATH_VERIFY_VERSION,
                "input": "full_response",
            },
        },
    }


@lru_cache(maxsize=1)
def _math_verify_metric() -> Callable[..., Tuple[Any, Any]]:
    try:
        observed_version = version("math-verify")
    except PackageNotFoundError as error:
        raise RuntimeError("full-response grading requires math-verify==0.8.0") from error
    if observed_version != MATH_VERIFY_VERSION:
        raise RuntimeError(
            "Math-Verify version must be %s, found %s"
            % (MATH_VERIFY_VERSION, observed_version)
        )
    try:
        from math_verify.metric import math_metric
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    except ImportError as error:
        raise RuntimeError(
            "full-response grading requires math-verify==%s" % MATH_VERIFY_VERSION
        ) from error
    return math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )


def _default_math_verify_scorer(prediction: str, gold: str) -> bool:
    score, _ = _math_verify_metric()(["\\boxed{%s}" % gold], [prediction])
    return bool(score)


def math_verify_full_response_grade(
    prediction: str,
    gold: str,
    scorer: Optional[MathVerifyScorer] = None,
) -> Grade:
    selected = _default_math_verify_scorer if scorer is None else scorer
    result = selected(prediction, gold)
    if not isinstance(result, bool):
        raise TypeError("Math-Verify scorer must return bool")
    return Grade(result, None, None, gold)


def grade_gsm8k_interfaces(
    prediction: str,
    gold: str,
    math_verify_scorer: Optional[MathVerifyScorer] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return three independent results; never collapse them into one score."""

    return {
        "released_last_boxed": released_last_boxed_grade(prediction, gold).to_dict(),
        "lm_eval_flexible_last_number": lm_eval_flexible_last_number_grade(
            prediction, gold
        ).to_dict(),
        "math_verify": math_verify_full_response_grade(
            prediction, gold, math_verify_scorer
        ).to_dict(),
    }
