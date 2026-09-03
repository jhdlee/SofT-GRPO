"""Gold-isolated records used at the student, teacher, and reward boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from .constants import (
    MATH_DATASET_ID,
    SDFT_DEMONSTRATION_PREFIX,
    SDFT_INSTRUCTION,
    STUDENT_PROMPT_SUFFIX,
)

_FORBIDDEN_STUDENT_KEYS = frozenset(
    {
        "answer",
        "demonstration",
        "gold",
        "gold_answer",
        "gold_cot",
        "gold_solution",
        "reward_model",
        "solution",
        "teacher_prompt",
    }
)


def _require_nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty string" % name)
    return value


@dataclass(frozen=True)
class MathExample:
    """A cleaned privileged example; never pass this object to rollout."""

    example_id: str
    source_index: int
    question: str
    gold_solution: str
    gold_cot: str
    gold_answer: str
    subject: str
    level: str
    split: str = "unsplit"

    def __post_init__(self) -> None:
        _require_nonempty_text(self.example_id, "example_id")
        if not isinstance(self.source_index, int) or self.source_index < 0:
            raise ValueError("source_index must be a non-negative integer")
        for name in (
            "question",
            "gold_solution",
            "gold_cot",
            "gold_answer",
            "subject",
            "level",
            "split",
        ):
            _require_nonempty_text(getattr(self, name), name)


@dataclass(frozen=True)
class StudentRecord:
    """The complete payload permitted to enter student rollout or inference."""

    example_id: str
    split: str
    prompt: Tuple[Mapping[str, str], ...]

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "example_id": self.example_id,
            "split": self.split,
            "prompt": [dict(message) for message in self.prompt],
        }
        validate_student_record(payload)
        return payload


@dataclass(frozen=True)
class TeacherRecord:
    """Privileged teacher-only content, joined to a rollout by example ID."""

    example_id: str
    split: str
    user_content: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "example_id": self.example_id,
            "split": self.split,
            "user_content": self.user_content,
        }


@dataclass(frozen=True)
class RewardRecord:
    """Verifier-only ground truth, kept out of the student record."""

    example_id: str
    split: str
    ground_truth: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "example_id": self.example_id,
            "split": self.split,
            "ground_truth": self.ground_truth,
        }


@dataclass(frozen=True)
class RecordBundle:
    student: StudentRecord
    teacher: TeacherRecord
    reward: RewardRecord

    def __post_init__(self) -> None:
        ids = {self.student.example_id, self.teacher.example_id, self.reward.example_id}
        splits = {self.student.split, self.teacher.split, self.reward.split}
        if len(ids) != 1 or len(splits) != 1:
            raise ValueError("student, teacher, and reward records must have one identity")


@dataclass(frozen=True)
class EvaluationRecord:
    example_id: str
    benchmark: str
    question: str
    gold_answer: str
    gold_solution: str = ""

    def student_record(self) -> StudentRecord:
        return build_student_record(
            example_id=self.example_id,
            split=self.benchmark,
            question=self.question,
        )


def render_student_user_content(question: str) -> str:
    return _require_nonempty_text(question, "question") + STUDENT_PROMPT_SUFFIX


def render_sdft_teacher_user_content(
    original_user_content: str,
    gold_cot: str,
    gold_answer: str,
) -> str:
    """Render the locked SDFT prompt with exactly one explicit answer line."""

    original = _require_nonempty_text(original_user_content, "original_user_content")
    rationale = _require_nonempty_text(gold_cot, "gold_cot")
    answer = _require_nonempty_text(gold_answer, "gold_answer")
    return (
        "\n"
        + original
        + "\n\n"
        + SDFT_DEMONSTRATION_PREFIX
        + rationale
        + "\nThe final answer is: \\boxed{"
        + answer
        + "}"
        + "\n\n"
        + SDFT_INSTRUCTION
        + "\n"
    )


def build_student_record(example_id: str, split: str, question: str) -> StudentRecord:
    record = StudentRecord(
        example_id=_require_nonempty_text(example_id, "example_id"),
        split=_require_nonempty_text(split, "split"),
        prompt=(
            {
                "role": "user",
                "content": render_student_user_content(question),
            },
        ),
    )
    record.to_dict()
    return record


def build_record_bundle(example: MathExample) -> RecordBundle:
    student = build_student_record(example.example_id, example.split, example.question)
    teacher = TeacherRecord(
        example_id=example.example_id,
        split=example.split,
        user_content=render_sdft_teacher_user_content(
            student.prompt[0]["content"], example.gold_cot, example.gold_answer
        ),
    )
    reward = RewardRecord(
        example_id=example.example_id,
        split=example.split,
        ground_truth=example.gold_answer,
    )
    return RecordBundle(student=student, teacher=teacher, reward=reward)


def build_verl_training_row(example: MathExample) -> Dict[str, Any]:
    """Bridge an isolated record bundle into the released VERL row schema.

    Privileged values live only under ``extra_info``. The rollout worker must
    call :func:`student_generation_payload` (or use VERL's derived
    ``raw_prompt_ids`` tensor) rather than forwarding this training row to the
    generation service.
    """

    bundle = build_record_bundle(example)
    original_user_content = bundle.student.prompt[0]["content"]
    return {
        "data_source": MATH_DATASET_ID,
        "prompt": [dict(message) for message in bundle.student.prompt],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": bundle.reward.ground_truth,
        },
        "extra_info": {
            "index": example.source_index,
            "split": example.split,
            "example_id": example.example_id,
            "opd_original_user_content": original_user_content,
            "opd_gold_cot": example.gold_cot,
            "opd_gold_answer": example.gold_answer,
        },
    }


def student_generation_payload(training_row: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy only the unprivileged prompt out of a VERL training row."""

    try:
        extra_info = training_row["extra_info"]
        prompt = training_row["prompt"]
    except KeyError as error:
        raise ValueError("VERL row is missing %s" % error.args[0]) from error
    if not isinstance(extra_info, Mapping):
        raise ValueError("VERL extra_info must be a mapping")
    expected_privileged = {
        "opd_original_user_content",
        "opd_gold_cot",
        "opd_gold_answer",
    }
    if not expected_privileged.issubset(extra_info):
        raise ValueError("VERL extra_info lacks the OPD privileged contract")
    payload = {
        "example_id": extra_info.get("example_id"),
        "split": extra_info.get("split"),
        "prompt": [dict(message) for message in prompt],
    }
    validate_student_record(payload)
    if payload["prompt"][0]["content"] != extra_info["opd_original_user_content"]:
        raise ValueError("OPD original user content differs from the student prompt")
    return payload


def _reject_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("student payload keys must be strings")
            if key.lower() in _FORBIDDEN_STUDENT_KEYS:
                raise ValueError("privileged field in student payload: %s" % key)
            _reject_forbidden_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_forbidden_keys(child)


def validate_student_record(payload: Mapping[str, Any]) -> None:
    """Reject structural gold leakage at the student API boundary."""

    if set(payload) != {"example_id", "split", "prompt"}:
        raise ValueError("student payload must contain only identity, split, and prompt")
    _require_nonempty_text(payload["example_id"], "example_id")
    _require_nonempty_text(payload["split"], "split")
    prompt = payload["prompt"]
    if not isinstance(prompt, (list, tuple)) or len(prompt) != 1:
        raise ValueError("student prompt must contain exactly one message")
    message = prompt[0]
    if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
        raise ValueError("student message schema is invalid")
    if message["role"] != "user":
        raise ValueError("student prompt must contain one user message")
    content = _require_nonempty_text(message["content"], "student user content")
    if not content.endswith(STUDENT_PROMPT_SUFFIX):
        raise ValueError("student prompt does not use the locked suffix")
    _reject_forbidden_keys(payload)
