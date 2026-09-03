"""Reproducible data and evaluation utilities for the OPD study.

This package is intentionally separate from the released SofT-GRPO trainer.
The trainer integration may consume these records, but importing ``opd_tools``
does not import Ray, Torch, VERL, or any network-facing dependency.
"""

from .constants import (
    GSM8K_DATASET_ID,
    GSM8K_DATASET_REVISION,
    MATH500_DATASET_ID,
    MATH500_DATASET_REVISION,
    MATH_DATASET_ID,
    MATH_DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
)
from .data import (
    MathCleaningReport,
    build_gsm8k_evaluation_records,
    build_math500_evaluation_records,
    clean_math_lighteval,
    prepare_math_example_splits,
    prepare_math_training_splits,
    prepare_verl_math_splits,
)
from .records import (
    EvaluationRecord,
    MathExample,
    RecordBundle,
    RewardRecord,
    StudentRecord,
    TeacherRecord,
    build_record_bundle,
    build_verl_training_row,
    student_generation_payload,
)

__all__ = [
    "EvaluationRecord",
    "GSM8K_DATASET_ID",
    "GSM8K_DATASET_REVISION",
    "MATH500_DATASET_ID",
    "MATH500_DATASET_REVISION",
    "MATH_DATASET_ID",
    "MATH_DATASET_REVISION",
    "MODEL_ID",
    "MODEL_REVISION",
    "MathCleaningReport",
    "MathExample",
    "RecordBundle",
    "RewardRecord",
    "StudentRecord",
    "TeacherRecord",
    "build_gsm8k_evaluation_records",
    "build_math500_evaluation_records",
    "build_record_bundle",
    "build_verl_training_row",
    "clean_math_lighteval",
    "prepare_math_training_splits",
    "prepare_math_example_splits",
    "prepare_verl_math_splits",
    "student_generation_payload",
]
