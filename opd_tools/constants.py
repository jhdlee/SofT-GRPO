"""Immutable source and protocol identifiers for the seed-11 study."""

from typing import Final

SOFTGRPO_UPSTREAM_COMMIT: Final = "8d3c61380b15c3400818da5ce41c62c293a1bfb4"
LM_EVAL_HARNESS_COMMIT: Final = "b954108c9baaaa934b4ad842033b31a97ee30816"
MATH_VERIFY_VERSION: Final = "0.8.0"

MODEL_ID: Final = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_REVISION: Final = "c46dac620b4e4f12c5662a2133376a2823458d0e"

MATH_DATASET_ID: Final = "DigitalLearningGmbH/MATH-lighteval"
MATH_DATASET_CONFIG: Final = "default"
MATH_DATASET_REVISION: Final = "92ace7ed9c5f22d9148ea70c403948eae7bed2e8"
MATH_SOURCE_TRAIN_SIZE: Final = 7_500
MATH_CLEAN_SIZE: Final = 7_497
MATH_TRAIN_SIZE: Final = 6_985
MATH_VALIDATION_SIZE: Final = 512
MATH_SPLIT_SEED: Final = 42
MATH_VALIDATION_IDS_SHA256: Final = (
    "126328bfe584655607174d96b16bc24f88de70399234bdff9b4428f14ee8e084"
)
MATH_EMPTY_ANSWER_INDICES: Final = (5_341, 5_343)
MATH_DUPLICATE_DROP_INDICES: Final = (959,)
MATH_DUPLICATE_KEEP_BY_DROP: Final = {959: 925}
MATH_RELEASED_EXTRACTOR_DISAGREEMENT_INDICES: Final = (252,)
MATH_RELEASED_EXTRACTOR_DISAGREEMENTS: Final = {252: ("17", "{17}.")}

MATH500_DATASET_ID: Final = "HuggingFaceH4/MATH-500"
MATH500_DATASET_CONFIG: Final = "default"
MATH500_DATASET_REVISION: Final = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
MATH500_TEST_SIZE: Final = 500

GSM8K_DATASET_ID: Final = "openai/gsm8k"
GSM8K_DATASET_CONFIG: Final = "main"
GSM8K_DATASET_REVISION: Final = "740312add88f781978c0658806c59bc2815b9866"
GSM8K_TEST_SIZE: Final = 1_319

# These evaluation records are repository-owned assets pinned transitively by
# SOFTGRPO_UPSTREAM_COMMIT. Hashes protect against local edits after checkout.
RELEASED_EVAL_FILE_SHA256: Final = {
    "aime2024": "ef02255e61539115f6274a0765ff10076df416db5f90d41500d472cb5184e560",
    "aime2025": "b9d730eab85774f6ce5dbb30d813520fea8c0f543e8224bc4de853f65084477a",
    "amc23": "a08542e241ab31d2c99dedbe84b2755400e5bd53e6d8d16678b14b897530254c",
}
RELEASED_EVAL_COUNTS: Final = {"aime2024": 30, "aime2025": 30, "amc23": 40}

STUDENT_PROMPT_SUFFIX: Final = (
    " Let's think step by step and output the final answer within \\boxed{}."
)
SDFT_DEMONSTRATION_PREFIX: Final = (
    "This is an example for a response to the question:\n"
)
SDFT_INSTRUCTION: Final = (
    "Now answer with a response of your own, including the thinking process."
)

DATA_PROTOCOL: Final = "opd-softgrpo-math-data-v1"
DATA_SCHEMA_VERSION: Final = 1
GSM8K_GRADER_PROTOCOL: Final = "opd-gsm8k-three-grader-v1"
MATERIALIZATION_PROTOCOL: Final = "opd-softgrpo-materialized-data-v1"
