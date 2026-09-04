import unittest
from unittest import mock

from opd_tools.aggregate_eval import (
    _Accumulator,
    _flatten_wandb_metrics,
    _score_record,
    aggregate,
    validate_common_source_identity,
)
from opd_tools.evaluation import (
    BENCHMARKS,
    COMMON_GENERATION_SEEDS,
    EVALUATION_SCHEMA_VERSION,
    EVALUATION_CELLS,
    HARD_TOKEN_GENERATION_SEEDS,
    MODEL_LABELS,
    graders_for_benchmark,
    evaluation_request_seed,
)


class AggregateEvaluationTests(unittest.TestCase):
    def test_generation_cells_require_one_expected_source_identity(self):
        expected = {"parent_commit": "a" * 40, "fork_commit": "b" * 40}
        self.assertEqual(
            validate_common_source_identity(
                [expected, dict(expected)],
                expected_parent_commit="a" * 40,
                expected_fork_commit="b" * 40,
            ),
            expected,
        )
        with self.assertRaisesRegex(ValueError, "one source identity"):
            validate_common_source_identity(
                [expected, {**expected, "fork_commit": "c" * 40}]
            )
        with self.assertRaisesRegex(ValueError, "commits are invalid"):
            validate_common_source_identity(
                [{"parent_commit": "", "fork_commit": "b" * 40}]
            )
        with self.assertRaisesRegex(ValueError, "aggregation checkout"):
            validate_common_source_identity(
                [expected],
                expected_parent_commit="c" * 40,
                expected_fork_commit="b" * 40,
            )

    def test_native_soft_invalid_boundary_is_incorrect_without_calling_graders(self):
        row = {
            "model_label": "initial",
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "benchmark": "math500",
            "example_id": "math500-0",
            "inference_mode": "native_soft",
            "sample_index": 0,
            "generation_seed": 11,
            "request_seed": evaluation_request_seed(11, "math500", "math500-0"),
            # Deliberately gradeable text, but no released soft-to-hard boundary.
            "response": "latent hard shadow with final answer \\boxed{1}",
            "response_token_count": 2,
            "finish_reason": "length",
            "capped": True,
            "latent_token_count": 2,
            "hard_token_count": 0,
            "close_tag": False,
            "soft_to_hard": False,
            "all_soft": True,
            "mixture_entropy_mean": 0.1,
            "top1_weight_mean": 0.9,
            "soft_hard_agreement": 1.0,
            "gold_answer": "1",
        }
        with mock.patch(
            "opd_tools.aggregate_eval.math_verify_full_response_grade"
        ) as math_verify, mock.patch(
            "opd_tools.aggregate_eval.released_last_boxed_grade"
        ) as released:
            scored = _score_record(row)
        self.assertEqual(
            scored["scores"],
            {"math_verify": False, "released_last_boxed": False},
        )
        math_verify.assert_not_called()
        released.assert_not_called()

    def _small_complete_accumulator(self):
        accumulator = _Accumulator()
        for model_index, (model, mode) in enumerate(EVALUATION_CELLS):
            for benchmark in BENCHMARKS:
                seeds = (
                    COMMON_GENERATION_SEEDS
                    if mode == "native_soft"
                    else HARD_TOKEN_GENERATION_SEEDS
                )
                for sample_index, seed in enumerate(seeds):
                    scores = {
                        grader: bool(model_index > 0 and sample_index == 0)
                        for grader in graders_for_benchmark(benchmark)
                    }
                    accumulator.add(
                        {
                            "model_label": model,
                            "benchmark": benchmark,
                            "example_id": benchmark + "-example",
                            "inference_mode": mode,
                            "sample_index": sample_index,
                            "generation_seed": seed,
                            "gold_answer": "1",
                            "scores": scores,
                            "response_token_count": 10,
                            "capped": False,
                            "latent_token_count": 8 if mode == "native_soft" else 0,
                            "hard_token_count": 2 if mode == "native_soft" else 10,
                            "close_tag": True,
                            "soft_to_hard": mode == "native_soft",
                            "all_soft": False,
                            "mixture_entropy_mean": 0.1
                            if mode == "native_soft"
                            else None,
                            "top1_weight_mean": 0.9
                            if mode == "native_soft"
                            else None,
                            "soft_hard_agreement": 1.0
                            if mode == "native_soft"
                            else None,
                        }
                    )
        return accumulator

    def test_report_schema_and_wandb_keys_are_benchmark_qualified(self):
        summary, metrics, comparisons = aggregate(self._small_complete_accumulator())
        self.assertTrue(summary["exploratory_single_seed"])
        self.assertEqual(summary["bootstrap"]["resamples"], 10_000)
        self.assertEqual(summary["bootstrap"]["seed"], 11)
        self.assertEqual(set(summary["models"]), set(MODEL_LABELS))
        initial = summary["models"]["initial"]["benchmarks"]["math500"]
        self.assertEqual(
            initial["native_soft"]["graders"]["math_verify"]["pass_at_1"], 0.0
        )
        baseline = summary["models"]["softgrpo_math_s11"]["benchmarks"]["math500"]
        self.assertEqual(
            baseline["native_soft"]["graders"]["math_verify"]["pass_at_1"], 1 / 32
        )
        flattened = _flatten_wandb_metrics(metrics, comparisons)
        self.assertIn(
            "eval/math500/softgrpo_math_s11/native_soft/math_verify/pass_at_1",
            flattened,
        )
        self.assertIn(
            "eval/math500/softgrpo_math_s11/native_soft/released_last_boxed/pass_at_1",
            flattened,
        )
        self.assertEqual(
            flattened["eval/math500/softgrpo_math_s11/native_soft/response_length_mean"],
            10.0,
        )
        self.assertIn(
            "eval_difference/math500/softgrpo_math_opd_s11_minus_softgrpo_math_s11/native_soft_vs_native_soft/math_verify/pass_at_1",
            flattened,
        )
        self.assertNotIn("opd/kl_weight", flattened)

    def test_accumulator_rejects_duplicate_pair(self):
        accumulator = _Accumulator()
        row = {
            "model_label": "hardgrpo_math_s11",
            "benchmark": "math500",
            "example_id": "x",
            "inference_mode": "hard_token",
            "sample_index": 0,
            "generation_seed": 11,
            "gold_answer": "1",
            "scores": {"math_verify": True, "released_last_boxed": True},
            "response_token_count": 1,
            "capped": False,
            "latent_token_count": 0,
            "hard_token_count": 1,
            "close_tag": False,
            "soft_to_hard": False,
            "all_soft": False,
            "mixture_entropy_mean": None,
            "top1_weight_mean": None,
            "soft_hard_agreement": None,
        }
        accumulator.add(row)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            accumulator.add(row)


if __name__ == "__main__":
    unittest.main()
