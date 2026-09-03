import unittest

from opd_tools.aggregate_eval import _Accumulator, _flatten_wandb_metrics, aggregate
from opd_tools.evaluation import (
    BENCHMARKS,
    COMMON_GENERATION_SEEDS,
    HARD_TOKEN_GENERATION_SEEDS,
    INFERENCE_MODES,
    MODEL_LABELS,
    graders_for_benchmark,
)


class AggregateEvaluationTests(unittest.TestCase):
    def _small_complete_accumulator(self):
        accumulator = _Accumulator()
        for model_index, model in enumerate(MODEL_LABELS):
            for benchmark in BENCHMARKS:
                for mode in INFERENCE_MODES:
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
            initial["native_soft"]["graders"]["math_verify"]["mean_at_32"], 0.0
        )
        baseline = summary["models"]["baseline"]["benchmarks"]["math500"]
        self.assertEqual(
            baseline["native_soft"]["graders"]["math_verify"]["mean_at_32"], 1 / 32
        )
        flattened = _flatten_wandb_metrics(metrics, comparisons)
        self.assertIn(
            "eval/math500/baseline/native_soft/math_verify/mean_at_32",
            flattened,
        )
        self.assertEqual(
            flattened["eval/math500/baseline/native_soft/response_length_mean"],
            10.0,
        )
        self.assertIn(
            "eval_difference/math500/opd_minus_baseline/native_soft/math_verify/mean_at_32",
            flattened,
        )
        self.assertNotIn("opd/kl_weight", flattened)

    def test_accumulator_rejects_duplicate_pair(self):
        accumulator = _Accumulator()
        row = {
            "model_label": "initial",
            "benchmark": "math500",
            "example_id": "x",
            "inference_mode": "hard_token",
            "sample_index": 0,
            "generation_seed": 11,
            "gold_answer": "1",
            "scores": {"math_verify": True},
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
