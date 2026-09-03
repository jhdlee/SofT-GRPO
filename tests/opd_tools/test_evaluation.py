import math
import json
import tempfile
import unittest
from pathlib import Path

from opd_tools.evaluation import (
    BOOTSTRAP_RESAMPLES,
    COMMON_GENERATION_SEEDS,
    HARD_TOKEN_GENERATION_SEEDS,
    GenerationRecord,
    correctness_by_example,
    evaluation_request_seed,
    example_level_metric,
    paired_bootstrap_difference,
    pass_at_k,
)
from opd_tools.generate_eval import (
    EVALUATION_SAMPLING_PROTOCOLS,
    _verify_shard,
    _write_shard,
    cleanup_stale_atomic_files,
    native_soft_diagnostics,
)


def generation_record(**overrides):
    values = {
        "model_label": "initial",
        "benchmark": "math500",
        "example_id": "math500-0",
        "inference_mode": "native_soft",
        "sample_index": 0,
        "generation_seed": COMMON_GENERATION_SEEDS[0],
        "request_seed": evaluation_request_seed(
            COMMON_GENERATION_SEEDS[0], "math500", "math500-0"
        ),
        "response": "reasoning</think>\\boxed{1}",
        "response_token_count": 3,
        "finish_reason": "stop",
        "capped": False,
        "latent_token_count": 2,
        "hard_token_count": 1,
        "close_tag": True,
        "soft_to_hard": True,
        "all_soft": False,
        "mixture_entropy_mean": 0.2,
        "top1_weight_mean": 0.9,
        "soft_hard_agreement": 1.0,
        "gold_answer": "1",
    }
    values.update(overrides)
    return GenerationRecord(**values)


class EvaluationFormulaTests(unittest.TestCase):
    def test_locked_common_seed_inventory(self):
        self.assertEqual(COMMON_GENERATION_SEEDS, tuple(range(11, 43)))
        self.assertEqual(HARD_TOKEN_GENERATION_SEEDS, (11,))
        seeds = {
            evaluation_request_seed(seed, "math500", "example")
            for seed in COMMON_GENERATION_SEEDS
        }
        self.assertEqual(len(seeds), 32)
        self.assertEqual(
            evaluation_request_seed(11, "math500", "example"),
            evaluation_request_seed(11, "math500", "example"),
        )
        self.assertNotEqual(
            evaluation_request_seed(11, "math500", "example"),
            evaluation_request_seed(11, "math500", "other"),
        )

    def test_pass_at_k_exact_boundaries(self):
        self.assertEqual(pass_at_k(32, 0, 8), 0.0)
        self.assertEqual(pass_at_k(32, 1, 32), 1.0)
        self.assertEqual(pass_at_k(32, 32, 8), 1.0)
        expected = 1.0 - math.comb(30, 8) / math.comb(32, 8)
        self.assertAlmostEqual(pass_at_k(32, 2, 8), expected)
        with self.assertRaises(ValueError):
            pass_at_k(4, 1, 5)

    def test_example_level_mean_and_pass(self):
        outcomes = {"a": (1,) + (0,) * 31, "b": (0,) * 32}
        means = example_level_metric(outcomes, "mean_at_32")
        passes = example_level_metric(outcomes, "pass_at_32")
        self.assertEqual(means, {"a": 1 / 32, "b": 0.0})
        self.assertEqual(passes, {"a": 1.0, "b": 0.0})

    def test_pairing_requires_exact_common_seed_set(self):
        rows = [
            {
                "example_id": "x",
                "generation_seed": seed,
                "scores": {"math_verify": seed == 11},
            }
            for seed in COMMON_GENERATION_SEEDS[:-1]
        ]
        with self.assertRaisesRegex(ValueError, "exact seed set"):
            correctness_by_example(rows, grader="math_verify")
        rows.append(
            {
                "example_id": "x",
                "generation_seed": COMMON_GENERATION_SEEDS[-1],
                "scores": {"math_verify": False},
            }
        )
        grouped = correctness_by_example(rows, grader="math_verify")
        self.assertEqual(len(grouped["x"]), 32)

    def test_bootstrap_is_paired_deterministic_and_exactly_10000(self):
        treatment = {"a": 1.0, "b": 0.0, "c": 1.0, "d": 1.0}
        control = {"a": 0.0, "b": 0.0, "c": 1.0, "d": 0.0}
        first = paired_bootstrap_difference(treatment, control)
        second = paired_bootstrap_difference(treatment, control)
        self.assertEqual(first, second)
        self.assertEqual(first["resamples"], BOOTSTRAP_RESAMPLES)
        self.assertEqual(first["bootstrap_seed"], 11)
        self.assertEqual(first["difference"], 0.5)
        self.assertLessEqual(first["ci_low"], first["difference"])
        self.assertGreaterEqual(first["ci_high"], first["difference"])
        with self.assertRaisesRegex(ValueError, "10,000"):
            paired_bootstrap_difference(treatment, control, resamples=9999)
        with self.assertRaisesRegex(ValueError, "identical"):
            paired_bootstrap_difference({"a": 1.0}, {"b": 0.0})


class EvaluationSchemaTests(unittest.TestCase):
    def test_generation_record_round_trip_and_unknown_field_rejection(self):
        record = generation_record()
        self.assertEqual(GenerationRecord.from_mapping(record.to_dict()), record)
        bad = record.to_dict()
        bad["unexpected"] = 1
        with self.assertRaisesRegex(ValueError, "unknown"):
            GenerationRecord.from_mapping(bad)

    def test_generation_shard_is_atomic_and_authenticated(self):
        record = generation_record()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed_11.jsonl"
            manifest = _write_shard(path, [record])
            self.assertEqual(manifest["row_count"], 1)
            self.assertEqual(
                _verify_shard(path, path.with_suffix(".manifest.json")), manifest
            )
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record.to_dict()) + "\n")
            with self.assertRaisesRegex(ValueError, "authentication"):
                _verify_shard(path, path.with_suffix(".manifest.json"))

    def test_hard_token_schema_rejects_latent_diagnostics(self):
        with self.assertRaisesRegex(ValueError, "hard-token"):
            generation_record(
                inference_mode="hard_token",
                latent_token_count=0,
                hard_token_count=3,
                soft_to_hard=False,
                all_soft=False,
                sample_index=0,
                generation_seed=11,
            )

    def test_native_soft_diagnostics_uses_released_support_sentinel(self):
        diagnostics = native_soft_diagnostics(
            response_token_ids=[10, 20, 30],
            topk_ids=[[10, 11, 12], [20, 21, 22], [30, 0, 0]],
            perturbed_logits=[[2.0, 1.0, 0.0], [4.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            gumbel_temperature=0.5,
            response_text="work</think> answer",
        )
        self.assertEqual(diagnostics["latent_token_count"], 2)
        self.assertEqual(diagnostics["hard_token_count"], 1)
        self.assertTrue(diagnostics["soft_to_hard"])
        self.assertTrue(diagnostics["close_tag"])
        self.assertFalse(diagnostics["all_soft"])
        self.assertEqual(diagnostics["soft_hard_agreement"], 1.0)
        self.assertGreater(diagnostics["top1_weight_mean"], 0.5)

    def test_released_and_training_matched_sampling_are_not_ambiguous(self):
        released = EVALUATION_SAMPLING_PROTOCOLS["released_anchor"]
        matched = EVALUATION_SAMPLING_PROTOCOLS["training_matched"]
        self.assertEqual((released["top_k"], released["max_new_tokens"]), (30, 32768))
        self.assertEqual(
            (matched["top_k"], matched["gumbel_softmax_temperature"]),
            (5, 0.1),
        )

    def test_cleanup_stale_atomic_files_is_narrow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "math500"
            benchmark.mkdir()
            stale = benchmark / ".seed_11.jsonl.deadbeef.tmp"
            stale.write_bytes(b"partial")
            committed = benchmark / "seed_11.jsonl"
            committed.write_bytes(b"complete")
            self.assertEqual(cleanup_stale_atomic_files(root), [stale])
            self.assertEqual(committed.read_bytes(), b"complete")

    def test_cleanup_stale_atomic_files_rejects_unknown_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unknown = root / ".unrelated.deadbeef.tmp"
            unknown.write_bytes(b"do not remove")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                cleanup_stale_atomic_files(root)
            self.assertTrue(unknown.exists())


if __name__ == "__main__":
    unittest.main()
