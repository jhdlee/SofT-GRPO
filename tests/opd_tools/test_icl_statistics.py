from __future__ import annotations

import math
import unittest

from opd_tools.icl import (
    BOOTSTRAP_RESAMPLES,
    normalized_answer_copy,
    paired_bootstrap_difference,
    paired_bootstrap_difference_in_differences,
    pass_at_k,
    pass_metrics_by_example,
    rationale_token_overlap_f1,
    rescue_harm_rates,
    summarize_pass_metrics,
)


class ICLStatisticsTests(unittest.TestCase):
    def test_pass_at_1_is_sample_accuracy_and_pass_at_8_is_any_success(self):
        outcomes = {
            "a": (True,) + (False,) * 7,
            "b": (False,) * 8,
        }
        by_example = pass_metrics_by_example(outcomes)
        self.assertEqual(by_example["a"], {"pass_at_1": 1 / 8, "pass_at_8": 1.0})
        self.assertEqual(by_example["b"], {"pass_at_1": 0.0, "pass_at_8": 0.0})
        summary = summarize_pass_metrics(outcomes)
        self.assertEqual(summary["pass_at_1"], 1 / 16)
        self.assertEqual(summary["pass_at_8"], 0.5)
        self.assertEqual(summary["samples_per_example"], 8)
        self.assertEqual(pass_at_k(8, 3, 1), 3 / 8)
        self.assertEqual(pass_at_k(8, 1, 8), 1.0)

    def test_pass_contract_rejects_non_eight_sample_reports(self):
        with self.assertRaisesRegex(ValueError, "eight"):
            pass_metrics_by_example({"a": (True,) * 7})
        with self.assertRaises(ValueError):
            pass_at_k(8, 1, 9)

    def test_bootstrap_is_paired_deterministic_and_exactly_10000(self):
        treatment = {"a": 1.0, "b": 1.0, "c": 0.0, "d": 1.0}
        control = {"a": 0.0, "b": 1.0, "c": 0.0, "d": 0.0}
        first = paired_bootstrap_difference(treatment, control)
        second = paired_bootstrap_difference(treatment, control)
        self.assertEqual(first, second)
        self.assertEqual(first["difference"], 0.5)
        self.assertEqual(first["resamples"], BOOTSTRAP_RESAMPLES)
        self.assertLessEqual(first["ci_low"], first["difference"])
        self.assertGreaterEqual(first["ci_high"], first["difference"])
        with self.assertRaisesRegex(ValueError, "10,000"):
            paired_bootstrap_difference(treatment, control, resamples=999)

    def test_difference_in_differences_uses_one_paired_resample_unit(self):
        result = paired_bootstrap_difference_in_differences(
            {"a": 1.0, "b": 1.0},
            {"a": 0.0, "b": 1.0},
            {"a": 0.0, "b": 1.0},
            {"a": 0.0, "b": 0.0},
        )
        # Post delta is (1, 0), starting delta is (0, 1).
        self.assertEqual(result["difference"], 0.0)
        self.assertIn("post_treatment", result["estimand"])

    def test_rescue_and_harm_pair_the_same_sample_positions(self):
        result = rescue_harm_rates(
            {"a": (True, False, True), "b": (False, True)},
            {"a": (False, True, True), "b": (False, True)},
        )
        self.assertEqual(result["rescued"], 1)
        self.assertEqual(result["harmed"], 1)
        self.assertEqual(result["control_incorrect"], 2)
        self.assertEqual(result["control_correct"], 3)
        self.assertEqual(result["rescue_rate"], 0.5)
        self.assertAlmostEqual(result["harm_rate"], 1 / 3)

    def test_rescue_and_harm_use_null_for_undefined_denominators(self):
        no_control_errors = rescue_harm_rates(
            {"a": (True, False)},
            {"a": (True, True)},
        )
        self.assertIsNone(no_control_errors["rescue_rate"])
        self.assertEqual(no_control_errors["harm_rate"], 0.5)

        no_control_successes = rescue_harm_rates(
            {"a": (True, False)},
            {"a": (False, False)},
        )
        self.assertEqual(no_control_successes["rescue_rate"], 0.5)
        self.assertIsNone(no_control_successes["harm_rate"])

    def test_copy_diagnostics_have_explicit_semantics(self):
        self.assertTrue(normalized_answer_copy("Work. \\boxed{007}", "7", "aime2024"))
        self.assertFalse(normalized_answer_copy("Work. \\boxed{008}", "7", "aime2024"))
        self.assertEqual(rationale_token_overlap_f1("alpha beta", "alpha beta"), 1.0)
        overlap = rationale_token_overlap_f1("alpha beta", "alpha gamma")
        self.assertGreater(overlap, 0.0)
        self.assertLess(overlap, 1.0)
        self.assertFalse(math.isnan(overlap))


if __name__ == "__main__":
    unittest.main()
