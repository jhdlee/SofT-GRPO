from __future__ import annotations

import unittest
from unittest.mock import patch

import opd_tools.graders as graders

from opd_tools.graders import (
    extract_lm_eval_flexible_number,
    gsm8k_grader_manifest,
    grade_gsm8k_interfaces,
    lm_eval_flexible_last_number_grade,
    math_verify_full_response_grade,
    released_last_boxed_grade,
)


class GraderTest(unittest.TestCase):
    def test_default_math_verify_rejects_unpinned_version(self):
        graders._math_verify_metric.cache_clear()
        with patch("opd_tools.graders.version", return_value="0.9.0"):
            with self.assertRaisesRegex(RuntimeError, "must be 0.8.0"):
                graders._math_verify_metric()
        graders._math_verify_metric.cache_clear()

    def test_grader_manifest_pins_all_three_sources(self):
        manifest = gsm8k_grader_manifest()
        self.assertEqual(manifest["protocol"], "opd-gsm8k-three-grader-v1")
        self.assertEqual(
            set(manifest["graders"]),
            {"released_last_boxed", "lm_eval_flexible_last_number", "math_verify"},
        )
        self.assertRegex(
            manifest["graders"]["lm_eval_flexible_last_number"][
                "lm_eval_harness_commit"
            ],
            r"^[0-9a-f]{40}$",
        )

    def test_released_last_boxed_uses_only_final_box(self):
        grade = released_last_boxed_grade(
            r"Trial \boxed{9}. Correction: \boxed{1/2}.", r"\frac{1}{2}"
        )
        self.assertTrue(grade.correct)
        self.assertEqual(grade.extracted_answer, "1/2")
        self.assertFalse(released_last_boxed_grade("The answer is 2.", "2").correct)

    def test_released_normalization_exception_falls_back_to_raw_equality(self):
        malformed_but_identical = r"1\text{ u}\text{ v}"
        grade = released_last_boxed_grade(
            r"work \boxed{" + malformed_but_identical + "}",
            malformed_but_identical,
        )
        self.assertTrue(grade.correct)
        self.assertIsNone(grade.normalized_prediction)

    def test_lm_eval_flexible_selects_last_regex_match(self):
        prediction = "We considered 10, then found the answer is $1,234."
        self.assertEqual(extract_lm_eval_flexible_number(prediction), "$1,234.")
        grade = lm_eval_flexible_last_number_grade(prediction, "1,234")
        self.assertTrue(grade.correct)
        self.assertEqual(grade.normalized_prediction, "1234")

    def test_lm_eval_accepts_single_digit_and_negative(self):
        self.assertTrue(lm_eval_flexible_last_number_grade("Answer: 7.", "7").correct)
        self.assertTrue(lm_eval_flexible_last_number_grade("Answer: -3", "-3").correct)
        self.assertFalse(lm_eval_flexible_last_number_grade("No numeric answer", "7").correct)

    def test_math_verify_accepts_injected_offline_scorer(self):
        calls = []

        def scorer(prediction, gold):
            calls.append((prediction, gold))
            return prediction.endswith(r"\boxed{2}") and gold == "2"

        grade = math_verify_full_response_grade("Full work. \\boxed{2}", "2", scorer)
        self.assertTrue(grade.correct)
        self.assertEqual(calls, [("Full work. \\boxed{2}", "2")])

    def test_three_interfaces_remain_separate(self):
        scores = grade_gsm8k_interfaces(
            "Reasoning mentions 1 and ends with answer 2.",
            "2",
            math_verify_scorer=lambda prediction, gold: True,
        )
        self.assertEqual(
            set(scores),
            {"released_last_boxed", "lm_eval_flexible_last_number", "math_verify"},
        )
        self.assertFalse(scores["released_last_boxed"]["correct"])
        self.assertTrue(scores["lm_eval_flexible_last_number"]["correct"])
        self.assertTrue(scores["math_verify"]["correct"])


if __name__ == "__main__":
    unittest.main()
