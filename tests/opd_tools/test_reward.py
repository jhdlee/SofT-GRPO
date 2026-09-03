from __future__ import annotations

import contextlib
import io
import unittest
import importlib.util
from pathlib import Path
from unittest.mock import patch

from opd_tools.graders import Grade
from opd_tools.reward import compute_score


class RewardTest(unittest.TestCase):
    def test_verl_file_path_loader_can_resolve_opd_tools_from_any_cwd(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "verl-0.4.x"
            / "verl"
            / "utils"
            / "reward_score"
            / "opd_math.py"
        )
        spec = importlib.util.spec_from_file_location("custom_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertIs(module.compute_score, compute_score)

    def test_reward_dict_preserves_released_scalar_and_numeric_extras(self):
        with patch(
            "opd_tools.reward.math_verify_full_response_grade",
            return_value=Grade(True, None, None, "2"),
        ):
            result = compute_score(
                data_source="math500",
                solution_str=r"Reasoning. \boxed{2}",
                ground_truth="2",
                extra_info={"split": "validation"},
            )
        self.assertEqual(
            result,
            {"score": 1.0, "released_reward": 1.0, "math_verify": 1.0},
        )
        self.assertTrue(all(type(value) is float for value in result.values()))

    def test_math_verify_diagnostic_does_not_change_training_reward(self):
        with patch(
            "opd_tools.reward.math_verify_full_response_grade",
            return_value=Grade(True, None, None, "2"),
        ):
            result = compute_score(
                "math", "The answer is 2.", "2", extra_info={"split": "validation"}
            )
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["released_reward"], 0.0)
        self.assertEqual(result["math_verify"], 1.0)

    def test_training_skips_expensive_math_verify(self):
        with patch(
            "opd_tools.reward.math_verify_full_response_grade",
            side_effect=AssertionError("Math-Verify must not run during training"),
        ):
            result = compute_score(
                "math",
                r"Work. \boxed{2}",
                "2",
                extra_info={"split": "train"},
            )
        self.assertEqual(result, {"score": 1.0, "released_reward": 1.0})

    def test_scalar_reward_matches_pinned_released_implementation(self):
        source_root = Path(__file__).resolve().parents[2]
        upstream_path = (
            source_root
            / "verl-0.4.x"
            / "verl"
            / "utils"
            / "reward_score"
            / "math_reward.py"
        )
        spec = importlib.util.spec_from_file_location(
            "pinned_upstream_math_reward", upstream_path
        )
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)

        cases = (
            (r"Work. \boxed{2}", "2"),
            (r"Work. \boxed{1/2}", r"\frac{1}{2}"),
            (r"Work. \boxed{.5}", "0.5"),
            (r"Earlier \boxed{1}; final \boxed{3}", "3"),
            (r"Work. \fbox{4}", "4"),
            (r"Work. \boxed 17$ trailing", "17"),
            (r"Work. \boxed{x=2}", "2"),
            (r"Work. \boxed{2\text{ cm}}", "2"),
            (r"No boxed answer: 2", "2"),
            (r"Work. \boxed{\sqrt}", r"\sqrt"),
            (r"Work. \boxed{2\text{ cm}\text{ twice}}", r"2\text{ cm}\text{ twice}"),
            (r"Work. \boxed{unbalanced", "unbalanced"),
        )
        for prediction, gold in cases:
            with self.subTest(prediction=prediction, gold=gold):
                # The released function prints normalization exceptions; those
                # messages are not part of reward semantics.
                with contextlib.redirect_stdout(io.StringIO()):
                    released = float(upstream.compute_score(prediction, gold))
                ours = compute_score(
                    "math",
                    prediction,
                    gold,
                    extra_info={"split": "train"},
                )
                self.assertEqual(ours["score"], released)


if __name__ == "__main__":
    unittest.main()
