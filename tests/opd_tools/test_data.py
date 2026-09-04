from __future__ import annotations

import json
import unittest

from opd_tools.constants import (
    GSM8K_DATASET_REVISION,
    MATH500_DATASET_REVISION,
    MATH_DATASET_REVISION,
    MODEL_REVISION,
)
from opd_tools.data import (
    build_gsm8k_evaluation_records,
    build_math500_evaluation_records,
    clean_math_lighteval,
    stratified_math_split,
)
from opd_tools.records import (
    build_record_bundle,
    build_verl_training_row,
    student_generation_payload,
    validate_student_record,
)


def math_row(index: int, subject: str = "Algebra", level: str = "Level 1"):
    return {
        "problem": "Problem %d" % index,
        "level": level,
        "solution": "Reasoning %d, hence \\boxed{%d}." % (index, index),
        "type": subject,
    }


class MathDataTest(unittest.TestCase):
    def test_source_revisions_are_immutable_commits(self):
        for revision in (
            MODEL_REVISION,
            MATH_DATASET_REVISION,
            MATH500_DATASET_REVISION,
            GSM8K_DATASET_REVISION,
        ):
            self.assertRegex(revision, r"^[0-9a-f]{40}$")

    def test_cleaning_detects_empty_box_and_whitespace_duplicate(self):
        rows = [math_row(index) for index in range(8)]
        rows[2]["solution"] = "Nonempty reasoning but no answer: \\boxed{}."
        rows[6]["problem"] = "Problem    3"
        rows[6]["solution"] = "A different valid proof ending in \\boxed{3}."

        cleaned, report = clean_math_lighteval(rows, enforce_pinned_contract=False)

        self.assertEqual([example.source_index for example in cleaned], [0, 1, 3, 4, 5, 7])
        self.assertEqual(report.empty_answer_source_indices, (2,))
        self.assertEqual(report.duplicate_drop_source_indices, (6,))
        self.assertEqual(report.duplicate_keep_by_drop, ((6, 3),))
        self.assertEqual(report.released_extractor_disagreement_source_indices, ())
        self.assertEqual(report.released_extractor_disagreements, ())

    def test_official_size_cleaning_and_split_contract(self):
        rows = [
            math_row(
                index,
                subject=("Algebra", "Geometry", "Number Theory")[index % 3],
                level="Level %d" % (index % 5 + 1),
            )
            for index in range(7_500)
        ]
        rows[5_341]["solution"] = "Valid rationale, empty final answer \\boxed{}."
        rows[5_343]["solution"] = "Another rationale, empty answer \\boxed{}."
        rows[959]["problem"] = rows[925]["problem"] + "  "
        rows[959]["solution"] = "Independent proof with final \\boxed{925}."
        rows[252]["solution"] = (
            "An intermediate display is \\boxed {17}.$ "
            "The final answer is \\boxed{17}."
        )

        cleaned, report = clean_math_lighteval(rows)
        splits = stratified_math_split(cleaned, validation_size=512, seed=42)

        self.assertEqual(report.source_count, 7_500)
        self.assertEqual(report.clean_count, 7_497)
        self.assertEqual(report.released_extractor_disagreement_source_indices, (252,))
        self.assertEqual(
            report.released_extractor_disagreements,
            ((252, "17", "{17}."),),
        )
        self.assertEqual(len(splits["train"]), 6_985)
        self.assertEqual(len(splits["validation"]), 512)
        train_ids = {example.example_id for example in splits["train"]}
        validation_ids = {example.example_id for example in splits["validation"]}
        self.assertFalse(train_ids & validation_ids)
        self.assertNotIn("math-train-000959", train_ids | validation_ids)

    def test_split_is_deterministic_and_stratified(self):
        rows = [
            math_row(
                index,
                subject="Algebra" if index < 15 else "Geometry",
                level="Level 1" if index % 2 else "Level 2",
            )
            for index in range(30)
        ]
        cleaned, _ = clean_math_lighteval(rows, enforce_pinned_contract=False)
        first = stratified_math_split(cleaned, validation_size=6, seed=42)
        second = stratified_math_split(cleaned, validation_size=6, seed=42)
        self.assertEqual(
            [item.example_id for item in first["validation"]],
            [item.example_id for item in second["validation"]],
        )
        self.assertEqual(
            {(item.subject, item.level) for item in first["validation"]},
            {(item.subject, item.level) for item in cleaned},
        )

    def test_student_channel_contains_no_privileged_structure(self):
        rows = [math_row(index) for index in range(4)]
        cleaned, _ = clean_math_lighteval(rows, enforce_pinned_contract=False)
        example = cleaned[0]
        # Assign the split in the same way the preparation path does.
        assigned = stratified_math_split(cleaned, validation_size=1)["train"][0]
        bundle = build_record_bundle(assigned)
        student = bundle.student.to_dict()
        validate_student_record(student)

        encoded = json.dumps(student)
        self.assertNotIn("gold_solution", encoded)
        self.assertNotIn("ground_truth", encoded)
        self.assertNotIn(example.gold_solution, encoded)
        self.assertNotIn(assigned.gold_solution, bundle.teacher.user_content)
        self.assertEqual(bundle.teacher.user_content.count(assigned.gold_cot), 1)
        self.assertEqual(
            bundle.teacher.user_content.count(
                "[Hint] The correct answer is %s." % assigned.gold_answer
            ),
            1,
        )
        self.assertEqual(
            bundle.teacher.user_content,
            "\n%s\n\n[Hint] The correct answer is %s. A common way to solve this is:\n"
            "%s\n\n"
            "[Instruction] If possible, derive the answer %s using an alternative, "
            "equally rigorous mathematical approach to the one provided above. Otherwise, "
            "improve the given reasoning by making it clearer, more complete, and logically "
            "sound. Do NOT state that you were given the answer or reference."
            % (
                bundle.student.prompt[0]["content"],
                assigned.gold_answer,
                assigned.gold_cot,
                assigned.gold_answer,
            ),
        )
        self.assertEqual(bundle.reward.ground_truth, assigned.gold_answer)

        contaminated = dict(student)
        contaminated["gold_answer"] = assigned.gold_answer
        with self.assertRaisesRegex(ValueError, "only identity"):
            validate_student_record(contaminated)

    def test_verl_bridge_confines_privileged_opd_fields_to_extra_info(self):
        rows = [math_row(index) for index in range(4)]
        cleaned, _ = clean_math_lighteval(rows, enforce_pinned_contract=False)
        assigned = stratified_math_split(cleaned, validation_size=1)["train"][0]
        training_row = build_verl_training_row(assigned)

        privileged_keys = {
            "opd_original_user_content",
            "opd_gold_cot",
            "opd_gold_answer",
        }
        self.assertTrue(privileged_keys.issubset(training_row["extra_info"]))
        self.assertEqual(
            training_row["extra_info"]["opd_original_user_content"],
            training_row["prompt"][0]["content"],
        )
        self.assertEqual(training_row["extra_info"]["opd_gold_cot"], assigned.gold_cot)
        self.assertNotIn("opd_gold_solution", training_row["extra_info"])
        generation = student_generation_payload(training_row)
        encoded = json.dumps(generation)
        for key in privileged_keys:
            self.assertNotIn(key, encoded)
        self.assertNotIn(assigned.gold_solution, encoded)
        self.assertNotIn("ground_truth", encoded)
        self.assertNotIn("reward_model", encoded)

    def test_evaluation_source_schemas(self):
        math500 = build_math500_evaluation_records(
            [
                {
                    "problem": "Compute 1+1.",
                    "solution": "It is \\boxed{2}.",
                    "answer": "2",
                    "subject": "Algebra",
                    "level": 1,
                    "unique_id": "test/algebra/1.json",
                }
            ],
            enforce_pinned_contract=False,
        )
        gsm8k = build_gsm8k_evaluation_records(
            [{"question": "Compute 1000+234.", "answer": "Work.\n#### 1,234"}],
            enforce_pinned_contract=False,
        )
        self.assertEqual(math500[0].gold_answer, "2")
        self.assertEqual(gsm8k[0].gold_answer, "1234")
        self.assertEqual(gsm8k[0].gold_solution, "Work.\n")
        self.assertNotIn("1234", json.dumps(gsm8k[0].student_record().to_dict()))


if __name__ == "__main__":
    unittest.main()
