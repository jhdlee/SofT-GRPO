from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from opd_tools.icl import (
    AIME2024_DATASET_REVISION,
    BENCHMARKS,
    COMMON_SAMPLE_SEEDS,
    CORE_CONDITIONS,
    EXPECTED_EXAMPLE_COUNTS,
    ICLEvaluationExample,
    MECHANISM_CONDITIONS,
    MODEL_LABELS,
    STUDY_BENCHMARKS,
    SOFTGRPO_MODEL_REVISION,
    STARTING_MODEL_REVISION,
    build_gsm8k_icl_examples,
    build_icl_matrix,
    build_shuffled_donor_map,
    canonicalize_problem_for_join,
    join_aime2024_icl_examples,
    join_aime2024_with_report,
    materialize_icl_dataset_from_rows,
    materialize_prompts,
    mechanism_subset_ids,
    model_source,
    render_icl_prompt,
    request_seed,
    validate_matrix_cell,
    verify_icl_dataset,
)


def example(index: int, benchmark: str = "math500", answer: str | None = None):
    return ICLEvaluationExample(
        example_id="%s-%03d" % (benchmark, index),
        benchmark=benchmark,
        source_index=index,
        question="Question %d with some explanatory text?" % index,
        gold_cot="Reasoning for problem %d without the final result." % index,
        gold_answer=str(index + 100 if answer is None else answer),
        subject="Algebra" if index < 4 else "Geometry",
        difficulty="Level %d" % (index % 2 + 1),
    )


def released_aime_row(problem: str, answer: str):
    return {
        "prompt": [{"from": "user", "value": problem}],
        "final_answer": answer,
    }


def aime_source_row(index: int, problem: str, answer: str):
    contest = "II" if index < 15 else "I"
    problem_number = index + 1 if index < 15 else index - 14
    return {
        "id": index,
        "problem": problem,
        "solution": "Detailed solution %d. Therefore $\\boxed{%s}$." % (index, answer),
        "answer": answer,
        "url": (
            "https://artofproblemsolving.com/wiki/index.php/"
            "2024_AIME_%s_Problems/Problem_%d" % (contest, problem_number)
        ),
        "year": "2024",
    }


def math500_row(index: int):
    return {
        "problem": "MATH problem %d" % index,
        "solution": "MATH work %d, so \\boxed{%d}." % (index, index + 1),
        "answer": str(index + 1),
        "subject": "Algebra",
        "level": index + 1,
        "unique_id": "math-%d" % index,
    }


class ICLDataTests(unittest.TestCase):
    def test_all_external_revisions_are_immutable_commits(self):
        for revision in (
            STARTING_MODEL_REVISION,
            SOFTGRPO_MODEL_REVISION,
            AIME2024_DATASET_REVISION,
        ):
            self.assertRegex(revision, r"^[0-9a-f]{40}$")
        self.assertEqual(MODEL_LABELS, ("starting", "softgrpo"))
        self.assertEqual(BENCHMARKS, ("math500", "gsm8k_test", "aime2024"))
        self.assertEqual(model_source("starting")["revision"], STARTING_MODEL_REVISION)
        self.assertEqual(
            model_source("softgrpo")["revision"], SOFTGRPO_MODEL_REVISION
        )
        changed = model_source("starting")
        changed["revision"] = "not-the-source"
        self.assertEqual(model_source("starting")["revision"], STARTING_MODEL_REVISION)

    def test_aime_join_is_bijective_canonicalizes_answers_and_strips_final_box(self):
        source = [
            aime_source_row(0, r"Compute $\\tfrac{1}{2}+\\tfrac{1}{2}$.", "7"),
            aime_source_row(1, "Second  problem.", "42"),
        ]
        released = [
            released_aime_row(r"Compute $\\frac{1}{2} + \\frac{1}{2}$.", "007"),
            released_aime_row("Second problem.", "042"),
        ]
        joined = join_aime2024_icl_examples(
            source, released, enforce_pinned_contract=False
        )
        self.assertEqual([item.gold_answer for item in joined], ["007", "042"])
        self.assertEqual(joined[0].question, released[0]["prompt"][0]["value"])
        self.assertNotIn("\\boxed", joined[0].gold_cot)
        self.assertIn("Detailed solution", joined[0].gold_cot)
        self.assertEqual(
            canonicalize_problem_for_join(source[0]["problem"]),
            canonicalize_problem_for_join(released[0]["prompt"][0]["value"]),
        )

    def test_aime_join_rejects_nonbijective_questions_and_wrong_solution(self):
        source = [aime_source_row(0, "Problem A", "7"), aime_source_row(1, "Problem B", "8")]
        released = [released_aime_row("Problem A", "7"), released_aime_row("Missing", "8")]
        with self.assertRaisesRegex(ValueError, "AIME"):
            join_aime2024_icl_examples(source, released, enforce_pinned_contract=False)

        released[1] = released_aime_row("Problem B", "8")
        source[1]["solution"] = "This incorrectly ends in \\boxed{9}."
        with self.assertRaisesRegex(ValueError, "solution"):
            join_aime2024_icl_examples(source, released, enforce_pinned_contract=False)

    def test_aime_join_accepts_conservative_answer_constrained_restatement(self):
        source_question = (
            "Every morning Aya goes for a 9-kilometer-long walk and stops at a coffee "
            "shop afterwards. One day she walks at s kilometers per hour."
        )
        released_question = (
            "Every morning, Aya does a 9 kilometer walk, and then finishes at the coffee "
            "shop. One day, she walks at s kilometers per hour."
        )
        joined, report = join_aime2024_with_report(
            [aime_source_row(0, source_question, "204")],
            [released_aime_row(released_question, "204")],
            enforce_pinned_contract=False,
        )
        self.assertEqual(len(joined), 1)
        self.assertFalse(report["rows"][0]["exact_normalized_match"])
        self.assertGreaterEqual(report["rows"][0]["similarity"], 0.60)

    def test_aime_answer_extraction_accepts_framebox_and_strict_terminal_fallback(self):
        released = [released_aime_row("A shared AIME problem statement", "7")]
        framed = aime_source_row(0, "A shared AIME problem statement", "7")
        framed["solution"] = (
            "First proof ending in $\\framebox{007}$. A later alternate proof is prose."
        )
        _, report = join_aime2024_with_report(
            [framed], released, enforce_pinned_contract=False
        )
        self.assertEqual(report["rows"][0]["answer_extraction"]["mode"], "framebox")

        terminal = aime_source_row(0, "A shared AIME problem statement", "7")
        terminal["solution"] = "A proof with a final calculation $2+5=7$."
        joined, report = join_aime2024_with_report(
            [terminal], released, enforce_pinned_contract=False
        )
        self.assertEqual(
            report["rows"][0]["answer_extraction"]["mode"],
            "terminal_math_expression",
        )
        self.assertNotIn("2+5=7", joined[0].gold_cot)

    def test_released_row_22_correction_is_digest_guarded_and_audited(self):
        released_path = (
            Path(__file__).resolve().parents[2]
            / "Soft-Thinking+noise+loss-main"
            / "datasets"
            / "aime2024.json"
        )
        released = json.loads(released_path.read_text(encoding="utf-8"))
        source = []
        for index, row in enumerate(released):
            answer = "197" if index == 22 else row["final_answer"]
            source.append(
                aime_source_row(index, row["prompt"][0]["value"], answer)
            )
        joined, report = join_aime2024_with_report(
            source, released, enforce_pinned_contract=False
        )
        self.assertEqual(len(joined), 30)
        self.assertEqual(joined[22].gold_answer, "197")
        self.assertEqual(len(report["corrections"]), 1)
        correction = report["corrections"][0]
        self.assertEqual(correction["released_index"], 22)
        self.assertEqual(correction["expected_raw_answer"], "480")
        self.assertEqual(correction["corrected_answer"], "197")
        self.assertRegex(correction["canonical_question_sha256"], r"^[0-9a-f]{64}$")

    def test_gsm8k_subset_is_seed42_deterministic_and_preserves_rationale(self):
        rows = [
            {
                "question": "GSM question %d" % index,
                "answer": "Line one for %d.\nLine two.\n#### %d" % (index, index + 1),
            }
            for index in range(520)
        ]
        first = build_gsm8k_icl_examples(rows, enforce_pinned_contract=False)
        second = build_gsm8k_icl_examples(rows, enforce_pinned_contract=False)
        self.assertEqual(len(first), 512)
        self.assertEqual(first, second)
        self.assertEqual(len({item.example_id for item in first}), 512)
        self.assertTrue(all("Line two.\n" in item.gold_cot for item in first))

    def test_shuffled_pairing_is_deterministic_bijective_and_answer_mismatched(self):
        examples = tuple(example(index) for index in range(8))
        first = build_shuffled_donor_map(examples)
        second = build_shuffled_donor_map(tuple(reversed(examples)))
        self.assertEqual(first, second)
        self.assertEqual(set(first), {item.example_id for item in examples})
        self.assertEqual(set(first.values()), set(first))
        by_id = {item.example_id: item for item in examples}
        for target_id, donor_id in first.items():
            self.assertNotEqual(target_id, donor_id)
            self.assertNotEqual(by_id[target_id].gold_answer, by_id[donor_id].gold_answer)

    def test_mechanism_subset_is_fixed_and_caps_aime_at_30(self):
        math = tuple(example(index) for index in range(200))
        self.assertEqual(mechanism_subset_ids(math), mechanism_subset_ids(tuple(math)))
        self.assertEqual(len(mechanism_subset_ids(math)), 128)
        aime = tuple(example(index, "aime2024", "%03d" % index) for index in range(30))
        self.assertEqual(len(mechanism_subset_ids(aime)), 30)

    def test_atomic_materialization_is_authenticated_and_idempotent(self):
        math = [math500_row(index) for index in range(2)]
        gsm = [
            {"question": "GSM %d" % index, "answer": "Work %d.\n#### %d" % (index, index + 3)}
            for index in range(2)
        ]
        aime = [aime_source_row(index, "AIME problem %d" % index, str(index + 7)) for index in range(2)]
        released = [
            released_aime_row("AIME problem %d" % index, str(index + 7))
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "icl-data"
            manifest = materialize_icl_dataset_from_rows(
                output,
                math,
                gsm,
                aime,
                released,
                enforce_pinned_contract=False,
            )
            self.assertEqual(manifest["counts"], {name: 2 for name in BENCHMARKS})
            self.assertEqual(
                verify_icl_dataset(output, enforce_pinned_contract=False), manifest
            )
            self.assertEqual(
                materialize_icl_dataset_from_rows(
                    output, [], [], [], [], enforce_pinned_contract=False
                ),
                manifest,
            )
            with self.assertRaisesRegex(ValueError, "pinned contract"):
                verify_icl_dataset(output)
            with (output / "shuffled_pairs.json").open("a", encoding="utf-8") as stream:
                stream.write("tamper")
            with self.assertRaisesRegex(ValueError, "authentication"):
                verify_icl_dataset(output, enforce_pinned_contract=False)


class ICLPromptTests(unittest.TestCase):
    def setUp(self):
        self.target = ICLEvaluationExample(
            example_id="math-a",
            benchmark="math500",
            source_index=0,
            question="What is six plus six?",
            gold_cot="Adding 6 and 6 gives 12. Thus the result 12 follows.",
            gold_answer="12",
            subject="Algebra",
            difficulty="Level 1",
        )
        self.donor = ICLEvaluationExample(
            example_id="math-b",
            benchmark="math500",
            source_index=1,
            question="What is seven plus six?",
            gold_cot="Adding the values gives 13.",
            gold_answer="13",
            subject="Algebra",
            difficulty="Level 1",
        )

    def test_no_demo_is_exact_and_generation_payload_has_no_gold_fields(self):
        prompt = render_icl_prompt(self.target, "no_demo")
        self.assertEqual(
            prompt.user_content,
            "What is six plus six? Let's think step by step and output the final answer within \\boxed{}.",
        )
        payload = prompt.generation_payload()
        encoded = json.dumps(payload)
        self.assertNotIn("gold_", encoded)
        self.assertNotIn(self.target.gold_cot, encoded)
        self.assertNotIn("The final answer is", encoded)

    def test_sdft_and_sdpg_matched_prompts_are_exact(self):
        original = (
            "What is six plus six? Let's think step by step and output the final answer "
            "within \\boxed{}."
        )
        sdft = render_icl_prompt(self.target, "sdft_matched")
        self.assertEqual(
            sdft.user_content,
            original
            + "\n\nThis is an example for a response to the question:\n"
            + self.target.gold_cot
            + "\nThe final answer is: \\boxed{12}\n\n"
            + "Now answer with a response of your own, including the thinking process.",
        )
        sdpg = render_icl_prompt(self.target, "sdpg_matched")
        self.assertEqual(
            sdpg.user_content,
            original
            + "\n\n[Hint] The correct answer is 12. A common way to solve this is:\n"
            + self.target.gold_cot
            + "\n\n[Instruction] If possible, derive the answer 12 using an alternative, "
            + "equally rigorous mathematical approach to the one provided above. Otherwise, "
            + "improve the given reasoning by making it clearer, more complete, and logically "
            + "sound. Do NOT state that you were given the answer or reference.",
        )

    def test_shuffled_prompts_require_a_distinct_answer_mismatched_donor(self):
        prompt = render_icl_prompt(
            self.target, "sdft_shuffled", shuffled_donor=self.donor
        )
        self.assertEqual(prompt.demonstration_example_id, self.donor.example_id)
        self.assertIn(self.donor.gold_cot, prompt.user_content)
        self.assertIn("\\boxed{13}", prompt.user_content)
        self.assertNotIn(self.target.gold_cot, prompt.user_content)
        with self.assertRaisesRegex(ValueError, "require a donor"):
            render_icl_prompt(self.target, "sdpg_shuffled")
        with self.assertRaisesRegex(ValueError, "different normalized answer"):
            render_icl_prompt(
                self.target,
                "sdpg_shuffled",
                shuffled_donor=ICLEvaluationExample(
                    example_id="same-answer",
                    benchmark="math500",
                    source_index=2,
                    question="Other",
                    gold_cot="Other work",
                    gold_answer="12",
                ),
            )

    def test_answer_only_and_rationale_only_controls(self):
        for family in ("sdft", "sdpg"):
            answer_only = render_icl_prompt(self.target, family + "_answer_only")
            self.assertIn("[Reasoning omitted.]", answer_only.user_content)
            self.assertNotIn(self.target.gold_cot, answer_only.user_content)
            self.assertIn("12", answer_only.user_content)

            rationale_only = render_icl_prompt(self.target, family + "_rationale_only")
            self.assertIn("[MASKED]", rationale_only.user_content)
            self.assertNotIn("12", rationale_only.user_content)
            self.assertNotIn("\\boxed{12}", rationale_only.user_content)

    def test_materialize_prompts_uses_registered_shuffled_map(self):
        examples = (self.target, self.donor)
        pairs = {self.target.example_id: self.donor.example_id, self.donor.example_id: self.target.example_id}
        prompts = materialize_prompts(examples, pairs, "sdft_shuffled")
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0].demonstration_example_id, self.donor.example_id)


class ICLMatrixTests(unittest.TestCase):
    def test_production_matrix_contains_only_the_18_registered_cells(self):
        matrix = build_icl_matrix()
        self.assertEqual(len(matrix), 18)
        self.assertTrue(all(cell.sample_count == 8 for cell in matrix))
        self.assertEqual({cell.benchmark for cell in matrix}, set(STUDY_BENCHMARKS))
        self.assertEqual({cell.condition for cell in matrix}, set(CORE_CONDITIONS))
        self.assertNotIn("gsm8k_test", {cell.benchmark for cell in matrix})
        self.assertFalse(any("shuffled" in cell.condition for cell in matrix))
        self.assertFalse(
            any(cell.condition in MECHANISM_CONDITIONS for cell in matrix)
        )
        trajectories_by_run = {}
        for model, mode in (
            ("starting", "native_soft"),
            ("starting", "hard_token"),
            ("softgrpo", "native_soft"),
        ):
            trajectories_by_run[(model, mode)] = sum(
                cell.example_count * cell.sample_count
                for cell in matrix
                if cell.model_label == model and cell.inference_mode == mode
            )
        self.assertEqual(set(trajectories_by_run.values()), {12_720})
        self.assertFalse(
            any(
                cell.model_label == "softgrpo" and cell.inference_mode == "hard_token"
                for cell in matrix
            )
        )
        self.assertEqual(
            {
                cell.condition
                for cell in matrix
                if cell.model_label == "starting" and cell.inference_mode == "hard_token"
            },
            set(CORE_CONDITIONS),
        )
        for cell in matrix:
            self.assertEqual(
                cell.example_count, EXPECTED_EXAMPLE_COUNTS[cell.benchmark]
            )

    def test_smoke_uses_16_examples_and_two_samples(self):
        matrix = build_icl_matrix(smoke=True)
        self.assertTrue(all(cell.example_count <= 16 for cell in matrix))
        self.assertTrue(all(cell.sample_count == 2 for cell in matrix))

    def test_matrix_validator_rejects_prohibited_hard_token_cells(self):
        with self.assertRaisesRegex(ValueError, "post-trained"):
            validate_matrix_cell("softgrpo", "hard_token", "no_demo", "math500")
        with self.assertRaisesRegex(ValueError, "unregistered ICL prompt condition"):
            validate_matrix_cell(
                "starting", "hard_token", MECHANISM_CONDITIONS[0], "math500"
            )
        with self.assertRaisesRegex(ValueError, "unregistered ICL prompt condition"):
            validate_matrix_cell(
                "starting", "native_soft", "sdft_shuffled", "math500"
            )
        with self.assertRaisesRegex(ValueError, "unregistered ICL benchmark"):
            validate_matrix_cell(
                "starting", "native_soft", "no_demo", "gsm8k_test"
            )

    def test_request_seeds_are_common_across_conditions_and_unique_by_sample(self):
        seeds = [request_seed("math500", "one", index) for index in range(8)]
        self.assertEqual(len(set(seeds)), 8)
        self.assertEqual(COMMON_SAMPLE_SEEDS, tuple(range(11, 19)))
        self.assertEqual(request_seed("math500", "one", 0), seeds[0])


if __name__ == "__main__":
    unittest.main()
