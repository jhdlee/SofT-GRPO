import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from opd_tools.evaluation import (
    COMMON_GENERATION_SEEDS,
    EVALUATION_PROTOCOL,
    EVALUATION_SCHEMA_VERSION,
    GenerationRecord,
    evaluation_request_seed,
)
from opd_tools.generate_eval import (
    EVALUATION_SAMPLING_PROTOCOLS,
    GENERATION_IMPLEMENTATION,
    _atomic_write,
    _canonical_json,
    _stable_wandb_id,
    _write_shard,
    expected_engine_mode,
    expected_sampling_source,
)
from opd_tools.graders import Grade
from opd_tools.constants import MODEL_ID, MODEL_REVISION, SOFTGRPO_UPSTREAM_COMMIT
from opd_tools import paper_anchor


REGRADER_ENV = {
    "OPD_PAPER_REGRADE_PARENT_COMMIT": "e" * 40,
    "OPD_PAPER_REGRADE_SUBMODULE_COMMIT": "f" * 40,
}


def _locked_distribution_version(distribution: str) -> str:
    return paper_anchor.UPSTREAM_GRADER_DEPENDENCY_VERSIONS[distribution]


def _record(*, example_id: str, sample_index: int, seed: int) -> GenerationRecord:
    return GenerationRecord(
        model_label="initial",
        benchmark="math500",
        example_id=example_id,
        inference_mode="native_soft",
        sample_index=sample_index,
        generation_seed=seed,
        request_seed=evaluation_request_seed(seed, "math500", example_id),
        response="reasoning</think> The answer is \\boxed{1}",
        response_token_count=4,
        finish_reason="stop",
        capped=False,
        latent_token_count=3,
        hard_token_count=1,
        close_tag=True,
        soft_to_hard=True,
        all_soft=False,
        mixture_entropy_mean=0.2,
        top1_weight_mean=0.9,
        soft_hard_agreement=1.0,
        gold_answer="1",
    )


def _generation_manifest(model_path: Path) -> dict:
    model = {
        "path": str(model_path.resolve()),
        "tree_sha256": "a" * 64,
        "files": [],
    }
    result = {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "softgrpo_upstream_commit": SOFTGRPO_UPSTREAM_COMMIT,
        "generation_implementation": GENERATION_IMPLEMENTATION,
        "sampling_source": expected_sampling_source("native_soft", "released_anchor"),
        "engine_mode": expected_engine_mode("native_soft"),
        "model_label": "initial",
        "model": model,
        "mode": "native_soft",
        "benchmarks": ["math500"],
        "generation_seeds": [11, 12],
        "sampling_protocol": "released_anchor",
        "sampling": EVALUATION_SAMPLING_PROTOCOLS["released_anchor"],
        "parallelism": dict(paper_anchor.PAPER_ANCHOR_PARALLELISM),
        "batch_size": 64,
        "max_running_requests": 16,
        "gpu_memory_utilization": 0.8,
        "context_length": 33_000,
        "data_manifest_content_sha256": "b" * 64,
        "cuda_visible_devices_source": "slurm",
        "parent_commit": "c" * 40,
        "fork_commit": "d" * 40,
    }
    result["wandb_run_id"] = _stable_wandb_id(result)
    return result


class PaperAnchorAuthenticationTests(unittest.TestCase):
    def _materialize_small_anchor(self, root: Path) -> dict:
        model_path = root / "model"
        model_path.mkdir()
        manifest = _generation_manifest(model_path)
        generation_path = (
            root / "raw/initial/native_soft/generation_manifest.json"
        )
        _atomic_write(generation_path, _canonical_json(manifest))
        shards = []
        for sample_index, seed in enumerate((11, 12)):
            data_path = (
                root
                / "raw/initial/native_soft/math500"
                / ("seed_%d.jsonl" % seed)
            )
            sidecar = _write_shard(
                data_path,
                [
                    _record(example_id="math500-0", sample_index=sample_index, seed=seed),
                    _record(example_id="math500-1", sample_index=sample_index, seed=seed),
                ],
            )
            shards.append(
                {
                    "path": data_path.relative_to(root).as_posix(),
                    "size": sidecar["size"],
                    "sha256": sidecar["sha256"],
                    "row_count": sidecar["row_count"],
                }
            )
        completion = {
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "generation_manifest_sha256": paper_anchor.file_sha256(generation_path),
            "model_label": "initial",
            "mode": "native_soft",
            "benchmarks": ["math500"],
            "sampling_protocol": "released_anchor",
            "shards_committed": 2,
            "expected_shards": 2,
            "rows_committed": 4,
            "shards": shards,
        }
        _atomic_write(
            root / "raw/initial/native_soft/completion.json",
            _canonical_json(completion),
        )
        return manifest

    def test_authenticates_only_exact_base_native_soft_math500_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._materialize_small_anchor(root)
            identity = {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "inventory_sha256": "c" * 64,
                "tree_sha256": "a" * 64,
            }
            with (
                patch.object(paper_anchor, "COMMON_GENERATION_SEEDS", (11, 12)),
                patch.object(paper_anchor, "PAPER_ANCHOR_SAMPLE_COUNT", 2),
                patch.object(paper_anchor, "PAPER_ANCHOR_EXAMPLE_COUNT", 2),
                patch.object(
                    paper_anchor,
                    "_authenticate_starting_model",
                    return_value=identity,
                ),
            ):
                authenticated = paper_anchor.authenticate_input(root)
                self.assertEqual(authenticated["model"], identity)
                self.assertEqual(authenticated["generation_seeds"], [11, 12])
                self.assertEqual(len(authenticated["shards"]), 2)

                # A seemingly useful extra cell is still forbidden: this report
                # has one paper-anchor cell, not an open-ended evaluation tree.
                extra = root / "raw/softgrpo/native_soft/math500/seed_11.jsonl"
                extra.parent.mkdir(parents=True)
                extra.write_text("{}\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "inventory differs"):
                    paper_anchor.authenticate_input(root)

            self.assertEqual(manifest["sampling_protocol"], "released_anchor")

    def test_rejects_a_training_matched_generation_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._materialize_small_anchor(root)
            manifest["sampling_protocol"] = "training_matched"
            path = root / "raw/initial/native_soft/generation_manifest.json"
            _atomic_write(path, _canonical_json(manifest))
            with (
                patch.object(paper_anchor, "COMMON_GENERATION_SEEDS", (11, 12)),
                patch.object(paper_anchor, "PAPER_ANCHOR_SAMPLE_COUNT", 2),
                patch.object(paper_anchor, "PAPER_ANCHOR_EXAMPLE_COUNT", 2),
                patch.object(
                    paper_anchor,
                    "_authenticate_starting_model",
                    return_value={
                        "id": MODEL_ID,
                        "revision": MODEL_REVISION,
                        "inventory_sha256": "c" * 64,
                        "tree_sha256": "a" * 64,
                    },
                ),
            ):
                with self.assertRaisesRegex(ValueError, "sampling_protocol"):
                    paper_anchor.authenticate_input(root)

    def test_rejects_completion_that_does_not_bind_the_exact_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._materialize_small_anchor(root)
            path = root / "raw/initial/native_soft/completion.json"
            completion = json.loads(path.read_text(encoding="utf-8"))
            completion["rows_committed"] -= 1
            _atomic_write(path, _canonical_json(completion))
            with (
                patch.object(paper_anchor, "COMMON_GENERATION_SEEDS", (11, 12)),
                patch.object(paper_anchor, "PAPER_ANCHOR_SAMPLE_COUNT", 2),
                patch.object(paper_anchor, "PAPER_ANCHOR_EXAMPLE_COUNT", 2),
                patch.object(
                    paper_anchor,
                    "_authenticate_starting_model",
                    return_value={
                        "id": MODEL_ID,
                        "revision": MODEL_REVISION,
                        "inventory_sha256": "c" * 64,
                        "tree_sha256": "a" * 64,
                    },
                ),
            ):
                with self.assertRaisesRegex(ValueError, "completion record"):
                    paper_anchor.authenticate_input(root)

    def test_starting_model_authentication_checks_pinned_identity_and_tree(self):
        model = {
            "path": "/tmp/paper-anchor-model",
            "tree_sha256": "a" * 64,
            "files": [],
        }
        asset = {
            "model": {
                "id": MODEL_ID,
                "requested_revision": MODEL_REVISION,
                "resolved_revision": MODEL_REVISION,
            },
            "inventory_sha256": "b" * 64,
        }
        with (
            patch.object(paper_anchor, "verify_model_snapshot", return_value=asset),
            patch.object(paper_anchor, "_tree_fingerprint", return_value=model),
        ):
            identity = paper_anchor._authenticate_starting_model(model)
        self.assertEqual(identity["id"], MODEL_ID)
        self.assertEqual(identity["revision"], MODEL_REVISION)

        wrong = dict(asset)
        wrong["model"] = dict(asset["model"], resolved_revision="wrong")
        with (
            patch.object(paper_anchor, "verify_model_snapshot", return_value=wrong),
            patch.object(paper_anchor, "_tree_fingerprint", return_value=model),
        ):
            with self.assertRaisesRegex(ValueError, "pinned starting checkpoint"):
                paper_anchor._authenticate_starting_model(model)


class PaperAnchorStatisticsTests(unittest.TestCase):
    @staticmethod
    def _grouped(example_count: int = 2) -> dict:
        grouped = {}
        for example_index in range(example_count):
            rows = {}
            for seed in COMMON_GENERATION_SEEDS:
                rows[seed] = {
                    "example_id": "math500-%d" % example_index,
                    "generation_seed": seed,
                    "correct": example_index == 0,
                    "upstream_rule_correct": example_index == 0,
                    "upstream_extracted_answer": "1" if example_index == 0 else "0",
                    "response_length": 100.0 + example_index,
                    "latent_length": 90.0,
                    "hard_answer_length": 10.0 + example_index,
                    "capped": 0.0,
                    "close_tag": 1.0,
                    "soft_to_hard": 1.0,
                    "all_soft": 0.0,
                    "mixture_entropy": 0.2,
                    "top1_weight": 0.9,
                    "soft_hard_agreement": 1.0,
                }
            grouped["math500-%d" % example_index] = rows
        return grouped

    def test_math_verify_scoring_uses_the_full_response_grader(self):
        record = _record(example_id="math500-0", sample_index=0, seed=11)
        grade = Grade(True, None, None, "1")
        with (
            patch.object(
                paper_anchor, "math_verify_full_response_grade", return_value=grade
            ) as scorer,
            patch.object(
                paper_anchor,
                "upstream_math500_rule_judge",
                return_value=(False, "0"),
            ) as upstream_scorer,
        ):
            row = paper_anchor._score_record(record.to_dict())
        scorer.assert_called_once_with(record.response, record.gold_answer)
        upstream_scorer.assert_called_once_with(record.response, record.gold_answer)
        self.assertTrue(row["correct"])
        self.assertFalse(row["upstream_rule_correct"])
        self.assertEqual(row["upstream_extracted_answer"], "0")

    def test_upstream_rule_judge_calls_the_authenticated_released_api(self):
        class Evaluator:
            def __init__(self):
                self.calls = []

            def rule_judge(self, response, gold_answer, finish_generation):
                self.calls.append((response, gold_answer, finish_generation))
                return True, ["parsed"]

        evaluator = Evaluator()
        with patch.object(
            paper_anchor, "_upstream_math500_evaluator", return_value=evaluator
        ):
            correct, extracted = paper_anchor.upstream_math500_rule_judge(
                "work \\boxed{1}", "1"
            )
        self.assertTrue(correct)
        self.assertEqual(extracted, "['parsed']")
        self.assertEqual(evaluator.calls, [("work \\boxed{1}", "1", False)])

    def test_upstream_grader_source_is_pinned_to_the_released_commit(self):
        with patch.object(
            paper_anchor,
            "distribution_version",
            side_effect=_locked_distribution_version,
        ):
            provenance = paper_anchor.upstream_math500_grader_provenance()
        self.assertEqual(
            provenance,
            {
                "path": "Soft-Thinking+noise+loss-main/matheval.py",
                "sha256": paper_anchor.UPSTREAM_MATH500_GRADER_SHA256,
                "api": "MATH500Evaluator.rule_judge",
                "upstream_commit": SOFTGRPO_UPSTREAM_COMMIT,
                "dependencies": {
                    "math-verify": "0.8.0",
                    "latex2sympy2-extended": "1.10.2",
                },
            },
        )

    def test_upstream_grader_dependency_versions_fail_closed(self):
        with patch.object(
            paper_anchor,
            "distribution_version",
            side_effect=_locked_distribution_version,
        ):
            self.assertEqual(
                paper_anchor.upstream_grader_dependency_versions(),
                paper_anchor.UPSTREAM_GRADER_DEPENDENCY_VERSIONS,
            )
        with patch.object(
            paper_anchor, "distribution_version", return_value="wrong"
        ):
            with self.assertRaisesRegex(RuntimeError, "requires math-verify==0.8.0"):
                paper_anchor.upstream_grader_dependency_versions()

    def test_regrader_source_requires_commits_and_hashes_implementation(self):
        identity = paper_anchor.required_regrader_source_identity(REGRADER_ENV)
        self.assertEqual(identity["parent_commit"], "e" * 40)
        self.assertEqual(identity["submodule_commit"], "f" * 40)
        self.assertEqual(
            identity["implementation_path"], "opd_tools/paper_anchor.py"
        )
        self.assertEqual(
            identity["implementation_sha256"],
            paper_anchor.file_sha256(Path(paper_anchor.__file__)),
        )
        legacy = paper_anchor.required_regrader_source_identity(
            {
                "OPD_PARENT_COMMIT": "a" * 40,
                "OPD_SUBMODULE_COMMIT": "b" * 40,
            }
        )
        self.assertEqual(legacy["parent_commit"], "a" * 40)
        self.assertEqual(legacy["submodule_commit"], "b" * 40)
        with self.assertRaisesRegex(RuntimeError, "is required"):
            paper_anchor.required_regrader_source_identity({})
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            paper_anchor.required_regrader_source_identity(
                {
                    **REGRADER_ENV,
                    "OPD_PARENT_COMMIT": "a" * 40,
                }
            )

    def test_reports_paper_mean32_aliases_pass_metrics_and_boundary_diagnostics(self):
        grouped = self._grouped()
        with (
            patch.object(paper_anchor, "PAPER_ANCHOR_EXAMPLE_COUNT", 2),
            patch.object(
                paper_anchor,
                "distribution_version",
                side_effect=_locked_distribution_version,
            ),
        ):
            summary, rows, wandb_metrics = paper_anchor.aggregate(grouped)
        metrics = summary["math_verify"]["metrics"]
        upstream_metrics = summary["upstream_rule_judge"]["metrics"]
        self.assertEqual(metrics["mean_at_32"]["estimate"], 0.5)
        self.assertEqual(metrics["pass_at_1"]["estimate"], 0.5)
        self.assertEqual(metrics["pass_at_1"]["alias_of"], "mean_at_32")
        self.assertEqual(metrics["sample_accuracy"]["estimate"], 0.5)
        self.assertEqual(metrics["pass_at_8"]["estimate"], 0.5)
        self.assertEqual(metrics["pass_at_16"]["estimate"], 0.5)
        self.assertEqual(metrics["pass_at_32"]["estimate"], 0.5)
        self.assertEqual(metrics["mean_at_32"]["resamples"], 10_000)
        self.assertEqual(upstream_metrics["pass_at_1"]["estimate"], 0.5)
        self.assertEqual(
            summary["upstream_rule_judge"]["provenance"]["sha256"],
            paper_anchor.UPSTREAM_MATH500_GRADER_SHA256,
        )
        self.assertTrue(summary["boundary_gate"]["valid"])
        self.assertEqual(
            summary["native_soft_diagnostics"]["soft_to_hard_mean"]["estimate"],
            1.0,
        )
        self.assertTrue(rows)
        self.assertEqual(
            wandb_metrics["paper_anchor/math500/math_verify/pass_at_1"], 0.5
        )
        self.assertEqual(
            wandb_metrics[
                "paper_anchor/math500/upstream_rule_judge/pass_at_1"
            ],
            0.5,
        )

    def test_upstream_rule_reports_boundary_sensitivity_without_replacing_score(self):
        grouped = self._grouped()
        grouped["math500-0"][COMMON_GENERATION_SEEDS[0]]["capped"] = 1.0
        with (
            patch.object(paper_anchor, "PAPER_ANCHOR_EXAMPLE_COUNT", 2),
            patch.object(
                paper_anchor,
                "distribution_version",
                side_effect=_locked_distribution_version,
            ),
        ):
            summary, rows, wandb_metrics = paper_anchor.aggregate(grouped)
        upstream = summary["upstream_rule_judge"]
        self.assertEqual(upstream["metrics"]["pass_at_1"]["estimate"], 0.5)
        sensitivity = upstream["boundary_sensitivity"]
        self.assertEqual(sensitivity["correct_invalid_response_count"], 1)
        self.assertLess(
            sensitivity["invalid_as_incorrect_pass_at_1"]["estimate"], 0.5
        )
        self.assertEqual(
            sensitivity["valid_boundary_only_sample_accuracy"]["estimate"], 0.5
        )
        self.assertTrue(
            any(row["category"] == "upstream_rule_judge_boundary" for row in rows)
        )
        self.assertEqual(
            wandb_metrics[
                "paper_anchor/math500/upstream_rule_judge/"
                "correct_invalid_response_count"
            ],
            1.0,
        )

    def test_bootstrap_is_deterministic_and_requires_ten_thousand_resamples(self):
        values = {"a": 0.0, "b": 1.0, "c": 1.0, "d": 0.0}
        self.assertEqual(
            paper_anchor.bootstrap_mean(values), paper_anchor.bootstrap_mean(values)
        )
        with self.assertRaisesRegex(ValueError, "10,000"):
            paper_anchor.bootstrap_mean(values, resamples=999)

    def test_report_writes_atomic_compact_files_and_uses_stable_wandb_id(self):
        input_manifest = {
            "model": {"tree_sha256": "a" * 64},
            "data_manifest_content_sha256": "b" * 64,
            "generation_manifest": {"sha256": "c" * 64},
            "shards": [{"sha256": "d" * 64}],
        }
        summary = {"metric": 0.5}
        rows = [{"metric": "mean_at_32", "estimate": 0.5}]
        metrics = {"paper_anchor/math500/math_verify/mean_at_32": 0.5}

        def publish(**kwargs):
            return kwargs["report_manifest"]["wandb_run_id"]

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(paper_anchor, "_publish_wandb", side_effect=publish),
            patch.object(
                paper_anchor,
                "distribution_version",
                side_effect=_locked_distribution_version,
            ),
            patch.dict(os.environ, REGRADER_ENV, clear=True),
        ):
            legacy_paths = [
                Path(directory) / "paper_anchor_summary.json",
                Path(directory) / "paper_anchor_metrics.csv",
                Path(directory) / "paper_anchor_manifest.json",
            ]
            for path in legacy_paths:
                path.write_text("legacy-authenticated-report\n", encoding="utf-8")
            first = paper_anchor.write_report(
                output_dir=Path(directory),
                input_manifest=input_manifest,
                summary=summary,
                metric_rows=rows,
                wandb_metrics=metrics,
            )
            second = paper_anchor.write_report(
                output_dir=Path(directory),
                input_manifest=input_manifest,
                summary=summary,
                metric_rows=rows,
                wandb_metrics=metrics,
            )
            self.assertEqual(first["wandb_run_id"], second["wandb_run_id"])
            for path in legacy_paths:
                self.assertEqual(
                    path.read_text(encoding="utf-8"),
                    "legacy-authenticated-report\n",
                )
            self.assertTrue(
                (
                    Path(directory)
                    / "paper_anchor_upstream_rule_summary.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    Path(directory)
                    / "paper_anchor_upstream_rule_metrics.csv"
                ).is_file()
            )
            stored = json.loads(
                (
                    Path(directory)
                    / "paper_anchor_upstream_rule_manifest.json"
                ).read_text()
            )
            self.assertEqual(stored, second)
            self.assertEqual(stored["report_variant"], "upstream-rule-judge-v1")
            self.assertIn("upstream_rule_judge", stored)
            self.assertEqual(stored["regrader_source"]["parent_commit"], "e" * 40)
            self.assertEqual(
                stored["regrader_source"]["submodule_commit"], "f" * 40
            )
            self.assertEqual(
                stored["regrader_source"]["implementation_sha256"],
                paper_anchor.file_sha256(Path(paper_anchor.__file__)),
            )
            self.assertTrue(
                stored["wandb_run_id"].startswith(
                    "paper-anchor-base-upstream-rule-"
                )
            )

    def test_wandb_identity_changes_with_regrader_commit_or_implementation(self):
        input_manifest = {
            "model": {"tree_sha256": "a" * 64},
            "data_manifest_content_sha256": "b" * 64,
            "generation_manifest": {"sha256": "c" * 64},
            "shards": [{"sha256": "d" * 64}],
        }
        grader = {
            "path": paper_anchor.UPSTREAM_MATH500_GRADER_PATH,
            "sha256": paper_anchor.UPSTREAM_MATH500_GRADER_SHA256,
            "api": paper_anchor.UPSTREAM_MATH500_GRADER_API,
            "upstream_commit": SOFTGRPO_UPSTREAM_COMMIT,
            "dependencies": dict(
                paper_anchor.UPSTREAM_GRADER_DEPENDENCY_VERSIONS
            ),
        }
        source = {
            "parent_commit": "a" * 40,
            "submodule_commit": "b" * 40,
            "implementation_path": paper_anchor.PAPER_ANCHOR_IMPLEMENTATION_PATH,
            "implementation_sha256": "c" * 64,
        }
        baseline = paper_anchor.stable_wandb_run_id(
            input_manifest, source, grader
        )
        changed_commit = paper_anchor.stable_wandb_run_id(
            input_manifest,
            {**source, "submodule_commit": "d" * 40},
            grader,
        )
        changed_implementation = paper_anchor.stable_wandb_run_id(
            input_manifest,
            {**source, "implementation_sha256": "e" * 64},
            grader,
        )
        changed_dependency = paper_anchor.stable_wandb_run_id(
            input_manifest,
            source,
            {
                **grader,
                "dependencies": {
                    **grader["dependencies"],
                    "math-verify": "different",
                },
            },
        )
        self.assertEqual(
            len({baseline, changed_commit, changed_implementation, changed_dependency}),
            4,
        )


if __name__ == "__main__":
    unittest.main()
