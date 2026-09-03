from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from opd_tools.prepare import (
    OUTPUT_FILENAMES,
    materialize_from_rows,
    verify_materialized_data,
)


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows

    @classmethod
    def from_list(cls, rows):
        return cls(rows)

    def to_parquet(self, path):
        # The unit test validates orchestration, atomic publication, logical
        # hashes, and byte authentication. Production uses datasets/pyarrow.
        Path(path).write_text(
            json.dumps(self.rows, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )


def math_row(index):
    return {
        "problem": "Question %d" % index,
        "level": "Level %d" % (index % 2 + 1),
        "solution": "Work, then \\boxed{%d}." % index,
        "type": "Algebra" if index < 6 else "Geometry",
    }


def released_row(question, answer):
    return {
        "prompt": [{"from": "user", "value": question}],
        "final_answer": answer,
    }


class PrepareTest(unittest.TestCase):
    def test_atomic_materialization_and_idempotent_verification(self):
        fake_datasets = types.SimpleNamespace(Dataset=FakeDataset)
        released = {
            "aime2024": [released_row("AIME24?", "1")],
            "aime2025": [released_row("AIME25?", "2")],
            "amc23": [released_row("AMC23?", "3")],
        }
        math500 = [
            {
                "problem": "MATH-500?",
                "solution": "Work. \\boxed{4}.",
                "answer": "4",
                "subject": "Algebra",
                "level": 1,
                "unique_id": "test/algebra/4.json",
            }
        ]
        gsm8k = [{"question": "GSM8K?", "answer": "Work.\n#### 5"}]
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "prepared"
            with patch.dict(sys.modules, {"datasets": fake_datasets}):
                manifest = materialize_from_rows(
                    output,
                    [math_row(index) for index in range(12)],
                    math500,
                    gsm8k,
                    released,
                    validation_size=3,
                    split_seed=42,
                    enforce_pinned_contract=False,
                )
            self.assertEqual(set(manifest["files"]), set(OUTPUT_FILENAMES))
            self.assertEqual(manifest["files"]["math_lighteval_train.parquet"]["row_count"], 9)
            self.assertEqual(manifest["files"]["math_lighteval_validation.parquet"]["row_count"], 3)
            self.assertEqual(
                set(path.name for path in output.iterdir()),
                set(OUTPUT_FILENAMES) | {"manifest.json"},
            )
            self.assertEqual(
                verify_materialized_data(output, enforce_pinned_contract=False), manifest
            )
            with self.assertRaisesRegex(ValueError, "pinned contract"):
                verify_materialized_data(output)
            # Existing authenticated output is a no-op and needs no writer.
            self.assertEqual(
                materialize_from_rows(
                    output, [], [], [], {}, enforce_pinned_contract=False
                ),
                manifest,
            )

    def test_verifier_rejects_parquet_tampering(self):
        fake_datasets = types.SimpleNamespace(Dataset=FakeDataset)
        released = {
            name: [released_row(name + "?", "1")]
            for name in ("aime2024", "aime2025", "amc23")
        }
        math500 = [
            {
                "problem": "MATH?",
                "solution": "Work. \\boxed{1}.",
                "answer": "1",
                "subject": "Algebra",
                "level": 1,
                "unique_id": "one",
            }
        ]
        gsm8k = [{"question": "GSM?", "answer": "Work.\n#### 1"}]
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "prepared"
            with patch.dict(sys.modules, {"datasets": fake_datasets}):
                materialize_from_rows(
                    output,
                    [math_row(index) for index in range(12)],
                    math500,
                    gsm8k,
                    released,
                    validation_size=3,
                    enforce_pinned_contract=False,
                )
            path = output / "gsm8k_test.parquet"
            path.write_bytes(path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "failed authentication"):
                verify_materialized_data(output, enforce_pinned_contract=False)

    def test_materializer_rejects_math500_overlap(self):
        fake_datasets = types.SimpleNamespace(Dataset=FakeDataset)
        released = {
            name: [released_row(name + "?", "1")]
            for name in ("aime2024", "aime2025", "amc23")
        }
        math500 = [
            {
                "problem": "Question   4",
                "solution": "Work. \\boxed{4}.",
                "answer": "4",
                "subject": "Algebra",
                "level": 1,
                "unique_id": "overlap",
            }
        ]
        gsm8k = [{"question": "GSM?", "answer": "Work.\n#### 1"}]
        with tempfile.TemporaryDirectory() as parent:
            with patch.dict(sys.modules, {"datasets": fake_datasets}):
                with self.assertRaisesRegex(ValueError, "overlaps MATH-500"):
                    materialize_from_rows(
                        Path(parent) / "prepared",
                        [math_row(index) for index in range(12)],
                        math500,
                        gsm8k,
                        released,
                        validation_size=3,
                        enforce_pinned_contract=False,
                    )


if __name__ == "__main__":
    unittest.main()
