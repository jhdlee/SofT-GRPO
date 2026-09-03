from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from opd_tools.data import prepare_math_training_splits
from opd_tools.manifest import (
    build_math_data_manifest,
    canonical_sha256,
    file_sha256,
    validate_manifest_content,
    write_manifest_atomic,
)


def row(index):
    return {
        "problem": "Question %d" % index,
        "level": "Level %d" % (index % 2 + 1),
        "solution": "Work, then \\boxed{%d}." % index,
        "type": "Algebra" if index < 6 else "Geometry",
    }


class ManifestTest(unittest.TestCase):
    def test_manifest_is_stable_and_authenticates_content(self):
        rows = [row(index) for index in range(12)]
        splits, report = prepare_math_training_splits(
            rows,
            validation_size=3,
            seed=42,
            enforce_pinned_contract=False,
        )
        first = build_math_data_manifest(splits, report)
        second = build_math_data_manifest(splits, report)
        self.assertEqual(first, second)
        self.assertEqual(len(first["manifest_content_sha256"]), 64)
        validate_manifest_content(first)

        tampered = json.loads(json.dumps(first))
        tampered["split"]["splits"]["train"]["count"] += 1
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_manifest_content(tampered)

    def test_atomic_manifest_write(self):
        rows = [row(index) for index in range(12)]
        splits, report = prepare_math_training_splits(
            rows,
            validation_size=3,
            enforce_pinned_contract=False,
        )
        manifest = build_math_data_manifest(splits, report)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_manifest_atomic(path, manifest)
            observed = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(observed, manifest)
            self.assertEqual(file_sha256(path), file_sha256(path))
            self.assertEqual(canonical_sha256(observed), canonical_sha256(manifest))


if __name__ == "__main__":
    unittest.main()
