from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

from opd_tools.assets import stage_model_snapshot, verify_model_snapshot
from opd_tools.constants import MODEL_ID, MODEL_REVISION


class AssetsTest(unittest.TestCase):
    def test_snapshot_is_resolved_hashed_atomic_and_idempotent(self):
        calls = []

        def info(**kwargs):
            calls.append(("info", kwargs))
            return types.SimpleNamespace(sha=MODEL_REVISION)

        def download(**kwargs):
            calls.append(("download", kwargs))
            root = Path(kwargs["local_dir"])
            (root / "config.json").write_text(
                json.dumps({"model_type": "qwen2"}), encoding="utf-8"
            )
            (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"fake weights")
            metadata = root / ".cache" / "huggingface"
            metadata.mkdir(parents=True)
            (metadata / "download.json").write_text("moving metadata", encoding="utf-8")
            return str(root)

        with tempfile.TemporaryDirectory() as parent:
            destination = Path(parent) / "assets" / "model"
            manifest = stage_model_snapshot(
                destination,
                cache_dir=Path(parent) / "cache",
                snapshot_download_fn=download,
                model_info_fn=info,
            )
            self.assertEqual(manifest["model"]["id"], MODEL_ID)
            self.assertEqual(manifest["model"]["resolved_revision"], MODEL_REVISION)
            self.assertEqual(
                manifest["transformers_local_path"], str(destination.resolve())
            )
            self.assertNotIn(".cache/huggingface/download.json", manifest["files"])
            self.assertEqual(verify_model_snapshot(destination), manifest)
            self.assertEqual(
                stage_model_snapshot(
                    destination,
                    snapshot_download_fn=lambda **kwargs: self.fail("redownloaded"),
                    model_info_fn=lambda **kwargs: self.fail("reresolved"),
                ),
                manifest,
            )
        self.assertEqual(calls[0][1]["revision"], MODEL_REVISION)
        self.assertEqual(calls[1][1]["revision"], MODEL_REVISION)

    def test_resolved_revision_must_equal_pin(self):
        with tempfile.TemporaryDirectory() as parent:
            with self.assertRaisesRegex(ValueError, "resolved to"):
                stage_model_snapshot(
                    Path(parent) / "model",
                    snapshot_download_fn=lambda **kwargs: self.fail("downloaded"),
                    model_info_fn=lambda **kwargs: types.SimpleNamespace(sha="0" * 40),
                )

    def test_file_tampering_is_rejected(self):
        def download(**kwargs):
            root = Path(kwargs["local_dir"])
            (root / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
            (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"weights")
            return str(root)

        with tempfile.TemporaryDirectory() as parent:
            destination = Path(parent) / "model"
            stage_model_snapshot(
                destination,
                snapshot_download_fn=download,
                model_info_fn=lambda **kwargs: types.SimpleNamespace(sha=MODEL_REVISION),
            )
            (destination / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "failed authentication"):
                verify_model_snapshot(destination)


if __name__ == "__main__":
    unittest.main()
