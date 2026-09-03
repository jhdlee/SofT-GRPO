from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from opd_tools.assets import stage_model_snapshot
from opd_tools.constants import MODEL_REVISION
from opd_tools.preflight import run_preflight
from opd_tools.prepare import materialize_from_rows


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows

    @classmethod
    def from_list(cls, rows):
        return cls(rows)

    def to_parquet(self, path):
        Path(path).write_text(json.dumps(self.rows), encoding="utf-8")


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return {"<think>": [101], "</think>": [102]}[text]

    def decode(self, ids, skip_special_tokens=False):
        return {101: "<think>", 102: "</think>"}[ids[0]]

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        if tokenize:
            return list(range(10 + len(messages[0]["content"].split())))
        return "<user>%s</user><assistant><think>\n" % messages[0]["content"]


def math_row(index):
    return {
        "problem": "Question %d" % index,
        "level": "Level %d" % (index % 2 + 1),
        "solution": "Reasoning, then \\boxed{%d}." % index,
        "type": "Algebra" if index < 6 else "Geometry",
    }


class PreflightTest(unittest.TestCase):
    def test_read_only_preflight_checks_both_asset_sets(self):
        fake_datasets = types.SimpleNamespace(Dataset=FakeDataset)
        released = {
            name: [
                {
                    "prompt": [{"from": "user", "value": name + " question"}],
                    "final_answer": "1",
                }
            ]
            for name in ("aime2024", "aime2025", "amc23")
        }
        math500 = [
            {
                "problem": "Held-out problem",
                "solution": "Work. \\boxed{1}.",
                "answer": "1",
                "subject": "Algebra",
                "level": 1,
                "unique_id": "held-out",
            }
        ]
        gsm8k = [{"question": "GSM question", "answer": "Work.\n#### 1"}]

        def download(**kwargs):
            root = Path(kwargs["local_dir"])
            (root / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
            (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"weights")
            return str(root)

        with tempfile.TemporaryDirectory() as parent:
            root = Path(parent)
            data_dir = root / "data"
            model_dir = root / "model"
            with patch.dict(sys.modules, {"datasets": fake_datasets}):
                materialize_from_rows(
                    data_dir,
                    [math_row(index) for index in range(12)],
                    math500,
                    gsm8k,
                    released,
                    validation_size=3,
                    enforce_pinned_contract=False,
                )
            stage_model_snapshot(
                model_dir,
                snapshot_download_fn=download,
                model_info_fn=lambda **kwargs: types.SimpleNamespace(sha=MODEL_REVISION),
            )
            result = run_preflight(
                data_dir,
                model_dir,
                parquet_reader=lambda path: json.loads(path.read_text(encoding="utf-8")),
                tokenizer_loader=lambda path: FakeTokenizer(),
                enforce_pinned_contract=False,
            )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["counts"], {"train": 9, "validation": 3, "math500": 1})
        self.assertEqual(result["student_payloads_checked"], 12)
        self.assertEqual(result["math_train_math500_overlap"], 0)
        self.assertEqual(result["tokenizer"]["token_ids"], {"<think>": 101, "</think>": 102})
        self.assertEqual(result["max_prompt_length"], 2_048)
        self.assertLessEqual(
            result["max_rendered_train_validation_prompt_tokens"], 2_048
        )


if __name__ == "__main__":
    unittest.main()
