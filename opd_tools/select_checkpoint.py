"""Resolve and authenticate a validation-selected VERL checkpoint export.

The command prints the ordinary Hugging Face export under the checkpoint's
``actor/huggingface`` directory.  It deliberately refuses to select the last
checkpoint when BEST is absent: test-set evaluation must use the checkpoint
chosen solely by the configured validation metric.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .checkpoint_compare import CHECKPOINT_SCHEMA_VERSION, verified_manifest


BEST_TRACKER = "best_checkpointed_iteration.txt"
DEFAULT_SELECTION_METRIC = "val/math_verify/mean_at_1"


@dataclass(frozen=True)
class SelectedCheckpoint:
    run_root: Path
    step: int
    checkpoint: Path
    model_export: Path
    selection_metric_name: str
    selection_metric_value: float
    payload_tree_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_root": str(self.run_root),
            "step": self.step,
            "checkpoint": str(self.checkpoint),
            "model_export": str(self.model_export),
            "selection_metric_name": self.selection_metric_name,
            "selection_metric_value": self.selection_metric_value,
            "payload_tree_sha256": self.payload_tree_sha256,
        }


def _read_selected_step(run_root: Path) -> int:
    tracker = run_root / BEST_TRACKER
    if not tracker.is_file() or tracker.is_symlink():
        raise RuntimeError(
            "BEST validation-checkpoint tracker is missing or unsafe: %s" % tracker
        )
    value = tracker.read_text(encoding="utf-8").strip()
    if not value.isdigit() or int(value) < 1:
        raise RuntimeError("BEST validation-checkpoint tracker is invalid: %r" % value)
    return int(value)


def resolve_selected_checkpoint(
    run_root: Path,
    *,
    expected_metric: str = DEFAULT_SELECTION_METRIC,
) -> SelectedCheckpoint:
    """Return the authenticated HF export selected by validation only."""

    unresolved_root = Path(run_root).expanduser().absolute()
    if not unresolved_root.is_dir() or unresolved_root.is_symlink():
        raise RuntimeError("run root is missing or unsafe: %s" % unresolved_root)
    root = unresolved_root.resolve()
    step = _read_selected_step(root)
    checkpoint = root / ("global_step_%d" % step)
    manifest = verified_manifest(root, step)
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("selected checkpoint has an unsupported manifest schema")
    if manifest.get("checkpoint_name") != checkpoint.name:
        raise RuntimeError("selected checkpoint name disagrees with its manifest")
    if manifest.get("selection_metric_name") != expected_metric:
        raise RuntimeError(
            "selected checkpoint uses %r rather than %r"
            % (manifest.get("selection_metric_name"), expected_metric)
        )
    metric_value = manifest.get("selection_metric_value")
    if (
        isinstance(metric_value, bool)
        or not isinstance(metric_value, (float, int))
        or not math.isfinite(float(metric_value))
    ):
        raise RuntimeError("selected checkpoint has no finite validation metric")
    payload_digest = manifest.get("payload_tree_sha256")
    if not isinstance(payload_digest, str) or len(payload_digest) != 64:
        raise RuntimeError("selected checkpoint has no valid payload-tree digest")

    actor = checkpoint / "actor"
    model_export = actor / "huggingface"
    if (
        not actor.is_dir()
        or actor.is_symlink()
        or not model_export.is_dir()
        or model_export.is_symlink()
    ):
        raise RuntimeError(
            "selected checkpoint has no safe actor/huggingface export; "
            "production must save actor.checkpoint.contents including hf_model"
        )
    required = (model_export / "config.json", model_export / "tokenizer_config.json")
    if any(not path.is_file() or path.is_symlink() for path in required):
        raise RuntimeError(
            "selected Hugging Face export lacks model/tokenizer configuration"
        )
    weights = [
        path
        for path in model_export.iterdir()
        if path.is_file()
        and not path.is_symlink()
        and path.suffix in {".safetensors", ".bin"}
    ]
    if not weights:
        raise RuntimeError("selected Hugging Face export contains no model weights")

    export_prefix = "actor/huggingface/"
    inventory_paths = {
        entry.get("path") for entry in manifest["files"] if isinstance(entry, dict)
    }
    for path in (*required, *weights):
        relative = path.relative_to(checkpoint).as_posix()
        if not relative.startswith(export_prefix) or relative not in inventory_paths:
            raise RuntimeError(
                "selected model export is absent from checkpoint inventory: %s" % path
            )

    return SelectedCheckpoint(
        run_root=root,
        step=step,
        checkpoint=checkpoint,
        model_export=model_export,
        selection_metric_name=expected_metric,
        selection_metric_value=float(metric_value),
        payload_tree_sha256=payload_digest,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-metric", default=DEFAULT_SELECTION_METRIC)
    parser.add_argument("--format", choices=("json", "path"), default="json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = resolve_selected_checkpoint(
        args.run_root, expected_metric=args.expected_metric
    )
    if args.format == "path":
        print(selected.model_export)
    else:
        print(json.dumps(selected.as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
