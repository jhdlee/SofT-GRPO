"""Resolve and authenticate a validation-selected VERL checkpoint export.

The command prints the ordinary Hugging Face export under the checkpoint's
``actor/huggingface`` directory.  It deliberately refuses to select the last
checkpoint when BEST is absent: test-set evaluation must use the checkpoint
chosen solely by the configured validation metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .checkpoint_compare import CHECKPOINT_SCHEMA_VERSION, verified_manifest
from .assets import verify_model_snapshot
from .manifest import file_sha256
from verl.opd.provenance import validate_checkpoint_provenance


BEST_TRACKER = "best_checkpointed_iteration.txt"
INITIAL_BEST_RECORD = "initial_best_reference.json"
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
    if not value.isdigit() or int(value) < 0:
        raise RuntimeError("BEST validation-checkpoint tracker is invalid: %r" % value)
    return int(value)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _resolve_initial_reference(
    root: Path, initial_model_path: Path | None, expected_metric: str
) -> SelectedCheckpoint:
    if initial_model_path is None:
        raise RuntimeError(
            "BEST points to the initial policy; --initial-model-path is required"
        )
    record_path = root / INITIAL_BEST_RECORD
    if not record_path.is_file() or record_path.is_symlink():
        raise RuntimeError("initial BEST reference is missing or unsafe")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise RuntimeError("initial BEST reference must be an object")
    digest = record.get("sha256")
    payload = {key: value for key, value in record.items() if key != "sha256"}
    if digest != _canonical_sha256(payload):
        raise RuntimeError("initial BEST reference digest mismatch")
    if payload.get("schema_version") != 1 or payload.get("global_step") != 0:
        raise RuntimeError("initial BEST reference schema is invalid")
    if payload.get("selection_metric_name") != expected_metric:
        raise RuntimeError(
            "initial policy was selected using %r rather than %r"
            % (payload.get("selection_metric_name"), expected_metric)
        )
    metric_value = payload.get("selection_metric_value")
    if (
        isinstance(metric_value, bool)
        or not isinstance(metric_value, (float, int))
        or not math.isfinite(float(metric_value))
    ):
        raise RuntimeError("initial BEST reference has no finite validation metric")

    model_root = Path(initial_model_path).expanduser().absolute()
    if model_root.is_symlink():
        raise RuntimeError("initial model path may not be a symlink")
    model_root = model_root.resolve()
    model_manifest = verify_model_snapshot(model_root)
    try:
        provenance = validate_checkpoint_provenance(payload.get("provenance"))
    except RuntimeError as error:
        raise RuntimeError("initial BEST reference provenance is invalid") from error
    provenance_model = provenance["model"]
    expected_identity = {
        "id": model_manifest["model"]["id"],
        "resolved_revision": model_manifest["model"]["resolved_revision"],
        "manifest_file_sha256": file_sha256(model_root / "manifest.json"),
        "manifest_content_sha256": model_manifest["manifest_content_sha256"],
        "inventory_sha256": model_manifest["inventory_sha256"],
    }
    if provenance_model != expected_identity:
        raise RuntimeError("initial BEST reference model identity differs")
    return SelectedCheckpoint(
        run_root=root,
        step=0,
        checkpoint=record_path,
        model_export=model_root,
        selection_metric_name=expected_metric,
        selection_metric_value=float(metric_value),
        payload_tree_sha256=model_manifest["inventory_sha256"],
    )


def resolve_selected_checkpoint(
    run_root: Path,
    *,
    expected_metric: str = DEFAULT_SELECTION_METRIC,
    initial_model_path: Path | None = None,
) -> SelectedCheckpoint:
    """Return the authenticated HF export selected by validation only."""

    unresolved_root = Path(run_root).expanduser().absolute()
    if not unresolved_root.is_dir() or unresolved_root.is_symlink():
        raise RuntimeError("run root is missing or unsafe: %s" % unresolved_root)
    root = unresolved_root.resolve()
    step = _read_selected_step(root)
    if step == 0:
        return _resolve_initial_reference(root, initial_model_path, expected_metric)
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
    parser.add_argument("--initial-model-path", type=Path)
    parser.add_argument("--format", choices=("json", "path"), default="json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = resolve_selected_checkpoint(
        args.run_root,
        expected_metric=args.expected_metric,
        initial_model_path=args.initial_model_path,
    )
    if args.format == "path":
        print(selected.model_export)
    else:
        print(json.dumps(selected.as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
