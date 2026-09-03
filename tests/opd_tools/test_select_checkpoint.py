from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from opd_tools.select_checkpoint import resolve_selected_checkpoint


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _fixture(root: Path, *, metric: str = "val/math_verify/mean_at_1") -> Path:
    checkpoint = root / "global_step_25"
    export = checkpoint / "actor" / "huggingface"
    export.mkdir(parents=True)
    (root / "best_checkpointed_iteration.txt").write_text("25\n", encoding="utf-8")
    files = {
        "data.pt": b"data",
        "driver_state.pt": b"driver",
        "rollout_metadata.json": b"rollout",
        "actor/model_world_size_8_rank_0.pt": b"actor",
        "actor/huggingface/config.json": b"{}\n",
        "actor/huggingface/tokenizer_config.json": b"{}\n",
        "actor/huggingface/model.safetensors": b"weights",
    }
    for relative, value in files.items():
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    inventory = [
        {"path": relative, "size": len(value), "sha256": _sha256(checkpoint / relative)}
        for relative, value in sorted(files.items())
    ]
    provenance_payload = {
        "schema_version": 1,
        "source": {"commit": "1" * 40},
        "resolved_hydra_config": {"resume_semantic_sha256": "2" * 64},
        "model": {"inventory_sha256": "3" * 64},
        "data": {"identity_sha256": "4" * 64},
        "environment": {"identity_sha256": "5" * 64},
    }
    resume_payload = {
        "source": provenance_payload["source"],
        "resolved_hydra_config_sha256": provenance_payload[
            "resolved_hydra_config"
        ]["resume_semantic_sha256"],
        "model": provenance_payload["model"],
        "data": provenance_payload["data"],
        "environment_sha256": provenance_payload["environment"]["identity_sha256"],
    }
    provenance = {
        **provenance_payload,
        "resume_identity_sha256": _canonical_sha256(resume_payload),
        "identity_sha256": _canonical_sha256(provenance_payload),
    }
    manifest = {
        "schema_version": 2,
        "checkpoint_name": "global_step_25",
        "global_step": 25,
        "selection_metric_name": metric,
        "selection_metric_value": 0.375,
        "payload_tree_sha256": "a" * 64,
        "actor_model_optimizer_tree_sha256": "b" * 64,
        "rollout_trajectory_sha256": "c" * 64,
        "provenance": provenance,
        "provenance_sha256": provenance["identity_sha256"],
        "resume_provenance_sha256": provenance["resume_identity_sha256"],
        "files": inventory,
    }
    (checkpoint / "checkpoint_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return export


def test_resolves_authenticated_best_export(tmp_path: Path) -> None:
    expected = _fixture(tmp_path)
    selected = resolve_selected_checkpoint(tmp_path)
    assert selected.step == 25
    assert selected.model_export == expected
    assert selected.selection_metric_value == pytest.approx(0.375)


def test_refuses_last_checkpoint_fallback(tmp_path: Path) -> None:
    _fixture(tmp_path)
    (tmp_path / "best_checkpointed_iteration.txt").unlink()
    with pytest.raises(RuntimeError, match="BEST"):
        resolve_selected_checkpoint(tmp_path)


def test_refuses_wrong_selection_metric(tmp_path: Path) -> None:
    _fixture(tmp_path, metric="val/released_reward/mean_at_1")
    with pytest.raises(RuntimeError, match="rather than"):
        resolve_selected_checkpoint(tmp_path)


def test_refuses_tampered_export(tmp_path: Path) -> None:
    export = _fixture(tmp_path)
    (export / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="(?:size|hash) mismatch"):
        resolve_selected_checkpoint(tmp_path)
