from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from opd_tools.select_checkpoint import resolve_selected_checkpoint
from opd_tools.assets import MODEL_ASSET_PROTOCOL
from opd_tools.constants import MODEL_ID, MODEL_REVISION
from opd_tools.manifest import canonical_sha256
from verl.opd.provenance import (
    CORE_RUNTIME_PACKAGES,
    INFORMATIONAL_PACKAGES,
    RESUME_INVOCATION_ONLY_CONFIG_FIELDS,
    SOFTGRPO_UPSTREAM_BASE_COMMIT,
)


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


def _valid_initial_provenance(model: dict) -> dict:
    data_payload = {
        "manifests": [{
            "configured_files": ["train.parquet", "validation.parquet"],
            "manifest_file_sha256": "4" * 64,
            "manifest_content_sha256": "5" * 64,
        }]
    }
    environment_payload = {
        "python": {"implementation": "CPython", "version": "3.11.13"},
        "core_packages": {package: "1.0" for package in CORE_RUNTIME_PACKAGES},
        "torch_cuda": "12.6",
    }
    payload = {
        "schema_version": 1,
        "source": {
            "commit": "1" * 40,
            "upstream_base_commit": SOFTGRPO_UPSTREAM_BASE_COMMIT,
        },
        "resolved_hydra_config": {
            "full_sha256": "2" * 64,
            "resume_semantic_sha256": "3" * 64,
            "excluded_invocation_fields": list(RESUME_INVOCATION_ONLY_CONFIG_FIELDS),
        },
        "model": model,
        "data": {**data_payload, "identity_sha256": _canonical_sha256(data_payload)},
        "environment": {
            **environment_payload,
            "informational_packages": {
                package: "1.0" for package in INFORMATIONAL_PACKAGES
            },
            "identity_sha256": _canonical_sha256(environment_payload),
        },
    }
    resume_payload = {
        "source": payload["source"],
        "resolved_hydra_config_sha256": payload["resolved_hydra_config"]["resume_semantic_sha256"],
        "model": payload["model"],
        "data": payload["data"],
        "environment_sha256": payload["environment"]["identity_sha256"],
    }
    return {
        **payload,
        "resume_identity_sha256": _canonical_sha256(resume_payload),
        "identity_sha256": _canonical_sha256(payload),
    }


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


def test_resolves_authenticated_initial_best_sentinel(tmp_path: Path) -> None:
    model = tmp_path / "pinned-starting-model"
    model.mkdir()
    for name, value in {
        "config.json": b"{}\n",
        "tokenizer_config.json": b"{}\n",
        "model.safetensors": b"weights",
    }.items():
        (model / name).write_bytes(value)
    inventory = {
        path.name: {"size": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(model.iterdir())
    }
    model_manifest = {
        "protocol": MODEL_ASSET_PROTOCOL,
        "model": {
            "id": MODEL_ID,
            "requested_revision": MODEL_REVISION,
            "resolved_revision": MODEL_REVISION,
        },
        "transformers_local_path": str(model.resolve()),
        "model_type": "qwen2",
        "files": inventory,
        "inventory_sha256": canonical_sha256(inventory),
    }
    model_manifest["manifest_content_sha256"] = canonical_sha256(model_manifest)
    (model / "manifest.json").write_text(json.dumps(model_manifest), encoding="utf-8")

    provenance_model = {
        "id": MODEL_ID,
        "resolved_revision": MODEL_REVISION,
        "manifest_file_sha256": _sha256(model / "manifest.json"),
        "manifest_content_sha256": model_manifest["manifest_content_sha256"],
        "inventory_sha256": model_manifest["inventory_sha256"],
    }
    payload = {
        "schema_version": 1,
        "global_step": 0,
        "selection_metric_name": "val/math_verify/mean_at_1",
        "selection_metric_value": 0.5,
        "selection_tiebreak_metric_name": "val/released_reward/mean_at_1",
        "selection_tiebreak_metric_value": 0.5,
        "provenance": _valid_initial_provenance(provenance_model),
    }
    record = {**payload, "sha256": _canonical_sha256(payload)}
    (tmp_path / "initial_best_reference.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    (tmp_path / "best_checkpointed_iteration.txt").write_text("0\n", encoding="utf-8")

    selected = resolve_selected_checkpoint(tmp_path, initial_model_path=model)
    assert selected.step == 0
    assert selected.model_export == model.resolve()
    assert selected.selection_metric_value == pytest.approx(0.5)

    # Re-sealing only the outer record must not hide corrupted nested
    # checkpoint provenance.
    record["provenance"]["environment"]["identity_sha256"] = "0" * 64
    tampered_payload = {key: value for key, value in record.items() if key != "sha256"}
    record["sha256"] = _canonical_sha256(tampered_payload)
    (tmp_path / "initial_best_reference.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="provenance is invalid"):
        resolve_selected_checkpoint(tmp_path, initial_model_path=model)


def test_initial_best_sentinel_requires_explicit_pinned_model(tmp_path: Path) -> None:
    tmp_path.joinpath("best_checkpointed_iteration.txt").write_text("0\n")
    with pytest.raises(RuntimeError, match="initial-model-path"):
        resolve_selected_checkpoint(tmp_path)
