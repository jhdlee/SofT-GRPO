import json
from pathlib import Path

import huggingface_hub
import pytest

import opd_tools.icl_assets as assets
from opd_tools.icl import (
    SOFTGRPO_MODEL_ID,
    SOFTGRPO_MODEL_REVISION,
    SOFTGRPO_MODEL_SUBFOLDER,
    STARTING_MODEL_ID,
    STARTING_MODEL_REVISION,
)


def _write_checkpoint(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")


def test_two_model_specs_are_exactly_pinned_with_softgrpo_subfolder():
    assert assets.MODEL_SPECS == {
        "starting": {
            "repo_id": STARTING_MODEL_ID,
            "revision": STARTING_MODEL_REVISION,
            "subfolder": None,
        },
        "softgrpo": {
            "repo_id": SOFTGRPO_MODEL_ID,
            "revision": SOFTGRPO_MODEL_REVISION,
            "subfolder": SOFTGRPO_MODEL_SUBFOLDER,
        },
    }


def test_prepare_stages_both_models_is_idempotent_and_authenticates(monkeypatch, tmp_path):
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"])
        if kwargs["repo_id"] == SOFTGRPO_MODEL_ID:
            _write_checkpoint(destination / SOFTGRPO_MODEL_SUBFOLDER)
        else:
            _write_checkpoint(destination)
        return str(destination)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    data_digest = "d" * 64

    def fake_prepare(path, cache):
        Path(path).mkdir(parents=True)
        return {"content_sha256": data_digest}

    def fake_verify(path):
        assert Path(path).is_dir()
        return {"content_sha256": data_digest}

    monkeypatch.setattr(assets, "prepare_icl_dataset", fake_prepare)
    monkeypatch.setattr(assets, "verify_icl_dataset", fake_verify)
    root = tmp_path / "assets"
    manifest = assets.prepare_icl_assets(root, tmp_path / "cache")
    assert len(calls) == 2
    starting_call = next(call for call in calls if call["repo_id"] == STARTING_MODEL_ID)
    soft_call = next(call for call in calls if call["repo_id"] == SOFTGRPO_MODEL_ID)
    assert starting_call["revision"] == STARTING_MODEL_REVISION
    assert starting_call["allow_patterns"] is None
    assert soft_call["revision"] == SOFTGRPO_MODEL_REVISION
    assert soft_call["allow_patterns"] == [
        SOFTGRPO_MODEL_SUBFOLDER + "/*",
        SOFTGRPO_MODEL_SUBFOLDER + "/**/*",
    ]
    assert json.loads((root / "models/softgrpo/.opd_source.json").read_text()) == assets.MODEL_SPECS["softgrpo"]
    assert manifest["data_content_sha256"] == data_digest

    # A second caller takes the same lock, authenticates, and performs no download.
    assert assets.prepare_icl_assets(root, tmp_path / "cache") == manifest
    assert len(calls) == 2

    (root / "models/starting/model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="starting model tree changed"):
        assets.verify_icl_assets(root)


def test_existing_model_without_pinned_source_marker_is_rejected(tmp_path):
    destination = tmp_path / "model"
    _write_checkpoint(destination)
    with pytest.raises(ValueError, match="pinned-source marker"):
        assets._stage_one(
            destination,
            assets.MODEL_SPECS["starting"],
            tmp_path / "cache",
        )
