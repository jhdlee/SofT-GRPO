"""Focused tests for checkpoint provenance construction and matching."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from verl.opd.provenance import (
    SOFTGRPO_UPSTREAM_BASE_COMMIT,
    _environment_identity,
    assert_checkpoint_provenance_matches,
    build_checkpoint_provenance,
    validate_checkpoint_provenance,
)


def _canonical_sha256(value) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_sealed_manifest(path: Path, payload: dict) -> dict:
    manifest = dict(payload)
    manifest["manifest_content_sha256"] = _canonical_sha256(manifest)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


@pytest.fixture()
def provenance_inputs(tmp_path):
    model_root = tmp_path / "model"
    data_root = tmp_path / "data"
    model_root.mkdir()
    data_root.mkdir()
    train = data_root / "train.parquet"
    validation = data_root / "validation.parquet"
    train.write_bytes(b"train")
    validation.write_bytes(b"validation")

    model_manifest = _write_sealed_manifest(
        model_root / "manifest.json",
        {
            "protocol": "test-model-v1",
            "model": {
                "id": "example/model",
                "requested_revision": "3" * 40,
                "resolved_revision": "3" * 40,
            },
            "inventory_sha256": "4" * 64,
        },
    )
    data_manifest = _write_sealed_manifest(
        data_root / "manifest.json",
        {
            "protocol": "test-data-v1",
            "files": {
                train.name: {"size": train.stat().st_size, "sha256": "5" * 64},
                validation.name: {
                    "size": validation.stat().st_size,
                    "sha256": "6" * 64,
                },
            },
        },
    )
    config = {
        "actor_rollout_ref": {
            "model": {"path": str(model_root)},
            "actor": {"optim": {"lr": 1e-6}},
        },
        "data": {
            "train_files": str(train),
            "val_files": [str(validation)],
        },
        "trainer": {
            "resume_mode": "disable",
            "resume_from_path": None,
            "max_rollout_iterations_per_invocation": 1,
            "requeue_signal_file": "/first/job/checkpoint.request",
            "total_training_steps": 327,
        },
    }
    environment = _environment_identity(
        package_versions={"torch": "2.6.0", "verl": "0.4.0"},
        python_version="3.11.13",
        python_implementation="CPython",
        torch_cuda_version="12.6",
    )
    return config, environment, model_manifest, data_manifest


def test_builder_records_all_required_job_start_identities(provenance_inputs):
    config, environment, model_manifest, data_manifest = provenance_inputs
    provenance = build_checkpoint_provenance(
        config,
        source_commit="1" * 40,
        environment_identity=environment,
    )
    assert validate_checkpoint_provenance(provenance) == provenance
    assert provenance["source"] == {
        "commit": "1" * 40,
        "upstream_base_commit": SOFTGRPO_UPSTREAM_BASE_COMMIT,
    }
    assert provenance["resolved_hydra_config"]["full_sha256"]
    assert provenance["resolved_hydra_config"]["resume_semantic_sha256"]
    assert provenance["model"]["resolved_revision"] == "3" * 40
    assert (
        provenance["model"]["manifest_content_sha256"]
        == model_manifest["manifest_content_sha256"]
    )
    assert (
        provenance["data"]["manifests"][0]["manifest_content_sha256"]
        == data_manifest["manifest_content_sha256"]
    )
    assert provenance["environment"] == environment


def test_config_hash_excludes_only_resume_invocation_controls(provenance_inputs):
    config, environment, _, _ = provenance_inputs
    first = build_checkpoint_provenance(
        config, source_commit="1" * 40, environment_identity=environment
    )
    resumed_config = deepcopy(config)
    resumed_config["trainer"].update(
        {
            "resume_mode": "auto",
            "resume_from_path": "/checkpoint/global_step_1",
            "max_rollout_iterations_per_invocation": None,
            "requeue_signal_file": "/requeued/job/checkpoint.request",
        }
    )
    resumed = build_checkpoint_provenance(
        resumed_config, source_commit="1" * 40, environment_identity=environment
    )
    assert (
        first["resolved_hydra_config"]["resume_semantic_sha256"]
        == resumed["resolved_hydra_config"]["resume_semantic_sha256"]
    )
    assert (
        first["resolved_hydra_config"]["full_sha256"]
        != resumed["resolved_hydra_config"]["full_sha256"]
    )

    changed_config = deepcopy(resumed_config)
    changed_config["actor_rollout_ref"]["actor"]["optim"]["lr"] = 2e-6
    changed = build_checkpoint_provenance(
        changed_config, source_commit="1" * 40, environment_identity=environment
    )
    assert (
        first["resolved_hydra_config"]["resume_semantic_sha256"]
        != changed["resolved_hydra_config"]["resume_semantic_sha256"]
    )


def test_informational_wandb_version_does_not_gate_resume(provenance_inputs):
    config, environment, _, _ = provenance_inputs
    first = build_checkpoint_provenance(
        config, source_commit="1" * 40, environment_identity=environment
    )
    changed_environment = deepcopy(environment)
    changed_environment["informational_packages"]["wandb"] = "999.0"
    environment_payload = {
        key: value
        for key, value in changed_environment.items()
        if key not in {"identity_sha256", "informational_packages"}
    }
    changed_environment["identity_sha256"] = _canonical_sha256(environment_payload)
    second = build_checkpoint_provenance(
        config, source_commit="1" * 40, environment_identity=changed_environment
    )
    assert first["identity_sha256"] != second["identity_sha256"]
    assert first["resume_identity_sha256"] == second["resume_identity_sha256"]
    assert_checkpoint_provenance_matches(first, second)


def test_builder_rejects_bad_source_commit_and_manifest_hash(provenance_inputs):
    config, environment, _, _ = provenance_inputs
    with pytest.raises(RuntimeError, match="full 40-character Git SHA"):
        build_checkpoint_provenance(
            config, source_commit="short", environment_identity=environment
        )

    model_manifest_path = Path(config["actor_rollout_ref"]["model"]["path"]) / "manifest.json"
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    model_manifest["inventory_sha256"] = "9" * 64
    model_manifest_path.write_text(json.dumps(model_manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="model manifest content hash mismatch"):
        build_checkpoint_provenance(
            config, source_commit="1" * 40, environment_identity=environment
        )
