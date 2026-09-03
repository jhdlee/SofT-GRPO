"""Dependency-light tests for the controller's atomic checkpoint protocol."""

import ast
import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pytest
import torch

from verl.opd.provenance import (
    CORE_RUNTIME_PACKAGES,
    RESUME_INVOCATION_ONLY_CONFIG_FIELDS,
    SOFTGRPO_UPSTREAM_BASE_COMMIT,
    assert_checkpoint_provenance_matches,
    validate_checkpoint_provenance,
)

SOURCE = Path(__file__).resolve().parents[2] / "verl" / "trainer" / "ppo" / "ray_trainer.py"
FUNCTIONS = {
    "_fsync_directory",
    "_atomic_write_text",
    "_checkpoint_step_from_name",
    "_sha256_file",
    "_checkpoint_files",
    "_checkpoint_inventory",
    "_inventory_digest",
    "_tensor_integrity_descriptor",
    "_canonical_json_digest",
    "_identity_string",
    "_build_rollout_integrity_record",
    "_verify_rollout_integrity_record",
    "_write_rollout_integrity_record",
    "_write_checkpoint_manifest",
    "_verify_checkpoint",
    "_find_latest_committed_checkpoint",
    "_verified_actor_state_digest",
    "_verified_rollout_trajectory_digest",
    "_read_step_tracker",
    "_maybe_update_best_checkpoint",
    "_prune_committed_checkpoints",
    "_requeue_requested",
    "_consume_requeue_request",
}


def _load_checkpoint_helpers():
    """Compile the real helpers without importing Ray or production extras."""

    parsed = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    selected = []
    for node in parsed.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id.startswith("_CHECKPOINT")
            or isinstance(target, ast.Name) and target.id in {"_BEST_CHECKPOINT_TRACKER", "_COMMITTED_CHECKPOINT_RE"}
            for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in FUNCTIONS:
            selected.append(node)
    namespace = {
        "hashlib": hashlib,
        "json": json,
        "np": np,
        "os": os,
        "Mapping": Mapping,
        "Optional": Optional,
        "re": re,
        "Sequence": Sequence,
        "shutil": shutil,
        "stat": stat,
        "torch": torch,
        "uuid": uuid,
        "assert_checkpoint_provenance_matches": assert_checkpoint_provenance_matches,
        "validate_checkpoint_provenance": validate_checkpoint_provenance,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    assert FUNCTIONS.issubset(namespace)
    return namespace


@pytest.fixture()
def checkpoint_helpers():
    return _load_checkpoint_helpers()


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


def _test_provenance(
    *,
    source_commit: str = "1" * 40,
    config_hash: str = "2" * 64,
    model_revision: str = "3" * 40,
    model_hash: str = "4" * 64,
    data_hash: str = "5" * 64,
    environment_version: str = "1.0",
):
    data_payload = {
        "manifests": [
            {
                "configured_files": ["train.parquet", "validation.parquet"],
                "manifest_file_sha256": data_hash,
                "manifest_content_sha256": "6" * 64,
            }
        ]
    }
    environment_payload = {
        "python": {"implementation": "CPython", "version": "3.11.13"},
        "core_packages": {
            package: environment_version if package == "torch" else "1.0"
            for package in CORE_RUNTIME_PACKAGES
        },
        "torch_cuda": "12.6",
    }
    payload = {
        "schema_version": 1,
        "source": {
            "commit": source_commit,
            "upstream_base_commit": SOFTGRPO_UPSTREAM_BASE_COMMIT,
        },
        "resolved_hydra_config": {
            "full_sha256": config_hash,
            "resume_semantic_sha256": config_hash,
            "excluded_invocation_fields": list(
                RESUME_INVOCATION_ONLY_CONFIG_FIELDS
            ),
        },
        "model": {
            "id": "example/model",
            "resolved_revision": model_revision,
            "manifest_file_sha256": model_hash,
            "manifest_content_sha256": "7" * 64,
            "inventory_sha256": "8" * 64,
        },
        "data": {
            **data_payload,
            "identity_sha256": _canonical_sha256(data_payload),
        },
        "environment": {
            **environment_payload,
            "informational_packages": {"wandb": "0.21.0"},
            "identity_sha256": _canonical_sha256(environment_payload),
        },
    }
    resume_payload = {
        "source": payload["source"],
        "resolved_hydra_config_sha256": payload["resolved_hydra_config"][
            "resume_semantic_sha256"
        ],
        "model": payload["model"],
        "data": payload["data"],
        "environment_sha256": payload["environment"]["identity_sha256"],
    }
    return {
        **payload,
        "resume_identity_sha256": _canonical_sha256(resume_payload),
        "identity_sha256": _canonical_sha256(payload),
    }


def _stage_checkpoint(
    helpers,
    root: Path,
    step: int,
    metric: float | None = None,
    provenance=None,
    with_opd_teacher: bool = False,
) -> Path:
    temporary = root / f".global_step_{step}.incomplete.test"
    (temporary / "actor").mkdir(parents=True)
    (temporary / "actor" / "model_world_size_2_rank_0.pt").write_bytes(f"model-{step}".encode())
    (temporary / "actor" / "optim_world_size_2_rank_0.pt").write_bytes(f"optim-{step}".encode())
    if with_opd_teacher:
        teacher = temporary / "actor" / "opd_teacher"
        teacher.mkdir()
        (teacher / "model_world_size_2_rank_0.pt").write_bytes(
            f"teacher-{step}".encode()
        )
        (teacher / "ema_state_world_size_2_rank_0.json").write_text(
            json.dumps({"update_count": step, "last_rollout_iteration": step - 1})
        )
    (temporary / "data.pt").write_bytes(f"data-{step}".encode())
    (temporary / "driver_state.pt").write_bytes(f"driver-{step}".encode())
    rollout_record = helpers["_build_rollout_integrity_record"](
        {
            "prompts": torch.tensor([[1, 2], [1, 2]]),
            "responses": torch.tensor([[3, 4, 0], [5, 6, 0]]),
            "response_mask": torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]]),
            "rollout_topk_ids": torch.tensor(
                [
                    [[1, 0], [2, 0], [3, 7], [4, 8], [0, 0]],
                    [[1, 0], [2, 0], [5, 9], [6, 10], [0, 0]],
                ]
            ),
            "rollout_topk_gumbels": torch.tensor(
                [
                    [[0.0, 0.0], [0.0, 0.0], [1.0, 0.2], [0.5, 0.1], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0], [1.1, 0.3], [0.7, 0.2], [0.0, 0.0]],
                ]
            ),
            "rollout_sampling_seed": torch.tensor([101, 102]),
            "gumbel_temperature": torch.tensor([0.1, 0.1]),
        },
        group_ids=[f"rollout-{step - 1:08d}-prompt-000000"] * 2,
        example_identities=[17, 17],
        rollout_iteration=step - 1,
    )
    helpers["_write_rollout_integrity_record"](str(temporary), rollout_record)
    manifest = helpers["_write_checkpoint_manifest"](
        str(temporary),
        checkpoint_name=f"global_step_{step}",
        global_step=step,
        completed_rollout_iteration=step - 1,
        next_rollout_iteration=step,
        optimizer_step=step * 2,
        total_rollout_iterations=327,
        world_size=2,
        reason="test",
        provenance=provenance or _test_provenance(),
        selection_metric_name="val/math_verify/mean_at_1" if metric is not None else None,
        selection_metric_value=metric,
    )
    assert manifest["dataloader_state_sha256"]
    assert manifest["actor_model_optimizer_tree_sha256"]
    assert manifest["rollout_trajectory_sha256"] == rollout_record["trajectory_sha256"]
    assert manifest["provenance_sha256"] == manifest["provenance"]["identity_sha256"]
    assert (
        manifest["resume_provenance_sha256"]
        == manifest["provenance"]["resume_identity_sha256"]
    )
    helpers["_verify_checkpoint"](str(temporary), require_committed_name=False)
    committed = root / f"global_step_{step}"
    os.replace(temporary, committed)
    return committed


def test_atomic_manifest_commit_and_stale_tracker_recovery(tmp_path, checkpoint_helpers):
    helpers = checkpoint_helpers
    first = _stage_checkpoint(helpers, tmp_path, 1)
    helpers["_verify_checkpoint"](str(first))
    helpers["_atomic_write_text"](str(tmp_path / "latest_checkpointed_iteration.txt"), "1\n")

    second = _stage_checkpoint(helpers, tmp_path, 2)
    assert helpers["_find_latest_committed_checkpoint"](str(tmp_path)) == str(second)
    assert helpers["_verified_rollout_trajectory_digest"](str(second)) == json.loads(
        (second / "rollout_metadata.json").read_text()
    )["trajectory_sha256"]


def test_manifest_authenticates_opd_teacher_and_ema_counter(
    tmp_path, checkpoint_helpers
):
    checkpoint = _stage_checkpoint(
        checkpoint_helpers, tmp_path, 1, with_opd_teacher=True
    )
    manifest = checkpoint_helpers["_verify_checkpoint"](str(checkpoint))
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["opd_teacher_tree_sha256"])
    state_path = (
        checkpoint
        / "actor"
        / "opd_teacher"
        / "ema_state_world_size_2_rank_0.json"
    )
    state_path.write_text(
        json.dumps({"update_count": 999, "last_rollout_iteration": 998})
    )
    with pytest.raises(RuntimeError, match="(?:size|hash) mismatch"):
        checkpoint_helpers["_verify_checkpoint"](str(checkpoint))


def test_hash_verification_fails_before_deserialization(tmp_path, checkpoint_helpers):
    checkpoint = _stage_checkpoint(checkpoint_helpers, tmp_path, 1)
    (checkpoint / "data.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="(?:size|hash) mismatch"):
        checkpoint_helpers["_verify_checkpoint"](str(checkpoint))


def test_rollout_counter_tampering_is_rejected(tmp_path, checkpoint_helpers):
    checkpoint = _stage_checkpoint(checkpoint_helpers, tmp_path, 1)
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["next_rollout_iteration"] = 99
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="next-rollout metadata"):
        checkpoint_helpers["_verify_checkpoint"](str(checkpoint))


@pytest.mark.parametrize(
    ("changed", "component"),
    [
        ({"source_commit": "a" * 40}, "source"),
        ({"config_hash": "b" * 64}, "resolved_hydra_config"),
        ({"model_revision": "c" * 40}, "model"),
        ({"model_hash": "d" * 64}, "model"),
        ({"data_hash": "e" * 64}, "data"),
        ({"environment_version": "2.0"}, "environment"),
    ],
)
def test_exact_resume_rejects_every_provenance_mismatch(
    tmp_path, checkpoint_helpers, changed, component
):
    checkpoint = _stage_checkpoint(checkpoint_helpers, tmp_path, 1)
    expected = _test_provenance(**changed)
    with pytest.raises(RuntimeError, match=rf"provenance mismatch: {component}"):
        checkpoint_helpers["_verify_checkpoint"](
            str(checkpoint), expected_provenance=expected
        )


def test_checkpoint_manifest_rejects_tampered_provenance_seal(
    tmp_path, checkpoint_helpers
):
    checkpoint = _stage_checkpoint(checkpoint_helpers, tmp_path, 1)
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["source"]["commit"] = "f" * 40
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(
        RuntimeError, match="provenance (?:resume )?identity hash mismatch"
    ):
        checkpoint_helpers["_verify_checkpoint"](str(checkpoint))


def test_legacy_schema_is_rejected(tmp_path, checkpoint_helpers):
    checkpoint = _stage_checkpoint(checkpoint_helpers, tmp_path, 1)
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="unsupported checkpoint manifest schema"):
        checkpoint_helpers["_verify_checkpoint"](str(checkpoint))


def test_authenticated_actor_digest_supports_zero_weight_parity(tmp_path, checkpoint_helpers):
    first_root = tmp_path / "disabled"
    second_root = tmp_path / "zero_weight"
    first_root.mkdir()
    second_root.mkdir()
    first = _stage_checkpoint(checkpoint_helpers, first_root, 1)
    second = _stage_checkpoint(checkpoint_helpers, second_root, 1)
    assert checkpoint_helpers["_verified_actor_state_digest"](
        str(first)
    ) == checkpoint_helpers["_verified_actor_state_digest"](str(second))


def test_rollout_digest_changes_for_any_continuous_action_change(checkpoint_helpers):
    helpers = checkpoint_helpers
    tensors = {
        "prompts": torch.tensor([[1, 2]]),
        "responses": torch.tensor([[3, 4]]),
        "response_mask": torch.ones((1, 2), dtype=torch.bool),
        "attention_mask": torch.ones((1, 4), dtype=torch.bool),
        "rollout_topk_ids": torch.tensor([[[1, 0], [2, 0], [3, 7], [4, 8]]]),
        "rollout_topk_gumbels": torch.tensor(
            [[[0.0, 0.0], [0.0, 0.0], [1.0, 0.2], [0.5, 0.1]]]
        ),
        "rollout_sampling_seed": torch.tensor([101]),
        "gumbel_temperature": torch.tensor([0.1]),
    }
    first = helpers["_build_rollout_integrity_record"](
        tensors,
        group_ids=["rollout-00000000-prompt-000000"],
        example_identities=[17],
        rollout_iteration=0,
    )
    changed = dict(tensors)
    changed["rollout_topk_gumbels"] = tensors["rollout_topk_gumbels"].clone()
    changed["rollout_topk_gumbels"][0, -1, -1] += 1e-3
    second = helpers["_build_rollout_integrity_record"](
        changed,
        group_ids=["rollout-00000000-prompt-000000"],
        example_identities=[17],
        rollout_iteration=0,
    )
    assert first["trajectory_sha256"] != second["trajectory_sha256"]
    assert (
        first["fields"]["rollout_topk_gumbels"]["sha256"]
        != second["fields"]["rollout_topk_gumbels"]["sha256"]
    )


def test_tensor_digest_supports_bfloat16_without_numpy_conversion(checkpoint_helpers):
    tensor = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    descriptor = checkpoint_helpers["_tensor_integrity_descriptor"](tensor)
    assert descriptor["dtype"] == "torch.bfloat16"
    assert descriptor["shape"] == [1, 2]
    assert re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"])


def test_incomplete_directory_is_never_resumable(tmp_path, checkpoint_helpers):
    (tmp_path / ".global_step_99.incomplete.crash").mkdir()
    assert checkpoint_helpers["_find_latest_committed_checkpoint"](str(tmp_path)) is None
    checkpoint_helpers["_atomic_write_text"](
        str(tmp_path / "latest_checkpointed_iteration.txt"),
        "99\n",
    )
    with pytest.raises(RuntimeError, match="not a real directory"):
        checkpoint_helpers["_find_latest_committed_checkpoint"](str(tmp_path))


def test_retention_keeps_latest_two_and_older_best(tmp_path, checkpoint_helpers):
    helpers = checkpoint_helpers
    for step, metric in [(1, 0.8), (2, 0.5), (3, 0.6), (4, 0.7)]:
        checkpoint = _stage_checkpoint(helpers, tmp_path, step, metric)
        manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text())
        helpers["_maybe_update_best_checkpoint"](
            str(tmp_path),
            global_step=step,
            metric_name="val/math_verify/mean_at_1",
            metric_value=metric,
            mode="max",
            verified_candidate_manifest=manifest,
        )
    removed = helpers["_prune_committed_checkpoints"](str(tmp_path), keep_latest=2)
    assert {Path(path).name for path in removed} == {"global_step_2"}
    assert {path.name for path in tmp_path.glob("global_step_*")} == {
        "global_step_1",
        "global_step_3",
        "global_step_4",
    }
    assert (tmp_path / "best_checkpointed_iteration.txt").read_text().strip() == "1"


def test_requeue_request_is_consumed_only_after_explicit_ack(tmp_path, checkpoint_helpers):
    signal_file = tmp_path / "checkpoint.request"
    assert not checkpoint_helpers["_requeue_requested"](str(signal_file))
    signal_file.touch()
    assert checkpoint_helpers["_requeue_requested"](str(signal_file))
    checkpoint_helpers["_consume_requeue_request"](str(signal_file))
    assert not signal_file.exists()


def test_config_exposes_resume_smoke_without_changing_horizon():
    config = (SOURCE.parents[1] / "config" / "ppo_trainer.yaml").read_text(encoding="utf-8")
    assert "max_rollout_iterations_per_invocation: null" in config
    assert "requeue_signal_file: null" in config
    assert "checkpoint_keep_latest: 2" in config
    source = SOURCE.read_text(encoding="utf-8")
    assert '"checkpoint_resume_provenance_sha256"' in source
    assert "expected_provenance=self.checkpoint_provenance" in source
