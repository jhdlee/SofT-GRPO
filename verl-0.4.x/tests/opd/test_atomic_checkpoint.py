"""Dependency-light tests for the controller's atomic checkpoint protocol."""

import ast
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import stat
import uuid
from pathlib import Path
from types import SimpleNamespace
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
    "_verify_initial_best_reference",
    "_write_initial_best_reference",
    "_maybe_update_best_checkpoint",
    "_remove_stale_checkpoint_trees",
    "_retire_checkpoint",
    "_repair_checkpoint_history",
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
            or isinstance(target, ast.Name)
            and target.id
            in {
                "_BEST_CHECKPOINT_TRACKER",
                "_INITIAL_BEST_RECORD",
                "_COMMITTED_CHECKPOINT_RE",
            }
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
    tiebreak_metric: float | None = None,
    provenance=None,
    with_opd_teacher: bool = False,
    semantic_state: bool = False,
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
    if semantic_state:
        from verl.opd.checkpoint_semantics import write_record
        write_record(temporary / "driver_semantic.json", {'rng_sha256': 'a' * 64, 'dataloader_sha256': 'b' * 64})
        for rank in range(2):
            write_record(temporary / 'actor' / f'semantic_state_world_size_2_rank_{rank}.json',
                         {'rank': rank, 'world_size': 2, 'model_sha256': 'c' * 64,
                          'optimizer_sha256': 'd' * 64, 'scheduler_sha256': 'e' * 64})
            write_record(temporary / 'actor' / f'worker_rng_world_size_2_rank_{rank}.json',
                         {'rank': rank, 'world_size': 2, 'state': {'seed': rank}})
    rollout_record = helpers["_build_rollout_integrity_record"](
        {
            "prompts": torch.tensor([[1, 2], [1, 2]]),
            "responses": torch.tensor([[3, 4, 0], [5, 6, 0]]),
            "response_mask": torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]]),
            "rollout_log_probs": torch.tensor(
                [[-0.1, -0.2, 0.0], [-0.3, -0.4, 0.0]]
            ),
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
        selection_tiebreak_metric_name=(
            "val/released_reward/mean_at_1"
            if tiebreak_metric is not None
            else None
        ),
        selection_tiebreak_metric_value=tiebreak_metric,
        semantic_state=semantic_state,
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


def test_semantic_checkpoint_keeps_byte_authentication_and_rejects_downgrade(tmp_path, checkpoint_helpers):
    checkpoint = _stage_checkpoint(checkpoint_helpers, tmp_path, 1, semantic_state=True)
    verify = checkpoint_helpers['_verify_checkpoint']
    assert verify(str(checkpoint), require_semantic=True)['semantic_identity']['schema'] == 'qwen_semantic_v1'
    path = checkpoint / 'checkpoint_manifest.json'
    manifest = json.loads(path.read_text())
    del manifest['semantic_identity']
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match='semantic identity'):
        verify(str(checkpoint))


def test_semantic_sidecars_are_part_of_authenticated_payload(tmp_path, checkpoint_helpers):
    checkpoint = _stage_checkpoint(checkpoint_helpers, tmp_path, 1, semantic_state=True)
    (checkpoint / 'actor/worker_rng_world_size_2_rank_1.json').write_text('{}')
    with pytest.raises(RuntimeError, match='(size|hash) mismatch'):
        checkpoint_helpers['_verify_checkpoint'](str(checkpoint), require_semantic=True)


def test_old_archives_authenticate_but_cannot_admit_semantic_profile(tmp_path, checkpoint_helpers):
    checkpoint = _stage_checkpoint(checkpoint_helpers, tmp_path, 1)
    checkpoint_helpers['_verify_checkpoint'](str(checkpoint))
    with pytest.raises(RuntimeError, match='required semantic'):
        checkpoint_helpers['_verify_checkpoint'](str(checkpoint), require_semantic=True)


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
        "rollout_log_probs": torch.tensor([[-0.1, -0.2]]),
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


def test_categorical_rollout_digest_requires_no_continuous_support_metadata(
    checkpoint_helpers,
):
    record = checkpoint_helpers["_build_rollout_integrity_record"](
        {
            "prompts": torch.tensor([[1, 2]]),
            "responses": torch.tensor([[3, 4]]),
            "response_mask": torch.ones((1, 2), dtype=torch.bool),
            "attention_mask": torch.ones((1, 4), dtype=torch.bool),
            "rollout_log_probs": torch.tensor([[-0.1, -0.2]]),
            "rollout_sampling_seed": torch.tensor([101]),
        },
        group_ids=["rollout-00000000-prompt-000000"],
        example_identities=[17],
        rollout_iteration=0,
    )

    assert record["replay_mode"] == "categorical"
    assert set(record["fields"]) == {
        "prompts",
        "responses",
        "response_mask",
        "attention_mask",
        "rollout_log_probs",
        "rollout_sampling_seed",
    }
    checkpoint_helpers["_verify_rollout_integrity_record"](
        record, expected_rollout_iteration=0
    )


def _rollout_integrity_batch(*, continuous):
    tensors = {
        "prompts": torch.tensor([[1, 2], [1, 2]]),
        "responses": torch.tensor([[3, 4], [5, 6]]),
        "response_mask": torch.ones((2, 2), dtype=torch.bool),
        "attention_mask": torch.ones((2, 4), dtype=torch.bool),
        "rollout_log_probs": torch.tensor([[-0.1, -0.2], [-0.3, -0.4]]),
        "rollout_sampling_seed": torch.tensor([101, 102]),
    }
    if continuous:
        tensors.update(
            rollout_topk_ids=torch.tensor(
                [[[1, 0], [2, 0], [3, 7], [4, 8]], [[1, 0], [2, 0], [5, 9], [6, 10]]]
            ),
            rollout_topk_gumbels=torch.arange(16, dtype=torch.float32).reshape(2, 4, 2) / 10,
            gumbel_temperature=torch.tensor([0.1, 0.1]),
        )
    return tensors


def _checkpoint_tensor_container(tensors, container):
    if container == "tensordict":
        # Other dependency-light tests install a TensorDict=object import stub
        # when this optional package is absent. Check the installed package,
        # so that stub cannot turn these tests into false failures or passes.
        try:
            importlib.metadata.version("tensordict")
        except importlib.metadata.PackageNotFoundError:
            pytest.skip("actual TensorDict checkpoint regression requires tensordict")
        from tensordict import TensorDict

        assert TensorDict is not object, "installed TensorDict was replaced by a test stub"
        return TensorDict(tensors, batch_size=[2])

    class RowIteratingBatch(dict):
        """Exercise the TensorDict iteration contract without GPU extras."""

        def __iter__(self):
            return iter([{name: tensor[row] for name, tensor in self.items()} for row in range(2)])

        def clone(self):
            return type(self)({name: tensor.clone() for name, tensor in self.items()})

    return RowIteratingBatch(tensors)


@pytest.mark.parametrize("container", ["row_iterating", "tensordict"])
@pytest.mark.parametrize("continuous", [False, True])
def test_tensordict_rollout_inventory_matches_dict_and_authenticates(checkpoint_helpers, continuous, container):
    """Use the actual worker container, whose iterator yields rows, not keys."""

    tensors = _rollout_integrity_batch(continuous=continuous)
    batch = _checkpoint_tensor_container(tensors, container)
    identity = {
        "group_ids": ["rollout-00000002-prompt-000000"] * 2,
        "example_identities": [17, 17],
        "rollout_iteration": 2,
    }
    expected = checkpoint_helpers["_build_rollout_integrity_record"](tensors, **identity)
    actual = checkpoint_helpers["_build_rollout_integrity_record"](batch, **identity)
    assert actual == expected
    assert actual["replay_mode"] == ("continuous" if continuous else "categorical")
    assert checkpoint_helpers["_verify_rollout_integrity_record"](
        actual, expected_rollout_iteration=2
    ) == actual
    assert set(actual["fields"]) == set(tensors)
    if continuous:
        assert actual["fields"]["rollout_topk_ids"]["shape"] == [2, 2, 2]

    # The container repair must retain authentication sensitivity to the actual
    # trajectory bytes rather than just accepting the tensor field names.
    changed = batch.clone()
    changed["rollout_log_probs"][0, 0] += 0.25
    changed_record = checkpoint_helpers["_build_rollout_integrity_record"](changed, **identity)
    assert changed_record["trajectory_sha256"] != actual["trajectory_sha256"]


@pytest.mark.parametrize("missing,match", [
    ("response_mask", "without fields.*response_mask"),
    ("rollout_topk_gumbels", "incomplete continuous replay inventory.*rollout_topk_gumbels"),
])
@pytest.mark.parametrize("container", ["row_iterating", "tensordict"])
def test_tensordict_rollout_inventory_still_rejects_missing_fields(checkpoint_helpers, missing, match, container):
    tensors = _rollout_integrity_batch(continuous=True)
    del tensors[missing]
    with pytest.raises(RuntimeError, match=match):
        checkpoint_helpers["_build_rollout_integrity_record"](
            _checkpoint_tensor_container(tensors, container),
            group_ids=["rollout-00000002-prompt-000000"] * 2,
            example_identities=[17, 17],
            rollout_iteration=2,
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


@pytest.mark.parametrize("failure", ["payload", "manifest", "non_object_manifest", "missing"])
def test_auto_resume_falls_back_to_newest_valid_checkpoint(tmp_path, checkpoint_helpers, failure):
    helpers = checkpoint_helpers
    earlier = _stage_checkpoint(helpers, tmp_path, 25)
    newest = _stage_checkpoint(helpers, tmp_path, 50)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("50\n")
    if failure == "payload":
        (newest / "data.pt").write_bytes(b"broken")
    elif failure == "manifest":
        (newest / "checkpoint_manifest.json").write_text("{")
    elif failure == "non_object_manifest":
        (newest / "checkpoint_manifest.json").write_text("[]")
    else:
        shutil.rmtree(newest)
    selected, manifest = helpers["_find_latest_committed_checkpoint"](
        str(tmp_path), return_manifest=True, expected_provenance=_test_provenance()
    )
    assert selected == str(earlier)
    assert manifest["global_step"] == 25
    # Resolution does not mutate evidence or trackers before state loads.
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "50\n"
    assert newest.exists() == (failure != "missing")


def test_auto_resume_never_silently_starts_fresh_after_all_checkpoints_fail(tmp_path, checkpoint_helpers):
    checkpoint = _stage_checkpoint(checkpoint_helpers, tmp_path, 25)
    (checkpoint / "data.pt").write_bytes(b"broken")
    with pytest.raises(RuntimeError, match="no valid committed checkpoint remains"):
        checkpoint_helpers["_find_latest_committed_checkpoint"](str(tmp_path))


@pytest.mark.parametrize("contents", [b"", b"not-a-step\n", b"\xff\xfe"])
def test_auto_resume_ignores_damaged_latest_tracker_with_valid_checkpoint(tmp_path, checkpoint_helpers, contents):
    checkpoint = _stage_checkpoint(checkpoint_helpers, tmp_path, 25)
    (tmp_path / "latest_checkpointed_iteration.txt").write_bytes(contents)
    assert checkpoint_helpers["_find_latest_committed_checkpoint"](str(tmp_path)) == str(checkpoint)


def test_damaged_latest_tracker_without_checkpoint_does_not_start_fresh(tmp_path, checkpoint_helpers):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("broken")
    with pytest.raises(RuntimeError, match="damaged latest checkpoint tracker has no retained checkpoint"):
        checkpoint_helpers["_find_latest_committed_checkpoint"](str(tmp_path))


@pytest.mark.parametrize("broken", [False, True])
def test_latest_tracker_symlink_is_rejected_even_with_valid_checkpoint(tmp_path, checkpoint_helpers, broken):
    _stage_checkpoint(checkpoint_helpers, tmp_path, 25)
    target = tmp_path / "target"
    if not broken:
        target.write_text("25\n")
    (tmp_path / "latest_checkpointed_iteration.txt").symlink_to(target)
    with pytest.raises(RuntimeError, match="invalid checkpoint tracker"):
        checkpoint_helpers["_find_latest_committed_checkpoint"](str(tmp_path))


def test_load_rejects_foreign_checkpoint_before_removing_abandoned_writes(tmp_path, checkpoint_helpers):
    helpers = checkpoint_helpers
    foreign = _stage_checkpoint(helpers, tmp_path, 25, provenance=_test_provenance(source_commit="a" * 40))
    abandoned = tmp_path / f".global_step_50.incomplete.{uuid.uuid4().hex}"
    abandoned.mkdir()
    (abandoned / "evidence").write_bytes(b"must remain until checkpoint verification succeeds")
    parsed = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    method = next(
        member for node in parsed.body if isinstance(node, ast.ClassDef)
        for member in node.body if isinstance(member, ast.FunctionDef) and member.name == "_load_checkpoint"
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), helpers)
    trainer = SimpleNamespace(
        checkpoint_provenance=_test_provenance(),
        config=SimpleNamespace(trainer=SimpleNamespace(
            default_local_dir=str(tmp_path), default_hdfs_dir=None,
            resume_mode="resume_path", resume_from_path=str(foreign),
        )),
    )
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        helpers["_load_checkpoint"](trainer)
    assert abandoned.is_dir()
    assert foreign.is_dir()


def test_auto_resume_rejects_authenticated_foreign_newest_checkpoint(tmp_path, checkpoint_helpers):
    _stage_checkpoint(checkpoint_helpers, tmp_path, 25)
    foreign = _stage_checkpoint(
        checkpoint_helpers, tmp_path, 50, provenance=_test_provenance(source_commit="a" * 40)
    )
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        checkpoint_helpers["_find_latest_committed_checkpoint"](
            str(tmp_path), expected_provenance=_test_provenance()
        )
    assert foreign.exists()


def test_auto_resume_preserves_required_semantic_verification(tmp_path, checkpoint_helpers):
    older = _stage_checkpoint(checkpoint_helpers, tmp_path, 25, semantic_state=True)
    _stage_checkpoint(checkpoint_helpers, tmp_path, 50)
    assert checkpoint_helpers["_find_latest_committed_checkpoint"](
        str(tmp_path), require_semantic=True
    ) == str(older)


def test_interrupted_retention_unpublishes_before_deleting(tmp_path, checkpoint_helpers, monkeypatch):
    helpers = checkpoint_helpers
    old = _stage_checkpoint(helpers, tmp_path, 25)
    latest = _stage_checkpoint(helpers, tmp_path, 50)
    real_rmtree = shutil.rmtree
    def interrupted(path):
        assert not old.exists()
        assert ".retired." in str(path)
        raise OSError("interrupted deletion")
    monkeypatch.setattr(shutil, "rmtree", interrupted)
    with pytest.raises(OSError, match="interrupted deletion"):
        helpers["_prune_committed_checkpoints"](str(tmp_path), keep_latest=1)
    assert helpers["_find_latest_committed_checkpoint"](str(tmp_path)) == str(latest)
    assert len(list(tmp_path.glob(".global_step_25.retired.*"))) == 1
    monkeypatch.setattr(shutil, "rmtree", real_rmtree)
    helpers["_remove_stale_checkpoint_trees"](str(tmp_path))
    assert not list(tmp_path.glob(".global_step_25.retired.*"))


def test_stale_tree_cleanup_only_removes_owned_temporary_directories(tmp_path, checkpoint_helpers):
    stale = tmp_path / f".global_step_50.incomplete.{uuid.uuid4().hex}"
    stale.mkdir()
    (stale / "large-payload").write_bytes(b"abandoned")
    preserved = tmp_path / ".global_step_50.incomplete.manual-evidence"
    preserved.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / f".global_step_51.incomplete.{uuid.uuid4().hex}"
    link.symlink_to(outside, target_is_directory=True)
    assert checkpoint_helpers["_remove_stale_checkpoint_trees"](str(tmp_path)) == [str(stale)]
    assert preserved.exists() and outside.exists() and link.is_symlink()


def test_in_place_rollback_rebuilds_best_and_frees_future_checkpoint_names(tmp_path, checkpoint_helpers):
    helpers = checkpoint_helpers
    for step, metric in [(25, 0.8), (50, 0.7), (75, 0.9)]:
        _stage_checkpoint(helpers, tmp_path, step, metric=metric)
    (tmp_path / "best_checkpointed_iteration.txt").write_text("75\n")
    resumed = helpers["_verify_checkpoint"](str(tmp_path / "global_step_50"))
    removed = helpers["_repair_checkpoint_history"](
        str(tmp_path), resumed_manifest=resumed, best_mode="max"
    )
    assert str(tmp_path / "global_step_75") in removed
    assert (tmp_path / "best_checkpointed_iteration.txt").read_text() == "25\n"
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "50\n"
    assert not (tmp_path / "checkpoint_recovery_step.txt").exists()
    # Repeating the discarded step can now publish its checkpoint successfully.
    _stage_checkpoint(helpers, tmp_path, 75, metric=0.85)


def test_recovery_finishes_best_update_after_commit_without_tracker_update(tmp_path, checkpoint_helpers):
    helpers = checkpoint_helpers
    _stage_checkpoint(helpers, tmp_path, 25, metric=0.7)
    newest = _stage_checkpoint(helpers, tmp_path, 50, metric=0.9)
    (tmp_path / "best_checkpointed_iteration.txt").write_text("25\n")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("25\n")
    helpers["_repair_checkpoint_history"](
        str(tmp_path), resumed_manifest=helpers["_verify_checkpoint"](str(newest)), best_mode="max"
    )
    assert (tmp_path / "best_checkpointed_iteration.txt").read_text() == "50\n"
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "50\n"


def test_final_commit_recovery_repairs_selection_and_finishes_retention(tmp_path, checkpoint_helpers):
    helpers = checkpoint_helpers
    for step, metric in [(25, 0.7), (75, 0.6), (100, 0.65), (109, 0.9)]:
        _stage_checkpoint(helpers, tmp_path, step, metric=metric)
    # The final checkpoint was committed, but publication of BEST/latest and
    # pruning never ran. An older interrupted write also still occupies disk.
    (tmp_path / "best_checkpointed_iteration.txt").write_text("25\n")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("100\n")
    stale = tmp_path / f".global_step_109.incomplete.{uuid.uuid4().hex}"
    stale.mkdir()
    (stale / "abandoned-payload").write_bytes(b"unfinished")
    final = tmp_path / "global_step_109"
    before = {path: path.read_bytes() for path in final.rglob("*") if path.is_file()}
    checkpoint = helpers["_verify_checkpoint"](str(final))
    helpers["_repair_checkpoint_history"](
        str(tmp_path), resumed_manifest=checkpoint, best_mode="max"
    )
    helpers["_prune_committed_checkpoints"](str(tmp_path), keep_latest=2)
    assert (tmp_path / "best_checkpointed_iteration.txt").read_text() == "109\n"
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "109\n"
    assert {path.name for path in tmp_path.glob("global_step_*")} == {"global_step_100", "global_step_109"}
    assert not stale.exists()
    assert not (tmp_path / "checkpoint_recovery_step.txt").exists()
    assert {path: path.read_bytes() for path in final.rglob("*") if path.is_file()} == before
    assert helpers["_verify_checkpoint"](str(final)) == checkpoint


def test_recovery_removes_corrupt_best_and_preserves_initial_policy_selection(tmp_path, checkpoint_helpers):
    helpers = checkpoint_helpers
    helpers["_write_initial_best_reference"](
        str(tmp_path), metric_name="val/math_verify/mean_at_1", metric_value=0.8,
        tiebreak_metric_name=None, tiebreak_metric_value=None, provenance=_test_provenance(),
    )
    resumed = _stage_checkpoint(helpers, tmp_path, 25, metric=0.7)
    corrupt = _stage_checkpoint(helpers, tmp_path, 50, metric=0.9)
    (corrupt / "data.pt").write_bytes(b"broken")
    (tmp_path / "best_checkpointed_iteration.txt").write_text("50\n")
    helpers["_repair_checkpoint_history"](
        str(tmp_path), resumed_manifest=helpers["_verify_checkpoint"](str(resumed)), best_mode="max"
    )
    assert not corrupt.exists()
    assert (tmp_path / "best_checkpointed_iteration.txt").read_text() == "0\n"
    assert helpers["_find_latest_committed_checkpoint"](str(tmp_path)) == str(resumed)


def test_recovery_never_deletes_authenticated_foreign_future_history(tmp_path, checkpoint_helpers):
    helpers = checkpoint_helpers
    resumed = _stage_checkpoint(helpers, tmp_path, 25, metric=0.7)
    foreign = _stage_checkpoint(
        helpers, tmp_path, 50, metric=0.9, provenance=_test_provenance(source_commit="a" * 40)
    )
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        helpers["_repair_checkpoint_history"](
            str(tmp_path), resumed_manifest=helpers["_verify_checkpoint"](str(resumed)), best_mode="max"
        )
    assert resumed.exists() and foreign.exists()
    assert not (tmp_path / "checkpoint_recovery_step.txt").exists()


def test_interrupted_rollback_cannot_resume_from_superseded_future(tmp_path, checkpoint_helpers):
    helpers = checkpoint_helpers
    resumed = _stage_checkpoint(helpers, tmp_path, 25, metric=0.7)
    future = _stage_checkpoint(helpers, tmp_path, 50, metric=0.9)
    retire = helpers["_retire_checkpoint"]
    def interrupted(path):
        raise OSError("interrupted retirement")
    helpers["_retire_checkpoint"] = interrupted
    manifest = helpers["_verify_checkpoint"](str(resumed))
    with pytest.raises(OSError, match="interrupted retirement"):
        helpers["_repair_checkpoint_history"](str(tmp_path), resumed_manifest=manifest, best_mode="max")
    assert future.exists()
    assert helpers["_find_latest_committed_checkpoint"](str(tmp_path)) == str(resumed)
    helpers["_retire_checkpoint"] = retire
    helpers["_repair_checkpoint_history"](str(tmp_path), resumed_manifest=manifest, best_mode="max")
    assert not future.exists()
    assert not (tmp_path / "checkpoint_recovery_step.txt").exists()


def test_initial_validation_competes_in_best_with_secondary_tie_break(
    tmp_path, checkpoint_helpers
):
    helpers = checkpoint_helpers
    provenance = _test_provenance()
    helpers["_write_initial_best_reference"](
        str(tmp_path),
        metric_name="val/math_verify/mean_at_1",
        metric_value=0.8,
        tiebreak_metric_name="val/released_reward/mean_at_1",
        tiebreak_metric_value=0.5,
        provenance=provenance,
    )
    assert (tmp_path / "best_checkpointed_iteration.txt").read_text().strip() == "0"

    worse = _stage_checkpoint(
        helpers,
        tmp_path,
        1,
        metric=0.79,
        tiebreak_metric=1.0,
        provenance=provenance,
    )
    worse_manifest = helpers["_verify_checkpoint"](str(worse))
    assert not helpers["_maybe_update_best_checkpoint"](
        str(tmp_path),
        global_step=1,
        metric_name="val/math_verify/mean_at_1",
        metric_value=0.79,
        mode="max",
        tiebreak_metric_name="val/released_reward/mean_at_1",
        tiebreak_metric_value=1.0,
        verified_candidate_manifest=worse_manifest,
    )

    better_tie = _stage_checkpoint(
        helpers,
        tmp_path,
        2,
        metric=0.8,
        tiebreak_metric=0.6,
        provenance=provenance,
    )
    better_tie_manifest = helpers["_verify_checkpoint"](str(better_tie))
    assert helpers["_maybe_update_best_checkpoint"](
        str(tmp_path),
        global_step=2,
        metric_name="val/math_verify/mean_at_1",
        metric_value=0.8,
        mode="max",
        tiebreak_metric_name="val/released_reward/mean_at_1",
        tiebreak_metric_value=0.6,
        verified_candidate_manifest=better_tie_manifest,
    )
    assert (tmp_path / "best_checkpointed_iteration.txt").read_text().strip() == "2"

    exact_tie = _stage_checkpoint(
        helpers,
        tmp_path,
        3,
        metric=0.8,
        tiebreak_metric=0.6,
        provenance=provenance,
    )
    exact_tie_manifest = helpers["_verify_checkpoint"](str(exact_tie))
    assert not helpers["_maybe_update_best_checkpoint"](
        str(tmp_path),
        global_step=3,
        metric_name="val/math_verify/mean_at_1",
        metric_value=0.8,
        mode="max",
        tiebreak_metric_name="val/released_reward/mean_at_1",
        tiebreak_metric_value=0.6,
        verified_candidate_manifest=exact_tie_manifest,
    )
    assert (tmp_path / "best_checkpointed_iteration.txt").read_text().strip() == "2"


def test_initial_best_reference_is_authenticated(tmp_path, checkpoint_helpers):
    helpers = checkpoint_helpers
    helpers["_write_initial_best_reference"](
        str(tmp_path),
        metric_name="val/math_verify/mean_at_1",
        metric_value=0.8,
        tiebreak_metric_name="val/released_reward/mean_at_1",
        tiebreak_metric_value=0.5,
        provenance=_test_provenance(),
    )
    record_path = tmp_path / "initial_best_reference.json"
    record = json.loads(record_path.read_text())
    record["selection_metric_value"] = 0.9
    record_path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="digest mismatch"):
        helpers["_prune_committed_checkpoints"](str(tmp_path), keep_latest=2)


def test_initial_reference_cannot_replace_a_trained_best(
    tmp_path, checkpoint_helpers
):
    (tmp_path / "best_checkpointed_iteration.txt").write_text("25\n")
    with pytest.raises(RuntimeError, match="existing trained BEST"):
        checkpoint_helpers["_write_initial_best_reference"](
            str(tmp_path),
            metric_name="val/math_verify/mean_at_1",
            metric_value=0.8,
            tiebreak_metric_name="val/released_reward/mean_at_1",
            tiebreak_metric_value=0.5,
            provenance=_test_provenance(),
        )
    assert not (tmp_path / "initial_best_reference.json").exists()


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
    assert "validation_seed_iteration: 0" in config
    source = SOURCE.read_text(encoding="utf-8")
    assert '"checkpoint_resume_provenance_sha256"' in source
    assert "expected_provenance=self.checkpoint_provenance" in source
    assert 'self.config.trainer.get("validation_seed_iteration", 0)' in source
    assert "if not self._resumed and selection_metric_name in val_metrics" in source
