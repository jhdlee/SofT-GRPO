"""Replay failure evidence stays finite, bounded, and durable before updates."""

import ast
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.trainer.ppo.opd_driver import (
    RolloutIntegrityConfig,
    build_replay_failure_diagnostics,
    compute_rollout_diagnostics,
    replay_integrity_mask,
    validate_categorical_rollout_integrity,
    validate_rollout_integrity,
)


SOURCE = Path(__file__).resolve().parents[2] / "verl/trainer/ppo"


def replay_inputs():
    responses = torch.tensor([[4, 99, 7, 8, 0], [4, 99, 7, 8, 9]])
    valid = responses.ne(0)
    supports = torch.stack([responses, torch.zeros_like(responses), torch.zeros_like(responses)], dim=-1)
    supports[:, 0, 1:] = torch.tensor([5, 6])
    rollout = torch.zeros_like(responses, dtype=torch.float32)
    actor = rollout.clone()
    actor[0, 0] = math.log(1.1)
    actor[1, 3] = math.log(2.0)
    actor[0, 1] = 1000  # Rewritten boundary is deliberately not compared.
    actor[0, 4] = float("nan")  # Padding is deliberately not compared.
    return {
        "responses": responses, "response_mask": valid,
        "rollout_log_probs": rollout, "actor_log_probs": actor,
        "comparison_mask": replay_integrity_mask(response_mask=valid, continuous_replay=True, rollout_topk_ids=supports, responses=responses, close_tag_token_id=99),
        "close_tag_token_id": 99, "prompt_indices": np.array([101, 202]),
        "rollout_ranks": torch.tensor([1, 0]), "rollout_sampling_seeds": torch.tensor([123, 456]),
        "rollout_topk_ids": supports, "rollout_topk_gumbels": supports.float() / 10,
        "rollout_topk_gumbel_noise": torch.full_like(supports, 0.125, dtype=torch.float32),
        "rollout_topk_retained_mask": supports.ne(0),
        "rollout_topk_probs": torch.full_like(supports, 0.25, dtype=torch.float32),
    }


def test_worst_replay_positions_preserve_balanced_identity_and_bounded_support_evidence():
    inputs = replay_inputs()
    result = build_replay_failure_diagnostics(**inputs, max_records=2)
    assert result["positions_retained"] == 2
    worst = result["worst_positions"][0]
    assert (worst["prompt_index"], worst["rollout_rank"], worst["request_seed"], worst["response_position"], worst["segment"]) == (202, 0, 456, 3, "hard_answer")
    assert worst["rollout_log_density"] == 0
    assert worst["actor_log_density"] == pytest.approx(math.log(2.0))
    assert worst["ratio_abs_error"] == pytest.approx(1)
    assert worst["support_ids"] == [8, 0, 0]
    assert worst["retained_mask"] == [True, False, False]
    assert worst["raw_gumbel_noise"] == [0.125] * 3
    assert worst["stored_probabilities"] == [0.25] * 3
    assert result["segments"]["boundary"]["compared_positions"] == 0
    assert result["segments"]["boundary"]["excluded_positions"] == 2
    assert result["segments"]["soft_prefix"]["compared_positions"] == 2
    assert result["segments"]["hard_answer"]["compared_positions"] == 5
    assert result["capped_response_count"] == 1
    assert result["close_tag_response_count"] == 2
    assert result["nonfinite_positions"] == 0
    json.dumps(result, allow_nan=False)


def test_replay_overflow_and_nonfinite_inputs_remain_finite_json_without_prompt_text():
    inputs = replay_inputs()
    inputs["actor_log_probs"][0, 0] = float("inf")
    inputs["rollout_topk_gumbel_noise"][0, 0, 0] = float("nan")
    inputs["prompt_indices"] = ["SECRET GOLD OR PROMPT TEXT", 202]
    result = build_replay_failure_diagnostics(**inputs)
    assert result["nonfinite_positions"] == 1
    assert result["segments"]["soft_prefix"]["nonfinite_positions"] == 1
    assert result["worst_positions"][0]["nonfinite"]
    assert result["worst_positions"][0]["actor_log_density"] is None
    assert result["worst_positions"][0]["ratio_abs_error"] is None
    assert result["worst_positions"][0]["raw_gumbel_noise"][0] is None
    assert result["worst_positions"][0]["prompt_index"] is None
    encoded = json.dumps(result, allow_nan=False)
    assert "SECRET" not in encoded
    assert "gold" not in encoded
    assert len(result["worst_positions"]) <= 8


@pytest.mark.parametrize("limit", [0, 9, True])
def test_replay_diagnostic_dump_bound_cannot_be_overridden(limit):
    with pytest.raises(ValueError):
        build_replay_failure_diagnostics(**replay_inputs(), max_records=limit)


def _load_failure_methods():
    """Exercise real driver/persistence methods without starting a Ray runtime."""
    namespace = {
        "np": np, "math": math, "time": time, "Path": Path,
        "build_replay_failure_diagnostics": build_replay_failure_diagnostics,
        "validate_rollout_integrity": validate_rollout_integrity,
        "validate_categorical_rollout_integrity": validate_categorical_rollout_integrity,
    }
    nodes = []
    for filename, class_name, method_names, function_names in (
        ("qwen_benchmark.py", "QwenTrainingBenchmarkTrainer", {"_persist", "record_benchmark_failure"}, {"_write", "_finite_failure_value"}),
        ("ray_trainer.py", "RayPPOTrainer", {"_validate_benchmark_before_update"}, set()),
    ):
        tree = ast.parse((SOURCE / filename).read_text())
        nodes.extend(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in function_names)
        original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        original.bases = []
        original.body = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in method_names]
        nodes.append(original)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "failure-methods", "exec"), namespace)
    return type("FailureTrainer", (namespace["QwenTrainingBenchmarkTrainer"], namespace["RayPPOTrainer"]), {})


@pytest.mark.parametrize("failure_kind", ["ratio", "cap"])
def test_real_pre_update_gate_persists_failure_before_update_and_retains_partial_timing(tmp_path, failure_kind):
    trainer = _load_failure_methods()()
    inputs = replay_inputs()
    if failure_kind == "cap":
        inputs["actor_log_probs"] = inputs["rollout_log_probs"].clone()
    trainer.measurement_path = tmp_path / "measurement.json"
    trainer.measurement = {"status": "running", "iterations": [], "startup_seconds": 3.5}
    trainer.rollout_integrity_config = RolloutIntegrityConfig(enabled=True, gate_first_n_iterations=1)
    trainer.continuous_replay = True
    trainer.close_tag_token_id = 99
    diagnostics = compute_rollout_diagnostics(responses=inputs["responses"], response_mask=inputs["response_mask"], rollout_topk_ids=inputs["rollout_topk_ids"], rollout_topk_gumbels=inputs["rollout_topk_gumbels"], gumbel_temperature=0.1, close_tag_token_id=99, decode=lambda ids: r"\boxed{1}")
    tensors = {key: inputs[key] for key in ("responses", "response_mask", "rollout_log_probs", "rollout_topk_ids", "rollout_topk_gumbels", "rollout_topk_gumbel_noise", "rollout_topk_retained_mask", "rollout_topk_probs")}
    tensors.update(rollout_rank=inputs["rollout_ranks"], rollout_sampling_seed=inputs["rollout_sampling_seeds"])
    batch = SimpleNamespace(batch=tensors, non_tensor_batch={"index": inputs["prompt_indices"], "gold_cot": ["SECRET GOLD"]}, meta_info={"rollout_timing": {"ranks": [{"rank": 0, "weight_sync_seconds": 1}]}, "privileged_prompt": "SECRET PROMPT"})
    updates = []
    with pytest.raises(RuntimeError, match="ratio error" if failure_kind == "ratio" else "cap rate"):
        trainer._validate_benchmark_before_update(batch=batch, diagnostics=diagnostics, replay_error=1 if failure_kind == "ratio" else 0,
            actor_log_probs=inputs["actor_log_probs"], comparison_mask=inputs["comparison_mask"], iteration=0,
            timing={"gen": 20, "old_log_prob": 5}, metrics={**diagnostics.metrics, "training/rollout_probs_diff_std": float("nan")}, started_at=time.perf_counter() - 30)
        updates.append("optimizer update")
    assert updates == []
    measured = json.loads(trainer.measurement_path.read_text())
    assert measured["status"] == "failed"
    assert measured["iterations"] == []
    assert measured["startup_seconds"] == 3.5
    failure = measured["failed_iterations"][0]
    assert failure == measured["failure"]
    assert failure["status"] == "failed_before_update"
    assert failure["optimizer_updates_completed"] is False
    assert failure["timing_s"]["gen"] == 20
    assert failure["timing_s"]["old_log_prob"] == 5
    assert failure["timing_s"]["step_partial"] >= 30
    assert "update_actor" not in failure["timing_s"]
    assert failure["metrics"]["training/rollout_probs_diff_std"] is None
    assert failure["nonfinite_metric_names"] == ["training/rollout_probs_diff_std"]
    assert failure["diagnostics"]["rollout_metrics"]["latent/cap_rate"] == 0.5
    assert "SECRET" not in trainer.measurement_path.read_text()
    assert trainer.rollout_integrity_config.max_replay_ratio_abs_error == 1e-4


def test_diagnostic_gate_precedes_worker_updates_and_legacy_without_hook_is_unchanged():
    tree = ast.parse((SOURCE / "ray_trainer.py").read_text())
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
    fit = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == "fit")
    call_lines = {}
    for node in ast.walk(fit):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            call_lines.setdefault(node.func.attr, []).append(node.lineno)
    assert max(call_lines["_validate_benchmark_before_update"]) < min(call_lines["compute_ref_log_prob"] + call_lines["update_critic"] + call_lines["update_actor"])
    method = _load_failure_methods()._validate_benchmark_before_update
    method(SimpleNamespace(), batch=None, diagnostics=None, replay_error=float("nan"), actor_log_probs=None, comparison_mask=None, iteration=0, timing={}, metrics={}, started_at=0)


def test_optional_action_fingerprints_preserve_legacy_hashes_and_support_bf16_metadata():
    tree = ast.parse((SOURCE / "qwen_benchmark.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_request_action_fingerprints")
    namespace = {"hashlib": hashlib, "torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "fingerprints", "exec"), namespace)
    fingerprint = namespace["_request_action_fingerprints"]
    inputs = replay_inputs()
    batch = {key: inputs[key] for key in ("responses", "rollout_topk_ids", "rollout_topk_gumbels", "rollout_log_probs")}
    valid = inputs["response_mask"][0]
    original = fingerprint(batch, 0, valid)
    assert original["tokens_sha256"] == hashlib.sha256(batch["responses"][0][valid].numpy().tobytes()).hexdigest()
    assert set(original) == {"tokens_sha256", "support_sha256", "perturbations_sha256", "log_probs_sha256"}
    for key in ("rollout_topk_probs", "rollout_topk_gumbel_noise", "rollout_topk_retained_mask"):
        batch[key] = inputs[key]
    batch["rollout_topk_probs"] = batch["rollout_topk_probs"].bfloat16()
    # Prefix support metadata belongs to the prompt and must not be hashed.
    for key in ("rollout_topk_ids", "rollout_topk_gumbels", "rollout_topk_probs", "rollout_topk_gumbel_noise", "rollout_topk_retained_mask"):
        tensor = batch[key]
        batch[key] = torch.cat((torch.zeros_like(tensor[:, :2]), tensor), dim=1)
    extended = fingerprint(batch, 0, valid)
    assert {key: extended[key] for key in original} == original
    assert set(extended) - set(original) == {"probabilities_sha256", "retained_mask_sha256", "raw_noise_sha256"}
    batch["rollout_topk_probs"][0, 2, 0] += 0.125
    changed = fingerprint(batch, 0, valid)
    assert changed["probabilities_sha256"] != extended["probabilities_sha256"]
    assert changed["support_sha256"] == extended["support_sha256"]
