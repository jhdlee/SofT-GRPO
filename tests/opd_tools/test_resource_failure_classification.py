"""Resource policy failures remain distinct from OOM and update outcomes."""
import json

import pytest

from opd_tools.training_capacity import CapacityRecorder, classify_failure


MESSAGES = (
    "physical-device resource gate failed: rank 1 used 99% >= 98%",
    "resource-integrity metrics must be finite",
    "rollout-integrity HBM gate failed: 88.700 GiB >= 72.000 GiB",
    "rollout-integrity host-RAM gate failed: 91.000% >= 90.000%",
)


@pytest.mark.parametrize("message", MESSAGES)
@pytest.mark.parametrize("stage", ("update_actor", "old_log_prob"))
@pytest.mark.parametrize("wrapped", (False, True))
def test_resource_gate_survives_ray_wrapping_and_precedes_stage_fallback(message, stage, wrapped):
    if wrapped:
        error = RuntimeError("ray::WorkerDict.actor_rollout_update_actor() RayTaskError(RuntimeError)")
        tail = "Traceback: collective post-policy check failed: rank 1 RuntimeError: " + message
    else:
        error, tail = RuntimeError(message), ""
    result = classify_failure(error, stage=stage, log_tail=tail)
    assert result["category"] == "resource_gate"
    assert result["stage"] == stage and result["error_type"] == "RuntimeError"


@pytest.mark.parametrize("oom", ("torch.OutOfMemoryError", "CUDA out of memory", "oom-kill", "oom_kill"))
def test_actual_oom_takes_precedence_over_resource_policy_context(oom):
    error = RuntimeError("physical-device resource gate measurement failed; " + oom)
    assert classify_failure(error, stage="update_actor")["category"] == "oom"


@pytest.mark.parametrize("message,stage,expected", (
    ("memory telemetry was recorded; replay ratio exceeded tolerance", "old_log_prob", "replay"),
    ("actor memory diagnostic attached; optimizer failed", "update_actor", "execution"),
    ("full-dose gradient integrity gate failed", "full_dose_gradient_gate", "gradient_gate"),
))
def test_generic_memory_words_do_not_change_existing_failure_categories(message, stage, expected):
    assert classify_failure(RuntimeError(message), stage=stage)["category"] == expected


def test_resource_failure_does_not_infer_optimizer_completion_from_rpc_error(tmp_path):
    path = tmp_path / "measurement.json"
    prior = {"rollout_iteration": 0, "optimizer_steps": 2}
    recorder = CapacityRecorder(path, {"iterations": [prior]})
    recorder.enter("update_actor", 1, {"gen": 400.0}, {}, {}, update_state="outcome_unknown")
    recorder.fail(RuntimeError("RayTaskError(RuntimeError): physical-device resource gate failed"))
    result = json.loads(path.read_text())
    assert result["failure"]["category"] == "resource_gate"
    assert result["failure"]["optimizer_update_state"] == "outcome_unknown"
    assert result["failure"]["optimizer_updates_completed"] is None
    assert result["failure"]["rollout_iteration"] == 1
    assert result["iterations"] == [prior]
