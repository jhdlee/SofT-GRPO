"""A new rollout cannot inherit the prior rollout's optimizer outcome."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

from opd_tools.training_capacity import CapacityRecorder


def test_second_rollout_timeout_records_no_current_updates_and_keeps_prior_evidence(tmp_path):
    source = Path(__file__).resolve().parents[2] / "verl-0.4.x/verl/trainer/ppo/ray_trainer.py"
    tree = ast.parse(source.read_text())
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "capacity_stage" and node.args
             and isinstance(node.args[0], ast.Constant) and node.args[0].value == "generation"]
    assert len(calls) == 1
    first = {"rollout_iteration": 0, "optimizer_steps": 2, "ema_updates": 1}
    measurement = {"iterations": [first.copy()]}
    path = tmp_path / "measurement.json"
    recorder = CapacityRecorder(path, measurement)
    recorder.enter("full_dose_gradient_gate", 0, {}, {}, {}, update_state="completed")
    assert recorder.snapshot()["optimizer_updates_completed"] is True

    # Execute the real trainer's entry call without importing Ray/actor workers.
    entry = ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=calls[0])], type_ignores=[]))
    exec(compile(entry, str(source), "exec"), {
        "capacity_stage": recorder.enter, "rollout_iteration": 1,
        "timing_raw": {}, "metrics": {}, "gen_batch": SimpleNamespace(meta_info={}),
    })
    durable = json.loads(path.read_text())["progress"]
    assert durable["rollout_iteration"] == 1
    assert durable["stage"] == "generation"
    assert durable["optimizer_update_state"] == "not_started"
    assert durable["optimizer_updates_completed"] is False

    recorder.fail(TimeoutError("allocation deadline during second rollout"))
    failed = json.loads(path.read_text())
    assert failed["failure"]["stage"] == "generation"
    assert failed["failure"]["optimizer_update_state"] == "not_started"
    assert failed["failure"]["optimizer_updates_completed"] is False
    assert failed["iterations"] == [first]
