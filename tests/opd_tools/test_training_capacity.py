"""Capacity contracts and failure persistence without CUDA, Ray or assets."""

import ast
import copy
import json
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import pytest

from opd_tools import training_capacity as capacity
from opd_tools.manifest import file_sha256


def valid_record():
    return {
        "rollout_iteration": 0, "trajectory_count": 512,
        "metrics": {
            "integrity/continuous_replay_active": 1, "replay/fallback_count": 0,
            "trainer/rollout_iteration": 0, "trainer/optimizer_steps_this_iteration": 2,
            "trainer/optimizer_step": 2, "opd/ema_updates_this_iteration": 1,
            "opd/ema_update_count": 1, "opd/beta_effective": 0.001,
            "replay/ratio_abs_error_max": 1e-4, "latent/soft_to_hard_rate": 0.02,
            "opd/latent_slot_count": 10, "opd/answer_slot_count": 1,
            "grad/opd_norm": 0.02, "grad/grpo_norm": 0.1,
            "grad/total_norm": 0.5, "actor/gradient_clipfrac": 0.5,
        },
        "actor_update_timing": {"ranks": [
            {"rank": rank, "optimizer_steps": 2, "ema_updates_this_iteration": 1,
             "ema_update_count": 1, "teacher_seconds": 2, "policy_update_seconds": 5,
             "worker_update_seconds": 7, "max_memory_allocated_gib": 70,
             "max_memory_reserved_gib": 74} for rank in range(2)
        ]},
    }


def test_contract_is_fresh_and_full_dose_single_invocation():
    first = capacity.capacity_contract()
    first["group_size"] = 1
    contract = capacity.capacity_contract()
    assert contract["group_size"] == 8
    assert contract["total_rollout_iterations"] == 109
    assert contract["invocation_iterations"] == 1
    assert contract["optimizer_steps"] == 2
    assert contract["ema_updates"] == 1
    assert contract["maximum_process_seconds"] == 1700
    assert contract["full_dose_gradient_gate_enabled"] is True
    assert contract["completion_gate_enabled"] is False


def test_capacity_overrides_keep_recipe_and_do_not_select_benchmark(tmp_path):
    values = dict(item.lstrip("+").split("=", 1) for item in capacity.capacity_overrides(tmp_path / "assets", tmp_path / "run"))
    for key, expected in {
        "data.train_batch_size": "64", "data.max_response_length": "8192",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": "2",
        "actor_rollout_ref.rollout.n": "8", "trainer.n_gpus_per_node": "2",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": "1",
        "trainer.total_training_steps": "null", "trainer.total_epochs": "1",
        "algorithm.opd.schedule": "constant", "algorithm.opd.beta_base": "0.001",
        "trainer.max_rollout_iterations_per_invocation": "1",
        "trainer.rollout_integrity.full_dose_gradient_gate_enabled": "true",
        "trainer.rollout_integrity.completion_gate_enabled": "false",
        "trainer.val_before_train": "false", "trainer.test_freq": "-1",
        "trainer.save_freq": "-1", "trainer.resume_mode": "disable",
        "actor_rollout_ref.model.qwen_replay_backend": "native_fa3_v1",
        "actor_rollout_ref.rollout.dispatch_mode": "bounded_async",
        "actor_rollout_ref.rollout.max_running_requests": "32",
        "actor_rollout_ref.rollout.async_queue_size": "64",
    }.items():
        assert values[key] == expected
    assert not any("training_benchmark_mode" in key for key in values)


def test_full_dose_capacity_accepts_low_completion_and_requires_real_boundary():
    record = valid_record()
    record["metrics"]["rollout/think_end_rate"] = 0.01
    assert capacity.validate_capacity_iteration(record)["accepted"]


@pytest.mark.parametrize("key,value", [
    ("opd/beta_effective", 0), ("opd/beta_effective", 0.001 / 11),
    ("trainer/optimizer_step", 6), ("opd/ema_update_count", 3),
    ("integrity/continuous_replay_active", 0), ("replay/fallback_count", 1),
    ("replay/ratio_abs_error_max", 1.01e-4), ("latent/soft_to_hard_rate", 0),
    ("grad/grpo_norm", 0), ("grad/opd_norm", 0.001),
    ("actor/gradient_clipfrac", 0.51), ("grad/total_norm", float("nan")),
    ("opd/answer_slot_count", 0), ("opd/latent_slot_count", True),
])
def test_capacity_rejects_invalid_production_evidence(key, value):
    record = valid_record()
    record["metrics"][key] = value
    with pytest.raises(ValueError):
        capacity.validate_capacity_iteration(record)


@pytest.mark.parametrize("mutation", ["trajectory_count", "duplicate_rank", "missing_rank", "rank_update", "memory_nan", "teacher_zero"])
def test_capacity_requires_actual_g8_population_and_rank_evidence(mutation):
    record = valid_record()
    ranks = record["actor_update_timing"]["ranks"]
    if mutation == "trajectory_count":
        record["trajectory_count"] = 64
    elif mutation == "duplicate_rank":
        ranks[1]["rank"] = 0
    elif mutation == "missing_rank":
        ranks.pop()
    elif mutation == "rank_update":
        ranks[0]["optimizer_steps"] = 1
    elif mutation == "memory_nan":
        ranks[1]["max_memory_allocated_gib"] = float("nan")
    else:
        ranks[0]["teacher_seconds"] = 0
    with pytest.raises(ValueError):
        capacity.validate_capacity_iteration(record)


@pytest.mark.parametrize("error,stage,log,category", [
    (RuntimeError("RayTaskError"), "update_actor", "CUDA out of memory", "oom"),
    (RuntimeError("fail"), "old_log_prob", "", "replay"),
    (RuntimeError("full-dose gradient integrity gate failed"), "full_dose_gradient_gate", "", "gradient_gate"),
    (TimeoutError("deadline"), "update_actor", "out of memory", "timeout"),
    (RuntimeError("fail"), "checkpoint", "", "checkpoint"),
    (RuntimeError("bad hash"), "asset_authentication", "", "authentication"),
    (RuntimeError("fail"), "worker_initialization", "", "startup"),
])
def test_failure_categories_keep_underlying_cause_and_stage(error, stage, log, category):
    result = capacity.classify_failure(error, stage=stage, log_tail=log)
    assert (result["category"], result["stage"]) == (category, stage)


@pytest.mark.parametrize("state,completed", [("not_started", False), ("outcome_unknown", None), ("completed", True)])
def test_failure_snapshot_preserves_post_update_and_late_metrics(tmp_path, state, completed):
    path = tmp_path / "measurement.json"
    recorder = capacity.CapacityRecorder(path, {"status": "running"})
    metrics = {f"padding/{index}": index for index in range(140)}
    metrics["grad/opd_norm"] = 0.02
    timing = {"gen": 30}
    meta = {"actor_update_timing": valid_record()["actor_update_timing"]}
    recorder.enter("update_actor", 0, timing, metrics, meta, update_state=state)
    metrics["actor/gradient_clipfrac"] = 1.0
    timing["update_actor"] = 40
    recorder.fail(RuntimeError("full-dose gradient gate failed"))
    result = json.loads(path.read_text())["failure"]
    assert result["optimizer_updates_completed"] is completed
    assert result["optimizer_update_state"] == state
    assert result["metrics"]["actor/gradient_clipfrac"] == 1.0
    assert result["metrics"]["grad/opd_norm"] == 0.02
    assert result["timing_s"]["update_actor"] == 40
    assert len(result["actor_update_timing"]["ranks"]) == 2
    assert result["stage_elapsed_seconds"] >= 0


def test_report_publishes_hashes_and_distinct_memory_scopes(tmp_path):
    report = {"status": "failed", "failure": {"category": "oom", "stage": "update_actor"},
              "source": {"parent_commit": "a" * 40, "fork_commit": "b" * 40},
              "resource_telemetry": {"peak_hbm_gib_per_gpu": {"0": 79}, "peak_host_ram_gib": 112},
              "measurement": {"progress": {"optimizer_update_state": "outcome_unknown"}}}
    capacity.publish_report(tmp_path, report)
    for filename in ("capacity.json", "REPORT.md"):
        assert (tmp_path / (filename + ".sha256")).read_text().split()[0] == file_sha256(tmp_path / filename)
    text = (tmp_path / "REPORT.md").read_text()
    assert "oom" in text and "outcome_unknown" in text and "sampled" in text and "node" in text
    assert "a" * 40 in text


def test_cleanup_kills_group_even_if_leader_already_exited(tmp_path, monkeypatch):
    runner = capacity.CapacityRunner(SimpleNamespace(run_root=tmp_path / "run", time_limit_seconds=100))
    signals = []
    monkeypatch.setattr(capacity.os, "killpg", lambda pid, signum: signals.append((pid, signum)))
    runner.child = SimpleNamespace(pid=123, wait=lambda timeout: 0)
    runner._terminate()
    assert signals == [(123, signal.SIGTERM), (123, signal.SIGKILL)]
    assert runner.child is None


@pytest.mark.parametrize("limit", [0, 59, 1701, 1800])
def test_cli_prohibits_over_budget_and_too_short_invocations(limit, tmp_path):
    with pytest.raises(SystemExit) as error:
        capacity.main(["run", "--assets-root", str(tmp_path), "--run-root", str(tmp_path / "run"), "--time-limit-seconds", str(limit)])
    assert error.value.code == 2


def test_actor_completion_is_persisted_before_full_dose_gate_and_checkpoint():
    """Check executable hook ordering, not just presence of callback strings."""
    source = Path(__file__).resolve().parents[2] / "verl-0.4.x/verl/trainer/ppo/ray_trainer.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
    fit = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "fit")
    calls = sorted((node for node in ast.walk(fit) if isinstance(node, ast.Call)), key=lambda node: node.lineno)
    actor = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "update_actor")
    gate = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "validate_full_dose_gradient_integrity")
    checkpoint = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "_save_checkpoint")
    hook = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "capacity_stage" and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "full_dose_gradient_gate")
    assert actor.lineno < hook.lineno < gate.lineno < checkpoint.lineno
    assert next(keyword.value.value for keyword in hook.keywords if keyword.arg == "update_state") == "completed"


@pytest.mark.parametrize("category,stage,state,monitor_fails", [
    ("oom", "update_actor", "outcome_unknown", False),
    ("oom", "update_actor", "outcome_unknown", True),
    ("replay", "old_log_prob", "not_started", False),
    ("gradient_gate", "full_dose_gradient_gate", "completed", False),
    ("timeout", "generation", "not_started", False),
])
def test_runner_failure_is_sealed_without_retry_and_cleanup_preserves_cause(tmp_path, monkeypatch, category, stage, state, monitor_fails):
    from opd_tools import icl_resource_monitor, qwen_training, training_benchmark

    root = tmp_path / "run"
    calls = []
    signals = []
    auth_calls = []
    monkeypatch.setattr(training_benchmark, "source_identity", lambda path: {"parent_commit": "a" * 40, "fork_commit": "b" * 40, "snapshot_verified": True})
    monkeypatch.setattr(qwen_training, "verify", lambda path: {"verified": True})
    monkeypatch.setattr(training_benchmark, "validate_phase_measurement", lambda measured, **kwargs: auth_calls.append(kwargs))
    monkeypatch.setattr(capacity.signal, "setitimer", lambda *args: None)
    monkeypatch.setattr(capacity.os, "killpg", lambda pid, signum: signals.append((pid, signum)))

    class Monitor:
        def __init__(self, **kwargs):
            pass
        def start(self):
            return self
        def stop(self):
            if monitor_fails:
                raise RuntimeError("monitor cannot join")
            return SimpleNamespace(to_dict=lambda: {"peak_hbm_gib_per_gpu": {"0": 79, "1": 78}, "peak_host_ram_gib": 125})

    monkeypatch.setattr(icl_resource_monitor, "ResourceMonitor", Monitor)

    class Child:
        pid = 123
        waits = 0
        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                assert 0 < timeout <= 55
                if category == "timeout":
                    raise subprocess.TimeoutExpired("trainer", timeout)
                return 1
            return 0

    def spawn(command, **kwargs):
        calls.append(command)
        assert kwargs["start_new_session"] is True
        assert kwargs["env"]["WANDB_MODE"] == "online"
        message = {"oom": "CUDA out of memory", "gradient_gate": "full-dose gradient integrity gate failed", "replay": "continuous replay acceptance failed", "timeout": "pending requests"}[category]
        kwargs["stdout"].write(message)
        kwargs["stdout"].flush()
        recorder = capacity.CapacityRecorder(root / "measurement.json", {"status": "running", "iterations": []})
        metrics = valid_record()["metrics"]
        recorder.enter(stage, 0, {"gen": 30}, metrics, {"actor_update_timing": valid_record()["actor_update_timing"] if state == "completed" else {}}, update_state=state)
        if category != "timeout":
            recorder.fail(RuntimeError(message))
        return Child()

    monkeypatch.setattr(capacity.subprocess, "Popen", spawn)
    runner = capacity.CapacityRunner(SimpleNamespace(run_root=root, assets_root=tmp_path / "assets", time_limit_seconds=100))
    assert runner.run() == 1
    assert len(calls) == 1
    assert signals == [(123, signal.SIGTERM), (123, signal.SIGKILL)]
    result = json.loads((root / "capacity.json").read_text())
    assert result["failure"]["category"] == category
    assert result["failure"]["stage"] == stage
    assert result["measurement"]["progress"]["optimizer_update_state"] == state
    assert result["measurement"]["progress"]["optimizer_updates_completed"] is {"not_started": False, "completed": True}.get(state)
    assert result["checkpoint_authenticated"] is False
    assert result["measurement_authenticated"] is True
    assert auth_calls[0]["phase"] == "capacity"
    assert not result["measurement"]["iterations"]
    assert (root / "capacity.json.sha256").read_text().split()[0] == file_sha256(root / "capacity.json")
    if monitor_fails:
        assert "cannot join" in result["resource_telemetry"]["monitor_error"]


def test_capacity_checkpoint_authentication_is_real_and_persisted_before_rehash():
    source = Path(__file__).resolve().parents[2] / "verl-0.4.x/verl/trainer/ppo/qwen_capacity.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_save_checkpoint")
    events = []
    namespace = {
        "Path": Path, "file_sha256": lambda path: "a" * 64,
        "_verify_checkpoint": lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("payload hash mismatch")),
    }
    # Compile the real subclass method against a fake successful save. A failed
    # full verifier must prevent the authenticated flag and preserve its stage.
    module = ast.Module(body=[ast.ClassDef(name="Capacity", bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[])], type_ignores=[])
    class Base:
        def _save_checkpoint(self, *args, **kwargs):
            events.append("save")
            return {}
    namespace["Base"] = Base
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    trainer = namespace["Capacity"]()
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(default_local_dir="/run/training"))
    trainer.checkpoint_provenance = {"identity": "pinned"}
    trainer.capacity = SimpleNamespace(current=(0, {}, {}, {}), enter=lambda stage, *args: events.append(stage), measurement={})
    with pytest.raises(RuntimeError, match="payload hash mismatch"):
        trainer._save_checkpoint()
    assert events == ["save", "checkpoint_authentication"]
    assert "checkpoint" not in trainer.capacity.measurement
