"""Production-only trainer instrumentation, numerical policy and invocation semantics."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import sys

from omegaconf import OmegaConf
import pytest

from verl.trainer.ppo.opd_driver import validate_production_gradient_integrity
from verl.trainer.ppo.qwen_production import attach_production_recorder, remaining_runtime_estimate


def trainer_stub(tmp_path, phase="uninterrupted"):
    config = OmegaConf.create({
        "trainer": {"training_profile": "qwen3-math-seven-arm-v1", "production_mode": True,
                    "production_phase": phase, "production_arm_id": "softgrpo_math_opd_s11",
                    "production_output": str(tmp_path / "measurement.json"),
                    "production_gradient_policy": "diagnostic_clipping",
                    "n_gpus_per_node": 4, "nnodes": 1,
                    "rollout_integrity": {"enabled": True, "completion_gate_enabled": False, "full_dose_gradient_gate_enabled": False},
                    "val_before_train": phase == "production", "test_freq": 25 if phase == "production" else -1,
                    "save_freq": 25 if phase == "production" else 1,
                    "max_rollout_iterations_per_invocation": None if phase == "production" else 2,
                    "default_local_dir": str(tmp_path / "training")},
        "actor_rollout_ref": {"actor": {"grad_clip": 1.0}}, "data": {"val_batch_size": 128},
    })
    return SimpleNamespace(config=config, total_rollout_iterations=109, optimizer_steps_per_rollout=2,
                           train_dataset=range(6985), val_dataset=range(512), global_steps=0,
                           checkpoint_provenance={"identity": "committed"},
                           fit=lambda: None, init_workers=lambda: None, _save_checkpoint=lambda: None,
                           _validate=lambda: {"val/math_verify/mean_at_1": 0.5})


@pytest.mark.parametrize("phase", ["production", "uninterrupted", "split", "resume", "full_dose", "zero_dose"])
def test_recorder_wraps_ordinary_trainer_and_persists_complete_iteration(tmp_path, monkeypatch, phase):
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(run=None))
    trainer = trainer_stub(tmp_path, phase)
    def fit():
        trainer.record_capacity_stage("update_actor", 0, {"gen": 1.0}, {}, {}, update_state="outcome_unknown")
        trainer.record_capacity_stage("iteration_metrics", 0, {}, {}, {}, update_state="completed")
        trainer.record_benchmark_iteration(0, {"update_actor": 2.0}, {"grad/opd_norm": 0.0}, {
            "capacity_rollout_trajectory_count": 512,
            "actor_update_timing": {"ranks": [{"rank": rank} for rank in range(4)]},
        })
    trainer.fit = fit
    attach_production_recorder(trainer)
    trainer.init_workers()
    trainer.fit()
    measured = json.loads((tmp_path / "measurement.json").read_text())
    assert measured["status"] == "complete"
    assert measured["phase"] == phase
    assert measured["completed_rollout_iterations"] == 1
    assert measured["iterations"][0]["trajectory_count"] == 512
    assert len(measured["iterations"][0]["actor_update_timing"]["ranks"]) == 4
    assert measured["progress"]["optimizer_updates_completed"] is True
    assert measured["checkpoint_provenance"] == trainer.checkpoint_provenance


@pytest.mark.parametrize("when", ["worker_initialization", "update_actor"])
def test_recorder_preserves_failure_without_claiming_update_or_checkpoint(tmp_path, monkeypatch, when):
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(run=None))
    trainer = trainer_stub(tmp_path)
    def fail():
        if when == "update_actor":
            trainer.record_capacity_stage(when, 0, {}, {}, {}, update_state="outcome_unknown")
        raise RuntimeError("CUDA out of memory")
    if when == "worker_initialization":
        trainer.init_workers = fail
    else:
        trainer.fit = fail
    attach_production_recorder(trainer)
    with pytest.raises(RuntimeError, match="out of memory"):
        (trainer.init_workers if when == "worker_initialization" else trainer.fit)()
    measured = json.loads((tmp_path / "measurement.json").read_text())
    assert measured["status"] == "failed"
    assert measured["failure"]["category"] == "oom"
    assert measured["failure"]["stage"] == when
    assert not measured["iterations"] and not measured["checkpoints"]


@pytest.mark.parametrize("key,value", [
    ("trainer.training_profile", "qwen3-training-benchmark-v1"),
    ("trainer.production_gradient_policy", "ignore_nans"),
    ("trainer.rollout_integrity.enabled", False),
    ("trainer.rollout_integrity.completion_gate_enabled", True),
    ("trainer.rollout_integrity.full_dose_gradient_gate_enabled", True),
    ("actor_rollout_ref.actor.grad_clip", 0.5),
    ("trainer.production_phase", "unknown"),
    ("trainer.test_freq", 10), ("trainer.save_freq", 10),
    ("trainer.max_rollout_iterations_per_invocation", 3), ("data.val_batch_size", 512),
])
def test_production_admission_rejects_changed_recipe_and_policies(tmp_path, key, value):
    trainer = trainer_stub(tmp_path, "production")
    OmegaConf.update(trainer.config, key, value)
    with pytest.raises(ValueError):
        attach_production_recorder(trainer)
    assert not (tmp_path / "measurement.json").exists()


@pytest.mark.parametrize("owner", ["trainer", "actor_rollout_ref"])
@pytest.mark.parametrize("policy", [None, {}, {"mode": "legacy_allocator_v1"},
    {"mode": "physical_device_v1", "max_device_used_fraction": 1.0, "sample_interval_seconds": 0.1},
    {"mode": "physical_device_v1", "max_device_used_fraction": 0.98, "sample_interval_seconds": 10.0}])
def test_revised_profile_rejects_missing_or_changed_resource_policy(tmp_path, owner, policy):
    trainer = trainer_stub(tmp_path)
    trainer.config.trainer.training_profile = "qwen3-math-seven-arm-lora-fa3-v1"
    expected = {"mode": "physical_device_v1", "max_device_used_fraction": 0.98, "sample_interval_seconds": 0.1}
    trainer.config.trainer.resource_policy = expected
    trainer.config.actor_rollout_ref.resource_policy = expected
    if policy is None:
        del trainer.config[owner].resource_policy
    else:
        trainer.config[owner].resource_policy = policy
    with pytest.raises(ValueError, match="physical-device resource policy"):
        attach_production_recorder(trainer)
    assert not (tmp_path / "measurement.json").exists()


def test_revised_profile_records_independent_physical_resource_policy(tmp_path):
    trainer = trainer_stub(tmp_path)
    trainer.config.trainer.training_profile = "qwen3-math-seven-arm-lora-fa3-v1"
    expected = {"mode": "physical_device_v1", "max_device_used_fraction": 0.98, "sample_interval_seconds": 0.1}
    trainer.config.trainer.resource_policy = expected
    trainer.config.actor_rollout_ref.resource_policy = expected
    attach_production_recorder(trainer)
    measured = json.loads((tmp_path / "measurement.json").read_text())
    assert measured["acceptance_policy"]["resource_policy"] == expected
    assert measured["acceptance_policy"]["optimizer_clip_norm"] == 1.0


def test_recorder_does_not_overwrite_a_prior_invocation(tmp_path):
    path = tmp_path / "measurement.json"
    path.write_text("prior")
    with pytest.raises(ValueError, match="fresh"):
        attach_production_recorder(trainer_stub(tmp_path))
    assert path.read_text() == "prior"


def test_validation_records_full_population_and_completed_rollout_axis(tmp_path):
    trainer = trainer_stub(tmp_path, "production")
    attach_production_recorder(trainer)
    for completed in (0, 25, 50, 75, 100, 109):
        trainer.global_steps = completed
        assert trainer._validate()["val/math_verify/mean_at_1"] == 0.5
    measured = json.loads((tmp_path / "measurement.json").read_text())
    assert [row["completed_rollout_iterations"] for row in measured["validations"]] == [0, 25, 50, 75, 100, 109]
    assert {row["example_count"] for row in measured["validations"]} == {512}


@pytest.mark.parametrize("standalone", [False, True])
@pytest.mark.parametrize("clip_frequency", [0.0, 0.5, 1.0])
def test_production_allows_clipping_frequency_and_eligible_empty_components(standalone, clip_frequency):
    metrics = {"actor/grad_norm": 0.0, "actor/gradient_clipfrac": clip_frequency, "grad/opd_norm": 0.0}
    if not standalone:
        metrics["grad/grpo_norm"] = 0.0
    validate_production_gradient_integrity(metrics, standalone=standalone)


@pytest.mark.parametrize("name", ["actor/grad_norm", "actor/gradient_clipfrac", "grad/opd_norm", "grad/grpo_norm"])
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -0.1])
def test_production_retains_finite_gradient_and_metric_checks(name, bad):
    metrics = {"actor/grad_norm": 0.1, "actor/gradient_clipfrac": 1.0, "grad/opd_norm": 0.0, "grad/grpo_norm": 0.1}
    if bad is None:
        metrics.pop(name)
    else:
        metrics[name] = bad
    with pytest.raises(RuntimeError):
        validate_production_gradient_integrity(metrics)


@pytest.mark.parametrize("completed,continues", [(108, True), (109, False), (110, False)])
@pytest.mark.parametrize("semantic_mode", [None, "qwen_semantic_v1"])
def test_real_fit_resume_guard_stops_before_duplicate_validation_or_rollout(completed, continues, semantic_mode, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "verl/trainer/ppo/ray_trainer.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
    fit = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "fit")
    start = next(index for index, node in enumerate(fit.body) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Attribute) and target.attr == "global_steps" for target in node.targets))
    stop = next(index for index, node in enumerate(fit.body[start:], start) if isinstance(node, ast.If) and "total_training_steps" in ast.unparse(node.test))
    method = ast.FunctionDef(name="guard", args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="self")], kwonlyargs=[], kw_defaults=[], defaults=[]), body=fit.body[start:stop + 1] + [ast.Return(value=ast.Constant(value=True))], decorator_list=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), "exec"), namespace)
    trainer = SimpleNamespace(global_steps=0, total_training_steps=109,
                              config=SimpleNamespace(trainer={"checkpoint_semantics": semantic_mode},
                                                     data=SimpleNamespace(seed=11)))
    calls = []
    monkeypatch.setattr('verl.opd.rng_state.seed_training_rng', lambda seed, **kw: calls.append(('seed', seed, kw)))
    def load():
        calls.append(('load',))
        trainer.global_steps = completed
    trainer._load_checkpoint = load
    assert bool(namespace["guard"](trainer)) is continues
    assert calls == ([('seed', 11, {'namespace': 'driver'})] if semantic_mode else []) + [('load',)]


def test_remaining_runtime_uses_active_critical_path_without_double_counting_overhead():
    measurement = {"completed_rollout_iterations": 2,
                   "configuration": {"algorithm": {"opd": {"enabled": True, "mode": "auxiliary"}}},
                   "validations": [{"timing_seconds": 50}],
                   "iterations": [
                       {"rollout_iteration": 0, "timing_s": {"step": 120, "save_checkpoint": 20, "gen": 60, "old_log_prob": 10, "ref": 5, "update_actor": 10}},
                       {"rollout_iteration": 1, "timing_s": {"step": 170, "testing": 50, "gen": 60, "old_log_prob": 10, "ref": 5, "update_actor": 30},
                        "actor_update_timing": {"ranks": [{"teacher_seconds": 10000} for _ in range(4)]}},
                   ]}
    result = remaining_runtime_estimate(measurement, gpus=4)
    assert result["complete"]
    assert result["active_core_mean_seconds"] == 120
    assert result["zero_dose_core_mean_seconds"] == 100
    assert result["remaining_rollout_iterations"] == 107
    assert result["remaining_scheduled_validations"] == result["remaining_scheduled_checkpoints"] == 5
    assert result["scenarios"]["central"]["remaining_seconds"] == 13190
    assert result["scenarios"]["token_1p5x"]["remaining_seconds"] == 18932.5
    assert result["scenarios"]["token_2x"]["remaining_seconds"] == 24675
    assert result["scenarios"]["central"]["remaining_gpu_hours"] == pytest.approx(4 * 13190 / 3600)


def test_remaining_runtime_labels_missing_active_and_checkpoint_costs_incomplete():
    measurement = {"completed_rollout_iterations": 1,
                   "configuration": {"algorithm": {"opd": {"enabled": True, "mode": "auxiliary"}}},
                   "iterations": [{"rollout_iteration": 0, "timing_s": {"step": 100, "gen": 80, "update_actor": 20}}]}
    result = remaining_runtime_estimate(measurement, gpus=4)
    assert not result["complete"]
    assert result["missing_stage_measurements"] == ["active_iteration", "validation", "checkpoint"]
    assert result["scenarios"]["central"]["remaining_seconds"] is None


@pytest.mark.parametrize("completed,events", [(24, 5), (25, 4), (100, 1), (109, 0)])
def test_remaining_runtime_counts_only_future_validation_and_save_events(completed, events):
    result = remaining_runtime_estimate({"completed_rollout_iterations": completed}, gpus=4)
    assert result["remaining_scheduled_validations"] == result["remaining_scheduled_checkpoints"] == events
    if completed == 109:
        assert result["complete"] and result["scenarios"]["central"]["remaining_seconds"] == 0
