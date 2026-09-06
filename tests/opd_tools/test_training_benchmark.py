from __future__ import annotations

import copy
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import opd_tools.training_benchmark as benchmark
from opd_tools.manifest import canonical_sha256
from opd_tools.training_runtime import MODEL_ID, MODEL_REVISION


def fingerprint(index=0, **changes):
    return {"example_id": f"math-{index}", "sample_index": 0, "prompt_ids_sha256": str(index) * 64, "request_seed": 123 + index, "token_count": 10,
            **{key + "_sha256": "a" * 64 for key in ("tokens", "support", "perturbations", "log_probs")}, **changes}


def test_fingerprint_comparison_tracks_output_differences_separately_from_identity():
    baseline = {"requests": [fingerprint(), fingerprint(1)]}
    candidate = copy.deepcopy(baseline)
    candidate["requests"][1].update(token_count=11, tokens_sha256="b" * 64)
    result = benchmark.compare_fingerprints(baseline, candidate)
    assert result["paired_requests"] == result["matching_request_seeds"] == 2
    assert result["token_count_different_requests"] == result["tokens_sha256_different_requests"] == 1
    assert result["support_sha256_different_requests"] == 0


def test_fingerprint_comparison_distinguishes_missing_new_metadata_from_equality():
    baseline = {"requests": [fingerprint(), fingerprint(1, probabilities_sha256="a" * 64)]}
    candidate = {"requests": [fingerprint(probabilities_sha256="a" * 64), fingerprint(1, probabilities_sha256="b" * 64)]}
    optional = benchmark.compare_fingerprints(baseline, candidate)["optional_replay_metadata"]
    assert optional["probabilities_sha256"] == {
        "paired_requests": 1, "different_requests": 1, "baseline_missing": 1, "candidate_missing": 0,
    }
    assert optional["retained_mask_sha256"]["different_requests"] is None
    assert optional["retained_mask_sha256"]["paired_requests"] == 0


@pytest.mark.parametrize("mutation", ["seed", "prompt", "sample", "order", "missing", "duplicate"])
def test_fingerprints_reject_wrong_or_incomplete_prompt_sample_identity(mutation):
    baseline = {"requests": [fingerprint(), fingerprint(1)]}
    candidate = copy.deepcopy(baseline)
    if mutation == "seed":
        candidate["requests"][0]["request_seed"] += 1
    elif mutation == "prompt":
        candidate["requests"][0]["example_id"] = "wrong"
    elif mutation == "sample":
        candidate["requests"][0]["sample_index"] = 1
    elif mutation == "order":
        candidate["requests"].reverse()
    elif mutation == "missing":
        del candidate["requests"][0]["prompt_ids_sha256"]
    else:
        baseline["requests"][1] = baseline["requests"][0]
        candidate = copy.deepcopy(baseline)
    with pytest.raises(ValueError):
        benchmark.compare_fingerprints(baseline, candidate)


def pilot_metrics(objective="standalone", iteration=0):
    active = objective == "standalone" or iteration > 0
    return {
        "integrity/continuous_replay_active": 1.0,
        "replay/fallback_count": 0.0,
        "replay/ratio_abs_error_max": 1e-4,
        "latent/soft_to_hard_rate": 0.25,
        "trainer/rollout_iteration": iteration,
        "trainer/optimizer_steps_this_iteration": 2.0,
        "trainer/optimizer_step": 2 * (iteration + 1),
        "opd/ema_updates_this_iteration": 1.0,
        "opd/ema_update_count": iteration + 1,
        "opd/beta_effective": 1.0 if objective == "standalone" else 0.001 * (iteration / 11),
        "opd/latent_slot_count": 120.0 if active else 0.0,
        "opd/answer_slot_count": 48.0 if active else 0.0,
        "grad/opd_norm": 0.01 if active else 0.0,
        "grad/total_norm": 0.3,
    }


@pytest.mark.parametrize("objective", ["standalone", "hybrid"])
@pytest.mark.parametrize("iteration", range(3))
def test_pilot_accepts_exact_full_horizon_schedule_and_update_evidence(objective, iteration):
    metrics = pilot_metrics(objective, iteration)
    original = copy.deepcopy(metrics)
    evidence = benchmark.validate_pilot_metrics(objective, iteration, metrics)
    assert evidence == {
        "accepted": True, "objective": objective, "rollout_iteration": iteration,
        "opd_active": objective == "standalone" or iteration > 0, "metrics": original,
    }
    assert metrics == original
    json.dumps(evidence, allow_nan=False)


@pytest.mark.parametrize(("name", "value"), [
    ("integrity/continuous_replay_active", 0),
    ("replay/fallback_count", 1),
    ("replay/ratio_abs_error_max", 0.00010000001),
    ("replay/ratio_abs_error_max", -1e-6),
    ("latent/soft_to_hard_rate", 0),
    ("latent/soft_to_hard_rate", 1.01),
    ("trainer/rollout_iteration", 2),
    ("trainer/optimizer_steps_this_iteration", 1),
    ("trainer/optimizer_steps_this_iteration", 3),
    ("trainer/optimizer_step", 3),
    ("opd/ema_updates_this_iteration", 0),
    ("opd/ema_updates_this_iteration", 2),
    ("opd/ema_update_count", 1),
    ("opd/beta_effective", 0.5),
    ("opd/latent_slot_count", 0),
    ("opd/answer_slot_count", 0),
    ("grad/opd_norm", 0),
    ("grad/opd_norm", -0.01),
    ("grad/total_norm", -0.01),
])
def test_pilot_rejects_broken_replay_schedule_cadence_or_active_gradients(name, value):
    metrics = pilot_metrics(iteration=1)
    metrics[name] = value
    with pytest.raises(ValueError, match=name):
        benchmark.validate_pilot_metrics("standalone", 1, metrics)


@pytest.mark.parametrize("name", list(pilot_metrics()))
@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), True, "1"])
def test_pilot_requires_every_evidence_metric_to_be_finite_numeric(name, invalid):
    metrics = pilot_metrics()
    if invalid is None:
        del metrics[name]
    else:
        metrics[name] = invalid
    with pytest.raises(ValueError, match=name):
        benchmark.validate_pilot_metrics("standalone", 0, metrics)


@pytest.mark.parametrize("name", [
    "opd/beta_effective", "opd/latent_slot_count", "opd/answer_slot_count", "grad/opd_norm",
])
def test_hybrid_zero_dose_keeps_teacher_kl_and_opd_gradients_inactive(name):
    metrics = pilot_metrics("hybrid", 0)
    metrics[name] = 1e-12
    with pytest.raises(ValueError, match=name):
        benchmark.validate_pilot_metrics("hybrid", 0, metrics)


@pytest.mark.parametrize("iteration", [1, 2])
def test_hybrid_pilot_rejects_shortened_warmup_or_wrong_ema_count(iteration):
    for name, value in (("opd/beta_effective", 0.001),
                        ("opd/ema_update_count", iteration),
                        ("opd/ema_update_count", iteration + 2)):
        metrics = pilot_metrics("hybrid", iteration)
        metrics[name] = value
        with pytest.raises(ValueError, match=name):
            benchmark.validate_pilot_metrics("hybrid", iteration, metrics)


@pytest.mark.parametrize(("objective", "iteration", "metrics"), [
    ("auxiliary", 0, {}), ("standalone", -1, {}), ("hybrid", 3, {}),
    ("standalone", True, {}), ("standalone", 0.0, {}), ("standalone", 0, []),
])
def test_pilot_rejects_nonpilot_invocations(objective, iteration, metrics):
    with pytest.raises(ValueError):
        benchmark.validate_pilot_metrics(objective, iteration, metrics)


def pilot_actor_timing(objective="standalone", iteration=0, ranks=2):
    active = objective == "standalone" or iteration > 0
    return {"ranks": [
        {"rank": rank, "teacher_seconds": 0.5 if active else 0.0,
         "policy_update_seconds": 1.0, "worker_update_seconds": 1.2,
         "optimizer_steps": 2.0, "ema_updates_this_iteration": 1.0,
         "ema_update_count": iteration + 1}
        for rank in reversed(range(ranks))
    ]}


@pytest.mark.parametrize("objective", ["standalone", "hybrid"])
@pytest.mark.parametrize("iteration", range(3))
@pytest.mark.parametrize("ranks", [1, 2])
def test_pilot_validates_and_preserves_every_rank_update(objective, iteration, ranks):
    timing = pilot_actor_timing(objective, iteration, ranks)
    original = copy.deepcopy(timing)
    evidence = benchmark.validate_pilot_metrics(
        objective, iteration, pilot_metrics(objective, iteration),
        actor_update_timing=timing, expected_ranks=ranks,
    )
    assert evidence["ranks"] == sorted(original["ranks"], key=lambda row: row["rank"])
    assert timing == original
    json.dumps(evidence, allow_nan=False)


@pytest.mark.parametrize(("name", "value"), [
    ("optimizer_steps", 1), ("optimizer_steps", 3),
    ("ema_updates_this_iteration", 0), ("ema_updates_this_iteration", 2),
    ("ema_update_count", 1), ("teacher_seconds", 0),
    ("teacher_seconds", float("nan")), ("teacher_seconds", float("inf")),
    ("policy_update_seconds", -1), ("worker_update_seconds", float("nan")),
])
def test_pilot_rejects_rank_one_failure_even_when_canonical_metrics_pass(name, value):
    timing = pilot_actor_timing(iteration=1)
    timing["ranks"][0][name] = value  # Deliberately rank 1; canonical metrics remain valid.
    with pytest.raises(ValueError, match=f"rank 1 metric {name}"):
        benchmark.validate_pilot_metrics(
            "standalone", 1, pilot_metrics(iteration=1),
            actor_update_timing=timing, expected_ranks=2,
        )


@pytest.mark.parametrize("name", ["optimizer_steps", "ema_updates_this_iteration", "ema_update_count",
                                  "teacher_seconds", "policy_update_seconds", "worker_update_seconds"])
def test_pilot_requires_complete_rank_evidence(name):
    timing = pilot_actor_timing()
    del timing["ranks"][0][name]
    with pytest.raises(ValueError, match=f"rank 1 metric {name}"):
        benchmark.validate_pilot_metrics(
            "standalone", 0, pilot_metrics(), actor_update_timing=timing, expected_ranks=2,
        )


def test_hybrid_zero_dose_keeps_teacher_replay_inactive_on_every_rank():
    timing = pilot_actor_timing("hybrid", 0)
    timing["ranks"][0]["teacher_seconds"] = 1e-12
    with pytest.raises(ValueError, match="rank 1 metric teacher_seconds"):
        benchmark.validate_pilot_metrics(
            "hybrid", 0, pilot_metrics("hybrid", 0), actor_update_timing=timing, expected_ranks=2,
        )


@pytest.mark.parametrize("field", ["ema_updates_this_iteration", "ema_update_count"])
@pytest.mark.parametrize("value", [0, 2])
@pytest.mark.parametrize("scope", ["canonical", "rank_one"])
def test_hybrid_zero_dose_rejects_missing_or_duplicate_ema(field, value, scope):
    metrics = pilot_metrics("hybrid", 0)
    timing = pilot_actor_timing("hybrid", 0)
    if scope == "canonical":
        metrics["opd/" + field] = value
        expected_error = "pilot metric opd/" + field
    else:
        timing["ranks"][0][field] = value
        expected_error = "rank 1 metric " + field
    with pytest.raises(ValueError, match=expected_error):
        benchmark.validate_pilot_metrics(
            "hybrid", 0, metrics, actor_update_timing=timing, expected_ranks=2,
        )


def test_hybrid_acceptance_follows_real_ema_state_across_zero_and_active_doses():
    import torch
    from verl.opd.ema import EMAUpdateState, freeze_teacher_, update_ema_once_

    student = torch.nn.Linear(1, 1, bias=False)
    teacher = copy.deepcopy(student)
    freeze_teacher_(teacher)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    state = EMAUpdateState()
    optimizer_steps = 0
    for iteration in range(3):
        # The zero-dose iteration still performs its two policy optimizer steps.
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            student(torch.ones(1, 1)).sum().backward()
            optimizer.step()
            optimizer_steps += 1
        before = state.update_count
        update_ema_once_(teacher, student, 0.99, iteration, state)
        metrics = pilot_metrics("hybrid", iteration)
        metrics.update({"trainer/optimizer_step": optimizer_steps,
                        "opd/ema_updates_this_iteration": state.update_count - before,
                        "opd/ema_update_count": state.update_count})
        timing = pilot_actor_timing("hybrid", iteration, ranks=1)
        timing["ranks"][0].update(ema_updates_this_iteration=state.update_count - before,
                                  ema_update_count=state.update_count)
        accepted = benchmark.validate_pilot_metrics(
            "hybrid", iteration, metrics, actor_update_timing=timing, expected_ranks=1,
        )
        assert accepted["opd_active"] is (iteration > 0)
        assert accepted["metrics"]["opd/ema_update_count"] == iteration + 1
        if iteration == 0:
            assert accepted["metrics"]["opd/beta_effective"] == 0
            assert accepted["metrics"]["grad/opd_norm"] == 0
            assert accepted["ranks"][0]["teacher_seconds"] == 0


@pytest.mark.parametrize("rank_ids", [[0], [0, 0], [0, 2], [0, "1"], [0, True], [0, 1, 2]])
def test_pilot_rejects_missing_duplicated_or_unexpected_rank_identity(rank_ids):
    timing = pilot_actor_timing()
    template = timing["ranks"][0]
    timing["ranks"] = [{**template, "rank": rank} for rank in rank_ids]
    with pytest.raises(ValueError, match="rank inventory"):
        benchmark.validate_pilot_metrics(
            "standalone", 0, pilot_metrics(), actor_update_timing=timing, expected_ranks=2,
        )


@pytest.mark.parametrize(("timing", "ranks"), [
    (None, 2), ({}, 2), ({"ranks": None}, 2), ({"ranks": [None, None]}, 2),
    (pilot_actor_timing(), None), (pilot_actor_timing(), True), (pilot_actor_timing(), 4),
])
def test_pilot_rank_validation_requires_both_timing_and_authorized_rank_count(timing, ranks):
    with pytest.raises(ValueError):
        benchmark.validate_pilot_metrics(
            "standalone", 0, pilot_metrics(), actor_update_timing=timing, expected_ranks=ranks,
        )


@pytest.mark.parametrize("failure", [None, "dirty", "gitlink", "environment"])
def test_source_identity_requires_clean_committed_and_pinned_parent_fork(monkeypatch, tmp_path, failure):
    fork = tmp_path / "project" / "3rdparty" / "SofT-GRPO"
    parent = fork.parents[1]
    def git(command, **kwargs):
        directory = Path(command[2])
        if command[3] == "status":
            return " M code.py" if failure == "dirty" else ""
        if command[-1] == "HEAD:3rdparty/SofT-GRPO":
            return "c" * 40 if failure == "gitlink" else "b" * 40
        return "a" * 40 if directory == parent else "b" * 40
    monkeypatch.setattr(benchmark.subprocess, "check_output", git)
    monkeypatch.setenv("OPD_QTB_PARENT_COMMIT", "c" * 40 if failure == "environment" else "a" * 40)
    monkeypatch.setenv("OPD_QTB_FORK_COMMIT", "b" * 40)
    if failure:
        with pytest.raises(ValueError):
            benchmark.source_identity(fork)
    else:
        assert benchmark.source_identity(fork) == {"parent_commit": "a" * 40, "fork_commit": "b" * 40, "snapshot_verified": True}


@pytest.fixture
def authenticated_phase(tmp_path):
    from verl.opd.provenance import _environment_identity, build_checkpoint_provenance

    def seal(path, value):
        value = {**value, "manifest_content_sha256": canonical_sha256(value)}
        benchmark.write_json(path, value)
        return value
    model_root, data_root = tmp_path / "model", tmp_path / "data"
    model_root.mkdir()
    data_root.mkdir()
    model = seal(model_root / "manifest.json", {"model": {"id": MODEL_ID, "resolved_revision": MODEL_REVISION}, "inventory_sha256": "c" * 64})
    for filename in ("train.parquet", "validation.parquet"):
        (data_root / filename).write_bytes(b"data")
    data = seal(data_root / "manifest.json", {"files": {name: {"size": 4, "sha256": "d" * 64} for name in ("train.parquet", "validation.parquet")}})
    configuration = {"actor_rollout_ref": {"model": {"path": str(model_root)}, "rollout": {"n": 8, "dispatch_mode": "bounded_async", "max_running_requests": 16, "async_queue_size": 32}}, "data": {"train_files": str(data_root / "train.parquet"), "val_files": str(data_root / "validation.parquet")}, "trainer": {"n_gpus_per_node": 2, "total_epochs": 1, "max_rollout_iterations_per_invocation": 3}}
    provenance = build_checkpoint_provenance(configuration, source_commit="b" * 40, environment_identity=_environment_identity(package_versions={"torch": "2.6.0", "verl": "0.4.0"}))
    measured = {"phase": "calibration", "variant": "bounded_async16", "status": "complete", "wandb_run_id": "run-1", "configuration": configuration, "checkpoint_provenance": provenance, "rows": [{"variant": "bounded_async16", "batch_index": 0}]}
    kwargs = {"phase": "calibration", "variant": "bounded_async16", "batches": [0], "overrides": ["trainer.n_gpus_per_node=2", "actor_rollout_ref.rollout.n=8", "actor_rollout_ref.rollout.dispatch_mode=bounded_async", "trainer.max_rollout_iterations_per_invocation=3"], "source": {"fork_commit": "b" * 40}, "assets": {"model_manifest_content_sha256": model["manifest_content_sha256"], "data_manifest_content_sha256": data["manifest_content_sha256"]}, "run_id": "run-1"}
    return measured, kwargs


def test_phase_authenticates_full_configuration_source_assets_and_identity(authenticated_phase):
    measured, kwargs = authenticated_phase
    kwargs["overrides"].append("hydra.run.dir=/outside/source")
    benchmark.validate_phase_measurement(measured, **kwargs)


@pytest.mark.parametrize("field", ["source", "model", "config", "gpus", "group", "wandb", "batch", "missing_batch", "assets", "seal"])
def test_phase_rejects_mismatched_or_forged_measurements(authenticated_phase, field):
    measured, kwargs = authenticated_phase
    if field == "source":
        kwargs["source"]["fork_commit"] = "c" * 40
    elif field == "model":
        measured["checkpoint_provenance"]["model"]["resolved_revision"] = "c" * 40
    elif field == "config":
        measured["configuration"]["trainer"]["total_epochs"] = 5
    elif field == "gpus":
        kwargs["overrides"][0] = "trainer.n_gpus_per_node=1"
    elif field == "group":
        kwargs["overrides"][1] = "actor_rollout_ref.rollout.n=1"
    elif field == "wandb":
        measured["wandb_run_id"] = "different-run"
    elif field == "batch":
        measured["rows"][0]["batch_index"] = 1
    elif field == "missing_batch":
        measured["rows"] = []
    elif field == "assets":
        kwargs["assets"]["data_manifest_content_sha256"] = "c" * 64
    else:
        measured["checkpoint_provenance"]["identity_sha256"] = "c" * 64
    with pytest.raises((ValueError, RuntimeError)):
        benchmark.validate_phase_measurement(measured, **kwargs)


@pytest.fixture
def runner(tmp_path, monkeypatch):
    instance = object.__new__(benchmark.CellRunner)
    instance.args = SimpleNamespace(root=tmp_path / "assets", objective="hybrid", gpus=2)
    instance.root = tmp_path / "cell"
    instance.root.mkdir()
    instance.started = time.monotonic()
    instance.deadline = instance.started + 7100
    instance.child = None
    instance.fork_root = Path(benchmark.__file__).resolve().parents[1]
    instance.source = {"parent_commit": "a" * 40, "fork_commit": "b" * 40}
    instance.assets = {}
    instance.cell = {"phases": [], "jobs": {"slurm_job_id": "123"}, "screen_rows": [], "confirm_rows": [], "iterations": [], "wandb_run_ids": []}
    monkeypatch.setattr(instance, "_terminate", lambda: setattr(instance, "child", None))
    return instance


@pytest.mark.parametrize("fails", [False, True])
def test_explicit_repair_pilot_never_becomes_a_dispatch_study_estimate(runner, monkeypatch, fails):
    from opd_tools import icl_resource_monitor
    calls = []
    runner.args.repair_pilot = "bounded_async32"

    def phase(*args):
        calls.append(args)
        if fails:
            raise RuntimeError("strict replay gate rejected")

    monkeypatch.setattr(runner, "phase", phase)
    monkeypatch.setattr(benchmark.signal, "signal", lambda *args: None)
    monitor = SimpleNamespace(start=lambda: monitor, stop=lambda: SimpleNamespace(to_dict=lambda: {}))
    monkeypatch.setattr(icl_resource_monitor, "ResourceMonitor", lambda **kwargs: monitor)
    if fails:
        with pytest.raises(RuntimeError, match="strict replay gate"):
            runner.run()
    else:
        runner.run()
    assert calls == [("pilot", "bounded_async32", [])]
    assert runner.cell["role"] == "repair_validation"
    assert runner.cell["status"] == ("incomplete" if fails else "repair_validation_complete")
    assert "estimate" not in runner.cell
    assert "dispatch_selection" not in runner.cell


def test_repair_pilot_cli_requires_explicit_thirty_minute_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "CellRunner", lambda args: pytest.fail("oversized repair allocation initialized"))
    with pytest.raises(SystemExit):
        benchmark.main(["--root", str(tmp_path), "--output-dir", str(tmp_path / "out"),
                        "--objective", "standalone", "--gpus", "2", "--repair-pilot", "bounded_async32"])


@pytest.mark.parametrize("objective,gpus", [("standalone", 1), ("standalone", 2), ("hybrid", 1), ("hybrid", 2)])
@pytest.mark.parametrize("backend", [None, "native_fa3_v1"])
def test_full_study_cli_keeps_default_and_accepts_explicit_native_backend(monkeypatch, tmp_path, objective, gpus, backend):
    observed = []
    monkeypatch.setattr(benchmark, "CellRunner", lambda args: SimpleNamespace(run=lambda: observed.append(args)))
    arguments = ["--root", str(tmp_path), "--output-dir", str(tmp_path / "out"),
                 "--objective", objective, "--gpus", str(gpus), "--time-limit-seconds", "7100"]
    if backend is not None:
        arguments += ["--qwen-replay-backend", backend]
    benchmark.main(arguments)
    assert len(observed) == 1
    args = observed[0]
    assert (args.objective, args.gpus, args.time_limit_seconds) == (objective, gpus, 7100)
    assert args.qwen_replay_backend == (backend or "disabled")
    assert args.repair_pilot is None


@pytest.mark.parametrize("repair,seconds", [(False, 7201), (True, 1801)])
def test_native_cli_preserves_full_study_and_repair_time_bounds(monkeypatch, tmp_path, repair, seconds):
    monkeypatch.setattr(benchmark, "CellRunner", lambda args: pytest.fail("oversized allocation initialized"))
    arguments = ["--root", str(tmp_path), "--output-dir", str(tmp_path / "out"),
                 "--objective", "standalone", "--gpus", "1", "--time-limit-seconds", str(seconds),
                 "--qwen-replay-backend", "native_fa3_v1"]
    if repair:
        arguments += ["--repair-pilot", "bounded_async32"]
    with pytest.raises(SystemExit):
        benchmark.main(arguments)


@pytest.mark.parametrize("backend", ["disabled", "native_fa3_v1"])
def test_cell_initial_metadata_records_requested_backend(monkeypatch, tmp_path, backend):
    from opd_tools import qwen_training

    monkeypatch.setattr(qwen_training, "verify", lambda root: {})
    monkeypatch.setattr(benchmark, "source_identity", lambda root: {"fork_commit": "b" * 40})
    args = SimpleNamespace(root=tmp_path / "assets", output_dir=tmp_path / "cell", objective="hybrid", gpus=2,
                           time_limit_seconds=7100, qwen_replay_backend=backend)
    instance = benchmark.CellRunner(args)
    assert instance.cell["configuration"]["qwen_replay_backend"] == backend
    assert json.loads((args.output_dir / "cell.json").read_text())["configuration"]["qwen_replay_backend"] == backend


@pytest.mark.parametrize("backend", ["disabled", "native_fa3_v1"])
@pytest.mark.parametrize("outcome", ["complete", "timeout", "exit_failure", "bad_json", "wrong_identity"])
def test_phase_preserves_partial_rows_authenticates_results_and_keeps_invocation_horizon(runner, monkeypatch, outcome, backend):
    runner.args.qwen_replay_backend = backend
    captured = {}
    def authenticate(measured, **kwargs):
        captured["authentication"] = kwargs
        if outcome == "wrong_identity":
            raise ValueError("wrong source")
    monkeypatch.setattr(benchmark, "validate_phase_measurement", authenticate)
    class Child:
        pid = 100
        def __init__(self, command, **kwargs):
            captured.update(command=command, **kwargs)
            output = Path(next(item.split("=", 1)[1] for item in command if item.startswith("++trainer.training_benchmark_output=")))
            if outcome == "bad_json":
                output.write_text("broken", encoding="utf-8")
            else:
                benchmark.write_json(output, {"phase": "calibration", "variant": "bounded_async16", "status": "complete", "wandb_online": True, "wandb_finished": True, "wandb_run_id": kwargs["env"]["WANDB_RUN_ID"], "rows": [{"variant": "bounded_async16", "batch_index": 0, "valid": True, "generated_tokens": 1000, "wall_seconds": 10}]})
        def wait(self, timeout):
            assert timeout <= 7100
            if outcome == "timeout":
                raise subprocess.TimeoutExpired(captured["command"], timeout)
            return 1 if outcome == "exit_failure" else 0
    monkeypatch.setattr(benchmark.subprocess, "Popen", Child)
    if outcome == "complete":
        runner.phase("calibration", "bounded_async16", [0])
    else:
        with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
            runner.phase("calibration", "bounded_async16", [0])
    cell = json.loads((runner.root / "cell.json").read_text())
    assert cell["phases"][0]["status"] == ("complete" if outcome == "complete" else "incomplete")
    if outcome != "bad_json":
        assert len(cell["screen_rows"]) == 1
        assert cell["screen_rows"][0]["valid"] == (outcome == "complete")
        assert cell["phases"][0]["measurement"]["rows"][0]["valid"] is True
    command = captured["command"]
    assert "trainer.total_epochs=1" in command
    assert "trainer.total_training_steps=null" in command
    assert "trainer.max_rollout_iterations_per_invocation=3" in command
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=1" in command
    assert "actor_rollout_ref.rollout.async_queue_size=32" in command
    assert ("actor_rollout_ref.model.qwen_replay_backend=native_fa3_v1" in command) == (backend == "native_fa3_v1")
    assert any(item.startswith("hydra.run.dir=" + str(runner.root)) for item in command)
    assert "hydra.job.chdir=false" in command
    assert captured["start_new_session"] is True
    assert captured["env"]["WANDB_RESUME"] == "never"
    invocation = json.loads(next(runner.root.glob("*/invocation.json")).read_text())
    assert invocation["command"] == command
    assert invocation["source"] == runner.source


def test_exhausted_budget_does_not_start_another_phase(runner, monkeypatch):
    runner.deadline = time.monotonic() + 50
    monkeypatch.setattr(benchmark.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must not start another process"))
    with pytest.raises(TimeoutError):
        runner.phase("pilot", "legacy_batch", [])
    assert runner.cell["phases"] == []


@pytest.mark.parametrize("pilot_complete", [True, False])
def test_cell_confirms_screen_winner_and_never_marks_missing_pilot_timing_complete(runner, monkeypatch, pilot_complete):
    import opd_tools.icl_resource_monitor as resource_monitor

    class Monitor:
        def __init__(self, **kwargs):
            pass
        def start(self):
            return self
        def stop(self):
            return self
        def to_dict(self):
            return {"sample_count": 1}
    monkeypatch.setattr(resource_monitor, "ResourceMonitor", Monitor)
    monkeypatch.setattr(benchmark.signal, "signal", lambda *args: None)
    runner.cell["configuration"] = {}
    calls = []
    def phase(kind, variant, batches):
        calls.append((kind, variant, batches))
        if kind == "calibration":
            wall = dict(legacy_batch=10, expanded_batch=9, bounded_async16=8, bounded_async32=7)[variant]
            if batches == [1] and variant == "bounded_async32":
                wall = 11
            row = {"variant": variant, "batch_index": batches[0], "wall_seconds": wall, "generated_tokens": 1000, "valid": True, "requests": [fingerprint(batches[0])]}
            runner.cell["screen_rows" if batches == [0] else "confirm_rows"].append(row)
        else:
            runner.cell.update(startup_seconds=1, validation_seconds=1, validation_example_count=128,
                               iterations=[{"rollout_iteration": index, "timing_s": {"step": 10, "gen": 2, "old_log_prob": 2, "update_actor": 3}} for index in range(3)])
            if pilot_complete:
                runner.cell["checkpoint_seconds"] = 1
    monkeypatch.setattr(runner, "phase", phase)
    if pilot_complete:
        runner.run()
    else:
        with pytest.raises(ValueError):
            runner.run()
    assert calls[:4] == [("calibration", variant, [0]) for variant in benchmark.VARIANT_CONFIGS]
    assert calls[4:] == [("calibration", "legacy_batch", [1]), ("calibration", "bounded_async32", [1]), ("pilot", "legacy_batch", [])]
    cell = json.loads((runner.root / "cell.json").read_text())
    assert cell["status"] == ("complete" if pilot_complete else "incomplete")
    assert cell["dispatch_selection"]["selected_variant"] == "legacy_batch"
    assert len(cell["output_comparisons"]) == 4
    assert (runner.root / "cell.json.sha256").is_file()
