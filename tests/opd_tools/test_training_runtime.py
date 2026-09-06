from __future__ import annotations

import copy
import hashlib
import json

import pytest

from opd_tools.training_runtime import (
    MAX_ALLOCATED_GPU_HOURS,
    MODEL_REVISION,
    VARIANTS,
    aggregate_cells,
    aggregate_repair_validation,
    atomic_write_json,
    estimate_runtime,
    main,
    pick_candidate,
    recommend_allocation,
    render_markdown,
    render_repair_markdown,
    select_dispatch,
)


def measurement(variant, batch=0, wall=10, tokens=1000, valid=True, **metadata):
    return {"variant": variant, "batch_index": batch, "wall_seconds": wall, "generated_tokens": tokens, "valid": valid, **metadata}


def iterations():
    return [
        {"rollout_iteration": index, "timing_s": {"step": step, "gen": gen, "reward": 2, "old_log_prob": 5, "ref": 2, "adv": 1, "update_actor": 20, "teacher_forward": 7}}
        for index, step, gen in ((0, 50, 10), (1, 60, 20), (2, 80, 30))
    ]


def complete_cell(objective="standalone", gpus=1):
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assets = {"model": {"id": "Qwen/Qwen3-0.6B", "revision": MODEL_REVISION}}
    assets["manifest_content_sha256"] = digest(assets)
    resolved_config = {"trainer": {"n_gpus_per_node": gpus}}
    provenance = {"source": {"commit": "b" * 40}, "model": {"id": "Qwen/Qwen3-0.6B", "resolved_revision": MODEL_REVISION}, "resolved_hydra_config": {"full_sha256": digest(resolved_config)}}
    return {
        "objective": objective,
        "gpus": gpus,
        "status": "complete",
        "startup_seconds": 100,
        "iterations": iterations(),
        "validation_seconds": 8,
        "validation_example_count": 128,
        "checkpoint_seconds": 3,
        "screen_rows": [measurement("legacy_batch"), measurement("bounded_async16", wall=8)],
        "confirm_rows": [measurement("legacy_batch", batch=1), measurement("bounded_async16", batch=1, wall=8)],
        "source": {"parent_commit": "a" * 40, "fork_commit": "b" * 40, "snapshot_verified": True, "implementation_sha256": "c" * 64, "files": [{"path": "file.py", "sha256": "d" * 64}]},
        "assets": assets,
        "configuration": {"model_id": "Qwen/Qwen3-0.6B", "model_revision": MODEL_REVISION, "objective": objective, "gpus": gpus, "dispatch_mode": "bounded_async", "max_running_requests": 16, "async_queue_size": 32},
        "phases": [{"status": "complete", "authenticated": True, "wandb_run_id": "abc123", "measurement": {"wandb_online": True, "wandb_finished": True, "wandb_run_id": "abc123", "configuration": resolved_config, "checkpoint_provenance": provenance}}],
        "jobs": [{"job_id": "123", "time_limit_seconds": 7200}],
        "wandb_run_ids": ["abc123"],
    }


def test_dispatch_registry_preserves_separate_queue_and_scheduler_capacity():
    assert tuple(VARIANTS) == ("legacy_batch", "expanded_batch", "bounded_async16", "bounded_async32")
    assert VARIANTS["legacy_batch"]["max_running_requests"] == 16
    assert VARIANTS["expanded_batch"]["async_queue_size"] == 32
    assert VARIANTS["bounded_async16"] == {"dispatch_mode": "bounded_async", "max_running_requests": 16, "async_queue_size": 32}
    assert VARIANTS["bounded_async32"] == {"dispatch_mode": "bounded_async", "max_running_requests": 32, "async_queue_size": 64}


def test_fastest_valid_screen_candidate_requires_independent_confirmation():
    screen = [measurement("legacy_batch"), measurement("expanded_batch", wall=9), measurement("bounded_async16", wall=8), measurement("bounded_async32", wall=7, valid=False)]
    assert pick_candidate(screen) == "bounded_async16"
    pending = select_dispatch(screen, [])
    assert pending["candidate_variant"] == "bounded_async16"
    assert pending["selected_variant"] == "legacy_batch"
    assert not pending["accepted"]
    result = select_dispatch(screen, [measurement("legacy_batch", batch=1, wall=20), measurement("bounded_async16", batch=1, wall=16)])
    assert result["accepted"]
    assert result["selected_variant"] == "bounded_async16"
    assert result["comparisons"][0]["wall_speedup"] == pytest.approx(1.25)
    assert result["comparisons"][1]["throughput_speedup"] == pytest.approx(1.25)


@pytest.mark.parametrize("wall,tokens", [(11, 2000), (9, 500), (10, 1100), (9, 900)])
def test_candidate_must_improve_wall_and_throughput_strictly(wall, tokens):
    screen = [measurement("legacy_batch"), measurement("expanded_batch", wall=wall, tokens=tokens)]
    confirm = [measurement("legacy_batch", batch=1), measurement("expanded_batch", batch=1, wall=8)]
    assert select_dispatch(screen, confirm)["selected_variant"] == "legacy_batch"


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -1, 0, True])
def test_nonfinite_missing_and_nonpositive_measurements_cannot_pass(bad):
    screen = [measurement("legacy_batch"), measurement("expanded_batch", wall=bad)]
    assert pick_candidate(screen) == "legacy_batch"
    screen = [measurement("legacy_batch", tokens=bad), measurement("expanded_batch", wall=8)]
    confirm = [measurement("legacy_batch", batch=1), measurement("expanded_batch", batch=1, wall=8)]
    assert not select_dispatch(screen, confirm)["accepted"]


def test_confirmation_failure_or_duplicate_measurement_fails_closed():
    screen = [measurement("legacy_batch"), measurement("expanded_batch", wall=8)]
    confirm = [measurement("legacy_batch", batch=1), measurement("expanded_batch", batch=1, wall=9, valid=False)]
    assert not select_dispatch(screen, confirm)["accepted"]
    confirm[-1]["valid"] = True
    confirm.append(copy.deepcopy(confirm[-1]))
    assert not select_dispatch(screen, confirm)["accepted"]


@pytest.mark.parametrize("rows", [None, {}, [None], "invalid"])
def test_malformed_dispatch_records_cannot_pass(rows):
    assert select_dispatch(rows, rows)["selected_variant"] == "legacy_batch"


def test_deterministic_tie_and_no_second_candidate_after_failed_confirmation():
    screen = [measurement("legacy_batch"), measurement("bounded_async16", wall=8), measurement("expanded_batch", wall=8)]
    assert pick_candidate(screen) == "expanded_batch"
    confirm = [measurement("legacy_batch", batch=1), measurement("bounded_async16", batch=1, wall=7)]
    assert select_dispatch(screen, confirm)["selected_variant"] == "legacy_batch"


def test_standalone_and_hybrid_exact_horizon_and_gpu_hours():
    standalone = estimate_runtime("standalone", 1, 100, iterations(), 8, 3)
    hybrid = estimate_runtime("hybrid", 2, 100, iterations(), 8, 3)
    assert standalone["active_iteration_count"] == 109
    assert hybrid["active_iteration_count"] == 108
    assert standalone["active_iteration_mean"] == {"wall_seconds": 70, "token_seconds": 52, "non_token_seconds": 18}
    assert standalone["scenarios"]["1.0"]["seconds"] == 100 + 109 * 70 + 8 * 4 * 6 + 3 * 5
    assert hybrid["scenarios"]["1.0"]["seconds"] == 100 + 50 + 108 * 70 + 8 * 4 * 6 + 3 * 5
    assert standalone["scenarios"]["2.0"]["seconds"] == 100 + 109 * (18 + 104) + 8 * 4 * 6 * 2 + 3 * 5
    assert hybrid["scenarios"]["2.0"]["seconds"] == 100 + (13 + 74) + 108 * (18 + 104) + 8 * 4 * 6 * 2 + 3 * 5
    assert hybrid["scenarios"]["1.5"]["gpu_hours"] == hybrid["scenarios"]["1.5"]["hours"] * 2
    assert standalone["preliminary"]
    assert "not confidence intervals" in standalone["scenario_note"]


def test_remove_nested_validation_save_and_never_double_count_teacher():
    original = estimate_runtime("standalone", 1, 100, iterations(), 8, 3)
    with_overhead = iterations()
    for row in with_overhead:
        row["timing_s"].update(testing=9, save_checkpoint=4, teacher_forward=9999)
        row["timing_s"]["step"] += 13
    observed = estimate_runtime("standalone", 1, 100, with_overhead, 8, 3)
    assert observed == original


def test_weight_transfer_stays_fixed_and_override_partitions_wall_time():
    source = iterations()
    original = estimate_runtime("standalone", 1, 100, source, 8, 3)
    for row in source:
        row["timing_s"]["weight_sync"] = 5
    fixed_sync = estimate_runtime("standalone", 1, 100, source, 8, 3)
    assert fixed_sync["scenarios"]["1.0"] == original["scenarios"]["1.0"]
    assert fixed_sync["scenarios"]["2.0"]["seconds"] == original["scenarios"]["2.0"]["seconds"] - 109 * 5
    for row in source:
        row["non_token_seconds"] = 25
    overridden = estimate_runtime("standalone", 1, 100, source, 8, 3)
    assert overridden["active_iteration_mean"]["token_seconds"] == 45
    assert overridden["active_iteration_mean"]["non_token_seconds"] == 25


@pytest.mark.parametrize("change", ["missing", "duplicate", "nan", "no_replay", "overlap", "bad_weight_sync", "bad_partition"])
def test_incomplete_or_inconsistent_training_timings_are_rejected(change):
    source = iterations()
    if change == "missing":
        source.pop()
    elif change == "duplicate":
        source[2]["rollout_iteration"] = 1
    elif change == "nan":
        source[1]["timing_s"]["update_actor"] = float("nan")
    elif change == "no_replay":
        del source[1]["timing_s"]["old_log_prob"]
    elif change == "overlap":
        source[1]["timing_s"]["update_actor"] = 200
    elif change == "bad_weight_sync":
        source[1]["timing_s"]["weight_sync"] = 200
    elif change == "bad_partition":
        source[1]["non_token_seconds"] = 200
    with pytest.raises(ValueError):
        estimate_runtime("standalone", 1, 100, source, 8, 3)


@pytest.mark.parametrize("objective,gpus", [("unknown", 1), ("hybrid", 4), ("hybrid", True)])
def test_only_planned_objectives_and_allocations(objective, gpus):
    with pytest.raises(ValueError):
        estimate_runtime(objective, gpus, 100, iterations(), 8, 3)


def test_aggregation_preserves_provenance_and_publishes_missing_cells(tmp_path):
    cell = complete_cell()
    cell["screen_rows"][1].update(rank_tail_seconds=[7.9, 8.0], peak_memory_bytes=[123, 456])
    atomic_write_json(tmp_path / "standalone-gpu1" / "cell.json", cell)
    report = aggregate_cells(tmp_path)
    assert len(report["cells"]) == 4
    measured = report["cells"][0]
    assert measured["status"] == "complete"
    for key in ("source", "configuration", "jobs", "wandb_run_ids", "screen_rows", "iterations"):
        assert measured[key] == cell[key]
    assert all(row["status"] == "incomplete" and row["estimate"] is None for row in report["cells"][1:])
    assert report["recipe"]["model_revision"] == MODEL_REVISION
    assert report["recipe"]["rollout_iterations"] == 109
    assert report["recipe"]["hybrid_warmup_iterations"] == 11
    assert report["budget"]["maximum_allocated_gpu_hours"] == MAX_ALLOCATED_GPU_HOURS == 12
    assert report["recommendations"][0]["gpus"] == 1
    assert report["recommendations"][1]["gpus"] is None
    markdown = render_markdown(report)
    assert "Preliminary" in markdown
    assert "abc123" in markdown
    assert "a" * 40 in markdown
    assert "incomplete" in markdown
    assert "confidence intervals" in markdown


def test_pin_mismatch_duplicate_and_timeout_never_supply_estimates(tmp_path):
    wrong_pin = complete_cell()
    wrong_pin["configuration"]["model_revision"] = "wrong"
    atomic_write_json(tmp_path / "wrong-pin" / "cell.json", wrong_pin)
    timed_out = complete_cell("hybrid", 1)
    timed_out.update(status="timeout", reason="two-hour allocation ended")
    atomic_write_json(tmp_path / "timeout" / "cell.json", timed_out)
    atomic_write_json(tmp_path / "duplicate-a" / "cell.json", complete_cell("hybrid", 2))
    atomic_write_json(tmp_path / "duplicate-b" / "cell.json", complete_cell("hybrid", 2))
    report = aggregate_cells(tmp_path)
    assert all(row["status"] == "incomplete" and row["estimate"] is None for row in report["cells"])
    assert "pinned" in report["cells"][0]["reason"]
    assert report["cells"][2]["reason"] == "two-hour allocation ended"
    assert "duplicate" in report["cells"][3]["reason"]


def test_malformed_configuration_becomes_explicit_incomplete_cell(tmp_path):
    cell = complete_cell()
    cell["configuration"] = None
    atomic_write_json(tmp_path / "standalone-gpu1" / "cell.json", cell)
    report = aggregate_cells(tmp_path)
    assert report["cells"][0]["status"] == "incomplete"
    assert report["cells"][0]["estimate"] is None
    assert "configuration" in report["cells"][0]["reason"]


@pytest.mark.parametrize("change", ["missing_model", "wrong_gpus", "source", "asset_seal", "phase_authentication", "wandb", "phase_source", "phase_config", "validation_count"])
def test_report_requires_authenticated_matching_provenance_before_recommending(tmp_path, change):
    cell = complete_cell()
    if change == "missing_model":
        del cell["configuration"]["model_revision"]
    elif change == "wrong_gpus":
        cell["configuration"]["gpus"] = 2
    elif change == "source":
        cell["source"]["snapshot_verified"] = False
    elif change == "asset_seal":
        cell["assets"]["manifest_content_sha256"] = "e" * 64
    elif change == "phase_authentication":
        cell["phases"][0]["authenticated"] = False
    elif change == "wandb":
        cell["phases"][0]["measurement"]["wandb_finished"] = False
    elif change == "phase_source":
        cell["phases"][0]["measurement"]["checkpoint_provenance"]["source"]["commit"] = "e" * 40
    elif change == "phase_config":
        cell["phases"][0]["measurement"]["configuration"]["trainer"]["n_gpus_per_node"] = 2
    else:
        cell["validation_example_count"] = 512
    atomic_write_json(tmp_path / "cell.json", cell)
    report = aggregate_cells(tmp_path)
    assert report["cells"][0]["status"] == "incomplete"
    assert report["cells"][0]["estimate"] is None
    assert report["recommendations"][0]["gpus"] is None


def test_recommend_smallest_conservative_fit_and_no_unmeasured_fallback(tmp_path):
    for gpus in (1, 2):
        cell = complete_cell(gpus=gpus)
        if gpus == 1:
            cell["startup_seconds"] = 40000
        atomic_write_json(tmp_path / f"standalone-gpu{gpus}" / "cell.json", cell)
    report = aggregate_cells(tmp_path)
    assert report["recommendations"][0]["gpus"] == 2
    assert report["recommendations"][1]["gpus"] is None
    assert recommend_allocation(report["cells"][:1], "standalone")["gpus"] is None
    for row in report["cells"]:
        if row.get("estimate"):
            row["estimate"]["eligible_under_12_hours"] = False
    assert recommend_allocation(report["cells"], "standalone")["gpus"] is None


def test_cli_atomic_json_and_markdown_with_empty_or_invalid_input(tmp_path):
    root = tmp_path / "input"
    bad = root / "broken" / "cell.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{ invalid", encoding="utf-8")
    output = tmp_path / "reports"
    assert main(["aggregate", "--input-root", str(root), "--output-dir", str(output)]) == 0
    report = json.loads((output / "training_runtime.json").read_text())
    assert len(report["input_errors"]) == 1
    assert all(row["estimate"] is None for row in report["cells"])
    assert "incomplete" in (output / "training_runtime.md").read_text()
    assert sorted(path.name for path in output.iterdir()) == ["training_runtime.json", "training_runtime.md"]
    with pytest.raises(ValueError):
        atomic_write_json(output / "training_runtime.json", {"value": float("nan")})
    assert json.loads((output / "training_runtime.json").read_text()) == report


def test_markdown_shows_actual_dispatch_lengths_rank_tails_memory_and_stage_costs(tmp_path):
    cell = complete_cell(gpus=2)
    for row in cell["screen_rows"]:
        row["requests"] = [{"token_count": 500}, {"token_count": 500}]
        row["rollout_timing"] = {"ranks": [{"rank": 0, "engine_generation_seconds": 6}, {"rank": 1, "engine_generation_seconds": 8}]}
        row["resource_telemetry"] = {"peak_hbm_gib_per_gpu": {"0": 12.5, "1": 13.75}, "peak_hbm_gib_aggregate": 25.75}
    cell["screen_rows"][1]["generated_tokens"] = 1100
    for row in cell["iterations"]:
        row["timing_s"]["weight_sync"] = 3
    atomic_write_json(tmp_path / "cell.json", cell)
    markdown = render_markdown(aggregate_cells(tmp_path))
    assert "| bounded_async16 / 0 | yes | 8.00 | 137.50 | 550.00 (+50.00) | r0: 6.00; r1: 8.00 | GPU 0: 12.50; GPU 1: 13.75; measured aggregate: 25.75 |" in markdown
    assert "| Active mean (iterations 1–2) | 70.00 | 25.00 | 5.00 | 2.00 | 20.00 | 3.00 |" in markdown
    assert "| Iteration 0 | 50.00 | 10.00 | 5.00 | 2.00 | 20.00 | 3.00 |" in markdown
    assert "| Startup | 100.00 |" in markdown
    assert "| Timing validation (128 examples) | 8.00 |" in markdown
    assert "| One authenticated checkpoint | 3.00 |" in markdown
    assert "not added together" in markdown
    assert "Δ unavailable" in markdown  # Confirmation rows have no request-length metadata.


def test_unmeasured_cells_and_missing_rank_memory_telemetry_are_explicit(tmp_path):
    cell = complete_cell()
    atomic_write_json(tmp_path / "cell.json", cell)
    markdown = render_markdown(aggregate_cells(tmp_path))
    assert "| legacy_batch / 0 | yes | 10.00 | 100.00 | unavailable (Δ unavailable) | unavailable | unavailable |" in markdown
    assert "| standalone | 2 | incomplete | unmeasured |" in markdown
    assert "Dispatch measurements, rank generation tails, and peak HBM: unavailable." in markdown


def test_teacher_diagnostics_show_max_rank_wall_without_changing_runtime(tmp_path):
    cell = complete_cell("hybrid", 2)
    original = estimate_runtime(cell["objective"], cell["gpus"], cell["startup_seconds"], cell["iterations"], cell["validation_seconds"], cell["checkpoint_seconds"])
    for row, maximum in zip(cell["iterations"], (0, 8, 12)):
        row["actor_update_timing"] = {"teacher_seconds_max": maximum, "ranks": [{"rank": 0, "teacher_seconds": maximum / 2}, {"rank": 1, "teacher_seconds": maximum}], "timing_method": "cuda_synchronized_wall", "timing_note": "synchronization cost included"}
    atomic_write_json(tmp_path / "cell.json", cell)
    report = aggregate_cells(tmp_path)
    assert report["cells"][3]["estimate"] == original
    markdown = render_markdown(report)
    assert "| Iteration 0 (zero dose) | 50.00 | 10.00 | 5.00 | 2.00 | 20.00 | unavailable | 0.00 |" in markdown
    assert "| Active mean (iterations 1–2) | 70.00 | 25.00 | 5.00 | 2.00 | 20.00 | unavailable | 10.00 |" in markdown
    assert "CUDA synchronization cost is included" in markdown
    assert "never added to actor time or the runtime estimate" in markdown


def submission_record():
    return {
        "schema_version": 1,
        "submission_id": "qwen-training-test",
        "source_snapshot": "/remote/snapshots/qwen-training",
        "parent_commit": "a" * 40,
        "fork_commit": "b" * 40,
        "profile": "qwen3-training-benchmark-v1",
        "job_limit_seconds": 7200,
        "maximum_allocated_gpu_hours": 12,
        "jobs": [
            {"objective": objective, "gpus": gpus, "job_id": 100 + index}
            for index, (objective, gpus) in enumerate(
                (("standalone", 1), ("standalone", 2), ("hybrid", 1), ("hybrid", 2))
            )
        ],
        "state": "submitted",
    }


def test_pending_submission_retains_job_source_and_registry_identity_without_measurements(tmp_path):
    submission = submission_record()
    path = tmp_path / "submission.json"
    atomic_write_json(path, submission)
    report = aggregate_cells(tmp_path)
    assert report["submission"]["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    for cell, job in zip(report["cells"], submission["jobs"]):
        assert cell["status"] == "incomplete"
        assert cell["estimate"] is None
        assert cell["reason"] == "job submitted; no cell measurement yet"
        assert cell["jobs"] == {"slurm_job_id": str(job["job_id"])}
        assert cell["source"]["parent_commit"] == submission["parent_commit"]
        assert cell["source"]["fork_commit"] == submission["fork_commit"]
        assert cell["source"]["source_snapshot"] == submission["source_snapshot"]
        assert "snapshot_verified" not in cell["source"]
        assert cell["source"]["identity_origin"] == "submission_manifest"
    assert all(row["gpus"] is None for row in report["recommendations"])
    markdown = render_markdown(report)
    assert "Submission `qwen-training-test`" in markdown
    assert "registry SHA-256" in markdown
    assert "job submitted; no cell measurement yet" in markdown
    assert '"slurm_job_id": "103"' in markdown
    assert "| hybrid | 2 | incomplete | unmeasured |" in markdown


@pytest.mark.parametrize("change", [
    "duplicate_cell", "duplicate_job", "missing_job", "unknown_objective",
    "string_gpu", "bool_gpu", "string_job", "bool_job", "negative_job",
    "short_parent", "uppercase_fork", "string_time", "bool_time", "over_time",
    "over_gpu_hours", "underdeclared_gpu_hours", "string_gpu_hours", "nan_gpu_hours",
    "wrong_profile", "partial_submission", "relative_snapshot", "bool_schema",
])
def test_invalid_submission_metadata_is_rejected_before_any_estimate(tmp_path, change):
    submission = submission_record()
    if change == "duplicate_cell":
        submission["jobs"][1]["gpus"] = 1
    elif change == "duplicate_job":
        submission["jobs"][1]["job_id"] = submission["jobs"][0]["job_id"]
    elif change == "missing_job":
        submission["jobs"].pop()
    elif change == "unknown_objective":
        submission["jobs"][0]["objective"] = "standalone_opd"
    elif change in ("string_gpu", "bool_gpu"):
        submission["jobs"][0]["gpus"] = "1" if change == "string_gpu" else True
    elif change in ("string_job", "bool_job", "negative_job"):
        submission["jobs"][0]["job_id"] = {"string_job": "100", "bool_job": True, "negative_job": -1}[change]
    elif change == "short_parent":
        submission["parent_commit"] = "abcdef"
    elif change == "uppercase_fork":
        submission["fork_commit"] = "B" * 40
    elif change in ("string_time", "bool_time", "over_time"):
        submission["job_limit_seconds"] = {"string_time": "7200", "bool_time": True, "over_time": 7201}[change]
    elif change in ("over_gpu_hours", "underdeclared_gpu_hours", "string_gpu_hours", "nan_gpu_hours"):
        submission["maximum_allocated_gpu_hours"] = {"over_gpu_hours": 12.1, "underdeclared_gpu_hours": 10, "string_gpu_hours": "12", "nan_gpu_hours": float("nan")}[change]
    elif change == "wrong_profile":
        submission["profile"] = "other"
    elif change == "partial_submission":
        submission["state"] = "submitting"
    elif change == "relative_snapshot":
        submission["source_snapshot"] = "snapshot"
    else:
        submission["schema_version"] = True
    (tmp_path / "submission.json").write_text(json.dumps(submission), encoding="utf-8")
    atomic_write_json(tmp_path / "standalone-gpu1" / "cell.json", complete_cell())
    report = aggregate_cells(tmp_path)
    assert report["submission"] is None
    assert any(error["path"].endswith("submission.json") for error in report["input_errors"])
    assert all(cell["status"] == "incomplete" and cell["estimate"] is None for cell in report["cells"])
    assert "submission identity invalid" in report["cells"][0]["reason"]


@pytest.mark.parametrize("change", [None, "job", "parent", "fork", "snapshot", "missing_source"])
def test_measured_cell_must_match_submission_source_and_job(tmp_path, change):
    submission = submission_record()
    cell = complete_cell()
    cell["jobs"] = {"slurm_job_id": str(submission["jobs"][0]["job_id"])}
    if change == "job":
        cell["jobs"]["slurm_job_id"] = "999"
    elif change in ("parent", "fork"):
        cell["source"][change + "_commit"] = "c" * 40
    elif change == "snapshot":
        cell["source"]["source_snapshot"] = "/wrong/snapshot"
    elif change == "missing_source":
        del cell["source"]
    atomic_write_json(tmp_path / "submission.json", submission)
    atomic_write_json(tmp_path / "standalone-gpu1" / "cell.json", cell)
    report = aggregate_cells(tmp_path)
    measured = report["cells"][0]
    assert measured["submission_job"] == submission["jobs"][0]
    assert measured["submission_identity_matches"] is (change is None)
    if change is None:
        assert measured["status"] == "complete"
        assert measured["estimate"] is not None
    else:
        assert measured["status"] == "incomplete"
        assert measured["estimate"] is None
        assert "submission identity" in measured["reason"]


def test_submission_may_use_a_smaller_consistent_budget_and_rejects_symlinks(tmp_path):
    submission = submission_record()
    submission.update(job_limit_seconds=3600, maximum_allocated_gpu_hours=6)
    original = tmp_path / "registry.json"
    atomic_write_json(original, submission)
    (tmp_path / "submission.json").symlink_to(original)
    rejected = aggregate_cells(tmp_path)
    assert rejected["submission"] is None
    assert "regular file" in rejected["input_errors"][0]["error"]
    (tmp_path / "submission.json").unlink()
    atomic_write_json(tmp_path / "submission.json", submission)
    assert aggregate_cells(tmp_path)["submission"]["job_limit_seconds"] == 3600


def test_report_surfaces_persisted_replay_failure_and_startup_without_fabricating_iterations(tmp_path):
    cell = complete_cell()
    cell.update(status="incomplete", reason="RuntimeError: pilot failed with exit code 1", iterations=[])
    del cell["startup_seconds"]
    failure = {
        "rollout_iteration": 0, "status": "failed_before_update", "stage": "pre_update_rollout_integrity",
        "error_type": "RuntimeError", "error": "RuntimeError: rollout/replay ratio error 0.125 exceeds 0.0001",
        "optimizer_updates_completed": False,
        "timing_s": {"step_partial": 35, "gen": 25, "old_log_prob": 8, "weight_sync": 1},
        "diagnostics": {"rollout_metrics": {"latent/cap_rate": 0.25, "latent/soft_to_hard_rate": 0.75},
                        "valid_boundary_count": 48, "worst_positions": [{"prompt_index": 123, "rollout_rank": 1, "response_position": 456, "segment": "soft_prefix", "rollout_log_density": -12, "actor_log_density": -11.88, "ratio_abs_error": 0.125}]},
    }
    cell["phases"].append({"phase": "pilot", "status": "incomplete", "measurement": {
        "status": "failed", "error": failure["error"], "failure": failure, "failed_iterations": [failure], "startup_seconds": 17,
    }})
    atomic_write_json(tmp_path / "cell.json", cell)
    report = aggregate_cells(tmp_path)
    measured = report["cells"][0]
    assert measured["status"] == "incomplete"
    assert measured["failure_status"] == "failed_before_update"
    assert measured["failure"] == failure
    assert measured["failed_iterations"] == [failure]
    assert measured["iterations"] == []
    assert measured["estimate"] is None
    assert measured["startup_seconds"] == 17
    assert "ratio error 0.125" in measured["reason"]
    assert "exit code 1" not in measured["reason"]
    markdown = render_markdown(report)
    assert "completed no optimizer updates" in markdown
    assert "prompt index 123, rollout rank 1, response position 456 (soft_prefix)" in markdown
    assert "rollout log density -12, actor log density -11.88" in markdown
    assert "| Startup | 17.00 |" in markdown
    assert "| Failed iteration 0 (partial; no update) | 35.00 | 25.00 | 8.00 |" in markdown


def test_old_failed_phases_surface_actual_error_without_new_diagnostic_fields(tmp_path):
    cell = complete_cell()
    cell.update(status="incomplete", reason="exit code 1", iterations=[])
    cell["phases"].append({"phase": "pilot", "status": "incomplete", "measurement": {"status": "failed", "error": "RuntimeError: rollout/replay ratio error 35310 exceeds 0.0001"}})
    atomic_write_json(tmp_path / "cell.json", cell)
    measured = aggregate_cells(tmp_path)["cells"][0]
    assert "ratio error 35310" in measured["reason"]
    assert measured["failure_status"] == "failed"
    assert measured["estimate"] is None


def repair_submission(tmp_path):
    return {"schema_version": 1, "role": "repair_validation", "state": "submitted", "submission_id": "repair-123",
            "job_id": 123, "objective": "standalone", "gpus": 2, "time_limit_seconds": 1800,
            "parent_commit": "a" * 40, "fork_commit": "b" * 40,
            "source_snapshot": "/scratch/snapshot", "run_root": str(tmp_path / "run"),
            "dispatch": dict(VARIANTS["bounded_async32"])}


@pytest.fixture
def repair_evidence(tmp_path):
    from opd_tools.manifest import canonical_sha256, file_sha256, write_manifest_atomic
    from opd_tools.training_runtime import MODEL_ID, _repair_overrides
    from opd_tools.training_benchmark import validate_pilot_metrics
    from verl.opd.provenance import _environment_identity, build_checkpoint_provenance

    submission = repair_submission(tmp_path)
    registry = tmp_path / "submission.json"
    atomic_write_json(registry, submission)
    model_root, data_root = tmp_path / "model", tmp_path / "data"
    model_root.mkdir()
    data_root.mkdir()
    def seal(path, value):
        value = {**value, "manifest_content_sha256": canonical_sha256(value)}
        write_manifest_atomic(path, value, validator=None)
        return value
    model = seal(model_root / "manifest.json", {"model": {"id": MODEL_ID, "resolved_revision": MODEL_REVISION}, "inventory_sha256": "c" * 64})
    for filename in ("train.parquet", "validation.parquet"):
        (data_root / filename).write_bytes(b"data")
    data = seal(data_root / "manifest.json", {"files": {name: {"size": 4, "sha256": "d" * 64} for name in ("train.parquet", "validation.parquet")}})
    config = {}
    for item in _repair_overrides(submission, "bounded_async32") + [
        "actor_rollout_ref.model.path=" + json.dumps(str(model_root)),
        "data.train_files=" + json.dumps(str(data_root / "train.parquet")),
        "data.val_files=" + json.dumps(str(data_root / "validation.parquet")),
    ]:
        key, encoded = item.lstrip("+").split("=", 1)
        target = config
        for part in key.split(".")[:-1]:
            target = target.setdefault(part, {})
        try:
            value = json.loads(encoded)
        except json.JSONDecodeError:
            value = encoded
        target[key.split(".")[-1]] = value
    provenance = build_checkpoint_provenance(config, source_commit="b" * 40, environment_identity=_environment_identity(package_versions={"torch": "2.6.0", "verl": "0.4.0"}))
    rows = iterations()
    for index, row in enumerate(rows):
        metrics = {"integrity/continuous_replay_active": 1.0, "replay/fallback_count": 0.0,
                   "replay/ratio_abs_error_max": 0.00001, "latent/soft_to_hard_rate": 0.98,
                   "latent/cap_rate": 0.02, "latent/close_tag_rate": 0.98,
                   "trainer/rollout_iteration": index, "trainer/optimizer_steps_this_iteration": 2.0,
                   "trainer/optimizer_step": 2 * (index + 1), "opd/ema_updates_this_iteration": 1.0,
                   "opd/ema_update_count": index + 1, "opd/beta_effective": 1.0,
                   "opd/latent_slot_count": 120.0, "opd/answer_slot_count": 48.0,
                   "grad/opd_norm": 0.01, "grad/total_norm": 0.3,
                   "integrity/checkpoint_committed": float(index == 2)}
        timing = {"teacher_seconds_max": 0.5, "ranks": [
            {"rank": rank, "teacher_seconds": 0.5, "policy_update_seconds": 1.0,
             "worker_update_seconds": 1.5, "optimizer_steps": 2.0,
             "ema_updates_this_iteration": 1.0, "ema_update_count": index + 1}
            for rank in range(2)]}
        row.update(metrics=metrics, actor_update_timing=timing,
                   pilot_acceptance=validate_pilot_metrics("standalone", index, metrics, actor_update_timing=timing, expected_ranks=2))
    rows[-1]["timing_s"]["save_checkpoint"] = 3.0
    measured = {"phase": "pilot", "variant": "bounded_async32", "status": "complete", "rows": [],
                "wandb_run_id": "run-1", "wandb_online": True, "wandb_finished": True,
                "configuration": config, "checkpoint_provenance": provenance,
                "iterations": rows, "failed_iterations": [], "startup_seconds": 17,
                "validation_seconds": 8, "validation_example_count": 128, "checkpoint_seconds": 3,
                "timing_only_validation_metrics": {"timing_val/accuracy": 0.5}}
    cell = complete_cell(gpus=2)
    assets = {"model": {"id": MODEL_ID, "revision": MODEL_REVISION},
              "model_manifest_content_sha256": model["manifest_content_sha256"],
              "data_manifest_content_sha256": data["manifest_content_sha256"]}
    assets["manifest_content_sha256"] = canonical_sha256(assets)
    cell.update(role="repair_validation", requested_dispatch="bounded_async32", status="repair_validation_complete",
                assets=assets, screen_rows=[], confirm_rows=[], jobs={"slurm_job_id": "123"}, wandb_run_ids=["run-1"])
    phase = {"phase": "pilot", "variant": "bounded_async32", "batches": [], "status": "complete",
             "authenticated": True, "wandb_run_id": "run-1", "measurement": measured}
    cell["phases"] = [phase]
    def persist():
        for key in ("startup_seconds", "iterations", "validation_seconds", "validation_example_count", "checkpoint_seconds", "checkpoint_provenance"):
            cell[key] = copy.deepcopy(measured[key])
        output = tmp_path / "phase.json"
        write_manifest_atomic(output, measured, validator=None)
        phase["sha256"] = file_sha256(output)
        atomic_write_json(tmp_path / "run" / "standalone-gpu2" / "cell.json", cell)
    persist()
    return registry, cell, measured, persist


def test_repair_pending_is_one_known_job_without_study_estimate(tmp_path):
    registry = tmp_path / "submission.json"
    atomic_write_json(registry, repair_submission(tmp_path))
    report = aggregate_repair_validation(registry)
    assert report["status"] == "pending"
    assert report["jobs"] == {"slurm_job_id": "123"}
    assert not report["input_errors"]
    assert report["submission"]["file_sha256"] == hashlib.sha256(registry.read_bytes()).hexdigest()
    assert "snapshot_verified" not in report["source"]
    assert report["full_training_estimate"] is report["allocation_recommendation"] is None
    assert "cells" not in report
    assert "four-cell runtime estimate" in render_repair_markdown(report)
    output = tmp_path / "reports"
    assert main(["repair", "--submission", str(registry), "--output-dir", str(output)]) == 0
    assert {path.name for path in output.iterdir()} == {"repair_report.md", "repair_report.json"}


def test_repair_complete_revalidates_pilot_and_hashes_without_loading_assets(repair_evidence, monkeypatch):
    registry, cell, measured, _ = repair_evidence
    import opd_tools.qwen_training as training
    monkeypatch.setattr(training, "verify", lambda *args: pytest.fail("report must not reread model assets"))
    report = aggregate_repair_validation(registry)
    assert report["status"] == "repair_validation_complete", report["reason"]
    assert report["measurement_authenticated"] is True
    assert len(report["pilot_gates"]) == 3
    assert not report["input_errors"]
    assert report["full_training_estimate"] is report["allocation_recommendation"] is None
    assert report["measurements"]["iterations"] == measured["iterations"]
    markdown = render_repair_markdown(report)
    assert "| Startup | 17.00 |" in markdown
    assert "| One authenticated checkpoint | 3.00 |" in markdown
    assert "Revalidated pilot iterations: 3/3" in markdown
    assert "Nested teacher time is never added" in markdown


@pytest.mark.parametrize("old_format", [False, True])
def test_repair_failure_preserves_exact_preupdate_reason_and_partial_timings(repair_evidence, old_format):
    registry, cell, measured, persist = repair_evidence
    failure = {"status": "failed_before_update", "stage": "pre_update_rollout_integrity", "rollout_iteration": 0,
               "error": "RuntimeError: rollout/replay ratio error 0.125 exceeds 0.0001", "optimizer_updates_completed": False,
               "timing_s": {"step_partial": 35, "gen": 25, "old_log_prob": 8},
               "diagnostics": {"worst_positions": [{"prompt_index": 3, "rollout_rank": 1, "response_position": 9,
                                                     "segment": "soft_prefix", "rollout_log_density": -12, "actor_log_density": -11.88}]}}
    measured.update(status="failed", iterations=[], error=failure["error"])
    if not old_format:
        measured.update(failure=failure, failed_iterations=[failure])
    cell.update(status="incomplete", reason="pilot failed with exit code 1")
    cell["phases"][0]["status"] = "incomplete"
    persist()
    report = aggregate_repair_validation(registry)
    assert report["status"] == "incomplete"
    assert "ratio error 0.125" in report["reason"]
    assert "exit code 1" not in report["reason"]
    assert not report["input_errors"]
    assert report["measurements"]["startup_seconds"] == 17
    assert not report["pilot_gates"]
    if not old_format:
        assert report["failure"] == failure
        text = render_repair_markdown(report)
        assert "completed no optimizer updates" in text
        assert "prompt index 3, rollout rank 1" in text
        assert "| Failed iteration 0 (partial; no update) | 35.00 |" in text


@pytest.mark.parametrize("key,value", [
    ("role", "training_benchmark"), ("state", "prepared"), ("schema_version", True),
    ("job_id", "123"), ("job_id", True), ("gpus", 4), ("gpus", True), ("objective", "standalone_opd"),
    ("time_limit_seconds", 1801), ("time_limit_seconds", True), ("internal_deadline_seconds", 1801),
    ("parent_commit", "bad"), ("fork_commit", "C" * 40), ("source_snapshot", "relative"),
    ("run_root", "relative"), ("dispatch", {"dispatch_mode": "bounded_async", "max_running_requests": True, "async_queue_size": 64}),
    ("jobs", []), ("maximum_repair_gpu_hours", 2),
])
def test_repair_manifest_rejects_wrong_scope_types_pins_or_allocation(tmp_path, key, value):
    submission = repair_submission(tmp_path)
    submission[key] = value
    registry = tmp_path / "submission.json"
    atomic_write_json(registry, submission)
    report = aggregate_repair_validation(registry)
    assert report["status"] == "incomplete"
    assert report["submission"] is None
    assert report["input_errors"]


@pytest.mark.parametrize("change", ["source", "job", "objective", "dispatch", "phase_hash", "config", "sealed_config", "seal", "assets", "wandb", "rank", "iteration", "acceptance", "checkpoint", "validation", "copy", "cap", "close"])
def test_repair_tampering_or_missing_acceptance_never_passes(repair_evidence, change):
    registry, cell, measured, persist = repair_evidence
    if change == "source": cell["source"]["fork_commit"] = "c" * 40
    elif change == "job": cell["jobs"]["slurm_job_id"] = "999"
    elif change == "objective": cell["objective"] = "hybrid"
    elif change == "dispatch": cell["requested_dispatch"] = "legacy_batch"
    elif change in ("config", "sealed_config"):
        measured["configuration"]["trainer"]["max_rollout_iterations_per_invocation"] = 2
        if change == "sealed_config":
            from verl.opd.provenance import build_checkpoint_provenance
            measured["checkpoint_provenance"] = build_checkpoint_provenance(
                measured["configuration"], source_commit="b" * 40,
                environment_identity=measured["checkpoint_provenance"]["environment"])
    elif change == "seal": measured["checkpoint_provenance"]["source"]["commit"] = "d" * 40
    elif change == "assets": cell["assets"]["model_manifest_content_sha256"] = "e" * 64
    elif change == "wandb": measured["wandb_finished"] = False
    elif change == "rank": measured["iterations"][1]["actor_update_timing"]["ranks"][1]["ema_update_count"] = 1
    elif change == "iteration": measured["iterations"].pop()
    elif change == "acceptance": measured["iterations"][1]["pilot_acceptance"]["accepted"] = False
    elif change == "checkpoint": measured["iterations"][-1]["metrics"]["integrity/checkpoint_committed"] = 0
    elif change == "validation": measured["validation_example_count"] = 512
    elif change == "cap": measured["iterations"][0]["metrics"]["latent/cap_rate"] = 0.25
    elif change == "close": measured["iterations"][0]["metrics"]["latent/close_tag_rate"] = 0.75
    persist()
    if change == "phase_hash": cell["phases"][0]["sha256"] = "f" * 64
    if change == "copy": cell["startup_seconds"] = 999
    atomic_write_json(registry.parent / "run" / "standalone-gpu2" / "cell.json", cell)
    report = aggregate_repair_validation(registry)
    assert report["status"] == "incomplete", change
    assert report["input_errors"]
    assert report["full_training_estimate"] is None


def test_repair_input_root_cannot_be_an_original_study_matrix(tmp_path):
    registry = tmp_path / "repair-submission.json"
    atomic_write_json(registry, repair_submission(tmp_path))
    matrix = tmp_path / "matrix"
    atomic_write_json(matrix / "standalone-gpu1" / "cell.json", complete_cell())
    report = aggregate_repair_validation(registry, matrix)
    assert "study matrices are separate" in report["reason"]
    original = aggregate_cells(matrix)
    assert len(original["cells"]) == 4
    assert original["cells"][0]["status"] == "complete"


def test_repair_running_without_phase_measurements_is_incomplete_not_invalid(repair_evidence):
    registry, cell, _, _ = repair_evidence
    cell.update(status="running", phases=[], iterations=[], wandb_run_ids=[])
    atomic_write_json(registry.parent / "run" / "standalone-gpu2" / "cell.json", cell)
    report = aggregate_repair_validation(registry)
    assert report["status"] == "incomplete"
    assert report["reason"] == "repair status is running"
    assert not report["input_errors"]


def test_repair_local_collection_override_retains_submitted_identity(repair_evidence, tmp_path):
    registry, cell, _, _ = repair_evidence
    collected = tmp_path / "local-copy"
    atomic_write_json(collected / "cell.json", cell)
    report = aggregate_repair_validation(registry, collected)
    assert report["status"] == "repair_validation_complete", report["reason"]
    assert report["submission"]["run_root"] != str(collected)
    report["reporter"] = {"parent_commit": "c" * 40, "fork_commit": "d" * 40}
    assert "Reporter source (separate from benchmark)" in render_repair_markdown(report)


def test_repair_malformed_measurement_still_publishes_an_incomplete_report(repair_evidence, tmp_path):
    registry, _, measured, persist = repair_evidence
    measured["iterations"] = [None]
    persist()
    report = aggregate_repair_validation(registry)
    assert report["status"] == "incomplete"
    assert report["input_errors"]
    assert "malformed measurement fields" in render_repair_markdown(report)
    output = tmp_path / "reports"
    assert main(["repair", "--submission", str(registry), "--output-dir", str(output)]) == 0
    assert json.loads((output / "repair_report.json").read_text())["status"] == "incomplete"
