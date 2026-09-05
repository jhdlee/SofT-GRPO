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
    atomic_write_json,
    estimate_runtime,
    main,
    pick_candidate,
    recommend_allocation,
    render_markdown,
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
