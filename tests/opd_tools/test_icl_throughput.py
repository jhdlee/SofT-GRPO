from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest

from opd_tools.icl import CORE_CONDITIONS, QWEN3_STUDY_ID, request_seed
from opd_tools.icl_throughput import (
    BOOTSTRAP_RESAMPLES,
    RUNTIME_LIMIT_SECONDS,
    BenchmarkObservation,
    RequestTiming,
    ThroughputExample,
    build_throughput_plan,
    choose_smallest_data_parallel_size,
    estimate_model_runtime,
    validate_observation,
)
from opd_tools.icl_eval import (
    _artifact_inventory,
    _async_queue_metrics,
    _compare_throughput_artifacts,
    _compare_throughput_manifests,
    _publish_resource_metrics,
    _validate_throughput_resource_evidence,
    _validate_throughput_timing_evidence,
)
from opd_tools.icl_runtime import TrajectoryMetadata


def _examples():
    values = []
    subjects = ("algebra", "geometry", "number_theory", "counting")
    levels = tuple("level_%d" % index for index in range(1, 6))
    for index in range(500):
        values.append(
            ThroughputExample(
                example_id="math-%03d" % index,
                benchmark="math500",
                subject=subjects[index % len(subjects)],
                level=levels[(index // len(subjects)) % len(levels)],
                prompt_tokens={
                    condition: 50 + index % 97 + offset * (index % 11)
                    for offset, condition in enumerate(CORE_CONDITIONS)
                },
            )
        )
    for index in range(30):
        values.append(
            ThroughputExample(
                example_id="aime-%02d" % index,
                benchmark="aime2024",
                subject="competition_math",
                level="aime2024",
                prompt_tokens={
                    condition: 100 + 7 * index + offset
                    for offset, condition in enumerate(CORE_CONDITIONS)
                },
            )
        )
    return tuple(values)


def _observations(plan, *, dp, generation_wall, model="qwen3_0p6b"):
    values = []
    for mode_index, mode in enumerate(("native_soft", "hard_token")):
        timings = tuple(
            RequestTiming(
                condition=request.condition,
                benchmark=request.benchmark,
                example_id=request.example_id,
                request_seed=request.request_seed,
                elapsed_seconds=10.0 + mode_index,
                generated_tokens=100 + mode_index,
            )
            for request in plan.requests
        )
        values.append(
            BenchmarkObservation(
                model_label=model,
                inference_mode=mode,
                data_parallel_size=dp,
                plan_sha256=plan.content_sha256,
                timings=timings,
                generation_wall_seconds=generation_wall,
                engine_load_seconds=2.0 + mode_index,
                finalization_seconds=3.0 + mode_index,
            )
        )
    return tuple(values)


def test_plan_is_deterministic_stratified_and_exactly_66_requests():
    examples = _examples()
    first = build_throughput_plan(examples)
    second = build_throughput_plan(tuple(reversed(examples)))
    assert first == second
    assert first.content_sha256 == second.content_sha256
    assert len(first.requests) == 66
    assert sum(
        request.benchmark == "math500" for request in first.requests
    ) == 60
    assert sum(
        request.benchmark == "aime2024" for request in first.requests
    ) == 6
    for condition in CORE_CONDITIONS:
        math = [
            request
            for request in first.requests
            if request.condition == condition and request.benchmark == "math500"
        ]
        aime = [
            request
            for request in first.requests
            if request.condition == condition and request.benchmark == "aime2024"
        ]
        assert len(math) == 20
        assert len(aime) == 2
        assert len({request.subject for request in math}) > 1
        assert len({request.level for request in math}) > 1
        assert len({request.length_quantile for request in math}) > 1
        assert {request.length_quantile for request in aime} == {0, 2}
        for request in math + aime:
            assert request.request_seed == request_seed(
                request.benchmark,
                request.example_id,
                0,
                study=QWEN3_STUDY_ID,
            )
    assert sum(
        count
        for by_benchmark in first.to_dict()["production_counts"].values()
        for count in by_benchmark.values()
    ) == 1_590


def test_plan_rejects_incomplete_duplicates_and_prompt_contract_drift():
    examples = list(_examples())
    with pytest.raises(ValueError, match="full MATH-500"):
        build_throughput_plan(examples[:-1])
    with pytest.raises(ValueError, match="unique"):
        build_throughput_plan([*examples[:-1], examples[0]])
    with pytest.raises(ValueError, match="three core"):
        ThroughputExample(
            example_id="bad",
            benchmark="math500",
            subject="algebra",
            level="level_1",
            prompt_tokens={"no_demo": 10},
        )


def test_observations_are_bound_to_identical_ids_and_seeds_for_dp1_dp2():
    plan = build_throughput_plan(_examples())
    dp1 = _observations(plan, dp=1, generation_wall=100.0)
    dp2 = _observations(plan, dp=2, generation_wall=60.0)
    for observation in (*dp1, *dp2):
        validate_observation(plan, observation)
        assert [timing.key for timing in observation.timings] == [
            request.key for request in plan.requests
        ]
    wrong = replace(
        dp1[0],
        timings=(
            replace(dp1[0].timings[0], request_seed=17),
            *dp1[0].timings[1:],
        ),
    )
    with pytest.raises(ValueError, match="identities or seeds"):
        validate_observation(plan, wrong)


def test_runtime_uses_10000_stratified_resamples_exact_weights_and_two_overheads():
    plan = build_throughput_plan(_examples())
    observations = _observations(plan, dp=1, generation_wall=660.0)
    estimate = estimate_model_runtime(plan, observations)
    # Both modes project their exact 1,590-request token workload at the
    # aggregate token throughput observed during the benchmark wall interval.
    assert estimate.generation_seconds == pytest.approx(31_800.0)
    # Two loads and two finalizations: (2+3) + (3+4).
    assert estimate.engine_load_seconds == 5.0
    assert estimate.finalization_seconds == 7.0
    assert estimate.overhead_seconds == 12.0
    assert estimate.total_seconds == pytest.approx(31_812.0)
    assert estimate.ci_low_seconds == pytest.approx(estimate.total_seconds)
    assert estimate.ci_high_seconds == pytest.approx(estimate.total_seconds)
    assert estimate.bootstrap_resamples == BOOTSTRAP_RESAMPLES
    payload = estimate.to_dict()
    assert payload["uncertainty_interpretation"] == (
        "conditional_workload_composition_at_observed_throughput"
    )
    assert payload["between_job_system_variability_included"] is False
    assert set(estimate.effective_concurrency) == {"native_soft", "hard_token"}
    with pytest.raises(ValueError, match="10,000"):
        estimate_model_runtime(plan, observations, resamples=999)


def test_parallelism_decision_runs_dp1_then_dp2_and_falls_back_to_dp8():
    plan = build_throughput_plan(_examples())
    empty = choose_smallest_data_parallel_size(plan, {})
    assert empty.status == "benchmark_required"
    assert empty.next_benchmark_data_parallel_size == 1

    fast_dp1 = _observations(plan, dp=1, generation_wall=660.0)
    decision = choose_smallest_data_parallel_size(plan, {1: fast_dp1})
    assert decision.selected_data_parallel_size == 1
    assert decision.next_benchmark_data_parallel_size is None

    slow_dp1 = _observations(plan, dp=1, generation_wall=1_500.0)
    decision = choose_smallest_data_parallel_size(plan, {1: slow_dp1})
    assert decision.status == "benchmark_required"
    assert decision.next_benchmark_data_parallel_size == 2
    assert decision.estimates[1].ci_high_seconds > RUNTIME_LIMIT_SECONDS

    fast_dp2 = _observations(plan, dp=2, generation_wall=600.0)
    decision = choose_smallest_data_parallel_size(
        plan, {1: slow_dp1, 2: fast_dp2}
    )
    assert decision.selected_data_parallel_size == 2

    slow_dp2 = _observations(plan, dp=2, generation_wall=1_500.0)
    decision = choose_smallest_data_parallel_size(
        plan, {1: slow_dp1, 2: slow_dp2}
    )
    assert decision.selected_data_parallel_size == 8


def test_runtime_rejects_mixed_model_mode_or_dp_evidence():
    plan = build_throughput_plan(_examples())
    observations = _observations(plan, dp=1, generation_wall=100.0)
    with pytest.raises(ValueError, match="native-soft and hard-token"):
        estimate_model_runtime(plan, (observations[0], observations[0]))
    with pytest.raises(ValueError, match="one model"):
        estimate_model_runtime(
            plan,
            (observations[0], replace(observations[1], model_label="other")),
        )
    with pytest.raises(ValueError, match="one DP"):
        estimate_model_runtime(
            plan,
            (observations[0], replace(observations[1], data_parallel_size=2)),
        )


def test_dp_equivalence_checks_exact_actions_and_tolerant_perturbed_values():
    fields = {
        "response": "reasoning</think>\\boxed{1}",
        "finish_reason": "stop",
        "response_token_count": 3,
        "capped": False,
        "latent_token_count": 2,
        "hard_token_count": 1,
        "close_tag": True,
        "soft_to_hard": True,
        "all_soft": False,
        "boxed_answer": True,
        "boundary_valid": True,
    }
    first = TrajectoryMetadata(
        response_token_ids=(10, 20, 2),
        latent_support_ids=((10, 11, 12, 13, 14), (20, 21, 22, 23, 24)),
        latent_perturbed_logits=((2.0, 1.0, 0.0, -1.0, -2.0),) * 2,
        latent_gumbel_noise=((0.5, 0.4, 0.3, 0.2, 0.1),) * 2,
    )
    second = TrajectoryMetadata(
        response_token_ids=first.response_token_ids,
        latent_support_ids=first.latent_support_ids,
        latent_perturbed_logits=((2.0001, 1.0, 0.0, -1.0, -2.0),) * 2,
        latent_gumbel_noise=first.latent_gumbel_noise,
    )
    result = _compare_throughput_artifacts(
        {("no_demo", "math500", "x"): (SimpleNamespace(**fields), first)},
        {("no_demo", "math500", "x"): (SimpleNamespace(**fields), second)},
    )
    assert result["request_count"] == 1
    assert result["latent_slot_count"] == 2
    assert result["top_five_support_exact"]
    assert 0 < result["perturbed_logits_abs_error_max"] < 5e-3

    changed_noise = replace(
        second,
        latent_gumbel_noise=((0.6, 0.4, 0.3, 0.2, 0.1),) * 2,
    )
    with pytest.raises(RuntimeError, match="Gumbel"):
        _compare_throughput_artifacts(
            {("no_demo", "math500", "x"): (SimpleNamespace(**fields), first)},
            {
                ("no_demo", "math500", "x"): (
                    SimpleNamespace(**fields),
                    changed_noise,
                )
            },
        )


def test_async_queue_metrics_capture_capacity_peak_and_replica_activity():
    rows = [
        {
            "request_index": 0,
            "submitted_at": 0.0,
            "completed_at": 2.0,
            "latency_seconds": 2.0,
        },
        {
            "request_index": 1,
            "submitted_at": 0.0,
            "completed_at": 1.0,
            "latency_seconds": 1.0,
        },
        {
            "request_index": 2,
            "submitted_at": 1.0,
            "completed_at": 3.0,
            "latency_seconds": 2.0,
        },
    ]
    metrics = _async_queue_metrics(rows, queue_size=4, data_parallel_size=2)
    assert metrics["maximum"] == 2
    assert metrics["capacity"] == 4
    assert metrics["mean"] == pytest.approx(5 / 3)
    assert metrics["mean_fraction"] == pytest.approx(5 / 12)
    assert "replica_busy_fraction" not in metrics


def test_resource_metrics_publish_real_assigned_gpu_values_without_pseudo_replicas():
    run = SimpleNamespace(summary={})
    _publish_resource_metrics(
        run,
        {
            "sample_count": 3,
            "host_metrics_available": True,
            "gpu_metrics_available": True,
            "gpu_selection_source": "SLURM_STEP_GPUS",
            "peak_hbm_gib_aggregate": 7.0,
            "peak_hbm_gib_per_gpu": {"3": 3.0, "5": 4.0},
            "gpu_utilization_mean_per_gpu": {"3": 30.0, "5": 40.0},
        },
    )
    assert run.summary["system/peak_hbm_gib_aggregate"] == 7.0
    assert run.summary["system/gpu/3/peak_hbm_gib"] == 3.0
    assert run.summary["system/gpu/5/peak_hbm_gib"] == 4.0
    assert run.summary["system/gpu/3/utilization_mean"] == 30.0
    assert run.summary["system/gpu/5/utilization_mean"] == 40.0
    assert run.summary["system/gpu_selection_source"] == "SLURM_STEP_GPUS"
    assert run.summary["system/gpu_metrics_available"] is True
    assert not any("replica" in name for name in run.summary)


def test_input_inventory_hashes_every_regular_report_artifact(tmp_path):
    first = tmp_path / "generation_manifest.json"
    second = tmp_path / "cell" / "chunk_00000.replay.npz"
    second.parent.mkdir()
    first.write_bytes(b"manifest\n")
    second.write_bytes(b"replay\n")
    inventory = _artifact_inventory(tmp_path, (second, first, first))
    assert [entry["path"] for entry in inventory["files"]] == [
        "cell/chunk_00000.replay.npz",
        "generation_manifest.json",
    ]
    by_path = {entry["path"]: entry for entry in inventory["files"]}
    assert by_path["generation_manifest.json"]["sha256"] == hashlib.sha256(
        b"manifest\n"
    ).hexdigest()
    assert len(inventory["content_sha256"]) == 64


def _manifest_for_dp(dp):
    return {
        "protocol": "generation-v1",
        "source_provenance": {
            "fork_commit": "a" * 40,
            "implementation_sha256": "b" * 64,
        },
        "model_label": "qwen3_0p6b",
        "mode": "native_soft",
        "sampling": {"temperature": 1.0},
        "parallelism": {
            "tensor_parallel_size": 1,
            "data_parallel_size": dp,
            "world_size": dp,
            "load_balance_method": "round_robin",
        },
        "request_queue_size": 32 * dp,
        "max_running_requests_per_replica": 16,
        "max_running_requests_aggregate": 16 * dp,
        "warmup": {
            "request_count": dp,
            "max_new_tokens": 32,
            "excluded_from_timing": True,
        },
        "wandb_run_id": "dp%d" % dp,
    }


def test_dp_manifests_bind_all_non_topology_config_and_reject_mixed_provenance():
    first = _manifest_for_dp(1)
    second = _manifest_for_dp(2)
    result = _compare_throughput_manifests(first, second)
    assert result["non_topology_config_exact"]
    assert len(result["normalized_manifest_sha256"]) == 64

    mixed = _manifest_for_dp(2)
    mixed["source_provenance"]["fork_commit"] = "c" * 40
    with pytest.raises(RuntimeError, match="source_provenance"):
        _compare_throughput_manifests(first, mixed)

    changed_sampling = _manifest_for_dp(2)
    changed_sampling["sampling"]["temperature"] = 0.6
    with pytest.raises(RuntimeError, match="sampling"):
        _compare_throughput_manifests(first, changed_sampling)

    changed_tp = _manifest_for_dp(2)
    changed_tp["parallelism"]["tensor_parallel_size"] = 2
    with pytest.raises(RuntimeError, match="parallelism"):
        _compare_throughput_manifests(first, changed_tp)


def _synthetic_timing_evidence():
    records = {
        ("no_demo", "math500", "math-1"): SimpleNamespace(
            request_seed=11,
            response_token_count=3,
            capped=False,
            all_soft=False,
            soft_to_hard=True,
        ),
        ("sdft_matched", "aime2024", "aime-1"): SimpleNamespace(
            request_seed=22,
            response_token_count=5,
            capped=True,
            all_soft=True,
            soft_to_hard=False,
        ),
    }
    artifacts = {key: (record, object()) for key, record in records.items()}
    locations = {
        ("no_demo", "math500", "math-1"): (0, 0),
        ("sdft_matched", "aime2024", "aime-1"): (0, 0),
    }
    rows = [
        {
            "timing_session_id": "one-session",
            "request_index": 0,
            "condition": "no_demo",
            "benchmark": "math500",
            "sample_index": 0,
            "chunk_index": 0,
            "chunk_row": 0,
            "example_id": "math-1",
            "submitted_at": 1.0,
            "completed_at": 3.0,
            "latency_seconds": 2.0,
            "response_tokens": 3,
            "request_seed": 11,
            "capped": False,
            "all_soft": False,
            "soft_to_hard": True,
        },
        {
            "timing_session_id": "one-session",
            "request_index": 1,
            "condition": "sdft_matched",
            "benchmark": "aime2024",
            "sample_index": 0,
            "chunk_index": 0,
            "chunk_row": 0,
            "example_id": "aime-1",
            "submitted_at": 2.0,
            "completed_at": 5.0,
            "latency_seconds": 3.0,
            "response_tokens": 5,
            "request_seed": 22,
            "capped": True,
            "all_soft": True,
            "soft_to_hard": False,
        },
    ]
    queue = _async_queue_metrics(rows, queue_size=32, data_parallel_size=1)
    payload = {
        "resumed": False,
        "request_count": 2,
        "response_tokens": 8,
        "response_length_mean": 4.0,
        "response_length_tokens": {"p50": 4.0, "p95": 4.9, "max": 5},
        "cap_rate": 0.5,
        "soft_to_hard_rate": 0.5,
        "generation_seconds": 4.0,
        "final_queue_drain_seconds": 3.0,
        "tokens_per_second": 2.0,
        "requests_per_hour": 1800.0,
        "latency_seconds": {"p50": 2.5, "p95": 2.95, "max": 3.0},
        "queue_occupancy": queue,
        "rows": rows,
    }
    return payload, artifacts, locations


def test_timing_evidence_is_recomputed_from_authenticated_completion_records():
    payload, artifacts, locations = _synthetic_timing_evidence()
    timings, recomputed = _validate_throughput_timing_evidence(
        payload,
        artifacts,
        locations,
        queue_size=32,
        data_parallel_size=1,
    )
    assert len(timings) == 2
    assert sum(value.generated_tokens for value in timings) == 8
    assert recomputed["cap_rate"] == 0.5
    assert recomputed["soft_to_hard_rate"] == 0.5

    tampered, artifacts, locations = _synthetic_timing_evidence()
    tampered["rows"][0]["response_tokens"] = 30
    with pytest.raises(ValueError, match="response_tokens"):
        _validate_throughput_timing_evidence(
            tampered,
            artifacts,
            locations,
            queue_size=32,
            data_parallel_size=1,
        )

    tampered, artifacts, locations = _synthetic_timing_evidence()
    tampered["latency_seconds"]["p95"] = 99.0
    with pytest.raises(ValueError, match="latency_seconds.p95"):
        _validate_throughput_timing_evidence(
            tampered,
            artifacts,
            locations,
            queue_size=32,
            data_parallel_size=1,
        )


def test_authenticated_partial_resume_timing_sessions_remain_usable():
    payload, artifacts, locations = _synthetic_timing_evidence()
    payload["resumed"] = True
    payload["rows"][1]["timing_session_id"] = "resumed-session"
    payload["rows"][1]["request_index"] = 0
    payload["generation_seconds"] = 5.0
    payload["final_queue_drain_seconds"] = 5.0
    payload["tokens_per_second"] = 1.6
    payload["requests_per_hour"] = 1440.0
    payload["queue_occupancy"] = _async_queue_metrics(
        payload["rows"], queue_size=32, data_parallel_size=1
    )
    timings, recomputed = _validate_throughput_timing_evidence(
        payload,
        artifacts,
        locations,
        queue_size=32,
        data_parallel_size=1,
    )
    assert len(timings) == 2
    assert recomputed["queue_occupancy"]["timing_session_count"] == 2
    assert recomputed["generation_seconds"] == 5.0

    duplicate, artifacts, locations = _synthetic_timing_evidence()
    duplicate["resumed"] = True
    duplicate["rows"][1]["request_index"] = 0
    with pytest.raises(ValueError, match="repeat within timing session"):
        _validate_throughput_timing_evidence(
            duplicate,
            artifacts,
            locations,
            queue_size=32,
            data_parallel_size=1,
        )


def _system_evidence(dp):
    gpu_ids = [str(index) for index in range(dp)]
    return {
        "sample_count": 4,
        "host_sample_count": 4,
        "gpu_sample_count": 4,
        "peak_hbm_gib_per_gpu": {gpu_id: 12.0 for gpu_id in gpu_ids},
        "peak_hbm_gib_aggregate": 12.0 * dp,
        "peak_host_ram_gib": 32.0,
        "cpu_utilization_mean": 50.0,
        "cpu_utilization_peak": 75.0,
        "gpu_utilization_mean": 80.0,
        "gpu_utilization_mean_per_gpu": {gpu_id: 80.0 for gpu_id in gpu_ids},
        "host_metrics_available": True,
        "gpu_metrics_available": True,
        "gpu_selection_source": "SLURM_STEP_GPUS",
        "gpu_selectors": gpu_ids,
        "expected_gpu_count": dp,
    }


def test_allocation_requires_usable_assigned_gpu_and_host_telemetry():
    result = _validate_throughput_resource_evidence(
        _system_evidence(2), data_parallel_size=2
    )
    assert result["validated"]
    assert result["gpu_ids"] == ["0", "1"]

    wrong_count = _system_evidence(2)
    wrong_count["expected_gpu_count"] = 1
    with pytest.raises(ValueError, match="expected GPU count"):
        _validate_throughput_resource_evidence(
            wrong_count, data_parallel_size=2
        )

    missing_utilization = _system_evidence(1)
    missing_utilization["gpu_utilization_mean_per_gpu"] = None
    with pytest.raises(ValueError, match="per-GPU telemetry"):
        _validate_throughput_resource_evidence(
            missing_utilization, data_parallel_size=1
        )
