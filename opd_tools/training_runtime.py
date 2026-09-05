"""Pure-CPU selection and preliminary runtime reports for Qwen3 training.

Input durations are critical-path wall times, never sums across GPU ranks.
``step`` encloses the training stages and any ``testing``/``save_checkpoint``;
``weight_sync`` (when provided) is nested in ``gen``. Teacher timings are nested
in ``update_actor`` and must not be counted a second time. Scenario multipliers
are explicit planning assumptions, not statistical confidence intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


PROTOCOL = "opd-qwen3-training-runtime-v1"
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
OBJECTIVES = ("standalone", "hybrid")
VARIANTS = {
    "legacy_batch": {"dispatch_mode": "legacy_batch", "max_running_requests": 16, "async_queue_size": 32},
    "expanded_batch": {"dispatch_mode": "expanded_batch", "max_running_requests": 16, "async_queue_size": 32},
    "bounded_async16": {"dispatch_mode": "bounded_async", "max_running_requests": 16, "async_queue_size": 32},
    "bounded_async32": {"dispatch_mode": "bounded_async", "max_running_requests": 32, "async_queue_size": 64},
}
TOKEN_STAGES = ("gen", "old_log_prob", "ref", "update_actor")
ROLLOUT_ITERATIONS = 109
OPTIMIZER_STEPS = 218
TRAIN_EXAMPLES = 6985
TRAIN_BATCH_SIZE = 64
MAX_RESPONSE_TOKENS = 8192
VALIDATION_EVENTS = 6
VALIDATION_EXAMPLES = 512
TIMING_VALIDATION_EXAMPLES = 128
CHECKPOINT_SAVES = 5
RUNTIME_LIMIT_HOURS = 12
JOB_LIMIT_HOURS = 2
MAX_ALLOCATED_GPU_HOURS = 12


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def _objective(value: str) -> str:
    value = {"standalone_opd": "standalone", "grpo_opd": "hybrid"}.get(value, value)
    if value not in OBJECTIVES:
        raise ValueError(f"unknown objective: {value!r}")
    return value


def _gpu_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2):
        raise ValueError("gpus must be 1 or 2")
    return value


def _row(rows: Sequence[Mapping[str, Any]], variant: str, batch: int) -> dict[str, Any] | None:
    if not isinstance(rows, (list, tuple)) or any(not isinstance(row, Mapping) for row in rows):
        return None
    matches = [row for row in rows if row.get("variant") == variant and row.get("batch_index") == batch]
    if len(matches) != 1 or matches[0].get("valid") is not True:
        return None
    row = matches[0]
    try:
        wall = _number(row.get("wall_seconds"), "wall_seconds", positive=True)
        tokens = _number(row.get("generated_tokens"), "generated_tokens", positive=True)
    except ValueError:
        return None
    return {**row, "wall_seconds": wall, "generated_tokens": tokens, "tokens_per_second": tokens / wall}


def pick_candidate(screen_rows: Sequence[Mapping[str, Any]]) -> str:
    """Choose the shortest valid nonlegacy batch-0 screen, deterministically."""
    candidates = [_row(screen_rows, variant, 0) for variant in VARIANTS if variant != "legacy_batch"]
    valid = [row for row in candidates if row is not None]
    if not valid:
        return "legacy_batch"
    return min(valid, key=lambda row: (row["wall_seconds"], list(VARIANTS).index(row["variant"])))["variant"]


def select_dispatch(
    screen_rows: Sequence[Mapping[str, Any]], confirm_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Accept the screened candidate only when both independent batches improve.

    A shorter response cannot pass on wall time alone: throughput must also
    strictly improve. Missing, invalid, or duplicate measurements fail closed.
    """
    candidate = pick_candidate(screen_rows)
    comparisons = []
    for batch, rows in ((0, screen_rows), (1, confirm_rows)):
        baseline = _row(rows, "legacy_batch", batch)
        proposed = _row(rows, candidate, batch) if candidate != "legacy_batch" else None
        comparison: dict[str, Any] = {"batch_index": batch, "valid": baseline is not None and proposed is not None}
        if comparison["valid"]:
            comparison.update({
                "legacy_wall_seconds": baseline["wall_seconds"],
                "candidate_wall_seconds": proposed["wall_seconds"],
                "legacy_tokens_per_second": baseline["tokens_per_second"],
                "candidate_tokens_per_second": proposed["tokens_per_second"],
                "wall_speedup": baseline["wall_seconds"] / proposed["wall_seconds"],
                "throughput_speedup": proposed["tokens_per_second"] / baseline["tokens_per_second"],
                "generated_tokens_difference": proposed["generated_tokens"] - baseline["generated_tokens"],
                "improved": proposed["wall_seconds"] < baseline["wall_seconds"] and proposed["tokens_per_second"] > baseline["tokens_per_second"],
            })
        else:
            comparison["improved"] = False
        comparisons.append(comparison)
    accepted = candidate != "legacy_batch" and all(item["improved"] for item in comparisons)
    selected = candidate if accepted else "legacy_batch"
    return {
        "selected_variant": selected,
        "candidate_variant": candidate,
        "accepted": accepted,
        "reason": "wall time and throughput improved on both batches" if accepted else "candidate lacked two valid improvements in both wall time and throughput",
        "comparisons": comparisons,
        "configuration": dict(VARIANTS[selected]),
    }


def _iteration_cost(row: Mapping[str, Any]) -> dict[str, float]:
    timings = row.get("timing_s")
    if not isinstance(timings, Mapping):
        raise ValueError("each iteration requires a timing_s mapping")
    values = {key: _number(value, f"timing_s.{key}") for key, value in timings.items()}
    for key in ("step", "gen", "old_log_prob", "update_actor"):
        if key not in values:
            raise ValueError(f"iteration is missing timing_s.{key}")
    training = values["step"] - values.get("testing", 0) - values.get("save_checkpoint", 0)
    _number(training, "training iteration wall seconds", positive=True)
    if "non_token_seconds" in row:
        fixed = _number(row["non_token_seconds"], "non_token_seconds")
        token = training - fixed
    else:
        weight_sync = values.get("weight_sync", 0)
        if weight_sync > values["gen"]:
            raise ValueError("weight_sync must be nested within gen")
        token = sum(values.get(stage, 0) for stage in TOKEN_STAGES) - weight_sync
        fixed = training - token
    tolerance = max(1e-6, training * 1e-6)
    if token < -tolerance or fixed < -tolerance:
        raise ValueError("token and non-token durations exceed the iteration wall time; check nested timers")
    token = min(training, max(0.0, token))
    return {"wall_seconds": training, "token_seconds": token, "non_token_seconds": training - token}


def estimate_runtime(
    objective: str,
    gpus: int,
    startup_seconds: float,
    iterations: Sequence[Mapping[str, Any]],
    validation_seconds: float,
    checkpoint_seconds: float,
) -> dict[str, Any]:
    """Extrapolate three measured iterations, six validations, and five saves.

    Iterations 1 and 2 estimate the active cost for both objectives. Hybrid also
    retains its zero-dose iteration 0 separately. The entire 128-example timing
    validation is conservatively treated as token-dependent when scaled.
    """
    objective = _objective(objective)
    gpus = _gpu_count(gpus)
    startup = _number(startup_seconds, "startup_seconds")
    validation = _number(validation_seconds, "validation_seconds", positive=True)
    checkpoint = _number(checkpoint_seconds, "checkpoint_seconds", positive=True)
    by_index = {}
    for row in iterations:
        index = row.get("rollout_iteration")
        if isinstance(index, bool) or not isinstance(index, int) or index not in (0, 1, 2) or index in by_index:
            raise ValueError("iterations must contain unique rollout_iteration values 0, 1, 2")
        by_index[index] = _iteration_cost(row)
    if set(by_index) != {0, 1, 2}:
        raise ValueError("all three completed iterations 0, 1, 2 are required")
    active = {key: mean(by_index[index][key] for index in (1, 2)) for key in by_index[1]}
    zero = by_index[0] if objective == "hybrid" else {key: 0.0 for key in active}
    active_count = ROLLOUT_ITERATIONS - (1 if objective == "hybrid" else 0)
    validation_total = validation * VALIDATION_EXAMPLES / TIMING_VALIDATION_EXAMPLES * VALIDATION_EVENTS
    checkpoint_total = checkpoint * CHECKPOINT_SAVES
    scenarios = {}
    for multiplier in (1.0, 1.5, 2.0):
        zero_time = zero["non_token_seconds"] + multiplier * zero["token_seconds"]
        active_time = active_count * (active["non_token_seconds"] + multiplier * active["token_seconds"])
        validation_time = multiplier * validation_total
        total = startup + zero_time + active_time + validation_time + checkpoint_total
        scenarios[str(multiplier)] = {
            "token_multiplier": multiplier,
            "seconds": total,
            "hours": total / 3600,
            "gpu_hours": total / 3600 * gpus,
            "components_seconds": {"startup": startup, "zero_dose_iteration": zero_time, "active_iterations": active_time, "validation": validation_time, "checkpoints": checkpoint_total},
        }
    return {
        "objective": objective,
        "gpus": gpus,
        "preliminary": True,
        "active_iteration_count": active_count,
        "active_iteration_mean": active,
        "zero_dose_iteration": zero if objective == "hybrid" else None,
        "observed_iterations": by_index,
        "scenarios": scenarios,
        "eligible_under_12_hours": scenarios["2.0"]["hours"] < RUNTIME_LIMIT_HOURS,
        "scenario_note": "Planning scenarios, not confidence intervals. Validation duration is conservatively scaled in full; startup and checkpoint costs stay fixed.",
    }


def recommend_allocation(cells: Sequence[Mapping[str, Any]], objective: str) -> dict[str, Any]:
    objective = _objective(objective)
    eligible = [cell for cell in cells if cell.get("objective") == objective and cell.get("status") == "complete" and (cell.get("estimate") or {}).get("eligible_under_12_hours") is True]
    if not eligible:
        return {"objective": objective, "gpus": None, "reason": "No complete measured allocation has a 2× planning scenario under 12 hours; further hardware calibration is needed."}
    selected = min(eligible, key=lambda cell: cell["gpus"])
    return {"objective": objective, "gpus": selected["gpus"], "reason": "Smallest complete measured allocation whose 2× planning scenario is under 12 hours."}


def atomic_write_text(path: Path | str, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path | str, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _validate_cell_provenance(cell: Mapping[str, Any], objective: str, gpus: int) -> None:
    configuration = cell.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("configuration must be a mapping")
    if configuration.get("model_id") != MODEL_ID or configuration.get("model_revision") != MODEL_REVISION:
        raise ValueError("configuration does not authenticate the pinned Qwen3 model")
    if configuration.get("objective") != objective or configuration.get("gpus") != gpus:
        raise ValueError("configuration objective/GPU identity differs from the cell")
    source = cell.get("source", {})
    if source.get("snapshot_verified") is not True or any(not isinstance(source.get(key), str) or re.fullmatch(r"[0-9a-f]{40}", source[key]) is None for key in ("parent_commit", "fork_commit")):
        raise ValueError("source must authenticate the clean parent/fork commits")
    assets = cell.get("assets", {})
    if assets.get("model") != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise ValueError("sealed assets do not identify the pinned Qwen3 model")
    if assets.get("manifest_content_sha256") != _canonical_sha256({key: value for key, value in assets.items() if key != "manifest_content_sha256"}):
        raise ValueError("asset manifest content hash differs")
    phases = cell.get("phases", [])
    if not phases or any(phase.get("status") != "complete" or phase.get("authenticated") is not True for phase in phases):
        raise ValueError("all benchmark phases must complete with authenticated measurements")
    for phase in phases:
        measured = phase.get("measurement", {})
        if measured.get("wandb_online") is not True or measured.get("wandb_finished") is not True or measured.get("wandb_run_id") != phase.get("wandb_run_id"):
            raise ValueError("phase W&B publication identity is incomplete")
        provenance = measured.get("checkpoint_provenance", {})
        model = provenance.get("model", {})
        if provenance.get("source", {}).get("commit") != source["fork_commit"] or model.get("id") != MODEL_ID or model.get("resolved_revision") != MODEL_REVISION:
            raise ValueError("phase checkpoint source/model identity differs")
        if provenance.get("resolved_hydra_config", {}).get("full_sha256") != _canonical_sha256(measured.get("configuration")):
            raise ValueError("phase resolved configuration hash differs")
    run_ids = [phase["wandb_run_id"] for phase in phases]
    if cell.get("wandb_run_ids") != run_ids or len(set(run_ids)) != len(run_ids):
        raise ValueError("cell W&B run identities do not match its phases")
    if not cell.get("jobs"):
        raise ValueError("benchmark job identity is missing")
    if cell.get("validation_example_count") != TIMING_VALIDATION_EXAMPLES:
        raise ValueError("timing validation must contain exactly 128 examples")


def aggregate_cells(input_root: Path | str) -> dict[str, Any]:
    """Read all expected cells; absent or invalid measurements stay explicit."""
    input_root = Path(input_root)
    found: dict[tuple[str, int], list[tuple[Path, dict[str, Any]]]] = {}
    errors = []
    for path in sorted(input_root.rglob("cell.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda text: (_ for _ in ()).throw(ValueError(f"nonfinite JSON number {text}")))
            key = (_objective(value["objective"]), _gpu_count(value["gpus"]))
            found.setdefault(key, []).append((path, value))
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append({"path": str(path), "error": str(error)})
    cells = []
    for objective in OBJECTIVES:
        for gpus in (1, 2):
            matches = found.get((objective, gpus), [])
            cell: dict[str, Any] = {"objective": objective, "gpus": gpus, "status": "incomplete", "estimate": None}
            if len(matches) != 1:
                cell["reason"] = "missing cell.json" if not matches else "duplicate cell.json files for this objective/GPU count"
                cell["input_paths"] = [str(path) for path, _ in matches]
                cells.append(cell)
                continue
            path, raw = matches[0]
            cell.update(raw)
            cell.update({"objective": objective, "gpus": gpus, "status": "incomplete", "estimate": None, "input_path": str(path), "measurement_status": raw.get("status")})
            cell["dispatch_selection"] = select_dispatch(raw.get("screen_rows", []), raw.get("confirm_rows", []))
            try:
                if raw.get("status") not in ("complete", "completed"):
                    raise ValueError(raw.get("reason") or f"benchmark status is {raw.get('status', 'missing')}")
                _validate_cell_provenance(raw, objective, gpus)
                cell["estimate"] = estimate_runtime(objective, gpus, raw.get("startup_seconds"), raw.get("iterations", []), raw.get("validation_seconds"), raw.get("checkpoint_seconds"))
                cell["status"] = "complete"
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                cell["reason"] = str(error)
            cells.append(cell)
    return {
        "protocol": PROTOCOL,
        "preliminary": True,
        "recipe": {"model": MODEL_ID, "model_revision": MODEL_REVISION, "train_examples": TRAIN_EXAMPLES, "train_batch_size": TRAIN_BATCH_SIZE, "rollout_iterations": ROLLOUT_ITERATIONS, "optimizer_steps": OPTIMIZER_STEPS, "max_response_tokens": MAX_RESPONSE_TOKENS, "hybrid_warmup_iterations": 11, "validation_examples": VALIDATION_EXAMPLES, "timing_validation_examples": TIMING_VALIDATION_EXAMPLES, "validation_events": VALIDATION_EVENTS, "checkpoint_saves": CHECKPOINT_SAVES},
        "budget": {"jobs": 4, "hours_per_job": JOB_LIMIT_HOURS, "maximum_allocated_gpu_hours": MAX_ALLOCATED_GPU_HOURS},
        "cells": cells,
        "input_errors": errors,
        "recommendations": [recommend_allocation(cells, objective) for objective in OBJECTIVES],
        "limitations": ["Planning scenarios are not confidence intervals.", "Only token-dependent training stages and timing-validation duration scale by 1.5×/2×; startup and checkpoint costs remain fixed.", "Timing validation is excluded from BEST selection.", "Full-dose gradient acceptance and exact next-update resume remain production gates.", "Incomplete cells do not supply runtime estimates or allocation recommendations."],
    }


def _markdown_value(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text.replace("|", "\\|").replace("\n", " ")


def _display_number(value: Any) -> str:
    try:
        return f"{_number(value, 'displayed measurement'):.2f}"
    except ValueError:
        return "unavailable"


def _mean_response_length(row: Mapping[str, Any]) -> float | None:
    requests = row.get("requests")
    if not isinstance(requests, list) or not requests:
        return None
    try:
        return _number(row.get("generated_tokens"), "generated_tokens") / len(requests)
    except ValueError:
        return None


def _dispatch_measurement_lines(cell: Mapping[str, Any]) -> list[str]:
    rows = [*cell.get("screen_rows", []), *cell.get("confirm_rows", [])]
    if not rows:
        return ["", "Dispatch measurements, rank generation tails, and peak HBM: unavailable.", ""]
    lines = ["", "| Dispatch / batch | Valid | Wall s | Tokens/s | Mean response tokens (Δ legacy) | Rank generation tails s | Peak HBM GiB |",
             "| --- | --- | ---: | ---: | ---: | --- | --- |"]
    for row in rows:
        baseline = next((item for item in rows if item.get("variant") == "legacy_batch" and item.get("batch_index") == row.get("batch_index")), {})
        average, baseline_average = _mean_response_length(row), _mean_response_length(baseline)
        response_text = _display_number(average)
        if average is not None and baseline_average is not None:
            response_text += f" ({average - baseline_average:+.2f})"
        else:
            response_text += " (Δ unavailable)"
        try:
            throughput = _number(row.get("generated_tokens"), "generated_tokens") / _number(row.get("wall_seconds"), "wall_seconds", positive=True)
        except ValueError:
            throughput = None
        ranks = row.get("rollout_timing", {}).get("ranks", [])
        tails = [f"r{_markdown_value(rank.get('rank', index))}: {_display_number(rank.get('engine_generation_seconds'))}" for index, rank in enumerate(ranks)]
        tail_text = "; ".join(tails) if tails else "unavailable"
        telemetry = row.get("resource_telemetry") or {}
        peaks = telemetry.get("peak_hbm_gib_per_gpu") or {}
        memory_text = "; ".join(f"GPU {_markdown_value(gpu)}: {_display_number(value)}" for gpu, value in sorted(peaks.items())) if peaks else "unavailable"
        if telemetry.get("peak_hbm_gib_aggregate") is not None:
            memory_text += f"; measured aggregate: {_display_number(telemetry['peak_hbm_gib_aggregate'])}"
        lines.append(f"| {_markdown_value(row.get('variant', 'unknown'))} / {row.get('batch_index', '?')} | {'yes' if row.get('valid') is True else 'no'} | {_display_number(row.get('wall_seconds'))} | {_display_number(throughput)} | {response_text} | {tail_text} | {memory_text} |")
    lines.extend(["", "Rank generation tails are each rank's elapsed engine-generation time; they are not added together. HBM values are sampled peaks, not allocation limits.", ""])
    return lines


def _stage_measurement_lines(cell: Mapping[str, Any]) -> list[str]:
    lines = ["", "| Measured phase | Wall s | Generation s | Replay s | Reference s | Actor incl. teacher s | Weight sync s (nested) | Teacher max-rank s (nested) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    by_index = {row.get("rollout_iteration"): row for row in cell.get("iterations", [])}
    def iteration_summary(indices):
        if any(index not in by_index for index in indices):
            return ["unavailable"] * 7
        records = [by_index[index] for index in indices]
        try:
            wall = mean(_iteration_cost(row)["wall_seconds"] for row in records)
        except ValueError:
            return ["unavailable"] * 7
        values = [_display_number(wall)]
        for stage in ("gen", "old_log_prob", "ref", "update_actor", "weight_sync"):
            try:
                values.append(_display_number(mean(_number(row["timing_s"].get(stage), stage) for row in records)))
            except ValueError:
                values.append("unavailable")
        try:
            teacher = mean(_number(row.get("actor_update_timing", {}).get("teacher_seconds_max"), "teacher_seconds_max") for row in records)
            values.append(_display_number(teacher))
        except ValueError:
            values.append("unavailable")
        return values
    lines.append(f"| Startup | {_display_number(cell.get('startup_seconds'))} | — | — | — | — | — | — |")
    zero_label = "Iteration 0 (zero dose)" if cell.get("objective") == "hybrid" else "Iteration 0"
    lines.append(f"| {zero_label} | {' | '.join(iteration_summary([0]))} |")
    lines.append(f"| Active mean (iterations 1–2) | {' | '.join(iteration_summary([1, 2]))} |")
    lines.append(f"| Timing validation (128 examples) | {_display_number(cell.get('validation_seconds'))} | — | — | — | — | — | — |")
    lines.append(f"| One authenticated checkpoint | {_display_number(cell.get('checkpoint_seconds'))} | — | — | — | — | — | — |")
    lines.extend(["", "Iteration wall times exclude nested validation and checkpoint saves. Actor time includes teacher work; weight synchronization is already inside generation. Teacher diagnostics use the maximum per-rank wall time, averaged across iterations 1–2 for the active mean. Qwen benchmark CUDA synchronization cost is included in these measurements. Nested teacher time is never added to actor time or the runtime estimate. Missing stage timers are unavailable, not zero.", ""])
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Preliminary Qwen3-0.6B MATH training runtime",
        "",
        f"Pinned model revision: `{MODEL_REVISION}`. One epoch: 6,985 examples, batch 64, 109 rollout iterations, 218 optimizer steps, 8,192 response tokens maximum.",
        "",
        "Four independent jobs are capped at two hours each (12 allocated GPU-hours maximum). Full training is not submitted. Estimates include six 512-example validations extrapolated from a timing-only 128-example subset, and five checkpoint saves.",
        "",
        "| Objective | H100s | Status | Dispatch | Central hours | 1.5× hours | 2× hours | Central GPU-hours |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for cell in report["cells"]:
        estimate = cell.get("estimate")
        times = [f"{estimate['scenarios'][key]['hours']:.2f}" for key in ("1.0", "1.5", "2.0")] if estimate else ["—"] * 3
        gpu_hours = f"{estimate['scenarios']['1.0']['gpu_hours']:.2f}" if estimate else "—"
        dispatch = cell.get("dispatch_selection", {}).get("selected_variant", "unmeasured")
        if not any(_row(cell.get("screen_rows", []), variant, 0) is not None for variant in VARIANTS):
            dispatch = "unmeasured"
        lines.append(f"| {cell['objective']} | {cell['gpus']} | {cell['status']} | {dispatch} | {' | '.join(times)} | {gpu_hours} |")
    lines.extend(["", "## Measurements and provenance", ""])
    for cell in report["cells"]:
        prefix = f"{cell['objective']} / {cell['gpus']} H100"
        if cell["status"] != "complete":
            lines.append(f"- **{prefix}:** incomplete — {_markdown_value(cell.get('reason', 'unfinished'))}.")
        selection = cell.get("dispatch_selection")
        if selection:
            comparisons = "; ".join(f"batch {row['batch_index']}: wall {row['wall_speedup']:.3f}×, throughput {row['throughput_speedup']:.3f}×, tokens Δ{row['generated_tokens_difference']:g}" if row["valid"] else f"batch {row['batch_index']}: incomplete/invalid" for row in selection["comparisons"])
            lines.append(f"- **{prefix} dispatch:** {_markdown_value(selection['candidate_variant'])}; {comparisons}. {_markdown_value(selection['reason'])}.")
        provenance = {key: cell[key] for key in ("jobs", "wandb_run_ids") if key in cell}
        source = cell.get("source", {})
        if isinstance(source, Mapping):
            provenance["source"] = {key: value for key, value in source.items() if isinstance(value, (str, int, float))}
        if provenance:
            lines.append(f"- **{prefix} provenance:** `{_markdown_value(provenance)}`.")
        lines.extend(_dispatch_measurement_lines(cell))
        lines.extend(_stage_measurement_lines(cell))
    lines.extend(["", "Full configurations, source hashes/inventories, raw comparison records, memory observations, rank timings, and iteration timing breakdowns are retained in `training_runtime.json` when supplied by the jobs.", "", "## Allocation recommendation", ""])
    for row in report["recommendations"]:
        choice = f"{row['gpus']} H100{'s' if row['gpus'] == 2 else ''}" if row["gpus"] else "undetermined"
        lines.append(f"- **{row['objective']}: {choice}.** {row['reason']}")
    lines.extend(["", "The 1.5× and 2× values are planning scenarios, not confidence intervals. They scale generation, replay/reference scoring, actor update (including its nested teacher work), and validation; measured weight synchronization stays fixed. Iteration totals exclude nested validation/checkpoint time and never add nested teacher timing twice.", "", "Full-dose gradient acceptance and exact next-update resume remain subsequent production gates. Timing-subset validation does not enter BEST selection.", ""])
    if report.get("input_errors"):
        lines.extend(["Input errors: " + _markdown_value(report["input_errors"]), ""])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    aggregate = subparsers.add_parser("aggregate", help="Publish complete and explicitly incomplete benchmark cells")
    aggregate.add_argument("--input-root", required=True, type=Path)
    aggregate.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    report = aggregate_cells(args.input_root)
    atomic_write_json(args.output_dir / "training_runtime.json", report)
    atomic_write_text(args.output_dir / "training_runtime.md", render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
