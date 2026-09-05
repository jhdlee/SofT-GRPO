"""End-to-end CLI for the native-soft ICL validation study.

Commands stage immutable assets, generate with upstream SofT-GRPO, replay
no-demo continuous prefixes, and aggregate the preregistered metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .icl import (
    CORE_CONDITIONS,
    ICLMatrixCell,
    LEGACY_STUDY_ID,
    QWEN3_STUDY_ID,
    STUDY_BENCHMARKS,
    SUPPORTED_CORE_CONDITIONS,
    build_icl_matrix,
    get_study_profile,
    load_icl_dataset,
    materialize_prompts,
    normalized_answer_copy,
    paired_bootstrap_difference,
    paired_bootstrap_difference_in_differences,
    rationale_token_overlap_f1,
    render_icl_prompt,
    rescue_harm_rates,
    request_seed,
    smoke_subset_ids,
    validate_matrix_cell,
)
from .icl_assets import prepare_icl_assets, verify_icl_assets
from .icl_replay import (
    ACTOR_ACTIVE_PROBABILITY_THRESHOLD,
    ActorAgreementTolerances,
    AtomicReplayStore,
    ReplayRecord,
    actor_agreement_gate,
    replay_chunk_metrics,
    replay_trajectory_many,
    stable_replay_run_id,
)
from .icl_runtime import (
    AtomicChunkStore,
    CompletionRecord,
    ReleasedSofTGRPOEngine,
    SamplingSettings,
    UPSTREAM_TEXT_PROMPT_TOKENIZATION,
    boundary_gate,
    canonical_json_bytes,
    generation_chunk_metrics,
    init_online_wandb,
    parse_sglang_completion,
    required_context_length,
    sha256_file,
    source_provenance,
    stable_wandb_run_id,
    unpack_trajectory,
    validate_atomic_reasoning_tokens,
)
from .icl_resource_monitor import ResourceMonitor
from .icl_throughput import (
    BenchmarkObservation,
    RequestTiming,
    ThroughputExample,
    build_throughput_plan,
    choose_smallest_data_parallel_size,
)
from .graders import (
    lm_eval_flexible_last_number_grade,
    math_verify_full_response_grade,
    released_last_boxed_grade,
)


def _atomic_json(path: Path, value: Any) -> None:
    from .icl_runtime import _atomic_bytes

    _atomic_bytes(path, canonical_json_bytes(value))


def _verify_assets_for_study(root: str | Path, study: str) -> Mapping[str, Any]:
    # Keep the old one-argument call observable for callers and tests that use
    # the legacy profile, while routing new studies through the explicit API.
    if study == LEGACY_STUDY_ID:
        return verify_icl_assets(root)
    return verify_icl_assets(root, study=study)


def _tokenizer_inventory_sha256(model_inventory: Mapping[str, Any]) -> str:
    """Hash the authenticated tokenizer subset already present in an asset tree."""

    prefixes = (
        "tokenizer",
        "special_tokens_map",
        "added_tokens",
        "vocab",
        "merges",
    )
    files = [
        dict(entry)
        for entry in model_inventory.get("files", ())
        if Path(str(entry.get("path", ""))).name.startswith(prefixes)
    ]
    if not files:
        raise ValueError("authenticated model inventory has no tokenizer files")
    return hashlib.sha256(canonical_json_bytes(files)).hexdigest()


def _finish_generation_resources(
    *, wandb_run: Any, engine: Any | None, succeeded: bool
) -> dict[str, float]:
    """Flush W&B before SGLang terminates all child processes."""

    if not succeeded:
        wandb_run.summary["generation/completed"] = False
    wandb_started = time.perf_counter()
    wandb_seconds = 0.0
    shutdown_seconds = 0.0
    try:
        wandb_run.finish()
        wandb_seconds = time.perf_counter() - wandb_started
    except Exception:
        wandb_seconds = time.perf_counter() - wandb_started
        # Preserve the generation/gate exception instead of replacing it with
        # a secondary W&B mailbox error during failure cleanup.
        if succeeded:
            raise
    finally:
        if engine is not None:
            shutdown_started = time.perf_counter()
            engine.shutdown()
            shutdown_seconds = time.perf_counter() - shutdown_started
    return {
        "wandb_finish_seconds": wandb_seconds,
        "engine_shutdown_seconds": shutdown_seconds,
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Return a deterministic linearly interpolated percentile."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _async_queue_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    queue_size: int,
    data_parallel_size: int,
) -> dict[str, Any]:
    """Summarize bounded frontend occupancy without inferring GPU routing."""

    if not rows:
        raise ValueError("queue metrics require request timing rows")
    by_session: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_session[str(row.get("timing_session_id", "single-session"))].append(row)
    duration = 0.0
    maximum = 0
    for session_rows in by_session.values():
        started = min(float(row["submitted_at"]) for row in session_rows)
        stopped = max(float(row["completed_at"]) for row in session_rows)
        session_duration = max(stopped - started, 1e-12)
        duration += session_duration
        events = []
        for row in session_rows:
            events.append((float(row["submitted_at"]), 1))
            events.append((float(row["completed_at"]), -1))
        # Process completions before submissions at the same instant. A request
        # replacing a completed request therefore never creates a spurious peak.
        occupancy = 0
        for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
            occupancy += delta
            if occupancy < 0:
                raise RuntimeError("asynchronous queue occupancy became negative")
            maximum = max(maximum, occupancy)
        if occupancy:
            raise RuntimeError("asynchronous queue did not drain to zero")
    summed_latency = sum(float(row["latency_seconds"]) for row in rows)
    return {
        "maximum": maximum,
        "capacity": queue_size,
        "mean": summed_latency / duration,
        "mean_fraction": summed_latency / duration / queue_size,
        "timing_session_count": len(by_session),
        "active_generation_seconds": duration,
    }


def _publish_resource_metrics(
    wandb_run: Any, resource_metrics: Mapping[str, Any]
) -> None:
    """Publish aggregate and assigned-device telemetry without fake routing labels."""

    for name, value in resource_metrics.items():
        if isinstance(value, (int, float, list)) and not isinstance(value, bool):
            wandb_run.summary["system/%s" % name] = value
    for name in ("host_metrics_available", "gpu_metrics_available"):
        if name in resource_metrics:
            wandb_run.summary["system/%s" % name] = bool(resource_metrics[name])
    if resource_metrics.get("gpu_selection_source"):
        wandb_run.summary["system/gpu_selection_source"] = resource_metrics[
            "gpu_selection_source"
        ]
    per_gpu_hbm = resource_metrics.get("peak_hbm_gib_per_gpu") or {}
    per_gpu_utilization = resource_metrics.get("gpu_utilization_mean_per_gpu") or {}
    for gpu_id, value in per_gpu_hbm.items():
        wandb_run.summary["system/gpu/%s/peak_hbm_gib" % gpu_id] = float(value)
    for gpu_id, value in per_gpu_utilization.items():
        wandb_run.summary["system/gpu/%s/utilization_mean" % gpu_id] = float(value)


LEGACY_RENDER_PROTOCOL = "checkpoint-native-assistant-fixed-think-newline-v1"
QWEN3_RENDER_PROTOCOL = "qwen3-enable-thinking-append-fixed-think-newline-v2"
QWEN3_MODEL_DISPLAY_NAMES = {
    "qwen3_0p6b": "Qwen3-0.6B",
    "qwen3_1p7b": "Qwen3-1.7B",
}
INFERENCE_MODE_DISPLAY_NAMES = {
    "native_soft": "soft-thinking",
    "hard_token": "discrete token CoT",
}
THROUGHPUT_WARMUP_PROTOCOL = "qwen3-dedicated-symmetric-warmup-v1"
THROUGHPUT_WARMUP_USER_CONTENT = (
    "Warm-up request only. Produce one short response."
)
THROUGHPUT_WARMUP_SEED = request_seed(
    "math500", "qwen3-throughput-dedicated-warmup", 0, study=QWEN3_STUDY_ID
)


def _render_protocol(study: str | None = None) -> str:
    return (
        QWEN3_RENDER_PROTOCOL
        if get_study_profile(study).study_id == QWEN3_STUDY_ID
        else LEGACY_RENDER_PROTOCOL
    )


def _render(
    tokenizer: Any, user_content: str, *, study: str | None = None
) -> str:
    profile = get_study_profile(study)
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if profile.study_id == QWEN3_STUDY_ID:
        kwargs["enable_thinking"] = True
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}], **kwargs
    )
    opener = profile.fixed_think_opener
    if profile.study_id == QWEN3_STUDY_ID:
        assistant_header = "<|im_start|>assistant\n"
        if not isinstance(rendered, str) or not rendered.endswith(assistant_header):
            raise RuntimeError(
                "Qwen3 template must end in its native assistant header before "
                "the registered fixed thinking opener is appended"
            )
        rendered += opener
    elif not isinstance(rendered, str) or not rendered.endswith(opener):
        raise RuntimeError(
            "checkpoint chat template must end in its registered native assistant %r opening"
            % opener
        )
    # This no-specials pass only validates that the rendered template is
    # tokenizable. Generation itself intentionally follows the released
    # SGLang text API, whose TokenizerManager encodes the complete string
    # again using the checkpoint's tokenizer defaults.
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    if not ids:
        raise RuntimeError("rendered chat prompt tokenized to empty")
    return rendered


def _throughput_warmup_contract(data_parallel_size: int) -> dict[str, Any]:
    if (
        isinstance(data_parallel_size, bool)
        or not isinstance(data_parallel_size, int)
        or data_parallel_size <= 0
    ):
        raise ValueError("throughput warmup DP size must be a positive integer")
    return {
        "protocol": THROUGHPUT_WARMUP_PROTOCOL,
        "request_count": data_parallel_size,
        "max_new_tokens": 32,
        "excluded_from_timing": True,
        "request_seed": THROUGHPUT_WARMUP_SEED,
        "user_content_sha256": hashlib.sha256(
            THROUGHPUT_WARMUP_USER_CONTENT.encode("utf-8")
        ).hexdigest(),
    }


def _run_symmetric_throughput_warmup(
    engine: Any,
    tokenizer: Any,
    *,
    study: str,
    data_parallel_size: int,
) -> None:
    if study != QWEN3_STUDY_ID:
        raise ValueError("the dedicated throughput warmup is registered only for Qwen3")
    prompt = _render(tokenizer, THROUGHPUT_WARMUP_USER_CONTENT, study=study)
    for _ in range(data_parallel_size):
        engine.warmup(
            prompt,
            THROUGHPUT_WARMUP_SEED,
            max_new_tokens=32,
        )


def _selected_cells(args: argparse.Namespace) -> list[Any]:
    study = getattr(args, "study", LEGACY_STUDY_ID)
    throughput_benchmark = bool(getattr(args, "throughput_benchmark", False))
    cells = [
        cell
        for cell in build_icl_matrix(
            smoke=bool(args.smoke) and not throughput_benchmark,
            study=study,
        )
        if cell.model_label == args.model and cell.inference_mode == args.mode
    ]
    if throughput_benchmark:
        counts = {"math500": 20, "aime2024": 2}
        cells = [
            ICLMatrixCell(
                model_label=cell.model_label,
                inference_mode=cell.inference_mode,
                condition=cell.condition,
                benchmark=cell.benchmark,
                subset=cell.subset,
                example_count=counts[cell.benchmark],
                sample_count=1,
            )
            for cell in cells
        ]
    if getattr(args, "benchmarks", None) and args.benchmarks != ["all"]:
        cells = [cell for cell in cells if cell.benchmark in args.benchmarks]
    if getattr(args, "conditions", None) and args.conditions != ["all"]:
        cells = [cell for cell in cells if cell.condition in args.conditions]
    if not cells:
        raise ValueError("the requested filters select no registered matrix cells")
    return cells


def _cell_prompts(
    cell: Any,
    *,
    examples: Sequence[Any],
    shuffled_pairs: Mapping[str, Mapping[str, str]],
    mechanism_ids: Mapping[str, Sequence[str]],
    selected_ids_override: Sequence[str] | None = None,
) -> tuple[list[Any], list[Any]]:
    benchmark_examples = [
        example for example in examples if example.benchmark == cell.benchmark
    ]
    selected_ids: Sequence[str] | None = selected_ids_override
    if selected_ids is None:
        selected_ids = (
            None
            if cell.condition in SUPPORTED_CORE_CONDITIONS
            else mechanism_ids[cell.benchmark]
        )
    eligible = benchmark_examples
    if selected_ids is not None:
        selected = set(selected_ids)
        eligible = [
            example
            for example in benchmark_examples
            if example.example_id in selected
        ]
    # Only the smoke matrix is smaller than its eligible full/mechanism pool.
    # A production mechanism cell already has exactly 128 eligible examples
    # and must not be reduced through the 16-example smoke selector.
    if selected_ids_override is None and cell.example_count < len(eligible):
        selected_ids = smoke_subset_ids(eligible)
    prompts = list(
        materialize_prompts(
            benchmark_examples,
            shuffled_pairs[cell.benchmark],
            cell.condition,
            selected_ids=selected_ids,
        )
    )
    prompts = prompts[: cell.example_count]
    by_id = {example.example_id: example for example in benchmark_examples}
    selected_examples = [by_id[prompt.example_id] for prompt in prompts]
    if len(prompts) != cell.example_count:
        raise RuntimeError("materialized prompt count differs from matrix cell")
    return selected_examples, prompts


def _build_rendered_throughput_plan(
    tokenizer: Any,
    *,
    examples: Sequence[Any],
    shuffled_pairs: Mapping[str, Mapping[str, str]],
    study: str,
) -> Any:
    """Build the sealed 66-request calibration plan from rendered prompts."""

    profile = get_study_profile(study)
    if profile.study_id != QWEN3_STUDY_ID:
        raise ValueError("throughput calibration is registered only for Qwen3")
    prompt_lengths: dict[tuple[str, str, str], int] = {}
    for benchmark in profile.benchmarks:
        benchmark_examples = [
            example for example in examples if example.benchmark == benchmark
        ]
        for condition in profile.core_conditions:
            prompts = materialize_prompts(
                benchmark_examples,
                shuffled_pairs[benchmark],
                condition,
            )
            for prompt in prompts:
                rendered = _render(tokenizer, prompt.user_content, study=study)
                prompt_lengths[(benchmark, prompt.example_id, condition)] = len(
                    tokenizer.encode(rendered)
                )
    calibration_examples = []
    for example in examples:
        if example.benchmark not in profile.benchmarks:
            continue
        calibration_examples.append(
            ThroughputExample(
                example_id=example.example_id,
                benchmark=example.benchmark,
                subject=example.subject or "unspecified",
                level=example.difficulty or example.benchmark,
                prompt_tokens={
                    condition: prompt_lengths[
                        (example.benchmark, example.example_id, condition)
                    ]
                    for condition in profile.core_conditions
                },
            )
        )
    return build_throughput_plan(calibration_examples)


def _throughput_selected_ids(plan: Any) -> dict[tuple[str, str], tuple[str, ...]]:
    result: dict[tuple[str, str], list[str]] = defaultdict(list)
    for request in plan.requests:
        result[(request.benchmark, request.condition)].append(request.example_id)
    return {key: tuple(values) for key, values in result.items()}


def _validate_generation_binding(
    manifest: Mapping[str, Any],
    *,
    assets: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    model_label: str,
    study: str | None = None,
) -> None:
    """Bind a generation manifest to the currently authenticated inputs."""

    expected = {
        "asset_manifest_sha256": assets["content_sha256"],
        "data_manifest_sha256": data_manifest["content_sha256"],
        "model_label": model_label,
        "model_tree_sha256": assets["models"][model_label]["tree_sha256"],
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(
                "generation manifest %s differs from authenticated inputs for %s"
                % (name, model_label)
            )
    profile = get_study_profile(study)
    manifest_study = manifest.get("study_id", LEGACY_STUDY_ID)
    if manifest_study != profile.study_id:
        raise ValueError("generation manifest differs from the registered study")
    if manifest.get("sampling") != SamplingSettings.for_study(profile.study_id).__dict__:
        raise ValueError("generation sampling settings differ from the registered protocol")
    if manifest.get("prompt_tokenization") != UPSTREAM_TEXT_PROMPT_TOKENIZATION:
        raise ValueError("generation prompt tokenization differs from released behavior")
    # Historical manifests predate an explicit rendering field. They remain
    # readable under the legacy profile; every new Qwen3 artifact must seal it.
    render_protocol = manifest.get("prompt_render_protocol")
    if render_protocol is not None and render_protocol != _render_protocol(
        profile.study_id
    ):
        raise ValueError("generation prompt rendering protocol changed")
    if profile.study_id != LEGACY_STUDY_ID and render_protocol is None:
        raise ValueError("Qwen3 generation manifest lacks its prompt rendering protocol")
    if profile.study_id == QWEN3_STUDY_ID:
        expected_qwen = {
            "study_protocol": profile.protocol,
            "model_source": assets["model_specs"][model_label],
            "fixed_think_opener": profile.fixed_think_opener,
            "reasoning_token_ids": {
                "think": profile.think_token_id,
                "close_think": profile.close_think_token_id,
            },
            "tokenizer_inventory_sha256": _tokenizer_inventory_sha256(
                assets["models"][model_label]
            ),
        }
        for name, value in expected_qwen.items():
            if manifest.get(name) != value:
                raise ValueError(
                    "generation manifest %s differs from the sealed Qwen3 protocol"
                    % name
                )


def _generation_chunk_identity(
    manifest: Mapping[str, Any],
    *,
    model_label: str,
    inference_mode: str,
    benchmark: str,
    condition: str,
    sample_index: int,
    chunk_index: int,
    example_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "generation_manifest_sha256": hashlib.sha256(
            canonical_json_bytes(dict(manifest))
        ).hexdigest(),
        "model_label": model_label,
        "mode": inference_mode,
        "benchmark": benchmark,
        "condition": condition,
        "sample_index": sample_index,
        "chunk_index": chunk_index,
        "example_ids": list(example_ids),
    }


def _validate_completion_records(
    records: Sequence[CompletionRecord],
    *,
    model_label: str,
    inference_mode: str,
    benchmark: str,
    condition: str,
    sample_index: int,
    example_ids: Sequence[str],
) -> None:
    """Reject authenticated chunks whose rows are attached to the wrong cell."""

    if len(records) != len(example_ids):
        raise RuntimeError("generation chunk row count differs from its expected examples")
    for replay_row, (record, example_id) in enumerate(
        zip(records, example_ids, strict=True)
    ):
        expected = {
            "model_label": model_label,
            "inference_mode": inference_mode,
            "benchmark": benchmark,
            "condition": condition,
            "sample_index": sample_index,
            "example_id": example_id,
            "replay_row": replay_row,
        }
        for name, value in expected.items():
            if getattr(record, name) != value:
                raise RuntimeError(
                    "generation record %s differs from its chunk identity" % name
                )


def _validate_replay_records(
    records: Sequence[ReplayRecord],
    *,
    source_records: Sequence[CompletionRecord],
    model_label: str,
    benchmark: str,
    sample_index: int,
) -> None:
    """Validate the exact source-row by prompted-context replay expansion."""

    expected_count = len(source_records) * len(CORE_CONDITIONS)
    if len(records) != expected_count:
        raise RuntimeError("replay chunk does not cover every core context")
    for index, record in enumerate(records):
        source = source_records[index // len(CORE_CONDITIONS)]
        condition = CORE_CONDITIONS[index % len(CORE_CONDITIONS)]
        expected = {
            "model_label": model_label,
            "benchmark": benchmark,
            "example_id": source.example_id,
            "sample_index": sample_index,
            "prompted_condition": condition,
            "latent_token_count": source.latent_token_count,
        }
        for name, value in expected.items():
            if getattr(record, name) != value:
                raise RuntimeError("replay record %s differs from its source" % name)
        expected_exclusion = (
            "zero_latent_slots" if source.latent_token_count == 0 else None
        )
        if record.replay_exclusion_reason != expected_exclusion:
            raise RuntimeError("replay exclusion does not match the source trajectory")


def run_prepare(args: argparse.Namespace) -> None:
    study = getattr(args, "study", LEGACY_STUDY_ID)
    if study == LEGACY_STUDY_ID:
        manifest = prepare_icl_assets(Path(args.root), Path(args.cache_dir))
    else:
        manifest = prepare_icl_assets(
            Path(args.root), Path(args.cache_dir), study=study
        )
    print(
        json.dumps(
            {
                "root": str(Path(args.root).resolve()),
                "content_sha256": manifest["content_sha256"],
                "paths": manifest["paths"],
            },
            indent=2,
        )
    )


def run_generate(args: argparse.Namespace) -> None:
    study = getattr(args, "study", LEGACY_STUDY_ID)
    profile = get_study_profile(study)
    throughput_benchmark = bool(getattr(args, "throughput_benchmark", False))
    assets = _verify_assets_for_study(args.root, study)
    asset_root = Path(args.root).expanduser().resolve()
    data_root = asset_root / "data"
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else asset_root / "generation"
    )
    examples, shuffled_pairs, mechanism_ids, data_manifest = load_icl_dataset(data_root)
    cells = _selected_cells(args)
    model_path = asset_root / "models" / args.model

    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("generation requires transformers") from error
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    think_start_id, think_end_id = validate_atomic_reasoning_tokens(
        tokenizer, study=study
    )
    throughput_plan = (
        _build_rendered_throughput_plan(
            tokenizer,
            examples=examples,
            shuffled_pairs=shuffled_pairs,
            study=study,
        )
        if throughput_benchmark
        else None
    )
    throughput_ids = (
        _throughput_selected_ids(throughput_plan)
        if throughput_plan is not None
        else {}
    )
    cell_payloads = []
    all_rendered = []
    for cell in cells:
        selected_examples, prompts = _cell_prompts(
            cell,
            examples=examples,
            shuffled_pairs=shuffled_pairs,
            mechanism_ids=mechanism_ids,
            selected_ids_override=throughput_ids.get(
                (cell.benchmark, cell.condition)
            ),
        )
        rendered = [
            _render(tokenizer, prompt.user_content, study=study)
            for prompt in prompts
        ]
        all_rendered.extend(rendered)
        cell_payloads.append((cell, selected_examples, prompts, rendered))

    settings = SamplingSettings.for_study(study)
    context_length = required_context_length(tokenizer, all_rendered, settings)
    model_config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    key_value_heads = int(
        getattr(model_config, "num_key_value_heads", 0)
        or getattr(model_config, "num_attention_heads", 0)
        or 0
    )
    if key_value_heads and key_value_heads % args.tensor_parallel_size:
        raise RuntimeError(
            "tensor parallel size %d does not divide %d key/value heads"
            % (args.tensor_parallel_size, key_value_heads)
        )
    maximum = int(getattr(model_config, "max_position_embeddings", 0) or 0)
    if study == QWEN3_STUDY_ID and maximum != profile.max_positions:
        raise RuntimeError(
            "model max_position_embeddings differs from the sealed %s contract: "
            "got %d, expected %d"
            % (study, maximum, profile.max_positions)
        )
    if maximum and context_length > maximum:
        raise RuntimeError(
            "maximum tokenized prompt + %d completion tokens + one guard "
            "position requires %d positions, model supports %d"
            % (settings.max_new_tokens, context_length, maximum)
        )
    queue_size = (
        args.queue_size
        if args.queue_size is not None
        else 2 * args.data_parallel_size * args.max_running_requests
    )
    config = {
        "protocol": (
            "opd-qwen3-icl-pass1-generation-v1"
            if study == QWEN3_STUDY_ID
            else "opd-softgrpo-native-soft-icl-generation-v3-math-aime-matched"
        ),
        "study_id": study,
        "study_protocol": profile.protocol,
        "source_provenance": source_provenance(),
        "asset_manifest_sha256": assets["content_sha256"],
        "data_manifest_sha256": data_manifest["content_sha256"],
        "model_label": args.model,
        "model_source": assets["model_specs"][args.model],
        "model_tree_sha256": assets["models"][args.model]["tree_sha256"],
        "tokenizer_inventory_sha256": _tokenizer_inventory_sha256(
            assets["models"][args.model]
        ),
        "reasoning_token_ids": {
            "think": think_start_id,
            "close_think": think_end_id,
        },
        "fixed_think_opener": profile.fixed_think_opener,
        "mode": args.mode,
        "smoke": bool(args.smoke),
        "throughput_benchmark": throughput_benchmark,
        "parallelism": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "data_parallel_size": args.data_parallel_size,
            "world_size": args.tensor_parallel_size * args.data_parallel_size,
            "load_balance_method": "round_robin",
        },
        "chunk_size": args.chunk_size,
        "request_queue_size": queue_size,
        "max_running_requests": args.max_running_requests,
        "max_running_requests_per_replica": args.max_running_requests,
        "max_running_requests_aggregate": (
            args.max_running_requests * args.data_parallel_size
        ),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "context_length": context_length,
        "sampling": settings.__dict__,
        "prompt_tokenization": UPSTREAM_TEXT_PROMPT_TOKENIZATION,
        "prompt_render_protocol": _render_protocol(study),
        "cells": [cell.to_dict() for cell in cells],
    }
    if throughput_plan is not None:
        config["throughput_plan"] = throughput_plan.to_dict()
        config["throughput_plan_sha256"] = throughput_plan.content_sha256
        config["warmup"] = _throughput_warmup_contract(
            args.data_parallel_size
        )
    run_id = stable_wandb_run_id(config, prefix="icl-generate")
    config["wandb_run_id"] = run_id
    manifest_path = output_root / args.model / args.mode / "generation_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != config:
            raise RuntimeError("generation invocation differs from its resume manifest")
    else:
        _atomic_json(manifest_path, config)
    completion_path = output_root / args.model / args.mode / "completion.json"
    existing_completion_payload = None
    if completion_path.is_file():
        existing_completion_payload = json.loads(
            completion_path.read_text(encoding="utf-8")
        )
        if existing_completion_payload.get(
            "generation_manifest_sha256"
        ) != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
            raise RuntimeError("existing completion marker does not bind to the manifest")
        throughput_path = output_root / args.model / args.mode / "throughput.json"
        if "throughput_sha256" in existing_completion_payload:
            if not throughput_path.is_file() or existing_completion_payload[
                "throughput_sha256"
            ] != hashlib.sha256(throughput_path.read_bytes()).hexdigest():
                raise RuntimeError("existing throughput artifact failed authentication")

    job_type = (
        "icl-throughput-benchmark"
        if throughput_benchmark
        else ("icl-generation-smoke" if args.smoke else "icl-generation")
    )
    wandb_run = init_online_wandb(run_id=run_id, config=config, job_type=job_type)
    engine = None
    resource_monitor = ResourceMonitor(interval_seconds=1.0)
    store = AtomicChunkStore(output_root)
    step = 0
    resumed = False
    cell_records: dict[tuple[str, str], list[CompletionRecord]] = defaultdict(list)
    request_timings: list[dict[str, Any]] = []
    engine_load_seconds = 0.0
    timing_payload: dict[str, Any] | None = None
    completion_payload: dict[str, Any] | None = None
    succeeded = False
    resource_metrics: dict[str, Any] | None = None
    timing_session_id = "%d-%d" % (os.getpid(), time.time_ns())
    try:
        resource_monitor.start()
        engine_load_started = time.perf_counter()
        engine = ReleasedSofTGRPOEngine(
            model_path=str(model_path),
            mode=args.mode,
            tensor_parallel_size=args.tensor_parallel_size,
            data_parallel_size=args.data_parallel_size,
            context_length=context_length,
            settings=settings,
            max_running_requests=args.max_running_requests,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        engine_load_seconds = time.perf_counter() - engine_load_started
        warmup_seconds = 0.0
        if throughput_benchmark:
            warmup_started = time.perf_counter()
            _run_symmetric_throughput_warmup(
                engine,
                tokenizer,
                study=study,
                data_parallel_size=args.data_parallel_size,
            )
            warmup_seconds = time.perf_counter() - warmup_started
        queued_prompts = []
        queued_seeds = []
        queued_refs = []
        pending_chunks = {}
        for cell, _, prompts, rendered in cell_payloads:
            # Arrange unfinished work in cell/chunk/sample-major order. This
            # keeps shared prefixes adjacent while allowing one continuous
            # queue to span every condition and benchmark in this invocation.
            for chunk_index, start in enumerate(
                range(0, len(prompts), args.chunk_size)
            ):
                stop = min(start + args.chunk_size, len(prompts))
                chunk_prompts = prompts[start:stop]
                for sample_index in range(cell.sample_count):
                    key = "%s/%s/%s/%s/sample_%02d/chunk_%05d" % (
                        args.model,
                        args.mode,
                        cell.benchmark,
                        cell.condition,
                        sample_index,
                        chunk_index,
                    )
                    identity = _generation_chunk_identity(
                        config,
                        model_label=args.model,
                        inference_mode=args.mode,
                        benchmark=cell.benchmark,
                        condition=cell.condition,
                        sample_index=sample_index,
                        chunk_index=chunk_index,
                        example_ids=[prompt.example_id for prompt in chunk_prompts],
                    )
                    committed = store.resume_state(key, expected_identity=identity)
                    if committed is not None:
                        old_records, _ = store.load(key)
                        cell_records[(cell.benchmark, cell.condition)].extend(old_records)
                        commit_metadata = committed.get("commit_metadata")
                        if commit_metadata is not None:
                            prior_timings = commit_metadata.get("request_timings")
                            if (
                                commit_metadata.get("protocol")
                                != "opd-icl-chunk-timing-v1"
                                or not isinstance(prior_timings, list)
                                or len(prior_timings) != len(old_records)
                                or [row.get("example_id") for row in prior_timings]
                                != [record.example_id for record in old_records]
                            ):
                                raise RuntimeError(
                                    "committed asynchronous timing metadata is malformed"
                                )
                            request_timings.extend(dict(row) for row in prior_timings)
                        elif throughput_benchmark:
                            raise RuntimeError(
                                "throughput benchmark chunk lacks resumable timing metadata"
                            )
                        resumed = True
                        step += 1
                        continue
                    pending_chunks[key] = {
                        "key": key,
                        "identity": identity,
                        "cell": cell,
                        "sample_index": sample_index,
                        "records": [None] * len(chunk_prompts),
                        "trajectories": [None] * len(chunk_prompts),
                        "submitted_at": [None] * len(chunk_prompts),
                        "completed_at": [None] * len(chunk_prompts),
                        "request_indices": [None] * len(chunk_prompts),
                        "chunk_index": chunk_index,
                        "remaining": len(chunk_prompts),
                    }
                    for row, prompt in enumerate(chunk_prompts):
                        queued_prompts.append(rendered[start + row])
                        queued_seeds.append(
                            request_seed(
                                cell.benchmark,
                                prompt.example_id,
                                sample_index,
                                study=study,
                            )
                        )
                        queued_refs.append((key, row, prompt))

        # Each completion is replaced in the SGLang queue before it is yielded
        # for parsing and persistence, so CPU bookkeeping does not create a
        # batch barrier. The queue drains only after the full invocation ends.
        if queued_prompts:
            completed = engine.generate_as_completed(
                queued_prompts, queued_seeds, queue_size=queue_size
            )
            for request_index, output, submitted_at, completed_at in completed:
                key, row, prompt = queued_refs[request_index]
                state = pending_chunks[key]
                if state["records"][row] is not None:
                    raise RuntimeError("SGLang completed one queued request twice")
                queued_cell = state["cell"]
                sample_index = state["sample_index"]
                record, trajectory = parse_sglang_completion(
                    output=output,
                    model_label=args.model,
                    mode=args.mode,
                    benchmark=queued_cell.benchmark,
                    condition=queued_cell.condition,
                    example_id=prompt.example_id,
                    sample_index=sample_index,
                    replay_row=row,
                    settings=settings,
                    think_end_id=think_end_id,
                    study=study,
                )
                state["records"][row] = record
                state["trajectories"][row] = trajectory
                state["submitted_at"][row] = submitted_at
                state["completed_at"][row] = completed_at
                state["request_indices"][row] = request_index
                state["remaining"] -= 1
                if state["remaining"]:
                    continue

                records = state["records"]
                trajectories = state["trajectories"]
                if any(value is None for value in records + trajectories):
                    raise RuntimeError(
                        "completed generation chunk contains an empty row"
                    )
                chunk_timing_rows = []
                for timing_row, record in enumerate(records):
                    submitted_at = float(state["submitted_at"][timing_row])
                    completed_at = float(state["completed_at"][timing_row])
                    chunk_timing_rows.append(
                        {
                            "timing_session_id": timing_session_id,
                            "request_index": int(
                                state["request_indices"][timing_row]
                            ),
                            "benchmark": queued_cell.benchmark,
                            "condition": queued_cell.condition,
                            "sample_index": sample_index,
                            "chunk_index": int(state["chunk_index"]),
                            "chunk_row": timing_row,
                            "example_id": record.example_id,
                            "submitted_at": submitted_at,
                            "completed_at": completed_at,
                            "latency_seconds": completed_at - submitted_at,
                            "response_tokens": record.response_token_count,
                            "request_seed": record.request_seed,
                            "capped": record.capped,
                            "all_soft": record.all_soft,
                            "soft_to_hard": record.soft_to_hard,
                        }
                    )
                store.commit(
                    state["key"],
                    records,
                    trajectories,
                    identity=state["identity"],
                    commit_metadata={
                        "protocol": "opd-icl-chunk-timing-v1",
                        "request_timings": chunk_timing_rows,
                    },
                )
                elapsed = max(
                    max(state["completed_at"]) - min(state["submitted_at"]),
                    1e-12,
                )
                step += 1
                cell_records[
                    (queued_cell.benchmark, queued_cell.condition)
                ].extend(records)
                metrics = generation_chunk_metrics(
                    records, elapsed_seconds=elapsed
                )
                metrics.update(
                    {
                        "generation/chunks_committed": step,
                        "generation/sample_index": sample_index,
                        "generation/benchmark": queued_cell.benchmark,
                        "generation/condition": queued_cell.condition,
                        "generation/chunk_index": state["chunk_index"],
                        "generation/request_queue_size": queue_size,
                        "generation/chunks_remaining": len(pending_chunks) - 1,
                        "integrity/resumed": int(resumed),
                    }
                )
                wandb_run.log(metrics, step=step)
                request_timings.extend(chunk_timing_rows)
                del pending_chunks[key]
        if pending_chunks:
            raise RuntimeError("SGLang queue ended before every chunk completed")

        invalid_cells = []
        demonstrated_boundary_count = 0
        if args.mode == "native_soft":
            for (benchmark, condition), records in cell_records.items():
                gate = boundary_gate(records)
                wandb_run.summary[
                    "boundary/%s/%s/capped_or_all_soft_rate"
                    % (benchmark, condition)
                ] = gate["failure_rate"]
                demonstrated_boundary_count += int(
                    gate["demonstrated_boundary_count"]
                )
                if benchmark in profile.boundary_gate_benchmarks and not gate["valid"]:
                    invalid_cells.append("%s/%s" % (benchmark, condition))
            if (args.smoke or throughput_benchmark) and invalid_cells:
                raise RuntimeError(
                    "native-soft smoke/benchmark exceeded the 5% capped/all-soft gate: %s"
                    % ", ".join(invalid_cells)
                )
            if (args.smoke or throughput_benchmark) and demonstrated_boundary_count == 0:
                raise RuntimeError(
                    "native-soft smoke/benchmark produced no real soft-to-hard categorical boxed answer"
                )
        wandb_run.summary[
            "generation/invalid_capped_or_all_soft_cells"
        ] = invalid_cells
        wandb_run.summary[
            "generation/demonstrated_boundary_count"
        ] = demonstrated_boundary_count
        wandb_run.summary["generation/completed"] = True
        wandb_run.summary["generation/output_root"] = str(output_root)
        if request_timings and existing_completion_payload is None:
            latencies = [row["latency_seconds"] for row in request_timings]
            queue_metrics = _async_queue_metrics(
                request_timings,
                queue_size=queue_size,
                data_parallel_size=args.data_parallel_size,
            )
            generation_seconds = float(queue_metrics["active_generation_seconds"])
            response_tokens = sum(
                int(row["response_tokens"]) for row in request_timings
            )
            session_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in request_timings:
                session_rows[str(row.get("timing_session_id", "single-session"))].append(row)
            final_queue_drain_seconds = sum(
                max(float(row["completed_at"]) for row in values)
                - max(float(row["submitted_at"]) for row in values)
                for values in session_rows.values()
            )
            timing_identities = {
                (
                    row["condition"],
                    row["benchmark"],
                    row["example_id"],
                    int(row["sample_index"]),
                )
                for row in request_timings
            }
            timing_complete = len(timing_identities) == len(request_timings)
            timing_payload = {
                "protocol": "opd-icl-async-throughput-v1",
                "study_id": study,
                "model_label": args.model,
                "mode": args.mode,
                "data_parallel_size": args.data_parallel_size,
                "queue_size": queue_size,
                "max_running_requests": args.max_running_requests,
                "max_running_requests_per_replica": args.max_running_requests,
                "max_running_requests_aggregate": (
                    args.max_running_requests * args.data_parallel_size
                ),
                "engine_load_seconds": engine_load_seconds,
                "warmup_seconds": warmup_seconds,
                "generation_seconds": generation_seconds,
                "final_queue_drain_seconds": final_queue_drain_seconds,
                "request_count": len(request_timings),
                "response_tokens": response_tokens,
                "response_length_mean": response_tokens / len(request_timings),
                "response_length_tokens": {
                    "p50": _percentile(
                        [row["response_tokens"] for row in request_timings], 0.50
                    ),
                    "p95": _percentile(
                        [row["response_tokens"] for row in request_timings], 0.95
                    ),
                    "max": max(row["response_tokens"] for row in request_timings),
                },
                "cap_rate": sum(bool(row["capped"]) for row in request_timings)
                / len(request_timings),
                "soft_to_hard_rate": sum(
                    bool(row["soft_to_hard"]) for row in request_timings
                )
                / len(request_timings),
                "tokens_per_second": response_tokens / generation_seconds,
                "requests_per_hour": len(request_timings)
                * 3600.0
                / generation_seconds,
                "latency_seconds": {
                    "p50": _percentile(latencies, 0.50),
                    "p95": _percentile(latencies, 0.95),
                    "max": max(latencies),
                },
                "queue_occupancy": queue_metrics,
                "eligible_for_allocation": timing_complete,
                "resumed": resumed,
                "rows": sorted(
                    request_timings,
                    key=lambda row: (
                        row.get("timing_session_id", "single-session"),
                        row["request_index"],
                    ),
                ),
            }
            for name in (
                "engine_load_seconds",
                "warmup_seconds",
                "generation_seconds",
                "final_queue_drain_seconds",
                "tokens_per_second",
                "requests_per_hour",
                "response_length_mean",
                "cap_rate",
                "soft_to_hard_rate",
            ):
                wandb_run.summary["throughput/%s" % name] = timing_payload[name]
            for name, value in timing_payload["latency_seconds"].items():
                wandb_run.summary["throughput/request_latency_%s" % name] = value
            for name in ("maximum", "capacity", "mean", "mean_fraction"):
                wandb_run.summary["throughput/queue_occupancy_%s" % name] = (
                    timing_payload["queue_occupancy"][name]
                )
            wandb_run.summary["throughput/eligible_for_allocation"] = bool(
                timing_payload["eligible_for_allocation"]
            )
        completion_payload = (
            dict(existing_completion_payload)
            if existing_completion_payload is not None
            else {
                "generation_manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
                "chunks_committed": step,
                "invalid_capped_or_all_soft_cells": invalid_cells,
                "demonstrated_boundary_count": demonstrated_boundary_count,
            }
        )
        wandb_run.log_artifact(
            str(manifest_path),
            name="%s-generation-manifest" % run_id,
            type="icl-generation-manifest",
        )
        succeeded = True
    finally:
        active_error = sys.exc_info()[1]
        telemetry_error: BaseException | None = None
        try:
            resource_metrics = resource_monitor.stop().to_dict()
            _publish_resource_metrics(wandb_run, resource_metrics)
        except BaseException as error:
            telemetry_error = error
            resource_metrics = {
                "monitor_error": "%s: %s" % (type(error).__name__, error)
            }
            try:
                wandb_run.summary["system/resource_monitor_failure"] = (
                    resource_metrics["monitor_error"]
                )
            except BaseException as summary_error:
                if hasattr(error, "add_note"):
                    error.add_note(
                        "W&B could not record the resource-monitor failure: %s: %s"
                        % (type(summary_error).__name__, summary_error)
                    )
        # The bundled SGLang Engine.shutdown() terminates every child of this
        # process, which includes W&B's service process. Flush W&B first so a
        # successful generation is publishable and a generation error is not
        # obscured by HandleAbandonedError during cleanup. This call must still
        # run when resource telemetry fails to stop or serialize.
        try:
            cleanup_timings = _finish_generation_resources(
                wandb_run=wandb_run, engine=engine, succeeded=succeeded
            )
        except BaseException as cleanup_error:
            if telemetry_error is not None and hasattr(cleanup_error, "add_note"):
                cleanup_error.add_note(
                    "Resource telemetry also failed during cleanup: %s: %s"
                    % (type(telemetry_error).__name__, telemetry_error)
                )
            raise
        if telemetry_error is not None:
            message = "resource telemetry failed during cleanup: %s: %s" % (
                type(telemetry_error).__name__,
                telemetry_error,
            )
            if active_error is not None:
                if hasattr(active_error, "add_note"):
                    active_error.add_note(message)
            else:
                raise RuntimeError(message) from telemetry_error
    if succeeded and completion_payload is not None:
        if timing_payload is not None:
            timing_payload["cleanup"] = cleanup_timings
            timing_payload["system"] = resource_metrics
            _atomic_json(
                output_root / args.model / args.mode / "throughput.json",
                timing_payload,
            )
            completion_payload["throughput_sha256"] = hashlib.sha256(
                canonical_json_bytes(timing_payload)
            ).hexdigest()
        _atomic_json(
            completion_path,
            completion_payload,
        )


def _load_hf_replay_model(model_path: Path, attention_implementation: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("continuous replay requires one CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    validate_atomic_reasoning_tokens(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
        attn_implementation=attention_implementation,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def run_replay(args: argparse.Namespace) -> None:
    study = getattr(args, "study", LEGACY_STUDY_ID)
    if study != LEGACY_STUDY_ID:
        raise ValueError("the Qwen3 pass@1 study does not schedule latent replay")
    assets = _verify_assets_for_study(args.root, study)
    asset_root = Path(args.root).expanduser().resolve()
    generation_root = (
        Path(args.generation_dir).expanduser().resolve()
        if args.generation_dir
        else asset_root / "generation"
    )
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else asset_root / "replay"
    )
    manifest_path = generation_root / args.model / "native_soft" / "generation_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("native-soft generation manifest is missing")
    generation_config = json.loads(manifest_path.read_text(encoding="utf-8"))
    if generation_config.get("model_label") != args.model or generation_config.get("mode") != "native_soft":
        raise ValueError("replay accepts only this model's native-soft generation")
    provenance = source_provenance()
    if generation_config.get("source_provenance") != provenance:
        raise ValueError("generation and replay source provenance differ")
    examples, pairs, subsets, data_manifest = load_icl_dataset(asset_root / "data")
    _validate_generation_binding(
        generation_config,
        assets=assets,
        data_manifest=data_manifest,
        model_label=args.model,
    )
    by_id = {example.example_id: example for example in examples}
    source_cells = [
        cell for cell in generation_config["cells"] if cell["condition"] == "no_demo"
    ]
    if args.benchmarks != ["all"]:
        source_cells = [cell for cell in source_cells if cell["benchmark"] in args.benchmarks]
    if not source_cells:
        raise ValueError("generation manifest contains no selected no-demo cells")
    model_path = asset_root / "models" / args.model
    model, tokenizer = _load_hf_replay_model(model_path, args.attention_implementation)
    settings = SamplingSettings()
    actor_tolerances = ActorAgreementTolerances()
    generation_store = AtomicChunkStore(generation_root)
    replay_store = AtomicReplayStore(output_root)
    config = {
        "protocol": "opd-softgrpo-native-soft-icl-replay-run-v1",
        "source_provenance": provenance,
        "asset_manifest_sha256": assets["content_sha256"],
        "data_manifest_sha256": data_manifest["content_sha256"],
        "model_tree_sha256": assets["models"][args.model]["tree_sha256"],
        "generation_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "model_label": args.model,
        "smoke": bool(generation_config["smoke"]),
        "hidden_chunk_size": args.hidden_chunk_size,
        "vocab_chunk_size": args.vocab_chunk_size,
        "benchmarks": [cell["benchmark"] for cell in source_cells],
        "conditions": list(CORE_CONDITIONS),
        "actor_agreement_tolerances": asdict(actor_tolerances),
    }
    run_id = stable_replay_run_id(config)
    config["wandb_run_id"] = run_id
    replay_manifest = output_root / args.model / "replay_manifest.json"
    if replay_manifest.exists():
        if json.loads(replay_manifest.read_text(encoding="utf-8")) != config:
            raise RuntimeError("replay invocation differs from its resume manifest")
    else:
        _atomic_json(replay_manifest, config)
    wandb_run = init_online_wandb(run_id=run_id, config=config, job_type="icl-replay")
    step = 0
    resumed = False
    succeeded = False
    source_replay_records: list[ReplayRecord] = []
    actor_records: list[ReplayRecord] = []
    try:
        for cell in source_cells:
            benchmark = cell["benchmark"]
            count = int(cell["example_count"])
            samples = int(cell["sample_count"])
            matrix_cell = ICLMatrixCell(**cell)
            _, source_prompts = _cell_prompts(
                matrix_cell,
                examples=examples,
                shuffled_pairs=pairs,
                mechanism_ids=subsets,
            )
            ordered_example_ids = [prompt.example_id for prompt in source_prompts]
            if len(ordered_example_ids) != count:
                raise RuntimeError("replay source cell differs from registered data ordering")
            chunks = math.ceil(count / int(generation_config["chunk_size"]))
            for sample_index in range(samples):
                for chunk_index in range(chunks):
                    source_key = "%s/native_soft/%s/no_demo/sample_%02d/chunk_%05d" % (
                        args.model,
                        benchmark,
                        sample_index,
                        chunk_index,
                    )
                    start = chunk_index * int(generation_config["chunk_size"])
                    stop = min(start + int(generation_config["chunk_size"]), count)
                    expected_example_ids = ordered_example_ids[start:stop]
                    source_identity = _generation_chunk_identity(
                        generation_config,
                        model_label=args.model,
                        inference_mode="native_soft",
                        benchmark=benchmark,
                        condition="no_demo",
                        sample_index=sample_index,
                        chunk_index=chunk_index,
                        example_ids=expected_example_ids,
                    )
                    source_manifest = generation_store.verify(
                        source_key, expected_identity=source_identity
                    )
                    source_records, arrays = generation_store.load(source_key)
                    _validate_completion_records(
                        source_records,
                        model_label=args.model,
                        inference_mode="native_soft",
                        benchmark=benchmark,
                        condition="no_demo",
                        sample_index=sample_index,
                        example_ids=expected_example_ids,
                    )
                    replay_key = "%s/%s/sample_%02d/chunk_%05d" % (
                        args.model,
                        benchmark,
                        sample_index,
                        chunk_index,
                    )
                    identity = {
                        "source_records_sha256": source_manifest["files"]["records"]["sha256"],
                        "source_replay_sha256": source_manifest["files"]["replay"]["sha256"],
                        "conditions": list(CORE_CONDITIONS),
                    }
                    if replay_store.resume_state(replay_key, expected_identity=identity) is not None:
                        resumed_records = replay_store.load(replay_key)
                        resumed_sources = [
                            record
                            for record in resumed_records
                            if record.prompted_condition == "no_demo"
                        ]
                        source_replay_records.extend(resumed_sources)
                        actor_records.extend(
                            record
                            for record in resumed_sources
                            if record.replay_exclusion_reason is None
                        )
                        resumed = True
                        step += 1
                        continue
                    results = []
                    for record in source_records:
                        if record.condition != "no_demo" or record.inference_mode != "native_soft":
                            raise RuntimeError("replay source is not a no-demo native-soft trajectory")
                        target = by_id[record.example_id]
                        prompt_texts = {}
                        for condition in CORE_CONDITIONS:
                            donor = None
                            if condition.endswith("_shuffled"):
                                donor = by_id[pairs[benchmark][record.example_id]]
                            prompt = render_icl_prompt(target, condition, shuffled_donor=donor)
                            prompt_texts[condition] = _render(tokenizer, prompt.user_content)
                        trajectory = unpack_trajectory(arrays, record.replay_row)
                        results.extend(
                            replay_trajectory_many(
                                model=model,
                                tokenizer=tokenizer,
                                trajectory=trajectory,
                                no_demo_prompt=prompt_texts["no_demo"],
                                prompted_prompts=prompt_texts,
                                model_label=args.model,
                                benchmark=benchmark,
                                example_id=record.example_id,
                                sample_index=sample_index,
                                settings=settings,
                                hidden_chunk_size=args.hidden_chunk_size,
                                vocab_chunk_size=args.vocab_chunk_size,
                            )
                        )
                    replay_store.commit(replay_key, results, identity=identity)
                    new_sources = [
                        record
                        for record in results
                        if record.prompted_condition == "no_demo"
                    ]
                    source_replay_records.extend(new_sources)
                    actor_records.extend(
                        record
                        for record in new_sources
                        if record.replay_exclusion_reason is None
                    )
                    step += 1
                    metrics = replay_chunk_metrics(results)
                    metrics.update(
                        {
                            "replay/chunks_committed": step,
                            "replay/sample_index": sample_index,
                            "integrity/resumed": int(resumed),
                        }
                    )
                    wandb_run.log(metrics, step=step)
        source_identities = {
            (
                record.model_label,
                record.benchmark,
                record.example_id,
                record.sample_index,
            )
            for record in source_replay_records
        }
        if len(source_identities) != len(source_replay_records):
            raise RuntimeError("replay source inventory contains duplicate trajectories")
        if actor_records:
            actor_gate = actor_agreement_gate(
                actor_records, tolerances=actor_tolerances
            )
        else:
            actor_gate = {
                "valid": None,
                "reason": "no_valid_latent_sources",
                "latent_slots": 0,
                "active_support_exact_rate": None,
                "centered_logprob_mae": None,
                "centered_logprob_abs_error_max": None,
                "tolerances": asdict(actor_tolerances),
                "active_probability_threshold": (
                    ACTOR_ACTIVE_PROBABILITY_THRESHOLD
                ),
            }
        excluded_source_count = sum(
            record.replay_exclusion_reason == "zero_latent_slots"
            for record in source_replay_records
        )
        wandb_run.summary["replay/source_trajectory_count"] = len(
            source_replay_records
        )
        wandb_run.summary["replay/source_valid_trajectory_count"] = len(
            actor_records
        )
        wandb_run.summary["replay/source_excluded_trajectory_count"] = (
            excluded_source_count
        )
        wandb_run.summary["replay/source_zero_latent_excluded_count"] = (
            excluded_source_count
        )
        for name in (
            "active_support_exact_rate",
            "centered_logprob_mae",
            "centered_logprob_abs_error_max",
        ):
            if actor_gate[name] is not None:
                wandb_run.summary[
                    "integrity/sglang_hf_%s" % name
                ] = actor_gate[name]
        for name, value in actor_gate["tolerances"].items():
            wandb_run.summary[
                "integrity/sglang_hf_tolerance/%s" % name
            ] = value
        wandb_run.summary[
            "integrity/sglang_hf_active_probability_threshold"
        ] = actor_gate["active_probability_threshold"]
        if actor_gate["valid"] is None:
            wandb_run.summary[
                "integrity/sglang_hf_actor_agreement_status"
            ] = "not_evaluated_no_valid_latent_sources"
        else:
            wandb_run.summary[
                "integrity/sglang_hf_actor_agreement_valid"
            ] = bool(actor_gate["valid"])
        if (
            generation_config["smoke"]
            and actor_gate["valid"] is False
        ):
            raise RuntimeError(
                "native-soft smoke failed SGLang/HF actor agreement: %s"
                % json.dumps(actor_gate, sort_keys=True)
            )
        wandb_run.summary["replay/completed"] = True
        wandb_run.summary["replay/output_root"] = str(output_root)
        _atomic_json(
            output_root / args.model / "completion.json",
            {
                "replay_manifest_sha256": hashlib.sha256(
                    replay_manifest.read_bytes()
                ).hexdigest(),
                "chunks_committed": step,
                "source_trajectory_count": len(source_replay_records),
                "source_valid_trajectory_count": len(actor_records),
                "source_excluded_trajectory_count": excluded_source_count,
                "source_zero_latent_excluded_count": excluded_source_count,
                "sglang_hf_actor_agreement": actor_gate,
            },
        )
        wandb_run.log_artifact(
            str(replay_manifest),
            name="%s-manifest" % run_id,
            type="icl-replay-manifest",
        )
        succeeded = True
    finally:
        if not succeeded:
            wandb_run.summary["replay/completed"] = False
        wandb_run.finish()
        del model


def _graders(benchmark: str) -> tuple[str, ...]:
    if benchmark == "gsm8k_test":
        return ("math_verify", "released_last_boxed", "lm_eval_flexible_last_number")
    return ("math_verify", "released_last_boxed")


def _grade(response: str, gold: str, grader: str, benchmark: str) -> bool:
    # AIME prompts preserve the canonical three-digit contest representation,
    # while all correctness interfaces compare its integer value.
    grading_gold = str(int(gold)) if benchmark == "aime2024" else gold
    if grader == "math_verify":
        return math_verify_full_response_grade(response, grading_gold).correct
    if grader == "released_last_boxed":
        result = released_last_boxed_grade(response, grading_gold)
        if benchmark != "aime2024" or result.normalized_prediction is None:
            return result.correct
        # AIME answer strings canonically carry three digits in prompts, but
        # both 007 and 7 denote the same contest answer. The released string
        # normalizer intentionally preserves leading zeros, so apply the
        # benchmark-specific integer equivalence at this interface.
        prediction = result.normalized_prediction
        if not prediction.isdigit() or len(prediction) > 3:
            return False
        return int(prediction) == int(gold)
    if grader == "lm_eval_flexible_last_number":
        return lm_eval_flexible_last_number_grade(response, grading_gold).correct
    raise ValueError("unknown grader: %s" % grader)


def _primary_grade(record: CompletionRecord, gold: str, grader: str, benchmark: str) -> bool:
    """Grade the preregistered output, treating invalid soft boundaries as wrong."""

    if record.inference_mode == "native_soft" and not record.boundary_valid:
        return False
    return _grade(record.response, gold, grader, benchmark)


def _generation_inventory(
    generation_root: Path,
    *,
    smoke: bool,
    study: str = LEGACY_STUDY_ID,
) -> tuple[dict[tuple[str, str], Mapping[str, Any]], list[Any]]:
    profile = get_study_profile(study)
    manifests = {}
    for model, mode in profile.allowed_model_modes:
        path = generation_root / model / mode / "generation_manifest.json"
        completion_path = generation_root / model / mode / "completion.json"
        if not path.is_file():
            raise ValueError("required generation manifest is missing: %s" % path)
        if not completion_path.is_file():
            raise ValueError("generation is not atomically marked complete: %s" % completion_path)
        value = json.loads(path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("generation_manifest_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("generation completion marker failed authentication")
        if value.get("model_label") != model or value.get("mode") != mode:
            raise ValueError("generation manifest identity differs: %s" % path)
        if value.get("study_id", LEGACY_STUDY_ID) != study:
            raise ValueError("generation manifest study differs: %s" % path)
        if bool(value.get("smoke")) != bool(smoke):
            raise ValueError("cannot mix smoke and production generation")
        manifests[(model, mode)] = value
    expected = list(build_icl_matrix(smoke=smoke, study=study))
    expected_by_run = defaultdict(list)
    for cell in expected:
        expected_by_run[(cell.model_label, cell.inference_mode)].append(cell.to_dict())
    for key, manifest in manifests.items():
        if manifest.get("cells") != expected_by_run[key]:
            raise ValueError("generation manifest does not contain its exact registered matrix")
    return manifests, expected


def _demo_for_record(
    *,
    condition: str,
    example: Any,
    by_id: Mapping[str, Any],
    shuffled_pairs: Mapping[str, Mapping[str, str]],
) -> Any | None:
    if condition == "no_demo":
        return None
    if condition.endswith("_shuffled"):
        return by_id[shuffled_pairs[example.benchmark][example.example_id]]
    return example


def _cell_metric_rows(
    cell_states: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    *,
    smoke: bool,
    study: str = LEGACY_STUDY_ID,
) -> list[dict[str, Any]]:
    profile = get_study_profile(study)
    rows = []
    for key in sorted(cell_states):
        state = cell_states[key]
        model, mode, benchmark, condition = key
        samples = state["sample_count"]
        for grader, by_example in sorted(state["outcomes"].items()):
            vectors_by_example = {}
            for example_id in sorted(by_example):
                values = by_example[example_id]
                if set(values) != set(range(samples)):
                    raise RuntimeError("cell has an incomplete common-seed outcome set")
                vectors_by_example[example_id] = tuple(
                    bool(values[index]) for index in range(samples)
                )
            pass1_by_example = {
                example_id: sum(values) / samples
                for example_id, values in vectors_by_example.items()
            }
            zeros = {example_id: 0.0 for example_id in pass1_by_example}
            pass1_bootstrap = paired_bootstrap_difference(
                pass1_by_example,
                zeros,
            )
            if samples == 8:
                pass8_by_example = {
                    example_id: float(any(values))
                    for example_id, values in vectors_by_example.items()
                }
                pass8_bootstrap = paired_bootstrap_difference(
                    pass8_by_example,
                    zeros,
                )
            else:
                pass8_bootstrap = None
            row = {
                    "model_label": model,
                    "inference_mode": mode,
                    "benchmark": benchmark,
                    "condition": condition,
                    "grader": grader,
                    "pass_at_1": pass1_bootstrap["difference"],
                    "pass_at_1_ci_low": pass1_bootstrap["ci_low"],
                    "pass_at_1_ci_high": pass1_bootstrap["ci_high"],
                    "bootstrap_resamples": pass1_bootstrap["resamples"],
                    "bootstrap_seed": pass1_bootstrap["bootstrap_seed"],
                    "example_count": len(vectors_by_example),
                    "samples_per_example": samples,
                    "smoke": smoke,
                }
            if 8 in profile.pass_ks:
                row.update(
                    {
                        "pass_at_8": (
                            None
                            if pass8_bootstrap is None
                            else pass8_bootstrap["difference"]
                        ),
                        "pass_at_8_ci_low": (
                            None
                            if pass8_bootstrap is None
                            else pass8_bootstrap["ci_low"]
                        ),
                        "pass_at_8_ci_high": (
                            None
                            if pass8_bootstrap is None
                            else pass8_bootstrap["ci_high"]
                        ),
                    }
                )
            if study == QWEN3_STUDY_ID:
                row["model_display_name"] = QWEN3_MODEL_DISPLAY_NAMES[model]
                row["inference_mode_label"] = INFERENCE_MODE_DISPLAY_NAMES[mode]
            rows.append(row)
    return rows


def _example_pass1(state: Mapping[str, Any], grader: str) -> dict[str, float]:
    samples = state["sample_count"]
    return {
        example_id: sum(bool(values[index]) for index in range(samples)) / samples
        for example_id, values in state["outcomes"][grader].items()
    }


def _example_pass8(state: Mapping[str, Any], grader: str) -> dict[str, float]:
    if state["sample_count"] != 8:
        raise ValueError("pass@8 contrasts require exactly eight samples")
    return {
        example_id: float(any(bool(values[index]) for index in range(8)))
        for example_id, values in state["outcomes"][grader].items()
    }


def _outcome_vectors(state: Mapping[str, Any], grader: str) -> dict[str, tuple[bool, ...]]:
    samples = state["sample_count"]
    return {
        example_id: tuple(bool(values[index]) for index in range(samples))
        for example_id, values in state["outcomes"][grader].items()
    }


def _legacy_comparison_rows(
    states: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    *,
    smoke: bool,
) -> list[dict[str, Any]]:
    if smoke:
        return []
    result = []

    def boundary_gate_valid(
        key: tuple[str, str, str, str], state: Mapping[str, Any]
    ) -> bool:
        if key[1] != "native_soft":
            return True
        return state["capped_or_all_soft"] / state["count"] <= 0.05

    def compare(key_t: tuple[str, str, str, str], key_c: tuple[str, str, str, str], name: str) -> None:
        treatment, control = states[key_t], states[key_c]
        treatment_gate = boundary_gate_valid(key_t, treatment)
        control_gate = boundary_gate_valid(key_c, control)
        common_graders = sorted(set(treatment["outcomes"]) & set(control["outcomes"]))
        for grader in common_graders:
            bootstrap = paired_bootstrap_difference(
                _example_pass1(treatment, grader), _example_pass1(control, grader)
            )
            rescue = rescue_harm_rates(
                _outcome_vectors(treatment, grader), _outcome_vectors(control, grader)
            )
            result.append(
                {
                    "comparison": name,
                    "model_label": key_t[0],
                    "inference_mode": key_t[1],
                    "benchmark": key_t[2],
                    "grader": grader,
                    "estimand": "pass_at_1",
                    "treatment_boundary_gate_valid": treatment_gate,
                    "control_boundary_gate_valid": control_gate,
                    "comparison_boundary_gate_valid": treatment_gate and control_gate,
                    **bootstrap,
                    **rescue,
                }
            )
            if treatment["sample_count"] == control["sample_count"] == 8:
                pass8_bootstrap = paired_bootstrap_difference(
                    _example_pass8(treatment, grader),
                    _example_pass8(control, grader),
                )
                result.append(
                    {
                        "comparison": name,
                        "model_label": key_t[0],
                        "inference_mode": key_t[1],
                        "benchmark": key_t[2],
                        "grader": grader,
                        "estimand": "pass_at_8",
                        "treatment_boundary_gate_valid": treatment_gate,
                        "control_boundary_gate_valid": control_gate,
                        "comparison_boundary_gate_valid": treatment_gate
                        and control_gate,
                        **pass8_bootstrap,
                    }
                )

    run_keys = sorted({key[:3] for key in states})
    for model, mode, benchmark in run_keys:
        for family in ("sdft", "sdpg"):
            matched = (model, mode, benchmark, family + "_matched")
            no_demo = (model, mode, benchmark, "no_demo")
            compare(matched, no_demo, family + "_matched_minus_no_demo")
        compare(
            (model, mode, benchmark, "sdft_matched"),
            (model, mode, benchmark, "sdpg_matched"),
            "sdft_matched_minus_sdpg_matched",
        )

    for benchmark in STUDY_BENCHMARKS:
        for family in ("sdft", "sdpg"):
            post_t = states[("softgrpo", "native_soft", benchmark, family + "_matched")]
            post_c = states[("softgrpo", "native_soft", benchmark, "no_demo")]
            start_t = states[("starting", "native_soft", benchmark, family + "_matched")]
            start_c = states[("starting", "native_soft", benchmark, "no_demo")]
            post_t_gate = boundary_gate_valid(
                ("softgrpo", "native_soft", benchmark, family + "_matched"),
                post_t,
            )
            post_c_gate = boundary_gate_valid(
                ("softgrpo", "native_soft", benchmark, "no_demo"), post_c
            )
            start_t_gate = boundary_gate_valid(
                ("starting", "native_soft", benchmark, family + "_matched"),
                start_t,
            )
            start_c_gate = boundary_gate_valid(
                ("starting", "native_soft", benchmark, "no_demo"), start_c
            )
            for grader in _graders(benchmark):
                for estimand, extractor in (
                    ("pass_at_1", _example_pass1),
                    ("pass_at_8", _example_pass8),
                ):
                    bootstrap = paired_bootstrap_difference_in_differences(
                        extractor(post_t, grader),
                        extractor(post_c, grader),
                        extractor(start_t, grader),
                        extractor(start_c, grader),
                    )
                    contrast_definition = bootstrap.pop("estimand")
                    result.append(
                        {
                            "comparison": (
                                "post_minus_start_%s_matched_minus_no_demo"
                                % family
                            ),
                            "model_label": "difference_in_differences",
                            "inference_mode": "native_soft",
                            "benchmark": benchmark,
                            "grader": grader,
                            "estimand": estimand,
                            "contrast_definition": contrast_definition,
                            "post_treatment_boundary_gate_valid": post_t_gate,
                            "post_control_boundary_gate_valid": post_c_gate,
                            "start_treatment_boundary_gate_valid": start_t_gate,
                            "start_control_boundary_gate_valid": start_c_gate,
                            "comparison_boundary_gate_valid": all(
                                (post_t_gate, post_c_gate, start_t_gate, start_c_gate)
                            ),
                            **bootstrap,
                        }
                    )
    return result


def _qwen3_comparison_rows(
    states: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build only preregistered paired pass@1 contrasts for Qwen3."""

    profile = get_study_profile(QWEN3_STUDY_ID)
    result: list[dict[str, Any]] = []

    def gate(
        key: tuple[str, str, str, str], state: Mapping[str, Any]
    ) -> tuple[bool, bool | None]:
        applied = (
            key[1] == "native_soft"
            and key[2] in profile.boundary_gate_benchmarks
        )
        valid = (
            state["capped_or_all_soft"] / state["count"] <= 0.05
            if applied
            else None
        )
        return applied, valid

    def compare(
        treatment_key: tuple[str, str, str, str],
        control_key: tuple[str, str, str, str],
        name: str,
    ) -> None:
        treatment = states[treatment_key]
        control = states[control_key]
        treatment_applied, treatment_valid = gate(treatment_key, treatment)
        control_applied, control_valid = gate(control_key, control)
        applied_values = [
            value
            for applied, value in (
                (treatment_applied, treatment_valid),
                (control_applied, control_valid),
            )
            if applied
        ]
        comparison_valid = (
            all(bool(value) for value in applied_values)
            if applied_values
            else None
        )
        for grader in sorted(
            set(treatment["outcomes"]) & set(control["outcomes"])
        ):
            bootstrap = paired_bootstrap_difference(
                _example_pass1(treatment, grader),
                _example_pass1(control, grader),
            )
            rescue = rescue_harm_rates(
                _outcome_vectors(treatment, grader),
                _outcome_vectors(control, grader),
            )
            result.append(
                {
                    "comparison": name,
                    "model_label": treatment_key[0],
                    "inference_mode": treatment_key[1],
                    "benchmark": treatment_key[2],
                    "grader": grader,
                    "estimand": "pass_at_1",
                    "treatment_model_label": treatment_key[0],
                    "control_model_label": control_key[0],
                    "treatment_inference_mode": treatment_key[1],
                    "control_inference_mode": control_key[1],
                    "treatment_condition": treatment_key[3],
                    "control_condition": control_key[3],
                    "treatment_boundary_gate_applied": treatment_applied,
                    "control_boundary_gate_applied": control_applied,
                    "treatment_boundary_gate_valid": treatment_valid,
                    "control_boundary_gate_valid": control_valid,
                    "comparison_boundary_gate_valid": comparison_valid,
                    "treatment_model_display_name": QWEN3_MODEL_DISPLAY_NAMES[
                        treatment_key[0]
                    ],
                    "control_model_display_name": QWEN3_MODEL_DISPLAY_NAMES[
                        control_key[0]
                    ],
                    "treatment_inference_mode_label": INFERENCE_MODE_DISPLAY_NAMES[
                        treatment_key[1]
                    ],
                    "control_inference_mode_label": INFERENCE_MODE_DISPLAY_NAMES[
                        control_key[1]
                    ],
                    **bootstrap,
                    **rescue,
                }
            )

    for model in profile.model_labels:
        for mode in profile.inference_modes:
            for benchmark in profile.benchmarks:
                no_demo = (model, mode, benchmark, "no_demo")
                sdft = (model, mode, benchmark, "sdft_matched")
                sdpg = (model, mode, benchmark, "sdpg_matched")
                compare(sdft, no_demo, "sdft_matched_minus_no_demo")
                compare(sdpg, no_demo, "sdpg_matched_minus_no_demo")
                compare(sdpg, sdft, "sdpg_matched_minus_sdft_matched")

    for model in profile.model_labels:
        for benchmark in profile.benchmarks:
            for condition in profile.core_conditions:
                compare(
                    (model, "native_soft", benchmark, condition),
                    (model, "hard_token", benchmark, condition),
                    "soft_thinking_minus_discrete_token_cot",
                )

    smaller, larger = profile.model_labels
    for mode in profile.inference_modes:
        for benchmark in profile.benchmarks:
            for condition in profile.core_conditions:
                compare(
                    (larger, mode, benchmark, condition),
                    (smaller, mode, benchmark, condition),
                    "qwen3_1p7b_minus_qwen3_0p6b",
                )
    return result


def _comparison_rows(
    states: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    *,
    smoke: bool,
    study: str = LEGACY_STUDY_ID,
) -> list[dict[str, Any]]:
    if smoke:
        return []
    if study == QWEN3_STUDY_ID:
        return _qwen3_comparison_rows(states)
    return _legacy_comparison_rows(states, smoke=False)


def _pearson(sum_x: float, sum_y: float, sum_x2: float, sum_y2: float, sum_xy: float, n: int) -> float | None:
    if n < 2:
        return None
    numerator = n * sum_xy - sum_x * sum_y
    denominator = math.sqrt(max(n * sum_x2 - sum_x * sum_x, 0.0) * max(n * sum_y2 - sum_y * sum_y, 0.0))
    return None if denominator == 0 else numerator / denominator


def _aggregate_replay(
    replay_root: Path,
    generation_root: Path,
    generation_manifests: Mapping[tuple[str, str], Mapping[str, Any]],
    states: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    expected_example_ids: Mapping[tuple[str, str, str, str], Sequence[str]],
    *,
    assets: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    store = AtomicReplayStore(replay_root)
    generation_store = AtomicChunkStore(generation_root)
    accumulators: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for model in ("starting", "softgrpo"):
        replay_manifest = replay_root / model / "replay_manifest.json"
        completion_path = replay_root / model / "completion.json"
        if not replay_manifest.is_file():
            raise ValueError("required replay manifest is missing: %s" % replay_manifest)
        if not completion_path.is_file():
            raise ValueError("replay is not atomically marked complete: %s" % completion_path)
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("replay_manifest_sha256") != hashlib.sha256(replay_manifest.read_bytes()).hexdigest():
            raise ValueError("replay completion marker failed authentication")
        generation = generation_manifests[(model, "native_soft")]
        replay_config = json.loads(replay_manifest.read_text(encoding="utf-8"))
        if (
            replay_config.get("source_provenance") != source_provenance()
            or replay_config.get("asset_manifest_sha256")
            != assets["content_sha256"]
            or replay_config.get("data_manifest_sha256")
            != data_manifest["content_sha256"]
            or replay_config.get("model_tree_sha256")
            != assets["models"][model]["tree_sha256"]
            or replay_config.get("model_label") != model
            or replay_config.get("generation_manifest_sha256")
            != hashlib.sha256(
                canonical_json_bytes(generation)
            ).hexdigest()
        ):
            raise ValueError("replay manifest provenance differs for %s" % model)
        for cell in generation["cells"]:
            if cell["condition"] != "no_demo":
                continue
            benchmark = cell["benchmark"]
            chunks = math.ceil(cell["example_count"] / generation["chunk_size"])
            for sample_index in range(cell["sample_count"]):
                for chunk_index in range(chunks):
                    key = "%s/%s/sample_%02d/chunk_%05d" % (
                        model,
                        benchmark,
                        sample_index,
                        chunk_index,
                    )
                    source_key = "%s/native_soft/%s/no_demo/sample_%02d/chunk_%05d" % (
                        model,
                        benchmark,
                        sample_index,
                        chunk_index,
                    )
                    start = chunk_index * int(generation["chunk_size"])
                    stop = min(
                        start + int(generation["chunk_size"]),
                        int(cell["example_count"]),
                    )
                    chunk_example_ids = expected_example_ids[
                        (model, "native_soft", benchmark, "no_demo")
                    ][start:stop]
                    source_identity = _generation_chunk_identity(
                        generation,
                        model_label=model,
                        inference_mode="native_soft",
                        benchmark=benchmark,
                        condition="no_demo",
                        sample_index=sample_index,
                        chunk_index=chunk_index,
                        example_ids=chunk_example_ids,
                    )
                    source_manifest = generation_store.verify(
                        source_key, expected_identity=source_identity
                    )
                    source_records, _ = generation_store.load(source_key)
                    _validate_completion_records(
                        source_records,
                        model_label=model,
                        inference_mode="native_soft",
                        benchmark=benchmark,
                        condition="no_demo",
                        sample_index=sample_index,
                        example_ids=chunk_example_ids,
                    )
                    replay_identity = {
                        "source_records_sha256": source_manifest["files"]["records"]["sha256"],
                        "source_replay_sha256": source_manifest["files"]["replay"]["sha256"],
                        "conditions": list(CORE_CONDITIONS),
                    }
                    data_path, _ = store.paths(key)
                    store.verify(key, expected_identity=replay_identity)
                    with data_path.open("r", encoding="utf-8") as stream:
                        records = [ReplayRecord.from_mapping(json.loads(line)) for line in stream]
                    _validate_replay_records(
                        records,
                        source_records=source_records,
                        model_label=model,
                        benchmark=benchmark,
                        sample_index=sample_index,
                    )
                    for record in records:
                        key4 = (model, "native_soft", benchmark, record.prompted_condition)
                        prompted_state = states[key4]
                        no_demo_state = states[(model, "native_soft", benchmark, "no_demo")]
                        prompted_correct = float(
                            prompted_state["outcomes"]["math_verify"][record.example_id][record.sample_index]
                        )
                        no_demo_correct = bool(
                            no_demo_state["outcomes"]["math_verify"][record.example_id][record.sample_index]
                        )
                        acc = accumulators.setdefault(
                            key4,
                            {
                                "slots": 0,
                                "records": 0,
                                "valid_records": 0,
                                "excluded_records": 0,
                                "forward_sum": 0.0,
                                "reverse_sum": 0.0,
                                "sequence_sum": 0.0,
                                "entropy_sum": 0.0,
                                "top1_sum": 0.0,
                                "actor_exact_slots": 0,
                                "actor_value_count": 0,
                                "actor_error_sum": 0.0,
                                "actor_error_max": 0.0,
                                "x": 0.0,
                                "y": 0.0,
                                "x2": 0.0,
                                "y2": 0.0,
                                "xy": 0.0,
                                "strata": defaultdict(lambda: {"n": 0, "kl": 0.0, "reward": 0.0}),
                            },
                        )
                        acc["records"] += 1
                        if record.replay_exclusion_reason is not None:
                            acc["excluded_records"] += 1
                            continue
                        acc["valid_records"] += 1
                        x = float(record.forward_kl_mean)
                        y = prompted_correct
                        acc["slots"] += record.latent_token_count
                        acc["forward_sum"] += float(record.forward_kl_sum)
                        acc["reverse_sum"] += float(record.reverse_kl_sum)
                        acc["sequence_sum"] += float(record.forward_kl_sum)
                        acc["entropy_sum"] += (
                            float(record.prompted_entropy_mean)
                            * record.latent_token_count
                        )
                        acc["top1_sum"] += (
                            float(record.prompted_top1_probability_mean)
                            * record.latent_token_count
                        )
                        acc["actor_exact_slots"] += (
                            record.sglang_hf_active_support_exact_slots
                        )
                        acc["actor_value_count"] += (
                            record.sglang_hf_centered_logprob_value_count
                        )
                        acc["actor_error_sum"] += (
                            float(record.sglang_hf_centered_logprob_abs_error_sum)
                        )
                        acc["actor_error_max"] = max(
                            acc["actor_error_max"],
                            float(record.sglang_hf_centered_logprob_abs_error_max),
                        )
                        acc["x"] += x
                        acc["y"] += y
                        acc["x2"] += x * x
                        acc["y2"] += y * y
                        acc["xy"] += x * y
                        stratum = acc["strata"]["correct" if no_demo_correct else "incorrect"]
                        stratum["n"] += 1
                        stratum["kl"] += x
                        stratum["reward"] += y
    rows = []
    for key, acc in sorted(accumulators.items()):
        model, mode, benchmark, condition = key
        row = {
            "model_label": model,
            "inference_mode": mode,
            "benchmark": benchmark,
            "condition": condition,
            "record_count": acc["records"],
            "source_trajectory_count": acc["records"],
            "valid_replay_trajectory_count": acc["valid_records"],
            "excluded_replay_trajectory_count": acc["excluded_records"],
            "zero_latent_excluded_count": acc["excluded_records"],
            "latent_slot_count": acc["slots"],
        }
        if acc["valid_records"]:
            row.update(
                {
                    "forward_kl_slot_mean": acc["forward_sum"] / acc["slots"],
                    "forward_kl_sequence_mean": (
                        acc["sequence_sum"] / acc["valid_records"]
                    ),
                    "reverse_kl_slot_mean": acc["reverse_sum"] / acc["slots"],
                    "prompted_entropy_slot_mean": (
                        acc["entropy_sum"] / acc["slots"]
                    ),
                    "prompted_top1_probability_slot_mean": (
                        acc["top1_sum"] / acc["slots"]
                    ),
                    "sglang_hf_active_support_exact_rate": (
                        acc["actor_exact_slots"] / acc["slots"]
                    ),
                    "sglang_hf_centered_logprob_mae": (
                        acc["actor_error_sum"] / acc["actor_value_count"]
                    ),
                    "sglang_hf_centered_logprob_abs_error_max": acc[
                        "actor_error_max"
                    ],
                    "reward_kl_pearson": _pearson(
                        acc["x"],
                        acc["y"],
                        acc["x2"],
                        acc["y2"],
                        acc["xy"],
                        acc["valid_records"],
                    ),
                }
            )
        else:
            row.update(
                {
                    "forward_kl_slot_mean": None,
                    "forward_kl_sequence_mean": None,
                    "reverse_kl_slot_mean": None,
                    "prompted_entropy_slot_mean": None,
                    "prompted_top1_probability_slot_mean": None,
                    "sglang_hf_active_support_exact_rate": None,
                    "sglang_hf_centered_logprob_mae": None,
                    "sglang_hf_centered_logprob_abs_error_max": None,
                    "reward_kl_pearson": None,
                }
            )
        for name, stratum in acc["strata"].items():
            row["no_demo_%s_count" % name] = stratum["n"]
            row["no_demo_%s_kl_mean" % name] = (
                None if not stratum["n"] else stratum["kl"] / stratum["n"]
            )
            row["no_demo_%s_reward_mean" % name] = (
                None if not stratum["n"] else stratum["reward"] / stratum["n"]
            )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        from .icl_runtime import _atomic_bytes

        _atomic_bytes(path, b"")
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    from io import StringIO

    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    from .icl_runtime import _atomic_bytes

    _atomic_bytes(path, stream.getvalue().encode("utf-8"))


def _flatten_wandb(
    metrics: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    replay: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for row in metrics:
        prefix = "eval/%s/%s/%s/%s/%s" % (
            row["benchmark"],
            row["model_label"],
            row["inference_mode"],
            row["condition"],
            row["grader"],
        )
        result[prefix + "/pass_at_1"] = row["pass_at_1"]
        result[prefix + "/pass_at_1_ci_low"] = row["pass_at_1_ci_low"]
        result[prefix + "/pass_at_1_ci_high"] = row["pass_at_1_ci_high"]
        if row.get("pass_at_8") is not None:
            result[prefix + "/pass_at_8"] = row["pass_at_8"]
            result[prefix + "/pass_at_8_ci_low"] = row["pass_at_8_ci_low"]
            result[prefix + "/pass_at_8_ci_high"] = row["pass_at_8_ci_high"]
    for row in diagnostics:
        prefix = "eval/%s/%s/%s/%s" % (
            row["benchmark"], row["model_label"], row["inference_mode"], row["condition"]
        )
        for name, value in row.items():
            if name not in {
                "benchmark",
                "model_label",
                "inference_mode",
                "condition",
            } and isinstance(value, (int, float)):
                result[prefix + "/" + name] = value
    for row in comparisons:
        prefix = "comparison/%s/%s/%s/%s/%s/%s" % (
            row["benchmark"],
            row["model_label"],
            row["inference_mode"],
            row["comparison"],
            row["grader"],
            row["estimand"],
        )
        for name in (
            "difference",
            "ci_low",
            "ci_high",
            "rescue_rate",
            "harm_rate",
            "treatment_boundary_gate_valid",
            "control_boundary_gate_valid",
            "post_treatment_boundary_gate_valid",
            "post_control_boundary_gate_valid",
            "start_treatment_boundary_gate_valid",
            "start_control_boundary_gate_valid",
            "comparison_boundary_gate_valid",
        ):
            if (
                name in row
                and isinstance(row[name], (int, float))
                and math.isfinite(float(row[name]))
            ):
                result[prefix + "/" + name] = row[name]
    for row in replay:
        prefix = "replay/%s/%s/%s" % (
            row["benchmark"],
            row["model_label"],
            row["condition"],
        )
        for name, value in row.items():
            if name not in {
                "benchmark",
                "model_label",
                "inference_mode",
                "condition",
            } and isinstance(value, (int, float)):
                result[prefix + "/" + name] = value
    return result


def _from_getgo_assessment(
    metrics: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    metric_index = {
        (
            row["model_label"],
            row["inference_mode"],
            row["benchmark"],
            row["condition"],
            row["grader"],
        ): row
        for row in metrics
    }
    diagnostic_index = {
        (row["model_label"], row["inference_mode"], row["benchmark"], row["condition"]): row
        for row in diagnostics
    }
    comparison_index = {
        (
            row["model_label"],
            row["inference_mode"],
            row["benchmark"],
            row["comparison"],
            row["grader"],
            row["estimand"],
        ): row
        for row in comparisons
    }
    result = []
    for benchmark in STUDY_BENCHMARKS:
        for family in ("sdft", "sdpg"):
            condition = family + "_matched"
            metric = metric_index[("starting", "native_soft", benchmark, condition, "math_verify")]
            matched_diagnostic = diagnostic_index[
                ("starting", "native_soft", benchmark, condition)
            ]
            no_demo_diagnostic = diagnostic_index[
                ("starting", "native_soft", benchmark, "no_demo")
            ]
            comparison = comparison_index[
                (
                    "starting",
                    "native_soft",
                    benchmark,
                    family + "_matched_minus_no_demo",
                    "math_verify",
                    "pass_at_1",
                )
            ]
            positive_paired_ci = comparison["ci_low"] > 0.0
            matched_boundary_valid = bool(
                matched_diagnostic["boundary_gate_valid"]
            )
            no_demo_boundary_valid = bool(
                no_demo_diagnostic["boundary_gate_valid"]
            )
            boundary_valid = matched_boundary_valid and no_demo_boundary_valid
            result.append(
                {
                    "benchmark": benchmark,
                    "prompt_family": family,
                    "native_soft_pass_at_1": metric["pass_at_1"],
                    "matched_minus_no_demo_ci_low": comparison["ci_low"],
                    "positive_paired_95ci": positive_paired_ci,
                    "matched_boundary_gate_valid": matched_boundary_valid,
                    "no_demo_boundary_gate_valid": no_demo_boundary_valid,
                    "boundary_gate_valid": boundary_valid,
                }
            )
    return result


def run_aggregate(args: argparse.Namespace) -> None:
    study = getattr(args, "study", LEGACY_STUDY_ID)
    profile = get_study_profile(study)
    assets = _verify_assets_for_study(args.root, study)
    asset_root = Path(args.root).expanduser().resolve()
    generation_root = (
        Path(args.generation_dir).expanduser().resolve()
        if args.generation_dir
        else asset_root / "generation"
    )
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else asset_root / "reports"
    )
    if study == LEGACY_STUDY_ID:
        manifests, cells = _generation_inventory(
            generation_root, smoke=args.smoke
        )
    else:
        manifests, cells = _generation_inventory(
            generation_root, smoke=args.smoke, study=study
        )
    provenance = source_provenance()
    if any(
        manifest.get("source_provenance") != provenance
        for manifest in manifests.values()
    ):
        raise ValueError("generation and aggregation source provenance differ")
    examples, shuffled_pairs, mechanism_ids, data_manifest = load_icl_dataset(
        asset_root / "data"
    )
    for (model_label, _), manifest in manifests.items():
        _validate_generation_binding(
            manifest,
            assets=assets,
            data_manifest=data_manifest,
            model_label=model_label,
            study=study,
        )
    by_id = {example.example_id: example for example in examples}
    store = AtomicChunkStore(generation_root)
    states = {}
    scored_rows = []
    diagnostic_rows = []
    for cell in cells:
        key4 = (cell.model_label, cell.inference_mode, cell.benchmark, cell.condition)
        state = {
            "sample_count": cell.sample_count,
            "outcomes": {grader: defaultdict(dict) for grader in _graders(cell.benchmark)},
            "count": 0,
            "response_tokens": 0,
            "capped": 0,
            "capped_or_all_soft": 0,
            "boundary_valid": 0,
            "soft_to_hard": 0,
            "all_soft": 0,
            "close_tag": 0,
            "latent_tokens": 0,
            "hard_answer_tokens": 0,
            "mixture_entropy_sum": 0.0,
            "top1_weight_sum": 0.0,
            "soft_hard_agreement_sum": 0.0,
            "overlap_sum": 0.0,
            "overlap_count": 0,
            "overlap_invalid_boundary_excluded": 0,
            "copy_sum": 0,
            "demo_count": 0,
        }
        manifest = manifests[(cell.model_label, cell.inference_mode)]
        _, expected_prompts = _cell_prompts(
            cell,
            examples=examples,
            shuffled_pairs=shuffled_pairs,
            mechanism_ids=mechanism_ids,
        )
        expected_example_ids = [prompt.example_id for prompt in expected_prompts]
        chunks = math.ceil(cell.example_count / manifest["chunk_size"])
        seen = set()
        for sample_index in range(cell.sample_count):
            for chunk_index in range(chunks):
                chunk_key = "%s/%s/%s/%s/sample_%02d/chunk_%05d" % (
                    cell.model_label, cell.inference_mode, cell.benchmark, cell.condition, sample_index, chunk_index
                )
                start = chunk_index * int(manifest["chunk_size"])
                stop = min(start + int(manifest["chunk_size"]), cell.example_count)
                chunk_example_ids = expected_example_ids[start:stop]
                chunk_identity = _generation_chunk_identity(
                    manifest,
                    model_label=cell.model_label,
                    inference_mode=cell.inference_mode,
                    benchmark=cell.benchmark,
                    condition=cell.condition,
                    sample_index=sample_index,
                    chunk_index=chunk_index,
                    example_ids=chunk_example_ids,
                )
                records, _ = store.load(
                    chunk_key, expected_identity=chunk_identity
                )
                _validate_completion_records(
                    records,
                    model_label=cell.model_label,
                    inference_mode=cell.inference_mode,
                    benchmark=cell.benchmark,
                    condition=cell.condition,
                    sample_index=sample_index,
                    example_ids=chunk_example_ids,
                )
                for record in records:
                    identity = (record.example_id, record.sample_index)
                    if identity in seen:
                        raise RuntimeError("duplicate generated example/sample")
                    seen.add(identity)
                    example = by_id[record.example_id]
                    if example.benchmark != cell.benchmark:
                        raise RuntimeError("generated example crossed benchmarks")
                    scores = {}
                    for grader in _graders(cell.benchmark):
                        correct = _primary_grade(
                            record, example.gold_answer, grader, cell.benchmark
                        )
                        scores[grader] = correct
                        state["outcomes"][grader][record.example_id][sample_index] = correct
                    demo = _demo_for_record(
                        condition=cell.condition,
                        example=example,
                        by_id=by_id,
                        shuffled_pairs=shuffled_pairs,
                    )
                    overlap = copy = None
                    if demo is not None:
                        copy = normalized_answer_copy(record.response, demo.gold_answer, cell.benchmark)
                        state["copy_sum"] += int(copy)
                        state["demo_count"] += 1
                        if (
                            record.inference_mode == "hard_token"
                            or record.boundary_valid
                        ):
                            generated_reasoning = record.response.split(
                                "</think>", 1
                            )[0]
                            overlap = rationale_token_overlap_f1(
                                generated_reasoning, demo.gold_cot
                            )
                            state["overlap_sum"] += overlap
                            state["overlap_count"] += 1
                        else:
                            state["overlap_invalid_boundary_excluded"] += 1
                    state["count"] += 1
                    state["response_tokens"] += record.response_token_count
                    state["capped"] += int(record.capped)
                    state["capped_or_all_soft"] += int(
                        record.capped or record.all_soft
                    )
                    state["boundary_valid"] += int(record.boundary_valid)
                    state["soft_to_hard"] += int(record.soft_to_hard)
                    state["all_soft"] += int(record.all_soft)
                    state["close_tag"] += int(record.close_tag)
                    state["latent_tokens"] += record.latent_token_count
                    state["hard_answer_tokens"] += max(
                        record.hard_token_count
                        - int(record.close_tag and record.soft_to_hard),
                        0,
                    )
                    if record.latent_token_count:
                        state["mixture_entropy_sum"] += record.mixture_entropy_mean * record.latent_token_count
                        state["top1_weight_sum"] += record.top1_weight_mean * record.latent_token_count
                        state["soft_hard_agreement_sum"] += record.soft_hard_agreement * record.latent_token_count
                    scored_rows.append(
                        {
                            "model_label": cell.model_label,
                            "inference_mode": cell.inference_mode,
                            "benchmark": cell.benchmark,
                            "condition": cell.condition,
                            "example_id": record.example_id,
                            "sample_index": sample_index,
                            "request_seed": record.request_seed,
                            "scores": scores,
                            "rationale_overlap_f1": overlap,
                            "demonstrated_answer_copy": copy,
                        }
                    )
        if len(seen) != cell.example_count * cell.sample_count:
            raise RuntimeError("cell is incomplete")
        states[key4] = state
        diagnostic = {
            "model_label": cell.model_label,
            "inference_mode": cell.inference_mode,
            "benchmark": cell.benchmark,
            "condition": cell.condition,
            "response_token_length_mean": state["response_tokens"] / state["count"],
            "cap_rate": state["capped"] / state["count"],
        }
        if study == QWEN3_STUDY_ID:
            diagnostic["model_display_name"] = QWEN3_MODEL_DISPLAY_NAMES[
                cell.model_label
            ]
            diagnostic["inference_mode_label"] = INFERENCE_MODE_DISPLAY_NAMES[
                cell.inference_mode
            ]
        if cell.inference_mode == "native_soft":
            gate_applied = cell.benchmark in profile.boundary_gate_benchmarks
            gate_valid = (
                state["capped_or_all_soft"] / state["count"] <= 0.05
                if gate_applied
                else None
            )
            diagnostic.update(
                {
                    "latent_token_length_mean": state["latent_tokens"] / state["count"],
                    "hard_answer_token_length_mean": state["hard_answer_tokens"] / state["count"],
                    "close_tag_rate": state["close_tag"] / state["count"],
                    "soft_to_hard_rate": state["soft_to_hard"] / state["count"],
                    "all_soft_rate": state["all_soft"] / state["count"],
                    "boundary_valid_rate": state["boundary_valid"] / state["count"],
                    "capped_or_all_soft_rate": state["capped_or_all_soft"]
                    / state["count"],
                    "boundary_gate_applied": gate_applied,
                    "boundary_gate_valid": gate_valid,
                    "mixture_entropy_slot_mean": (
                        None
                        if not state["latent_tokens"]
                        else state["mixture_entropy_sum"] / state["latent_tokens"]
                    ),
                    "top1_weight_slot_mean": (
                        None
                        if not state["latent_tokens"]
                        else state["top1_weight_sum"] / state["latent_tokens"]
                    ),
                    "soft_hard_agreement_slot_mean": (
                        None
                        if not state["latent_tokens"]
                        else state["soft_hard_agreement_sum"] / state["latent_tokens"]
                    ),
                }
            )
        if state["demo_count"]:
            diagnostic["demonstrated_answer_copy_rate"] = state["copy_sum"] / state["demo_count"]
            diagnostic["demonstrated_answer_copy_count"] = state["demo_count"]
            diagnostic["rationale_overlap_valid_count"] = state["overlap_count"]
            diagnostic["rationale_overlap_invalid_boundary_excluded_count"] = state[
                "overlap_invalid_boundary_excluded"
            ]
            diagnostic["rationale_overlap_f1_mean"] = (
                None
                if not state["overlap_count"]
                else state["overlap_sum"] / state["overlap_count"]
            )
        diagnostic_rows.append(diagnostic)

    metric_rows = _cell_metric_rows(states, smoke=args.smoke, study=study)
    comparison_rows = _comparison_rows(states, smoke=args.smoke, study=study)
    replay_rows: list[dict[str, Any]] = []
    from_getgo = (
        []
        if args.smoke or study != LEGACY_STUDY_ID
        else _from_getgo_assessment(metric_rows, diagnostic_rows, comparison_rows)
    )
    compact_provenance = {
        "source": provenance,
        "assets": assets,
        "data": data_manifest,
        "generation": {
            "%s/%s" % key: value for key, value in sorted(manifests.items())
        },
        "generation_completion": {
            "%s/%s" % key: json.loads(
                (
                    generation_root
                    / key[0]
                    / key[1]
                    / "completion.json"
                ).read_text(encoding="utf-8")
            )
            for key in sorted(manifests)
        },
    }
    compact_provenance_bytes = canonical_json_bytes(compact_provenance)
    config = {
        "protocol": (
            "opd-qwen3-icl-pass1-report-v1"
            if study == QWEN3_STUDY_ID
            else "opd-softgrpo-native-soft-icl-report-v2-math-aime-matched"
        ),
        "study_id": study,
        "study_protocol": profile.protocol,
        "source_provenance": provenance,
        "asset_manifest_sha256": assets["content_sha256"],
        "data_manifest_sha256": data_manifest["content_sha256"],
        "generation_manifest_sha256": {
            "%s/%s" % key: hashlib.sha256(canonical_json_bytes(value)).hexdigest()
            for key, value in manifests.items()
        },
        "compact_provenance_sha256": hashlib.sha256(
            compact_provenance_bytes
        ).hexdigest(),
        "smoke": bool(args.smoke),
    }
    report = {
        "config": config,
        "metrics": metric_rows,
        "diagnostics": diagnostic_rows,
        "comparisons": comparison_rows,
        "replay": replay_rows,
        "icl_from_getgo_assessment": from_getgo,
        "invalid_icl_cells": [
            "%s/%s/%s/%s"
            % (row["model_label"], row["inference_mode"], row["benchmark"], row["condition"])
            for row in diagnostic_rows
            if row["inference_mode"] == "native_soft"
            and row.get("boundary_gate_applied")
            and not row["boundary_gate_valid"]
        ],
        "notes": {
            "pass_at_1": (
                "single sampled outcome per example"
                if study == QWEN3_STUDY_ID
                else "canonical c/n estimator over common samples"
            ),
            "native_soft_scoring": "boundary-invalid samples are incorrect in every primary grader",
            "native_soft_cell_gate": (
                "applied only to registered boundary-gate benchmarks; invalid when "
                "capped-or-all-soft rate exceeds 5%. AIME is diagnostic only. "
                "Boundary-invalid samples are scored incorrect."
            ),
            "aime2024": "30-example intervals are exploratory and imprecise",
            "condition_control": (
                "matched conditions are compared with no_demo; without shuffled "
                "controls, effects cannot be attributed specifically to gold relevance"
            ),
            "inference": (
                "paired confidence intervals are pointwise and unadjusted; the "
                "assessment table is descriptive and defines no omnibus success rule"
            ),
            "replay": "deferred from this core ICL evaluation",
            "seed": "single-seed-11 exploratory evaluation",
        },
    }
    if 8 in profile.pass_ks:
        report["notes"]["pass_at_8"] = (
            "probability at least one of eight succeeds; omitted for smoke n=2"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "report.json"
    scored_path = output_root / "scored.jsonl"
    provenance_path = output_root / "provenance.json"
    _atomic_json(report_path, report)
    from .icl_runtime import _atomic_bytes

    _atomic_bytes(provenance_path, compact_provenance_bytes)
    _atomic_bytes(
        scored_path,
        b"".join(canonical_json_bytes(row) for row in scored_rows),
    )
    _write_csv(output_root / "metrics.csv", metric_rows)
    _write_csv(output_root / "comparisons.csv", comparison_rows)
    _write_csv(output_root / "replay.csv", replay_rows)
    run_id = stable_wandb_run_id(config, prefix="icl-report")
    wandb_run = init_online_wandb(
        run_id=run_id,
        config=config,
        job_type="icl-aggregate",
    )
    try:
        wandb_run.log(
            _flatten_wandb(
                metric_rows,
                diagnostic_rows,
                comparison_rows,
                replay_rows,
            ),
            step=0,
        )
        for path in (
            report_path,
            output_root / "metrics.csv",
            output_root / "comparisons.csv",
            output_root / "replay.csv",
            provenance_path,
        ):
            if path.exists():
                wandb_run.log_artifact(
                    str(path),
                    name="%s-%s" % (run_id, path.stem),
                    type="icl-report",
                )
        wandb_run.summary["evaluation/completed"] = True
        wandb_run.summary["evaluation/report_path"] = str(report_path)
    finally:
        wandb_run.finish()


def _artifact_inventory(root: Path, paths: Sequence[Path]) -> dict[str, Any]:
    """Hash every regular input file consumed by a benchmark report."""

    resolved_root = root.expanduser().resolve()
    entries = []
    seen = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        try:
            relative = path.relative_to(resolved_root).as_posix()
        except ValueError as error:
            raise RuntimeError("throughput artifact lies outside its evidence root") from error
        if relative in seen:
            continue
        seen.add(relative)
        if not path.is_file() or raw_path.is_symlink():
            raise RuntimeError("throughput artifact is not a regular file: %s" % raw_path)
        entries.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    entries.sort(key=lambda value: value["path"])
    if not entries:
        raise RuntimeError("throughput evidence inventory is empty")
    payload = {
        "protocol": "opd-icl-throughput-input-inventory-v1",
        "files": entries,
    }
    payload["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def _normalized_throughput_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only fields already proven to be deterministic DP topology values."""

    normalized = json.loads(json.dumps(dict(manifest)))
    parallelism = normalized.get("parallelism")
    if isinstance(parallelism, dict):
        parallelism.pop("data_parallel_size", None)
        parallelism.pop("world_size", None)
    for name in (
        "request_queue_size",
        "max_running_requests_aggregate",
        "wandb_run_id",
    ):
        normalized.pop(name, None)
    warmup = normalized.get("warmup")
    if isinstance(warmup, dict):
        warmup.pop("request_count", None)
    return normalized


def _compare_throughput_manifests(
    dp1: Mapping[str, Any], dp2: Mapping[str, Any]
) -> dict[str, Any]:
    """Require DP1/DP2 to differ only in individually validated topology fields."""

    first = _normalized_throughput_manifest(dp1)
    second = _normalized_throughput_manifest(dp2)
    if first != second:
        changed = sorted(
            name
            for name in set(first) | set(second)
            if first.get(name) != second.get(name)
        )
        raise RuntimeError(
            "DP1/DP2 generation manifests differ outside topology-derived fields: %s"
            % ", ".join(changed)
        )
    digest = hashlib.sha256(canonical_json_bytes(first)).hexdigest()
    return {
        "non_topology_config_exact": True,
        "normalized_manifest_sha256": digest,
    }


def _assert_recomputed_number(
    observed: Any, expected: float, *, label: str
) -> None:
    if isinstance(observed, bool):
        raise ValueError("throughput %s is not numeric" % label)
    try:
        value = float(observed)
    except (TypeError, ValueError) as error:
        raise ValueError("throughput %s is not numeric" % label) from error
    if not math.isfinite(value) or not math.isclose(
        value, float(expected), rel_tol=1e-12, abs_tol=1e-9
    ):
        raise ValueError("throughput %s differs from authenticated timing rows" % label)


def _validate_throughput_resource_evidence(
    system: Any, *, data_parallel_size: int
) -> dict[str, Any]:
    """Reject allocation evidence without usable assigned-GPU/host samples."""

    if not isinstance(system, Mapping):
        raise ValueError("throughput resource evidence is missing")
    if system.get("host_metrics_available") is not True:
        raise ValueError("throughput host telemetry is unavailable")
    if system.get("gpu_metrics_available") is not True:
        raise ValueError("throughput assigned-GPU telemetry is unavailable")
    if system.get("expected_gpu_count") != data_parallel_size:
        raise ValueError("throughput telemetry expected GPU count differs from topology")
    selectors = system.get("gpu_selectors")
    if (
        not isinstance(selectors, (list, tuple))
        or len(selectors) != data_parallel_size
        or len(set(selectors)) != data_parallel_size
        or not system.get("gpu_selection_source")
    ):
        raise ValueError("throughput telemetry lacks an exact GPU assignment")
    for name in ("sample_count", "host_sample_count", "gpu_sample_count"):
        value = system.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("throughput telemetry %s must be positive" % name)
    if (
        system["host_sample_count"] > system["sample_count"]
        or system["gpu_sample_count"] > system["sample_count"]
    ):
        raise ValueError("throughput telemetry sample counts are inconsistent")
    for name in (
        "peak_hbm_gib_aggregate",
        "peak_host_ram_gib",
        "cpu_utilization_mean",
        "cpu_utilization_peak",
        "gpu_utilization_mean",
    ):
        value = system.get(name)
        if isinstance(value, bool):
            raise ValueError("throughput telemetry %s is invalid" % name)
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("throughput telemetry %s is invalid" % name) from error
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError("throughput telemetry %s is invalid" % name)
    for name in ("cpu_utilization_mean", "cpu_utilization_peak", "gpu_utilization_mean"):
        if float(system[name]) > 100.0:
            raise ValueError("throughput telemetry %s exceeds 100 percent" % name)
    per_gpu_hbm = system.get("peak_hbm_gib_per_gpu")
    per_gpu_util = system.get("gpu_utilization_mean_per_gpu")
    if (
        not isinstance(per_gpu_hbm, Mapping)
        or not isinstance(per_gpu_util, Mapping)
        or set(per_gpu_hbm) != set(per_gpu_util)
        or len(per_gpu_hbm) != data_parallel_size
    ):
        raise ValueError("throughput per-GPU telemetry differs from assigned topology")
    for gpu_id in per_gpu_hbm:
        for name, raw, maximum in (
            ("HBM", per_gpu_hbm[gpu_id], None),
            ("utilization", per_gpu_util[gpu_id], 100.0),
        ):
            if isinstance(raw, bool):
                raise ValueError("throughput per-GPU %s is invalid" % name)
            try:
                value = float(raw)
            except (TypeError, ValueError) as error:
                raise ValueError("throughput per-GPU %s is invalid" % name) from error
            if (
                not math.isfinite(value)
                or value < 0.0
                or (maximum is not None and value > maximum)
            ):
                raise ValueError("throughput per-GPU %s is invalid" % name)
    aggregate_hbm = float(system["peak_hbm_gib_aggregate"])
    per_gpu_hbm_values = [float(value) for value in per_gpu_hbm.values()]
    if not max(per_gpu_hbm_values) <= aggregate_hbm <= sum(per_gpu_hbm_values):
        raise ValueError("throughput aggregate HBM differs from per-GPU samples")
    return {
        "validated": True,
        "expected_gpu_count": data_parallel_size,
        "host_sample_count": system["host_sample_count"],
        "gpu_sample_count": system["gpu_sample_count"],
        "gpu_ids": sorted(str(value) for value in per_gpu_hbm),
    }


def _validate_throughput_timing_evidence(
    throughput: Mapping[str, Any],
    artifacts: Mapping[tuple[str, str, str], tuple[Any, Any]],
    artifact_locations: Mapping[tuple[str, str, str], tuple[int, int]],
    *,
    queue_size: int,
    data_parallel_size: int,
) -> tuple[tuple[RequestTiming, ...], dict[str, Any]]:
    """Recompute report-critical timing and response summaries from chunk rows."""

    timing_rows = throughput.get("rows")
    if not isinstance(timing_rows, list) or len(timing_rows) != len(artifacts):
        raise ValueError("throughput benchmark timing rows are incomplete")
    if set(artifact_locations) != set(artifacts):
        raise RuntimeError("throughput artifact locations are incomplete")
    sessions = {
        row.get("timing_session_id")
        for row in timing_rows
        if isinstance(row, Mapping)
    }
    if (
        None in sessions
        or not sessions
        or any(not isinstance(value, str) or not value for value in sessions)
    ):
        raise ValueError("throughput evidence contains an invalid timing session")
    if type(throughput.get("resumed")) is not bool:
        raise ValueError("throughput resumed state must be explicit")

    rows_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    request_indices_by_session: dict[str, list[int]] = defaultdict(list)
    for row in timing_rows:
        if not isinstance(row, Mapping):
            raise ValueError("throughput timing row must be a mapping")
        key = (
            str(row.get("condition", "")),
            str(row.get("benchmark", "")),
            str(row.get("example_id", "")),
        )
        if key in rows_by_key:
            raise ValueError("throughput timing rows contain a duplicate request")
        rows_by_key[key] = row
        request_index = row.get("request_index")
        if isinstance(request_index, bool) or not isinstance(request_index, int):
            raise ValueError("throughput request_index must be an integer")
        request_indices_by_session[str(row["timing_session_id"])].append(
            request_index
        )
    if set(rows_by_key) != set(artifacts):
        raise ValueError("throughput timing identities differ from authenticated chunks")
    for session_id, request_indices in request_indices_by_session.items():
        if (
            any(value < 0 for value in request_indices)
            or len(set(request_indices)) != len(request_indices)
        ):
            raise ValueError(
                "throughput request indices repeat within timing session %s"
                % session_id
            )

    timings = []
    for key in sorted(artifacts):
        record, _ = artifacts[key]
        row = rows_by_key[key]
        expected_chunk, expected_row = artifact_locations[key]
        exact_values = {
            "sample_index": 0,
            "chunk_index": expected_chunk,
            "chunk_row": expected_row,
            "request_seed": record.request_seed,
            "response_tokens": record.response_token_count,
            "capped": record.capped,
            "all_soft": record.all_soft,
            "soft_to_hard": record.soft_to_hard,
        }
        for name, expected in exact_values.items():
            observed = row.get(name)
            if type(expected) is bool:
                matches = type(observed) is bool and observed is expected
            else:
                matches = (
                    not isinstance(observed, bool)
                    and isinstance(observed, int)
                    and observed == expected
                )
            if not matches:
                raise ValueError(
                    "throughput timing %s differs from authenticated chunk record"
                    % name
                )
        submitted = row.get("submitted_at")
        completed = row.get("completed_at")
        if isinstance(submitted, bool) or isinstance(completed, bool):
            raise ValueError("throughput request timestamps must be numeric")
        try:
            submitted_value = float(submitted)
            completed_value = float(completed)
        except (TypeError, ValueError) as error:
            raise ValueError("throughput request timestamps must be numeric") from error
        if (
            not math.isfinite(submitted_value)
            or not math.isfinite(completed_value)
            or completed_value <= submitted_value
        ):
            raise ValueError("throughput request timestamps are invalid")
        elapsed = completed_value - submitted_value
        _assert_recomputed_number(
            row.get("latency_seconds"), elapsed, label="row latency_seconds"
        )
        timings.append(
            RequestTiming(
                condition=key[0],
                benchmark=key[1],
                example_id=key[2],
                request_seed=record.request_seed,
                elapsed_seconds=elapsed,
                generated_tokens=record.response_token_count,
            )
        )

    response_tokens = sum(record.response_token_count for record, _ in artifacts.values())
    response_lengths = [record.response_token_count for record, _ in artifacts.values()]
    latencies = [timing.elapsed_seconds for timing in timings]
    queue_metrics = _async_queue_metrics(
        timing_rows,
        queue_size=queue_size,
        data_parallel_size=data_parallel_size,
    )
    generation_seconds = float(queue_metrics["active_generation_seconds"])
    rows_by_session: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in timing_rows:
        rows_by_session[str(row["timing_session_id"])].append(row)
    final_queue_drain_seconds = sum(
        max(float(row["completed_at"]) for row in values)
        - max(float(row["submitted_at"]) for row in values)
        for values in rows_by_session.values()
    )
    recomputed = {
        "request_count": len(timing_rows),
        "response_tokens": response_tokens,
        "response_length_mean": response_tokens / len(timing_rows),
        "response_length_tokens": {
            "p50": _percentile(response_lengths, 0.50),
            "p95": _percentile(response_lengths, 0.95),
            "max": max(response_lengths),
        },
        "cap_rate": sum(record.capped for record, _ in artifacts.values())
        / len(artifacts),
        "soft_to_hard_rate": sum(
            record.soft_to_hard for record, _ in artifacts.values()
        )
        / len(artifacts),
        "generation_seconds": generation_seconds,
        "final_queue_drain_seconds": final_queue_drain_seconds,
        "tokens_per_second": response_tokens / generation_seconds,
        "requests_per_hour": len(timing_rows) * 3600.0 / generation_seconds,
        "latency_seconds": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies),
        },
        "queue_occupancy": queue_metrics,
    }
    for name in (
        "response_length_mean",
        "cap_rate",
        "soft_to_hard_rate",
        "generation_seconds",
        "final_queue_drain_seconds",
        "tokens_per_second",
        "requests_per_hour",
    ):
        _assert_recomputed_number(throughput.get(name), recomputed[name], label=name)
    for section in ("response_length_tokens", "latency_seconds"):
        observed_section = throughput.get(section)
        if not isinstance(observed_section, Mapping):
            raise ValueError("throughput %s summary is missing" % section)
        for name, expected in recomputed[section].items():
            _assert_recomputed_number(
                observed_section.get(name), expected, label="%s.%s" % (section, name)
            )
    observed_queue = throughput.get("queue_occupancy")
    if not isinstance(observed_queue, Mapping):
        raise ValueError("throughput queue_occupancy summary is missing")
    for name in (
        "maximum",
        "capacity",
        "mean",
        "mean_fraction",
        "timing_session_count",
        "active_generation_seconds",
    ):
        _assert_recomputed_number(
            observed_queue.get(name),
            recomputed["queue_occupancy"][name],
            label="queue_occupancy.%s" % name,
        )
    for name in ("request_count", "response_tokens"):
        observed = throughput.get(name)
        if isinstance(observed, bool) or not isinstance(observed, int):
            raise ValueError("throughput %s must be an integer" % name)
        if observed != recomputed[name]:
            raise ValueError("throughput %s differs from authenticated rows" % name)
    return tuple(timings), recomputed


def _load_throughput_observation(
    generation_root: Path,
    *,
    model_label: str,
    inference_mode: str,
    data_parallel_size: int,
    plan: Any,
    examples: Sequence[Any],
    shuffled_pairs: Mapping[str, Mapping[str, str]],
    mechanism_ids: Mapping[str, Sequence[str]],
    assets: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
) -> tuple[
    BenchmarkObservation,
    dict[tuple[str, str, str], tuple[Any, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    run_root = generation_root / model_label / inference_mode
    manifest_path = run_root / "generation_manifest.json"
    completion_path = run_root / "completion.json"
    throughput_path = run_root / "throughput.json"
    for path in (manifest_path, completion_path, throughput_path):
        if not path.is_file():
            raise ValueError("throughput benchmark artifact is missing: %s" % path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    throughput = json.loads(throughput_path.read_text(encoding="utf-8"))
    _validate_generation_binding(
        manifest,
        assets=assets,
        data_manifest=data_manifest,
        model_label=model_label,
        study=QWEN3_STUDY_ID,
    )
    if manifest.get("source_provenance") != source_provenance():
        raise ValueError("throughput benchmark source provenance differs")
    if not manifest.get("throughput_benchmark") or manifest.get("smoke"):
        raise ValueError("allocation evidence is not a dedicated throughput benchmark")
    if manifest.get("mode") != inference_mode:
        raise ValueError("throughput benchmark mode differs from its path")
    expected_queue = {1: 32, 2: 64}[data_parallel_size]
    expected_aggregate_running = {1: 16, 2: 32}[data_parallel_size]
    expected_parallelism = {
        "tensor_parallel_size": 1,
        "data_parallel_size": data_parallel_size,
        "world_size": data_parallel_size,
        "load_balance_method": "round_robin",
    }
    if manifest.get("parallelism") != expected_parallelism:
        raise ValueError("throughput benchmark topology differs from its DP label")
    if (
        manifest.get("chunk_size") != 66
        or float(manifest.get("gpu_memory_utilization", -1.0)) != 0.8
        or manifest.get("request_queue_size") != expected_queue
        or manifest.get("max_running_requests_per_replica") != 16
        or manifest.get("max_running_requests_aggregate")
        != expected_aggregate_running
    ):
        raise ValueError("throughput benchmark queue/running-request contract changed")
    if manifest.get("warmup") != _throughput_warmup_contract(
        data_parallel_size
    ):
        raise ValueError("throughput benchmark warmup contract changed")
    if manifest.get("throughput_plan") != plan.to_dict():
        raise ValueError("throughput benchmark selected a different request plan")
    if manifest.get("throughput_plan_sha256") != plan.content_sha256:
        raise ValueError("throughput benchmark request-plan hash differs")
    if completion.get("generation_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("throughput completion does not authenticate its manifest")
    if completion.get("throughput_sha256") != sha256_file(throughput_path):
        raise ValueError("throughput completion does not authenticate timing evidence")
    if (
        throughput.get("protocol") != "opd-icl-async-throughput-v1"
        or throughput.get("study_id") != QWEN3_STUDY_ID
        or throughput.get("model_label") != model_label
        or throughput.get("mode") != inference_mode
        or throughput.get("request_count") != 66
        or throughput.get("data_parallel_size") != data_parallel_size
        or throughput.get("queue_size") != expected_queue
        or throughput.get("max_running_requests") != 16
        or throughput.get("max_running_requests_per_replica") != 16
        or throughput.get("max_running_requests_aggregate")
        != expected_aggregate_running
        or throughput.get("eligible_for_allocation") is not True
    ):
        raise ValueError("throughput timing evidence is incomplete or invalid")
    resource_reconciliation = _validate_throughput_resource_evidence(
        throughput.get("system"), data_parallel_size=data_parallel_size
    )

    selected_ids = _throughput_selected_ids(plan)
    cells = [ICLMatrixCell(**value) for value in manifest["cells"]]
    expected_cells = {
        (condition, benchmark): 20 if benchmark == "math500" else 2
        for condition in CORE_CONDITIONS
        for benchmark in STUDY_BENCHMARKS
    }
    if {
        (cell.condition, cell.benchmark): cell.example_count for cell in cells
    } != expected_cells:
        raise ValueError("throughput benchmark cell inventory is not exactly 66 requests")
    store = AtomicChunkStore(generation_root)
    artifacts: dict[tuple[str, str, str], tuple[Any, Any]] = {}
    artifact_locations: dict[tuple[str, str, str], tuple[int, int]] = {}
    input_paths = [manifest_path, completion_path, throughput_path]
    for cell in cells:
        _, prompts = _cell_prompts(
            cell,
            examples=examples,
            shuffled_pairs=shuffled_pairs,
            mechanism_ids=mechanism_ids,
            selected_ids_override=selected_ids[(cell.benchmark, cell.condition)],
        )
        for chunk_index, start in enumerate(
            range(0, len(prompts), int(manifest["chunk_size"]))
        ):
            chunk_prompts = prompts[start : start + int(manifest["chunk_size"])]
            key = "%s/%s/%s/%s/sample_00/chunk_%05d" % (
                model_label,
                inference_mode,
                cell.benchmark,
                cell.condition,
                chunk_index,
            )
            identity = _generation_chunk_identity(
                manifest,
                model_label=model_label,
                inference_mode=inference_mode,
                benchmark=cell.benchmark,
                condition=cell.condition,
                sample_index=0,
                chunk_index=chunk_index,
                example_ids=[prompt.example_id for prompt in chunk_prompts],
            )
            input_paths.extend(store.paths(key))
            records, arrays = store.load(key, expected_identity=identity)
            _validate_completion_records(
                records,
                model_label=model_label,
                inference_mode=inference_mode,
                benchmark=cell.benchmark,
                condition=cell.condition,
                sample_index=0,
                example_ids=[prompt.example_id for prompt in chunk_prompts],
            )
            for row, record in enumerate(records):
                key3 = (record.condition, record.benchmark, record.example_id)
                if key3 in artifacts:
                    raise RuntimeError("throughput benchmark contains a duplicate request")
                artifacts[key3] = (record, unpack_trajectory(arrays, row))
                artifact_locations[key3] = (chunk_index, row)
    if len(artifacts) != 66:
        raise RuntimeError("throughput benchmark does not contain exactly 66 requests")

    timings, timing_reconciliation = _validate_throughput_timing_evidence(
        throughput,
        artifacts,
        artifact_locations,
        queue_size=expected_queue,
        data_parallel_size=data_parallel_size,
    )
    cleanup = throughput.get("cleanup")
    if not isinstance(cleanup, Mapping):
        raise ValueError("throughput finalization timing evidence is missing")
    finalization_values = []
    for name in ("wandb_finish_seconds", "engine_shutdown_seconds"):
        raw_value = cleanup.get(name)
        if isinstance(raw_value, bool):
            raise ValueError("throughput finalization timing is invalid")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError("throughput finalization timing is invalid") from error
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("throughput finalization timing is invalid")
        finalization_values.append(value)
    finalization_seconds = sum(finalization_values)
    representative_overhead = {
        "protocol": "opd-icl-representative-current-invocation-overhead-v1",
        "semantics": (
            "engine load, warmup, and finalization are measured on the completing "
            "invocation and applied once per inference mode as representative overhead"
        ),
        "engine_load_seconds": throughput.get("engine_load_seconds"),
        "warmup_seconds": throughput.get("warmup_seconds"),
        "finalization_seconds": finalization_seconds,
    }
    for name in ("engine_load_seconds", "warmup_seconds"):
        raw_value = representative_overhead[name]
        if isinstance(raw_value, bool):
            raise ValueError("throughput representative overhead timing is invalid")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "throughput representative overhead timing is invalid"
            ) from error
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("throughput representative overhead timing is invalid")
        representative_overhead[name] = value
    observation = BenchmarkObservation(
        model_label=model_label,
        inference_mode=inference_mode,
        data_parallel_size=data_parallel_size,
        plan_sha256=plan.content_sha256,
        timings=timings,
        generation_wall_seconds=float(throughput["generation_seconds"]),
        engine_load_seconds=representative_overhead["engine_load_seconds"],
        finalization_seconds=finalization_seconds,
    )
    inventory = _artifact_inventory(generation_root, input_paths)
    evidence = {
        "generation_manifest": manifest,
        "generation_manifest_sha256": sha256_file(manifest_path),
        "input_artifacts": inventory,
        "timing_reconciliation": timing_reconciliation,
        "resource_reconciliation": resource_reconciliation,
        "representative_overhead": representative_overhead,
    }
    return observation, artifacts, throughput, evidence


def _compare_throughput_artifacts(
    dp1: Mapping[tuple[str, str, str], tuple[Any, Any]],
    dp2: Mapping[tuple[str, str, str], tuple[Any, Any]],
    *,
    probability_tolerance: float = 5e-3,
) -> dict[str, Any]:
    if set(dp1) != set(dp2):
        raise RuntimeError("DP1 and DP2 completed different request identities")
    perturbed_error = 0.0
    probability_error = 0.0
    latent_slots = 0
    exact_fields = (
        "response",
        "finish_reason",
        "response_token_count",
        "capped",
        "latent_token_count",
        "hard_token_count",
        "close_tag",
        "soft_to_hard",
        "all_soft",
        "boxed_answer",
        "boundary_valid",
    )
    for key in sorted(dp1):
        record1, trajectory1 = dp1[key]
        record2, trajectory2 = dp2[key]
        if any(getattr(record1, name) != getattr(record2, name) for name in exact_fields):
            raise RuntimeError("DP1/DP2 response metadata differs for %r" % (key,))
        if trajectory1.response_token_ids != trajectory2.response_token_ids:
            raise RuntimeError("DP1/DP2 response token IDs differ for %r" % (key,))
        if trajectory1.latent_support_ids != trajectory2.latent_support_ids:
            raise RuntimeError("DP1/DP2 top-five supports differ for %r" % (key,))
        if trajectory1.latent_gumbel_noise != trajectory2.latent_gumbel_noise:
            raise RuntimeError("DP1/DP2 Gumbel draws differ for %r" % (key,))
        logits1 = np.asarray(trajectory1.latent_perturbed_logits, dtype=np.float64)
        logits2 = np.asarray(trajectory2.latent_perturbed_logits, dtype=np.float64)
        if logits1.shape != logits2.shape:
            raise RuntimeError("DP1/DP2 perturbed-logit shapes differ for %r" % (key,))
        if logits1.size:
            current_error = float(np.max(np.abs(logits1 - logits2)))
            perturbed_error = max(perturbed_error, current_error)
            if current_error > probability_tolerance:
                raise RuntimeError(
                    "DP1/DP2 perturbed logits exceed replay tolerance for %r" % (key,)
                )
            scaled1 = logits1 / 0.1
            scaled2 = logits2 / 0.1
            scaled1 -= scaled1.max(axis=-1, keepdims=True)
            scaled2 -= scaled2.max(axis=-1, keepdims=True)
            probabilities1 = np.exp(scaled1)
            probabilities2 = np.exp(scaled2)
            probabilities1 /= probabilities1.sum(axis=-1, keepdims=True)
            probabilities2 /= probabilities2.sum(axis=-1, keepdims=True)
            current_probability_error = float(
                np.max(np.abs(probabilities1 - probabilities2))
            )
            probability_error = max(probability_error, current_probability_error)
            if current_probability_error > probability_tolerance:
                raise RuntimeError(
                    "DP1/DP2 reconstructed probabilities exceed replay tolerance for %r"
                    % (key,)
                )
            latent_slots += logits1.shape[0]
    return {
        "request_count": len(dp1),
        "latent_slot_count": latent_slots,
        "response_and_finish_exact": True,
        "response_token_ids_exact": True,
        "top_five_support_exact": True,
        "gumbel_noise_exact": True,
        "perturbed_logits_abs_error_max": perturbed_error,
        "reconstructed_probability_abs_error_max": probability_error,
        "tolerance": probability_tolerance,
    }


def run_benchmark_report(args: argparse.Namespace) -> None:
    study = getattr(args, "study", QWEN3_STUDY_ID)
    if study != QWEN3_STUDY_ID:
        raise ValueError("throughput allocation is registered only for Qwen3")
    assets = _verify_assets_for_study(args.root, study)
    asset_root = Path(args.root).expanduser().resolve()
    examples, shuffled_pairs, mechanism_ids, data_manifest = load_icl_dataset(
        asset_root / "data"
    )
    roots = {
        1: Path(args.dp1_generation_dir).expanduser().resolve(),
        2: Path(args.dp2_generation_dir).expanduser().resolve(),
    }
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("throughput reporting requires transformers") from error

    report_models = {}
    recommendations = {}
    input_artifact_sha256s: dict[str, Any] = {}
    resource_table = {
        1: {"gpus": 1, "cpus": 16, "memory": "128G", "queue_size": 32},
        2: {"gpus": 2, "cpus": 32, "memory": "256G", "queue_size": 64},
        8: {
            "gpus": 8,
            "cpus": 112,
            "memory": "1904000M",
            "queue_size": 256,
        },
    }
    for model_label in get_study_profile(study).model_labels:
        tokenizer = AutoTokenizer.from_pretrained(
            str(asset_root / "models" / model_label), local_files_only=True
        )
        plan = _build_rendered_throughput_plan(
            tokenizer,
            examples=examples,
            shuffled_pairs=shuffled_pairs,
            study=study,
        )
        observations_by_dp = {}
        comparisons = {}
        telemetry = {}
        evidence_by_dp = {}
        input_artifact_sha256s[model_label] = {}
        for dp, generation_root in roots.items():
            observations = []
            artifacts_by_mode = {}
            telemetry[str(dp)] = {}
            evidence_by_dp[str(dp)] = {}
            input_artifact_sha256s[model_label][str(dp)] = {}
            for mode in get_study_profile(study).inference_modes:
                observation, artifacts, throughput, evidence = _load_throughput_observation(
                    generation_root,
                    model_label=model_label,
                    inference_mode=mode,
                    data_parallel_size=dp,
                    plan=plan,
                    examples=examples,
                    shuffled_pairs=shuffled_pairs,
                    mechanism_ids=mechanism_ids,
                    assets=assets,
                    data_manifest=data_manifest,
                )
                observations.append(observation)
                artifacts_by_mode[mode] = artifacts
                telemetry[str(dp)][mode] = {
                    key: value for key, value in throughput.items() if key != "rows"
                }
                evidence_by_dp[str(dp)][mode] = evidence
                input_artifact_sha256s[model_label][str(dp)][mode] = {
                    entry["path"]: entry["sha256"]
                    for entry in evidence["input_artifacts"]["files"]
                }
            observations_by_dp[dp] = tuple(observations)
            comparisons[str(dp)] = artifacts_by_mode
        equivalence = {}
        for mode in get_study_profile(study).inference_modes:
            equivalence[mode] = _compare_throughput_artifacts(
                comparisons["1"][mode], comparisons["2"][mode]
            )
            equivalence[mode].update(
                _compare_throughput_manifests(
                    evidence_by_dp["1"][mode]["generation_manifest"],
                    evidence_by_dp["2"][mode]["generation_manifest"],
                )
            )
        decision = choose_smallest_data_parallel_size(plan, observations_by_dp)
        selected = decision.selected_data_parallel_size
        if selected not in resource_table:
            raise RuntimeError("throughput calibration did not produce a resource choice")
        recommendations[model_label] = {
            "data_parallel_size": selected,
            **resource_table[selected],
            "selection_rule": (
                "smallest measured DP topology whose stratified-bootstrap 95% "
                "upper runtime is at most 18 hours; otherwise unextrapolated DP8"
            ),
            "runtime_limit_hours": 18.0,
        }
        report_models[model_label] = {
            "throughput_plan_sha256": plan.content_sha256,
            "throughput_plan": plan.to_dict(),
            "dp1_dp2_equivalence": equivalence,
            "decision": decision.to_dict(),
            "telemetry": telemetry,
            "input_evidence": {
                dp: {
                    mode: {
                        key: value
                        for key, value in evidence.items()
                        if key != "generation_manifest"
                    }
                    for mode, evidence in by_mode.items()
                }
                for dp, by_mode in evidence_by_dp.items()
            },
        }
    input_artifact_set_sha256 = hashlib.sha256(
        canonical_json_bytes(input_artifact_sha256s)
    ).hexdigest()
    config = {
        "protocol": "opd-qwen3-icl-throughput-report-v1",
        "study_id": study,
        "source_provenance": source_provenance(),
        "asset_manifest_sha256": assets["content_sha256"],
        "data_manifest_sha256": data_manifest["content_sha256"],
        "input_artifact_sha256s": input_artifact_sha256s,
        "input_artifact_set_sha256": input_artifact_set_sha256,
        "bootstrap_resamples": 10_000,
        "selection_runtime_upper_hours": 18.0,
        "production_allocation_hours": 24.0,
        "timing_session_treatment": (
            "sum active spans within each authenticated timing session; exclude "
            "idle time between resumable invocations"
        ),
        "overhead_treatment": (
            "use the completing invocation's engine-load and finalization timing "
            "once per inference mode as representative production overhead"
        ),
    }
    report = {"config": config, "models": report_models}
    recommendation_payload = {
        "protocol": "opd-qwen3-icl-adaptive-allocation-v1",
        "study_id": study,
        "source_provenance": config["source_provenance"],
        "throughput_report_sha256": hashlib.sha256(
            canonical_json_bytes(report)
        ).hexdigest(),
        "recommendations": recommendations,
    }
    report_path = output_root / "throughput-report.json"
    recommendation_path = output_root / "allocation-recommendations.json"
    _atomic_json(report_path, report)
    _atomic_json(recommendation_path, recommendation_payload)
    run_id = stable_wandb_run_id(config, prefix="icl-throughput-report")
    wandb_run = init_online_wandb(
        run_id=run_id, config=config, job_type="icl-throughput-report"
    )
    try:
        metrics = {}
        for model_label, value in report_models.items():
            metrics["allocation/%s/data_parallel_size" % model_label] = (
                recommendations[model_label]["data_parallel_size"]
            )
            for dp, estimate in value["decision"]["estimates"].items():
                metrics[
                    "throughput/%s/dp%s/runtime_upper_hours" % (model_label, dp)
                ] = estimate["upper_hours"]
        wandb_run.log(metrics, step=0)
        for path in (report_path, recommendation_path):
            wandb_run.log_artifact(
                str(path),
                name="%s-%s" % (run_id, path.stem),
                type="icl-throughput-report",
            )
        wandb_run.summary["throughput/completed"] = True
        wandb_run.summary[
            "throughput/input_artifact_set_sha256"
        ] = input_artifact_set_sha256
    finally:
        wandb_run.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="stage pinned models and evaluation data")
    prepare.add_argument("--study", default=LEGACY_STUDY_ID)
    prepare.add_argument("--root", required=True)
    prepare.add_argument("--cache-dir", required=True)
    prepare.set_defaults(handler=run_prepare)

    generate = commands.add_parser("generate", help="run exact upstream generation")
    generate.add_argument("--study", default=LEGACY_STUDY_ID)
    generate.add_argument("--root", required=True)
    generate.add_argument("--model", required=True)
    generate.add_argument("--mode", required=True)
    generate.add_argument("--output-dir")
    generate.add_argument(
        "--benchmarks", nargs="+", default=["all"], choices=("all",) + STUDY_BENCHMARKS
    )
    generate.add_argument(
        "--conditions", nargs="+", default=["all"], choices=("all",) + CORE_CONDITIONS
    )
    generate.add_argument("--tensor-parallel-size", type=int, default=1)
    generate.add_argument("--data-parallel-size", type=int, default=1)
    generate.add_argument("--chunk-size", type=int, default=64)
    generate.add_argument("--max-running-requests", type=int, default=16)
    generate.add_argument(
        "--queue-size",
        type=int,
        help=(
            "maximum asynchronous requests kept in flight; defaults to twice "
            "the aggregate data-parallel running-request capacity"
        ),
    )
    generate.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    generate.add_argument("--smoke", action="store_true")
    generate.add_argument(
        "--throughput-benchmark",
        action="store_true",
        help="run the sealed 66-request Qwen3 DP calibration workload",
    )
    generate.set_defaults(handler=run_generate)

    benchmark_report = commands.add_parser(
        "benchmark-report",
        help="authenticate DP1/DP2 calibration and choose production resources",
    )
    benchmark_report.add_argument("--study", default=QWEN3_STUDY_ID)
    benchmark_report.add_argument("--root", required=True)
    benchmark_report.add_argument("--dp1-generation-dir", required=True)
    benchmark_report.add_argument("--dp2-generation-dir", required=True)
    benchmark_report.add_argument("--output-dir", required=True)
    benchmark_report.set_defaults(handler=run_benchmark_report)

    replay = commands.add_parser("replay", help="replay no-demo native-soft trajectories")
    replay.add_argument("--study", default=LEGACY_STUDY_ID)
    replay.add_argument("--root", required=True)
    replay.add_argument("--model", required=True)
    replay.add_argument("--generation-dir")
    replay.add_argument("--output-dir")
    replay.add_argument(
        "--benchmarks", nargs="+", default=["all"], choices=("all",) + STUDY_BENCHMARKS
    )
    replay.add_argument("--hidden-chunk-size", type=int, default=32)
    replay.add_argument("--vocab-chunk-size", type=int, default=8192)
    replay.add_argument(
        "--attention-implementation",
        default="flash_attention_2",
        choices=("flash_attention_2", "sdpa", "eager"),
    )
    replay.set_defaults(handler=run_replay)

    aggregate = commands.add_parser("aggregate", help="grade and aggregate the complete matrix")
    aggregate.add_argument("--study", default=LEGACY_STUDY_ID)
    aggregate.add_argument("--root", required=True)
    aggregate.add_argument("--generation-dir")
    aggregate.add_argument("--output-dir")
    aggregate.add_argument("--smoke", action="store_true")
    aggregate.set_defaults(handler=run_aggregate)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    profile = get_study_profile(args.study)
    if args.command == "generate":
        validate_matrix_cell(
            args.model,
            args.mode,
            profile.core_conditions[0],
            profile.benchmarks[0],
            study=args.study,
        )
        if (
            args.tensor_parallel_size <= 0
            or args.data_parallel_size <= 0
            or args.chunk_size <= 0
            or args.max_running_requests <= 0
            or (args.queue_size is not None and args.queue_size <= 0)
        ):
            raise ValueError(
                "GPU, chunk, queue, and running-request counts must be positive"
            )
        if not 0.0 < args.gpu_memory_utilization < 1.0:
            raise ValueError("gpu-memory-utilization must be in (0, 1)")
        if "all" in args.benchmarks and args.benchmarks != ["all"]:
            raise ValueError("all cannot be combined with explicit benchmarks")
        if "all" in args.conditions and args.conditions != ["all"]:
            raise ValueError("all cannot be combined with explicit conditions")
        if args.smoke and args.throughput_benchmark:
            raise ValueError("smoke and throughput-benchmark are mutually exclusive")
        if args.throughput_benchmark:
            if args.study != QWEN3_STUDY_ID:
                raise ValueError("throughput-benchmark is registered only for Qwen3")
            if args.benchmarks != ["all"] or args.conditions != ["all"]:
                raise ValueError(
                    "throughput-benchmark requires all registered benchmarks and conditions"
                )
        allocated = os.environ.get("SLURM_GPUS_ON_NODE")
        world_size = args.tensor_parallel_size * args.data_parallel_size
        step_allocated = os.environ.get("OPD_EXPECTED_VISIBLE_GPUS", allocated)
        if step_allocated is not None and int(step_allocated) != world_size:
            raise RuntimeError(
                "tensor-parallel-size * data-parallel-size disagrees with "
                "SLURM_GPUS_ON_NODE/the explicit Slurm step GPU allocation"
            )
    if args.command == "replay" and (
        args.hidden_chunk_size <= 0 or args.vocab_chunk_size <= 0
    ):
        raise ValueError("replay chunk sizes must be positive")
    if args.command == "replay" and "all" in args.benchmarks and args.benchmarks != ["all"]:
        raise ValueError("all cannot be combined with explicit benchmarks")
    if args.command == "replay":
        if args.study != LEGACY_STUDY_ID:
            raise ValueError("the Qwen3 pass@1 study does not schedule latent replay")
        if args.model not in profile.model_labels:
            raise ValueError("unsupported model for replay study")
    args.handler(args)


if __name__ == "__main__":
    main()
