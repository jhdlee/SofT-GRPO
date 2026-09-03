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
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .icl import (
    BENCHMARKS,
    CORE_CONDITIONS,
    ICLMatrixCell,
    PROMPT_CONDITIONS,
    build_icl_matrix,
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
    source_provenance,
    stable_wandb_run_id,
    unpack_trajectory,
    validate_atomic_reasoning_tokens,
)
from .graders import (
    lm_eval_flexible_last_number_grade,
    math_verify_full_response_grade,
    released_last_boxed_grade,
)


def _atomic_json(path: Path, value: Any) -> None:
    from .icl_runtime import _atomic_bytes

    _atomic_bytes(path, canonical_json_bytes(value))


def _finish_generation_resources(
    *, wandb_run: Any, engine: Any | None, succeeded: bool
) -> None:
    """Flush W&B before SGLang terminates all child processes."""

    if not succeeded:
        wandb_run.summary["generation/completed"] = False
    try:
        wandb_run.finish()
    except Exception:
        # Preserve the generation/gate exception instead of replacing it with
        # a secondary W&B mailbox error during failure cleanup.
        if succeeded:
            raise
    finally:
        if engine is not None:
            engine.shutdown()


def _render(tokenizer: Any, user_content: str) -> str:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str) or not rendered.endswith("<think>\n"):
        raise RuntimeError(
            "checkpoint chat template must end in its native assistant <think>\\n opening"
        )
    # This no-specials pass only validates that the rendered template is
    # tokenizable. Generation itself intentionally follows the released
    # SGLang text API, whose TokenizerManager encodes the string again with
    # tokenizer defaults (and therefore adds the checkpoint's second BOS).
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    if not ids:
        raise RuntimeError("rendered chat prompt tokenized to empty")
    return rendered


def _selected_cells(args: argparse.Namespace) -> list[Any]:
    cells = [
        cell
        for cell in build_icl_matrix(smoke=args.smoke)
        if cell.model_label == args.model and cell.inference_mode == args.mode
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
) -> tuple[list[Any], list[Any]]:
    benchmark_examples = [
        example for example in examples if example.benchmark == cell.benchmark
    ]
    selected_ids: Sequence[str] | None = (
        None
        if cell.condition in CORE_CONDITIONS
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
    if cell.example_count < len(eligible):
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


def _validate_generation_binding(
    manifest: Mapping[str, Any],
    *,
    assets: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    model_label: str,
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
    if manifest.get("sampling") != SamplingSettings().__dict__:
        raise ValueError("generation sampling settings differ from the registered protocol")
    if manifest.get("prompt_tokenization") != UPSTREAM_TEXT_PROMPT_TOKENIZATION:
        raise ValueError("generation prompt tokenization differs from released behavior")


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
    manifest = prepare_icl_assets(Path(args.root), Path(args.cache_dir))
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
    assets = verify_icl_assets(args.root)
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
    _, think_end_id = validate_atomic_reasoning_tokens(tokenizer)
    cell_payloads = []
    all_rendered = []
    for cell in cells:
        selected_examples, prompts = _cell_prompts(
            cell,
            examples=examples,
            shuffled_pairs=shuffled_pairs,
            mechanism_ids=mechanism_ids,
        )
        rendered = [_render(tokenizer, prompt.user_content) for prompt in prompts]
        all_rendered.extend(rendered)
        cell_payloads.append((cell, selected_examples, prompts, rendered))

    settings = SamplingSettings()
    context_length = required_context_length(tokenizer, all_rendered, settings)
    model_config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    maximum = int(getattr(model_config, "max_position_embeddings", 0) or 0)
    if maximum and context_length > maximum:
        raise RuntimeError(
            "maximum tokenized prompt + %d completion tokens + one guard "
            "position requires %d positions, model supports %d"
            % (settings.max_new_tokens, context_length, maximum)
        )
    config = {
        "protocol": "opd-softgrpo-native-soft-icl-generation-v2-32768",
        "source_provenance": source_provenance(),
        "asset_manifest_sha256": assets["content_sha256"],
        "data_manifest_sha256": data_manifest["content_sha256"],
        "model_label": args.model,
        "model_tree_sha256": assets["models"][args.model]["tree_sha256"],
        "mode": args.mode,
        "smoke": bool(args.smoke),
        "num_gpus": args.num_gpus,
        "chunk_size": args.chunk_size,
        "max_running_requests": args.max_running_requests,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "context_length": context_length,
        "sampling": settings.__dict__,
        "prompt_tokenization": UPSTREAM_TEXT_PROMPT_TOKENIZATION,
        "cells": [cell.to_dict() for cell in cells],
    }
    run_id = stable_wandb_run_id(config, prefix="icl-generate")
    config["wandb_run_id"] = run_id
    manifest_path = output_root / args.model / args.mode / "generation_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != config:
            raise RuntimeError("generation invocation differs from its resume manifest")
    else:
        _atomic_json(manifest_path, config)

    wandb_run = init_online_wandb(
        run_id=run_id, config=config, job_type="icl-generation-smoke" if args.smoke else "icl-generation"
    )
    engine = None
    store = AtomicChunkStore(output_root)
    step = 0
    resumed = False
    cell_records: dict[tuple[str, str], list[CompletionRecord]] = defaultdict(list)
    succeeded = False
    try:
        engine = ReleasedSofTGRPOEngine(
            model_path=str(model_path),
            mode=args.mode,
            num_gpus=args.num_gpus,
            context_length=context_length,
            settings=settings,
            max_running_requests=args.max_running_requests,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        for cell, _, prompts, rendered in cell_payloads:
            for sample_index in range(cell.sample_count):
                for chunk_index, start in enumerate(range(0, len(prompts), args.chunk_size)):
                    stop = min(start + args.chunk_size, len(prompts))
                    key = "%s/%s/%s/%s/sample_%02d/chunk_%05d" % (
                        args.model,
                        args.mode,
                        cell.benchmark,
                        cell.condition,
                        sample_index,
                        chunk_index,
                    )
                    chunk_prompts = prompts[start:stop]
                    identity = {
                        "generation_manifest_sha256": hashlib.sha256(
                            canonical_json_bytes(config)
                        ).hexdigest(),
                        "model_label": args.model,
                        "mode": args.mode,
                        "benchmark": cell.benchmark,
                        "condition": cell.condition,
                        "sample_index": sample_index,
                        "chunk_index": chunk_index,
                        "example_ids": [prompt.example_id for prompt in chunk_prompts],
                    }
                    committed = store.resume_state(key, expected_identity=identity)
                    if committed is not None:
                        old_records, _ = store.load(key)
                        cell_records[(cell.benchmark, cell.condition)].extend(old_records)
                        resumed = True
                        step += 1
                        continue
                    seeds = [
                        request_seed(cell.benchmark, prompt.example_id, sample_index)
                        for prompt in chunk_prompts
                    ]
                    started = time.perf_counter()
                    outputs = engine.generate(rendered[start:stop], seeds)
                    records = []
                    trajectories = []
                    for row, (prompt, output) in enumerate(zip(chunk_prompts, outputs, strict=True)):
                        record, trajectory = parse_sglang_completion(
                            output=output,
                            model_label=args.model,
                            mode=args.mode,
                            benchmark=cell.benchmark,
                            condition=cell.condition,
                            example_id=prompt.example_id,
                            sample_index=sample_index,
                            replay_row=row,
                            settings=settings,
                            think_end_id=think_end_id,
                        )
                        records.append(record)
                        trajectories.append(trajectory)
                    store.commit(key, records, trajectories, identity=identity)
                    elapsed = max(time.perf_counter() - started, 1e-12)
                    step += 1
                    cell_records[(cell.benchmark, cell.condition)].extend(records)
                    metrics = generation_chunk_metrics(records, elapsed_seconds=elapsed)
                    metrics.update(
                        {
                            "generation/chunks_committed": step,
                            "generation/sample_index": sample_index,
                            "integrity/resumed": int(resumed),
                        }
                    )
                    wandb_run.log(metrics, step=step)

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
                if not gate["valid"]:
                    invalid_cells.append("%s/%s" % (benchmark, condition))
            if args.smoke and invalid_cells:
                raise RuntimeError(
                    "native-soft smoke exceeded the 5% capped/all-soft gate: %s"
                    % ", ".join(invalid_cells)
                )
            if args.smoke and demonstrated_boundary_count == 0:
                raise RuntimeError(
                    "native-soft smoke produced no real soft-to-hard categorical boxed answer"
                )
        wandb_run.summary[
            "generation/invalid_capped_or_all_soft_cells"
        ] = invalid_cells
        wandb_run.summary[
            "generation/demonstrated_boundary_count"
        ] = demonstrated_boundary_count
        wandb_run.summary["generation/completed"] = True
        wandb_run.summary["generation/output_root"] = str(output_root)
        _atomic_json(
            output_root / args.model / args.mode / "completion.json",
            {
                "generation_manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
                "chunks_committed": step,
                "invalid_capped_or_all_soft_cells": invalid_cells,
                "demonstrated_boundary_count": demonstrated_boundary_count,
            },
        )
        wandb_run.log_artifact(
            str(manifest_path),
            name="%s-generation-manifest" % run_id,
            type="icl-generation-manifest",
        )
        succeeded = True
    finally:
        # The bundled SGLang Engine.shutdown() terminates every child of this
        # process, which includes W&B's service process. Flush W&B first so a
        # successful generation is publishable and a generation error is not
        # obscured by HandleAbandonedError during cleanup.
        _finish_generation_resources(
            wandb_run=wandb_run, engine=engine, succeeded=succeeded
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
    assets = verify_icl_assets(args.root)
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
) -> tuple[dict[tuple[str, str], Mapping[str, Any]], list[Any]]:
    manifests = {}
    for model, mode in (
        ("starting", "native_soft"),
        ("starting", "hard_token"),
        ("softgrpo", "native_soft"),
    ):
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
        if bool(value.get("smoke")) != bool(smoke):
            raise ValueError("cannot mix smoke and production generation")
        manifests[(model, mode)] = value
    expected = list(build_icl_matrix(smoke=smoke))
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
) -> list[dict[str, Any]]:
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
            rows.append(
                {
                    "model_label": model,
                    "inference_mode": mode,
                    "benchmark": benchmark,
                    "condition": condition,
                    "grader": grader,
                    "pass_at_1": pass1_bootstrap["difference"],
                    "pass_at_1_ci_low": pass1_bootstrap["ci_low"],
                    "pass_at_1_ci_high": pass1_bootstrap["ci_high"],
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
                    "bootstrap_resamples": pass1_bootstrap["resamples"],
                    "bootstrap_seed": pass1_bootstrap["bootstrap_seed"],
                    "example_count": len(vectors_by_example),
                    "samples_per_example": samples,
                    "smoke": smoke,
                }
            )
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


def _comparison_rows(
    states: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    *,
    smoke: bool,
) -> list[dict[str, Any]]:
    if smoke:
        return []
    result = []

    def compare(key_t: tuple[str, str, str, str], key_c: tuple[str, str, str, str], name: str) -> None:
        treatment, control = states[key_t], states[key_c]
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
                        **pass8_bootstrap,
                    }
                )

    run_keys = sorted({key[:3] for key in states})
    for model, mode, benchmark in run_keys:
        for family in ("sdft", "sdpg"):
            matched = (model, mode, benchmark, family + "_matched")
            no_demo = (model, mode, benchmark, "no_demo")
            shuffled = (model, mode, benchmark, family + "_shuffled")
            compare(matched, no_demo, family + "_matched_minus_no_demo")
            compare(matched, shuffled, family + "_matched_minus_shuffled")
        compare(
            (model, mode, benchmark, "sdft_matched"),
            (model, mode, benchmark, "sdpg_matched"),
            "sdft_matched_minus_sdpg_matched",
        )

    # Mechanism controls use fixed subsets. Restrict the full matched arm to
    # exactly the control IDs before any paired estimand is computed.
    for model in ("starting", "softgrpo"):
        for benchmark in BENCHMARKS:
            for family in ("sdft", "sdpg"):
                matched = states[(model, "native_soft", benchmark, family + "_matched")]
                for control_suffix in ("answer_only", "rationale_only"):
                    control = states[(model, "native_soft", benchmark, family + "_" + control_suffix)]
                    for grader in _graders(benchmark):
                        treatment_pass = _example_pass1(matched, grader)
                        control_pass = _example_pass1(control, grader)
                        subset_ids = set(control_pass)
                        if not subset_ids or not subset_ids.issubset(treatment_pass):
                            raise RuntimeError("mechanism controls are not a subset of matched ICL")
                        restricted_pass = {key: treatment_pass[key] for key in subset_ids}
                        treatment_outcomes = _outcome_vectors(matched, grader)
                        control_outcomes = _outcome_vectors(control, grader)
                        restricted_outcomes = {
                            key: treatment_outcomes[key] for key in subset_ids
                        }
                        bootstrap = paired_bootstrap_difference(
                            restricted_pass, control_pass
                        )
                        rescue = rescue_harm_rates(
                            restricted_outcomes, control_outcomes
                        )
                        result.append(
                            {
                                "comparison": "%s_matched_minus_%s"
                                % (family, control_suffix),
                                "model_label": model,
                                "inference_mode": "native_soft",
                                "benchmark": benchmark,
                                "grader": grader,
                                "estimand": "pass_at_1",
                                **bootstrap,
                                **rescue,
                            }
                        )
                        treatment_pass8 = _example_pass8(matched, grader)
                        control_pass8 = _example_pass8(control, grader)
                        restricted_pass8 = {
                            key: treatment_pass8[key] for key in subset_ids
                        }
                        pass8_bootstrap = paired_bootstrap_difference(
                            restricted_pass8,
                            control_pass8,
                        )
                        result.append(
                            {
                                "comparison": "%s_matched_minus_%s"
                                % (family, control_suffix),
                                "model_label": model,
                                "inference_mode": "native_soft",
                                "benchmark": benchmark,
                                "grader": grader,
                                "estimand": "pass_at_8",
                                **pass8_bootstrap,
                            }
                        )

    for benchmark in BENCHMARKS:
        for family in ("sdft", "sdpg"):
            for control_condition, control_label in (
                ("no_demo", "no_demo"),
                (family + "_shuffled", "shuffled"),
            ):
                post_t = states[("softgrpo", "native_soft", benchmark, family + "_matched")]
                post_c = states[("softgrpo", "native_soft", benchmark, control_condition)]
                start_t = states[("starting", "native_soft", benchmark, family + "_matched")]
                start_c = states[("starting", "native_soft", benchmark, control_condition)]
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
                        result.append(
                            {
                                "comparison": (
                                    "post_minus_start_%s_matched_minus_%s"
                                    % (family, control_label)
                                ),
                                "model_label": "difference_in_differences",
                                "inference_mode": "native_soft",
                                "benchmark": benchmark,
                                "grader": grader,
                                "estimand": estimand,
                                **bootstrap,
                            }
                        )
    return result


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
        if row["pass_at_8"] is not None:
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
        for name in ("difference", "ci_low", "ci_high", "rescue_rate", "harm_rate"):
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
    replay: Sequence[Mapping[str, Any]],
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
    replay_index = {
        (row["model_label"], row["benchmark"], row["condition"]): row
        for row in replay
    }
    result = []
    for benchmark in BENCHMARKS:
        for family in ("sdft", "sdpg"):
            condition = family + "_matched"
            metric = metric_index[("starting", "native_soft", benchmark, condition, "math_verify")]
            diagnostic = diagnostic_index[("starting", "native_soft", benchmark, condition)]
            comparison = comparison_index[
                (
                    "starting",
                    "native_soft",
                    benchmark,
                    family + "_matched_minus_shuffled",
                    "math_verify",
                    "pass_at_1",
                )
            ]
            matched_replay = replay_index[("starting", benchmark, condition)]
            shuffled_replay = replay_index[("starting", benchmark, family + "_shuffled")]
            positive_paired_ci = comparison["ci_low"] > 0.0
            boundary_valid = bool(diagnostic["boundary_gate_valid"])
            # There was no absolute KL cutoff in the preregistration. Use the
            # shuffled context only as a transparent relative proximity
            # diagnostic rather than inventing an absolute threshold post hoc.
            relative_proximity = (
                matched_replay["forward_kl_slot_mean"]
                <= shuffled_replay["forward_kl_slot_mean"]
            )
            result.append(
                {
                    "benchmark": benchmark,
                    "prompt_family": family,
                    "native_soft_pass_at_1": metric["pass_at_1"],
                    "matched_minus_shuffled_ci_low": comparison["ci_low"],
                    "positive_paired_95ci": positive_paired_ci,
                    "boundary_gate_valid": boundary_valid,
                    "matched_forward_kl_slot_mean": matched_replay["forward_kl_slot_mean"],
                    "shuffled_forward_kl_slot_mean": shuffled_replay["forward_kl_slot_mean"],
                    "matched_no_farther_than_shuffled_diagnostic": relative_proximity,
                    "overall_criterion": (
                        "not_pre_registered_due_to_unspecified_reward_and_kl_thresholds"
                    ),
                }
            )
    return result


def run_aggregate(args: argparse.Namespace) -> None:
    assets = verify_icl_assets(args.root)
    asset_root = Path(args.root).expanduser().resolve()
    generation_root = (
        Path(args.generation_dir).expanduser().resolve()
        if args.generation_dir
        else asset_root / "generation"
    )
    replay_root = (
        Path(args.replay_dir).expanduser().resolve()
        if args.replay_dir
        else asset_root / "replay"
    )
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else asset_root / "reports"
    )
    manifests, cells = _generation_inventory(generation_root, smoke=args.smoke)
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
        )
    by_id = {example.example_id: example for example in examples}
    store = AtomicChunkStore(generation_root)
    states = {}
    expected_ids_by_cell: dict[tuple[str, str, str, str], list[str]] = {}
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
        expected_ids_by_cell[key4] = expected_example_ids
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
                store.verify(chunk_key, expected_identity=chunk_identity)
                records, _ = store.load(chunk_key)
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
        if cell.inference_mode == "native_soft":
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
                    "boundary_gate_valid": state["capped_or_all_soft"]
                    / state["count"]
                    <= 0.05,
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

    metric_rows = _cell_metric_rows(states, smoke=args.smoke)
    comparison_rows = _comparison_rows(states, smoke=args.smoke)
    replay_rows = _aggregate_replay(
        replay_root,
        generation_root,
        manifests,
        states,
        expected_ids_by_cell,
        assets=assets,
        data_manifest=data_manifest,
    )
    from_getgo = (
        []
        if args.smoke
        else _from_getgo_assessment(
            metric_rows, diagnostic_rows, comparison_rows, replay_rows
        )
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
        "replay": {
            model: json.loads(
                (replay_root / model / "replay_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            for model in ("starting", "softgrpo")
        },
        "replay_completion": {
            model: json.loads(
                (replay_root / model / "completion.json").read_text(
                    encoding="utf-8"
                )
            )
            for model in ("starting", "softgrpo")
        },
    }
    compact_provenance_bytes = canonical_json_bytes(compact_provenance)
    config = {
        "protocol": "opd-softgrpo-native-soft-icl-report-v1",
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
            if row["inference_mode"] == "native_soft" and not row["boundary_gate_valid"]
        ],
        "notes": {
            "pass_at_1": "canonical c/n estimator over common samples",
            "pass_at_8": "probability at least one of eight succeeds; omitted for smoke n=2",
            "native_soft_scoring": "boundary-invalid samples are incorrect in every primary grader",
            "native_soft_cell_gate": (
                "invalid only when capped-or-all-soft rate exceeds 5%; smoke "
                "separately requires a demonstrated categorical boxed transition "
                "per checkpoint"
            ),
            "aime2024": "30-example intervals are exploratory and imprecise",
            "seed": "single-seed-11 exploratory evaluation",
        },
    }
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="stage pinned models and evaluation data")
    prepare.add_argument("--root", required=True)
    prepare.add_argument("--cache-dir", required=True)
    prepare.set_defaults(handler=run_prepare)

    generate = commands.add_parser("generate", help="run exact upstream generation")
    generate.add_argument("--root", required=True)
    generate.add_argument("--model", required=True, choices=("starting", "softgrpo"))
    generate.add_argument("--mode", required=True, choices=("native_soft", "hard_token"))
    generate.add_argument("--output-dir")
    generate.add_argument("--benchmarks", nargs="+", default=["all"], choices=("all",) + BENCHMARKS)
    generate.add_argument(
        "--conditions", nargs="+", default=["all"], choices=("all",) + PROMPT_CONDITIONS
    )
    generate.add_argument("--num-gpus", type=int, default=1)
    generate.add_argument("--chunk-size", type=int, default=8)
    generate.add_argument("--max-running-requests", type=int, default=16)
    generate.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    generate.add_argument("--smoke", action="store_true")
    generate.set_defaults(handler=run_generate)

    replay = commands.add_parser("replay", help="replay no-demo native-soft trajectories")
    replay.add_argument("--root", required=True)
    replay.add_argument("--model", required=True, choices=("starting", "softgrpo"))
    replay.add_argument("--generation-dir")
    replay.add_argument("--output-dir")
    replay.add_argument("--benchmarks", nargs="+", default=["all"], choices=("all",) + BENCHMARKS)
    replay.add_argument("--hidden-chunk-size", type=int, default=32)
    replay.add_argument("--vocab-chunk-size", type=int, default=8192)
    replay.add_argument(
        "--attention-implementation",
        default="flash_attention_2",
        choices=("flash_attention_2", "sdpa", "eager"),
    )
    replay.set_defaults(handler=run_replay)

    aggregate = commands.add_parser("aggregate", help="grade and aggregate the complete matrix")
    aggregate.add_argument("--root", required=True)
    aggregate.add_argument("--generation-dir")
    aggregate.add_argument("--replay-dir")
    aggregate.add_argument("--output-dir")
    aggregate.add_argument("--smoke", action="store_true")
    aggregate.set_defaults(handler=run_aggregate)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        if args.num_gpus <= 0 or args.chunk_size <= 0 or args.max_running_requests <= 0:
            raise ValueError("GPU, chunk, and running-request counts must be positive")
        if not 0.0 < args.gpu_memory_utilization < 1.0:
            raise ValueError("gpu-memory-utilization must be in (0, 1)")
        if "all" in args.benchmarks and args.benchmarks != ["all"]:
            raise ValueError("all cannot be combined with explicit benchmarks")
        if "all" in args.conditions and args.conditions != ["all"]:
            raise ValueError("all cannot be combined with explicit conditions")
        allocated = os.environ.get("SLURM_GPUS_ON_NODE")
        if allocated is not None and int(allocated) != args.num_gpus:
            raise RuntimeError("--num-gpus disagrees with SLURM_GPUS_ON_NODE")
    if args.command == "replay" and (
        args.hidden_chunk_size <= 0 or args.vocab_chunk_size <= 0
    ):
        raise ValueError("replay chunk sizes must be positive")
    if args.command == "replay" and "all" in args.benchmarks and args.benchmarks != ["all"]:
        raise ValueError("all cannot be combined with explicit benchmarks")
    args.handler(args)


if __name__ == "__main__":
    main()
