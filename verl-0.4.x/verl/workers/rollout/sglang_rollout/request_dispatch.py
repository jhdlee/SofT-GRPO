"""Bounded, order-preserving submission for one frozen-policy rollout.

This module deliberately imports no inference or distributed runtime. Sampling
seeds are resolved by the existing deterministic_sampling helpers before entry;
expansion here must never derive those seeds a second time.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from typing import Any


LOCAL_DISPATCH_MODES = ("legacy_batch", "expanded_batch", "bounded_async")
DISPATCH_MODES = (*LOCAL_DISPATCH_MODES, "shared_queue")


def positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_dispatch_options(mode: str, queue_size: int, max_running_requests: int | None) -> None:
    if mode not in DISPATCH_MODES:
        raise ValueError(f"dispatch_mode must be one of {DISPATCH_MODES}; got {mode!r}")
    positive_integer(queue_size, "async_queue_size")
    if max_running_requests is not None:
        positive_integer(max_running_requests, "max_running_requests")


def benchmark_response_cap(meta_info: Mapping[str, Any], response_cap: int) -> int:
    """Apply an explicitly marked, per-call benchmark warmup reduction only."""

    if "benchmark_max_new_tokens" not in meta_info:
        return response_cap
    cap = positive_integer(meta_info["benchmark_max_new_tokens"], "benchmark_max_new_tokens")
    if cap > response_cap:
        raise ValueError("benchmark_max_new_tokens must not exceed the configured response cap")
    return cap


def poison_engine(engine: Any, reason: str) -> None:
    """Permanently retire an engine after a partial or failed rollout.

    Cancelling an async_generate coroutine does not abort its scheduler request
    in the pinned SGLang runtime. Killing the engine's child processes prevents
    requests from surviving into weight synchronization or memory release.
    """

    if engine is None:
        return
    engine._opd_poisoned = str(reason)
    if not getattr(engine, "_opd_shutdown_after_failure", False):
        engine._opd_shutdown_after_failure = True
        try:
            engine.shutdown()
        except Exception:
            # Preserve the original collective failure; the job supervisor also
            # tears down Ray's process tree. The poison guard still forbids reuse.
            pass


def require_healthy_engine(engine: Any) -> None:
    reason = getattr(engine, "_opd_poisoned", None)
    if reason:
        raise RuntimeError(f"SGLang engine is poisoned and cannot be reused: {reason}")


def require_idle_engine(engine: Any) -> None:
    """Forbid policy/cache/memory transitions until the whole batch completes."""

    require_healthy_engine(engine)
    if getattr(engine, "_opd_batch_outstanding", False):
        raise RuntimeError("SGLang rollout requests remain outstanding")


def check_collective_error(engine: Any, distributed: Any, error: BaseException | None, stage: str) -> None:
    """Make every rollout rank fail before any following collective/transition."""

    message = None if error is None else f"{type(error).__name__}: {error}"[:2000]
    errors = [message]
    if distributed.is_initialized():
        errors = [None] * distributed.get_world_size()
        try:
            distributed.all_gather_object(errors, message)
        except BaseException as collective_error:
            poison_engine(engine, f"collective {stage} communication failed: {collective_error}")
            raise
    failures = [f"rank {rank}: {item}" for rank, item in enumerate(errors) if item is not None]
    if failures:
        reason = f"SGLang collective {stage} failed: " + "; ".join(failures)
        poison_engine(engine, reason)
        raise RuntimeError(reason) from error


def _prompt_sampling_params(
    input_ids: Sequence[Sequence[int]],
    image_data: Sequence[Any],
    sampling_params: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    if not input_ids or len(input_ids) != len(image_data):
        raise ValueError("prompt IDs and image data must be nonempty and aligned")
    if isinstance(sampling_params, Mapping):
        params = [dict(sampling_params) for _ in input_ids]
    else:
        if len(sampling_params) != len(input_ids):
            raise ValueError("sampling parameters must align with prompts")
        params = [dict(item) for item in sampling_params]
    counts = [positive_integer(item.get("n", 1), "sampling n") for item in params]
    if len(set(counts)) != 1:
        raise ValueError("all prompts must use the same rollout group size")
    return params, counts


def expanded_requests(
    input_ids: Sequence[Sequence[int]],
    image_data: Sequence[Any],
    sampling_params: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    expanded_sampling_seeds: Sequence[int] | None,
) -> list[dict[str, Any]]:
    """Flatten G in prompt-major order, using already-expanded seeds once."""

    params, counts = _prompt_sampling_params(input_ids, image_data, sampling_params)
    total = sum(counts)
    if expanded_sampling_seeds is not None and len(expanded_sampling_seeds) != total:
        raise ValueError("expanded seeds must provide one seed per trajectory")
    if expanded_sampling_seeds is None and any(item.get("seed") is not None for item in params):
        raise ValueError("seeded expansion requires the canonical expanded sampling seeds")

    requests = []
    for prompt_index, (ids, image, params_row, count) in enumerate(zip(input_ids, image_data, params, counts)):
        for sample_index in range(count):
            index = len(requests)
            item = dict(params_row)
            item["n"] = 1
            if expanded_sampling_seeds is not None:
                item["seed"] = int(expanded_sampling_seeds[index])
            requests.append({
                "index": index,
                "prompt_index": prompt_index,
                "sample_index": sample_index,
                "input_ids": list(ids),
                "image_data": image,
                "sampling_params": item,
            })
    return requests


def _validate_outputs(outputs: Any, expected_count: int) -> list[Mapping[str, Any]]:
    if not isinstance(outputs, (list, tuple)) or len(outputs) != expected_count:
        raise RuntimeError(f"SGLang must return exactly {expected_count} ordered completions")
    if not all(isinstance(output, Mapping) for output in outputs):
        raise RuntimeError("SGLang returned a non-mapping completion")
    # Keep every native metadata field, without converting through chat text.
    return list(outputs)


async def dispatch_generation(
    engine: Any,
    *,
    mode: str,
    queue_size: int,
    input_ids: Sequence[Sequence[int]],
    image_data: Sequence[Any],
    sampling_params: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    expanded_sampling_seeds: Sequence[int] | None,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Finish the entire frozen-policy batch, retaining canonical row order."""

    validate_dispatch_options(mode, queue_size, None)
    if mode == "shared_queue":
        raise ValueError("shared_queue requires the distributed shared-queue coordinator")
    require_idle_engine(engine)
    _, counts = _prompt_sampling_params(input_ids, image_data, sampling_params)
    request_count = sum(counts)
    # The compatibility path passes the original engine parameters unchanged,
    # including callers using engine-managed seeds without OPD seed metadata.
    requests = (
        None if mode == "legacy_batch"
        else expanded_requests(input_ids, image_data, sampling_params, expanded_sampling_seeds)
    )
    started = time.perf_counter()
    peak_pending = 0
    pending: dict[asyncio.Task[Any], int] = {}
    engine._opd_batch_outstanding = True
    try:
        if mode == "legacy_batch":
            outputs = await engine.async_generate(
                prompt=None, input_ids=list(input_ids), image_data=list(image_data),
                sampling_params=sampling_params, return_logprob=True,
            )
        elif mode == "expanded_batch":
            outputs = await engine.async_generate(
                prompt=None,
                input_ids=[item["input_ids"] for item in requests],
                image_data=[item["image_data"] for item in requests],
                sampling_params=[item["sampling_params"] for item in requests],
                return_logprob=True,
            )
        else:
            outputs = [None] * len(requests)
            next_index = 0

            def submit(index: int) -> None:
                item = requests[index]
                task = asyncio.create_task(engine.async_generate(
                    prompt=None, input_ids=item["input_ids"],
                    image_data=item["image_data"], sampling_params=item["sampling_params"],
                    return_logprob=True,
                ))
                pending[task] = index

            while next_index < len(requests) and len(pending) < queue_size:
                submit(next_index)
                next_index += 1
            peak_pending = len(pending)
            while pending:
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in sorted(done, key=lambda item: pending[item]):
                    index = pending.pop(task)
                    output = task.result()
                    if not isinstance(output, Mapping):
                        raise RuntimeError("SGLang returned a non-mapping completion")
                    if outputs[index] is not None:
                        raise RuntimeError("SGLang completed a trajectory more than once")
                    outputs[index] = output
                    if next_index < len(requests):
                        submit(next_index)
                        next_index += 1
                peak_pending = max(peak_pending, len(pending))
                # Run newly submitted calls before any synchronous processing.
                await asyncio.sleep(0)
        outputs = _validate_outputs(outputs, request_count)
        return outputs, {
            "engine_generation_seconds": time.perf_counter() - started,
            "request_count": request_count,
            "frontend_peak_pending": peak_pending,
            "dispatch_mode": mode,
        }
    except BaseException as error:
        # Retire the backend before cancelling its frontend coroutines. No
        # partial result, retry, subsequent cache flush, or update is permitted.
        poison_engine(engine, f"{type(error).__name__}: {error}")
        raise
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        engine._opd_batch_outstanding = False
