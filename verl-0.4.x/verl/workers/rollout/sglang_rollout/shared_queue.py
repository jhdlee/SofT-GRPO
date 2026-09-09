"""Asynchronous consumers for the shared frozen-policy rollout coordinator.

Coordinator state is imported from the dependency-light OPD tools package so
fresh Ray actors do not import the inference runtime.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from typing import Any

from opd_tools.shared_rollout_queue import SharedRolloutError, SharedRolloutQueue

from .request_dispatch import poison_engine, positive_integer, require_idle_engine


async def dispatch_shared_generation(
    engine: Any, client: Any, requests: Sequence[Mapping[str, Any]], queue_size: int,
    rank: int, policy_identity: Any, *, poll_interval_seconds: float = 0.1,
) -> tuple[list[Mapping[str, Any]], dict[str, Any], list[int]]:
    """Drain one global pool while retaining original ownership and metadata.

``client`` provides async coordinator methods plus ``put_output(output)`` and
``get_outputs(references)``. Engines remain outstanding through the global
completion barrier and result transport, including engines that finish early.
    """

    started = time.perf_counter()
    pending: dict[asyncio.Task[Any], dict[str, Any]] = {}
    peak_pending = completed_count = 0
    first_submission = last_completion = None
    owns_outstanding_flag = False
    try:
        # Even entry failures must reach the coordinator: peers may already be
        # waiting for this rank to register and cannot enter their collective.
        positive_integer(queue_size, "queue_size")
        if isinstance(poll_interval_seconds, bool) or not isinstance(poll_interval_seconds, (int, float)) or not 0 < poll_interval_seconds <= 5:
            raise ValueError("poll_interval_seconds must be positive and at most five seconds")
        require_idle_engine(engine)
        engine._opd_batch_outstanding = True
        owns_outstanding_flag = True
        await client.register(rank, requests, policy_identity)
        registration_seconds = time.perf_counter() - started
        while True:
            state = await client.status()
            if state["state"] == "done":
                if pending:
                    raise SharedRolloutError("global queue completed with local requests outstanding")
                break
            while len(pending) < queue_size:
                item = await client.claim(rank)
                if item["state"] != "request":
                    break
                request = item["request"]
                if first_submission is None:
                    first_submission = time.perf_counter()
                task = asyncio.create_task(engine.async_generate(
                    prompt=None, input_ids=request["input_ids"], image_data=request["image_data"],
                    sampling_params=request["sampling_params"], return_logprob=True,
                ))
                pending[task] = item
                peak_pending = max(peak_pending, len(pending))
            if not pending:
                await asyncio.sleep(poll_interval_seconds)
                continue
            finished, _ = await asyncio.wait(pending, timeout=poll_interval_seconds, return_when=asyncio.FIRST_COMPLETED)
            for task in sorted(finished, key=lambda item: pending[item]["global_index"]):
                item = pending.pop(task)
                output = task.result()
                if not isinstance(output, Mapping):
                    raise SharedRolloutError("SGLang returned a non-mapping completion")
                reference = await client.put_output(output)
                await client.complete(rank, item["global_index"], reference)
                completed_count += 1
                last_completion = time.perf_counter()
            await asyncio.sleep(0)
        generation_seconds = time.perf_counter() - started
        transfer_started = time.perf_counter()
        records = await client.results_for(rank)
        if len(records) != len(requests) or [item["local_index"] for item in records] != list(range(len(requests))):
            raise SharedRolloutError("shared rollout results have incorrect ownership or order")
        outputs = await client.get_outputs([item["payload_reference"] for item in records])
        if not isinstance(outputs, (list, tuple)) or len(outputs) != len(requests) or not all(isinstance(item, Mapping) for item in outputs):
            raise SharedRolloutError("shared rollout transport returned incorrect completions")
        # A peer can fail during result transport, after generation itself ends.
        await client.status()
        local_seconds = 0.0 if first_submission is None else last_completion - first_submission
        return list(outputs), {
            "dispatch_mode": "shared_queue", "request_count": len(requests),
            "frontend_peak_pending": peak_pending,
            "engine_generation_seconds": generation_seconds,
            "shared_queue_local_generation_seconds": local_seconds,
            "shared_queue_global_wait_seconds": max(0.0, generation_seconds - local_seconds),
            "shared_queue_registration_seconds": registration_seconds,
            "shared_queue_result_transport_seconds": time.perf_counter() - transfer_started,
            "shared_queue_executed_request_count": completed_count,
            "shared_queue_owned_request_count": len(requests),
        }, [item["execution_rank"] for item in records]
    except BaseException as error:
        try:
            await asyncio.wait_for(client.fail(rank, f"{type(error).__name__}: {error}"), timeout=5)
        except BaseException:
            # Communication failure still requires retiring this local engine.
            pass
        poison_engine(engine, f"{type(error).__name__}: {error}")
        raise
    finally:
        # Frontend cancellation cannot abort the SGLang scheduler request. The
        # failure path above shuts down that scheduler before cancelling these.
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if owns_outstanding_flag:
            engine._opd_batch_outstanding = False
