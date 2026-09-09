"""Ray transport for one shared, frozen-policy rollout queue.

The coordinator handles request descriptions and nested object references only.
Native completions stay in Ray's object store and are fetched directly by their
original owner rank. Distributed creation and cleanup barriers belong to the
rollout adapter, not this transport.
"""

from __future__ import annotations

import asyncio
import importlib
from typing import Any

from opd_tools.shared_rollout_queue import SharedRolloutQueue


def create_shared_queue(world_size: int, capacity_per_rank: int) -> Any:
    """Create one coordinator from rank zero inside the existing Ray cluster."""

    ray = importlib.import_module("ray")
    if not ray.is_initialized():
        raise RuntimeError("Shared rollout queue requires an initialized Ray cluster")
    coordinator = ray.remote(
        num_cpus=0, max_restarts=0, max_task_retries=0,
    )(SharedRolloutQueue)
    return coordinator.remote(world_size, capacity_per_rank)


def close_shared_queue(actor: Any) -> None:
    """Retire the coordinator after all ranks reach the cleanup boundary."""

    if actor is None:
        return
    ray = importlib.import_module("ray")
    try:
        ray.kill(actor, no_restart=True)
    except ray.exceptions.RayActorError:
        # An engine failure may already have killed this coordinator. Cleanup
        # must not mask the original rollout error in that case.
        pass


def abort_shared_queue(actor: Any, rank: int, reason: str) -> None:
    """Notify peers of adapter failure even if its asyncio loop cannot start.

    The adapter catches notification errors, retires its local engine, and
    enters the matched generation-error collective. The timeout prevents a
    failed coordinator from indefinitely delaying that cleanup boundary.
    """

    if actor is None:
        return
    ray = importlib.import_module("ray")
    ray.get(actor.fail.remote(rank, reason), timeout=10)


class RayQueueClient:
    """Asynchronous queue RPCs and direct object-store completion transport."""

    def __init__(self, actor: Any):
        self.actor = actor
        self._ray = importlib.import_module("ray")
        # Keep producer-owned references alive through the complete rollout.
        # The coordinator also retains nested references until owner retrieval.
        self._output_refs: list[Any] = []

    async def register(self, rank: int, requests: list[Any], policy_identity: Any) -> Any:
        return await self.actor.register.remote(rank, requests, policy_identity)

    async def claim(self, rank: int) -> Any:
        return await self.actor.claim.remote(rank)

    async def complete(self, rank: int, global_index: int, payload_reference: Any) -> Any:
        # Keep the wrapper intact: a top-level ObjectRef argument would be
        # dereferenced by Ray before reaching the coordinator actor.
        self._unwrap_reference(payload_reference)
        return await self.actor.complete.remote(rank, global_index, payload_reference)

    async def fail(self, rank: int, reason: str) -> Any:
        return await self.actor.fail.remote(rank, reason)

    async def status(self) -> Any:
        return await self.actor.status.remote()

    async def results_for(self, rank: int) -> Any:
        return await self.actor.results_for.remote(rank)

    async def put_output(self, output: Any) -> list[Any]:
        """Store a native completion without blocking other engine requests."""

        reference = await asyncio.to_thread(self._ray.put, output)
        self._output_refs.append(reference)
        return [reference]

    async def get_outputs(self, references: list[Any]) -> list[Any]:
        """Fetch wrapped references directly, preserving canonical row order."""

        refs = [self._unwrap_reference(item) for item in references]
        return list(await asyncio.gather(*refs))

    def _unwrap_reference(self, wrapped: Any) -> Any:
        if (
            not isinstance(wrapped, (list, tuple))
            or len(wrapped) != 1
            or not isinstance(wrapped[0], self._ray.ObjectRef)
        ):
            raise TypeError("Native completion reference must be wrapped as [ObjectRef]")
        return wrapped[0]
