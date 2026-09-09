"""Dependency-light coordinator state for one frozen-policy shared rollout.

Only request descriptions and opaque output references visit this actor. Its
fresh Ray process must not import the model runtime merely to manage a queue.
"""

from __future__ import annotations

import copy
import json
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any


def positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class SharedRolloutError(RuntimeError):
    """The entire frozen-policy batch must be discarded."""


class SharedRolloutQueue:
    """Serialized coordinator state for one complete distributed rollout.

The actor wrapping this class must serialize its method calls. Each rank gets
an initial reservation so an early polling rank cannot take the entire queue
before another rank starts. All remaining requests form one common pool;
running responses are never migrated or retried.
"""

    def __init__(self, world_size: int, per_rank_capacity: int):
        self.world_size = positive_integer(world_size, "world_size")
        self.per_rank_capacity = positive_integer(per_rank_capacity, "per_rank_capacity")
        self._registrations: dict[int, list[dict[str, Any]]] = {}
        self._policy_identity: str | None = None
        self._requests: list[dict[str, Any]] = []
        self._reserved: dict[int, deque[int]] = {}
        self._available: deque[int] = deque()
        self._claims: dict[int, int] = {}
        self._inflight = [0] * world_size
        self._results: dict[int, dict[str, Any]] = {}
        self._failure: str | None = None

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise SharedRolloutError(self._failure)

    def _reject(self, reason: str) -> None:
        if self._failure is None:
            self._failure = reason
        self._raise_if_failed()

    def _rank(self, rank: int) -> int:
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < self.world_size:
            self._reject(f"invalid rollout rank {rank!r}")
        return rank

    def register(self, rank: int, requests: Sequence[Mapping[str, Any]], policy_identity: Any) -> dict[str, Any]:
        self._raise_if_failed()
        rank = self._rank(rank)
        if rank in self._registrations:
            self._reject(f"rank {rank} registered more than once")
        if policy_identity is None or policy_identity == "" or policy_identity == {}:
            self._reject("a nonempty frozen policy identity is required")
        try:
            identity = json.dumps(policy_identity, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            self._reject(f"invalid frozen policy identity: {error}")
        if self._policy_identity is not None and identity != self._policy_identity:
            self._reject(f"rank {rank} has a different frozen policy identity")
        if not isinstance(requests, (list, tuple)):
            self._reject("expanded requests must be a list or tuple")
        owned = []
        seen = set()
        for local_index, request in enumerate(requests):
            if not isinstance(request, Mapping):
                self._reject("expanded request must be a mapping")
            if type(request.get("index")) is not int or request["index"] != local_index:
                self._reject(f"rank {rank} request indices must be canonical and unique")
            pair = (request.get("prompt_index"), request.get("sample_index"))
            if any(type(value) is not int or value < 0 for value in pair) or pair in seen:
                self._reject(f"rank {rank} has invalid or duplicate prompt/sample identities")
            seen.add(pair)
            ids = request.get("input_ids")
            params = request.get("sampling_params")
            if not isinstance(ids, (list, tuple)) or not ids or any(type(value) is not int or value < 0 for value in ids):
                self._reject("expanded request must contain nonempty token IDs")
            if not isinstance(params, Mapping) or type(params.get("n")) is not int or params["n"] != 1:
                self._reject("shared queue requires already-expanded requests with n=1")
            if "seed" in params and params["seed"] is not None and type(params["seed"]) is not int:
                self._reject("expanded request seed must be an integer when supplied")
            if "image_data" not in request:
                self._reject("expanded request is missing image_data")
            owned.append(copy.deepcopy(dict(request)))
        self._registrations[rank] = owned
        self._policy_identity = identity
        if len(self._registrations) == self.world_size:
            for owner in range(self.world_size):
                for local_index, request in enumerate(self._registrations[owner]):
                    self._requests.append({
                        "global_index": len(self._requests), "owner_rank": owner,
                        "local_index": local_index, "request": request,
                    })
            if not self._requests:
                self._reject("a shared rollout must contain at least one request")
            initial = min(self.per_rank_capacity, len(self._requests) // self.world_size)
            self._reserved = {
                participant: deque(range(participant * initial, (participant + 1) * initial))
                for participant in range(self.world_size)
            }
            self._available = deque(range(initial * self.world_size, len(self._requests)))
        return self.status()

    def claim(self, rank: int) -> dict[str, Any]:
        self._raise_if_failed()
        rank = self._rank(rank)
        if rank not in self._registrations:
            self._reject(f"rank {rank} claimed work before registration")
        if len(self._registrations) < self.world_size:
            return {"state": "waiting"}
        if len(self._results) == len(self._requests):
            return {"state": "done"}
        if self._inflight[rank] >= self.per_rank_capacity:
            return {"state": "waiting"}
        queue = self._reserved[rank] or self._available
        if not queue:
            return {"state": "waiting"}
        global_index = queue.popleft()
        self._claims[global_index] = rank
        self._inflight[rank] += 1
        return {"state": "request", **copy.deepcopy(self._requests[global_index])}

    def complete(self, rank: int, global_index: int, payload_reference: Any) -> dict[str, Any]:
        self._raise_if_failed()
        rank = self._rank(rank)
        if type(global_index) is not int or global_index not in self._claims:
            self._reject(f"rank {rank} completed an unclaimed request {global_index!r}")
        if self._claims[global_index] != rank:
            self._reject(f"rank {rank} completed a request claimed by another rank")
        if global_index in self._results:
            self._reject(f"request {global_index} completed more than once")
        request = self._requests[global_index]
        # Do not copy, inspect, dereference, or materialize this transport token.
        self._results[global_index] = {
            "payload_reference": payload_reference, "execution_rank": rank,
            "local_index": request["local_index"], "global_index": global_index,
        }
        self._inflight[rank] -= 1
        return self.status()

    def fail(self, rank: int, reason: str) -> dict[str, Any]:
        # Failure reporting stays idempotent even when every peer reports it.
        if self._failure is None:
            self._failure = f"shared rollout failed on rank {rank}: {str(reason)[:2000]}"
        return {"state": "failed", "failure": self._failure}

    def status(self) -> dict[str, Any]:
        self._raise_if_failed()
        registered = len(self._registrations)
        state = "registering" if registered < self.world_size else (
            "done" if len(self._results) == len(self._requests) else "running"
        )
        return {
            "state": state, "registered_ranks": registered, "world_size": self.world_size,
            "request_count": len(self._requests), "claimed_count": len(self._claims),
            "completed_count": len(self._results), "inflight_count": sum(self._inflight),
        }

    def results_for(self, rank: int) -> list[dict[str, Any]]:
        self._raise_if_failed()
        rank = self._rank(rank)
        if self.status()["state"] != "done":
            raise SharedRolloutError("shared rollout results are unavailable before every response completes")
        return [dict(self._results[item["global_index"]]) for item in self._requests if item["owner_rank"] == rank]
