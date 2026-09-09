"""Shared scheduling, ownership, and failure barriers without Ray or GPUs."""

import asyncio
import copy
import importlib.util
import sys
import types
from pathlib import Path

import pytest


FORK_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FORK_ROOT))
ROLLOUT = FORK_ROOT / "verl-0.4.x/verl/workers/rollout/sglang_rollout"
PACKAGE = "opd_shared_queue_tested"
if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(ROLLOUT)]
    sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(f"{PACKAGE}.shared_queue", ROLLOUT / "shared_queue.py")
shared = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = shared
spec.loader.exec_module(shared)


POLICY = {"iteration": 9, "checkpoint": "same-weights", "sampling_contract": "same-config"}


def requests(rank, count=8):
    return [{
        "index": index, "prompt_index": index // 2, "sample_index": index % 2,
        "input_ids": [100 + rank, index + 1], "image_data": None,
        "sampling_params": {"n": 1, "seed": rank * 10000 + index, "max_new_tokens": 8192},
    } for index in range(count)]


def native_output(request):
    seed = request["sampling_params"]["seed"]
    return {
        "text": f"response-{seed}",
        "meta_info": {
            "seed": seed, "input_ids": request["input_ids"],
            "output_token_logprobs": [(-0.3, 19, None)],
            "output_topk_idx_list": [[19, 20]],
            "output_topk_gumbel_list": [[0.7, -0.2]],
            "output_topk_gumbel_noise_list": [[0.4, -0.6]],
            "output_topk_retained_mask_list": [[True, False]],
            "output_topk_prob_list": [[0.8, 0.2]],
            "unrecognized_future_field": {"must_survive": True},
        },
    }


class Client:
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.outputs = []

    async def register(self, *args):
        await asyncio.sleep(0)
        return self.coordinator.register(*args)

    async def claim(self, *args):
        await asyncio.sleep(0)
        return self.coordinator.claim(*args)

    async def complete(self, *args):
        await asyncio.sleep(0)
        return self.coordinator.complete(*args)

    async def fail(self, *args):
        return self.coordinator.fail(*args)

    async def status(self):
        await asyncio.sleep(0)
        return self.coordinator.status()

    async def results_for(self, rank):
        return self.coordinator.results_for(rank)

    async def put_output(self, output):
        self.outputs.append(output)
        return [len(self.outputs) - 1]

    async def get_outputs(self, references):
        return [self.outputs[reference[0]] for reference in references]


class Engine:
    def __init__(self, rank, delay=0.001):
        self.rank = rank
        self.delay = delay
        self.calls = []
        self.active = self.peak = self.shutdown_calls = 0
        self.events = []

    async def async_generate(self, **kwargs):
        assert kwargs["return_logprob"] is True
        assert kwargs["sampling_params"]["n"] == 1
        self.calls.append(copy.deepcopy(kwargs))
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            return native_output(kwargs)
        finally:
            self.active -= 1
            self.events.append("finished")

    def shutdown(self):
        self.shutdown_calls += 1
        self.events.append("shutdown")


async def consume(engine, client, owned, cap=2, policy=POLICY):
    return await shared.dispatch_shared_generation(
        engine, client, owned, cap, engine.rank, policy, poll_interval_seconds=0.001,
    )


def test_shared_queue_uses_faster_engine_more_and_restores_native_results():
    async def run():
        coordinator = shared.SharedRolloutQueue(2, 2)
        client = Client(coordinator)
        engines = [Engine(0, delay=0.035), Engine(1, delay=0.001)]
        owned = [requests(rank, 20) for rank in range(2)]
        original = copy.deepcopy(owned)
        results = await asyncio.wait_for(asyncio.gather(*[
            consume(engine, client, owned[rank]) for rank, engine in enumerate(engines)
        ]), timeout=3)
        assert len(engines[1].calls) > len(engines[0].calls)
        assert len(engines[0].calls) >= 2  # Initial reservation cannot be stolen.
        assert sum(len(engine.calls) for engine in engines) == 40
        seeds = [call["sampling_params"]["seed"] for engine in engines for call in engine.calls]
        assert len(seeds) == len(set(seeds)) == 40
        assert owned == original
        for rank, (outputs, timing, execution_ranks) in enumerate(results):
            assert outputs == [native_output(item) for item in owned[rank]]
            assert len(execution_ranks) == len(owned[rank])
            assert set(execution_ranks).issubset({0, 1})
            assert timing["frontend_peak_pending"] == engines[rank].peak == 2
            assert timing["shared_queue_executed_request_count"] == len(engines[rank].calls)
            assert timing["shared_queue_owned_request_count"] == timing["request_count"] == 20
            assert timing["engine_generation_seconds"] >= timing["shared_queue_local_generation_seconds"]
            assert timing["shared_queue_result_transport_seconds"] >= 0
            assert not engines[rank]._opd_batch_outstanding
            assert engines[rank].shutdown_calls == engines[rank].active == 0
        assert coordinator.status()["state"] == "done"
        assert coordinator.status()["completed_count"] == 40

    asyncio.run(run())


def test_registration_barrier_and_initial_capacity_are_enforced():
    queue = shared.SharedRolloutQueue(2, 2)
    queue.register(1, requests(1), POLICY)
    assert queue.claim(1) == {"state": "waiting"}
    queue.register(0, requests(0), POLICY)
    first = [queue.claim(1), queue.claim(1)]
    assert queue.claim(1) == {"state": "waiting"}
    assert len({item["global_index"] for item in first}) == 2
    for item in first:
        queue.complete(1, item["global_index"], object())
    # Rank 1 can take common work, but rank 0's reserved work stays available.
    common = queue.claim(1)
    reserved = [queue.claim(0), queue.claim(0)]
    assert common["global_index"] >= 4
    assert [item["global_index"] for item in reserved] == [0, 1]


def test_opaque_results_wait_for_whole_batch_and_preserve_ownership():
    class Opaque:
        def __deepcopy__(self, memo):
            raise AssertionError("output reference must not be inspected or copied")

    queue = shared.SharedRolloutQueue(2, 1)
    queue.register(1, requests(1, 1), POLICY)
    queue.register(0, requests(0, 1), POLICY)
    first, second = queue.claim(0), queue.claim(1)
    reference = Opaque()
    queue.complete(0, first["global_index"], reference)
    with pytest.raises(shared.SharedRolloutError, match="before every response"):
        queue.results_for(0)
    queue.complete(1, second["global_index"], object())
    result = queue.results_for(0)
    assert result[0]["payload_reference"] is reference
    assert result[0]["execution_rank"] == 0
    assert result[0]["local_index"] == 0
    assert queue.claim(0) == {"state": "done"}


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows[0].update(index=1), "indices"),
    (lambda rows: rows[1].update(prompt_index=0, sample_index=0), "identities"),
    (lambda rows: rows[0]["sampling_params"].update(n=8), "n=1"),
    (lambda rows: rows[0]["sampling_params"].update(seed="derive again"), "seed"),
    (lambda rows: rows[0].update(input_ids=[]), "token IDs"),
    (lambda rows: rows[0].pop("image_data"), "image_data"),
])
def test_malformed_requests_fail_every_peer(mutation, match):
    queue = shared.SharedRolloutQueue(2, 2)
    queue.register(0, requests(0), POLICY)
    malformed = requests(1)
    mutation(malformed)
    with pytest.raises(shared.SharedRolloutError, match=match):
        queue.register(1, malformed, POLICY)
    with pytest.raises(shared.SharedRolloutError, match=match):
        queue.claim(0)


@pytest.mark.parametrize("action,match", [
    (lambda queue: queue.register(0, requests(0), POLICY), "more than once"),
    (lambda queue: queue.register(1, requests(1), {**POLICY, "iteration": 10}), "different frozen policy"),
    (lambda queue: queue.claim(1), "before registration"),
    (lambda queue: queue.claim(2), "invalid rollout rank"),
])
def test_protocol_violations_fail_whole_rollout(action, match):
    queue = shared.SharedRolloutQueue(2, 1)
    queue.register(0, requests(0), POLICY)
    with pytest.raises(shared.SharedRolloutError, match=match):
        action(queue)
    with pytest.raises(shared.SharedRolloutError, match=match):
        queue.status()


@pytest.mark.parametrize("violation,match", [("unclaimed", "unclaimed"), ("wrong_rank", "another rank"), ("duplicate", "more than once")])
def test_invalid_completions_fail_every_peer(violation, match):
    queue = shared.SharedRolloutQueue(2, 1)
    for rank in range(2):
        queue.register(rank, requests(rank), POLICY)
    item = queue.claim(0)
    if violation == "duplicate":
        queue.complete(0, item["global_index"], object())
    with pytest.raises(shared.SharedRolloutError, match=match):
        queue.complete(1 if violation == "wrong_rank" else 0,
                       999 if violation == "unclaimed" else item["global_index"], object())
    with pytest.raises(shared.SharedRolloutError, match=match):
        queue.claim(1)


def test_failure_poisons_peers_before_cancelling_outstanding_requests():
    class FailingEngine(Engine):
        async def async_generate(self, **kwargs):
            if self.rank == 0 and kwargs["input_ids"][-1] == 1:
                await asyncio.sleep(0.015)
                raise ValueError("scheduler failure")
            self.active += 1
            try:
                await asyncio.Event().wait()
            finally:
                self.active -= 1
                self.events.append("cancelled")

    async def run():
        client = Client(shared.SharedRolloutQueue(2, 2))
        engines = [FailingEngine(rank) for rank in range(2)]
        results = await asyncio.wait_for(asyncio.gather(*[
            consume(engine, client, requests(engine.rank)) for engine in engines
        ], return_exceptions=True), timeout=2)
        assert isinstance(results[0], ValueError)
        assert isinstance(results[1], shared.SharedRolloutError)
        for engine in engines:
            assert engine.shutdown_calls == 1
            assert engine.events.index("shutdown") < engine.events.index("cancelled")
            assert engine.active == 0
            assert engine._opd_poisoned
            assert not engine._opd_batch_outstanding
        with pytest.raises(shared.SharedRolloutError, match="scheduler failure"):
            client.coordinator.results_for(0)

    asyncio.run(run())


def test_finished_rank_stays_outstanding_and_observes_later_peer_failure():
    class LaterFailureEngine(Engine):
        async def async_generate(self, **kwargs):
            if self.rank == 1:
                await asyncio.sleep(0.04)
                assert engines[0]._opd_batch_outstanding
                with pytest.raises(RuntimeError, match="outstanding"):
                    shared.require_idle_engine(engines[0])
                raise ValueError("late peer failure")
            return await super().async_generate(**kwargs)

    async def run():
        nonlocal engines
        client = Client(shared.SharedRolloutQueue(2, 1))
        engines = [LaterFailureEngine(rank) for rank in range(2)]
        results = await asyncio.wait_for(asyncio.gather(*[
            consume(engine, client, requests(engine.rank, 1), cap=1) for engine in engines
        ], return_exceptions=True), timeout=2)
        assert all(isinstance(result, BaseException) for result in results)
        assert all(engine.shutdown_calls == 1 for engine in engines)
        assert len(engines[0].calls) == 1

    engines = []
    asyncio.run(run())


def test_cancellation_fails_peers_and_retires_both_engines():
    class BlockingEngine(Engine):
        async def async_generate(self, **kwargs):
            self.started.set()
            self.active += 1
            try:
                await asyncio.Event().wait()
            finally:
                self.active -= 1
                self.events.append("cancelled")

    async def run():
        client = Client(shared.SharedRolloutQueue(2, 1))
        engines = [BlockingEngine(rank) for rank in range(2)]
        for engine in engines:
            engine.started = asyncio.Event()
        tasks = [asyncio.create_task(consume(engine, client, requests(engine.rank), cap=1)) for engine in engines]
        await asyncio.gather(*[engine.started.wait() for engine in engines])
        tasks[0].cancel()
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=2)
        assert isinstance(results[0], asyncio.CancelledError)
        assert isinstance(results[1], shared.SharedRolloutError)
        for engine in engines:
            assert engine.shutdown_calls == 1
            assert engine.events == ["shutdown", "cancelled"]
            assert engine.active == 0
            assert not engine._opd_batch_outstanding

    asyncio.run(run())


def test_no_memory_transition_until_owned_results_arrive():
    class TransportClient(Client):
        async def get_outputs(self, references):
            assert engine._opd_batch_outstanding
            with pytest.raises(RuntimeError, match="outstanding"):
                shared.require_idle_engine(engine)
            return await super().get_outputs(references)

    engine = Engine(0)
    asyncio.run(consume(engine, TransportClient(shared.SharedRolloutQueue(1, 2)), requests(0)))
    shared.require_idle_engine(engine)


def test_result_transport_failure_marks_global_failure_and_retires_engine():
    class BrokenTransport(Client):
        async def get_outputs(self, references):
            raise OSError("object lost")

    client = BrokenTransport(shared.SharedRolloutQueue(1, 2))
    engine = Engine(0)
    with pytest.raises(OSError, match="object lost"):
        asyncio.run(consume(engine, client, requests(0)))
    assert engine.shutdown_calls == 1
    with pytest.raises(shared.SharedRolloutError, match="object lost"):
        client.coordinator.status()


def test_empty_owner_can_execute_another_owners_requests():
    async def run():
        client = Client(shared.SharedRolloutQueue(2, 1))
        engines = [Engine(rank) for rank in range(2)]
        result = await asyncio.wait_for(asyncio.gather(
            consume(engines[0], client, [], cap=1),
            consume(engines[1], client, requests(1, 4), cap=1),
        ), timeout=2)
        assert result[0][0] == result[0][2] == []
        assert result[0][1]["shared_queue_executed_request_count"] > 0
        assert result[1][0] == [native_output(item) for item in requests(1, 4)]

    asyncio.run(run())


def test_entry_failure_reaches_peers_waiting_for_registration():
    async def run():
        client = Client(shared.SharedRolloutQueue(2, 1))
        engines = [Engine(rank) for rank in range(2)]
        engines[0]._opd_poisoned = "earlier failure"
        results = await asyncio.wait_for(asyncio.gather(*[
            consume(engine, client, requests(engine.rank), cap=1) for engine in engines
        ], return_exceptions=True), timeout=2)
        assert all(isinstance(result, RuntimeError) for result in results)
        assert all("earlier failure" in str(result) for result in results)
        assert all(engine.shutdown_calls == 1 for engine in engines)
        assert not any(engine.calls for engine in engines)

    asyncio.run(run())
