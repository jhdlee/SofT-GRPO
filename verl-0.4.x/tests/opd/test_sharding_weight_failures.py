"""Exercise actual manager phases with two concurrent CPU rank stand-ins."""

import __future__
import ast
import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_async_rollout_dispatch import dispatch


PATH = Path(__file__).resolve().parents[2] / "verl/workers/sharding_manager/fsdp_sglang.py"


class Collectives:
    def __init__(self, world=2):
        self.world = world
        self.condition = threading.Condition()
        self.slots = {}
        self.counters = [0] * world

    def exchange(self, rank, label, value):
        with self.condition:
            index = self.counters[rank]
            self.counters[rank] += 1
            slot = self.slots.setdefault(index, {})
            slot[rank] = (label, value)
            self.condition.notify_all()
            if not self.condition.wait_for(lambda: len(slot) == self.world, timeout=3):
                raise AssertionError(f"unmatched collective {index}: {slot}")
            assert len({item[0] for item in slot.values()}) == 1, f"phase mismatch: {slot}"
            return [slot[peer][1] for peer in range(self.world)]


def _manager(rank, collectives, *, failure=None, failure_rank=1, tp=1):
    events, stages = [], []

    def fail(stage):
        if failure == stage and rank == failure_rank:
            raise RuntimeError("injected " + stage)

    class Dist:
        phase = "inventory"

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size():
            return collectives.world

        @classmethod
        def all_gather_object(cls, output, value):
            output[:] = collectives.exchange(rank, cls.phase, value)

        @staticmethod
        def gather_object(obj, object_gather_list, dst, group):
            events.append("tp gather")
            if tp == 1:
                object_gather_list[:] = [obj]
            else:
                values = collectives.exchange(rank, "tp gather", obj)
                if rank == dst:
                    object_gather_list[:] = values

    def check(engine, distributed, error, stage):
        stages.append(stage)
        previous = Dist.phase
        Dist.phase = stage
        try:
            dispatch.check_collective_error(engine, distributed, error, stage)
        finally:
            Dist.phase = previous

    class Engine:
        shutdown_calls = 0
        _opd_batch_outstanding = failure == "outstanding" and rank == failure_rank

        async def resume_memory_occupation(self):
            events.append("resume")
            fail("resume")

        async def update_weights_from_tensor(self, **kwargs):
            name = kwargs["named_tensors"][0][0]
            events.append(("update", name, kwargs["flush_cache"]))
            fail("update")

        async def release_memory_occupation(self):
            events.append("release")
            fail("release")

        def shutdown(self):
            self.shutdown_calls += 1

    def state_dict():
        events.append("state dict")
        fail("state dict")
        names = ("weight_a",) if failure == "inventory" and rank == failure_rank else ("weight_a", "weight_b")
        return {name: (rank, name) for name in names}

    def serialize(tensor):
        events.append("serialize")
        fail("serialization")
        return tensor

    def materialize(tensor):
        events.append("materialize")
        fail("materialization")
        return tensor

    def load(_):
        events.append("load")
        fail("load")

    def offload(_):
        events.append("offload")
        fail("offload")

    tree = ast.parse(PATH.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FSDPSGLangShardingManager")
    cls.bases = []
    cls.body = [node for node in cls.body if not isinstance(node, ast.FunctionDef) or node.name != "__init__"]
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = []
    namespace = {
        "asyncio": asyncio, "time": time, "dist": Dist,
        "check_collective_error": check, "poison_engine": dispatch.poison_engine,
        "require_idle_engine": dispatch.require_idle_engine,
        "logger": None, "log_gpu_memory_usage": lambda *_, **__: None,
        "load_fsdp_model_to_gpu": load, "offload_fsdp_model_to_cpu": offload,
        "fsdp_version": lambda _: 1, "_preprocess_tensor_for_update_weights": materialize,
        "MultiprocessingSerializer": SimpleNamespace(serialize=serialize),
        "LocalSerializedTensor": lambda **kwargs: kwargs,
        "torch": SimpleNamespace(cuda=SimpleNamespace(
            empty_cache=lambda: None, current_device=lambda: rank,
            get_rng_state=lambda: "actor rng", set_rng_state=lambda state: events.append(("rng", state)),
        )),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(PATH), "exec", flags=__future__.annotations.compiler_flag), namespace)
    manager = namespace[cls.name]()
    manager.inference_engine = Engine()
    manager.module = SimpleNamespace(state_dict=state_dict, train=lambda: events.append("train"))
    manager.offload_param = True
    manager._opd_poisoned = None
    manager._opd_rng_switched = False
    manager.last_rollout_timing = {}
    manager.torch_random_states, manager.gen_random_states = "actor rng", "generation rng"
    members = [rank] if tp == 1 else list(range(tp))
    manager.device_mesh = {"infer_tp": SimpleNamespace(
        get_local_rank=lambda: 0 if tp == 1 else rank,
        mesh=SimpleNamespace(size=lambda: (tp,), tolist=lambda: members),
        get_group=lambda: tuple(members),
    )}
    manager.events, manager.stages = events, stages
    return manager


def _run_pair(operation, *, failure=None, failure_rank=1, tp=1):
    collectives = Collectives()
    managers = [_manager(rank, collectives, failure=failure, failure_rank=failure_rank, tp=tp) for rank in range(2)]
    errors = [None, None]

    def run(rank):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            operation(managers[rank], loop)
        except BaseException as error:
            errors[rank] = error
        finally:
            loop.close()

    workers = [threading.Thread(target=run, args=(rank,), daemon=True) for rank in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)
    assert not any(worker.is_alive() for worker in workers), "manager phases deadlocked"
    assert managers[0].stages == managers[1].stages
    return managers, errors


@pytest.mark.parametrize("entry", ["sync", "async"])
@pytest.mark.parametrize("failure", ["load", "state dict", "resume", "inventory", "materialization", "serialization", "update", "offload", "outstanding"])
def test_one_dp_failure_stops_both_ranks_before_generation(entry, failure):
    def operation(manager, loop):
        if entry == "sync":
            manager.__enter__()
        else:
            loop.run_until_complete(manager.wake_up())
        manager.events.append("generation")

    managers, errors = _run_pair(operation, failure=failure)
    for manager, error in zip(managers, errors):
        assert isinstance(error, RuntimeError), repr(error)
        assert "collective" in str(error)
        assert manager._opd_poisoned
        assert manager.inference_engine.shutdown_calls == 1
        assert "generation" not in manager.events
        assert "release" not in manager.events
        if failure != "offload":
            assert ("update", "weight_b", True) not in manager.events
        if failure in {"load", "state dict", "resume", "inventory", "materialization", "serialization", "outstanding"}:
            assert not any(isinstance(event, tuple) and event[0] == "update" for event in manager.events)


@pytest.mark.parametrize("tp", [1, 2])
def test_success_preserves_tp_gathers_and_flushes_only_final_weight(tp):
    def operation(manager, loop):
        manager.__enter__()
        manager.events.append("generation")
        manager.__exit__(None, None, None)

    managers, errors = _run_pair(operation, tp=tp)
    assert errors == [None, None]
    for rank, manager in enumerate(managers):
        assert manager.events.count("tp gather") == 2
        updates = [event for event in manager.events if isinstance(event, tuple) and event[0] == "update"]
        assert updates == ([("update", "weight_a", False), ("update", "weight_b", True)] if tp == 1 or rank == 0 else [])
        assert not manager._opd_poisoned
        assert not manager._opd_rng_switched
        assert manager.last_rollout_timing["weight_sync_seconds"] >= 0


@pytest.mark.parametrize("failure", ["serialization", "update"])
def test_tp_peer_failure_exits_at_matching_global_phase(failure):
    def operation(manager, loop):
        manager.__enter__()

    managers, errors = _run_pair(operation, failure=failure, failure_rank=0, tp=2)
    assert all(isinstance(error, RuntimeError) and "collective" in str(error) for error in errors)
    assert all(manager._opd_poisoned for manager in managers)
    if failure == "serialization":
        assert all("tp gather" not in manager.events for manager in managers)


@pytest.mark.parametrize("failure", ["release", "outstanding"])
def test_release_refuses_live_or_failed_scheduler_on_every_rank(failure):
    def operation(manager, loop):
        loop.run_until_complete(manager.release_memory())

    managers, errors = _run_pair(operation, failure=failure)
    assert all(isinstance(error, RuntimeError) and "collective" in str(error) for error in errors)
    assert all(manager._opd_poisoned for manager in managers)
    if failure == "outstanding":
        assert all("release" not in manager.events for manager in managers)


def test_direct_update_refuses_outstanding_generation_before_resume():
    def operation(manager, loop):
        loop.run_until_complete(manager.update_weights({"weight_a": 1}))

    managers, errors = _run_pair(operation, failure="outstanding")
    assert all(isinstance(error, RuntimeError) and "outstanding" in str(error) for error in errors)
    assert all("resume" not in manager.events for manager in managers)


def test_poisoned_manager_refuses_later_update_and_release_collectively():
    def operation(manager, loop):
        with pytest.raises(RuntimeError, match="injected resume"):
            manager.__enter__()
        with pytest.raises(RuntimeError, match="poisoned"):
            loop.run_until_complete(manager.update_weights({"weight_a": 1}))
        with pytest.raises(RuntimeError, match="poisoned"):
            loop.run_until_complete(manager.release_memory())

    managers, errors = _run_pair(operation, failure="resume")
    assert errors == [None, None]
    for manager in managers:
        assert manager.events.count("resume") == 1
        assert "release" not in manager.events
        assert manager.inference_engine.shutdown_calls == 1


def test_failure_of_error_collective_itself_poisons_local_manager(monkeypatch):
    manager = _manager(0, Collectives())
    distributed = manager._finish_stage.__globals__["dist"]

    def broken_collective(*_):
        raise RuntimeError("collective backend disconnected")

    monkeypatch.setattr(distributed, "all_gather_object", broken_collective)
    with pytest.raises(RuntimeError, match="backend disconnected"):
        manager._guard_stage("local readiness", lambda: None)
    assert manager._opd_poisoned
    assert manager.inference_engine.shutdown_calls == 1
    assert "release" not in manager.events
