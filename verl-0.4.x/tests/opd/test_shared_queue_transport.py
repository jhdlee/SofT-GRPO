"""CPU-only checks for Ray queue lifecycle and opaque completion transport."""

from __future__ import annotations

import asyncio
import importlib.util
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


FORK_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FORK_ROOT))
ROLLOUT = FORK_ROOT / "verl-0.4.x/verl/workers/rollout/sglang_rollout"
PACKAGE = "opd_shared_queue_transport_tested"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROLLOUT)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(
    PACKAGE + ".shared_queue_transport", ROLLOUT / "shared_queue_transport.py"
)
transport = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = transport
spec.loader.exec_module(transport)


class ObjectRef:
    def __init__(self, value=None, error=None, delay=0):
        self.value = value
        self.error = error
        self.delay = delay

    def __await__(self):
        async def retrieve():
            await asyncio.sleep(self.delay)
            if self.error is not None:
                raise self.error
            return self.value

        return retrieve().__await__()


class ActorError(Exception):
    pass


class RemoteMethod:
    def __init__(self, actor, name):
        self.actor = actor
        self.name = name

    def remote(self, *args):
        # Ray resolves top-level ObjectRef arguments, but leaves nested refs
        # intact. Simulate this rule so accidental payload transfer is detected.
        resolved = tuple(arg.value if isinstance(arg, ObjectRef) else arg for arg in args)
        self.actor.calls.append((self.name, resolved))
        response = self.actor.responses.get(self.name)
        if isinstance(response, BaseException):
            return ObjectRef(error=response)
        return ObjectRef(response)


class Actor:
    def __init__(self):
        self.calls = []
        self.responses = {}
        for method in ("register", "claim", "complete", "fail", "status", "results_for"):
            setattr(self, method, RemoteMethod(self, method))


class FakeRay:
    ObjectRef = ObjectRef
    exceptions = SimpleNamespace(RayActorError=ActorError)

    def __init__(self):
        self.initialized = True
        self.actor = Actor()
        self.created = []
        self.killed = []
        self.put_calls = []
        self.get_calls = []
        self.kill_error = None

    def is_initialized(self):
        return self.initialized

    def remote(self, **options):
        def decorate(cls):
            def create(*args):
                self.created.append((cls, options, args))
                return self.actor

            return SimpleNamespace(remote=create)

        return decorate

    def kill(self, actor, *, no_restart):
        self.killed.append((actor, no_restart))
        if self.kill_error is not None:
            raise self.kill_error

    def put(self, output):
        self.put_calls.append((output, threading.get_ident()))
        return ObjectRef(output)

    def get(self, reference, *, timeout):
        self.get_calls.append((reference, timeout))
        if reference.error is not None:
            raise reference.error
        return reference.value


@pytest.fixture
def ray(monkeypatch):
    fake = FakeRay()
    monkeypatch.setitem(sys.modules, "ray", fake)
    return fake


def test_coordinator_is_zero_cpu_without_restart_or_task_retry(ray):
    actor = transport.create_shared_queue(4, 32)
    assert actor is ray.actor
    assert ray.created == [(
        transport.SharedRolloutQueue,
        {"num_cpus": 0, "max_restarts": 0, "max_task_retries": 0},
        (4, 32),
    )]


def test_coordinator_never_starts_an_implicit_cluster(ray):
    ray.initialized = False
    with pytest.raises(RuntimeError, match="initialized Ray cluster"):
        transport.create_shared_queue(4, 32)
    assert ray.created == []


def test_coordinator_normal_import_never_loads_model_or_ray_runtime():
    script = '''
import importlib.abc
import sys

sys.path.insert(0, sys.argv[1])
blocked = {"torch", "sglang", "verl", "ray"}

class BlockRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise AssertionError("coordinator imported model runtime: " + fullname)

sys.meta_path.insert(0, BlockRuntimeImports())
from opd_tools.shared_rollout_queue import SharedRolloutQueue, SharedRolloutError

assert SharedRolloutQueue.__module__ == "opd_tools.shared_rollout_queue"
assert SharedRolloutError.__module__ == "opd_tools.shared_rollout_queue"
assert SharedRolloutQueue(4, 32).status()["state"] == "registering"
assert not blocked.intersection(name.split(".", 1)[0] for name in sys.modules)
print("dependency-light coordinator import passed")
'''
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(FORK_ROOT)],
        check=True, capture_output=True, text=True, timeout=10,
    )
    assert result.stdout.strip() == "dependency-light coordinator import passed"


def test_all_queue_methods_await_rpc_results_and_preserve_arguments(ray):
    client = transport.RayQueueClient(ray.actor)
    requests = [{"sampling_params": {"n": 1, "seed": 123}}]
    policy = {"iteration": 2, "weight_version": "same-policy"}
    reference = [ObjectRef({"native": "payload"})]
    calls = [
        ("register", (2, requests, policy)),
        ("claim", (2,)),
        ("complete", (2, 7, reference)),
        ("fail", (2, "engine failed")),
        ("status", ()),
        ("results_for", (2,)),
    ]

    async def run():
        for name, args in calls:
            response = {"method": name}
            ray.actor.responses[name] = response
            assert await getattr(client, name)(*args) is response

    asyncio.run(run())
    assert ray.actor.calls == calls


def test_native_payload_bypasses_coordinator_and_returns_in_owner_order(ray):
    client = transport.RayQueueClient(ray.actor)
    payloads = [
        {"text": "answer", "meta_info": {"retained": [[True, False]], "seed": 91}},
        {"text": "other", "meta_info": {"tokens": [7, 2], "seed": 19}},
    ]

    async def run():
        refs = [await client.put_output(output) for output in payloads]
        # Complete in reverse order, as happens when engines have different
        # response lengths. The queue returns references in canonical order.
        await client.complete(1, 1, refs[1])
        await client.complete(0, 0, refs[0])
        coordinator_references = [call[1][2] for call in ray.actor.calls]
        assert coordinator_references == [refs[1], refs[0]]
        assert all(isinstance(wrapper[0], ObjectRef) for wrapper in coordinator_references)
        ray.actor.responses["results_for"] = [
            {"payload_reference": refs[0]}, {"payload_reference": refs[1]}
        ]
        rows = await client.results_for(0)
        outputs = await client.get_outputs([row["payload_reference"] for row in rows])
        assert outputs == payloads
        assert all(result is original for result, original in zip(outputs, payloads))
        assert client._output_refs == [wrapper[0] for wrapper in refs]

    asyncio.run(run())


def test_object_store_put_does_not_block_the_event_loop(ray):
    client = transport.RayQueueClient(ray.actor)
    started = threading.Event()
    release = threading.Event()
    event_loop_thread = threading.get_ident()

    def slow_put(output):
        ray.put_calls.append((output, threading.get_ident()))
        started.set()
        if not release.wait(timeout=2):
            raise RuntimeError("event loop did not release the object-store writer")
        return ObjectRef(output)

    ray.put = slow_put

    async def run():
        task = asyncio.create_task(client.put_output({"large": "native completion"}))
        try:
            while not started.is_set():
                await asyncio.sleep(0.001)
            assert not task.done()
            release.set()
            assert isinstance((await task)[0], ObjectRef)
        finally:
            release.set()

    asyncio.run(asyncio.wait_for(run(), timeout=3))
    assert ray.put_calls[0][1] != event_loop_thread


@pytest.mark.parametrize("method,args", [("claim", (1,)), ("status", ()), ("results_for", (0,))])
def test_remote_failure_propagates_without_hidden_retries(ray, method, args):
    ray.actor.responses[method] = ActorError("coordinator failed")
    client = transport.RayQueueClient(ray.actor)
    with pytest.raises(ActorError, match="coordinator failed"):
        asyncio.run(getattr(client, method)(*args))
    assert ray.actor.calls == [(method, args)]


@pytest.mark.parametrize("wrapped", [ObjectRef("naked"), [], ["payload"], [ObjectRef(1), ObjectRef(2)]])
def test_invalid_wrappers_never_send_payload_to_coordinator(ray, wrapped):
    client = transport.RayQueueClient(ray.actor)
    with pytest.raises(TypeError, match="wrapped"):
        asyncio.run(client.complete(0, 0, wrapped))
    assert ray.actor.calls == []
    with pytest.raises(TypeError, match="wrapped"):
        asyncio.run(client.get_outputs([wrapped]))


def test_missing_object_fails_owner_fetch(ray):
    client = transport.RayQueueClient(ray.actor)
    with pytest.raises(RuntimeError, match="object lost"):
        asyncio.run(client.get_outputs([[ObjectRef(error=RuntimeError("object lost"))]]))
    assert asyncio.run(client.get_outputs([])) == []


def test_cleanup_kills_only_coordinator_and_tolerates_already_dead_actor(ray):
    transport.close_shared_queue(None)
    assert ray.killed == []
    transport.close_shared_queue(ray.actor)
    assert ray.killed == [(ray.actor, True)]
    ray.kill_error = ActorError("already dead")
    transport.close_shared_queue(ray.actor)
    assert ray.killed == [(ray.actor, True), (ray.actor, True)]
    ray.kill_error = RuntimeError("unexpected control-plane failure")
    with pytest.raises(RuntimeError, match="control-plane"):
        transport.close_shared_queue(ray.actor)


def test_adapter_abort_notifies_peers_without_a_running_asyncio_loop(ray):
    transport.abort_shared_queue(None, 1, "no coordinator yet")
    assert ray.get_calls == []
    transport.abort_shared_queue(ray.actor, 1, "event loop unavailable")
    assert ray.actor.calls == [("fail", (1, "event loop unavailable"))]
    assert ray.get_calls[0][1] == 10
    assert ray.killed == []


def test_adapter_abort_exposes_notification_failure_for_collective_cleanup(ray):
    ray.actor.responses["fail"] = ActorError("coordinator unavailable")
    with pytest.raises(ActorError, match="coordinator unavailable"):
        transport.abort_shared_queue(ray.actor, 2, "engine unavailable")
    assert ray.actor.calls == [("fail", (2, "engine unavailable"))]
    assert ray.get_calls[0][1] == 10
