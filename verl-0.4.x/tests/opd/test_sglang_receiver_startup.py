"""Run actual native receiver/flush methods across the first engine resume."""

import __future__
import ast
import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_async_rollout_dispatch import dispatch


VERL_ROOT = Path(__file__).resolve().parents[2]
NATIVE_ROOT = VERL_ROOT.parent / "Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python/sglang/srt"


def _methods(path, class_name, names, namespace):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    assert len(methods) == len(names)
    for method in methods:
        method.decorator_list = []
    cls = ast.ClassDef(name=class_name, bases=[], keywords=[], body=methods, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace[class_name]


def _engine(*, flush_success=True, bootstrap=True):
    events = []

    async def exception_wrapper(function):
        await function()

    native = _methods(
        NATIVE_ROOT / "managers/tokenizer_manager.py", "TokenizerManager",
        {"auto_create_handle_loop", "handle_loop", "flush_cache", "release_memory_occupation", "resume_memory_occupation"},
        {
            "asyncio": asyncio, "time": time, "print_exception_wrapper": exception_wrapper,
            # Avoid installing a process signal handler in a CPU regression.
            "threading": SimpleNamespace(current_thread=lambda: "worker", main_thread=lambda: "main"),
            "logger": SimpleNamespace(warning=lambda *_: None),
            "FlushCacheReqInput": lambda: SimpleNamespace(kind="flush"),
        },
    )
    manager = native()
    manager.no_create_loop = False
    manager.asyncio_tasks = set()
    replies = asyncio.Queue()

    async def receive():
        events.append("receiver awaiting reply")
        return await replies.get()

    def deliver(item):
        kind, future = item
        events.append("received " + kind)
        if not future.done():
            future.set_result([SimpleNamespace(success=flush_success if kind == "flush" else True)])

    async def communicator(request):
        # The transport may queue a reply immediately, but only the native
        # handle_loop dispatches it and resolves the awaited communicator.
        events.append(("request", request.kind, manager.no_create_loop, len(manager.asyncio_tasks)))
        future = asyncio.get_running_loop().create_future()
        await replies.put((request.kind, future))
        return await future

    async def watchdog():
        await asyncio.Event().wait()

    manager.recv_from_detokenizer = SimpleNamespace(recv_pyobj=receive)
    manager._result_dispatcher = deliver
    manager.sigterm_watchdog = watchdog
    manager.flush_cache_communicator = communicator
    manager.release_memory_occupation_communicator = communicator
    manager.resume_memory_occupation_communicator = communicator
    if not bootstrap:
        # Reproduce the pre-fix first flush: no receiver has been started yet.
        manager.auto_create_handle_loop = lambda: None

    engine_class = _methods(
        VERL_ROOT / "verl/workers/rollout/sglang_rollout/sglang_rollout.py", "AsyncEngine",
        {"flush_cache", "release_memory_occupation", "resume_memory_occupation"},
        {
            "require_idle_engine": dispatch.require_idle_engine,
            "ReleaseMemoryOccupationReqInput": lambda: SimpleNamespace(kind="release"),
            "ResumeMemoryOccupationReqInput": lambda: SimpleNamespace(kind="resume"),
        },
    )
    engine = engine_class()
    engine.tokenizer_manager = manager
    engine._need_reload = True
    return engine, events


async def _close(engine):
    tasks = list(engine.tokenizer_manager.asyncio_tasks)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def test_first_resume_starts_native_reply_receiver_before_flush_request():
    async def run():
        engine, events = _engine()
        try:
            await asyncio.wait_for(engine.resume_memory_occupation(), timeout=1)
            requests = [event for event in events if isinstance(event, tuple)]
            assert requests == [
                ("request", "flush", True, 2),
                ("request", "release", True, 2),
                ("request", "resume", True, 2),
            ]
            assert "received flush" in events
            assert events.index("received flush") < events.index(requests[1])
            assert engine._need_reload is False
            # Native auto_create_handle_loop is idempotent; repeated flushes
            # do not create additional receiver/watchdog tasks.
            await asyncio.wait_for(engine.flush_cache(), timeout=1)
            assert len(engine.tokenizer_manager.asyncio_tasks) == 2
        finally:
            await _close(engine)

    asyncio.run(run())


def test_missing_native_receiver_reproduces_startup_stall_before_memory_pause():
    async def run():
        engine, events = _engine(bootstrap=False)
        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(engine.resume_memory_occupation(), timeout=0.05)
            assert events == [("request", "flush", False, 0)]
            assert engine._need_reload is True
        finally:
            await _close(engine)

    asyncio.run(run())


def test_busy_first_flush_still_prevents_native_memory_pause():
    async def run():
        engine, events = _engine(flush_success=False)
        try:
            with pytest.raises(RuntimeError, match="scheduler requests may remain outstanding"):
                await asyncio.wait_for(engine.resume_memory_occupation(), timeout=1)
            assert [event[1] for event in events if isinstance(event, tuple)] == ["flush"]
            assert engine._need_reload is True
        finally:
            await _close(engine)

    asyncio.run(run())


@pytest.mark.parametrize("guard", ["outstanding", "poisoned"])
def test_refused_engine_does_not_start_receiver_or_send_first_flush(guard):
    async def run():
        engine, events = _engine()
        if guard == "outstanding":
            engine._opd_batch_outstanding = True
        else:
            engine._opd_poisoned = "retired"
        with pytest.raises(RuntimeError, match="outstanding|poisoned"):
            await engine.resume_memory_occupation()
        assert not engine.tokenizer_manager.no_create_loop
        assert not engine.tokenizer_manager.asyncio_tasks
        assert not events

    asyncio.run(run())
