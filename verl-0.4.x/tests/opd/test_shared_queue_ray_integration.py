"""Opt-in, real Ray/Gloo transport checks with four CPU rollout workers.

Run with OPD_RUN_SHARED_QUEUE_RAY_TEST=1 on an allocated CPU node. The actual
adapter methods, coordinator actor, nested ObjectRefs, and Gloo collectives are
used; only the inference engine and response tensors are small CPU fixtures.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import socket
import sys
import tempfile
import types
from datetime import timedelta
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("OPD_RUN_SHARED_QUEUE_RAY_TEST") != "1",
    reason="opt-in real CPU Ray/Gloo integration",
)
TEST_DIRECTORY = Path(__file__).resolve().parent
FORK_ROOT = TEST_DIRECTORY.parents[2]
sys.path.insert(0, str(FORK_ROOT))
ROLLOUT = TEST_DIRECTORY.parents[1] / "verl/workers/rollout/sglang_rollout"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_transport():
    from opd_tools.shared_rollout_queue import SharedRolloutQueue

    package_name = "opd_shared_queue_real_ray"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROLLOUT)]
    sys.modules[package_name] = package
    transport = _load_module(package_name + ".shared_queue_transport", ROLLOUT / "shared_queue_transport.py")
    # Only the local inference consumer bypasses SGLang's eager package init.
    # Fresh coordinator actors import the real dependency-light module normally.
    core = _load_module(package_name + ".shared_queue", ROLLOUT / "shared_queue.py")
    assert transport.SharedRolloutQueue is SharedRolloutQueue
    assert SharedRolloutQueue.__module__ == "opd_tools.shared_rollout_queue"
    return transport, core


class RayAdapterWorker:
    def run(self, rank, world_size, init_method, failure_mode):
        import numpy as np
        import torch
        import torch.distributed as dist

        torch.set_num_threads(1)
        fixture = _load_module("opd_actual_adapter_ray_fixture", TEST_DIRECTORY / "test_async_rollout_dispatch.py")
        transport, core = _load_transport()
        inject_failure = failure_mode != "success"
        expected_failure = {
            "engine": "injected real-Ray engine failure",
            "client_init": "injected real-Ray client initialization failure",
            "result_transport": "injected real-Ray result transport failure",
        }.get(failure_mode)
        created = []
        closed = []
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size,
                                timeout=timedelta(seconds=60))

        def create_queue(world, capacity):
            actor = transport.create_shared_queue(world, capacity)
            created.append(actor)
            return actor

        def close_queue(actor):
            transport.close_shared_queue(actor)
            closed.append(actor)

        class FailureAwareClient(transport.RayQueueClient):
            def __init__(self, actor):
                if failure_mode == "client_init" and rank == 3:
                    raise RuntimeError(expected_failure)
                super().__init__(actor)

            async def get_outputs(self, references):
                if failure_mode == "result_transport" and rank == 3:
                    # The core invokes transport only after all responses have
                    # completed. Other ranks can already be at the collective.
                    status = await self.status()
                    assert status["state"] == "done"
                    raise OSError(expected_failure)
                return await super().get_outputs(references)

        class TimedEngine(fixture.Engine):
            def __init__(self, *, failure=False, delay=0):
                super().__init__()
                self.failure = failure
                self.delay = delay
                self.submitted = 0
                self.cancelled = 0

            async def async_generate(self, **kwargs):
                self.submitted += 1
                call_number = self.submitted
                try:
                    await asyncio.sleep(self.delay)
                    if self.failure and call_number == 1:
                        raise ValueError("injected real-Ray engine failure")
                    return await super().async_generate(**kwargs)
                except asyncio.CancelledError:
                    assert self.shutdown_calls > 0, "backend must retire before frontend cancellation"
                    self.cancelled += 1
                    raise

        def adapter_for(mode, engine):
            adapter = fixture.load_adapter()
            adapter.config.update(dispatch_mode=mode, async_queue_size=2, max_running_requests=2,
                                  tensor_model_parallel_size=1, mode="sync", multi_turn={"enable": False})
            adapter._rank = rank
            adapter._tp_rank = 0
            adapter._tp_size = 1
            adapter._device_mesh_cpu = {"tp": types.SimpleNamespace(
                get_group=lambda: None, mesh=torch.tensor([rank]),
            )}
            adapter._engine = engine
            adapter._test_namespace.update(
                dist=dist, create_shared_queue=create_queue, close_shared_queue=close_queue,
                abort_shared_queue=transport.abort_shared_queue, RayQueueClient=FailureAwareClient,
                dispatch_shared_generation=core.dispatch_shared_generation,
                expanded_requests=fixture.dispatch.expanded_requests,
            )
            return adapter

        def owned_prompts():
            batch = fixture.prompts()
            batch.non_tensor_batch["index"] += rank * 1000
            return batch

        try:
            reference = None
            if not inject_failure:
                reference = adapter_for("bounded_async", TimedEngine())._batch_level_generate_sequences(owned_prompts())
            # The slow rank keeps its initial two requests while faster ranks
            # drain the remaining common pool. No response lengths are known.
            engine = TimedEngine(failure=failure_mode == "engine" and rank == 3,
                                 delay=0.3 if rank == 3 else 0.001 * (rank + 1))
            adapter = adapter_for("shared_queue", engine)
            try:
                result = adapter._batch_level_generate_sequences(owned_prompts())
            except BaseException as error:
                if not inject_failure:
                    raise
                assert expected_failure in str(error), str(error)
                assert engine.shutdown_calls == 1
                assert engine.flush_calls == 0
                assert not getattr(engine, "_opd_batch_outstanding", False)
                summary = {"rank": rank, "failed": True, "error": str(error)[:1000],
                           "shutdown_calls": engine.shutdown_calls, "cancelled": engine.cancelled}
            else:
                assert not inject_failure, "injected failure must discard the whole batch"
                assert set(result.batch) == set(reference.batch)
                for name, value in reference.batch.items():
                    if name != "rollout_rank":
                        assert torch.equal(value, result.batch[name]), name
                for name, value in reference.non_tensor_batch.items():
                    assert np.array_equal(value, result.non_tensor_batch[name]), name
                assert engine.shutdown_calls == 0
                assert engine.flush_calls == 1
                assert not engine._opd_batch_outstanding
                summary = {
                    "rank": rank, "failed": False, "submitted": engine.submitted,
                    "seed_rows": result.batch["rollout_sampling_seed"].tolist(),
                    "execution_ranks": result.batch["rollout_rank"].tolist(),
                    "peak_pending": result.meta_info["rollout_timing"]["frontend_peak_pending"],
                    "owned_count": result.meta_info["rollout_timing"]["request_count"],
                    "executed_count": result.meta_info["rollout_timing"]["shared_queue_executed_request_count"],
                }
            assert len(created) == len(closed) == (1 if rank == 0 else 0)
            summary["created_actors"] = created
            summary["closed_actor_count"] = len(closed)
            # Test the matched post-rollout phase as well: a hanging generation
            # or cleanup collective must prevent this barrier and fail the test.
            dist.barrier()
            return summary
        finally:
            dist.destroy_process_group()
            loop.close()
            asyncio.set_event_loop(None)


@pytest.fixture(scope="module")
def real_ray():
    import ray

    if ray.is_initialized():
        pytest.fail("integration requires its own isolated local Ray runtime")
    # Ray's socket paths must be short; these are ephemeral local processes.
    temporary = tempfile.TemporaryDirectory(prefix="opdq-", dir="/tmp")
    pythonpath = os.pathsep.join((str(TEST_DIRECTORY), str(FORK_ROOT)))
    if os.environ.get("PYTHONPATH"):
        pythonpath += os.pathsep + os.environ["PYTHONPATH"]
    try:
        ray.init(
            num_cpus=4, num_gpus=0, include_dashboard=False,
            object_store_memory=512 * 1024 * 1024, _temp_dir=temporary.name,
            runtime_env={"env_vars": {"PYTHONPATH": pythonpath, "CUDA_VISIBLE_DEVICES": "",
                                       "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}},
        )
        yield ray
    finally:
        ray.shutdown()
        temporary.cleanup()


@pytest.mark.parametrize("failure_mode", ["success", "engine", "client_init", "result_transport"], ids=[
    "native-results-and-order", "engine-failure-and-cleanup",
    "client-failure-before-registration", "owner-transport-failure-after-generation",
])
def test_four_real_ray_workers_complete_actual_adapter_collectives(real_ray, failure_mode):
    ray = real_ray
    inject_failure = failure_mode != "success"
    host = ray.util.get_node_ip_address()
    with socket.socket() as listener:
        listener.bind((host, 0))
        port = listener.getsockname()[1]
    init_method = f"tcp://{host}:{port}"
    worker_class = ray.remote(num_cpus=1, num_gpus=0, max_restarts=0)(RayAdapterWorker)
    workers = [worker_class.remote() for _ in range(4)]
    try:
        summaries = ray.get([
            worker.run.remote(rank, 4, init_method, failure_mode)
            for rank, worker in enumerate(workers)
        ], timeout=180)
        assert [item["rank"] for item in summaries] == list(range(4))
        assert all(item["failed"] == inject_failure for item in summaries)
        if not inject_failure:
            assert sum(item["submitted"] for item in summaries) == 64
            assert sum(item["executed_count"] for item in summaries) == 64
            assert all(item["owned_count"] == 16 for item in summaries)
            assert all(item["peak_pending"] <= 2 for item in summaries)
            assert max(item["submitted"] for item in summaries[:3]) > summaries[3]["submitted"]
            execution_ranks = [rank for item in summaries for rank in item["execution_ranks"]]
            assert len(execution_ranks) == 64
            assert set(execution_ranks) == set(range(4))
            seeds = [seed for item in summaries for seed in item["seed_rows"]]
            assert len(seeds) == len(set(seeds)) == 64
        assert sum(item["closed_actor_count"] for item in summaries) == 1
        coordinator = summaries[0]["created_actors"][0]
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(coordinator.status.remote(), timeout=30)
    finally:
        for worker in workers:
            ray.kill(worker, no_restart=True)
