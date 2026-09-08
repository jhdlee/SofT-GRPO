"""Admission process isolation plus the real two-GPU integration entrypoint."""
import json
from pathlib import Path

import pytest
import torch

from verl.opd import qwen_lora_fsdp_probe as probe


@pytest.mark.skipif(torch.cuda.device_count() != 2, reason="requires exactly two NVIDIA GPUs")
def test_real_native_lora_fsdp_two_rank_admission():
    if torch.version.hip is not None or any(torch.cuda.get_device_capability(rank) != (9, 0) for rank in range(2)):
        pytest.skip("requires Hopper SM90")
    result = probe.validate_native_lora_fsdp_cuda()
    assert result["status"] == "passed" and result["world_size"] == 2
    assert [row["rank"] for row in result["ranks"]] == [0, 1]
    for row in result["ranks"]:
        assert row["optimizer_steps"] == 2 and row["dense_ema_updates"] == 1 and row["wrapper_count"] == 3
        assert all(row[field] for field in ("frozen_base_unchanged", "base_gradients_absent",
            "adapter_gradients_finite", "adapter_update_nonzero", "disabled_reference_exact",
            "dense_export_exact", "current_actor_detached"))


class Process:
    def __init__(self, alive):
        self.alive, self.terminated, self.killed = alive, False, False

    def is_alive(self): return self.alive
    def terminate(self): self.terminated = True
    def kill(self): self.killed, self.alive = True, False
    def join(self, timeout): pass


@pytest.mark.parametrize("outcome", ["complete", "timeout", "worker_failure"])
def test_probe_bounds_and_cleans_only_its_fresh_workers(monkeypatch, tmp_path, outcome):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda rank: (9, 0))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    workers = [Process(outcome != "complete") for _ in range(2)]
    class Context:
        processes = workers
        def join(self, timeout):
            assert timeout == 1
            if outcome == "worker_failure": raise RuntimeError("child failed")
            return outcome == "complete"
    def spawn(function, *, args, nprocs, join):
        assert function is probe._rank_probe and nprocs == 2 and join is False
        assert args[1] == str(Path(probe.__file__).resolve())
        for rank in range(2):
            (Path(args[0]) / f"rank-{rank}.json").write_text(json.dumps({"rank": rank}))
        return Context()
    monkeypatch.setattr(torch.multiprocessing, "spawn", spawn)
    times = iter([0, 121])
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(times))
    if outcome == "complete":
        assert probe.validate_native_lora_fsdp_cuda()["ranks"] == [{"rank": 0}, {"rank": 1}]
        assert not any(worker.terminated or worker.killed for worker in workers)
    else:
        with pytest.raises(TimeoutError if outcome == "timeout" else RuntimeError):
            probe.validate_native_lora_fsdp_cuda()
        assert all(worker.terminated and worker.killed for worker in workers)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("initialized,count,capability", [(True,2,(9,0)), (False,1,(9,0)), (False,4,(9,0)), (False,2,(8,0))])
def test_admission_refuses_shared_groups_wrong_topology_or_nonhopper(monkeypatch, initialized, count, capability):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: initialized)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda rank: capability)
    monkeypatch.setattr(torch.multiprocessing, "spawn", lambda *a, **k: pytest.fail("must not spawn"))
    with pytest.raises(RuntimeError):
        probe.validate_native_lora_fsdp_cuda()


def test_child_refuses_a_different_import_before_distributed_initialization():
    with pytest.raises(RuntimeError, match="different source module"):
        probe._rank_probe(0, "unused", "/different/installed/module.py")
