"""Exercise the worker's benchmark timing path without FSDP/GPU dependencies."""

import __future__
import ast
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class Data:
    def __init__(self, meta_info):
        self.meta_info = meta_info

    def to(self, device):
        return self


class Timer:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.last = time.perf_counter() - self.started


class Sharding:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def preprocess_data(self, data):
        return data

    def postprocess_data(self, data):
        return data


class Config(dict):
    def __getattr__(self, key):
        return self[key]


@pytest.mark.parametrize("profile", [None, "qwen3-training-benchmark-v1"])
def test_worker_preserves_all_rank_timing_and_separate_teacher_max_only_in_benchmark(profile, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "verl/workers/fsdp_workers.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "update_actor")
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    gathered = []

    def gather(results, local):
        gathered.append(local)
        results[:] = [local, {
            **local, "rank": 1, "teacher_seconds": 0.2,
            "policy_update_seconds": local["policy_update_seconds"] + 0.3,
            "worker_update_seconds": local["worker_update_seconds"] + 0.5,
        }]

    device = SimpleNamespace(
        current_device=lambda: "cpu", max_memory_allocated=lambda: 1,
        max_memory_reserved=lambda: 2,
    )
    namespace = {
        "torch": torch, "Timer": Timer, "DataProto": Data,
        "dist": SimpleNamespace(all_gather_object=gather),
        "get_torch_device": lambda: device,
        "psutil": SimpleNamespace(
            virtual_memory=lambda: SimpleNamespace(used=1, percent=1), cpu_percent=lambda **kwargs: 1,
        ),
    }
    exec(compile(module, str(source), "exec", flags=__future__.annotations.compiler_flag), namespace)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: pytest.fail("CPU timing must not synchronize CUDA"))
    actor_metrics = {
        "perf/teacher_seconds": 0.0,
        "trainer/optimizer_steps_this_iteration": 2.0,
        "opd/ema_updates_this_iteration": 1.0,
        "opd/ema_update_count": 3.0,
    }
    worker = SimpleNamespace(
        opd_config=SimpleNamespace(prompt_profile=profile),
        _is_actor=True, _is_offload_param=False, _is_offload_optimizer=False,
        config=Config(rollout={}, actor=SimpleNamespace(ppo_epochs=1)),
        ulysses_sharding_manager=Sharding(),
        actor=SimpleNamespace(update_policy=lambda data: dict(actor_metrics)),
        flops_counter=SimpleNamespace(estimate_flops=lambda *args: (1, 1)),
        actor_lr_scheduler=SimpleNamespace(get_last_lr=lambda: [1e-6], step=lambda: None),
        world_size=2, rank=0,
    )
    result = namespace["update_actor"](worker, Data({"global_token_num": [1, 1]}))
    if profile is None:
        assert not gathered
        assert "actor_update_timing" not in result.meta_info
        assert "perf/teacher_seconds_max" not in result.meta_info["metrics"]
    else:
        timing = result.meta_info["actor_update_timing"]
        assert len(gathered) == 1
        assert [row["rank"] for row in timing["ranks"]] == [0, 1]
        assert timing["ranks"][0]["optimizer_steps"] == 2.0
        assert timing["ranks"][0]["ema_update_count"] == 3.0
        assert timing["teacher_seconds_max"] == 0.2
        assert result.meta_info["metrics"]["perf/teacher_seconds_max"] == 0.2
        assert timing["worker_update_seconds_max"] == timing["ranks"][1]["worker_update_seconds"]
        assert timing["timing_method"] == "cpu_wall"
