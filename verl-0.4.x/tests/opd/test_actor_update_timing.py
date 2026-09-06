"""Exercise the worker's benchmark timing path without FSDP/GPU dependencies."""

import __future__
import ast
import copy
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class Data:
    def __init__(self, meta_info=None, tensors=None):
        self.meta_info = meta_info or {}
        self.batch = tensors or {}

    @classmethod
    def from_dict(cls, tensors, meta_info):
        return cls(meta_info=meta_info, tensors=tensors)

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


def load_worker_method(name, namespace):
    source = Path(__file__).resolve().parents[2] / "verl/workers/fsdp_workers.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(source), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


@pytest.mark.parametrize("profile", [None, "qwen3-training-benchmark-v1"])
@pytest.mark.parametrize("standalone", [False, True])
@pytest.mark.parametrize("sampling", [
    {"temperature": 1.0, "add_noise_dirichlet": False, "add_noise_gumbel_softmax": True},
    {"temperature": 0.7, "add_noise_dirichlet": False, "add_noise_gumbel_softmax": False,
     "enable_soft_thinking": False},
    {"temperature": 0.85, "add_noise_dirichlet": True, "add_noise_gumbel_softmax": False,
     "enable_soft_thinking": True},
])
def test_worker_preserves_sampling_contract_and_rank_timing(profile, standalone, sampling, monkeypatch):
    gathered = []

    def gather(results, local):
        gathered.append(local)
        results[:] = [local, {
            **local, "rank": 1, "teacher_seconds": 0.2,
            "policy_update_seconds": local["policy_update_seconds"] + 0.3,
            "worker_update_seconds": local["worker_update_seconds"] + 0.5,
            "max_memory_allocated_gib": 17.0,
            "max_memory_reserved_gib": 19.0,
        }]

    device = SimpleNamespace(
        current_device=lambda: "cpu", max_memory_allocated=lambda: 11 * 1024**3,
        max_memory_reserved=lambda: 13 * 1024**3,
    )
    namespace = {
        "torch": torch, "Timer": Timer, "DataProto": Data,
        "dist": SimpleNamespace(all_gather_object=gather),
        "get_torch_device": lambda: device,
        "psutil": SimpleNamespace(
            virtual_memory=lambda: SimpleNamespace(used=1, percent=1), cpu_percent=lambda **kwargs: 1,
        ),
    }
    update_actor = load_worker_method("update_actor", namespace)
    compute_log_prob = load_worker_method("compute_log_prob", namespace)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: pytest.fail("CPU timing must not synchronize CUDA"))
    actor_metrics = {
        "perf/teacher_seconds": 0.0,
        "trainer/optimizer_steps_this_iteration": 2.0,
        "opd/ema_updates_this_iteration": 1.0,
        "opd/ema_update_count": 3.0,
    }
    expected_sampling = {
        key: sampling[key] for key in ("temperature", "add_noise_dirichlet", "add_noise_gumbel_softmax")
    }
    expected_sampling["continuous_replay"] = sampling.get("enable_soft_thinking", True)
    driver_meta = {"global_token_num": [1, 1], "rollout_iteration": 7, "opd_beta": 1.0}
    # A recompute RPC receives a serialized copy.  Neither objective may rely
    # on mutations of that copy reaching the next update RPC.
    driver_data = Data(meta_info=copy.deepcopy(driver_meta))

    def assert_sampling(data):
        assert {key: data.meta_info[key] for key in expected_sampling} == expected_sampling
        assert {key: data.meta_info[key] for key in driver_meta} == driver_meta

    def recompute(data, calculate_entropy):
        assert calculate_entropy
        assert_sampling(data)
        return torch.zeros(2, 1), torch.ones(2, 1)

    def update(data):
        assert_sampling(data)
        # Standalone must acquire sampling metadata without importing the old
        # policy tensors that the objective intentionally excludes.
        assert ("old_log_probs" in data.batch) == (not standalone)
        return dict(actor_metrics)

    worker = SimpleNamespace(
        opd_config=SimpleNamespace(prompt_profile=profile),
        _is_actor=True, _is_offload_param=False, _is_offload_optimizer=False,
        config=Config(rollout=Config(
            **sampling, log_prob_micro_batch_size_per_gpu=2,
            log_prob_max_token_len_per_gpu=64, log_prob_use_dynamic_bsz=False,
        ), actor=SimpleNamespace(ppo_epochs=1)),
        ulysses_sharding_manager=Sharding(),
        actor=SimpleNamespace(update_policy=update, compute_log_prob=recompute, actor_module=SimpleNamespace()),
        flops_counter=SimpleNamespace(estimate_flops=lambda *args: (1, 1)),
        actor_lr_scheduler=SimpleNamespace(get_last_lr=lambda: [1e-6], step=lambda: None),
        world_size=1, rank=0,
    )
    replay_result = compute_log_prob(worker, copy.deepcopy(driver_data))
    assert driver_data.meta_info == driver_meta
    if not standalone:
        driver_data.batch.update(replay_result.batch)
        driver_data.meta_info.update(replay_result.meta_info)
        # Config remains authoritative, as it is in recompute/reference
        # scoring: stale caller metadata cannot select a different density.
        driver_data.meta_info.update({
            "temperature": 9.0, "add_noise_dirichlet": not sampling["add_noise_dirichlet"],
            "add_noise_gumbel_softmax": not sampling["add_noise_gumbel_softmax"],
            "continuous_replay": not expected_sampling["continuous_replay"],
        })
    worker.world_size = 2
    result = update_actor(worker, driver_data)
    if profile is None:
        assert not gathered
        assert "actor_update_timing" not in result.meta_info
        assert "perf/teacher_seconds_max" not in result.meta_info["metrics"]
        assert result.meta_info["metrics"]["perf/max_memory_allocated_gb"] == 11.0
        assert result.meta_info["metrics"]["perf/max_memory_reserved_gb"] == 13.0
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
        # Keep the raw rank inventory while publishing the actual larger
        # rank-one peaks before DataProto.concat discards its metadata.
        assert [row["max_memory_allocated_gib"] for row in timing["ranks"]] == [11.0, 17.0]
        assert [row["max_memory_reserved_gib"] for row in timing["ranks"]] == [13.0, 19.0]
        assert timing["max_memory_allocated_gib"] == 17.0
        assert timing["max_memory_reserved_gib"] == 19.0
        assert result.meta_info["metrics"]["perf/max_memory_allocated_gb"] == 17.0
        assert result.meta_info["metrics"]["perf/max_memory_reserved_gb"] == 19.0
        assert "PyTorch process-lifetime" in timing["memory_scope"]
