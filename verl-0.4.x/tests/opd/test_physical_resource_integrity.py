"""Physical/virtual memory separation and real worker transition ordering."""
import copy
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from verl.opd.metrics import validate_physical_resource_limits
from verl.opd.resource_integrity import PhysicalMemorySampler, ResourceGuard, normalize_resource_policy
from test_actor_update_timing import Config, Data, Sharding, Timer, load_worker_method

POLICY = {"mode": "physical_device_v1", "max_device_used_fraction": .98, "sample_interval_seconds": .1}


@pytest.mark.parametrize("value", [None, {}])
def test_absent_policy_preserves_legacy(value):
    assert normalize_resource_policy(value) is None


@pytest.mark.parametrize("changes", [{"mode": "off"}, {"max_device_used_fraction": 1},
    {"max_device_used_fraction": float("nan")}, {"max_device_used_fraction": True},
    {"sample_interval_seconds": 0}, {"sample_interval_seconds": float("inf")}, {"extra": 1}])
def test_policy_rejects_unbounded_or_ambiguous_settings(changes):
    with pytest.raises(ValueError): normalize_resource_policy({**POLICY, **changes})


def test_sampler_observes_transient_pressure_between_entry_and_exit():
    observed = threading.Event(); calls = 0
    def read():
        nonlocal calls
        calls += 1
        if calls == 2:
            observed.set()
            return 1, 100
        return 60, 100
    sampler = PhysicalMemorySampler(read, .01).start()
    assert observed.wait(1)
    result = sampler.stop()
    assert result["sample_count"] >= 3
    assert result["device_used_peak_bytes"] == 99
    assert result["start_free_bytes"] == result["final_free_bytes"] == 60
    assert sampler.stop() == result and not sampler.thread.is_alive()
    with pytest.raises(RuntimeError, match="physical-device resource gate"):
        validate_physical_resource_limits(result, max_device_used_fraction=.98)


@pytest.mark.parametrize("value", [(float("nan"), 100), (1, 0), (-1, 100), (101, 100), (True, 100)])
def test_sensor_values_fail_closed(value):
    with pytest.raises(RuntimeError, match="physical-device resource gate"):
        PhysicalMemorySampler(lambda: value, .01).start()


def test_short_sampler_requires_final_sample_and_stable_capacity():
    values = iter([(50, 100), (60, 101)])
    sampler = PhysicalMemorySampler(lambda: next(values), 1).start()
    with pytest.raises(RuntimeError, match="changing device capacity"): sampler.stop()
    assert not sampler.thread.is_alive()


def test_background_sensor_error_is_not_silently_ignored():
    failed = threading.Event(); count = 0
    def read():
        nonlocal count
        count += 1
        if count > 1:
            failed.set(); raise RuntimeError("sensor unavailable")
        return 40, 100
    sampler = PhysicalMemorySampler(read, .01).start()
    assert failed.wait(1)
    with pytest.raises(RuntimeError, match="sampling failed.*sensor unavailable"): sampler.stop()
    assert not sampler.thread.is_alive()


class Exchanges:
    def __init__(self):
        self.condition = threading.Condition(); self.rounds = {}; self.indices = [0, 0]

    def rank(self, rank):
        def gather(output, value):
            with self.condition:
                index = self.indices[rank]; self.indices[rank] += 1
                slot = self.rounds.setdefault(index, {}); slot[rank] = copy.deepcopy(value)
                self.condition.notify_all()
                if not self.condition.wait_for(lambda: len(slot) == 2, timeout=2):
                    raise RuntimeError("unmatched resource collective")
                output[:] = [slot[0], slot[1]]
        return SimpleNamespace(is_initialized=lambda: True, get_world_size=lambda: 2, all_gather_object=gather)


def run_ranks(operations):
    errors = [None, None]; outputs = [None, None]
    def run(rank):
        try: outputs[rank] = operations[rank]()
        except BaseException as error: errors[rank] = error
    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join(4)
    assert not any(thread.is_alive() for thread in threads), "resource failure stranded a peer"
    return outputs, errors


def worker(rank, collective, events, *, policy=POLICY, metrics_error=False, profile=None):
    device = SimpleNamespace(current_device=lambda: rank,
        max_memory_allocated=lambda: float("nan") if metrics_error else 88.7 * 1024**3,
        max_memory_reserved=lambda: 95. * 1024**3)
    namespace = {"torch": torch, "Timer": Timer, "DataProto": Data, "dist": collective.rank(rank),
        "get_torch_device": lambda: device,
        "psutil": SimpleNamespace(virtual_memory=lambda: SimpleNamespace(used=1, percent=96.),
                                  cpu_percent=lambda **kwargs: 25.),
        "load_fsdp_model_to_gpu": lambda model: events.append((rank, "load")),
        "offload_fsdp_model_to_cpu": lambda model: events.append((rank, "offload")),
        "log_gpu_memory_usage": lambda *args, **kwargs: None, "logger": None}
    method = load_worker_method("update_actor", namespace)
    def update(data):
        events.append((rank, "optimizer_1")); events.append((rank, "optimizer_2"))
        return {"trainer/optimizer_steps_this_iteration": 2, "perf/teacher_seconds": 0.}
    obj = SimpleNamespace(opd_config=SimpleNamespace(prompt_profile=profile),
        _is_actor=True, _is_offload_param=True, _is_offload_optimizer=False,
        config=Config(resource_policy=copy.deepcopy(policy), rollout_integrity={"enabled": True},
            rollout=Config(name="vllm", temperature=1., add_noise_dirichlet=False,
                           add_noise_gumbel_softmax=False, enable_soft_thinking=False),
            actor=SimpleNamespace(ppo_epochs=1)), ulysses_sharding_manager=Sharding(),
        actor=SimpleNamespace(update_policy=update), actor_module_fsdp=object(),
        flops_counter=SimpleNamespace(estimate_flops=lambda *args: (1, 1)),
        actor_lr_scheduler=SimpleNamespace(get_last_lr=lambda: [1e-6], step=lambda: events.append((rank, "scheduler"))),
        world_size=2, rank=rank)
    return obj, lambda: method(obj, Data(meta_info={"global_token_num": [1, 1]}))


@pytest.fixture
def cpu_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: ((30 if device == 0 else 20), 100))


def test_logical_oversubscription_and_whole_node_host_usage_are_diagnostic(cpu_cuda):
    from verl.opd.chat import QWEN3_TRAINING_PROFILE
    events = []; exchanges = Exchanges()
    workers = [worker(rank, exchanges, events, profile=QWEN3_TRAINING_PROFILE) for rank in range(2)]
    outputs, errors = run_ranks([item[1] for item in workers])
    assert errors == [None, None]
    assert sum(event[1] == "scheduler" for event in events) == 2
    assert sum(event[1] == "offload" for event in events) == 2
    timing = outputs[0].meta_info["actor_update_timing"]
    assert timing["logical_allocator_peaks_diagnostic_only"] is True
    assert timing["physical_device_used_fraction_peak"] == .8
    assert timing["physical_device_free_min_gib"] == 20 / 1024**3
    assert [row["physical_memory"]["device_used_peak_bytes"] for row in timing["ranks"]] == [70, 80]
    assert all(row["physical_memory"]["sample_count"] >= 2 for row in timing["ranks"])


@pytest.mark.parametrize("fault", ["physical", "metric", "legacy"])
def test_one_rank_resource_failure_collectively_blocks_scheduler_and_offload(cpu_cuda, monkeypatch, fault):
    if fault == "physical":
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1 if device else 40, 100))
    events = []; exchanges = Exchanges()
    workers = [worker(rank, exchanges, events, policy={} if fault == "legacy" else POLICY,
                      metrics_error=fault == "metric" and rank == 1) for rank in range(2)]
    _, errors = run_ranks([item[1] for item in workers])
    assert all(isinstance(error, RuntimeError) and "resource-integrity" in str(error) for error in errors)
    assert sum(event[1].startswith("optimizer") for event in events) == 4
    assert not any(event[1] in ("scheduler", "offload") for event in events)
    assert all(getattr(item[0], "_opd_resource_failure", None) for item in workers)
    if fault == "physical": assert '"device_used_peak_bytes": 99' in str(errors[0])
    if fault == "legacy": assert "72.000" in str(errors[0])


@pytest.mark.parametrize("fault", ["prior_failure", "initial_sensor"])
def test_one_rank_start_failure_blocks_all_optimizers(cpu_cuda, monkeypatch, fault):
    events = []; exchanges = Exchanges()
    workers = [worker(rank, exchanges, events) for rank in range(2)]
    if fault == "prior_failure": workers[1][0]._opd_resource_failure = "prior failure"
    else:
        def read(device):
            if device: raise RuntimeError("sensor unavailable")
            return 50, 100
        monkeypatch.setattr(torch.cuda, "mem_get_info", read)
    _, errors = run_ranks([item[1] for item in workers])
    assert all(isinstance(error, RuntimeError) for error in errors)
    assert not events


def test_actor_failure_keeps_original_error_and_stops_sampler(cpu_cuda):
    obj = SimpleNamespace(config=Config(resource_policy=POLICY, rollout_integrity={"enabled": True}))
    dist = SimpleNamespace(is_initialized=lambda: False)
    guard = ResourceGuard(obj, distributed=dist, device=0)
    with pytest.raises(RuntimeError, match="original actor failure"):
        with guard: raise RuntimeError("original actor failure")
    assert guard.monitor.stopped and not guard.monitor.thread.is_alive()


def test_missing_validation_cannot_succeed_or_reuse_worker(cpu_cuda):
    obj = SimpleNamespace(config=Config(resource_policy=POLICY, rollout_integrity={"enabled": True}))
    dist = SimpleNamespace(is_initialized=lambda: False)
    with pytest.raises(RuntimeError, match="without a completed collective validation"):
        with ResourceGuard(obj, distributed=dist, device=0): pass
    with pytest.raises(RuntimeError, match="cannot be reused"):
        with ResourceGuard(obj, distributed=dist, device=0): pytest.fail("poisoned worker entered")


def test_sampler_cleanup_failure_preserves_actor_error_and_poisons_worker(cpu_cuda, monkeypatch):
    obj = SimpleNamespace(config=Config(resource_policy=POLICY, rollout_integrity={"enabled": True}))
    dist = SimpleNamespace(is_initialized=lambda: False)
    guard = ResourceGuard(obj, distributed=dist, device=0)
    with pytest.raises(RuntimeError, match="original actor failure"):
        with guard:
            guard.monitor.stop()  # End the real thread before injecting cleanup failure.
            guard.monitor.stopped = False
            monkeypatch.setattr(guard.monitor, "stop", lambda: (_ for _ in ()).throw(RuntimeError("cannot join sampler")))
            raise RuntimeError("original actor failure")
    assert "sampler cleanup failed" in obj._opd_resource_failure
