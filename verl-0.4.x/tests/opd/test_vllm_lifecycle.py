"""Execute real vLLM adapter/manager methods without a GPU engine dependency."""
import ast
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.opd.vllm_lifecycle import finish_vllm_stage, poison_vllm, require_idle_vllm
from test_deterministic_sampling import seed_module

ROOT = Path(__file__).resolve().parents[2] / 'verl/workers'


def methods(path, name, names, namespace):
    tree = ast.parse(path.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    selected = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for method in selected:
        method.decorator_list = []
    cls = ast.ClassDef(name=name, bases=[], keywords=[], body=selected, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace[name]


class Exchanges:
    def __init__(self):
        self.condition = threading.Condition(); self.values = {}; self.indices = [0, 0]

    def rank(self, rank):
        def exchange(output, value):
            with self.condition:
                index = self.indices[rank]; self.indices[rank] += 1
                slot = self.values.setdefault(index, {}); slot[rank] = value
                self.condition.notify_all()
                if not self.condition.wait_for(lambda: len(slot) == 2, timeout=3):
                    raise RuntimeError('unmatched rank collective')
                output[:] = [slot[0], slot[1]]
        return SimpleNamespace(is_initialized=lambda: True, get_world_size=lambda: 2, all_gather_object=exchange)


def manager(rank, exchange, events, *, enabled=True, sleep_error=False):
    def sleep(**kw):
        events.append((rank, 'sleep'))
        if sleep_error:
            raise RuntimeError('sleep failed')
    engine = SimpleNamespace(sleep=sleep, shutdown=lambda: events.append((rank, 'shutdown')))
    device = SimpleNamespace(empty_cache=lambda: events.append((rank, 'empty')),
                             get_rng_state=lambda: 'generation', set_rng_state=lambda state: events.append((rank, 'rng', state)),
                             synchronize=lambda: None)
    cls = methods(ROOT / 'sharding_manager/fsdp_vllm.py', 'FSDPVLLMShardingManager',
                  {'__enter__', '__exit__', '_guard_stage', '_release_after_rollout'},
                  {'torch': SimpleNamespace(distributed=exchange.rank(rank)), 'time': time,
                   'get_torch_device': lambda: device, 'vllm_version': '0.8.5', 'OrderedDict': OrderedDict})
    obj = cls(); obj.inference_engine = engine; obj._frozen_batch_guard = enabled
    obj._opd_rng_switched = True; obj.torch_random_states = 'actor'; obj.device_mesh = object()
    obj.module = SimpleNamespace(train=lambda: events.append((rank, 'train')))
    obj.last_rollout_timing = {}
    return obj


def run_ranks(operations):
    errors = [None, None]
    def run(rank):
        try: operations[rank]()
        except BaseException as error: errors[rank] = error
    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join(timeout=4)
    assert not any(thread.is_alive() for thread in threads), 'unmatched collective'
    return errors


@pytest.mark.parametrize('failure', ['request', 'outstanding', 'scheduler', 'assembly'])
def test_failed_or_outstanding_rank_prevents_all_memory_release_and_updates(failure):
    events = []; collective = Exchanges()
    managers = [manager(rank, collective, events) for rank in range(2)]
    error = RuntimeError(failure) if failure in ('request', 'assembly') else None
    if failure == 'outstanding': managers[1].inference_engine._opd_batch_outstanding = True
    if failure == 'scheduler': managers[1].inference_engine.llm_engine = SimpleNamespace(has_unfinished_requests=lambda: True)
    def operation(rank):
        managers[rank].__exit__(type(error) if rank == 1 else None, error if rank == 1 else None, None)
        events.append((rank, 'optimizer'))
    errors = run_ranks([lambda: operation(0), lambda: operation(1)])
    assert all(isinstance(error, RuntimeError) for error in errors)
    assert not any(event[1] in ('sleep', 'empty', 'train', 'optimizer') for event in events)
    assert sum(event[1] == 'shutdown' for event in events) == 2
    assert all(obj.inference_engine._opd_poisoned and not obj._opd_rng_switched for obj in managers)


def test_healthy_fast_rank_waits_for_whole_batch_before_sleep():
    events = []; collective = Exchanges(); slow_done = threading.Event()
    managers = [manager(rank, collective, events) for rank in range(2)]
    def slow():
        time.sleep(.05)
        assert not any(event[1] == 'sleep' for event in events)
        slow_done.set(); managers[1].__exit__(None, None, None)
    errors = run_ranks([lambda: managers[0].__exit__(None, None, None), slow])
    assert errors == [None, None] and slow_done.is_set()
    assert sum(event[1] == 'sleep' for event in events) == 2
    assert all(obj.last_rollout_timing['release_memory_seconds'] >= 0 for obj in managers)


def test_memory_release_failure_poisoned_collectively_before_optimizer():
    events = []; collective = Exchanges()
    managers = [manager(rank, collective, events, sleep_error=rank == 1) for rank in range(2)]
    errors = run_ranks([lambda: managers[0].__exit__(None, None, None), lambda: managers[1].__exit__(None, None, None)])
    assert all(isinstance(error, RuntimeError) for error in errors)
    assert all(obj.inference_engine._opd_poisoned for obj in managers)


def test_poison_prevents_following_weight_transfer():
    events = []; collective = Exchanges()
    managers = [manager(rank, collective, events) for rank in range(2)]
    poison_vllm(managers[1].inference_engine, 'prior failure')
    errors = run_ranks([lambda: managers[0].__enter__(), lambda: managers[1].__enter__()])
    assert all(isinstance(error, RuntimeError) for error in errors)
    assert not any(event[1] in ('empty', 'sleep', 'train') for event in events)


def test_legacy_exit_retains_noncollective_default():
    events = []; obj = manager(0, Exchanges(), events, enabled=False)
    obj.__exit__(None, None, None)
    assert (0, 'sleep') in events and not hasattr(obj.inference_engine, '_opd_poisoned')


def test_executor_shutdown_and_permanent_poison_even_when_shutdown_fails():
    calls = []
    def shutdown(): calls.append(1); raise RuntimeError('shutdown unavailable')
    engine = SimpleNamespace(llm_engine=SimpleNamespace(model_executor=SimpleNamespace(shutdown=shutdown)))
    poison_vllm(engine, 'request failed'); poison_vllm(engine, 'collective failed')
    assert calls == [1] and 'shutdown unavailable' in engine._opd_shutdown_error
    with pytest.raises(RuntimeError, match='poisoned'): require_idle_vllm(engine)


def test_failed_collective_poisoned():
    engine = SimpleNamespace(shutdown=lambda: None)
    def broken(*args): raise RuntimeError('NCCL failure')
    dist = SimpleNamespace(is_initialized=lambda: True, get_world_size=lambda: 2, all_gather_object=broken)
    with pytest.raises(RuntimeError, match='NCCL failure'): finish_vllm_stage(engine, dist, None, 'completion')
    assert engine._opd_poisoned


class Config(dict):
    __getattr__ = dict.__getitem__


def rollout(*, groups=8, fail=None):
    def data(batch, non_tensor_batch): return SimpleNamespace(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={})
    def pad(rows, value, max_length): return torch.tensor([row + [value] * (max_length - len(row)) for row in rows])
    namespace = {'DataProto': data, 'time': time, 'torch': torch, 'np': np, 'deepcopy': deepcopy,
                 'vllm_version': '0.8.5', 'TensorDict': lambda tensors, **kw: tensors,
                 'pad_2d_list_to_length': pad, 'get_response_mask': lambda response_id, **kw: (response_id != 0).long(),
                 'derive_request_seed': seed_module.derive_request_seed, 'expand_parallel_seeds': seed_module.expand_parallel_seeds,
                 '_repeat_interleave': lambda value, n: value.repeat_interleave(n, dim=0),
                 '_repeat_non_tensor_batch': lambda values, repeats, **kw: {k: np.repeat(v, repeats, axis=0) for k, v in values.items()}}
    cls = methods(ROOT / 'rollout/vllm_rollout/vllm_rollout_spmd.py', 'vLLMRollout',
                  {'generate_sequences', '_generate_sequences_impl', 'update_sampling_params'}, namespace)
    cls.update_sampling_params = contextmanager(cls.update_sampling_params)
    obj = cls(); obj._frozen_batch_guard = True; obj.lora_kwargs = {}; obj.pad_token_id = 0
    obj.config = Config(response_length=4, deterministic_sampling=True,
                        val_kwargs=SimpleNamespace(top_k=-1, top_p=1, temperature=.6))
    obj.sampling_params = SimpleNamespace(n=groups, seed=0, temperature=1)
    obj.requests = []; obj.shutdowns = []
    def generate(prompts, sampling_params, **kw):
        assert obj.inference_engine._opd_batch_outstanding
        obj.requests.extend(zip(prompts, sampling_params))
        if fail == 'generation': raise RuntimeError('generate failed')
        logprobs = [{3: SimpleNamespace(logprob=-.2)}, {4: SimpleNamespace(logprob=-.4)}]
        if fail == 'assembly': logprobs = None
        if fail == 'densities': logprobs = logprobs[:1]
        return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[3, 4], logprobs=logprobs)]) for _ in prompts]
    obj.inference_engine = SimpleNamespace(generate=generate, shutdown=lambda: obj.shutdowns.append(1))
    prompts = SimpleNamespace(batch={'input_ids': torch.tensor([[1, 2], [1, 5]]), 'attention_mask': torch.ones(2, 2, dtype=torch.long),
                                     'position_ids': torch.tensor([[0, 1], [0, 1]])},
                              non_tensor_batch={'raw_prompt_ids': np.array([[1, 2], [1, 5]], dtype=object), 'index': np.array([17, 28]),
                                                'privileged': np.array(['gold-a', 'gold-b'], dtype=object)},
                              meta_info={'eos_token_id': 4, 'rollout_iteration': 3, 'rollout_seed': 11})
    return obj, prompts


@pytest.mark.parametrize('groups', [1, 8])
def test_real_adapter_preserves_seed_groups_densities_and_metadata(groups):
    obj, prompts = rollout(groups=groups)
    result = obj.generate_sequences(prompts)
    seeds = [params.seed for _, params in obj.requests]
    assert len(seeds) == len(set(seeds)) == 2 * groups
    assert all(params.n == 1 for _, params in obj.requests)
    assert result.batch['rollout_sampling_seed'].tolist() == seeds
    assert result.non_tensor_batch['privileged'].tolist() == ['gold-a'] * groups + ['gold-b'] * groups
    assert result.batch['rollout_log_probs'].shape == (2 * groups, 4)
    assert result.meta_info['rollout_timing']['generated_tokens'] == 4 * groups
    assert result.meta_info['rollout_timing']['tensor_assembly_seconds'] >= 0
    assert not obj.inference_engine._opd_batch_outstanding and not obj.shutdowns


@pytest.mark.parametrize('failure', ['generation', 'assembly', 'densities'])
def test_real_adapter_poisoned_on_generation_or_tensor_assembly_error(failure):
    obj, prompts = rollout(fail=failure)
    with pytest.raises((RuntimeError, TypeError)): obj.generate_sequences(prompts)
    assert obj.inference_engine._opd_poisoned and obj.shutdowns == [1]
    assert obj.inference_engine._opd_batch_outstanding
    with pytest.raises(RuntimeError, match='poisoned'): obj.generate_sequences(prompts)
    assert len(obj.requests) == 16


def test_real_adapter_validation_uses_preexpanded_sample_indices_once():
    obj, prompts = rollout(groups=8)
    prompts.meta_info['validate'] = True
    prompts.non_tensor_batch['rollout_sample_index'] = np.array([3, 7])
    result = obj.generate_sequences(prompts)
    assert len(obj.requests) == 2 and all(params.n == 1 for _, params in obj.requests)
    assert result.non_tensor_batch['rollout_sample_index'].tolist() == [3, 7]
    assert obj.sampling_params.n == 8 and obj.sampling_params.temperature == 1


def test_preparation_failure_is_poisoned_before_any_request():
    obj, prompts = rollout()
    del prompts.meta_info['rollout_seed']
    with pytest.raises(RuntimeError, match='rollout_seed'): obj.generate_sequences(prompts)
    assert not obj.requests and obj.shutdowns == [1] and obj.inference_engine._opd_poisoned
