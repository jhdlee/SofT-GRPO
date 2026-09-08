import json
import inspect
import ast
import importlib.metadata
import importlib.util
import threading
from contextlib import nullcontext
import os
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
import sys
import warnings
import time

import pytest
import torch
from safetensors.torch import load_file

from verl.opd import qwen_weight_export as export
from verl.opd import qwen_vllm_arithmetic
from verl.opd import qwen_vllm_attention
from verl.opd.checkpoint_semantics import collective_checkpoint_stage
from verl.opd.provenance import _canonical_sha256, _model_identity
from verl.opd.qwen_lora import effective_projection_weight, qwen_lora_config
from test_qwen_lora import actor, projection
from test_vllm_lifecycle import Exchanges, methods, run_ranks


@pytest.fixture(autouse=True)
def cpu_fault_injection(monkeypatch):
    # These threaded stand-ins exercise collective ordering and local files.
    # Native CUDA export/merge arithmetic has its own real GPU acceptance.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    # Structural/numerical installer contracts have their own CPU references.
    # These stand-ins isolate matched entry phases and injected rank failures.
    monkeypatch.setattr(qwen_vllm_arithmetic, 'install_qwen_vllm_replay_arithmetic',
                        lambda model, *args, **kwargs: model.test_install_arithmetic())
    monkeypatch.setattr(qwen_vllm_arithmetic, 'verify_qwen_vllm_weights',
                        lambda model, weights: model.test_verify_weights(weights))
    monkeypatch.setattr(qwen_vllm_attention, 'install_qwen_vllm_attention',
                        lambda impl, **kwargs: impl.test_install_attention())
    monkeypatch.setattr(qwen_vllm_attention, 'qwen_vllm_attention_telemetry', lambda *args, **kwargs: [])


class ThreadRanks:
    def __init__(self):
        self.local = threading.local(); self.exchange = Exchanges()
    def is_initialized(self): return True
    def get_world_size(self): return 2
    def get_rank(self): return self.local.rank
    def all_gather_object(self, output, value):
        self.exchange.rank(self.get_rank()).all_gather_object(output, value)
    def operations(self, function):
        def run(rank): self.local.rank = rank; return function(rank)
        return [lambda: run(0), lambda: run(1)]


def test_transfer_failure_reaches_peers_before_any_tensor_collective(monkeypatch):
    dist = ThreadRanks(); monkeypatch.setattr(export, 'dist', dist)
    gathers = []
    class Shard:
        device = torch.device('cpu')
        def detach(self): return self
        def to(self, device):
            if dist.get_rank() == 1: raise RuntimeError('injected allocation failure')
            return self
        def full_tensor(self): gathers.append(dist.get_rank()); return torch.ones(2)
    errors = run_ranks(dist.operations(lambda rank: export._materialize(Shard())))
    assert all(isinstance(error, RuntimeError) and 'allocation failure' in str(error) for error in errors)
    assert not gathers


def test_rank_inventory_mismatch_stops_before_materialization(monkeypatch):
    dist = ThreadRanks(); monkeypatch.setattr(export, 'dist', dist)
    monkeypatch.setattr(export, '_materialize', lambda *args, **kw: pytest.fail('must not materialize'))
    errors = run_ranks(dist.operations(lambda rank: export.dense_rollout_weights({str(rank): torch.ones(2)}, {})))
    assert all(isinstance(error, RuntimeError) and 'inventories differ' in str(error) for error in errors)


def test_dense_export_has_exact_effective_bf16_actor_projection():
    model = actor()
    with torch.no_grad(): projection(model).qwen_lora_B.fill_(.3)
    dense = export.dense_rollout_weights(model.state_dict(), qwen_lora_config(model))
    name = 'model.layers.0.self_attn.q_proj.weight'
    x = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8) / 7
    assert torch.equal(torch.nn.functional.linear(x, dense[name]),
                       torch.nn.functional.linear(x, effective_projection_weight(projection(model))))
    assert not any('qwen_lora' in name for name in dense)
    assert all(not value.requires_grad for value in dense.values())


def test_full_tuning_dense_export_casts_floats_and_preserves_integer_buffers():
    source = {'z.weight': torch.ones(2, requires_grad=True), 'a.counter': torch.arange(2)}
    result = export.dense_rollout_weights(source)
    assert list(result) == ['a.counter', 'z.weight']
    assert result['z.weight'].dtype == torch.bfloat16 and not result['z.weight'].requires_grad
    assert torch.equal(result['a.counter'], source['a.counter'])
    with pytest.raises(RuntimeError, match='explicit merge configuration'):
        export.dense_rollout_weights(actor().state_dict())


def native_entry_managers(dist, events, *, failure=None, lora=False, vllm_bindings=None):
    """Run actual entry/export/load methods with two real matched phase streams."""
    source = Path(__file__).resolve().parents[2] / 'verl/workers/sharding_manager/fsdp_vllm.py'
    objects = []
    for rank in range(2):
        def build(rank):
            def event(name):
                events.append((rank, name))
                if rank == 1 and name == failure:
                    raise RuntimeError('injected ' + name)
            class Shard:
                device = torch.device('cpu')
                def __init__(self, value): self.value = value
                def detach(self): return self
                def to(self, device): event('local transfer'); return self
                def full_tensor(self):
                    event('gather entered')
                    values = [None, None]
                    dist.all_gather_object(values, 'tensor gather')
                    assert values == ['tensor gather', 'tensor gather'], 'mismatched collective phase'
                    event('gather completed')
                    return self.value.detach()
            model = actor() if lora else SimpleNamespace()
            model._opd_qwen_replay_arithmetic = True
            state = model.state_dict() if lora else {'z.weight': torch.ones(2), 'a.counter': torch.arange(2)}
            if failure == 'configuration' and rank == 1: model.peft_config = {}
            def state_dict(): event('state dict'); return {name: Shard(value) for name, value in state.items()}
            model.state_dict = state_dict
            def wake_up(tags=None): event('wake ' + tags[0])
            engine = SimpleNamespace(wake_up=wake_up, shutdown=lambda: events.append((rank, 'shutdown')))
            engine.llm_engine = SimpleNamespace(get_vllm_config=lambda: object())
            cache_calls = 0
            rng = 'actor'
            def empty_cache():
                nonlocal cache_calls
                cache_calls += 1; event('empty initial' if cache_calls == 1 else 'empty final')
            def set_rng(value):
                nonlocal rng
                rng = value
                if value == 'generation': event('RNG switch')
                else: events.append((rank, 'RNG restored'))
            device = SimpleNamespace(empty_cache=empty_cache, get_rng_state=lambda: rng,
                                     set_rng_state=set_rng, synchronize=lambda: event('synchronize'),
                                     current_device=lambda: 'cpu')
            def load_weights(values):
                weights = dict(values)
                assert weights and all(isinstance(value, torch.Tensor) for value in weights.values())
                assert all(value.dtype == torch.bfloat16 for value in weights.values() if value.is_floating_point())
                assert not any('qwen_lora' in name for name in weights)
                event('load weights'); return list(weights)
            cls = methods(source, 'FSDPVLLMShardingManager', {'__enter__', '_enter_native', '_guard_stage', 'update_params'}, {
                'torch': SimpleNamespace(distributed=dist), 'time': time, 'inspect': inspect,
                'get_torch_device': lambda: device, 'vllm_version': None, 'vllm_package_version': '0.8.5', 'OrderedDict': dict,
                'load_fsdp_model_to_gpu': lambda model: event('actor load'),
                'offload_fsdp_model_to_cpu': lambda model: event('actor offload'),
                'DTensor': Shard, 'patch_vllm_moe_model_weight_loader': lambda model: None,
                'logger': SimpleNamespace(info=lambda *args: None),
                'log_gpu_memory_usage': lambda *args, **kwargs: None,
                **(vllm_bindings or {}),
            })
            obj = cls(); obj.module = model; obj.inference_engine = engine
            def install_arithmetic(): event('arithmetic install'); return {'recipe': 'test'}
            def verify_weights(weights): event('weights verified'); return {'loaded_weights_exact': True}
            def install_attention(): event('attention install'); return {'num_splits': 1}
            attention_layer = SimpleNamespace(self_attn=SimpleNamespace(attn=SimpleNamespace(
                impl=SimpleNamespace(test_install_attention=install_attention))))
            obj.model_runner = SimpleNamespace(model=SimpleNamespace(load_weights=load_weights,
                test_install_arithmetic=install_arithmetic, test_verify_weights=verify_weights,
                model=SimpleNamespace(layers=[attention_layer])))
            obj.model_config = object()
            obj._frozen_batch_guard = True; obj._opd_rng_switched = False
            obj.tp_size = 1; obj.offload_param = True; obj.device_mesh = object(); obj.full_params = False
            obj.gen_random_states = 'generation'; obj.torch_random_states = 'actor'
            return obj
        objects.append(build(rank))
    return objects


def modern_vllm_bindings(monkeypatch, installed_version):
    """Execute the actual version selector/imports, without loading CUDA vLLM."""
    base = Path(__file__).resolve().parents[2] / 'verl'
    vllm = ModuleType('vllm'); vllm.LLM = object()
    distributed = ModuleType('vllm.distributed'); distributed.parallel_state = object()
    monkeypatch.setitem(sys.modules, 'vllm', vllm)
    monkeypatch.setitem(sys.modules, 'vllm.distributed', distributed)
    original_version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, 'version',
                        lambda package: installed_version if package == 'vllm' else original_version(package))
    spec = importlib.util.spec_from_file_location('_test_modern_vllm', base / 'third_party/vllm/__init__.py')
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    assert shim.package_version == installed_version and shim.vllm_version is None
    monkeypatch.setitem(sys.modules, 'verl.third_party.vllm', shim)
    manager_path = base / 'workers/sharding_manager/fsdp_vllm.py'
    imports = [node for node in ast.parse(manager_path.read_text()).body
               if isinstance(node, ast.ImportFrom) and node.module == 'verl.third_party.vllm']
    bindings = {}
    exec(compile(ast.Module(body=imports, type_ignores=[]), str(manager_path), 'exec'), bindings)
    return bindings


def test_modern_installed_vllm_accepts_native_entry_with_none_legacy_selector(monkeypatch):
    bindings = modern_vllm_bindings(monkeypatch, '0.8.5')
    dist = ThreadRanks(); monkeypatch.setattr(export, 'dist', dist)
    events = []; objects = native_entry_managers(dist, events, lora=True, vllm_bindings=bindings)
    errors = run_ranks(dist.operations(lambda rank: objects[rank].__enter__()))
    assert errors == [None, None], errors
    assert sum(name == 'load weights' for _, name in events) == 2
    assert not any(name == 'shutdown' for _, name in events)


@pytest.mark.parametrize('installed_version,tp_size', [('0.8.4', 1), ('0.8.5.post1', 1), ('0.9.0', 1), ('0.8.5', 2)])
def test_native_entry_rejects_wrong_installed_version_or_tp_before_transfer(monkeypatch, installed_version, tp_size):
    bindings = modern_vllm_bindings(monkeypatch, installed_version)
    dist = ThreadRanks(); monkeypatch.setattr(export, 'dist', dist)
    events = []; objects = native_entry_managers(dist, events, vllm_bindings=bindings)
    objects[1].tp_size = tp_size
    errors = run_ranks(dist.operations(lambda rank: objects[rank].__enter__()))
    assert all(isinstance(error, RuntimeError) and 'requires TP1 and vLLM 0.8.5' in str(error) for error in errors), errors
    assert not any(name in ('state dict', 'gather entered', 'wake weights', 'load weights') for _, name in events)
    assert sum(name == 'shutdown' for _, name in events) == 2


@pytest.mark.parametrize('lora', [False, True])
def test_native_entry_materializes_full_and_lora_weights_before_local_loader(monkeypatch, lora):
    dist = ThreadRanks(); monkeypatch.setattr(export, 'dist', dist)
    events = []; objects = native_entry_managers(dist, events, lora=lora)
    def enter(rank): objects[rank].__enter__(); events.append((rank, 'generation allowed'))
    errors = run_ranks(dist.operations(enter))
    assert errors == [None, None], errors
    assert sum(name == 'generation allowed' for _, name in events) == 2
    assert max(index for index, (_, name) in enumerate(events) if name == 'gather completed') < min(
        index for index, (_, name) in enumerate(events) if name == 'load weights')
    assert all(obj._opd_rng_switched and obj.base_sync_done for obj in objects)
    assert all(obj.last_rollout_timing['weight_transfer_seconds'] >= 0 for obj in objects)
    assert all(obj.last_rollout_timing['categorical_arithmetic']['loaded_weights_exact'] for obj in objects)
    for rank in range(2):
        sequence = [name for r, name in events if r == rank]
        assert sequence.index('load weights') < sequence.index('arithmetic install') < sequence.index('weights verified') < sequence.index('attention install') < sequence.index('generation allowed')
    assert not any(name == 'shutdown' for _, name in events)


def test_legacy_entry_does_not_use_native_collective_export_or_entry_guards(monkeypatch):
    dist = ThreadRanks(); events = []
    objects = native_entry_managers(dist, events)
    monkeypatch.setattr(export, 'dense_rollout_weights', lambda *args, **kwargs: pytest.fail('legacy native export'))
    for obj in objects:
        obj._frozen_batch_guard = False
        obj.module.state_dict = lambda: {'weight': torch.ones(2, dtype=torch.bfloat16)}
        obj._guard_stage = lambda *args: pytest.fail('legacy collective guard')
    errors = run_ranks(dist.operations(lambda rank: objects[rank].__enter__()))
    assert errors == [None, None], errors
    assert sum(name == 'load weights' for _, name in events) == 2
    assert not dist.exchange.values and not any(name == 'shutdown' for _, name in events)


@pytest.mark.parametrize('failure', [
    'actor load', 'state dict', 'configuration', 'local transfer', 'gather completed',
    'wake weights', 'load weights', 'arithmetic install', 'weights verified', 'attention install', 'actor offload', 'empty final', 'wake kv_cache',
    'RNG switch', 'synchronize',
])
def test_native_entry_failure_poisoned_collectively_before_generation(monkeypatch, failure):
    dist = ThreadRanks(); monkeypatch.setattr(export, 'dist', dist)
    events = []; objects = native_entry_managers(dist, events, failure=failure)
    def enter(rank): objects[rank].__enter__(); events.append((rank, 'generation allowed'))
    errors = run_ranks(dist.operations(enter))
    assert all(isinstance(error, RuntimeError) and 'collective' in str(error) for error in errors), errors
    assert not any(name == 'generation allowed' for _, name in events)
    assert sum(name == 'shutdown' for _, name in events) == 2
    assert all(obj.inference_engine._opd_poisoned and not obj._opd_rng_switched for obj in objects)
    if failure == 'actor load': assert not any(name == 'state dict' for _, name in events)
    if failure == 'local transfer': assert not any(name == 'gather entered' for _, name in events)
    if failure in ('state dict', 'configuration', 'local transfer', 'gather completed'):
        assert not any(name == 'wake weights' for _, name in events)


def base_model(tmp_path):
    root = tmp_path / 'base'; root.mkdir()
    payload = {'model': {'id': 'Qwen/Qwen3-0.6B', 'resolved_revision': 'c1899de289a04d12100db370d81485cdf75e47ca'},
               'inventory_sha256': 'a' * 64}
    payload['manifest_content_sha256'] = _canonical_sha256(payload)
    (root / 'manifest.json').write_text(json.dumps(payload))
    return root


def test_real_adapter_payload_binds_frozen_base_and_pinned_revision(tmp_path):
    model = actor(); base = base_model(tmp_path)
    checkpoint = tmp_path / 'checkpoint'
    export.save_native_adapter(model, checkpoint, base_model_path=str(base))
    directory = checkpoint / 'lora_adapter'
    actual = load_file(directory / 'adapter_model.safetensors')
    expected = {}
    for name, tensor in model.state_dict().items():
        for native, peft in (('.qwen_lora_A', '.lora_A.weight'), ('.qwen_lora_B', '.lora_B.weight')):
            if name.endswith(native): expected['base_model.model.' + name.removesuffix(native) + peft] = tensor
    assert set(actual) == set(expected)
    assert all(torch.equal(actual[name], tensor) and tensor.dtype == torch.float32 for name, tensor in expected.items())
    config = json.loads((directory / 'adapter_config.json').read_text())
    identity = json.loads((directory / 'frozen_base_identity.json').read_text())
    assert identity == _model_identity(str(base))
    assert config['base_model_name_or_path'] == identity['id'] and config['revision'] == identity['resolved_revision']
    with pytest.raises(RuntimeError, match='publication'):
        export.save_native_adapter(model, checkpoint, base_model_path=str(base))


def test_adapter_publication_failure_propagates_to_every_rank(tmp_path, monkeypatch):
    dist = ThreadRanks(); monkeypatch.setattr(export, 'dist', dist)
    base = base_model(tmp_path); models = [actor(), actor()]
    def fail(*args, **kw): raise OSError('disk full')
    monkeypatch.setattr('safetensors.torch.save_file', fail)
    errors = run_ranks(dist.operations(lambda rank: export.save_native_adapter(models[rank], tmp_path / 'checkpoint', base_model_path=str(base))))
    assert all(isinstance(error, RuntimeError) and 'disk full' in str(error) for error in errors)
    assert not (tmp_path / 'checkpoint/lora_adapter/adapter_config.json').exists()


def test_checkpoint_local_io_failure_prevents_following_rank_barrier():
    dist = ThreadRanks(); following = []
    def operation(rank):
        def write():
            if rank == 1: raise OSError('RNG sidecar write failed')
            return 'saved'
        collective_checkpoint_stage('RNG publication', write, distributed=dist)
        following.append(rank)
    errors = run_ranks(dist.operations(operation))
    assert all(isinstance(error, RuntimeError) and 'RNG sidecar' in str(error) for error in errors)
    assert not following


@pytest.mark.parametrize('operation', ['save', 'restore'])
def test_actual_rng_sidecar_io_error_is_collective(tmp_path, monkeypatch, operation):
    from verl.opd import rng_state
    dist = ThreadRanks(); following = []
    if operation == 'restore':
        for rank in range(2): rng_state.save_worker_rng(tmp_path, rank, 2)
        (tmp_path / 'worker_rng_world_size_2_rank_1.json').unlink()
    monkeypatch.setattr(rng_state, 'collective_checkpoint_stage',
                        lambda label, fn: collective_checkpoint_stage(label, fn, distributed=dist))
    def invoke(rank):
        if operation == 'save':
            rng_state.save_worker_rng(tmp_path if rank == 0 else tmp_path / 'missing', rank, 2)
        else:
            rng_state.restore_worker_rng(tmp_path, rank, 2)
        following.append(rank)
    errors = run_ranks(dist.operations(invoke))
    assert all(isinstance(error, RuntimeError) and 'worker RNG' in str(error) for error in errors)
    assert not following


@pytest.mark.parametrize('failure', ['rank state', 'metadata'])
def test_actual_fsdp_checkpoint_local_publication_failure_precedes_barrier(tmp_path, failure):
    dist = ThreadRanks(); barriers = []
    dist.barrier = lambda: barriers.append(dist.get_rank())
    def save(tensor, path):
        if failure == 'rank state' and dist.get_rank() == 1: raise OSError('injected shard write')
        torch.save(tensor, path)
    source = Path(__file__).resolve().parents[2] / 'verl/utils/checkpoint/fsdp_checkpoint_manager.py'
    cls = methods(source, 'FSDPCheckpointManager', {'save_checkpoint'}, {
        'torch': SimpleNamespace(save=save, distributed=dist), 'os': os, 'warnings': warnings,
        'ShardedStateDictConfig': lambda **kw: None, 'ShardedOptimStateDictConfig': lambda **kw: None,
        'is_cuda_available': False, 'get_fsdp_state_ctx': lambda *args: nullcontext(),
        'StateDictType': SimpleNamespace(SHARDED_STATE_DICT='sharded'), 'fsdp_version': lambda model: 0,
        'collective_checkpoint_stage': lambda label, fn: collective_checkpoint_stage(label, fn, distributed=dist),
    })
    objects = []
    for rank in range(2):
        obj = cls(); obj.model = torch.nn.Linear(2, 2); obj.optimizer = obj.lr_scheduler = None
        obj.rank = rank; obj.world_size = 2; obj.semantic_state = True
        obj.previous_saved_paths = []; obj.checkpoint_contents = ['model', 'optimizer', 'extra']
        def mkdir(path):
            Path(path).mkdir(parents=True, exist_ok=True)
            return path
        obj.local_mkdir = mkdir
        obj.get_rng_state = lambda: {}
        def metadata(path):
            if failure == 'metadata': raise OSError('injected metadata write')
        obj.model.config = SimpleNamespace(save_pretrained=metadata)
        obj.model.can_generate = lambda: False
        obj.processing_class = SimpleNamespace(save_pretrained=lambda path: None)
        objects.append(obj)
    errors = run_ranks(dist.operations(lambda rank: objects[rank].save_checkpoint(str(tmp_path), global_step=1)))
    assert all(isinstance(error, RuntimeError) and 'publication' in str(error) for error in errors), errors
    # The initial directory-ready barrier is allowed; the post-publication
    # barrier must never run after a peer's local write failed.
    assert sorted(barriers) == [0, 1] and all(not obj.previous_saved_paths for obj in objects)
