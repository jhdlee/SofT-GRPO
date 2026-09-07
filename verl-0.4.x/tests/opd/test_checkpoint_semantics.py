import copy
import json

import numpy as np
import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

from verl.opd.checkpoint_semantics import (
    checkpoint_semantic_identity, model_state_record, read_record, semantic_sha256,
    verify_model_state, write_record,
)


def distributed_tensor(monkeypatch, *, placements=(Shard(0),), mesh_values=(0,), dtype=torch.float32):
    # Exercise the real DTensor/DeviceMesh classes without starting a process
    # group. Hashing a local shard requires no collective or network socket.
    monkeypatch.setattr('torch.distributed.device_mesh.get_rank', lambda: 0)
    mesh = DeviceMesh('cpu', list(mesh_values), mesh_dim_names=('dp',), _init_backend=False)
    tensor = torch.arange(6, dtype=dtype).reshape(2, 3)
    spec = DTensorSpec(mesh, placements, TensorMeta(torch.Size([2, 3]), (3, 1), dtype))
    return DTensor(tensor, spec, requires_grad=False)


def test_real_dtensor_ignores_only_cached_process_metadata(monkeypatch):
    tensor = distributed_tensor(monkeypatch)
    before = semantic_sha256({'model': tensor})
    tensor.device_mesh._hash = 19427658783
    tensor.device_mesh._thread_id = 938427
    assert semantic_sha256({'model': tensor}) == before
    tensor.to_local()[0, 0] = 99
    assert semantic_sha256({'model': tensor}) != before


def test_distributed_layout_and_dtype_remain_exact(monkeypatch):
    baseline = semantic_sha256(distributed_tensor(monkeypatch))
    assert semantic_sha256(distributed_tensor(monkeypatch, placements=(Replicate(),))) != baseline
    assert semantic_sha256(distributed_tensor(monkeypatch, mesh_values=(0, 1))) != baseline
    assert semantic_sha256(distributed_tensor(monkeypatch, dtype=torch.bfloat16)) != baseline


def test_mapping_order_storage_alias_and_archive_names_do_not_enter_identity(tmp_path):
    a = {'x': torch.arange(6.).reshape(2, 3), 2: {'lr': .001}, 'rng': np.arange(4, dtype=np.uint32)}
    b = dict(reversed(list(a.items())))
    b['x'] = a['x'].clone()
    torch.save(a, tmp_path / 'first.pt')
    torch.save(b, tmp_path / 'second.pt')
    assert (tmp_path / 'first.pt').read_bytes() != (tmp_path / 'second.pt').read_bytes()
    assert semantic_sha256(a) == semantic_sha256(b)
    assert semantic_sha256({2: 1}) != semantic_sha256({'2': 1})
    assert semantic_sha256([1]) != semantic_sha256((1,))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), np.array([object()]), object()])
def test_unsupported_or_nonfinite_scalar_state_fails(value):
    with pytest.raises((ValueError, TypeError)):
        semantic_sha256(value)


def test_model_optimizer_scheduler_checked_independently_before_load(tmp_path):
    model = {'qwen_lora_A': torch.ones(2, 3)}
    optimizer = {'state': {0: {'exp_avg': torch.zeros(2, 3)}}, 'param_groups': [{'lr': .01}]}
    scheduler = {'last_epoch': 2, '_last_lr': [.01]}
    path = tmp_path / 'state.json'
    write_record(path, model_state_record(model, optimizer, scheduler, rank=1, world_size=4))
    assert verify_model_state(path, model, optimizer, scheduler, rank=1, world_size=4)
    for index, field in enumerate(('model', 'optimizer', 'scheduler')):
        state = [copy.deepcopy(model), copy.deepcopy(optimizer), copy.deepcopy(scheduler)]
        state[index]['changed'] = 1
        with pytest.raises(RuntimeError, match=field):
            verify_model_state(path, *state, rank=1, world_size=4)
    with pytest.raises(RuntimeError, match='rank'):
        verify_model_state(path, model, optimizer, scheduler, rank=0, world_size=4)


def semantic_tree(root, *, teacher=True):
    (root / 'actor').mkdir(parents=True)
    write_record(root / 'driver_semantic.json', {'rng_sha256': 'a' * 64, 'dataloader_sha256': 'b' * 64})
    for rank in range(4):
        suffix = f'world_size_4_rank_{rank}.json'
        state = model_state_record({'w': torch.tensor([rank])}, {}, {}, rank=rank, world_size=4)
        write_record(root / 'actor' / f'semantic_state_{suffix}', state)
        write_record(root / 'actor' / f'worker_rng_{suffix}', {'rank': rank, 'world_size': 4, 'state': {'seed': rank}})
        if teacher:
            (root / 'actor/opd_teacher').mkdir(exist_ok=True)
            write_record(root / 'actor/opd_teacher' / f'semantic_state_{suffix}', state)
            (root / 'actor/opd_teacher' / f'ema_state_{suffix}').write_text(json.dumps({'update_count': 2, 'last_rollout_iteration': 1}))


def test_all_ranks_rng_driver_and_teacher_inventory_required(tmp_path):
    semantic_tree(tmp_path)
    identity = checkpoint_semantic_identity(tmp_path, world_size=4, require=True)
    assert identity['teacher_model_ema_sha256'] and identity['driver_rng_sha256'] == 'a' * 64
    path = tmp_path / 'actor/worker_rng_world_size_4_rank_3.json'
    path.unlink()
    with pytest.raises(RuntimeError, match='missing'):
        checkpoint_semantic_identity(tmp_path, world_size=4, require=True)


def test_old_format_optional_but_partial_new_format_fails(tmp_path):
    assert checkpoint_semantic_identity(tmp_path, world_size=4) is None
    with pytest.raises(RuntimeError, match='required'):
        checkpoint_semantic_identity(tmp_path, world_size=4, require=True)
    semantic_tree(tmp_path, teacher=False)
    assert checkpoint_semantic_identity(tmp_path, world_size=4)['teacher_model_ema_sha256'] is None
    (tmp_path / 'driver_semantic.json').unlink()
    with pytest.raises(RuntimeError, match='required'):
        checkpoint_semantic_identity(tmp_path, world_size=4)


def test_records_are_exclusive_and_authenticated(tmp_path):
    path = tmp_path / 'state.json'
    write_record(path, {'x': 1})
    with pytest.raises(FileExistsError):
        write_record(path, {'x': 2})
    state = json.loads(path.read_text()); state['x'] = 2
    path.write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match='digest'):
        read_record(path)
