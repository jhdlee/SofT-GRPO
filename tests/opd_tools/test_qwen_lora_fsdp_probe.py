"""CPU checks of isolated two/four-rank probe orchestration (no CUDA execution)."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def probe():
    path = Path(__file__).resolve().parents[2] / 'verl-0.4.x/verl/opd/qwen_lora_fsdp_probe.py'
    spec = importlib.util.spec_from_file_location('qwen_lora_fsdp_probe_cpu_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('world_size', [2, 4])
def test_probe_passes_requested_world_size_to_every_fresh_rank(probe, monkeypatch, tmp_path, world_size):
    torch, dist = ModuleType('torch'), ModuleType('torch.distributed')
    dist.is_initialized = lambda: False
    torch.distributed = dist
    torch.cuda = SimpleNamespace(device_count=lambda: world_size, get_device_capability=lambda rank: (9, 0))
    torch.version = SimpleNamespace(hip=None)
    calls = []
    def spawn(target, args, nprocs, join):
        calls.append((target, args, nprocs, join))
        directory, source, ranks = args
        assert Path(source).resolve() == Path(probe.__file__).resolve()
        assert ranks == nprocs == world_size
        for rank in range(nprocs):
            (Path(directory) / f'rank-{rank}.json').write_text(json.dumps({'rank': rank}))
        return SimpleNamespace(join=lambda timeout: True, processes=[])
    torch.multiprocessing = SimpleNamespace(spawn=spawn)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'torch.distributed', dist)
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    result = probe.validate_native_lora_fsdp_cuda(world_size=world_size)
    assert result['world_size'] == world_size
    assert result['ranks'] == [{'rank': rank} for rank in range(world_size)]
    assert len(calls) == 1 and calls[0][0] is probe._rank_probe and calls[0][3] is False
    assert not list(tmp_path.iterdir())
    torch.cuda.device_count = lambda: 4 if world_size == 2 else 2
    with pytest.raises(RuntimeError, match='isolated NVIDIA'):
        probe.validate_native_lora_fsdp_cuda(world_size=world_size)
    assert len(calls) == 1


@pytest.mark.parametrize('world_size', [False, True, 0, 1, 3, 8, 2.0])
def test_probe_rejects_unsupported_topology_before_importing_cuda(probe, world_size):
    with pytest.raises(ValueError, match='world size'):
        probe.validate_native_lora_fsdp_cuda(world_size=world_size)
