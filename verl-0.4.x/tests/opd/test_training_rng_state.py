import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.opd.checkpoint_semantics import semantic_sha256
from verl.opd.rng_state import (
    capture_rng_state, preserve_training_rng, restore_rng_state, restore_worker_rng,
    save_worker_rng, seed_training_rng,
)


@pytest.fixture(autouse=True)
def preserve_test_process_rng():
    with preserve_training_rng():
        yield


def draw():
    return random.random(), float(np.random.rand()), torch.rand(3).tolist()


def test_seed_is_reproducible_and_rank_namespaced():
    seed_training_rng(11, 0); a = draw()
    seed_training_rng(11, 0); assert draw() == a
    seed_training_rng(11, 1); assert draw() != a
    seed_training_rng(11, 0, namespace='driver'); assert draw() != a


@pytest.mark.parametrize('seed,rank', [(True, 0), (11, -1), (-1, 0), ('11', 0)])
def test_seed_contract_strict(seed, rank):
    with pytest.raises(ValueError):
        seed_training_rng(seed, rank)


def test_all_process_rng_restored_exactly():
    seed_training_rng(11, 3)
    saved = capture_rng_state(); expected = draw()
    seed_training_rng(239)
    restore_rng_state(saved)
    assert semantic_sha256(capture_rng_state()) == semantic_sha256(saved)
    assert draw() == expected


def test_teacher_or_checkpoint_random_work_cannot_change_actor_stream_on_error():
    seed_training_rng(11)
    saved = capture_rng_state()
    with pytest.raises(RuntimeError, match='teacher'):
        with preserve_training_rng():
            draw(); seed_training_rng(900); raise RuntimeError('teacher failed')
    assert semantic_sha256(capture_rng_state()) == semantic_sha256(saved)


def test_worker_bundle_restores_actor_and_independent_rollout_generator(tmp_path):
    seed_training_rng(11, 2)
    rollout = SimpleNamespace(gen_random_states=torch.Generator().manual_seed(812).get_state())
    saved = capture_rng_state(); generation = rollout.gen_random_states.clone()
    save_worker_rng(tmp_path, 2, 4, rollout)
    draw(); rollout.gen_random_states = torch.Generator().manual_seed(13).get_state()
    restore_worker_rng(tmp_path, 2, 4, rollout)
    assert semantic_sha256(capture_rng_state()) == semantic_sha256(saved)
    assert torch.equal(rollout.gen_random_states, generation)
    with pytest.raises(RuntimeError, match='owning'):
        restore_worker_rng(tmp_path, 2, 4)


def test_worker_without_rollout_generator_is_supported(tmp_path):
    save_worker_rng(tmp_path, 0, 4)
    assert restore_worker_rng(tmp_path, 0, 4)
    with pytest.raises(RuntimeError, match='missing rollout'):
        restore_worker_rng(tmp_path, 0, 4, SimpleNamespace(gen_random_states=torch.get_rng_state()))


def test_exact_next_update_with_stochastic_gradient_and_optimizer_resume(tmp_path):
    import copy

    seed_training_rng(11)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)

    def update():
        optimizer.zero_grad()
        x = torch.rand(4, 3) * (random.random() + float(np.random.rand()))
        torch.nn.functional.dropout(model(x), p=.4, training=True).square().mean().backward()
        optimizer.step(); scheduler.step()

    update()
    saved = copy.deepcopy((model.state_dict(), optimizer.state_dict(), scheduler.state_dict()))
    save_worker_rng(tmp_path, 0, 1)
    update()
    expected = semantic_sha256((model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), capture_rng_state()))
    # A fresh engine/teacher can consume each stream, but cannot own its restore.
    seed_training_rng(941)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
    model.load_state_dict(saved[0]); optimizer.load_state_dict(saved[1]); scheduler.load_state_dict(saved[2])
    draw()
    restore_worker_rng(tmp_path, 0, 1)
    update()
    assert semantic_sha256((model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), capture_rng_state())) == expected


def test_cuda_state_cannot_be_silently_ignored(monkeypatch):
    state = capture_rng_state()
    state['torch_cuda'] = '01'
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='without CUDA'):
        restore_rng_state(state)


def test_current_device_cuda_stream_restores_independently(monkeypatch):
    # Exercise the real capture/restore branch on CPU CI; CUDA runtime admission
    # additionally checks the actual per-worker stream on each physical GPU.
    cuda = [torch.Generator().manual_seed(81).get_state()]
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda: cuda[0].clone())
    monkeypatch.setattr(torch.cuda, 'set_rng_state', lambda state: cuda.__setitem__(0, state.clone()))
    saved = capture_rng_state()
    cuda[0] = torch.Generator().manual_seed(99).get_state()
    draw()
    restore_rng_state(saved)
    assert semantic_sha256(capture_rng_state()) == semantic_sha256(saved)
