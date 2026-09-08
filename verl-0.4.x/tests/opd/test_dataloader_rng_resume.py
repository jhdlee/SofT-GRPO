"""Iterator reconstruction must not spend the restored training RNG twice."""
import ast
import copy
from importlib.machinery import PathFinder
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.opd.checkpoint_semantics import semantic_sha256
from verl.opd.rng_state import capture_rng_state, preserve_training_rng, restore_rng_state, resume_dataloader_iterator


@pytest.fixture(autouse=True)
def preserve_test_rng():
    with preserve_training_rng():
        yield


class StochasticDataset(torch.utils.data.Dataset):
    def __len__(self): return 8
    def __getitem__(self, index):
        return index, torch.rand(3) * (random.random() + float(np.random.rand()))


def test_exact_next_stochastic_update_preserves_real_getitem_and_dropout_draws():
    random.seed(19); np.random.seed(19); torch.manual_seed(19)
    dataset = StochasticDataset()
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=list(range(8)), num_workers=0)
    iterator = iter(loader)
    def update(batch):
        optimizer.zero_grad()
        torch.nn.functional.dropout(model(batch[1]), p=.4, training=True).square().mean().backward()
        optimizer.step()
    update(next(iterator))
    checkpoint = copy.deepcopy((model.state_dict(), optimizer.state_dict(), capture_rng_state()))
    expected_batch = next(iterator)
    update(expected_batch)
    expected = semantic_sha256((model.state_dict(), optimizer.state_dict(), capture_rng_state()))
    # Reconstructing objects and loading checkpoint state happen before RNG restore.
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    model.load_state_dict(checkpoint[0]); optimizer.load_state_dict(checkpoint[1])
    resumed = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=list(range(2, 8)), num_workers=0)
    restore_rng_state(checkpoint[2])
    before = capture_rng_state()
    iterator = resume_dataloader_iterator(resumed)
    assert capture_rng_state() == before
    actual_batch = next(iterator)
    assert capture_rng_state() != before  # legitimate stochastic data work remains owned by training
    assert torch.equal(actual_batch[0], expected_batch[0]) and torch.equal(actual_batch[1], expected_batch[1])
    update(actual_batch)
    assert semantic_sha256((model.state_dict(), optimizer.state_dict(), capture_rng_state())) == expected


def test_plain_torch_iterator_reproduces_exact_one_int64_draw_regression():
    torch.manual_seed(197)
    saved = torch.get_rng_state()
    loader = torch.utils.data.DataLoader(list(range(4)), batch_size=2, num_workers=0)
    iter(loader)
    observed = torch.get_rng_state()
    assert not torch.equal(saved, observed)
    generator = torch.Generator(); generator.set_state(saved)
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    assert torch.equal(observed, generator.get_state())
    torch.set_rng_state(saved)
    resume_dataloader_iterator(loader)
    assert torch.equal(saved, torch.get_rng_state())


def test_iterator_constructor_failure_restores_every_rng_and_propagates():
    class FailingLoader:
        def __iter__(self):
            random.random(); np.random.rand(); torch.rand(3)
            raise RuntimeError("restore failed")
    before = capture_rng_state()
    with pytest.raises(RuntimeError, match="restore failed"):
        resume_dataloader_iterator(FailingLoader())
    assert capture_rng_state() == before


@pytest.mark.parametrize("resumed,semantic,epoch,preserved", [
    (True, "qwen_semantic_v1", 0, True),
    (False, "qwen_semantic_v1", 0, False),
    (True, "qwen_semantic_v1", 1, False),
    (True, None, 0, False),
])
def test_trainer_only_protects_first_resumed_semantic_iterator(resumed, semantic, epoch, preserved):
    source = Path(__file__).resolve().parents[2] / "verl/trainer/ppo/ray_trainer.py"
    tree = ast.parse(source.read_text())
    fit = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "fit")
    loop = next(node for node in ast.walk(fit) if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                and node.target.id == "epoch")
    admission = loop.body[0]
    loader = torch.utils.data.DataLoader(list(range(4)), batch_size=2, num_workers=0)
    trainer = SimpleNamespace(_resumed=resumed, train_dataloader=loader,
                              config=SimpleNamespace(trainer={"checkpoint_semantics": semantic}))
    before = capture_rng_state()
    namespace = {"self": trainer, "epoch": epoch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[admission], type_ignores=[])), str(source), "exec"), namespace)
    assert (capture_rng_state() == before) is preserved
    assert torch.equal(next(namespace["train_iterator"]), torch.tensor([0, 1]))


@pytest.mark.parametrize("workers", [0, 2])
def test_pinned_stateful_loader_restores_same_next_batch_state_and_driver_rng(workers):
    if PathFinder.find_spec("torchdata") is None:
        pytest.skip("pinned runtime provides torchdata; local dependency is optional")
    from torchdata.stateful_dataloader import StatefulDataLoader
    def make():
        return StatefulDataLoader(list(range(24)), batch_size=4, num_workers=workers,
                                   sampler=torch.utils.data.SequentialSampler(list(range(24))))
    torch.manual_seed(251)
    loader = make(); original = iter(loader)
    next(original)
    checkpoint, rng = copy.deepcopy(loader.state_dict()), capture_rng_state()
    expected = next(original)
    expected_state, expected_rng = copy.deepcopy(loader.state_dict()), capture_rng_state()
    resumed = make(); resumed.load_state_dict(checkpoint)
    restore_rng_state(rng)
    iterator = resume_dataloader_iterator(resumed)
    assert torch.equal(next(iterator), expected)
    assert semantic_sha256(resumed.state_dict()) == semantic_sha256(expected_state)
    assert capture_rng_state() == expected_rng
