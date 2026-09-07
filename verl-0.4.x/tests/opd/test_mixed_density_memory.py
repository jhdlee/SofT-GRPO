"""Mixed replay values and first-order gradients without duplicate dense branches."""

import pytest
import torch

from verl.opd.density import fixed_support_gumbel_log_probs, mixed_support_log_probs


def _categorical(logits, labels, **kwargs):
    assert kwargs.get("inplace_backward") is False
    return logits.float().log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)


def _inputs(dtype, mode="mixed", duplicates=False):
    torch.manual_seed(871)
    # Both inputs and supports have noncontiguous storage.
    logits = torch.randn(43, 11, dtype=dtype).T.detach().requires_grad_()
    ids = torch.tensor([[1, 3, 9, 21, 42]]).expand(11, -1).clone()
    retained = torch.ones_like(ids, dtype=torch.bool)
    retained[::2, -2:] = False
    hard = torch.arange(11) % 3 != 0
    if mode == "soft":
        hard[:] = False
    elif mode == "hard":
        hard[:] = True
    if duplicates:
        ids[~hard, 1] = ids[~hard, 0]
    ids[hard, 1:] = 0
    retained[hard, 1:] = False
    actions = (torch.randn_like(ids, dtype=torch.float32) * 2).requires_grad_()
    labels = torch.arange(11) + 1
    upstream = torch.linspace(-1.2, 2.0, 22)[::2]
    upstream[0] = 0
    return logits, ids, actions, retained, labels, upstream


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["mixed", "soft", "hard"])
@pytest.mark.parametrize("duplicates", [False, True])
def test_mixed_values_and_signed_gradients_match_independent_branches(dtype, mode, duplicates):
    logits, ids, actions, retained, labels, upstream = _inputs(dtype, mode, duplicates)
    oracle = logits.detach().clone().requires_grad_()
    original = logits.detach().clone()
    expected = torch.where(
        (ids[:, 1:] == 0).all(-1),
        _categorical(oracle, labels, inplace_backward=False),
        fixed_support_gumbel_log_probs(oracle, ids, actions, retained),
    )
    actual = mixed_support_log_probs(
        logits, ids, actions, retained, labels,
        categorical_log_probs=_categorical, row_chunk_size=2,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Another differentiable consumer detects destructive CE input reuse.
    other = logits[:, 7].float().square().sum() * 0.13
    expected_other = oracle[:, 7].float().square().sum() * 0.13
    ((actual * upstream).sum() + other).backward()
    ((expected * upstream).sum() + expected_other).backward()
    tolerance = {"rtol": 2e-6, "atol": 2e-6} if dtype == torch.float32 else {}
    torch.testing.assert_close(logits.grad, oracle.grad, **tolerance)
    torch.testing.assert_close(logits.detach(), original, rtol=0, atol=0)
    assert actions.grad is None


def test_backward_recomputes_only_hard_rows_in_bounded_groups():
    logits, ids, actions, retained, labels, upstream = _inputs(torch.bfloat16)
    calls, saved = [], []

    def categorical(value, targets, **kwargs):
        calls.append((value.shape[0], torch.is_grad_enabled(), targets.tolist()))
        return _categorical(value, targets, **kwargs)

    with torch.autograd.graph.saved_tensors_hooks(lambda value: saved.append(value) or value, lambda value: value):
        score = mixed_support_log_probs(
            logits, ids, actions, retained, labels,
            categorical_log_probs=categorical, row_chunk_size=2,
        )
    vocabulary_saved = [v for v in saved if v.shape == logits.shape]
    assert len(vocabulary_saved) == 1
    assert vocabulary_saved[0].data_ptr() == logits.data_ptr()
    (score * upstream).sum().backward()
    assert calls[0][:2] == (11, False)
    hard = (ids[:, 1:] == 0).all(-1)
    assert all(rows <= 2 and enabled for rows, enabled, _ in calls[1:])
    assert [label for _, _, labels_in_call in calls[1:] for label in labels_in_call] == labels[hard].tolist()


def test_backward_has_one_parent_gradient_and_bounded_fp32_vocabulary_work():
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    rows, vocabulary, chunk = 11, 4096, 2

    class Allocations(TorchDispatchMode):
        def __init__(self):
            self.fp32_sizes = []
            self.parent_storages = set()

        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            result = function(*args, **(kwargs or {}))
            for value in tree_flatten(result)[0]:
                if not isinstance(value, torch.Tensor):
                    continue
                if value.dtype == torch.float32:
                    self.fp32_sizes.append(value.numel())
                if value.shape == (rows, vocabulary):
                    self.parent_storages.add(value.untyped_storage().data_ptr())
            return result

    logits = torch.randn(rows, vocabulary, dtype=torch.bfloat16, requires_grad=True)
    ids = torch.arange(5).expand(rows, -1).clone()
    retained = torch.ones_like(ids, dtype=torch.bool)
    ids[1::2, 1:] = 0
    retained[1::2, 1:] = False
    actions = torch.randn(rows, 5)
    score = mixed_support_log_probs(logits, ids, actions, retained, ids[:, 0], categorical_log_probs=_categorical, row_chunk_size=chunk)
    allocations = Allocations()
    with allocations:
        score.sum().backward()
    assert max(allocations.fp32_sizes) <= chunk * vocabulary
    # Leaf AccumulateGrad adopts the single custom backward allocation.
    assert allocations.parent_storages == {logits.grad.untyped_storage().data_ptr()}


@pytest.mark.parametrize("chunk", [0, -1, True, 1.5])
def test_invalid_row_chunk_is_rejected(chunk):
    logits, ids, actions, retained, labels, _ = _inputs(torch.float32)
    with pytest.raises(ValueError, match="row_chunk_size"):
        mixed_support_log_probs(logits, ids, actions, retained, labels, categorical_log_probs=_categorical, row_chunk_size=chunk)


def test_second_derivative_is_explicitly_unsupported():
    logits, ids, actions, retained, labels, _ = _inputs(torch.float32)
    score = mixed_support_log_probs(logits, ids, actions, retained, labels, categorical_log_probs=_categorical)
    gradient = torch.autograd.grad(score.sum(), logits, create_graph=True)[0]
    with pytest.raises(RuntimeError):
        torch.autograd.grad(gradient.sum(), logits)
