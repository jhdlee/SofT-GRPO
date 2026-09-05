"""Numerical checks against independent top-k/top-p behavior sampling."""

import ast
from pathlib import Path

import pytest
import torch

from verl.opd.density import fixed_support_gumbel_log_probs


def _record_action(logits, noise, *, top_p=0.95, temperature=0.1):
    """Reference the generation operations, without calling replay helpers."""

    probabilities = logits.float().softmax(-1)
    k = noise.numel()
    values, ids = probabilities.topk(k)
    filtered = torch.zeros_like(probabilities).scatter(-1, ids, values)
    filtered /= filtered.sum()
    values, ids = filtered.sort(descending=True)
    # Nucleus sampling retains the token which first crosses the threshold.
    keep = values.cumsum(-1) - values < top_p
    filtered = torch.zeros_like(filtered).scatter(-1, ids, values * keep)
    filtered /= filtered.sum()
    support_probabilities, ids = filtered.topk(k)
    support_probabilities /= support_probabilities.sum()
    base = (support_probabilities + 1e-6).log()
    noise = noise.float().clamp(-1.5, 3)
    perturbed = base + noise
    density_entries = base > -3
    expected = (-noise - (-noise).exp())[density_entries].mean()
    permutation = (perturbed / temperature).softmax(-1).argsort(descending=True)
    return (
        ids[permutation], perturbed[permutation],
        support_probabilities[permutation] > 0, expected,
    )


def _old_density(logits, ids, perturbed):
    probabilities = logits.softmax(-1).gather(-1, ids)
    base = (probabilities / probabilities.sum(-1, keepdim=True) + 1e-6).log()
    noise = (perturbed - base).clamp(-1.5, 3)
    density_entries = (base > -3).float()
    return ((-noise - (-noise).exp()) * density_entries).sum(-1) / density_entries.sum(-1)


@pytest.mark.parametrize("top_p", [0.1, 0.5, 0.95, 1.0])
def test_unchanged_policy_matches_independent_sampling(top_p):
    generator = torch.Generator().manual_seed(928)
    for _ in range(24):
        logits = torch.randn(37, generator=generator) * 1.7
        noise = torch.randn(5, generator=generator) * 2
        ids, perturbed, retained, expected = _record_action(logits, noise, top_p=top_p)
        actual = fixed_support_gumbel_log_probs(logits, ids, perturbed, retained)
        torch.testing.assert_close(actual, expected, rtol=0, atol=8e-7)
        assert abs(torch.exp(actual - expected).item() - 1.0) < 1e-6


def test_top_p_counterexample_crosses_released_density_selection_threshold():
    logits = torch.tensor([0.55, 0.22, 0.16, 0.049, 0.021]).log()
    ids, perturbed, retained, expected = _record_action(
        logits, torch.tensor([-0.2, 0.1, 0.5, -1.5, 2.0]),
    )
    assert retained.sum() == 4
    # The fourth retained token moves from .049 to .049/.979 > exp(-3).
    # Omitting behavior top-p changes both normalization and the score average.
    assert logits.softmax(-1)[3].log() < -3
    assert (0.049 / 0.979 + 1e-6) > torch.exp(torch.tensor(-3.0))
    incorrect = _old_density(logits, ids, perturbed)
    assert abs(torch.exp(incorrect - expected).item() - 1.0) > 0.4
    actual = fixed_support_gumbel_log_probs(logits, ids, perturbed, retained)
    torch.testing.assert_close(actual, expected, rtol=0, atol=3e-7)


def test_arbitrary_zero_probability_filler_ids_cannot_change_density_or_gradient():
    logits = torch.tensor([1.4, 0.7, -0.1, 100.0, -100.0, 15.0, -3.0], requires_grad=True)
    ids = torch.tensor([2, 3, 0, 4, 1])
    retained = torch.tensor([True, False, True, False, True])
    base = logits.detach().gather(-1, ids).masked_fill(~retained, -torch.inf).softmax(-1)
    action = ((base + 1e-6).log() + torch.tensor([0.1, -1.5, 0.4, 3.0, -0.2])).requires_grad_()
    first = fixed_support_gumbel_log_probs(logits, ids, action, retained)
    other_ids = ids.clone()
    other_ids[~retained] = torch.tensor([5, 6])
    second = fixed_support_gumbel_log_probs(logits, other_ids, action, retained)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    first.backward()
    assert logits.grad[[0, 1, 2]].abs().sum() > 0
    torch.testing.assert_close(logits.grad[[3, 4, 5, 6]], torch.zeros(4), rtol=0, atol=0)
    assert action.grad is None


def test_updated_policy_keeps_behavior_support_and_remains_differentiable():
    behavior_logits = torch.tensor([2.0, 1.5, 1.0, -2.0, -3.0, -4.0])
    ids, perturbed, retained, _ = _record_action(
        behavior_logits, torch.tensor([0.1, 0.3, -0.4, 0.7, -0.2]),
    )
    current_logits = (behavior_logits + torch.tensor([0.03, -0.04, 0.01, 0.1, -0.1, 50.0])).requires_grad_()
    actual = fixed_support_gumbel_log_probs(current_logits, ids, perturbed, retained)
    actual.backward()
    assert current_logits.argmax().item() == 5  # New current-policy top token is outside behavior support.
    assert current_logits.grad[5] == 0
    assert current_logits.grad.abs().sum() > 0
    # Finite differences establish a real current-policy gradient, rather than
    # simply reporting the stored behavior score as replay likelihood.
    for token_id in ids[retained].tolist():
        delta = torch.zeros_like(current_logits)
        delta[token_id] = 1e-3
        high = fixed_support_gumbel_log_probs(current_logits.detach() + delta, ids, perturbed, retained)
        low = fixed_support_gumbel_log_probs(current_logits.detach() - delta, ids, perturbed, retained)
        torch.testing.assert_close(current_logits.grad[token_id], (high - low) / 2e-3, rtol=0.02, atol=1e-4)


def test_mixed_continuous_and_categorical_wrapper_preserves_legacy():
    # Execute the actual wrapper without importing unrelated TensorDict or GPU
    # dependencies. This test must also run by itself in the CPU environment.
    path = Path(__file__).resolve().parents[2] / "verl/utils/torch_functional.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "logprobs_from_logits_topk_gumbel")
    namespace = {
        "torch": torch, "FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE": True,
        "logprobs_from_logits_flash_attn": lambda logits, labels, **_: logits.float().log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), namespace)
    replay = namespace["logprobs_from_logits_topk_gumbel"]
    logits = torch.tensor([[0.55, 0.22, 0.16, 0.049, 0.021], [0.4, 0.1, 0.2, 0.2, 0.1]]).log()
    ids, perturbed, retained, expected = _record_action(logits[0], torch.tensor([-0.2, 0.1, 0.5, -1.5, 2.0]))
    ids = torch.stack([ids, torch.tensor([3, 0, 0, 0, 0])])
    perturbed = torch.stack([perturbed, torch.zeros(5)])
    retained = torch.stack([retained, torch.tensor([True, False, False, False, False])])
    labels = ids[:, 0]
    corrected = replay(
        logits.unsqueeze(0), ids.unsqueeze(0), perturbed.unsqueeze(0), labels.unsqueeze(0),
        rollout_topk_retained_mask=retained.unsqueeze(0),
    )
    expected_categorical = logits[1].log_softmax(-1)[3]
    torch.testing.assert_close(corrected, torch.stack([expected, expected_categorical]).unsqueeze(0), rtol=0, atol=3e-7)
    legacy = replay(logits, ids, perturbed, labels)
    torch.testing.assert_close(legacy, torch.stack([_old_density(logits[0], ids[0], perturbed[0]), expected_categorical]))


@pytest.mark.parametrize("case", ["dtype", "shape", "empty"])
def test_invalid_behavior_mask_fails_closed(case):
    logits, ids, action = torch.zeros(2, 9), torch.zeros(2, 5, dtype=torch.long), torch.zeros(2, 5)
    retained = torch.ones(2, 5, dtype=torch.bool)
    if case == "dtype":
        retained = retained.float()
    elif case == "shape":
        retained = retained[:, :4]
    else:
        retained[1] = False
    with pytest.raises((TypeError, ValueError), match="Boolean|identical shapes|at least one token"):
        fixed_support_gumbel_log_probs(logits, ids, action, retained)
