import pytest
import torch

from verl.opd import (
    ddp_scaled_local_loss,
    full_vocab_kl,
    latent_kl_sum_and_count,
    latent_mask_from_topk_support,
    safe_mean_from_sum,
)


def _manual_kl(p, q):
    return torch.sum(p * (p.log() - q.log()), dim=-1)


@pytest.mark.parametrize("direction", ["teacher_to_student", "student_to_teacher"])
def test_full_vocab_kl_is_exact_fp32_and_detaches_teacher(direction):
    student = torch.tensor([[0.1, -0.5, 0.7]], dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.tensor([[0.6, -0.2, 0.0]], dtype=torch.bfloat16, requires_grad=True)
    actual = full_vocab_kl(student, teacher, direction=direction)

    student_probs = torch.softmax(student.float(), dim=-1)
    teacher_probs = torch.softmax(teacher.detach().float(), dim=-1)
    if direction == "teacher_to_student":
        expected = _manual_kl(teacher_probs, student_probs)
    else:
        expected = _manual_kl(student_probs, teacher_probs)

    assert actual.dtype is torch.float32
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert teacher.grad is None


def test_full_vocab_kl_zero_for_equal_logits():
    logits = torch.tensor([[1.0, 2.0, -1.0]], requires_grad=True)
    loss = full_vocab_kl(logits, logits.detach().clone())
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-7, rtol=0)


@pytest.mark.parametrize("direction", ["teacher_to_student", "student_to_teacher"])
def test_vocabulary_chunking_matches_unchunked_value_and_gradient(direction):
    torch.manual_seed(7)
    teacher = torch.randn(2, 3, 11)
    student_full = torch.randn(2, 3, 11, requires_grad=True)
    student_chunked = student_full.detach().clone().requires_grad_(True)

    full = full_vocab_kl(student_full, teacher, direction=direction).sum()
    chunked = full_vocab_kl(
        student_chunked,
        teacher,
        direction=direction,
        vocab_chunk_size=3,
    ).sum()
    full.backward()
    chunked.backward()

    torch.testing.assert_close(chunked, full, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(student_chunked.grad, student_full.grad, atol=2e-6, rtol=2e-6)


def test_latent_mask_uses_released_categorical_support_sentinel():
    response_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    supports = torch.tensor(
        [[[4, 3, 2, 1, 5], [9, 0, 0, 0, 0], [7, 6, 0, 0, 0], [8, 5, 4, 3, 2]]]
    )
    expected = torch.tensor([[True, False, True, False]])
    assert torch.equal(latent_mask_from_topk_support(response_mask, supports), expected)


def test_positive_gate_preserves_all_latent_slot_denominator():
    token_kl = torch.tensor([[1.0, 3.0], [5.0, 7.0]], requires_grad=True)
    latent_mask = torch.ones_like(token_kl, dtype=torch.bool)
    advantages = torch.tensor([1.0, -1.0])

    numerator, denominator, active = latent_kl_sum_and_count(
        token_kl,
        latent_mask,
        trajectory_advantages=advantages,
        gate="positive_advantage",
    )
    assert numerator.item() == 4.0
    assert denominator.item() == 4
    assert active.item() == 2
    assert safe_mean_from_sum(numerator, denominator).item() == 1.0


def test_empty_latent_mask_returns_differentiable_zero():
    values = torch.tensor([[2.0]], requires_grad=True)
    numerator, denominator, _ = latent_kl_sum_and_count(values, torch.zeros_like(values, dtype=torch.bool))
    loss = safe_mean_from_sum(numerator, denominator)
    assert loss.item() == 0.0
    loss.backward()
    assert values.grad.item() == 0.0


def test_ddp_scaling_recovers_global_slot_weighted_gradient():
    local_value = torch.tensor(3.0, requires_grad=True)
    loss = ddp_scaled_local_loss(local_value, torch.tensor(4), world_size=2)
    assert loss.item() == 1.5
    loss.backward()
    assert local_value.grad.item() == 0.5


def test_shapes_and_gate_inputs_are_validated():
    with pytest.raises(ValueError, match="identical shapes"):
        full_vocab_kl(torch.zeros(2, 3), torch.zeros(2, 4))
    with pytest.raises(ValueError, match="trajectory_advantages"):
        latent_kl_sum_and_count(torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool), gate="positive_advantage")
