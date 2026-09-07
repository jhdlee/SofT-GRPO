"""Dense-oracle and saved-activation regressions for full-vocabulary OPD KL."""

import pytest
import torch

from verl.opd import OPDConfig, PrivilegedReplay
from verl.opd.losses import full_vocab_kl, full_vocab_kl_with_statistics


def _dense_kl(student, teacher, direction, temperature):
    student_logp = torch.log_softmax(student.float() / temperature, dim=-1)
    teacher_logp = torch.log_softmax(teacher.detach().float() / temperature, dim=-1)
    if direction == "teacher_to_student":
        return (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1)
    return (student_logp.exp() * (student_logp - teacher_logp)).sum(dim=-1)


@pytest.mark.parametrize("direction", ["teacher_to_student", "student_to_teacher"])
@pytest.mark.parametrize("temperature", [0.65, 1.0, 2.3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("chunk_size", [1, 7, None])
def test_chunked_kl_matches_dense_values_weighted_gradients_and_statistics(direction, temperature, dtype, chunk_size):
    torch.manual_seed(41)
    # A strided, multidimensional input exercises vocabulary and position
    # indexing; nonuniform upstream values also exercise masked/weighted loss.
    student = torch.randn(2, 3, 38, dtype=dtype)[..., ::2].requires_grad_()
    teacher = torch.randn_like(student).requires_grad_()
    oracle_student = student.detach().clone().requires_grad_()
    weights = torch.tensor([[0.0, 1.7, -0.4], [2.1, 0.3, 0.0]])
    actual, student_logz, teacher_logz, entropy = full_vocab_kl_with_statistics(
        student, teacher, direction, temperature, chunk_size
    )
    expected = _dense_kl(oracle_student, teacher, direction, temperature)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(student_logz, torch.logsumexp(student.float() / temperature, -1), rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(teacher_logz, torch.logsumexp(teacher.float() / temperature, -1), rtol=2e-6, atol=2e-6)
    teacher_logp = torch.log_softmax(teacher.detach().float() / temperature, dim=-1)
    torch.testing.assert_close(entropy, -(teacher_logp.exp() * teacher_logp).sum(-1), rtol=2e-6, atol=2e-6)
    assert actual.dtype == torch.float32
    assert all(not value.requires_grad and value.grad_fn is None for value in (student_logz, teacher_logz, entropy))
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    # Respect input-dtype rounding, while requiring the existing FP32 replay
    # loss tolerance for float inputs; no replay acceptance tolerance changes.
    tolerance = {} if dtype != torch.float32 else {"rtol": 2e-6, "atol": 2e-6}
    torch.testing.assert_close(student.grad, oracle_student.grad, **tolerance)
    assert teacher.grad is None
    assert torch.count_nonzero(student.grad[0, 0]) == 0
    assert torch.count_nonzero(student.grad[1, 2]) == 0


@pytest.mark.parametrize("direction", ["teacher_to_student", "student_to_teacher"])
@pytest.mark.parametrize("vocab_size", [257, 1021])
def test_backward_saves_only_original_logits_and_per_position_statistics(direction, vocab_size):
    student = torch.randn(3, 5, vocab_size, dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.randn_like(student)
    saved = []

    def pack(value):
        saved.append(value)
        return value

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        result = full_vocab_kl(student, teacher, direction=direction, vocab_chunk_size=13)
    assert len(saved) == 5
    vocabulary_tensors = [value for value in saved if value.shape == student.shape]
    assert len(vocabulary_tensors) == 2
    assert {value.data_ptr() for value in vocabulary_tensors} == {student.data_ptr(), teacher.data_ptr()}
    assert all(value.dtype == torch.bfloat16 for value in vocabulary_tensors)
    assert sum(value.numel() for value in saved) == 2 * student.numel() + 3 * result.numel()
    result.sum().backward()
    assert torch.isfinite(student.grad).all()


def test_higher_order_derivatives_are_explicitly_unsupported():
    student = torch.randn(2, 7, requires_grad=True)
    loss = full_vocab_kl(student, torch.randn_like(student), vocab_chunk_size=3).sum()
    gradient = torch.autograd.grad(loss, student, create_graph=True)[0]
    with pytest.raises(RuntimeError):
        torch.autograd.grad(gradient.sum(), student)


@pytest.mark.parametrize("kwargs", [
    {"temperature": 0}, {"temperature": -0.1}, {"temperature": float("nan")},
    {"temperature": float("inf")}, {"direction": "unknown"},
    {"vocab_chunk_size": 0}, {"vocab_chunk_size": -1},
    {"vocab_chunk_size": True}, {"vocab_chunk_size": 1.5},
])
def test_invalid_kl_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        full_vocab_kl(torch.zeros(2, 3), torch.zeros(2, 3), **kwargs)


@pytest.mark.parametrize("student,teacher,error", [
    (torch.zeros(2, 3), torch.zeros(2, 4), ValueError),
    (torch.zeros(2, 0), torch.zeros(2, 0), ValueError),
    (torch.tensor(0.0), torch.tensor(0.0), ValueError),
    (torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 3), TypeError),
    (torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.long), TypeError),
    (torch.zeros(2, 3), torch.empty(2, 3, device="meta"), ValueError),
])
def test_invalid_kl_inputs_are_rejected(student, teacher, error):
    with pytest.raises(error):
        full_vocab_kl(student, teacher, vocab_chunk_size=2)


@pytest.mark.parametrize("direction", ["teacher_to_student", "student_to_teacher"])
@pytest.mark.parametrize("gate", ["all", "positive_advantage"])
def test_replay_metrics_match_dense_oracle_without_vocab_sized_diagnostic_graph(direction, gate):
    # Construct only the loss portion: teacher forward/tokenization is covered
    # separately by test_privileged_replay, and contributes no autograd here.
    replay = object.__new__(PrivilegedReplay)
    replay.config = OPDConfig(kl_direction=direction, trajectory_gate=gate, temperature=0.7, loss_support="all_response")
    torch.manual_seed(73)
    student = torch.randn(7, 29, dtype=torch.bfloat16, requires_grad=True)
    query_indices = torch.tensor([6, 0, 4, 1, 5])
    teacher = torch.randn(5, 29, dtype=torch.bfloat16)
    latent_mask = torch.tensor([[True, False, True, False], [False, True, False, False]])
    answer_mask = torch.tensor([[False, True, False, False], [True, False, False, False]])
    objective_mask = latent_mask | answer_mask
    support_ids = torch.tensor([[0, 3, 17], [2, 7, 11], [1, 5, 28]])
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda value: saved.append(value) or value, lambda value: value):
        result = replay.loss_from_teacher_logits(
            student_logits=student, student_query_indices=query_indices,
            teacher_logits=teacher, teacher_seconds=0.25,
            latent_mask=latent_mask, objective_mask=objective_mask, answer_mask=answer_mask,
            advantages=torch.tensor([1.0, -2.0]), latent_support_ids=support_ids, vocab_chunk_size=7,
        )
    # KL retains the parent student storage directly, alongside the detached
    # teacher. There is no selected student copy or vocabulary-sized FP32
    # diagnostic graph.
    vocabulary_tensors = [value for value in saved if value.ndim == 2 and value.shape[-1] == 29]
    assert len(vocabulary_tensors) == 2
    assert all(value.dtype == torch.bfloat16 for value in vocabulary_tensors)
    assert {value.data_ptr() for value in vocabulary_tensors} == {student.data_ptr(), teacher.data_ptr()}

    selected = student.detach()[query_indices].float().requires_grad_()
    dense = _dense_kl(selected, teacher, direction, 0.7)
    active = torch.tensor([True, True, True, gate == "all", gate == "all"])
    flat_latent = latent_mask[objective_mask]
    flat_answer = answer_mask[objective_mask]
    torch.testing.assert_close(result.kl_sum, dense[active].sum(), rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(result.latent_kl_sum, dense[active & flat_latent].sum(), rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(result.answer_kl_sum, dense[active & flat_answer].sum(), rtol=2e-6, atol=2e-6)
    oracle_gradient = torch.autograd.grad(dense[active].sum(), selected)[0]
    expected_support_gradient = oracle_gradient[flat_latent].gather(-1, support_ids)
    torch.testing.assert_close(result.opd_support_gradient, expected_support_gradient, rtol=2e-6, atol=2e-6)
    expected_support_logits = student.detach()[query_indices][flat_latent].gather(-1, support_ids)
    torch.testing.assert_close(result.student_support_logits, expected_support_logits)
    teacher_logp = torch.log_softmax(teacher.float() / 0.7, dim=-1)
    assert result.teacher_entropy_sum == pytest.approx(-(teacher_logp.exp() * teacher_logp).sum(-1)[active].sum().item(), rel=2e-6, abs=2e-6)
    assert result.denominator_slots == 5
    assert result.selected_slots == int(active.sum())
    assert result.latent_slots == 3
    assert result.answer_slots == 2
    assert not result.opd_support_gradient.requires_grad
    assert not result.student_support_logits.requires_grad
    result.kl_sum.backward()
    assert torch.isfinite(student.grad).all()
    expected_gradient = torch.zeros_like(student)
    expected_gradient[query_indices] = oracle_gradient.to(student.dtype)
    torch.testing.assert_close(student.grad, expected_gradient)


@pytest.mark.parametrize("support_ids,error", [
    (torch.tensor([[-1, 2], [1, 3]]), ValueError),
    (torch.tensor([[0, 7], [1, 3]]), ValueError),
    (torch.tensor([[0.0, 2.0], [1.0, 3.0]]), TypeError),
])
def test_replay_paired_support_gather_rejects_invalid_ids(support_ids, error):
    replay = object.__new__(PrivilegedReplay)
    replay.config = OPDConfig()
    with pytest.raises(error, match="latent_support_ids"):
        replay.loss_from_teacher_logits(
            student_logits=torch.randn(2, 7, requires_grad=True),
            student_query_indices=torch.arange(2), teacher_logits=torch.randn(2, 7),
            teacher_seconds=0.0, latent_mask=torch.ones(1, 2, dtype=torch.bool),
            advantages=torch.ones(1), latent_support_ids=support_ids, vocab_chunk_size=3,
        )


@pytest.mark.parametrize("direction", ["teacher_to_student", "student_to_teacher"])
@pytest.mark.parametrize("temperature", [0.65, 1.0, 2.3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("chunk_size", [1, 7, None])
@pytest.mark.parametrize("duplicate_rows", [False, True])
def test_indexed_kl_matches_dense_parent_gradients_and_statistics(
    direction, temperature, dtype, chunk_size, duplicate_rows,
):
    torch.manual_seed(117)
    student = torch.randn(9, 38, dtype=dtype)[:, ::2].requires_grad_()
    query_indices = torch.tensor([8, 1, 8, 1, 0] if duplicate_rows else [8, 1, 7, 3, 0])
    teacher = torch.randn(5, 19, dtype=dtype, requires_grad=True)
    oracle_student = student.detach().clone().requires_grad_()
    weights = torch.tensor([0.0, 1.7, -0.4, 2.1, 0.3])
    actual, student_logz, teacher_logz, entropy = full_vocab_kl_with_statistics(
        student, teacher, direction, temperature, chunk_size,
        student_row_indices=query_indices,
    )
    selected = oracle_student.index_select(0, query_indices)
    expected = _dense_kl(selected, teacher, direction, temperature)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(student_logz, torch.logsumexp(selected.float() / temperature, -1), rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(teacher_logz, torch.logsumexp(teacher.float() / temperature, -1), rtol=2e-6, atol=2e-6)
    teacher_logp = torch.log_softmax(teacher.detach().float() / temperature, -1)
    torch.testing.assert_close(entropy, -(teacher_logp.exp() * teacher_logp).sum(-1), rtol=2e-6, atol=2e-6)
    assert all(not value.requires_grad and value.grad_fn is None for value in (student_logz, teacher_logz, entropy))
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    tolerance = {} if dtype != torch.float32 else {"rtol": 2e-6, "atol": 2e-6}
    torch.testing.assert_close(student.grad, oracle_student.grad, **tolerance)
    assert teacher.grad is None
    assert torch.count_nonzero(student.grad[2]) == 0


def test_indexed_kl_saves_parent_storage_and_allocates_one_parent_gradient():
    from torch.utils._python_dispatch import TorchDispatchMode

    student = torch.randn(13, 103, requires_grad=True)
    teacher = torch.randn(7, 103)
    indices = torch.tensor([12, 2, 9, 0, 12, 5, 3])
    source_storages = {student.untyped_storage().data_ptr(), teacher.untyped_storage().data_ptr()}
    parent_gradient_storages = set()
    index_add_storages = set()
    saved = []

    class CheckVocabularyAllocations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten.index_select.default:
                assert args[0].shape[-1] <= 11
            output = func(*args, **(kwargs or {}))
            if isinstance(output, torch.Tensor) and output.ndim == 2 and output.shape[-1] == 103:
                storage = output.untyped_storage().data_ptr()
                if storage not in source_storages:
                    # The only new complete-vocabulary matrix may be the
                    # required parent gradient, never a selected-row matrix.
                    assert output.shape == student.shape
                    parent_gradient_storages.add(storage)
            if func is torch.ops.aten.index_add_.default:
                assert args[0].shape[-1] <= 11
                index_add_storages.add(args[0].untyped_storage().data_ptr())
            return output

    with CheckVocabularyAllocations():
        with torch.autograd.graph.saved_tensors_hooks(lambda value: saved.append(value) or value, lambda value: value):
            values = full_vocab_kl(student, teacher, vocab_chunk_size=11, student_row_indices=indices)
        gradient = torch.autograd.grad(values.sum(), student)[0]
    assert len(saved) == 6
    assert saved[0].data_ptr() == student.data_ptr()
    assert saved[1].data_ptr() == teacher.data_ptr()
    assert saved[-1].data_ptr() == indices.data_ptr()
    assert sum(value.numel() for value in saved) == student.numel() + teacher.numel() + 4 * indices.numel()
    assert parent_gradient_storages == index_add_storages == {gradient.untyped_storage().data_ptr()}
    assert torch.isfinite(gradient).all()


def test_indexed_kl_empty_selection_has_differentiable_zero_parent_gradient():
    student = torch.randn(3, 17, requires_grad=True)
    values = full_vocab_kl(
        student, torch.empty(0, 17), vocab_chunk_size=5,
        student_row_indices=torch.empty(0, dtype=torch.long),
    )
    assert values.shape == (0,)
    values.sum().backward()
    torch.testing.assert_close(student.grad, torch.zeros_like(student))


@pytest.mark.parametrize("student,teacher,indices,error,match", [
    (torch.zeros(3, 7), torch.zeros(2, 7), [0, 1], ValueError, "one-dimensional"),
    (torch.zeros(3, 7), torch.zeros(2, 7), torch.tensor([[0, 1]]), ValueError, "one-dimensional"),
    (torch.zeros(3, 7), torch.zeros(2, 7), torch.tensor([0.0, 1.0]), TypeError, "torch.long"),
    (torch.zeros(3, 7), torch.zeros(2, 7), torch.tensor([0, 1], dtype=torch.int32), TypeError, "torch.long"),
    (torch.zeros(3, 7), torch.zeros(2, 7), torch.tensor([0, -1]), ValueError, "out-of-range"),
    (torch.zeros(3, 7), torch.zeros(2, 7), torch.tensor([0, 3]), ValueError, "out-of-range"),
    (torch.zeros(3, 7), torch.zeros(3, 7), torch.tensor([0, 1]), ValueError, "align"),
    (torch.zeros(3, 7), torch.zeros(2, 8), torch.tensor([0, 1]), ValueError, "align"),
    (torch.zeros(3, 2, 7), torch.zeros(2, 2, 7), torch.tensor([0, 1]), ValueError, "two-dimensional"),
    (torch.zeros(3, 7), torch.zeros(2, 7), torch.empty(2, dtype=torch.long, device="meta"), ValueError, "logits device"),
])
def test_invalid_indexed_kl_inputs_are_rejected(student, teacher, indices, error, match):
    with pytest.raises(error, match=match):
        full_vocab_kl(student, teacher, vocab_chunk_size=3, student_row_indices=indices)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA indexed KL arithmetic")
@pytest.mark.parametrize("direction", ["teacher_to_student", "student_to_teacher"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_indexed_kl_duplicate_rows_match_dense_parent_gradient(direction, dtype):
    torch.manual_seed(193)
    student = torch.randn(17, 514, device="cuda", dtype=dtype)[:, ::2].requires_grad_()
    teacher = torch.randn(7, 257, device="cuda", dtype=dtype, requires_grad=True)
    indices = torch.tensor([16, 1, 9, 1, 0, 16, 3], device="cuda")
    weights = torch.tensor([0.0, 1.7, -0.4, 2.1, 0.3, 0.8, -0.2], device="cuda")
    oracle_student = student.detach().clone().requires_grad_()
    actual = full_vocab_kl(student, teacher, direction, 0.7, 31, student_row_indices=indices)
    expected = _dense_kl(oracle_student.index_select(0, indices), teacher, direction, 0.7)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    gradient = torch.autograd.grad((actual * weights).sum(), student)[0]
    expected_gradient = torch.autograd.grad((expected * weights).sum(), oracle_student)[0]
    tolerance = {} if dtype != torch.float32 else {"rtol": 2e-6, "atol": 2e-6}
    torch.testing.assert_close(gradient, expected_gradient, **tolerance)
    assert teacher.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA indexed KL memory")
def test_cuda_indexed_kl_full_qwen_vocabulary_has_one_parent_gradient_memory_bound():
    # This isolates the indexed loss allocation contract; the complete actor
    # graph and native projection still require a separate GPU integration run.
    student = torch.randn(1024, 151936, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.randn(768, 151936, device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(768, device="cuda").flip(0)
    chunk_size = 2048
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    values = full_vocab_kl(student, teacher, vocab_chunk_size=chunk_size, student_row_indices=indices)
    gradient = torch.autograd.grad(values.sum(), student)[0]
    torch.cuda.synchronize()
    peak_increment = torch.cuda.max_memory_allocated() - baseline
    # Allow one BF16 parent gradient, generous live FP32 chunk workspaces and
    # allocator/statistic overhead. Two selected-vocabulary student/gradient
    # copies would exceed this bound even without other graph activations.
    bound = student.numel() * student.element_size() + 24 * indices.numel() * chunk_size * 4 + 32 * 1024**2
    assert peak_increment <= bound, (baseline, peak_increment, bound)
    assert gradient.shape == student.shape
    assert torch.isfinite(gradient).all()
