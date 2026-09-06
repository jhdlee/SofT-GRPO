"""CUDA gates for memory-bounded, full-vocabulary OPD KL.

The large synthetic gate isolates KL allocation at two 8192-token responses
and Qwen3's 151936-token vocabulary. It is not an end-to-end actor memory or
training acceptance result. No model downloads or checkpoints are involved.
"""

import json

import pytest
import torch

from verl.opd.losses import full_vocab_kl


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="OPD allocation regression requires NVIDIA CUDA"
)


def _cuda_device():
    if torch.version.hip is not None or not torch.cuda.is_bf16_supported():
        pytest.skip("OPD allocation regression requires NVIDIA BF16 support")
    return torch.device("cuda", torch.cuda.current_device())


@pytest.mark.parametrize("direction", ["teacher_to_student", "student_to_teacher"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_chunked_kl_matches_dense_value_and_weighted_student_gradient(direction, dtype):
    device = _cuda_device()
    temperature = 0.7
    # Fixed arithmetic patterns avoid dependence on any global RNG state.
    coordinate = torch.arange(13 * 67, device=device, dtype=torch.float32).reshape(13, 67)
    student = ((coordinate.remainder(43) - 21) / 13).to(dtype).requires_grad_()
    teacher = ((coordinate.remainder(37) - 18) / 11).to(dtype).requires_grad_()
    oracle_student = student.detach().clone().requires_grad_()
    weights = torch.tensor([0.0, 0.3, -0.7, 1.2, 0.1, 2.0, 0.0, -0.1, 1.0, 0.2, 0.8, 0.0, 1.4], device=device)
    actual = full_vocab_kl(student, teacher, direction, temperature, vocab_chunk_size=17)
    student_logp = torch.log_softmax(oracle_student.float() / temperature, dim=-1)
    teacher_logp = torch.log_softmax(teacher.detach().float() / temperature, dim=-1)
    if direction == "teacher_to_student":
        expected = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1)
    else:
        expected = (student_logp.exp() * (student_logp - teacher_logp)).sum(dim=-1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    tolerance = {"atol": 2e-6, "rtol": 2e-6} if dtype == torch.float32 else {}
    torch.testing.assert_close(student.grad, oracle_student.grad, **tolerance)
    assert teacher.grad is None
    assert torch.isfinite(student.grad).all()
    assert torch.count_nonzero(student.grad[weights == 0]) == 0


def test_qwen_two_full_responses_kl_has_bounded_cuda_allocations(record_property):
    device = _cuda_device()
    gib = 1024 ** 3
    if torch.cuda.get_device_properties(device).total_memory < 24 * gib:
        pytest.skip("two full Qwen response KL allocation gate requires at least 24 GiB GPU memory")
    positions, vocabulary, chunk_size = 2 * 8192, 151936, 8192
    dtype = torch.bfloat16
    shape = (positions, vocabulary)
    fp32_chunk_bytes = positions * chunk_size * torch.empty((), dtype=torch.float32).element_size()
    output_gradient_bytes = positions * vocabulary * torch.empty((), dtype=dtype).element_size()
    # Eight live FP32 chunks conservatively cover normalizer reductions,
    # log-probabilities, probabilities, products, and overlapping expression
    # temporaries. Per-position outputs and CUDA allocator rounding receive a
    # further 64 MiB. A dense FP32 vocabulary tensor alone is about 9.27 GiB;
    # the forward limit is about 4.06 GiB, independent of vocabulary size.
    temporary_bound = 8 * fp32_chunk_bytes + 64 * 1024 ** 2
    backward_bound = output_gradient_bytes + temporary_bound
    record_property("shape", json.dumps(list(shape)))
    record_property("response_rows", 2)
    record_property("response_tokens_per_row", 8192)
    record_property("dtype", str(dtype))
    record_property("vocab_chunk_size", chunk_size)
    record_property("direction", "teacher_to_student")
    record_property("temperature", 0.7)
    record_property("fp32_chunk_bytes", fp32_chunk_bytes)
    record_property("output_gradient_bytes", output_gradient_bytes)
    record_property("forward_increment_bound_bytes", temporary_bound)
    record_property("backward_increment_bound_bytes", backward_bound)

    # Warm CUDA kernels and allocator bookkeeping on tiny inputs, then begin
    # fresh allocation measurements with both real, materialized logits live.
    warm_student = torch.zeros(2, 19, device=device, dtype=dtype, requires_grad=True)
    full_vocab_kl(warm_student, torch.ones_like(warm_student), vocab_chunk_size=7).sum().backward()
    del warm_student
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    coordinate = torch.arange(vocabulary, device=device, dtype=torch.float32)
    student_pattern = ((coordinate.remainder(113) - 56) / 31).to(dtype)
    teacher_pattern = ((coordinate.remainder(127) - 63) / 29).to(dtype)
    student = student_pattern.expand(shape).contiguous().requires_grad_()
    teacher = teacher_pattern.expand(shape).contiguous().requires_grad_()
    del coordinate, student_pattern, teacher_pattern
    assert student.is_contiguous() and teacher.is_contiguous()
    assert student.numel() * student.element_size() == output_gradient_bytes
    assert teacher.numel() * teacher.element_size() == output_gradient_bytes
    try:
        torch.cuda.synchronize(device)
        forward_baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        token_kl = full_vocab_kl(student, teacher, temperature=0.7, vocab_chunk_size=chunk_size)
        torch.cuda.synchronize(device)
        forward_peak = torch.cuda.max_memory_allocated(device)
        forward_increment = forward_peak - forward_baseline
        record_property("forward_baseline_bytes", forward_baseline)
        record_property("forward_peak_bytes", forward_peak)
        record_property("forward_increment_bytes", forward_increment)
        assert forward_increment <= temporary_bound, (
            f"KL forward allocation at shape={shape}, dtype={dtype}, chunk={chunk_size}: "
            f"increment={forward_increment} bytes exceeds bound={temporary_bound} bytes"
        )
        assert token_kl.shape == (positions,) and token_kl.dtype == torch.float32
        assert torch.isfinite(token_kl).all()
        assert torch.all(token_kl > 0)

        torch.cuda.synchronize(device)
        backward_baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        token_kl.mean().backward()
        torch.cuda.synchronize(device)
        backward_peak = torch.cuda.max_memory_allocated(device)
        backward_increment = backward_peak - backward_baseline
        record_property("backward_baseline_bytes", backward_baseline)
        record_property("backward_peak_bytes", backward_peak)
        record_property("backward_increment_bytes", backward_increment)
        assert backward_increment <= backward_bound, (
            f"KL backward allocation at shape={shape}, dtype={dtype}, chunk={chunk_size}: "
            f"increment={backward_increment} bytes exceeds bound={backward_bound} bytes"
        )
        assert student.grad is not None and student.grad.dtype == dtype
        assert teacher.grad is None
        # Capture peaks before validation. Check finite gradients in bounded
        # vocabulary slices without copying the full output or casting to FP32.
        for gradient_chunk in student.grad.split(chunk_size, dim=-1):
            assert torch.isfinite(gradient_chunk).all()
        del gradient_chunk
        assert torch.count_nonzero(student.grad[0]) > 0
        record_property("student_gradient_finite", True)
        record_property("teacher_gradient_is_none", True)
        record_property("status", "passed")
    finally:
        student.grad = None
        del student, teacher
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
