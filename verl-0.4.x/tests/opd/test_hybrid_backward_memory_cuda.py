"""Synthetic native-head + full hybrid-loss CUDA gradient/allocation gates.

The large case has two 512-token prefixes and two 8192-token responses, Qwen3's
151936-token vocabulary and its 1024-wide head. It exercises the actual retained
density, privileged replay loss and shared projection backward without generating
rollouts or loading a transformer. Passing is not FSDP/G8 training acceptance.
Use pytest --junitxml=PATH to retain the measured properties as an artifact.
"""

import hashlib
import json
import math
from pathlib import Path
import time

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="hybrid backward allocation gate requires NVIDIA CUDA"
)


@pytest.fixture
def device():
    if torch.version.hip is not None or not torch.cuda.is_bf16_supported():
        pytest.skip("native projection requires NVIDIA BF16 support")
    from verl.opd.qwen_replay_backend import validate_qwen_replay_runtime
    from verl.utils import torch_functional

    validate_qwen_replay_runtime()
    assert torch_functional.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE, "real FlashAttention CE is required"
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield torch.device("cuda", torch.cuda.current_device())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


class _DelimiterTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [{"<think>": 1, "</think>": 2}[text]]

    def decode(self, ids, skip_special_tokens=False):
        assert skip_special_tokens is False
        return {1: "<think>", 2: "</think>"}[int(ids[0])]


def _layout(device, prefix, response, vocabulary):
    rows = 2 * (prefix + response)
    positions = torch.arange(rows, device=device)
    support = ((positions[:, None] * 7 + torch.arange(5, device=device)) % (vocabulary - 1) + 1).long()
    retained = torch.ones_like(support, dtype=torch.bool)
    query = (torch.arange(2, device=device)[:, None] * (prefix + response)
             + prefix - 1 + torch.arange(response, device=device)[None, :])
    response_mask = torch.ones((2, response), device=device, dtype=torch.bool)
    response_mask[:, -1] = False  # padding must not contribute to either loss
    boundary = response * 3 // 4
    objective_mask = response_mask.clone()
    objective_mask[:, boundary] = False  # overwritten transition excluded from OPD
    latent = objective_mask & (torch.arange(response, device=device)[None, :] < boundary)
    answer = objective_mask & ~latent
    hard = query[:, boundary:].flatten()
    support[hard, 1:] = 0
    retained[hard, 1:] = False
    advantages = torch.tensor([1.0, -0.7], device=device)[:, None].expand(2, response)
    return dict(rows=rows, support=support, retained=retained, labels=support[:, 0], query=query,
                response_mask=response_mask, objective_mask=objective_mask, latent=latent,
                answer=answer, advantages=advantages)


def _actions(logits, layout):
    with torch.no_grad():
        support_logits = logits.gather(-1, layout["support"]).float()
        base = (support_logits.masked_fill(~layout["retained"], -torch.inf).softmax(-1) + 1e-6).log()
        noise = torch.tensor([0.3, -0.2, 0.1, 0.5, -0.4], device=logits.device)
        return (base + noise).requires_grad_()


def _legacy_density(logits, layout, actions):
    """Independent previous dense-promotion support graph plus the real CE op."""
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    support_logits = logits.float().gather(-1, layout["support"])
    support_logits = support_logits.masked_fill(~layout["retained"], -torch.inf)
    base = (support_logits.softmax(-1) + 1e-6).log()
    noise = (actions.detach().float() - base).clamp(-1.5, 3.0)
    selected = (base > -3).float()
    soft = ((-noise - (-noise).exp()) * selected).sum(-1) / selected.sum(-1)
    # Never allow the oracle CE backward to overwrite the shared logits.
    hard = -cross_entropy_loss(logits, layout["labels"], inplace_backward=False)[0]
    return torch.where((layout["support"][:, 1:] == 0).all(-1), hard, soft)


def _hybrid_loss(logits, teacher, layout, actions, *, oracle=False, shared=None):
    from verl.opd import OPDConfig, PrivilegedReplay
    from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
    from verl.utils.torch_functional import logprobs_from_logits_topk_gumbel

    if oracle:
        density = _legacy_density(logits, layout, actions)
    else:
        density = logprobs_from_logits_topk_gumbel(
            logits, layout["support"], actions, layout["labels"],
            inplace_backward=True, rollout_topk_retained_mask=layout["retained"],
        )
    response_density = density[layout["query"]]
    if shared is None:
        offsets = torch.tensor([-0.3, -0.1, 0.0, 0.1, 0.3], device=logits.device)
        delta = offsets[torch.arange(response_density.numel(), device=logits.device) % 5]
        shared = dict(old=response_density.detach() + delta.reshape_as(response_density),
                      reference=response_density.detach() + 0.17)
    policy, *_ = compute_policy_loss(
        old_log_prob=shared["old"], log_prob=response_density,
        advantages=layout["advantages"], response_mask=layout["response_mask"],
        cliprange=0.2, cliprange_low=0.2, cliprange_high=0.2,
        clip_ratio_c=3.0, loss_agg_mode="token-mean",
    )
    reference = agg_loss(
        loss_mat=kl_penalty(response_density, shared["reference"], "low_var_kl"),
        loss_mask=layout["response_mask"], loss_agg_mode="token-mean",
    )
    indices = layout["query"][layout["objective_mask"]]
    if oracle:
        # Dense FP32 and ordinary indexed autograd are intentionally confined
        # to the tiny numerical oracle; never construct them at Qwen scale.
        student_logp = logits.index_select(0, indices).float().log_softmax(-1)
        teacher_logp = teacher.detach().float().log_softmax(-1)
        kl = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1).mean()
    else:
        replay = PrivilegedReplay(
            torch.nn.Identity(), _DelimiterTokenizer(),
            OPDConfig(loss_support="all_response", schedule="constant", beta_base=0.001),
        )
        result = replay.loss_from_teacher_logits(
            student_logits=logits, student_query_indices=indices, teacher_logits=teacher.detach(),
            teacher_seconds=0.0, latent_mask=layout["latent"],
            objective_mask=layout["objective_mask"], answer_mask=layout["answer"],
            advantages=layout["advantages"],
            latent_support_ids=layout["support"][layout["query"][layout["latent"]]],
            vocab_chunk_size=8192,
        )
        assert result.latent_slots > 0 and result.answer_slots > 0
        kl = result.kl_sum / result.denominator_slots
    return policy + 0.001 * reference + 0.001 * kl, density, shared


def test_cuda_hybrid_shared_projection_matches_legacy_density_and_dense_kl(device):
    from verl.opd.batch_invariant_linear import batch_invariant_linear

    generator = torch.Generator(device=device).manual_seed(5701)
    layout = _layout(device, prefix=3, response=9, vocabulary=67)
    hidden = torch.randn(layout["rows"], 35, generator=generator, device=device, dtype=torch.bfloat16).requires_grad_()
    weight = (torch.randn(67, 35, generator=generator, device=device, dtype=torch.bfloat16) / math.sqrt(35)).requires_grad_()
    teacher = torch.randn(int(layout["objective_mask"].sum()), 67, generator=generator,
                          device=device, dtype=torch.bfloat16).requires_grad_()
    logits = batch_invariant_linear(hidden, weight)
    logits.retain_grad()
    oracle_logits = logits.detach().clone().requires_grad_()
    actions = _actions(logits, layout)
    actual, density, shared = _hybrid_loss(logits, teacher, layout, actions)
    expected, expected_density, _ = _hybrid_loss(oracle_logits, teacher, layout, actions,
                                               oracle=True, shared=shared)
    torch.testing.assert_close(density, expected_density, rtol=0, atol=2e-6)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    actual.backward()
    expected.backward()
    # BF16 gradient comparison uses the same dtype tolerance as the existing
    # CUDA KL dense-oracle gate. Existing strict replay tests remain unchanged.
    torch.testing.assert_close(logits.grad, oracle_logits.grad)
    gradient = logits.grad.float()
    torch.testing.assert_close(hidden.grad, (gradient @ weight.detach().float()).to(hidden.dtype))
    torch.testing.assert_close(weight.grad, (gradient.T @ hidden.detach().float()).to(weight.dtype))
    assert teacher.grad is None and actions.grad is None
    assert torch.isfinite(hidden.grad).all() and torch.isfinite(weight.grad).all()
    assert hidden.grad.abs().sum() > 0 and weight.grad.abs().sum() > 0
    # The standalone head has no attention edges: inactive prefix rows must
    # receive no gradient from masked response losses.
    assert torch.count_nonzero(hidden.grad[:2]) == 0


def test_qwen_full_hybrid_head_backward_has_bounded_cuda_peak(device, record_property):
    from verl.opd.batch_invariant_linear import batch_invariant_linear
    from verl.opd.qwen_replay_backend import qwen_replay_arithmetic_identity

    gib = 1024 ** 3
    if torch.cuda.get_device_properties(device).total_memory < 75 * gib:
        pytest.skip("full hybrid native-head allocation gate requires an 80 GB-class GPU")
    prefix, response, vocabulary, hidden_size, chunk = 512, 8192, 151936, 1024, 8192
    layout = _layout(device, prefix, response, vocabulary)
    rows = layout["rows"]
    dense_bf16 = rows * vocabulary * 2
    fp32_chunk = rows * chunk * 4
    head_bytes = vocabulary * hidden_size * 2
    # Array-derived conservative live-storage bounds. The forward allowance
    # includes two packed BF16 outputs and eight FP32 vocabulary chunks. The
    # backward allowance adds three packed BF16 branch/scatter buffers, four
    # head-sized buffers and 1 GiB of small tensors/workspace. Their combined
    # absolute limit is below the failed actor's 53.72 GiB live allocation;
    # allocator caching receives two further dense buffers/eight chunks/2 GiB.
    forward_bound = 2 * dense_bf16 + 8 * fp32_chunk + gib
    backward_bound = 3 * dense_bf16 + 8 * fp32_chunk + 4 * head_bytes + gib
    absolute_extra_bound = 5 * dense_bf16 + 8 * fp32_chunk + 4 * head_bytes + gib

    warm_x = torch.ones(2, 8, device=device, dtype=torch.bfloat16, requires_grad=True)
    warm_w = torch.ones(16, 8, device=device, dtype=torch.bfloat16, requires_grad=True)
    batch_invariant_linear(warm_x, warm_w).sum().backward()
    del warm_x, warm_w
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    generator = torch.Generator(device=device).manual_seed(5702)
    hidden = torch.randn(rows, hidden_size, generator=generator, device=device, dtype=torch.bfloat16).requires_grad_()
    weight = (torch.randn(vocabulary, hidden_size, generator=generator, device=device,
                          dtype=torch.bfloat16) / math.sqrt(hidden_size)).requires_grad_()
    pattern = ((torch.arange(vocabulary, device=device) % 127).float() - 63) / 29
    teacher = pattern.to(torch.bfloat16).expand(int(layout["objective_mask"].sum()), -1).contiguous().requires_grad_()
    del pattern
    measured = dict(schema_version=1, scope="synthetic_native_head_and_full_hybrid_loss_only",
                    packed_rows=rows, response_rows=2, prefix_tokens=prefix, response_tokens=response,
                    vocabulary=vocabulary, hidden_size=hidden_size, dtype="bfloat16", vocab_chunk_size=chunk,
                    beta=0.001, reference_coefficient=0.001, teacher_direction="teacher_to_student",
                    arithmetic=qwen_replay_arithmetic_identity(),
                    test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    verl_root = Path(__file__).resolve().parents[2] / "verl"
    measured["loss_implementation_sha256"] = {
        name: hashlib.sha256((verl_root / name).read_bytes()).hexdigest()
        for name in ("opd/losses.py", "opd/replay.py", "opd/density.py",
                     "utils/torch_functional.py", "trainer/ppo/core_algos.py")
    }
    try:
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        absolute_bound = baseline + absolute_extra_bound
        reserved_bound = absolute_bound + 2 * dense_bf16 + 8 * fp32_chunk + 2 * gib
        assert absolute_bound < 53.72 * gib and reserved_bound < 77.96 * gib
        measured.update(initial_allocated_bytes=baseline, initial_reserved_bytes=torch.cuda.memory_reserved(device),
                        forward_increment_bound_bytes=forward_bound, backward_increment_bound_bytes=backward_bound,
                        absolute_allocated_bound_bytes=absolute_bound, absolute_reserved_bound_bytes=reserved_bound)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        logits = batch_invariant_linear(hidden, weight)
        actions = _actions(logits, layout)
        loss, density, shared = _hybrid_loss(logits, teacher, layout, actions)
        torch.cuda.synchronize(device)
        forward_peak = torch.cuda.max_memory_allocated(device)
        forward_reserved = torch.cuda.max_memory_reserved(device)
        measured.update(forward_seconds=time.perf_counter() - started,
                        forward_peak_allocated_bytes=forward_peak, forward_peak_reserved_bytes=forward_reserved,
                        forward_increment_bytes=forward_peak - baseline, loss=float(loss.detach()))
        assert forward_peak - baseline <= forward_bound
        assert torch.isfinite(loss)
        backward_baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(device)
        backward_peak = torch.cuda.max_memory_allocated(device)
        backward_reserved = torch.cuda.max_memory_reserved(device)
        measured.update(backward_seconds=time.perf_counter() - started,
                        backward_baseline_allocated_bytes=backward_baseline,
                        backward_peak_allocated_bytes=backward_peak, backward_peak_reserved_bytes=backward_reserved,
                        backward_increment_bytes=backward_peak - backward_baseline)
        assert backward_peak - backward_baseline <= backward_bound
        assert max(forward_peak, backward_peak) <= absolute_bound
        assert max(forward_reserved, backward_reserved) <= reserved_bound
        # Inspect in small slices after capturing peaks. Avoid a dense FP32
        # gradient copy or a full-vocabulary Boolean finite-check allocation.
        for parameter in (hidden, weight):
            assert parameter.grad is not None and parameter.grad.dtype == torch.bfloat16
            assert any(torch.count_nonzero(part).item() for part in parameter.grad.split(256))
            for part in parameter.grad.split(256):
                assert torch.isfinite(part).all()
        assert teacher.grad is None and actions.grad is None
        assert torch.count_nonzero(hidden.grad[:prefix - 1]) == 0
        measured.update(student_gradients_finite=True, teacher_gradient_is_none=True,
                        action_gradient_is_none=True, status="passed")
    except BaseException as error:
        measured.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record_property("hybrid_backward_probe", json.dumps(measured, sort_keys=True, allow_nan=False))
        hidden.grad = None
        weight.grad = None
        teacher.grad = None
