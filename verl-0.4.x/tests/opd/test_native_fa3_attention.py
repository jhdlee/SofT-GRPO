"""The diagnostic FA3 bridge preserves saved-forward data and causal gradients."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

bridge = importlib.import_module("verl.opd.native_fa3_attention")


def _inputs(length=5, dim=8, *, dtype=torch.double, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(104)
    return tuple((torch.randn(1, heads, length, dim, dtype=dtype, device=device, generator=generator) * 0.3).requires_grad_()
                 for heads in (4, 2, 2))


def test_cpu_gradient_check_and_no_parameter_or_input_mutation():
    q, k, v = _inputs(length=3, dim=2)
    original = [tensor.detach().clone() for tensor in (q, k, v)]
    parameter = torch.nn.Parameter(torch.randn(2, dtype=torch.double))
    module = SimpleNamespace(weight=parameter)
    run = lambda a, b, c: bridge.native_fa3_attention(module, a, b, c, scaling=0.37)[0]
    assert torch.autograd.gradcheck(run, (q, k, v))
    assert module.weight is parameter and parameter.grad is None
    for actual, saved in zip((q, k, v), original):
        assert torch.equal(actual, saved)


def test_cpu_causality_gqa_shape_and_gradients():
    q, k, v = _inputs(length=7)
    output, weights = bridge.native_fa3_attention(None, q, k, v, position_ids=torch.arange(7)[None])
    assert output.shape == (1, 7, 4, 8) and weights is None
    changed_k, changed_v = k.detach().clone(), v.detach().clone()
    changed_k[:, :, 3:] += 1
    changed_v[:, :, 3:] -= 1
    changed_output, _ = bridge.native_fa3_attention(None, q, changed_k, changed_v)
    assert torch.equal(output[:, :3], changed_output[:, :3])
    output[:, :3].square().sum().backward()
    for tensor in (q, k, v):
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0
        assert torch.count_nonzero(tensor.grad[:, :, 3:]) == 0


@pytest.mark.parametrize("options", [
    {"attention_mask": torch.tensor([[1, 1, 1, 1, 0]])},
    {"dropout": 0.1}, {"sliding_window": 2}, {"is_causal": False},
    {"position_ids": torch.tensor([[0, 1, 0, 1, 2]])},
    {"position_ids": torch.tensor([[0, 1, 2, 4, 5]])},
    {"scaling": float("nan")}, {"scaling": 0.0},
    {"alibi_slopes": torch.ones(4)}, {"past_key_values": object()},
])
def test_unsupported_attention_semantics_fail_explicitly(options):
    with pytest.raises(ValueError):
        bridge.native_fa3_attention(None, *_inputs(), **options)


def test_rejects_second_row_and_misaligned_kv():
    q, k, v = _inputs()
    with pytest.raises(ValueError, match="one unpacked row"):
        bridge.native_fa3_attention(None, q.expand(2, -1, -1, -1), k, v)
    with pytest.raises(ValueError, match="matching head dimensions"):
        bridge.native_fa3_attention(None, q, k[:, :, :-1], v)


def test_actual_autograd_bridge_uses_saved_fa3_output_lse_without_second_forward(monkeypatch):
    calls = []
    q, k, v = (tensor.transpose(1, 2).squeeze(0).detach().contiguous().requires_grad_() for tensor in _inputs())
    output = torch.full_like(q, 0.125)
    lse = torch.arange(q.shape[1] * q.shape[0], dtype=torch.float32).reshape(q.shape[1], q.shape[0])

    def forward(a, b, c, cumulative, length, scale):
        assert a is q and b is k and c is v
        assert cumulative.dtype == torch.int32 and cumulative.tolist() == [0, 5]
        assert length == 5 and scale == 0.37
        calls.append("forward")
        return output, lse

    def backward(gradient, a, b, c, saved_output, saved_lse, cumulative, length, scale):
        assert a is q and b is k and c is v
        assert saved_output.data_ptr() == output.data_ptr()
        assert saved_lse.data_ptr() == lse.data_ptr()
        assert torch.equal(gradient, torch.ones_like(output))
        assert cumulative.tolist() == [0, 5] and length == 5 and scale == 0.37
        calls.append("backward")
        return torch.full_like(q, 2), torch.full_like(k, 3), torch.full_like(v, 4)

    monkeypatch.setattr(bridge, "_fa3_forward", forward)
    monkeypatch.setattr(bridge, "_fa2_backward", backward)
    actual = bridge._NativeFA3Attention.apply(q, k, v, 0.37)
    assert torch.equal(actual, output)
    actual.sum().backward()
    assert calls == ["forward", "backward"]
    for tensor, expected in ((q, 2), (k, 3), (v, 4)):
        assert torch.equal(tensor.grad, torch.full_like(tensor, expected))


@pytest.mark.parametrize("wrong_lse", [torch.zeros(5, 4), torch.zeros(4, 5, dtype=torch.double)])
def test_rejects_lse_layout_or_dtype_instead_of_guessing(monkeypatch, wrong_lse):
    q, k, v = (tensor.transpose(1, 2).squeeze(0).contiguous() for tensor in _inputs())
    monkeypatch.setattr(bridge, "_fa3_forward", lambda *args: (torch.zeros_like(q), wrong_lse))
    with pytest.raises(RuntimeError, match="LSE must be float32"):
        bridge._NativeFA3Attention.apply(q, k, v, 0.5)


def _stub_import(monkeypatch, package, module_name, member, implementation):
    monkeypatch.setitem(sys.modules, package, ModuleType(package))
    module = ModuleType(module_name)
    setattr(module, member, implementation)
    monkeypatch.setitem(sys.modules, module_name, module)


def test_pinned_forward_signature_and_extra_outputs(monkeypatch):
    q, k, v = (tensor.transpose(1, 2).squeeze(0).contiguous() for tensor in _inputs())
    cumulative = torch.tensor([0, 5], dtype=torch.int32)
    output, lse = torch.zeros_like(q), torch.zeros(4, 5)

    def native(*args, **kwargs):
        assert len(args) == 7
        assert args[0] is q and args[1] is k and args[2] is v
        assert args[3] is cumulative and args[4] is cumulative and args[5:] == (5, 5)
        assert kwargs == dict(softmax_scale=0.37, causal=True, num_splits=1, return_softmax_lse=True)
        return output, lse, "unused_split_output", "unused_split_lse"

    _stub_import(monkeypatch, "sgl_kernel", "sgl_kernel.flash_attn", "flash_attn_varlen_func", native)
    actual, actual_lse = bridge._fa3_forward(q, k, v, cumulative, 5, 0.37)
    assert actual is output and actual_lse is lse


def test_pinned_backward_writes_buffers_and_does_not_return_them(monkeypatch):
    q, k, v = (tensor.transpose(1, 2).squeeze(0).contiguous() for tensor in _inputs())
    output, lse = torch.zeros_like(q), torch.zeros(4, 5)
    cumulative = torch.tensor([0, 5], dtype=torch.int32)

    def backward(*args, **kwargs):
        assert len(args) == 13
        assert args[1] is q and args[2] is k and args[3] is v
        assert args[4] is output and args[5] is lse
        assert args[9] is cumulative and args[10] is cumulative and args[11:] == (5, 5)
        assert kwargs == dict(dropout_p=0.0, softmax_scale=0.37, causal=True, window_size_left=-1,
                              window_size_right=-1, softcap=0.0, alibi_slopes=None, deterministic=True, rng_state=None)
        for buffer, number in zip(args[6:9], (2, 3, 4)):
            buffer.fill_(number)
        return torch.tensor(-999.)  # softmax_d, deliberately not the gradients

    _stub_import(monkeypatch, "flash_attn", "flash_attn.flash_attn_interface", "_flash_attn_varlen_backward", backward)
    result = bridge._fa2_backward(torch.ones_like(q), q, k, v, output, lse, cumulative, 5, 0.37)
    for actual, number, original in zip(result, (2, 3, 4), (q, k, v)):
        assert actual.shape == original.shape and torch.equal(actual, torch.full_like(original, number))


def _layout(lengths=(3, 1, 5), *, device="cpu", mask=None):
    positions = torch.cat([torch.arange(length, device=device) for length in lengths])[None]
    cumulative = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32, device=device)
    layout = bridge.prepare_fa3_attention_layout(
        positions, total_tokens=sum(lengths), device=device, attention_mask=mask,
        opd_cu_seqlens=cumulative, opd_max_seqlen=max(lengths),
    )
    return positions, cumulative, layout


def test_packed_cpu_outputs_gradcheck_and_no_cross_row_gradients():
    positions, _, layout = _layout()
    q, k, v = _inputs(length=9, dim=2)
    run = lambda a, b, c: bridge.native_fa3_attention(
        None, a, b, c, scaling=0.37, opd_attention_layout=layout,
    )[0]
    actual = run(q, k, v)
    expected = torch.cat([
        bridge.native_fa3_attention(None, q[:, :, b:e], k[:, :, b:e], v[:, :, b:e], scaling=0.37)[0]
        for b, e in layout.segments
    ], 1)
    assert torch.equal(actual, expected)
    assert torch.autograd.gradcheck(run, (q, k, v))
    changed_k, changed_v = k.detach().clone(), v.detach().clone()
    changed_k[:, :, :4] += 10
    changed_v[:, :, :4] -= 10
    assert torch.equal(actual[:, 4:], run(q, changed_k, changed_v)[:, 4:])
    actual[:, 4:6].square().sum().backward()
    for tensor in (q, k, v):
        assert torch.count_nonzero(tensor.grad[:, :, :4]) == 0
        assert torch.count_nonzero(tensor.grad[:, :, 6:]) == 0
        assert tensor.grad[:, :, 4:6].abs().sum() > 0
    assert positions.tolist() == [[0, 1, 2, 0, 0, 1, 2, 3, 4]]


def test_prepared_layout_reused_without_metadata_host_reads(monkeypatch):
    positions, cumulative, layout = _layout()
    q, k, v = _inputs(length=9)
    # Caller-owned metadata can change without changing the validated copy.
    cumulative.fill_(-1)
    assert layout.cu_seqlens.tolist() == [0, 3, 4, 9]

    def forbidden(*args, **kwargs):
        raise AssertionError("attention layer attempted repeated metadata validation/host synchronization")

    monkeypatch.setattr(bridge, "prepare_fa3_attention_layout", forbidden)
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    for _ in range(3):
        output, _ = bridge.native_fa3_attention(None, q, k, v, opd_attention_layout=layout)
        assert output.shape == (1, 9, 4, 8)


@pytest.mark.parametrize("field", ["position_ids", "attention_mask", "cu_seqlens"])
def test_prepared_layout_rejects_mutated_metadata(field):
    mask = torch.ones(1, 9, dtype=torch.int64)
    _, _, layout = _layout(mask=mask)
    getattr(layout, field).add_(1)
    with pytest.raises(ValueError, match="modified in place"):
        bridge.native_fa3_attention(None, *_inputs(length=9), opd_attention_layout=layout)


def test_layout_rejects_other_forward_metadata_and_mixed_raw_arguments():
    positions, cumulative, layout = _layout()
    q, k, v = _inputs(length=9)
    with pytest.raises(ValueError, match="different position IDs"):
        bridge.native_fa3_attention(None, q, k, v, position_ids=positions.clone(), opd_attention_layout=layout)
    with pytest.raises(ValueError, match="not both"):
        bridge.native_fa3_attention(None, q, k, v, opd_attention_layout=layout,
                                   opd_cu_seqlens=cumulative, opd_max_seqlen=5)
    with pytest.raises(ValueError, match="token count"):
        bridge.native_fa3_attention(None, q[:, :, :-1], k[:, :, :-1], v[:, :, :-1], opd_attention_layout=layout)


def test_all_ones_teacher_mask_and_raw_packed_metadata_are_supported():
    q, k, v = _inputs()
    masked, _ = bridge.native_fa3_attention(None, q, k, v, attention_mask=torch.ones(1, 5))
    plain, _ = bridge.native_fa3_attention(None, q, k, v)
    assert torch.equal(masked, plain)
    positions, cumulative, layout = _layout()
    q, k, v = _inputs(length=9)
    raw, _ = bridge.native_fa3_attention(None, q, k, v, position_ids=positions,
                                       opd_cu_seqlens=cumulative, opd_max_seqlen=5)
    prepared, _ = bridge.native_fa3_attention(None, q, k, v, opd_attention_layout=layout)
    assert torch.equal(raw, prepared)


@pytest.mark.parametrize("boundaries,maximum,positions", [
    ([1, 3, 5], 3, [0, 1, 2, 0, 1]),
    ([0, 3, 4], 3, [0, 1, 2, 0, 1]),
    ([0, 3, 3, 5], 3, [0, 1, 2, 0, 1]),
    ([0, 4, 3, 5], 4, [0, 1, 2, 0, 1]),
    ([0, 3, 5], 4, [0, 1, 2, 0, 1]),
    ([0, 3, 5], True, [0, 1, 2, 0, 1]),
    ([0, 3, 5], 3, [0, 1, 2, 3, 4]),
    ([0, 3, 5], 3, [0, 1, 2, 0, 2]),
    ([0, 3, 5], 3, [1, 2, 3, 0, 1]),
    ([0, 3, 5], 3, None),
])
def test_invalid_packed_boundaries_or_positions_fail_before_kernels(boundaries, maximum, positions):
    with pytest.raises(ValueError):
        bridge.prepare_fa3_attention_layout(
            None if positions is None else torch.tensor([positions]), total_tokens=5, device="cpu",
            opd_cu_seqlens=torch.tensor(boundaries, dtype=torch.int32), opd_max_seqlen=maximum,
        )


def test_layout_rejects_noncontiguous_or_wrong_dtype_and_half_supplied_metadata():
    positions = torch.tensor([[0, 1, 2, 0, 1]])
    for cumulative, maximum in ((torch.tensor([0, 3, 5]), 3),
                                 (torch.tensor([0, 9, 3, 9, 5], dtype=torch.int32)[::2], 3),
                                 (None, 3), (torch.tensor([0, 3, 5], dtype=torch.int32), None)):
        with pytest.raises(ValueError):
            bridge.prepare_fa3_attention_layout(positions, total_tokens=5, device="cpu",
                                                opd_cu_seqlens=cumulative, opd_max_seqlen=maximum)


def test_packed_autograd_preserves_total_lse_width_and_max_segment_length(monkeypatch):
    _, _, layout = _layout()
    q, k, v = (tensor.transpose(1, 2).squeeze(0).detach().contiguous().requires_grad_() for tensor in _inputs(length=9))
    calls = []

    def forward(a, b, c, cumulative, maximum, scale):
        assert cumulative is layout.cu_seqlens and maximum == 5
        calls.append("forward")
        return torch.ones_like(a), torch.zeros(4, 9, dtype=torch.float32)

    def backward(gradient, a, b, c, output, lse, cumulative, maximum, scale):
        assert cumulative is layout.cu_seqlens and maximum == 5 and lse.shape == (4, 9)
        calls.append("backward")
        return torch.ones_like(a), torch.ones_like(b), torch.ones_like(c)

    monkeypatch.setattr(bridge, "_fa3_forward", forward)
    monkeypatch.setattr(bridge, "_fa2_backward", backward)
    bridge._NativeFA3Attention.apply(q, k, v, 0.37, layout).sum().backward()
    assert calls == ["forward", "backward"]
    assert all(tensor.grad is not None for tensor in (q, k, v))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires allocated H100 and pinned FA3/FA2 kernels")
def test_h100_actual_forward_and_mathematical_gradients():
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("pinned sgl-kernel FA3 requires Hopper")
    from flash_attn import flash_attn_func
    from sgl_kernel.flash_attn import flash_attn_varlen_func

    q, k, v = _inputs(length=33, dim=128, dtype=torch.bfloat16, device="cuda")
    original = [tensor.detach().clone() for tensor in (q, k, v)]
    output, _ = bridge.native_fa3_attention(None, q, k, v, scaling=0.13)
    packed = [tensor.transpose(1, 2).squeeze(0).contiguous() for tensor in (q, k, v)]
    cumulative = torch.tensor([0, 33], dtype=torch.int32, device="cuda")
    direct, direct_lse, *_ = flash_attn_varlen_func(
        *packed, cumulative, cumulative, 33, 33, softmax_scale=0.13, causal=True,
        num_splits=1, return_softmax_lse=True,
    )
    assert direct_lse.shape == (4, 33) and direct_lse.dtype == torch.float32
    assert torch.equal(output.squeeze(0), direct)
    output[:, :17].float().square().sum().backward()
    actual_gradients = [tensor.grad.detach().clone() for tensor in (q, k, v)]

    reference = [tensor.detach().double().requires_grad_() for tensor in (q, k, v)]
    dense = bridge._dense_attention(*reference, 0.13)
    dense[:, :17].square().sum().backward()
    fa2_inputs = [tensor.detach().transpose(1, 2).contiguous().requires_grad_() for tensor in (q, k, v)]
    fa2_output = flash_attn_func(*fa2_inputs, dropout_p=0.0, softmax_scale=0.13, causal=True, deterministic=True)
    fa2_output[:, :17].float().square().sum().backward()
    for actual, math_input, fa2_input, input_tensor, saved in zip(actual_gradients, reference, fa2_inputs, (q, k, v), original):
        assert torch.equal(input_tensor, saved)
        assert torch.isfinite(actual).all() and actual.abs().sum() > 0
        assert torch.count_nonzero(actual[:, :, 17:]) == 0
        # BF16 gradients are validated as mathematical gradients, not exact
        # derivatives of each kernel's rounding or a relaxed replay ratio gate.
        torch.testing.assert_close(actual.float(), math_input.grad.float(), rtol=0.03, atol=0.003)
        torch.testing.assert_close(actual.float(), fa2_input.grad.transpose(1, 2).float(), rtol=0.03, atol=0.003)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires allocated H100 and pinned FA3/FA2 kernels")
def test_h100_packed_forward_gradients_and_row_isolation():
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("pinned sgl-kernel FA3 requires Hopper")
    from flash_attn import flash_attn_varlen_func as fa2_varlen
    from sgl_kernel.flash_attn import flash_attn_varlen_func as fa3_varlen

    lengths = (17, 1, 129)
    positions, _, layout = _layout(lengths, device="cuda:0")
    q, k, v = _inputs(length=sum(lengths), dim=128, dtype=torch.bfloat16, device="cuda:0")
    output, _ = bridge.native_fa3_attention(None, q, k, v, scaling=0.13, opd_attention_layout=layout)
    packed = [tensor.transpose(1, 2).squeeze(0).contiguous() for tensor in (q, k, v)]
    direct, lse, *_ = fa3_varlen(*packed, layout.cu_seqlens, layout.cu_seqlens, 129, 129,
                               softmax_scale=0.13, causal=True, num_splits=1, return_softmax_lse=True)
    assert torch.equal(output.squeeze(0), direct) and lse.shape == (4, sum(lengths))
    for begin, end in layout.segments:
        single_cumulative = torch.tensor([0, end - begin], dtype=torch.int32, device=q.device)
        single = fa3_varlen(*(tensor[begin:end] for tensor in packed), single_cumulative, single_cumulative,
                            end - begin, end - begin, softmax_scale=0.13, causal=True, num_splits=1)
        torch.testing.assert_close(direct[begin:end], single, rtol=0, atol=0)
    output[:, 18:27].float().square().sum().backward()
    actual_gradients = [tensor.grad.detach().clone() for tensor in (q, k, v)]
    reference = [tensor.detach().double().requires_grad_() for tensor in (q, k, v)]
    dense, _ = bridge.native_fa3_attention(None, *(tensor.cpu() for tensor in reference), scaling=0.13,
        opd_cu_seqlens=layout.cu_seqlens.cpu(), opd_max_seqlen=129, position_ids=positions.cpu())
    dense[:, 18:27].square().sum().backward()
    fa2_inputs = [tensor.detach().requires_grad_() for tensor in packed]
    fa2_output = fa2_varlen(*fa2_inputs, layout.cu_seqlens, layout.cu_seqlens, 129, 129,
                            dropout_p=0.0, softmax_scale=0.13, causal=True, deterministic=True)
    fa2_output[18:27].float().square().sum().backward()
    for actual, math_input, fa2_input in zip(actual_gradients, reference, fa2_inputs):
        assert torch.isfinite(actual).all() and actual[:, :, 18:27].abs().sum() > 0
        assert torch.count_nonzero(actual[:, :, :18]) == 0
        assert torch.count_nonzero(actual[:, :, 27:]) == 0
        torch.testing.assert_close(actual.float(), math_input.grad.float(), rtol=0.03, atol=0.003)
        torch.testing.assert_close(actual.float(), fa2_input.grad.transpose(0, 1)[None].float(), rtol=0.03, atol=0.003)
