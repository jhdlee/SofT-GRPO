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
    {"attention_mask": torch.ones(1, 5)},
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
