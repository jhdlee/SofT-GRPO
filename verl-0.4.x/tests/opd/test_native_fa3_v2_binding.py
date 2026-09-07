"""Validate the source-pinned FA3 extension ABI without requiring a GPU build."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

PACKAGE = (Path(__file__).resolve().parents[3] / "Soft-Thinking+noise+loss-main" /
           "sglang_soft_thinking_pkg/sgl-kernel/native-fa3/opd_fa3")


@pytest.fixture
def interface(monkeypatch):
    native = SimpleNamespace()
    package = ModuleType("opd_fa3")
    package._C = native
    monkeypatch.setitem(sys.modules, "opd_fa3", package)
    spec = importlib.util.spec_from_file_location("opd_fa3_interface_fixture", PACKAGE / "interface.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "is_fa3_supported", lambda device=None: True)
    return module, native


def inputs():
    return [torch.zeros(5, h, 128, dtype=torch.bfloat16) for h in (4, 2, 2)]


def test_varlen_forward_preserves_boundary_and_gqa_metadata_with_pinned_split(interface):
    module, native = interface
    q, k, v = inputs()
    cumulative = torch.tensor([0, 2, 5], dtype=torch.int32)
    expected = (torch.ones_like(q), torch.zeros(4, 5), None, None)

    def forward(*args):
        assert len(args) == 33
        assert args[0] is q and args[1] is k and args[2] is v
        assert args[7] is cumulative and args[8] is cumulative
        assert args[12:14] == (3, 3)
        assert args[23:33] == (.125, True, -1, -1, 0., False, None, 1, None, 0)
        return expected

    native.fwd = forward
    result = module.flash_attn_varlen_func(q, k, v, cumulative, cumulative, 3, 3,
                                          causal=True, softmax_scale=.125, return_softmax_lse=True)
    assert all(a is b for a, b in zip(result, expected))


def test_native_backward_returns_actual_kernel_gradients_and_fixed_determinism(interface):
    module, native = interface
    q, k, v = inputs()
    cumulative = torch.tensor([0, 5], dtype=torch.int32)
    out, lse = torch.ones_like(q), torch.zeros(4, 5)
    gradient = torch.full_like(q, .5)
    expected = tuple(torch.full_like(t, i + 1) for i, t in enumerate((q, k, v)))

    def backward(*args):
        assert len(args) == 22
        assert args[0] is gradient and args[1] is q and args[2] is k and args[3] is v
        assert args[4] is out and args[5] is lse
        assert args[6:9] == (None, None, None)
        assert args[9] is cumulative and args[10] is cumulative
        assert args[11:] == (None, None, 5, 5, .125, True, -1, -1, 0., True, 0)
        return (*expected, "scratch_only", "scratch_only")

    native.bwd = backward
    actual = module.flash_attn_varlen_backward(gradient, q, k, v, out, lse, cumulative,
                                              cumulative, 5, 5, softmax_scale=.125)
    assert all(a is b for a, b in zip(actual, expected))


@pytest.mark.parametrize("option", [dict(num_splits=0), dict(num_splits=2), dict(softcap=1.),
                                    dict(window_size=(1, 1)), dict(qv=torch.zeros(5, 4, 128))])
def test_omitted_kernel_features_fail_before_dispatch(interface, option):
    module, native = interface
    native.fwd = lambda *args: pytest.fail("unsupported input reached the kernel")
    cumulative = torch.tensor([0, 5], dtype=torch.int32)
    with pytest.raises(ValueError, match="requires num_splits=1"):
        module.flash_attn_varlen_func(*inputs(), cumulative, cumulative, 5, 5, **option)


@pytest.mark.parametrize("dtype,dim", [(torch.float16, 128), (torch.float32, 128), (torch.bfloat16, 64)])
def test_wrong_dtype_or_head_dimension_fails_before_dispatch(interface, dtype, dim):
    module, native = interface
    native.fwd = lambda *args: pytest.fail("unsupported input reached the kernel")
    cumulative = torch.tensor([0, 5], dtype=torch.int32)
    q, k, v = [torch.zeros(5, h, dim, dtype=dtype) for h in (4, 2, 2)]
    with pytest.raises(ValueError, match="BF16 head128"):
        module.flash_attn_varlen_func(q, k, v, cumulative, cumulative, 5, 5)


def test_backward_cannot_disable_determinism(interface):
    module, native = interface
    q, k, v = inputs()
    cumulative = torch.tensor([0, 5], dtype=torch.int32)
    with pytest.raises(ValueError, match="deterministic native backward"):
        module.flash_attn_varlen_backward(q, q, k, v, q, torch.zeros(4, 5), cumulative,
                                          cumulative, 5, 5, softmax_scale=.125, deterministic=False)
