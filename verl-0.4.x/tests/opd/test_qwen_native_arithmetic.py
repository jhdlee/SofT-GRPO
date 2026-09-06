"""Gradient and isolation checks for the opt-in forward-parity experiment."""

import copy
from types import SimpleNamespace

import pytest
import torch

from verl.opd.qwen_native_arithmetic import (
    _packed_linear,
    install_probe_candidate,
    native_rms_norm,
    native_rope,
    native_silu_mul,
)


@pytest.mark.parametrize("with_residual", [False, True])
def test_native_norm_gradients_include_weight_and_residual(with_residual):
    torch.manual_seed(21)
    x = torch.randn(2, 3, 8, dtype=torch.double, requires_grad=True)
    w = torch.randn(8, dtype=torch.double, requires_grad=True)
    residual = torch.randn_like(x, requires_grad=True)
    if with_residual:
        assert torch.autograd.gradcheck(lambda a, b, c: native_rms_norm(a, b, 1e-5, c), (x, w, residual))
    else:
        assert torch.autograd.gradcheck(lambda a, b: native_rms_norm(a, b, 1e-5), (x, w))


def test_native_norm_preserves_inputs_and_pre_rounding_sum():
    torch.manual_seed(13)
    x = torch.randn(3, 128).bfloat16().requires_grad_()
    residual = torch.randn_like(x, requires_grad=True)
    w = torch.randn(128).bfloat16().requires_grad_()
    original_x, original_r = x.detach().clone(), residual.detach().clone()
    y, carry = native_rms_norm(x, w, residual=residual)
    total = x.float() + residual.float()
    expected = (total * torch.rsqrt(total.square().mean(-1, keepdim=True) + 1e-6) * w.float()).bfloat16()
    assert torch.equal(y, expected)
    assert torch.equal(carry, total.bfloat16())
    assert torch.equal(x, original_x) and torch.equal(residual, original_r)
    (y.float().square().sum() + carry.float().square().sum()).backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0 for t in (x, w, residual))


def test_native_silu_packed_backward():
    value = torch.randn(3, 16, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(native_silu_mul, (value,))


def test_native_rope_backward_and_input_isolation():
    q = torch.randn(3, 4, 8, dtype=torch.double, requires_grad=True)
    k = torch.randn(3, 2, 8, dtype=torch.double, requires_grad=True)
    angles = torch.randn(11, 4, dtype=torch.double)
    cache = torch.cat((angles.cos(), angles.sin()), -1)
    positions = torch.tensor([0, 3, 9])
    original_q, original_k = q.detach().clone(), k.detach().clone()
    assert torch.autograd.gradcheck(lambda a, b: native_rope(a, b, positions, cache), (q, k))
    assert torch.equal(q, original_q) and torch.equal(k, original_k)


def test_packed_projection_keeps_separate_parameter_gradients():
    torch.manual_seed(31)
    modules = [torch.nn.Linear(8, size, bias=False).double() for size in (8, 4, 4)]
    value = torch.randn(2, 3, 8, dtype=torch.double, requires_grad=True)
    result = _packed_linear(value, modules)
    expected = torch.cat([module(value) for module in modules], -1)
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)
    result.square().sum().backward()
    assert all(module.weight.grad is not None and module.weight.grad.abs().sum() > 0 for module in modules)


def test_pinned_candidate_causal_gradients_and_checkpoint_identity():
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.51.1":
        pytest.skip("model-level candidate targets the pinned transformers 4.51.1 environment")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(17)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                        max_position_embeddings=32, attention_dropout=0.0, use_cache=False,
                        tie_word_embeddings=True)
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).bfloat16().eval()
    untouched = copy.deepcopy(model)
    parameters = dict(model.named_parameters())
    keys = set(model.state_dict())
    install_probe_candidate(model)
    assert set(model.state_dict()) == keys
    assert all(dict(model.named_parameters())[name] is value for name, value in parameters.items())
    assert not getattr(untouched, "_opd_native_arithmetic_candidate", False)
    embeddings = torch.randn(1, 7, 32).bfloat16().requires_grad_()
    logits = model(inputs_embeds=embeddings, use_cache=False).logits
    changed = embeddings.detach().clone(); changed[:, 4:] += 1
    changed_logits = model(inputs_embeds=changed, use_cache=False).logits
    assert torch.equal(logits[:, :4], changed_logits[:, :4])
    logits[:, :3].float().square().sum().backward()
    assert torch.count_nonzero(embeddings.grad[:, 3:]) == 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert model.model.layers[0].input_layernorm.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in untouched.parameters())


@pytest.mark.parametrize("backend,positions,mask,reject", [
    ("eager", [0, 1, 2, 0, 1], None, True),
    ("sdpa", [0, 1, 2, 0, 1], None, True),
    ("eager", [0, 0], None, True),
    ("flash_attention_2", [0, 1, 2, 0, 1], None, False),
    ("eager", [0, 1, 2, 3, 4], None, False),
    ("eager", [0, 0, 1, 2, 0], [0, 1, 1, 1, 0], False),
])
def test_candidate_rejects_unsupported_packed_attention_before_model_layers(backend, positions, mask, reject):
    class ReachedAttention(Exception):
        pass

    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(8, 8)
            self.layers = torch.nn.ModuleList()
            self.gradient_checkpointing = False

        def _update_causal_mask(self, *args):
            raise ReachedAttention

    model = SimpleNamespace(model=Core(), config=SimpleNamespace(
        model_type="qwen3", rope_scaling=None, head_dim=8,
        rope_theta=1000000., max_position_embeddings=16, _attn_implementation=backend,
    ))
    install_probe_candidate(model)
    kwargs = dict(inputs_embeds=torch.zeros(1, len(positions), 8),
                  position_ids=torch.tensor([positions]), use_cache=False)
    if mask is not None:
        kwargs["attention_mask"] = torch.tensor([mask])
    if reject:
        with pytest.raises(ValueError, match="packed position resets require flash_attention_2"):
            model.model(**kwargs)
    else:
        with pytest.raises(ReachedAttention):
            model.model(**kwargs)


def test_candidate_unsupported_model_fails_without_mutation():
    class Other:
        class Config:
            model_type = "other"
        config = Config()
    model = Other()
    with pytest.raises(ValueError, match="dense Qwen3"):
        install_probe_candidate(model)
    assert not getattr(model, "_opd_native_arithmetic_candidate", False)
