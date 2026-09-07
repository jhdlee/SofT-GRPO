"""Gradient and isolation checks for the opt-in forward-parity experiment."""

import copy
from types import SimpleNamespace

import pytest
import torch

from verl.opd.qwen_native_arithmetic import (
    _packed_linear,
    install_probe_candidate,
    install_qwen_replay_arithmetic,
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


@pytest.mark.parametrize("kind", ["weight_dtype", "residual_dtype", "residual_shape", "weight_shape", "epsilon"])
def test_native_norm_rejects_unsupported_kernel_contract(kind):
    x, w, residual = torch.randn(2, 8).bfloat16(), torch.randn(8).bfloat16(), torch.randn(2, 8).bfloat16()
    epsilon = 1e-6
    if kind == "weight_dtype": w = w.float()
    if kind == "residual_dtype": residual = residual.float()
    if kind == "residual_shape": residual = residual.reshape(1, 16)
    if kind == "weight_shape": w = w.unsqueeze(0)
    if kind == "epsilon": epsilon = float("nan")
    with pytest.raises(ValueError):
        native_rms_norm(x, w, epsilon, residual)


@pytest.mark.parametrize("kind", ["key_dtype", "position_dtype", "negative", "outside", "cache_dtype", "cache_shape", "key_head_dim"])
def test_native_rope_rejects_unsupported_pointer_contract(kind):
    q, k = torch.randn(3, 2, 8), torch.randn(3, 1, 8)
    positions, cache = torch.tensor([0, 1, 2]), torch.randn(4, 8)
    if kind == "key_dtype": k = k.double()
    if kind == "position_dtype": positions = positions.int()
    if kind == "negative": positions[0] = -1
    if kind == "outside": positions[-1] = 4
    if kind == "cache_dtype": cache = cache.bfloat16()
    if kind == "cache_shape": cache = cache[:, :4]
    if kind == "key_head_dim": k = k[:, :, :4]
    with pytest.raises(ValueError):
        native_rope(q, k, positions, cache)


def _small_replay_model():
    """Real module/parameter interfaces without requiring a particular HF release."""
    class Norm(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(width))
            self.variance_epsilon = 1e-6

    class Layer(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.input_layernorm = Norm(64)
            self.post_attention_layernorm = Norm(64)
            self.self_attn = torch.nn.Module()
            self.self_attn.layer_idx = index
            self.self_attn.q_proj = torch.nn.Linear(64, 128, bias=False)
            self.self_attn.k_proj = torch.nn.Linear(64, 64, bias=False)
            self.self_attn.v_proj = torch.nn.Linear(64, 64, bias=False)
            self.self_attn.o_proj = torch.nn.Linear(128, 64, bias=False)
            self.self_attn.q_norm = Norm(64)
            self.self_attn.k_norm = Norm(64)
            self.self_attn.attention_dropout = 0.0
            self.self_attn.scaling = 64 ** -0.5
            self.self_attn.sliding_window = None
            self.mlp = torch.nn.Module()
            self.mlp.gate_proj = torch.nn.Linear(64, 128, bias=False)
            self.mlp.up_proj = torch.nn.Linear(64, 128, bias=False)
            self.mlp.down_proj = torch.nn.Linear(128, 64, bias=False)

    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(32, 64)
            self.layers = torch.nn.ModuleList([Layer(0), Layer(1)])
            self.norm = Norm(64)
            self.gradient_checkpointing = False

        def _update_causal_mask(self, *args):
            raise AssertionError("production layout must bypass the HF mask builder")

    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type="qwen3", rope_scaling=None, head_dim=64, hidden_act="silu",
                                   rope_theta=1000000., max_position_embeddings=32, attention_dropout=0.0,
                                   num_attention_heads=2, num_key_value_heads=1, _attn_implementation="flash_attention_2")
    model.model = Core()
    model.lm_head = torch.nn.Linear(64, 32, bias=False)
    model.lm_head.weight = model.model.embed_tokens.weight
    return model.bfloat16()


def _install_cpu_replay_reference(monkeypatch, model, **kwargs):
    # Only device resolution is mocked: exercise the real production installer,
    # core, layer, layout, and differentiable CPU reference implementations.
    import verl.opd.qwen_native_arithmetic as arithmetic
    monkeypatch.setattr(arithmetic, "_replay_cache_device", lambda device: torch.device("cpu"))
    return install_qwen_replay_arithmetic(model, cache_device="cuda:0", **kwargs)


def test_fp32_master_native_decoder_adapters_match_base_and_receive_causal_gradients(monkeypatch):
    import verl.opd.qwen_native_arithmetic as arithmetic
    from verl.opd.native_fa3_attention import native_fa3_attention
    from verl.opd.qwen_lora import install_qwen_lora
    from verl.opd.batch_invariant_linear import batch_invariant_linear

    torch.manual_seed(58)
    base = _small_replay_model()
    model = copy.deepcopy(base).float()
    install_qwen_lora(model, rank=4, alpha=8)
    _install_cpu_replay_reference(monkeypatch, base)
    arithmetic._install_native_arithmetic(
        model, linear=batch_invariant_linear, attention=native_fa3_attention,
        production=True, fp32_masters=True, cache_device="cuda:0",
    )
    embeddings = torch.randn(1, 6, 64).bfloat16().requires_grad_()
    positions = torch.tensor([[0, 1, 2, 0, 1, 2]])
    kwargs = dict(inputs_embeds=embeddings, position_ids=positions,
                  opd_cu_seqlens=torch.tensor([0, 3, 6], dtype=torch.int32), opd_max_seqlen=3)
    actual = model.lm_head(model.model(**kwargs).last_hidden_state)
    expected = base.lm_head(base.model(**kwargs).last_hidden_state)
    assert torch.equal(actual, expected)
    actual[:, :2].float().square().sum().backward()
    assert torch.count_nonzero(embeddings.grad[:, 2:]) == 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if n.endswith("qwen_lora_B"))
    assert all(p.grad is None for n, p in model.named_parameters() if "qwen_lora" not in n)


@pytest.mark.parametrize("kind", ["model", "scaled_rope", "checkpointing", "cache_device", "head_dim", "dropout", "window", "missing_head"])
def test_production_install_rejects_before_mutating_model(kind):
    model = _small_replay_model()
    if kind == "model": model.config.model_type = "other"
    if kind == "scaled_rope": model.config.rope_scaling = {"rope_type": "dynamic"}
    if kind == "checkpointing": model.model.gradient_checkpointing = True
    if kind == "head_dim": model.config.head_dim = 8
    if kind == "dropout": model.config.attention_dropout = 0.1
    if kind == "window": model.model.layers[0].self_attn.sliding_window = 8
    if kind == "missing_head": del model.lm_head
    parameters = dict(model.named_parameters())
    original_forwards = [layer.forward for layer in model.model.layers]
    with pytest.raises(ValueError):
        install_qwen_replay_arithmetic(model, cache_device="cpu")
    assert not hasattr(model.model, "_opd_native_rope_cache")
    assert not getattr(model, "_opd_native_arithmetic_candidate", False)
    assert all(dict(model.named_parameters())[name] is parameter for name, parameter in parameters.items())
    assert [layer.forward for layer in model.model.layers] == original_forwards


def test_production_packed_layout_prepared_once_causal_gradients_and_parameter_identity(monkeypatch):
    import verl.opd.native_fa3_attention as fa3
    torch.manual_seed(71)
    model = _small_replay_model()
    original_parameters, original_keys = dict(model.named_parameters()), set(model.state_dict())
    preparation, attention_layouts = [], []
    original_prepare, original_attention = fa3.prepare_fa3_attention_layout, fa3.native_fa3_attention

    def prepare(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        preparation.append(result)
        return result

    def attention(*args, **kwargs):
        attention_layouts.append(kwargs["opd_attention_layout"])
        return original_attention(*args, **kwargs)

    monkeypatch.setattr(fa3, "prepare_fa3_attention_layout", prepare)
    monkeypatch.setattr(fa3, "native_fa3_attention", attention)
    emitted = []
    _install_cpu_replay_reference(monkeypatch, model, emit=lambda name, value: emitted.append((name, value)))
    value = torch.randn(1, 6, 64).bfloat16().requires_grad_()
    positions = torch.tensor([[0, 1, 2, 0, 1, 2]])
    kwargs = dict(position_ids=positions, opd_cu_seqlens=torch.tensor([0, 3, 6], dtype=torch.int32), opd_max_seqlen=3,
                  use_cache=False)
    output = model.lm_head(model.model(inputs_embeds=value, **kwargs).last_hidden_state)
    assert len(preparation) == 1 and len(attention_layouts) == 2
    assert all(layout is preparation[0] for layout in attention_layouts)
    assert emitted[0][0] == "embedding" and emitted[-1][0] == "final_norm"
    assert all(not value.requires_grad for _, value in emitted)
    changed = value.detach().clone(); changed[:, :3] += 2
    changed_output = model.lm_head(model.model(inputs_embeds=changed, **kwargs).last_hidden_state)
    assert torch.equal(output[:, 3:], changed_output[:, 3:])
    output[:, 3:].float().square().sum().backward()
    assert torch.count_nonzero(value.grad[:, :3]) == 0
    assert value.grad[:, 3:].abs().sum() > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    assert all(dict(model.named_parameters())[name] is parameter for name, parameter in original_parameters.items())
    assert set(model.state_dict()) == original_keys
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_production_teacher_all_ones_mask_copy_and_ema_fixed_point(monkeypatch):
    import verl.opd.qwen_native_arithmetic as arithmetic
    from verl.opd.ema import freeze_teacher_, update_ema_module_
    student = _install_cpu_replay_reference(monkeypatch, _small_replay_model())
    teacher = freeze_teacher_(copy.deepcopy(student))
    student_cache = student.model._opd_native_rope_cache
    teacher_cache = teacher.model._opd_native_rope_cache
    assert student_cache.dtype == teacher_cache.dtype == torch.float32
    assert teacher_cache is not student_cache
    assert dict(student.named_buffers()).keys() == dict(teacher.named_buffers()).keys()
    before = teacher_cache.clone()
    update_ema_module_(teacher, student, 0.99)
    assert torch.equal(teacher_cache, before)
    received = []
    original_rope = arithmetic.native_rope

    def rope(*args, **kwargs):
        received.append(args[3])
        return original_rope(*args, **kwargs)

    monkeypatch.setattr(arithmetic, "native_rope", rope)
    with torch.no_grad():
        result = teacher.model(input_ids=torch.tensor([[1, 2, 3, 4]]), attention_mask=torch.ones(1, 4, dtype=torch.long),
                               position_ids=torch.arange(4).unsqueeze(0), use_cache=False)
    assert result.last_hidden_state.shape == (1, 4, 64)
    assert len(received) == 2 and all(cache is teacher_cache for cache in received)
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in teacher.parameters())


@pytest.mark.parametrize("kind,match", [("missing_layout", "packed position resets"), ("return_dict", "return_dict=True"),
                                        ("cache_cast", "FP32 RoPE cache"), ("position_range", "outside the initialized cache")])
def test_production_forward_rejects_invalid_contract_before_layers(monkeypatch, kind, match):
    model = _install_cpu_replay_reference(monkeypatch, _small_replay_model())
    for layer in model.model.layers:
        monkeypatch.setattr(layer, "forward", lambda *args, **kwargs: pytest.fail("invalid input reached a decoder layer"))
    kwargs = dict(input_ids=torch.tensor([[1, 2, 3, 4]]), position_ids=torch.tensor([[0, 1, 2, 3]]), use_cache=False)
    if kind == "missing_layout": kwargs["position_ids"] = torch.tensor([[0, 1, 0, 1]])
    if kind == "return_dict": kwargs["return_dict"] = False
    if kind == "cache_cast": model.bfloat16()
    if kind == "position_range": kwargs["position_ids"] = torch.tensor([[30, 31, 32, 33]])
    with pytest.raises(ValueError, match=match):
        model.model(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA cache construction requires a GPU")
def test_production_installer_builds_native_fp32_cache_on_gpu_with_cpu_parameters():
    from sglang.srt.layers.rotary_embedding import RotaryEmbedding
    model = _small_replay_model().float()
    parameters = dict(model.named_parameters())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        install_qwen_replay_arithmetic(model, cache_device=torch.device("cuda", torch.cuda.current_device()))
    cache = model.model._opd_native_rope_cache
    assert cache.is_cuda and cache.dtype == torch.float32
    with torch.device(cache.device):
        native = RotaryEmbedding(64, 64, 32, 1000000., True, torch.bfloat16)
    assert torch.equal(cache, native.cos_sin_cache)
    assert all(parameter.device.type == "cpu" and parameter.dtype == torch.float32 for parameter in model.parameters())
    assert all(dict(model.named_parameters())[name] is parameter for name, parameter in parameters.items())
