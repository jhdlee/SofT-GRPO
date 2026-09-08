"""CPU structural/lifecycle coverage; GPU replay admission remains separate."""
import copy
from types import SimpleNamespace

import pytest
import torch

from verl.opd import qwen_vllm_arithmetic as arithmetic
from verl.opd.qwen_native_arithmetic import native_rms_norm, native_rope, native_silu_mul


class UnquantizedLinearMethod:
    def apply(self, owner, value, bias=None):
        return torch.nn.functional.linear(value, owner.weight, bias)


class Norm(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(.5, 1.5, size).bfloat16())
        self.variance_epsilon = 1e-6
        self.variance_size_override = None
    def forward(self, value, residual=None):
        raise AssertionError("original normalization called")
    forward_native = forward


class Operation(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("original operation called")


class FlashAttentionImpl:
    vllm_flash_attn_version = 3


class Qwen3ForCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3", hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=128,
            max_position_embeddings=32, rope_theta=1e6, vocab_size=64, rms_norm_eps=1e-6,
            tie_word_embeddings=True, rope_scaling=None, attention_bias=False,
            hidden_act="silu", sliding_window=None)
        self.quant_config = self.lora_config = None
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(64, 256, dtype=torch.bfloat16)
        self.model.norm = Norm(256)
        self.lm_head = self.model.embed_tokens
        self.logits_processor = SimpleNamespace(scale=1., soft_cap=None, org_vocab_size=64,
                                                _get_logits=lambda *args: None)
        self.model.layers = torch.nn.ModuleList()
        rotary = Operation()
        for _ in range(2):
            layer = torch.nn.Module(); layer.self_attn = torch.nn.Module()
            a = layer.self_attn
            a.head_dim = 128; a.num_heads = 2; a.num_kv_heads = 1
            a.attn = SimpleNamespace(impl=FlashAttentionImpl())
            a.rotary_emb = rotary; a.q_norm = Norm(128); a.k_norm = Norm(128)
            layer.input_layernorm = Norm(256); layer.post_attention_layernorm = Norm(256)
            layer.mlp = torch.nn.Module(); layer.mlp.act_fn = Operation()
            for owner, name, ins, outs in ((a, "qkv_proj", 256, 512), (a, "o_proj", 256, 256),
                (layer.mlp, "gate_up_proj", 256, 1024), (layer.mlp, "down_proj", 512, 256)):
                module = torch.nn.Linear(ins, outs, bias=False, dtype=torch.bfloat16)
                module.quant_method = UnquantizedLinearMethod()
                setattr(owner, name, module)
            self.model.layers.append(layer)


def engine_config():
    return SimpleNamespace(model_config=SimpleNamespace(enforce_eager=True),
        compilation_config=SimpleNamespace(level=0),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1))


def install(model):
    return arithmetic.install_qwen_vllm_replay_arithmetic(model, copy.deepcopy(model.config),
        backend="native_fa3_v2", tensor_parallel_size=1, engine_config=engine_config())


@pytest.fixture
def cpu_install(monkeypatch):
    # Only GPU platform admission is replaced. Real model inventory and actual
    # native CPU-reference primitives execute in every installer test below.
    monkeypatch.setattr(arithmetic, "_runtime_check", lambda *args, **kwargs: None)


def test_complete_installer_matches_native_operations_and_preserves_parameters(cpu_install):
    model = Qwen3ForCausalLM(); before = {n: p.clone() for n, p in model.named_parameters()}
    attention = model.model.layers[0].self_attn; impl = attention.attn.impl
    rng = torch.get_rng_state().clone()
    identity = install(model)
    assert torch.equal(rng, torch.get_rng_state())
    assert identity["recipe"] == arithmetic.RECIPE and identity["attention"] == "stock_vllm_fa3"
    assert attention.attn.impl is impl
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())
    x = torch.randn(3, 256).bfloat16(); residual = x.clone(); initial = x.clone()
    norm = model.model.layers[0].input_layernorm
    actual = norm(x, residual)
    expected = native_rms_norm(x, norm.weight, 1e-6, residual)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert torch.equal(x, initial) and torch.equal(residual, initial)
    q, k = x.clone(), x[:, :128].clone(); positions = torch.tensor([0, 1, 2])
    actual = attention.rotary_emb(positions, q, k)
    expected = native_rope(q.reshape(3, 2, 128), k.reshape(3, 1, 128), positions, model._opd_qwen_vllm_rope_cache)
    assert all(torch.equal(a, b.flatten(1)) for a, b in zip(actual, expected))
    qnorm = attention.q_norm
    assert torch.equal(qnorm.forward_native(q.reshape(3, 2, 128)), native_rms_norm(q.reshape(3, 2, 128), qnorm.weight))
    gate = torch.randn(3, 1024).bfloat16()
    assert torch.equal(model.model.layers[0].mlp.act_fn(gate), native_silu_mul(gate))


def test_partial_install_failure_restores_every_previous_operation(cpu_install, monkeypatch):
    model = Qwen3ForCausalLM(); a = model.model.layers[0].self_attn
    method, rotary = a.qkv_proj.quant_method, a.rotary_emb
    original = arithmetic.types.MethodType
    def fail_at_final_norm(function, owner):
        if owner is model.model.norm:
            raise RuntimeError("injected final normalization installation failure")
        return original(function, owner)
    monkeypatch.setattr(arithmetic.types, "MethodType", fail_at_final_norm)
    with pytest.raises(RuntimeError, match="injected"):
        install(model)
    assert a.qkv_proj.quant_method is method
    assert "forward" not in rotary.__dict__ and "forward_native" not in a.q_norm.__dict__
    assert not hasattr(model, "_opd_qwen_vllm_arithmetic")


@pytest.mark.parametrize("fault", ["positions", "offsets"])
def test_native_rotary_rejects_context_or_offset_mismatch(cpu_install, fault):
    model = Qwen3ForCausalLM(); install(model)
    rotary = model.model.layers[0].self_attn.rotary_emb
    with pytest.raises(ValueError):
        rotary(torch.tensor([0, 32 if fault == "positions" else 1]), torch.zeros(2, 256).bfloat16(),
               torch.zeros(2, 128).bfloat16(), offsets=torch.tensor([0]) if fault == "offsets" else None)


def test_reentry_uses_current_weights_and_checks_hooks(cpu_install):
    model = Qwen3ForCausalLM(); identity = install(model)
    cache = model._opd_qwen_vllm_rope_cache; proj = model.model.layers[0].self_attn.qkv_proj
    hook = proj.quant_method; x = torch.randn(3, 256).bfloat16()
    with torch.no_grad(): proj.weight.add_(.25)
    assert install(model) == identity and proj.quant_method is hook
    assert model._opd_qwen_vllm_rope_cache is cache
    assert torch.equal(hook.apply(proj, x), torch.nn.functional.linear(x, proj.weight))
    logits = model.logits_processor._get_logits(x, model.lm_head)
    assert torch.equal(logits, torch.nn.functional.linear(x, model.lm_head.weight))
    model.model.norm.forward = lambda *args: None
    with pytest.raises(ValueError, match="hook changed"):
        install(model)


def test_reentry_detects_replaced_quant_apply(cpu_install):
    model = Qwen3ForCausalLM(); install(model)
    model.model.layers[0].self_attn.qkv_proj.quant_method.apply = lambda *args: None
    with pytest.raises(ValueError, match="hook changed"):
        install(model)


@pytest.mark.parametrize("fault", ["version", "tp", "backend", "cpu", "compile", "eager", "actual_tp", "pp", "tf32"])
def test_runtime_rejects_unsupported_execution(monkeypatch, fault):
    monkeypatch.setattr(arithmetic.importlib.metadata, "version", lambda _: "0.8.4" if fault == "version" else "0.8.5")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", fault == "tf32")
    config = engine_config()
    if fault == "compile": config.compilation_config.level = 3
    if fault == "eager": config.model_config.enforce_eager = False
    if fault == "actual_tp": config.parallel_config.tensor_parallel_size = 2
    if fault == "pp": config.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(ValueError):
        arithmetic._runtime_check(torch.device("cpu" if fault == "cpu" else "cuda:0"),
            backend="disabled" if fault == "backend" else "native_fa3_v2", tensor_parallel_size=2 if fault == "tp" else 1,
            engine_config=config)


@pytest.mark.parametrize("fault", ["dtype", "inventory", "tie", "fa2", "norm", "head", "quant", "lora", "logits", "rope"])
def test_model_restrictions_fail_before_hooks(cpu_install, fault):
    model = Qwen3ForCausalLM(); norm = model.model.norm; old = norm.forward
    if fault == "dtype": model.model.norm.weight.data = model.model.norm.weight.float()
    if fault == "inventory": model.extra = torch.nn.Parameter(torch.ones(1).bfloat16())
    if fault == "tie": model.lm_head = copy.deepcopy(model.lm_head)
    if fault == "fa2": model.model.layers[0].self_attn.attn.impl.vllm_flash_attn_version = 2
    if fault == "norm": model.model.layers[0].input_layernorm.variance_size_override = 128
    if fault == "head": model.config.head_dim = 64
    if fault == "quant": model.quant_config = object()
    if fault == "lora": model.lora_config = object()
    if fault == "logits": model.logits_processor.scale = .5
    if fault == "rope": model.config.rope_scaling = {"factor": 2}
    with pytest.raises(ValueError): install(model)
    assert norm.forward == old and not hasattr(model, "_opd_qwen_vllm_arithmetic")


def exported(model):
    result = {n: p.detach().clone() for n, p in model.named_parameters()}
    for index in range(2):
        p = f"model.layers.{index}.self_attn."
        q, k, v = result.pop(p + "qkv_proj.weight").split([256, 128, 128])
        result.update({p + name + ".weight": value for name, value in zip(("q_proj", "k_proj", "v_proj"), (q, k, v))})
        p = f"model.layers.{index}.mlp."
        gate, up = result.pop(p + "gate_up_proj.weight").chunk(2)
        result[p + "gate_proj.weight"], result[p + "up_proj.weight"] = gate, up
    result["lm_head.weight"] = result["model.embed_tokens.weight"].clone()
    return result


def test_current_fused_and_tied_export_verification():
    model = Qwen3ForCausalLM(); weights = exported(model)
    assert arithmetic.verify_qwen_vllm_weights(model, weights)["loaded_weights_exact"]
    with torch.no_grad(): model.model.layers[0].self_attn.o_proj.weight.add_(.25)
    with pytest.raises(ValueError, match="current actor export"):
        arithmetic.verify_qwen_vllm_weights(model, weights)
    assert arithmetic.verify_qwen_vllm_weights(model, exported(model))["loaded_weights_exact"]


@pytest.mark.parametrize("fault", ["missing", "extra", "tied", "fp32"])
def test_export_verification_rejects_wrong_inventory_or_dtype(fault):
    model = Qwen3ForCausalLM(); weights = exported(model)
    if fault == "missing": weights.pop("model.layers.0.self_attn.k_proj.weight")
    if fault == "extra": weights["unknown"] = torch.ones(1).bfloat16()
    if fault == "tied": weights["lm_head.weight"].add_(.25)
    if fault == "fp32": weights["model.norm.weight"] = weights["model.norm.weight"].float()
    with pytest.raises(ValueError): arithmetic.verify_qwen_vllm_weights(model, weights)
