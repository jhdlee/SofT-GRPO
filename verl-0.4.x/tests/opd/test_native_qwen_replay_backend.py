"""CPU safety contracts for the opt-in native Qwen replay backend."""
import argparse
import ast
import copy
import dataclasses
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
NATIVE = ROOT / "Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python/sglang/srt"
spec = importlib.util.spec_from_file_location("opd_test_native_qwen_replay", NATIVE / "layers/qwen_replay_arithmetic.py")
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


def args(**changes):
    values = dict(opd_qwen_replay_backend="native_fa3_v1", tp_size=1, dp_size=1, ep_size=1,
                  device="cuda", dtype="bfloat16", kv_cache_dtype="auto", attention_backend="fa3",
                  disable_cuda_graph=True, disable_overlap_schedule=True, disable_radix_cache=True)
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("change", [
    {"opd_qwen_replay_backend": "unknown"}, {"tp_size": 2}, {"dp_size": 2}, {"ep_size": 2},
    {"device": "cpu"}, {"device": None}, {"dtype": "auto"}, {"dtype": "float16"},
    {"attention_backend": "flashinfer"}, {"attention_backend": None},
    {"disable_cuda_graph": False}, {"disable_overlap_schedule": False}, {"disable_radix_cache": False},
    {"quantization": "fp8"}, {"quantization_param_path": "scales.json"}, {"kv_cache_dtype": "fp8_e4m3"},
    {"torchao_config": "int8wo"}, {"lora_paths": ["adapter"]}, {"speculative_algorithm": "EAGLE"},
    {"speculative_draft_model_path": "draft"}, {"enable_dp_attention": True}, {"enable_ep_moe": True},
])
def test_selected_backend_rejects_unsupported_recipe_before_import_or_model_mutation(change):
    with pytest.raises(ValueError, match="native_fa3_v1|opd_qwen_replay_backend"):
        backend.validate_qwen_replay_server_args(args(**change))


def test_disabled_recipe_is_unmodified_and_does_not_require_verl_or_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "verl.opd.batch_invariant_linear", None)
    old = SimpleNamespace(opd_qwen_replay_backend="disabled", device="cpu", dtype="auto", tp_size=8)
    model = object()
    assert backend.install_qwen_replay_backend(model, None, old) == {"backend": "disabled"}
    assert backend.validate_qwen_replay_server_args(SimpleNamespace()) is False


def test_fa3_split_policy_is_instance_scoped_and_preserves_disabled_defaults():
    calls = []
    def varlen(*a, num_splits=1, **kw):
        calls.append(("varlen", num_splits))
        return a, kw
    def cache(*a, num_splits=0, **kw):
        calls.append(("cache", num_splits))
        return a, kw
    default = backend.fa3_replay_callables(SimpleNamespace(), varlen, cache)
    selected = backend.fa3_replay_callables(args(), varlen, cache)
    assert default == (varlen, cache)
    assert selected[0] is not varlen and selected[1] is not cache
    assert selected[0](1, causal=True) == ((1,), {"causal": True})
    selected[1](2)
    default[0](3); default[1](4)
    assert calls == [("varlen", 1), ("cache", 1), ("varlen", 1), ("cache", 0)]


class Unquantized:
    def apply(self, layer, value, bias=None):
        return torch.nn.functional.linear(value, layer.weight, bias)


class Projection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 4))
        self.quant_method = Unquantized()
    def forward(self, value, bias=None):
        return self.quant_method.apply(self, value, bias)


class FakeQwen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.self_attn, layer.mlp = torch.nn.Module(), torch.nn.Module()
        layer.self_attn.qkv_proj, layer.self_attn.o_proj = Projection(), Projection()
        layer.mlp.gate_up_proj, layer.mlp.down_proj = Projection(), Projection()
        self.model.layers = torch.nn.ModuleList([layer])
        self.logits_processor = SimpleNamespace(config=SimpleNamespace(vocab_size=4))


def test_private_quant_method_and_head_keep_parameter_and_default_instance_identity():
    model, untouched = FakeQwen(), FakeQwen()
    before = dict(model.named_parameters())
    state = copy.deepcopy(model.state_dict())
    original = model.model.layers[0].self_attn.qkv_proj.quant_method
    observed = []
    def linear(x, weight, bias=None):
        observed.append(weight)
        return torch.nn.functional.linear(x, weight, bias)
    assert backend._install_linear(model, linear, Unquantized) == 4
    module = model.model.layers[0].self_attn.qkv_proj
    assert module.quant_method is not original
    value = torch.randn(3, 4)
    assert torch.equal(module(value), original.apply(module, value))
    assert torch.equal(model.logits_processor._get_logits(value, module, None), module(value).float())
    assert all(dict(model.named_parameters())[name] is parameter for name, parameter in before.items())
    assert all(torch.equal(model.state_dict()[name], saved) for name, saved in state.items())
    assert "apply" not in untouched.model.layers[0].self_attn.qkv_proj.quant_method.__dict__
    with pytest.raises(ValueError, match="embedding bias"):
        model.logits_processor._get_logits(value, module, None, embedding_bias=torch.zeros(4))


def test_invalid_projection_structure_fails_before_any_quant_method_is_changed():
    model = FakeQwen()
    original = model.model.layers[0].self_attn.qkv_proj.quant_method
    model.model.layers[0].mlp.down_proj.quant_method = object()
    with pytest.raises(ValueError, match="expected 4"):
        backend._install_linear(model, torch.nn.functional.linear, Unquantized)
    assert model.model.layers[0].self_attn.qkv_proj.quant_method is original


def qwen_config(**changes):
    fields = dict(model_type="qwen3", hidden_size=1024, intermediate_size=3072,
                  num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
                  head_dim=128, vocab_size=151936)
    fields.update(changes)
    return SimpleNamespace(**fields)


def test_loaded_model_guard_rejects_other_architectures_and_cpu_weights():
    Qwen3ForCausalLM = type("Qwen3ForCausalLM", (FakeQwen,), {})
    model = Qwen3ForCausalLM()
    with pytest.raises(ValueError, match="unscaled dense"):
        backend._validate_loaded_model(FakeQwen(), qwen_config())
    with pytest.raises(ValueError, match="hidden_size"):
        backend._validate_loaded_model(model, qwen_config(hidden_size=2048))
    with pytest.raises(ValueError, match="unscaled dense"):
        backend._validate_loaded_model(model, qwen_config(rope_scaling={"factor": 2}))
    with pytest.raises(ValueError, match="BF16 CUDA"):
        backend._validate_loaded_model(model, qwen_config())


def test_full_install_exposes_small_provenance_and_is_not_reinstalled(monkeypatch):
    from verl.opd import qwen_replay_backend
    model = FakeQwen()
    monkeypatch.setattr(backend, "_validate_loaded_model", lambda model, config: None)
    monkeypatch.setattr(qwen_replay_backend, "validate_qwen_replay_runtime", lambda: dict(qwen_replay_backend.QWEN_REPLAY_RUNTIME))
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.linear", SimpleNamespace(UnquantizedLinearMethod=Unquantized))
    result = backend.install_qwen_replay_backend(model, qwen_config(), args())
    assert result["backend"] == "native_fa3_v1" and result["projection_module_count"] == 4
    assert result["num_splits"] == 1 and result["dtype"] == "bfloat16"
    assert result["runtime_versions"] == qwen_replay_backend.QWEN_REPLAY_RUNTIME
    assert set(result["source_sha256"]) == {"native_integration", "linear", "attention_backend"}
    assert all(len(value) == 64 for value in result["source_sha256"].values())
    with pytest.raises(ValueError, match="already installed"):
        backend.install_qwen_replay_backend(model, qwen_config(), args())


def test_server_args_default_and_cli_enum_use_the_actual_dataclass_definition():
    tree = ast.parse((NATIVE / "server_args.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ServerArgs")
    field = next(node for node in cls.body if isinstance(node, ast.AnnAssign) and node.target.id == "opd_qwen_replay_backend")
    assert ast.literal_eval(field.value) == "disabled"
    cli = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "add_cli_args")
    call = next(node for node in ast.walk(cli) if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--opd-qwen-replay-backend")
    assert ast.literal_eval(next(x.value for x in call.keywords if x.arg == "choices")) == ["disabled", "native_fa3_v1"]
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__")
    assert isinstance(init.body[0], ast.If) and "validate_qwen_replay_server_args" in ast.unparse(init.body[0])


def test_model_install_precedes_pool_and_graph_initialization_and_all_fa3_calls_are_instance_scoped():
    tree = ast.parse((NATIVE / "model_executor/model_runner.py").read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "ModelRunner")
    initialize = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "initialize")
    calls = sorted((n.lineno, ast.unparse(n.func)) for n in ast.walk(initialize) if isinstance(n, ast.Call))
    line = lambda name:next(number for number, target in calls if target.endswith(name))
    assert line("load_model") < line("install_qwen_replay_backend") < line("init_memory_pool") < line("init_cuda_graphs")
    attention = ast.parse((NATIVE / "layers/attention/flashattention_backend.py").read_text())
    native_calls = [node.func for node in ast.walk(attention) if isinstance(node, ast.Call)
                    and ((isinstance(node.func, ast.Name) and node.func.id.startswith("flash_attn_"))
                         or (isinstance(node.func, ast.Attribute) and node.func.attr.startswith("_flash_attn_")))]
    assert len(native_calls) == 12
    assert all(isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self" for node in native_calls)
