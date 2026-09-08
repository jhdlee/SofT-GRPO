"""Pinned FSDP original ownership, without CUDA or a process group."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.fsdp import FlatParameter
from torch.distributed.fsdp._flat_param import ParamInfo, SharedParamInfo

from verl.opd.qwen_lora import (
    disable_qwen_lora, effective_weight_fp32, install_qwen_lora,
    merged_weight_fp32, native_base_parameter, validate_qwen_lora_frozen,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3")
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4)
        layer = nn.Module()
        layer.q_proj = nn.Linear(4, 4)
        layer.norm = nn.LayerNorm(4)
        self.model.layers = nn.ModuleList([layer])
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight


def installed():
    return install_qwen_lora(TinyModel(), rank=2, alpha=4, target_modules=("q_proj",))


def temporary_views(model):
    """Use real FlatParameter/split views and Torch 2.6's ownership records."""
    infos, originals, aliases, shared, primary = [], [], [], [], {}
    for path, module in model.named_modules():
        for name, parameter in module._parameters.items():
            if parameter is None:
                continue
            if id(parameter) in primary:
                owner, owner_name, owner_path = primary[id(parameter)]
                aliases.append(SharedParamInfo(name, module, path, owner_name, owner, owner_path))
                shared.append(parameter)
            else:
                primary[id(parameter)] = (module, name, path)
                infos.append(ParamInfo(name, module, path))
                originals.append(parameter)
    flat = FlatParameter(torch.cat([p.detach().flatten() for p in originals]), requires_grad=True)
    flat._params, flat._param_infos = originals, infos
    flat._shared_params, flat._shared_param_infos = shared, aliases
    flat._tensors = [part.reshape(original.shape) for part, original in
                     zip(flat.split([p.numel() for p in originals]), originals)]
    for info, view in zip(infos, flat._tensors):
        info.module._parameters[info.param_name] = view
    for info in aliases:
        info.module._parameters[info.param_name] = getattr(info.prim_module, info.prim_param_name)
    return flat


def test_owned_mixed_flat_view_detaches_only_base_and_retains_adapter_gradients():
    model = installed(); module = model.model.layers[0].q_proj
    flat = temporary_views(model)
    assert module.weight.requires_grad and module.weight._base is flat
    base = native_base_parameter(module)
    assert not base.requires_grad and base.data_ptr() == module.weight.data_ptr()
    aa = module.qwen_lora_A.detach().clone().requires_grad_()
    bb = module.qwen_lora_B.detach().clone().requires_grad_()
    expected = base + (bb @ aa) * module._opd_lora_scale
    actual = effective_weight_fp32(module)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward(); expected.square().sum().backward()
    chunks = flat.grad.split([p.numel() for p in flat._params])
    for info, gradient in zip(flat._param_infos, chunks):
        if info.module is module and info.param_name in ("qwen_lora_A", "qwen_lora_B"):
            oracle = aa.grad if info.param_name.endswith("A") else bb.grad
            torch.testing.assert_close(gradient.reshape_as(oracle), oracle, rtol=2e-6, atol=2e-6)
        else:
            assert not gradient.count_nonzero()
    assert bb.grad.abs().sum() > 0


@pytest.mark.parametrize("name", ["weight", "bias"])
def test_hidden_original_unfreeze_is_rejected_including_disabled_reference(name):
    model = installed(); module = model.model.layers[0].q_proj
    original = getattr(module, name)
    temporary_views(model)
    original.requires_grad_(True)
    with pytest.raises(ValueError, match="original base must remain frozen"):
        native_base_parameter(module, name)
    with pytest.raises(ValueError, match="original base must remain frozen"):
        validate_qwen_lora_frozen(model)
    if name == "weight":
        with disable_qwen_lora(model), pytest.raises(ValueError, match="original base"):
            effective_weight_fp32(module)


@pytest.mark.parametrize("corruption", ["unowned", "stale", "cross_owner", "missing_original"])
def test_arbitrary_or_stale_tensor_views_do_not_establish_frozen_ownership(corruption):
    model = installed(); module = model.model.layers[0].q_proj
    flat = temporary_views(model)
    if corruption == "unowned":
        module._parameters["weight"] = torch.ones(16, requires_grad=True).view(4, 4)
    elif corruption == "stale":
        module._parameters["weight"] = module.weight.view_as(module.weight)
    elif corruption == "cross_owner":
        module._parameters["weight"] = model.model.embed_tokens.weight[:4]
    else:
        flat._params = None
    with pytest.raises(ValueError, match="native LoRA base view"):
        native_base_parameter(module)


def test_tied_root_alias_checks_original_and_detaches_shared_current_view():
    model = installed(); original = model.model.embed_tokens.weight
    temporary_views(model)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert not native_base_parameter(model.lm_head).requires_grad
    assert not native_base_parameter(model.model.embed_tokens).requires_grad
    assert validate_qwen_lora_frozen(model) > 0
    original.requires_grad_(True)
    with pytest.raises(ValueError, match="original base"):
        native_base_parameter(model.lm_head)


def test_disabled_reference_preserves_input_gradient_without_base_or_adapter_graph():
    model = installed(); module = model.model.layers[0].q_proj
    flat = temporary_views(model)
    value = torch.randn(3, 4, requires_grad=True)
    with disable_qwen_lora(model):
        weight = effective_weight_fp32(module)
        assert not weight.requires_grad
        torch.nn.functional.linear(value, weight).square().sum().backward()
    assert value.grad.abs().sum() > 0 and flat.grad is None


def test_original_parameter_and_standalone_trainability_checks_remain_strict():
    model = installed(); module = model.model.layers[0].q_proj
    assert native_base_parameter(module) is module.weight
    module.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="original base"):
        effective_weight_fp32(module)
    with pytest.raises(ValueError, match="base must remain frozen"):
        merged_weight_fp32(module.weight, module.qwen_lora_A, module.qwen_lora_B, 2.)
    ordinary = nn.Linear(4, 4)
    assert native_base_parameter(ordinary) is ordinary.weight and ordinary.weight.requires_grad


def test_inventory_contains_only_names_and_survives_copy_and_state_reconstruction():
    model = installed()
    for module in model.modules():
        names = getattr(module, "_opd_qwen_lora_frozen_parameter_names", ())
        assert isinstance(names, tuple) and all(isinstance(name, str) for name in names)
    copied = copy.deepcopy(model)
    assert validate_qwen_lora_frozen(copied) == validate_qwen_lora_frozen(model)
    restored = installed()
    restored.load_state_dict(model.state_dict())
    assert validate_qwen_lora_frozen(restored) == validate_qwen_lora_frozen(model)
    assert not any("frozen_parameter" in name for name in model.state_dict())


@pytest.mark.parametrize("corruption", ["missing", "partial", "new_base"])
def test_global_freeze_validator_rejects_incomplete_inventory(corruption):
    model = installed(); module = model.model.layers[0].q_proj
    if corruption == "missing":
        del module._opd_qwen_lora_frozen_parameter_names
    elif corruption == "partial":
        module._opd_qwen_lora_frozen_parameter_names = ("weight",)
    else:
        module.register_parameter("new_base", nn.Parameter(torch.zeros(4)))
    with pytest.raises(ValueError, match="inventory differs"):
        validate_qwen_lora_frozen(model)
