"""Native arithmetic isolates frozen FSDP views while preserving adapter graphs."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch.distributed.fsdp import FlatParameter
from torch.distributed.fsdp._flat_param import ParamInfo, SharedParamInfo

from verl.opd import qwen_native_arithmetic as arithmetic
from verl.opd.batch_invariant_linear import batch_invariant_linear
from verl.opd.native_fa3_attention import native_fa3_attention
from verl.opd.qwen_lora import install_qwen_lora, disable_qwen_lora
from verl.opd.qwen_lora_ema import _dense_pairs
from test_qwen_native_arithmetic import _small_replay_model


def model_with_lora(monkeypatch, *, lora=True):
    torch.manual_seed(81)
    model = _small_replay_model().float()
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.bias = torch.nn.Parameter(torch.linspace(-.03, .03, module.out_features))
    if lora:
        # The unadapted projections must remain protected too.
        install_qwen_lora(model, rank=4, alpha=8, target_modules=('q_proj', 'v_proj'))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith('qwen_lora_B'):
                    parameter.normal_(std=.02)
    monkeypatch.setattr(arithmetic, '_replay_cache_device', lambda device: torch.device('cpu'))
    arithmetic._install_native_arithmetic(model, linear=batch_invariant_linear, attention=native_fa3_attention,
                                          cache_device='cuda:0', production=True, fp32_masters=True)
    return model


def temporary_root_decoder_views(model):
    """Use real FlatParameter storage and Torch2.6 ownership/view metadata."""
    groups = {'': []}
    for path, module in model.named_modules():
        owner = next((f'model.layers.{i}' for i in range(len(model.model.layers))
                      if path == f'model.layers.{i}' or path.startswith(f'model.layers.{i}.')), '')
        for name, parameter in module._parameters.items():
            if parameter is not None:
                groups.setdefault(owner, []).append((path, module, name, parameter))
    flats = []
    for rows in groups.values():
        primary, aliases, seen = [], [], {}
        for path, module, name, parameter in rows:
            if id(parameter) in seen:
                aliases.append((path, module, name, parameter, seen[id(parameter)]))
            else:
                seen[id(parameter)] = len(primary)
                primary.append((path, module, name, parameter))
        originals = [row[3] for row in primary]
        flat = FlatParameter(torch.cat([p.detach().flatten() for p in originals]),
                             requires_grad=any(p.requires_grad for p in originals))
        views = [piece.view_as(parameter) for piece, parameter in zip(flat.split([p.numel() for p in originals]), originals)]
        flat._params = originals
        flat._param_infos = [ParamInfo(name, module, path) for path, module, name, _ in primary]
        flat._tensors = views
        flat._shared_param_infos, flat._shared_params = [], []
        for (_, module, name, _), view in zip(primary, views):
            module._parameters[name] = view
        for path, module, name, parameter, index in aliases:
            prim_path, prim_module, prim_name, _ = primary[index]
            flat._shared_param_infos.append(SharedParamInfo(name, module, path, prim_name, prim_module, prim_path))
            flat._shared_params.append(parameter)
            module._parameters[name] = views[index]
        flats.append((flat, primary))
    return flats


def logits(model, value):
    hidden = model.model(inputs_embeds=value, position_ids=torch.tensor([[0, 1, 2, 0, 1, 2]]),
                         opd_cu_seqlens=torch.tensor([0, 3, 6], dtype=torch.int32), opd_max_seqlen=3,
                         use_cache=False).last_hidden_state
    return model.lm_head(hidden)


def test_mixed_flat_views_preserve_exact_output_input_and_adapter_gradients(monkeypatch):
    baseline = model_with_lora(monkeypatch)
    viewed = model_with_lora(monkeypatch)
    flats = temporary_root_decoder_views(viewed)
    assert len(flats) == 3
    assert viewed.model.layers[0].self_attn.q_proj.weight.requires_grad
    assert viewed.model.layers[0].self_attn.k_proj.weight.requires_grad
    assert viewed.model.layers[0].input_layernorm.weight.requires_grad
    assert viewed.model.layers[0].self_attn.q_proj.bias.requires_grad
    value = torch.randn(1, 6, 64, dtype=torch.bfloat16).requires_grad_()
    other = value.detach().clone().requires_grad_()
    expected, actual = logits(baseline, value), logits(viewed, other)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected[:, 3:5].float().square().sum().backward()
    actual[:, 3:5].float().square().sum().backward()
    torch.testing.assert_close(other.grad, value.grad, rtol=0, atol=0)
    assert torch.count_nonzero(other.grad[:, :3]) == 0
    baseline_parameters = dict(baseline.named_parameters())
    observed_nonzero = 0
    for flat, rows in flats:
        if not flat.requires_grad:
            assert flat.grad is None
            continue
        for piece, (path, _, name, original) in zip(flat.grad.split([row[3].numel() for row in rows]), rows):
            if name.startswith('qwen_lora_'):
                wanted = baseline_parameters[path + '.' + name].grad
                torch.testing.assert_close(piece.view_as(original), wanted, rtol=0, atol=0)
                observed_nonzero += int(torch.count_nonzero(piece))
            else:
                assert torch.count_nonzero(piece) == 0
            assert original.grad is None
    assert observed_nonzero > 0
    assert viewed.lm_head.weight is viewed.model.embed_tokens.weight


def test_external_embedding_and_disabled_current_actor_paths_keep_frozen_base(monkeypatch):
    baseline = model_with_lora(monkeypatch)
    viewed = model_with_lora(monkeypatch)
    temporary_root_decoder_views(viewed)
    ids = torch.tensor([[1, 2, 3]])
    expected, actual = baseline.model.embed_tokens(ids), viewed.model.embed_tokens(ids)
    assert expected.dtype == actual.dtype == torch.float32
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    inputs = torch.randn(1, 6, 64, dtype=torch.bfloat16).requires_grad_()
    with disable_qwen_lora(baseline), disable_qwen_lora(viewed):
        torch.testing.assert_close(logits(viewed, inputs), logits(baseline, inputs), rtol=0, atol=0)
        logits(viewed, inputs).float().square().mean().backward()
    assert inputs.grad is not None and inputs.grad.abs().sum() > 0
    assert not viewed.model.layers[0].self_attn.q_proj._opd_lora_disabled


@pytest.mark.parametrize('leaf', ['norm', 'bias', 'embedding', 'untargeted_projection'])
def test_original_base_unfreezing_is_rejected_by_native_paths(monkeypatch, leaf):
    model = model_with_lora(monkeypatch)
    if leaf == 'embedding':
        model.model.embed_tokens.weight.requires_grad_(True)
        with pytest.raises(ValueError, match='original base must remain frozen'):
            model.model.embed_tokens(torch.tensor([1, 2]))
        return
    module = model.model.layers[0]
    parameter = {'norm': module.input_layernorm.weight, 'bias': module.self_attn.q_proj.bias,
                 'untargeted_projection': module.self_attn.k_proj.weight}[leaf]
    parameter.requires_grad_(True)
    with pytest.raises(ValueError, match='original base must remain frozen'):
        logits(model, torch.randn(1, 6, 64, dtype=torch.bfloat16))


def test_full_finetuning_norm_bias_embedding_and_projection_gradients_remain(monkeypatch):
    model = model_with_lora(monkeypatch, lora=False)
    value = model.model.embed_tokens(torch.tensor([[1, 2, 3, 4, 5, 6]])).bfloat16()
    logits(model, value).float().square().sum().backward()
    layer = model.model.layers[0]
    for parameter in (layer.input_layernorm.weight, layer.self_attn.q_proj.bias,
                      layer.self_attn.k_proj.weight, model.model.embed_tokens.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


def test_dense_ema_rejects_unfrozen_nonadapter_source_before_detaching():
    from test_qwen_lora import TinyQwen
    student, teacher = TinyQwen(), TinyQwen()
    teacher.load_state_dict(student.state_dict())
    teacher.requires_grad_(False)
    install_qwen_lora(student, rank=4, alpha=8)
    student.model.layers[0].norm.weight.requires_grad_(True)
    with pytest.raises(ValueError, match='original base must remain frozen'):
        list(_dense_pairs(teacher, student))


def test_vllm_unfrozen_base_fails_both_ranks_before_state_capture(monkeypatch):
    from test_qwen_weight_export import ThreadRanks, native_entry_managers, run_ranks, export
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    dist = ThreadRanks()
    monkeypatch.setattr(export, 'dist', dist)
    events = []
    managers = native_entry_managers(dist, events, lora=True)
    managers[1].module.model.embed_tokens.weight.requires_grad_(True)
    errors = run_ranks(dist.operations(lambda rank: managers[rank].__enter__()))
    assert all(isinstance(error, RuntimeError) and 'original base must remain frozen' in str(error) for error in errors)
    assert not any(name in ('state dict', 'gather entered', 'wake weights', 'load weights') for _, name in events)
    assert sum(name == 'shutdown' for _, name in events) == 2


def test_sglang_unfrozen_base_fails_both_ranks_before_state_capture():
    from test_qwen_lora import actor
    from test_sharding_weight_failures import _run_pair
    def operation(manager, loop):
        model = actor()
        model.state_dict = manager.module.state_dict
        manager.module = model
        # Thread stand-ins expose their rank through this harmless test mesh.
        if manager.device_mesh['infer_tp'].mesh.tolist() == [1]:
            model.model.embed_tokens.weight.requires_grad_(True)
        manager._prepare_weights()
    managers, errors = _run_pair(operation)
    assert all(isinstance(error, RuntimeError) and 'original base must remain frozen' in str(error) for error in errors)
    assert not any('state dict' in manager.events for manager in managers)
    assert all(manager.inference_engine.shutdown_calls == 1 for manager in managers)
