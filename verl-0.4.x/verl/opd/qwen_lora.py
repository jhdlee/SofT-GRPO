"""FP32 LoRA masters with one shared dense-weight merge for native Qwen.

These adapters deliberately do not install PEFT forward hooks: the native
decoder accesses projections inside its owning FSDP block. The same merge is
used for differentiable replay and detached dense inference export. Checkpoint
configuration is explicit; it is never inferred from an adapter's shape.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import math
from typing import Mapping

import torch
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None


QWEN_LORA_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)
QWEN_LORA_MERGE_RULE = "fp32_fixed_tile_ba_scale_add_then_bf16_v1"
_A = "qwen_lora_A"
_B = "qwen_lora_B"


def validate_qwen_lora_config(config):
    if not isinstance(config, Mapping):
        raise ValueError("native LoRA configuration must be a mapping")
    required = {"rank", "alpha", "target_modules", "dropout", "bias", "seed", "merge_rule"}
    if set(config) != required:
        raise ValueError("native LoRA configuration fields differ")
    rank, alpha, seed = config["rank"], config["alpha"], config["seed"]
    if type(rank) is not int or not 1 <= rank <= 256:
        raise ValueError("native LoRA rank must be an integer in [1, 256]")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("native LoRA alpha must be finite and positive")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("native LoRA seed must be a nonnegative int64")
    targets = config["target_modules"]
    if (not isinstance(targets, (list, tuple)) or not targets
            or any(not isinstance(name, str) or name not in QWEN_LORA_TARGET_MODULES for name in targets)
            or len(set(targets)) != len(targets)):
        raise ValueError("native LoRA targets must be a nonempty unique subset of the seven Qwen attention/MLP projections")
    if config["dropout"] != 0.0 or config["bias"] != "none" or config["merge_rule"] != QWEN_LORA_MERGE_RULE:
        raise ValueError("native LoRA requires dropout zero, no bias adaptation, and the sealed merge rule")
    return {**config, "alpha": float(alpha), "target_modules": [name for name in QWEN_LORA_TARGET_MODULES if name in targets]}


def _unwrap(model):
    # Avoid a dependency on worker or FSDP construction code.
    while hasattr(model, "_fsdp_wrapped_module"):
        model = model._fsdp_wrapped_module
    return model


def has_qwen_lora(model):
    return getattr(_unwrap(model), "_opd_qwen_lora_config", None) is not None


def qwen_lora_config(model):
    value = getattr(_unwrap(model), "_opd_qwen_lora_config", None)
    if value is None:
        raise ValueError("model has no native Qwen LoRA adapters")
    return validate_qwen_lora_config(value)


def install_qwen_lora(model, *, rank=32, alpha=64, target_modules=QWEN_LORA_TARGET_MODULES, seed=11):
    """Freeze the FP32 model and add deterministic FP32 A/B parameters in place.

    Run before the native arithmetic installer and before FSDP wrapping. The
    local CPU generator leaves global training/sampling RNG streams untouched.
    B starts at zero; A uses the ordinary Kaiming-uniform linear initialization.
    """
    config = validate_qwen_lora_config(dict(rank=rank, alpha=alpha, target_modules=list(target_modules),
                                          dropout=0.0, bias="none", seed=seed, merge_rule=QWEN_LORA_MERGE_RULE))
    if has_qwen_lora(model) or getattr(model, "_opd_native_arithmetic_candidate", False):
        raise ValueError("native LoRA must be installed once, before replay arithmetic")
    if getattr(getattr(model, "config", None), "model_type", None) != "qwen3":
        raise ValueError("native LoRA supports dense Qwen3 only")
    if hasattr(model, "peft_config"):
        raise ValueError("native LoRA cannot wrap a PEFT model")
    parameters = list(model.parameters())
    if not parameters or any(p.dtype != torch.float32 or p.is_meta for p in parameters):
        raise ValueError("native LoRA requires materialized FP32 base parameters")
    targets = [(name, module) for name, module in model.named_modules()
               if name.rsplit(".", 1)[-1] in config["target_modules"]]
    expected = len(model.model.layers) * len(config["target_modules"])
    if not expected or len(targets) != expected:
        raise ValueError("native LoRA projection inventory differs from the Qwen decoder")
    for name, module in targets:
        if type(module) is not nn.Linear or any(hasattr(module, key) for key in (_A, _B)):
            raise ValueError(f"native LoRA requires an unmodified Linear projection: {name}")
    # Allocate everything before mutating the model's trainable state.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    prepared = []
    for _, module in targets:
        a = torch.empty(rank, module.in_features, dtype=torch.float32)
        a.uniform_(-1 / math.sqrt(module.in_features), 1 / math.sqrt(module.in_features), generator=generator)
        prepared.append((module, a.to(module.weight.device),
                         torch.zeros(module.out_features, rank, dtype=torch.float32, device=module.weight.device)))
    for parameter in parameters:
        parameter.requires_grad_(False)
    for module, a, b in prepared:
        module.register_parameter(_A, nn.Parameter(a))
        module.register_parameter(_B, nn.Parameter(b))
        module._opd_lora_scale = float(alpha) / rank
        module._opd_lora_disabled = False
    model._opd_qwen_lora_config = config
    return model


@contextmanager
def disable_qwen_lora(model):
    """Temporarily score the immutable base; restore nested contexts on error."""
    qwen_lora_config(model)
    modules = [module for module in model.modules() if hasattr(module, _A)]
    previous = [module._opd_lora_disabled for module in modules]
    try:
        for module in modules:
            module._opd_lora_disabled = True
        yield model
    finally:
        for module, disabled in zip(modules, previous):
            module._opd_lora_disabled = disabled


if triton is not None:
    @triton.jit
    def _merge_kernel(W, A, B, OUT, N: tl.constexpr, K: tl.constexpr, R: tl.constexpr, SCALE: tl.constexpr):
        rows = tl.program_id(0) * 32 + tl.arange(0, 32)
        columns = tl.program_id(1) * 64 + tl.arange(0, 64)
        reduction = tl.arange(0, 32)
        accumulator = tl.zeros((32, 64), tl.float32)
        for block in range(tl.cdiv(R, 32)):
            indices = block * 32 + reduction
            b = tl.load(B + rows[:, None] * R + indices[None, :],
                        (rows[:, None] < N) & (indices[None, :] < R), 0.0)
            a = tl.load(A + indices[:, None] * K + columns[None, :],
                        (indices[:, None] < R) & (columns[None, :] < K), 0.0)
            accumulator = tl.dot(b, a, accumulator, input_precision="ieee")
        base = tl.load(W + rows[:, None] * K + columns[None, :],
                       (rows[:, None] < N) & (columns[None, :] < K), 0.0)
        output = base + accumulator * SCALE
        tl.store(OUT + rows[:, None] * K + columns[None, :], output,
                 (rows[:, None] < N) & (columns[None, :] < K))


class _MergeLoRA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, a, b, scale):
        ctx.save_for_backward(a, b)
        ctx.scale = scale
        with torch.autocast(device_type=weight.device.type, enabled=False):
            if not weight.is_cuda:
                return weight + (b @ a) * scale
            output = torch.empty_like(weight)
            with torch.cuda.device(weight.device):
                _merge_kernel[(triton.cdiv(weight.shape[0], 32), triton.cdiv(weight.shape[1], 64))](
                    weight, a, b, output, *weight.shape, a.shape[0], scale,
                    num_warps=4, num_stages=2, enable_fp_fusion=False,
                )
            return output

    @staticmethod
    def backward(ctx, gradient):
        a, b = ctx.saved_tensors
        if gradient.is_cuda and torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError("native LoRA backward requires matmul.allow_tf32=False")
        with torch.autocast(device_type=gradient.device.type, enabled=False):
            # Match the explicit graph's backward ordering: the scale belongs
            # to the product before its two matrix-product derivatives.
            gradient = gradient.float() * ctx.scale
            da = b.transpose(0, 1) @ gradient if ctx.needs_input_grad[1] else None
            db = gradient @ a.transpose(0, 1) if ctx.needs_input_grad[2] else None
        return None, da, db, None


def merged_weight_fp32(weight, a, b, scale):
    """Canonical merge, differentiable only with respect to FP32 adapters."""
    if any(hasattr(t, "full_tensor") or hasattr(t, "local_shards") for t in (weight, a, b)):
        raise ValueError("materialize FSDP/DTensor masters collectively before native LoRA merge")
    if (weight.ndim != 2 or a.ndim != 2 or b.ndim != 2
            or a.shape[1] != weight.shape[1] or b.shape != (weight.shape[0], a.shape[0])
            or not 1 <= a.shape[0] <= 256):
        raise ValueError("native LoRA base/adapter dimensions differ")
    if any(t.dtype != torch.float32 or t.device != weight.device or not t.is_contiguous() for t in (weight, a, b)):
        raise ValueError("native LoRA merge requires contiguous FP32 masters on one device")
    if weight.requires_grad:
        raise ValueError("native LoRA base must remain frozen")
    if isinstance(scale, bool) or not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("native LoRA scale must be finite and positive")
    if weight.device.type not in ("cpu", "cuda"):
        raise ValueError("native LoRA supports CPU references and NVIDIA CUDA only")
    if weight.is_cuda and (torch.version.hip is not None or triton is None):
        raise ValueError("native LoRA requires NVIDIA CUDA and Triton")
    return _MergeLoRA.apply(weight, a, b, float(scale))


def effective_weight_fp32(module):
    if hasattr(module, _A) and not getattr(module, "_opd_lora_disabled", False):
        return merged_weight_fp32(module.weight, getattr(module, _A), getattr(module, _B), module._opd_lora_scale)
    if module.weight.dtype != torch.float32:
        raise ValueError("v2 replay requires FP32 base/adapter masters")
    return module.weight


def effective_projection_weight(module, *, dtype=torch.bfloat16):
    return effective_weight_fp32(module).to(dtype=dtype)


def _adapter_inventory(state_dict, config):
    config = validate_qwen_lora_config(config)
    weights = [name for name in state_dict if name.endswith(".weight")
               and name.rsplit(".", 2)[-2] in config["target_modules"]]
    if not weights:
        raise ValueError("native LoRA checkpoint has no adapted projections")
    expected = {name[:-len("weight")] + suffix for name in weights for suffix in (_A, _B)}
    actual = {name for name in state_dict if name.rsplit(".", 1)[-1] in (_A, _B)}
    if expected != actual:
        raise ValueError("native LoRA checkpoint adapter inventory is incomplete or unexpected")
    for name in weights:
        prefix = name[:-len("weight")]
        w, a, b = state_dict[name], state_dict[prefix + _A], state_dict[prefix + _B]
        if w.ndim != 2 or a.shape != (config["rank"], w.shape[1]) or b.shape != (w.shape[0], config["rank"]):
            raise ValueError(f"native LoRA checkpoint adapter dimensions differ: {name}")
    return config, weights, expected


@torch.no_grad()
def merge_qwen_lora_state_dict(state_dict, config, *, dtype=torch.bfloat16):
    """Export already-materialized full FP32 tensors under canonical HF names.

    Call on the same NVIDIA arithmetic runtime used for actor replay. FSDP
    state-dict materialization and process collectives belong to the caller.
    """
    config, weights, adapter_names = _adapter_inventory(state_dict, config)
    weights = set(weights)
    result = OrderedDict()
    for name, tensor in state_dict.items():
        if name in adapter_names:
            continue
        if name in weights:
            prefix = name[:-len("weight")]
            tensor = merged_weight_fp32(tensor.detach(), state_dict[prefix + _A].detach(),
                                         state_dict[prefix + _B].detach(), config["alpha"] / config["rank"])
        result[name] = tensor.detach().to(dtype=dtype) if tensor.is_floating_point() else tensor.detach()
    return result


def peft_adapter_config(config, *, base_model_name_or_path="Qwen/Qwen3-0.6B"):
    config = validate_qwen_lora_config(config)
    return {"peft_type": "LORA", "task_type": "CAUSAL_LM", "inference_mode": True,
            "base_model_name_or_path": str(base_model_name_or_path), "r": config["rank"],
            "lora_alpha": config["alpha"], "lora_dropout": 0.0, "bias": "none",
            "target_modules": config["target_modules"], "fan_in_fan_out": False,
            "use_rslora": False, "use_dora": False}


def peft_adapter_state_dict(state_dict, config):
    """Return standard PEFT adapter keys from full or adapter-only state."""
    config = validate_qwen_lora_config(config)
    names = {name for name in state_dict if name.rsplit(".", 1)[-1] in (_A, _B)}
    prefixes = sorted({name.rsplit(".", 1)[0] + "." for name in names})
    if not prefixes or names != {prefix + suffix for prefix in prefixes for suffix in (_A, _B)}:
        raise ValueError("native LoRA adapter export requires complete A/B pairs")
    result = OrderedDict()
    for prefix in prefixes:
        if prefix.rstrip(".").rsplit(".", 1)[-1] not in config["target_modules"]:
            raise ValueError("native LoRA adapter export has an unexpected target")
        a, b = state_dict[prefix + _A], state_dict[prefix + _B]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != config["rank"] or b.shape[1] != config["rank"]:
            raise ValueError("native LoRA adapter export dimensions differ from its config")
        for suffix, peft_suffix in ((_A, "lora_A.weight"), (_B, "lora_B.weight")):
            tensor = state_dict[prefix + suffix]
            if tensor.dtype != torch.float32 or hasattr(tensor, "full_tensor") or hasattr(tensor, "local_shards"):
                raise ValueError("native LoRA adapter export requires materialized FP32 masters")
            result["base_model.model." + prefix + peft_suffix] = tensor.detach().clone()
    return result


def effective_update_statistics(before, after):
    """Measure actual BF16 inference changes, separately from adapter updates."""
    if not before or before.keys() != after.keys():
        raise ValueError("effective update inventories differ or are empty")
    changed = count = 0
    squared = maximum = 0.0
    for name, old in before.items():
        new = after[name]
        if old.dtype != torch.bfloat16 or new.dtype != torch.bfloat16 or old.shape != new.shape:
            raise ValueError("effective update diagnostics require matching BF16 weights")
        delta = new.detach().float().cpu() - old.detach().float().cpu()
        if not bool(torch.isfinite(delta).all()):
            raise ValueError("effective weights changed to nonfinite values")
        changed += int(delta.count_nonzero())
        count += delta.numel()
        squared += float(delta.double().square().sum())
        maximum = max(maximum, float(delta.abs().max()) if delta.numel() else 0.0)
    return {"changed_elements": changed, "total_elements": count,
            "changed_fraction": changed / count if count else 0.0,
            "max_abs_change": maximum, "rms_change": math.sqrt(squared / count) if count else 0.0}


@torch.no_grad()
def lora_diagnostics(model, previous=None):
    """Return (metrics, CPU-BF16 snapshot), gathering one FSDP unit at a time.

    Every rank participates. Neither the snapshots nor the returned metrics
    require a subsequent all-reduce: they describe the same full actor. The
    first call compares to frozen base weights; later calls use the explicitly
    supplied previous actor-version snapshot. These are diagnostics, not gates.
    """
    from .qwen_lora_ema import _fsdp_units, _owned_tensors, _summon

    qwen_lora_config(model)
    snapshots, baseline = OrderedDict(), OrderedDict()
    adapter_squared = 0.0
    for unit_name, unit in _fsdp_units(model).items():
        with _summon(unit):
            for name, (tensor, module, leaf) in _owned_tensors(unit).items():
                if leaf in (_A, _B):
                    if not bool(torch.isfinite(tensor).all()):
                        raise ValueError("native LoRA adapter parameter is nonfinite")
                    adapter_squared += float(tensor.detach().double().square().sum())
                if leaf == "weight" and hasattr(module, _A):
                    key = ".".join(part for part in (unit_name, name) if part)
                    snapshots[key] = effective_projection_weight(module).detach().cpu().clone()
                    if previous is None:
                        baseline[key] = tensor.detach().to(torch.bfloat16).cpu().clone()
    stats = effective_update_statistics(baseline if previous is None else previous, snapshots)
    return {"lora/adapter_parameter_l2": math.sqrt(adapter_squared),
            "lora/effective_change_baseline": "frozen_base" if previous is None else "previous_actor_version",
            **{"lora/effective_" + key: value for key, value in stats.items()}}, snapshots


@torch.no_grad()
def adapter_gradient_statistics(model):
    """Collect sharded adapter gradients before optimizer.step/zero_grad.

    Native FSDP use_orig_params=True exposes each original parameter's local
    shard. Global SUM reductions therefore count each element exactly once.
    This helper must be called by all ranks at the same update boundary.
    """
    qwen_lora_config(model)
    first = next(model.parameters())
    # A squared norm, B squared norm, nonfinite count, nonzero count, observed
    # gradient elements, and trainable parameter elements.
    statistics = torch.zeros(6, dtype=torch.float64, device=first.device)
    for name, parameter in model.named_parameters():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in (_A, _B):
            continue
        statistics[5] += parameter.numel()
        gradient = parameter.grad
        if gradient is not None:
            finite = torch.isfinite(gradient)
            statistics[0 if leaf == _A else 1] += gradient.detach().double().square().sum()
            statistics[2] += (~finite).sum()
            statistics[3] += gradient.count_nonzero()
            statistics[4] += gradient.numel()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(statistics)
    a_sq, b_sq, nonfinite, nonzero, observed, elements = statistics.cpu().tolist()
    return {"lora/grad_norm": math.sqrt(a_sq + b_sq), "lora/a_grad_norm": math.sqrt(a_sq),
            "lora/b_grad_norm": math.sqrt(b_sq), "lora/grad_finite": nonfinite == 0,
            "lora/grad_nonfinite_elements": int(nonfinite), "lora/grad_nonzero_elements": int(nonzero),
            "lora/grad_observed_elements": int(observed), "lora/trainable_parameter_elements": int(elements)}


def validate_native_lora_cuda(*, device=None):
    """Bounded controlled-update admission probe; never uses training examples.

    A deliberately large disposable learning rate proves that adapter updates
    can change the BF16 inference model. It is not a recommendation for the
    training learning rate, nor a threshold on real-batch update magnitudes.
    CPU RNG is restored; the probe does not consume any CUDA random numbers.
    """
    from types import SimpleNamespace
    from .batch_invariant_linear import batch_invariant_linear

    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("native LoRA admission probe requires NVIDIA CUDA")
    device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("native LoRA v2 admission probe requires Hopper SM90")
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.random.fork_rng(devices=[]), torch.cuda.device(device):
            model = nn.Module()
            model.config = SimpleNamespace(model_type="qwen3")
            model.model = nn.Module()
            layer = nn.Module(); layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Linear(32, 64, bias=False, dtype=torch.float32)
            with torch.no_grad():
                layer.self_attn.q_proj.weight.copy_(torch.arange(64 * 32).reshape(64, 32).remainder(17).sub(8).float() / 128)
            model.model.layers = nn.ModuleList([layer])
            model.to(device)
            install_qwen_lora(model, rank=4, alpha=8, target_modules=("q_proj",), seed=11)
            projection = layer.self_attn.q_proj
            base = projection.weight.detach().clone()
            initial = effective_projection_weight(projection).detach().cpu()
            inputs = torch.arange(5 * 32).reshape(5, 32).remainder(13).sub(6).float().div(16).to(device=device, dtype=torch.bfloat16)
            target = torch.arange(64).remainder(7).float().div(8).to(device).unsqueeze(0)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.0)
            gradient_norms = []
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                result = batch_invariant_linear(inputs, effective_projection_weight(projection))
                (result.float() - target).square().mean().backward()
                gradients = (projection.qwen_lora_A.grad, projection.qwen_lora_B.grad)
                if any(g is None or not bool(torch.isfinite(g).all()) for g in gradients):
                    raise RuntimeError("native LoRA controlled update produced missing/nonfinite adapter gradients")
                if not bool(gradients[1].count_nonzero()):
                    raise RuntimeError("native LoRA controlled update did not reach the B adapter")
                gradient_norms.append(float(torch.nn.utils.clip_grad_norm_([projection.qwen_lora_A, projection.qwen_lora_B], 1.0)))
                optimizer.step()
            effective = effective_projection_weight(projection)
            exported = merge_qwen_lora_state_dict(model.state_dict(), qwen_lora_config(model))
            exported_weight = exported["model.layers.0.self_attn.q_proj.weight"]
            if not torch.equal(effective, exported_weight):
                raise RuntimeError("native LoRA dense export differs from actor effective weights")
            if not torch.equal(batch_invariant_linear(inputs, effective), batch_invariant_linear(inputs, exported_weight)):
                raise RuntimeError("native LoRA dense export projection differs from replay")
            if projection.weight.grad is not None or not torch.equal(base, projection.weight):
                raise RuntimeError("native LoRA controlled update changed or differentiated frozen base weights")
            statistics = effective_update_statistics({"weight": initial}, {"weight": effective.detach().cpu()})
            if statistics["changed_elements"] <= 0:
                raise RuntimeError("native LoRA controlled update did not change BF16 inference weights")
            return {"schema_version": 1, "status": "passed", "rank": 4, "alpha": 8,
                    "disposable_learning_rate": 1e-2, "optimizer_steps": 2,
                    "adapter_gradient_norms": gradient_norms, "frozen_base_unchanged": True,
                    "dense_export_weight_exact": True, "dense_export_projection_exact": True,
                    "teacher_or_training_data_used": False, "effective_update": statistics}
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
