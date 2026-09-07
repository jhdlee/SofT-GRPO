"""Collective native dense exports with the same LoRA merge as actor replay."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.distributed as dist


def collective_stage(label, operation):
    """Finish recoverable local work on every rank before advancing a phase."""
    result, error = None, None
    try:
        result = operation()
    except BaseException as failure:
        error = f"{type(failure).__name__}: {failure}"
    errors = [error]
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    if any(value is not None for value in errors):
        raise RuntimeError(f"collective native export failed at {label}: {errors}")
    return result


def _materialize(tensor, *, stage=collective_stage, label="tensor"):
    def transfer():
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else tensor.device
        return tensor.detach().to(device)
    # A local allocation failure must reach all peers before any peer enters
    # DTensor.full_tensor's process collective. Do not wrap these two checked
    # stages in an outer error collective with a competing phase number.
    local = stage(label + " local transfer", transfer)
    return stage(label + " gather", lambda: local.full_tensor() if hasattr(local, "full_tensor") else local)


def _parameter_inventory(state_dict, *, stage=collective_stage):
    names = stage("parameter names", lambda: sorted(state_dict))
    def check_inventory():
        if dist.is_initialized():
            inventories = [None] * dist.get_world_size()
            dist.all_gather_object(inventories, names)
            if any(inventory != names for inventory in inventories):
                raise RuntimeError("native export rank parameter inventories differ")
    stage("native parameter inventory", check_inventory)
    return names


def dense_rollout_weights(state_dict, config=None, *, stage=collective_stage):
    from verl.opd.qwen_lora import merge_qwen_lora_state_dict

    names = _parameter_inventory(state_dict, stage=stage)
    dense = {}
    for name in names:
        dense[name] = _materialize(state_dict[name], stage=stage, label=name)
    def effective_weights():
        if any(hasattr(value, "full_tensor") or hasattr(value, "local_shards") for value in dense.values()):
            raise ValueError("native rollout export requires fully materialized tensors")
        if config is not None:
            # Merge on the actor CUDA device, never through a CPU BLAS variant.
            return merge_qwen_lora_state_dict(dense, config, dtype=torch.bfloat16)
        if any(name.endswith((".qwen_lora_A", ".qwen_lora_B")) for name in dense):
            raise ValueError("native adapter weights require an explicit merge configuration")
        return {name: tensor.to(dtype=torch.bfloat16) if tensor.is_floating_point() else tensor
                for name, tensor in dense.items()}
    return stage("effective BF16 weights", effective_weights)


def save_native_adapter(module, local_path, *, base_model_path):
    from safetensors.torch import save_file
    from verl.opd.qwen_lora import peft_adapter_config, peft_adapter_state_dict, qwen_lora_config
    from verl.opd.provenance import _model_identity

    config = collective_stage("adapter configuration", lambda: qwen_lora_config(module))
    state = collective_stage("adapter state dict", module.state_dict)
    adapters = {}
    for name in _parameter_inventory(state):
        if name.endswith((".qwen_lora_A", ".qwen_lora_B")):
            tensor = _materialize(state[name], label="adapter " + name)
            adapters[name] = collective_stage("adapter CPU " + name, lambda: tensor.cpu().contiguous())
    converted = collective_stage("adapter export mapping", lambda: peft_adapter_state_dict(adapters, config))
    base = collective_stage("frozen base identity", lambda: _model_identity(base_model_path))
    def publish():
        if not dist.is_initialized() or dist.get_rank() == 0:
            directory = Path(local_path) / "lora_adapter"
            directory.mkdir(parents=True, exist_ok=False)
            save_file(converted, str(directory / "adapter_model.safetensors"))
            adapter_config = peft_adapter_config(config, base_model_name_or_path=base["id"])
            adapter_config["revision"] = base["resolved_revision"]
            (directory / "adapter_config.json").write_text(json.dumps(
                adapter_config,
                sort_keys=True, indent=2, allow_nan=False,
            ) + "\n")
            (directory / "frozen_base_identity.json").write_text(json.dumps(
                base, sort_keys=True, indent=2, allow_nan=False,
            ) + "\n")
    collective_stage("adapter publication", publish)
