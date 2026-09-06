"""Fail-closed contracts for installing the same arithmetic on both policies."""

import pytest
from omegaconf import OmegaConf

from verl.opd.qwen_replay_backend import validate_qwen_replay_worker, qwen_replay_arithmetic_identity, validate_qwen_replay_runtime, QWEN_REPLAY_RUNTIME


def _settings():
    config = OmegaConf.create({"model": {"qwen_replay_backend": "native_fa3_v1", "lora_rank": 0},
                              "actor": {"strategy": "fsdp"},
                              "rollout": {"name": "sglang", "tensor_model_parallel_size": 1, "qwen_replay_backend": "native_fa3_v1"}})
    options = dict(use_remove_padding=True, use_fused_kernels=False, enable_gradient_checkpointing=False,
                   enable_activation_offload=False, use_liger=False, sequence_parallel_size=1)
    return config, options


def test_native_policy_pair_accepts_supported_fsdp_recipe():
    config, options = _settings()
    assert validate_qwen_replay_worker(config, **options) == "native_fa3_v1"


@pytest.mark.parametrize("path,value", [("model.qwen_replay_backend", "unknown"), ("rollout.qwen_replay_backend", "disabled"),
                                       ("actor.strategy", "fsdp2"), ("model.lora_rank", 4),
                                       ("rollout.tensor_model_parallel_size", 2), ("rollout.name", "vllm")])
def test_mismatched_or_unsupported_policy_pair_fails(path, value):
    config, options = _settings()
    OmegaConf.update(config, path, value)
    with pytest.raises(ValueError):
        validate_qwen_replay_worker(config, **options)


@pytest.mark.parametrize("option,value", [("use_remove_padding", False), ("use_fused_kernels", True),
                                         ("enable_gradient_checkpointing", True), ("enable_activation_offload", True),
                                         ("use_liger", True), ("sequence_parallel_size", 2)])
def test_unsupported_forward_paths_rejected_before_installation(option, value):
    config, options = _settings()
    options[option] = value
    with pytest.raises(ValueError):
        validate_qwen_replay_worker(config, **options)


def test_disabled_backend_preserves_existing_launch_options():
    config, options = _settings()
    config.model.qwen_replay_backend = "disabled"
    config.rollout.qwen_replay_backend = "disabled"
    config.actor.strategy = "fsdp2"
    options.update(use_remove_padding=False, use_fused_kernels=True, enable_gradient_checkpointing=True)
    assert validate_qwen_replay_worker(config, **options) == "disabled"


def test_native_rollout_cannot_be_paired_with_ordinary_actor_arithmetic():
    config, options = _settings()
    config.model.qwen_replay_backend = "disabled"
    with pytest.raises(ValueError, match="must match"):
        validate_qwen_replay_worker(config, **options)


def test_arithmetic_identity_binds_all_shared_kernel_sources():
    identity = qwen_replay_arithmetic_identity()
    assert identity["recipe"] == "native_fa3_v1" and identity["attention_num_splits"] == 1
    assert set(identity["implementation_sha256"]) == {"qwen_replay_backend.py", "qwen_native_arithmetic.py", "batch_invariant_linear.py", "native_fa3_attention.py"}
    assert all(len(value) == 64 for value in identity["implementation_sha256"].values())


def test_versioned_recipe_cannot_silently_use_different_kernel_dependencies():
    versions = {**QWEN_REPLAY_RUNTIME, "torch": "2.6.0+cu124"}
    assert validate_qwen_replay_runtime(versions.__getitem__) == versions
    versions["sgl-kernel"] = "0.1.2"
    with pytest.raises(RuntimeError, match="calibrated kernel runtime"):
        validate_qwen_replay_runtime(versions.__getitem__)
