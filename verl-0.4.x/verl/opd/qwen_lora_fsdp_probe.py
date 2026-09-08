"""Disposable two- or four-rank native-LoRA/FSDP admission; no training assets or Ray."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time


def _rank_probe(rank, directory, expected_module_path, world_size=2):
    if str(Path(__file__).resolve()) != expected_module_path:
        raise RuntimeError("native LoRA FSDP child imported a different source module")
    if not __debug__:
        raise RuntimeError("native LoRA FSDP admission requires its assertions enabled")
    from datetime import timedelta
    from functools import partial

    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, MixedPrecision, StateDictType
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    from .ema import EMAUpdateState, freeze_teacher_
    from .qwen_lora import (adapter_gradient_statistics, disable_qwen_lora, install_qwen_lora,
                            merge_qwen_lora_state_dict, qwen_lora_config, validate_qwen_lora_frozen)
    from .qwen_lora_ema import initialize_dense_teacher_, update_dense_ema_once_
    from .qwen_native_arithmetic import install_qwen_replay_arithmetic

    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=(Path(directory) / "rendezvous").as_uri(),
                            rank=rank, world_size=world_size, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(3187)
        config = Qwen3Config(vocab_size=64, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=128,
                            max_position_embeddings=32, attention_dropout=0., use_cache=False,
                            tie_word_embeddings=True, bos_token_id=1, eos_token_id=2, pad_token_id=0)
        config._attn_implementation = "eager"
        student_model, teacher_model = Qwen3ForCausalLM(config).float(), Qwen3ForCausalLM(config).float()
        teacher_model.load_state_dict(student_model.state_dict())
        frozen_before = {name: value.detach().clone() for name, value in student_model.state_dict().items()}
        install_qwen_lora(student_model, rank=4, alpha=8, seed=11)
        adapter_before = {name: value.detach().clone() for name, value in student_model.state_dict().items()
                          if "qwen_lora" in name}
        for model in (student_model, teacher_model):
            install_qwen_replay_arithmetic(model, cache_device=device, backend="native_fa3_v2")
        precision = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32,
                                   buffer_dtype=torch.float32, cast_forward_inputs=False, cast_root_forward_inputs=False)
        policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen3DecoderLayer})

        def wrap(model):
            return FSDP(model, device_id=device, use_orig_params=True, auto_wrap_policy=policy,
                        mixed_precision=precision, sync_module_states=True)

        student, teacher = wrap(student_model), freeze_teacher_(wrap(teacher_model))
        assert len(FSDP.fsdp_modules(student)) == len(FSDP.fsdp_modules(teacher)) == 3
        validate_qwen_lora_frozen(student)
        initialize_dense_teacher_(teacher, student)
        ids = torch.tensor([[1, 3 + rank, 5, 7, 9, 11]], device=device)
        kwargs = dict(input_ids=ids, position_ids=torch.tensor([[0, 1, 2, 0, 1, 2]], device=device),
                      opd_cu_seqlens=torch.tensor([0, 3, 6], dtype=torch.int32, device=device),
                      opd_max_seqlen=3, use_cache=False)
        with torch.no_grad(), disable_qwen_lora(student):
            disabled_before = student(**kwargs).logits.detach().clone()
        optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                                      lr=1e-2, weight_decay=0.)
        target = torch.arange(64, device=device).remainder(7).float().div(8)
        gradient_norms = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                current_teacher = student(**kwargs).logits
                assert not current_teacher.requires_grad
            logits = student(**kwargs).logits
            loss = (logits.float() - target).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            stats = adapter_gradient_statistics(student)
            assert stats["lora/grad_finite"] and stats["lora/b_grad_norm"] > 0
            gradient_norms.append(stats["lora/grad_norm"])
            assert all(p.grad is None for name, p in student.named_parameters() if "qwen_lora" not in name)
            assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
            assert torch.isfinite(student.clip_grad_norm_(1.0))
            optimizer.step()
            validate_qwen_lora_frozen(student)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), disable_qwen_lora(student):
            assert torch.equal(disabled_before, student(**kwargs).logits)
        with torch.no_grad():
            current_teacher = student(**kwargs).logits
            assert not current_teacher.requires_grad
        state = EMAUpdateState()
        update_dense_ema_once_(teacher, student, .5, 0, state)
        assert state.update_count == 1
        assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
        with FSDP.state_dict_type(student, StateDictType.FULL_STATE_DICT,
                                  FullStateDictConfig(offload_to_cpu=False, rank0_only=False)):
            full = student.state_dict()
        assert all(torch.equal(full[name].cpu(), value) for name, value in frozen_before.items())
        assert any(not torch.equal(full[name].cpu(), value) for name, value in adapter_before.items())
        dense = merge_qwen_lora_state_dict(full, qwen_lora_config(student), dtype=torch.float32)
        # Detached dense export must execute exactly the same native arithmetic.
        exported = Qwen3ForCausalLM(config).float().to(device)
        exported.load_state_dict(dense)
        freeze_teacher_(exported)
        install_qwen_replay_arithmetic(exported, cache_device=device, backend="native_fa3_v2")
        with torch.no_grad():
            assert torch.equal(current_teacher, exported(**kwargs).logits)
        torch.cuda.synchronize(device)
        result = dict(rank=rank, optimizer_steps=2, frozen_base_unchanged=True,
                      base_gradients_absent=True, adapter_gradients_finite=True, adapter_update_nonzero=True,
                      disabled_reference_exact=True, dense_export_exact=True, current_actor_detached=True,
                      dense_ema_updates=1, wrapper_count=3, adapter_gradient_norms=gradient_norms,
                      installed_module_path=str(Path(__file__).resolve()))
        (Path(directory) / f"rank-{rank}.json").write_text(json.dumps(result, allow_nan=False))
    finally:
        dist.destroy_process_group()


def validate_native_lora_fsdp_cuda(*, world_size=2):
    """Run a fresh bounded distributed probe before any rollout engines start."""
    if type(world_size) is not int or world_size not in (2, 4):
        raise ValueError("native LoRA FSDP admission supports world size two or four")
    import torch
    import torch.distributed as dist

    if dist.is_initialized() or torch.cuda.device_count() != world_size or torch.version.hip is not None:
        raise RuntimeError(f"native LoRA FSDP admission requires exactly {world_size} isolated NVIDIA GPUs")
    if any(torch.cuda.get_device_capability(rank) != (9, 0) for rank in range(world_size)):
        raise RuntimeError("native LoRA FSDP admission requires Hopper SM90 GPUs")
    with tempfile.TemporaryDirectory(prefix="opd-lora-fsdp-", dir=os.environ.get("TMPDIR")) as directory:
        context = torch.multiprocessing.spawn(_rank_probe, args=(directory, str(Path(__file__).resolve()), world_size),
                                               nprocs=world_size, join=False)
        try:
            deadline = time.monotonic() + 120
            while not context.join(timeout=1):
                if time.monotonic() >= deadline:
                    raise TimeoutError("native LoRA distributed FSDP admission exceeded 120 seconds")
            ranks = [json.loads((Path(directory) / f"rank-{rank}.json").read_text()) for rank in range(world_size)]
            return {"schema_version": 1, "status": "passed", "world_size": world_size, "ranks": ranks}
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            for process in context.processes:
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
