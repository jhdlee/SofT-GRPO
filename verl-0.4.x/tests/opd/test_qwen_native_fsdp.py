"""CUDA integration of replay arithmetic with real FSDP1 and an EMA teacher.

Random tiny weights and synthetic privileged prefixes exercise the integration;
this is neither a MATH rollout nor a replacement for the strict pilot gate.
No checkpoints or model/tokenizer downloads are used.
"""

import copy
from datetime import timedelta
from functools import partial

import pytest
import torch
import torch.distributed as dist


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native replay FSDP integration requires NVIDIA CUDA")
def test_qwen_replay_fsdp_step_teacher_ema_and_offload(tmp_path):
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.51.1":
        pytest.skip("native replay integration targets pinned transformers 4.51.1")
    if torch.version.hip is not None or not torch.cuda.is_bf16_supported():
        pytest.skip("native replay integration requires NVIDIA BF16 kernels")
    if dist.is_initialized():
        pytest.skip("this test owns an isolated NCCL process group")

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    from verl.opd.ema import EMAUpdateState, freeze_teacher_, teacher_gradient_isolation_violations, update_ema_once_
    from verl.opd.losses import full_vocab_kl
    from verl.opd.masks import latent_kl_sum_and_count
    from verl.opd.qwen_native_arithmetic import install_qwen_replay_arithmetic
    from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu

    device = torch.device("cuda", torch.cuda.current_device())
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    group_created = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        dist.init_process_group(
            "nccl", init_method=(tmp_path / "nccl-rendezvous").as_uri(), rank=0, world_size=1,
            timeout=timedelta(seconds=90),
        )
        group_created = True
        torch.manual_seed(9107)
        config = Qwen3Config(
            vocab_size=256, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=128,
            max_position_embeddings=64, attention_dropout=0.0, use_cache=False,
            tie_word_embeddings=True, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )
        # Construction uses no HF attention kernel. The production installer
        # replaces the attention path before either model executes a forward.
        config._attn_implementation = "eager"
        with torch.device("cpu"):
            actor_model = Qwen3ForCausalLM(copy.deepcopy(config)).float()
            teacher_model = Qwen3ForCausalLM(copy.deepcopy(config)).float()
        teacher_model.load_state_dict(actor_model.state_dict())
        for model in (actor_model, teacher_model):
            model.config._attn_implementation = "flash_attention_2"
            original_parameters = dict(model.named_parameters())
            original_keys = set(model.state_dict())
            install_qwen_replay_arithmetic(model, cache_device=device)
            assert set(model.state_dict()) == original_keys
            assert all(dict(model.named_parameters())[name] is value for name, value in original_parameters.items())
            assert all(value.device.type == "cpu" and value.dtype == torch.float32 for value in model.parameters())
        freeze_teacher_(teacher_model)

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
        wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen3DecoderLayer})

        def wrap(model):
            return FSDP(model, auto_wrap_policy=wrap_policy, mixed_precision=mixed_precision,
                        device_id=device, use_orig_params=False, sync_module_states=True)

        actor, teacher = wrap(actor_model), wrap(teacher_model)
        actor.train()
        teacher.eval()
        assert len(FSDP.fsdp_modules(actor)) == len(FSDP.fsdp_modules(teacher)) == 3
        cache = actor_model.model._opd_native_rope_cache
        teacher_cache = teacher_model.model._opd_native_rope_cache
        original_cache, cache_pointer = cache.clone(), cache.data_ptr()
        assert cache.dtype == teacher_cache.dtype == torch.float32
        assert cache.device == teacher_cache.device == device
        assert torch.equal(cache, teacher_cache)
        optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-3, weight_decay=0.01)

        # Separate gradient check: loss on the beginning of row two cannot
        # reach row one or future positions of row two through packed attention.
        cumulative = torch.tensor([0, 4, 10], dtype=torch.int32, device=device)
        positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 4, 5]], device=device)
        causal_input = torch.randn(1, 10, 256, dtype=torch.bfloat16, device=device).requires_grad_()
        causal_kwargs = dict(position_ids=positions, opd_cu_seqlens=cumulative, opd_max_seqlen=6, use_cache=False)
        with torch.no_grad():
            baseline = actor(inputs_embeds=causal_input.detach(), **causal_kwargs).logits
            changed_input = causal_input.detach().clone()
            changed_input[:, :4] += 2
            changed_input[:, 7:] -= 2
            changed = actor(inputs_embeds=changed_input, **causal_kwargs).logits
            assert torch.equal(baseline[:, 4:7], changed[:, 4:7])
        causal_logits = actor(inputs_embeds=causal_input, **causal_kwargs).logits
        causal_logits[:, 4:7].float().square().mean().backward()
        assert torch.count_nonzero(causal_input.grad[:, :4]) == 0
        assert torch.count_nonzero(causal_input.grad[:, 7:]) == 0
        assert causal_input.grad[:, 4:7].abs().sum() > 0
        optimizer.zero_grad(set_to_none=True)

        # Both student rows receive detached response embeddings. The teacher
        # receives exactly row two's response after two extra prefix embeddings.
        response = torch.randn(3, 256, dtype=torch.bfloat16, device=device)
        first_prefix = torch.randn(2, 256, dtype=torch.bfloat16, device=device)
        second_prefix = torch.randn(3, 256, dtype=torch.bfloat16, device=device)
        privilege = torch.randn(2, 256, dtype=torch.bfloat16, device=device)
        first_row, second_row = torch.cat((first_prefix, response)), torch.cat((second_prefix, response))
        student_input = torch.cat((first_row, second_row)).unsqueeze(0).detach()
        teacher_input = torch.cat((second_prefix, privilege, response)).unsqueeze(0).detach()
        assert not student_input.requires_grad and not teacher_input.requires_grad
        assert torch.equal(student_input[0, -3:], teacher_input[0, -3:])
        student_kwargs = dict(
            position_ids=torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5]], device=device),
            opd_cu_seqlens=torch.tensor([0, 5, 11], dtype=torch.int32, device=device), opd_max_seqlen=6, use_cache=False,
        )
        teacher_parameters_before = {name: value.detach().clone() for name, value in teacher.named_parameters()}
        with torch.no_grad():
            teacher_logits = teacher(
                inputs_embeds=teacher_input, attention_mask=torch.ones(1, 8, dtype=torch.long, device=device),
                position_ids=torch.arange(8, device=device).unsqueeze(0),
                logits_to_keep=torch.tensor([4, 5, 6], device=device), use_cache=False,
            ).logits
        assert teacher_logits.shape == (1, 3, 256) and not teacher_logits.requires_grad
        assert teacher_gradient_isolation_violations(teacher) == (0, 0)
        student_logits = actor(inputs_embeds=student_input, **student_kwargs).logits[:, [7, 8, 9]]
        assert student_logits.shape == teacher_logits.shape
        token_kl = full_vocab_kl(student_logits, teacher_logits, direction="teacher_to_student", temperature=1.0)
        numerator, denominator, active = latent_kl_sum_and_count(
            token_kl, torch.tensor([[True, False, True]], device=device),
        )
        assert int(denominator) == int(active) == 2
        loss = numerator / denominator
        assert torch.isfinite(loss) and float(loss.detach()) > 0
        loss.backward()
        assert all(value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
                   for value in actor.parameters())
        assert all(value.dtype == torch.float32 and value.grad.dtype == torch.float32 for value in actor.parameters())
        assert teacher_gradient_isolation_violations(teacher) == (0, 0)
        actor_before = [value.detach().clone() for value in actor.parameters()]
        optimizer.step()
        assert all(torch.isfinite(value).all() for value in actor.parameters())
        assert any(not torch.equal(before, value) for before, value in zip(actor_before, actor.parameters()))
        assert all(torch.equal(value, teacher_parameters_before[name]) for name, value in teacher.named_parameters())
        assert all(torch.isfinite(value).all() for state in optimizer.state.values() for value in state.values()
                   if torch.is_tensor(value))
        optimizer.zero_grad(set_to_none=True)

        state = EMAUpdateState()
        update_ema_once_(teacher, actor, 0.99, 0, state)
        assert state.update_count == 1 and state.last_rollout_iteration == 0
        with pytest.raises(RuntimeError, match="already been updated"):
            update_ema_once_(teacher, actor, 0.99, 0, state)
        assert teacher_gradient_isolation_violations(teacher) == (0, 0)
        assert torch.equal(cache, original_cache) and torch.equal(teacher_cache, original_cache)
        assert all(torch.isfinite(value).all() for value in teacher.parameters())
        assert any(not torch.equal(value, teacher_parameters_before[name]) for name, value in teacher.named_parameters())

        # Exercise the worker's actual parameter-only offload helpers. Buffers
        # must stay FP32/CUDA, and reloaded updated parameters must score exactly.
        with torch.no_grad():
            expected_after_step = actor(inputs_embeds=student_input, **student_kwargs).logits
        offload_fsdp_model_to_cpu(actor, empty_cache=False)
        assert all(value.device.type == "cpu" for value in actor.parameters())
        assert actor_model.model._opd_native_rope_cache is cache
        assert cache.data_ptr() == cache_pointer and cache.device == device and cache.dtype == torch.float32
        assert torch.equal(cache, original_cache)
        load_fsdp_model_to_gpu(actor)
        assert all(value.device == device for value in actor.parameters())
        with torch.no_grad():
            actual_after_reload = actor(inputs_embeds=student_input, **student_kwargs).logits
        assert torch.equal(actual_after_reload, expected_after_step)
        assert cache.data_ptr() == cache_pointer and torch.equal(cache, original_cache)
        torch.cuda.synchronize(device)
    finally:
        try:
            if group_created:
                dist.destroy_process_group()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
