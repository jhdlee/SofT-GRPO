"""Short GPU localization of native/HF Qwen3 forward drift.

No scheduler server, sampling, optimizer, teacher, or model-weight export. This
diagnostic is not rollout acceptance. Activations stay in process; JSON contains
scalar comparisons and at most eight vocabulary/support entries per comparison.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import types

import torch
import torch.nn.functional as F

from .manifest import canonical_sha256, file_sha256
from .training_runtime import atomic_write_json


def difference(left, right):
    """Finite diagnostics without hiding dtype, shape, or nonfinite mismatches."""
    if left.shape != right.shape:
        return {"shape_matches": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    a, b = left.detach().float(), right.detach().float()
    finite = torch.isfinite(a) & torch.isfinite(b)
    delta = (a - b).abs()[finite]
    return {"shape_matches": True, "shape": list(a.shape), "left_dtype": str(left.dtype),
            "right_dtype": str(right.dtype), "nonfinite_elements": int((~finite).sum()),
            "exact_elements": int((a == b).sum()), "elements": a.numel(),
            "max_abs": float(delta.max()) if delta.numel() else None,
            "mean_abs": float(delta.mean()) if delta.numel() else None,
            "rms": float(delta.square().mean().sqrt()) if delta.numel() else None}


def logits_difference(left, right, limit=8):
    """Compare the final causal position, including threshold-sensitive supports."""
    a, b = left[-1].float(), right[-1].float()
    result = difference(a, b)
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return result
    ids = (a - b).abs().topk(min(limit, a.numel())).indices
    support = a.topk(5).indices
    pa, pb = torch.softmax(a[support], -1), torch.softmax(b[support], -1)
    la, lb = torch.log(pa + 1e-6), torch.log(pb + 1e-6)
    result.update(
        worst=[{"token_id": int(i), "native_logit": float(a[i]), "comparison_logit": float(b[i])} for i in ids],
        native_top5_support=[{"token_id": int(i), "native_logit": float(a[i]), "comparison_logit": float(b[i]),
                             "native_conditional_logp": float(la[j]), "comparison_conditional_logp": float(lb[j]),
                             "native_score_mask": bool(la[j] > -3), "comparison_score_mask": bool(lb[j] > -3)}
                            for j, i in enumerate(support)],
        support_note="Fixed native final-position top-five logits, conditional normalization plus released epsilon; synthetic diagnostic support, not a sampled action or replay gate.")
    return result


class Trace:
    """Clone before native in-place norms can overwrite observations."""
    def __init__(self):
        self.values = {}
        self.tokens = 0
        self.layer = 0

    def emit(self, name, value):
        if isinstance(value, (tuple, list)):
            value = value[0]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"trace {name} is not a tensor")
        # Callers must put token dimension first (or batch=1, tokens second).
        value = value.reshape(self.tokens, -1) if name != "logits" else value.reshape(-1, value.shape[-1])
        self.values.setdefault(name, []).append(value.detach().clone())

    def finish(self):
        return {name: torch.cat(parts, dim=0) for name, parts in self.values.items()}


def compare_traces(native, other):
    common = native.keys() & other.keys()
    rows = {name: difference(native[name], other[name]) for name in sorted(common) if name != "logits"}
    for name, row in rows.items():
        if row["shape_matches"] and row["exact_elements"] != row["elements"]:
            different = (native[name] != other[name]).reshape(native[name].shape[0], -1).sum(-1)
            row["differing_rows"] = {
                "prefix_elements": int(different[:-4].sum()),
                "decode_elements_by_row": different[-4:].tolist(),
                "first_row_indices": different.nonzero().flatten()[:8].tolist(),
            }
    return {"stages": rows, "missing_native": sorted(other.keys() - native.keys()),
            "missing_comparison": sorted(native.keys() - other.keys()),
            "final_logits": logits_difference(native["logits"], other["logits"])}


def synthetic_sequence(tokenizer, length):
    """Public fixed text and mixtures; no training rows or privileged answers."""
    if type(length) is not int or not 16 <= length <= 256:
        raise ValueError("sequence length must be between 16 and 256")
    text = "A short arithmetic example: seventeen plus twenty-five equals forty-two. Check each step carefully. "
    ids = tokenizer.encode(text * 20, add_special_tokens=False)[:length]
    if len(ids) != length:
        raise ValueError("tokenizer did not produce the requested short sequence")
    support = torch.zeros(length, 5, dtype=torch.long)
    support[:, 0] = torch.tensor(ids)
    probs = torch.zeros(length, 5, dtype=torch.float32)
    probs[:, 0] = 1
    # Eight positions ending immediately before a final hard token; decode four
    # positions therefore exercises both soft and hard inputs using the same KV.
    for pos in range(length - 9, length - 1):
        support[pos] = torch.tensor([(ids[pos] + 31 * k) % tokenizer.vocab_size for k in range(5)])
        probs[pos] = torch.tensor([0.51, 0.20, 0.14, 0.10, 0.05])
    return torch.tensor(ids, dtype=torch.long), support, probs


@contextlib.contextmanager
def native_hooks(model, trace):
    handles = []
    def hook(module, name):
        handles.append(module.register_forward_hook(lambda mod, args, out: trace.emit(name, out)))
    for i, layer in enumerate(model.model.layers):
        prefix = f"layer.{i}."
        def before(mod, args, index=i):
            trace.layer = index
            hidden, residual = args[1], args[3]
            total = hidden if residual is None else (hidden.float() + residual.float()).to(hidden.dtype)
            trace.emit(f"layer.{index}.block_input", total)
            if index == 0:
                trace.emit("embedding", hidden)
        handles.append(layer.register_forward_pre_hook(before))
        hook(layer.input_layernorm, prefix + "norm_in")
        hook(layer.post_attention_layernorm, prefix + "norm_post")
        hook(layer.self_attn.qkv_proj, prefix + "qkv")
        hook(layer.self_attn.q_norm, prefix + "q_norm")
        hook(layer.self_attn.k_norm, prefix + "k_norm")
        def attention_input(mod, args, index=i):
            trace.emit(f"layer.{index}.q_rope", args[0])
            trace.emit(f"layer.{index}.k_rope", args[1])
        handles.append(layer.self_attn.attn.register_forward_pre_hook(attention_input))
        hook(layer.self_attn.attn, prefix + "attention")
        hook(layer.self_attn.o_proj, prefix + "attention_projected")
        hook(layer.mlp.gate_up_proj, prefix + "gate_up")
        hook(layer.mlp, prefix + "mlp")
        def after(mod, args, out, index=i):
            hidden, residual = out
            trace.emit(f"layer.{index}.block_total", (hidden.float() + residual.float()).to(hidden.dtype))
        handles.append(layer.register_forward_hook(after))
    hook(model.model.norm, "final_norm")
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextlib.contextmanager
def hf_hooks(model, trace):
    """Observe the actual HF forward, including its separate linear calls."""
    from transformers.models.qwen3 import modeling_qwen3
    handles, parts = [], {}
    original_rope = modeling_qwen3.apply_rotary_pos_emb
    def rope(*args, **kwargs):
        q, k = original_rope(*args, **kwargs)
        trace.emit(f"layer.{trace.layer}.q_rope", q.transpose(1, 2).contiguous())
        trace.emit(f"layer.{trace.layer}.k_rope", k.transpose(1, 2).contiguous())
        return q, k
    def hook(module, name):
        handles.append(module.register_forward_hook(lambda mod, args, out: trace.emit(name, out)))
    for i, layer in enumerate(model.model.layers):
        prefix = f"layer.{i}."
        def before(mod, args, kwargs, index=i):
            trace.layer = index
            hidden = args[0] if args else kwargs["hidden_states"]
            trace.emit(f"layer.{index}.block_input", hidden)
            if index == 0:
                trace.emit("embedding", hidden)
        handles.append(layer.register_forward_pre_hook(before, with_kwargs=True))
        for stage, module in (("norm_in", layer.input_layernorm), ("norm_post", layer.post_attention_layernorm),
                              ("q_norm", layer.self_attn.q_norm), ("k_norm", layer.self_attn.k_norm),
                              ("attention_projected", layer.self_attn.o_proj), ("mlp", layer.mlp), ("block_total", layer)):
            hook(module, prefix + stage)
        for group, names, parent in (("qkv", ("q_proj", "k_proj", "v_proj"), layer.self_attn),
                                      ("gate_up", ("gate_proj", "up_proj"), layer.mlp)):
            for name in names:
                def projection(mod, args, out, index=i, group=group, name=name, names=names):
                    parts[index, group, name] = out.detach().clone()
                    if name == names[-1]:
                        trace.emit(f"layer.{index}.{group}", torch.cat([parts.pop((index, group, n)) for n in names], -1))
                handles.append(getattr(parent, name).register_forward_hook(projection))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda mod, args, index=i: trace.emit(f"layer.{index}.attention", args[0])))
    hook(model.model.norm, "final_norm")
    modeling_qwen3.apply_rotary_pos_emb = rope
    try:
        yield
    finally:
        modeling_qwen3.apply_rotary_pos_emb = original_rope
        for handle in handles:
            handle.remove()


def make_native_batch(runner, ids):
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    # No requests survive a completed synchronous case; reuse only KV storage.
    runner.req_to_token_pool.clear()
    runner.token_to_kv_pool_allocator.clear()
    params = SamplingParams(temperature=1, top_k=5, noise_factor=1, max_new_tokens=8)
    params.normalize(None)
    req = Req(rid="parity", origin_input_text="", origin_input_ids=ids.tolist(),
              sampling_params=params, enable_soft_thinking=True, max_topk=5)
    req.prefix_indices, req.fill_ids = [], req.origin_input_ids
    req.extend_input_len, req.logprob_start_len = len(req.fill_ids), len(req.fill_ids) - 1
    batch = ScheduleBatch.init_new(reqs=[req], req_to_token_pool=runner.req_to_token_pool,
                                  token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
                                  tree_cache=None, model_config=runner.model_config, enable_overlap=False,
                                  spec_algorithm=SpeculativeAlgorithm.NONE, enable_custom_logit_processor=False)
    batch.prepare_for_extend()
    return batch


def native_forward(runner, ids, support, probs, *, cached=False):
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    trace = Trace()
    prefix = len(ids) - 4 if cached else len(ids)
    batch = make_native_batch(runner, ids[:prefix])
    with torch.no_grad(), native_hooks(runner.model, trace):
        for start, end in [(0, prefix)] + ([(i, i + 1) for i in range(prefix, len(ids))] if cached else []):
            if start:
                batch.output_ids = ids[start:end].cuda()
                batch.reqs[0].topk_idx = (support[start] if support is not None else torch.tensor([int(ids[start]), 0, 0, 0, 0])).cuda()
                batch.reqs[0].topk_prob = (probs[start] if probs is not None else torch.tensor([1., 0., 0., 0., 0.])).cuda()
                batch.prepare_for_decode()
            forward = ForwardBatch.init_new(batch.get_model_worker_batch(), runner)
            forward.topk_indices = forward.topk_probs = None
            if support is not None:
                forward.topk_indices = support[start:end].cuda()
                forward.topk_probs = probs[start:end].cuda()
            trace.tokens = end - start
            output = runner.forward(forward)
            trace.emit("logits", output.next_token_logits)
        torch.cuda.synchronize()
    return trace.finish()


def hf_forward(model, ids, embedding=None, *, candidate_emit_state=None):
    trace = Trace()
    trace.tokens = len(ids)
    model.train()  # Qwen3 dropout is zero; the production replay model is in train mode.
    if candidate_emit_state is not None:
        candidate_emit_state["trace"] = trace
    hooks = hf_hooks(model, trace) if candidate_emit_state is None else contextlib.nullcontext()
    with torch.no_grad(), hooks:
        kwargs = {"input_ids": ids.cuda().unsqueeze(0)} if embedding is None else {"inputs_embeds": embedding.unsqueeze(0)}
        out = model(**kwargs, attention_mask=None, position_ids=torch.arange(len(ids), device="cuda").unsqueeze(0), use_cache=False)
        trace.emit("logits", out.logits)
        torch.cuda.synchronize()
    return trace.finish()


def packed_projection_controls(native, hf, trace, linear=F.linear):
    """Hold activations/weights fixed to isolate packed versus separate GEMMs."""
    rows = []
    for i, (nlayer, hlayer) in enumerate(zip(native.model.layers, hf.model.layers)):
        for group, stage, nmodule, hmodules in (
            ("qkv", "norm_in", nlayer.self_attn.qkv_proj, [hlayer.self_attn.q_proj, hlayer.self_attn.k_proj, hlayer.self_attn.v_proj]),
            ("gate_up", "norm_post", nlayer.mlp.gate_up_proj, [hlayer.mlp.gate_proj, hlayer.mlp.up_proj]),
        ):
            weight = torch.cat([m.weight for m in hmodules])
            if not torch.equal(weight, nmodule.weight):
                raise RuntimeError(f"pinned packed weights differ at layer {i} {group}")
            x = trace[f"layer.{i}.{stage}"]
            packed = linear(x, weight)
            separate = torch.cat([F.linear(x, m.weight) for m in hmodules], -1)
            rows.append({"layer": i, "group": group, "weights_equal": True,
                         "packed_vs_separate_same_input": difference(packed, separate),
                         "native_observed_vs_packed_same_input": difference(trace[f"layer.{i}.{group}"], packed)})
    return rows


def projection_shape_controls(value, weight, linear=F.linear):
    """Identical rows/weights under prefill, decode, and reordered schedules."""
    full = linear(value, weight)
    prefix = len(value) - 4
    chunked = torch.cat([linear(value[:prefix], weight)] + [linear(value[i:i + 1], weight) for i in range(prefix, len(value))])
    permutation = torch.arange(len(value) - 1, -1, -1, device=value.device)
    permuted = linear(value[permutation], weight)[permutation]
    return {"full_vs_prefix": difference(full[:prefix], chunked[:prefix]),
            "full_vs_decode": difference(full[prefix:], chunked[prefix:]),
            "full_vs_reordered": difference(full, permuted)}


def install_native_probe_linear(model, linear):
    """Instance-only TP1 unquantized projection/head control, never a server default."""
    from sglang.srt.layers.linear import UnquantizedLinearMethod

    processor = model.logits_processor
    if processor.do_tensor_parallel_all_gather or processor.do_tensor_parallel_all_gather_dp_attn or processor.final_logit_softcapping:
        raise ValueError("probe linear replacement requires TP1 without logit softcapping")
    modules = [module for module in model.modules() if isinstance(getattr(module, "quant_method", None), UnquantizedLinearMethod)]
    if not modules:
        raise ValueError("no native unquantized projections found")
    for module in modules:
        method = copy.copy(module.quant_method)
        def apply(method_self, layer, value, bias=None):
            return linear(value, layer.weight, bias)
        method.apply = types.MethodType(apply, method)
        module.quant_method = method
    def get_logits(processor_self, hidden_states, lm_head, logits_metadata, embedding_bias=None):
        if embedding_bias is not None:
            raise ValueError("probe native head does not support embedding bias")
        logits = linear(hidden_states.to(lm_head.weight.dtype), lm_head.weight)
        if processor_self.logit_scale is not None:
            logits = logits * processor_self.logit_scale
        return logits[:, :processor_self.config.vocab_size].float()
    processor._get_logits = types.MethodType(get_logits, processor)
    return len(modules)


def fix_native_fa3_split_count():
    """Probe process only: the pinned backend otherwise lets decode choose splits."""
    from sglang.srt.layers.attention import flashattention_backend as backend
    for name in ("flash_attn_varlen_func", "flash_attn_with_kvcache"):
        original = getattr(backend, name)
        def fixed(*args, _original=original, **kwargs):
            kwargs["num_splits"] = 1
            return _original(*args, **kwargs)
        setattr(backend, name, fixed)


def run(args, *, wandb_run=None):
    from .qwen_training import verify
    from .training_benchmark import source_identity
    from sglang.bench_one_batch import load_model
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs, ServerArgs
    from transformers import AutoModelForCausalLM
    from verl.models.transformers.monkey_patch import apply_monkey_patch

    source = source_identity(Path(__file__).resolve().parents[1])
    source["source_snapshot"] = str(Path(__file__).resolve().parents[3])
    source["probe_file_sha256"] = file_sha256(Path(__file__))
    assets = verify(args.assets)
    if not torch.cuda.is_available():
        raise RuntimeError("this probe requires an allocated CUDA GPU")
    if torch.cuda.device_count() != 1 or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("the probe requires exactly one visible H100")
    torch.manual_seed(11)
    torch.cuda.set_device(0)
    model_path = str(Path(args.assets).resolve() / "model")
    server = ServerArgs(model_path=model_path, dtype="bfloat16", device="cuda", tp_size=1,
                        context_length=512, max_total_tokens=1024, max_running_requests=2,
                        mem_fraction_static=0.2, chunked_prefill_size=512, max_prefill_tokens=512,
                        attention_backend=args.attention_backend, disable_cuda_graph=True, disable_radix_cache=True,
                        disable_overlap_schedule=True, enable_soft_thinking=True, max_topk=5, random_seed=11)
    _set_envs_and_config(server)
    # Explicit FP32 backward policy for the differentiable projection candidate.
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.attention_backend == "fa3":
        fix_native_fa3_split_count()
    runner, tokenizer = load_model(server, PortArgs.init_new(server), 0)
    # bench_one_batch's generic loader omits these fork-specific model fields.
    runner.model_config.enable_soft_thinking = True
    runner.model_config.max_topk = 5
    linear = F.linear
    native_linear_count = 0
    if args.batch_invariant_linear:
        from verl.opd.batch_invariant_linear import batch_invariant_linear
        linear = batch_invariant_linear
        native_linear_count = install_native_probe_linear(runner.model, linear)
    def load_hf():
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16,
                                                    attn_implementation="flash_attention_2").cuda()
        apply_monkey_patch(model, use_remove_padding=True, ulysses_sp_size=1, use_fused_kernels=False)
        return model
    hf = load_hf()
    if not torch.equal(runner.model.model.embed_tokens.weight[:hf.config.vocab_size], hf.model.embed_tokens.weight):
        raise RuntimeError("native/HF embedding weights differ")
    ids, support, probs = synthetic_sequence(tokenizer, args.length)
    result = {"schema_version": 1, "role": "qwen3_forward_parity_diagnostic", "status": "running",
              "source": source, "assets_manifest_sha256": assets["manifest_content_sha256"],
              "jobs": {"slurm_job_id": os.environ.get("SLURM_JOB_ID")},
              "device": {"name": torch.cuda.get_device_name(0), "visible_count": torch.cuda.device_count(),
                         "memory_bytes": torch.cuda.get_device_properties(0).total_memory},
              "wandb": {"enabled": wandb_run is not None, "run_id": wandb_run.id if wandb_run else None,
                        "url": wandb_run.url if wandb_run else None, "finished": False},
              "model": assets["model"], "sequence": {"length": len(ids), "ids_sha256": canonical_sha256(ids.tolist()),
                      "support_sha256": canonical_sha256(support.tolist()), "probabilities_sha256": canonical_sha256(probs.tolist()),
                      "soft_positions": list(range(len(ids) - 9, len(ids) - 1)), "cached_decode_steps": 4},
              "configuration": {"native": {"attention_backend": args.attention_backend, "tp_size": 1, "disable_cuda_graph": True,
                                              "fixed_num_splits": 1 if args.attention_backend == "fa3" else None},
                                "hf": {"attention_backend": "flash_attention_2", "parameter_dtype": "bfloat16", "train_mode": True,
                                       "remove_padding": True, "batch_size": 1, "fsdp": False}},
              "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "flash-attn", "flashinfer-python")},
              "cases": [], "full_training_estimate": None,
              "limitations": ["Synthetic short fixed inputs; no inference about long-trajectory frequency or training acceptance.",
                              "No FSDP wrapping, optimizer, teacher or EMA. Optional synthetic backward is not full OPD acceptance.",
                              "Hooks clone actual intermediate tensors and disable CUDA graphs; timing is diagnostic, not throughput.",
                              "Native layer outputs retain a separate residual; block_total is the BF16 sum for comparison only.",
                              "Full-prefill versus cached decode changes GEMM/attention shapes; final-head controls must be interpreted separately."]}
    result["configuration"]["projection"] = {
        "policy": "fixed_tile_triton_32_64_32" if args.batch_invariant_linear else "pytorch_default",
        "native_module_count": native_linear_count,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "flashinfer_use_tensor_core_env": os.environ.get("SGLANG_FLASHINFER_USE_TENSOR_CORE"),
    }
    if args.batch_invariant_linear:
        result["configuration"]["projection"]["source_sha256"] = file_sha256(Path(importlib.import_module("verl.opd.batch_invariant_linear").__file__))
        result["limitations"].append("Fixed-tile cases modify native inference and the replay candidate together; parity would validate this proposed common arithmetic, not the released F.linear inference path.")
    atomic_write_json(args.output, result)
    candidate_hf, candidate_emit_state = None, {}
    if args.candidate:
        module, name = args.candidate.split(":", 1)
        imported = importlib.import_module(module)
        # A separate instance prevents candidate patches from contaminating
        # subsequent baseline cases. Its own callbacks replace module hooks.
        candidate_hf = load_hf()
        installer_kwargs = {"emit": lambda key, tensor: candidate_emit_state["trace"].emit(key, tensor)}
        if args.batch_invariant_linear:
            installer_kwargs["linear"] = linear
        if args.candidate_attention_fa3:
            from verl.opd.native_fa3_attention import native_fa3_attention
            installer_kwargs["attention"] = native_fa3_attention
        installed = getattr(imported, name)(candidate_hf, **installer_kwargs)
        if installed is not None:
            candidate_hf = installed
        result["candidate"] = {"callable": args.candidate, "source_sha256": file_sha256(Path(imported.__file__))}
        result["candidate"]["attention"] = "fa3_single_split_forward_fa2_backward" if args.candidate_attention_fa3 else "flash_attention_2"
        if args.candidate_attention_fa3:
            result["candidate"]["attention_source_sha256"] = file_sha256(Path(importlib.import_module("verl.opd.native_fa3_attention").__file__))
            result["limitations"].append("Diagnostic FA3 candidate uses the installed FA2 analytic backward with actual FA3 output/LSE; synthetic gradient validation does not establish full OPD/FSDP acceptance.")
    for soft in (False, True):
        start = time.monotonic()
        native = native_forward(runner, ids, support if soft else None, probs if soft else None)
        cached = native_forward(runner, ids, support if soft else None, probs if soft else None, cached=True)
        backend = runner.attn_backend
        backend_identity = {"class": type(backend).__name__,
                            "decode_use_tensor_cores": getattr(backend, "decode_use_tensor_cores", None),
                            "ragged_prefill_backend_after_plan": getattr(getattr(backend, "prefill_wrapper_ragged", None), "_backend", None),
                            "paged_prefill_backends_after_plan": [getattr(wrapper, "_backend", None) for wrapper in getattr(backend, "prefill_wrappers_paged", [])]}
        embed = None
        if soft:
            table = hf.model.embed_tokens(support.cuda())
            normalized = probs.cuda() / probs.cuda().sum(-1, keepdim=True)
            embed = torch.sum(normalized.unsqueeze(-1) * table, dim=1, dtype=table.dtype)
        baseline = hf_forward(hf, ids, embed)
        row = {"input": "soft_mixtures" if soft else "hard_tokens", "native_attention_backend_observed": backend_identity,
               "native_prefill_vs_cached": compare_traces(native, cached),
               "native_prefill_vs_hf": compare_traces(native, baseline),
               "native_cached_vs_hf": compare_traces(cached, baseline),
               "packed_projection_controls": packed_projection_controls(runner.model, hf, native, linear),
               "head_same_input_last_position": difference(native["logits"][-1:], linear(native["final_norm"][-1:], hf.lm_head.weight).float()),
               "qkv_shape_controls": projection_shape_controls(native["layer.0.norm_in"], runner.model.model.layers[0].self_attn.qkv_proj.weight, linear),
               "qkv_pytorch_shape_controls": projection_shape_controls(native["layer.0.norm_in"], runner.model.model.layers[0].self_attn.qkv_proj.weight)}
        previous_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        try:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            row["qkv_pytorch_full_accumulation_shape_controls"] = projection_shape_controls(native["layer.0.norm_in"], runner.model.model.layers[0].self_attn.qkv_proj.weight)
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous_reduction
        # Feed exact native embedding values to HF, eliminating embedding arithmetic.
        controlled = hf_forward(hf, ids, native["embedding"])
        row["native_prefill_vs_hf_native_embedding"] = compare_traces(native, controlled)
        if candidate_hf is not None:
            candidate_trace = hf_forward(candidate_hf, ids, native["embedding"], candidate_emit_state=candidate_emit_state)
            row["native_prefill_vs_candidate"] = compare_traces(native, candidate_trace)
            row["native_cached_vs_candidate"] = compare_traces(cached, candidate_trace)
        row["diagnostic_wall_seconds"] = time.monotonic() - start
        result["cases"].append(row)
        if wandb_run is not None:
            wandb_run.log({row["input"] + "/" + key + "/final_logit_max_abs": value["final_logits"]["max_abs"]
                           for key, value in row.items() if isinstance(value, dict) and "final_logits" in value})
        atomic_write_json(args.output, result)
    if args.backward_sanity:
        if candidate_hf is None:
            raise ValueError("--backward-sanity requires --candidate")
        trace = Trace()
        trace.tokens = len(ids)
        candidate_emit_state["trace"] = trace
        candidate_hf.zero_grad(set_to_none=True)
        logits = candidate_hf(input_ids=ids.cuda().unsqueeze(0), use_cache=False).logits
        loss = logits[0, -1, :32].float().square().mean()
        loss.backward()
        grads = [parameter.grad for parameter in candidate_hf.parameters() if parameter.grad is not None]
        result["candidate_backward_sanity"] = {
            "objective": "mean square of first 32 logits at final hard-token position; synthetic, no optimizer",
            "loss": float(loss.detach()), "parameters_with_grad": len(grads),
            "all_finite": bool(grads) and all(bool(torch.isfinite(grad).all()) for grad in grads),
            "parameters_with_nonzero_grad": sum(bool(torch.count_nonzero(grad)) for grad in grads)}
        atomic_write_json(args.output, result)
        if not result["candidate_backward_sanity"]["all_finite"]:
            raise RuntimeError("candidate synthetic backward produced missing/nonfinite gradients")
    result["status"] = "diagnostic_complete"
    result["peak_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated()
    atomic_write_json(args.output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--candidate", help="Optional module:function installer on a separate HF model; receives (model, emit=callback), returns model or None, and emits canonical per-layer stages.")
    parser.add_argument("--wandb", action="store_true", help="Publish scalar diagnostic metrics online to the existing benchmark project.")
    parser.add_argument("--backward-sanity", action="store_true", help="After all forward traces, run one short synthetic candidate backward without an optimizer.")
    parser.add_argument("--batch-invariant-linear", action="store_true", help="Diagnostic fixed-tile matmul in native projections/head and the separate candidate; no training/default changes.")
    parser.add_argument("--attention-backend", choices=("flashinfer", "fa3"), default="flashinfer")
    parser.add_argument("--candidate-attention-fa3", action="store_true", help="Diagnostic one-split FA3 forward with existing FA2 analytic backward.")
    args = parser.parse_args(argv)
    if not 16 <= args.length <= 256:
        parser.error("--length must be between 16 and 256")
    if args.output.exists():
        parser.error("--output already exists; preserve previous probe artifacts")
    if args.backward_sanity and not args.candidate:
        parser.error("--backward-sanity requires --candidate")
    if args.batch_invariant_linear and not args.candidate:
        parser.error("--batch-invariant-linear requires --candidate")
    if args.candidate_attention_fa3 and (not args.candidate or args.attention_backend != "fa3"):
        parser.error("--candidate-attention-fa3 requires --candidate and --attention-backend fa3")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    try:
        if args.wandb:
            import wandb
            wandb_run = wandb.init(project="opd-qwen3-training-benchmark", entity=os.environ.get("WANDB_ENTITY"),
                                  job_type="forward-parity-diagnostic", mode="online", dir=str(args.output.parent),
                                  config={"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "length": args.length, "candidate": args.candidate,
                                          "batch_invariant_linear": args.batch_invariant_linear,
                                          "attention_backend": args.attention_backend, "candidate_attention_fa3": args.candidate_attention_fa3,
                                          "flashinfer_use_tensor_core_env": os.environ.get("SGLANG_FLASHINFER_USE_TENSOR_CORE")})
        result = run(args, wandb_run=wandb_run)
        if wandb_run is not None:
            wandb_run.finish(exit_code=0)
            result["wandb"]["finished"] = True
            atomic_write_json(args.output, result)
    except BaseException as error:
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        result = json.loads(args.output.read_text()) if args.output.exists() else {"schema_version": 1, "role": "qwen3_forward_parity_diagnostic"}
        if wandb_run is not None:
            result["wandb"] = {"enabled": True, "run_id": wandb_run.id, "url": wandb_run.url, "finished": True, "exit_code": 1}
        result.update(status="failed", error=f"{type(error).__name__}: {error}"[:2000])
        atomic_write_json(args.output, result)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
