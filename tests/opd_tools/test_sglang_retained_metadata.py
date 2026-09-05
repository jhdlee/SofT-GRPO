"""CPU execution of native sampler metadata and engine output routing."""

import ast
import copy
import dataclasses
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SRT = Path(__file__).resolve().parents[2] / "Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python/sglang/srt"


def load_nodes(path, names, namespace, *, class_name=None):
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    selected = [copy.deepcopy(node) for node in body if getattr(node, "name", None) in names]
    assert {node.name for node in selected} == set(names)
    for node in selected:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("noise_mode", ["gumbel", "gaussian", "dirichlet"])
def test_sampler_retained_mask_follows_native_action_sort_and_categorical_sentinel(noise_mode):
    # The renormalizers stand in for CUDA FlashInfer kernels. Everything after
    # their output is the actual sampler forward, including noise and sorting.
    filtered = torch.tensor([[0.60, 0.0, 0.25, 0.0, 0.15, 0.0, 0.0]]).repeat(2, 1)
    if noise_mode == "dirichlet":
        # The released Gamma branch requires strictly positive concentration.
        filtered = torch.tensor([[0.45, 0.20, 0.15, 0.12, 0.08, 0.0, 0.0]]).repeat(2, 1)
    draw = torch.tensor([[-1.4, 2.7, 0.4, -0.2, 1.3]]).repeat(2, 1)
    namespace = {
        "torch": torch, "global_server_args_dict": {"sampling_backend": "flashinfer"},
        "SYNC_TOKEN_IDS_ACROSS_TP": False,
        "top_k_renorm_prob": lambda probs, ks: probs,
        "top_p_renorm_prob": lambda probs, ps: filtered.clone(),
        "_request_local_gumbel": lambda logits, info: draw.clone(),
        "_request_local_categorical": lambda probs, info: probs.argmax(-1, keepdim=True),
    }
    forward = load_nodes(SRT / "layers/sampler.py", ["forward"], namespace, class_name="Sampler")["forward"]
    info = SimpleNamespace(
        has_custom_logit_processor=False, is_all_greedy=False, temperatures=torch.ones(2, 1),
        soft_thinking_modes=torch.tensor([True, False]), top_ps=torch.tensor([0.95, 0.95]),
        after_thinking_top_ps=torch.ones(2), top_ks=torch.full((2,), 5), after_thinking_top_ks=torch.full((2,), 5),
        min_ps=torch.zeros(2), after_thinking_min_ps=torch.zeros(2), need_min_p_sampling=False,
        need_after_thinking_min_p_sampling=False, max_topk=5, noise_factor=torch.ones(2),
        gumbel_softmax_temperatures=torch.full((2, 1), 0.1), noise_gumbel=True, noise_on_logits=True,
        deterministic_random_mask=torch.zeros(2, dtype=torch.bool), random_counters=torch.zeros(2, dtype=torch.int64),
        grammars=None, device="cpu",
    )
    original_logits = torch.tensor([[4.0, 1.0, 3.0, 0.0, 2.0, -1.0, -2.0]]).repeat(2, 1)
    output = SimpleNamespace(next_token_logits=original_logits.clone())
    torch.manual_seed(791)
    tokens = forward(SimpleNamespace(use_nan_detection=False), output, info, True, [], [],
                     enable_soft_thinking=True, add_noise_dirichlet=noise_mode == "dirichlet",
                     add_noise_gumbel_softmax=noise_mode == "gumbel")

    probabilities, support = torch.topk(filtered, 5, dim=-1)
    probabilities /= probabilities.sum(-1, keepdim=True)
    retained = probabilities > 0
    torch.manual_seed(791)
    if noise_mode == "gumbel":
        perturbed = (probabilities + 1e-6).log() + draw
        weights = (perturbed / 0.1).softmax(-1)
    elif noise_mode == "dirichlet":
        perturbed = torch.distributions.Gamma(probabilities, torch.ones_like(probabilities)).rsample()
        weights = perturbed / perturbed.sum(-1, keepdim=True)
    else:
        perturbed = torch.distributions.Normal(probabilities, torch.ones_like(probabilities) * 0.05).rsample()
        weights = perturbed / perturbed.sum(-1, keepdim=True)
    weights, permutation = weights.sort(-1, descending=True)
    support = support.gather(-1, permutation)
    expected_mask = retained.gather(-1, permutation)
    expected_mask[1] = torch.tensor([True, False, False, False, False])
    assert output.topk_retained_mask.dtype == torch.bool
    assert torch.equal(output.topk_retained_mask, expected_mask)
    # Metadata is observational: the exact existing stochastic action survives.
    assert torch.equal(output.topk_probs[0], weights[0])
    assert torch.equal(output.topk_indices[0], support[0])
    assert torch.equal(output.topk_gumbels[0], perturbed.gather(-1, permutation)[0])
    assert int(tokens[0]) == int(support[0, 0])
    assert torch.equal(output.topk_probs[1], torch.tensor([1.0, 0, 0, 0, 0]))
    assert torch.equal(output.topk_indices[1], torch.tensor([0, 0, 0, 0, 0]))
    if noise_mode == "gumbel":
        assert torch.equal(output.topk_gumbel_noise[0], draw.gather(-1, permutation)[0])
        density_mask = ((probabilities + 1e-6).log() > -3).float()
        expected_density = ((-draw - (-draw).exp()) * density_mask).sum(-1) / density_mask.sum(-1)
        assert torch.equal(output.next_token_gumbel_logprobs, expected_density)
    assert torch.equal(output.topk_gumbel_noise[1], torch.zeros(5))
    assert torch.equal(output.next_token_logprobs, original_logits.softmax(-1).log().gather(-1, tokens.long()[:, None]).squeeze(-1))


def test_greedy_sampler_marks_only_the_categorical_head_retained():
    namespace = {"torch": torch, "SYNC_TOKEN_IDS_ACROSS_TP": False}
    forward = load_nodes(SRT / "layers/sampler.py", ["forward"], namespace, class_name="Sampler")["forward"]
    output = SimpleNamespace(next_token_logits=torch.tensor([[1.0, 4.0, 2.0]]),
                             topk_probs=torch.zeros(1, 5), topk_indices=torch.zeros(1, 5, dtype=torch.int64))
    info = SimpleNamespace(has_custom_logit_processor=False, is_all_greedy=True, grammars=None)
    tokens = forward(SimpleNamespace(use_nan_detection=False), output, info, False, [], [], enable_soft_thinking=True)
    assert tokens.tolist() == [1]
    assert output.topk_retained_mask.tolist() == [[True, False, False, False, False]]


@pytest.mark.parametrize("case", ["latent", "natural_close", "early_close", "categorical"])
def test_request_records_exact_mask_and_flushes_aligned_boolean_metadata(case):
    namespace = load_nodes(
        SRT / "managers/schedule_batch.py", ["update_topk_info", "get_output_topk_retained_mask_list"],
        {"torch": torch}, class_name="Req",
    )
    mask = torch.tensor([[True, False, True, False, True]])
    output = SimpleNamespace(
        topk_gumbels=torch.tensor([[0.2, 0.1, -4.0, -5.0, -6.0]]),
        topk_gumbel_noise=torch.zeros(1, 5), topk_retained_mask=mask.clone(),
        topk_probs=torch.tensor([[0.9, 0.05, 0.03, 0.01, 0.01]]),
        topk_indices=torch.tensor([[3, 7, 2, 8, 5]]), entropy=torch.tensor([0.0]),
    )
    request = SimpleNamespace(
        sampling_params=SimpleNamespace(soft_thinking_mode=case != "categorical", think_end_str_id=9,
                                        early_stopping_entropy_threshold=0.1 if case == "early_close" else 0,
                                        early_stopping_length_threshold=1),
        output_ids=[9 if case == "natural_close" else 3], low_entropy_steps=0,
        output_topk_prob_list_tmp=[], output_topk_gumbel_list_tmp=[], output_topk_gumbel_noise_list_tmp=[],
        output_topk_retained_mask_list_tmp=[], output_topk_retained_mask_list=[], output_topk_idx_list_tmp=[],
    )
    namespace["update_topk_info"](request, output, 0)
    observed = namespace["get_output_topk_retained_mask_list"](request)
    expected = mask.tolist() if case == "latent" else [[True, False, False, False, False]]
    assert observed == expected
    assert all(type(value) is bool for value in observed[0])
    assert not request.output_topk_retained_mask_list_tmp
    assert namespace["get_output_topk_retained_mask_list"](request) == expected
    if case in ("natural_close", "early_close"):
        assert request.output_ids == [9]
        assert request.topk_idx.tolist() == [9, 0, 0, 0, 0]


def io_types():
    return load_nodes(SRT / "managers/io_struct.py", ["BatchTokenIDOut", "BatchStrOut", "BatchEmbeddingOut", "BatchMultimodalOut"], {"dataclass": dataclasses.dataclass})


@pytest.mark.parametrize("type_name", ["BatchTokenIDOut", "BatchStrOut", "BatchEmbeddingOut", "BatchMultimodalOut"])
def test_actual_multi_tokenizer_routing_preserves_compact_fields_and_dataclass_contract(type_name):
    namespace = io_types()
    route = load_nodes(SRT / "managers/multi_tokenizer_mixin.py", ["_handle_output_by_index"], namespace)["_handle_output_by_index"]
    cls = namespace[type_name]
    data = {field.name: [field.name + "-0", field.name + "-1"] for field in dataclasses.fields(cls)}
    if "output_topk_retained_mask_list" in data:
        data["output_topk_retained_mask_list"] = [[[True, False, True]], [[True, True, False]]]
    original = cls(**data)
    routed = route(original, 1)
    for field in dataclasses.fields(cls):
        assert getattr(routed, field.name) == [data[field.name][1]]


def test_scheduler_positional_constructor_and_detokenizer_preserve_field_alignment():
    namespace = io_types()
    path = SRT / "managers/scheduler_output_processor_mixin.py"
    constructors = [node for node in ast.walk(ast.parse(path.read_text()))
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "BatchTokenIDOut"]
    assert len(constructors) == 1
    constructor = constructors[0]
    fields = [field.name for field in dataclasses.fields(namespace["BatchTokenIDOut"])]
    assert len(constructor.args) == len(fields)
    for field, argument in zip(fields, constructor.args):
        assert isinstance(argument, ast.Name)
        namespace[argument.id] = [field]
    output = eval(compile(ast.Expression(constructor), str(path), "eval"), namespace)
    assert all(getattr(output, name) == [name] for name in fields)

    path = SRT / "managers/detokenizer_manager.py"
    constructor = next(node for node in ast.walk(ast.parse(path.read_text()))
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "BatchStrOut")
    namespace.update(recv_obj=output, output_strs=["decoded"])
    decoded = eval(compile(ast.Expression(constructor), str(path), "eval"), namespace)
    for name in ("output_topk_retained_mask_list", "output_topk_gumbel_noise_list", "output_topk_gumbel_list", "output_topk_indices_list"):
        assert getattr(decoded, name) == getattr(output, name)


def test_tokenizer_exposes_retained_mask_without_conversion():
    path = SRT / "managers/tokenizer_manager.py"
    assignments = [node for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
                           and target.slice.value == "output_topk_retained_mask_list" for target in node.targets)]
    assert len(assignments) == 1
    masks = [[[True, False]], [[False, True]]]
    namespace = {"meta_info": {}, "recv_obj": SimpleNamespace(output_topk_retained_mask_list=masks), "i": 1}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["meta_info"]["output_topk_retained_mask_list"] is masks[1]


def test_native_weighted_embedding_equals_actual_actor_reconstruction_after_fp32_update():
    # Reuse only the CPU dependency stubs; both functions under comparison are
    # compiled from their actual production source by the two test loaders.
    helper_path = SRT.parents[4] / "verl-0.4.x/tests/opd/test_actor_retained_support.py"
    spec = importlib.util.spec_from_file_location("native_metadata_actor_cpu_helpers", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    native = load_nodes(SRT / "layers/vocab_parallel_embedding.py", ["weighted_forward"],
                        {"torch": torch}, class_name="VocabParallelEmbedding")["weighted_forward"]

    class RecordingModel(helper.Model):
        def __call__(self, **kwargs):
            self.last_kwargs = kwargs
            return super().__call__(**kwargs)

    actor, batch = helper._actor(), helper._batch()
    actor.actor_module = RecordingModel()
    with torch.no_grad():
        # FP32 master values deliberately differ from their inference BF16 copy.
        actor.actor_module.embed.weight.copy_(
            torch.linspace(-0.77777, 1.22223, 64).reshape(16, 4)
        )
    forward = helper._load_forward(lambda **kwargs: torch.zeros_like(kwargs["labels"], dtype=torch.float32))
    forward(actor, batch, temperature=1.0)
    active = batch["attention_mask"].bool()
    support = batch["rollout_topk_ids"][active]
    weights = batch["rollout_topk_probs"][active]
    table = actor.actor_module.embed.weight.detach().bfloat16()
    engine_embedding = SimpleNamespace(quant_method=SimpleNamespace(embedding=lambda _, ids: table[ids]))
    expected = native(engine_embedding, weights, support)
    assert torch.equal(actor.actor_module.last_inputs[0], expected)
    # Packed row boundaries survive inputs_embeds dispatch to model attention.
    assert actor.actor_module.last_kwargs["position_ids"].tolist() == [[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]]
    assert actor.actor_module.last_kwargs["attention_mask"] is None
    assert actor.actor_module.last_kwargs["use_cache"] is False
