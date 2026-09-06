"""Bounded real-forward observations, causal alignment, and microbatch restoration."""

import __future__
import ast
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from verl.opd.replay_diagnostics import capture_replay_diagnostics
from verl.trainer.ppo.opd_driver import build_replay_failure_diagnostics, replay_integrity_mask
from test_actor_retained_support import _actor, _batch, _load_forward


def fixture():
    batch, length, prompt, vocab, k = 2, 28, 4, 32, 3
    responses = torch.full((batch, length), 7, dtype=torch.long)
    responses[:, 12] = 29
    attention = torch.ones(batch, prompt + length, dtype=torch.long)
    attention[0, :2] = 0
    attention[1, -3:] = 0
    indices = attention.flatten().nonzero().flatten()
    # Different logits at every packed query make one-token and cross-row
    # misalignment observable, while keeping the fixture small.
    logits = torch.arange(indices.numel() * vocab, dtype=torch.float32).reshape(-1, vocab) / 29
    logits[:, 7] += torch.arange(indices.numel()) / 100
    logits.requires_grad_()
    support = torch.zeros(batch, length, k, dtype=torch.long)
    support[..., 0] = responses
    support[:, :12, 1:] = torch.tensor([10, 2])
    retained = torch.zeros_like(support, dtype=torch.bool)
    retained[..., 0] = True
    retained[:, :12, 1] = True
    dense_lookup = torch.full((batch * (prompt + length),), -1, dtype=torch.long)
    dense_lookup[indices] = torch.arange(indices.numel())
    row = torch.arange(batch).unsqueeze(-1)
    query = dense_lookup[row * (prompt + length) + prompt + torch.arange(length) - 1].clamp_min(0)
    actor = logits.detach().log_softmax(-1)[query, responses]
    # Soft errors dominate globally, so an independent hard list is necessary.
    log_difference = torch.cat((torch.linspace(1, 2, 12), torch.zeros(1), torch.linspace(.01, .2, 15))).expand(batch, -1)
    rollout = actor - log_difference
    return dict(logits=logits, packed_indices=indices, attention_mask=attention, responses=responses,
                actor_log_probs=actor, rollout_log_probs=rollout, support_ids=support,
                retained_mask=retained, perturbed_logits=torch.zeros_like(support, dtype=torch.float32),
                close_tag_token_id=29), query


def driver_inputs(inputs, captured):
    valid = inputs["attention_mask"][:, -inputs["responses"].shape[-1]:]
    return dict(rollout_log_probs=inputs["rollout_log_probs"], actor_log_probs=inputs["actor_log_probs"],
                response_mask=valid, responses=inputs["responses"], close_tag_token_id=29,
                comparison_mask=replay_integrity_mask(response_mask=valid, continuous_replay=True,
                    rollout_topk_ids=inputs["support_ids"], responses=inputs["responses"], close_tag_token_id=29),
                rollout_topk_ids=inputs["support_ids"], actor_replay_diagnostics=captured)


def test_capture_is_bounded_causal_detached_and_does_not_change_logits():
    inputs, query = fixture()
    before = inputs["logits"].detach().clone()
    captured = capture_replay_diagnostics(**inputs)
    positions = captured["actor_replay_positions"]
    assert positions.shape == (2, 16)
    assert positions[0, :8].tolist() == list(range(11, 3, -1))
    assert positions[0, 8:].tolist() == list(range(27, 19, -1))
    assert positions[1, 8:].tolist() == list(range(24, 16, -1))
    assert not positions.eq(12).any()  # first close action stays excluded
    assert captured["actor_replay_score_mask"].dtype == torch.bool
    assert captured["actor_replay_support_ids"].dtype == torch.int64
    for row in range(2):
        for slot, position in enumerate(positions[row].tolist()):
            ids = inputs["support_ids"][row, position]
            expected = before[query[row, position], ids]
            torch.testing.assert_close(captured["actor_replay_support_logits"][row, slot], expected, rtol=0, atol=0)
            if slot >= 8:
                expected_normalizer = before[query[row, position]].logsumexp(-1)
                torch.testing.assert_close(captured["actor_replay_categorical_log_normalizer"][row, slot], expected_normalizer)
            else:
                q = expected.masked_fill(~inputs["retained_mask"][row, position], -torch.inf).softmax(-1)
                torch.testing.assert_close(captured["actor_replay_support_probabilities"][row, slot], q)
                assert captured["actor_replay_score_mask"][row, slot].tolist() == ((q + 1e-6).log() > -3).tolist()
    assert all(not value.requires_grad and value.grad_fn is None for value in captured.values())
    assert max(value.numel() for value in captured.values()) == 2 * 16 * 3
    torch.testing.assert_close(inputs["logits"], before, rtol=0, atol=0)


def test_nonfinite_ties_are_stable_and_empty_segment_slots_are_sentinels():
    inputs, _ = fixture()
    inputs["actor_log_probs"][:, :12] = float("nan")
    inputs["attention_mask"][1, 4 + 12:] = 0
    captured = capture_replay_diagnostics(**inputs)
    assert captured["actor_replay_positions"][0, :8].tolist() == list(range(8))
    assert captured["actor_replay_positions"][1, 8:].tolist() == [-1] * 8
    assert not captured["actor_replay_score_mask"][1, 8:].any()
    result = build_replay_failure_diagnostics(**driver_inputs(inputs, captured))
    assert all(item["actor_replay"]["available"] for item in result["worst_positions"])
    json.dumps(result, allow_nan=False)


def test_driver_keeps_global_worst_and_separate_hard_evidence_after_row_permutation():
    inputs, _ = fixture()
    captured = capture_replay_diagnostics(**inputs)
    kwargs = driver_inputs(inputs, captured)
    order = torch.tensor([1, 0])
    kwargs = {key: value[order] if isinstance(value, torch.Tensor) else value for key, value in kwargs.items()}
    kwargs["actor_replay_diagnostics"] = {key: value[order] for key, value in captured.items()}
    kwargs["prompt_indices"] = [902, 901]
    result = build_replay_failure_diagnostics(**kwargs)
    assert len(result["worst_positions"]) == len(result["worst_hard_positions"]) == 8
    assert all(item["segment"] == "soft_prefix" for item in result["worst_positions"])
    assert all(item["segment"] == "hard_answer" for item in result["worst_hard_positions"])
    for item in result["worst_positions"] + result["worst_hard_positions"]:
        actual = item["actor_replay"]
        assert actual["available"] and actual["source"] == "existing_actor_replay_forward"
        assert actual["log_density"] == item["actor_log_density"]
        assert item["prompt_index"] == [902, 901][item["batch_row"]]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("field", ["actor_replay_support_ids", "actor_replay_log_density"])
def test_driver_rejects_misaligned_captured_identity_or_density(field):
    inputs, _ = fixture()
    captured = capture_replay_diagnostics(**inputs)
    captured[field] = captured[field] + 1
    with pytest.raises(ValueError, match="does not match|do not match"):
        build_replay_failure_diagnostics(**driver_inputs(inputs, captured))


def test_actual_actor_capture_uses_one_forward_and_preserves_default_result():
    calls = []
    model_calls = []
    def score(**kwargs):
        calls.append(1)
        return kwargs["logits"].float().log_softmax(-1).gather(-1, kwargs["labels"].unsqueeze(-1)).squeeze(-1)
    forward, actor, batch = _load_forward(score), _actor(), _batch()
    original_model = actor.actor_module
    class CountModel:
        def get_input_embeddings(self):
            return original_model.get_input_embeddings()
        def __call__(self, **kwargs):
            model_calls.append(1)
            return original_model(**kwargs)
    actor.actor_module = CountModel()
    default = forward(actor, batch, temperature=1.0)
    assert len(calls) == len(model_calls) == 1 and default[3] is None
    batch.update(_collect_replay_diagnostics=True, _replay_diagnostics_close_tag_id=6,
                 rollout_log_probs=torch.zeros_like(batch["responses"], dtype=torch.float32))
    collected = forward(actor, batch, temperature=1.0)
    assert len(calls) == len(model_calls) == 2  # one model/score pass per invocation
    torch.testing.assert_close(default[1], collected[1], rtol=0, atol=0)
    assert collected[3]["actor_replay_positions"].shape == (2, 16)
    assert collected[3]["actor_replay_positions"][0, :2].tolist() == [0, -1]
    assert collected[3]["actor_replay_positions"][0, 8:10].tolist() == [2, -1]


@pytest.mark.parametrize("setting", ["use_ulysses_sp", "use_fused_kernels"])
def test_capture_rejects_unsupported_execution_before_model_call(setting):
    forward, actor, batch = _load_forward(lambda **kwargs: pytest.fail("model must not run")), _actor(), _batch()
    setattr(actor, setting, True)
    batch["_collect_replay_diagnostics"] = True
    with pytest.raises(RuntimeError, match="capture requires"):
        forward(actor, batch, temperature=1.0)


class Batch(dict):
    @property
    def batch_size(self):
        return (len(next(iter(self.values()))),)

    def take(self, indices):
        return Batch({key: value[indices] for key, value in self.items()})

    def split(self, size):
        return [self.take(slice(start, start + size)) for start in range(0, self.batch_size[0], size)]


class Proto:
    def __init__(self, batch, meta_info):
        self.batch, self.meta_info, self.non_tensor_batch = batch, meta_info, {}

    def select(self, batch_keys):
        return Proto(Batch({key: self.batch[key] for key in batch_keys}), self.meta_info)


def load_compute():
    path = Path(__file__).resolve().parents[2] / "verl/workers/actor/dp_actor.py"
    cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "DataParallelPPOActor")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "compute_log_prob")
    method.decorator_list = []
    def rearrange(batch, max_token_len):
        return [batch.take([2, 0]), batch.take([1])], [[2, 0], [1]]
    namespace = dict(torch=torch, DataProto=Proto, itertools=itertools,
                     rearrange_micro_batches=rearrange, get_reverse_idx=lambda values: sorted(range(len(values)), key=values.__getitem__))
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace["compute_log_prob"]


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_compute_api_and_dynamic_permutation_restore_every_returned_row(capture, dynamic):
    source = Batch({key: torch.arange(3).reshape(3, 1) for key in (
        "responses", "input_ids", "attention_mask", "position_ids", "rollout_log_probs", "rollout_topk_ids",
        "rollout_topk_gumbels", "gumbel_temperature", "rollout_topk_retained_mask", "rollout_topk_probs")})
    meta = dict(micro_batch_size=2, temperature=1, use_dynamic_bsz=dynamic, max_token_len=100,
                add_noise_dirichlet=False, add_noise_gumbel_softmax=True, replay_diagnostics_close_tag_id=6)
    calls = []
    def forward(micro_batch, **kwargs):
        values = micro_batch["responses"].float()
        calls.append(values[:, 0].tolist())
        assert ("_collect_replay_diagnostics" in micro_batch) == capture
        return values + 10, values, None, {"actor_replay_positions": values.long()} if capture else None
    actor = SimpleNamespace(actor_module=SimpleNamespace(eval=lambda: None),
                            ulysses_sequence_parallel_size=1, _forward_micro_batch=forward)
    output = load_compute()(actor, Proto(source, meta), calculate_entropy=True, collect_replay_diagnostics=capture)
    assert len(output) == (3 if capture else 2)
    assert len(calls) == 2
    torch.testing.assert_close(output[0][:, 0], torch.arange(3).float())
    torch.testing.assert_close(output[1][:, 0], torch.arange(3).float() + 10)
    if capture:
        torch.testing.assert_close(output[2]["actor_replay_positions"][:, 0], torch.arange(3))


@pytest.mark.parametrize("capture", [False, True])
def test_real_worker_return_preserves_bounded_per_row_tensors_and_default_actor_api(capture):
    path = Path(__file__).resolve().parents[2] / "verl/workers/fsdp_workers.py"
    cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "compute_log_prob")
    method.decorator_list = []
    class WorkerProto:
        def __init__(self, tensors=None, meta_info=None):
            self.batch, self.meta_info = tensors or {}, meta_info or {}
        def to(self, device):
            return self
        @classmethod
        def from_dict(cls, tensors, meta_info):
            return cls(tensors, meta_info)
    namespace = dict(DataProto=WorkerProto, get_torch_device=lambda: SimpleNamespace(current_device=lambda: "cpu"))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    observed_calls = []
    positions = torch.tensor([[2] * 16, [3] * 16])
    def compute(**kwargs):
        observed_calls.append(kwargs)
        output = (torch.zeros(2, 4), torch.ones(2, 4))
        return (*output, {"actor_replay_positions": positions}) if kwargs.get("collect_replay_diagnostics") else output
    rollout = SimpleNamespace(log_prob_micro_batch_size_per_gpu=2, log_prob_max_token_len_per_gpu=100,
                              log_prob_use_dynamic_bsz=False, temperature=1,
                              add_noise_dirichlet=False, add_noise_gumbel_softmax=True, get=lambda key, default: default)
    class Sharding:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def preprocess_data(self, data):
            return data
        def postprocess_data(self, data):
            return data
    worker = SimpleNamespace(_is_actor=True, _is_offload_param=False, world_size=1,
                             actor=SimpleNamespace(actor_module=SimpleNamespace(), compute_log_prob=compute),
                             config=SimpleNamespace(rollout=rollout), ulysses_sharding_manager=Sharding())
    result = namespace["compute_log_prob"](worker, WorkerProto(meta_info={"collect_replay_diagnostics": capture}))
    assert len(observed_calls) == 1
    assert ("collect_replay_diagnostics" in observed_calls[0]) == capture
    assert ("actor_replay_positions" in result.batch) == capture
    if capture:
        torch.testing.assert_close(result.batch["actor_replay_positions"], positions, rtol=0, atol=0)
