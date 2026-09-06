"""Execute actor replay packing to verify causal alignment of recorded filtering."""

import __future__
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_forward(score):
    path = Path(__file__).resolve().parents[2] / "verl/workers/actor/dp_actor.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DataParallelPPOActor")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_forward_micro_batch")

    def unpad(values, mask):
        indices = mask.flatten().nonzero().flatten()
        lengths = mask.sum(-1).to(torch.int32)
        cumulative = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0, dtype=torch.int32)))
        return values.flatten(0, 1)[indices], indices, cumulative, int(lengths.max())

    def pad(hidden_states, indices, batch, seqlen):
        result = torch.zeros(batch * seqlen, hidden_states.shape[-1], dtype=hidden_states.dtype)
        result[indices] = hidden_states
        return result.reshape(batch, seqlen, -1)

    namespace = {
        "torch": torch, "unpad_input": unpad, "pad_input": pad,
        "rearrange": lambda values, pattern: values.flatten(0, 1),
        "index_first_axis": lambda values, indices: values[indices],
        "FSDP": SimpleNamespace(summon_full_params=lambda *args, **kwargs: nullcontext()),
        "logprobs_from_logits_topk_gumbel": score,
    }
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace["_forward_micro_batch"]


def _batch():
    ids = torch.tensor([[0, 3, 4, 5, 6, 2], [7, 8, 9, 10, 2, 0]])
    support = torch.zeros(2, 6, 5, dtype=torch.long)
    support[..., 0] = ids
    support[:, 3] = torch.tensor([[5, 7, 8, 9, 11], [10, 3, 5, 6, 11]])
    retained = torch.zeros_like(support, dtype=torch.bool)
    retained[..., 0] = True
    retained[0, 3] = torch.tensor([True, False, True, True, False])
    retained[1, 3] = torch.tensor([True, True, False, False, False])
    return {
        "input_ids": ids, "responses": ids[:, 3:], "rollout_topk_ids": support,
        "rollout_topk_gumbels": torch.zeros_like(support, dtype=torch.float32),
        "rollout_topk_retained_mask": retained,
        "rollout_topk_probs": retained.float() * 0.33333331,
        "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]]),
        "position_ids": torch.tensor([[0, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 0]]),
        "gumbel_temperature": torch.tensor([0.1, 0.1]),
    }


class Model:
    def __init__(self):
        self.embed = torch.nn.Embedding(16, 4)

    def get_input_embeddings(self):
        return self.embed

    def __call__(self, *, inputs_embeds, **kwargs):
        assert not inputs_embeds.requires_grad  # Previous sampled actions are detached.
        self.last_inputs = inputs_embeds
        self.last_kwargs = kwargs
        return SimpleNamespace(logits=torch.zeros(*inputs_embeds.shape[:-1], 16))


def _actor():
    return SimpleNamespace(
        opd_replay=None, use_remove_padding=True, use_ulysses_sp=False, use_fused_kernels=False,
        device_name="cpu", actor_module=Model(),
        opd_config=SimpleNamespace(prompt_profile="qwen3-training-benchmark-v1"),
    )


def test_filter_mask_tracks_response_targets_through_left_and_right_padding():
    observed = []

    def score(*, logits, rollout_topk_ids, rollout_topk_gumbels, labels,
              inplace_backward, rollout_topk_retained_mask):
        observed.append(rollout_topk_retained_mask)
        assert rollout_topk_retained_mask.shape == rollout_topk_ids.shape
        # Encode distinct Boolean patterns so a one-token or cross-row shift
        # cannot accidentally pass this causal target-alignment check.
        return (rollout_topk_retained_mask.long() * torch.tensor([1, 2, 4, 8, 16])).sum(-1).float()

    forward = _load_forward(score)
    actor, batch = _actor(), _batch()
    _, values, _, _ = forward(actor, batch, temperature=1.0)
    assert len(observed) == 1
    torch.testing.assert_close(values[0], torch.tensor([13.0, 1.0, 1.0]))
    torch.testing.assert_close(values[1, :2], torch.tensor([3.0, 1.0]))
    weights = batch["rollout_topk_probs"][batch["attention_mask"].bool()]
    weights = weights / weights.sum(-1, keepdim=True)
    table = actor.actor_module.embed.weight.detach().to(torch.bfloat16)
    supports = batch["rollout_topk_ids"][batch["attention_mask"].bool()]
    native_prefix = torch.sum(table[supports] * weights.unsqueeze(-1), dim=1, dtype=table.dtype)
    torch.testing.assert_close(actor.actor_module.last_inputs[0], native_prefix, rtol=0, atol=0)


def test_qwen_replay_rejects_missing_filter_mask_before_model_forward():
    forward = _load_forward(lambda **kwargs: pytest.fail("density evaluation must not run"))
    batch = _batch()
    batch.pop("rollout_topk_retained_mask")
    with pytest.raises(RuntimeError, match="requires the recorded retained support mask"):
        forward(_actor(), batch, temperature=1.0)


def test_native_replay_passes_real_row_boundaries_without_changing_action_order():
    forward = _load_forward(lambda **kwargs: torch.zeros(kwargs["labels"].shape))
    actor = _actor()
    actor.qwen_replay_backend = "native_fa3_v1"
    batch = _batch()
    forward(actor, batch, temperature=1.0)
    metadata = actor.actor_module.last_kwargs
    assert torch.equal(metadata["opd_cu_seqlens"], torch.tensor([0, 5, 10], dtype=torch.int32))
    assert metadata["opd_max_seqlen"] == 5
    assert torch.equal(metadata["position_ids"], torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]]))
