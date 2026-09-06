"""Exercise support-only diagnostics and the standalone density graph boundary."""

from types import SimpleNamespace

import pytest
import torch

from verl.opd import categorical_suffix_mask, latent_mask_from_topk_support, opd_loss_support_mask
from verl.opd.config import LossSupport, ObjectiveMode
from test_actor_retained_support import Model, _actor, _batch, _load_forward


class DifferentiableModel(Model):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(16, dtype=torch.float32) / 19)

    def __call__(self, *, inputs_embeds, **kwargs):
        assert not inputs_embeds.requires_grad
        self.last_logits = self.weight.expand(*inputs_embeds.shape[:-1], 16).clone()
        return SimpleNamespace(logits=self.last_logits)


@pytest.mark.parametrize("mode,compute_opd", [
    (ObjectiveMode.STANDALONE, True),
    (ObjectiveMode.AUXILIARY, True),
    (ObjectiveMode.AUXILIARY, False),
])
def test_update_keeps_causal_support_values_without_vocab_copy_and_only_required_graph(mode, compute_opd, monkeypatch):
    observed = []

    def score(*, logits, **kwargs):
        observed.append(logits.requires_grad)
        # A differentiable scoring stand-in makes retaining the unused
        # standalone policy graph observable without a GPU CE kernel.
        return logits.square().mean(-1)

    forward = _load_forward(score)

    def pad(hidden_states, indices, batch, seqlen):
        result = hidden_states.new_zeros(batch * seqlen, *hidden_states.shape[1:])
        result[indices] = hidden_states
        return result.reshape(batch, seqlen, *hidden_states.shape[1:])

    forward.__globals__.update(
        pad_input=pad,
        ObjectiveMode=ObjectiveMode,
        latent_mask_from_topk_support=latent_mask_from_topk_support,
        opd_loss_support_mask=opd_loss_support_mask,
        categorical_suffix_mask=categorical_suffix_mask,
    )
    actor, batch = _actor(), _batch()
    actor.actor_module = DifferentiableModel()
    actor.opd_config.mode = mode
    actor.opd_config.loss_support = LossSupport.ALL_RESPONSE
    actor.config = {}
    batch["extra_info"] = [{"row": 0}, {"row": 1}]

    def teacher_logits(*, response_embeddings, latent_mask, **kwargs):
        assert not response_embeddings.requires_grad
        return torch.zeros(int(latent_mask.sum()), 16), 0.0

    def loss_from_teacher_logits(*, student_logits, **kwargs):
        # Loss and causal-mask equivalence are covered by privileged replay
        # tests; this fixture isolates the surrounding actor memory contract.
        return SimpleNamespace(kl_sum=student_logits.square().sum())

    actor.opd_replay = SimpleNamespace(
        think_end_id=6, teacher_logits=teacher_logits,
        loss_from_teacher_logits=loss_from_teacher_logits,
    )
    original_select = torch.Tensor.index_select

    def forbid_vocabulary_row_copy(tensor, dim, index):
        if tensor.ndim == 2 and tensor.shape[-1] == 16 and dim == 0:
            pytest.fail("support diagnostics copied complete vocabulary rows")
        return original_select(tensor, dim, index)

    monkeypatch.setattr(torch.Tensor, "index_select", forbid_vocabulary_row_copy)
    _, log_probs, result, diagnostic = forward(
        actor, batch, temperature=1.0,
        compute_opd=compute_opd, collect_gradient_info=True,
    )
    needs_density_grad = mode is not ObjectiveMode.STANDALONE
    assert observed == [needs_density_grad]
    assert log_probs.requires_grad == needs_density_grad
    assert diagnostic["support_logits"].shape == (2, 5)
    assert not diagnostic["support_logits"].requires_grad
    # Left-padded first prompt has two valid tokens; the second has three.
    # Their causal queries are packed positions 1 and 7 respectively.
    logits = actor.actor_module.last_logits.squeeze(0).detach()
    support_ids = batch["rollout_topk_ids"][:, 3]
    expected = torch.stack([logits[1, support_ids[0]], logits[7, support_ids[1]]])
    torch.testing.assert_close(diagnostic["support_logits"], expected, rtol=0, atol=0)
    expected_density = logits.square().mean(-1)[1:4]
    torch.testing.assert_close(log_probs[0], expected_density, rtol=0, atol=0)
    loss = result.kl_sum if compute_opd else log_probs.sum()
    loss.backward()
    assert actor.actor_module.weight.grad is not None
    assert torch.isfinite(actor.actor_module.weight.grad).all()
    assert actor.actor_module.weight.grad.abs().sum() > 0
