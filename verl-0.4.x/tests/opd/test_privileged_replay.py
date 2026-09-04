from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import verl.opd.replay as replay_module
from verl.opd import OPDConfig, PrivilegedReplay, full_vocab_kl


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text == "<think>":
            return [1]
        if text == "</think>":
            return [2]
        # The synthetic native prompt contains its atomic opening tag.
        return [3, 1]

    def decode(self, token_ids, skip_special_tokens=False):
        assert skip_special_tokens is False
        if list(token_ids) == [1]:
            return "<think>"
        if list(token_ids) == [2]:
            return "</think>"
        return "prompt"

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert add_generation_prompt is True
        assert tokenize is False
        return "synthetic privileged prompt <think>"


class _RecordingCausalModel(nn.Module):
    def __init__(self, vocab_size=11, hidden_size=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.last_inputs = None
        self.last_query_indices = None

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        *,
        inputs_embeds,
        attention_mask,
        position_ids,
        use_cache,
        logits_to_keep,
    ):
        assert use_cache is False
        assert attention_mask.shape == inputs_embeds.shape[:2]
        assert position_ids.shape == inputs_embeds.shape[:2]
        self.last_inputs = inputs_embeds.detach().clone()
        self.last_query_indices = logits_to_keep.detach().clone()
        # A causal toy representation: hidden state t depends only on 0..t.
        hidden = inputs_embeds.cumsum(dim=1)
        return SimpleNamespace(logits=self.lm_head(hidden[:, logits_to_keep, :]))


def _replay(*, gate="all", loss_support="latent_only"):
    config = OPDConfig(trajectory_gate=gate, loss_support=loss_support)
    return PrivilegedReplay(_RecordingCausalModel(), _Tokenizer(), config)


def test_fsdp_embedding_lookup_summons_only_the_root_unit(monkeypatch):
    calls = []

    class _FakeFSDP(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(7, 3)

        def get_input_embeddings(self):
            return self.embedding

        @staticmethod
        @contextmanager
        def summon_full_params(module, **kwargs):
            calls.append((module, kwargs))
            yield

    monkeypatch.setattr(replay_module, "FSDP", _FakeFSDP)
    module = _FakeFSDP()
    values = replay_module._model_embeddings(
        module,
        torch.tensor([1, 2]),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert values.shape == (2, 3)
    assert calls == [
        (
            module,
            {"recurse": False, "writeback": False, "with_grads": False},
        )
    ]


def test_teacher_replay_uses_causal_query_positions_and_detaches_prefix():
    replay = _replay()
    response_embeddings = torch.randn(1, 3, 5, requires_grad=True)
    response_mask = torch.tensor([[True, True, True]])
    latent_mask = torch.tensor([[True, True, False]])

    logits, _ = replay.teacher_logits(
        response_embeddings=response_embeddings,
        response_mask=response_mask,
        latent_mask=latent_mask,
        extra_infos=[
            {
                "opd_original_user_content": "Question",
                "opd_gold_cot": "Reasoning",
                "opd_gold_answer": "4",
            }
        ],
    )

    # The synthetic teacher prompt has two tokens. Action zero is queried at
    # the last prompt position and action one after replaying response action 0.
    assert replay.teacher_module.last_query_indices.tolist() == [1, 2]
    assert logits.shape == (2, 11)
    assert logits.requires_grad is False
    assert replay.teacher_module.last_inputs.requires_grad is False
    assert response_embeddings.grad is None
    assert all(parameter.grad is None for parameter in replay.teacher_module.parameters())

    with torch.no_grad():
        expected_hidden = replay.teacher_module.last_inputs.cumsum(dim=1)
        expected = replay.teacher_module.lm_head(expected_hidden[:, [1, 2], :]).squeeze(0)
    torch.testing.assert_close(logits, expected)


def test_teacher_replay_fails_if_teacher_forward_reenables_autograd():
    class _LeakingTeacher(_RecordingCausalModel):
        def forward(self, **kwargs):
            with torch.enable_grad():
                return super().forward(**kwargs)

    replay = PrivilegedReplay(_LeakingTeacher(), _Tokenizer(), OPDConfig())
    with pytest.raises(RuntimeError, match="teacher logits retained an autograd graph"):
        replay.teacher_logits(
            response_embeddings=torch.randn(1, 2, 5),
            response_mask=torch.tensor([[True, True]]),
            latent_mask=torch.tensor([[True, False]]),
            extra_infos=[
                {
                    "opd_original_user_content": "Question",
                    "opd_gold_cot": "Reasoning",
                    "opd_gold_answer": "4",
                }
            ],
        )


@pytest.mark.parametrize(
    ("gate", "expected_selected"),
    [("all", 4), ("positive_advantage", 2)],
)
def test_replay_loss_supports_tokenwise_verl_advantages_and_detaches_teacher(
    gate,
    expected_selected,
):
    replay = _replay(gate=gate)
    torch.manual_seed(19)
    student_logits = torch.randn(4, 11, requires_grad=True)
    teacher_logits = torch.randn(4, 11)
    latent_mask = torch.tensor(
        [[True, True, False], [True, False, True]],
    )
    # VERL repeats each sequence-level GRPO advantage over response positions.
    advantages = torch.tensor(
        [[2.0, 2.0, 0.0], [-3.0, 0.0, -3.0]],
    )
    support_ids = torch.tensor(
        [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]],
    )

    result = replay.loss_from_teacher_logits(
        student_logits=student_logits,
        student_query_indices=torch.arange(4),
        teacher_logits=teacher_logits,
        teacher_seconds=0.25,
        latent_mask=latent_mask,
        advantages=advantages,
        latent_support_ids=support_ids,
        vocab_chunk_size=3,
    )

    per_slot = full_vocab_kl(student_logits, teacher_logits)
    expected_sum = per_slot.sum() if gate == "all" else per_slot[:2].sum()
    torch.testing.assert_close(result.kl_sum, expected_sum)
    assert result.denominator_slots == 4
    assert result.selected_slots == expected_selected
    assert result.teacher_seconds == 0.25

    result.kl_sum.backward()
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()
    if gate == "positive_advantage":
        assert torch.count_nonzero(student_logits.grad[:2]).item() > 0
        assert torch.count_nonzero(student_logits.grad[2:]).item() == 0
        assert torch.count_nonzero(result.opd_support_gradient[2:]).item() == 0


def test_positive_advantage_gate_rejects_misaligned_advantages():
    replay = _replay(gate="positive_advantage")
    with pytest.raises(ValueError, match="advantages must have shape"):
        replay.loss_from_teacher_logits(
            student_logits=torch.randn(2, 7, requires_grad=True),
            student_query_indices=torch.arange(2),
            teacher_logits=torch.randn(2, 7),
            teacher_seconds=0.0,
            latent_mask=torch.tensor([[True], [True]]),
            advantages=torch.randn(2, 2, 2),
            latent_support_ids=torch.tensor([[0, 1], [1, 2]]),
        )


@pytest.mark.parametrize(
    ("gate", "expected_selected"),
    [("all", 5), ("positive_advantage", 3)],
)
def test_all_response_replay_splits_latent_and_answer_kl_with_one_denominator(
    gate,
    expected_selected,
):
    replay = _replay(gate=gate, loss_support="all_response")
    torch.manual_seed(23)
    student_logits = torch.randn(5, 11, requires_grad=True)
    teacher_logits = torch.randn(5, 11)
    latent_mask = torch.tensor([[True, True, False], [True, False, False]])
    answer_mask = torch.tensor([[False, False, True], [False, True, False]])
    objective_mask = latent_mask | answer_mask
    advantages = torch.tensor([[2.0, 2.0, 2.0], [-1.0, -1.0, 0.0]])
    latent_support_ids = torch.tensor(
        [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
    )

    result = replay.loss_from_teacher_logits(
        student_logits=student_logits,
        student_query_indices=torch.arange(5),
        teacher_logits=teacher_logits,
        teacher_seconds=0.5,
        latent_mask=latent_mask,
        objective_mask=objective_mask,
        answer_mask=answer_mask,
        advantages=advantages,
        latent_support_ids=latent_support_ids,
    )

    per_slot = full_vocab_kl(student_logits, teacher_logits)
    gate_mask = torch.ones(5, dtype=torch.bool)
    if gate == "positive_advantage":
        gate_mask[3:] = False
    flat_latent = latent_mask[objective_mask]
    flat_answer = answer_mask[objective_mask]
    torch.testing.assert_close(result.kl_sum, per_slot[gate_mask].sum())
    torch.testing.assert_close(
        result.latent_kl_sum, per_slot[gate_mask & flat_latent].sum()
    )
    torch.testing.assert_close(
        result.answer_kl_sum, per_slot[gate_mask & flat_answer].sum()
    )
    assert result.denominator_slots == 5
    assert result.latent_slots == 3
    assert result.answer_slots == 2
    assert result.selected_slots == expected_selected
    assert result.opd_support_gradient.shape == (3, 3)
    assert result.student_support_logits.shape == (3, 3)


def test_replay_loss_rejects_attached_teacher_and_disconnected_student_logits():
    replay = _replay()
    common = {
        "student_query_indices": torch.arange(2),
        "teacher_seconds": 0.0,
        "latent_mask": torch.tensor([[True, True]]),
        "advantages": torch.ones(1, 2),
        "latent_support_ids": torch.tensor([[0, 1], [1, 2]]),
    }
    with pytest.raises(RuntimeError, match="teacher logits must be fully detached"):
        replay.loss_from_teacher_logits(
            student_logits=torch.randn(2, 7, requires_grad=True),
            teacher_logits=torch.randn(2, 7, requires_grad=True),
            **common,
        )
    with pytest.raises(RuntimeError, match="student logits are disconnected"):
        replay.loss_from_teacher_logits(
            student_logits=torch.randn(2, 7),
            teacher_logits=torch.randn(2, 7),
            **common,
        )
