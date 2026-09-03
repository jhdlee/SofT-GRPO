from types import SimpleNamespace

import pytest
import torch
from torch import nn

from opd_tools.icl_replay import (
    ActorAgreementTolerances,
    AtomicReplayStore,
    actor_agreement_gate,
    compare_sglang_hf_actor,
    full_vocab_pair_metrics,
    reconstruct_latent_embeddings,
    replay_chunk_metrics,
    replay_trajectory,
    replay_trajectory_many,
)
from opd_tools.icl_runtime import SamplingSettings, TrajectoryMetadata


class _ToyBackbone(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding
        self.query_lengths = []

    def forward(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        attention_mask,
        position_ids,
        cache_position,
        past_key_values=None,
        use_cache,
        return_dict,
    ):
        assert use_cache and return_dict
        values = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        self.query_lengths.append(values.shape[1])
        previous = (
            torch.zeros((1, 1, values.shape[-1]), dtype=values.dtype)
            if past_key_values is None
            else past_key_values
        )
        hidden = values.cumsum(dim=1) + previous
        return SimpleNamespace(
            last_hidden_state=hidden,
            past_key_values=hidden[:, -1:, :].detach(),
        )


class _RecordingHead(nn.Linear):
    def __init__(self, hidden, vocab):
        super().__init__(hidden, vocab, bias=False)
        self.rows = []

    def forward(self, value):
        self.rows.append(value.shape[0])
        return super().forward(value)


class _ToyModel(nn.Module):
    def __init__(self, vocab=17, hidden=7):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.model = _ToyBackbone(self.embedding)
        self.lm_head = _RecordingHead(hidden, vocab)
        self.config = SimpleNamespace(max_position_embeddings=128)

    def get_input_embeddings(self):
        return self.embedding


class _Tokenizer:
    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is True
        return [1, 2] if text == "no demo" else [3, 4, 5]


def _trajectory():
    return TrajectoryMetadata(
        response_token_ids=(2, 7, 9, 11),
        latent_support_ids=(
            (2, 3, 4, 5, 6),
            (7, 8, 9, 10, 11),
            (9, 10, 11, 12, 13),
        ),
        latent_perturbed_logits=(
            (3.0, 2.0, 1.0, 0.0, -1.0),
            (2.0, 1.0, 0.0, -1.0, -2.0),
            (4.0, 2.0, 0.0, -1.0, -2.0),
        ),
        latent_gumbel_noise=(
            (1.0, 0.0, -0.5, 0.5, -1.0),
            (0.5, -0.5, 0.0, 1.0, -1.0),
            (1.5, 0.5, -0.5, 0.0, -1.0),
        ),
    )


def _zero_latent_trajectory():
    return TrajectoryMetadata(
        response_token_ids=(2, 11),
        latent_support_ids=(),
        latent_perturbed_logits=(),
        latent_gumbel_noise=(),
    )


def test_reconstruct_latent_embeddings_matches_manual_mixture_and_detaches():
    torch.manual_seed(7)
    model = _ToyModel()
    trajectory = _trajectory()
    actual = reconstruct_latent_embeddings(
        model, trajectory, gumbel_temperature=0.1
    )
    supports = torch.tensor(trajectory.latent_support_ids)
    logits = torch.tensor(trajectory.latent_perturbed_logits)
    expected = (
        torch.softmax(logits / 0.1, dim=-1).unsqueeze(-1)
        * model.embedding(supports)
    ).sum(dim=1)
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad


def test_full_vocab_kl_direction_and_chunking_are_exact_fp32():
    torch.manual_seed(8)
    p = torch.randn(4, 13, dtype=torch.bfloat16)
    q = torch.randn(4, 13, dtype=torch.bfloat16)
    actual = full_vocab_pair_metrics(p, q, vocab_chunk_size=3)
    p_log = torch.log_softmax(p.float(), dim=-1)
    q_log = torch.log_softmax(q.float(), dim=-1)
    expected_forward = (q_log.exp() * (q_log - p_log)).sum(dim=-1)
    expected_reverse = (p_log.exp() * (p_log - q_log)).sum(dim=-1)
    torch.testing.assert_close(actual["forward_kl"], expected_forward)
    torch.testing.assert_close(actual["reverse_kl"], expected_reverse)
    assert actual["forward_kl"].dtype == torch.float32


def _matching_sglang_metadata(logits, settings):
    probabilities = torch.softmax(logits.float() / settings.temperature, dim=-1)
    values, support = torch.topk(probabilities, settings.top_k, dim=-1)
    values /= values.sum(dim=-1, keepdim=True)
    ascending, order = torch.sort(values, dim=-1)
    keep_ascending = torch.cumsum(ascending, dim=-1) >= 1.0 - settings.top_p
    keep = torch.zeros_like(keep_ascending)
    keep.scatter_(1, order, keep_ascending)
    values = torch.where(keep, values, torch.zeros_like(values))
    values /= values.sum(dim=-1, keepdim=True)
    noise = torch.tensor(
        [[0.4, -0.2, 0.8, -0.5, 0.1], [0.2, 0.7, -0.3, 0.5, -0.4]],
        dtype=torch.float32,
    )
    perturbed = torch.log(values + 1e-6) + noise
    _, permutation = torch.sort(
        torch.softmax(perturbed / settings.gumbel_temperature, dim=-1),
        dim=-1,
        descending=True,
    )
    return (
        torch.gather(support, 1, permutation),
        torch.gather(perturbed, 1, permutation),
        torch.gather(noise, 1, permutation),
    )


def test_sglang_hf_actor_agreement_is_exact_for_matching_no_demo_logits():
    settings = SamplingSettings()
    logits = torch.tensor(
        [
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.5, -1.0, -2.0],
            [0.5, 1.5, 4.5, 2.5, 3.5, -0.5, -1.5, -2.5],
        ],
        dtype=torch.bfloat16,
    )
    support, perturbed, noise = _matching_sglang_metadata(logits, settings)
    observation = compare_sglang_hf_actor(
        logits,
        support_ids=support,
        perturbed_logits=perturbed,
        gumbel_noise=noise,
        settings=settings,
    )
    assert observation["active_support_exact_slots"] == 2
    assert observation["centered_logprob_abs_error_max"] < 1e-6


def test_sglang_hf_actor_agreement_detects_support_and_logit_drift():
    settings = SamplingSettings()
    logits = torch.tensor(
        [
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.5, -1.0, -2.0],
            [0.5, 1.5, 4.5, 2.5, 3.5, -0.5, -1.5, -2.5],
        ],
        dtype=torch.float32,
    )
    support, perturbed, noise = _matching_sglang_metadata(logits, settings)
    support[0, 0] = 7
    perturbed[1, 0] += 1.0
    observation = compare_sglang_hf_actor(
        logits,
        support_ids=support,
        perturbed_logits=perturbed,
        gumbel_noise=noise,
        settings=settings,
    )
    assert observation["active_support_exact_slots"] < 2
    assert observation["centered_logprob_abs_error_max"] > 0.25


def test_kv_replay_scores_exact_causal_latent_positions_and_no_gradients():
    torch.manual_seed(9)
    model = _ToyModel()
    record = replay_trajectory(
        model=model,
        tokenizer=_Tokenizer(),
        trajectory=_trajectory(),
        no_demo_prompt="no demo",
        prompted_prompt="with demonstration",
        model_label="starting",
        benchmark="math500",
        example_id="math500-0",
        sample_index=0,
        prompted_condition="sdft_matched",
        settings=SamplingSettings(),
        hidden_chunk_size=2,
        vocab_chunk_size=5,
    )
    assert record.latent_token_count == 3
    assert record.forward_kl_mean > 0
    assert record.reverse_kl_mean > 0
    # Each policy: prompt forward, then only the first two actions. The final
    # action has no successor query and is never fed into the backbone.
    assert model.model.query_lengths == [2, 3, 2, 2]
    # LM-head chunks contain only selected queries, never full prompt states.
    assert max(model.lm_head.rows) <= 2
    assert sum(model.lm_head.rows) == 6
    assert all(parameter.grad is None for parameter in model.parameters())


def test_identical_context_has_zero_kl():
    torch.manual_seed(10)
    record = replay_trajectory(
        model=_ToyModel(),
        tokenizer=_Tokenizer(),
        trajectory=_trajectory(),
        no_demo_prompt="no demo",
        prompted_prompt="no demo",
        model_label="starting",
        benchmark="math500",
        example_id="math500-0",
        sample_index=0,
        prompted_condition="no_demo",
    )
    assert record.forward_kl_sum == pytest.approx(0.0, abs=1e-7)
    assert record.reverse_kl_sum == pytest.approx(0.0, abs=1e-7)


def test_many_context_replay_shares_no_demo_backbone_and_preserves_zero_control():
    torch.manual_seed(12)
    model = _ToyModel()
    records = replay_trajectory_many(
        model=model,
        tokenizer=_Tokenizer(),
        trajectory=_trajectory(),
        no_demo_prompt="no demo",
        prompted_prompts={
            "no_demo": "no demo",
            "sdft_matched": "with demonstration",
            "sdpg_matched": "another demonstration",
        },
        model_label="starting",
        benchmark="math500",
        example_id="math500-0",
        sample_index=0,
        hidden_chunk_size=2,
    )
    assert [record.prompted_condition for record in records] == [
        "no_demo",
        "sdft_matched",
        "sdpg_matched",
    ]
    assert records[0].forward_kl_sum == pytest.approx(0.0, abs=1e-7)
    # One no-demo prompt and two prompted contexts, then one action chunk each.
    assert model.model.query_lengths == [2, 3, 3, 2, 2, 2]
    # For each latent-query chunk the no-demo LM-head result is shared.
    assert len(model.lm_head.rows) == 6


def test_zero_latent_source_is_explicitly_excluded_for_all_five_contexts(tmp_path):
    model = _ToyModel()
    prompts = {
        "no_demo": "no demo",
        "sdft_matched": "with demonstration",
        "sdft_shuffled": "another demonstration",
        "sdpg_matched": "with demonstration",
        "sdpg_shuffled": "another demonstration",
    }
    records = replay_trajectory_many(
        model=model,
        tokenizer=_Tokenizer(),
        trajectory=_zero_latent_trajectory(),
        no_demo_prompt="no demo",
        prompted_prompts=prompts,
        model_label="starting",
        benchmark="math500",
        example_id="math500-1",
        sample_index=0,
    )
    assert [record.prompted_condition for record in records] == list(prompts)
    assert all(record.replay_exclusion_reason == "zero_latent_slots" for record in records)
    assert all(record.forward_kl_mean is None for record in records)
    assert model.model.query_lengths == []
    metrics = replay_chunk_metrics(records)
    assert metrics["replay/source_trajectory_count"] == 1
    assert metrics["replay/source_valid_trajectory_count"] == 0
    assert metrics["replay/source_zero_latent_excluded_count"] == 1
    assert "replay/forward_kl_slot_mean" not in metrics
    store = AtomicReplayStore(tmp_path)
    store.commit("starting/math500/zero", records, identity={"source": "zero"})
    assert store.load("starting/math500/zero") == records
    with pytest.raises(ValueError, match="valid replay records only"):
        actor_agreement_gate([records[0]])


def test_replay_metrics_exclude_zero_latent_sources_without_zero_imputation():
    prompts = {
        "no_demo": "no demo",
        "sdft_matched": "with demonstration",
        "sdft_shuffled": "another demonstration",
        "sdpg_matched": "with demonstration",
        "sdpg_shuffled": "another demonstration",
    }
    valid = replay_trajectory_many(
        model=_ToyModel(),
        tokenizer=_Tokenizer(),
        trajectory=_trajectory(),
        no_demo_prompt="no demo",
        prompted_prompts=prompts,
        model_label="starting",
        benchmark="math500",
        example_id="math500-0",
        sample_index=0,
    )
    excluded = replay_trajectory_many(
        model=_ToyModel(),
        tokenizer=_Tokenizer(),
        trajectory=_zero_latent_trajectory(),
        no_demo_prompt="no demo",
        prompted_prompts=prompts,
        model_label="starting",
        benchmark="math500",
        example_id="math500-1",
        sample_index=0,
    )
    valid_metrics = replay_chunk_metrics(valid)
    mixed_metrics = replay_chunk_metrics([*valid, *excluded])
    assert mixed_metrics["replay/source_trajectory_count"] == 2
    assert mixed_metrics["replay/source_valid_trajectory_count"] == 1
    assert mixed_metrics["replay/source_excluded_trajectory_count"] == 1
    assert mixed_metrics["replay/forward_kl_slot_mean"] == pytest.approx(
        valid_metrics["replay/forward_kl_slot_mean"]
    )
    assert mixed_metrics["replay/forward_kl_sequence_mean"] == pytest.approx(
        valid_metrics["replay/forward_kl_sequence_mean"]
    )


def test_replay_store_resumes_and_authenticates(tmp_path):
    torch.manual_seed(11)
    record = replay_trajectory(
        model=_ToyModel(),
        tokenizer=_Tokenizer(),
        trajectory=_trajectory(),
        no_demo_prompt="no demo",
        prompted_prompt="with demonstration",
        model_label="starting",
        benchmark="math500",
        example_id="math500-0",
        sample_index=0,
        prompted_condition="sdpg_matched",
    )
    store = AtomicReplayStore(tmp_path)
    identity = {"source": "chunk-sha", "condition": "sdpg_matched"}
    first = store.commit("starting/math500/chunk0", [record], identity=identity)
    assert store.commit("starting/math500/chunk0", [record], identity=identity) == first
    assert store.load("starting/math500/chunk0") == [record]
    assert replay_chunk_metrics([record])["replay/latent_slots"] == 3
    data, _ = store.paths("starting/math500/chunk0")
    data.write_bytes(data.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="authentication"):
        store.verify("starting/math500/chunk0")


def test_actor_agreement_gate_declares_and_applies_smoke_tolerances():
    torch.manual_seed(13)
    record = replay_trajectory(
        model=_ToyModel(),
        tokenizer=_Tokenizer(),
        trajectory=_trajectory(),
        no_demo_prompt="no demo",
        prompted_prompt="no demo",
        model_label="starting",
        benchmark="math500",
        example_id="math500-0",
        sample_index=0,
        prompted_condition="no_demo",
    )
    gate = actor_agreement_gate(
        [record],
        tolerances=ActorAgreementTolerances(
            min_active_support_exact_rate=0.0,
            max_centered_logprob_mae=100.0,
            max_centered_logprob_abs_error=100.0,
        ),
    )
    assert gate["valid"]
    assert gate["tolerances"]["min_active_support_exact_rate"] == 0.0
    assert not actor_agreement_gate([record])["valid"]
    with pytest.raises(ValueError, match="duplicate"):
        actor_agreement_gate([record, record])
