"""Bounded observations from an existing actor replay forward; no extra model call."""

from __future__ import annotations

import torch

from .masks import categorical_suffix_mask, opd_loss_support_mask


REPLAY_DIAGNOSTIC_PREFIX = "actor_replay_"
REPLAY_RECORDS_PER_SEGMENT = 8


@torch.no_grad()
def capture_replay_diagnostics(
    *, logits: torch.Tensor, packed_indices: torch.Tensor,
    attention_mask: torch.Tensor, responses: torch.Tensor,
    actor_log_probs: torch.Tensor, rollout_log_probs: torch.Tensor,
    support_ids: torch.Tensor, retained_mask: torch.Tensor,
    perturbed_logits: torch.Tensor, close_tag_token_id: int,
) -> dict[str, torch.Tensor]:
    """Retain the worst eight soft and eight hard positions per response.

    Per-row storage makes ordinary DataProto slicing/concatenation sufficient
    for rank and microbatch alignment. This contains every global worst-eight
    position, with ties ordered by response position. Only selected support
    logits are gathered, never a response-by-vocabulary or selected-by-vocab
    copy. All tensors are detached observations of the current forward.
    """
    batch, length = responses.shape
    width = attention_mask.shape[-1]
    if any(value.shape != responses.shape for value in (actor_log_probs, rollout_log_probs)):
        raise ValueError("diagnostic densities must align with responses")
    if (support_ids.shape[:2] != responses.shape or support_ids.ndim != 3
            or not 2 <= support_ids.shape[-1] <= 8
            or retained_mask.shape != support_ids.shape or retained_mask.dtype != torch.bool
            or perturbed_logits.shape != support_ids.shape):
        raise ValueError("diagnostics require aligned bounded retained support")
    valid = attention_mask[:, -length:].bool()
    compared = opd_loss_support_mask(
        valid, support_ids, loss_support="all_response", responses=responses,
        close_tag_token_id=close_tag_token_id,
    )
    hard = categorical_suffix_mask(valid, support_ids, responses, close_tag_token_id)
    errors = ((actor_log_probs.float() - rollout_log_probs.float()).exp() - 1).abs()
    errors = torch.where(torch.isfinite(errors), errors, torch.inf)
    # Stable ordering matches the driver's row-major stable ranking, including
    # nonfinite values. Invalid slots sort after every valid error.
    position_parts = []
    for segment in (compared & ~hard, compared & hard):
        order = torch.argsort(errors.masked_fill(~segment, -1), dim=-1, descending=True, stable=True)
        selected = order[:, :REPLAY_RECORDS_PER_SEGMENT]
        selected = selected.masked_fill(~segment.gather(-1, selected), -1)
        if selected.shape[-1] < REPLAY_RECORDS_PER_SEGMENT:
            selected = torch.nn.functional.pad(selected, (0, REPLAY_RECORDS_PER_SEGMENT - selected.shape[-1]), value=-1)
        position_parts.append(selected)
    positions = torch.cat(position_parts, dim=-1)
    active = positions >= 0
    safe_positions = positions.clamp_min(0)
    rows = torch.arange(batch, device=responses.device).unsqueeze(-1).expand_as(positions)
    dense_query = rows * width + (width - length) + safe_positions - 1
    dense_to_packed = torch.full((batch * width,), -1, dtype=torch.long, device=logits.device)
    dense_to_packed[packed_indices] = torch.arange(packed_indices.numel(), device=logits.device)
    packed_query = dense_to_packed[dense_query]
    if bool(((packed_query < 0) & active).any()):
        raise ValueError("captured action has no valid preceding replay query")
    packed_query = packed_query.clamp_min(0)
    selected_ids = support_ids[rows, safe_positions]
    selected_retained = retained_mask[rows, safe_positions]
    selected_logits = logits.detach()[packed_query.unsqueeze(-1), selected_ids].float()
    probabilities = selected_logits.masked_fill(~selected_retained, -torch.inf).softmax(-1)
    support_log_probs = (probabilities + 1e-6).log()
    score_mask = support_log_probs > -3
    inferred_gumbels = (perturbed_logits[rows, safe_positions].float() - support_log_probs).clamp(-1.5, 3)
    selected_logit = logits.detach()[packed_query, responses[rows, safe_positions]].float()
    selected_density = actor_log_probs[rows, safe_positions].float()
    selected_hard = hard[rows, safe_positions] & active

    def bounded(value):
        mask = active if value.ndim == 2 else active.unsqueeze(-1)
        return torch.where(mask, value, torch.zeros_like(value)).detach()

    return {
        "actor_replay_positions": positions.detach(),
        "actor_replay_support_ids": bounded(selected_ids),
        "actor_replay_support_logits": bounded(selected_logits),
        "actor_replay_support_probabilities": bounded(probabilities),
        "actor_replay_support_log_probs": bounded(support_log_probs),
        "actor_replay_score_mask": bounded(score_mask),
        "actor_replay_inferred_gumbels": bounded(inferred_gumbels),
        "actor_replay_log_density": bounded(selected_density),
        "actor_replay_is_categorical": selected_hard.detach(),
        "actor_replay_categorical_selected_logit": torch.where(selected_hard, selected_logit, 0).detach(),
        # For categorical positions CE already supplies log p for this token;
        # logit - log p yields its full-vocabulary normalizer without another
        # full-vocabulary reduction. This is derived, not separately evaluated.
        "actor_replay_categorical_log_normalizer": torch.where(selected_hard, selected_logit - selected_density, 0).detach(),
    }
