"""Latent-slot masks, trajectory gating, and loss reductions."""

from __future__ import annotations

from typing import Optional, Union

import torch

from .config import LossSupport, TrajectoryGate


def latent_mask_from_topk_support(response_mask: torch.Tensor, rollout_topk_ids: torch.Tensor) -> torch.Tensor:
    """Identify released SofT-GRPO continuous-action positions.

    The released rollout represents a categorical position with support
    ``[hard_token_id, 0, 0, ...]``.  Any valid response position that does not
    have this sentinel layout is a continuous latent position.
    """

    if rollout_topk_ids.ndim != response_mask.ndim + 1:
        raise ValueError("rollout_topk_ids must have one support dimension beyond response_mask")
    if rollout_topk_ids.shape[:-1] != response_mask.shape:
        raise ValueError("response_mask and rollout_topk_ids leading dimensions must match")
    if rollout_topk_ids.shape[-1] < 2:
        raise ValueError("rollout_topk_ids must contain at least two support IDs")

    valid_response = response_mask.to(dtype=torch.bool)
    categorical_sentinel = (rollout_topk_ids[..., 1:] == 0).all(dim=-1)
    return valid_response & ~categorical_sentinel


def _validate_boundary_inputs(
    response_mask: torch.Tensor,
    rollout_topk_ids: torch.Tensor,
    responses: torch.Tensor,
) -> None:
    if responses.shape != response_mask.shape:
        raise ValueError("responses and response_mask must have identical shapes")
    # Reuse the released-support validation, including the support-width check.
    latent_mask_from_topk_support(response_mask, rollout_topk_ids)


def categorical_suffix_mask(
    response_mask: torch.Tensor,
    rollout_topk_ids: torch.Tensor,
    responses: torch.Tensor,
    close_tag_token_id: int,
) -> torch.Tensor:
    """Return sampled categorical actions strictly after the first close tag.

    Masks in this module are response-*action* aligned: position ``t`` selects
    the causal logits that predict response action ``t``.  Consequently the
    categorical ``</think>`` transition action itself is excluded, while EOS
    and other valid sampled suffix actions after it remain eligible.

    Released SofT-GRPO encodes categorical actions with support
    ``[token_id, 0, 0, ...]``.  A malformed continuous action after the close
    tag is deliberately excluded rather than being relabeled categorical.
    """

    _validate_boundary_inputs(response_mask, rollout_topk_ids, responses)
    if isinstance(close_tag_token_id, bool) or not isinstance(close_tag_token_id, int):
        raise TypeError("close_tag_token_id must be an integer")

    valid = response_mask.to(dtype=torch.bool)
    close = valid & responses.eq(close_tag_token_id)
    # Shift the inclusive prefix state right by one so the close action is not
    # part of the answer suffix.  Multiple close tokens remain suffix actions
    # once the first transition has occurred.
    seen_close_inclusive = close.to(dtype=torch.int64).cumsum(dim=-1) > 0
    seen_close_before = torch.zeros_like(seen_close_inclusive)
    if seen_close_before.shape[-1] > 1:
        seen_close_before[..., 1:] = seen_close_inclusive[..., :-1]

    categorical = (rollout_topk_ids[..., 1:] == 0).all(dim=-1)
    return valid & categorical & seen_close_before


def opd_loss_support_mask(
    response_mask: torch.Tensor,
    rollout_topk_ids: torch.Tensor,
    *,
    loss_support: Union[LossSupport, str] = LossSupport.LATENT_ONLY,
    responses: Optional[torch.Tensor] = None,
    close_tag_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Build the response-action mask for the configured OPD objective.

    ``latent_only`` intentionally delegates to the original released-support
    detector and therefore preserves historical behavior exactly.

    ``all_response`` includes valid continuous positions strictly before the
    first atomic ``</think>`` and valid categorical positions strictly after
    it.  It excludes the close transition itself, padding, and malformed mode
    positions.  If a capped trajectory never closes, all of its valid
    continuous positions remain selected and it contributes no answer slots.
    """

    try:
        support_type = LossSupport(loss_support)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown OPD loss support: {loss_support!r}") from exc

    latent = latent_mask_from_topk_support(response_mask, rollout_topk_ids)
    if support_type is LossSupport.LATENT_ONLY:
        return latent
    if responses is None or close_tag_token_id is None:
        raise ValueError(
            "responses and close_tag_token_id are required for all_response support"
        )

    _validate_boundary_inputs(response_mask, rollout_topk_ids, responses)
    valid = response_mask.to(dtype=torch.bool)
    close = valid & responses.eq(close_tag_token_id)
    seen_close_inclusive = close.to(dtype=torch.int64).cumsum(dim=-1) > 0
    pre_close_latent = latent & ~seen_close_inclusive
    answer = categorical_suffix_mask(
        response_mask,
        rollout_topk_ids,
        responses,
        close_tag_token_id,
    )
    return pre_close_latent | answer


def trajectory_gate_mask(
    trajectory_advantages: torch.Tensor,
    gate: Union[TrajectoryGate, str] = TrajectoryGate.ALL,
) -> torch.Tensor:
    """Return a one-dimensional boolean keep mask for trajectories."""

    if trajectory_advantages.ndim != 1:
        raise ValueError("trajectory_advantages must be one-dimensional")
    try:
        gate_type = TrajectoryGate(gate)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown trajectory gate: {gate!r}") from exc
    if gate_type is TrajectoryGate.ALL:
        return torch.ones_like(trajectory_advantages, dtype=torch.bool)
    return trajectory_advantages > 0


def gated_latent_mask(
    latent_mask: torch.Tensor,
    trajectory_advantages: Optional[torch.Tensor] = None,
    gate: Union[TrajectoryGate, str] = TrajectoryGate.ALL,
) -> torch.Tensor:
    """Apply a trajectory-level gate to a ``[batch, ...]`` latent mask."""

    if latent_mask.ndim < 1:
        raise ValueError("latent_mask must have a batch dimension")
    gate_type = TrajectoryGate(gate)
    if gate_type is TrajectoryGate.ALL:
        return latent_mask.to(dtype=torch.bool)
    if trajectory_advantages is None:
        raise ValueError("trajectory_advantages are required for positive_advantage gating")
    if trajectory_advantages.shape != (latent_mask.shape[0],):
        raise ValueError("trajectory_advantages must have shape [batch]")
    keep = trajectory_gate_mask(trajectory_advantages, gate_type)
    keep_shape = (keep.shape[0],) + (1,) * (latent_mask.ndim - 1)
    return latent_mask.to(dtype=torch.bool) & keep.reshape(keep_shape)


def latent_kl_sum_and_count(
    token_kl: torch.Tensor,
    latent_mask: torch.Tensor,
    trajectory_advantages: Optional[torch.Tensor] = None,
    gate: Union[TrajectoryGate, str] = TrajectoryGate.ALL,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return numerator, normalization count, and active-slot count.

    The denominator is always the number of *all* valid latent slots.  Thus the
    optional positive-advantage gate reduces the dose rather than renormalizing
    over only selected trajectories, matching the intended gated objective.
    """

    if token_kl.shape != latent_mask.shape:
        raise ValueError("token_kl and latent_mask must have identical shapes")
    if not token_kl.is_floating_point():
        raise TypeError("token_kl must be floating point")

    base_mask = latent_mask.to(dtype=torch.bool)
    selected_mask = gated_latent_mask(base_mask, trajectory_advantages, gate)
    numerator = token_kl.masked_fill(~selected_mask, 0.0).sum()
    denominator = base_mask.sum()
    active_count = selected_mask.sum()
    return numerator, denominator, active_count


def safe_mean_from_sum(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Divide scalar tensors without NaNs, returning a differentiable zero."""

    if numerator.numel() != 1 or denominator.numel() != 1:
        raise ValueError("numerator and denominator must be scalar tensors")
    denominator_float = denominator.to(device=numerator.device, dtype=numerator.dtype)
    positive = (denominator_float > 0).to(dtype=numerator.dtype)
    return numerator / denominator_float.clamp_min(1.0) * positive


def ddp_scaled_local_loss(
    local_numerator: torch.Tensor,
    global_denominator: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    """Scale a local numerator for DDP's gradient averaging.

    ``global_denominator`` must already have been all-reduced.  Multiplication by
    ``world_size`` ensures the subsequent average of rank-local gradients equals
    the gradient of the global slot-weighted mean.
    """

    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    return safe_mean_from_sum(local_numerator * world_size, global_denominator)
