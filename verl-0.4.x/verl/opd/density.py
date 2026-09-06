"""Differentiable replay of the released fixed-support Gumbel score."""

from __future__ import annotations

import torch


def fixed_support_gumbel_log_probs(
    logits: torch.Tensor,
    rollout_topk_ids: torch.Tensor,
    rollout_topk_gumbels: torch.Tensor,
    rollout_topk_retained_mask: torch.Tensor,
) -> torch.Tensor:
    """Evaluate recorded actions on their behavior-selected filtered support.

    SGLang applies top-k and top-p before selecting a fixed-width token-ID
    array. When top-p keeps fewer than K tokens, the array also contains
    arbitrary IDs with zero sampling probability. The recorded Boolean mask
    identifies the retained entries *before* Gumbel noise, in the same sorted
    order as the recorded IDs and perturbed logits.

    Keep that behavior support fixed while normalizing the current actor's
    logits. Recomputing a nucleus from the current policy would change the
    action support, and treating zero-probability fillers as retained tokens
    would change the density even before an update. The released epsilon,
    clipped Gumbel score, and current-policy ``log p > -3`` averaging rule are
    unchanged. This is the released score, not a replacement distribution or
    a Jacobian-corrected Concrete density. Only current actor logits receive
    gradients; the recorded action remains detached.
    """

    if rollout_topk_retained_mask.dtype != torch.bool:
        raise TypeError("rollout_topk_retained_mask must be Boolean")
    if rollout_topk_ids.shape != rollout_topk_gumbels.shape or rollout_topk_ids.shape != rollout_topk_retained_mask.shape:
        raise ValueError("recorded IDs, perturbed logits, and retained mask must have identical shapes")
    if logits.ndim != rollout_topk_ids.ndim or logits.shape[:-1] != rollout_topk_ids.shape[:-1]:
        raise ValueError("current logits and recorded support must have matching leading dimensions")
    if not bool(rollout_topk_retained_mask.any(dim=-1).all()):
        raise ValueError("every recorded support row must retain at least one token")

    # SGLang promotes logits to FP32 before sampling. Work on just the recorded
    # support: the full-vocabulary normalizer cancels, and arbitrary filler
    # logits must have neither probability mass nor a gradient.
    # Gather before promotion: casting the packed full-vocabulary matrix first
    # creates a multi-GiB FP32 temporary for long Qwen responses. Recorded top-k
    # IDs are unique on retained entries; repeated padding IDs are masked out.
    # Preserve the previous FP32 gradient accumulation for callers supplying
    # duplicate *retained* IDs, which do not occur in generated top-k supports.
    sorted_retained = rollout_topk_ids.masked_fill(~rollout_topk_retained_mask, -1).sort(dim=-1).values
    duplicate_retained = (sorted_retained[..., 1:] == sorted_retained[..., :-1]) & (sorted_retained[..., 1:] >= 0)
    if bool(duplicate_retained.any()):
        support_logits = logits.float().gather(-1, rollout_topk_ids)
    else:
        support_logits = logits.gather(-1, rollout_topk_ids).float()
    support_logits = support_logits.masked_fill(~rollout_topk_retained_mask, -torch.inf)
    support_log_probs = (torch.softmax(support_logits, dim=-1) + 1e-6).log()
    reparameterized = (rollout_topk_gumbels.detach().float() - support_log_probs).clamp(-1.5, 3.0)
    scores = -reparameterized - (-reparameterized).exp()
    density_mask = (support_log_probs > -3.0).to(scores.dtype)
    return (scores * density_mask).sum(dim=-1) / density_mask.sum(dim=-1)
