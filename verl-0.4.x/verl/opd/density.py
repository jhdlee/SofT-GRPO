"""Differentiable replay of the released fixed-support Gumbel score."""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable


def _validate_support(logits, ids, actions, retained):
    if retained.dtype != torch.bool:
        raise TypeError("rollout_topk_retained_mask must be Boolean")
    if ids.shape != actions.shape or ids.shape != retained.shape:
        raise ValueError("recorded IDs, perturbed logits, and retained mask must have identical shapes")
    if logits.ndim != ids.ndim or logits.shape[:-1] != ids.shape[:-1]:
        raise ValueError("current logits and recorded support must have matching leading dimensions")
    if not bool(retained.any(dim=-1).all()):
        raise ValueError("every recorded support row must retain at least one token")


def _gumbel_scores(support_logits, actions, retained):
    support_logits = support_logits.masked_fill(~retained, -torch.inf)
    support_log_probs = (torch.softmax(support_logits, dim=-1) + 1e-6).log()
    reparameterized = (actions.detach().float() - support_log_probs).clamp(-1.5, 3.0)
    scores = -reparameterized - (-reparameterized).exp()
    density_mask = (support_log_probs > -3.0).to(scores.dtype)
    return (scores * density_mask).sum(dim=-1) / density_mask.sum(dim=-1)


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

    _validate_support(logits, rollout_topk_ids, rollout_topk_gumbels, rollout_topk_retained_mask)

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
    return _gumbel_scores(support_logits, rollout_topk_gumbels, rollout_topk_retained_mask)


class _MixedSupportLogProbs(torch.autograd.Function):
    """Return one dense gradient for disjoint soft and categorical positions.

    Ordinary ``where(CE(logits), gather(logits))`` creates separate full-size
    gradients for both branches. Here the categorical kernel's forward stays
    unchanged, while backward recomputes only bounded groups of hard rows and
    accumulates the small fixed-support derivative into the same output.
    Never overwrite the shared input: OPD also needs the original logits.
    """

    @staticmethod
    def forward(ctx, logits, ids, actions, retained, labels, categorical, row_chunk_size):
        soft = _gumbel_scores(logits.gather(-1, ids).float(), actions, retained)
        hard = categorical(logits, labels, inplace_backward=False)
        is_hard = (ids[:, 1:] == 0).all(-1)
        ctx.save_for_backward(logits, ids, actions.detach(), retained, labels)
        ctx.categorical = categorical
        ctx.row_chunk_size = row_chunk_size
        ctx.set_materialize_grads(False)
        return torch.where(is_hard, hard, soft)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        if upstream is None or not ctx.needs_input_grad[0]:
            return (None,) * 7
        logits, ids, actions, retained, labels = ctx.saved_tensors
        is_hard = (ids[:, 1:] == 0).all(-1)
        # Recompute the released epsilon/clamp/mask derivative with ordinary
        # autograd on K columns, preserving its exact boundary conventions.
        with torch.enable_grad():
            support = logits.gather(-1, ids).detach().float().requires_grad_(True)
            soft = _gumbel_scores(support, actions, retained)
            sensitivity = upstream.float().masked_fill(is_hard, 0)
            support_gradient = torch.autograd.grad(soft, support, sensitivity)[0]
        gradient = torch.zeros_like(logits)
        sorted_ids = ids.masked_fill(~retained, -1).sort(-1).values
        duplicates = (sorted_ids[:, 1:] == sorted_ids[:, :-1]) & (sorted_ids[:, 1:] >= 0)
        if bool(duplicates.any()):
            # Match the existing duplicate-ID contract: add in FP32 before
            # rounding to the input dtype, with only row_chunk_size rows live.
            for start in range(0, logits.shape[0], ctx.row_chunk_size):
                stop = start + ctx.row_chunk_size
                chunk = torch.zeros_like(logits[start:stop], dtype=torch.float32)
                chunk.scatter_add_(-1, ids[start:stop], support_gradient[start:stop])
                gradient[start:stop].copy_(chunk)
                del chunk
        else:
            gradient.scatter_add_(-1, ids, support_gradient.to(logits.dtype))
        del support_gradient, support, soft
        hard_rows = torch.nonzero(is_hard, as_tuple=False).flatten()
        for rows in hard_rows.split(ctx.row_chunk_size):
            if rows.numel() == 0:
                continue
            with torch.enable_grad():
                chunk = logits.index_select(0, rows).detach().requires_grad_(True)
                score = ctx.categorical(chunk, labels.index_select(0, rows), inplace_backward=False)
                chunk_gradient = torch.autograd.grad(score, chunk, upstream.index_select(0, rows))[0]
            gradient.index_copy_(0, rows, chunk_gradient)
            del chunk, score, chunk_gradient
        return gradient, None, None, None, None, None, None


def mixed_support_log_probs(
    logits, rollout_topk_ids, rollout_topk_gumbels, rollout_topk_retained_mask, labels,
    *, categorical_log_probs, row_chunk_size=128,
):
    """Replay mixed actions with one vocabulary-sized first-order gradient.

    ``categorical_log_probs`` is the existing FlashAttention CE wrapper in
    training. Keeping it injected also permits independent CPU gradient
    oracles, without installing a categorical fallback in the rollout path.
    Forward values retain the same categorical kernel and fixed-support score.
    """
    _validate_support(logits, rollout_topk_ids, rollout_topk_gumbels, rollout_topk_retained_mask)
    if logits.ndim != 2 or labels.shape != logits.shape[:-1]:
        raise ValueError("mixed replay requires packed [rows, vocabulary] logits and one label per row")
    if type(row_chunk_size) is not int or row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be a positive integer")
    return _MixedSupportLogProbs.apply(
        logits, rollout_topk_ids, rollout_topk_gumbels, rollout_topk_retained_mask,
        labels, categorical_log_probs, row_chunk_size,
    )
