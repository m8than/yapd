"""Prior-calibrated PMI scoring for Chatterbox-Flash block decoding.

The decoder ranks masked positions by a self-calibrated pointwise mutual
information score that suppresses the dominant-token bias of discrete speech
codecs (no per-token silence IDs needed):

    s_i = log p_i(x_i) - lambda * log p_bar(x_i)

where ``p_bar`` is a reference marginal — either the precomputed unconditional
block prior (see :meth:`ChatterboxFlashT3._compute_uncond_block_prior`) or the
step-0 block-mean ``mean_j p_j``.
"""

from __future__ import annotations

import torch
from torch import Tensor

_LOG_EPS = 1e-8


def _marginal_for_sampled(
    probs_bl: Tensor,
    sampled: Tensor,
    marginal_prior: Tensor | None,
) -> Tensor:
    """Return per-position marginal probability for each sampled token.

    If ``marginal_prior`` is given (shape ``(V,)``), index into it directly.
    Otherwise compute the block-mean marginal on the fly.
    """
    if marginal_prior is not None:
        return marginal_prior[sampled]
    bl = probs_bl.size(0)
    sampled_cols = sampled.unsqueeze(0).expand(bl, -1)
    return probs_bl.gather(1, sampled_cols).mean(dim=0)


def self_calibrated_pmi(
    probs_bl: Tensor,
    sampled: Tensor,
    eps: float = _LOG_EPS,
    marginal_prior: Tensor | None = None,
    prior_lambda: float = 1.0,
) -> Tensor:
    """
    Per-position PMI-style score vs a marginal prior.

    s_i = log p_i(x_i) - lambda * log p_bar(x_i),
    p_bar(v) = mean_j p_j(v).

    When ``marginal_prior`` (shape ``(V,)``) is provided, it is used as
    ``p_bar`` instead of the current block-mean — typically the precomputed
    unconditional block prior for a consistent, uninformed reference.
    """
    pi = probs_bl.gather(1, sampled.unsqueeze(1)).squeeze(1)
    p_bar_sampled = _marginal_for_sampled(probs_bl, sampled, marginal_prior)
    return torch.log(pi.clamp_min(eps)) - (
        float(prior_lambda) * torch.log(p_bar_sampled.clamp_min(eps))
    )


def compute_pmi(
    probs_bl: Tensor,
    sampled: Tensor,
    *,
    marginal_prior: Tensor | None = None,
    prior_lambda: float = 1.0,
    eps: float = _LOG_EPS,
) -> Tensor:
    """Compute self-calibrated PMI.

    When ``marginal_prior`` (shape ``(V,)``) is given, it is used as the
    denominator instead of computing ``probs_bl.mean(dim=0)`` on the fly.
    """
    return self_calibrated_pmi(
        probs_bl,
        sampled,
        eps=eps,
        marginal_prior=marginal_prior,
        prior_lambda=prior_lambda,
    )
