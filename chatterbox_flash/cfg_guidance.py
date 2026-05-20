"""Classifier-free guidance helpers for Chatterbox-Flash.

We only keep the ``zero_text_batch`` recipe used in the paper: at each block
step the forward batch is doubled to ``2 * B``; rows ``0..B-1`` carry the full
text + conditioning, rows ``B..2B-1`` carry a null branch where the text and
the entire conditioning prefix are zeroed (``cfg_null_cond_mode="zero_all"``).
The two halves of the logits are then combined via
``log p' = log p_c + w * (log p_c - log p_u)``.

The same combination is applied on the **PMI scale** as well (see
:func:`pmi_cfg_combine`) — this is the only PMI/CFG combination supported;
no alternative ``cfg_prior_mode`` is exposed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def omnivoice_cfg_log_probs(
    c_log_probs: Tensor,
    u_log_probs: Tensor,
    guidance_scale: float,
) -> Tensor:
    """``log p' = log p_c + w * (log p_c - log p_u)`` (elementwise)."""
    if guidance_scale == 0.0:
        return c_log_probs
    return c_log_probs + guidance_scale * (c_log_probs - u_log_probs)


def apply_zero_text_cfg_from_logits(
    logits_cond: Tensor,
    logits_u: Tensor,
    guidance_scale: float,
) -> Tensor:
    """CFG combination from two logit tensors of identical shape.

    Operates on log-softmax so the result is a valid scoring/sampling logit
    (softmax recovers the guided distribution).
    """
    if guidance_scale == 0.0:
        return logits_cond
    c_log = F.log_softmax(logits_cond, dim=-1)
    u_log = F.log_softmax(logits_u, dim=-1)
    return omnivoice_cfg_log_probs(c_log, u_log, guidance_scale)


def pmi_cfg_combine(
    pmi_c: Tensor,
    pmi_u: Tensor,
    guidance_scale: float,
) -> Tensor:
    """Guidance-combine PMI scores: ``pmi = (1+w) * pmi_c - w * pmi_u``."""
    w = float(guidance_scale)
    return (1.0 + w) * pmi_c - w * pmi_u
