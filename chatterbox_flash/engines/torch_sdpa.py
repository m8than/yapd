"""Pure-PyTorch fallback inference engine for Chatterbox-Flash.

Implements the same engine protocol as :class:`FlashInferEngine` but on
top of the stock HuggingFace transformers ``LlamaModel`` SDPA backend.
No CUDA-graph capture, no paged buffer, no flashinfer kernels — just
``transformers.DynamicCache`` with the standard causal/bidirectional
attention masks.

This is the engine selected automatically when ``flashinfer-python`` is
not importable, e.g. on Ampere/Turing GPUs or in environments where the
flashinfer wheels aren't available.  It produces numerically matching
outputs (up to fp32/bf16 reduction-order differences) at lower
throughput.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


CacheStarts = Union[int, List[int]]


def _crop_dynamic_cache(cache, max_length: int) -> None:
    """Truncate a :class:`transformers.DynamicCache` to ``max_length`` tokens.

    Compatible with both the new ``cache.crop()`` API (transformers ≥ 4.46)
    and the older direct-tensor layout.
    """
    if max_length is None:
        return
    if hasattr(cache, "crop"):
        cache.crop(max_length)
    else:
        for li in range(len(cache.key_cache)):
            cache.key_cache[li] = cache.key_cache[li][:, :, :max_length, :]
            cache.value_cache[li] = cache.value_cache[li][:, :, :max_length, :]
        cache._seen_tokens = max_length


class TorchSDPAEngine:
    """Pure-PyTorch SDPA + :class:`DynamicCache` fallback engine."""

    has_cuda_graph: bool = False

    def __init__(
        self,
        model,
        max_seq_len: int,
        dtype: torch.dtype,
        *,
        batch_size: int = 1,
        page_size: int = 16,  # ignored; protocol parity
    ) -> None:
        from transformers import DynamicCache  # local import; lazy

        self._model = model
        self._max_seq_len = int(max_seq_len)
        self._dtype = dtype
        self._batch_size = max(1, int(batch_size))
        self._device = model.device

        self._DynamicCache = DynamicCache

        self._dc = DynamicCache()
        # ``cache_lens[bi]`` is the *committed* (advance_cache-ed) length per
        # sample.  In the SDPA path the underlying DynamicCache is uniform
        # across the batch, so the per-sample bookkeeping is mostly for
        # protocol parity with FlashInfer; the per-sample length is read only
        # by capacity checks / introspection.
        self._cache_lens: List[int] = [0] * self._batch_size
        self._prefix_lens: List[int] = [0] * self._batch_size

    # ------------------------------------------------------------------ #
    #  Introspection / lifecycle
    # ------------------------------------------------------------------ #

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def cache_len(self) -> int:
        """Committed cache length (uniform across the batch under SDPA)."""
        return self._cache_lens[0]

    @cache_len.setter
    def cache_len(self, value: int) -> None:
        self._cache_lens = [int(value)] * self._batch_size

    @property
    def cache_lens(self) -> List[int]:
        return self._cache_lens

    @property
    def prefix_lens(self) -> List[int]:
        return self._prefix_lens

    def reset(self) -> None:
        self._dc = self._DynamicCache()
        self._cache_lens = [0] * self._batch_size
        self._prefix_lens = [0] * self._batch_size

    def can_reuse(
        self,
        max_seq_len: int,
        dtype: torch.dtype,
        *,
        batch_size: int = 1,
        page_size: int = 16,
    ) -> bool:
        return (
            self._max_seq_len >= max_seq_len
            and self._dtype == dtype
            and self._batch_size == max(1, batch_size)
        )

    def snapshot_kv(self):
        """Lightweight snapshot for CFG uncond restore."""
        ks = [t.clone() for t in self._dc.key_cache]
        vs = [t.clone() for t in self._dc.value_cache]
        return (ks, vs, list(self._cache_lens), list(self._prefix_lens))

    def restore_kv(self, snap) -> None:
        ks, vs, cl, pl = snap
        self._dc = self._DynamicCache()
        for k, v in zip(ks, vs):
            self._dc.key_cache.append(k.clone())
            self._dc.value_cache.append(v.clone())
        self._dc._seen_tokens = ks[0].shape[-2] if ks else 0
        self._cache_lens = list(cl)
        self._prefix_lens = list(pl)

    # ------------------------------------------------------------------ #
    #  Prefix forward
    # ------------------------------------------------------------------ #

    def prefix_forward(
        self,
        prefix_emb_or_list: Union[Tensor, List[Tensor]],
    ) -> Tensor:
        """Run the cond/text/SOS prefix once and seed the cache."""
        if isinstance(prefix_emb_or_list, list):
            return self._prefix_forward_ragged(prefix_emb_or_list)

        prefix_emb = prefix_emb_or_list
        B_pfx = prefix_emb.size(0)
        if B_pfx not in (1, self._batch_size):
            raise ValueError(
                f"SDPA prefix_forward: prefix batch {B_pfx} must be 1 or "
                f"engine batch {self._batch_size}",
            )
        if B_pfx == 1 and self._batch_size > 1:
            prefix_emb = prefix_emb.expand(self._batch_size, -1, -1).contiguous()

        lp = prefix_emb.size(1)
        device = self._device
        pos_ids = (
            torch.arange(lp, device=device)
            .unsqueeze(0)
            .expand(self._batch_size, -1)
            .contiguous()
        )

        cfg = self._model.tfmr.config
        old_impl = getattr(cfg, "_attn_implementation", "sdpa")
        cfg._attn_implementation = "sdpa"
        try:
            out = self._model.tfmr(
                input_ids=None,
                inputs_embeds=prefix_emb.to(self._dtype),
                position_ids=pos_ids,
                past_key_values=self._dc,
                use_cache=True,
                return_dict=True,
            )
        finally:
            cfg._attn_implementation = old_impl

        self._dc = out.past_key_values
        for bi in range(self._batch_size):
            self._cache_lens[bi] = lp
            self._prefix_lens[bi] = lp

        shift_ctx = out.last_hidden_state[:, lp - 1 : lp, :].contiguous()
        return shift_ctx

    def _prefix_forward_ragged(self, emb_list: List[Tensor]) -> Tensor:
        """Cross-text path: pad to max length, mask out padding in attention.

        Each sample gets its own prefix length; the cache is built from a
        left-padded forward.  Padding tokens are attention-masked so they
        contribute zero KV, but the cache slots still exist — downstream
        block forwards know to skip them via per-sample ``cache_starts``.
        """
        if len(emb_list) == 1:
            return self.prefix_forward(emb_list[0])

        B = len(emb_list)
        if B != self._batch_size:
            raise ValueError(
                f"SDPA ragged prefix: list length {B} != engine batch "
                f"{self._batch_size}",
            )
        lp_list = [int(e.size(1)) for e in emb_list]
        lp_max = max(lp_list)
        dim = emb_list[0].size(-1)
        device = self._device

        padded = torch.zeros(B, lp_max, dim, device=device, dtype=self._dtype)
        for bi, e in enumerate(emb_list):
            padded[bi, : e.size(1)] = e.to(self._dtype).squeeze(0)
        # Additive mask: 0 for visible, -inf for padding (additive on attn scores).
        # transformers SDPA expects 4D mask (B, 1, Q, K).
        pad_mask = torch.zeros(B, lp_max, device=device, dtype=self._dtype)
        for bi, lp_i in enumerate(lp_list):
            if lp_i < lp_max:
                pad_mask[bi, lp_i:] = float("-inf")
        attn_mask = pad_mask[:, None, None, :].expand(B, 1, lp_max, lp_max)
        # Combine with causal: HF SDPA does causal internally if attention_mask
        # is 2D (B, K); for the 4D additive mask path we have to OR in causal.
        causal = torch.triu(
            torch.full((lp_max, lp_max), float("-inf"), device=device, dtype=self._dtype),
            diagonal=1,
        )
        attn_mask = attn_mask + causal[None, None]

        pos_ids = (
            torch.arange(lp_max, device=device).unsqueeze(0).expand(B, -1).contiguous()
        )

        cfg = self._model.tfmr.config
        old_impl = getattr(cfg, "_attn_implementation", "sdpa")
        cfg._attn_implementation = "sdpa"
        try:
            out = self._model.tfmr(
                input_ids=None,
                inputs_embeds=padded,
                position_ids=pos_ids,
                past_key_values=self._dc,
                use_cache=True,
                return_dict=True,
                attention_mask=attn_mask,
            )
        finally:
            cfg._attn_implementation = old_impl

        self._dc = out.past_key_values
        for bi, lp_i in enumerate(lp_list):
            self._cache_lens[bi] = lp_i
            self._prefix_lens[bi] = lp_i

        # SOS hidden lives at index lp_i - 1 within each row.
        idx = torch.tensor(
            [lp_i - 1 for lp_i in lp_list], device=device, dtype=torch.long,
        )
        shift_ctx = out.last_hidden_state.gather(
            1,
            idx[:, None, None].expand(B, 1, out.last_hidden_state.size(-1)),
        )
        return shift_ctx.contiguous()

    # ------------------------------------------------------------------ #
    #  Block forward
    # ------------------------------------------------------------------ #

    def block_forward(
        self,
        block_emb: Tensor,
        cache_starts: CacheStarts,
    ) -> Tensor:
        """Transient block forward; KV is written but not committed.

        Caller must re-invoke with the same ``cache_starts`` between
        unmasking steps; the cache is re-cropped each time so the new
        block's keys/values overwrite the previous step's tentative KV.
        """
        B = block_emb.size(0)
        bl = block_emb.size(1)
        device = self._device
        dtype = self._dtype

        if isinstance(cache_starts, int):
            cs_uniform: Optional[int] = cache_starts
        else:
            cs_list = list(cache_starts)
            if len(set(cs_list)) == 1:
                cs_uniform = cs_list[0]
            else:
                cs_uniform = None

        if cs_uniform is not None:
            _crop_dynamic_cache(self._dc, cs_uniform)
            pos_ids = (
                torch.arange(cs_uniform, cs_uniform + bl, device=device)
                .unsqueeze(0).expand(B, -1).contiguous()
            )
            kv_len = cs_uniform + bl
            # Full-visible mask within the block (already-causal prefix in cache).
            attn_mask = torch.zeros((1, 1, bl, kv_len), dtype=dtype, device=device)
        else:
            cs_max = max(cs_list)
            _crop_dynamic_cache(self._dc, cs_max)
            # Pad each row's positions to cs_max + bl; mask out the prefix
            # slots beyond cs_i so the block forward only attends to that
            # sample's true prefix.
            pos_rows = torch.zeros(B, bl, device=device, dtype=torch.long)
            for bi, cs in enumerate(cs_list):
                pos_rows[bi] = torch.arange(cs, cs + bl, device=device)
            pos_ids = pos_rows
            kv_len = cs_max + bl
            attn_mask = torch.zeros(B, 1, bl, kv_len, dtype=dtype, device=device)
            for bi, cs in enumerate(cs_list):
                if cs < cs_max:
                    attn_mask[bi, 0, :, cs:cs_max] = float("-inf")

        cfg = self._model.tfmr.config
        old_impl = getattr(cfg, "_attn_implementation", "sdpa")
        cfg._attn_implementation = "sdpa"
        try:
            out = self._model.tfmr(
                input_ids=None,
                inputs_embeds=block_emb.to(dtype),
                position_ids=pos_ids,
                past_key_values=self._dc,
                use_cache=True,
                return_dict=True,
                attention_mask=attn_mask,
            )
        finally:
            cfg._attn_implementation = old_impl

        self._dc = out.past_key_values
        return out.last_hidden_state

    def block_forward_graph(
        self,
        block_emb: Tensor,
        cache_starts: CacheStarts,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """SDPA has no CUDA-graph fast path → delegate to eager."""
        return self.block_forward(block_emb, cache_starts), None

    def advance_cache(self, block_len: int) -> None:
        """Commit ``block_len`` newly-written entries; bump per-sample lens."""
        for bi in range(self._batch_size):
            self._cache_lens[bi] += int(block_len)

    def set_shift_ctx(self, ctx: Tensor) -> None:  # noqa: D401
        """No-op; the SDPA path computes the shift in-line in the caller."""
        return None

    def capture_cuda_graph(
        self,
        block_size: int,
        *,
        speech_head: Optional[torch.nn.Linear] = None,
    ) -> None:  # noqa: D401
        """No CUDA-graph support; intentionally a no-op."""
        logger.debug(
            "TorchSDPAEngine.capture_cuda_graph(block_size=%d) is a no-op "
            "(SDPA backend has no CUDA-graph fast path).",
            block_size,
        )
        return None
