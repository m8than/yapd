"""Abstract inference-engine protocol shared by the FlashInfer and the
torch-SDPA backends.

The block-diffusion ``generate()`` loop is engine-agnostic: it only ever
talks to the engine through these six methods.  Concrete backends differ
in how they store the KV cache (paged buffer vs. HuggingFace
:class:`~transformers.DynamicCache`), how they batch the prefix
forward, and whether they can capture the per-block forward as a CUDA
graph.

Contract
--------
The engine manages a single KV cache shared by the conditioning prefix
and the speech-block forwards.  Lifecycle for one ``generate()`` call::

    engine.reset()
    shift_ctx = engine.prefix_forward(prefix_emb_or_list)   # (B, 1, dim)
    engine.capture_cuda_graph(BS, speech_head=speech_head)   # optional

    for b in range(num_blocks):
        for k in range(K):
            engine.set_shift_ctx(shift_ctx)                  # graph engines only
            hidden, logits = engine.block_forward_graph(block_emb, cs)
            #   or: hidden = engine.block_forward(block_emb, cs)

        engine.advance_cache(block_len)
        # caller commits the block's clean tokens via block_forward(...)
        # before advance_cache; shift_ctx is updated from that hidden.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Tuple, Union, runtime_checkable

import torch
from torch import Tensor


CacheStarts = Union[int, List[int]]


@runtime_checkable
class InferenceEngine(Protocol):
    """Minimal protocol implemented by all Chatterbox-Flash engines."""

    has_cuda_graph: bool
    """``True`` iff :meth:`capture_cuda_graph` was called and succeeded."""

    @property
    def cache_len(self) -> int:
        """Current per-row KV cache occupancy (uniform across the batch)."""
        ...

    def reset(self) -> None:
        """Clear cache contents and free CUDA-graph state for the next call."""
        ...

    def can_reuse(
        self,
        max_seq_len: int,
        dtype: torch.dtype,
        *,
        batch_size: int,
        page_size: int = 16,
    ) -> bool:
        """Whether this engine can be re-used for a request with these shape/dtype."""
        ...

    # --- prefix / block forwards -------------------------------------- #

    def prefix_forward(
        self,
        prefix_emb_or_list: Union[Tensor, List[Tensor]],
    ) -> Tensor:
        """Encode the conditioning prefix once.

        ``prefix_emb_or_list`` is either a stacked tensor ``(B_fwd, lp, dim)``
        (same-text path) or a list of length ``B_fwd`` of per-row prefixes
        with different lengths (cross-text path).  Writes the prefix KV
        into the engine's cache and returns ``shift_ctx`` — the hidden
        state of the SOS row, shape ``(B_fwd, 1, dim)``.
        """
        ...

    def block_forward(
        self,
        block_emb: Tensor,
        cache_starts: CacheStarts,
    ) -> Tensor:
        """Eager (non-CUDA-graph) block forward.

        ``block_emb`` shape: ``(B_fwd, bl, dim)``.
        ``cache_starts`` is either an int (uniform start position) or a
        per-row list (cross-text / compact-null-prefix path).
        Returns hidden state ``(B_fwd, bl, dim)``; logits are computed by
        the caller via ``speech_head``.
        """
        ...

    def block_forward_graph(
        self,
        block_emb: Tensor,
        cache_starts: CacheStarts,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """CUDA-graph replay of the block forward.

        Engines that don't support CUDA graphs (e.g. torch SDPA) should
        return ``(self.block_forward(block_emb, cache_starts), None)`` so
        the caller can fall back uniformly.  When the engine pre-computes
        logits inside the graph (FlashInfer), the second element of the
        return tuple is the logits tensor ``(B_fwd, bl, V)``.
        """
        ...

    def advance_cache(self, block_len: int) -> None:
        """Commit ``block_len`` newly-written entries to the cache."""
        ...

    def set_shift_ctx(self, ctx: Tensor) -> None:
        """Stage the SOS / previous-block-end hidden state for the graph.

        FlashInfer copies this into a graph-captured buffer at the start
        of each block; the SDPA engine ignores it (it computes the shift
        in-line during :meth:`block_forward`).
        """
        ...

    def capture_cuda_graph(
        self,
        block_size: int,
        *,
        speech_head: Optional[torch.nn.Linear] = None,
    ) -> None:
        """Capture the per-block forward as a CUDA graph.

        ``block_size`` is the fixed ``D`` the graph is captured for.
        Engines that don't support CUDA graphs should leave
        ``has_cuda_graph = False``.  Optionally folds the speech head
        into the graph if supplied.
        """
        ...
