"""MLX (Apple Silicon) inference engine for Chatterbox-Flash.

Runs the LLaMA backbone of the T3 decoder natively on Metal via the
`mlx <https://github.com/ml-explore/mlx>`_ framework, while keeping the
small auxiliary modules (speech / text / cond embeddings, speech head,
conditioning encoder) on PyTorch.  The PyTorch ↔ MLX boundary is at
the engine surface, so the block-diffusion ``generate()`` loop does not
need to know about MLX at all.

Memory layout
-------------
* T3 conditioning encoder + speech_emb + text_emb + speech_head live
  on the PyTorch device (CPU on a Mac; the host model is loaded with
  ``device='cpu'`` for the MLX backend).
* The LLaMA stack (embed-free, 30 layers for Llama_520M) is rebuilt in
  MLX, with weights converted once from the PyTorch ``state_dict``.
* A per-layer KV cache lives entirely inside MLX.

Tentative-then-commit pattern
-----------------------------
The block-diffusion loop calls :meth:`block_forward` multiple times for
the same block (one per inner-loop step), each time overwriting the
previous step's tentative KV.  We implement this by tracking a
*committed* length per layer and cropping the cache back to that length
at the start of each block forward.  :meth:`advance_cache` bumps the
committed length so the next block sees the just-written entries as
real prefix.

Status
------
Experimental.  Tested with mlx ≥ 0.13 and mlx-lm ≥ 0.10.  This module
imports MLX lazily; the rest of the package works without MLX installed.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


MLX_AVAILABLE: bool = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_lm") is not None
)


def _require_mlx() -> tuple["module", "module"]:  # type: ignore[name-defined]
    if not MLX_AVAILABLE:
        raise ImportError(
            "MLX backend requires the 'mlx' and 'mlx-lm' packages.\n"
            "  Install (Apple Silicon Mac only):\n"
            "    pip install mlx mlx-lm\n"
            "  or:\n"
            "    pip install 'chatterbox-flash[mlx]'",
        )
    import mlx.core as mx
    import mlx_lm  # noqa: F401  (force import, validate version)
    return mx, mlx_lm


# --------------------------------------------------------------------------- #
#  Weight conversion: PyTorch HF LlamaModel state_dict  →  MLX state_dict
# --------------------------------------------------------------------------- #


def _torch_to_mx(t: torch.Tensor, mx_module):  # noqa: ANN001
    """Convert a torch tensor to an mx.array via numpy (zero extra copy)."""
    arr = t.detach().to(torch.float32).cpu().numpy()
    return mx_module.array(arr)


def _convert_llama_weights(
    tfmr_state: dict[str, torch.Tensor],
    mx_module,
) -> dict[str, "mx_module.array"]:  # type: ignore[name-defined]
    """Convert HF LlamaModel state_dict into an MLX-flat dict.

    Both PyTorch and mlx-lm store nn.Linear as ``(out, in)`` so no
    transposition is needed.  RoPE inv_freq / cos_cached / sin_cached
    buffers are recomputed inside MLX and therefore skipped.
    """
    SKIP_SUBSTRINGS = (
        "rotary_emb.inv_freq",
        "rotary_emb.cos_cached",
        "rotary_emb.sin_cached",
    )
    out: dict[str, object] = {}
    for k, v in tfmr_state.items():
        if any(s in k for s in SKIP_SUBSTRINGS):
            continue
        out[k] = _torch_to_mx(v, mx_module)
    return out


# --------------------------------------------------------------------------- #
#  Embed-free MLX LlamaModel wrapper
# --------------------------------------------------------------------------- #


class _EmbedFreeLlama:
    """Thin wrapper around mlx-lm's LlamaModel that accepts inputs_embeds.

    Reuses the entire mlx-lm LLaMA layer stack (incl. RoPE LLaMA3 scaling
    that ships with mlx-lm) but bypasses the input ``embed_tokens`` layer
    — we already have the speech / text / cond embeddings on the
    PyTorch side.
    """

    def __init__(
        self,
        tfmr_state: dict[str, torch.Tensor],
        hf_config,  # noqa: ANN001
        rotary_emb=None,  # noqa: ANN001
    ) -> None:
        mx, _ = _require_mlx()
        from mlx_lm.models.llama import LlamaModel, ModelArgs

        self._mx = mx
        args = ModelArgs(
            model_type="llama",
            hidden_size=int(hf_config.hidden_size),
            num_hidden_layers=int(hf_config.num_hidden_layers),
            intermediate_size=int(hf_config.intermediate_size),
            num_attention_heads=int(hf_config.num_attention_heads),
            num_key_value_heads=int(hf_config.num_key_value_heads),
            rms_norm_eps=float(hf_config.rms_norm_eps),
            vocab_size=int(hf_config.vocab_size),
            head_dim=int(getattr(hf_config, "head_dim", 0))
            or int(hf_config.hidden_size // hf_config.num_attention_heads),
            max_position_embeddings=int(
                getattr(hf_config, "max_position_embeddings", 4096)
            ),
            rope_theta=float(getattr(hf_config, "rope_theta", 10000.0)),
            rope_traditional=False,
            rope_scaling=dict(hf_config.rope_scaling)
            if getattr(hf_config, "rope_scaling", None) is not None
            else None,
            tie_word_embeddings=bool(hf_config.tie_word_embeddings),
        )
        self._args = args
        self._llama = LlamaModel(args)

        # Convert weights (skip rotary buffers; mlx-lm rebuilds them).
        mlx_state = _convert_llama_weights(tfmr_state, mx)
        from mlx.utils import tree_unflatten
        self._llama.load_weights(list(mlx_state.items()), strict=False)

        # Override RoPE with HF's *exact* inv_freq. Reconstructing it from the
        # config (rope_theta / rope_scaling) is fragile: across transformers
        # versions ``config.rope_theta`` may be absent, in which case the
        # ModelArgs above silently falls back to base=10000 while HF actually
        # used e.g. 500000 — giving a position-dependent RoPE mismatch that
        # scrambles generation. HF's ``rotary_emb.inv_freq`` already bakes in
        # the base *and* any (llama3 / yarn / …) scaling, so using it verbatim
        # is correct regardless of config layout or scaling type.
        self._override_rope_from_hf(rotary_emb)

        # Optional weight-only quantization of the backbone (4- / 8-bit).
        # Driven by an env var so the tts/model signatures stay untouched
        # (same convention as CHATTERBOX_FLASH_ENGINE). Must run AFTER
        # load_weights so the fp weights are present to quantize, and only
        # touches this LLaMA stack — S3Gen / VoiceEncoder live on the torch
        # side and are unaffected.
        self._quant_bits = self._maybe_quantize()

        # Force materialization to catch missing keys early.
        mx.eval(self._llama.parameters())
        self._cache_cls = None
        self._init_cache_cls()

    def _override_rope_from_hf(self, rotary_emb) -> None:  # noqa: ANN001
        """Replace each layer's RoPE with one driven by HF's exact inv_freq."""
        mx = self._mx
        if rotary_emb is None or not hasattr(rotary_emb, "inv_freq"):
            logger.warning(
                "MLX backbone: HF rotary_emb/inv_freq unavailable; falling back "
                "to config-reconstructed RoPE (may mismatch if config lacks "
                "rope_theta).",
            )
            return

        import numpy as _np
        import mlx.nn as nn

        inv = rotary_emb.inv_freq.detach().to(torch.float32).cpu().numpy()
        # mx.fast.rope takes per-pair ``freqs`` = wavelengths = 1 / inv_freq.
        freqs = mx.array((1.0 / inv).astype(_np.float32))
        att_scale = float(getattr(rotary_emb, "attention_scaling", 1.0) or 1.0)
        dims = int(self._args.head_dim)
        traditional = bool(self._args.rope_traditional)

        class _HFRoPE(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dims = dims
                self._freqs = freqs
                self.scale = att_scale
                self.traditional = traditional

            def __call__(self, x, offset=0):  # noqa: ANN001
                y = mx.fast.rope(
                    x, self.dims, traditional=self.traditional,
                    base=None, scale=1.0, offset=offset, freqs=self._freqs,
                )
                return y if self.scale == 1.0 else y * self.scale

        for layer in self._llama.layers:
            layer.self_attn.rope = _HFRoPE()
        logger.info(
            "MLX backbone RoPE overridden from HF inv_freq "
            "(dims=%d, attention_scaling=%.4f).", dims, att_scale,
        )

    def _maybe_quantize(self) -> Optional[int]:
        import os

        raw = os.environ.get("CHATTERBOX_FLASH_MLX_QUANT_BITS", "").strip()
        if not raw:
            return None
        try:
            bits = int(raw)
        except ValueError:
            logger.warning("Ignoring invalid CHATTERBOX_FLASH_MLX_QUANT_BITS=%r", raw)
            return None
        if bits not in (2, 3, 4, 6, 8):
            logger.warning("Unsupported MLX quant bits=%d; skipping.", bits)
            return None
        group_size = int(os.environ.get("CHATTERBOX_FLASH_MLX_QUANT_GROUP", "64"))

        import mlx.nn as nn

        # Skip layers whose dims aren't divisible by group_size — mlx requires
        # in_features % group_size == 0, and the speech/text heads live on the
        # torch side anyway, so any odd-shaped Linear here is left in fp.
        def _can_quant(_path: str, module) -> bool:  # noqa: ANN001
            if not hasattr(module, "to_quantized"):
                return False
            w = getattr(module, "weight", None)
            return w is not None and w.ndim == 2 and (w.shape[-1] % group_size == 0)

        nn.quantize(
            self._llama, group_size=group_size, bits=bits,
            class_predicate=_can_quant,
        )
        logger.info(
            "MLX backbone quantized to %d-bit (group_size=%d).", bits, group_size,
        )
        return bits

    def _init_cache_cls(self) -> None:
        try:
            from mlx_lm.models.cache import KVCache
            self._cache_cls = KVCache
        except ImportError:  # very old mlx-lm
            from mlx_lm.models.base import KVCache  # type: ignore
            self._cache_cls = KVCache

    def make_cache(self) -> list:
        return [self._cache_cls() for _ in range(self._args.num_hidden_layers)]

    def forward(
        self,
        inputs_embeds_mx,
        cache: list,
        *,
        mask=None,
    ):
        """Run the LLaMA stack on continuous embeddings, return hidden_states."""
        mx = self._mx
        x = inputs_embeds_mx
        # Manually mirror mlx-lm LlamaModel.__call__ but without embed_tokens.
        for layer, c in zip(self._llama.layers, cache):
            x = layer(x, mask=mask, cache=c)
        x = self._llama.norm(x)
        return x


# --------------------------------------------------------------------------- #
#  Engine
# --------------------------------------------------------------------------- #


CacheStarts = Union[int, List[int]]


class MLXEngine:
    """Apple-Silicon MLX engine. Implements the chatterbox_flash engine protocol."""

    has_cuda_graph: bool = False  # MLX has no CUDA-graph concept

    def __init__(
        self,
        model,  # ChatterboxFlashT3
        max_seq_len: int,
        dtype: torch.dtype,
        *,
        batch_size: int = 1,
        page_size: int = 16,  # unused; protocol parity
    ) -> None:
        self._model = model
        self._max_seq_len = int(max_seq_len)
        self._dtype = dtype
        self._batch_size = max(1, int(batch_size))
        self._device = model.device

        mx, _ = _require_mlx()
        self._mx = mx
        if dtype == torch.bfloat16:
            self._mx_dtype = mx.bfloat16
        elif dtype == torch.float16:
            self._mx_dtype = mx.float16
        else:
            self._mx_dtype = mx.float32

        # Build the MLX backbone exactly once per engine instance.
        tfmr_state = {k: v for k, v in model.tfmr.state_dict().items()}
        self._backbone = _EmbedFreeLlama(
            tfmr_state, model.tfmr.config,
            rotary_emb=getattr(model.tfmr, "rotary_emb", None),
        )
        self._cache = self._backbone.make_cache()

        # Per-row batch-1 caches, populated only on the ragged (cross-text /
        # per-row prefix length) path.  A single dense batch-B cache cannot
        # represent rows with differing sequence lengths, so the ragged path
        # serializes over these instead of slicing ``self._cache`` live.
        self._row_caches: Optional[List[list]] = None

        # Per-row bookkeeping.  MLX's KV cache is shared across the batch
        # dim (same as the SDPA path) so this is uniform across rows.
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
        self._cache = self._backbone.make_cache()
        self._row_caches = None
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

    # ------------------------------------------------------------------ #
    #  Tensor boundary
    # ------------------------------------------------------------------ #

    def _torch_to_mx(self, t: Tensor):
        """torch (cpu / mps / cuda) → mx.array (Metal). Routes via numpy on host."""
        return self._mx.array(t.detach().to(torch.float32).cpu().numpy()).astype(
            self._mx_dtype,
        )

    def _mx_to_torch(self, a) -> Tensor:
        """mx.array → torch on the host device, restoring our dtype."""
        # mx.array → numpy → torch (always lands on the model's device).
        arr = np.array(a.astype(self._mx.float32))
        return torch.from_numpy(arr).to(device=self._device, dtype=self._dtype)

    # ------------------------------------------------------------------ #
    #  KV-cache cropping (tentative-then-commit pattern)
    # ------------------------------------------------------------------ #

    def _crop_cache(self, cache: list, length: int) -> None:
        """Truncate every layer of ``cache`` to ``length`` time steps."""
        mx = self._mx
        for c in cache:
            # mlx-lm's KVCache exposes ``keys`` / ``values`` (shape
            # ``(B, n_kv_h, T, head_dim)``) and an ``offset`` int.
            if c.keys is None:
                continue
            c.keys = c.keys[..., :length, :]
            c.values = c.values[..., :length, :]
            c.offset = int(length)
        mx.eval(*[(c.keys, c.values) for c in cache if c.keys is not None])

    def _crop_cache_to(self, length: int) -> None:
        """Truncate the shared (uniform-path) KV cache to ``length`` steps."""
        self._crop_cache(self._cache, length)

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
                f"MLX prefix_forward: prefix batch {B_pfx} must be 1 or "
                f"engine batch {self._batch_size}",
            )
        if B_pfx == 1 and self._batch_size > 1:
            prefix_emb = prefix_emb.expand(self._batch_size, -1, -1).contiguous()

        lp = int(prefix_emb.size(1))
        # MLX expects a 3D input (B, T, D).
        x_mx = self._torch_to_mx(prefix_emb)
        # The conditioning prefix must be processed CAUSALLY (matching the HF
        # SDPA path, which gets an implicit causal mask). _EmbedFreeLlama
        # bypasses LlamaModel.__call__, so mask=None would leave the prefix
        # fully bidirectional and corrupt the cached context. Use mlx's native
        # "causal" string mask — exactly what mlx-lm passes for a causal
        # forward — so it is dtype-safe regardless of the backbone weight dtype.
        out_mx = self._backbone.forward(x_mx, self._cache, mask="causal")
        self._mx.eval(out_mx)

        for bi in range(self._batch_size):
            self._cache_lens[bi] = lp
            self._prefix_lens[bi] = lp

        # SOS is at index lp - 1.
        shift_ctx = self._mx_to_torch(out_mx[:, lp - 1 : lp, :])
        return shift_ctx.contiguous()

    def _prefix_forward_ragged(self, emb_list: List[Tensor]) -> Tensor:
        """Pad-and-mask path for per-row prefix lengths (cross-text / null prefix)."""
        if len(emb_list) == 1:
            return self.prefix_forward(emb_list[0])

        mx = self._mx
        B = len(emb_list)
        if B != self._batch_size:
            raise ValueError(
                f"MLX ragged prefix: list length {B} != engine batch "
                f"{self._batch_size}",
            )
        lp_list = [int(e.size(1)) for e in emb_list]
        lp_max = max(lp_list)
        dim = emb_list[0].size(-1)

        padded = torch.zeros(B, lp_max, dim, dtype=self._dtype, device=self._device)
        for bi, e in enumerate(emb_list):
            padded[bi, : e.size(1)] = e.squeeze(0).to(self._dtype)

        # Additive padding mask, combined with causal triangular mask.
        x_mx = self._torch_to_mx(padded)
        neg_inf = mx.array(-1e9, dtype=self._mx_dtype)
        pad_mask = mx.zeros((B, lp_max), dtype=self._mx_dtype)
        for bi, lp_i in enumerate(lp_list):
            if lp_i < lp_max:
                pad_mask[bi, lp_i:] = neg_inf
        causal = mx.triu(
            mx.full((lp_max, lp_max), neg_inf, dtype=self._mx_dtype),
            k=1,
        )
        attn_mask = pad_mask[:, None, None, :] + causal[None, None]
        out_mx = self._backbone.forward(x_mx, self._cache, mask=attn_mask)
        mx.eval(out_mx)

        for bi, lp_i in enumerate(lp_list):
            self._cache_lens[bi] = lp_i
            self._prefix_lens[bi] = lp_i

        # Split the padded batch-B prefix cache into per-row batch-1 caches,
        # each trimmed to that row's *valid* prefix length (the padded tail
        # carries masked garbage and must be dropped).  The ragged
        # ``block_forward`` serializes over these so a batch-1 block input
        # never gets concatenated against batch-B cache keys.
        row_caches: List[list] = []
        for bi, lp_i in enumerate(lp_list):
            rc = self._backbone.make_cache()
            for src, dst in zip(self._cache, rc):
                if src.keys is None:
                    continue
                dst.keys = src.keys[bi : bi + 1, :, :lp_i, :]
                dst.values = src.values[bi : bi + 1, :, :lp_i, :]
                dst.offset = int(lp_i)
            mx.eval(*[(c.keys, c.values) for c in rc if c.keys is not None])
            row_caches.append(rc)
        self._row_caches = row_caches

        # SOS row index per sample (lp_i - 1).  Gather on the torch side.
        hidden = self._mx_to_torch(out_mx)
        idx = torch.tensor(
            [lp_i - 1 for lp_i in lp_list], device=self._device, dtype=torch.long,
        )
        shift_ctx = hidden.gather(
            1, idx[:, None, None].expand(B, 1, hidden.size(-1)),
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
        """Tentative block forward; KV is written, advance_cache commits it."""
        B = int(block_emb.size(0))
        bl = int(block_emb.size(1))

        if isinstance(cache_starts, int):
            cs_uniform: Optional[int] = cache_starts
        else:
            cs_list = list(cache_starts)
            if len(set(cs_list)) == 1:
                cs_uniform = cs_list[0]
            else:
                cs_uniform = None

        if cs_uniform is not None:
            self._crop_cache_to(int(cs_uniform))
            x_mx = self._torch_to_mx(block_emb)
            # mlx-lm builds a causal mask internally when T > 1 and the
            # cache has an offset; the block sees the cropped prefix +
            # itself.  No explicit mask needed here.
            out_mx = self._backbone.forward(x_mx, self._cache)
            self._mx.eval(out_mx)
            return self._mx_to_torch(out_mx)

        # Ragged: per-row cache_starts differ (cross-text / null-prefix).
        # mlx-lm exposes no per-row position offset, and a dense batch-B
        # cache cannot hold rows of differing length, so we serialize over
        # the per-row batch-1 caches seeded at prefix time.  Each row's
        # cache is cropped to that row's committed length (cs_i), the block
        # is forwarded tentatively, and the write persists in the row cache
        # for the next step / commit — mirroring the uniform path per row.
        # We deliberately serialize to keep correctness > throughput.
        mx = self._mx
        if self._row_caches is None:
            raise RuntimeError(
                "MLX ragged block_forward requires per-row caches; "
                "prefix_forward must run with a per-row prefix list first.",
            )
        outs = []
        for bi, cs_i in enumerate(cs_list):  # type: ignore[name-defined]
            rc = self._row_caches[bi]
            self._crop_cache(rc, int(cs_i))
            x_mx_i = self._torch_to_mx(block_emb[bi : bi + 1])
            out_mx_i = self._backbone.forward(x_mx_i, rc)
            mx.eval(out_mx_i)
            outs.append(self._mx_to_torch(out_mx_i))
        return torch.cat(outs, dim=0)

    def block_forward_graph(
        self,
        block_emb: Tensor,
        cache_starts: CacheStarts,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """No graph capture on MLX → delegate to eager."""
        return self.block_forward(block_emb, cache_starts), None

    def advance_cache(self, block_len: int) -> None:
        """Commit ``block_len`` newly-written entries."""
        for bi in range(self._batch_size):
            self._cache_lens[bi] += int(block_len)

    def set_shift_ctx(self, ctx: Tensor) -> None:  # noqa: D401
        """No graph buffer to stage; the caller folds shift_ctx in-line."""
        return None

    def capture_cuda_graph(
        self,
        block_size: int,
        *,
        speech_head: Optional[torch.nn.Linear] = None,
    ) -> None:  # noqa: D401
        """MLX has no CUDA-graph concept; intentionally a no-op."""
        logger.debug(
            "MLXEngine.capture_cuda_graph(block_size=%d) is a no-op.",
            block_size,
        )
        return None


__all__ = [
    "MLXEngine",
    "MLX_AVAILABLE",
]
