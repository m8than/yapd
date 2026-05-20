"""
FlashInfer attention engine with **paged KV cache** for batched
block-diffusion inference.

Supports ``batch_size >= 1`` via ``BatchPrefillWithPagedKVCacheWrapper``.
Each sample gets a static slice of pages in the pool; prefix pages are
computed once and replicated for same-text multi-generation, or computed
independently per sample for cross-text batching.

KV cache layout (NHD, per layer):
    ``(total_pages, 2, page_size, num_kv_heads, head_dim)``

Static page allocation: sample *i* → pages
``[i * pages_per_sample : (i+1) * pages_per_sample]``.

Fused projections (same as before):
  - QKV: 1 GEMM per layer instead of 3.
  - Gate-up MLP: 1 GEMM instead of 2 (Llama SwiGLU).
"""

from __future__ import annotations

import logging
import math
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

try:
    import flashinfer
    _FLASHINFER_AVAILABLE = True
    # Module-level alias avoids a name collision with the engine's
    # ``self._fi_fused_add_rmsnorm`` helper, which forwards through to the
    # real flashinfer kernel.
    _flashinfer_fused_add_rmsnorm = flashinfer.norm.fused_add_rmsnorm
    try:
        from flashinfer.rope import apply_rope_inplace as _fi_apply_rope
        _FI_ROPE_AVAILABLE = True
    except (ImportError, AttributeError):
        _FI_ROPE_AVAILABLE = False
except ImportError:
    _FLASHINFER_AVAILABLE = False
    _FI_ROPE_AVAILABLE = False
    _flashinfer_fused_add_rmsnorm = None  # type: ignore[assignment]


def flashinfer_available() -> bool:
    return _FLASHINFER_AVAILABLE


class FlashInferEngine:
    """FlashInfer engine with paged KV cache for batched block-diffusion."""

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        model,
        max_seq_len: int,
        dtype: torch.dtype,
        batch_size: int = 1,
        page_size: int = 16,
    ):
        if not _FLASHINFER_AVAILABLE:
            raise ImportError(
                "FlashInfer not installed.  Install with: "
                "pip install flashinfer-python"
            )

        device = model.device
        config = model.tfmr.config

        # ---- Hopper PDL (programmatic dependent launch) ----
        # PDL lets back-to-back kernels overlap their epilogue/prologue on
        # Hopper, which helps the memory-bound norm / activation kernels in the
        # decoder hot loop. Auto-detected from the device capability; can be
        # forced via ``RESEMBLETRON_FLASHINFER_PDL`` (``0`` to disable).
        import os
        _pdl_env = os.environ.get("RESEMBLETRON_FLASHINFER_PDL")
        if _pdl_env is not None:
            self._enable_pdl: bool = _pdl_env not in ("0", "false", "False")
        elif device.type == "cuda":
            cap = torch.cuda.get_device_capability(device)
            self._enable_pdl = cap[0] >= 9
        else:
            self._enable_pdl = False
        # ---- fp16 qk reduction for bf16 attention ----
        # Costs one extra cast on QK but lets the tensor cores reduce in fp16,
        # which is materially faster on Hopper. Numerically equivalent in
        # practice for inference-time bf16 models (vLLM/SGLang default).
        self._use_fp16_qk_red: bool = dtype in (torch.float16, torch.bfloat16)

        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)

        B = max(1, batch_size)
        ps = page_size
        P = math.ceil(max_seq_len / ps)
        total_pages = B * P

        # ---- Paged KV caches ----
        self._paged_kv: List[Tensor] = [
            torch.zeros(
                total_pages, 2, ps, num_kv_heads, head_dim,
                device=device, dtype=dtype,
            )
            for _ in range(num_layers)
        ]

        self._max_seq_len = max_seq_len
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._device = device
        self._dtype = dtype
        self._batch_size = B
        self._page_size = ps
        self._pages_per_sample = P

        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        self._q_dim = q_dim
        self._kv_dim = kv_dim
        hidden_size = num_heads * head_dim
        self._hidden_size = hidden_size

        # ---- Fused projection weights ----
        self._fused_qkv_w: List[Tensor] = []
        self._fused_gate_up_w: List[Tensor] = []
        self._o_proj_w: List[Tensor] = []
        self._down_proj_w: List[Tensor] = []
        self._input_ln: list = []
        self._post_ln: list = []
        self._mlp_modules: list = []
        # ---- Cached RMSNorm weight/eps for FlashInfer fused kernels ----
        # ``flashinfer.norm.{rmsnorm, fused_add_rmsnorm}`` needs the raw weight
        # tensor + scalar eps. Caching avoids re-fetching attrs in the hot loop.
        self._input_ln_w: List[Tensor] = []
        self._input_ln_eps: List[float] = []
        self._post_ln_w: List[Tensor] = []
        self._post_ln_eps: List[float] = []

        for layer in model.tfmr.layers:
            attn = layer.self_attn
            mlp = layer.mlp
            self._fused_qkv_w.append(
                torch.cat(
                    [attn.q_proj.weight.data, attn.k_proj.weight.data,
                     attn.v_proj.weight.data], dim=0,
                )
            )
            self._o_proj_w.append(attn.o_proj.weight)
            if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
                self._fused_gate_up_w.append(
                    torch.cat(
                        [mlp.gate_proj.weight.data, mlp.up_proj.weight.data],
                        dim=0,
                    )
                )
                self._down_proj_w.append(mlp.down_proj.weight)
            self._input_ln.append(layer.input_layernorm)
            self._post_ln.append(layer.post_attention_layernorm)
            self._mlp_modules.append(mlp)
            self._input_ln_w.append(layer.input_layernorm.weight.data)
            self._input_ln_eps.append(
                float(layer.input_layernorm.variance_epsilon)
            )
            self._post_ln_w.append(
                layer.post_attention_layernorm.weight.data
            )
            self._post_ln_eps.append(
                float(layer.post_attention_layernorm.variance_epsilon)
            )

        self._mlp_act_fn = model.tfmr.layers[0].mlp.act_fn
        self._final_norm = model.tfmr.norm
        self._final_norm_w = model.tfmr.norm.weight.data
        self._final_norm_eps = float(model.tfmr.norm.variance_epsilon)
        # FlashInfer rmsnorm kernels support fp16 / bf16. fp32 falls back to the
        # legacy LlamaRMSNorm module path.
        self._use_fused_norm: bool = dtype in (torch.float16, torch.bfloat16)
        self._use_fused_silu: bool = (
            self._use_fused_norm and bool(self._fused_gate_up_w)
        )

        # ---- RoPE ----
        # transformers ≥5.9 consolidates rope settings into a single
        # ``config.rope_parameters`` dict and DROPS the top-level
        # ``config.rope_theta`` / ``config.rope_scaling`` attributes; older
        # versions expose them at top level. Read both so we get the real
        # values regardless of layout (the T3 backbone uses theta=500000 with
        # llama3 scaling, both of which now live only in rope_parameters).
        _rope_params = getattr(config, "rope_parameters", None) or {}
        self._rope_theta: float = float(
            getattr(config, "rope_theta", None)
            or _rope_params.get("rope_theta")
            or 10000.0
        )
        # ``flashinfer.rope.apply_rope_inplace`` only implements plain-theta
        # RoPE (``rope_scale=1.0``); it does NOT apply llama3-style frequency
        # scaling. With llama3 scaling the inplace kernel encodes wrong
        # positions and the model degenerates into repetitive output. When any
        # non-plain rope_type is configured, fall back to the precomputed HF
        # cos/sin table below (built from ``model.tfmr.rotary_emb``, which bakes
        # in the scaling) — exactly matching the torch_sdpa path.
        _rope_scaling = getattr(config, "rope_scaling", None) or _rope_params or None
        # transformers has used both "rope_type" (≥4.40) and the legacy "type"
        # key; accept either, and treat anything other than default/linear
        # (e.g. "llama3", "dynamic", "yarn") as needing the table fallback.
        _rope_type = (
            (_rope_scaling.get("rope_type") or _rope_scaling.get("type"))
            if _rope_scaling
            else None
        )
        _plain_rope = _rope_type in (None, "default", "linear")
        self._use_rope_inplace: bool = (
            _FI_ROPE_AVAILABLE and device.type == "cuda" and _plain_rope
        )

        if self._use_rope_inplace:
            self._rope_indptr = torch.zeros(
                B + 1, dtype=torch.int32, device=device,
            )
            self._rope_offset = torch.zeros(
                B, dtype=torch.int32, device=device,
            )
            self._rope_indptr_1 = torch.zeros(
                2, dtype=torch.int32, device=device,
            )
            self._rope_offset_1 = torch.zeros(
                1, dtype=torch.int32, device=device,
            )

        dim = config.hidden_size
        _dummy = torch.zeros(1, 1, dim, device=device, dtype=dtype)
        _full_pos = torch.arange(max_seq_len, device=device).unsqueeze(0)
        self._rope_cos, self._rope_sin = model.tfmr.rotary_emb(
            _dummy, _full_pos,
        )
        self._rope_cos = self._rope_cos.detach()
        self._rope_sin = self._rope_sin.detach()

        from transformers.models.llama.modeling_llama import (
            apply_rotary_pos_emb,
        )
        self._apply_rotary_pos_emb = apply_rotary_pos_emb

        # ---- Per-sample cache state ----
        self._cache_lens: List[int] = [0] * B
        self._prefix_lens: List[int] = [0] * B

        # ---- Eager paged wrapper (block forward without CUDA graph) ----
        self._fi_workspace = torch.empty(
            256 * 1024 * 1024, dtype=torch.uint8, device=device,
        )
        self._fi_eager_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self._fi_workspace, kv_layout="NHD",
        )
        # ---- Ragged prefill wrapper (cross-text prefix forward, eager only) ----
        # Used to fuse per-sample ``single_prefill_with_kv_cache`` calls into one
        # ragged-batched prefill per layer. Has its own workspace so its plan
        # state cannot clobber the paged wrapper.
        self._fi_ragged_workspace = torch.empty(
            64 * 1024 * 1024, dtype=torch.uint8, device=device,
        )
        self._fi_ragged_wrapper = (
            flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                self._fi_ragged_workspace, kv_layout="NHD",
            )
        )

        # ---- CUDA graph state ----
        self._use_cuda_graph: bool = False
        self._cuda_graph = None
        self._graph_pool = None
        self._g_block_size: int = 0
        self._g_inputs_embeds: Tensor | None = None
        self._g_output_hidden: Tensor | None = None
        self._g_cos: Tensor | None = None
        self._g_sin: Tensor | None = None
        self._fi_graph_wrapper = None
        self._g_page_write_idx: Tensor | None = None
        self._g_page_write_off: Tensor | None = None
        self._last_plan_cache_start: int | tuple = -1
        self._plan_qo_indptr: Tensor | None = None
        self._plan_paged_kv_indptr: Tensor | None = None
        self._plan_paged_kv_indices: Tensor | None = None
        self._plan_paged_kv_last_page_len: Tensor | None = None

        # ---- Speech head in graph (optional) ----
        self._has_speech_head: bool = False
        self._speech_head_w: Tensor | None = None
        self._speech_head_b: Tensor | None = None
        self._g_shift_ctx: Tensor | None = None
        self._g_logits: Tensor | None = None

        logger.info(
            "FlashInferEngine: max_seq=%d layers=%d heads=%d/%d dim=%d "
            "batch=%d page_size=%d pages/sample=%d total_pages=%d "
            "rope_inplace=%s dtype=%s",
            max_seq_len, num_layers, num_heads, num_kv_heads, head_dim,
            B, ps, P, total_pages, self._use_rope_inplace, dtype,
        )

    # ------------------------------------------------------------------ #
    #  Page-index helpers
    # ------------------------------------------------------------------ #

    def _compute_page_write_indices(
        self, cache_starts: int | List[int], bl: int,
    ) -> tuple[Tensor, Tensor]:
        """Flat page indices + offsets for writing ``B * bl`` KV entries."""
        B = self._batch_size
        P = self._pages_per_sample
        ps = self._page_size
        dev = self._device

        if isinstance(cache_starts, int):
            pos = torch.arange(cache_starts, cache_starts + bl, device=dev)
            p_in_s = pos // ps
            off = pos % ps
            base = torch.arange(B, device=dev, dtype=torch.long) * P
            return (
                (base[:, None] + p_in_s[None, :]).reshape(-1),
                off.repeat(B),
            )

        parts_idx: list[Tensor] = []
        parts_off: list[Tensor] = []
        for bi in range(B):
            cs = cache_starts[bi]
            pos = torch.arange(cs, cs + bl, device=dev)
            parts_idx.append(bi * P + pos // ps)
            parts_off.append(pos % ps)
        return torch.cat(parts_idx), torch.cat(parts_off)

    def _plan_paged_attention(
        self,
        wrapper,
        kv_lens: int | List[int],
        seq_len: int,
        causal: bool = False,
        *,
        qo_indptr_buf: Tensor | None = None,
        paged_kv_indptr_buf: Tensor | None = None,
        paged_kv_indices_buf: Tensor | None = None,
        paged_kv_last_page_len_buf: Tensor | None = None,
    ):
        """Call ``wrapper.plan()`` with page-table metadata."""
        B = self._batch_size
        P = self._pages_per_sample
        ps = self._page_size
        dev = self._device

        kv_list = [kv_lens] * B if isinstance(kv_lens, int) else list(kv_lens)
        pages_used = [math.ceil(kl / ps) for kl in kv_list]
        total_pu = sum(pages_used)

        use_buf = qo_indptr_buf is not None

        qo = qo_indptr_buf if use_buf else torch.zeros(
            B + 1, dtype=torch.int32, device=dev,
        )
        kvi = paged_kv_indptr_buf if use_buf else torch.zeros(
            B + 1, dtype=torch.int32, device=dev,
        )
        kvidx = paged_kv_indices_buf if use_buf else torch.empty(
            total_pu, dtype=torch.int32, device=dev,
        )
        kvlp = paged_kv_last_page_len_buf if use_buf else torch.empty(
            B, dtype=torch.int32, device=dev,
        )

        if use_buf:
            qo[0] = 0
            kvi[0] = 0

        idx = 0
        for bi in range(B):
            pu = pages_used[bi]
            qo[bi + 1] = qo[bi] + seq_len
            kvi[bi + 1] = kvi[bi] + pu
            kvidx[idx: idx + pu] = torch.arange(
                bi * P, bi * P + pu, dtype=torch.int32, device=dev,
            )
            idx += pu
            last = kv_list[bi] % ps
            kvlp[bi] = last if last > 0 else ps

        wrapper.plan(
            qo_indptr=qo,
            paged_kv_indptr=kvi,
            paged_kv_indices=kvidx[:total_pu],
            paged_kv_last_page_len=kvlp,
            num_qo_heads=self._num_heads,
            num_kv_heads=self._num_kv_heads,
            head_dim_qk=self._head_dim,
            page_size=ps,
            causal=causal,
            q_data_type=self._dtype,
            use_fp16_qk_reduction=self._use_fp16_qk_red,
        )

    # ------------------------------------------------------------------ #
    #  MLP helper (shared by prefix + block + graph paths)
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  FlashInfer kernel wrappers (forward PDL flag once)
    # ------------------------------------------------------------------ #

    def _fi_rmsnorm(self, x: Tensor, w: Tensor, eps: float) -> Tensor:
        return flashinfer.norm.rmsnorm(
            x, w, eps, enable_pdl=self._enable_pdl,
        )

    def _fi_fused_add_rmsnorm(
        self, x: Tensor, residual: Tensor, w: Tensor, eps: float,
    ) -> None:
        _flashinfer_fused_add_rmsnorm(
            x, residual, w, eps, enable_pdl=self._enable_pdl,
        )

    def _mlp_forward(self, layer_idx: int, hidden: Tensor) -> Tensor:
        """Llama SwiGLU MLP. Accepts any leading dims ``(..., dim)``.

        When fused gate-up weight + bf16/fp16 are available, the
        ``silu(gate) * up`` elementwise op is replaced by FlashInfer's
        ``silu_and_mul`` (single fused kernel, 1 read + 1 write of the
        intermediate tensor instead of 2 reads + 1 write).
        """
        if not self._fused_gate_up_w:
            return self._mlp_modules[layer_idx](hidden)

        gate_up = F.linear(hidden, self._fused_gate_up_w[layer_idx])
        if self._use_fused_silu:
            orig_shape = gate_up.shape
            gate_up_2d = gate_up.reshape(-1, orig_shape[-1])
            act_2d = flashinfer.activation.silu_and_mul(
                gate_up_2d, enable_pdl=self._enable_pdl,
            )
            act = act_2d.view(*orig_shape[:-1], act_2d.size(-1))
        else:
            mid = gate_up.size(-1) // 2
            act = self._mlp_act_fn(gate_up[..., :mid]) * gate_up[..., mid:]
        return F.linear(act, self._down_proj_w[layer_idx])

    # ------------------------------------------------------------------ #
    #  Prefix forward  (single-request per sample, paged KV write)
    # ------------------------------------------------------------------ #

    def prefix_forward(
        self,
        prefix_emb_or_list: Tensor | List[Tensor],
        prefix_lens: List[int] | None = None,
    ) -> Tensor:
        """Compute prefix KV and store in paged cache.

        Same-text mode
            ``prefix_emb_or_list`` is ``(1, lp, dim)``.
            Computed once for sample 0, pages copied to samples 1..B-1.

        Cross-text mode
            ``prefix_emb_or_list`` is a list of ``(1, lp_i, dim)``.
            Each sample processed independently.

        Returns:
            ``(B, 1, dim)`` shift context (SOS hidden per sample).
        """
        B = self._batch_size

        if isinstance(prefix_emb_or_list, list):
            # Cross-text: pack all samples into one ragged batched prefill so
            # we issue O(L) kernel launches instead of O(L * B).
            return self._prefix_forward_batched(prefix_emb_or_list)

        emb = prefix_emb_or_list[:1] if prefix_emb_or_list.size(0) > 1 else prefix_emb_or_list
        ctx = self._prefix_forward_one(emb, sample_idx=0)
        lp = prefix_emb_or_list.size(1)
        pfx_pages = math.ceil(lp / self._page_size)
        P = self._pages_per_sample
        for li in range(self._num_layers):
            src = self._paged_kv[li][:pfx_pages]
            for bi in range(1, B):
                s = bi * P
                self._paged_kv[li][s: s + pfx_pages] = src
        for bi in range(1, B):
            self._cache_lens[bi] = lp
            self._prefix_lens[bi] = lp
        return ctx.expand(B, -1, -1).contiguous()

    def _prefix_forward_one(
        self, prefix_emb: Tensor, sample_idx: int,
    ) -> Tensor:
        """Single-sample prefix prefill, writes into ``sample_idx`` page slice.

        Decoder body uses the vLLM-style ``(hidden, residual)`` carry pattern
        with :func:`flashinfer.norm.fused_add_rmsnorm` (in-place residual add
        + RMSNorm in a single kernel). Attention is the optimized
        :func:`flashinfer.single_prefill_with_kv_cache` (causal).
        """
        lp = prefix_emb.size(1)
        page_off = sample_idx * self._pages_per_sample
        ps = self._page_size
        dev = self._device

        pos = torch.arange(lp, device=dev)
        pw_idx = page_off + pos // ps
        pw_off = pos % ps

        if self._use_rope_inplace:
            self._rope_indptr_1[1] = lp
            self._rope_offset_1[0] = 0

        nh = self._num_heads
        nkv = self._num_kv_heads
        hd = self._head_dim
        qd = self._q_dim
        kvd = self._kv_dim

        if self._use_fused_norm:
            return self._prefix_forward_one_fused(
                prefix_emb, sample_idx, lp, pw_idx, pw_off,
                nh, nkv, hd, qd, kvd,
            )

        hidden = prefix_emb

        for li in range(self._num_layers):
            residual = hidden
            hidden = self._input_ln[li](hidden)

            qkv = F.linear(hidden, self._fused_qkv_w[li])
            q = qkv[:, :, :qd].view(1, lp, nh, hd)
            k = qkv[:, :, qd: qd + kvd].view(1, lp, nkv, hd)
            v = qkv[:, :, qd + kvd:].view(1, lp, nkv, hd)

            qs = q.squeeze(0)
            ks = k.squeeze(0)
            vs = v.squeeze(0)

            if self._use_rope_inplace:
                _fi_apply_rope(
                    qs, ks, self._rope_indptr_1, self._rope_offset_1,
                    interleave=False, rope_scale=1.0,
                    rope_theta=self._rope_theta,
                )
            else:
                cos = self._rope_cos[:, :lp, :]
                sin = self._rope_sin[:, :lp, :]
                q, k = self._apply_rotary_pos_emb(
                    q, k, cos, sin, unsqueeze_dim=2,
                )
                qs = q.squeeze(0)
                ks = k.squeeze(0)

            attn_out = flashinfer.single_prefill_with_kv_cache(
                qs, ks, vs, causal=True, kv_layout="NHD",
                use_fp16_qk_reduction=self._use_fp16_qk_red,
            )

            self._paged_kv[li][pw_idx, 0, pw_off] = ks
            self._paged_kv[li][pw_idx, 1, pw_off] = vs

            attn_out = attn_out.unsqueeze(0).reshape(1, lp, -1)
            attn_out = F.linear(attn_out, self._o_proj_w[li])

            hidden = residual + attn_out
            residual = hidden
            hidden = self._post_ln[li](hidden)
            hidden = self._mlp_forward(li, hidden)
            hidden = residual + hidden

        hidden = self._final_norm(hidden)
        self._cache_lens[sample_idx] = lp
        self._prefix_lens[sample_idx] = lp
        return hidden[:, lp - 1: lp, :]

    def _prefix_forward_one_fused(
        self,
        prefix_emb: Tensor,
        sample_idx: int,
        lp: int,
        pw_idx: Tensor,
        pw_off: Tensor,
        nh: int,
        nkv: int,
        hd: int,
        qd: int,
        kvd: int,
    ) -> Tensor:
        """fused_add_rmsnorm + silu_and_mul variant of :meth:`_prefix_forward_one`."""
        L = self._num_layers
        # Work in 2D ``(lp, dim)`` so fused_add_rmsnorm (which wants 2D contig)
        # and silu_and_mul (any leading dims) both apply cleanly.
        # Clone so we don't mutate the caller's ``prefix_emb``.
        residual = prefix_emb.reshape(lp, -1).contiguous().clone()
        hidden = self._fi_rmsnorm(
            residual, self._input_ln_w[0], self._input_ln_eps[0],
        )

        for li in range(L):
            qkv = F.linear(hidden, self._fused_qkv_w[li])
            qs = qkv[:, :qd].view(lp, nh, hd)
            ks = qkv[:, qd: qd + kvd].view(lp, nkv, hd)
            vs = qkv[:, qd + kvd:].view(lp, nkv, hd)

            if self._use_rope_inplace:
                _fi_apply_rope(
                    qs, ks, self._rope_indptr_1, self._rope_offset_1,
                    interleave=False, rope_scale=1.0,
                    rope_theta=self._rope_theta,
                )
            else:
                # Apply RoPE via HF helper (expects ``(B, S, h, hd)``).
                q3 = qs.unsqueeze(0)
                k3 = ks.unsqueeze(0)
                cos = self._rope_cos[:, :lp, :]
                sin = self._rope_sin[:, :lp, :]
                q3, k3 = self._apply_rotary_pos_emb(
                    q3, k3, cos, sin, unsqueeze_dim=2,
                )
                qs = q3.squeeze(0)
                ks = k3.squeeze(0)

            attn_out = flashinfer.single_prefill_with_kv_cache(
                qs, ks, vs, causal=True, kv_layout="NHD",
                use_fp16_qk_reduction=self._use_fp16_qk_red,
            )

            self._paged_kv[li][pw_idx, 0, pw_off] = ks
            self._paged_kv[li][pw_idx, 1, pw_off] = vs

            attn_out_2d = attn_out.reshape(lp, -1)
            attn_out_2d = F.linear(attn_out_2d, self._o_proj_w[li]).contiguous()

            # residual += attn_out; attn_out = rmsnorm(residual, post_ln_w)
            self._fi_fused_add_rmsnorm(
                attn_out_2d, residual,
                self._post_ln_w[li], self._post_ln_eps[li],
            )

            mlp_out = self._mlp_forward(li, attn_out_2d)
            if not mlp_out.is_contiguous():
                mlp_out = mlp_out.contiguous()

            if li + 1 < L:
                self._fi_fused_add_rmsnorm(
                    mlp_out, residual,
                    self._input_ln_w[li + 1], self._input_ln_eps[li + 1],
                )
                hidden = mlp_out
            else:
                residual.add_(mlp_out)
                hidden = residual

        # Final norm and per-sample SOS hidden.
        out = self._fi_rmsnorm(
            hidden, self._final_norm_w, self._final_norm_eps,
        )
        self._cache_lens[sample_idx] = lp
        self._prefix_lens[sample_idx] = lp
        return out[lp - 1: lp, :].view(1, 1, -1)

    # ------------------------------------------------------------------ #
    #  Cross-text ragged batched prefix forward
    # ------------------------------------------------------------------ #

    def _prefix_forward_batched(
        self, emb_list: List[Tensor],
    ) -> Tensor:
        """Cross-text prefix prefill via ragged batched attention.

        Replaces the per-sample loop of
        :meth:`_prefix_forward_one` with a single
        :class:`flashinfer.BatchPrefillWithRaggedKVCacheWrapper.run`
        per layer. Sample ``bi`` may have its own prefix length ``lp_i``;
        embeddings are packed into a flat ``(sum_i lp_i, dim)`` tensor.

        Each sample's KV is scattered into its own paged-cache slice
        ``[bi*P : bi*P + ceil(lp_i/ps)]``.

        Returns:
            ``(B_actual, 1, dim)`` per-sample SOS hidden.
        """
        B_actual = len(emb_list)
        if B_actual == 1:
            # Single sample: paged ragged setup overhead is wasteful. Reuse the
            # well-tuned single-sample path which calls single_prefill directly.
            return self._prefix_forward_one(emb_list[0], 0)

        dev = self._device
        ps = self._page_size
        P = self._pages_per_sample
        nh = self._num_heads
        nkv = self._num_kv_heads
        hd = self._head_dim
        qd = self._q_dim
        kvd = self._kv_dim
        L = self._num_layers
        dtype = self._dtype

        lp_list = [int(e.size(1)) for e in emb_list]
        total_lp = sum(lp_list)

        # Pack embeddings to (total_lp, dim).
        packed = torch.cat(
            [e.reshape(-1, e.size(-1)) for e in emb_list], dim=0,
        ).to(dtype=dtype).contiguous()

        # qo_indptr == kv_indptr for self-attention prefill.
        indptr_host = [0]
        for lp_i in lp_list:
            indptr_host.append(indptr_host[-1] + lp_i)
        qo_indptr = torch.tensor(
            indptr_host, dtype=torch.int32, device=dev,
        )

        # Paged-write indices: sample bi's positions [0, lp_i) → pages
        # [bi*P, bi*P + ceil(lp_i/ps)).
        pw_idx_parts: list[Tensor] = []
        pw_off_parts: list[Tensor] = []
        for bi, lp_i in enumerate(lp_list):
            pos = torch.arange(lp_i, device=dev)
            pw_idx_parts.append(bi * P + pos // ps)
            pw_off_parts.append(pos % ps)
        pw_idx = torch.cat(pw_idx_parts)
        pw_off = torch.cat(pw_off_parts)

        # RoPE: each sample starts from position 0 (prefix). Reuse indptr as
        # rope_indptr; offsets are all zero. ``apply_rope_inplace`` wants int32.
        if self._use_rope_inplace:
            rope_indptr = qo_indptr.to(torch.int32)
            rope_offset = torch.zeros(
                B_actual, dtype=torch.int32, device=dev,
            )
        else:
            # Pre-build packed cos/sin once; shape (total_lp, head_dim).
            cos_parts = [self._rope_cos[0, :lp_i, :] for lp_i in lp_list]
            sin_parts = [self._rope_sin[0, :lp_i, :] for lp_i in lp_list]
            packed_cos = torch.cat(cos_parts, dim=0)
            packed_sin = torch.cat(sin_parts, dim=0)

        # Plan ragged wrapper once for the whole layer stack (causal).
        self._fi_ragged_wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=qo_indptr,
            num_qo_heads=nh,
            num_kv_heads=nkv,
            head_dim_qk=hd,
            causal=True,
            q_data_type=dtype,
            use_fp16_qk_reduction=self._use_fp16_qk_red,
        )

        use_fused = self._use_fused_norm
        if use_fused:
            residual = packed.clone()
            hidden = self._fi_rmsnorm(
                residual, self._input_ln_w[0], self._input_ln_eps[0],
            )
        else:
            hidden = packed  # (total_lp, dim)

        for li in range(L):
            if use_fused:
                qkv = F.linear(hidden, self._fused_qkv_w[li])
            else:
                # Module-style norm requires 3D ``(1, total_lp, dim)``.
                normed = self._input_ln[li](hidden.unsqueeze(0)).squeeze(0)
                qkv = F.linear(normed, self._fused_qkv_w[li])

            qs = qkv[:, :qd].view(total_lp, nh, hd)
            ks = qkv[:, qd: qd + kvd].view(total_lp, nkv, hd)
            vs = qkv[:, qd + kvd:].view(total_lp, nkv, hd)

            if self._use_rope_inplace:
                _fi_apply_rope(
                    qs, ks, rope_indptr, rope_offset,
                    interleave=False, rope_scale=1.0,
                    rope_theta=self._rope_theta,
                )
            else:
                # ``apply_rotary_pos_emb`` expects ``(B, S, h, hd)``; mimic with
                # a synthetic batch dim over the packed sequence.
                cos = packed_cos.unsqueeze(0)
                sin = packed_sin.unsqueeze(0)
                q3 = qs.unsqueeze(0)
                k3 = ks.unsqueeze(0)
                q3, k3 = self._apply_rotary_pos_emb(
                    q3, k3, cos, sin, unsqueeze_dim=2,
                )
                qs = q3.squeeze(0)
                ks = k3.squeeze(0)

            self._paged_kv[li][pw_idx, 0, pw_off] = ks
            self._paged_kv[li][pw_idx, 1, pw_off] = vs

            attn_out = self._fi_ragged_wrapper.run(qs, ks, vs)
            attn_out_2d = attn_out.reshape(total_lp, -1)
            attn_out_2d = F.linear(
                attn_out_2d, self._o_proj_w[li],
            ).contiguous()

            if use_fused:
                self._fi_fused_add_rmsnorm(
                    attn_out_2d, residual,
                    self._post_ln_w[li], self._post_ln_eps[li],
                )
                mlp_out = self._mlp_forward(li, attn_out_2d)
                if not mlp_out.is_contiguous():
                    mlp_out = mlp_out.contiguous()
                if li + 1 < L:
                    self._fi_fused_add_rmsnorm(
                        mlp_out, residual,
                        self._input_ln_w[li + 1], self._input_ln_eps[li + 1],
                    )
                    hidden = mlp_out
                else:
                    residual.add_(mlp_out)
                    hidden = residual
            else:
                # Non-fused fallback: keep classic residual carry.
                hidden = hidden + attn_out_2d  # residual after attn
                residual_after_attn = hidden
                hidden_n = self._post_ln[li](hidden.unsqueeze(0)).squeeze(0)
                mlp_out = self._mlp_forward(li, hidden_n)
                hidden = residual_after_attn + mlp_out

        if use_fused:
            out = self._fi_rmsnorm(
                hidden, self._final_norm_w, self._final_norm_eps,
            )
        else:
            out = self._final_norm(hidden.unsqueeze(0)).squeeze(0)

        # Per-sample SOS hidden = out[lp_i - 1] within each sample's slice.
        shift_idx = (qo_indptr[1:] - 1).to(torch.long)
        shift_ctx = out.index_select(0, shift_idx).unsqueeze(1)  # (B_actual, 1, dim)

        for bi, lp_i in enumerate(lp_list):
            self._cache_lens[bi] = lp_i
            self._prefix_lens[bi] = lp_i

        return shift_ctx.contiguous()

    # ------------------------------------------------------------------ #
    #  Batched block forward  (non-causal, paged attention)
    # ------------------------------------------------------------------ #

    def block_forward(
        self,
        block_emb: Tensor,
        cache_starts: int | List[int],
    ) -> Tensor:
        """Batched block forward with non-causal paged attention.

        Args:
            block_emb:    ``(B, bl, dim)``
            cache_starts: int (uniform) or list[int] (per-sample).

        Returns:
            ``(B, bl, dim)`` after final RMSNorm.
        """
        B = self._batch_size
        bl = block_emb.size(1)

        pw_idx, pw_off = self._compute_page_write_indices(cache_starts, bl)

        if isinstance(cache_starts, int):
            kv_lens: int | List[int] = cache_starts + bl
        else:
            kv_lens = [cs + bl for cs in cache_starts]

        self._plan_paged_attention(
            self._fi_eager_wrapper, kv_lens, seq_len=bl, causal=False,
        )

        if self._use_rope_inplace:
            for bi in range(B):
                self._rope_indptr[bi + 1] = (bi + 1) * bl
            if isinstance(cache_starts, int):
                self._rope_offset[:] = cache_starts
            else:
                for bi in range(B):
                    self._rope_offset[bi] = cache_starts[bi]

        nh = self._num_heads
        nkv = self._num_kv_heads
        hd = self._head_dim
        qd = self._q_dim
        kvd = self._kv_dim

        if self._use_fused_norm:
            return self._block_forward_fused(
                block_emb, cache_starts, B, bl, pw_idx, pw_off,
                nh, nkv, hd, qd, kvd,
            )

        hidden = block_emb

        for li in range(self._num_layers):
            residual = hidden
            hidden = self._input_ln[li](hidden)

            qkv = F.linear(hidden, self._fused_qkv_w[li])
            q = qkv[:, :, :qd].view(B, bl, nh, hd)
            k = qkv[:, :, qd: qd + kvd].view(B, bl, nkv, hd)
            v = qkv[:, :, qd + kvd:].view(B, bl, nkv, hd)

            qs = q.reshape(B * bl, nh, hd)
            ks = k.reshape(B * bl, nkv, hd)
            vs = v.reshape(B * bl, nkv, hd)

            if self._use_rope_inplace:
                _fi_apply_rope(
                    qs, ks, self._rope_indptr, self._rope_offset,
                    interleave=False, rope_scale=1.0,
                    rope_theta=self._rope_theta,
                )
            else:
                if isinstance(cache_starts, int):
                    cos = self._rope_cos[
                        :, cache_starts: cache_starts + bl, :
                    ]
                    sin = self._rope_sin[
                        :, cache_starts: cache_starts + bl, :
                    ]
                else:
                    cos = torch.cat([
                        self._rope_cos[:, cs: cs + bl, :]
                        for cs in cache_starts
                    ], dim=0)
                    sin = torch.cat([
                        self._rope_sin[:, cs: cs + bl, :]
                        for cs in cache_starts
                    ], dim=0)
                q, k = self._apply_rotary_pos_emb(
                    q, k, cos, sin, unsqueeze_dim=2,
                )
                qs = q.reshape(B * bl, nh, hd)
                ks = k.reshape(B * bl, nkv, hd)

            self._paged_kv[li][pw_idx, 0, pw_off] = ks
            self._paged_kv[li][pw_idx, 1, pw_off] = vs

            attn_out = self._fi_eager_wrapper.run(
                qs, self._paged_kv[li],
            )
            attn_out = attn_out.reshape(B, bl, -1)
            attn_out = F.linear(attn_out, self._o_proj_w[li])

            hidden = residual + attn_out
            residual = hidden
            hidden = self._post_ln[li](hidden)
            hidden = self._mlp_forward(li, hidden)
            hidden = residual + hidden

        return self._final_norm(hidden)

    def _block_forward_fused(
        self,
        block_emb: Tensor,
        cache_starts: int | List[int],
        B: int,
        bl: int,
        pw_idx: Tensor,
        pw_off: Tensor,
        nh: int,
        nkv: int,
        hd: int,
        qd: int,
        kvd: int,
    ) -> Tensor:
        """``block_forward`` body with fused_add_rmsnorm + silu_and_mul.

        Carries ``(hidden, residual)`` as 2D ``(B*bl, dim)``. ``residual`` is
        a private clone of ``block_emb`` so the caller's tensor is not mutated.
        """
        L = self._num_layers
        N = B * bl

        residual = block_emb.reshape(N, -1).contiguous().clone()
        hidden = self._fi_rmsnorm(
            residual, self._input_ln_w[0], self._input_ln_eps[0],
        )

        for li in range(L):
            qkv = F.linear(hidden, self._fused_qkv_w[li])
            qs = qkv[:, :qd].view(N, nh, hd)
            ks = qkv[:, qd: qd + kvd].view(N, nkv, hd)
            vs = qkv[:, qd + kvd:].view(N, nkv, hd)

            if self._use_rope_inplace:
                _fi_apply_rope(
                    qs, ks, self._rope_indptr, self._rope_offset,
                    interleave=False, rope_scale=1.0,
                    rope_theta=self._rope_theta,
                )
            else:
                if isinstance(cache_starts, int):
                    cos = self._rope_cos[
                        :, cache_starts: cache_starts + bl, :
                    ]
                    sin = self._rope_sin[
                        :, cache_starts: cache_starts + bl, :
                    ]
                else:
                    cos = torch.cat([
                        self._rope_cos[:, cs: cs + bl, :]
                        for cs in cache_starts
                    ], dim=0)
                    sin = torch.cat([
                        self._rope_sin[:, cs: cs + bl, :]
                        for cs in cache_starts
                    ], dim=0)
                q3 = qs.view(B, bl, nh, hd)
                k3 = ks.view(B, bl, nkv, hd)
                q3, k3 = self._apply_rotary_pos_emb(
                    q3, k3, cos, sin, unsqueeze_dim=2,
                )
                qs = q3.reshape(N, nh, hd)
                ks = k3.reshape(N, nkv, hd)

            self._paged_kv[li][pw_idx, 0, pw_off] = ks
            self._paged_kv[li][pw_idx, 1, pw_off] = vs

            attn_out = self._fi_eager_wrapper.run(
                qs, self._paged_kv[li],
            )
            attn_out_2d = attn_out.reshape(N, -1)
            attn_out_2d = F.linear(attn_out_2d, self._o_proj_w[li]).contiguous()

            self._fi_fused_add_rmsnorm(
                attn_out_2d, residual,
                self._post_ln_w[li], self._post_ln_eps[li],
            )

            mlp_out = self._mlp_forward(li, attn_out_2d)
            if not mlp_out.is_contiguous():
                mlp_out = mlp_out.contiguous()

            if li + 1 < L:
                self._fi_fused_add_rmsnorm(
                    mlp_out, residual,
                    self._input_ln_w[li + 1], self._input_ln_eps[li + 1],
                )
                hidden = mlp_out
            else:
                residual.add_(mlp_out)
                hidden = residual

        out = self._fi_rmsnorm(
            hidden, self._final_norm_w, self._final_norm_eps,
        )
        return out.view(B, bl, -1)

    # ------------------------------------------------------------------ #
    #  Cache management
    # ------------------------------------------------------------------ #

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def cache_len(self) -> int:
        """Backward-compat: cache length of sample 0."""
        return self._cache_lens[0]

    @cache_len.setter
    def cache_len(self, value: int):
        self._cache_lens = [value] * self._batch_size

    @property
    def cache_lens(self) -> List[int]:
        return self._cache_lens

    @property
    def prefix_lens(self) -> List[int]:
        return self._prefix_lens

    def advance_cache(self, block_len: int):
        for bi in range(self._batch_size):
            self._cache_lens[bi] += block_len

    def reset(self):
        B = self._batch_size
        self._cache_lens = [0] * B
        self._prefix_lens = [0] * B
        self._last_plan_cache_start = -1
        # CUDA graph / pool are kept across generate() calls. Per-call prefix and
        # paged-KV state are rebuilt in prefix_forward; block_forward_graph replans
        # when cache_starts change. capture_cuda_graph early-returns when the same
        # block_size is already captured (avoids allocator issues + recapture cost).

    def snapshot_kv(self):
        """Deep copy of paged KV + lengths for :meth:`restore_kv` (CFG uncond pass)."""
        kv_copy = [t.clone() for t in self._paged_kv]
        return (
            kv_copy,
            list(self._cache_lens),
            list(self._prefix_lens),
            int(getattr(self, "_last_plan_cache_start", -1)),
        )

    def restore_kv(self, snap) -> None:
        """Restore state from :meth:`snapshot_kv`."""
        kv_copy, cache_lens, prefix_lens, last_plan = snap
        for li, t in enumerate(kv_copy):
            self._paged_kv[li].copy_(t)
        self._cache_lens = cache_lens
        self._prefix_lens = prefix_lens
        self._last_plan_cache_start = last_plan

    # ------------------------------------------------------------------ #
    #  CUDA Graph
    # ------------------------------------------------------------------ #

    @property
    def has_cuda_graph(self) -> bool:
        return self._use_cuda_graph

    def can_reuse(
        self,
        max_seq_len: int,
        dtype: torch.dtype,
        batch_size: int = 1,
        page_size: int = 16,
    ) -> bool:
        return (
            self._max_seq_len >= max_seq_len
            and self._dtype == dtype
            and self._batch_size == max(1, batch_size)
            and self._page_size == page_size
        )

    def capture_cuda_graph(
        self,
        block_size: int,
        speech_head: "torch.nn.Linear | None" = None,
    ):
        if self._use_cuda_graph and self._g_block_size == block_size:
            return

        import gc
        import os
        import shutil
        import glob as glob_mod

        device = self._device
        dtype = self._dtype
        bl = block_size
        B = self._batch_size
        P = self._pages_per_sample
        ps = self._page_size
        hs = self._hidden_size

        cache_root = os.path.expanduser("~/.cache/flashinfer")
        for partial in glob_mod.glob(
            os.path.join(cache_root, "**/cached_ops/batch_prefill_*"),
            recursive=True,
        ):
            if os.path.isdir(partial):
                so_files = glob_mod.glob(os.path.join(partial, "*.so"))
                if not so_files:
                    logger.info(
                        "FlashInferEngine: cleaning partial JIT: %s", partial,
                    )
                    shutil.rmtree(partial, ignore_errors=True)

        # ---- Plan buffers (pre-allocated for CUDA-graph wrapper) ----
        max_plan_pages = B * P
        self._plan_qo_indptr = torch.zeros(
            B + 1, dtype=torch.int32, device=device,
        )
        self._plan_paged_kv_indptr = torch.zeros(
            B + 1, dtype=torch.int32, device=device,
        )
        self._plan_paged_kv_indices = torch.zeros(
            max_plan_pages, dtype=torch.int32, device=device,
        )
        self._plan_paged_kv_last_page_len = torch.zeros(
            B, dtype=torch.int32, device=device,
        )

        fi_ws = torch.empty(
            256 * 1024 * 1024, dtype=torch.uint8, device=device,
        )
        self._fi_graph_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            fi_ws,
            kv_layout="NHD",
            use_cuda_graph=True,
            qo_indptr_buf=self._plan_qo_indptr,
            paged_kv_indptr_buf=self._plan_paged_kv_indptr,
            paged_kv_indices_buf=self._plan_paged_kv_indices,
            paged_kv_last_page_len_buf=self._plan_paged_kv_last_page_len,
        )

        # ---- Graph IO ----
        self._g_inputs_embeds = torch.zeros(
            B, bl, hs, device=device, dtype=dtype,
        )
        self._g_output_hidden = torch.zeros(
            B, bl, hs, device=device, dtype=dtype,
        )
        self._g_block_size = bl

        uniform_cache = len(set(self._cache_lens)) <= 1
        if uniform_cache:
            cs0 = self._cache_lens[0]
            pw_idx, pw_off = self._compute_page_write_indices(cs0, bl)
            kv_plan: int | List[int] = cs0 + bl
        else:
            starts = list(self._cache_lens)
            pw_idx, pw_off = self._compute_page_write_indices(starts, bl)
            kv_plan = [c + bl for c in self._cache_lens]

        self._g_page_write_idx = pw_idx.clone()
        self._g_page_write_off = pw_off.clone()

        if self._use_rope_inplace:
            for bi in range(B):
                self._rope_indptr[bi + 1] = (bi + 1) * bl
            if uniform_cache:
                self._rope_offset[:] = self._cache_lens[0]
            else:
                for bi in range(B):
                    self._rope_offset[bi] = self._cache_lens[bi]
        else:
            self._g_cos = torch.zeros(
                B, bl, self._head_dim, device=device, dtype=dtype,
            )
            self._g_sin = torch.zeros(
                B, bl, self._head_dim, device=device, dtype=dtype,
            )
            for bi in range(B):
                cs_bi = self._cache_lens[bi]
                self._g_cos[bi].copy_(self._rope_cos[0, cs_bi: cs_bi + bl, :])
                self._g_sin[bi].copy_(self._rope_sin[0, cs_bi: cs_bi + bl, :])

        # ---- Speech head in graph ----
        if speech_head is not None:
            self._speech_head_w = speech_head.weight.detach()
            self._speech_head_b = (
                speech_head.bias.detach()
                if speech_head.bias is not None
                else None
            )
            vocab_size = speech_head.weight.size(0)
            self._g_shift_ctx = torch.zeros(
                B, 1, hs, device=device, dtype=dtype,
            )
            self._g_logits = torch.zeros(
                B, bl, vocab_size, device=device, dtype=dtype,
            )
            self._has_speech_head = True
            logger.info(
                "FlashInferEngine: speech_head in graph (vocab=%d)",
                vocab_size,
            )

        # ---- Initial plan ----
        logger.info(
            "FlashInferEngine: CUDA graph JIT compiling batch_prefill "
            "(MAX_JOBS=%s) ...",
            os.environ.get("MAX_JOBS", "unset"),
        )
        self._plan_paged_attention(
            self._fi_graph_wrapper, kv_plan, seq_len=bl, causal=False,
            qo_indptr_buf=self._plan_qo_indptr,
            paged_kv_indptr_buf=self._plan_paged_kv_indptr,
            paged_kv_indices_buf=self._plan_paged_kv_indices,
            paged_kv_last_page_len_buf=self._plan_paged_kv_last_page_len,
        )
        # Tuple matches :meth:`block_forward_graph` when ``cache_starts`` is a list;
        # same-text int ``cache_starts`` triggers one replan on first replay (harmless).
        self._last_plan_cache_start = tuple(self._cache_lens)

        # ---- Warmup ----
        logger.info("FlashInferEngine: CUDA graph warmup forward ...")
        self._graph_body()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.synchronize()

        # ---- Capture ----
        logger.info("FlashInferEngine: CUDA graph capturing ...")
        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph, pool=self._graph_pool):
            self._graph_body()

        if self._graph_pool is None:
            self._graph_pool = self._cuda_graph.pool()

        self._use_cuda_graph = True
        logger.info(
            "FlashInferEngine: CUDA graph captured "
            "(block_size=%d, batch=%d, speech_head=%s)",
            bl, B, self._has_speech_head,
        )

    def _graph_body(self):
        """Graph-capturable batched forward for block denoising.

        In fused-norm mode the layer body carries ``(hidden, residual)`` as 2D
        ``(B*bl, dim)`` and uses :func:`flashinfer.norm.fused_add_rmsnorm` for
        every "residual add + RMSNorm" pair. The persistent
        ``self._g_inputs_embeds`` buffer doubles as the running residual; it is
        overwritten via ``copy_(block_emb)`` before each replay, so in-place
        mutation inside the graph is safe.
        """
        B = self._batch_size
        bl = self._g_block_size
        nh = self._num_heads
        nkv = self._num_kv_heads
        hd = self._head_dim
        qd = self._q_dim
        kvd = self._kv_dim
        use_ri = self._use_rope_inplace
        L = self._num_layers
        N = B * bl

        if self._use_fused_norm:
            # 2D view of the persistent input buffer; in-place mutation OK
            # because the buffer is reset every replay (see ``block_forward_graph``).
            residual = self._g_inputs_embeds.view(N, -1)
            hidden = self._fi_rmsnorm(
                residual, self._input_ln_w[0], self._input_ln_eps[0],
            )

            for li in range(L):
                qkv = F.linear(hidden, self._fused_qkv_w[li])
                qs = qkv[:, :qd].view(N, nh, hd)
                ks = qkv[:, qd: qd + kvd].view(N, nkv, hd)
                vs = qkv[:, qd + kvd:].view(N, nkv, hd)

                if use_ri:
                    _fi_apply_rope(
                        qs, ks, self._rope_indptr, self._rope_offset,
                        interleave=False, rope_scale=1.0,
                        rope_theta=self._rope_theta,
                    )
                else:
                    q3 = qs.view(B, bl, nh, hd)
                    k3 = ks.view(B, bl, nkv, hd)
                    q3, k3 = self._apply_rotary_pos_emb(
                        q3, k3, self._g_cos, self._g_sin, unsqueeze_dim=2,
                    )
                    qs = q3.reshape(N, nh, hd)
                    ks = k3.reshape(N, nkv, hd)

                self._paged_kv[li][
                    self._g_page_write_idx, 0, self._g_page_write_off
                ] = ks
                self._paged_kv[li][
                    self._g_page_write_idx, 1, self._g_page_write_off
                ] = vs

                attn_out = self._fi_graph_wrapper.run(
                    qs, self._paged_kv[li],
                )
                attn_out_2d = attn_out.reshape(N, -1)
                attn_out_2d = F.linear(
                    attn_out_2d, self._o_proj_w[li],
                ).contiguous()

                self._fi_fused_add_rmsnorm(
                    attn_out_2d, residual,
                    self._post_ln_w[li], self._post_ln_eps[li],
                )

                mlp_out = self._mlp_forward(li, attn_out_2d)
                if not mlp_out.is_contiguous():
                    mlp_out = mlp_out.contiguous()

                if li + 1 < L:
                    self._fi_fused_add_rmsnorm(
                        mlp_out, residual,
                        self._input_ln_w[li + 1], self._input_ln_eps[li + 1],
                    )
                    hidden = mlp_out
                else:
                    residual.add_(mlp_out)
                    hidden = residual

            out = self._fi_rmsnorm(
                hidden, self._final_norm_w, self._final_norm_eps,
            )
            self._g_output_hidden[:] = out.view(B, bl, -1)
        else:
            hidden = self._g_inputs_embeds

            for li in range(L):
                residual = hidden
                hidden = self._input_ln[li](hidden)

                qkv = F.linear(hidden, self._fused_qkv_w[li])
                q = qkv[:, :, :qd].view(B, bl, nh, hd)
                k = qkv[:, :, qd: qd + kvd].view(B, bl, nkv, hd)
                v = qkv[:, :, qd + kvd:].view(B, bl, nkv, hd)

                qs = q.reshape(B * bl, nh, hd)
                ks = k.reshape(B * bl, nkv, hd)
                vs = v.reshape(B * bl, nkv, hd)

                if use_ri:
                    _fi_apply_rope(
                        qs, ks, self._rope_indptr, self._rope_offset,
                        interleave=False, rope_scale=1.0,
                        rope_theta=self._rope_theta,
                    )
                else:
                    cos = self._g_cos
                    sin = self._g_sin
                    q, k = self._apply_rotary_pos_emb(
                        q, k, cos, sin, unsqueeze_dim=2,
                    )
                    qs = q.reshape(B * bl, nh, hd)
                    ks = k.reshape(B * bl, nkv, hd)

                self._paged_kv[li][
                    self._g_page_write_idx, 0, self._g_page_write_off
                ] = ks
                self._paged_kv[li][
                    self._g_page_write_idx, 1, self._g_page_write_off
                ] = vs

                attn_out = self._fi_graph_wrapper.run(
                    qs, self._paged_kv[li],
                )
                attn_out = attn_out.reshape(B, bl, -1)
                attn_out = F.linear(attn_out, self._o_proj_w[li])

                hidden = residual + attn_out
                residual = hidden
                hidden = self._post_ln[li](hidden)
                hidden = self._mlp_forward(li, hidden)
                hidden = residual + hidden

            self._g_output_hidden[:] = self._final_norm(hidden)

        if self._has_speech_head:
            shift_hidden = torch.cat(
                [self._g_shift_ctx, self._g_output_hidden[:, : bl - 1]],
                dim=1,
            )
            self._g_logits[:] = F.linear(
                shift_hidden,
                self._speech_head_w,
                self._speech_head_b,
            )

    def set_shift_ctx(self, ctx: Tensor):
        """Copy shift context into graph buffer (once per block).

        Args:
            ctx: ``(B, 1, dim)`` or ``(1, 1, dim)`` (broadcast).
        """
        if self._g_shift_ctx is not None:
            if ctx.size(0) < self._batch_size:
                self._g_shift_ctx[:] = ctx.expand(
                    self._batch_size, -1, -1,
                )
            else:
                self._g_shift_ctx.copy_(ctx)

    def block_forward_graph(
        self,
        block_emb: Tensor,
        cache_starts: int | List[int],
    ) -> tuple[Tensor, Tensor | None]:
        """Batched block forward via CUDA graph replay.

        Falls back to eager :meth:`block_forward` for partial blocks.

        Args:
            block_emb:    ``(B, bl, dim)``
            cache_starts: int (uniform) or list[int] per-sample.

        Returns:
            ``(hidden, logits)`` — ``(B, bl, dim)`` and
            ``(B, bl, vocab)`` or ``None``.
        """
        bl = block_emb.size(1)
        if bl != self._g_block_size:
            return self.block_forward(block_emb, cache_starts), None

        self._g_inputs_embeds.copy_(block_emb)

        cs_key: int | tuple
        if isinstance(cache_starts, int):
            cs_key = cache_starts
        else:
            cs_key = tuple(cache_starts)

        if cs_key != self._last_plan_cache_start:
            B = self._batch_size
            bl_ = bl

            pw_idx, pw_off = self._compute_page_write_indices(
                cache_starts, bl_,
            )
            self._g_page_write_idx.copy_(pw_idx)
            self._g_page_write_off.copy_(pw_off)

            if self._use_rope_inplace:
                for bi in range(B):
                    self._rope_indptr[bi + 1] = (bi + 1) * bl_
                if isinstance(cache_starts, int):
                    self._rope_offset[:] = cache_starts
                else:
                    for bi in range(B):
                        self._rope_offset[bi] = cache_starts[bi]
            else:
                if isinstance(cache_starts, int):
                    for bi in range(B):
                        self._g_cos[bi].copy_(
                            self._rope_cos[0, cache_starts: cache_starts + bl_, :],
                        )
                        self._g_sin[bi].copy_(
                            self._rope_sin[0, cache_starts: cache_starts + bl_, :],
                        )
                else:
                    for bi in range(B):
                        csb = cache_starts[bi]
                        self._g_cos[bi].copy_(
                            self._rope_cos[0, csb: csb + bl_, :],
                        )
                        self._g_sin[bi].copy_(
                            self._rope_sin[0, csb: csb + bl_, :],
                        )

            if isinstance(cache_starts, int):
                kv_lens: int | List[int] = cache_starts + bl_
            else:
                kv_lens = [c + bl_ for c in cache_starts]

            self._plan_paged_attention(
                self._fi_graph_wrapper, kv_lens, seq_len=bl_,
                causal=False,
                qo_indptr_buf=self._plan_qo_indptr,
                paged_kv_indptr_buf=self._plan_paged_kv_indptr,
                paged_kv_indices_buf=self._plan_paged_kv_indices,
                paged_kv_last_page_len_buf=self._plan_paged_kv_last_page_len,
            )
            self._last_plan_cache_start = cs_key

        self._cuda_graph.replay()

        logits = (
            self._g_logits if self._has_speech_head else None
        )
        return self._g_output_hidden, logits
