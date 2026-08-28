"""Continuous-batching inference engine for Chatterbox-Flash on ROCm.

Replaces the lock-step ``ChatterboxFlashT3.generate()`` loop with a slot-based
scheduler: every T3 "tick" runs one batched block-diffusion step across all
active requests, each of which may sit at a different block / denoise-step /
sequence position. Requests join at any tick boundary (batched ragged prefill)
and leave on EOS, freeing their KV slot immediately.

Faithful to the stock decoding math (paper "best" config):
  * OmniVoice r_n count schedule (t_shift) as per-step unmask floor
  * PMI quantile early decoding (time_shift_tau)
  * precomputed unconditional block prior as PMI denominator
  * zero_text_batch CFG (cond + zero_all null rows) with pmi_cfg combination
  * gumbel position noise when position_temperature > 0

Deviations (distribution-preserving):
  * token sampling uses gumbel-max instead of torch.multinomial (identical
    categorical distribution, no host sync)
  * all per-row Python loops in the unmask step are vectorized across the batch
  * softmax/log/quantile math runs in fp32 (stock code mixes bf16/fp32)

The Llama backbone forward is re-implemented directly on the module weights
(qkv/o/mlp/rmsnorm/rope) with a preallocated per-slot KV cache written via
index_put_ at per-row offsets — no HuggingFace DynamicCache, no per-tick
allocation churn.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from server.scheduler import SchedulerQueue

logger = logging.getLogger(__name__)

NEG_INF = float("-inf")


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #


@dataclass
class Request:
    rid: int
    text: str
    text_tokens: Tensor              # (1, Lt) long, CPU
    n_speech: int                    # speech-token budget, multiple of 16
    # streaming sink: callable(bytes | None); None = end of stream
    sink: object = None
    # Dynamic LoRA model selection. Slot zero is the base model.
    lora_slot: int = 0
    model_id: str = "ResembleAI/chatterbox-turbo"
    priority: int = 0
    # engine state
    slot: int = -1
    lp: int = 0                      # prefix length
    cpu_len: int = 0                 # prefix + committed tokens (CPU mirror)
    committed: list = field(default_factory=list)   # filtered token ids (CPU)
    emitted_tok: int = 0             # tokens handed to vocoder emit boundary
    voc_tail: object = None          # np.ndarray crossfade tail
    first_chunk_sent: bool = False
    t3_done: bool = False
    voc_jobs_pending: int = 0
    finished: bool = False
    # metrics (perf_counter timestamps)
    t_recv: float = 0.0
    t_admit: float = 0.0
    t_first_commit: float = 0.0
    t_first_pcm: float = 0.0
    t_done: float = 0.0
    n_tokens: int = 0
    audio_samples: int = 0


# --------------------------------------------------------------------------- #
# Bare-metal Llama forward over a slotted KV cache
# --------------------------------------------------------------------------- #


def _gumbel_like(x: Tensor) -> Tensor:
    """Standard Gumbel(0,1) noise, numerically safe."""
    u = torch.rand_like(x).clamp_min(1e-20)
    return -torch.log((-torch.log(u)).clamp_min(1e-20))


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class LlamaCore:
    """Weight views + custom forward for the T3 Llama backbone.

    QKV and gate/up projections are pre-fused into single GEMMs; RMSNorm
    uses the fused ``F.rms_norm`` kernel. Attention reads only the first
    ``L_used`` KV columns (callers cap it to the longest live sequence).
    """

    def __init__(self, tfmr, dtype: torch.dtype, device: torch.device):
        self.layers = list(tfmr.layers)
        self.norm = tfmr.norm
        self.rotary = tfmr.rotary_emb
        self.n_layers = len(self.layers)
        cfg = tfmr.config
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.eps = cfg.rms_norm_eps
        self.dtype = dtype
        self.device = device
        assert self.n_heads == self.n_kv, "GQA not needed for Llama_520M"
        with torch.no_grad():
            self.wqkv = [
                torch.cat([l.self_attn.q_proj.weight,
                           l.self_attn.k_proj.weight,
                           l.self_attn.v_proj.weight], dim=0).contiguous()
                for l in self.layers
            ]
            self.wgu = [
                torch.cat([l.mlp.gate_proj.weight, l.mlp.up_proj.weight],
                          dim=0).contiguous()
                for l in self.layers
            ]

    def forward(
        self,
        x: Tensor,                 # (B, T, D)
        pos_ids: Tensor,           # (B, T) long
        write_cols: Tensor,        # (B, T) long — KV columns to write
        kc: Tensor, vc: Tensor,    # (n_layers, S, H, L, hd); rows 0..B-1 active
        attn_mask: Tensor,         # (B, 1, T, L_used) or (B, 1, T, T) additive
        self_attn_only: bool,      # True → attend over this forward's own KV
        l_used: int | None = None, # KV columns to attend over (default: all)
    ) -> Tensor:
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        cos, sin = self.rotary(x, pos_ids)            # (B, T, hd)
        cos = cos.unsqueeze(1).to(x.dtype)
        sin = sin.unsqueeze(1).to(x.dtype)
        row = torch.arange(B, device=x.device).unsqueeze(1)  # (B, 1)
        if l_used is None:
            l_used = kc.size(3)

        h = x
        for li, lyr in enumerate(self.layers):
            attn = lyr.self_attn
            hn = F.rms_norm(h, (D,), lyr.input_layernorm.weight, self.eps)
            qkv = F.linear(hn, self.wqkv[li]).view(B, T, 3 * H, hd).transpose(1, 2)
            q, k, v = qkv.split(H, dim=1)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            # scatter KV into cache at per-row columns
            kcl, vcl = kc[li], vc[li]
            kcl[row, :, write_cols] = k.transpose(1, 2)
            vcl[row, :, write_cols] = v.transpose(1, 2)
            if self_attn_only:
                ka, va = k, v
            else:
                ka = kcl[:B, :, :l_used]
                va = vcl[:B, :, :l_used]
            o = F.scaled_dot_product_attention(q, ka, va, attn_mask=attn_mask)
            o = o.transpose(1, 2).reshape(B, T, H * hd)
            h = h + attn.o_proj(o)
            hn = F.rms_norm(h, (D,), lyr.post_attention_layernorm.weight, self.eps)
            gu = F.linear(hn, self.wgu[li])
            g, u = gu.chunk(2, dim=-1)
            h = h + lyr.mlp.down_proj(F.silu(g) * u)
        return F.rms_norm(h, (D,), self.norm.weight, self.eps)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


class EngineBase:
    """Decode-agnostic scheduler plumbing shared by the block-diffusion
    (flash) and autoregressive (turbo) engines: request queue, vocoder
    chunk handoff, and the run loop. Subclasses implement ``tick()`` and
    ``_free_slot()`` and must call :meth:`_init_sched`."""

    CODEC_V: int

    def _init_sched(self) -> None:
        self.active: list[Request] = []
        self.scheduler: SchedulerQueue[Request] = SchedulerQueue()
        self.vocoder = None          # set by server wiring
        self.stop_flag = False
        self.tick_count = 0
        self.tick_time = 0.0

    # ---- public API (any thread) ---- #

    def submit(self, req: Request) -> None:
        self.scheduler.submit(req)

    @property
    def queue_depth(self) -> int:
        return self.scheduler.depth()

    # ---- token → vocoder handoff ---- #

    def _push_committed(self, r: Request, toks: list[int], now: float) -> None:
        add = [t for t in toks if t < self.CODEC_V]
        if add and not r.committed:
            r.t_first_commit = now
        r.committed.extend(add)
        self._maybe_vocode(r, final=False)

    def _finish_t3(self, r: Request, now: float) -> None:
        r.t3_done = True
        r.n_tokens = len(r.committed)
        self._maybe_vocode(r, final=True)

    def _maybe_vocode(self, r: Request, final: bool) -> None:
        voc = self.vocoder
        avail = len(r.committed) - r.emitted_tok
        while True:
            if not r.first_chunk_sent:
                target = voc.chunk_first
            elif getattr(voc, "topup_sizes", ()):
                cursor = voc.chunk_first
                target = voc.chunk
                for size in voc.topup_sizes:
                    if r.emitted_tok < cursor + size:
                        target = size
                        break
                    cursor += size
            elif (getattr(voc, "chunk_second", 0)
                  and r.emitted_tok < getattr(voc, "topup_until", 0)):
                target = voc.chunk_second
            else:
                target = voc.chunk
            if r.t3_done and avail > 0 and avail <= target:
                voc.enqueue(r, final=True)
                r.emitted_tok = len(r.committed)
                r.first_chunk_sent = True
                return
            if avail >= target:
                is_last = r.t3_done and avail == target
                voc.enqueue(r, final=is_last, n_emit=target)
                r.emitted_tok += target
                r.first_chunk_sent = True
                avail -= target
                if is_last:
                    return
            else:
                break
        if r.t3_done and avail == 0:
            voc.finalize_empty(r)

    # ---- main loop ---- #

    def tick(self) -> None:
        raise NotImplementedError

    def _free_slot(self, req: Request) -> None:
        raise NotImplementedError

    def run(self) -> None:
        torch.manual_seed(1234)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            while not self.stop_flag:
                try:
                    self.tick()
                except Exception:
                    logger.exception("engine tick failed; aborting active requests")
                    for r in list(self.active):
                        self._free_slot(r)
                        if r.sink is not None:
                            r.sink(None)
                    self.active.clear()


class FlashEngine(EngineBase):
    """Continuous-batching block-diffusion scheduler for ChatterboxFlashT3."""

    def __init__(
        self,
        t3,
        cond_emb: Tensor,            # (1, Lc, D) prepared conditioning, model dtype
        *,
        max_active: int = 128,
        max_text_tokens: int = 160,
        max_speech_tokens: int = 624,
        num_steps: int = 10,
        temperature: float = 0.6,
        time_shift_tau: float = 0.1,
        omnivoice_t_shift: float = 0.5,
        cfg_scale: float = 1.0,
        position_temperature: float = 5.0,
        prefill_max: int = 16,
    ):
        self.t3 = t3
        self.dev = t3.device
        self.dtype = next(t3.parameters()).dtype
        self.hp = t3.hp
        self.BS = 16
        self.K = num_steps
        self.temp = temperature
        self.tau = time_shift_tau
        self.cfg = cfg_scale
        self.posT = position_temperature
        self.MASK = t3.mask_token_id
        self.STOP = int(self.hp.stop_speech_token)
        self.SOS = int(self.hp.start_speech_token)
        self.CODEC_V = int(self.hp.start_speech_token)   # valid codec ids < this
        self.V = int(self.hp.speech_tokens_dict_size)
        self.max_active = max_active
        self.max_text_tokens = max_text_tokens
        self.max_speech = max_speech_tokens
        self.prefill_max = prefill_max
        self.n_branch = 2 if cfg_scale > 0 else 1

        self.core = LlamaCore(t3.tfmr, self.dtype, self.dev)
        D = t3.dim
        H, hd = self.core.n_heads, self.core.head_dim
        NL = self.core.n_layers

        self.cond_emb = cond_emb.to(self.dev, self.dtype)          # (1, Lc, D)
        Lc = cond_emb.size(1)
        self.lp_max = Lc + max_text_tokens + 1
        self.L = self.lp_max + max_speech_tokens + self.BS
        S = max_active * self.n_branch

        logger.info(
            "KV cache: %d layers x %d slots x %d heads x %d cols x %d = %.1f GiB",
            NL, S, H, self.L, hd,
            2 * NL * S * H * self.L * hd * self.dtype.itemsize / 2**30,
        )
        self.kc = torch.zeros(NL, S, H, self.L, hd, dtype=self.dtype, device=self.dev)
        self.vc = torch.zeros(NL, S, H, self.L, hd, dtype=self.dtype, device=self.dev)

        M = max_active
        self.N_max = max_speech_tokens
        self.xt = torch.full((M, self.N_max), self.MASK, dtype=torch.long, device=self.dev)
        self.blk_start = torch.zeros(M, dtype=torch.long, device=self.dev)
        self.k_step = torch.zeros(M, dtype=torch.long, device=self.dev)
        self.cache_len = torch.zeros(M, dtype=torch.long, device=self.dev)
        self.n_speech = torch.zeros(M, dtype=torch.long, device=self.dev)
        self.is_commit = torch.zeros(M, dtype=torch.bool, device=self.dev)
        self.shift_ctx = torch.zeros(M * self.n_branch, 1, D, dtype=self.dtype, device=self.dev)

        # precomputed tables
        from chatterbox_flash.model import _omnivoice_unmask_schedule
        sched = _omnivoice_unmask_schedule(self.BS, self.K, omnivoice_t_shift)
        self.sched_t = torch.tensor(sched, dtype=torch.long, device=self.dev)
        q = [max(0.0, 1.0 - self.tau * (k + 1) / self.K) for k in range(self.K)]
        self.q_t = torch.tensor(q, dtype=torch.float32, device=self.dev)
        self.ar16 = torch.arange(self.BS, device=self.dev)
        self.arL = torch.arange(self.L, device=self.dev)

        # PMI unconditional block prior (V,)
        self.prior = t3._compute_uncond_block_prior(
            self.BS, self.SOS, self.MASK, self.dtype, self.dev,
        ).float().clamp_min(1e-8)
        self.log_prior = self.prior.log()

        # SOS embedding vector (1, 1, D)
        sos = torch.full((1, 1), self.SOS, device=self.dev, dtype=torch.long)
        self.sos_emb = t3._embed_speech_tokens(sos).to(self.dtype)

        # scheduling state (owned by GPU thread; waiting deque shared)
        self._init_sched()
        self.commit_blocks = 0


    # ---------------- prefix embedding ------------------------------------ #

    def _build_prefixes(self, reqs: list[Request]) -> tuple[Tensor, list[int]]:
        """Returns padded (G*n_branch, T, D) prefix embeddings, interleaved
        [cond_i, null_i] per request, plus per-request prefix lengths."""
        t3 = self.t3
        embs = []
        lps = []
        for r in reqs:
            tt = r.text_tokens.to(self.dev)
            te = t3.text_emb(tt)
            if t3.hp.input_pos_emb == "learned":
                te = te + t3.text_pos_emb(tt)
            pfx = torch.cat([self.cond_emb, te.to(self.dtype), self.sos_emb], dim=1)
            lps.append(pfx.size(1))
            embs.append(pfx)
        T = max(lps)
        G = len(reqs)
        nb = self.n_branch
        out = torch.zeros(G * nb, T, self.t3.dim, dtype=self.dtype, device=self.dev)
        for i, (pfx, lp) in enumerate(zip(embs, lps)):
            out[i * nb, :lp] = pfx[0]
            if nb == 2:
                # zero_all null prefix: zeros everywhere except SOS at lp-1
                out[i * nb + 1, lp - 1] = self.sos_emb[0, 0]
        return out, lps

    # ---------------- admission / prefill ---------------------------------- #

    def _admit(self) -> None:
        free = self.max_active - len(self.active)
        if free <= 0:
            return
        batch = self.scheduler.take(
            min(free, self.prefill_max),
            select_key=lambda request: (
                request.priority,
                request.t_recv,
                request.rid,
            ),
        )
        if not batch:
            return

        x, lps = self._build_prefixes(batch)
        nb = self.n_branch
        G = len(batch)
        T = x.size(1)
        base = len(self.active)

        # copy rows into slots [base, base+G): forward rows must be 0..B-1
        # contiguous, so we prefill in a temporary layout then place KV rows.
        pos = torch.arange(T, device=self.dev).unsqueeze(0).expand(G * nb, -1)
        # causal + column-validity mask (Gnb, 1, T, T)
        lp_t = torch.tensor(lps, device=self.dev).repeat_interleave(nb)
        causal = torch.triu(torch.full((T, T), NEG_INF, device=self.dev), 1)
        colv = torch.where(
            torch.arange(T, device=self.dev)[None, :] < lp_t[:, None], 0.0, NEG_INF,
        )
        mask = (causal[None] + colv[:, None, :]).unsqueeze(1).to(self.dtype)

        # prefill writes into the *destination* slot rows; build row-mapped
        # temporary views by running forward with kc/vc narrowed to the target
        # rows. Rows must be contiguous from 0 for LlamaCore, so use
        # index-mapped scratch: write into slots directly with narrow(base*nb).
        kc = self.kc.narrow(1, base * nb, G * nb)
        vc = self.vc.narrow(1, base * nb, G * nb)
        hidden = self.core.forward(
            x, pos, pos, kc, vc, mask, self_attn_only=True,
        )
        # shift ctx = hidden at lp-1 per row
        idx = (lp_t - 1)[:, None, None].expand(-1, 1, hidden.size(-1))
        sc = hidden.gather(1, idx)                       # (Gnb, 1, D)
        self.shift_ctx[base * nb: (base + G) * nb] = sc

        now = time.perf_counter()
        arange_g = torch.arange(base, base + G, device=self.dev)
        self.blk_start[arange_g] = 0
        self.k_step[arange_g] = 0
        self.is_commit[arange_g] = False
        self.cache_len[arange_g] = torch.tensor(lps, device=self.dev)
        self.xt[arange_g] = self.MASK
        for i, r in enumerate(batch):
            r.slot = base + i
            r.lp = lps[i]
            r.cpu_len = lps[i]
            r.t_admit = now
            self.n_speech[base + i] = r.n_speech
            self.active.append(r)

    # ---------------- slot free / compaction ------------------------------- #

    def _free_slot(self, req: Request) -> None:
        """Swap-with-last compaction keeping active rows contiguous."""
        i = req.slot
        last = len(self.active) - 1
        nb = self.n_branch
        if i != last:
            mv = self.active[last]
            # move KV rows
            self.kc[:, i * nb:(i + 1) * nb] = self.kc[:, last * nb:(last + 1) * nb]
            self.vc[:, i * nb:(i + 1) * nb] = self.vc[:, last * nb:(last + 1) * nb]
            self.shift_ctx[i * nb:(i + 1) * nb] = self.shift_ctx[last * nb:(last + 1) * nb]
            for t in (self.blk_start, self.k_step, self.cache_len,
                      self.n_speech, self.is_commit):
                t[i] = t[last]
            self.xt[i] = self.xt[last]
            mv.slot = i
            self.active[i] = mv
        self.active.pop()
        req.slot = -1

    # ---------------- one scheduler tick ----------------------------------- #

    def _step_gpu(self, B: int, l_used: int):
        """One batched block-diffusion step over rows [0, B). No host syncs;
        all state lives in preallocated buffers. Attention reads only the
        first ``l_used`` KV columns (longest live sequence)."""
        nb = self.n_branch

        blk_idx = self.blk_start[:B, None] + self.ar16          # (B, 16)
        xt_blk = self.xt[:B].gather(1, blk_idx)                 # (B, 16)
        emb = self.t3._embed_speech_tokens(xt_blk).to(self.dtype)
        x = emb.repeat_interleave(nb, dim=0) if nb == 2 else emb

        pos1 = self.cache_len[:B, None] + self.ar16             # (B, 16)
        pos = pos1.repeat_interleave(nb, dim=0) if nb == 2 else pos1
        # column mask: visible = [0, cache_len + 16)
        colv1 = torch.where(
            self.arL[None, :l_used] < (self.cache_len[:B, None] + self.BS),
            0.0, NEG_INF,
        )
        colv = colv1.repeat_interleave(nb, dim=0) if nb == 2 else colv1
        mask = colv[:, None, None, :].to(self.dtype)            # (Bnb,1,1,l_used)

        hidden = self.core.forward(
            x, pos, pos, self.kc, self.vc, mask,
            self_attn_only=False, l_used=l_used,
        )                                                       # (Bnb, 16, D)

        shift_hidden = torch.cat(
            [self.shift_ctx[:B * nb], hidden[:, : self.BS - 1]], dim=1,
        )
        logits = self.t3.speech_head(shift_hidden).float()      # (Bnb, 16, V)

        commit_rows = self.is_commit[:B].clone()                # (B,)
        denoise_rows = ~commit_rows

        # ---- commit bookkeeping ----
        last_h = hidden[:, self.BS - 1: self.BS]                # (Bnb, 1, D)
        cm2 = commit_rows.repeat_interleave(nb) if nb == 2 else commit_rows
        self.shift_ctx[:B * nb] = torch.where(
            cm2[:, None, None], last_h, self.shift_ctx[:B * nb],
        )

        # ---- denoise step (vectorized across rows) ----
        if nb == 2:
            lc = logits.view(B, 2, self.BS, self.V)
            logits_c, logits_u = lc[:, 0], lc[:, 1]
            lsc = F.log_softmax(logits_c, dim=-1)
            lsu = F.log_softmax(logits_u, dim=-1)
            guided = lsc + self.cfg * (lsc - lsu)
            probs_c = logits_c.softmax(-1)
            probs_u = logits_u.softmax(-1)
        else:
            guided = logits
            probs_c = logits.softmax(-1)
            probs_u = None

        if self.temp > 0:
            g = _gumbel_like(guided)
            sampled = (guided / self.temp + g).argmax(-1)       # (B, 16)
        else:
            sampled = guided.argmax(-1)

        is_mask = xt_blk == self.MASK                           # (B, 16)
        lp_c = probs_c.gather(2, sampled.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8).log()
        pmi = lp_c - self.log_prior[sampled]
        if probs_u is not None:
            lp_u = probs_u.gather(2, sampled.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8).log()
            pmi_u = lp_u - self.log_prior[sampled]
            pmi = (1.0 + self.cfg) * pmi - self.cfg * pmi_u
        if self.posT > 0:
            gn = _gumbel_like(pmi)
            pmi = torch.where(is_mask, pmi + gn, pmi)

        m = is_mask.sum(1)                                       # (B,)
        ks = self.k_step[:B]
        # quantile threshold over masked pmi (ascending sort, linear interp)
        asc = torch.where(is_mask, pmi, torch.inf).sort(1).values
        p = self.q_t[ks.clamp(max=self.K - 1)] * (m - 1).clamp_min(0).float()
        lo = p.floor().long().clamp_min(0)
        hi = p.ceil().long().clamp_min(0)
        v_lo = asc.gather(1, lo[:, None]).squeeze(1)
        v_hi = asc.gather(1, hi[:, None]).squeeze(1)
        tau_k = v_lo + (p - lo.float()) * (v_hi - v_lo)
        n_quant = ((pmi >= tau_k[:, None]) & is_mask).sum(1)
        if self.tau <= 0:
            n_quant = torch.zeros_like(n_quant)
        n_sched = torch.minimum(self.sched_t[ks.clamp(max=self.K - 1)], m)
        n_unmask = torch.minimum(torch.maximum(n_sched, n_quant), m)
        n_unmask = torch.where(ks >= self.K - 1, m, n_unmask)
        n_unmask = torch.where(m == 0, torch.zeros_like(n_unmask), n_unmask)

        order = torch.where(is_mask, pmi, -torch.inf).argsort(1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, self.ar16.expand(B, -1))
        do_unmask = is_mask & (rank < n_unmask[:, None])

        xt_new = torch.where(do_unmask, sampled, xt_blk)
        # only denoise rows mutate xt / k_step
        xt_final = torch.where(denoise_rows[:, None], xt_new, xt_blk)
        self.xt[:B] = self.xt[:B].scatter(1, blk_idx, xt_final)

        n_mask_after = (xt_final == self.MASK).sum(1)
        hit_eos = (xt_final == self.STOP).any(1)
        self.k_step[:B] = torch.where(denoise_rows, ks + 1, ks)

        # commit rows: advance cache; move to next block. Clamps keep inert
        # padding rows (freed slots) in-range forever.
        adv = commit_rows
        self.cache_len[:B] = torch.where(
            adv, (self.cache_len[:B] + self.BS).clamp(max=self.L - self.BS),
            self.cache_len[:B],
        )
        self.blk_start[:B] = torch.where(
            adv, (self.blk_start[:B] + self.BS).clamp(max=self.N_max - self.BS),
            self.blk_start[:B],
        )
        self.k_step[:B] = torch.where(adv, torch.zeros_like(ks), self.k_step[:B])
        self.is_commit[:B] = torch.where(adv, torch.zeros_like(adv), self.is_commit[:B])
        # denoise rows whose block just finished start committing next tick
        to_commit = denoise_rows & ~hit_eos & (
            (n_mask_after == 0) | (self.k_step[:B] >= self.K)
        )
        self.is_commit[:B] = self.is_commit[:B] | to_commit

        flags = torch.stack([
            commit_rows.long(),
            n_mask_after,
            hit_eos.long(),
            self.k_step[:B],
            self.blk_start[:B],
            self.n_speech[:B],
        ], dim=1)
        return flags, xt_final

    @torch.inference_mode()
    def tick(self) -> None:
        self._admit()
        A = len(self.active)
        if A == 0:
            self.scheduler.wait(0.005)
            return
        t0 = time.perf_counter()

        l_used = min(self.L, max(r.cpu_len for r in self.active) + self.BS)
        flags, xt_final = self._step_gpu(A, l_used)
        flags_cpu = flags.cpu()
        xt_final_cpu = xt_final.cpu()

        now = time.perf_counter()
        to_free: list[Request] = []
        for i in range(A):
            r = self.active[i]
            was_commit, nmask, eos, kk, bstart, nsp = flags_cpu[i].tolist()
            if was_commit:
                # block [bstart-16, bstart) is now final
                r.cpu_len = r.lp + bstart
                toks = xt_final_cpu[i].tolist()
                self._push_committed(r, toks, now)
                self.commit_blocks += 1
                if bstart >= nsp:
                    self._finish_t3(r, now)
                    to_free.append(r)
            elif eos:
                toks = xt_final_cpu[i].tolist()
                cut = toks.index(self.STOP)
                self._push_committed(r, toks[:cut], now)
                self._finish_t3(r, now)
                to_free.append(r)

        for r in to_free:
            self._free_slot(r)

        self.tick_count += 1
        self.tick_time += time.perf_counter() - t0

    # ---------------- stats ------------------------------------------------ #

    def stats(self) -> dict:
        avg_tick = self.tick_time / max(1, self.tick_count)
        return dict(
            active=len(self.active),
            waiting=self.queue_depth,
            ticks=self.tick_count,
            avg_tick_ms=round(avg_tick * 1e3, 3),
            commit_blocks=self.commit_blocks,
        )
