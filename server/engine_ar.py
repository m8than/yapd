"""Continuous-batching autoregressive engine for Chatterbox-Turbo.

Turbo's T3 is a 350M GPT2-medium backbone decoded token-by-token — no
diffusion, no CFG. Each scheduler tick advances every active request by one
speech token: a (B, 1) forward over per-slot KV caches, then a vectorized
replica of the stock HF sampler chain (temperature → top-k → top-p →
repetition-penalty → categorical sample, in exactly that order, matching
``T3.inference_turbo``). Requests join at any tick via ragged batched
prefill (the first token is sampled straight from the prefill's last
hidden state) and leave on EOS / token cap.

Matches stock semantics:
  * repetition penalty over previously *generated* tokens (SOS penalized
    only for the very first sample, as upstream does)
  * non-codec samples (ids >= 6561) stay in the AR context but are
    filtered from the vocoder stream
  * on finish, 3 silence tokens (S3GEN_SIL) are appended before vocoding
Sampling uses gumbel-max instead of torch.multinomial (identical
categorical distribution, no host sync).
"""

from __future__ import annotations

import logging
import time

import torch
import torch.nn.functional as F
from torch import Tensor

from server.engine import EngineBase, Request, _gumbel_like, NEG_INF

logger = logging.getLogger(__name__)

S3GEN_SIL = 4299


class GPT2Core:
    """Custom forward over the GPT2 blocks with a slotted KV cache."""

    def __init__(self, tfmr, dtype: torch.dtype, device: torch.device,
                 lora_bank=None):
        self.blocks = list(tfmr.h)
        self.wpe = tfmr.wpe
        self.ln_f = tfmr.ln_f
        cfg = tfmr.config
        self.n_layers = len(self.blocks)
        self.n_heads = cfg.n_head
        self.D = cfg.n_embd
        self.head_dim = self.D // self.n_heads
        self.eps = cfg.layer_norm_epsilon
        self.scale = (self.D // self.n_heads) ** -0.5
        self.dtype = dtype
        self.device = device
        self.lora_bank = lora_bank

    def _project(self, x: Tensor, weight: Tensor, bias: Tensor,
                 *, layer: int, module: str, groups):
        out = torch.addmm(bias, x, weight)
        if self.lora_bank is not None:
            self.lora_bank.apply_(
                out, x, layer=layer, module=module, groups=groups,
            )
        return out

    def forward(
        self,
        x: Tensor,                 # (B, T, D) input embeddings (pre-wpe)
        pos_ids: Tensor,           # (B, T) long
        write_cols: Tensor,        # (B, T) long
        kc: Tensor, vc: Tensor,    # (n_layers, S, H, L, hd)
        attn_mask: Tensor,         # (B, 1, T, L_used) or (B, 1, T, T)
        self_attn_only: bool,
        l_used: int | None = None,
        lora_groups=None,
    ) -> Tensor:
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        row = torch.arange(B, device=x.device).unsqueeze(1)
        if l_used is None:
            l_used = kc.size(3)

        h = x + self.wpe(pos_ids).to(x.dtype)
        for li, blk in enumerate(self.blocks):
            hn = F.layer_norm(h, (D,), blk.ln_1.weight, blk.ln_1.bias, self.eps)
            # Conv1D: y = x @ W + b, W is (in, out)
            qkv = self._project(
                hn.view(-1, D), blk.attn.c_attn.weight,
                blk.attn.c_attn.bias, layer=li, module="attn.c_attn",
                groups=lora_groups,
            ).view(B, T, 3, H, hd)
            q = qkv[:, :, 0].transpose(1, 2)
            k = qkv[:, :, 1].transpose(1, 2)
            v = qkv[:, :, 2].transpose(1, 2)
            kcl, vcl = kc[li], vc[li]
            kcl[row, :, write_cols] = k.transpose(1, 2)
            vcl[row, :, write_cols] = v.transpose(1, 2)
            if self_attn_only:
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            elif T == 1:
                # single-token decode: explicit bmm+softmax reads KV at
                # ~1.9 TB/s vs ~1.5 for the efficient-attention kernel
                ka = kcl[:B, :, :l_used]
                va = vcl[:B, :, :l_used]
                s = torch.matmul(q, ka.transpose(-1, -2)) * self.scale
                s = s + attn_mask
                o = torch.matmul(s.softmax(-1), va)
            else:
                ka = kcl[:B, :, :l_used]
                va = vcl[:B, :, :l_used]
                o = F.scaled_dot_product_attention(q, ka, va, attn_mask=attn_mask)
            o = o.transpose(1, 2).reshape(-1, D)
            attn_out = self._project(
                o, blk.attn.c_proj.weight, blk.attn.c_proj.bias,
                layer=li, module="attn.c_proj", groups=lora_groups,
            ).view(B, T, D)
            h = h + attn_out
            hn2 = F.layer_norm(h, (D,), blk.ln_2.weight, blk.ln_2.bias, self.eps)
            m = self._project(
                hn2.view(-1, D), blk.mlp.c_fc.weight, blk.mlp.c_fc.bias,
                layer=li, module="mlp.c_fc", groups=lora_groups,
            )
            m = F.gelu(m, approximate="tanh")
            mlp_out = self._project(
                m, blk.mlp.c_proj.weight, blk.mlp.c_proj.bias,
                layer=li, module="mlp.c_proj", groups=lora_groups,
            ).view(B, T, D)
            h = h + mlp_out
        return F.layer_norm(h, (D,), self.ln_f.weight, self.ln_f.bias, self.eps)


class TurboEngine(EngineBase):
    """One-speech-token-per-tick continuous batching for ChatterboxTurboTTS."""

    def __init__(
        self,
        t3,
        cond_emb: Tensor,            # (1, Lc, D)
        *,
        max_active: int = 256,
        max_text_tokens: int = 160,
        max_speech_tokens: int = 624,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        prefill_max: int = 32,
        lora_bank=None,
    ):
        self.t3 = t3
        self.dev = t3.device
        self.dtype = next(t3.parameters()).dtype
        hp = t3.hp
        self.STOP = int(hp.stop_speech_token)      # 6562
        self.SOS = int(hp.start_speech_token)      # 6561
        self.CODEC_V = int(hp.start_speech_token)
        self.V = int(hp.speech_tokens_dict_size)   # 6563
        self.temp = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.rep_p = repetition_penalty
        self.max_active = max_active
        self.max_text_tokens = max_text_tokens
        self.max_speech = max_speech_tokens
        self.prefill_max = prefill_max

        self.lora_bank = lora_bank
        self.core = GPT2Core(t3.tfmr, self.dtype, self.dev, lora_bank)
        D = self.core.D
        H, hd, NL = self.core.n_heads, self.core.head_dim, self.core.n_layers

        self.cond_emb = cond_emb.to(self.dev, self.dtype)
        Lc = cond_emb.size(1)
        self.lp_max = Lc + max_text_tokens + 1
        self.L = self.lp_max + max_speech_tokens + 8
        S = max_active
        logger.info(
            "turbo KV cache: %d layers x %d slots x %d heads x %d cols = %.1f GiB",
            NL, S, H, self.L,
            2 * NL * S * H * self.L * hd * self.dtype.itemsize / 2**30,
        )
        self.kc = torch.zeros(NL, S, H, self.L, hd, dtype=self.dtype, device=self.dev)
        self.vc = torch.zeros(NL, S, H, self.L, hd, dtype=self.dtype, device=self.dev)

        self.cache_len = torch.zeros(S, dtype=torch.long, device=self.dev)
        self.last_tok = torch.full((S, 1), self.SOS, dtype=torch.long, device=self.dev)
        self.seen = torch.zeros(S, self.V, dtype=torch.bool, device=self.dev)
        self.arL = torch.arange(self.L, device=self.dev)

        sos = torch.full((1, 1), self.SOS, device=self.dev, dtype=torch.long)
        self.sos_emb = t3.speech_emb(sos).to(self.dtype)

        # Conditioning KV depends on adapter weights. Cache one template per
        # registered adapter so admission remains memcpy-only.
        self.Lc = Lc
        n_adapters = len(lora_bank.adapters) if lora_bank is not None else 1
        self.tpl_kc = torch.zeros(
            NL, n_adapters, H, Lc, hd, dtype=self.dtype, device=self.dev,
        )
        self.tpl_vc = torch.zeros_like(self.tpl_kc)
        with torch.inference_mode():
            posc = torch.arange(Lc, device=self.dev).unsqueeze(0)
            causal = torch.triu(
                torch.full((Lc, Lc), NEG_INF, device=self.dev), 1,
            )[None, None].to(self.dtype)
            for adapter_slot in range(n_adapters):
                tkc = self.tpl_kc[:, adapter_slot:adapter_slot + 1]
                tvc = self.tpl_vc[:, adapter_slot:adapter_slot + 1]
                groups = self._lora_groups([adapter_slot], Lc)
                self.core.forward(
                    self.cond_emb, posc, posc, tkc, tvc, causal,
                    self_attn_only=True, lora_groups=groups,
                )

        self._init_sched()
        self.gen_tokens = 0

    # ---------------- sampling (stock HF chain, vectorized) ---------------- #

    def _sample(self, logits: Tensor, seen: Tensor) -> Tensor:
        """temperature → top-k → top-p → repetition penalty → categorical.

        ``logits`` (B, V) fp32; ``seen`` (B, V) bool history mask. Order and
        semantics replicate the LogitsProcessorList in T3.inference_turbo,
        but all work after top-k happens on the (B, k) candidate set:
        tokens outside top-k have probability 0 under the stock chain, and
        top-p over the survivors is unchanged (HF removes ascending-cum
        <= 1-p  ⟺  keep iff descending-cum *before* the token < p).
        """
        if self.temp > 0 and self.temp != 1.0:
            logits = logits / self.temp
        k = min(self.top_k, self.V) if self.top_k > 0 else self.V
        vals, idx = logits.topk(k, dim=-1)            # sorted descending
        if self.top_p < 1.0:
            probs = vals.softmax(-1)
            cum_before = probs.cumsum(-1) - probs
            vals = torch.where(cum_before < self.top_p, vals, NEG_INF)
        if self.rep_p != 1.0:
            seen_k = seen.gather(1, idx)
            pen = torch.where(vals > 0, vals / self.rep_p, vals * self.rep_p)
            vals = torch.where(seen_k, pen, vals)
        choice = (vals + _gumbel_like(vals)).argmax(-1, keepdim=True)  # (B,1)
        return idx.gather(1, choice).squeeze(1)        # (B,)

    def _lora_groups(self, slots: list[int], tokens_per_row: int):
        if self.lora_bank is None:
            return None
        groups = []
        arange = torch.arange(tokens_per_row, device=self.dev)
        for slot in sorted(set(slots)):
            if slot == 0:
                continue
            rows = [i for i, value in enumerate(slots) if value == slot]
            row_tensor = torch.tensor(rows, device=self.dev)
            indices = (
                row_tensor[:, None] * tokens_per_row + arange[None, :]
            ).reshape(-1)
            groups.append((slot, indices))
        return groups or None

    # ---------------- admission / prefill ---------------------------------- #

    def _build_text_embs(self, reqs: list[Request]) -> tuple[Tensor, list[int]]:
        """Padded (G, Tt, D) [text | SOS] embeddings + per-request lengths
        (text_len + 1). The conditioning prefix is NOT included — its KV
        comes from the precomputed template."""
        embs, tls = [], []
        for r in reqs:
            tt = r.text_tokens.to(self.dev)
            te = self.t3.text_emb(tt).to(self.dtype)   # GPT2: no learned pos here
            e = torch.cat([te, self.sos_emb], dim=1)
            tls.append(e.size(1))
            embs.append(e)
        T = max(tls)
        out = torch.zeros(len(reqs), T, self.core.D, dtype=self.dtype, device=self.dev)
        for i, (e, tl) in enumerate(zip(embs, tls)):
            out[i, :tl] = e[0]
        return out, tls

    @torch.inference_mode()
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

        x, tls = self._build_text_embs(batch)
        G, T = x.size(0), x.size(1)
        Lc = self.Lc
        base = len(self.active)

        adapter_slots = [request.lora_slot for request in batch]
        adapter_slots_t = torch.tensor(adapter_slots, device=self.dev)
        self.kc[:, base:base + G, :, :Lc] = self.tpl_kc.index_select(
            1, adapter_slots_t,
        )
        self.vc[:, base:base + G, :, :Lc] = self.tpl_vc.index_select(
            1, adapter_slots_t,
        )

        # 2. text+SOS prefill attending over [cond template | causal text]
        tl_t = torch.tensor(tls, device=self.dev)
        lp_t = tl_t + Lc                                        # full prefix len
        pos = (Lc + torch.arange(T, device=self.dev)).unsqueeze(0).expand(G, -1)
        l_used = Lc + T
        ar = torch.arange(T, device=self.dev)
        # columns [0, Lc): always visible; column Lc+j: causal (j <= i) and real
        causal = torch.where(ar[None, :] <= ar[:, None], 0.0, NEG_INF)  # (T, T)
        real = torch.where(ar[None, :] < tl_t[:, None], 0.0, NEG_INF)   # (G, T)
        mask = torch.zeros(G, 1, T, l_used, device=self.dev)
        mask[:, 0, :, Lc:] = causal[None] + real[:, None, :]
        mask = mask.to(self.dtype)

        kc = self.kc.narrow(1, base, G)
        vc = self.vc.narrow(1, base, G)
        hidden = self.core.forward(
            x, pos, pos, kc, vc, mask, self_attn_only=False, l_used=l_used,
            lora_groups=self._lora_groups(adapter_slots, T),
        )
        idx = (tl_t - 1)[:, None, None].expand(-1, 1, hidden.size(-1))
        last_h = hidden.gather(1, idx).squeeze(1)               # (G, D)
        logits = self.t3.speech_head(last_h).float()            # (G, V)

        # first sample: history = {SOS} (stock passes speech_start_token)
        seen0 = torch.zeros(G, self.V, dtype=torch.bool, device=self.dev)
        seen0[:, self.SOS] = True
        tok0 = self._sample(logits, seen0)                      # (G,)

        rows = torch.arange(base, base + G, device=self.dev)
        self.cache_len[rows] = lp_t
        self.last_tok[rows] = tok0[:, None]
        self.seen[rows] = False
        self.seen[rows, tok0] = True

        tok0_cpu = tok0.cpu().tolist()
        lps = (tl_t + Lc).cpu().tolist()
        now = time.perf_counter()
        for i, r in enumerate(batch):
            r.slot = base + i
            r.lp = lps[i]
            r.cpu_len = lps[i]
            r.t_admit = now
            self.active.append(r)
            self._consume(r, tok0_cpu[i], now)

    # ---------------- per-token bookkeeping --------------------------------- #

    def _consume(self, r: Request, tok: int, now: float) -> bool:
        """Register one sampled token; returns True if the request finished."""
        r.cpu_len += 1
        gen = r.cpu_len - r.lp
        if tok == self.STOP or gen >= self.max_speech:
            r.committed.extend([S3GEN_SIL] * 3)   # stock trailing silence
            self._finish_t3(r, now)
            return True
        self.gen_tokens += 1
        self._push_committed(r, [tok], now)       # filters ids >= CODEC_V
        return False

    def _free_slot(self, req: Request) -> None:
        i = req.slot
        last = len(self.active) - 1
        if i != last:
            mv = self.active[last]
            self.kc[:, i] = self.kc[:, last]
            self.vc[:, i] = self.vc[:, last]
            for t in (self.cache_len, self.last_tok, self.seen):
                t[i] = t[last]
            mv.slot = i
            self.active[i] = mv
        self.active.pop()
        req.slot = -1
    @torch.inference_mode()
    def tick(self) -> None:
        self._admit()
        A = len(self.active)
        if A == 0:
            self.scheduler.wait(0.005)
            return
        t0 = time.perf_counter()

        l_used = min(self.L, max(r.cpu_len for r in self.active) + 1)
        x = self.t3.speech_emb(self.last_tok[:A]).to(self.dtype)   # (A, 1, D)
        pos = self.cache_len[:A, None]                             # (A, 1)
        colv = torch.where(
            self.arL[None, :l_used] < (self.cache_len[:A, None] + 1), 0.0, NEG_INF,
        )
        mask = colv[:, None, None, :].to(self.dtype)
        hidden = self.core.forward(
            x, pos, pos, self.kc, self.vc, mask,
            self_attn_only=False, l_used=l_used,
            lora_groups=self._lora_groups(
                [request.lora_slot for request in self.active], 1,
            ),
        )
        logits = self.t3.speech_head(hidden[:, 0]).float()         # (A, V)
        tok = self._sample(logits, self.seen[:A])                  # (A,)

        self.cache_len[:A] += 1
        self.last_tok[:A] = tok[:, None]
        self.seen[:A].scatter_(1, tok[:, None], True)

        tok_cpu = tok.cpu().tolist()                               # 1 sync/tick
        now = time.perf_counter()
        to_free = []
        for i in range(A):
            if self._consume(self.active[i], tok_cpu[i], now):
                to_free.append(self.active[i])
        for r in to_free:
            self._free_slot(r)

        self.tick_count += 1
        self.tick_time += time.perf_counter() - t0

    def stats(self) -> dict:
        avg_tick = self.tick_time / max(1, self.tick_count)
        return dict(
            active=len(self.active),
            waiting=self.queue_depth,
            ticks=self.tick_count,
            avg_tick_ms=round(avg_tick * 1e3, 3),
            gen_tokens=self.gen_tokens,
        )
