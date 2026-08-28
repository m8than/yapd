"""Numerical parity: LlamaCore custom forward vs stock HF LlamaModel."""

import torch
import torch.nn.functional as F

from chatterbox_flash import ChatterboxFlashTTS
from server.engine import LlamaCore

torch.manual_seed(0)

tts = ChatterboxFlashTTS.from_pretrained("ResembleAI/chatterbox-flash", device="cuda")
t3 = tts.t3
dt = next(t3.parameters()).dtype
dev = t3.device
core = LlamaCore(t3.tfmr, dt, dev)

B, T, D = 2, 37, t3.dim
H, hd = core.n_heads, core.head_dim
NL = core.n_layers
L = 128

with torch.inference_mode():
    x = (torch.randn(B, T, D, device=dev) * 0.1).to(dt)
    pos = torch.arange(T, device=dev).unsqueeze(0).expand(B, -1)

    # HF reference
    cfg = t3.tfmr.config
    cfg._attn_implementation = "sdpa"
    out = t3.tfmr(inputs_embeds=x, position_ids=pos, use_cache=False, return_dict=True)
    ref = out.last_hidden_state

    # custom prefill
    kc = torch.zeros(NL, B, H, L, hd, dtype=dt, device=dev)
    vc = torch.zeros(NL, B, H, L, hd, dtype=dt, device=dev)
    causal = torch.triu(torch.full((T, T), float("-inf"), device=dev), 1)
    mask = causal[None, None].expand(B, 1, T, T).to(dt)
    mine = core.forward(x, pos, pos, kc, vc, mask, self_attn_only=True)

    diff = (ref.float() - mine.float()).abs()
    print("prefill max diff", diff.max().item(), "mean", diff.mean().item(),
          "ref scale", ref.float().abs().mean().item())

    # decode continuation parity: run T+16 tokens in one HF pass vs
    # prefill(T) + block(16) through the cache
    xb = (torch.randn(B, 16, D, device=dev) * 0.1).to(dt)
    x_full = torch.cat([x, xb], 1)
    pos_full = torch.arange(T + 16, device=dev).unsqueeze(0).expand(B, -1)
    ref_full = t3.tfmr(inputs_embeds=x_full, position_ids=pos_full,
                       use_cache=False, return_dict=True).last_hidden_state
    # NOTE: HF full pass is causal within the block; engine block is
    # full-visible. For parity use a causal block mask here.
    posb = torch.arange(T, T + 16, device=dev).unsqueeze(0).expand(B, -1)
    colm = torch.where(torch.arange(L, device=dev)[None, :] < T + 16, 0.0, float("-inf"))
    causal_blk = torch.triu(torch.full((16, 16), float("-inf"), device=dev), 1)
    m2 = colm[:, None, None, :].expand(B, 1, 16, L).clone()
    m2[:, :, :, T:T + 16] += causal_blk[None, None]
    mine_blk = core.forward(xb, posb, posb, kc, vc, m2.to(dt), self_attn_only=False)
    diff2 = (ref_full[:, T:].float() - mine_blk.float()).abs()
    print("decode max diff", diff2.max().item(), "mean", diff2.mean().item())
