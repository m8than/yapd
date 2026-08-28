"""Vocoder subprocess: owns S3Gen on the same GPU with its own GIL.

Parent (engine process) ships token-window jobs over an mp.Queue; this
process reuses :class:`server.vocoder.Vocoder` for micro-batching, chunked
CFM/HiFT synthesis and crossfade, and returns PCM chunks over another queue.

Protocol
--------
in_q  : ("job", rid, window: list[int], emit_off: int, first: bool, final: bool)
        ("end", rid)                      close stream (no more audio)
        ("stop",)
out_q : ("ready",)                        emitted once after model load
        (rid, pcm_bytes | None)           None closes the stream
"""

from __future__ import annotations

import logging
import time


class RemoteReq:
    """Per-rid state stand-in for engine.Request inside the subprocess."""

    __slots__ = ("rid", "voc_tail", "audio_samples", "t_first_pcm",
                 "finished", "t_done", "sink", "send_tail")

    def __init__(self, rid, out_q):
        self.rid = rid
        self.voc_tail = None
        self.audio_samples = 0
        self.t_first_pcm = 0.0
        self.finished = False
        self.t_done = 0.0
        self.sink = lambda data: out_q.put((rid, data))
        self.send_tail = lambda arr: out_q.put(
            (rid, ("tail", arr.astype("float32").tobytes())))


def build_vocoder(cfg: dict):
    """Load a fresh S3Gen + reference embedding and return a ready Vocoder.
    Shared by the vocoder subprocess and the in-process first-chunk lane."""
    log = logging.getLogger("voc_proc")
    import torch
    import librosa
    from server.vocoder import Vocoder
    device = cfg.get("device", "cuda")
    target_device = torch.device(device)
    torch.cuda.set_device(0 if target_device.index is None else target_device)
    if cfg.get("model") == "turbo":
        from pathlib import Path
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from chatterbox.models.s3gen import S3Gen

        ckpt_dir = Path(snapshot_download(
            cfg["repo"], allow_patterns=["s3gen_meanflow.safetensors"],
        ))
        s3gen = S3Gen(meanflow=True)
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen_meanflow.safetensors"), strict=True,
        )
        s3gen.to(device).eval()
    else:
        from chatterbox_flash import ChatterboxFlashTTS

        tts = ChatterboxFlashTTS.from_pretrained(cfg["repo"], device=device)
        # engine-side T3/VE are dead weight here; free them immediately.
        del tts.t3, tts.ve
        s3gen = tts.s3gen
    torch.cuda.empty_cache()

    wav24, _ = librosa.load(cfg["ref_wav"], sr=24000)
    if cfg.get("model") == "turbo":
        # turbo loudness-normalizes the reference before embedding
        try:
            import math
            import pyloudnorm as ln
            meter = ln.Meter(24000)
            gain = 10.0 ** ((-27 - meter.integrated_loudness(wav24)) / 20.0)
            if math.isfinite(gain) and gain > 0.0:
                wav24 = wav24 * gain
        except Exception:
            log.warning("loudness norm failed; using raw reference")
    with torch.inference_mode():
        voc_ref = s3gen.embed_ref(
            wav24[: int(cfg["voc_ref_seconds"] * 24000)], 24000, device=device,
        )
    if cfg.get("fused_snake"):
        from server.kernels import install_fused_snake
        count = install_fused_snake(s3gen.mel2wav)
        log.info("installed fused HIP Snake on %d modules", count)
    if cfg["voc_dtype"] == "bf16":
        s3gen.flow.to(torch.bfloat16)

    return Vocoder(
        s3gen, voc_ref, torch.device(device),
        chunk_first=cfg["chunk_first"],
        topup_sizes=tuple(cfg.get("topup_sizes") or ()),
        chunk_second=cfg.get("chunk_second", 0),
        chunk=cfg["chunk"],
        topup_until=cfg.get("topup_until", 0),
        lookback=cfg["lookback"],
        first_max_batch=cfg.get("first_voc_batch", 24),
        max_batch=cfg["voc_batch"],
        pre_roll=cfg.get("pre_roll", 0),
        n_cfm_timesteps=cfg.get("n_cfm", 1),
    )


def main(cfg: dict, in_q, out_q) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s voc %(levelname)s %(message)s")
    log = logging.getLogger("voc_proc")
    import torch
    import numpy as np
    from server.vocoder import Vocoder, VocJob

    t0 = time.time()
    device = cfg.get("device", "cuda")
    voc = build_vocoder(cfg)
    s3gen = voc.s3gen

    # warm the MIOpen find-db over the bucketed (B, W) shape grid so no
    # request ever pays conv-tuning latency (results persist on disk)
    with torch.inference_mode():
        for W in voc.W_BUCKETS:
            for B in voc.B_BUCKETS:
                if B > cfg["voc_batch"] * 2:
                    break
                tok = torch.randint(0, 6561, (B, W), device=device)
                tl = torch.full((B,), W, dtype=torch.long, device=device)
                mels, _ = s3gen.flow.inference(
                    token=tok, token_len=tl,
                    prompt_token=voc.prompt_token,
                    prompt_token_len=voc.prompt_token_len,
                    prompt_feat=voc.prompt_feat, prompt_feat_len=None,
                    embedding=voc.embedding, finalize=True,
                    n_timesteps=voc.n_cfm, meanflow=True,
                )
                s3gen.mel2wav.inference(
                    speech_feat=mels.float(),
                    cache_source=torch.zeros(B, 1, 0, device=device),
                )
    torch.cuda.synchronize()
    log.info("vocoder ready in %.1fs (shape grid warmed)", time.time() - t0)

    reqs: dict[int, RemoteReq] = {}
    import threading

    def pump():
        while True:
            msg = in_q.get()
            if msg[0] == "stop":
                voc.stop_flag = True
                voc.wake.set()
                voc.wake_first.set()
                return
            if msg[0] == "end":
                rid = msg[1]
                r = reqs.setdefault(rid, RemoteReq(rid, out_q))
                with voc.lock:
                    voc.q.append(VocJob(r, [], 0, False, True, end_marker=True))
                voc.wake.set()
                continue
            _, rid, window, emit_off, first, final, tail, priority, deadline = msg
            r = reqs.setdefault(rid, RemoteReq(rid, out_q))
            if tail is not None:
                # first chunk was vocoded in the engine process; its crossfade
                # tail arrives with the rid's first remote job
                r.voc_tail = (np.frombuffer(tail, np.float32).copy()
                              if tail else None)
                r.audio_samples = max(r.audio_samples, 1)
            job = VocJob(
                r, window, emit_off, first, final, deadline=deadline,
            )
            with voc.lock:
                if first:
                    voc.q_first.append(job)
                elif priority:
                    voc.q_topup[len(window)].append(job)
                else:
                    voc.q.append(job)
            (voc.wake_first if priority else voc.wake).set()

    threading.Thread(target=pump, daemon=True).start()

    # periodic cleanup of finished rid state
    def sweep():
        while True:
            time.sleep(30)
            dead = [rid for rid, r in reqs.items() if r.finished]
            for rid in dead:
                reqs.pop(rid, None)

    threading.Thread(target=sweep, daemon=True).start()

    out_q.put(("ready",))
    voc.run()   # blocks until stop
