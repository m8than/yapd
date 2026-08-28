"""Standalone engine sanity test: N concurrent requests through the
continuous-batching engine + streaming vocoder, wavs written to /tmp.

    HIP_VISIBLE_DEVICES=0 python -m server.test_engine
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
import soundfile as sf
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("test")

TEXTS = [
    "Sometimes it's better to just let things slide, you know?",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "I don't spend time in traffic anymore and I love it.",
    "Machine learning systems can now generate speech that sounds remarkably natural.",
    "Please arrive fifteen minutes early so we can get you checked in.",
    "The weather forecast says heavy rain is expected across the valley tomorrow afternoon.",
    "She sells seashells by the seashore, and the shells she sells are surely seashells.",
    "Our flight departs at seven forty five in the morning from gate twenty two.",
]


def main():
    from chatterbox_flash import ChatterboxFlashTTS
    from server.engine import FlashEngine, Request
    from server.vocoder import Vocoder
    from server.app import _prep_factory

    t0 = time.time()
    tts = ChatterboxFlashTTS.from_pretrained("ResembleAI/chatterbox-flash", device="cuda")
    tts.prepare_conditionals("assets/ref.wav", exaggeration=0.5)
    with torch.inference_mode():
        cond_emb = tts.t3.prepare_conditioning(tts.conds.t3)
    log.info("loaded in %.1fs cond_emb=%s", time.time() - t0, tuple(cond_emb.shape))

    engine = FlashEngine(tts.t3, cond_emb, max_active=16)
    voc = Vocoder(tts.s3gen, tts.conds.gen, engine.dev)
    engine.vocoder = voc
    threading.Thread(target=engine.run, daemon=True).start()
    threading.Thread(target=voc.run, daemon=True).start()

    prep = _prep_factory(tts, 160)

    for round_no in range(2):
        done = threading.Event()
        remaining = [len(TEXTS)]
        results = {}

        def make_sink(i):
            chunks = []
            tt = {"first": None}

            def sink(data):
                if data is None:
                    results[i]["pcm"] = b"".join(chunks)
                    remaining[0] -= 1
                    if remaining[0] == 0:
                        done.set()
                else:
                    if tt["first"] is None:
                        tt["first"] = time.perf_counter()
                        results[i]["t_first"] = tt["first"]
                    chunks.append(data)
            return sink

        t_start = time.perf_counter()
        reqs = []
        for i, text in enumerate(TEXTS):
            toks = prep(text)
            nsp = min(max(6 * toks.size(1), 300), engine.max_speech)
            nsp = (nsp + 15) // 16 * 16
            results[i] = {}
            r = Request(rid=i, text=text, text_tokens=toks, n_speech=nsp,
                        sink=make_sink(i))
            r.t_recv = time.perf_counter()
            reqs.append(r)
            engine.submit(r)

        assert done.wait(timeout=600), "timed out"
        wall = time.perf_counter() - t_start

        total_audio = 0.0
        for i, r in enumerate(reqs):
            pcm = np.frombuffer(results[i]["pcm"], dtype=np.int16).astype(np.float32) / 32767
            dur = len(pcm) / 24000
            total_audio += dur
            ttfp = results[i].get("t_first", t_start) - r.t_recv
            rms = float(np.sqrt((pcm ** 2).mean())) if len(pcm) else 0.0
            log.info(
                "req %d: %5.2fs audio  ttfp %5.3fs  toks %3d  rms %.3f  | %s",
                i, dur, ttfp, r.n_tokens, rms, r.text[:40],
            )
            if round_no == 1:
                sf.write(f"/tmp/flash_test_{i}.wav", pcm, 24000)
        log.info("round %d: wall %.2fs  audio %.1fs  xRT %.1f  engine=%s voc=%s",
                 round_no, wall, total_audio, total_audio / wall,
                 engine.stats(), voc.stats())


if __name__ == "__main__":
    main()
