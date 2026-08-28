"""Continuous-batching Chatterbox-Flash/Turbo server (single GPU, ROCm).

POST /tts        {"text": "..."}  -> chunked stream of raw PCM s16le @ 24 kHz
GET  /healthz    liveness
GET  /stats      engine/vocoder gauges + rolling request metrics

Run:
    python -m server.app --port 8020 --max-active 128
"""

from __future__ import annotations

import argparse
import gc
import asyncio
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import median

import torch
from server.models import (
    FLASH_MODEL,
    TURBO_MODEL,
    canonical_model,
    parse_model_options,
    worker_capabilities,
)
from server.http_api import (
    OPENAI_BUILTIN_VOICES,
    PcmStream,
    internal_tts,
    openai_models,
    openai_speech,
)

logger = logging.getLogger("chatterbox.server")


def _prep_factory(tts, max_text_tokens):
    import torch.nn.functional as F
    from chatterbox_flash.text_norm import en_us_cleaner

    lock = threading.Lock()

    def prep(text: str):
        with lock:
            norm = en_us_cleaner(text)
        toks = tts.tokenizer.text_to_tokens(norm)          # (1, Lt) cpu
        toks = F.pad(toks, (1, 0), value=tts.t3.hp.start_text_token)
        toks = F.pad(toks, (0, 1), value=tts.t3.hp.stop_text_token)
        if toks.size(1) > max_text_tokens:
            raise ValueError(f"text too long: {toks.size(1)} > {max_text_tokens} tokens")
        return toks

    return prep


def _prep_factory_turbo(tokenizer, max_text_tokens):
    from chatterbox.tts_turbo import punc_norm

    def prep(text: str):
        toks = tokenizer(punc_norm(text), return_tensors="pt",
                         truncation=True).input_ids            # (1, Lt) cpu
        if toks.size(1) > max_text_tokens:
            raise ValueError(f"text too long: {toks.size(1)} > {max_text_tokens} tokens")
        return toks

    return prep


def build(args):
    from server.voc_client import VocClient, SplitVoc

    # spawn the vocoder subprocess first so its model load overlaps ours
    voc_cfg = dict(
        model=args.model,
        repo="ResembleAI/chatterbox-flash" if args.model == "flash"
             else "ResembleAI/chatterbox-turbo",
        ref_wav=args.ref_wav,
        voc_ref_seconds=args.voc_ref_seconds,
        voc_dtype=args.voc_dtype,
        chunk_first=args.chunk_first,
        chunk_second=args.chunk_second,
        topup_sizes=args.topup_sizes,
        topup_until=args.topup_until,
        chunk=args.chunk,
        lookback=args.lookback,
        voc_batch=args.voc_batch,
        first_voc_batch=args.first_voc_batch,
        pre_roll=round(args.pre_roll_ms * 24),
        n_cfm=args.n_cfm,
        fused_snake=args.fused_snake,
        procs=args.voc_procs,
        device=args.voc_device,
        devices=args.voc_devices,
    )
    vocoder = None if args.in_process_vocoder else VocClient(voc_cfg)
    first_client = (
        VocClient(dict(voc_cfg, voc_batch=16, procs=1))
        if args.first_voc_proc and not args.in_process_vocoder else None
    )

    device = "cuda"
    logger.info("loading %s model...", args.model)
    if args.model == "flash" and args.lora_modules:
        raise ValueError("dynamic LoRA is currently supported for Turbo only")
    if args.model == "flash":
        from chatterbox_flash import ChatterboxFlashTTS
        from server.engine import FlashEngine

        tts = ChatterboxFlashTTS.from_pretrained("ResembleAI/chatterbox-flash", device=device)
        tts.prepare_conditionals(args.ref_wav, exaggeration=0.5)
        with torch.inference_mode():
            cond_emb = tts.t3.prepare_conditioning(tts.conds.t3)
        engine = FlashEngine(
            tts.t3, cond_emb,
            max_active=args.max_active,
            max_text_tokens=args.max_text_tokens,
            max_speech_tokens=args.max_speech_tokens,
            cfg_scale=args.cfg_scale,
        )
        prep = _prep_factory(tts, args.max_text_tokens)

        def speech_budget(toks) -> int:
            n = max(6 * toks.size(1), 300)
            n = min(n, engine.max_speech)
            return (n + 15) // 16 * 16
    else:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        from server.engine_ar import TurboEngine

        tts = ChatterboxTurboTTS.from_pretrained(device=device)
        tts.prepare_conditionals(args.ref_wav)
        tts.t3.to(torch.bfloat16)
        tts.conds.t3.to(dtype=torch.bfloat16)
        from server.lora import LoRABank
        lora_bank = LoRABank.load(
            args.lora_modules,
            t3=tts.t3,
            device=torch.device(device),
            dtype=torch.bfloat16,
            max_rank=args.max_lora_rank,
            base_model_id=args.base_model_id,
        )
        logger.info("registered LoRA models: %s", lora_bank.names)
        with torch.inference_mode():
            cond_emb = tts.t3.prepare_conditioning(tts.conds.t3)
        engine = TurboEngine(
            tts.t3, cond_emb,
            max_active=args.max_active,
            max_text_tokens=args.max_text_tokens,
            max_speech_tokens=args.max_speech_tokens,
            top_k=args.top_k,
            lora_bank=lora_bank,
        )
        prep = _prep_factory_turbo(tts.tokenizer, args.max_text_tokens)

        def speech_budget(toks) -> int:
            return engine.max_speech

    if args.in_process_vocoder:
        from server.vocoder import Vocoder

        if args.voc_dtype == "bf16":
            tts.s3gen.flow.to(torch.bfloat16)
        if args.fused_snake:
            from server.kernels import install_fused_snake
            install_fused_snake(tts.s3gen.mel2wav)
        vocoder = Vocoder(
            tts.s3gen, tts.conds.gen, torch.device(device),
            chunk_first=args.chunk_first,
            chunk_second=args.chunk_second,
            topup_sizes=tuple(args.topup_sizes or ()),
            topup_until=args.topup_until,
            chunk=args.chunk,
            lookback=args.lookback,
            pre_roll=round(args.pre_roll_ms * 24),
            max_batch=args.voc_batch,
            first_max_batch=args.first_voc_batch,
            n_cfm_timesteps=args.n_cfm,
        )
        threading.Thread(
            target=vocoder.run, name="vocoder", daemon=True,
        ).start()
        engine.vocoder = vocoder
        return tts, engine, vocoder, prep, speech_budget

    # S3Gen lives in the vocoder subprocess; free the parent copy.
    tts.s3gen = None
    torch.cuda.empty_cache()
    assert vocoder is not None
    if first_client is None:
        vocoder.wait_ready()
        engine.vocoder = vocoder
        return tts, engine, vocoder, prep, speech_budget
    split = SplitVoc(first_client, vocoder)
    split.wait_ready()
    engine.vocoder = split
    return tts, engine, split, prep, speech_budget


def _warmup(engine, prep, speech_budget, n=8):
    """Push a few requests through the full stack to trigger MIOpen tuning
    and kernel JIT before any timed traffic."""
    from server.engine import Request

    done = threading.Event()
    remaining = [n]
    texts = [
        "This is a warm up utterance to compile all the kernels.",
        "The quick brown fox jumps over the lazy dog near the river bank today.",
        "Testing one two three, testing the vocoder path with a longer sentence to cover more shapes.",
    ]

    def sink_factory():
        def sink(data):
            if data is None:
                remaining[0] -= 1
                if remaining[0] == 0:
                    done.set()
        return sink

    t0 = time.perf_counter()
    for i in range(n):
        text = texts[i % len(texts)]
        toks = prep(text)
        r = Request(rid=-1 - i, text=text, text_tokens=toks,
                    n_speech=speech_budget(toks), sink=sink_factory())
        r.t_recv = time.perf_counter()
        engine.submit(r)
    if not done.wait(timeout=300):
        raise RuntimeError("warmup timed out")
    logger.info("warmup done in %.1fs", time.perf_counter() - t0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8020)
    p.add_argument("--ref-wav", default="assets/ref.wav")
    p.add_argument("--model", choices=["flash", "turbo"], default="flash")
    p.add_argument("--max-active", type=int, default=128)
    p.add_argument("--max-text-tokens", type=int, default=160)
    p.add_argument("--first-voc-batch", type=int, default=24)
    p.add_argument("--max-speech-tokens", type=int, default=624)
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--chunk-first", type=int, default=12)
    p.add_argument("--chunk-second", type=int, default=0,
                   help="priority top-up chunk size; 0 disables")
    p.add_argument("--topup-sizes", type=int, nargs="+",
                   help="ordered priority top-up sizes after the first chunk")
    p.add_argument("--top-k", type=int, default=1000,
                   help="Turbo sampler candidate count")
    p.add_argument("--topup-until", type=int, default=0,
                   help="priority top-ups repeat until this emitted-token boundary")
    p.add_argument("--chunk", type=int, default=80)
    p.add_argument("--pre-roll-ms", type=float, default=0.0,
                   help="leading PCM jitter buffer; does not delay TTFB")
    p.add_argument("--lookback", type=int, default=16)
    p.add_argument("--voc-batch", type=int, default=48)
    p.add_argument("--voc-ref-seconds", type=float, default=6.0)
    p.add_argument("--voc-dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--n-cfm", type=int, default=None,
                   help="CFM steps; default 1 for turbo (distilled), 2 for flash")
    p.add_argument("--voc-procs", type=int, default=2)
    p.add_argument("--voc-device", default="cuda",
                   help="vocoder device visible to subprocesses, e.g. cuda:1")
    p.add_argument("--voc-devices", nargs="+",
                   help="one logical CUDA device per vocoder worker")
    p.add_argument("--in-process-vocoder", action="store_true",
                   help="run S3Gen in the engine process on a separate HIP stream")
    p.add_argument("--fused-snake", action="store_true",
                   help="use the native fused HIP HiFT Snake kernel")
    p.add_argument("--first-voc-proc", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="dedicated first-chunk process; disable on a dedicated vocoder GPU")
    p.add_argument(
        "--lora-modules", nargs="*", default=[],
        metavar="NAME=PATH",
        help="static PEFT adapters exposed as request model IDs",
    )
    p.add_argument("--max-lora-rank", type=int, default=64)
    p.add_argument(
        "--base-model-id", default=None,
        help="model request value selecting the unadapted base",
    )
    args = p.parse_args()
    if args.n_cfm is None:
        args.n_cfm = 1 if args.model == "turbo" else 2
    if args.base_model_id is None:
        args.base_model_id = TURBO_MODEL if args.model == "turbo" else FLASH_MODEL
    else:
        args.base_model_id = canonical_model(args.base_model_id)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    import uvloop
    uvloop.install()
    from aiohttp import web

    from server.engine import Request

    tts, engine, vocoder, prep, speech_budget = build(args)
    lora_bank = getattr(engine, "lora_bank", None)
    voices = ["base", *(lora_bank.names if lora_bank is not None else [])]
    capabilities = worker_capabilities(
        args.base_model_id,
        voices=voices,
        max_input=args.max_text_tokens,
        input_unit="tokens",
        model_options={
            "voice": {
                "type": "string",
                "enum": voices,
                "default": "base",
                "description": (
                    "Turbo PEFT adapter" if args.model == "turbo"
                    else "preloaded reference conditioning"
                ),
            },
        },
    )
    logger.info("capabilities=%s", json.dumps(capabilities, sort_keys=True))

    t_engine = threading.Thread(target=engine.run, name="t3-engine", daemon=True)
    t_engine.start()
    _warmup(engine, prep, speech_budget)
    gc.collect()
    gc.freeze()
    gc.set_threshold(100_000, 100, 100)

    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="prep")
    rid_counter = [0]
    metrics_lock = threading.Lock()
    metrics: list[dict] = []

    async def create_pcm_stream(body: dict) -> PcmStream:
        loop = asyncio.get_running_loop()
        text = str(body["text"]).strip()
        if not text:
            raise ValueError("text must not be empty")
        requested_model = canonical_model(str(
            body.get("model", args.base_model_id)
        ))
        if requested_model != args.base_model_id:
            raise ValueError(f"unknown model: {requested_model}")
        options = parse_model_options(body, {"voice"})
        voice = str(options.get("voice", "base"))
        if voice not in voices and voice in OPENAI_BUILTIN_VOICES:
            voice = "base"
        if lora_bank is not None:
            lora_slot = lora_bank.resolve(voice)
        elif voice != "base":
            raise ValueError(f"unknown voice: {voice}")
        else:
            lora_slot = 0
        priority = 1 if body.get("priority") == "background" else 0
        toks = await loop.run_in_executor(pool, prep, text)
        nsp = speech_budget(toks)

        rid_counter[0] += 1
        rid = rid_counter[0]
        queue: asyncio.Queue = asyncio.Queue()

        def sink(data):
            loop.call_soon_threadsafe(queue.put_nowait, data)

        req = Request(
            rid=rid,
            text=text,
            text_tokens=toks,
            n_speech=nsp,
            sink=sink,
            lora_slot=lora_slot,
            model_id=args.base_model_id,
            priority=priority,
        )
        req.t_recv = time.perf_counter()

        async def chunks():
            engine.submit(req)
            try:
                while True:
                    data = await asyncio.wait_for(queue.get(), timeout=300)
                    if data is None:
                        break
                    yield data
            except asyncio.TimeoutError:
                logger.warning("generation timeout rid=%d", rid)
            finally:
                if req.t_done:
                    metric = {
                        "rid": rid,
                        "queue_s": (
                            round(req.t_admit - req.t_recv, 4)
                            if req.t_admit else None
                        ),
                        "ttfp_s": (
                            round(req.t_first_pcm - req.t_recv, 4)
                            if req.t_first_pcm else None
                        ),
                        "total_s": round(req.t_done - req.t_recv, 4),
                        "audio_s": round(req.audio_samples / 24000, 3),
                        "n_tokens": req.n_tokens,
                    }
                    with metrics_lock:
                        metrics.append(metric)
                        if len(metrics) > 200000:
                            del metrics[:100000]

        return PcmStream(chunks(), {
            "Content-Type": "application/octet-stream",
            "X-Sample-Rate": "24000",
            "X-Request-Id": str(rid),
            "X-Model": args.base_model_id,
            "X-Voice": voice,
        })

    async def tts_handler(request: "web.Request"):
        return await internal_tts(request, create_pcm_stream)

    async def openai_speech_handler(request: "web.Request"):
        return await openai_speech(
            request, create_pcm_stream, default_model=args.base_model_id,
        )
    async def stats_handler(request):
        with metrics_lock:
            done = list(metrics[-2000:])
            n_total = len(metrics)
        ttfp = sorted(m["ttfp_s"] for m in done if m["ttfp_s"])
        s = dict(engine=engine.stats(), vocoder=vocoder.stats(), completed=n_total)
        if ttfp:
            s["ttfp_recent"] = dict(
                p50=ttfp[len(ttfp) // 2],
                p95=ttfp[int(len(ttfp) * 0.95)],
                n=len(ttfp),
            )
        return web.json_response(s)

    async def health_handler(request):
        return web.json_response({
            "ok": True,
            "base_model": args.base_model_id,
            "models": [args.base_model_id],
            "voices": voices,
            "capabilities": capabilities,
            "lora_adapters": (
                lora_bank.metadata() if lora_bank is not None else []
            ),
        })

    async def models_handler(request):
        return openai_models(capabilities, owned_by="resemble-ai")
    app = web.Application()
    app.router.add_post("/tts", tts_handler)
    app.router.add_post("/v1/audio/speech", openai_speech_handler)
    app.router.add_get("/v1/models", models_handler)
    app.router.add_get("/stats", stats_handler)
    app.router.add_get("/healthz", health_handler)
    logger.info("serving on %s:%d", args.host, args.port)
    web.run_app(app, host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
