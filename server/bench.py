"""Load benchmark for the Chatterbox-Flash streaming TTS server.

Closed-loop concurrency sweep: C virtual users each hold one connection and
issue back-to-back POST /tts requests for a fixed duration. Per request we
record:

  * TTFP  — time to first PCM byte (request sent -> first body chunk)
  * TTLB  — time to last byte (full utterance latency)
  * audio duration, server RTF, delivery realtime factor

Aggregates: p50/p90/p95/p99 TTFP & TTLB, request throughput, aggregate
audio-seconds generated per wall second (xRT).

    python -m server.bench --url http://127.0.0.1:8020 \
        --concurrency 1 8 32 128 512 1000 2000 --duration 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time

import aiohttp

TEXTS = [
    "Sometimes it's better to just let things slide, you know?",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "I don't spend time in traffic anymore and I love it.",
    "Machine learning systems can now generate speech that sounds remarkably natural.",
    "Please arrive fifteen minutes early so we can get you checked in.",
    "The weather forecast says heavy rain is expected across the valley tomorrow afternoon.",
    "She sells seashells by the seashore, and the shells she sells are seashells for sure.",
    "Our flight departs at seven forty five in the morning from gate twenty two.",
    "Thank you for calling customer support, how can I help you today?",
    "The meeting has been rescheduled to next Thursday at three in the afternoon.",
    "A gentle breeze drifted through the open window carrying the scent of rain.",
    "Could you send me the quarterly report before the end of the day?",
    "He finished the marathon in just under four hours despite the heat.",
    "The museum's new exhibit features artifacts from the early bronze age.",
    "Remember to water the plants while we are away next week.",
    "Interest rates are expected to remain unchanged for the rest of the year.",
]

# Length-mixed corpus approximating production traffic: ~30% short prompts,
# ~50% medium sentences, ~20% long multi-clause ones; includes numerals,
# currency, dates and abbreviations to exercise text normalization.
TEXTS_MIXED = [
    # short (agent acknowledgements, UI strings)
    "Sure, give me one moment.",
    "Your order has shipped.",
    "I didn't catch that, could you repeat it?",
    "The total comes to $42.50.",
    # medium
    "Your appointment with Dr. Alvarez is confirmed for March 3rd at 2:30 PM.",
    "The package weighing 4.2 kg was delivered to 118 Elm St. this morning.",
    "Traffic on I-95 is backed up for about 12 miles, so plan an extra 45 minutes.",
    "To reset your password, enter the 6 digit code we just sent to your phone.",
    "The quarterly revenue grew 18% year over year, beating analyst expectations.",
    "A cold front moving in from the northwest will drop temperatures to 28°F overnight.",
    "Flight UA 2301 to Denver now boards at gate B14 instead of gate C7.",
    "The museum opens at 9 AM on weekdays and stays open until 8 PM on Saturdays.",
    # long (narration, article sentences)
    "After weeks of negotiation, the two companies announced a merger agreement valued "
    "at roughly $3.4 billion, pending approval from regulators in both countries.",
    "The hikers followed the narrow trail along the ridge for nearly two hours, pausing "
    "occasionally to photograph the valley below, before descending toward the lake as "
    "the light began to fade.",
    "Researchers at the university published a study on Tuesday suggesting that regular "
    "short walks throughout the day may be more beneficial for long term health than a "
    "single extended workout session.",
    "When the storm finally passed, residents emerged to find streets littered with "
    "branches, several power lines down along the main road, and the river running "
    "higher than it had in almost a decade.",
]

SR = 24000
BYTES_PER_S = SR * 2


async def worker(session, urls, corpus, stop_at, results, errors, wid,
                 realtime=False, gap=0.0):
    rng = random.Random(wid)
    url = urls[wid % len(urls)]
    while time.perf_counter() < stop_at:
        text = rng.choice(corpus)
        t0 = time.perf_counter()
        ttfp = None
        nbytes = 0
        try:
            async with session.post(f"{url}/tts", json={"text": text}) as resp:
                if resp.status != 200:
                    errors.append(f"http {resp.status}")
                    continue
                async for chunk in resp.content.iter_chunked(65536):
                    if ttfp is None and chunk:
                        ttfp = time.perf_counter() - t0
                    nbytes += len(chunk)
            t1 = time.perf_counter()
            if ttfp is None or nbytes == 0:
                errors.append("empty")
                continue
            audio_s = nbytes / BYTES_PER_S
            results.append(dict(
                ttfp=ttfp,
                ttlb=t1 - t0,
                audio_s=audio_s,
                stall=max(0.0, (t1 - t0 - ttfp) - audio_s),
            ))
            if realtime:
                # live listener: play the audio out in real time, then idle
                # for a conversational gap before the next utterance
                play_left = max(0.0, audio_s - (t1 - t0 - ttfp))
                await asyncio.sleep(play_left + gap)
        except Exception as e:
            errors.append(type(e).__name__)


def pct(v, q):
    if not v:
        return float("nan")
    v = sorted(v)
    i = min(len(v) - 1, int(q * len(v)))
    return v[i]


async def run_level(urls, corpus, conc, duration, ramp_s, realtime=False, gap=0.0):
    conn = aiohttp.TCPConnector(limit=conc + 16, force_close=False)
    timeout = aiohttp.ClientTimeout(total=600, sock_read=600)
    results: list[dict] = []
    errors: list[str] = []
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as s:
        t_start = time.perf_counter()
        stop_at = t_start + duration
        tasks = []
        for w in range(conc):
            tasks.append(asyncio.create_task(
                worker(s, urls, corpus, stop_at, results, errors, w,
                       realtime=realtime, gap=gap)))
            if ramp_s and conc > 1:
                await asyncio.sleep(ramp_s / conc)
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - t_start
        # server stats snapshot (first shard)
        try:
            async with s.get(f"{urls[0]}/stats") as r:
                server_stats = await r.json()
        except Exception:
            server_stats = {}

    ttfp = [r["ttfp"] for r in results]
    ttlb = [r["ttlb"] for r in results]
    audio = sum(r["audio_s"] for r in results)
    out = dict(
        concurrency=conc,
        duration_s=round(wall, 1),
        requests=len(results),
        errors=len(errors),
        rps=round(len(results) / wall, 2),
        audio_xrt=round(audio / wall, 1),
        ttfp_p50=round(pct(ttfp, 0.50), 3),
        ttfp_p90=round(pct(ttfp, 0.90), 3),
        ttfp_p95=round(pct(ttfp, 0.95), 3),
        ttfp_p99=round(pct(ttfp, 0.99), 3),
        ttfp_max=round(max(ttfp), 3) if ttfp else None,
        ttlb_p50=round(pct(ttlb, 0.50), 3),
        ttlb_p95=round(pct(ttlb, 0.95), 3),
        ttlb_p99=round(pct(ttlb, 0.99), 3),
        stall_p95=round(pct([r["stall"] for r in results], 0.95), 3),
        avg_audio_s=round(audio / max(1, len(results)), 2),
        server=server_stats.get("engine", {}),
    )
    if errors:
        from collections import Counter
        out["error_kinds"] = dict(Counter(errors))
    return out


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", nargs="+", default=["http://127.0.0.1:8020"],
                   help="one or more shard URLs; workers round-robin")
    p.add_argument("--corpus", choices=["short", "mixed"], default="short")
    p.add_argument("--concurrency", type=int, nargs="+",
                   default=[1, 8, 32, 128, 512, 1000, 2000])
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--ramp", type=float, default=2.0,
                   help="seconds over which workers are started")
    p.add_argument("--realtime", action="store_true",
                   help="pace each worker at 1x playback + --gap between utterances")
    p.add_argument("--gap", type=float, default=0.0,
                   help="idle seconds between utterances per stream (realtime mode)")
    p.add_argument("--out", default="/tmp/flash_bench.json")
    args = p.parse_args()

    corpus = TEXTS if args.corpus == "short" else TEXTS_MIXED
    all_out = []
    for c in args.concurrency:
        print(f"=== concurrency {c} (duration {args.duration}s, "
              f"{len(args.url)} shards, corpus={args.corpus}) ===", flush=True)
        r = await run_level(args.url, corpus, c, args.duration, args.ramp,
                            realtime=args.realtime, gap=args.gap)
        all_out.append(r)
        print(json.dumps(r, indent=2), flush=True)
        with open(args.out, "w") as f:
            json.dump(all_out, f, indent=2)
        await asyncio.sleep(3)

    # summary table
    hdr = f"{'conc':>5} {'reqs':>6} {'rps':>7} {'xRT':>7} {'ttfp p50':>9} {'p95':>7} {'p99':>7} {'ttlb p50':>9} {'p95':>7} {'err':>5}"
    print("\n" + hdr)
    for r in all_out:
        print(f"{r['concurrency']:>5} {r['requests']:>6} {r['rps']:>7} {r['audio_xrt']:>7} "
              f"{r['ttfp_p50']:>9} {r['ttfp_p95']:>7} {r['ttfp_p99']:>7} "
              f"{r['ttlb_p50']:>9} {r['ttlb_p95']:>7} {r['errors']:>5}")


if __name__ == "__main__":
    import uvloop
    uvloop.install()
    asyncio.run(main())
