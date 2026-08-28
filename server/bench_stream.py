"""One-shot sustained-stream benchmark for the production TTS contract.

Starts N long TTS requests together.  Unlike bench.py's closed-loop utterance
load, every connection remains continuously active, and chunk timing is used
to verify that audio can play at 1x without underruns after a fixed startup
buffer while generation remains at or above the requested XRT.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time

import aiohttp
BYTES_PER_AUDIO_S = 24000 * 2

TEXT = (
    "The quarterly planning meeting began shortly after sunrise. Representatives "
    "from every department reviewed customer feedback, engineering milestones, "
    "operating costs, and the delivery schedule. After careful review, the "
    "committee approved the proposal."
)
LONG_TEXT = (
    "The quarterly planning meeting began shortly after sunrise. Representatives "
    "from every department reviewed customer feedback, engineering milestones, "
    "operating costs, and the delivery schedule for the coming year. The team "
    "agreed to simplify the product, improve reliability, document every critical "
    "workflow, and measure progress against clear outcomes. After a careful review, "
    "the committee approved the proposal and assigned each action to an owner. "
    "Everyone left with a shared understanding of the priorities and the work ahead."
)
VERY_LONG_TEXT = LONG_TEXT + " " + LONG_TEXT




def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


async def one(session: aiohttp.ClientSession, url: str, text: str,
              start: asyncio.Event, startup_s: float, target_xrt: float,
              delay_s: float = 0.0, stream_seconds: float = 0.0,
              continuation_delay_s: float = 0.0,
              api_segmentation: bool = True,
              model_id: str | None = None) -> dict:
    await start.wait()
    if delay_s:
        await asyncio.sleep(delay_s)
    t0 = time.perf_counter()
    events: list[tuple[float, float]] = []
    nbytes = 0
    backend_shard = None
    cycles = 0
    end_at = t0 + stream_seconds
    while True:
        async with session.post(
                f"{url}/tts",
                json={
                    "text": text, "segment": api_segmentation,
                    **({"model": model_id} if model_id else {}),
                }) as response:
            if backend_shard is None:
                backend_shard = response.headers.get("X-Backend-Shard")
            response.raise_for_status()
            while True:
                data, _ = await response.content.readchunk()
                if not data:
                    if response.content.at_eof():
                        break
                    continue
                now = time.perf_counter()
                audio_s = len(data) / BYTES_PER_AUDIO_S
                nbytes += len(data)
                events.append((now, audio_s))
        cycles += 1
        if stream_seconds <= 0.0 or time.perf_counter() >= end_at:
            break
        if cycles == 1 and continuation_delay_s > 0.0:
            await asyncio.sleep(continuation_delay_s)
    done = time.perf_counter()
    if not events:
        raise RuntimeError("empty stream")

    first = events[0][0]
    total_audio = nbytes / BYTES_PER_AUDIO_S
    generation_s = max(1e-9, events[-1][0] - first)

    # Playback starts startup_s after request submission.  Immediately before
    # each chunk arrives, consume wall time since the prior event; a negative
    # buffer is an audible underrun.  Include the tail through response EOF.
    play_start = t0 + startup_s
    buffer_s = 0.0
    min_buffer_s = 0.0
    last = play_start
    max_gap_s = 0.0
    for at, audio_s in events:
        if at > play_start:
            buffer_s -= at - max(last, play_start)
            min_buffer_s = min(min_buffer_s, buffer_s)
            max_gap_s = max(max_gap_s, at - max(last, play_start))
        buffer_s += audio_s
        last = max(at, play_start)
    if done > last:
        buffer_s -= done - last
        min_buffer_s = min(min_buffer_s, buffer_s)

    return {
        "ttfb": first - t0,
        "shard": url,
        "backend_shard": backend_shard,
        "audio_s": total_audio,
        "total_s": done - t0,
        "generation_xrt": total_audio / generation_s,
        "target_xrt_met": total_audio / generation_s >= target_xrt,
        "max_chunk_gap_s": max_gap_s,
        "underrun_s": max(0.0, -min_buffer_s),
        "smooth": min_buffer_s >= 0.0,
        "chunks": len(events),
        "cycles": cycles,
        "_events": events,
        "_started": t0,
        "_done": done,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", nargs="+", default=["http://127.0.0.1:8020"])
    parser.add_argument("--concurrency", type=int, default=1000)
    parser.add_argument("--startup", type=float, default=0.4)
    parser.add_argument("--target-xrt", type=float, default=1.5)
    parser.add_argument("--out", default="/tmp/turbo_sustained.json")
    parser.add_argument("--ramp", type=float, default=0.0,
                        help="seconds over which the concurrent connections start")
    parser.add_argument("--long", action="store_true",
                        help="use a 300-token utterance (server max-text >=384)")
    parser.add_argument("--very-long", action="store_true",
                        help="use a 600-token utterance (server max-text >=768)")
    parser.add_argument("--weights", type=int, nargs="+",
                        help="per-URL stream weights/counts for weighted routing")
    parser.add_argument("--stream-seconds", type=float, default=0.0,
                        help="keep each logical stream active for at least this long")
    parser.add_argument("--no-fill-barrier", action="store_true",
                        help="allow continuations while the fleet is still filling")
    parser.add_argument("--no-api-segmentation", action="store_true",
                        help="send each benchmark utterance as one backend request")
    parser.add_argument(
        "--model", nargs="+",
        help="model/LoRA IDs assigned round-robin to logical streams",
    )
    args = parser.parse_args()

    if args.weights is not None and len(args.weights) != len(args.url):
        parser.error("--weights must contain one value per --url")
    text = VERY_LONG_TEXT if args.very_long else (LONG_TEXT if args.long else TEXT)
    weights = args.weights or [1] * len(args.url)
    assigned = [0] * len(args.url)
    routes = []
    for _ in range(args.concurrency):
        shard = min(range(len(args.url)),
                    key=lambda j: assigned[j] / max(1, weights[j]))
        routes.append(args.url[shard])
        assigned[shard] += 1
    connector = aiohttp.TCPConnector(limit=args.concurrency + 16)
    timeout = aiohttp.ClientTimeout(total=600, sock_read=600)
    start = asyncio.Event()
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for i in range(args.concurrency):
            delay = args.ramp * i / max(1, args.concurrency - 1)
            tasks.append(asyncio.create_task(one(
                session, routes[i], text, start, args.startup, args.target_xrt,
                delay, args.stream_seconds,
                0.0 if args.no_fill_barrier else max(0.0, args.ramp - delay),
                not args.no_api_segmentation,
                args.model[i % len(args.model)] if args.model else None,
            )))
        wall_start = time.perf_counter()
        start.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        wall_s = time.perf_counter() - wall_start

    errors = [repr(r) for r in results if isinstance(r, BaseException)]
    good = [r for r in results if not isinstance(r, BaseException)]
    ttfb = [r["ttfb"] for r in good]
    xrt = [r["generation_xrt"] for r in good]
    gaps = [r["max_chunk_gap_s"] for r in good]
    audio = sum(r["audio_s"] for r in good)
    overlap_start = wall_start + args.ramp
    overlap_end = min((r["_done"] for r in good), default=overlap_start)
    overlap_audio = sum(
        audio_s for r in good for at, audio_s in r["_events"]
        if overlap_start <= at <= overlap_end
    )
    overlap_s = max(0.0, overlap_end - overlap_start)
    report = {
        "concurrency": args.concurrency,
        "shards": len(args.url),
        "startup_s": args.startup,
        "target_xrt": args.target_xrt,
        "wall_s": round(wall_s, 3),
        "ramp_s": args.ramp,
        "all_active_overlap_s": round(overlap_s, 3),
        "all_active_audio_xrt": round(overlap_audio / overlap_s, 2) if overlap_s else 0.0,
        "streams": len(good),
        "errors": len(errors),
        "aggregate_audio_xrt": round(audio / wall_s, 2),
        "ttfb_p50": round(percentile(ttfb, .50), 3),
        "ttfb_p95": round(percentile(ttfb, .95), 3),
        "ttfb_p99": round(percentile(ttfb, .99), 3),
        "generation_xrt_p01": round(percentile(xrt, .01), 3),
        "generation_xrt_p50": round(percentile(xrt, .50), 3),
        "max_chunk_gap_p95": round(percentile(gaps, .95), 3),
        "smooth_streams": sum(r["smooth"] for r in good),
        "target_xrt_streams": sum(r["target_xrt_met"] for r in good),
        "max_underrun_s": round(max((r["underrun_s"] for r in good), default=0), 3),
        "error_kinds": errors,
        "results": good,
    }
    for result in good:
        result.pop("_events", None)
        result.pop("_started", None)
        result.pop("_done", None)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    import uvloop
    uvloop.install()
    asyncio.run(main())
