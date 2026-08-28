"""Continuously batched Kokoro-82M streaming server."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch
from aiohttp import web
from server.models import (
    KOKORO_MODEL,
    canonical_model,
    parse_model_options,
    worker_capabilities,
)
from server.scheduler import SchedulerQueue
from server.http_api import (
    OPENAI_BUILTIN_VOICES,
    PcmStream,
    internal_tts,
    openai_models,
    openai_speech,
)

KOKORO_OPENAI_VOICES = {
    "alloy": "af_alloy",
    "echo": "am_echo",
    "fable": "bm_fable",
    "nova": "af_nova",
    "onyx": "am_onyx",
}

logger = logging.getLogger("kokoro.server")
SAMPLE_RATE = 24000
BASE_MODEL = KOKORO_MODEL
LANGUAGES = set("abefhijpz")
DEFAULT_WARMUP_TEXT = (
    "The quickest way to understand streaming speech is to hear it begin "
    "before the sentence is finished generating."
)
DEFAULT_VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
    "am_michael", "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    "ef_dora", "em_alex", "em_santa", "ff_siwis",
    "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
    "if_sara", "im_nicola",
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
    "pf_dora", "pm_alex", "pm_santa",
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
]


@dataclass
class KokoroRequest:
    rid: int
    phonemes: deque[str]
    token_ids: deque[list[int]]
    voice: str
    speed: float
    sink: object
    priority: int = 0
    cancelled: bool = False
    started: float = field(default_factory=time.perf_counter)
    audio_seconds: float = 0.0


class KokoroBatchScheduler:
    """Deadline-ordered continuous scheduler with exact-length microbatches."""

    def __init__(self, model, voice_packs: dict[str, torch.Tensor], device,
                 *, max_batch: int = 16,
                 background_batch_size: int = 32,
                 batch_wait_ms: float = 2.0,
                 max_batch_wait_ms: float = 500.0,
                 startup_buffer_ms: float = 400.0):
        self.model = model
        self.voice_packs = voice_packs
        self.device = torch.device(device)
        self.max_batch = max_batch
        self.background_batch_size = min(
            background_batch_size, max_batch,
        )
        self.batch_wait = batch_wait_ms / 1000
        self.startup_buffer = startup_buffer_ms / 1000
        self.max_batch_wait = max_batch_wait_ms / 1000
        self.scheduler: SchedulerQueue[KokoroRequest] = SchedulerQueue()
        self.ready = threading.Event()
        self.startup_error: BaseException | None = None
        self.stop = False
        self.requests = self.chunks = self.errors = self.batches = 0
        self.batch_rows = self.audio_samples = 0
        self.gpu_seconds = 0.0

    def submit(self, request: KokoroRequest) -> None:
        self.scheduler.submit(request)

    def depth(self) -> int:
        return self.scheduler.depth()

    def _token_ids(self, request: KokoroRequest) -> list[int]:
        return request.token_ids[0]

    def _batch_limit(
        self, request: KokoroRequest, token_length: int,
    ) -> int:
        if request.priority > 0:
            return self.background_batch_size
        if token_length <= 128 and request.speed >= 0.9:
            return self.max_batch
        return min(self.max_batch, 64)
    def _batch_buckets(self) -> list[int]:
        buckets = [
            size for size in (1, 8, 16, 24, 28, 32, 64, 128, 256, 512)
            if size <= self.max_batch
        ]
        if buckets[-1] != self.max_batch:
            buckets.append(self.max_batch)
        return buckets

    def _batch_bucket(self, size: int) -> int:
        return next(
            (
                bucket for bucket in self._batch_buckets()
                if bucket >= size
            ),
            self.max_batch,
        )

    def _compatible(
        self, seed: KokoroRequest, request: KokoroRequest,
    ) -> bool:
        seed_length = len(self._token_ids(seed)) + 2
        request_length = len(self._token_ids(request)) + 2
        return (
            request.priority == seed.priority
            and request_length == seed_length
            and self._batch_limit(request, request_length)
                == self._batch_limit(seed, seed_length)
        )

    def _take_batch(self) -> list[KokoroRequest]:
        selected = self.scheduler.take_group(
            select_key=lambda request: (
                request.priority,
                request.started + self.startup_buffer
                + request.audio_seconds,
                request.rid,
            ),
            limit_for=lambda request: self._batch_limit(
                request, len(self._token_ids(request)) + 2,
            ),
            compatible=self._compatible,
        )
        if not selected:
            return []
        first = selected[0]
        token_length = len(self._token_ids(first)) + 2
        batch_limit = self._batch_limit(first, token_length)
        deadline = time.perf_counter() + self.max_batch_wait
        quiet_rounds = 0
        while len(selected) < batch_limit and self.batch_wait:
            time.sleep(self.batch_wait)
            added = self.scheduler.extend_group(
                selected,
                limit=batch_limit,
                compatible=self._compatible,
            )
            quiet_rounds = 0 if added else quiet_rounds + 1
            if (
                time.perf_counter() >= deadline
                or (first.priority == 0 and quiet_rounds >= 5)
            ):
                break
        return selected

    @torch.inference_mode()
    def _forward(self, requests: list[KokoroRequest]) -> list[bytes]:
        real_count = len(requests)
        padded_count = self._batch_bucket(real_count)
        if padded_count > real_count:
            requests = [
                *requests,
                *([requests[-1]] * (padded_count - real_count)),
            ]
        token_rows = []
        styles = []
        speeds = []
        for request in requests:
            phonemes = request.phonemes[0]
            token_ids = self._token_ids(request)
            if not token_ids or len(token_ids) + 2 > self.model.context_length:
                raise ValueError("invalid Kokoro phoneme length")
            token_rows.append(torch.tensor(
                [0, *token_ids, 0], dtype=torch.long, device=self.device,
            ))
            styles.append(
                self.voice_packs[request.voice][min(len(phonemes), 510) - 1]
            )
            speeds.append(request.speed)
        # Exact token lengths avoid bidirectional recurrent padding effects.
        input_ids = torch.stack(token_rows)
        lengths = torch.full(
            (len(requests),), input_ids.shape[1],
            dtype=torch.long, device=self.device,
        )
        style = torch.cat(styles, dim=0)
        speed = torch.tensor(speeds, dtype=torch.float32, device=self.device)
        start = time.perf_counter()
        audio, audio_lengths = self.model.forward_with_tokens(
            input_ids, style, speed, lengths,
        )
        pcm_gpu = audio.clamp(-1, 1).mul(32767).round().to(torch.int16)
        lengths = audio_lengths.cpu().tolist()
        pcm = pcm_gpu.cpu().numpy().astype("<i2", copy=False)
        self.gpu_seconds += time.perf_counter() - start
        return [
            pcm[row, :lengths[row]].tobytes()
            for row in range(real_count)
        ]

    @torch.inference_mode()
    def warmup(self) -> None:
        voice_name = (
            "af_heart" if "af_heart" in self.voice_packs
            else next(iter(self.voice_packs))
        )
        samples = [
            "ðɪs ɪz ə mˈidiəm bˈætʃ wˈɔɹmʌp sˈɛntəns",
            (
                "ðə kwˈɪkɪst wˈA tʊ ˌʌndəɹstˈænd stɹˈimɪŋ spˈiʧ ɪz tə "
                "hˈɪɹ ɪt bəɡˈɪn bəfˈɔɹ ðə sˈɛntᵊns ɪz fˈɪnəʃt "
                "ʤˈɛnəɹˌATɪŋ."
            ),
            (
                "ðˌɪs bˈækɡɹˌWnd ɹəkwˈɛst kəntˈɪnjəwəsli "
                "ˈɛksəɹsˌIzᵻz ðə stɹˈimɪŋ spˈiʧ ˈɛnʤən wˌIl "
                "ˌɪntəɹˈæktɪv ʤˌɛnəɹˈAʃən ɹəmˈAnz əvˈAləbᵊl."
            ),
        ]
        for sample_index, phonemes in enumerate(samples):
            token_ids = [
                value for value in map(self.model.vocab.get, phonemes)
                if value is not None
            ]
            if sample_index == 0:
                batch_sizes = [1]
            elif sample_index == 1:
                batch_sizes = self._batch_buckets()
            else:
                batch_sizes = [1, self.background_batch_size]
            for batch_size in batch_sizes:
                requests = [
                    KokoroRequest(
                        rid=-row - 1,
                        phonemes=deque([phonemes]),
                        token_ids=deque([token_ids]),
                        voice=voice_name,
                        speed=1.0,
                        sink=lambda _: None,
                    )
                    for row in range(batch_size)
                ]
                self._forward(requests)
        self.gpu_seconds = 0.0
        logger.info(
            "Kokoro warmup complete for batches %s",
            self._batch_buckets(),
        )

    @torch.inference_mode()
    def run(self) -> None:
        torch.cuda.set_device(self.device)
        try:
            self.warmup()
        except BaseException as exc:
            self.startup_error = exc
            self.ready.set()
            raise
        self.ready.set()
        while not self.stop:
            requests = self._take_batch()
            if not requests:
                self.scheduler.wait(0.005)
                continue
            live = [request for request in requests if not request.cancelled]
            if not live:
                continue
            try:
                pcm_rows = self._forward(live)
                self.batches += 1
                self.batch_rows += len(live)
                for request, pcm in zip(live, pcm_rows):
                    request.phonemes.popleft()
                    request.token_ids.popleft()
                    self.audio_samples += len(pcm) // 2
                    request.audio_seconds += len(pcm) / (2 * SAMPLE_RATE)
                    self.chunks += 1
                    request.sink(pcm)
                    if request.phonemes and not request.cancelled:
                        self.scheduler.submit(request)
                    else:
                        self.requests += 1
                        request.sink(None)
            except Exception:
                self.errors += len(live)
                logger.exception("Kokoro batch of %d failed", len(live))
                for request in live:
                    request.sink(None)

    def stats(self) -> dict:
        return {
            "queue": self.depth(),
            "requests": self.requests,
            "chunks": self.chunks,
            "errors": self.errors,
            "batches": self.batches,
            "avg_batch": round(self.batch_rows / max(self.batches, 1), 2),
            "audio_xrt_gpu": round(
                self.audio_samples / SAMPLE_RATE / max(self.gpu_seconds, 1e-9), 2,
            ),
            "gpu_seconds": round(self.gpu_seconds, 3),
        }


def voice_language(voice: str) -> str:
    language = voice[0] if voice else ""
    if language not in LANGUAGES:
        raise ValueError(f"cannot infer language from Kokoro voice {voice!r}")
    return language


def split_text(text: str, limit: int) -> list[str]:
    parts = []
    rest = " ".join(text.split())
    while len(rest) > limit:
        window = rest[:limit + 1]
        cut = max(
            (
                window.rfind(marker) + 1
                for marker in (". ", "! ", "? ", "; ", ", ")
            ),
            default=-1,
        )
        if cut < limit // 2:
            cut = window.rfind(" ")
        if cut < 1:
            cut = limit
        parts.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        parts.append(rest)
    if len(parts) > 1 and len(parts[-1].split()) < 4:
        words = f"{parts[-2]} {parts[-1]}".split()
        if len(words) < 8:
            parts[-2:] = [" ".join(words)]
        else:
            midpoint = len(words) // 2
            parts[-2:] = [
                " ".join(words[:midpoint]),
                " ".join(words[midpoint:]),
            ]
    return parts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8030)
    parser.add_argument("--repo-id", default=BASE_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["fp32", "bf16", "fp16"], default="fp32",
    )
    parser.add_argument("--voices", nargs="+", default=DEFAULT_VOICES)
    parser.add_argument("--prep-workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--background-batch-size", type=int, default=32)
    parser.add_argument("--batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--max-batch-wait-ms", type=float, default=500.0)
    parser.add_argument("--startup-buffer-ms", type=float, default=400.0)
    parser.add_argument("--max-queue", type=int, default=4096)
    parser.add_argument("--max-chars", type=int, default=4096)
    parser.add_argument("--chunk-chars", type=int, default=512)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    from kokoro import KPipeline
    from huggingface_hub import snapshot_download
    from server.kokoro_batch import KModel

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.dtype]
    logger.info(
        "loading batched %s on %s as %s", args.repo_id, device, args.dtype,
    )
    model = KModel(repo_id=args.repo_id).to(
        device=device, dtype=dtype,
    ).eval()
    voice_dir = Path(snapshot_download(
        repo_id=args.repo_id, allow_patterns="voices/*.pt",
    )) / "voices"
    voice_packs = {
        voice: torch.load(
            voice_dir / f"{voice}.pt",
            map_location="cpu", weights_only=True,
        ).to(device=device, dtype=dtype)
        for voice in args.voices
    }
    default_voice = (
        "af_heart" if "af_heart" in voice_packs else next(iter(voice_packs))
    )
    capabilities = worker_capabilities(
        BASE_MODEL,
        voices=list(args.voices),
        max_input=args.max_chars,
        input_unit="characters",
        model_options={
            "voice": {
                "type": "string",
                "enum": list(args.voices),
                "default": default_voice,
            },
            "language": {
                "type": "string",
                "enum": sorted(LANGUAGES),
                "description": "Defaults to the selected voice's language",
            },
            "speed": {
                "type": "number",
                "minimum": 0.5,
                "maximum": 2.0,
                "default": 1.0,
            },
        },
    )
    logger.info("capabilities=%s", json.dumps(capabilities, sort_keys=True))
    scheduler = KokoroBatchScheduler(
        model, voice_packs, device,
        max_batch=args.batch_size,
        background_batch_size=args.background_batch_size,
        batch_wait_ms=args.batch_wait_ms,
        max_batch_wait_ms=args.max_batch_wait_ms,
        startup_buffer_ms=args.startup_buffer_ms,
    )
    threading.Thread(
        target=scheduler.run,
        name="kokoro-batch-gpu",
        daemon=True,
    ).start()
    scheduler.ready.wait()
    if scheduler.startup_error is not None:
        raise scheduler.startup_error

    thread_local = threading.local()
    prep_pool = ThreadPoolExecutor(
        max_workers=args.prep_workers,
        thread_name_prefix="kokoro-g2p",
    )
    prep_cache: OrderedDict[tuple[str, str], Future] = OrderedDict()
    prep_cache_lock = threading.Lock()
    def prepare(text: str, language: str) -> list[str]:
        if not hasattr(thread_local, "pipelines"):
            thread_local.pipelines = {}
        pipeline = thread_local.pipelines.get(language)
        if pipeline is None:
            pipeline = KPipeline(
                lang_code=language,
                repo_id=args.repo_id,
                model=False,
            )
            thread_local.pipelines[language] = pipeline
        phonemes = []
        for segment in split_text(text, args.chunk_chars):
            phonemes.extend(
                result.phonemes
                for result in pipeline(
                    segment,
                    voice=None,
                    split_pattern=None,
                    model=False,
                )
                if result.phonemes
            )
        return phonemes

    def prepare_cached(text: str, language: str) -> list[str]:
        key = (text, language)
        with prep_cache_lock:
            future = prep_cache.get(key)
            leader = future is None
            if leader:
                future = Future()
                prep_cache[key] = future
            else:
                prep_cache.move_to_end(key)
        if leader:
            try:
                future.set_result(prepare(text, language))
            except BaseException as exc:
                future.set_exception(exc)
                with prep_cache_lock:
                    prep_cache.pop(key, None)
                raise
            with prep_cache_lock:
                while len(prep_cache) > 1024:
                    oldest_key, oldest = next(iter(prep_cache.items()))
                    if not oldest.done():
                        break
                    prep_cache.pop(oldest_key)
        return future.result()

    prepare_cached(DEFAULT_WARMUP_TEXT, "a")

    request_id = [0]

    async def create_pcm_stream(body: dict) -> PcmStream:
        loop = asyncio.get_running_loop()
        request_started = time.perf_counter()
        text = str(body["text"]).strip()
        model_id = canonical_model(str(body.get("model", BASE_MODEL)))
        if model_id != BASE_MODEL:
            raise ValueError(f"unknown Kokoro model: {model_id}")
        options = parse_model_options(
            body, {"voice", "language", "speed"},
        )
        voice = str(options.get("voice", default_voice))
        if voice not in voice_packs and voice in OPENAI_BUILTIN_VOICES:
            mapped = KOKORO_OPENAI_VOICES.get(voice, default_voice)
            voice = mapped if mapped in voice_packs else default_voice
        if voice not in voice_packs:
            raise ValueError(f"unknown or unloaded Kokoro voice: {voice}")
        language = str(options.get("language", voice_language(voice)))
        if language != voice_language(voice):
            raise ValueError("Kokoro voice/language mismatch")
        speed = float(options.get("speed", 1.0))
        priority = 1 if body.get("priority") == "background" else 0
        if not 0.5 <= speed <= 2.0:
            raise ValueError("Kokoro speed must be in [0.5, 2.0]")
        if not text or len(text) > args.max_chars:
            raise ValueError(f"text length must be 1..{args.max_chars}")
        if scheduler.depth() >= args.max_queue:
            raise web.HTTPServiceUnavailable(text="Kokoro queue full")
        phoneme_chunks = await loop.run_in_executor(
            prep_pool, prepare_cached, text, language,
        )
        if not phoneme_chunks:
            raise ValueError("Kokoro phonemization produced no chunks")

        request_id[0] += 1
        queue: asyncio.Queue = asyncio.Queue()

        def sink(data):
            loop.call_soon_threadsafe(queue.put_nowait, data)

        job = KokoroRequest(
            rid=request_id[0],
            phonemes=deque(phoneme_chunks),
            token_ids=deque([
                [
                    value for value in map(model.vocab.get, phonemes)
                    if value is not None
                ]
                for phonemes in phoneme_chunks
            ]),
            voice=voice,
            speed=speed,
            sink=sink,
            priority=priority,
            started=request_started,
        )

        async def chunks():
            completed = False
            scheduler.submit(job)
            try:
                while True:
                    data = await asyncio.wait_for(queue.get(), timeout=300)
                    if data is None:
                        completed = True
                        break
                    yield data
            finally:
                if not completed:
                    job.cancelled = True

        return PcmStream(chunks(), {
            "Content-Type": "application/octet-stream",
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Model": BASE_MODEL,
            "X-Voice": voice,
            "X-Kokoro-Voice": voice,
            "X-Segment-Preroll-Ms": "0",
            "X-Request-Id": str(job.rid),
        })

    async def tts(request: web.Request) -> web.StreamResponse:
        return await internal_tts(request, create_pcm_stream)

    async def openai_speech_handler(request: web.Request) -> web.StreamResponse:
        return await openai_speech(
            request, create_pcm_stream, default_model=BASE_MODEL,
        )

    async def health(request):
        return web.json_response({
            "ok": True,
            "base_model": BASE_MODEL,
            "models": [BASE_MODEL],
            "voices": args.voices,
            "capabilities": capabilities,
            "sample_rate": SAMPLE_RATE,
            "batch_size": args.batch_size,
        })

    async def stats(request):
        return web.json_response(scheduler.stats())

    async def models_handler(request):
        return openai_models(capabilities, owned_by="hexgrad")
    app = web.Application(client_max_size=1024 * 1024)
    app.router.add_post("/tts", tts)
    app.router.add_post("/v1/audio/speech", openai_speech_handler)
    app.router.add_get("/v1/models", models_handler)
    app.router.add_get("/healthz", health)
    app.router.add_get("/stats", stats)
    logger.info(
        "serving batched Kokoro on %s:%d with %d voices B=%d",
        args.host, args.port, len(args.voices), args.batch_size,
    )
    web.run_app(
        app, host=args.host, port=args.port,
        access_log=None, backlog=args.max_queue,
    )


if __name__ == "__main__":
    import uvloop
    uvloop.install()
    main()
