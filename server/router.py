"""Weighted least-connections streaming router for Turbo GPU shards."""
from __future__ import annotations

import argparse
import gc
import asyncio
import logging
import json
from pathlib import Path

from server.models import TURBO_MODEL, canonical_model, model_owner
import aiohttp
from aiohttp import web

log = logging.getLogger("turbo.router")


class Router:
    def __init__(self, upstreams: list[str], capacities: list[int],
                 segment_chars: int = 512, segment_preroll_ms: float = 80.0,
                 segment_lookahead_s: float = 4.0,
                 default_model: str = TURBO_MODEL):
        if len(upstreams) != len(capacities):
            raise ValueError("one capacity is required per upstream")
        if any(capacity <= 0 for capacity in capacities):
            raise ValueError("capacities must be positive")
        self.upstreams = [url.rstrip("/") for url in upstreams]
        self.capacities = capacities
        self.segment_chars = segment_chars
        self.segment_preroll_bytes = round(segment_preroll_ms * 48)
        self.segment_lookahead_s = segment_lookahead_s
        self.default_model = canonical_model(default_model)
        self.active = [0] * len(upstreams)
        self.changed = asyncio.Condition()
        self.session: aiohttp.ClientSession | None = None
        self.model_sets: list[set[str]] = [set() for _ in upstreams]
        self.voice_sets: list[set[str]] = [set() for _ in upstreams]
        self.capability_sets: list[dict | None] = [None for _ in upstreams]
        self.stress_tasks: list[asyncio.Task] = []
        self.stress_started = 0.0
        self.stress_requests = 0
        self.stress_errors = 0
        self.stress_bytes = 0
        self.stress_model = self.default_model

    async def start(self, app: web.Application) -> None:
        connector = aiohttp.TCPConnector(limit=sum(self.capacities) + 64)
        timeout = aiohttp.ClientTimeout(total=600, sock_read=600)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        await self.refresh_models()
        log.info(
            "capabilities=%s",
            json.dumps(self.merged_capabilities(), sort_keys=True),
        )

    async def close(self, app: web.Application) -> None:
        await self.stop_stress()
        if self.session is not None:
            await self.session.close()

    async def refresh_models(self) -> None:
        assert self.session is not None
        async def fetch(url):
            try:
                async with self.session.get(f"{url}/healthz") as response:
                    if response.status != 200:
                        return set(), set(), None
                    payload = await response.json()
                    return (
                        set(payload.get("models", [])),
                        set(payload.get("voices", [])),
                        payload.get("capabilities"),
                    )
            except Exception:
                return set(), set(), None
        discovered = list(await asyncio.gather(
            *(fetch(url) for url in self.upstreams)
        ))
        self.model_sets = [item[0] for item in discovered]
        self.voice_sets = [item[1] for item in discovered]
        self.capability_sets = [item[2] for item in discovered]

    def merged_capabilities(self) -> dict[str, dict]:
        merged = {}
        for model in sorted(set().union(*self.model_sets)):
            shards = [
                index for index, models in enumerate(self.model_sets)
                if model in models
            ]
            source = next(
                (
                    self.capability_sets[index] for index in shards
                    if self.capability_sets[index] is not None
                ),
                None,
            )
            if source is None:
                continue
            capability = json.loads(json.dumps(source))
            voices = sorted(set().union(*(
                self.voice_sets[index] for index in shards
            )))
            capability["voices"] = voices
            voice_schema = (
                capability.get("request", {})
                .get("common", {})
                .get("model_options", {})
                .get("properties", {})
                .get("voice")
            )
            if voice_schema is not None:
                voice_schema["enum"] = voices
            merged[model] = capability
        return merged

    async def acquire(
        self,
        model: str | None = None,
        *,
        voice: str | None = None,
        reserve: int = 0,
    ) -> int:
        route_model = canonical_model(model or self.default_model)
        supported = [
            i for i, models in enumerate(self.model_sets)
            if route_model in models
            and (voice is None or voice in self.voice_sets[i])
            and self.capacities[i] > reserve
        ]
        if not supported:
            detail = f" model {route_model}"
            if voice is not None:
                detail += f" with voice {voice}"
            raise ValueError(f"no backend serves{detail}")
        async with self.changed:
            await self.changed.wait_for(
                lambda: any(
                    self.active[i] < self.capacities[i] - reserve
                    for i in supported
                )
            )
            choices = [
                i for i in supported
                if self.active[i] < self.capacities[i] - reserve
            ]
            shard = min(
                choices,
                key=lambda i: self.active[i] / self.capacities[i],
            )
            self.active[shard] += 1
            return shard

    async def release(self, shard: int) -> None:
        async with self.changed:
            self.active[shard] -= 1
            self.changed.notify_all()

    def split_text(self, text: str) -> list[str]:
        parts = []
        rest = " ".join(text.split())
        limit = self.segment_chars
        while len(rest) > limit:
            window = rest[:limit + 1]
            cut = -1
            for marker in (". ", "! ", "? ", "; ", ": ", ", "):
                at = window.rfind(marker)
                if at >= int(limit * 0.55):
                    cut = max(cut, at + 1)
            if cut < 0:
                cut = window.rfind(" ")
            if cut < 1:
                cut = limit
            parts.append(rest[:cut].strip())
            rest = rest[cut:].strip()
        if rest:
            parts.append(rest)
        return parts

    async def tts(self, request: web.Request) -> web.StreamResponse:
        try:
            payload = json.loads(await request.read())
            text = str(payload["text"]).strip()
            unknown_fields = sorted(
                set(payload) - {
                    "model", "text", "priority", "model_options", "segment",
                }
            )
            if unknown_fields:
                raise ValueError(
                    f"unsupported request fields: {', '.join(unknown_fields)}"
                )
            options = payload.get("model_options", {})
            if not isinstance(options, dict):
                raise ValueError("model_options must be an object")
            voice = options.get("voice")
            if voice is not None and not isinstance(voice, str):
                raise ValueError("model_options.voice must be a string")
        except Exception as exc:
            return web.json_response({"error": str(exc) or "bad request"}, status=400)
        segments = (self.split_text(text)
                    if payload.get("segment", True) else [text])
        if not segments:
            return web.json_response({"error": "empty text"}, status=400)

        requested_model = canonical_model(str(
            payload.get("model", self.default_model)
        ))
        if voice is None:
            voice_schema = (
                self.merged_capabilities().get(requested_model, {})
                .get("request", {})
                .get("common", {})
                .get("model_options", {})
                .get("properties", {})
                .get("voice", {})
            )
            voice = voice_schema.get("default")
            if voice is not None:
                options["voice"] = voice
                payload["model_options"] = options
        payload["model"] = requested_model
        try:
            shard = await self.acquire(requested_model, voice=voice)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=404)
        assert self.session is not None
        response = None
        started = asyncio.get_running_loop().time()
        audio_bytes = 0
        try:
            for index, segment in enumerate(segments):
                if index:
                    elapsed = asyncio.get_running_loop().time() - started
                    playback_elapsed = max(0.0, elapsed - 0.4)
                    lead = audio_bytes / 48000 - playback_elapsed
                    if lead > self.segment_lookahead_s:
                        await asyncio.sleep(lead - self.segment_lookahead_s)
                segment_payload = dict(payload, text=segment)
                segment_payload.pop("segment", None)
                async with self.session.post(
                        f"{self.upstreams[shard]}/tts",
                        json=segment_payload) as backend:
                    if backend.status != 200:
                        error = await backend.read()
                        if response is None:
                            return web.Response(status=backend.status, body=error,
                                                content_type=backend.content_type)
                        raise RuntimeError(
                            f"segment {index + 1} failed: HTTP {backend.status}")
                    if response is None:
                        response = web.StreamResponse(
                            status=200,
                            headers={
                                "Content-Type": backend.headers.get(
                                    "Content-Type", "application/octet-stream"),
                                "X-Sample-Rate": backend.headers.get(
                                    "X-Sample-Rate", "24000"),
                                "X-Backend-Shard": str(shard),
                                "X-Text-Segments": str(len(segments)),
                                "X-Model": backend.headers.get(
                                    "X-Model", str(payload.get("model", "base"))),
                                "X-Voice": backend.headers.get(
                                    "X-Voice", voice or "base"),
                            },
                        )
                        await response.prepare(request)
                    preroll_ms = float(backend.headers.get(
                        "X-Segment-Preroll-Ms",
                        self.segment_preroll_bytes / 48,
                    ))
                    skip = round(preroll_ms * 48) if index else 0
                    async for chunk in backend.content.iter_any():
                        if skip:
                            drop = min(skip, len(chunk))
                            chunk, skip = chunk[drop:], skip - drop
                        if chunk:
                            audio_bytes += len(chunk)
                            await response.write(chunk)
            assert response is not None
            await response.write_eof()
            return response
        except (ConnectionResetError, asyncio.CancelledError):
            raise
        except Exception:
            log.exception("backend shard %d failed", shard)
            if response is not None:
                try:
                    await response.write_eof()
                except Exception:
                    pass
                return response
            raise web.HTTPBadGateway()
        finally:
            await self.release(shard)

    async def _stress_worker(self, worker_id: int) -> None:
        assert self.session is not None
        text = (
            "This background request continuously exercises the streaming "
            "speech engine while interactive generation remains available."
        )
        while True:
            shard = await self.acquire(self.stress_model, reserve=1)
            try:
                async with self.session.post(
                        f"{self.upstreams[shard]}/tts",
                        json={
                            "text": text,
                            "model": self.stress_model,
                            "priority": "background",
                        }) as backend:
                    if backend.status != 200:
                        self.stress_errors += 1
                        await backend.read()
                    else:
                        async for chunk in backend.content.iter_any():
                            self.stress_bytes += len(chunk)
                        self.stress_requests += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                self.stress_errors += 1
                log.exception("stress worker %d failed", worker_id)
            finally:
                await self.release(shard)

    async def stop_stress(self) -> None:
        tasks, self.stress_tasks = self.stress_tasks, []
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stress_start(self, request: web.Request) -> web.Response:
        body = await request.json()
        concurrency = max(1, min(int(body.get("concurrency", 1)), 8192))
        model = canonical_model(str(body.get("model", self.default_model)))
        await self.refresh_models()
        if model not in set().union(*self.model_sets):
            return web.json_response(
                {"error": f"no backend serves model: {model}"}, status=400,
            )
        await self.stop_stress()
        self.stress_model = model
        self.stress_started = asyncio.get_running_loop().time()
        self.stress_requests = self.stress_errors = self.stress_bytes = 0
        self.stress_tasks = [
            asyncio.create_task(self._stress_worker(i))
            for i in range(concurrency)
        ]
        return await self.stress_status(request)

    async def stress_stop(self, request: web.Request) -> web.Response:
        await self.stop_stress()
        return await self.stress_status(request)

    async def stress_status(self, request: web.Request) -> web.Response:
        elapsed = (asyncio.get_running_loop().time() - self.stress_started
                   if self.stress_started else 0.0)
        audio_s = self.stress_bytes / 48000
        return web.json_response({
            "running": len(self.stress_tasks),
            "requests": self.stress_requests,
            "errors": self.stress_errors,
            "audio_xrt": round(audio_s / elapsed, 1) if elapsed else 0.0,
            "model": self.stress_model,
            "active": self.active,
            "capacity": self.capacities,
        })

    async def home(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(
            Path(__file__).with_name("web.html"),
            headers={"Cache-Control": "no-store"},
        )

    async def models(self, request: web.Request) -> web.Response:
        await self.refresh_models()
        capabilities = self.merged_capabilities()
        available = set().union(*self.model_sets)
        return web.json_response({
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "owned_by": model_owner(model),
                    "capabilities": capabilities.get(model),
                }
                for model in sorted(available)
            ],
        })

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "ok": True,
            "active": self.active,
            "capacity": self.capacities,
            "upstreams": self.upstreams,
            "models": sorted(set().union(*self.model_sets)),
            "capabilities": self.merged_capabilities(),
        })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8019)
    parser.add_argument("--upstream", nargs="+", required=True)
    parser.add_argument("--capacity", nargs="+", type=int, required=True)
    parser.add_argument("--segment-chars", type=int, default=512)
    parser.add_argument("--default-model", default=TURBO_MODEL)
    parser.add_argument("--segment-preroll-ms", type=float, default=80.0)
    parser.add_argument("--segment-lookahead", type=float, default=4.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    router = Router(
        args.upstream,
        args.capacity,
        segment_chars=args.segment_chars,
        segment_preroll_ms=args.segment_preroll_ms,
        segment_lookahead_s=args.segment_lookahead,
        default_model=args.default_model,
    )
    app = web.Application(client_max_size=1024 * 1024)
    app.router.add_get("/", router.home)
    app.router.add_get("/v1/models", router.models)
    app.router.add_post("/tts", router.tts)
    app.router.add_get("/healthz", router.health)
    app.router.add_post("/stress/start", router.stress_start)
    app.router.add_post("/stress/stop", router.stress_stop)
    app.router.add_get("/stress", router.stress_status)
    app.on_startup.append(router.start)
    app.on_cleanup.append(router.close)
    gc.collect()
    gc.freeze()
    gc.set_threshold(100_000, 100, 100)
    web.run_app(
        app, host=args.host, port=args.port,
        access_log=None, backlog=8192,
    )


if __name__ == "__main__":
    import uvloop
    uvloop.install()
    main()
