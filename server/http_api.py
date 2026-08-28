"""Shared HTTP surfaces for direct speech model workers."""
from __future__ import annotations

import asyncio
import base64
import io
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

import numpy as np
from aiohttp import web

from server.models import KOKORO_MODEL, canonical_model

OPENAI_TTS_MODELS = {
    "tts-1",
    "tts-1-hd",
    "gpt-4o-mini-tts",
    "gpt-4o-mini-tts-2025-12-15",
}
OPENAI_BUILTIN_VOICES = {
    "alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx",
    "sage", "shimmer", "verse", "marin", "cedar",
}


@dataclass
class PcmStream:
    chunks: AsyncIterator[bytes]
    headers: dict[str, str]


StreamFactory = Callable[[dict], Awaitable[PcmStream]]

_FORMATS = {
    "mp3": ("mp3", "libmp3lame", "audio/mpeg"),
    "opus": ("ogg", "libopus", "audio/ogg"),
    "aac": ("adts", "aac", "audio/aac"),
    "flac": ("flac", "flac", "audio/flac"),
    "wav": ("wav", "pcm_s16le", "audio/wav"),
}


def openai_error(
    message: str,
    *,
    status: int = 400,
    param: str | None = None,
    code: str | None = None,
    error_type: str = "invalid_request_error",
) -> web.Response:
    return web.json_response(
        {
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            },
        },
        status=status,
    )


def translate_openai_request(
    body: dict, *, default_model: str | None = None,
) -> tuple[dict, str, str]:
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    allowed = {
        "input", "model", "voice", "instructions", "response_format",
        "speed", "stream_format",
    }
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise ValueError(f"unknown parameter: {unknown[0]}")
    for required in ("input", "model", "voice"):
        if required not in body:
            raise KeyError(required)

    text = body["input"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("input must be a non-empty string")
    if len(text) > 4096:
        raise ValueError("input must contain at most 4096 characters")

    if not isinstance(body["model"], str) or not body["model"]:
        raise ValueError("model must be a non-empty string")
    model = canonical_model(str(body["model"]))
    if model in OPENAI_TTS_MODELS and default_model is not None:
        model = default_model
    voice_value = body["voice"]
    if isinstance(voice_value, dict):
        if set(voice_value) != {"id"} or not isinstance(voice_value["id"], str):
            raise ValueError("voice object must contain exactly one string id")
        voice = voice_value["id"]
    elif isinstance(voice_value, str) and voice_value:
        voice = voice_value
    else:
        raise ValueError("voice must be a non-empty string or an object with id")

    instructions = body.get("instructions")
    if instructions not in (None, ""):
        raise ValueError(f"instructions are not supported by model {model}")

    response_format = str(body.get("response_format", "mp3")).lower()
    if response_format not in {*_FORMATS, "pcm"}:
        raise ValueError(f"unsupported response_format: {response_format}")
    stream_format = str(body.get("stream_format", "audio")).lower()
    if stream_format not in {"audio", "sse"}:
        raise ValueError(f"unsupported stream_format: {stream_format}")

    options = {"voice": voice}
    if "speed" in body:
        speed = float(body["speed"])
        if not 0.25 <= speed <= 4.0:
            raise ValueError("speed must be between 0.25 and 4.0")
        if model == KOKORO_MODEL:
            if not 0.5 <= speed <= 2.0:
                raise ValueError("Kokoro speed must be between 0.5 and 2.0")
            options["speed"] = speed
        elif speed != 1.0:
            raise ValueError(f"speed is not supported by model {model}")

    return {
        "model": model,
        "text": text,
        "model_options": options,
    }, response_format, stream_format


async def _collect(chunks: AsyncIterator[bytes]) -> bytes:
    rows = []
    async for chunk in chunks:
        rows.append(chunk)
    return b"".join(rows)


def _encode_pcm(pcm: bytes, response_format: str) -> tuple[bytes, str]:
    import av

    container_format, codec, content_type = _FORMATS[response_format]
    output = io.BytesIO()
    with av.open(output, mode="w", format=container_format) as container:
        stream = container.add_stream(codec, rate=24000)
        stream.layout = "mono"
        samples = np.frombuffer(pcm, dtype="<i2")
        offset = 0
        frame_samples = 4096
        while offset < len(samples):
            row = np.ascontiguousarray(samples[offset:offset + frame_samples])
            frame = av.AudioFrame.from_ndarray(
                row.reshape(1, -1), format="s16", layout="mono",
            )
            frame.sample_rate = 24000
            for packet in stream.encode(frame):
                container.mux(packet)
            offset += len(row)
        for packet in stream.encode(None):
            container.mux(packet)
    return output.getvalue(), content_type


async def _formatted_chunks(
    audio: PcmStream, response_format: str,
) -> AsyncIterator[bytes]:
    if response_format == "pcm":
        async for chunk in audio.chunks:
            yield chunk
        return
    pcm = await _collect(audio.chunks)
    encoded, _ = await asyncio.to_thread(_encode_pcm, pcm, response_format)
    for offset in range(0, len(encoded), 16384):
        yield encoded[offset:offset + 16384]


def openai_models(capabilities: dict, *, owned_by: str) -> web.Response:
    return web.json_response({
        "object": "list",
        "data": [{
            "id": capabilities["model"],
            "object": "model",
            "created": 0,
            "owned_by": owned_by,
            "capabilities": capabilities,
        }],
    })


async def internal_tts(request: web.Request, factory: StreamFactory) -> web.StreamResponse:
    try:
        body = await request.json()
        audio = await factory(body)
    except web.HTTPException:
        raise
    except Exception as exc:
        return web.json_response({"error": str(exc) or "bad request"}, status=400)

    response = web.StreamResponse(headers=audio.headers)
    await response.prepare(request)
    try:
        async for chunk in audio.chunks:
            await response.write(chunk)
        await response.write_eof()
    except (ConnectionResetError, asyncio.CancelledError):
        await audio.chunks.aclose()
    return response


async def openai_speech(
    request: web.Request,
    factory: StreamFactory,
    *,
    default_model: str,
) -> web.StreamResponse:
    try:
        body = await request.json()
        internal, response_format, stream_format = translate_openai_request(
            body, default_model=default_model,
        )
        audio = await factory(internal)
    except KeyError as exc:
        param = str(exc.args[0])
        return openai_error(
            f"Missing required parameter: '{param}'.", param=param,
            code="missing_required_parameter",
        )
    except web.HTTPException as exc:
        return openai_error(
            exc.text or exc.reason,
            status=exc.status,
            code="service_unavailable" if exc.status == 503 else None,
            error_type=(
                "server_error" if exc.status >= 500 else "invalid_request_error"
            ),
        )
    except Exception as exc:
        return openai_error(str(exc) or "Invalid request")

    if stream_format == "sse":
        response = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Model": audio.headers.get("X-Model", internal["model"]),
            "X-Voice": audio.headers.get("X-Voice", ""),
        })
        await response.prepare(request)
        try:
            async for chunk in _formatted_chunks(audio, response_format):
                event = {
                    "type": "speech.audio.delta",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }
                await response.write(
                    f"event: speech.audio.delta\ndata: {json.dumps(event)}\n\n".encode()
                )
            await response.write(
                b'event: speech.audio.done\ndata: {"type":"speech.audio.done"}\n\n'
            )
            await response.write_eof()
        except (ConnectionResetError, asyncio.CancelledError):
            await audio.chunks.aclose()
        return response

    if response_format == "pcm":
        headers = {
            **audio.headers,
            "Content-Type": "application/octet-stream",
            "X-Audio-Format": "pcm_s16le",
        }
        response = web.StreamResponse(headers=headers)
        await response.prepare(request)
        try:
            async for chunk in audio.chunks:
                await response.write(chunk)
            await response.write_eof()
        except (ConnectionResetError, asyncio.CancelledError):
            await audio.chunks.aclose()
        return response

    try:
        pcm = await _collect(audio.chunks)
        encoded, content_type = await asyncio.to_thread(
            _encode_pcm, pcm, response_format,
        )
    except Exception as exc:
        return openai_error(
            str(exc) or "Audio encoding failed",
            status=500,
            code="audio_encoding_failed",
            error_type="server_error",
        )
    return web.Response(
        body=encoded,
        content_type=content_type,
        headers={
            "X-Model": audio.headers.get("X-Model", internal["model"]),
            "X-Voice": audio.headers.get("X-Voice", ""),
            "X-Audio-Format": response_format,
        },
    )
