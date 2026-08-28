"""CPU-only contracts for the direct OpenAI speech adapter."""
from __future__ import annotations

import io

import av
import numpy as np

from server.http_api import _encode_pcm, translate_openai_request
from server.models import KOKORO_MODEL, TURBO_MODEL


def test_openai_request_translates_to_internal_contract() -> None:
    internal, response_format, stream_format = translate_openai_request({
        "model": KOKORO_MODEL,
        "input": "Hello.",
        "voice": "af_heart",
        "speed": 1.25,
        "response_format": "pcm",
    })
    assert internal == {
        "model": KOKORO_MODEL,
        "text": "Hello.",
        "model_options": {"voice": "af_heart", "speed": 1.25},
    }
    assert response_format == "pcm"
    assert stream_format == "audio"


def test_openai_model_and_voice_object_are_accepted() -> None:
    internal, response_format, _ = translate_openai_request(
        {
            "model": "tts-1",
            "input": "Hello.",
            "voice": {"id": "narrator"},
        },
        default_model=TURBO_MODEL,
    )
    assert internal["model"] == TURBO_MODEL
    assert internal["model_options"] == {"voice": "narrator"}
    assert response_format == "mp3"


def test_openai_validation_rejects_unsupported_controls() -> None:
    invalid = [
        {"model": TURBO_MODEL, "input": "Hello.", "voice": "base", "speed": 2},
        {"model": TURBO_MODEL, "input": "Hello.", "voice": "base", "instructions": "whisper"},
        {"model": TURBO_MODEL, "input": "Hello.", "voice": "base", "unknown": True},
        {"model": TURBO_MODEL, "input": "Hello."},
    ]
    for body in invalid:
        try:
            translate_openai_request(body)
        except (KeyError, ValueError):
            pass
        else:
            raise AssertionError(f"invalid OpenAI request accepted: {body}")


def test_every_openai_audio_container_decodes() -> None:
    time = np.arange(24000, dtype=np.float32) / 24000
    pcm = (np.sin(2 * np.pi * 440 * time) * 12000).astype("<i2").tobytes()
    expected_types = {
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
    }
    for response_format, expected_type in expected_types.items():
        encoded, content_type = _encode_pcm(pcm, response_format)
        assert content_type == expected_type
        assert encoded
        with av.open(io.BytesIO(encoded), mode="r") as container:
            decoded_samples = sum(
                frame.samples for frame in container.decode(audio=0)
            )
        assert decoded_samples > 0
