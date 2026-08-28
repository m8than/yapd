"""Canonical model identifiers shared by workers and the fleet router."""
from __future__ import annotations

KOKORO_MODEL = "hexgrad/Kokoro-82M"
TURBO_MODEL = "ResembleAI/chatterbox-turbo"
FLASH_MODEL = "ResembleAI/chatterbox-flash"

MODEL_ALIASES = {
    "kokoro": KOKORO_MODEL,
    "turbo": TURBO_MODEL,
    "chatterbox-turbo": TURBO_MODEL,
    "flash": FLASH_MODEL,
    "chatterbox-flash": FLASH_MODEL,
}


def canonical_model(model: str) -> str:
    """Resolve a built-in short name without rewriting voices or LoRAs."""
    return MODEL_ALIASES.get(model, model)


def model_owner(model: str) -> str:
    return "kokoro" if model == KOKORO_MODEL or model.startswith("kokoro/") else "chatterbox"


def parse_model_options(body: dict, allowed: set[str]) -> dict:
    misplaced = sorted(set(body) & allowed)
    if misplaced:
        raise ValueError(
            f"{', '.join(misplaced)} must be inside model_options"
        )
    unknown_fields = sorted(
        set(body) - {"model", "text", "priority", "model_options"}
        - allowed
    )
    if unknown_fields:
        raise ValueError(
            f"unsupported request fields: {', '.join(unknown_fields)}"
        )
    options = body.get("model_options", {})
    if not isinstance(options, dict):
        raise ValueError("model_options must be an object")
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(
            f"unsupported model_options: {', '.join(unknown)}"
        )
    return options


def worker_capabilities(
    model: str,
    *,
    voices: list[str],
    max_input: int,
    input_unit: str,
    model_options: dict,
) -> dict:
    return {
        "model": model,
        "request": {
            "required": ["text"],
            "common": {
                "model": {"type": "string", "default": model},
                "text": {
                    "type": "string",
                    "max_length": max_input,
                    "length_unit": input_unit,
                },
                "priority": {
                    "type": "string",
                    "enum": ["interactive", "background"],
                    "default": "interactive",
                },
                "model_options": {
                    "type": "object",
                    "additional_properties": False,
                    "properties": model_options,
                },
            },
        },
        "voices": voices,
        "output": {
            "streaming": True,
            "format": "pcm_s16le",
            "sample_rate": 24000,
            "channels": 1,
        },
    }
