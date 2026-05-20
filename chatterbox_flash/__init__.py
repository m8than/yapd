"""Chatterbox-Flash — prior-calibrated block-diffusion zero-shot TTS.

Public API:
    ChatterboxFlashTTS     — end-to-end TTS pipeline (T3 + S3Gen + VE + tokenizer)
    ChatterboxFlashT3      — block-diffusion T3 decoder (low-level)
    Conditionals           — cacheable speaker / prompt conditioning bundle
    en_us_cleaner          — English text normalization used at training / eval time
"""

from .engines import FLASHINFER_AVAILABLE, MLX_AVAILABLE, build_engine
from .model import ChatterboxFlashT3
from .text_norm import en_us_cleaner
from .tts import ChatterboxFlashTTS, Conditionals

__version__ = "0.1.0"

__all__ = [
    "ChatterboxFlashT3",
    "ChatterboxFlashTTS",
    "Conditionals",
    "FLASHINFER_AVAILABLE",
    "MLX_AVAILABLE",
    "build_engine",
    "en_us_cleaner",
    "__version__",
]
