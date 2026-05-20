"""English text normalization for Chatterbox-Flash inference.

This is a stripped-down port of Resemble's ``en_us_cleaner`` pipeline that
produces identical cleaned text but drops the ``SliceMap`` alignment plumbing
required for training. It expands numbers, currencies, units, time, phone
numbers and a small set of abbreviations so the input text matches the form
the model was trained on, which is essential for WER consistency on the
LibriSpeech-PC and Seed-TTS evaluation benchmarks.

The single entry point is :func:`en_us_cleaner`.
"""

from .cleaners import en_us_cleaner

__all__ = ["en_us_cleaner"]
