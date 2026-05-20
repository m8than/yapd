"""Backwards-compatibility shim — :class:`FlashInferEngine` now lives in
:mod:`chatterbox_flash.engines.flashinfer`.

The original module path is preserved so existing scripts that did

    from chatterbox_flash.attention_flashinfer import FlashInferEngine

continue to work.  New code should import from
``chatterbox_flash.engines`` (or call :func:`chatterbox_flash.engines.build_engine`)
instead.
"""

from __future__ import annotations

from .engines.flashinfer import (  # noqa: F401
    FlashInferEngine,
    flashinfer_available,
)

__all__ = ["FlashInferEngine", "flashinfer_available"]
