"""Inference engines for Chatterbox-Flash.

Three backends are provided, all implementing the :class:`InferenceEngine`
protocol consumed by :class:`~chatterbox_flash.model.ChatterboxFlashT3`:

* :class:`FlashInferEngine` — paged KV cache + custom attention kernels
  via the ``flashinfer-python`` package; supports CUDA-graph capture of
  the block forward.  This is what the paper numbers were produced with
  and what should be used for any throughput-sensitive workload on
  CUDA.

* :class:`TorchSDPAEngine` — pure PyTorch fallback using the HuggingFace
  transformers SDPA backend with a :class:`DynamicCache`.  No extra
  dependencies, runs on any CUDA / CPU / MPS device the underlying T3
  supports.  No CUDA graphs, slower per step, but produces numerically
  matching output (modulo small fp32/bf16 reduction order differences).

* :class:`MLXEngine` *(experimental)* — Apple-Silicon native Metal
  backend.  Wraps the mlx-lm LLaMA backbone and bypasses its
  ``embed_tokens`` layer so the chatterbox-flash custom embeddings can
  be used unchanged.  Requires ``mlx`` and ``mlx-lm`` installed.

Use :func:`build_engine` to instantiate; it auto-falls-back to SDPA when
FlashInfer is unavailable.  MLX must be selected explicitly via
``backend="mlx"`` (or the ``CHATTERBOX_FLASH_ENGINE=mlx`` env var).
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Literal

import torch

from .base import InferenceEngine
from .torch_sdpa import TorchSDPAEngine

logger = logging.getLogger(__name__)


FLASHINFER_AVAILABLE: bool = importlib.util.find_spec("flashinfer") is not None
MLX_AVAILABLE: bool = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_lm") is not None
)


_Backend = Literal["auto", "flashinfer", "torch", "mlx"]


def _resolve_backend(requested: _Backend) -> Literal["flashinfer", "torch", "mlx"]:
    """Pick a backend honoring ``CHATTERBOX_FLASH_ENGINE`` env override."""
    env = os.environ.get("CHATTERBOX_FLASH_ENGINE", "").strip().lower()
    if env in ("flashinfer", "torch", "sdpa", "mlx"):
        requested = "torch" if env == "sdpa" else env  # type: ignore[assignment]

    if requested == "auto":
        return "flashinfer" if FLASHINFER_AVAILABLE else "torch"
    if requested == "flashinfer":
        if not FLASHINFER_AVAILABLE:
            raise ImportError(
                "flashinfer-python is not installed. Install with "
                "`pip install flashinfer-python` or pass backend='torch'.",
            )
        return "flashinfer"
    if requested == "mlx":
        if not MLX_AVAILABLE:
            raise ImportError(
                "MLX backend requires the 'mlx' and 'mlx-lm' packages. "
                "Install with `pip install mlx mlx-lm` "
                "(Apple Silicon Mac only).",
            )
        return "mlx"
    if requested in ("torch", "sdpa"):
        return "torch"
    raise ValueError(
        f"backend must be one of 'auto', 'flashinfer', 'torch', 'mlx'; "
        f"got {requested!r}",
    )


def build_engine(
    model: torch.nn.Module,
    max_seq_len: int,
    dtype: torch.dtype,
    *,
    backend: _Backend = "auto",
    batch_size: int = 1,
    page_size: int = 16,
) -> InferenceEngine:
    """Construct an inference engine.

    Parameters
    ----------
    model: the :class:`ChatterboxFlashT3` instance.
    max_seq_len: upper bound on ``prefix_len + total_speech_len`` for the
        cache reservation; needed by the FlashInfer paged buffer.  The
        torch SDPA / MLX engines use this only as a hint.
    dtype: model parameter dtype (bf16 / fp16 / fp32).
    backend: ``"auto"`` (default, prefer FlashInfer) / ``"flashinfer"`` /
        ``"torch"`` / ``"mlx"``.  Environment variable
        ``CHATTERBOX_FLASH_ENGINE`` overrides this.
    batch_size: number of forward-batch rows (with zero-text-batch CFG
        this is ``2 × B_usr``).
    page_size: FlashInfer paged-cache page size (ignored elsewhere).
    """
    chosen = _resolve_backend(backend)
    if chosen == "flashinfer":
        from .flashinfer import FlashInferEngine
        logger.debug(
            "build_engine: backend=flashinfer max_seq=%d dtype=%s batch=%d page=%d",
            max_seq_len, dtype, batch_size, page_size,
        )
        return FlashInferEngine(
            model, max_seq_len, dtype,
            batch_size=batch_size, page_size=page_size,
        )
    if chosen == "mlx":
        from .mlx import MLXEngine
        logger.debug(
            "build_engine: backend=mlx max_seq=%d dtype=%s batch=%d",
            max_seq_len, dtype, batch_size,
        )
        return MLXEngine(
            model, max_seq_len, dtype, batch_size=batch_size,
        )
    logger.debug(
        "build_engine: backend=torch (SDPA fallback) max_seq=%d dtype=%s batch=%d",
        max_seq_len, dtype, batch_size,
    )
    return TorchSDPAEngine(
        model, max_seq_len, dtype, batch_size=batch_size,
    )


def __getattr__(name: str):  # noqa: D401
    """Lazy attribute access — only import optional backends if asked for."""
    if name == "FlashInferEngine":
        from .flashinfer import FlashInferEngine
        return FlashInferEngine
    if name == "MLXEngine":
        from .mlx import MLXEngine
        return MLXEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "InferenceEngine",
    "TorchSDPAEngine",
    "FlashInferEngine",
    "MLXEngine",
    "FLASHINFER_AVAILABLE",
    "MLX_AVAILABLE",
    "build_engine",
]
