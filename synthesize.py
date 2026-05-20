#!/usr/bin/env python3
"""Single entry-point synthesis CLI for Chatterbox-Flash.

Minimal driver for synthesizing one or more sentences that share a single
reference voice. One ``--backend`` value picks the hardware path; everything
else (device, dtype, CUDA graph) is wired automatically. Inference-only.

    ``--backend gpu``         torch SDPA on cuda, bf16, CUDA graph on (default)
    ``--backend flashinfer``  FlashInfer paged-KV + CUDA graph on cuda, bf16
    ``--backend cpu``         torch SDPA on cpu, fp32
    ``--backend mlx``         Apple-Silicon native Metal, fp16

One reference wav (``--audio_prompt``) is cloned, then every text passed via
``--text`` and/or ``--text_file`` is synthesized in batched forwards of
``--batch_size`` rows (a single text is just a batch of one) and written to
``--output_dir/{stem}_{i:03d}.wav``. Per-batch and aggregate generation-time /
RTF are printed.

Examples
--------
    # Default — --backend gpu, one or more sentences:
    python synthesize.py \\
        --audio_prompt reference.wav \\
        --text "I don't spend time in traffic anymore and I love it." \\
               "Nah, don't stress about it too much, you know?"

    # FlashInfer:    python synthesize.py ... --backend flashinfer
    # CPU only:      python synthesize.py ... --backend cpu
    # Apple Silicon: python synthesize.py ... --backend mlx

    # One sentence per line from a file, 8 rows per batched forward:
    python synthesize.py --audio_prompt ref.wav --text_file sentences.txt \\
        --batch_size 8
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import soundfile as sf
import torch

from chatterbox_flash import ChatterboxFlashTTS

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_HF_REPO = "ResembleAI/chatterbox-flash"

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

# Backend → (internal engine name, torch device, dtype, cuda_graph) tuple.
# The user-facing ``--backend`` value picks one of these rows; everything
# else (device, dtype, graph capture) follows automatically so the four
# hardware paths stay cleanly separated.
_BACKENDS: dict[str, dict] = {
    "gpu":        {"engine": "torch",      "device": "cuda", "dtype": "bf16", "cuda_graph": True},
    "flashinfer": {"engine": "flashinfer", "device": "cuda", "dtype": "bf16", "cuda_graph": True},
    "cpu":        {"engine": "torch",      "device": "cpu",  "dtype": "fp16", "cuda_graph": False},
    "mlx":        {"engine": "mlx",        "device": "cpu",  "dtype": "fp16", "cuda_graph": False},
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--audio_prompt", type=Path, required=True,
        help="Reference wav used as the shared zero-shot voice for every text.",
    )
    p.add_argument(
        "--text", type=str, nargs="+", default=None,
        help="One or more texts to synthesize (space-separated, each quoted).",
    )
    p.add_argument(
        "--text_file", type=Path, default=None,
        help="File with one sentence per line ('#' lines and blanks ignored). "
        "Combined with any --text entries.",
    )
    p.add_argument(
        "--output_dir", type=Path, default=REPO_ROOT / "test_out",
        help="Directory for the generated wavs. Default: ./test_out",
    )
    p.add_argument(
        "--hf_repo", type=str, default=DEFAULT_HF_REPO,
        help=f"HuggingFace Hub repo id. Default: {DEFAULT_HF_REPO}. "
        "Private repos require $HF_TOKEN or a cached `huggingface-cli login`.",
    )

    p.add_argument(
        "--backend", type=str, default="gpu", choices=list(_BACKENDS),
        help="Hardware path (default 'gpu'). 'gpu' = torch SDPA on cuda + "
        "bf16; 'flashinfer' = FlashInfer paged-KV + CUDA-graph on cuda + "
        "bf16; 'cpu' = torch SDPA on cpu + fp16; 'mlx' = Apple-Silicon native "
        "Metal + fp16.",
    )
    p.add_argument(
        "--dtype", type=str, default=None, choices=["bf16", "fp16", "fp32"],
        help="Override the backend's default compute dtype. Default (None) "
        "uses the per-backend default (gpu/flashinfer bf16, cpu/mlx fp16). "
        "Use fp32 on cpu if a fp16 op is unsupported or slower on your CPU.",
    )
    p.add_argument(
        "--quantize_bits", type=int, default=None, choices=[4, 8],
        help="Quantize the T3 LLaMA backbone to 4- or 8-bit. MLX backend "
        "only (uses mlx.nn.quantize); the S3Gen vocoder and voice encoder "
        "stay in --dtype. Ignored with a warning on other backends.",
    )
    p.add_argument(
        "--batch_size", type=int, default=8,
        help="Rows per batched forward. Texts are split into groups of this "
        "size; a single text is just a batch of one. Default: 8.",
    )

    # Decoding knobs — defaults reproduce the paper's best configuration.
    p.add_argument(
        "--block_size", type=int, default=16,
        help="Block-diffusion block size (speech tokens decoded per block). "
        "Default 16 matches the paper / released checkpoint.",
    )
    p.add_argument("--num_steps", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--time_shift_tau", type=float, default=0.5)
    p.add_argument("--omnivoice_schedule_t_shift", type=float, default=0.5)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--position_temperature", type=float, default=5.0)
    p.add_argument(
        "--n_cfm_timesteps", type=int, default=2,
        help="S3Gen CFM steps for vocoding. The released checkpoint is "
        "meanflow-distilled (S3Gen instantiated with meanflow=True in "
        "tts.py), so the canonical setting is 2 — matches the internal "
        "S3GenStreamer defaults.",
    )
    p.add_argument(
        "--max_speech_tokens", type=int, default=None,
        help="Cap the per-utterance speech-token budget. Default (None) uses "
        "max(text_tokens*6, 300). Lower this to bound trailing-silence "
        "hallucinations when EOS is not sampled in time (eg. CPU/fp32 paths "
        "with default temperature).",
    )
    p.add_argument(
        "--no_text_norm", action="store_true",
        help="Skip English text normalization (number/abbreviation expansion).",
    )
    p.add_argument(
        "--no_cuda_graph", action="store_true",
        help="Disable CUDA-graph capture of the per-block forward.",
    )
    p.add_argument(
        "--disable_warmup", action="store_true",
        help="Skip the throwaway warmup generate. By default every backend "
        "runs one warmup pass so the timed loop measures steady-state "
        "throughput without cold-start costs (kernel JIT, MKL/Metal planning, "
        "first S3Gen invocation) folded into the first batch.",
    )
    return p


def _collect_texts(args: argparse.Namespace) -> list[str]:
    texts: list[str] = list(args.text or [])
    if args.text_file is not None:
        if not args.text_file.is_file():
            raise SystemExit(f"--text_file not found: {args.text_file}")
        texts += [
            line.strip()
            for line in args.text_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    if not texts:
        raise SystemExit("No texts given — pass --text and/or --text_file.")
    return texts


def main() -> None:
    args = _build_parser().parse_args()
    texts = _collect_texts(args)

    if not args.audio_prompt.is_file():
        raise SystemExit(f"Reference audio not found: {args.audio_prompt}")

    cfg = _BACKENDS[args.backend]
    engine = cfg["engine"]
    device = cfg["device"]
    dtype_str = args.dtype or cfg["dtype"]
    dtype = _DTYPE_MAP[dtype_str]

    # 4-/8-bit quantization is wired through the engine via an env var so the
    # tts/model call signatures stay untouched (same convention as
    # CHATTERBOX_FLASH_ENGINE). Only the MLX engine reads it.
    if args.quantize_bits is not None:
        if engine == "mlx":
            os.environ["CHATTERBOX_FLASH_MLX_QUANT_BITS"] = str(args.quantize_bits)
        else:
            print(
                f"[batch] --quantize_bits is MLX-only; ignored for "
                f"backend={args.backend}.",
            )
            os.environ.pop("CHATTERBOX_FLASH_MLX_QUANT_BITS", None)
    else:
        os.environ.pop("CHATTERBOX_FLASH_MLX_QUANT_BITS", None)
    use_cuda_graph = (
        cfg["cuda_graph"]
        and not args.no_cuda_graph
        and device.startswith("cuda")
    )
    batch_size = max(1, args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.audio_prompt.stem

    print(
        f"[batch] backend={args.backend} (engine={engine} device={device} "
        f"dtype={dtype_str} cuda_graph={use_cuda_graph}) "
        f"batch_size={batch_size} block_size={args.block_size}"
        + (f" quant={args.quantize_bits}bit" if (
            args.quantize_bits is not None and engine == "mlx") else "")
        + f" | {len(texts)} text(s)",
    )

    t0 = time.time()
    print(f"[batch] loading from HF Hub: {args.hf_repo}")
    tts = ChatterboxFlashTTS.from_pretrained(
        args.hf_repo, device=device, dtype=dtype,
        drf_block_size=args.block_size,
    )
    print(f"[batch] loaded checkpoint in {time.time() - t0:.2f}s")

    t0 = time.time()
    conds = tts.prepare_conditionals(str(args.audio_prompt))
    print(f"[batch] conditioned on {args.audio_prompt.name} in {time.time() - t0:.2f}s")

    gen_kwargs = dict(
        normalize_text=not args.no_text_norm,
        num_steps=args.num_steps,
        temperature=args.temperature,
        time_shift_tau=args.time_shift_tau,
        omnivoice_schedule_t_shift=args.omnivoice_schedule_t_shift,
        cfg_scale=args.cfg_scale,
        position_temperature=args.position_temperature,
        use_cuda_graph=use_cuda_graph,
        backend=engine,
        n_cfm_timesteps=args.n_cfm_timesteps,
        max_speech_tokens=args.max_speech_tokens,
    )

    def _sync() -> None:
        # CUDA kernels launch asynchronously, so we must drain the queue at
        # every timing boundary or the RTF numbers would measure launch time,
        # not execution. No-op on cpu / mlx (mlx times itself via mx.eval).
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    # Warmup: one throwaway generate to fold every backend's cold-start cost
    # out of the timed loop — FlashInfer JIT + per-block CUDA-graph capture,
    # CUDA/cuDNN kernel autotuning, CPU MKL planning, mlx Metal kernel
    # compilation, and the first S3Gen vocoder invocation. The engine is
    # cached on the model, so the real loop reuses the warmed state. Shape it
    # like the real batch (batch_size rows) so any size-specialized kernels /
    # graphs are primed for the steady-state path. Skip with --disable_warmup.
    if not args.disable_warmup:
        _sync()
        t_warm = time.time()
        warm_chunk = texts[: min(batch_size, len(texts))]
        _ = tts.generate_batch(
            warm_chunk, conds_list=[conds] * len(warm_chunk), **gen_kwargs,
        )
        _sync()
        print(f"[batch] {engine} warmup in {time.time() - t_warm:.2f}s")

    # Always go through the batched path; with FlashInfer the cached engine is
    # already warm, so per-batch timing measures steady-state throughput. The
    # same reference voice is shared across every row.
    wavs: list[torch.Tensor] = []
    _sync()
    t_all = time.time()
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        # The previous iteration's post-generate _sync() (or the pre-loop one)
        # already drained the queue, so the GPU is idle here — t0 is accurate.
        t0 = time.time()
        chunk_wavs = tts.generate_batch(
            chunk, conds_list=[conds] * len(chunk), **gen_kwargs,
        )
        _sync()
        dt = time.time() - t0
        dur = sum(w.shape[-1] for w in chunk_wavs) / tts.sr
        print(
            f"[batch] rows {start}-{start + len(chunk) - 1} ({len(chunk)}) "
            f"in {dt:.2f}s (audio {dur:.2f}s, RTF={dt / max(dur, 1e-6):.3f})",
        )
        wavs.extend(chunk_wavs)
    # Last iteration's post-generate _sync() already drained the queue.
    total_dt = time.time() - t_all
    total_dur = sum(w.shape[-1] for w in wavs) / tts.sr
    print(
        f"[batch] total: {len(wavs)} text(s) in {total_dt:.2f}s "
        f"(audio {total_dur:.2f}s, RTF={total_dt / max(total_dur, 1e-6):.3f})",
    )

    for i, wav in enumerate(wavs):
        out_path = args.output_dir / f"{stem}_{i:03d}.wav"
        sf.write(str(out_path), wav.detach().float().cpu().numpy().squeeze(), tts.sr)
        print(f"[batch] wrote {out_path}")


if __name__ == "__main__":
    main()
