"""Micro-batched chunked-streaming S3Gen vocoder worker.

Turns committed speech tokens into 24 kHz PCM incrementally:

  * per request, tokens are vocoded in windows [emit_start - lookback, emit_end)
    so each chunk sees left context; only [emit_start, emit_end) is emitted
  * chunk joins are blended with a 20 ms linear crossfade (the last FADE
    samples of every chunk are withheld and blended into the next one)
  * jobs from all requests are grouped into padded micro-batches; the flow
    encoder/CFM/HiFT all run batched (ragged lengths via token_len)

Runs on its own thread + HIP stream so vocoding overlaps T3 decode ticks.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)

def _silence_cfm_progress() -> None:
    """The stock CFM euler loop prints + tqdm-wraps every call; at server
    scale that is pure stdout overhead. Shadow them at module level."""
    import chatterbox.models.s3gen.flow_matching as fm
    fm.tqdm = lambda it, **kw: it
    fm.print = lambda *a, **k: None


SR = 24000
SAMPLES_PER_TOKEN = 960          # 25 tokens/s -> 2 mel frames -> 2*480 samples
MEL_PER_TOKEN = 2


@dataclass
class VocJob:
    req: object
    window: list                  # token ids (window = lookback + emit)
    emit_off: int                 # tokens inside window before emit point
    first: bool                   # first chunk of the request
    final: bool                   # last chunk -> close stream after
    end_marker: bool = False      # no audio; just close the stream
    deadline: float = float("inf")  # playback depletion deadline


class Vocoder:
    B_BUCKETS = (4, 8, 16, 24, 32, 48, 64, 96)
    W_BUCKETS = (4, 8, 12, 13, 15, 16, 20, 21, 24, 28, 32, 36, 40, 44, 52, 60, 64, 68, 76, 80, 84, 96)

    def __init__(
        self,
        s3gen,
        ref_dict: dict,
        device,
        *,
        chunk_first: int = 12,
        chunk: int = 80,
        chunk_second: int = 0,
        topup_sizes: tuple[int, ...] = (),
        lookback: int = 16,
        topup_until: int = 0,
        fade: int = 480,
        max_batch: int = 48,
        pre_roll: int = 0,
        first_max_batch: int = 24,
        n_cfm_timesteps: int = 1,
    ):
        _silence_cfm_progress()
        self.s3gen = s3gen
        self.device = device
        self.dtype = s3gen.dtype
        self.chunk_first = chunk_first
        self.chunk_second = chunk_second
        self.topup_sizes = tuple(topup_sizes)
        self.chunk = chunk
        self.topup_until = topup_until or (
            chunk_first + (sum(self.topup_sizes) if self.topup_sizes else chunk_second)
        )
        self.lookback = lookback
        self.fade = fade
        self.max_batch = max_batch
        self.pre_roll = pre_roll
        self.first_max_batch = first_max_batch
        self.n_cfm = n_cfm_timesteps

        # pre-cast ref dict once (shared voice)
        self.ref = {}
        for k, v in ref_dict.items():
            if torch.is_tensor(v):
                v = v.to(device=device)
                if v.dtype.is_floating_point:
                    v = v.to(self.dtype)
            self.ref[k] = v
        self.prompt_token = self.ref["prompt_token"]           # (1, P)
        self.prompt_token_len = self.ref["prompt_token_len"]
        self.prompt_feat = self.ref["prompt_feat"]             # (1, M1, 80)
        self.embedding = self.ref["embedding"]

        self.trim_fade = s3gen.trim_fade.detach().float().cpu().numpy()
        self.fade_in = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        self.fade_out = 1.0 - self.fade_in

        self.q: deque[VocJob] = deque()           # bulk lane
        self.q_topup: dict[int, deque[VocJob]] = defaultdict(deque)
        self.q_first: deque[VocJob] = deque()     # first-chunk highest priority
        self.topup_last_width = -1
        self.first_streak = 0
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.wake_first = threading.Event()
        self.stop_flag = False

        # stats
        self.batches = 0
        self.jobs = 0
        self.gpu_time = 0.0

    # ------------- producer side (engine thread) -------------------------- #

    def enqueue(self, req, final: bool, n_emit: int | None = None) -> None:
        e_start = req.emitted_tok
        e_end = len(req.committed) if n_emit is None else e_start + n_emit
        first = e_start == 0
        if self.topup_sizes:
            priority = first or e_start == self.chunk_first
        else:
            priority = first or (
                self.chunk_second > 0 and e_start < self.topup_until
            )
        emit_width = n_emit if priority and not first and n_emit else self.chunk
        w_start = 0 if first else max(
            0, e_end - (self.lookback + emit_width))
        w_start = min(w_start, e_start)
        base = req.t_recv or time.perf_counter()
        deadline = base + 0.4 + e_start / 25.0
        job = VocJob(
            req=req,
            window=req.committed[w_start:e_end],
            emit_off=e_start - w_start,
            first=first,
            final=final,
            deadline=deadline,
        )
        with self.lock:
            if first:
                self.q_first.append(job)
            elif priority:
                self.q_topup[len(job.window)].append(job)
            else:
                self.q.append(job)
        (self.wake_first if priority else self.wake).set()

    def finalize_empty(self, req) -> None:
        with self.lock:
            self.q.append(VocJob(req, [], 0, False, True, end_marker=True))
        self.wake.set()

    # ------------- consumer side (vocoder thread) ------------------------- #

    def run(self) -> None:
        """Two lanes on separate CUDA streams: first-chunk batches overlap
        in-flight bulk batches instead of queueing behind them (~200-400ms)."""
        t = threading.Thread(target=self._run_lane, args=(True,), daemon=True)
        t.start()
        self._run_lane(False)

    def _run_lane(self, priority: bool) -> None:
        stream = torch.cuda.Stream(priority=-1 if priority else 0)
        wake = self.wake_first if priority else self.wake
        take = self._take_first if priority else self._take_bulk
        with torch.cuda.stream(stream):
            while not self.stop_flag:
                jobs = take()
                if not jobs:
                    wake.wait(timeout=0.005)
                    wake.clear()
                    continue
                try:
                    self._process(jobs)
                except Exception:
                    logger.exception("vocoder batch failed; closing %d streams", len(jobs))
                    for j in jobs:
                        self._emit(j.req, None)

    def _take_first(self) -> list[VocJob]:
        # First chunks dominate, with bounded service for homogeneous top-ups.
        with self.lock:
            widths = sorted(w for w in self.q_topup if self.q_topup[w])
            width = next((w for w in widths if w > self.topup_last_width),
                         widths[0] if widths else None)
            use_first = bool(self.q_first) and (
                not widths or self.first_streak < 3)
            src = self.q_first if use_first else (
                self.q_topup[width] if width is not None else self.q_first)
            n = len(src)
        if 0 < n < 4:
            time.sleep(0.006)
        with self.lock:
            widths = sorted(w for w in self.q_topup if self.q_topup[w])
            width = next((w for w in widths if w > self.topup_last_width),
                         widths[0] if widths else None)
            use_first = bool(self.q_first) and (
                not widths or self.first_streak < 3)
            src = self.q_first if use_first else (
                self.q_topup[width] if width is not None else self.q_first)
            n = min(len(src), self.first_max_batch)
            jobs = [src.popleft() for _ in range(n)]
            if jobs:
                if use_first:
                    self.first_streak += 1
                else:
                    self.first_streak = 0
                    if width is not None:
                        self.topup_last_width = width
            return jobs

    def _take_bulk(self) -> list[VocJob]:
        with self.lock:
            n_avail = len(self.q)
        if 0 < n_avail < self.max_batch // 2:
            # brief coalescing window so flow/hift run at useful batch sizes
            time.sleep(0.004)
        with self.lock:
            out: list[VocJob] = []
            skipped: list[VocJob] = []
            while self.q and len(out) < self.max_batch:
                j = self.q.popleft()
                if (not j.end_marker and len(j.window) > 0
                        and j.req.audio_samples == 0):
                    # its first chunk is still in the priority lane; chunk 2
                    # must not overtake chunk 1 (crossfade tail dependency)
                    skipped.append(j)
                else:
                    out.append(j)
            self.q.extendleft(reversed(skipped))
            return out

    @torch.inference_mode()
    def _process(self, jobs: list[VocJob]) -> None:
        t0 = time.perf_counter()
        real = [j for j in jobs if not j.end_marker and len(j.window) > 0]
        for j in jobs:
            if j.end_marker or len(j.window) == 0:
                self._finish(j.req)

        if not real:
            return
        B = len(real)
        lens = [len(j.window) for j in real]
        # Bucket width first. A synthetic full-width row is needed only when
        # T is wider than every real row; exact-width batches already carry
        # token_len.max()==T and must not be inflated (64 -> 96 was 50% waste).
        T = next((t for t in self.W_BUCKETS if t >= max(lens)), max(lens))
        needs_width_row = T > max(lens)
        min_batch = B + int(needs_width_row)
        Bp = next((b for b in self.B_BUCKETS if b >= min_batch), min_batch)
        tok = torch.zeros(Bp, T, dtype=torch.long, device=self.device)
        for i, j in enumerate(real):
            tok[i, : lens[i]] = torch.tensor(j.window, dtype=torch.long, device=self.device)
        pad_lens = lens + [T] * (Bp - B)
        tok_len = torch.tensor(pad_lens, dtype=torch.long, device=self.device)

        mels, _ = self.s3gen.flow.inference(
            token=tok,
            token_len=tok_len,
            prompt_token=self.prompt_token,
            prompt_token_len=self.prompt_token_len,
            prompt_feat=self.prompt_feat,
            prompt_feat_len=None,
            embedding=self.embedding,
            finalize=True,
            n_timesteps=self.n_cfm,
            meanflow=True,
        )                                            # (B, 80, 2*T)
        mels = mels.float()                          # HiFT runs fp32
        wavs, _ = self.s3gen.mel2wav.inference(
            speech_feat=mels,
            cache_source=torch.zeros(Bp, 1, 0, device=self.device),
        )                                            # (Bp, samples)
        wavs_np = wavs.float().cpu().numpy()
        self.gpu_time += time.perf_counter() - t0
        self.batches += 1
        self.jobs += len(jobs)

        for i, j in enumerate(real):
            n_tok = lens[i]
            wav = wavs_np[i, : n_tok * SAMPLES_PER_TOKEN]
            self._postprocess(j, wav)

    def _postprocess(self, j: VocJob, wav: np.ndarray) -> None:
        r = j.req
        fade = self.fade
        off = j.emit_off * SAMPLES_PER_TOKEN
        if j.first:
            seg = wav[off:]
            if len(seg) >= len(self.trim_fade):
                seg[: len(self.trim_fade)] *= self.trim_fade
            out = seg
        else:
            lo = max(0, off - fade)
            seg = wav[lo:]
            n_ov = off - lo
            tail = r.voc_tail
            if tail is not None and n_ov > 0:
                n = min(len(tail), n_ov, len(seg))
                seg[:n] = tail[:n] * self.fade_out[:n] + seg[:n] * self.fade_in[:n]
            out = seg
        if j.first and self.pre_roll:
            out = np.concatenate((
                np.zeros(self.pre_roll, dtype=np.float32), out,
            ))
        if not j.final and len(out) > fade:
            r.voc_tail = out[-fade:].copy()
            out = out[:-fade]
        else:
            r.voc_tail = None

        if j.first and r.voc_tail is not None:
            # first-chunk lane in a dedicated subprocess: hand the crossfade
            # tail to the parent so it can route chunk 2 to the bulk pool
            st = getattr(r, "send_tail", None)
            if st is not None:
                st(r.voc_tail)

        pcm = np.clip(out, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2").tobytes()
        r.audio_samples += len(out)
        if r.t_first_pcm == 0.0:
            r.t_first_pcm = time.perf_counter()
        self._emit(r, pcm)
        if j.final:
            self._finish(r)

    def _finish(self, r) -> None:
        if r.finished:
            return
        r.finished = True
        r.t_done = time.perf_counter()
        self._emit(r, None)

    def _emit(self, r, data) -> None:
        if r.sink is not None:
            r.sink(data)

    def stats(self) -> dict:
        return dict(
            voc_batches=self.batches,
            voc_jobs=self.jobs,
            voc_queue=len(self.q),
            voc_gpu_s=round(self.gpu_time, 2),
        )
