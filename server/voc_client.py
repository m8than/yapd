"""Parent-side client for a pool of vocoder subprocesses.

Implements the producer interface the engine expects (``enqueue`` /
``finalize_empty`` / ``chunk_first`` / ``chunk``). Jobs are routed to a
worker by request id (per-request affinity keeps chunk order and crossfade
tails on one worker); a reader thread per worker routes returned PCM chunks
to per-request sinks and metrics.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class VocClient:
    def __init__(self, cfg: dict):
        import torch.multiprocessing as mp

        ctx = mp.get_context("spawn")
        from server.voc_proc import main as voc_main

        devices = cfg.get("devices") or [cfg.get("device", "cuda")]
        self.n = len(devices) if cfg.get("devices") else max(1, int(cfg.get("procs", 1)))
        self.in_qs = [ctx.Queue() for _ in range(self.n)]
        self.out_qs = [ctx.Queue() for _ in range(self.n)]
        self.procs = []
        for i in range(self.n):
            child_cfg = dict(cfg, device=devices[i % len(devices)])
            p = ctx.Process(target=voc_main,
                            args=(child_cfg, self.in_qs[i], self.out_qs[i]),
                            daemon=True)
            p.start()
            self.procs.append(p)

        self.chunk_first = cfg["chunk_first"]
        self.chunk_second = cfg.get("chunk_second", 0)
        self.topup_sizes = tuple(cfg.get("topup_sizes") or ())
        self.topup_until = cfg.get("topup_until", 0) or (
            self.chunk_first
            + (sum(self.topup_sizes) if self.topup_sizes else self.chunk_second)
        )
        self.chunk = cfg["chunk"]
        self.lookback = cfg["lookback"]
        self._reqs: dict[int, object] = {}
        self._lock = threading.Lock()
        self.jobs = 0
        self.chunks_out = 0
        self._ready = False

    def wait_ready(self, timeout: float = 900) -> None:
        if self._ready:
            return
        for i in range(self.n):
            msg = self.out_qs[i].get(timeout=timeout)
            assert msg == ("ready",), msg
        self._ready = True
        for i in range(self.n):
            threading.Thread(target=self._read_loop, args=(self.out_qs[i],),
                             daemon=True).start()

    def _route(self, rid: int):
        return self.in_qs[rid % self.n]

    # ---- engine-facing producer API (engine thread) ----------------------- #

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
        self.send_job(
            req, req.committed[w_start:e_end], e_start - w_start,
            first, final, priority=priority, deadline=deadline,
        )

    def send_job(self, req, window, emit_off, first, final, tail=None,
                 priority=None, deadline=float("inf")) -> None:
        with self._lock:
            self._reqs[req.rid] = req
        self.jobs += 1
        self._route(req.rid).put((
            "job", req.rid, window, emit_off, first, final, tail,
            first if priority is None else priority, deadline,
        ))

    def finalize_empty(self, req) -> None:
        with self._lock:
            self._reqs[req.rid] = req
        self._route(req.rid).put(("end", req.rid))

    # ---- consumer side ----------------------------------------------------- #

    def _read_loop(self, out_q) -> None:
        import numpy as np

        while True:
            rid, data = out_q.get()
            with self._lock:
                req = self._reqs.get(rid)
            if req is None:
                continue
            if isinstance(data, tuple):
                # ("tail", bytes): crossfade tail from the first-chunk proc;
                # must be set before audio_samples goes nonzero (queue FIFO)
                req.voc_tail = np.frombuffer(data[1], np.float32).copy()
                continue
            if data is None:
                req.finished = True
                req.t_done = time.perf_counter()
                with self._lock:
                    self._reqs.pop(rid, None)
                if req.sink is not None:
                    req.sink(None)
            else:
                if req.t_first_pcm == 0.0:
                    req.t_first_pcm = time.perf_counter()
                req.audio_samples += len(data) // 2
                self.chunks_out += 1
                if req.sink is not None:
                    req.sink(data)

    def stats(self) -> dict:
        return dict(voc_jobs=self.jobs, voc_chunks=self.chunks_out,
                    voc_procs=self.n,
                    voc_alive=sum(p.is_alive() for p in self.procs))

    def stop(self) -> None:
        for q in self.in_qs:
            q.put(("stop",))


class SplitVoc:
    """Routes first chunks to a dedicated vocoder subprocess (own GIL, never
    behind a bulk batch -> minimal TTFP) and bulk chunks to the main pool.
    A rid's bulk jobs are held until its first chunk has emitted (chunk order
    + crossfade tail dependency); the tail, returned by the first-chunk proc,
    is forwarded with the rid's first bulk job.
    """

    def __init__(self, first: VocClient, remote: VocClient):
        from collections import deque
        self.topup_until = first.topup_until

        self.first = first
        self.remote = remote
        self.chunk_first = first.chunk_first
        self.chunk_second = first.chunk_second
        self.topup_sizes = first.topup_sizes
        self.chunk = remote.chunk
        self.lookback = remote.lookback
        self._pending = deque()   # (req, window, emit_off, final)
        self._sent_tail: set[int] = set()
        self._plock = threading.Lock()
        threading.Thread(target=self._flush_loop, daemon=True).start()

    # ---- engine-facing producer API --------------------------------------- #

    def enqueue(self, req, final: bool, n_emit: int | None = None) -> None:
        e_start = req.emitted_tok
        if e_start == 0:
            self.first.enqueue(req, final=final, n_emit=n_emit)
            return
        # snapshot the window now: the engine bumps emitted_tok right after
        e_end = len(req.committed) if n_emit is None else e_start + n_emit
        w_start = max(0, e_end - (self.lookback + self.chunk))
        w_start = min(w_start, e_start)
        job = (req, req.committed[w_start:e_end], e_start - w_start, final)
        with self._plock:
            self._pending.append(job)
        self._flush()

    def finalize_empty(self, req) -> None:
        self.remote.finalize_empty(req)

    # ---- pending flush ----------------------------------------------------- #

    def _flush(self) -> None:
        with self._plock:
            held: set[int] = set()
            for _ in range(len(self._pending)):
                req, window, emit_off, final = self._pending.popleft()
                rid = req.rid
                # audio_samples > 0 implies the local first chunk emitted and
                # req.voc_tail is final (set before the counter is bumped)
                if rid in held or req.audio_samples == 0:
                    held.add(rid)
                    self._pending.append((req, window, emit_off, final))
                    continue
                tail = None
                if rid not in self._sent_tail:
                    t = req.voc_tail
                    tail = t.astype("float32").tobytes() if t is not None else b""
                    self._sent_tail.add(rid)
                self.remote.send_job(req, window, emit_off, False, final, tail)
                if final:
                    self._sent_tail.discard(rid)

    def _flush_loop(self) -> None:
        while True:
            time.sleep(0.002)
            if self._pending:
                try:
                    self._flush()
                except Exception:
                    logger.exception("splitvoc flush failed")

    # ---- misc -------------------------------------------------------------- #

    def stats(self) -> dict:
        out = self.remote.stats()
        f = self.first.stats()
        out.update(
            first_jobs=f["voc_jobs"],
            first_chunks=f["voc_chunks"],
            first_alive=f["voc_alive"],
            split_pending=len(self._pending),
        )
        return out

    def wait_ready(self, timeout: float = 900) -> None:
        self.first.wait_ready(timeout)
        self.remote.wait_ready(timeout)

    def stop(self) -> None:
        self.first.stop()
        self.remote.stop()
