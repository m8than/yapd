"""Shared admission primitives for heterogeneous speech model engines.

The execution cadence remains model-specific: autoregressive and block-diffusion
engines yield at decode ticks, while Kokoro yields after a complete acoustic
forward. This module owns the common concurrent waiting queue, priority
selection, compatible batching, wakeups, and shutdown state.
"""
from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from typing import Generic, TypeVar


RequestT = TypeVar("RequestT")


class SchedulerQueue(Generic[RequestT]):
    """Thread-safe priority queue with model-defined batch compatibility.

    A deque is intentional. Speech queues are normally short, while selecting a
    compatible microbatch requires examining every waiter regardless of the
    backing data structure. It also preserves FIFO order among equal priorities.
    """

    def __init__(self) -> None:
        self._items: deque[RequestT] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = False

    def submit(self, request: RequestT) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("scheduler is closed")
            self._items.append(request)
        self._wake.set()

    def depth(self) -> int:
        with self._lock:
            return len(self._items)

    def __bool__(self) -> bool:
        return self.depth() > 0

    def take(
        self,
        limit: int,
        *,
        select_key: Callable[[RequestT], object] | None = None,
    ) -> list[RequestT]:
        """Remove up to ``limit`` requests, priority first and FIFO on ties."""
        if limit <= 0:
            return []
        with self._lock:
            if not self._items:
                return []
            if select_key is None:
                return [self._items.popleft() for _ in range(min(limit, len(self._items)))]
            ordered = sorted(
                enumerate(self._items),
                key=lambda item: (select_key(item[1]), item[0]),
            )
            chosen = {index for index, _ in ordered[:limit]}
            selected = [request for index, request in ordered[:limit]]
            self._items = deque(
                request for index, request in enumerate(self._items)
                if index not in chosen
            )
            return selected

    def take_group(
        self,
        *,
        select_key: Callable[[RequestT], object],
        limit_for: Callable[[RequestT], int],
        compatible: Callable[[RequestT, RequestT], bool],
    ) -> list[RequestT]:
        """Select the most urgent request and its FIFO-compatible peers."""
        with self._lock:
            if not self._items:
                return []
            seed = min(
                enumerate(self._items),
                key=lambda item: (select_key(item[1]), item[0]),
            )[1]
            limit = max(1, limit_for(seed))
            selected: list[RequestT] = []
            remaining: deque[RequestT] = deque()
            removed_seed = False
            for request in self._items:
                is_seed = request is seed and not removed_seed
                if is_seed:
                    removed_seed = True
                if len(selected) < limit and (is_seed or compatible(seed, request)):
                    selected.append(request)
                else:
                    remaining.append(request)
            self._items = remaining
            return selected

    def extend_group(
        self,
        selected: list[RequestT],
        *,
        limit: int,
        compatible: Callable[[RequestT, RequestT], bool],
    ) -> int:
        """Append newly arrived compatible peers to an existing batch."""
        if not selected or len(selected) >= limit:
            return 0
        seed = selected[0]
        added = 0
        with self._lock:
            remaining: deque[RequestT] = deque()
            for request in self._items:
                if len(selected) < limit and compatible(seed, request):
                    selected.append(request)
                    added += 1
                else:
                    remaining.append(request)
            self._items = remaining
        return added

    def wait(self, timeout: float) -> None:
        self._wake.wait(timeout)
        self._wake.clear()

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._wake.set()
