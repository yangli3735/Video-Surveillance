"""
utils/frame_buffer.py
Thread-safe structures for passing frames between pipeline stages.

FrameQueue  — standard producer/consumer queue (camera → inference)
RingBuffer  — fixed-size circular frame buffer for pre-event recording
"""
import queue
import threading
from collections import deque
from typing import Optional

import numpy as np


class FrameQueue:
    """
    Thin wrapper around queue.Queue with non-blocking put (drops oldest when full).
    This prevents slow consumers from stalling the camera thread.
    """

    def __init__(self, maxsize: int = 4):
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)

    def put(self, item, block: bool = False) -> None:
        """Put a frame; if queue is full, drop the oldest and insert new one."""
        if not block:
            try:
                self._q.put_nowait(item)
            except queue.Full:
                try:
                    self._q.get_nowait()   # discard oldest
                except queue.Empty:
                    pass
                self._q.put_nowait(item)
        else:
            self._q.put(item)

    def get(self, timeout: float = 1.0):
        """Get a frame; returns None on timeout."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def empty(self) -> bool:
        return self._q.empty()

    def qsize(self) -> int:
        return self._q.qsize()


class RingBuffer:
    """
    Thread-safe ring buffer that keeps the last N frames.
    Used for pre-event clip recording (frames before a detection fires).
    """

    def __init__(self, maxlen: int = 90):
        self._buf: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def put(self, frame: np.ndarray) -> None:
        with self._lock:
            self._buf.append(frame.copy())

    def get_all(self) -> list:
        """Return a snapshot of all buffered frames (oldest first)."""
        with self._lock:
            return list(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)
