"""
modules/recording/recorder.py
================================
Responsibility:
  - Save JPEG snapshots on event trigger
  - Save MP4 video clips on event trigger (includes pre-event frames from RingBuffer)
  - Run disk I/O in its OWN thread to avoid blocking the main pipeline
  - Manage save directory and file naming

Design notes:
  - On event: snapshot is written immediately from the event frame
  - On event: video writer opens, writes pre-event ring buffer frames,
    then continues writing frames for post_event_record_seconds
  - A simple state machine: IDLE → RECORDING → IDLE
  - Thread-safe: recording is triggered via a queue from the event callback
"""
import os
import queue
import threading
import time
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np

from config import cfg
from modules.event.event_handler import Event
from utils.frame_buffer import RingBuffer
from utils.logger import get_logger

log = get_logger(__name__)


class _Cmd(Enum):
    TRIGGER = auto()
    FRAME = auto()
    STOP = auto()


class Recorder:
    """
    Event-driven video and snapshot recorder.

    Usage:
        recorder = Recorder(ring_buffer)
        recorder.start()

        # Wire into EventHandler:
        event_handler.register_callback(recorder.on_event)

        # Feed every annotated frame so the recorder can continue the clip:
        recorder.push_frame(annotated_frame)

        recorder.stop()
        recorder.join()
    """

    def __init__(self, ring_buffer: RingBuffer):
        rec_cfg = cfg["recording"]
        self._enabled: bool = rec_cfg.get("enabled", True)
        self._save_dir: str = rec_cfg.get("save_dir", "data/recordings")
        self._snapshot_on_event: bool = rec_cfg.get("snapshot_on_event", True)
        self._video_on_event: bool = rec_cfg.get("video_on_event", True)
        self._post_seconds: float = rec_cfg.get("post_event_record_seconds", 5.0)
        self._fps: int = rec_cfg.get("video_fps", 20)
        self._fourcc_str: str = rec_cfg.get("video_codec", "mp4v")

        cam_cfg = cfg["camera"]
        self._width: int = cam_cfg["width"]
        self._height: int = cam_cfg["height"]

        self._ring = ring_buffer
        self._cmd_queue: queue.Queue = queue.Queue()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=60)

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="RecorderThread", daemon=True)

        os.makedirs(self._save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._enabled:
            log.info("Recorder starting (save_dir=%s)", self._save_dir)
            self._thread.start()

    def stop(self) -> None:
        log.info("Recorder stop requested")
        self._cmd_queue.put((_Cmd.STOP, None))
        self._stop_event.set()

    def join(self, timeout: float = 10.0) -> None:
        self._thread.join(timeout=timeout)

    def on_event(self, event: Event) -> None:
        """EventHandler callback — called from main thread."""
        if self._enabled:
            self._cmd_queue.put((_Cmd.TRIGGER, event))

    def push_frame(self, frame: np.ndarray) -> None:
        """Called every frame from the main loop (non-blocking)."""
        if not self._enabled:
            return
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # drop frame rather than blocking

    # ------------------------------------------------------------------
    # Internal recording loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        writer: Optional[cv2.VideoWriter] = None
        recording_until: float = 0.0
        fourcc = cv2.VideoWriter_fourcc(*self._fourcc_str)

        while not self._stop_event.is_set():
            # ---- Check for commands ----
            try:
                cmd, payload = self._cmd_queue.get_nowait()
                if cmd == _Cmd.STOP:
                    break
                if cmd == _Cmd.TRIGGER:
                    event: Event = payload
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    tag = f"{event.trigger_class}_{ts}"

                    # ---- Snapshot ----
                    if self._snapshot_on_event:
                        snap_path = os.path.join(self._save_dir, f"{tag}.jpg")
                        cv2.imwrite(snap_path, event.result.frame)
                        log.info("Snapshot saved: %s", snap_path)

                    # ---- Start / extend video clip ----
                    if self._video_on_event:
                        if writer is None:
                            vid_path = os.path.join(self._save_dir, f"{tag}.mp4")
                            writer = cv2.VideoWriter(
                                vid_path, fourcc, self._fps,
                                (self._width, self._height),
                            )
                            # Write pre-event frames from ring buffer
                            for pre_frame in self._ring.get_all():
                                writer.write(pre_frame)
                            log.info("Video recording started: %s", vid_path)

                        recording_until = time.monotonic() + self._post_seconds

            except queue.Empty:
                pass

            # ---- Write frames while recording ----
            if writer is not None:
                try:
                    frame = self._frame_queue.get(timeout=0.05)
                    writer.write(frame)
                except queue.Empty:
                    pass

                if time.monotonic() > recording_until:
                    writer.release()
                    writer = None
                    log.info("Video recording ended")

        # ---- Clean up ----
        if writer is not None:
            writer.release()
        log.info("Recorder thread exited cleanly")
