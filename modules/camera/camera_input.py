"""
modules/camera/camera_input.py
============================
Responsibility:
  - Open camera (USB via index, or CSI via GStreamer pipeline string)
  - Capture frames continuously in a background thread
  - Push frames into a FrameQueue for downstream consumers
  - Populate a RingBuffer for pre-event recording
  - Expose FPS telemetry

Design notes:
  - Runs in its OWN thread (thread_camera=true in config)
  - If the camera disconnects, it tries to reconnect up to MAX_RETRIES times
  - Frames are BGR numpy arrays (OpenCV standard)
"""
import threading
import time

import cv2

from config import cfg
from utils.frame_buffer import FrameQueue, RingBuffer
from utils.logger import get_logger

log = get_logger(__name__)

_MAX_RETRIES = 5
_RETRY_DELAY = 2.0  # seconds


class CameraInput:
    """
    Threaded camera capture module.

    Usage:
        cam = CameraInput(frame_queue, ring_buffer)
        cam.start()          # begins background thread
        ...
        cam.stop()           # signals thread to exit
        cam.join()           # waits for thread to finish
    """

    def __init__(self, frame_queue: FrameQueue, ring_buffer: RingBuffer):
        cam_cfg = cfg["camera"]
        self._source = cam_cfg["source"]
        self._width: int = cam_cfg["width"]
        self._height: int = cam_cfg["height"]
        self._target_fps: int = cam_cfg["fps"]
        self._use_gstreamer: bool = cam_cfg.get("use_gstreamer", False)

        self._queue = frame_queue
        self._ring = ring_buffer

        self._cap: cv2.VideoCapture | None = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="CameraThread", daemon=True)

        # FPS tracking
        self._frame_count: int = 0
        self._fps: float = 0.0
        self._fps_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        log.info("CameraInput starting (source=%s)", self._source)
        self._thread.start()

    def stop(self) -> None:
        log.info("CameraInput stop requested")
        self._stop_event.set()

    def join(self, timeout: float = 5.0) -> None:
        self._thread.join(timeout=timeout)

    @property
    def fps(self) -> float:
        with self._fps_lock:
            return self._fps

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open_capture(self) -> cv2.VideoCapture:
        if self._use_gstreamer:
            cap = cv2.VideoCapture(self._source, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(self._source)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            cap.set(cv2.CAP_PROP_FPS, self._target_fps)
            # Reduce internal OpenCV buffer to minimise latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self) -> None:
        retries = 0

        while not self._stop_event.is_set():
            # ---- Open camera ----
            self._cap = self._open_capture()
            if not self._cap.isOpened():
                retries += 1
                log.warning("Camera open failed (attempt %d/%d)", retries, _MAX_RETRIES)
                if retries >= _MAX_RETRIES:
                    log.error("Camera could not be opened after %d retries. Exiting thread.", _MAX_RETRIES)
                    return
                time.sleep(_RETRY_DELAY)
                continue

            log.info(
                "Camera opened — %dx%d @ %d fps",
                int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(self._cap.get(cv2.CAP_PROP_FPS)),
            )
            retries = 0
            t_start = time.monotonic()
            frame_count = 0

            # ---- Capture loop ----
            while not self._stop_event.is_set():
                ret, frame = self._cap.read()
                if not ret:
                    log.warning("Frame read failed — camera may have disconnected")
                    break

                self._queue.put(frame)
                self._ring.put(frame)

                # Update FPS every second
                frame_count += 1
                elapsed = time.monotonic() - t_start
                if elapsed >= 1.0:
                    with self._fps_lock:
                        self._fps = frame_count / elapsed
                    frame_count = 0
                    t_start = time.monotonic()

            self._cap.release()
            if not self._stop_event.is_set():
                log.info("Attempting camera reconnect…")
                time.sleep(_RETRY_DELAY)

        log.info("CameraInput thread exited cleanly")
