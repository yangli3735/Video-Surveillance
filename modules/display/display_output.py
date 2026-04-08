"""
modules/display/display_output.py
===================================
Responsibility:
  - Render the annotated frame in an OpenCV window (local mode, V1)
  - Overlay system telemetry: FPS, inference time, detection count, timestamp
  - Handle keyboard shortcuts:
      q / ESC  → request shutdown
      s        → manual snapshot
      r        → toggle recording on/off
  - Future hook: stream frames via MJPEG to a Flask/FastAPI web dashboard

Design notes:
  - MUST be called from the MAIN thread on Linux (OpenCV GUI requirement)
  - Returns False when the user requests exit
"""
import time
from typing import Callable, Optional

import cv2
import numpy as np

from config import cfg
from modules.inference.yolo_detector import DetectionResult
from utils.logger import get_logger

log = get_logger(__name__)

# Overlay colours
_WHITE = (255, 255, 255)
_GREEN = (0, 220, 0)
_YELLOW = (0, 200, 255)
_RED = (0, 0, 255)
_SHADOW = (30, 30, 30)


class DisplayOutput:
    """
    Local OpenCV display module.

    Usage:
        display = DisplayOutput(camera_fps_fn=cam.fps)
        ...
        keep_running = display.show(detection_result)
        if not keep_running:
            shutdown()
    """

    def __init__(
        self,
        camera_fps_fn: Optional[Callable[[], float]] = None,
        manual_snapshot_fn: Optional[Callable[[np.ndarray], None]] = None,
    ):
        disp_cfg = cfg["display"]
        self._enabled: bool = disp_cfg.get("enabled", True)
        self._window: str = disp_cfg.get("window_name", "Surveillance")
        self._show_fps: bool = disp_cfg.get("show_fps", True)

        self._camera_fps_fn = camera_fps_fn
        self._manual_snapshot_fn = manual_snapshot_fn

        self._frame_count: int = 0
        self._display_fps: float = 0.0
        self._t_last: float = time.monotonic()

        if self._enabled:
            cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                self._window,
                cfg["camera"]["width"],
                cfg["camera"]["height"],
            )
            log.info("DisplayOutput window '%s' opened", self._window)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(self, result: DetectionResult) -> bool:
        """
        Render frame with telemetry overlay.
        Returns True to continue, False to exit.
        """
        if not self._enabled:
            return True

        frame = result.frame.copy()
        self._draw_overlay(frame, result)
        cv2.imshow(self._window, frame)

        return self._handle_keys(frame)

    def close(self) -> None:
        if self._enabled:
            cv2.destroyAllWindows()
            log.info("DisplayOutput closed")

    # ------------------------------------------------------------------
    # Overlay rendering
    # ------------------------------------------------------------------

    def _draw_overlay(self, frame: np.ndarray, result: DetectionResult) -> None:
        h, w = frame.shape[:2]

        # ---- Update display FPS ----
        self._frame_count += 1
        now = time.monotonic()
        elapsed = now - self._t_last
        if elapsed >= 1.0:
            self._display_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._t_last = now

        lines = []
        if self._show_fps:
            cam_fps = self._camera_fps_fn() if self._camera_fps_fn else 0.0
            lines.append(f"Cam: {cam_fps:.1f} fps  Disp: {self._display_fps:.1f} fps")
        if result.inference_ms > 0:
            lines.append(f"Infer: {result.inference_ms:.1f} ms")
        lines.append(f"Objects: {len(result.detections)}")
        lines.append(time.strftime("%Y-%m-%d %H:%M:%S"))

        # Translucent background bar
        bar_h = len(lines) * 22 + 8
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (260, bar_h), _SHADOW, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, text in enumerate(lines):
            y = 20 + i * 22
            cv2.putText(frame, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1, cv2.LINE_AA)

    def _handle_keys(self, frame: np.ndarray) -> bool:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            log.info("User requested exit via keyboard")
            return False
        if key == ord("s") and self._manual_snapshot_fn:
            self._manual_snapshot_fn(frame)
            log.info("Manual snapshot triggered")
        return True
