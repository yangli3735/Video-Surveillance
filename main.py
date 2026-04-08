"""
main.py — Smart Camera Surveillance System
============================================
Application entry point for NVIDIA Jetson Orin Nano.

Pipeline (data flow):
  CameraInput  ──[raw_queue]──►  YOLODetector  ──[result_queue]──►  main loop
       │                                                                  │
    [ring_buf]                                                     EventHandler
                                                                    │        │
                                                               Recorder   GPIOController
                                                                    │
                                                            DisplayOutput (main thread)

Thread layout:
  Thread-1  CameraInput      (grab frames, push to raw_queue + ring_buf)
  Thread-2  YOLODetector     (pop raw_queue, infer, push to result_queue)
  Thread-3  Recorder         (pop events, write MP4/JPEG to disk)
  Main      main loop        (pop result_queue, EventHandler, DisplayOutput)

Run:
    python main.py
    python main.py --config config/settings.yaml   (explicit config — future)
"""
import signal
import sys
import time

from config import cfg
from modules.camera.camera_input import CameraInput
from modules.display.display_output import DisplayOutput
from modules.event.event_handler import EventHandler
from modules.gpio.gpio_controller import GPIOController
from modules.inference.yolo_detector import YOLODetector
from modules.recording.recorder import Recorder
from utils.frame_buffer import FrameQueue, RingBuffer
from utils.logger import get_logger

log = get_logger(__name__)


# ------------------------------------------------------------------
# Global shutdown flag (set by signal handler or display exit)
# ------------------------------------------------------------------
_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    log.info("Signal %d received — initiating shutdown", signum)
    _shutdown = True


# ------------------------------------------------------------------
# Manual snapshot helper (wired into DisplayOutput key handler)
# ------------------------------------------------------------------
_manual_snap_recorder: Recorder | None = None


def _manual_snapshot(frame):
    import os
    import cv2
    save_dir = cfg["recording"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"manual_{ts}.jpg")
    cv2.imwrite(path, frame)
    log.info("Manual snapshot: %s", path)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    global _shutdown

    log.info("=" * 60)
    log.info("Smart Camera Surveillance System — starting up")
    log.info("Platform: NVIDIA Jetson Orin Nano")
    log.info("=" * 60)

    # ---- Signal handlers ----
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ---- Shared data structures ----
    queue_size: int = cfg["app"].get("queue_maxsize", 4)
    pre_buf_fps = cfg["camera"]["fps"]
    pre_buf_secs = cfg["recording"].get("pre_event_buffer_seconds", 3)

    raw_queue = FrameQueue(maxsize=queue_size)
    result_queue = FrameQueue(maxsize=queue_size)
    ring_buffer = RingBuffer(maxlen=int(pre_buf_fps * pre_buf_secs))

    # ---- Build modules ----
    camera = CameraInput(frame_queue=raw_queue, ring_buffer=ring_buffer)
    detector = YOLODetector(frame_queue=raw_queue, result_queue=result_queue)
    recorder = Recorder(ring_buffer=ring_buffer)
    gpio = GPIOController()
    event_handler = EventHandler()
    display = DisplayOutput(
        camera_fps_fn=camera.fps.__get__(camera),  # bound property getter
        manual_snapshot_fn=_manual_snapshot,
    )

    # ---- Wire event callbacks ----
    event_handler.register_callback(recorder.on_event)
    event_handler.register_callback(gpio.on_event)

    # ---- Setup hardware ----
    gpio.setup()

    # ---- Start background threads ----
    camera.start()
    detector.start()
    recorder.start()

    log.info("All threads started — running main loop (press q or Ctrl+C to stop)")

    # ---- Main loop (runs on main thread for OpenCV GUI) ----
    try:
        while not _shutdown:
            result = result_queue.get(timeout=1.0)
            if result is None:
                continue

            # Event evaluation
            event_handler.process(result)

            # Feed recorder with the annotated frame for ongoing clip
            recorder.push_frame(result.frame)

            # Display
            keep_running = display.show(result)
            if not keep_running:
                _shutdown = True

    except Exception as exc:
        log.exception("Unexpected error in main loop: %s", exc)
    finally:
        log.info("Shutting down…")
        camera.stop()
        detector.stop()
        recorder.stop()

        camera.join()
        detector.join()
        recorder.join()

        gpio.cleanup()
        display.close()
        log.info("Shutdown complete.")


if __name__ == "__main__":
    main()
