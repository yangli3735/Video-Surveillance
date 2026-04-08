"""
modules/inference/yolo_detector.py
===================================
Responsibility:
  - Load a YOLO model (Ultralytics YOLOv8 or YOLOv11)
  - Optionally load/export a TensorRT engine for maximum Jetson performance
  - Run inference on incoming frames from a FrameQueue
  - Push DetectionResult objects to a results queue for the event handler
  - Annotate frames with bounding boxes and labels
  - Run in its OWN thread (thread_inference=true in config)

TensorRT path:
  First run  → export model to TensorRT engine (saved to engine_path)
  Later runs → load the compiled engine directly (much faster startup + inference)

Inference rate is throttled by infer_every_n_frames to balance latency vs. CPU/GPU load.
"""
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from config import cfg
from utils.frame_buffer import FrameQueue
from utils.logger import get_logger

log = get_logger(__name__)


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------

@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) in pixel coords


@dataclass
class DetectionResult:
    frame: np.ndarray                    # annotated BGR frame
    detections: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


# ------------------------------------------------------------------
# Colour palette for bounding boxes  (class_index → BGR)
# ------------------------------------------------------------------

_COLOURS = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
]


def _colour(class_id: int):
    return _COLOURS[class_id % len(_COLOURS)]


# ------------------------------------------------------------------
# Detector class
# ------------------------------------------------------------------

class YOLODetector:
    """
    Threaded YOLO inference module.

    Usage:
        detector = YOLODetector(frame_queue, result_queue)
        detector.start()
        ...
        detector.stop()
        detector.join()
    """

    def __init__(self, frame_queue: FrameQueue, result_queue: FrameQueue):
        inf_cfg = cfg["inference"]
        self._model_path: str = inf_cfg["model_path"]
        self._engine_path: str = inf_cfg["engine_path"]
        self._use_tensorrt: bool = inf_cfg.get("use_tensorrt", False)
        self._conf: float = inf_cfg["confidence_threshold"]
        self._iou: float = inf_cfg["iou_threshold"]
        self._device: str = inf_cfg.get("device", "cuda")
        self._skip: int = max(1, inf_cfg.get("infer_every_n_frames", 1))

        self._in_queue = frame_queue
        self._out_queue = result_queue

        self._model = None   # loaded in _load_model()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="InferenceThread", daemon=True)

        self._frame_idx: int = 0
        self._last_result: Optional[DetectionResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        log.info("YOLODetector starting (model=%s, TRT=%s)", self._model_path, self._use_tensorrt)
        self._load_model()
        self._thread.start()

    def stop(self) -> None:
        log.info("YOLODetector stop requested")
        self._stop_event.set()

    def join(self, timeout: float = 10.0) -> None:
        self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError:
            log.error("ultralytics not installed. Run: pip install ultralytics")
            raise

        if self._use_tensorrt:
            import os
            if os.path.exists(self._engine_path):
                log.info("Loading TensorRT engine from %s", self._engine_path)
                self._model = YOLO(self._engine_path)
            else:
                log.info("TensorRT engine not found — exporting from %s …", self._model_path)
                base = YOLO(self._model_path)
                base.export(format="engine", device=self._device)
                self._model = YOLO(self._engine_path)
                log.info("TensorRT engine saved to %s", self._engine_path)
        else:
            log.info("Loading YOLO model from %s", self._model_path)
            self._model = YOLO(self._model_path)

        log.info("Model loaded — classes: %d", len(self._model.names))

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            frame = self._in_queue.get(timeout=1.0)
            if frame is None:
                continue

            self._frame_idx += 1

            # Throttle inference rate
            if self._frame_idx % self._skip != 0:
                # Re-use last detections but with the new frame so display stays smooth
                if self._last_result is not None:
                    result = DetectionResult(
                        frame=self._annotate(frame.copy(), self._last_result.detections),
                        detections=self._last_result.detections,
                        inference_ms=0.0,
                    )
                    self._out_queue.put(result)
                else:
                    self._out_queue.put(DetectionResult(frame=frame))
                continue

            t0 = time.perf_counter()
            try:
                results = self._model(
                    frame,
                    conf=self._conf,
                    iou=self._iou,
                    device=self._device,
                    verbose=False,
                )
            except Exception as exc:
                log.error("Inference error: %s", exc)
                self._out_queue.put(DetectionResult(frame=frame))
                continue

            elapsed_ms = (time.perf_counter() - t0) * 1000
            detections = self._parse(results)
            annotated = self._annotate(frame.copy(), detections)

            result = DetectionResult(
                frame=annotated,
                detections=detections,
                inference_ms=elapsed_ms,
            )
            self._last_result = result
            self._out_queue.put(result)

        log.info("YOLODetector thread exited cleanly")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse(self, results) -> List[Detection]:
        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(Detection(
                    class_name=names[cls_id],
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                ))
        return detections

    def _annotate(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        show_labels = cfg["display"].get("show_labels", True)
        show_conf = cfg["display"].get("show_confidence", True)

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            colour = _colour(i)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            if show_labels:
                label = det.class_name
                if show_conf:
                    label += f" {det.confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                y_label = max(y1 - 4, label_size[1] + 4)
                cv2.rectangle(
                    frame,
                    (x1, y_label - label_size[1] - 4),
                    (x1 + label_size[0], y_label),
                    colour, -1,
                )
                cv2.putText(
                    frame, label, (x1, y_label - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
                )
        return frame
