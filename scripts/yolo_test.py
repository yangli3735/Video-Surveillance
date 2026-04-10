#!/usr/bin/env python3
"""
scripts/yolo_test.py — Week 1 YOLOv8 Detection Validation
==========================================================
Platform : NVIDIA Jetson Orin Nano · JetPack 6.1 · Docker
Camera   : CSI on CAM0 (sensor-id=0) via nvarguscamerasrc
Purpose  : Run YOLOv8n on live camera feed, draw bounding boxes (focus on
           person class), display FPS, and log detections.
           Press 'q' to quit.

Run inside the Docker container:
    python3 scripts/yolo_test.py
"""

import sys
import time
import logging

import cv2
import yaml
from ultralytics import YOLO

# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/yolo_test.log", mode="a"),
    ],
)
log = logging.getLogger("yolo_test")

# ---------------------------------------------------------------------------
#  Load config
# ---------------------------------------------------------------------------
CONFIG_PATH = "config/settings.yaml"
try:
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
except FileNotFoundError:
    log.warning("Config file %s not found — using defaults", CONFIG_PATH)
    cfg = {}

cam_cfg = cfg.get("camera", {})
inf_cfg = cfg.get("inference", {})
display_cfg = cfg.get("display", {})

# GStreamer pipeline for CSI camera on CAM0
GST_PIPELINE = cam_cfg.get(
    "gstreamer_pipeline",
    (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=1"
    ),
)

MODEL_PATH = inf_cfg.get("model_path", "data/models/yolov8n.pt")
ENGINE_PATH = inf_cfg.get("engine_path", "data/models/yolov8n.engine")
USE_TENSORRT = inf_cfg.get("use_tensorrt", False)
CONF_THRESH = inf_cfg.get("confidence_threshold", 0.50)
IOU_THRESH = inf_cfg.get("iou_threshold", 0.45)
DEVICE = inf_cfg.get("device", "cuda")

WINDOW_NAME = display_cfg.get("window_name", "YOLO-Test")
SHOW_FPS = display_cfg.get("show_fps", True)
SHOW_LABELS = display_cfg.get("show_labels", True)
SHOW_CONF = display_cfg.get("show_confidence", True)

# COCO class index for "person"
PERSON_CLASS_ID = 0

# Colors: person = green, others = blue
COLOR_PERSON = (0, 255, 0)
COLOR_OTHER = (255, 180, 0)

# ---------------------------------------------------------------------------
#  Load YOLO model
# ---------------------------------------------------------------------------
def load_model() -> YOLO:
    """Load the YOLO model — TensorRT engine if enabled, otherwise .pt."""
    if USE_TENSORRT:
        log.info("Loading TensorRT engine: %s", ENGINE_PATH)
        model = YOLO(ENGINE_PATH, task="detect")
    else:
        log.info("Loading PyTorch model: %s", MODEL_PATH)
        model = YOLO(MODEL_PATH)
    return model

# ---------------------------------------------------------------------------
#  Draw detections
# ---------------------------------------------------------------------------
def draw_detections(frame, results) -> int:
    """Draw bounding boxes and labels. Returns person count."""
    person_count = 0

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            is_person = cls_id == PERSON_CLASS_ID
            if is_person:
                person_count += 1

            color = COLOR_PERSON if is_person else COLOR_OTHER
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if SHOW_LABELS:
                class_name = result.names.get(cls_id, str(cls_id))
                label = f"{class_name}"
                if SHOW_CONF:
                    label += f" {conf:.2f}"

                # Label background
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                )
                cv2.rectangle(
                    frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1
                )
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

    return person_count

# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Load model
    model = load_model()

    # Open CSI camera
    log.info("Opening CSI camera with GStreamer pipeline:")
    log.info("  %s", GST_PIPELINE)

    cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        log.error("Failed to open camera. See camera_test.py troubleshooting.")
        sys.exit(1)

    log.info("Camera opened. Starting detection loop…")

    frame_count = 0
    fps = 0.0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Empty frame — retrying…")
                continue

            frame_count += 1
            elapsed = time.time() - t_start

            # Run YOLO inference
            results = model.predict(
                frame,
                conf=CONF_THRESH,
                iou=IOU_THRESH,
                device=DEVICE,
                verbose=False,
            )

            # Draw detections
            person_count = draw_detections(frame, results)

            # Update FPS every 0.5 seconds
            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                t_start = time.time()

            # HUD overlay
            if SHOW_FPS:
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                f"Persons: {person_count}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, frame)

            # Log periodically (roughly every 5 seconds)
            if frame_count == 1 and fps > 0:
                log.info("FPS: %.1f  |  Persons detected: %d", fps, person_count)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("User pressed 'q' — exiting.")
                break

    except KeyboardInterrupt:
        log.info("Interrupted — shutting down.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log.info("Camera released. Done.")


if __name__ == "__main__":
    main()
