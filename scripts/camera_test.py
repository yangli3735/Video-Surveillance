#!/usr/bin/env python3
"""
scripts/camera_test.py — Week 1 CSI Camera Validation
======================================================
Platform : NVIDIA Jetson Orin Nano · JetPack 6.1 · Docker
Camera   : CSI on CAM0 (sensor-id=0) via nvarguscamerasrc
Purpose  : Open the camera, show a live window with real-time FPS overlay,
           and log basic metrics.  Press 'q' to quit.

Run inside the Docker container:
    python3 scripts/camera_test.py
"""

import sys
import time
import logging

import cv2
import yaml

# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/camera_test.log", mode="a"),
    ],
)
log = logging.getLogger("camera_test")

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
display_cfg = cfg.get("display", {})

# GStreamer pipeline for CSI camera on CAM0
# This pipeline uses nvarguscamerasrc (NVIDIA Argus) to capture from the CSI
# port, converts NVMM GPU memory to CPU-accessible BGR via nvvidconv +
# videoconvert, and feeds it to OpenCV through appsink.
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

WINDOW_NAME = display_cfg.get("window_name", "Camera-Test")

# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("Opening CSI camera with GStreamer pipeline:")
    log.info("  %s", GST_PIPELINE)

    # cv2.CAP_GSTREAMER tells OpenCV to interpret the string as a GStreamer
    # pipeline instead of a device path or URL.
    cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        log.error("Failed to open camera. Checklist:")
        log.error("  1. Is the Argus daemon running on the host?")
        log.error("     → sudo systemctl status nvargus-daemon")
        log.error("  2. Did you mount the Argus socket?")
        log.error("     → -v /tmp/argus_socket:/tmp/argus_socket")
        log.error("  3. Did you pass --runtime nvidia?")
        log.error("  4. Does 'gst-inspect-1.0 nvarguscamerasrc' succeed?")
        sys.exit(1)

    log.info("Camera opened successfully.")

    frame_count = 0
    fps = 0.0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Empty frame received — retrying…")
                continue

            frame_count += 1
            elapsed = time.time() - t_start

            # Update FPS every 0.5 seconds
            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                t_start = time.time()

            # Draw FPS on the frame
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

            cv2.imshow(WINDOW_NAME, frame)

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
