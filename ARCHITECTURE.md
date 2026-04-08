# Smart Camera Surveillance System — Architecture & Development Guide
**Platform:** NVIDIA Jetson Orin Nano Developer Kit

---

## Thread Layout

| Thread | Module | Responsibility |
|--------|--------|----------------|
| Thread-1 | `CameraInput` | Grabs frames, pushes to `raw_queue` and `ring_buffer` |
| Thread-2 | `YOLODetector` | Pops frames, runs YOLO, pushes `DetectionResult` |
| Thread-3 | `Recorder` | Writes JPEG snapshots and MP4 clips on events |
| Main | `EventHandler` + `DisplayOutput` | Evaluates rules, fires callbacks, renders window |

---

## Pipeline Data Flow

```
CameraInput ──[raw_queue]──► YOLODetector ──[result_queue]──► main loop
     │                                                              │
 [ring_buf]                                                  EventHandler
 (pre-event                                                   │         │
  circular                                                Recorder  GPIOController
  buffer)                                                    │
                                                        DisplayOutput (main thread)
```

---

## Suggested Development Order (MVP → Full)

| Step | Task |
|------|------|
| 1 | `pip install ultralytics PyYAML` and download `yolov8n.pt` |
| 2 | Test camera: `python -c "import cv2; cap=cv2.VideoCapture(0); print(cap.read()[0])"` |
| 3 | Run `python main.py` — you should see a labelled video window |
| 4 | Tune `config/settings.yaml`: camera source, watch_classes, confidence |
| 5 | Enable `recording.enabled: true`, trigger an event, check `data/recordings/` |
| 6 | Enable `use_tensorrt: true` for 3-5× faster inference on-device |
| 7 | Wire buzzer, set `gpio.enabled: true`, test event → buzzer |
| 8 | Add Flask route in a new `modules/web/` module for MJPEG stream |

---

## Key Configuration Knobs (`config/settings.yaml`)

| Setting | Default | Effect |
|---------|---------|--------|
| `camera.source` | CSI GStreamer pipeline | CAM0 port on Jetson Orin Nano (`sensor-id=0`) |
| `camera.use_gstreamer` | `true` | Enabled — CAM0 is a CSI port requiring GStreamer |
| `inference.use_tensorrt` | `false` | Export + cache TRT engine (much faster on Jetson) |
| `inference.infer_every_n_frames` | `2` | Skip frames to save GPU budget |
| `inference.confidence_threshold` | `0.50` | Minimum score for a detection to be shown |
| `event.watch_classes` | `[person, car]` | Only these classes fire events |
| `event.cooldown_seconds` | `5` | Prevents event spam for the same class |
| `recording.enabled` | `true` | Master switch for all recording activity |
| `recording.pre_event_buffer_seconds` | `3` | Seconds before detection saved in clip |
| `recording.post_event_record_seconds` | `5` | Seconds to keep recording after event ends |
| `gpio.enabled` | `false` | Set `true` when buzzer is physically wired |
| `gpio.buzzer_pin` | `18` | BCM pin number for the buzzer |
| `display.enabled` | `true` | Master switch for OpenCV window |
| `display.show_fps` | `true` | Overlay FPS and inference time on screen |

---

## Module Summary

| File | Description |
|------|-------------|
| `main.py` | Entry point — wires all modules, owns the main loop |
| `config/settings.yaml` | All tunable parameters in one place |
| `config/__init__.py` | Loads YAML → `cfg` dict, importable from any module |
| `modules/camera/camera_input.py` | Threaded USB/CSI capture with auto-reconnect |
| `modules/inference/yolo_detector.py` | Threaded YOLO inference + bounding-box drawing |
| `modules/event/event_handler.py` | Rule engine: watch-list × confidence × cooldown → callbacks |
| `modules/recording/recorder.py` | Threaded disk writer: JPEG snapshots + MP4 event clips |
| `modules/display/display_output.py` | OpenCV window with FPS/inference telemetry overlay |
| `modules/gpio/gpio_controller.py` | Jetson.GPIO abstraction: buzzer output, PIR sensor input |
| `utils/logger.py` | Rotating-file + console logger (one call: `get_logger(__name__)`) |
| `utils/frame_buffer.py` | `FrameQueue` (pipeline backpressure) + `RingBuffer` (pre-event clip) |

---

## Planned Future Modules

| Module | Location | Notes |
|--------|----------|-------|
| Web dashboard | `modules/web/` | Flask + MJPEG stream or Socket.IO |
| Face recognition | `modules/face/` | InsightFace or dlib |
| Action recognition | `modules/action/` | Skeleton-based or CNN temporal model |
| Database storage | `modules/database/` | SQLAlchemy + SQLite/PostgreSQL |
| Cloud notification | `modules/cloud/` | REST webhook / MQTT / AWS SNS |
| Alert scheduler | `modules/alert/` | Time-based rules (arm/disarm by time of day) |

---

## TensorRT Acceleration (Step 6 Detail)

YOLOv8n on Jetson Orin Nano — typical performance:

| Mode | Approx. Inference Time |
|------|------------------------|
| PyTorch (FP32) | ~35–60 ms/frame |
| TensorRT (FP16) | ~8–15 ms/frame |

To enable:
1. Set `inference.use_tensorrt: true` in `config/settings.yaml`
2. On first run the system auto-exports the `.pt` model to a `.engine` file
3. Subsequent runs load the engine directly (fast startup)

> The engine is device-specific — it must be built on the Jetson, not on a PC.

---

## CSI Camera GStreamer Pipeline Example

For IMX219 (Raspberry Pi Camera v2) on Jetson:

```yaml
camera:
  use_gstreamer: true
  source: "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
```

> To change resolution, adjust `width`/`height` in both the GStreamer string and the top-level `width`/`height` fields so they match.

---

## GPIO Pin Reference (BCM Numbering)

| Signal | Default BCM Pin | Jetson Orin Nano Header Pin |
|--------|-----------------|----------------------------|
| Buzzer OUT | 18 | Pin 12 |
| PIR Sensor IN | 24 | Pin 18 |

Adjust `gpio.buzzer_pin` and `gpio.motion_sensor_pin` in `settings.yaml` as needed.

---

## Quick Start

```bash
# 1. Install dependencies
pip install ultralytics PyYAML

# 2. Download model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mv yolov8n.pt data/models/

# 3. Run
python main.py
```

Keyboard shortcuts while running:

| Key | Action |
|-----|--------|
| `q` or `ESC` | Quit |
| `s` | Save manual snapshot |
