# Intelligent Security & Abnormal Behavior Analysis System

> **Scope**: Week 1 — environment setup, camera validation, YOLOv8 detection, TensorRT export path  
> **Platform**: NVIDIA Jetson Orin Nano · JetPack 6.1 · L4T R36.5.0 · CUDA 12.6  
> **Runtime**: Docker with NVIDIA runtime (Option B)

---

## 1. Project Overview

This project builds an edge-based intelligent security camera system on the
NVIDIA Jetson Orin Nano.  Week 1 focuses exclusively on:

| Goal | Validation |
|------|-----------|
| Environment setup | Docker image builds, all imports work |
| CSI camera access | Live feed via `nvarguscamerasrc`, FPS overlay |
| YOLOv8 detection | Person detection with bounding boxes on live feed |
| TensorRT export | `.pt` → `.engine` conversion runs on-device |

Later phases (tracking, fall detection, event pipeline, web UI, recording,
GPIO alerts) are **not** in scope for Week 1.

---

## 2. Hardware & Software Stack

| Component | Detail |
|-----------|--------|
| Device | NVIDIA Jetson Orin Nano |
| JetPack | 6.1 (L4T R36.5.0) |
| CUDA | 12.6 |
| Camera | CSI camera on **CAM0** (`sensor-id=0`) |
| Docker | NVIDIA Container Runtime (`--runtime nvidia`) |
| DNN | YOLOv8n (Ultralytics) |
| Inference | PyTorch → TensorRT FP16 engine |

---

## 3. CSI Camera — GStreamer Pipeline Explained

Jetson CSI cameras are **not** standard V4L2 `/dev/video*` webcams.  They are
accessed through the **NVIDIA Argus** camera daemon, which exposes frames via
the GStreamer element `nvarguscamerasrc`.

The full pipeline used throughout this project:

```
nvarguscamerasrc sensor-id=0 ! \
  video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! \
  nvvidconv ! video/x-raw, format=BGRx ! \
  videoconvert ! video/x-raw, format=BGR ! \
  appsink drop=1
```

### Pipeline breakdown

| Stage | Purpose |
|-------|---------|
| `nvarguscamerasrc sensor-id=0` | Capture from CSI port CAM0 via the Argus daemon |
| `video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1` | Request 1280×720 @ 30 fps in NVMM (GPU) memory |
| `nvvidconv` | Hardware-accelerated pixel format conversion (runs on VIC engine) |
| `video/x-raw, format=BGRx` | Output BGRx (4-byte aligned BGR) |
| `videoconvert` | Software conversion to 3-channel BGR |
| `video/x-raw, format=BGR` | Final format that OpenCV `cv2.Mat` expects |
| `appsink drop=1` | Deliver frames to the application; drop stale frames to reduce latency |

> **Key point**: `cv2.VideoCapture()` must be called with `cv2.CAP_GSTREAMER`
> as the second argument so OpenCV uses the GStreamer backend.

### Why CSI instead of USB?

The Orin Nano's CSI interface offers:
- Zero-copy GPU memory access (NVMM) — no CPU bounce buffer
- Hardware ISP (image signal processor) for exposure, white balance, noise reduction
- Lower and more predictable latency than USB cameras

---

## 4. Project Structure

```
camera_project/
├── Dockerfile              # Docker build file (JetPack 6.1 / L4T R36.4.0 base)
├── requirements.txt        # Python deps (excluding torch/opencv — see comments)
├── README.md               # ← you are here
├── main.py                 # Application entry point (future)
├── config/
│   └── settings.yaml       # All runtime parameters (camera, YOLO, display, …)
├── scripts/
│   ├── verify_env.sh       # Smoke-test: CUDA, torch, OpenCV, GStreamer, TensorRT
│   ├── camera_test.py      # CSI camera open + FPS display
│   ├── yolo_test.py        # YOLOv8 person detection on live camera feed
│   └── export_engine.sh    # Export YOLOv8n .pt → TensorRT .engine
├── modules/                # Application modules (camera, display, inference, …)
├── utils/                  # Shared utilities (logger, frame buffer, …)
├── data/
│   ├── models/             # YOLO weights (.pt) and TensorRT engines (.engine)
│   └── recordings/         # Event snapshots / clips (future)
└── logs/                   # Runtime logs
```

---

## 5. Quick Start

### 5.1 Build the Docker Image

```bash
cd ~/camera_project

docker build -t camera-system:week1 .
```

> The first build may take 10–20 minutes (downloading base image + pip packages).
> Subsequent rebuilds use the Docker cache and are much faster.

### 5.2 Run the Container

```bash
docker run --rm -it \
    --runtime nvidia \
    --network host \
    -v /tmp/argus_socket:/tmp/argus_socket \
    -v "$(pwd)/data":/app/data \
    -v "$(pwd)/logs":/app/logs \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    camera-system:week1 \
    bash
```

**Flag explanation**:

| Flag | Why |
|------|-----|
| `--runtime nvidia` | Enables GPU access inside the container |
| `-v /tmp/argus_socket:…` | Shares the Argus daemon socket so `nvarguscamerasrc` can access the CSI camera |
| `-v $(pwd)/data:/app/data` | Persist model files and recordings across container restarts |
| `-v $(pwd)/logs:/app/logs` | Persist log files |
| `-e DISPLAY` + `-v /tmp/.X11-unix:…` | Forward X11 display so OpenCV windows render on the local monitor |
| `--network host` | Simplifies network access (optional, convenient for debugging) |

### 5.3 Run Week 1 Validation Scripts (inside the container)

```bash
# 1. Verify environment
bash scripts/verify_env.sh

# 2. Camera test — opens a live window with FPS overlay (press 'q' to quit)
python3 scripts/camera_test.py

# 3. YOLO detection — live person detection with bounding boxes (press 'q' to quit)
python3 scripts/yolo_test.py

# 4. Export YOLOv8n to TensorRT engine (takes a few minutes on first run)
bash scripts/export_engine.sh
```

---

## 6. Week 1 Validation Checklist

- [ ] `docker build` completes without errors
- [ ] `verify_env.sh` reports CUDA, torch, OpenCV (with GStreamer), ultralytics, TensorRT all OK
- [ ] `camera_test.py` opens the CSI camera and shows a live feed with FPS ≥ 25
- [ ] `yolo_test.py` detects persons with bounding boxes; FPS displayed on screen
- [ ] `export_engine.sh` produces `data/models/yolov8n.engine`
- [ ] Container restarts do not lose model files (mounted volume)

---

## 7. Troubleshooting

### Camera not opening / black screen

```bash
# Check that the Argus daemon is running on the HOST (not inside the container):
sudo systemctl status nvargus-daemon

# If stopped:
sudo systemctl start nvargus-daemon
```

Make sure `-v /tmp/argus_socket:/tmp/argus_socket` is in your `docker run` command.

### `nvarguscamerasrc` not found inside container

The GStreamer plugin is loaded from the host via the NVIDIA runtime.  Verify:

```bash
gst-inspect-1.0 nvarguscamerasrc
```

If missing, ensure `--runtime nvidia` is passed in the `docker run` command.

### OpenCV cannot open GStreamer pipeline

```python
# Quick check — run inside the container:
import cv2
print(cv2.getBuildInformation())
# Look for "GStreamer: YES" in the output.
```

If GStreamer shows NO, the pip-installed `opencv-python` may be shadowing the
system build.  Fix:

```bash
pip uninstall -y opencv-python opencv-contrib-python
python3 -c "import cv2; print(cv2.__version__)"
```

### CUDA not available in PyTorch

```python
import torch
print(torch.cuda.is_available())   # Should be True
print(torch.version.cuda)          # Should show 12.x
```

If False, the Jetson PyTorch wheel may have been replaced by a generic pip
build.  Rebuild the Docker image — the constraint file in the Dockerfile
prevents this.

---

## 8. Base Image Compatibility Note

The Dockerfile uses `nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.5-py3`,
targeting L4T R36.4.0 (JetPack 6.1 GA).  Your host runs L4T **R36.5.0**.

This is expected and acceptable:

- R36.4.0 and R36.5.0 share the same major kernel ABI
- The NVIDIA driver interface is backward-compatible within the R36 family
- Container user-space libraries (CUDA, TensorRT, cuDNN) communicate with
  the host kernel driver, which handles the minor version difference

If you encounter issues, check NGC for an updated tag:
<https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch>

---

*Week 1 complete.  Future phases will build on this validated foundation.*
