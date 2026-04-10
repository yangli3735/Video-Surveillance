# =============================================================================
#  Dockerfile — Intelligent Security & Abnormal Behavior Analysis System
#  Target : NVIDIA Jetson Orin Nano · JetPack 6.1 · Docker path (Option B)
#  Scope  : Week 1 — environment, camera, YOLOv8, TensorRT export
# =============================================================================
#
#  BASE-IMAGE COMPATIBILITY NOTE
#  ─────────────────────────────
#  The official L4T PyTorch image used below is tagged r36.4.0, built for
#  JetPack 6.1 GA (L4T R36.4.0).  Your host reports L4T R36.5.0.
#
#  Minor point-release mismatches within the same R36 family share the
#  same kernel ABI and NVIDIA driver interface, so an r36.4.0 container
#  runs correctly on an R36.5.0 host in practice.  This is a common and
#  accepted real-world compatibility compromise on Jetson.
#
#  If NVIDIA publishes an r36.5.0 tag later, simply update the FROM line.
#  To browse all available tags:
#    https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch
#
# =============================================================================

FROM nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.5-py3

ENV DEBIAN_FRONTEND=noninteractive

# ── 1. System packages ─────────────────────────────────────────────────────
#  GStreamer runtime + plugins (required by nvarguscamerasrc for CSI camera)
#  v4l-utils for camera enumeration / debugging
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgstreamer1.0-0 \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-tools \
        libgstreamer-plugins-base1.0-dev \
        libglib2.0-dev \
        v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Library paths — make CUDA / TensorRT / cuDNN always discoverable ────
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}
RUN ldconfig

# ── 3. Python dependencies ─────────────────────────────────────────────────
#
#  CRITICAL — Protect Jetson-specific PyTorch
#  ──────────────────────────────────────────
#  The base image ships a Jetson-specific PyTorch wheel compiled for aarch64
#  with CUDA 12.x support.  We MUST NOT let pip silently replace it with a
#  generic PyPI build (x86-only or CPU-only).
#
#  Strategy: freeze the currently installed torch* versions into a pip
#  constraint file, then install all remaining packages under that constraint.
#
RUN pip freeze | grep -iE "^(torch|torchvision)==" > /tmp/torch-pin.txt && \
    echo "── Pinned Jetson packages ──" && cat /tmp/torch-pin.txt

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
        -c /tmp/torch-pin.txt \
        -r /tmp/requirements.txt

#  ultralytics pulls opencv-python from PyPI.  That generic build lacks
#  GStreamer and CUDA support.  Remove it so Python falls back to the
#  system-built cv2 (which has both).
RUN pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true

# ── 4. Refresh linker cache after all installs ─────────────────────────────
RUN ldconfig

# ── 5. Smoke-test critical imports (fail the build early if broken) ────────
RUN python3 -c "import torch; print(f'PyTorch {torch.__version__}  CUDA available: {torch.cuda.is_available()}')"
RUN python3 -c "import cv2;   print(f'OpenCV  {cv2.__version__}')"
RUN python3 -c "from ultralytics import YOLO; print('ultralytics import OK')"

# ── 6. Application layout ──────────────────────────────────────────────────
WORKDIR /app
COPY . /app

CMD ["python3", "main.py"]
