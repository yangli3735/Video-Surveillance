#!/usr/bin/env bash
# =============================================================================
#  verify_env.sh — Week 1 Environment Verification
#  Platform: NVIDIA Jetson Orin Nano · JetPack 6.1 · Docker (Option B)
#
#  Run inside the Docker container:
#    bash scripts/verify_env.sh
# =============================================================================

set -euo pipefail

PASS=0
FAIL=0
WARN=0

pass() { echo "  [PASS] $1"; ((PASS++)); }
fail() { echo "  [FAIL] $1"; ((FAIL++)); }
warn() { echo "  [WARN] $1"; ((WARN++)); }

echo "============================================================"
echo "  Environment Verification — Week 1"
echo "============================================================"
echo ""

# ── 1. CUDA Toolkit ────────────────────────────────────────────────────────
echo "── CUDA ──"
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9.]+')
    pass "nvcc found — CUDA $CUDA_VER"
else
    # nvcc may not be on PATH but CUDA libs are still usable
    if [ -d /usr/local/cuda ]; then
        warn "nvcc not on PATH, but /usr/local/cuda exists"
    else
        fail "CUDA toolkit not found"
    fi
fi
echo ""

# ── 2. NVIDIA GPU ──────────────────────────────────────────────────────────
echo "── GPU ──"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
        && pass "nvidia-smi OK" \
        || warn "nvidia-smi present but query failed (normal on some Jetson images)"
else
    warn "nvidia-smi not found (common in L4T containers; GPU may still work via CUDA)"
fi
echo ""

# ── 3. Python ──────────────────────────────────────────────────────────────
echo "── Python ──"
PYVER=$(python3 --version 2>&1)
pass "$PYVER"
echo ""

# ── 4. PyTorch + CUDA ─────────────────────────────────────────────────────
echo "── PyTorch ──"
python3 -c "
import torch
ver   = torch.__version__
cuda  = torch.cuda.is_available()
cudav = torch.version.cuda if cuda else 'N/A'
dev   = torch.cuda.get_device_name(0) if cuda else 'N/A'
print(f'  torch {ver}  |  CUDA available: {cuda}  |  CUDA version: {cudav}  |  device: {dev}')
" && pass "PyTorch import OK" || fail "PyTorch import failed"
echo ""

# ── 5. OpenCV + GStreamer ──────────────────────────────────────────────────
echo "── OpenCV ──"
python3 -c "
import cv2
ver = cv2.__version__
info = cv2.getBuildInformation()
gst = 'YES' if 'GStreamer:                   YES' in info or 'GStreamer: YES' in info else 'NO'
cuda_cv = 'YES' if 'NVIDIA CUDA' in info and 'YES' in info.split('NVIDIA CUDA')[1][:30] else 'check manually'
print(f'  OpenCV {ver}  |  GStreamer: {gst}  |  CUDA: {cuda_cv}')
if gst == 'NO':
    print('  ⚠  GStreamer support missing — CSI camera will NOT work.')
    print('     Likely cause: pip opencv-python is shadowing system cv2.')
    print('     Fix: pip uninstall opencv-python opencv-contrib-python')
" && pass "OpenCV import OK" || fail "OpenCV import failed"
echo ""

# ── 6. Ultralytics (YOLO) ─────────────────────────────────────────────────
echo "── Ultralytics ──"
python3 -c "
import ultralytics
print(f'  ultralytics {ultralytics.__version__}')
from ultralytics import YOLO
" && pass "ultralytics import OK" || fail "ultralytics import failed"
echo ""

# ── 7. TensorRT ───────────────────────────────────────────────────────────
echo "── TensorRT ──"
python3 -c "
import tensorrt as trt
print(f'  TensorRT {trt.__version__}')
" 2>/dev/null && pass "TensorRT import OK" || warn "TensorRT Python binding not importable (engine export may still work via torch2trt / ultralytics CLI)"
echo ""

# ── 8. GStreamer CLI ───────────────────────────────────────────────────────
echo "── GStreamer CLI ──"
if command -v gst-launch-1.0 &>/dev/null; then
    GST_VER=$(gst-launch-1.0 --version | head -1)
    pass "$GST_VER"
    # Check nvarguscamerasrc element
    if gst-inspect-1.0 nvarguscamerasrc &>/dev/null; then
        pass "nvarguscamerasrc element available"
    else
        warn "nvarguscamerasrc not found — CSI camera may not work inside this container"
        echo "       Make sure you run with --runtime nvidia and -v /tmp/argus_socket:/tmp/argus_socket"
    fi
else
    fail "gst-launch-1.0 not found"
fi
echo ""

# ── Summary ────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Results:  ${PASS} passed   ${FAIL} failed   ${WARN} warnings"
echo "============================================================"

if [ "$FAIL" -gt 0 ]; then
    echo "  ⚠  Some checks failed. Review the output above."
    exit 1
else
    echo "  ✓  Environment looks ready for Week 1."
    exit 0
fi
