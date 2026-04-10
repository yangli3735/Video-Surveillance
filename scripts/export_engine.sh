#!/usr/bin/env bash
# =============================================================================
#  export_engine.sh — Week 1 TensorRT Engine Export
#  Platform: NVIDIA Jetson Orin Nano · JetPack 6.1 · Docker
#
#  Exports YOLOv8n from PyTorch (.pt) to TensorRT (.engine) using the
#  Ultralytics CLI.  The resulting engine is optimized for the Orin Nano GPU
#  and uses FP16 precision for best throughput.
#
#  Run inside the Docker container:
#    bash scripts/export_engine.sh
# =============================================================================

set -euo pipefail

MODEL_DIR="data/models"
PT_MODEL="${MODEL_DIR}/yolov8n.pt"
ENGINE_FILE="${MODEL_DIR}/yolov8n.engine"
IMGSZ=640

echo "============================================================"
echo "  YOLOv8n → TensorRT Engine Export"
echo "============================================================"
echo ""

# ── 1. Ensure model directory exists ───────────────────────────────────────
mkdir -p "$MODEL_DIR"

# ── 2. Download YOLOv8n.pt if not present ──────────────────────────────────
if [ -f "$PT_MODEL" ]; then
    echo "[INFO] Found existing model: $PT_MODEL"
else
    echo "[INFO] Downloading YOLOv8n weights…"
    python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')   # downloads to current dir
import shutil, os
src = 'yolov8n.pt'
dst = '${PT_MODEL}'
if os.path.exists(src) and src != dst:
    shutil.move(src, dst)
print(f'Saved to {dst}')
"
    echo "[INFO] Download complete: $PT_MODEL"
fi

echo ""

# ── 3. Export to TensorRT engine ───────────────────────────────────────────
#  - format=engine  → TensorRT
#  - half=True      → FP16 precision (Orin Nano has FP16 tensor cores)
#  - device=0       → use GPU 0
#  - imgsz=$IMGSZ   → input resolution
#
#  First export takes several minutes while TensorRT builds and optimizes
#  the engine.  The .engine file is device-specific and NOT portable across
#  different GPU architectures.
echo "[INFO] Starting TensorRT export (this may take a few minutes)…"
echo ""

python3 -c "
from ultralytics import YOLO

model = YOLO('${PT_MODEL}')
model.export(
    format='engine',
    half=True,
    device=0,
    imgsz=${IMGSZ},
)
print()
print('Export command completed.')
"

echo ""

# ── 4. Verify output ──────────────────────────────────────────────────────
# ultralytics places the .engine next to the .pt by default
EXPECTED="${MODEL_DIR}/yolov8n.engine"

if [ -f "$EXPECTED" ]; then
    SIZE=$(du -h "$EXPECTED" | cut -f1)
    echo "[PASS] TensorRT engine created: $EXPECTED ($SIZE)"
    echo ""
    echo "  To use the engine, set 'use_tensorrt: true' in config/settings.yaml"
    echo "  or run:  python3 scripts/yolo_test.py"
    echo "  (after updating the config)."
else
    # ultralytics might save with a slightly different name/path
    FOUND=$(find "$MODEL_DIR" -name "*.engine" -type f 2>/dev/null | head -1)
    if [ -n "$FOUND" ]; then
        SIZE=$(du -h "$FOUND" | cut -f1)
        echo "[PASS] TensorRT engine created at unexpected path: $FOUND ($SIZE)"
        echo "  Update engine_path in config/settings.yaml if needed."
    else
        echo "[FAIL] No .engine file found in $MODEL_DIR"
        echo "  Check the export output above for errors."
        exit 1
    fi
fi

echo ""
echo "============================================================"
echo "  Export complete."
echo "============================================================"
