#!/bin/bash
# ─── VLM3D Challenge — End-to-End Docker Test ─────────────────────────────────
# Run this LOCALLY on your development machine (or on SONIC with Docker access)
# BEFORE submitting to the challenge.
#
# Usage:
#   bash test_docker.sh                     # uses sample_slice.png (auto-generates a test NIfTI)
#   bash test_docker.sh /path/to/my.nii.gz  # test with a real scan
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

IMAGE_NAME="vlm3d_inference:latest"
TEST_INPUT_DIR="$(pwd)/test_input"
TEST_OUTPUT_DIR="$(pwd)/test_output"
SCAN_ARG="${1:-}"     # optional path to a real NIfTI file

echo "============================================================"
echo " VLM3D Docker End-to-End Test"
echo "============================================================"

# ── Step 1: Create test I/O directories ──────────────────────────────────────
rm -rf "$TEST_INPUT_DIR" "$TEST_OUTPUT_DIR"
mkdir -p "$TEST_INPUT_DIR" "$TEST_OUTPUT_DIR"

# ── Step 2: Prepare a test scan ──────────────────────────────────────────────
if [ -n "$SCAN_ARG" ] && [ -f "$SCAN_ARG" ]; then
    echo "[test] Using provided scan: $SCAN_ARG"
    cp "$SCAN_ARG" "$TEST_INPUT_DIR/test_scan.nii.gz"
else
    echo "[test] No scan provided — generating a synthetic NIfTI for smoke-test..."
    python3 - <<'PYEOF'
import numpy as np
import nibabel as nib
import os

# Synthetic 160x160x112 volume with random CT-like values
data = np.random.randint(-1000, 400, size=(160, 160, 112), dtype=np.int16)
affine = np.diag([1.5, 1.5, 1.5, 1.0])
img = nib.Nifti1Image(data, affine)
nib.save(img, "test_input/synthetic_test.nii.gz")
print("[test] Synthetic volume saved to test_input/synthetic_test.nii.gz")
PYEOF
fi

# ── Step 3: Build Docker image ────────────────────────────────────────────────
echo ""
echo "[test] Building Docker image '${IMAGE_NAME}'..."
echo "       (this takes 5-10 minutes the first time due to MONAI install)"
docker build -t "${IMAGE_NAME}" .
echo "[test] Docker image built successfully."

# ── Step 4: Run inference container ──────────────────────────────────────────
echo ""
echo "[test] Running inference container..."
docker run --rm \
    --gpus all \
    -v "${TEST_INPUT_DIR}:/input:ro" \
    -v "${TEST_OUTPUT_DIR}:/output" \
    "${IMAGE_NAME}"

# ── Step 5: Validate output ────────────────────────────────────────────────────
echo ""
echo "[test] Checking output files..."
OUTPUT_FILES=$(ls "$TEST_OUTPUT_DIR"/*.json 2>/dev/null || true)

if [ -z "$OUTPUT_FILES" ]; then
    echo "❌ FAIL: No JSON output files found in $TEST_OUTPUT_DIR"
    exit 1
fi

for f in $OUTPUT_FILES; do
    echo "   Output file: $f"
    BINARY=$(python3 -c "import json; d=json.load(open('$f')); print(d['binary_prediction'])")
    LENGTH=$(python3 -c "import json; d=json.load(open('$f')); print(len(d['binary_prediction']))")
    echo "   Binary vector (length=$LENGTH): $BINARY"
    if [ "$LENGTH" != "18" ]; then
        echo "❌ FAIL: Expected 18-length binary vector, got $LENGTH"
        exit 1
    fi
done

echo ""
echo "============================================================"
echo " ✅ PASS — Container ran successfully."
echo "    Output dir: $TEST_OUTPUT_DIR"
echo "============================================================"
