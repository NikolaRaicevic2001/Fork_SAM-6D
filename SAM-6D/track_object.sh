#!/bin/bash
# Quick-start script for RealSense D455 object tracking

# ============================================
# Configuration - EDIT THESE PATHS
# ============================================
OBJECT_NAME="masterChef"              # Object directory name under Data/myObject/
CAD_FILE="obj_000001.ply"             # CAD model filename
SEGMENTOR_MODEL="fastsam"             # 'sam' or 'fastsam'
DET_SCORE_THRESH=0.2                  # Detection score threshold

# ============================================
# Set Paths
# ============================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OBJECT_DIR="${SCRIPT_DIR}/Data/myObject/${OBJECT_NAME}"
CAD_PATH="${OBJECT_DIR}/${CAD_FILE}"

# ============================================
# Validation
# ============================================
if [ ! -d "$OBJECT_DIR" ]; then
    echo "Error: Object directory not found: $OBJECT_DIR"
    exit 1
fi

if [ ! -f "$CAD_PATH" ]; then
    echo "Error: CAD file not found: $CAD_PATH"
    exit 1
fi

TEMPLATE_DIR="${OBJECT_DIR}/outputs/templates"
if [ ! -d "$TEMPLATE_DIR" ] || [ -z "$(ls -A ${TEMPLATE_DIR}/rgb_*.png 2>/dev/null)" ]; then
    echo "Warning: Templates not found in ${TEMPLATE_DIR}"
    echo "Please render templates first:"
    echo "  cd Render && blenderproc run render_custom_templates.py --output_dir ${OBJECT_DIR}/outputs --cad_path ${CAD_PATH}"
    exit 1
fi

# ============================================
# Run Tracking
# ============================================
echo "========================================="
echo "SAM-6D RealSense D455 Object Tracking"
echo "========================================="
echo "Object: $OBJECT_NAME"
echo "CAD Model: $CAD_PATH"
echo "Segmentor: $SEGMENTOR_MODEL"
echo "Detection Threshold: $DET_SCORE_THRESH"
echo "========================================="
echo ""

cd "$SCRIPT_DIR"
python realsense_tracker.py \
    --object_dir "$OBJECT_DIR" \
    --cad_file "$CAD_FILE" \
    --segmentor_model "$SEGMENTOR_MODEL" \
    --det_score_thresh "$DET_SCORE_THRESH"
