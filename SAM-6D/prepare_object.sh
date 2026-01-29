#!/bin/bash
set -euo pipefail

ROOT="/home/nikolaraicevic/Workspace/External/SAM-6D/SAM-6D"
BLENDER="$HOME/blender/blender-3.3.1-linux-x64/blender"

OBJECT_DIR="myObject/bigRedCube"
OBJECT_NAME="bigRedCube"
STL="$ROOT/Data/$OBJECT_DIR/$OBJECT_NAME.stl"
PLY="$ROOT/Data/$OBJECT_DIR/${OBJECT_NAME}.ply"

echo "Processing: $STL"
test -f "$STL" || { echo "ERROR: STL not found: $STL"; exit 1; }

mkdir -p "$ROOT/Data/$OBJECT_DIR/outputs"

"$BLENDER" --background \
  --python "$ROOT/process_mesh.py" \
  -- "$STL" "$PLY"

echo "Done."