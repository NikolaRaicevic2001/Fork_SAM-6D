#!/bin/bash
set -euo pipefail

ROOT="/home/nikolaraicevic/Workspace/External/SAM-6D/SAM-6D"
BLENDER="$HOME/blender/blender-3.3.1-linux-x64/blender"

OBJECT_DIR="myObject/bigRedCube"
OBJECT_NAME="bigRedCube"

STL="$ROOT/Data/$OBJECT_DIR/$OBJECT_NAME.stl"
RAW_PLY="$ROOT/Data/$OBJECT_DIR/${OBJECT_NAME}_raw.ply"
SAM6D_PLY="$ROOT/Data/$OBJECT_DIR/${OBJECT_NAME}_sam6d.ply"

echo "Processing: $STL"
test -f "$STL" || { echo "ERROR: STL not found: $STL"; exit 1; }

mkdir -p "$ROOT/Data/$OBJECT_DIR/outputs"

"$BLENDER" --background \
  --python "$ROOT/process_mesh.py" \
  -- "$STL" "$RAW_PLY"

python - << EOF
import open3d as o3d
mesh = o3d.io.read_triangle_mesh("$RAW_PLY")
mesh.compute_vertex_normals()
pcd = mesh.sample_points_uniformly(number_of_points=60000)
o3d.io.write_point_cloud("$SAM6D_PLY", pcd)
print("Wrote:", "$SAM6D_PLY")
EOF

echo "Done."