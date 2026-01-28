import bpy
import sys

# -------------------------------------------------
# Parse arguments
# -------------------------------------------------
argv = sys.argv
argv = argv[argv.index("--") + 1:]
input_path = argv[0]
output_path = argv[1]

# -------------------------------------------------
# Reset Blender scene
# -------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)

# -------------------------------------------------
# Import STL
# -------------------------------------------------
bpy.ops.import_mesh.stl(filepath=input_path)

obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

if obj.type != 'MESH':
    raise RuntimeError(f"Imported object is not a mesh: {obj.type}")

# -------------------------------------------------
# Units: mm -> meters
# -------------------------------------------------
obj.scale = (0.001, 0.001, 0.001)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# -------------------------------------------------
# Geometry processing (same as your working version)
# -------------------------------------------------
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.subdivide(number_cuts=10)     # keep if this worked for you
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')

# -------------------------------------------------
# ADD SOLID COLOR USING VERTEX COLORS (CRITICAL)
# -------------------------------------------------
mesh = obj.data

# Create vertex color layer if missing
if not mesh.vertex_colors:
    mesh.vertex_colors.new(name="Col")

color_layer = mesh.vertex_colors.active

# Paint entire mesh solid red
for poly in mesh.polygons:
    for loop_idx in poly.loop_indices:
        color_layer.data[loop_idx].color = (1.0, 0.0, 0.0, 1.0)  # RGBA

# -------------------------------------------------
# Export PLY with vertex colors
# -------------------------------------------------
bpy.ops.export_mesh.ply(
    filepath=output_path,
    use_ascii=True,
    use_normals=True,
    use_colors=True    # THIS is what makes segmentation work
)

print("STL -> vertex-colored PLY written to:", output_path)
