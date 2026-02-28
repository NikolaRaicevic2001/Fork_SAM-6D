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

bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
obj.location = (0.0, 0.0, 0.0)
bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

scale = 1.0
obj.scale = (scale, scale, scale)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

if obj.type != 'MESH':
    raise RuntimeError(f"Imported object is not a mesh: {obj.type}")

print("Dimensions (Blender units):", obj.dimensions)

# -------------------------------------------------
# Geometry processing 
# -------------------------------------------------
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.subdivide(number_cuts=3)     
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
        color_layer.data[loop_idx].color = (1.0, 0.502, 0.0, 1.0)         # RGBA: Organge
        # color_layer.data[loop_idx].color = (0.0, 0.0, 1.0, 1.0)           # RGBA: Blue
        # color_layer.data[loop_idx].color = (0.118, 0.165, 0.337, 1.0)     # RGBA: Dark Blue
        # color_layer.data[loop_idx].color = (0.0, 0.7, 0.298, 1.0)         # RGBA: Green

# -------------------------------------------------
# Export PLY with vertex colors
# -------------------------------------------------
bpy.ops.export_mesh.ply(
    filepath=output_path,
    use_ascii=True,
    use_normals=True,
    use_colors=True    
)

print("STL -> vertex-colored PLY written to:", output_path)
