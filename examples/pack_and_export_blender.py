"""End-to-end example: pack and export entirely within Blender.

This script runs entirely in Blender's Python environment with the
spectral_packer C++ core module installed.

Run with: ~/blender-4.5.0-linux-x64/blender --background --python examples/pack_and_export_blender.py
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_BLEND = Path("/tmp/packed_10_objects_blender.blend")
OUTPUT_RENDERS = Path("/tmp/packed_renders_blender")

print("=" * 60)
print("END-TO-END TEST: Pack and Export in Blender")
print("=" * 60)

# Step 1: Verify imports
print("\n[Step 1] Verifying imports...")
import spectral_packer
print(f"  spectral_packer version: {spectral_packer.__version__}")
print(f"  CUDA available: {spectral_packer.is_cuda_available()}")

if not spectral_packer.is_cuda_available():
    raise RuntimeError("C++ core module not available!")

from spectral_packer import BinPacker, export_to_blend
print("  BinPacker and export_to_blend imported successfully")

# Step 2: Find STL files
print("\n[Step 2] Finding STL files...")
stl_files = sorted(DATA_DIR.glob("*.stl"))[:10]
print(f"  Found {len(stl_files)} STL files:")
for f in stl_files:
    print(f"    {f.name}")

if len(stl_files) < 10:
    raise RuntimeError(f"Need 10 STL files in {DATA_DIR}, found {len(stl_files)}")

# Step 3: Pack the objects
print("\n[Step 3] Packing objects...")
packer = BinPacker(
    tray_size=(120, 120, 80),
    voxel_resolution=64,
    num_orientations=6,
)

result = packer.pack_files_for_export(stl_files)

print(f"  Placed: {result.num_placed}/{len(stl_files)}")
print(f"  Density: {result.density:.1%}")

if result.num_placed == 0:
    raise RuntimeError("No objects were placed!")

# Step 4: Export to .blend
print("\n[Step 4] Exporting to .blend...")
export_to_blend(result, OUTPUT_BLEND)
print(f"  Exported to: {OUTPUT_BLEND}")

# Step 5: Verify the scene
print("\n[Step 5] Verifying scene...")
import bpy

mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH' and not obj.name.startswith('Tray')]
print(f"  Found {len(mesh_objects)} mesh objects:")
for obj in mesh_objects:
    loc = obj.location
    print(f"    {obj.name}: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")

if len(mesh_objects) != result.num_placed:
    raise RuntimeError(f"Expected {result.num_placed} objects, found {len(mesh_objects)}")

# Step 6: Render views
print("\n[Step 6] Rendering views...")
import math
import mathutils

# Calculate scene bounds
min_coords = [float('inf')] * 3
max_coords = [float('-inf')] * 3
for obj in mesh_objects:
    for v in obj.bound_box:
        world_v = obj.matrix_world @ mathutils.Vector(v)
        for i in range(3):
            min_coords[i] = min(min_coords[i], world_v[i])
            max_coords[i] = max(max_coords[i], world_v[i])

center = [(min_coords[i] + max_coords[i]) / 2 for i in range(3)]
max_size = max(max_coords[i] - min_coords[i] for i in range(3))

# Set up rendering
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'
scene.cycles.samples = 64  # Lower for faster test
scene.render.resolution_x = 800
scene.render.resolution_y = 600

# Set up world
world = bpy.data.worlds.get('World') or bpy.data.worlds.new('World')
scene.world = world
world.use_nodes = True
nodes = world.node_tree.nodes
nodes.clear()
bg = nodes.new('ShaderNodeBackground')
bg.inputs['Color'].default_value = (0.15, 0.15, 0.2, 1)
bg.inputs['Strength'].default_value = 0.5
output = nodes.new('ShaderNodeOutputWorld')
world.node_tree.links.new(bg.outputs['Background'], output.inputs['Surface'])

# Remove existing lights
for obj in list(bpy.data.objects):
    if obj.type == 'LIGHT':
        bpy.data.objects.remove(obj)

# Add sun light
bpy.ops.object.light_add(type='SUN', location=(50, -30, 80))
sun = bpy.context.active_object
sun.data.energy = 3
sun.rotation_euler = (math.radians(45), math.radians(20), math.radians(30))

# Set up materials
colors = [
    (0.9, 0.25, 0.2, 1), (0.2, 0.75, 0.3, 1), (0.2, 0.4, 0.9, 1),
    (0.95, 0.7, 0.1, 1), (0.7, 0.2, 0.8, 1), (0.1, 0.8, 0.8, 1),
    (0.95, 0.5, 0.5, 1), (0.5, 0.8, 0.2, 1), (0.3, 0.3, 0.7, 1),
    (0.9, 0.6, 0.4, 1),
]

for i, obj in enumerate(mesh_objects):
    mat = bpy.data.materials.new(name=f"Material_{i}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = colors[i % len(colors)]
    bsdf.inputs["Roughness"].default_value = 0.4
    obj.data.materials.clear()
    obj.data.materials.append(mat)

# Hide tray
for obj in bpy.data.objects:
    if obj.name.startswith('Tray'):
        obj.hide_render = True

# Add camera
bpy.ops.object.camera_add()
camera = bpy.context.active_object
scene.camera = camera
camera.data.lens = 50

cam_distance = max_size * 2.5

def point_camera_at(camera, target):
    direction = mathutils.Vector(target) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

# Render views
OUTPUT_RENDERS.mkdir(exist_ok=True)

views = [
    ("front", (center[0], center[1] - cam_distance, center[2] + max_size * 0.3)),
    ("iso", (center[0] + cam_distance * 0.7, center[1] - cam_distance * 0.7, center[2] + cam_distance * 0.5)),
]

for view_name, cam_loc in views:
    camera.location = cam_loc
    point_camera_at(camera, center)

    output_path = OUTPUT_RENDERS / f"packed_{view_name}.png"
    scene.render.filepath = str(output_path)

    print(f"  Rendering {view_name}...")
    bpy.ops.render.render(write_still=True)
    print(f"    Saved: {output_path}")

# Final summary
print("\n" + "=" * 60)
print("TEST PASSED!")
print("=" * 60)
print(f"  Objects packed: {result.num_placed}/10")
print(f"  Density: {result.density:.1%}")
print(f"  Blend file: {OUTPUT_BLEND}")
print(f"  Renders: {OUTPUT_RENDERS}")
