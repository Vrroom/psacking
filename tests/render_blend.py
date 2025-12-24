"""Render the packed .blend file from multiple views."""

import bpy
import math
import mathutils
from pathlib import Path

# Load the blend file
blend_file = "/tmp/packed_10_objects.blend"
bpy.ops.wm.open_mainfile(filepath=blend_file)

print("=" * 60)
print("Rendering packed objects")
print("=" * 60)

# List objects in scene
mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH' and not obj.name.startswith('Tray')]
print(f"\nFound {len(mesh_objects)} mesh objects:")
for obj in mesh_objects:
    print(f"  {obj.name}: location={tuple(round(x, 2) for x in obj.location)}")

# Calculate bounding box of all objects
min_coords = [float('inf')] * 3
max_coords = [float('-inf')] * 3
for obj in mesh_objects:
    for v in obj.bound_box:
        world_v = obj.matrix_world @ mathutils.Vector(v)
        for i in range(3):
            min_coords[i] = min(min_coords[i], world_v[i])
            max_coords[i] = max(max_coords[i], world_v[i])

center = [(min_coords[i] + max_coords[i]) / 2 for i in range(3)]
size = [max_coords[i] - min_coords[i] for i in range(3)]
max_size = max(size)

print(f"\nScene bounds: {min_coords} to {max_coords}")
print(f"Scene center: {center}")
print(f"Scene size: {size}, max={max_size:.1f}")

# Set up rendering
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'
scene.cycles.samples = 128
scene.render.resolution_x = 1024
scene.render.resolution_y = 768
scene.render.film_transparent = False

# Set world background to gradient
world = bpy.data.worlds.get('World') or bpy.data.worlds.new('World')
scene.world = world
world.use_nodes = True
nodes = world.node_tree.nodes
nodes.clear()
bg = nodes.new('ShaderNodeBackground')
bg.inputs['Color'].default_value = (0.15, 0.15, 0.2, 1)  # Dark blue-gray
bg.inputs['Strength'].default_value = 0.5
output = nodes.new('ShaderNodeOutputWorld')
world.node_tree.links.new(bg.outputs['Background'], output.inputs['Surface'])

# Remove existing lights
for obj in bpy.data.objects:
    if obj.type == 'LIGHT':
        bpy.data.objects.remove(obj)

# Add key light (sun)
bpy.ops.object.light_add(type='SUN', location=(50, -30, 80))
sun = bpy.context.active_object
sun.name = "Key_Light"
sun.data.energy = 3
sun.data.angle = math.radians(5)
sun.rotation_euler = (math.radians(45), math.radians(20), math.radians(30))

# Add fill light (area)
bpy.ops.object.light_add(type='AREA', location=(-40, -40, 40))
fill = bpy.context.active_object
fill.name = "Fill_Light"
fill.data.energy = 200
fill.data.size = 30
fill.rotation_euler = (math.radians(60), 0, math.radians(-45))

# Add rim light
bpy.ops.object.light_add(type='AREA', location=(0, 60, 30))
rim = bpy.context.active_object
rim.name = "Rim_Light"
rim.data.energy = 150
rim.data.size = 20
rim.rotation_euler = (math.radians(120), 0, math.radians(180))

# Set up materials for the meshes (give them distinct colors)
colors = [
    (0.9, 0.25, 0.2, 1),   # Red
    (0.2, 0.75, 0.3, 1),   # Green
    (0.2, 0.4, 0.9, 1),    # Blue
    (0.95, 0.7, 0.1, 1),   # Yellow/Orange
    (0.7, 0.2, 0.8, 1),    # Purple
    (0.1, 0.8, 0.8, 1),    # Cyan
    (0.95, 0.5, 0.5, 1),   # Pink
    (0.5, 0.8, 0.2, 1),    # Lime
    (0.3, 0.3, 0.7, 1),    # Indigo
    (0.9, 0.6, 0.4, 1),    # Peach
]

for i, obj in enumerate(mesh_objects):
    mat = bpy.data.materials.new(name=f"Material_{i}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = colors[i % len(colors)]
    bsdf.inputs["Metallic"].default_value = 0.1
    bsdf.inputs["Roughness"].default_value = 0.4
    bsdf.inputs["Specular IOR Level"].default_value = 0.5
    obj.data.materials.clear()
    obj.data.materials.append(mat)

# Hide tray boundary from render
for obj in bpy.data.objects:
    if obj.name.startswith('Tray'):
        obj.hide_render = True

# Add camera
bpy.ops.object.camera_add(location=(0, 0, 0))
camera = bpy.context.active_object
camera.name = "RenderCamera"
scene.camera = camera
camera.data.lens = 50

# Camera distance based on scene size
cam_distance = max_size * 2.5

# Camera views
camera_views = [
    ("front", (center[0], center[1] - cam_distance, center[2] + max_size * 0.3)),
    ("top", (center[0], center[1], center[2] + cam_distance)),
    ("iso", (center[0] + cam_distance * 0.7, center[1] - cam_distance * 0.7, center[2] + cam_distance * 0.5)),
]

def point_camera_at(camera, target):
    direction = mathutils.Vector(target) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

# Render each view
output_dir = Path("/tmp/packed_renders")
output_dir.mkdir(exist_ok=True)

for view_name, cam_loc in camera_views:
    print(f"\nRendering {view_name} view...")

    camera.location = cam_loc
    point_camera_at(camera, center)

    output_path = output_dir / f"packed_{view_name}.png"
    scene.render.filepath = str(output_path)

    bpy.ops.render.render(write_still=True)
    print(f"  Saved: {output_path}")

print("\n" + "=" * 60)
print(f"Renders saved to {output_dir}")
print("=" * 60)
