"""Render cross-section animation - slice reveal from front to back.

Run with:
    ~/blender-4.5.0-linux-x64/blender renders/teaser_pack.blend --background --python examples/render_crosssection.py

Options:
    --preview     Low-res fast preview (30 frames, 25% resolution)
    --iso         Use isometric view instead of front view

Examples:
    # Preview with front view (default)
    blender ... --python examples/render_crosssection.py -- --preview

    # Preview with isometric view
    blender ... --python examples/render_crosssection.py -- --preview --iso

    # Full quality isometric
    blender ... --python examples/render_crosssection.py -- --iso
"""

import bpy
import math
import mathutils
import sys
from pathlib import Path

# Parse flags
PREVIEW = '--preview' in sys.argv
ISO_VIEW = '--iso' in sys.argv

# Configuration
if PREVIEW:
    CROSSSECTION_FRAMES = 30  # 1 second preview
    FPS = 30
    RENDER_SAMPLES = 8
    RESOLUTION_SCALE = 25
    print("*** PREVIEW MODE: Low-res, 30 frames ***")
else:
    CROSSSECTION_FRAMES = 90  # 3 seconds
    FPS = 30
    RENDER_SAMPLES = 64
    RESOLUTION_SCALE = 100

VIEW_MODE = "iso" if ISO_VIEW else "front"
print(f"*** VIEW MODE: {VIEW_MODE} ***")

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "renders"

print("=" * 70)
print("CROSS-SECTION - Front to Back Slice Reveal")
print("=" * 70)

# Get scene
scene = bpy.context.scene
scene.render.fps = FPS
scene.cycles.samples = RENDER_SAMPLES
scene.render.use_persistent_data = True
scene.render.resolution_percentage = RESOLUTION_SCALE

# Find mesh objects (excluding tray boundary and old cutter)
mesh_objects = [obj for obj in bpy.data.objects
                if obj.type == 'MESH'
                and not obj.name.startswith('Tray')
                and not obj.name.startswith('CrossSection')
                and not obj.name.startswith('Slicer')]
print(f"Found {len(mesh_objects)} mesh objects")

# Remove any existing boolean modifiers from previous runs
for obj in mesh_objects:
    for mod in list(obj.modifiers):
        if 'CrossSection' in mod.name or 'Slice' in mod.name:
            obj.modifiers.remove(mod)

# Remove old cutter objects
for obj in list(bpy.data.objects):
    if 'Cutter' in obj.name or 'Slicer' in obj.name:
        bpy.data.objects.remove(obj)

# Calculate scene bounds
min_coords = [float('inf')] * 3
max_coords = [float('-inf')] * 3
for obj in mesh_objects:
    for v in obj.bound_box:
        world_v = obj.matrix_world @ mathutils.Vector(v)
        for i in range(3):
            min_coords[i] = min(min_coords[i], world_v[i])
            max_coords[i] = max(max_coords[i], world_v[i])

center = mathutils.Vector([(min_coords[i] + max_coords[i]) / 2 for i in range(3)])
scene_size = [max_coords[i] - min_coords[i] for i in range(3)]
max_size = max(scene_size)

print(f"Scene bounds: X[{min_coords[0]:.1f}, {max_coords[0]:.1f}] Y[{min_coords[1]:.1f}, {max_coords[1]:.1f}] Z[{min_coords[2]:.1f}, {max_coords[2]:.1f}]")

# Get or create camera
camera = scene.camera
if camera is None:
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    scene.camera = camera

def point_camera_at(camera, target):
    direction = mathutils.Vector(target) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

if ISO_VIEW:
    # Isometric view - zoomed in more
    cam_distance = max_size * 1.6
    camera.location = (
        center.x + cam_distance * 0.9,
        center.y - cam_distance * 0.9,
        center.z + cam_distance * 0.6
    )
    point_camera_at(camera, center)
    camera.data.lens = 45  # Tighter lens
    camera.data.type = 'PERSP'
    print("Camera: Isometric perspective view (zoomed in)")
else:
    # Front view (looking along +Y axis) - zoomed in more
    cam_distance = max_size * 2.0
    camera.location = (center.x, min_coords[1] - cam_distance, center.z)
    camera.rotation_euler = (math.radians(90), 0, 0)
    camera.data.lens = 35
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = max(scene_size[0], scene_size[2]) * 1.05  # Tighter
    print("Camera: Front orthographic view (zoomed in)")

# Boolean box slicer - FRONT face cuts from front to back
# Box extends far in +Y direction so only front face does the cutting

bpy.ops.mesh.primitive_cube_add(size=1)
slicer = bpy.context.active_object
slicer.name = "Slicer_Box"

# Scale: wide in X and Z, very long in Y (extends away from camera)
# primitive_cube_add(size=1) creates cube with half-extent 0.5
# After scaling, actual half-extent = scale * 0.5
slicer_scale_x = scene_size[0] * 3
slicer_scale_z = scene_size[2] * 3
slicer_scale_y = scene_size[1] * 6  # Scale factor (actual half-extent = this * 0.5)

slicer.scale = (slicer_scale_x, slicer_scale_y, slicer_scale_z)

# Actual Y half-extent after scaling
slicer_half_y = slicer_scale_y * 0.5

# Hide slicer in render
slicer.hide_render = True
slicer.display_type = 'WIRE'

print(f"Slice sweep: Cutting plane moves FRONT to BACK (full → empty)")
print(f"Scene Y range: [{min_coords[1]:.1f}, {max_coords[1]:.1f}]")

# Add boolean INTERSECT to all mesh objects
print("Adding boolean INTERSECT modifiers...")
for obj in mesh_objects:
    bool_mod = obj.modifiers.new(name="Slice_Bool", type='BOOLEAN')
    bool_mod.operation = 'INTERSECT'
    bool_mod.object = slicer
    bool_mod.solver = 'FAST'

# Animate: FRONT face moves from front of scene to back
# Front face Y = slicer_center_y - slicer_half_y
# So slicer_center_y = front_face_y + slicer_half_y

scene.frame_start = 1
scene.frame_end = CROSSSECTION_FRAMES
slicer.animation_data_clear()

# Front face sweep positions (cutting plane moves front to back)
front_face_start = min_coords[1] - 5   # Frame 1: front face before scene (full visible)
front_face_end = max_coords[1] + 5     # Last frame: front face past back (nothing visible)

print(f"Front face (cutting plane) sweep: Y from {front_face_start:.1f} to {front_face_end:.1f}")

for frame in range(1, CROSSSECTION_FRAMES + 1):
    scene.frame_set(frame)

    t = (frame - 1) / (CROSSSECTION_FRAMES - 1)
    front_face_y = front_face_start + t * (front_face_end - front_face_start)

    # Center is slicer_half_y behind front face
    slicer_center_y = front_face_y + slicer_half_y

    slicer.location = (center.x, slicer_center_y, center.z)
    slicer.keyframe_insert(data_path="location", frame=frame)

# Set output directory
suffix = f"_{VIEW_MODE}"
if PREVIEW:
    suffix += "_preview"
crosssection_dir = OUTPUT_DIR / f"crosssection{suffix}"
crosssection_dir.mkdir(exist_ok=True)

scene.render.filepath = str(crosssection_dir / "frame_")
scene.render.image_settings.file_format = 'PNG'

print(f"\nRendering {CROSSSECTION_FRAMES} frames to {crosssection_dir}/")
bpy.ops.render.render(animation=True)

print("\nCross-section complete!")

# Clean up: remove boolean modifiers and slicer
print("Cleaning up...")
for obj in mesh_objects:
    for mod in list(obj.modifiers):
        if 'Slice' in mod.name:
            obj.modifiers.remove(mod)

bpy.data.objects.remove(slicer)

# Compile video
import subprocess

video_name = f"crosssection{suffix}.mp4"
crosssection_video = OUTPUT_DIR / video_name
cmd = [
    "ffmpeg", "-y", "-framerate", str(FPS),
    "-i", str(crosssection_dir / "frame_%04d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "-crf", "18",
    str(crosssection_video)
]
print(f"\nCompiling {crosssection_video}...")
subprocess.run(cmd, capture_output=True)

print(f"\nDone! Video saved to: {crosssection_video}")
