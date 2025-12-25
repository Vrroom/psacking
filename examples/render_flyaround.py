"""Render 360° flyaround of packed tray.

Run with:
    ~/blender-4.5.0-linux-x64/blender renders/teaser_pack.blend --background --python examples/render_flyaround.py

Options:
    --preview     Low-res fast preview (30 frames, 25% resolution)

Examples:
    # Preview
    blender ... --python examples/render_flyaround.py -- --preview

    # Full quality
    blender ... --python examples/render_flyaround.py
"""

import bpy
import math
import mathutils
import sys
from pathlib import Path

# Parse flags
PREVIEW = '--preview' in sys.argv

# Configuration
if PREVIEW:
    FLYAROUND_FRAMES = 30  # 1 second preview
    FPS = 30
    RENDER_SAMPLES = 8
    RESOLUTION_SCALE = 25
    print("*** PREVIEW MODE: Low-res, 30 frames ***")
else:
    FLYAROUND_FRAMES = 120  # 4 seconds
    FPS = 30
    RENDER_SAMPLES = 64
    RESOLUTION_SCALE = 100

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "renders"

print("=" * 70)
print("360° FLYAROUND")
print("=" * 70)

# Get scene
scene = bpy.context.scene
scene.render.fps = FPS
scene.cycles.samples = RENDER_SAMPLES
scene.render.use_persistent_data = True
scene.render.resolution_percentage = RESOLUTION_SCALE

# Find mesh objects (excluding tray boundary)
mesh_objects = [obj for obj in bpy.data.objects
                if obj.type == 'MESH' and not obj.name.startswith('Tray')]
print(f"Found {len(mesh_objects)} mesh objects")

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

print(f"Scene center: {center}")
print(f"Scene size: {scene_size}")

# Get or create camera
camera = scene.camera
if camera is None:
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    scene.camera = camera

# Camera settings - balanced zoom
camera.data.lens = 42
camera.data.type = 'PERSP'

def point_camera_at(camera, target):
    direction = mathutils.Vector(target) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

# Camera orbit parameters - balanced
orbit_radius = max_size * 1.85
orbit_height = center.z + max_size * 0.57

print(f"Orbit radius: {orbit_radius:.1f}")
print(f"Orbit height: {orbit_height:.1f}")

# Clear existing animation data
camera.animation_data_clear()

# Create keyframes for circular orbit
scene.frame_start = 1
scene.frame_end = FLYAROUND_FRAMES

for frame in range(1, FLYAROUND_FRAMES + 1):
    scene.frame_set(frame)

    # Calculate angle (full 360°)
    angle = (frame - 1) / FLYAROUND_FRAMES * 2 * math.pi

    # Camera position on circle
    cam_x = center.x + orbit_radius * math.cos(angle)
    cam_y = center.y + orbit_radius * math.sin(angle)
    cam_z = orbit_height

    camera.location = (cam_x, cam_y, cam_z)
    point_camera_at(camera, center)

    # Insert keyframes
    camera.keyframe_insert(data_path="location", frame=frame)
    camera.keyframe_insert(data_path="rotation_euler", frame=frame)

# Set output
if PREVIEW:
    flyaround_dir = OUTPUT_DIR / "flyaround_preview"
else:
    flyaround_dir = OUTPUT_DIR / "flyaround"
flyaround_dir.mkdir(exist_ok=True)

scene.render.filepath = str(flyaround_dir / "frame_")
scene.render.image_settings.file_format = 'PNG'

print(f"\nRendering {FLYAROUND_FRAMES} frames to {flyaround_dir}/")
bpy.ops.render.render(animation=True)

print("\nFlyaround complete!")

# Compile video with ffmpeg
import subprocess

if PREVIEW:
    flyaround_video = OUTPUT_DIR / "flyaround_preview.mp4"
else:
    flyaround_video = OUTPUT_DIR / "flyaround_360.mp4"

cmd = [
    "ffmpeg", "-y", "-framerate", str(FPS),
    "-i", str(flyaround_dir / "frame_%04d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "-crf", "18",
    str(flyaround_video)
]
print(f"\nCompiling {flyaround_video}...")
subprocess.run(cmd, capture_output=True)

print(f"\nDone! Video saved to: {flyaround_video}")
