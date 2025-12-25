"""Generate GitHub README teaser image from Thingi10K packing.

This script creates a polished visualization of 3D bin packing for the README:
1. Filters Thingi10K objects to similar physical sizes (20-50mm for dense packing)
2. Uses pitch=1.0 so 1 voxel = 1mm (consistent physical scale)
3. Packs into a 240x123x100mm tray (matching paper's benchmark)
4. Keeps packing until no more objects fit (consecutive failures)
5. Exports to Blender and renders high-quality images

Run with:
    ~/blender-4.5.0-linux-x64/blender --background --python examples/github_teaser.py
"""

import random
import math
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import trimesh

# Configuration
SIZE_RANGE = (20, 50)  # Object size range in mm
MAX_CONSECUTIVE_FAILURES = 30  # Stop after this many failures in a row
TRAY_SIZE_MM = (240, 123, 100)  # Tray size in mm (= voxels with pitch=1.0)
VOXEL_RESOLUTION = 128  # Max voxels per object dimension
PITCH = 1.0  # 1 voxel = 1mm
NUM_ORIENTATIONS = 24  # Full rotation set for best packing
SEED = 42  # Random seed for reproducibility

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
THINGI_DIR = Path("/home/ubuntu/general-purpose/cpsc424_final_project/thingi")
OUTPUT_BLEND = PROJECT_ROOT / "renders" / "teaser_pack.blend"
OUTPUT_RENDERS = PROJECT_ROOT / "renders"

print("=" * 70)
print("GITHUB README TEASER - Spectral 3D Bin Packing")
print("=" * 70)
print(f"  Tray size: {TRAY_SIZE_MM[0]}x{TRAY_SIZE_MM[1]}x{TRAY_SIZE_MM[2]} mm")
print(f"  Pitch: {PITCH} (1 voxel = 1mm)")

# Step 1: Import and verify
print("\n[Step 1] Importing spectral_packer...")
import spectral_packer
from spectral_packer import BinPacker, export_to_blend
from spectral_packer.voxelizer import Voxelizer, VoxelizationInfo
from spectral_packer.packer import PackingResult, MeshPlacementInfo, PlacementInfo
print(f"  Version: {spectral_packer.__version__}")
print(f"  CUDA available: {spectral_packer.is_cuda_available()}")

# Step 2: Collect and filter STL files by physical size
print(f"\n[Step 2] Scanning Thingi10K for objects {SIZE_RANGE[0]}-{SIZE_RANGE[1]}mm...")
all_stl_files = sorted(THINGI_DIR.glob("*.stl"))
print(f"  Total STL files: {len(all_stl_files)}")

candidates = []
scan_start = time.time()
for i, stl_file in enumerate(all_stl_files):
    if (i + 1) % 500 == 0:
        print(f"  Scanned {i + 1}/{len(all_stl_files)}...")
    try:
        mesh = trimesh.load(str(stl_file), force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        max_extent = mesh.extents.max()
        if SIZE_RANGE[0] <= max_extent <= SIZE_RANGE[1]:
            candidates.append((stl_file, max_extent))
    except Exception:
        pass  # Skip problematic files

scan_time = time.time() - scan_start
print(f"  Found {len(candidates)} objects in size range ({scan_time:.1f}s)")

if len(candidates) == 0:
    raise RuntimeError(f"No objects found in {SIZE_RANGE[0]}-{SIZE_RANGE[1]}mm range!")

# Show size distribution
sizes = [s for _, s in candidates]
print(f"  Size range: {min(sizes):.1f}mm - {max(sizes):.1f}mm")
print(f"  Mean size: {sum(sizes)/len(sizes):.1f}mm")

# Step 3: Pre-voxelize all candidates with fixed pitch
print(f"\n[Step 3] Pre-voxelizing {len(candidates)} candidates with pitch={PITCH}...")
voxelizer = Voxelizer(resolution=VOXEL_RESOLUTION, pitch=PITCH)

voxelized_candidates = []
vox_start = time.time()
for i, (stl_file, size) in enumerate(candidates):
    if (i + 1) % 100 == 0:
        print(f"  Voxelized {i + 1}/{len(candidates)}...")
    try:
        voxel, info = voxelizer.voxelize_file_with_info(stl_file)
        voxelized_candidates.append((stl_file, voxel, info, size))
    except Exception as e:
        pass  # Skip problematic files

vox_time = time.time() - vox_start
print(f"  Successfully voxelized {len(voxelized_candidates)} objects ({vox_time:.1f}s)")

# Step 4: Iterative packing until tray is full
print(f"\n[Step 4] Packing into {TRAY_SIZE_MM} mm tray...")
print(f"  Will stop after {MAX_CONSECUTIVE_FAILURES} consecutive failures")

from spectral_packer import (
    fft_search_placement_with_cache,
    place_in_tray,
    calculate_distance,
)
from spectral_packer.rotations import get_orientations, make_contiguous

random.seed(SEED)

# Initialize empty tray
tray = np.zeros(TRAY_SIZE_MM, dtype=np.int32)
generation = 0
object_id = 0

placed_items: List[Tuple[Path, VoxelizationInfo, Tuple[int, int, int], int]] = []
consecutive_failures = 0
total_attempts = 0

pack_start = time.time()

while consecutive_failures < MAX_CONSECUTIVE_FAILURES:
    # Pick a random candidate
    stl_file, voxel, info, size = random.choice(voxelized_candidates)
    total_attempts += 1

    # Try all orientations and find best placement
    best_position = None
    best_score = float('inf')
    best_orientation_idx = 0
    best_rotated_item = voxel

    orientations = get_orientations(voxel, NUM_ORIENTATIONS)
    tray_distance = calculate_distance(tray)

    for orient_idx, rotated_item in enumerate(orientations):
        rotated_item = make_contiguous(rotated_item.astype(np.int32))

        # Skip if item doesn't fit in tray
        if any(rotated_item.shape[i] > TRAY_SIZE_MM[i] for i in range(3)):
            continue

        position, found, score = fft_search_placement_with_cache(
            rotated_item, tray, tray_distance, generation
        )

        if found and score < best_score:
            best_position = position
            best_score = score
            best_orientation_idx = orient_idx
            best_rotated_item = rotated_item

    if best_position is not None:
        # Place the item
        object_id += 1
        tray = place_in_tray(best_rotated_item, tray, best_position, object_id)
        generation += 1

        placed_items.append((stl_file, info, best_position, best_orientation_idx))
        consecutive_failures = 0

        # Progress update every 10 items
        if len(placed_items) % 10 == 0:
            density = np.sum(tray > 0) / tray.size
            print(f"  Placed {len(placed_items)} objects, density: {density:.1%}")
    else:
        consecutive_failures += 1

pack_time = time.time() - pack_start

# Calculate final stats
total_volume = np.sum(tray > 0)
tray_volume = np.prod(TRAY_SIZE_MM)
density = total_volume / tray_volume

print(f"\n  Packing complete!")
print(f"  Placed: {len(placed_items)} objects")
print(f"  Attempts: {total_attempts}")
print(f"  Density: {density:.1%}")
print(f"  Pack time: {pack_time:.1f}s ({pack_time/max(1, len(placed_items)):.2f}s per placed object)")

if len(placed_items) == 0:
    raise RuntimeError("No objects were placed!")

# Step 5: Build PackingResult for export
print("\n[Step 5] Building export data...")

# Create placement infos
placements = []
mesh_placements = []

for i, (stl_file, info, position, orient_idx) in enumerate(placed_items):
    placement = PlacementInfo(
        item_index=i,
        position=position,
        score=0.0,
        success=True,
        volume=int(np.sum(tray == (i + 1))),
        orientation_index=orient_idx,
    )
    placements.append(placement)

    mesh_placement = MeshPlacementInfo(
        mesh_path=stl_file,
        voxel_info=info,
        voxel_position=position,
        orientation_index=orient_idx,
        success=True,
    )
    mesh_placements.append(mesh_placement)

# Create PackingResult
result = PackingResult(
    tray=tray,
    placements=placements,
    num_placed=len(placed_items),
    num_failed=total_attempts - len(placed_items),
    total_volume=total_volume,
    density=density,
)
result.mesh_placements = mesh_placements

# Step 6: Export to Blender
print("\n[Step 6] Exporting to .blend file...")
OUTPUT_RENDERS.mkdir(exist_ok=True)
export_to_blend(result, OUTPUT_BLEND)
print(f"  Exported to: {OUTPUT_BLEND}")

# Step 7: Set up Blender scene and render
print("\n[Step 7] Setting up Blender scene...")
import bpy
import mathutils

# Get mesh objects (excluding tray boundary)
mesh_objects = [obj for obj in bpy.data.objects
                if obj.type == 'MESH' and not obj.name.startswith('Tray')]
print(f"  Found {len(mesh_objects)} mesh objects")

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
scene_size = [max_coords[i] - min_coords[i] for i in range(3)]
max_size = max(scene_size)

print(f"  Scene bounds: {scene_size[0]:.1f} x {scene_size[1]:.1f} x {scene_size[2]:.1f}")
print(f"  Scene center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")

# Set up high-quality rendering
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'
scene.cycles.samples = 256  # High quality
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080

# Set up world background (soft gradient)
world = bpy.data.worlds.get('World') or bpy.data.worlds.new('World')
scene.world = world
world.use_nodes = True
nodes = world.node_tree.nodes
nodes.clear()
bg = nodes.new('ShaderNodeBackground')
bg.inputs['Color'].default_value = (0.95, 0.95, 0.97, 1)  # Very light gray
bg.inputs['Strength'].default_value = 1.0
output = nodes.new('ShaderNodeOutputWorld')
world.node_tree.links.new(bg.outputs['Background'], output.inputs['Surface'])

# Remove existing lights
for obj in list(bpy.data.objects):
    if obj.type == 'LIGHT':
        bpy.data.objects.remove(obj)

# Add three-point lighting
# Key light (sun)
bpy.ops.object.light_add(type='SUN', location=(100, -50, 150))
sun = bpy.context.active_object
sun.name = "Key_Light"
sun.data.energy = 3
sun.rotation_euler = (math.radians(45), math.radians(15), math.radians(30))

# Fill light (area)
bpy.ops.object.light_add(type='AREA', location=(-80, 80, 100))
fill = bpy.context.active_object
fill.name = "Fill_Light"
fill.data.energy = 400
fill.data.size = 30
fill.rotation_euler = (math.radians(60), math.radians(-20), math.radians(-45))

# Rim light
bpy.ops.object.light_add(type='AREA', location=(50, 100, 50))
rim = bpy.context.active_object
rim.name = "Rim_Light"
rim.data.energy = 200
rim.data.size = 20
rim.rotation_euler = (math.radians(70), math.radians(30), math.radians(120))

# Assign materials with varied colors (vibrant palette)
print("\n[Step 8] Assigning materials...")

colors = [
    (0.90, 0.25, 0.20, 1),  # Red
    (0.20, 0.75, 0.35, 1),  # Green
    (0.20, 0.45, 0.90, 1),  # Blue
    (0.95, 0.70, 0.10, 1),  # Yellow/Orange
    (0.75, 0.20, 0.80, 1),  # Purple
    (0.15, 0.80, 0.85, 1),  # Cyan
    (0.95, 0.50, 0.25, 1),  # Orange
    (0.50, 0.80, 0.25, 1),  # Lime
    (0.45, 0.35, 0.75, 1),  # Indigo
    (0.90, 0.45, 0.55, 1),  # Pink
    (0.25, 0.60, 0.50, 1),  # Teal
    (0.80, 0.65, 0.45, 1),  # Tan
]

for i, obj in enumerate(mesh_objects):
    mat = bpy.data.materials.new(name=f"Material_{i:03d}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = colors[i % len(colors)]
    bsdf.inputs["Roughness"].default_value = 0.3
    bsdf.inputs["Metallic"].default_value = 0.0
    obj.data.materials.clear()
    obj.data.materials.append(mat)

# Hide tray boundary for cleaner render
for obj in bpy.data.objects:
    if obj.name.startswith('Tray'):
        obj.hide_render = True

# Add camera
print("\n[Step 9] Setting up camera...")
bpy.ops.object.camera_add()
camera = bpy.context.active_object
scene.camera = camera
camera.data.lens = 50

cam_distance = max_size * 1.8

def point_camera_at(camera, target):
    direction = mathutils.Vector(target) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

# Render views
print("\n[Step 10] Rendering high-quality views...")
OUTPUT_RENDERS.mkdir(exist_ok=True)

views = [
    ("iso", (center[0] + cam_distance * 0.9, center[1] - cam_distance * 0.9, center[2] + cam_distance * 0.7)),
    ("top", (center[0], center[1], center[2] + cam_distance * 1.5)),
]

for view_name, cam_loc in views:
    camera.location = cam_loc
    point_camera_at(camera, center)

    output_path = OUTPUT_RENDERS / f"teaser_{view_name}.png"
    scene.render.filepath = str(output_path)

    print(f"  Rendering {view_name}...")
    render_start = time.time()
    bpy.ops.render.render(write_still=True)
    render_time = time.time() - render_start
    print(f"    Saved: {output_path} ({render_time:.1f}s)")

# Save final blend file with isometric camera
camera.location = views[0][1]
point_camera_at(camera, center)
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND))

# Summary
print("\n" + "=" * 70)
print("TEASER GENERATION COMPLETE!")
print("=" * 70)
print(f"  Objects packed: {len(placed_items)}")
print(f"  Total attempts: {total_attempts}")
print(f"  Packing density: {density:.1%}")
print(f"  Pack time: {pack_time:.1f}s")
print(f"\nOutput files:")
print(f"  Blend file: {OUTPUT_BLEND}")
print(f"  Renders:")
for view_name, _ in views:
    print(f"    - {OUTPUT_RENDERS / f'teaser_{view_name}.png'}")
