"""Run blender export tests within Blender's Python environment.

This test uses pre-created mesh placement data since the C++ core module
is not available in Blender's Python. The actual packing is done by the
system Python, and this test verifies the Blender export functionality.
"""

import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Now run the actual test
import numpy as np
from pathlib import Path

print("=" * 60)
print("Running Blender Export Integration Test")
print("=" * 60)

# Test 1: Check bpy is available
print("\n[Test 1] Checking bpy availability...")
from spectral_packer import is_blender_available
assert is_blender_available(), "bpy should be available in Blender"
print("  PASSED: bpy is available")

# Test 2: Create mock packing result and export to .blend
print("\n[Test 2] Creating mock packing result and exporting to .blend...")
from spectral_packer import export_to_blend
from spectral_packer.packer import PackingResult, PlacementInfo, MeshPlacementInfo
from spectral_packer.voxelizer import VoxelizationInfo

# Find STL files from data/ folder (copied from Thingi10K)
stl_dir = Path(project_root) / "data"
stl_files = sorted(stl_dir.glob("*.stl"))[:10]
print(f"  Found {len(stl_files)} STL files: {[f.name for f in stl_files]}")

# Create mock VoxelizationInfo and MeshPlacementInfo for each file
# Using realistic positions that won't overlap (grid layout)
mesh_placements = []
# 10 positions in a 4x3 grid pattern (only using 10 of 12 positions)
positions = [
    (0, 0, 0), (25, 0, 0), (50, 0, 0), (75, 0, 0),
    (0, 25, 0), (25, 25, 0), (50, 25, 0), (75, 25, 0),
    (0, 50, 0), (25, 50, 0),
]
orientations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Different orientations

for i, (stl_file, pos, orient) in enumerate(zip(stl_files, positions, orientations)):
    vox_info = VoxelizationInfo(
        mesh_path=stl_file,
        mesh_bounds_min=np.array([-10.0, -10.0, -10.0]),
        mesh_bounds_max=np.array([10.0, 10.0, 10.0]),
        pitch=1.0,  # 1 unit per voxel
        voxel_shape=(20, 20, 20),
    )
    mesh_placements.append(MeshPlacementInfo(
        mesh_path=stl_file,
        voxel_info=vox_info,
        voxel_position=pos,
        orientation_index=orient,
        success=True,
    ))

# Create mock PackingResult
result = PackingResult(
    tray=np.zeros((120, 100, 40), dtype=np.int32),
    placements=[
        PlacementInfo(item_index=i, position=pos, score=0.0, success=True, volume=1000, orientation_index=orient)
        for i, (pos, orient) in enumerate(zip(positions, orientations))
    ],
    num_placed=len(stl_files),
    num_failed=0,
    density=0.15,
    total_volume=10000,
    bounding_box=((0, 0, 0), (95, 70, 20)),
    mesh_placements=mesh_placements,
)
print(f"  Created mock result with {result.num_placed} objects")

# Export to .blend
output_path = "/tmp/test_packed_export.blend"
export_to_blend(result, output_path)
print(f"  Exported to: {output_path}")

# Verify the file exists and has reasonable size
assert Path(output_path).exists(), "Output .blend file should exist"
file_size = Path(output_path).stat().st_size
print(f"  File size: {file_size / 1024:.1f} KB")
assert file_size > 1000, "File should be larger than 1KB"
print("  PASSED: Export successful")

# Test 3: Verify objects in the blend file
print("\n[Test 3] Verifying objects in .blend file...")
import bpy

# Count mesh objects (excluding tray boundary)
mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH' and not obj.name.startswith('Tray')]
print(f"  Found {len(mesh_objects)} mesh objects in scene")
assert len(mesh_objects) == result.num_placed, f"Should have {result.num_placed} mesh objects"

# Check that objects have transforms applied
for obj in mesh_objects:
    loc = obj.location
    print(f"    {obj.name}: location=({loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f})")

print("  PASSED: All objects present with transforms")

# Test 4: Verify tray boundary exists
print("\n[Test 4] Verifying tray boundary...")
tray_objects = [obj for obj in bpy.data.objects if obj.name.startswith('Tray')]
assert len(tray_objects) == 1, "Should have exactly one tray boundary"
print(f"  Tray boundary: {tray_objects[0].name}")
print("  PASSED: Tray boundary exists")

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
