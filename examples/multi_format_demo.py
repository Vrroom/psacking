#!/usr/bin/env python3
"""
Multi-Format Mesh Loading Demo
==============================

This example demonstrates loading 3D meshes from various file formats
using the spectral_packer library.

It shows how to:
1. Load meshes from different formats (STL, OBJ, PLY, etc.)
2. Get mesh information and statistics
3. Validate and repair meshes
4. Voxelize meshes for packing

Requirements:
    - spectral_packer package
    - trimesh (pip install trimesh)
    - numpy

Usage:
    python multi_format_demo.py [mesh_file1] [mesh_file2] ...

    If no files are provided, it will create and use sample meshes.
"""

import sys
from pathlib import Path
import numpy as np


def create_sample_stl(output_path: Path) -> None:
    """Create a sample binary STL file of a cube."""
    import struct

    triangles = [
        # Front face (z=0)
        ((0, 0, -1), ((0, 0, 0), (1, 1, 0), (1, 0, 0))),
        ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0))),
        # Back face (z=1)
        ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1))),
        ((0, 0, 1), ((0, 0, 1), (1, 1, 1), (0, 1, 1))),
        # Bottom face (y=0)
        ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1))),
        ((0, -1, 0), ((0, 0, 0), (1, 0, 1), (0, 0, 1))),
        # Top face (y=1)
        ((0, 1, 0), ((0, 1, 0), (1, 1, 1), (1, 1, 0))),
        ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1))),
        # Left face (x=0)
        ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1))),
        ((-1, 0, 0), ((0, 0, 0), (0, 1, 1), (0, 1, 0))),
        # Right face (x=1)
        ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1))),
        ((1, 0, 0), ((1, 0, 0), (1, 1, 1), (1, 0, 1))),
    ]

    # Header (80 bytes) + num triangles (4 bytes)
    data = b"Binary STL cube" + b"\x00" * 65 + struct.pack("<I", len(triangles))

    for normal, (v1, v2, v3) in triangles:
        data += struct.pack("<3f", *normal)
        data += struct.pack("<3f", *v1)
        data += struct.pack("<3f", *v2)
        data += struct.pack("<3f", *v3)
        data += struct.pack("<H", 0)

    output_path.write_bytes(data)


def demonstrate_mesh_loading(mesh_path: Path) -> None:
    """Demonstrate mesh loading capabilities for a single file."""
    from spectral_packer import load_mesh, get_mesh_info, MeshLoadError

    print(f"\nLoading: {mesh_path.name}")
    print("-" * 50)

    # Get mesh info without loading
    try:
        info = get_mesh_info(mesh_path)
        print(f"  Format:      {info['format']}")
        print(f"  Vertices:    {info['num_vertices']}")
        print(f"  Faces:       {info['num_faces']}")
        print(f"  Watertight:  {info['is_watertight']}")
        print(f"  Surface:     {info['surface_area']:.4f}")
        if info['volume'] is not None:
            print(f"  Volume:      {info['volume']:.4f}")
        bbox = info['bounding_box']
        print(f"  Extents:     {bbox['extents']}")
        print(f"  File size:   {info['file_size_bytes']} bytes")
    except Exception as e:
        print(f"  Error getting info: {e}")
        return

    # Load the mesh
    try:
        vertices, faces = load_mesh(
            mesh_path,
            validate=True,
            repair=True,
            center=True,
            scale=1.0
        )
        print(f"\n  Loaded successfully:")
        print(f"    Vertices shape: {vertices.shape}")
        print(f"    Faces shape:    {faces.shape}")
        print(f"    Vertices dtype: {vertices.dtype}")
        print(f"    Vertices range: [{vertices.min():.3f}, {vertices.max():.3f}]")
    except MeshLoadError as e:
        print(f"  Load error: {e}")
    except Exception as e:
        print(f"  Error: {e}")


def demonstrate_voxelization(mesh_path: Path) -> None:
    """Demonstrate mesh voxelization."""
    from spectral_packer import Voxelizer, load_mesh

    print(f"\nVoxelizing: {mesh_path.name}")
    print("-" * 50)

    voxelizer = Voxelizer(resolution=32)

    try:
        # Try direct voxelization (uses C++ for STL)
        grid = voxelizer.voxelize_file(mesh_path)
        print(f"  Grid shape:    {grid.shape}")
        print(f"  Occupied:      {np.sum(grid > 0)} voxels")
        print(f"  Fill ratio:    {np.mean(grid > 0):.1%}")
    except Exception as e:
        print(f"  Voxelization error: {e}")


def main():
    """Run the multi-format demo."""
    from spectral_packer import SUPPORTED_FORMATS

    print("Spectral Packer - Multi-Format Demo")
    print("=" * 50)

    # List supported formats
    print("\nSupported Formats:")
    for ext, desc in SUPPORTED_FORMATS.items():
        print(f"  {ext:8s} - {desc}")

    # Get mesh files from command line or create samples
    if len(sys.argv) > 1:
        mesh_files = [Path(p) for p in sys.argv[1:]]
    else:
        # Create a sample mesh in temp directory
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        sample_stl = temp_dir / "sample_cube.stl"

        print(f"\nNo mesh files provided. Creating sample at: {sample_stl}")
        create_sample_stl(sample_stl)
        mesh_files = [sample_stl]

    # Process each mesh file
    for mesh_path in mesh_files:
        if not mesh_path.exists():
            print(f"\nFile not found: {mesh_path}")
            continue

        demonstrate_mesh_loading(mesh_path)
        demonstrate_voxelization(mesh_path)

    # Demonstrate packing with voxelized meshes
    print("\n" + "=" * 50)
    print("Packing Demo with Voxelized Meshes")
    print("=" * 50)

    try:
        from spectral_packer import BinPacker, Voxelizer, is_cuda_available

        if not is_cuda_available():
            print("\nCUDA not available. Skipping packing demo.")
            return

        voxelizer = Voxelizer(resolution=32)
        packer = BinPacker(tray_size=(50, 50, 50))

        # Voxelize all meshes
        voxels = []
        for mesh_path in mesh_files:
            if mesh_path.exists():
                try:
                    grid = voxelizer.voxelize_file(mesh_path)
                    voxels.append(grid)
                except Exception as e:
                    print(f"Could not voxelize {mesh_path.name}: {e}")

        if voxels:
            # Pack the voxelized meshes
            result = packer.pack_voxels(voxels)
            print(f"\nPacked {result.num_placed}/{len(voxels)} items")
            print(f"Packing density: {result.density:.1%}")

    except ImportError as e:
        print(f"\nPacking demo skipped: {e}")


if __name__ == "__main__":
    main()
