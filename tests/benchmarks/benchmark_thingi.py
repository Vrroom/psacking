#!/usr/bin/env python3
"""
Benchmark packing with Thingi10K meshes (similar to paper's experiments).

Paper setup:
- Tray: 480mm × 245mm × 200mm
- Voxel resolution: 2mm → 240 × 123 × 100 voxels
- Objects: 6000+ from Thingi10K

Usage:
    python benchmark_thingi.py --num-objects 50 --num-orientations 24
    python benchmark_thingi.py --num-objects 100 --visualize
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from spectral_packer import BinPacker, voxelize_stl


# Paper's tray dimensions at 2mm voxel resolution
TRAY_SIZE_MM = (480, 245, 200)
VOXEL_RESOLUTION_MM = 2
TRAY_SIZE_VOXELS = (
    TRAY_SIZE_MM[0] // VOXEL_RESOLUTION_MM,  # 240
    TRAY_SIZE_MM[1] // VOXEL_RESOLUTION_MM,  # 122 (we'll use 123)
    TRAY_SIZE_MM[2] // VOXEL_RESOLUTION_MM,  # 100
)
# Adjusted to match paper's 240×123×100
TRAY_SIZE_VOXELS = (240, 123, 100)


def find_thingi_stls(thingi_dir: str, limit: int = None) -> list:
    """Find STL files in Thingi10K directory."""
    pattern = os.path.join(thingi_dir, "*.stl")
    stl_files = sorted(glob.glob(pattern))
    if limit:
        stl_files = stl_files[:limit]
    return stl_files


def voxelize_meshes(stl_files: list, resolution: int = 64, verbose: bool = True):
    """Voxelize multiple STL files."""
    voxels = []
    failed = []

    for i, stl_path in enumerate(stl_files):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Voxelizing {i+1}/{len(stl_files)}...", end='\r')

        try:
            voxel = voxelize_stl(stl_path, resolution)
            if voxel is not None and np.sum(voxel) > 0:
                voxels.append(voxel)
            else:
                failed.append(stl_path)
        except Exception as e:
            failed.append(stl_path)

    if verbose:
        print(f"  Voxelized {len(voxels)}/{len(stl_files)} meshes successfully")
        if failed:
            print(f"  Failed: {len(failed)} meshes")

    return voxels


def run_benchmark(
    thingi_dir: str,
    num_objects: int = 50,
    num_orientations: int = 1,
    voxel_resolution: int = 64,
    tray_size: tuple = TRAY_SIZE_VOXELS,
    visualize: bool = False,
    verbose: bool = True,
):
    """Run packing benchmark with Thingi10K meshes."""

    print("=" * 70)
    print("SPECTRAL PACKING BENCHMARK")
    print("=" * 70)
    print(f"Tray size (voxels): {tray_size}")
    print(f"Num objects:        {num_objects}")
    print(f"Num orientations:   {num_orientations}")
    print(f"Voxel resolution:   {voxel_resolution}")
    print("=" * 70)

    # Find STL files
    print("\n[1/3] Finding STL files...")
    stl_files = find_thingi_stls(thingi_dir, limit=num_objects)
    if not stl_files:
        print(f"ERROR: No STL files found in {thingi_dir}")
        return None
    print(f"  Found {len(stl_files)} STL files")

    # Voxelize meshes
    print("\n[2/3] Voxelizing meshes...")
    t_voxelize_start = time.perf_counter()
    voxels = voxelize_meshes(stl_files, resolution=voxel_resolution, verbose=verbose)
    t_voxelize = time.perf_counter() - t_voxelize_start
    print(f"  Voxelization time: {t_voxelize:.2f}s")

    if not voxels:
        print("ERROR: No meshes were successfully voxelized")
        return None

    # Calculate total volume of all objects
    total_item_volume = sum(np.sum(v > 0) for v in voxels)
    tray_volume = np.prod(tray_size)
    theoretical_max_density = min(1.0, total_item_volume / tray_volume)

    print(f"\n  Total item volume: {total_item_volume} voxels")
    print(f"  Tray volume:       {tray_volume} voxels")
    print(f"  Theoretical max:   {theoretical_max_density:.1%}")

    # Pack
    print(f"\n[3/3] Packing with {num_orientations} orientation(s)...")
    packer = BinPacker(
        tray_size=tray_size,
        num_orientations=num_orientations,
    )

    t_pack_start = time.perf_counter()
    result = packer.pack_voxels(voxels, sort_by_volume=True)
    t_pack = time.perf_counter() - t_pack_start

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Items placed:       {result.num_placed}/{len(voxels)}")
    print(f"Items failed:       {result.num_failed}")
    print(f"Packing density:    {result.density:.2%}")
    print(f"Total volume:       {result.total_volume} voxels")
    print(f"Packing time:       {t_pack:.2f}s")
    print(f"Time per item:      {1000*t_pack/len(voxels):.1f}ms")
    print("=" * 70)

    if result.bounding_box:
        bbox_min, bbox_max = result.bounding_box
        bbox_dims = tuple(mx - mn + 1 for mn, mx in zip(bbox_min, bbox_max))
        print(f"Bounding box:       {bbox_dims}")

    # Visualize if requested
    if visualize:
        print("\nGenerating visualization...")
        visualize_3d_packing(result, tray_size)

    return {
        'num_objects': len(voxels),
        'num_placed': result.num_placed,
        'num_failed': result.num_failed,
        'density': result.density,
        'total_volume': result.total_volume,
        'pack_time': t_pack,
        'voxelize_time': t_voxelize,
        'tray_size': tray_size,
        'num_orientations': num_orientations,
    }


def visualize_3d_packing(result, tray_size):
    """Create visualization of 3D packing result."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("matplotlib not available")
        return

    tray = result.tray

    # Create 3 views: XY (top), XZ (front), YZ (side)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Top view (XY) - sum along Z
    xy_view = np.max(tray, axis=2)

    # Front view (XZ) - sum along Y
    xz_view = np.max(tray, axis=1)

    # Side view (YZ) - sum along X
    yz_view = np.max(tray, axis=0)

    views = [
        (xy_view, "Top View (XY)", "X", "Y"),
        (xz_view, "Front View (XZ)", "X", "Z"),
        (yz_view, "Side View (YZ)", "Y", "Z"),
    ]

    max_val = max(1, tray.max())
    colors = ['white'] + list(plt.cm.tab20.colors) * (max_val // 20 + 1)
    cmap = ListedColormap(colors[:max_val + 1])

    for ax, (view, title, xlabel, ylabel) in zip(axes, views):
        ax.imshow(view.T, cmap=cmap, origin='lower',
                  interpolation='nearest', vmin=0, vmax=max_val)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    plt.suptitle(
        f"Packing Result: {result.num_placed} items, {result.density:.1%} density",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()

    filename = '/tmp/benchmark_packing_3d.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*60}")
    print(f"SAVED 3D VISUALIZATION: {filename}")
    print(f"{'='*60}")


def compare_orientations(thingi_dir: str, num_objects: int = 30):
    """Compare packing with different orientation settings."""
    print("\n" + "=" * 70)
    print("ORIENTATION COMPARISON")
    print("=" * 70)

    results = {}
    for n_orient in [1, 4, 6, 24]:
        print(f"\n--- Testing {n_orient} orientations ---")
        result = run_benchmark(
            thingi_dir,
            num_objects=num_objects,
            num_orientations=n_orient,
            verbose=False,
        )
        if result:
            results[n_orient] = result

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Orientations':<15} {'Placed':<10} {'Density':<12} {'Time':<10}")
    print("-" * 50)
    for n_orient, r in results.items():
        print(f"{n_orient:<15} {r['num_placed']:<10} {r['density']:.2%}       {r['pack_time']:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark packing with Thingi10K")
    parser.add_argument(
        "--thingi-dir",
        default="/home/ubuntu/general-purpose/cpsc424_final_project/thingi",
        help="Path to Thingi10K STL files"
    )
    parser.add_argument(
        "--num-objects", "-n",
        type=int,
        default=50,
        help="Number of objects to pack"
    )
    parser.add_argument(
        "--num-orientations", "-o",
        type=int,
        default=1,
        choices=[1, 4, 6, 24],
        help="Number of orientations to try"
    )
    parser.add_argument(
        "--voxel-resolution", "-r",
        type=int,
        default=64,
        help="Voxel resolution for mesh voxelization"
    )
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Generate visualization"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare different orientation settings"
    )
    parser.add_argument(
        "--tray-size",
        type=int,
        nargs=3,
        default=list(TRAY_SIZE_VOXELS),
        help="Tray size in voxels (X Y Z)"
    )

    args = parser.parse_args()

    if args.compare:
        compare_orientations(args.thingi_dir, args.num_objects)
    else:
        run_benchmark(
            thingi_dir=args.thingi_dir,
            num_objects=args.num_objects,
            num_orientations=args.num_orientations,
            voxel_resolution=args.voxel_resolution,
            tray_size=tuple(args.tray_size),
            visualize=args.visualize,
        )


if __name__ == "__main__":
    main()
