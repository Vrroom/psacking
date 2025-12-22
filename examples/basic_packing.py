#!/usr/bin/env python3
"""
Basic Packing Example
=====================

This example demonstrates the basic usage of the spectral_packer library
for 3D bin packing.

It shows how to:
1. Create a BinPacker with a specified tray size
2. Pack voxelized items into the tray
3. Access packing results and statistics

Requirements:
    - spectral_packer package (with CUDA support)
    - numpy

Usage:
    python basic_packing.py
"""

import numpy as np


def create_sample_items():
    """Create some sample voxelized items for packing.

    Returns a list of 3D numpy arrays representing various shapes.
    """
    items = []

    # A 5x5x5 cube
    cube = np.ones((5, 5, 5), dtype=np.int32)
    items.append(cube)

    # A 3x3x8 tall box
    tall_box = np.ones((3, 3, 8), dtype=np.int32)
    items.append(tall_box)

    # A 6x6x2 flat box
    flat_box = np.ones((6, 6, 2), dtype=np.int32)
    items.append(flat_box)

    # An L-shaped item
    l_shape = np.zeros((5, 5, 5), dtype=np.int32)
    l_shape[0:3, 0:2, 0:5] = 1
    l_shape[0:3, 2:4, 0:2] = 1
    items.append(l_shape)

    # Several small cubes
    for _ in range(5):
        small_cube = np.ones((2, 2, 2), dtype=np.int32)
        items.append(small_cube)

    return items


def main():
    """Run the basic packing example."""
    from spectral_packer import BinPacker, is_cuda_available

    # Check if CUDA is available
    if not is_cuda_available():
        print("Warning: CUDA core module not available.")
        print("The packing algorithm requires GPU acceleration.")
        print("Please ensure the package was built with CUDA support.")
        return

    print("Spectral Packer - Basic Example")
    print("=" * 40)

    # Create a packer with a 50x50x50 voxel tray
    print("\nCreating packer with 50x50x50 tray...")
    packer = BinPacker(
        tray_size=(50, 50, 50),
        voxel_resolution=64,  # Resolution for any mesh voxelization
    )

    # Create sample items
    print("Creating sample items...")
    items = create_sample_items()
    print(f"  Created {len(items)} items")

    # Calculate total volume of items
    total_item_volume = sum(np.sum(item > 0) for item in items)
    print(f"  Total item volume: {total_item_volume} voxels")

    # Pack the items
    print("\nPacking items...")
    result = packer.pack_voxels(items, sort_by_volume=True)

    # Print results
    print("\nPacking Results:")
    print("-" * 40)
    print(f"  Items placed:    {result.num_placed}/{len(items)}")
    print(f"  Items failed:    {result.num_failed}")
    print(f"  Packing density: {result.density:.1%}")
    print(f"  Total volume:    {result.total_volume} voxels")

    if result.bounding_box:
        bbox_min, bbox_max = result.bounding_box
        print(f"  Bounding box:    {bbox_min} to {bbox_max}")

    # Print individual placement details
    print("\nPlacement Details:")
    print("-" * 40)
    for p in result.placements:
        status = "Placed" if p.success else "Failed"
        if p.success:
            print(f"  Item {p.item_index}: {status} at {p.position}, "
                  f"score={p.score:.2f}, volume={p.volume}")
        else:
            print(f"  Item {p.item_index}: {status}, volume={p.volume}")

    # Print summary
    print("\n" + result.summary())

    # Analyze the tray
    print("\nTray Analysis:")
    print("-" * 40)
    unique_ids = np.unique(result.tray)
    print(f"  Unique IDs in tray: {unique_ids.tolist()}")
    for item_id in unique_ids:
        if item_id > 0:
            count = np.sum(result.tray == item_id)
            print(f"  Item {item_id}: {count} voxels")


if __name__ == "__main__":
    main()
