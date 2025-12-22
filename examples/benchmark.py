#!/usr/bin/env python3
"""
Performance Benchmark
=====================

This example benchmarks the spectral packing algorithm with various
configurations and workloads.

It measures:
1. Voxelization time for different resolutions
2. Placement search time vs tray size
3. Packing time vs number of items
4. FFT operation performance

Requirements:
    - spectral_packer package (with CUDA support)
    - numpy

Usage:
    python benchmark.py [--quick]

    Use --quick for a faster benchmark with fewer iterations.
"""

import sys
import time
import numpy as np
from typing import List, Tuple


def create_random_items(
    num_items: int,
    min_size: int = 3,
    max_size: int = 10,
    seed: int = 42
) -> List[np.ndarray]:
    """Create random voxelized items for benchmarking."""
    rng = np.random.RandomState(seed)
    items = []

    for _ in range(num_items):
        size = rng.randint(min_size, max_size + 1, size=3)
        item = np.ones(tuple(size), dtype=np.int32)
        items.append(item)

    return items


def benchmark_placement_search(
    tray_sizes: List[Tuple[int, int, int]],
    item_size: Tuple[int, int, int] = (5, 5, 5),
    num_trials: int = 5
) -> None:
    """Benchmark placement search for different tray sizes."""
    from spectral_packer import fft_search_placement

    print("\n1. Placement Search Benchmark")
    print("=" * 60)
    print(f"{'Tray Size':<20} {'Mean (ms)':<15} {'Std (ms)':<15}")
    print("-" * 60)

    item = np.ones(item_size, dtype=np.int32)

    for tray_size in tray_sizes:
        tray = np.zeros(tray_size, dtype=np.int32)

        times = []
        for _ in range(num_trials):
            start = time.perf_counter()
            position, found, score = fft_search_placement(item, tray)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        mean_time = np.mean(times)
        std_time = np.std(times)
        print(f"{str(tray_size):<20} {mean_time:<15.2f} {std_time:<15.2f}")


def benchmark_packing(
    tray_size: Tuple[int, int, int],
    num_items_list: List[int],
    item_size_range: Tuple[int, int] = (3, 8),
    num_trials: int = 3
) -> None:
    """Benchmark full packing for different numbers of items."""
    from spectral_packer import BinPacker

    print("\n2. Full Packing Benchmark")
    print("=" * 60)
    print(f"Tray size: {tray_size}")
    print(f"{'Num Items':<15} {'Mean (ms)':<15} {'Placed':<15} {'Density':<15}")
    print("-" * 60)

    packer = BinPacker(tray_size=tray_size)

    for num_items in num_items_list:
        times = []
        placed_counts = []
        densities = []

        for trial in range(num_trials):
            items = create_random_items(
                num_items,
                min_size=item_size_range[0],
                max_size=item_size_range[1],
                seed=42 + trial
            )

            start = time.perf_counter()
            result = packer.pack_voxels(items)
            end = time.perf_counter()

            times.append((end - start) * 1000)
            placed_counts.append(result.num_placed)
            densities.append(result.density)

        mean_time = np.mean(times)
        mean_placed = np.mean(placed_counts)
        mean_density = np.mean(densities)
        print(f"{num_items:<15} {mean_time:<15.2f} {mean_placed:<15.1f} {mean_density:<15.1%}")


def benchmark_fft_operations(
    grid_sizes: List[int],
    num_trials: int = 5
) -> None:
    """Benchmark FFT convolution and correlation operations."""
    from spectral_packer import dft_conv3, dft_corr3

    print("\n3. FFT Operations Benchmark")
    print("=" * 60)
    print(f"{'Grid Size':<15} {'Conv (ms)':<15} {'Corr (ms)':<15}")
    print("-" * 60)

    for size in grid_sizes:
        a = np.ones((size, size, size), dtype=np.int32)
        b = np.ones((size, size, size), dtype=np.int32)

        # Benchmark convolution
        conv_times = []
        for _ in range(num_trials):
            start = time.perf_counter()
            result = dft_conv3(a, b)
            end = time.perf_counter()
            conv_times.append((end - start) * 1000)

        # Benchmark correlation
        corr_times = []
        for _ in range(num_trials):
            start = time.perf_counter()
            result = dft_corr3(a, b)
            end = time.perf_counter()
            corr_times.append((end - start) * 1000)

        mean_conv = np.mean(conv_times)
        mean_corr = np.mean(corr_times)
        print(f"{size}x{size}x{size:<8} {mean_conv:<15.2f} {mean_corr:<15.2f}")


def benchmark_distance_calculation(
    grid_sizes: List[int],
    num_trials: int = 5
) -> None:
    """Benchmark distance field calculation."""
    from spectral_packer import calculate_distance

    print("\n4. Distance Field Benchmark")
    print("=" * 60)
    print(f"{'Grid Size':<15} {'Mean (ms)':<15} {'Std (ms)':<15}")
    print("-" * 60)

    for size in grid_sizes:
        # Create a grid with some occupied voxels
        grid = np.zeros((size, size, size), dtype=np.int32)
        grid[size//4:size//2, size//4:size//2, size//4:size//2] = 1

        times = []
        for _ in range(num_trials):
            start = time.perf_counter()
            result = calculate_distance(grid)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        mean_time = np.mean(times)
        std_time = np.std(times)
        print(f"{size}x{size}x{size:<8} {mean_time:<15.2f} {std_time:<15.2f}")


def run_quick_benchmark():
    """Run a quick benchmark with reduced parameters."""
    print("Running QUICK benchmark (reduced iterations)")
    print()

    benchmark_placement_search(
        tray_sizes=[(32, 32, 32), (64, 64, 64)],
        num_trials=2
    )

    benchmark_packing(
        tray_size=(50, 50, 50),
        num_items_list=[5, 10],
        num_trials=2
    )

    benchmark_fft_operations(
        grid_sizes=[16, 32],
        num_trials=2
    )

    benchmark_distance_calculation(
        grid_sizes=[16, 32],
        num_trials=2
    )


def run_full_benchmark():
    """Run the full benchmark suite."""
    print("Running FULL benchmark")
    print()

    benchmark_placement_search(
        tray_sizes=[
            (32, 32, 32),
            (64, 64, 64),
            (100, 100, 100),
            (128, 128, 128),
        ],
        num_trials=5
    )

    benchmark_packing(
        tray_size=(100, 100, 100),
        num_items_list=[5, 10, 20, 50],
        num_trials=3
    )

    benchmark_fft_operations(
        grid_sizes=[16, 32, 64, 128],
        num_trials=5
    )

    benchmark_distance_calculation(
        grid_sizes=[16, 32, 64, 128],
        num_trials=5
    )


def main():
    """Run the benchmark."""
    from spectral_packer import is_cuda_available

    print("Spectral Packer - Performance Benchmark")
    print("=" * 60)

    if not is_cuda_available():
        print("\nError: CUDA core module not available.")
        print("Please ensure the package was built with CUDA support.")
        return

    # Check for quick mode
    quick_mode = "--quick" in sys.argv

    try:
        if quick_mode:
            run_quick_benchmark()
        else:
            run_full_benchmark()
    except Exception as e:
        print(f"\nBenchmark error: {e}")
        raise

    print("\n" + "=" * 60)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
