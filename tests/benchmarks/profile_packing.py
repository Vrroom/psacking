#!/usr/bin/env python3
"""
Profile spectral packing operations to identify bottlenecks.

Usage:
    python -m tests.benchmarks.profile_packing
    python tests/benchmarks/profile_packing.py --num-items 10 --tray-size 128 128 128
    python tests/benchmarks/profile_packing.py --cprofile  # Enable cProfile analysis
"""
import argparse
import cProfile
import io
import pstats
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
import sys

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spectral_packer import (
    BinPacker,
    fft_search_placement,
    place_in_tray,
    dft_conv3,
    dft_corr3,
    calculate_distance,
)


@dataclass
class TimingResult:
    """Timing result for a single operation."""
    operation: str
    duration_ms: float
    iterations: int = 1

    @property
    def avg_ms(self) -> float:
        return self.duration_ms / self.iterations


def get_gpu_utilization() -> Optional[float]:
    """Get current GPU utilization percentage."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split('\n')[0])
    except Exception:
        pass
    return None


def get_gpu_memory() -> Tuple[Optional[float], Optional[float]]:
    """Get GPU memory usage (used_mb, total_mb)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            return float(parts[0].strip()), float(parts[1].strip())
    except Exception:
        pass
    return None, None


def create_random_item(size: Tuple[int, int, int], fill_ratio: float = 0.3) -> np.ndarray:
    """Create a random voxel item."""
    item = np.random.random(size) < fill_ratio
    return item.astype(np.int32)


def create_cube_item(size: int) -> np.ndarray:
    """Create a solid cube item."""
    return np.ones((size, size, size), dtype=np.int32)


def time_function(func, *args, iterations: int = 1, warmup: int = 0, **kwargs) -> TimingResult:
    """Time a function with optional warmup."""
    name = func.__name__

    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Timed runs
    start = time.perf_counter()
    for _ in range(iterations):
        result = func(*args, **kwargs)
    end = time.perf_counter()

    return TimingResult(
        operation=name,
        duration_ms=(end - start) * 1000,
        iterations=iterations
    )


def pad_to_size(arr: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
    """Pad array to target size (for FFT operations that require same-sized inputs)."""
    padded = np.zeros(target_size, dtype=arr.dtype)
    slices = tuple(slice(0, s) for s in arr.shape)
    padded[slices] = arr
    return padded


def profile_fft_operations(tray_size: Tuple[int, int, int], item_size: int = 20):
    """Profile low-level FFT operations."""
    print("\n" + "=" * 70)
    print("FFT OPERATIONS PROFILE")
    print("=" * 70)
    print(f"Tray size: {tray_size}, Item size: {item_size}^3")

    tray = np.zeros(tray_size, dtype=np.int32)
    item = create_cube_item(item_size)
    item_padded = pad_to_size(item, tray_size)

    results = []

    # Profile dft_conv3 (requires same-sized inputs)
    r = time_function(dft_conv3, item_padded, tray, iterations=5, warmup=1)
    results.append(r)
    print(f"  dft_conv3:           {r.avg_ms:8.2f} ms avg ({r.iterations} runs)")

    # Profile dft_corr3 (requires same-sized inputs)
    r = time_function(dft_corr3, item_padded, tray, iterations=5, warmup=1)
    results.append(r)
    print(f"  dft_corr3:           {r.avg_ms:8.2f} ms avg ({r.iterations} runs)")

    # Profile calculate_distance
    occupied_tray = place_in_tray(item, tray.copy(), (0, 0, 0), 1)
    r = time_function(calculate_distance, occupied_tray, iterations=5, warmup=1)
    results.append(r)
    print(f"  calculate_distance:  {r.avg_ms:8.2f} ms avg ({r.iterations} runs)")

    # Profile fft_search_placement (handles padding internally)
    r = time_function(fft_search_placement, item, tray, iterations=5, warmup=1)
    results.append(r)
    print(f"  fft_search_placement:{r.avg_ms:8.2f} ms avg ({r.iterations} runs)")

    return results


def profile_placement_scaling(tray_size: Tuple[int, int, int], item_sizes: List[int]):
    """Profile placement time vs item size."""
    print("\n" + "=" * 70)
    print("PLACEMENT TIME vs ITEM SIZE")
    print("=" * 70)
    print(f"Tray size: {tray_size}")
    print(f"{'Item Size':<12} {'Volume':<12} {'Time (ms)':<12}")
    print("-" * 40)

    tray = np.zeros(tray_size, dtype=np.int32)

    for size in item_sizes:
        if size > min(tray_size):
            continue
        item = create_cube_item(size)
        volume = size ** 3

        r = time_function(fft_search_placement, item, tray, iterations=3, warmup=1)
        print(f"{size}^3         {volume:<12} {r.avg_ms:<12.2f}")


def profile_tray_filling(tray_size: Tuple[int, int, int], num_items: int, item_size: int = 15):
    """Profile packing time as tray fills up."""
    print("\n" + "=" * 70)
    print("PACKING TIME vs TRAY FILL LEVEL")
    print("=" * 70)
    print(f"Tray size: {tray_size}, Item size: {item_size}^3, Num items: {num_items}")
    print(f"{'Item #':<10} {'Fill %':<12} {'Time (ms)':<12} {'Position':<20}")
    print("-" * 60)

    tray = np.zeros(tray_size, dtype=np.int32)
    tray_volume = np.prod(tray_size)

    for i in range(num_items):
        item = create_cube_item(item_size)

        start = time.perf_counter()
        pos, found, score = fft_search_placement(item, tray)
        end = time.perf_counter()

        time_ms = (end - start) * 1000
        fill_pct = 100 * np.sum(tray > 0) / tray_volume

        if found:
            tray = place_in_tray(item, tray, pos, i + 1)
            print(f"{i+1:<10} {fill_pct:<12.1f} {time_ms:<12.2f} {pos}")
        else:
            print(f"{i+1:<10} {fill_pct:<12.1f} {time_ms:<12.2f} FAILED")
            break


def profile_orientation_overhead(tray_size: Tuple[int, int, int], num_items: int = 5):
    """Profile overhead of different orientation counts."""
    print("\n" + "=" * 70)
    print("ORIENTATION SAMPLING OVERHEAD")
    print("=" * 70)
    print(f"Tray size: {tray_size}, Num items: {num_items}")

    # Create random items
    items = [create_random_item((15, 20, 10), fill_ratio=0.5) for _ in range(num_items)]

    print(f"\n{'Orientations':<15} {'Total Time':<15} {'Per Item':<15} {'Placed':<10}")
    print("-" * 60)

    for n_orient in [1, 4, 6, 24]:
        packer = BinPacker(tray_size=tray_size, num_orientations=n_orient)

        start = time.perf_counter()
        result = packer.pack_voxels(items, sort_by_volume=True)
        end = time.perf_counter()

        total_ms = (end - start) * 1000
        per_item_ms = total_ms / num_items

        print(f"{n_orient:<15} {total_ms:<15.1f} {per_item_ms:<15.1f} {result.num_placed:<10}")


def profile_gpu_utilization(tray_size: Tuple[int, int, int], duration_sec: float = 10):
    """Profile GPU utilization during continuous packing."""
    print("\n" + "=" * 70)
    print("GPU UTILIZATION PROFILE")
    print("=" * 70)

    # Get baseline
    baseline_util = get_gpu_utilization()
    used_mb, total_mb = get_gpu_memory()

    print(f"Initial GPU utilization: {baseline_util:.1f}%" if baseline_util else "GPU query unavailable")
    if used_mb:
        print(f"Initial GPU memory: {used_mb:.0f} / {total_mb:.0f} MB ({100*used_mb/total_mb:.1f}%)")

    print(f"\nRunning continuous packing for {duration_sec}s...")

    tray = np.zeros(tray_size, dtype=np.int32)
    item = create_cube_item(15)

    start = time.perf_counter()
    iterations = 0
    util_samples = []

    while time.perf_counter() - start < duration_sec:
        fft_search_placement(item, tray)
        iterations += 1

        # Sample GPU utilization periodically
        if iterations % 10 == 0:
            util = get_gpu_utilization()
            if util is not None:
                util_samples.append(util)

    elapsed = time.perf_counter() - start

    print(f"\nCompleted {iterations} FFT searches in {elapsed:.2f}s")
    print(f"Throughput: {iterations/elapsed:.1f} searches/sec")

    if util_samples:
        print(f"\nGPU Utilization during packing:")
        print(f"  Min:  {min(util_samples):.1f}%")
        print(f"  Max:  {max(util_samples):.1f}%")
        print(f"  Avg:  {sum(util_samples)/len(util_samples):.1f}%")


def full_profile(tray_size: Tuple[int, int, int], num_items: int):
    """Run complete profiling suite."""
    print("=" * 70)
    print("SPECTRAL PACKER PROFILING SUITE")
    print("=" * 70)
    print(f"Tray size: {tray_size}")
    print(f"Num items: {num_items}")

    # System info
    util = get_gpu_utilization()
    used_mb, total_mb = get_gpu_memory()
    if util is not None:
        print(f"GPU utilization: {util:.1f}%")
    if used_mb:
        print(f"GPU memory: {used_mb:.0f}/{total_mb:.0f} MB")

    # Run profiles
    profile_fft_operations(tray_size)
    profile_placement_scaling(tray_size, [5, 10, 15, 20, 25, 30])
    profile_tray_filling(tray_size, num_items=20, item_size=15)
    profile_orientation_overhead(tray_size, num_items=5)
    profile_gpu_utilization(tray_size, duration_sec=5)

    print("\n" + "=" * 70)
    print("PROFILING COMPLETE")
    print("=" * 70)


def profile_with_cprofile(tray_size: Tuple[int, int, int], num_items: int,
                          num_orientations: int = 1, save_profile: bool = True):
    """
    Profile packing operations using cProfile for detailed Python-level analysis.

    This provides cumulative timing for all Python functions, helping identify
    bottlenecks in the Python layer (orientation generation, array processing, etc.)
    Note: C++ functions appear as single calls since cProfile can't see inside them.
    """
    print("\n" + "=" * 70)
    print("cPROFILE ANALYSIS")
    print("=" * 70)
    print(f"Tray size: {tray_size}")
    print(f"Num items: {num_items}")
    print(f"Num orientations: {num_orientations}")

    # Create test items
    items = [create_random_item((15, 20, 10), fill_ratio=0.5) for _ in range(num_items)]

    # Create profiler
    profiler = cProfile.Profile()

    # Profile the packing operation
    print("\nProfiling BinPacker.pack_voxels()...")
    profiler.enable()

    packer = BinPacker(tray_size=tray_size, num_orientations=num_orientations)
    result = packer.pack_voxels(items, sort_by_volume=True)

    profiler.disable()

    # Print results summary
    print(f"\nPacking result: {result.num_placed}/{num_items} items placed")
    print(f"Density: {result.density:.1%}")

    # Print cProfile statistics
    print("\n" + "-" * 70)
    print("TOP 30 FUNCTIONS BY CUMULATIVE TIME")
    print("-" * 70)

    # Create stats object
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')

    # Print to stdout
    stats.print_stats(30)

    # Also show callers for top functions
    print("\n" + "-" * 70)
    print("TOP 10 FUNCTIONS BY TOTAL TIME (self time)")
    print("-" * 70)
    stats.sort_stats('tottime')
    stats.print_stats(10)

    # Save profile to file if requested
    if save_profile:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_path = Path(f"/tmp/spectral_packer_profile_{timestamp}.prof")
        profiler.dump_stats(str(profile_path))
        print(f"\n{'=' * 70}")
        print(f"Profile saved to: {profile_path}")
        print(f"View with: python -m pstats {profile_path}")
        print(f"Or use snakeviz: snakeviz {profile_path}")
        print(f"{'=' * 70}")

    return profiler


def profile_cprofile_comparison(tray_size: Tuple[int, int, int], num_items: int):
    """Compare cProfile results across different orientation counts."""
    print("\n" + "=" * 70)
    print("cPROFILE COMPARISON: ORIENTATION SCALING")
    print("=" * 70)

    items = [create_random_item((15, 20, 10), fill_ratio=0.5) for _ in range(num_items)]

    for n_orient in [1, 4, 24]:
        print(f"\n{'='*70}")
        print(f"ORIENTATIONS: {n_orient}")
        print(f"{'='*70}")

        profiler = cProfile.Profile()
        profiler.enable()

        packer = BinPacker(tray_size=tray_size, num_orientations=n_orient)
        result = packer.pack_voxels(items.copy(), sort_by_volume=True)

        profiler.disable()

        print(f"Result: {result.num_placed}/{num_items} placed, {result.density:.1%} density")

        stats = pstats.Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats('cumulative')

        # Show just top 10 for comparison
        print("\nTop 10 by cumulative time:")
        stats.print_stats(10)


def main():
    parser = argparse.ArgumentParser(description="Profile spectral packing operations")
    parser.add_argument(
        "--tray-size", "-t",
        type=int, nargs=3,
        default=[128, 128, 128],
        help="Tray size (X Y Z)"
    )
    parser.add_argument(
        "--num-items", "-n",
        type=int, default=10,
        help="Number of items for full profile"
    )
    parser.add_argument(
        "--profile",
        choices=["full", "fft", "scaling", "filling", "orientations", "gpu", "cprofile", "cprofile-compare"],
        default="full",
        help="Profile type to run"
    )
    parser.add_argument(
        "--num-orientations",
        type=int, default=1,
        choices=[1, 4, 6, 24],
        help="Number of orientations for cprofile mode"
    )
    parser.add_argument(
        "--cprofile",
        action="store_true",
        help="Shortcut for --profile cprofile"
    )

    args = parser.parse_args()
    tray_size = tuple(args.tray_size)

    # Handle --cprofile shortcut
    profile_type = "cprofile" if args.cprofile else args.profile

    if profile_type == "full":
        full_profile(tray_size, args.num_items)
    elif profile_type == "fft":
        profile_fft_operations(tray_size)
    elif profile_type == "scaling":
        profile_placement_scaling(tray_size, [5, 10, 15, 20, 25, 30])
    elif profile_type == "filling":
        profile_tray_filling(tray_size, num_items=args.num_items)
    elif profile_type == "orientations":
        profile_orientation_overhead(tray_size, num_items=args.num_items)
    elif profile_type == "gpu":
        profile_gpu_utilization(tray_size, duration_sec=10)
    elif profile_type == "cprofile":
        profile_with_cprofile(tray_size, args.num_items,
                              num_orientations=args.num_orientations)
    elif profile_type == "cprofile-compare":
        profile_cprofile_comparison(tray_size, args.num_items)


if __name__ == "__main__":
    main()
