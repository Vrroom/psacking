"""
Tests to verify/disprove the correctness of the GPU distance computation.

This test suite investigates potential bugs in the calculate_distance function
implemented in fft3.cu.
"""
import numpy as np
import pytest

from spectral_packer import calculate_distance


def cpu_manhattan_distance(grid: np.ndarray) -> np.ndarray:
    """
    Reference implementation of Manhattan (L1) distance using BFS.

    This is the gold standard for comparison.
    """
    from collections import deque

    shape = grid.shape
    dist = np.full(shape, np.iinfo(np.int32).max, dtype=np.int32)
    visited = np.zeros(shape, dtype=bool)
    queue = deque()

    # Initialize: occupied cells have distance 0
    for idx in np.ndindex(shape):
        if grid[idx] == 1:
            dist[idx] = 0
            visited[idx] = True
            queue.append(idx)

    # BFS
    directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

    while queue:
        x, y, z = queue.popleft()
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            if (0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]
                and not visited[nx, ny, nz]):
                visited[nx, ny, nz] = True
                dist[nx, ny, nz] = dist[x, y, z] + 1
                queue.append((nx, ny, nz))

    return dist


class TestDistanceCorrectness:
    """Tests for GPU distance computation correctness."""

    def test_single_occupied_center_cube(self):
        """Test with a single occupied voxel in the center of a cube."""
        # Use a cube (N=M=L) to avoid the dimension swap bug
        grid = np.zeros((5, 5, 5), dtype=np.int32)
        grid[2, 2, 2] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        # Check distances match
        np.testing.assert_array_equal(
            gpu_dist, cpu_dist,
            err_msg="GPU distance differs from CPU reference for cube grid"
        )

    def test_single_occupied_rectangular_grid(self):
        """
        Test with a rectangular grid where dimensions are NOT equal.

        This tests the suspected indexing bug when M != L.
        """
        # Non-cubic: N=5, M=7, L=3 (all different)
        grid = np.zeros((5, 7, 3), dtype=np.int32)
        grid[2, 3, 1] = 1  # Center-ish occupied voxel

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        print(f"\nGrid shape: {grid.shape}")
        print(f"Occupied at: (2, 3, 1)")
        print(f"\nGPU result at occupied cell: {gpu_dist[2, 3, 1]}")
        print(f"CPU result at occupied cell: {cpu_dist[2, 3, 1]}")

        # Check if the occupied cell has distance 0
        assert gpu_dist[2, 3, 1] == 0, f"Occupied cell should have distance 0, got {gpu_dist[2, 3, 1]}"

        # Check a few specific distances
        # The cell at (0, 0, 0) should have distance |2-0| + |3-0| + |1-0| = 6
        expected_dist_000 = abs(2-0) + abs(3-0) + abs(1-0)
        print(f"\nDistance at (0,0,0): GPU={gpu_dist[0,0,0]}, CPU={cpu_dist[0,0,0]}, expected={expected_dist_000}")

        # Full comparison
        if not np.array_equal(gpu_dist, cpu_dist):
            # Find first mismatch
            mismatch = np.where(gpu_dist != cpu_dist)
            if len(mismatch[0]) > 0:
                first_idx = (mismatch[0][0], mismatch[1][0], mismatch[2][0])
                print(f"\nFirst mismatch at {first_idx}:")
                print(f"  GPU: {gpu_dist[first_idx]}")
                print(f"  CPU: {cpu_dist[first_idx]}")
                print(f"  Total mismatches: {len(mismatch[0])}")

        np.testing.assert_array_equal(
            gpu_dist, cpu_dist,
            err_msg="GPU distance differs from CPU reference for rectangular grid (M != L)"
        )

    def test_rectangular_grid_m_equals_l(self):
        """Test rectangular grid where M == L but N is different."""
        # N=7, M=5, L=5 (M equals L)
        grid = np.zeros((7, 5, 5), dtype=np.int32)
        grid[3, 2, 2] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(
            gpu_dist, cpu_dist,
            err_msg="GPU distance differs from CPU reference when M == L"
        )

    def test_multiple_occupied_voxels(self):
        """Test with multiple occupied voxels."""
        grid = np.zeros((6, 6, 6), dtype=np.int32)
        grid[0, 0, 0] = 1
        grid[5, 5, 5] = 1
        grid[3, 3, 3] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(
            gpu_dist, cpu_dist,
            err_msg="GPU distance differs from CPU for multiple occupied voxels"
        )

    def test_large_rectangular_grid(self):
        """Test a larger rectangular grid to stress test the algorithm."""
        # Large grid with different dimensions
        grid = np.zeros((20, 15, 10), dtype=np.int32)
        # Place occupied voxels in various locations
        grid[5, 5, 5] = 1
        grid[15, 10, 5] = 1
        grid[0, 0, 0] = 1
        grid[19, 14, 9] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        print(f"\nLarge grid test - shape: {grid.shape}")

        if not np.array_equal(gpu_dist, cpu_dist):
            mismatch_count = np.sum(gpu_dist != cpu_dist)
            print(f"Mismatches: {mismatch_count} out of {grid.size}")

        np.testing.assert_array_equal(
            gpu_dist, cpu_dist,
            err_msg="GPU distance differs from CPU for large rectangular grid"
        )

    def test_iteration_convergence(self):
        """
        Test if 8 iterations is sufficient for convergence.

        The algorithm uses a fixed 8 iterations. For large grids where
        the maximum distance exceeds what can be propagated in 8 iterations,
        results may be incorrect.
        """
        # Create a thin elongated grid where distance could be large
        # If the grid is 100 long in one dimension, max L1 distance could be ~100
        # With 8 iterations of sweeping, this should still work because
        # each iteration does a full forward+backward pass
        grid = np.zeros((50, 3, 3), dtype=np.int32)
        grid[0, 1, 1] = 1  # Occupied at one end

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        # The far end should have distance 49
        print(f"\nElongated grid test")
        print(f"Distance at far end (49,1,1): GPU={gpu_dist[49,1,1]}, CPU={cpu_dist[49,1,1]}")

        np.testing.assert_array_equal(
            gpu_dist, cpu_dist,
            err_msg="GPU distance iteration convergence test failed"
        )


class TestIndexingBug:
    """Specific tests to demonstrate the indexing bug in calculate_distance."""

    def test_specific_indexing_case(self):
        """
        Detailed test showing the indexing issue.

        When M != L, the GPU implementation appears to swap the j and k indices,
        causing incorrect distance values to be computed.
        """
        # Create a 3x4x5 grid (N=3, M=4, L=5)
        # The GPU code receives dimensions as (N, L, M) = (3, 5, 4)
        # and uses indexing (i*L+j)*M + k = (i*5+j)*4 + k
        # but data was copied using i*(M*L) + j*L + k = i*20 + j*5 + k

        N, M, L = 3, 4, 5
        grid = np.zeros((N, M, L), dtype=np.int32)

        # Set occupied voxel at (1, 2, 3)
        # Copy-in index: 1*20 + 2*5 + 3 = 33
        grid[1, 2, 3] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        print(f"\nSpecific indexing test - grid shape: ({N}, {M}, {L})")
        print(f"Occupied voxel at (1, 2, 3)")
        print(f"\nOccupied cell distance: GPU={gpu_dist[1,2,3]}, CPU={cpu_dist[1,2,3]}")

        # Check corner distances
        corners = [(0,0,0), (0,0,L-1), (0,M-1,0), (0,M-1,L-1),
                   (N-1,0,0), (N-1,0,L-1), (N-1,M-1,0), (N-1,M-1,L-1)]

        print("\nCorner distances:")
        for corner in corners:
            expected = abs(corner[0]-1) + abs(corner[1]-2) + abs(corner[2]-3)
            print(f"  {corner}: GPU={gpu_dist[corner]}, CPU={cpu_dist[corner]}, expected L1={expected}")

        np.testing.assert_array_equal(
            gpu_dist, cpu_dist,
            err_msg="Indexing bug demonstrated: M != L causes incorrect distances"
        )

    def test_diagnose_swapped_indices(self):
        """
        Check if the bug manifests as swapped M and L dimensions.
        """
        N, M, L = 4, 6, 3  # Deliberately different M and L
        grid = np.zeros((N, M, L), dtype=np.int32)

        # Place a single occupied voxel
        grid[2, 4, 1] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        print(f"\nDiagnostic test - shape: ({N}, {M}, {L})")
        print(f"Occupied at (2, 4, 1)")

        # Check if the GPU thinks the occupied cell is elsewhere
        # If indices are swapped, (2, 4, 1) might appear at (2, 1, 4)
        print(f"\ngpu_dist[2,4,1] = {gpu_dist[2,4,1]} (should be 0)")
        print(f"gpu_dist[2,1,4] would be out of bounds for L={L}")

        # Find where GPU thinks distance is 0
        zero_locs = np.where(gpu_dist == 0)
        print(f"\nLocations where GPU distance = 0:")
        for i, j, k in zip(*zero_locs):
            print(f"  ({i}, {j}, {k})")

        np.testing.assert_array_equal(
            gpu_dist, cpu_dist,
            err_msg="Index swap bug detected"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
