"""
Comprehensive 3D distance computation tests.

These tests verify correct Manhattan distance computation on various
3D grid configurations with different sizes, shapes, and occupancy patterns.
"""
import numpy as np
import pytest
from collections import deque

from spectral_packer import calculate_distance


def cpu_manhattan_distance(grid):
    """Reference CPU implementation using BFS."""
    shape = grid.shape
    dist = np.full(shape, np.iinfo(np.int32).max, dtype=np.int32)
    visited = np.zeros(shape, dtype=bool)
    queue = deque()
    for idx in np.ndindex(shape):
        if grid[idx] == 1:
            dist[idx] = 0
            visited[idx] = True
            queue.append(idx)
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


class TestDistance3DCorners:
    """Test distance computation with sources at corners."""

    def test_single_corner_cube(self):
        """Single source at corner of a cube."""
        N = 30
        grid = np.zeros((N, N, N), dtype=np.int32)
        grid[0, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)
        # Far corner should have distance 3*(N-1)
        assert gpu_dist[N-1, N-1, N-1] == 3 * (N - 1)

    def test_opposite_corners(self):
        """Sources at two opposite corners."""
        N = 30
        grid = np.zeros((N, N, N), dtype=np.int32)
        grid[0, 0, 0] = 1
        grid[N-1, N-1, N-1] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)
        # Center should be equidistant from both corners
        mid = N // 2
        expected_mid_dist = mid + mid + mid  # or (N-1-mid)*3
        assert gpu_dist[mid, mid, mid] <= expected_mid_dist

    def test_all_eight_corners(self):
        """Sources at all eight corners."""
        N = 20
        grid = np.zeros((N, N, N), dtype=np.int32)
        for i in [0, N-1]:
            for j in [0, N-1]:
                for k in [0, N-1]:
                    grid[i, j, k] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)
        # Center should have distance to nearest corner
        # For N=20, mid=10, nearest corners are (19,19,19) etc at distance 9*3=27
        mid = N // 2
        dist_to_near_corner = 3 * (N - 1 - mid)
        assert gpu_dist[mid, mid, mid] == dist_to_near_corner

    def test_corner_rectangular(self):
        """Corner source on rectangular (non-cubic) grid."""
        grid = np.zeros((40, 30, 20), dtype=np.int32)
        grid[0, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)
        # Far corner distance
        assert gpu_dist[39, 29, 19] == 39 + 29 + 19


class TestDistance3DShapes:
    """Test with different grid shapes."""

    def test_thin_slab_xy(self):
        """Thin slab in XY plane (small Z)."""
        grid = np.zeros((50, 50, 3), dtype=np.int32)
        grid[25, 25, 1] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)

    def test_thin_slab_xz(self):
        """Thin slab in XZ plane (small Y)."""
        grid = np.zeros((50, 3, 50), dtype=np.int32)
        grid[25, 1, 25] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)

    def test_thin_slab_yz(self):
        """Thin slab in YZ plane (small X)."""
        grid = np.zeros((3, 50, 50), dtype=np.int32)
        grid[1, 25, 25] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)

    def test_elongated_x(self):
        """Long grid along X axis."""
        grid = np.zeros((100, 10, 10), dtype=np.int32)
        grid[50, 5, 5] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)

    def test_elongated_y(self):
        """Long grid along Y axis."""
        grid = np.zeros((10, 100, 10), dtype=np.int32)
        grid[5, 50, 5] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)

    def test_elongated_z(self):
        """Long grid along Z axis."""
        grid = np.zeros((10, 10, 100), dtype=np.int32)
        grid[5, 5, 50] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)

    def test_all_different_dimensions(self):
        """Grid with all different dimensions."""
        grid = np.zeros((37, 53, 41), dtype=np.int32)
        grid[18, 26, 20] = 1  # Center-ish

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)


class TestDistance3DPatterns:
    """Test with various occupancy patterns."""

    def test_center_source(self):
        """Single source at center."""
        N = 31  # Odd so center is exact
        mid = N // 2
        grid = np.zeros((N, N, N), dtype=np.int32)
        grid[mid, mid, mid] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)
        # All corners should have same distance
        corner_dist = gpu_dist[0, 0, 0]
        assert corner_dist == 3 * mid
        assert gpu_dist[N-1, 0, 0] == corner_dist
        assert gpu_dist[0, N-1, 0] == corner_dist
        assert gpu_dist[0, 0, N-1] == corner_dist

    def test_edge_sources(self):
        """Sources along one edge."""
        N = 25
        grid = np.zeros((N, N, N), dtype=np.int32)
        for i in range(N):
            grid[i, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)

    def test_face_sources(self):
        """Sources covering one face."""
        N = 20
        grid = np.zeros((N, N, N), dtype=np.int32)
        grid[0, :, :] = 1  # Entire x=0 face

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)
        # Distance should equal x coordinate
        for i in range(N):
            assert gpu_dist[i, N//2, N//2] == i

    def test_plane_source(self):
        """Source plane in the middle."""
        N = 25
        mid = N // 2
        grid = np.zeros((N, N, N), dtype=np.int32)
        grid[mid, :, :] = 1  # Entire x=mid plane

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)

    def test_sparse_random(self):
        """Sparse random occupancy (~1%)."""
        np.random.seed(123)
        N = 40
        grid = np.zeros((N, N, N), dtype=np.int32)
        num_sources = N * N * N // 100
        indices = np.random.choice(N * N * N, size=num_sources, replace=False)
        for idx in indices:
            i, j, k = idx // (N * N), (idx // N) % N, idx % N
            grid[i, j, k] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)

    def test_dense_random(self):
        """Dense random occupancy (~20%)."""
        np.random.seed(456)
        N = 30
        grid = np.zeros((N, N, N), dtype=np.int32)
        num_sources = N * N * N // 5
        indices = np.random.choice(N * N * N, size=num_sources, replace=False)
        for idx in indices:
            i, j, k = idx // (N * N), (idx // N) % N, idx % N
            grid[i, j, k] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)

    def test_checkerboard_3d(self):
        """3D checkerboard pattern (every other voxel)."""
        N = 20
        grid = np.zeros((N, N, N), dtype=np.int32)
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if (i + j + k) % 2 == 0:
                        grid[i, j, k] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)
        # All distances should be 0 or 1
        assert gpu_dist.max() == 1


class TestDistance3DStress:
    """Stress tests with larger 3D grids."""

    def test_cube_64(self):
        """64x64x64 cube with corner source."""
        N = 64
        grid = np.zeros((N, N, N), dtype=np.int32)
        grid[0, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)

    def test_cube_64_multiple_sources(self):
        """64x64x64 cube with multiple sources."""
        N = 64
        grid = np.zeros((N, N, N), dtype=np.int32)
        # Sources at various locations
        grid[0, 0, 0] = 1
        grid[N-1, N-1, N-1] = 1
        grid[N//2, N//2, N//2] = 1
        grid[N//4, 3*N//4, N//2] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)

    def test_rectangular_large(self):
        """Large rectangular grid 80x60x40."""
        grid = np.zeros((80, 60, 40), dtype=np.int32)
        grid[40, 30, 20] = 1
        grid[0, 0, 0] = 1
        grid[79, 59, 39] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)

    def test_cube_100(self):
        """100x100x100 cube - 1M voxels."""
        N = 100
        grid = np.zeros((N, N, N), dtype=np.int32)
        grid[0, 0, 0] = 1
        grid[N-1, N-1, N-1] = 1

        gpu_dist = calculate_distance(grid)

        # Just verify corner distances (full BFS would be slow)
        assert gpu_dist[0, 0, 0] == 0
        assert gpu_dist[N-1, N-1, N-1] == 0
        # Center (50,50,50) is closer to (99,99,99) than (0,0,0)
        # Distance to (99,99,99) = 49+49+49 = 147
        mid = N // 2
        expected_dist = 3 * (N - 1 - mid)  # 147
        assert gpu_dist[mid, mid, mid] == expected_dist


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
