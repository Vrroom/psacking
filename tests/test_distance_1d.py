"""
Tests for 1D distance computation along each axis.

These tests use grids that are essentially 1D (two dimensions have size 1)
to isolate the sweep behavior along each axis independently.
"""
import numpy as np
import pytest
from collections import deque

try:
    from spectral_packer import calculate_distance
    HAS_BINDINGS = True
except ImportError:
    HAS_BINDINGS = False


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


@pytest.mark.skipif(not HAS_BINDINGS, reason="C++ bindings not available")
class TestDistance1DBasic:
    """Basic 1D distance tests along each axis."""

    def test_x_axis_n20(self):
        """Test 1D distance along X axis with N=20."""
        N = 20
        grid = np.zeros((N, 1, 1), dtype=np.int32)
        grid[0, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        expected = list(range(N))
        actual = [int(gpu_dist[i, 0, 0]) for i in range(N)]

        assert actual == expected, f"X-axis distances incorrect"

    def test_y_axis_n20(self):
        """Test 1D distance along Y axis with N=20."""
        N = 20
        grid = np.zeros((1, N, 1), dtype=np.int32)
        grid[0, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        expected = list(range(N))
        actual = [int(gpu_dist[0, j, 0]) for j in range(N)]

        assert actual == expected, f"Y-axis distances incorrect"

    def test_z_axis_n20(self):
        """Test 1D distance along Z axis with N=20."""
        N = 20
        grid = np.zeros((1, 1, N), dtype=np.int32)
        grid[0, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        expected = list(range(N))
        actual = [int(gpu_dist[0, 0, k]) for k in range(N)]

        assert actual == expected, f"Z-axis distances incorrect"

    def test_x_axis_n50(self):
        """Test 1D distance along X axis with N=50."""
        N = 50
        grid = np.zeros((N, 1, 1), dtype=np.int32)
        grid[0, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        assert gpu_dist[N-1, 0, 0] == N - 1

    def test_y_axis_n50(self):
        """Test 1D distance along Y axis with N=50."""
        N = 50
        grid = np.zeros((1, N, 1), dtype=np.int32)
        grid[0, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        assert gpu_dist[0, N-1, 0] == N - 1

    def test_z_axis_n50(self):
        """Test 1D distance along Z axis with N=50."""
        N = 50
        grid = np.zeros((1, 1, N), dtype=np.int32)
        grid[0, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        assert gpu_dist[0, 0, N-1] == N - 1


@pytest.mark.skipif(not HAS_BINDINGS, reason="C++ bindings not available")
class TestDistance1DMultipleOccupancies:
    """1D tests with different occupancy patterns."""

    def test_two_sources_ends(self):
        """Two occupied voxels at both ends."""
        N = 100
        grid = np.zeros((N, 1, 1), dtype=np.int32)
        grid[0, 0, 0] = 1
        grid[N-1, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)
        # Middle point should have distance to nearest source
        # Index 50 is 50 from 0 and 49 from 99, so min = 49
        assert gpu_dist[N//2, 0, 0] == min(N//2, N - 1 - N//2)

    def test_three_sources_evenly_spaced(self):
        """Three occupied voxels evenly spaced."""
        N = 99  # Divisible by 3
        grid = np.zeros((N, 1, 1), dtype=np.int32)
        grid[0, 0, 0] = 1
        grid[N//3, 0, 0] = 1
        grid[2*N//3, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)

    def test_source_at_far_end(self):
        """Single source at far end (tests backward propagation)."""
        N = 100
        grid = np.zeros((N, 1, 1), dtype=np.int32)
        grid[N-1, 0, 0] = 1

        gpu_dist = calculate_distance(grid)

        expected = [N - 1 - i for i in range(N)]
        actual = [int(gpu_dist[i, 0, 0]) for i in range(N)]
        assert actual == expected

    def test_source_in_middle(self):
        """Single source in the middle."""
        N = 101
        mid = N // 2
        grid = np.zeros((N, 1, 1), dtype=np.int32)
        grid[mid, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)

    def test_alternating_occupancy(self):
        """Every 10th voxel is occupied."""
        N = 100
        grid = np.zeros((N, 1, 1), dtype=np.int32)
        for i in range(0, N, 10):
            grid[i, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)
        # Sources at 0,10,20,...,90. Max distance is at index 99, distance 9 from 90
        assert gpu_dist.max() == 9

    def test_random_occupancy(self):
        """Random occupancy pattern."""
        np.random.seed(42)
        N = 200
        grid = np.zeros((N, 1, 1), dtype=np.int32)
        # Randomly occupy ~10% of voxels
        occupied = np.random.choice(N, size=N//10, replace=False)
        for idx in occupied:
            grid[idx, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)


@pytest.mark.skipif(not HAS_BINDINGS, reason="C++ bindings not available")
class TestDistance1DStress:
    """Stress tests with large 1D grids."""

    def test_x_axis_n256(self):
        """X-axis with N=256."""
        N = 256
        grid = np.zeros((N, 1, 1), dtype=np.int32)
        grid[0, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        assert gpu_dist[N-1, 0, 0] == N - 1

    def test_x_axis_n512(self):
        """X-axis with N=512."""
        N = 512
        grid = np.zeros((N, 1, 1), dtype=np.int32)
        grid[0, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        assert gpu_dist[N-1, 0, 0] == N - 1

    def test_x_axis_n1000(self):
        """X-axis with N=1000."""
        N = 1000
        grid = np.zeros((N, 1, 1), dtype=np.int32)
        grid[0, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        assert gpu_dist[N-1, 0, 0] == N - 1

    def test_y_axis_n1000(self):
        """Y-axis with N=1000."""
        N = 1000
        grid = np.zeros((1, N, 1), dtype=np.int32)
        grid[0, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        assert gpu_dist[0, N-1, 0] == N - 1

    def test_z_axis_n1000(self):
        """Z-axis with N=1000."""
        N = 1000
        grid = np.zeros((1, 1, N), dtype=np.int32)
        grid[0, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        assert gpu_dist[0, 0, N-1] == N - 1

    def test_n1000_multiple_sources(self):
        """N=1000 with multiple sources."""
        N = 1000
        grid = np.zeros((N, 1, 1), dtype=np.int32)
        grid[0, 0, 0] = 1
        grid[N//4, 0, 0] = 1
        grid[N//2, 0, 0] = 1
        grid[3*N//4, 0, 0] = 1
        grid[N-1, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)

    def test_n1000_dense_occupancy(self):
        """N=1000 with every 50th voxel occupied."""
        N = 1000
        grid = np.zeros((N, 1, 1), dtype=np.int32)
        for i in range(0, N, 50):
            grid[i, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        cpu_dist = cpu_manhattan_distance(grid)

        np.testing.assert_array_equal(gpu_dist, cpu_dist)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
