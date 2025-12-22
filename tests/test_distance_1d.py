"""
Tests for 1D distance computation along each axis.

These tests use grids that are essentially 1D (two dimensions have size 1)
to isolate the sweep behavior along each axis independently.
"""
import numpy as np
import pytest

try:
    from spectral_packer import calculate_distance
    HAS_BINDINGS = True
except ImportError:
    HAS_BINDINGS = False


@pytest.mark.skipif(not HAS_BINDINGS, reason="C++ bindings not available")
class TestDistance1D:
    """Test distance computation on 1D grids along each axis."""

    def test_x_axis_n20(self):
        """Test 1D distance along X axis with N=20."""
        N = 20
        grid = np.zeros((N, 1, 1), dtype=np.int32)
        grid[0, 0, 0] = 1  # Occupied at one end

        gpu_dist = calculate_distance(grid)

        # Check all distances along X axis
        expected = list(range(N))
        actual = [gpu_dist[i, 0, 0] for i in range(N)]

        print(f"\nX-axis (shape {grid.shape}):")
        print(f"Expected: {expected}")
        print(f"Actual:   {actual}")

        assert actual == expected, f"X-axis distances incorrect"

    def test_y_axis_n20(self):
        """Test 1D distance along Y axis with N=20."""
        N = 20
        grid = np.zeros((1, N, 1), dtype=np.int32)
        grid[0, 0, 0] = 1  # Occupied at one end

        gpu_dist = calculate_distance(grid)

        # Check all distances along Y axis
        expected = list(range(N))
        actual = [gpu_dist[0, j, 0] for j in range(N)]

        print(f"\nY-axis (shape {grid.shape}):")
        print(f"Expected: {expected}")
        print(f"Actual:   {actual}")

        assert actual == expected, f"Y-axis distances incorrect"

    def test_z_axis_n20(self):
        """Test 1D distance along Z axis with N=20."""
        N = 20
        grid = np.zeros((1, 1, N), dtype=np.int32)
        grid[0, 0, 0] = 1  # Occupied at one end

        gpu_dist = calculate_distance(grid)

        # Check all distances along Z axis
        expected = list(range(N))
        actual = [gpu_dist[0, 0, k] for k in range(N)]

        print(f"\nZ-axis (shape {grid.shape}):")
        print(f"Expected: {expected}")
        print(f"Actual:   {actual}")

        assert actual == expected, f"Z-axis distances incorrect"

    def test_x_axis_n50(self):
        """Test 1D distance along X axis with N=50."""
        N = 50
        grid = np.zeros((N, 1, 1), dtype=np.int32)
        grid[0, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        far_dist = gpu_dist[N-1, 0, 0]

        print(f"\nX-axis N=50: far end distance = {far_dist}, expected = {N-1}")
        assert far_dist == N - 1

    def test_y_axis_n50(self):
        """Test 1D distance along Y axis with N=50."""
        N = 50
        grid = np.zeros((1, N, 1), dtype=np.int32)
        grid[0, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        far_dist = gpu_dist[0, N-1, 0]

        print(f"\nY-axis N=50: far end distance = {far_dist}, expected = {N-1}")
        assert far_dist == N - 1

    def test_z_axis_n50(self):
        """Test 1D distance along Z axis with N=50."""
        N = 50
        grid = np.zeros((1, 1, N), dtype=np.int32)
        grid[0, 0, 0] = 1

        gpu_dist = calculate_distance(grid)
        far_dist = gpu_dist[0, 0, N-1]

        print(f"\nZ-axis N=50: far end distance = {far_dist}, expected = {N-1}")
        assert far_dist == N - 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
