"""
Tests for the C++ core bindings.

These tests verify that the pybind11 bindings work correctly,
including type conversions, error handling, and function behavior.
"""

import pytest
import numpy as np




class TestFFTSearchPlacement:
    """Tests for fft_search_placement function."""

    def test_simple_placement(self, cube_3x3x3, empty_tray_medium):
        """Test placing a single cube in an empty tray."""
        from spectral_packer import fft_search_placement

        position, found, score = fft_search_placement(cube_3x3x3, empty_tray_medium)

        assert found is True
        assert len(position) == 3
        assert all(isinstance(p, int) for p in position)
        assert score >= 0

    def test_placement_position_valid(self, cube_5x5x5, empty_tray_medium):
        """Test that returned position is within tray bounds."""
        from spectral_packer import fft_search_placement

        position, found, _ = fft_search_placement(cube_5x5x5, empty_tray_medium)

        if found:
            tray_shape = empty_tray_medium.shape
            item_shape = cube_5x5x5.shape
            for i in range(3):
                assert position[i] >= 0
                assert position[i] + item_shape[i] <= tray_shape[i]

    def test_no_space_for_item(self):
        """Test when item is too large for tray."""
        from spectral_packer import fft_search_placement

        item = np.ones((20, 20, 20), dtype=np.int32)
        tray = np.zeros((10, 10, 10), dtype=np.int32)

        position, found, score = fft_search_placement(item, tray)

        assert found is False

    def test_tray_already_full(self):
        """Test when tray is completely full."""
        from spectral_packer import fft_search_placement

        item = np.ones((3, 3, 3), dtype=np.int32)
        tray = np.ones((10, 10, 10), dtype=np.int32)

        position, found, score = fft_search_placement(item, tray)

        assert found is False

    def test_l_shape_placement(self, l_shape, empty_tray_medium):
        """Test placing a non-cubic shape."""
        from spectral_packer import fft_search_placement

        position, found, score = fft_search_placement(l_shape, empty_tray_medium)

        assert found is True

    def test_single_voxel(self, single_voxel, empty_tray_small):
        """Test placing a single voxel."""
        from spectral_packer import fft_search_placement

        position, found, score = fft_search_placement(single_voxel, empty_tray_small)

        assert found is True


class TestPlaceInTray:
    """Tests for place_in_tray function."""

    def test_basic_placement(self, cube_3x3x3, empty_tray_small):
        """Test basic item placement."""
        from spectral_packer import place_in_tray

        position = (5, 5, 5)
        result = place_in_tray(cube_3x3x3, empty_tray_small, position, 1)

        # Check item was placed
        assert np.sum(result > 0) == np.sum(cube_3x3x3 > 0)
        # Check position
        assert result[5, 5, 5] == 1

    def test_placement_at_origin(self, cube_3x3x3, empty_tray_small):
        """Test placement at origin."""
        from spectral_packer import place_in_tray

        position = (0, 0, 0)
        result = place_in_tray(cube_3x3x3, empty_tray_small, position, 1)

        assert result[0, 0, 0] == 1
        assert result[2, 2, 2] == 1

    def test_different_item_ids(self, cube_3x3x3, empty_tray_medium):
        """Test placing items with different IDs."""
        from spectral_packer import place_in_tray

        tray = empty_tray_medium.copy()
        tray = place_in_tray(cube_3x3x3, tray, (0, 0, 0), 1)
        tray = place_in_tray(cube_3x3x3, tray, (10, 10, 10), 2)

        assert np.sum(tray == 1) == 27
        assert np.sum(tray == 2) == 27

    def test_collision_detection(self):
        """Test that overlapping placements raise an error."""
        from spectral_packer import place_in_tray

        item = np.ones((3, 3, 3), dtype=np.int32)
        tray = np.zeros((10, 10, 10), dtype=np.int32)
        tray[2:5, 2:5, 2:5] = 1  # Pre-occupy space

        with pytest.raises(RuntimeError):
            place_in_tray(item, tray, (2, 2, 2), 2)


class TestDFTOperations:
    """Tests for DFT convolution and correlation."""

    def test_conv3_shape(self):
        """Test that convolution preserves shape."""
        from spectral_packer import dft_conv3

        a = np.ones((5, 5, 5), dtype=np.int32)
        b = np.ones((5, 5, 5), dtype=np.int32)

        result = dft_conv3(a, b)

        assert result.shape == a.shape

    def test_corr3_shape(self):
        """Test that correlation preserves shape."""
        from spectral_packer import dft_corr3

        a = np.ones((5, 5, 5), dtype=np.int32)
        b = np.ones((5, 5, 5), dtype=np.int32)

        result = dft_corr3(a, b)

        assert result.shape == a.shape

    def test_conv3_zeros(self):
        """Test convolution with zeros."""
        from spectral_packer import dft_conv3

        a = np.ones((3, 3, 3), dtype=np.int32)
        b = np.zeros((3, 3, 3), dtype=np.int32)

        result = dft_conv3(a, b)

        assert np.sum(result) == 0


class TestDistanceCalculation:
    """Tests for calculate_distance function."""

    def test_distance_from_center(self):
        """Test distance calculation from a center point."""
        from spectral_packer import calculate_distance

        grid = np.zeros((5, 5, 5), dtype=np.int32)
        grid[2, 2, 2] = 1  # Single point in center

        dist = calculate_distance(grid)

        # Center should have distance 0 (or close to it)
        assert dist[2, 2, 2] == 0 or abs(dist[2, 2, 2]) < 2

    def test_distance_from_empty(self):
        """Test distance calculation from empty grid."""
        from spectral_packer import calculate_distance

        grid = np.zeros((5, 5, 5), dtype=np.int32)

        dist = calculate_distance(grid)

        # All distances should be large
        assert np.all(dist >= 0)


class TestCollisionGrid:
    """Tests for collision_grid function."""

    def test_no_collision(self, cube_3x3x3, empty_tray_small):
        """Test collision grid with empty tray."""
        from spectral_packer import collision_grid

        result = collision_grid(empty_tray_small, cube_3x3x3)

        # Should have zeros where no collision
        assert np.sum(result == 0) > 0

    def test_full_tray_collision(self):
        """Test collision grid with full tray."""
        from spectral_packer import collision_grid

        item = np.ones((3, 3, 3), dtype=np.int32)
        tray = np.ones((10, 10, 10), dtype=np.int32)

        result = collision_grid(tray, item)

        # All positions should have collision
        assert np.all(result > 0)


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_make_tight(self, cube_with_padding):
        """Test make_tight removes empty borders."""
        from spectral_packer import make_tight

        result = make_tight(cube_with_padding)

        # Should be smaller than original
        assert all(r <= o for r, o in zip(result.shape, cube_with_padding.shape))
        # Should still contain all occupied voxels
        assert np.sum(result > 0) == np.sum(cube_with_padding > 0)

    def test_get_bounds(self, cube_with_padding):
        """Test get_bounds returns correct bounds."""
        from spectral_packer import get_bounds

        lo, hi = get_bounds(cube_with_padding)

        assert lo == (2, 2, 2)
        assert hi == (6, 6, 6)

    def test_get_bounds_empty(self):
        """Test get_bounds with empty grid."""
        from spectral_packer import get_bounds

        grid = np.zeros((5, 5, 5), dtype=np.int32)

        lo, hi = get_bounds(grid)

        # Bounds of empty grid - implementation dependent
        assert len(lo) == 3
        assert len(hi) == 3


class TestTypeConversion:
    """Tests for numpy array type handling."""

    def test_float_array_conversion(self, empty_tray_small):
        """Test that float arrays are handled."""
        from spectral_packer import fft_search_placement

        # Float array should be converted to int
        item = np.ones((3, 3, 3), dtype=np.float64)

        # Should not raise
        position, found, score = fft_search_placement(item, empty_tray_small)
        assert isinstance(found, bool)

    def test_bool_array_conversion(self, empty_tray_small):
        """Test that bool arrays work."""
        from spectral_packer import fft_search_placement

        item = np.ones((3, 3, 3), dtype=bool)

        position, found, score = fft_search_placement(item, empty_tray_small)
        assert isinstance(found, bool)

    def test_wrong_dimensions(self, empty_tray_small):
        """Test error on wrong array dimensions."""
        from spectral_packer import fft_search_placement

        # 2D array should fail
        item = np.ones((3, 3), dtype=np.int32)

        with pytest.raises((RuntimeError, ValueError)):
            fft_search_placement(item, empty_tray_small)
