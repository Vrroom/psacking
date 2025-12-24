"""
Tests for core FFT operations, ported from C++ run_tests().

These tests verify the correctness of the fundamental FFT-based operations
used in spectral packing: convolution, correlation, and collision detection.
"""

import pytest
import numpy as np


pytestmark = pytest.mark.skipif(
    not pytest.importorskip("spectral_packer")._CORE_AVAILABLE,
    reason="CUDA core module not available"
)


class TestFFTConvolution:
    """Tests for dft_conv3 (3D convolution via FFT)."""

    def test_conv3_1d_case_1(self):
        """Test 1D convolution: [1,1,1] * [1,2,3] = [1,3,6]."""
        from spectral_packer import dft_conv3

        a = np.array([[[1, 1, 1]]], dtype=np.int32)
        b = np.array([[[1, 2, 3]]], dtype=np.int32)
        expected = np.array([[[1, 3, 6]]], dtype=np.int32)

        result = dft_conv3(a, b)

        np.testing.assert_array_equal(result, expected)

    def test_conv3_1d_case_2(self):
        """Test 1D convolution along different axis."""
        from spectral_packer import dft_conv3

        a = np.array([[[1], [11], [2], [5]]], dtype=np.int32)
        b = np.array([[[12], [12], [6], [1]]], dtype=np.int32)
        expected = np.array([[[12], [144], [162], [151]]], dtype=np.int32)

        result = dft_conv3(a, b)

        np.testing.assert_array_equal(result, expected)

    def test_conv3_1d_case_3(self):
        """Test 1D convolution with zeros."""
        from spectral_packer import dft_conv3

        a = np.array([[[1]], [[0]], [[1]], [[0]]], dtype=np.int32)
        b = np.array([[[1]], [[8]], [[12]], [[4]]], dtype=np.int32)
        expected = np.array([[[1]], [[8]], [[13]], [[12]]], dtype=np.int32)

        result = dft_conv3(a, b)

        np.testing.assert_array_equal(result, expected)


class TestFFTCorrelation:
    """Tests for dft_corr3 (3D cross-correlation via FFT)."""

    def test_corr3_1d_case_1(self):
        """Test 1D correlation: corr([1,1,1], [1,2,3]) = [6,3,1]."""
        from spectral_packer import dft_corr3

        a = np.array([[[1, 1, 1]]], dtype=np.int32)
        b = np.array([[[1, 2, 3]]], dtype=np.int32)
        expected = np.array([[[6, 3, 1]]], dtype=np.int32)

        result = dft_corr3(a, b)

        np.testing.assert_array_equal(result, expected)

    def test_corr3_1d_case_2(self):
        """Test 1D correlation along different axis."""
        from spectral_packer import dft_corr3

        a = np.array([[[1], [11], [2], [5]]], dtype=np.int32)
        b = np.array([[[12], [12], [6], [1]]], dtype=np.int32)
        expected = np.array([[[161], [186], [84], [60]]], dtype=np.int32)

        result = dft_corr3(a, b)

        np.testing.assert_array_equal(result, expected)

    def test_corr3_1d_case_3(self):
        """Test 1D correlation with zeros."""
        from spectral_packer import dft_corr3

        a = np.array([[[1]], [[0]], [[1]], [[0]]], dtype=np.int32)
        b = np.array([[[1]], [[8]], [[12]], [[4]]], dtype=np.int32)
        expected = np.array([[[13]], [[8]], [[1]], [[0]]], dtype=np.int32)

        result = dft_corr3(a, b)

        np.testing.assert_array_equal(result, expected)


class TestDistanceField:
    """Tests for calculate_distance (distance transform)."""

    def test_distance_3x3x3_center(self):
        """Test distance from center point in 3x3x3 grid."""
        from spectral_packer import calculate_distance

        # Single occupied voxel at center
        tray = np.zeros((3, 3, 3), dtype=np.int32)
        tray[1, 1, 1] = 1

        # Expected Manhattan-style distance field
        expected = np.array([
            [[2, 1, 2], [1, 0, 1], [2, 1, 2]],
            [[1, 0, 1], [0, 0, 0], [1, 0, 1]],  # Center layer
            [[2, 1, 2], [1, 0, 1], [2, 1, 2]]
        ], dtype=np.int32)

        result = calculate_distance(tray)

        # Check center is 0
        assert result[1, 1, 1] == 0
        # Check corners have distance >= 2
        assert result[0, 0, 0] >= 2
        assert result[2, 2, 2] >= 2

    def test_distance_1x3x3_center(self):
        """Test distance from center in 1x3x3 grid."""
        from spectral_packer import calculate_distance

        tray = np.zeros((3, 1, 3), dtype=np.int32)
        tray[1, 0, 1] = 1

        expected = np.array([
            [[2, 1, 2]],
            [[1, 0, 1]],
            [[2, 1, 2]]
        ], dtype=np.int32)

        result = calculate_distance(tray)

        np.testing.assert_array_equal(result, expected)

    def test_distance_1x1x4_off_center(self):
        """Test distance with off-center point."""
        from spectral_packer import calculate_distance

        tray = np.zeros((3, 1, 4), dtype=np.int32)
        tray[1, 0, 1] = 1

        expected = np.array([
            [[2, 1, 2, 3]],
            [[1, 0, 1, 2]],
            [[2, 1, 2, 3]]
        ], dtype=np.int32)

        result = calculate_distance(tray)

        np.testing.assert_array_equal(result, expected)


class TestCollisionGrid:
    """Tests for collision_grid function."""

    def test_collision_grid_simple(self):
        """Test collision grid with simple overlapping shapes."""
        from spectral_packer import collision_grid

        # Tray with some occupied voxels
        a = np.zeros((4, 1, 4), dtype=np.int32)
        a[0, 0, 1] = 1
        a[1, 0, 0] = 1
        a[2, 0, 0] = 1
        a[2, 0, 1] = 1
        a[2, 0, 3] = 1

        # Item pattern
        b = np.zeros((4, 1, 4), dtype=np.int32)
        b[0, 0, 0] = 1
        b[0, 0, 1] = 1
        b[0, 0, 2] = 1
        b[1, 0, 1] = 1

        result = collision_grid(a, b)

        # At origin, placing b would overlap with a at (0,0,1)
        assert result[0, 0, 0] > 0  # Collision

    def test_collision_grid_no_overlap(self):
        """Test collision grid when item fits."""
        from spectral_packer import collision_grid

        # Empty tray
        tray = np.zeros((10, 10, 10), dtype=np.int32)
        item = np.ones((3, 3, 3), dtype=np.int32)

        result = collision_grid(tray, item)

        # Should have many valid positions (collision = 0)
        assert np.sum(result == 0) > 0


class TestSearchPlacement:
    """Tests for fft_search_placement function."""

    def test_simple_placement_in_small_tray(self):
        """Test placing a small item in a partially occupied tray."""
        from spectral_packer import fft_search_placement, place_in_tray

        # Item: vertical bar
        item = np.ones((3, 1, 1), dtype=np.int32)

        # Tray with some occupied corners
        tray = np.zeros((3, 3, 3), dtype=np.int32)
        tray[0, 1, 0] = 1
        tray[0, 2, 0] = 1
        tray[1, 2, 0] = 1

        position, found, score = fft_search_placement(item, tray)

        assert found is True

        # Verify item can actually be placed there
        result = place_in_tray(item, tray, position, 2)
        assert np.sum(result == 2) == np.sum(item > 0)

    def test_multiple_placements(self):
        """Test sequential placement of multiple items."""
        from spectral_packer import fft_search_placement, place_in_tray

        tray = np.zeros((3, 3, 3), dtype=np.int32)

        items = [
            np.ones((3, 1, 1), dtype=np.int32),  # Vertical bar
            np.ones((1, 1, 2), dtype=np.int32),  # Small L
            np.ones((1, 1, 1), dtype=np.int32),  # Single voxel
        ]

        placed = 0
        for i, item in enumerate(items, start=1):
            position, found, score = fft_search_placement(item, tray)
            if found:
                tray = place_in_tray(item, tray, position, i)
                placed += 1

        # Should place at least the first item
        assert placed >= 1


class TestVoxelGridOperations:
    """Tests for voxel grid utility functions."""

    def test_make_tight(self):
        """Test that make_tight removes empty borders correctly."""
        from spectral_packer import make_tight

        # Grid with padding
        grid = np.zeros((5, 5, 5), dtype=np.int32)
        grid[1:3, 1:3, 1:4] = 1

        result = make_tight(grid)

        # Should remove the empty border
        assert result.shape == (2, 2, 3)
        assert np.sum(result) == np.sum(grid)

    def test_get_bounds(self):
        """Test get_bounds returns correct bounding box."""
        from spectral_packer import get_bounds

        grid = np.zeros((5, 5, 5), dtype=np.int32)
        grid[1:3, 0:2, 1:4] = 1

        lo, hi = get_bounds(grid)

        assert lo == (1, 0, 1)
        assert hi == (2, 1, 3)

    def test_get_bounds_non_contiguous(self):
        """Test get_bounds with non-contiguous occupied voxels."""
        from spectral_packer import get_bounds

        grid = np.zeros((5, 5, 5), dtype=np.int32)
        grid[0, 0, 1] = 1
        grid[0, 1, 0] = 1
        grid[0, 1, 1] = 1
        grid[1, 0, 1] = 1
        grid[1, 1, 0] = 1
        grid[1, 1, 1] = 1
        grid[2, 0, 0] = 1

        lo, hi = get_bounds(grid)

        assert lo == (0, 0, 0)
        assert hi == (2, 1, 1)


class TestFFTSearchWithCache:
    """Tests for fft_search_placement_with_cache (GPU-accelerated path)."""

    def test_cache_consistency(self):
        """Test that cached and non-cached search give same results."""
        from spectral_packer import (
            fft_search_placement,
            fft_search_placement_with_cache,
            calculate_distance
        )

        item = np.ones((3, 3, 3), dtype=np.int32)
        tray = np.zeros((20, 20, 20), dtype=np.int32)

        # Non-cached path
        pos1, found1, score1 = fft_search_placement(item, tray)

        # Cached path
        tray_phi = calculate_distance(tray)
        pos2, found2, score2 = fft_search_placement_with_cache(
            item, tray, tray_phi, generation=0
        )

        assert found1 == found2
        # Positions might differ if scores are equal, but both should be valid
        if found1 and found2:
            assert pos1[0] >= 0 and pos2[0] >= 0


class TestFFTSearchBatch:
    """Tests for fft_search_batch (batch orientation search).

    Note: Each test uses a unique generation number to ensure the GPU cache
    is properly invalidated between tests. The cache is global and persists
    across test functions.
    """
    # Counter to ensure unique generation numbers across tests
    _gen = 1000

    @classmethod
    def next_gen(cls):
        cls._gen += 1
        return cls._gen

    def test_single_orientation_matches_cached_search(self):
        """Batch search with one orientation should match single cached search."""
        from spectral_packer import (
            fft_search_placement_with_cache,
            fft_search_batch,
            calculate_distance
        )

        item = np.ones((3, 3, 3), dtype=np.int32)
        tray = np.zeros((20, 20, 20), dtype=np.int32)
        tray_phi = calculate_distance(tray)

        gen = self.next_gen()

        # Single cached search
        pos1, found1, score1 = fft_search_placement_with_cache(
            item, tray, tray_phi, generation=gen
        )

        # Batch search with single orientation (same tray, new gen to force refresh)
        pos2, found2, score2 = fft_search_batch(
            [item], tray, tray_phi, generation=gen + 1
        )

        assert found1 == found2
        assert score1 == score2
        if found1:
            assert pos1 == pos2

    def test_multiple_orientations_finds_best(self):
        """Batch search should find the best placement across orientations."""
        from spectral_packer import fft_search_batch, calculate_distance

        # Create an L-shaped item
        item = np.zeros((3, 3, 3), dtype=np.int32)
        item[0, :, 0] = 1  # Vertical bar
        item[0, 0, :] = 1  # Horizontal bar

        # Create rotated versions (90 degree rotations)
        orientations = [
            item,
            np.rot90(item, k=1, axes=(0, 1)),
            np.rot90(item, k=2, axes=(0, 1)),
            np.rot90(item, k=3, axes=(0, 1)),
        ]

        tray = np.zeros((10, 10, 10), dtype=np.int32)
        tray_phi = calculate_distance(tray)

        pos, found, score = fft_search_batch(
            orientations, tray, tray_phi, generation=self.next_gen()
        )

        assert found is True
        # Position should be valid
        assert all(p >= 0 for p in pos)

    def test_batch_with_partially_filled_tray(self):
        """Batch search should work with obstacles in the tray."""
        from spectral_packer import fft_search_batch, calculate_distance, place_in_tray

        item = np.ones((2, 2, 2), dtype=np.int32)
        orientations = [item]  # Cube has same shape in all orientations

        tray = np.zeros((10, 10, 10), dtype=np.int32)
        # Fill bottom corner
        tray[0:3, 0:3, 0:3] = 1
        tray_phi = calculate_distance(tray)

        pos, found, score = fft_search_batch(
            orientations, tray, tray_phi, generation=self.next_gen()
        )

        assert found is True

        # Verify no collision by checking the placed region doesn't overlap
        x, y, z = pos
        item_region = tray[x:x+2, y:y+2, z:z+2]
        assert np.sum(item_region) == 0, f"Collision at position {pos}"

        # Verify placement works
        result = place_in_tray(item, tray, pos, 2)
        assert np.sum(result == 2) == np.sum(item)

    def test_batch_item_too_large(self):
        """Batch search should return not found when item exceeds tray."""
        from spectral_packer import fft_search_batch, calculate_distance

        # Item larger than tray
        item = np.ones((15, 15, 15), dtype=np.int32)
        orientations = [item]

        tray = np.zeros((10, 10, 10), dtype=np.int32)
        tray_phi = calculate_distance(tray)

        pos, found, score = fft_search_batch(
            orientations, tray, tray_phi, generation=self.next_gen()
        )

        assert found is False

    def test_batch_no_valid_placement(self):
        """Batch search should return not found when tray is full."""
        from spectral_packer import fft_search_batch, calculate_distance

        item = np.ones((3, 3, 3), dtype=np.int32)
        orientations = [item]

        # Completely full tray
        tray = np.ones((5, 5, 5), dtype=np.int32)
        tray_phi = calculate_distance(tray)

        pos, found, score = fft_search_batch(
            orientations, tray, tray_phi, generation=self.next_gen()
        )

        assert found is False

    def test_batch_generation_cache_invalidation(self):
        """Generation counter should trigger cache refresh when tray changes."""
        from spectral_packer import fft_search_batch, calculate_distance, place_in_tray

        item = np.ones((3, 3, 3), dtype=np.int32)
        orientations = [item]

        tray = np.zeros((15, 15, 15), dtype=np.int32)
        tray_phi = calculate_distance(tray)

        gen = self.next_gen()

        # First placement
        pos1, found1, score1 = fft_search_batch(
            orientations, tray, tray_phi, generation=gen
        )
        assert found1 is True

        # Place the item
        tray = place_in_tray(item, tray, pos1, 1)
        tray_phi = calculate_distance(tray)

        # Second placement with NEW generation (cache should refresh)
        pos2, found2, score2 = fft_search_batch(
            orientations, tray, tray_phi, generation=gen + 1
        )
        assert found2 is True
        # Should find different position (first spot is now occupied)
        assert pos1 != pos2

    def test_batch_sequential_packing(self):
        """Test packing multiple items sequentially using batch search."""
        from spectral_packer import fft_search_batch, calculate_distance, place_in_tray

        tray = np.zeros((20, 20, 20), dtype=np.int32)

        items = [
            np.ones((4, 4, 4), dtype=np.int32),
            np.ones((3, 3, 3), dtype=np.int32),
            np.ones((2, 2, 2), dtype=np.int32),
            np.ones((2, 2, 2), dtype=np.int32),
        ]

        placed = 0
        generation = self.next_gen()

        for i, item in enumerate(items, start=1):
            tray_phi = calculate_distance(tray)
            orientations = [item]  # Cubes look the same in all orientations

            pos, found, score = fft_search_batch(
                orientations, tray, tray_phi, generation=generation
            )

            if found:
                tray = place_in_tray(item, tray, pos, i)
                placed += 1
                generation += 1

        # Should place all items
        assert placed == len(items)
        # Verify total voxels placed
        total_voxels = sum(np.sum(item) for item in items)
        assert np.sum(tray > 0) == total_voxels

    def test_batch_prefers_lower_position(self):
        """Batch search should prefer placements with lower z (height penalty)."""
        from spectral_packer import fft_search_batch, calculate_distance

        item = np.ones((2, 2, 2), dtype=np.int32)
        orientations = [item]

        # Empty tray
        tray = np.zeros((10, 10, 10), dtype=np.int32)
        tray_phi = calculate_distance(tray)

        pos, found, score = fft_search_batch(
            orientations, tray, tray_phi, generation=self.next_gen()
        )

        assert found is True
        # Should prefer z=0 (lowest position) due to height penalty
        assert pos[2] == 0

    def test_tetris_bricks_various_lengths(self):
        """Pack bricks of various lengths into a larger grid (tetris-style)."""
        from spectral_packer import BinPacker

        # Create bricks of various lengths (all 2 units wide, 1 unit tall)
        bricks = []
        for length in [2, 3, 4, 5, 6, 4, 3, 2, 5, 4, 3, 6, 2, 4, 5, 3]:
            brick = np.ones((length, 2, 1), dtype=np.int32)
            bricks.append(brick)

        # Use BinPacker with larger tray
        packer = BinPacker(tray_size=(32, 32, 16), num_orientations=4)
        result = packer.pack_voxels(bricks, sort_by_volume=False)

        # Should place all bricks
        assert result.num_placed == len(bricks), \
            f"Only placed {result.num_placed} of {len(bricks)} bricks"

        # Verify total voxels
        total_voxels = sum(np.sum(b) for b in bricks)
        assert np.sum(result.tray > 0) == total_voxels

    def test_tetris_mixed_shapes(self):
        """Pack a mix of tetris-like shapes into a grid."""
        from spectral_packer import BinPacker

        # Classic tetris-like shapes (flat, z=1)
        shapes = []

        # I-piece (4x1)
        i_piece = np.ones((4, 1, 1), dtype=np.int32)
        shapes.extend([i_piece] * 4)

        # O-piece (2x2)
        o_piece = np.ones((2, 2, 1), dtype=np.int32)
        shapes.extend([o_piece] * 4)

        # L-piece
        l_piece = np.zeros((3, 2, 1), dtype=np.int32)
        l_piece[0:3, 0, 0] = 1
        l_piece[2, 1, 0] = 1
        shapes.extend([l_piece] * 3)

        # T-piece
        t_piece = np.zeros((3, 2, 1), dtype=np.int32)
        t_piece[0:3, 0, 0] = 1
        t_piece[1, 1, 0] = 1
        shapes.extend([t_piece] * 3)

        # S-piece
        s_piece = np.zeros((3, 2, 1), dtype=np.int32)
        s_piece[1:3, 0, 0] = 1
        s_piece[0:2, 1, 0] = 1
        shapes.extend([s_piece] * 2)

        # Use BinPacker with 4 orientations (rotations in xy plane)
        packer = BinPacker(tray_size=(40, 40, 10), num_orientations=4)
        result = packer.pack_voxels(shapes, sort_by_volume=False)

        # Should place all shapes in this spacious tray
        assert result.num_placed == len(shapes), \
            f"Only placed {result.num_placed} of {len(shapes)} shapes"

        # Verify total voxels
        total_voxels = sum(np.sum(s) for s in shapes)
        assert np.sum(result.tray > 0) == total_voxels
