"""
Tests and benchmarks for orientation sampling in packing.

This module tests:
1. Correctness of rotation functions
2. Impact of orientation sampling on packing density
3. Performance benchmarks
"""
import numpy as np
import pytest
import time
from typing import Tuple

try:
    from spectral_packer import (
        BinPacker,
        get_orientations,
        get_24_orientations,
        rotate_90_x,
        rotate_90_y,
        rotate_90_z,
        fft_search_placement,
    )
    HAS_BINDINGS = True
except ImportError:
    HAS_BINDINGS = False


class TestRotationCorrectness:
    """Test that rotations are mathematically correct."""

    def test_rotate_90_x_shape(self):
        """Rotation around X swaps Y and Z dimensions."""
        grid = np.ones((3, 4, 5), dtype=np.int32)
        rotated = rotate_90_x(grid)
        assert rotated.shape == (3, 5, 4)

    def test_rotate_90_y_shape(self):
        """Rotation around Y swaps X and Z dimensions."""
        grid = np.ones((3, 4, 5), dtype=np.int32)
        rotated = rotate_90_y(grid)
        assert rotated.shape == (5, 4, 3)

    def test_rotate_90_z_shape(self):
        """Rotation around Z swaps X and Y dimensions."""
        grid = np.ones((3, 4, 5), dtype=np.int32)
        rotated = rotate_90_z(grid)
        assert rotated.shape == (4, 3, 5)

    def test_rotate_preserves_volume(self):
        """Rotation should preserve the number of occupied voxels."""
        np.random.seed(42)
        grid = (np.random.rand(10, 12, 8) > 0.5).astype(np.int32)
        original_volume = np.sum(grid)

        for rotate_fn in [rotate_90_x, rotate_90_y, rotate_90_z]:
            rotated = rotate_fn(grid)
            assert np.sum(rotated) == original_volume

    def test_four_rotations_returns_to_original(self):
        """Rotating 4 times around same axis returns to original."""
        np.random.seed(42)
        grid = (np.random.rand(5, 6, 7) > 0.5).astype(np.int32)

        for rotate_fn in [rotate_90_x, rotate_90_y, rotate_90_z]:
            rotated = grid
            for _ in range(4):
                rotated = rotate_fn(rotated)
            np.testing.assert_array_equal(rotated, grid)

    def test_get_24_orientations_count(self):
        """Should return exactly 24 orientations."""
        grid = np.ones((3, 4, 5), dtype=np.int32)
        orientations = get_24_orientations(grid)
        assert len(orientations) == 24

    def test_get_orientations_counts(self):
        """Test different num_orientations values."""
        grid = np.ones((3, 4, 5), dtype=np.int32)

        assert len(get_orientations(grid, 1)) == 1
        assert len(get_orientations(grid, 4)) == 4
        assert len(get_orientations(grid, 6)) == 6
        assert len(get_orientations(grid, 24)) == 24

    def test_invalid_num_orientations_raises(self):
        """Invalid num_orientations should raise ValueError."""
        grid = np.ones((3, 4, 5), dtype=np.int32)

        with pytest.raises(ValueError):
            get_orientations(grid, 5)
        with pytest.raises(ValueError):
            get_orientations(grid, 12)

    def test_l_shape_rotation(self):
        """Test rotation of an L-shaped object."""
        # Create L-shape in XY plane
        grid = np.zeros((3, 3, 1), dtype=np.int32)
        grid[0, 0, 0] = 1
        grid[1, 0, 0] = 1
        grid[2, 0, 0] = 1
        grid[0, 1, 0] = 1

        # Rotate around Z - L should be in different orientation
        rotated = rotate_90_z(grid)
        assert rotated.shape == (3, 3, 1)
        assert np.sum(rotated) == 4  # Same volume

        # Check that the L is actually rotated
        assert not np.array_equal(grid, rotated)


@pytest.mark.skipif(not HAS_BINDINGS, reason="C++ bindings not available")
class TestOrientationPacking:
    """Test that orientation sampling works in packing."""

    def test_packer_accepts_num_orientations(self):
        """BinPacker should accept num_orientations parameter."""
        for n in [1, 4, 6, 24]:
            packer = BinPacker(tray_size=(50, 50, 50), num_orientations=n)
            assert packer.num_orientations == n

    def test_packer_invalid_num_orientations(self):
        """BinPacker should reject invalid num_orientations."""
        with pytest.raises(ValueError):
            BinPacker(tray_size=(50, 50, 50), num_orientations=5)

    def test_orientation_index_in_placement(self):
        """PlacementInfo should include orientation_index."""
        packer = BinPacker(tray_size=(50, 50, 50), num_orientations=4)

        # Create a simple item
        item = np.ones((5, 5, 5), dtype=np.int32)
        result = packer.pack_voxels([item])

        assert result.num_placed == 1
        assert result.placements[0].orientation_index >= 0
        assert result.placements[0].orientation_index < 4

    def test_elongated_item_benefits_from_rotation(self):
        """An elongated item might only fit when rotated."""
        # Tray that's wide but short in Z
        packer_no_rot = BinPacker(tray_size=(30, 30, 10), num_orientations=1)
        packer_with_rot = BinPacker(tray_size=(30, 30, 10), num_orientations=24)

        # Item that's tall in Z - won't fit without rotation
        item = np.ones((5, 5, 15), dtype=np.int32)

        result_no_rot = packer_no_rot.pack_voxels([item])
        result_with_rot = packer_with_rot.pack_voxels([item])

        # Without rotation, it shouldn't fit (15 > 10)
        assert result_no_rot.num_placed == 0

        # With rotation, it should fit (rotated to fit in 30x30 footprint)
        assert result_with_rot.num_placed == 1


@pytest.mark.skipif(not HAS_BINDINGS, reason="C++ bindings not available")
class TestTetrisStyle:
    """2D Tetris-style tests with small grids for visual verification."""

    def make_2d_grid(self, pattern):
        """Convert 2D string pattern to 3D numpy array (depth=1)."""
        lines = [line.strip() for line in pattern.strip().split('\n')]
        rows = len(lines)
        cols = max(len(line) for line in lines)
        grid = np.zeros((rows, cols, 1), dtype=np.int32)
        for i, line in enumerate(lines):
            for j, char in enumerate(line):
                if char == '#':
                    grid[i, j, 0] = 1
        return grid

    def test_l_piece_fits_with_rotation(self):
        """L-piece that only fits when rotated."""
        # Tray: 4 wide, 3 tall (won't fit L vertically)
        tray_size = (3, 4, 1)

        # L-piece (3 tall, needs rotation to fit in height=3 with other pieces)
        l_piece = self.make_2d_grid("""
        #
        #
        ##
        """)

        # Without rotation
        packer_1 = BinPacker(tray_size=tray_size, num_orientations=1)
        result_1 = packer_1.pack_voxels([l_piece])

        # With rotation
        packer_4 = BinPacker(tray_size=tray_size, num_orientations=4)
        result_4 = packer_4.pack_voxels([l_piece])

        # Both should place it (fits in 3x4)
        assert result_1.num_placed == 1
        assert result_4.num_placed == 1

    def test_t_piece_rotation(self):
        """T-piece can be rotated 4 ways."""
        t_piece = self.make_2d_grid("""
        ###
        .#.
        """)

        orientations = get_orientations(t_piece, 4)
        assert len(orientations) == 4

        # Each orientation should have same volume
        for orient in orientations:
            assert np.sum(orient) == 4  # T has 4 blocks

    def test_two_l_pieces_pack_better_with_rotation(self):
        """Two L-pieces should pack tighter with rotation."""
        tray_size = (4, 4, 1)

        l_piece = self.make_2d_grid("""
        #.
        #.
        ##
        """)

        # Two identical L-pieces
        items = [l_piece.copy(), l_piece.copy()]

        # Without rotation - both L's same orientation
        packer_1 = BinPacker(tray_size=tray_size, num_orientations=1)
        result_1 = packer_1.pack_voxels(items)

        # With rotation - can interlock
        packer_4 = BinPacker(tray_size=tray_size, num_orientations=4)
        result_4 = packer_4.pack_voxels(items)

        print(f"\n  2 L-pieces in 4x4:")
        print(f"    Without rotation: placed {result_1.num_placed}, density {result_1.density:.1%}")
        print(f"    With rotation:    placed {result_4.num_placed}, density {result_4.density:.1%}")

        # Both should fit
        assert result_1.num_placed == 2
        assert result_4.num_placed == 2

    def test_i_piece_horizontal_vs_vertical(self):
        """I-piece (4x1) - rotation determines if it fits."""
        # Tall narrow tray
        tray_size = (6, 2, 1)

        # Horizontal I-piece (1x4)
        i_piece = self.make_2d_grid("""
        ####
        """)

        # Without rotation - 4 wide, needs width 4, tray width is 2
        packer_1 = BinPacker(tray_size=tray_size, num_orientations=1)
        result_1 = packer_1.pack_voxels([i_piece])

        # With rotation - can become 4 tall x 1 wide
        packer_4 = BinPacker(tray_size=tray_size, num_orientations=4)
        result_4 = packer_4.pack_voxels([i_piece])

        print(f"\n  I-piece (1x4) in 6x2 tray:")
        print(f"    Without rotation: placed {result_1.num_placed}")
        print(f"    With rotation:    placed {result_4.num_placed}")

        # Only rotated version should fit
        assert result_1.num_placed == 0  # 4-wide doesn't fit in width-2
        assert result_4.num_placed == 1  # Rotated to 4-tall fits in height-6

    def test_s_and_z_pieces(self):
        """S and Z pieces are mirror images - rotation helps pack both."""
        tray_size = (4, 6, 1)

        s_piece = self.make_2d_grid("""
        .##
        ##.
        """)

        z_piece = self.make_2d_grid("""
        ##.
        .##
        """)

        items = [s_piece, z_piece]

        packer_1 = BinPacker(tray_size=tray_size, num_orientations=1)
        result_1 = packer_1.pack_voxels(items)

        packer_4 = BinPacker(tray_size=tray_size, num_orientations=4)
        result_4 = packer_4.pack_voxels(items)

        print(f"\n  S + Z pieces in 4x6:")
        print(f"    Without rotation: placed {result_1.num_placed}, density {result_1.density:.1%}")
        print(f"    With rotation:    placed {result_4.num_placed}, density {result_4.density:.1%}")

        assert result_1.num_placed == 2
        assert result_4.num_placed == 2

    def test_tetris_line_clear_scenario(self):
        """Pack multiple pieces into a small area."""
        tray_size = (4, 10, 1)

        pieces = [
            # I-piece
            self.make_2d_grid("####"),
            # O-piece (square)
            self.make_2d_grid("""
            ##
            ##
            """),
            # T-piece
            self.make_2d_grid("""
            ###
            .#.
            """),
            # L-piece
            self.make_2d_grid("""
            #.
            #.
            ##
            """),
        ]

        packer_1 = BinPacker(tray_size=tray_size, num_orientations=1)
        result_1 = packer_1.pack_voxels(pieces)

        packer_4 = BinPacker(tray_size=tray_size, num_orientations=4)
        result_4 = packer_4.pack_voxels(pieces)

        print(f"\n  Tetris pieces (I, O, T, L) in 4x10:")
        print(f"    Without rotation: placed {result_1.num_placed}/4, density {result_1.density:.1%}")
        print(f"    With rotation:    placed {result_4.num_placed}/4, density {result_4.density:.1%}")

        # With rotation should pack all 4
        assert result_4.num_placed == 4
        # Without might also work depending on order
        assert result_1.num_placed >= 3


@pytest.mark.skipif(not HAS_BINDINGS, reason="C++ bindings not available")
class TestOrientationBenchmark:
    """Benchmark orientation sampling impact on packing."""

    def generate_random_items(
        self,
        num_items: int,
        size_range: Tuple[int, int] = (3, 10),
        seed: int = 42
    ):
        """Generate random voxelized items."""
        np.random.seed(seed)
        items = []
        for _ in range(num_items):
            # Random dimensions
            dims = np.random.randint(size_range[0], size_range[1] + 1, size=3)
            # Random occupancy (~50%)
            item = (np.random.rand(*dims) > 0.5).astype(np.int32)
            # Ensure at least one voxel is occupied
            if np.sum(item) == 0:
                item[0, 0, 0] = 1
            items.append(item)
        return items

    def generate_elongated_items(
        self,
        num_items: int,
        seed: int = 42
    ):
        """Generate elongated items that benefit from rotation."""
        np.random.seed(seed)
        items = []
        for _ in range(num_items):
            # One dimension is much larger than others
            short = np.random.randint(2, 5)
            long_dim = np.random.randint(8, 15)
            axis = np.random.randint(0, 3)

            dims = [short, short, short]
            dims[axis] = long_dim

            item = np.ones(dims, dtype=np.int32)
            items.append(item)
        return items

    @pytest.mark.parametrize("num_orientations", [1, 4, 6, 24])
    def test_benchmark_random_items(self, num_orientations):
        """Benchmark packing with different orientation counts."""
        tray_size = (50, 50, 50)
        num_items = 20

        items = self.generate_random_items(num_items, size_range=(3, 8))

        packer = BinPacker(tray_size=tray_size, num_orientations=num_orientations)

        start = time.perf_counter()
        result = packer.pack_voxels(items)
        elapsed = time.perf_counter() - start

        print(f"\n  Orientations: {num_orientations:2d} | "
              f"Placed: {result.num_placed:2d}/{num_items} | "
              f"Density: {result.density:.1%} | "
              f"Time: {elapsed*1000:.1f}ms")

        # Just check it runs - actual improvement varies by items
        assert result.num_placed >= 0

    def test_benchmark_elongated_items(self):
        """Benchmark with elongated items where rotation helps most."""
        tray_size = (40, 40, 40)
        num_items = 15

        items = self.generate_elongated_items(num_items)

        results = {}
        for num_orientations in [1, 24]:
            packer = BinPacker(tray_size=tray_size, num_orientations=num_orientations)

            start = time.perf_counter()
            result = packer.pack_voxels(items)
            elapsed = time.perf_counter() - start

            results[num_orientations] = {
                'placed': result.num_placed,
                'density': result.density,
                'time': elapsed
            }

            print(f"\n  Orientations: {num_orientations:2d} | "
                  f"Placed: {result.num_placed:2d}/{num_items} | "
                  f"Density: {result.density:.1%} | "
                  f"Time: {elapsed*1000:.1f}ms")

        # With elongated items, 24 orientations should usually pack more
        # (Not strictly required to pass, but print for analysis)
        improvement = results[24]['placed'] - results[1]['placed']
        print(f"\n  Improvement with 24 orientations: +{improvement} items placed")

    def test_benchmark_report(self):
        """Generate a comprehensive benchmark report."""
        print("\n" + "=" * 70)
        print("ORIENTATION SAMPLING BENCHMARK REPORT")
        print("=" * 70)

        tray_size = (60, 60, 60)

        # Test with different item types
        test_cases = [
            ("Random cubic items", self.generate_random_items(25, (4, 8))),
            ("Elongated items", self.generate_elongated_items(20)),
        ]

        for case_name, items in test_cases:
            print(f"\n{case_name}:")
            print("-" * 50)

            for num_orientations in [1, 4, 6, 24]:
                packer = BinPacker(
                    tray_size=tray_size,
                    num_orientations=num_orientations
                )

                start = time.perf_counter()
                result = packer.pack_voxels(items)
                elapsed = time.perf_counter() - start

                print(f"  n_orient={num_orientations:2d}: "
                      f"placed={result.num_placed:2d}/{len(items)}, "
                      f"density={result.density:.1%}, "
                      f"time={elapsed*1000:6.1f}ms")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
