"""
Visual tests for packing - generates PNG images to verify correctness.

These tests create actual visualizations of packing results that can be
inspected manually to verify the algorithm is working correctly.
"""
import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

from spectral_packer import BinPacker, fft_search_placement, place_in_tray

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def make_2d_piece(pattern):
    """Convert ASCII pattern to 3D numpy array (depth=1)."""
    lines = [line for line in pattern.strip().split('\n')]
    # Find the actual content bounds
    rows = len(lines)
    cols = max(len(line) for line in lines) if lines else 0

    grid = np.zeros((rows, cols, 1), dtype=np.int32)
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char == '#':
                grid[i, j, 0] = 1
    return grid


def get_tetris_pieces():
    """Return all 7 standard Tetris pieces."""
    pieces = {
        'I': make_2d_piece("####"),
        'O': make_2d_piece("""
##
##
"""),
        'T': make_2d_piece("""
###
.#.
"""),
        'S': make_2d_piece("""
.##
##.
"""),
        'Z': make_2d_piece("""
##.
.##
"""),
        'J': make_2d_piece("""
#..
###
"""),
        'L': make_2d_piece("""
..#
###
"""),
    }
    return pieces


def visualize_2d_tray(tray, title="Packing Result", filename=None):
    """
    Visualize a 2D tray (or 3D with depth=1) as a colored grid.

    Each unique item ID gets a different color.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping visualization")
        return None

    # Handle 3D array with depth=1
    if tray.ndim == 3:
        grid = tray[:, :, 0]
    else:
        grid = tray

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Get unique values (0 = empty, 1+ = different items)
    unique_vals = np.unique(grid)
    n_items = len(unique_vals) - 1  # Exclude 0 (empty)

    # Create colormap: white for empty, distinct colors for items
    colors = ['white']  # 0 = empty
    color_cycle = plt.cm.tab10.colors  # 10 distinct colors
    for i in range(max(1, int(grid.max()))):
        colors.append(color_cycle[i % len(color_cycle)])
    cmap = ListedColormap(colors[:int(grid.max()) + 1])

    # Plot the grid
    im = ax.imshow(grid, cmap=cmap, interpolation='nearest',
                   vmin=0, vmax=max(1, grid.max()))

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

    # Add cell values
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid[i, j]
            if val > 0:
                ax.text(j, i, str(int(val)), ha='center', va='center',
                       fontsize=8, fontweight='bold', color='white')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Add legend
    legend_text = f"Items placed: {n_items}\n"
    legend_text += f"Grid size: {grid.shape[0]}x{grid.shape[1]}\n"
    occupied = np.sum(grid > 0)
    total = grid.size
    legend_text += f"Density: {occupied}/{total} = {100*occupied/total:.1f}%"
    ax.text(1.02, 0.98, legend_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n{'='*60}")
        print(f"SAVED VISUALIZATION: {filename}")
        print(f"{'='*60}\n")

    plt.close()
    return filename


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
class TestVisualPacking:
    """Visual tests that generate PNG images for manual inspection."""

    def test_all_tetris_pieces_no_rotation(self):
        """Pack all 7 Tetris pieces WITHOUT rotation and visualize."""
        tray_size = (10, 20, 1)
        packer = BinPacker(tray_size=tray_size, num_orientations=1)

        pieces = get_tetris_pieces()
        items = list(pieces.values())

        result = packer.pack_voxels(items, sort_by_volume=False)

        # Save visualization
        with tempfile.NamedTemporaryFile(suffix='_no_rotation.png', delete=False,
                                          dir='/tmp') as f:
            filename = f.name

        visualize_2d_tray(
            result.tray,
            title=f"Tetris Pieces - NO ROTATION\n"
                  f"Placed: {result.num_placed}/7, Density: {result.density:.1%}",
            filename=filename
        )

        print(f"\nPieces: {list(pieces.keys())}")
        print(f"Placed: {result.num_placed}/7")
        print(f"Density: {result.density:.1%}")
        for p in result.placements:
            status = "PLACED" if p.success else "FAILED"
            print(f"  Item {p.item_index}: {status} at {p.position}, orientation={p.orientation_index}")

        assert os.path.exists(filename)
        assert result.num_placed >= 5  # Should place most pieces

    def test_all_tetris_pieces_with_rotation(self):
        """Pack all 7 Tetris pieces WITH rotation and visualize."""
        tray_size = (10, 20, 1)
        packer = BinPacker(tray_size=tray_size, num_orientations=4)

        pieces = get_tetris_pieces()
        items = list(pieces.values())

        result = packer.pack_voxels(items, sort_by_volume=False)

        # Save visualization
        with tempfile.NamedTemporaryFile(suffix='_with_rotation.png', delete=False,
                                          dir='/tmp') as f:
            filename = f.name

        visualize_2d_tray(
            result.tray,
            title=f"Tetris Pieces - WITH ROTATION (4 orientations)\n"
                  f"Placed: {result.num_placed}/7, Density: {result.density:.1%}",
            filename=filename
        )

        print(f"\nPieces: {list(pieces.keys())}")
        print(f"Placed: {result.num_placed}/7")
        print(f"Density: {result.density:.1%}")
        for p in result.placements:
            status = "PLACED" if p.success else "FAILED"
            print(f"  Item {p.item_index}: {status} at {p.position}, orientation={p.orientation_index}")

        assert os.path.exists(filename)
        assert result.num_placed == 7  # Should place all with rotation

    def test_tight_packing_challenge(self):
        """Try to pack pieces into a tight space - rotation should help."""
        # 4x10 tray - very constrained
        tray_size = (4, 10, 1)

        pieces = [
            make_2d_piece("####"),      # I piece (4 wide)
            make_2d_piece("##\n##"),    # O piece (2x2)
            make_2d_piece("###\n.#."),  # T piece (3 wide)
            make_2d_piece("#.\n#.\n##"), # L piece (2x3)
        ]

        # Without rotation
        packer_no_rot = BinPacker(tray_size=tray_size, num_orientations=1)
        result_no_rot = packer_no_rot.pack_voxels(pieces, sort_by_volume=False)

        with tempfile.NamedTemporaryFile(suffix='_tight_no_rot.png', delete=False,
                                          dir='/tmp') as f:
            filename_no_rot = f.name
        visualize_2d_tray(
            result_no_rot.tray,
            title=f"Tight Packing (4x10) - NO ROTATION\n"
                  f"Placed: {result_no_rot.num_placed}/4",
            filename=filename_no_rot
        )

        # With rotation
        packer_rot = BinPacker(tray_size=tray_size, num_orientations=4)
        result_rot = packer_rot.pack_voxels(pieces, sort_by_volume=False)

        with tempfile.NamedTemporaryFile(suffix='_tight_with_rot.png', delete=False,
                                          dir='/tmp') as f:
            filename_rot = f.name
        visualize_2d_tray(
            result_rot.tray,
            title=f"Tight Packing (4x10) - WITH ROTATION\n"
                  f"Placed: {result_rot.num_placed}/4",
            filename=filename_rot
        )

        print(f"\nTight packing comparison:")
        print(f"  Without rotation: {result_no_rot.num_placed}/4 placed")
        print(f"  With rotation:    {result_rot.num_placed}/4 placed")

        assert os.path.exists(filename_no_rot)
        assert os.path.exists(filename_rot)

    def test_step_by_step_placement(self):
        """Show step-by-step placement of each piece."""
        tray_size = (8, 12, 1)

        pieces = [
            ("I", make_2d_piece("####")),
            ("O", make_2d_piece("##\n##")),
            ("T", make_2d_piece("###\n.#.")),
            ("L", make_2d_piece("#.\n#.\n##")),
            ("J", make_2d_piece(".#\n.#\n##")),
        ]

        tray = np.zeros(tray_size, dtype=np.int32)
        filenames = []

        print(f"\nStep-by-step placement in {tray_size[0]}x{tray_size[1]} tray:")

        for idx, (name, piece) in enumerate(pieces):
            # Try with rotation
            from spectral_packer import get_orientations
            from spectral_packer.rotations import make_contiguous

            best_pos = None
            best_score = float('inf')
            best_piece = piece

            for orient_idx, rotated in enumerate(get_orientations(piece, 4)):
                rotated = make_contiguous(rotated.astype(np.int32))
                if any(rotated.shape[i] > tray_size[i] for i in range(3)):
                    continue
                pos, found, score = fft_search_placement(rotated, tray)
                if found and score < best_score:
                    best_pos = pos
                    best_score = score
                    best_piece = rotated

            if best_pos is not None:
                item_id = idx + 1
                tray = place_in_tray(best_piece, tray, best_pos, item_id)
                print(f"  Step {idx+1}: Placed '{name}' at {best_pos}")

                with tempfile.NamedTemporaryFile(
                    suffix=f'_step{idx+1}_{name}.png', delete=False, dir='/tmp'
                ) as f:
                    filename = f.name

                visualize_2d_tray(
                    tray,
                    title=f"Step {idx+1}: After placing '{name}'\n"
                          f"Position: {best_pos}",
                    filename=filename
                )
                filenames.append(filename)
            else:
                print(f"  Step {idx+1}: FAILED to place '{name}'")

        print(f"\nGenerated {len(filenames)} step-by-step images")
        for f in filenames:
            print(f"  {f}")

    def test_many_small_pieces(self):
        """Pack many small pieces to test density."""
        tray_size = (15, 15, 1)

        # Create 20 small pieces (1x2, 2x1, 1x3, 2x2)
        pieces = []
        for _ in range(5):
            pieces.append(make_2d_piece("##"))      # 1x2
            pieces.append(make_2d_piece("#\n#"))    # 2x1
            pieces.append(make_2d_piece("###"))     # 1x3
            pieces.append(make_2d_piece("##\n##"))  # 2x2

        packer = BinPacker(tray_size=tray_size, num_orientations=4)
        result = packer.pack_voxels(pieces)

        with tempfile.NamedTemporaryFile(suffix='_many_pieces.png', delete=False,
                                          dir='/tmp') as f:
            filename = f.name

        visualize_2d_tray(
            result.tray,
            title=f"Many Small Pieces (20 items)\n"
                  f"Placed: {result.num_placed}/20, Density: {result.density:.1%}",
            filename=filename
        )

        print(f"\nMany pieces test:")
        print(f"  Placed: {result.num_placed}/20")
        print(f"  Density: {result.density:.1%}")

        assert result.num_placed >= 15  # Should place most

    def test_comparison_side_by_side(self):
        """Create side-by-side comparison of with/without rotation."""
        tray_size = (8, 16, 1)

        pieces = get_tetris_pieces()
        items = list(pieces.values())

        # Pack without rotation
        packer1 = BinPacker(tray_size=tray_size, num_orientations=1)
        result1 = packer1.pack_voxels(items, sort_by_volume=True)

        # Pack with rotation
        packer2 = BinPacker(tray_size=tray_size, num_orientations=4)
        result2 = packer2.pack_voxels(items, sort_by_volume=True)

        # Create side-by-side visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        for ax, result, title in [
            (axes[0], result1, "WITHOUT Rotation"),
            (axes[1], result2, "WITH Rotation (4 orientations)")
        ]:
            grid = result.tray[:, :, 0]

            colors = ['white']
            color_cycle = plt.cm.tab10.colors
            for i in range(max(1, int(grid.max()))):
                colors.append(color_cycle[i % len(color_cycle)])
            cmap = ListedColormap(colors[:int(grid.max()) + 1])

            ax.imshow(grid, cmap=cmap, interpolation='nearest',
                     vmin=0, vmax=max(1, grid.max()))

            ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
            ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    val = grid[i, j]
                    if val > 0:
                        ax.text(j, i, str(int(val)), ha='center', va='center',
                               fontsize=10, fontweight='bold', color='white')

            ax.set_title(f"{title}\nPlaced: {result.num_placed}/7, "
                        f"Density: {result.density:.1%}",
                        fontsize=12, fontweight='bold')

        plt.suptitle("Rotation Impact on Packing", fontsize=14, fontweight='bold')
        plt.tight_layout()

        with tempfile.NamedTemporaryFile(suffix='_comparison.png', delete=False,
                                          dir='/tmp') as f:
            filename = f.name

        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n{'='*60}")
        print(f"SIDE-BY-SIDE COMPARISON: {filename}")
        print(f"{'='*60}")
        print(f"\nWithout rotation: {result1.num_placed}/7 placed, {result1.density:.1%} density")
        print(f"With rotation:    {result2.num_placed}/7 placed, {result2.density:.1%} density")

        assert os.path.exists(filename)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
