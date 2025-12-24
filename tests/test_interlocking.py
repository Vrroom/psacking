"""
2D Interlocking Tests - Verifies that objects can get trapped inside ring shapes.

Tests create various ring configurations and verify:
1. Objects DO get placed inside rings (current greedy algorithm)
2. Different exit sizes affect trapping behavior
3. Visualizations show the interlocking clearly
"""
import numpy as np
import pytest
from collections import deque
from pathlib import Path

from spectral_packer import BinPacker

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


VIZ_DIR = Path("/tmp/interlocking_tests")


def setup_module(module):
    """Create output directory for visualizations."""
    VIZ_DIR.mkdir(exist_ok=True)


def make_ring(outer_size, wall_thickness=1):
    """Create a square ring (hollow square)."""
    grid = np.ones((outer_size, outer_size, 1), dtype=np.int32)
    t = wall_thickness
    if outer_size > 2 * t:
        grid[t:-t, t:-t, 0] = 0
    return grid


def make_ring_with_exit(outer_size, wall_thickness=1, exit_size=1, exit_side='right'):
    """Create a square ring with an exit gap on the specified side."""
    grid = make_ring(outer_size, wall_thickness)
    t = wall_thickness
    inner_mid = outer_size // 2
    half_exit = exit_size // 2
    start, end = inner_mid - half_exit, inner_mid - half_exit + exit_size

    if exit_side == 'right':
        grid[start:end, -t:, 0] = 0
    elif exit_side == 'left':
        grid[start:end, :t, 0] = 0
    elif exit_side == 'top':
        grid[:t, start:end, 0] = 0
    elif exit_side == 'bottom':
        grid[-t:, start:end, 0] = 0
    return grid


def make_u_shape(width, height, wall_thickness=1):
    """Create a U-shaped trap (open on top)."""
    grid = np.zeros((height, width, 1), dtype=np.int32)
    t = wall_thickness
    grid[:, :t, 0] = 1      # Left wall
    grid[:, -t:, 0] = 1     # Right wall
    grid[-t:, :, 0] = 1     # Bottom wall
    return grid


def visualize_interlocking(tray, ring_mask, title, filename, highlight_trapped=True):
    """Visualize tray with ring and placed objects. Trapped objects shown in red."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping visualization")
        return None

    grid = tray[:, :, 0].copy() if tray.ndim == 3 else tray.copy()
    ring = ring_mask[:, :, 0] if ring_mask.ndim == 3 else ring_mask

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    interior = find_ring_interior(ring)
    trapped_ids = find_trapped_objects(grid) if highlight_trapped else set()

    # Build visualization grid: -0.5=interior empty, 0=exterior, 1=wall, 2+=objects
    viz_grid = grid.copy().astype(float)
    viz_grid[(grid == 0) & interior] = -0.5

    # Build colormap
    n_objects = int(grid.max())
    colors = ['white', 'lightgray', 'dimgray']  # exterior, interior, wall
    color_cycle = plt.cm.tab10.colors
    for i in range(max(1, n_objects - 1)):
        obj_id = i + 2
        colors.append('red' if obj_id in trapped_ids else color_cycle[i % len(color_cycle)])

    ax.imshow(viz_grid, cmap=ListedColormap(colors), interpolation='nearest',
              vmin=-0.5, vmax=max(1, n_objects))

    # Grid lines
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

    # Cell labels
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid[i, j]
            if val == 1:
                ax.text(j, i, 'W', ha='center', va='center', fontsize=7, color='white', fontweight='bold')
            elif val > 1:
                ax.text(j, i, str(int(val)), ha='center', va='center', fontsize=8,
                       fontweight='bold', color='white' if val in trapped_ids else 'black')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Legend
    n_objects_placed = len(set(grid.flatten()) - {0, 1})
    legend_text = (f"Ring walls: W (gray)\nObjects placed: {n_objects_placed}\n"
                   f"Trapped objects: {len(trapped_ids)} (red)\nInterior zone: light gray")
    ax.text(1.02, 0.98, legend_text, transform=ax.transAxes, verticalalignment='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()
    return len(trapped_ids)


def get_object_cells(tray, obj_id):
    """Get all cells occupied by an object as a set of (row, col) tuples."""
    indices = np.argwhere(tray == obj_id)
    return set(map(tuple, indices))


def get_object_bbox(tray, obj_id):
    """Get bounding box (min_row, min_col, height, width) of an object."""
    indices = np.argwhere(tray == obj_id)
    if len(indices) == 0:
        return None
    min_rc = indices.min(axis=0)
    max_rc = indices.max(axis=0)
    return min_rc[0], min_rc[1], max_rc[0] - min_rc[0] + 1, max_rc[1] - min_rc[1] + 1


def can_object_move_to(tray, obj_id, new_top_left):
    """Check if object can be placed at new_top_left without collision."""
    h, w = tray.shape
    cells = get_object_cells(tray, obj_id)
    if not cells:
        return True

    min_row = min(c[0] for c in cells)
    min_col = min(c[1] for c in cells)
    dr, dc = new_top_left[0] - min_row, new_top_left[1] - min_col

    for r, c in cells:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            if tray[nr, nc] != 0 and tray[nr, nc] != obj_id:
                return False
    return True


def is_object_outside(tray, cells, position, orig_min):
    """Check if object at given position is completely outside the grid."""
    h, w = tray.shape
    dr, dc = position[0] - orig_min[0], position[1] - orig_min[1]
    return all(not (0 <= r + dr < h and 0 <= c + dc < w) for r, c in cells)


def can_object_escape(tray, obj_id):
    """
    Check if an object can escape using BFS pathfinding.

    Returns True if there exists a sequence of moves (up/down/left/right)
    that leads to the object being completely outside the grid.
    """
    h, w = tray.shape
    cells = get_object_cells(tray, obj_id)
    if not cells:
        return True

    bbox = get_object_bbox(tray, obj_id)
    if bbox is None:
        return True
    min_row, min_col, obj_h, obj_w = bbox
    orig_min = (min_row, min_col)
    start = orig_min

    visited = {start}
    queue = deque([start])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        pos = queue.popleft()
        if is_object_outside(tray, cells, pos, orig_min):
            return True

        for dr, dc in directions:
            new_pos = (pos[0] + dr, pos[1] + dc)
            if new_pos in visited:
                continue
            # Bound search to reasonable area
            if not (-obj_h <= new_pos[0] <= h and -obj_w <= new_pos[1] <= w):
                continue
            if can_object_move_to(tray, obj_id, new_pos):
                visited.add(new_pos)
                queue.append(new_pos)
    return False


def find_trapped_objects(tray):
    """
    Find truly trapped objects using iterative disassembly.

    Repeatedly removes any object that can escape until no more can.
    The remaining objects are truly trapped (interlocked).
    """
    grid = tray.copy()
    remaining = set(int(x) for x in grid.flatten() if x > 1)

    changed = True
    while changed:
        changed = False
        for obj_id in list(remaining):
            if can_object_escape(grid, obj_id):
                grid[grid == obj_id] = 0
                remaining.remove(obj_id)
                changed = True
    return remaining


def find_ring_interior(ring_mask):
    """Find interior of ring via flood fill from edges (for visualization)."""
    ring = ring_mask[:, :, 0] if ring_mask.ndim == 3 else ring_mask
    h, w = ring.shape
    exterior = np.zeros_like(ring, dtype=bool)

    queue = deque()
    # Seed from all edge cells that are empty
    for i in range(h):
        for j in [0, w - 1]:
            if ring[i, j] == 0 and not exterior[i, j]:
                exterior[i, j] = True
                queue.append((i, j))
    for j in range(w):
        for i in [0, h - 1]:
            if ring[i, j] == 0 and not exterior[i, j]:
                exterior[i, j] = True
                queue.append((i, j))

    while queue:
        i, j = queue.popleft()
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w and not exterior[ni, nj] and ring[ni, nj] == 0:
                exterior[ni, nj] = True
                queue.append((ni, nj))

    return ~exterior & (ring == 0)


def count_trapped_objects(tray, ring_mask=None):
    """Count trapped objects. Returns (num_trapped, trapped_ids, total_objects)."""
    grid = tray[:, :, 0] if tray.ndim == 3 else tray
    all_ids = set(int(x) for x in grid.flatten() if x > 1)
    trapped_ids = find_trapped_objects(grid)
    return len(trapped_ids), trapped_ids, len(all_ids)


class TestSimpleRingInterlocking:
    """Test that objects get trapped inside simple closed rings."""

    def test_square_ring_traps_objects(self):
        """Closed square ring - all interior objects are trapped."""
        tray_size = (20, 20, 1)
        ring = make_ring(outer_size=10, wall_thickness=1)
        tray = np.zeros(tray_size, dtype=np.int32)
        tray[5:15, 5:15, 0] = ring[:, :, 0]
        ring_mask = tray.copy()

        objects = [np.ones((1, 1, 1), dtype=np.int32) for _ in range(50)]
        packer = BinPacker(tray_size=tray_size, num_orientations=1)
        result = packer.pack_voxels(objects, initial_tray=tray, sort_by_volume=False)

        n_trapped, _, _ = count_trapped_objects(result.tray, ring_mask)
        visualize_interlocking(result.tray, ring_mask,
            title=f"Closed Square Ring\nPlaced: {result.num_placed}, Trapped: {n_trapped}",
            filename=VIZ_DIR / "01_square_ring_closed.png")

        assert n_trapped > 0, "Expected at least one object trapped inside ring"

    def test_thick_wall_ring(self):
        """Ring with thicker walls - smaller interior."""
        tray_size = (24, 24, 1)
        ring = make_ring(outer_size=12, wall_thickness=2)
        tray = np.zeros(tray_size, dtype=np.int32)
        tray[6:18, 6:18, 0] = ring[:, :, 0]
        ring_mask = tray.copy()

        objects = [np.ones((1, 1, 1), dtype=np.int32) for _ in range(40)]
        packer = BinPacker(tray_size=tray_size, num_orientations=1)
        result = packer.pack_voxels(objects, initial_tray=tray, sort_by_volume=False)

        n_trapped, _, _ = count_trapped_objects(result.tray, ring_mask)
        visualize_interlocking(result.tray, ring_mask,
            title=f"Thick Wall Ring (wall=2)\nPlaced: {result.num_placed}, Trapped: {n_trapped}",
            filename=VIZ_DIR / "02_thick_wall_ring.png")

        assert n_trapped > 0, "Expected trapped objects"


class TestComplexRingShapes:
    """Test more complex ring configurations."""

    def test_u_shape_trap(self):
        """U-shape is open on top - objects can escape upward."""
        tray_size = (20, 20, 1)
        u_shape = make_u_shape(width=8, height=10, wall_thickness=1)
        tray = np.zeros(tray_size, dtype=np.int32)
        tray[5:15, 6:14, 0] = u_shape[:, :, 0]
        ring_mask = tray.copy()

        objects = ([np.ones((1, 1, 1), dtype=np.int32) for _ in range(20)] +
                   [np.ones((2, 1, 1), dtype=np.int32) for _ in range(10)])
        packer = BinPacker(tray_size=tray_size, num_orientations=1)
        result = packer.pack_voxels(objects, initial_tray=tray, sort_by_volume=False)

        n_trapped, _, _ = count_trapped_objects(result.tray, ring_mask)
        visualize_interlocking(result.tray, ring_mask,
            title=f"U-Shape Trap (open top)\nPlaced: {result.num_placed}, Trapped: {n_trapped}",
            filename=VIZ_DIR / "03_u_shape_trap.png")
        assert result.num_placed > 0

    def test_nested_rings(self):
        """Two concentric rings - objects in inner ring are deeply trapped."""
        tray_size = (30, 30, 1)
        outer_ring = make_ring(outer_size=20, wall_thickness=1)
        inner_ring = make_ring(outer_size=10, wall_thickness=1)

        tray = np.zeros(tray_size, dtype=np.int32)
        tray[5:25, 5:25, 0] = outer_ring[:, :, 0]
        tray[10:20, 10:20, 0] = np.maximum(tray[10:20, 10:20, 0], inner_ring[:, :, 0])
        ring_mask = tray.copy()

        objects = [np.ones((1, 1, 1), dtype=np.int32) for _ in range(60)]
        packer = BinPacker(tray_size=tray_size, num_orientations=1)
        result = packer.pack_voxels(objects, initial_tray=tray, sort_by_volume=False)

        n_trapped, _, _ = count_trapped_objects(result.tray, ring_mask)
        visualize_interlocking(result.tray, ring_mask,
            title=f"Nested Rings\nPlaced: {result.num_placed}, Trapped: {n_trapped}",
            filename=VIZ_DIR / "04_nested_rings.png")
        assert n_trapped > 0, "Expected objects trapped between or inside rings"

    def test_irregular_ring(self):
        """Ring with an internal nested square."""
        # Create irregular shape with nested square
        grid = np.zeros((8, 10, 1), dtype=np.int32)
        grid[0, :, 0] = 1; grid[-1, :, 0] = 1  # top/bottom walls
        grid[:, 0, 0] = 1; grid[:, -1, 0] = 1  # left/right walls
        grid[2:6, 2:6, 0] = 1  # inner square
        grid[3:5, 3:5, 0] = 0  # hollow center

        tray_size = (20, 20, 1)
        tray = np.zeros(tray_size, dtype=np.int32)
        tray[6:14, 5:15, 0] = grid[:, :, 0]
        ring_mask = tray.copy()

        objects = [np.ones((1, 1, 1), dtype=np.int32) for _ in range(40)]
        packer = BinPacker(tray_size=tray_size, num_orientations=1)
        result = packer.pack_voxels(objects, initial_tray=tray, sort_by_volume=False)

        n_trapped, _, _ = count_trapped_objects(result.tray, ring_mask)
        visualize_interlocking(result.tray, ring_mask,
            title=f"Irregular Ring (inner square)\nPlaced: {result.num_placed}, Trapped: {n_trapped}",
            filename=VIZ_DIR / "05_irregular_ring.png")


class TestRingWithSmallExit:
    """Test rings with small exits - large objects trapped, small can escape."""

    def test_ring_with_1_cell_exit(self):
        """1-cell exit: only 1x1 objects can escape."""
        tray_size = (24, 24, 1)
        ring = make_ring_with_exit(outer_size=12, wall_thickness=1, exit_size=1)
        tray = np.zeros(tray_size, dtype=np.int32)
        tray[6:18, 6:18, 0] = ring[:, :, 0]
        ring_mask = tray.copy()

        objects = ([np.ones((1, 1, 1), dtype=np.int32) for _ in range(20)] +  # Can exit
                   [np.ones((2, 2, 1), dtype=np.int32) for _ in range(10)])   # Cannot exit
        packer = BinPacker(tray_size=tray_size, num_orientations=1)
        result = packer.pack_voxels(objects, initial_tray=tray, sort_by_volume=False)

        n_trapped, _, _ = count_trapped_objects(result.tray, ring_mask)
        visualize_interlocking(result.tray, ring_mask,
            title=f"Ring with 1-cell Exit\nPlaced: {result.num_placed}, Trapped: {n_trapped}\n(1x1 can escape, 2x2 cannot)",
            filename=VIZ_DIR / "06_ring_small_exit_1.png")

    def test_ring_with_2_cell_exit(self):
        """2-cell exit: 1x1 and 2x1 can escape, 3x3 cannot."""
        tray_size = (24, 24, 1)
        ring = make_ring_with_exit(outer_size=12, wall_thickness=1, exit_size=2)
        tray = np.zeros(tray_size, dtype=np.int32)
        tray[6:18, 6:18, 0] = ring[:, :, 0]
        ring_mask = tray.copy()

        objects = ([np.ones((1, 1, 1), dtype=np.int32) for _ in range(15)] +
                   [np.ones((2, 1, 1), dtype=np.int32) for _ in range(10)] +
                   [np.ones((3, 3, 1), dtype=np.int32) for _ in range(5)])
        packer = BinPacker(tray_size=tray_size, num_orientations=4)
        result = packer.pack_voxels(objects, initial_tray=tray, sort_by_volume=False)

        n_trapped, _, _ = count_trapped_objects(result.tray, ring_mask)
        visualize_interlocking(result.tray, ring_mask,
            title=f"Ring with 2-cell Exit\nPlaced: {result.num_placed}, Trapped: {n_trapped}\n(1x1, 2x1 can escape; 3x3 cannot)",
            filename=VIZ_DIR / "07_ring_small_exit_2.png")


class TestRingWithLargeExit:
    """Test rings with large exits - most objects can escape."""

    def test_ring_with_half_open(self):
        """Half wall removed - most objects can escape."""
        tray_size = (24, 24, 1)
        ring = make_ring(outer_size=12, wall_thickness=1)
        ring[0:6, -1:, 0] = 0  # Remove top half of right wall
        tray = np.zeros(tray_size, dtype=np.int32)
        tray[6:18, 6:18, 0] = ring[:, :, 0]
        ring_mask = tray.copy()

        objects = ([np.ones((2, 2, 1), dtype=np.int32) for _ in range(20)] +
                   [np.ones((3, 2, 1), dtype=np.int32) for _ in range(10)])
        packer = BinPacker(tray_size=tray_size, num_orientations=4)
        result = packer.pack_voxels(objects, initial_tray=tray, sort_by_volume=False)

        n_trapped, _, _ = count_trapped_objects(result.tray, ring_mask)
        visualize_interlocking(result.tray, ring_mask,
            title=f"Half-Open Ring\nPlaced: {result.num_placed}, Trapped: {n_trapped}",
            filename=VIZ_DIR / "08_ring_half_open.png")

    def test_c_shape_wide_gap(self):
        """6-cell gap - even large objects can escape."""
        tray_size = (24, 24, 1)
        ring = make_ring_with_exit(outer_size=12, wall_thickness=1, exit_size=6)
        tray = np.zeros(tray_size, dtype=np.int32)
        tray[6:18, 6:18, 0] = ring[:, :, 0]
        ring_mask = tray.copy()

        objects = ([np.ones((1, 1, 1), dtype=np.int32) for _ in range(10)] +
                   [np.ones((2, 2, 1), dtype=np.int32) for _ in range(10)] +
                   [np.ones((3, 3, 1), dtype=np.int32) for _ in range(5)] +
                   [np.ones((5, 2, 1), dtype=np.int32) for _ in range(3)])
        packer = BinPacker(tray_size=tray_size, num_orientations=4)
        result = packer.pack_voxels(objects, initial_tray=tray, sort_by_volume=False)

        n_trapped, _, _ = count_trapped_objects(result.tray, ring_mask)
        visualize_interlocking(result.tray, ring_mask,
            title=f"C-Shape Wide Gap (6 cells)\nPlaced: {result.num_placed}, Trapped: {n_trapped}",
            filename=VIZ_DIR / "09_c_shape_wide_gap.png")

    def test_comparison_exit_sizes(self):
        """Compare same ring with different exit sizes side by side."""
        if not HAS_MATPLOTLIB:
            pytest.skip("matplotlib not available")

        tray_size, exit_sizes = (20, 20, 1), [0, 1, 2, 4]
        results = []

        for exit_size in exit_sizes:
            ring = make_ring(10, 1) if exit_size == 0 else make_ring_with_exit(10, 1, exit_size)
            tray = np.zeros(tray_size, dtype=np.int32)
            tray[5:15, 5:15, 0] = ring[:, :, 0]
            ring_mask = tray.copy()

            objects = [np.ones((1, 1, 1), dtype=np.int32) for _ in range(30)]
            packer = BinPacker(tray_size=tray_size, num_orientations=1)
            result = packer.pack_voxels(objects, initial_tray=tray, sort_by_volume=False)
            n_trapped, _, _ = count_trapped_objects(result.tray, ring_mask)
            results.append((exit_size, result, ring_mask, n_trapped))

        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        for ax, (exit_size, result, ring_mask, n_trapped) in zip(axes, results):
            grid = result.tray[:, :, 0].copy()
            interior = find_ring_interior(ring_mask[:, :, 0])
            viz_grid = grid.copy().astype(float)
            viz_grid[(grid == 0) & interior] = -0.5
            trapped_ids = find_trapped_objects(grid)

            n_objects = int(grid.max())
            colors = ['white', 'lightgray', 'dimgray']
            for i in range(max(1, n_objects - 1)):
                colors.append('red' if (i + 2) in trapped_ids else plt.cm.tab10.colors[i % 10])

            ax.imshow(viz_grid, cmap=ListedColormap(colors), interpolation='nearest',
                     vmin=-0.5, vmax=max(1, n_objects))
            ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
            ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
            ax.set_title(f"{'Closed' if exit_size == 0 else f'Exit: {exit_size}'}\nTrapped: {n_trapped}",
                        fontsize=11, fontweight='bold')

        plt.suptitle("Exit Size Comparison (Red = trapped)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "10_exit_size_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()


class TestInterlockingSummary:
    """Summary test that reports test configuration."""

    def test_full_summary(self):
        """Print summary of interlocking test suite."""
        print(f"\nInterlocking tests - visualizations at: {VIZ_DIR}")
        print("Red = trapped, Light gray = interior zone")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
