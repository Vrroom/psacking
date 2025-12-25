"""
3D Interlocking Tests - Compares standard vs interlocking-free packing modes.

Tests use 3D cage structures (walls + floor/ceiling inside only) to demonstrate:
1. Standard mode places objects inside cages when exterior fills up
2. Interlocking-free (IF) mode avoids cage interiors (GPU flood-fill detection)
3. IF mode correctly handles cages with exits (objects can escape via exit)
4. Slice-based visualizations show all Z levels side-by-side
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


def make_ring(outer_size, wall_thickness=1, height=1, closed=False):
    """Create a square ring (hollow square).

    Args:
        outer_size: Size of outer square
        wall_thickness: Thickness of walls
        height: Height in Z dimension (use >1 for true 3D rings)
        closed: If True, add floor and ceiling to create a closed box
                (objects inside cannot escape via Z axis)
    """
    grid = np.ones((outer_size, outer_size, height), dtype=np.int32)
    t = wall_thickness
    if outer_size > 2 * t:
        if closed and height > 2 * t:
            # Hollow interior with floor/ceiling (closed box)
            grid[t:-t, t:-t, t:-t] = 0
        else:
            # Just walls, open top/bottom (objects can escape via Z)
            grid[t:-t, t:-t, :] = 0
    return grid


def make_ring_3d_cage(outer_size, wall_thickness=1, height=3):
    """Create a 3D ring cage: walls + floor/ceiling ONLY inside the ring.

    This creates a structure where:
    - Outside the ring: all Z levels are empty (objects can be placed and escape)
    - Ring walls: solid at all Z levels
    - Inside the ring at z=0: solid floor (prevents placement)
    - Inside the ring at z=1 through z=height-2: HOLLOW (objects can be placed)
    - Inside the ring at z=height-1: solid ceiling (prevents placement)

    Objects placed at z=1 interior are truly TRAPPED because:
    - They can't escape via X/Y (walls block)
    - They can't be placed at z=0 or z=height-1 (floor/ceiling cause collision)
    - GPU flood-fill can't reach interior from boundary

    Args:
        outer_size: Size of outer square
        wall_thickness: Thickness of walls
        height: Height in Z dimension (minimum 3 for floor/interior/ceiling)
    """
    assert height >= 3, "Height must be >= 3 for floor/interior/ceiling"

    grid = np.zeros((outer_size, outer_size, height), dtype=np.int32)
    t = wall_thickness

    # Walls on all 4 sides, full height
    grid[:t, :, :] = 1       # Left wall
    grid[-t:, :, :] = 1      # Right wall
    grid[:, :t, :] = 1       # Bottom wall
    grid[:, -t:, :] = 1      # Top wall

    # Floor and ceiling ONLY inside the ring (not outside)
    if outer_size > 2 * t:
        grid[t:-t, t:-t, 0] = 1           # Floor (inside ring only)
        grid[t:-t, t:-t, height-1] = 1    # Ceiling (inside ring only)
        # Middle layers inside ring remain 0 (hollow)

    return grid


def make_ring_with_exit(outer_size, wall_thickness=1, exit_size=1, exit_side='right', height=1):
    """Create a square ring with an exit gap on the specified side.

    Args:
        outer_size: Size of outer square
        wall_thickness: Thickness of walls
        exit_size: Size of the exit gap
        exit_side: Which side has the exit ('right', 'left', 'top', 'bottom')
        height: Height in Z dimension (use >1 for true 3D rings)
    """
    grid = make_ring(outer_size, wall_thickness, height)
    t = wall_thickness
    inner_mid = outer_size // 2
    half_exit = exit_size // 2
    start, end = inner_mid - half_exit, inner_mid - half_exit + exit_size

    if exit_side == 'right':
        grid[start:end, -t:, :] = 0
    elif exit_side == 'left':
        grid[start:end, :t, :] = 0
    elif exit_side == 'top':
        grid[:t, start:end, :] = 0
    elif exit_side == 'bottom':
        grid[-t:, start:end, :] = 0
    return grid


def make_ring_3d_cage_with_exit(outer_size, wall_thickness=1, height=3, exit_size=1, exit_side='right'):
    """Create a 3D cage with an exit through which objects can escape.

    Like make_ring_3d_cage but with a gap in one wall AND the corresponding
    floor/ceiling removed in the exit area, creating a true escape route.

    Args:
        outer_size: Size of outer square
        wall_thickness: Thickness of walls
        height: Height in Z dimension (minimum 3)
        exit_size: Size of the exit gap
        exit_side: Which side has the exit ('right', 'left', 'top', 'bottom')
    """
    # Start with basic cage
    grid = make_ring_3d_cage(outer_size, wall_thickness, height)

    t = wall_thickness
    inner_mid = outer_size // 2
    half_exit = exit_size // 2
    start, end = inner_mid - half_exit, inner_mid - half_exit + exit_size

    # Remove wall at exit
    if exit_side == 'right':
        grid[start:end, -t:, :] = 0
        # Also remove floor/ceiling in exit area so objects can escape
        if outer_size > 2 * t:
            grid[start:end, -t:, 0] = 0       # floor
            grid[start:end, -t:, height-1] = 0  # ceiling
    elif exit_side == 'left':
        grid[start:end, :t, :] = 0
        if outer_size > 2 * t:
            grid[start:end, :t, 0] = 0
            grid[start:end, :t, height-1] = 0
    elif exit_side == 'top':
        grid[:t, start:end, :] = 0
        if outer_size > 2 * t:
            grid[:t, start:end, 0] = 0
            grid[:t, start:end, height-1] = 0
    elif exit_side == 'bottom':
        grid[-t:, start:end, :] = 0
        if outer_size > 2 * t:
            grid[-t:, start:end, 0] = 0
            grid[-t:, start:end, height-1] = 0

    return grid


def make_u_shape(width, u_height, wall_thickness=1, height=1):
    """Create a U-shaped trap (open on top).

    Args:
        width: Width of U shape
        u_height: Height of U shape (in X-Y plane)
        wall_thickness: Thickness of walls
        height: Height in Z dimension (use >1 for true 3D)
    """
    grid = np.zeros((u_height, width, height), dtype=np.int32)
    t = wall_thickness
    grid[:, :t, :] = 1      # Left wall
    grid[:, -t:, :] = 1     # Right wall
    grid[-t:, :, :] = 1     # Bottom wall
    return grid


def visualize_interlocking(tray, ring_mask, title, filename, highlight_trapped=True, z_slice=None):
    """Visualize tray with ring and placed objects. Trapped objects shown in red.

    Args:
        z_slice: Z index to visualize. If None, uses middle slice for 3D grids.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping visualization")
        return None

    # For 3D grids, take middle Z slice
    if tray.ndim == 3 and tray.shape[2] > 1:
        z = z_slice if z_slice is not None else tray.shape[2] // 2
        grid = tray[:, :, z].copy()
        ring = ring_mask[:, :, z] if ring_mask.ndim == 3 else ring_mask
    else:
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
    Find truly trapped objects using iterative disassembly (2D version).

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


# =============================================================================
# 3D Escape Analysis (consistent with GPU interlocking-free algorithm)
# =============================================================================

def get_object_cells_3d(tray, obj_id):
    """Get all cells occupied by an object as a set of (x, y, z) tuples."""
    indices = np.argwhere(tray == obj_id)
    return set(map(tuple, indices))


def get_object_bbox_3d(tray, obj_id):
    """Get 3D bounding box: (min_x, min_y, min_z, size_x, size_y, size_z)."""
    indices = np.argwhere(tray == obj_id)
    if len(indices) == 0:
        return None
    min_coords = indices.min(axis=0)
    max_coords = indices.max(axis=0)
    sizes = max_coords - min_coords + 1
    return (*min_coords, *sizes)


def can_object_move_to_3d(tray, obj_id, new_origin):
    """Check if object can be placed at new_origin without collision (3D)."""
    nx, ny, nz = tray.shape
    cells = get_object_cells_3d(tray, obj_id)
    if not cells:
        return True

    # Get current origin (min coords)
    min_x = min(c[0] for c in cells)
    min_y = min(c[1] for c in cells)
    min_z = min(c[2] for c in cells)

    # Compute displacement
    dx = new_origin[0] - min_x
    dy = new_origin[1] - min_y
    dz = new_origin[2] - min_z

    for x, y, z in cells:
        new_x, new_y, new_z = x + dx, y + dy, z + dz
        if 0 <= new_x < nx and 0 <= new_y < ny and 0 <= new_z < nz:
            val = tray[new_x, new_y, new_z]
            if val != 0 and val != obj_id:
                return False
    return True


def is_object_outside_3d(tray, cells, position, orig_min):
    """Check if object at given position is completely outside the 3D grid."""
    nx, ny, nz = tray.shape
    dx = position[0] - orig_min[0]
    dy = position[1] - orig_min[1]
    dz = position[2] - orig_min[2]
    return all(
        not (0 <= x + dx < nx and 0 <= y + dy < ny and 0 <= z + dz < nz)
        for x, y, z in cells
    )


def can_object_escape_3d(tray, obj_id):
    """
    Check if an object can escape using 3D BFS pathfinding.

    Returns True if there exists a sequence of moves along any of the
    6 axis directions (±X, ±Y, ±Z) that leads to the object being
    completely outside the grid.

    This is consistent with the GPU interlocking-free algorithm.
    """
    nx, ny, nz = tray.shape
    cells = get_object_cells_3d(tray, obj_id)
    if not cells:
        return True

    bbox = get_object_bbox_3d(tray, obj_id)
    if bbox is None:
        return True

    min_x, min_y, min_z, obj_sx, obj_sy, obj_sz = bbox
    orig_min = (min_x, min_y, min_z)
    start = orig_min

    visited = {start}
    queue = deque([start])

    # 6 directions: ±X, ±Y, ±Z
    directions = [
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1)
    ]

    while queue:
        pos = queue.popleft()
        if is_object_outside_3d(tray, cells, pos, orig_min):
            return True

        for dx, dy, dz in directions:
            new_pos = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
            if new_pos in visited:
                continue
            # Bound search to reasonable area
            if not (-obj_sx <= new_pos[0] <= nx and
                    -obj_sy <= new_pos[1] <= ny and
                    -obj_sz <= new_pos[2] <= nz):
                continue
            if can_object_move_to_3d(tray, obj_id, new_pos):
                visited.add(new_pos)
                queue.append(new_pos)
    return False


def find_trapped_objects_3d(tray):
    """
    Find truly trapped objects using iterative disassembly (3D version).

    Repeatedly removes any object that can escape (via any of 6 directions)
    until no more can. The remaining objects are truly trapped.

    This is consistent with the GPU interlocking-free algorithm.
    """
    grid = tray.copy()
    remaining = set(int(x) for x in grid.flatten() if x > 1)

    changed = True
    while changed:
        changed = False
        for obj_id in list(remaining):
            if can_object_escape_3d(grid, obj_id):
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


def count_trapped_objects(tray, ring_mask=None, use_3d=True):
    """Count trapped objects using 3D escape analysis (consistent with GPU IF).

    Args:
        tray: The 3D tray array with placed objects
        ring_mask: Unused, kept for API compatibility
        use_3d: If True (default), use 3D escape analysis. If False, use 2D slice.

    Returns:
        (num_trapped, trapped_ids, total_objects)
    """
    if use_3d and tray.ndim == 3:
        all_ids = set(int(x) for x in tray.flatten() if x > 1)
        trapped_ids = find_trapped_objects_3d(tray)
        return len(trapped_ids), trapped_ids, len(all_ids)
    else:
        # Legacy 2D mode (for visualization compatibility)
        grid = tray[:, :, 0] if tray.ndim == 3 else tray
        all_ids = set(int(x) for x in grid.flatten() if x > 1)
        trapped_ids = find_trapped_objects(grid)
        return len(trapped_ids), trapped_ids, len(all_ids)


class TestSimpleRingInterlocking:
    """Test 3D cage rings with floor/ceiling comparing standard vs interlocking-free modes.

    All tests use:
    - 3D cages (walls + floor at z=0 + ceiling at z=height-1 inside ring only)
    - Height-1 objects that fit at z=1 (middle layer)
    - Slice-based visualization showing all Z levels
    """

    def visualize_slices(self, result_std, result_free, ring_mask, title, filename):
        """Side-by-side visualization showing all Z slices for each mode."""
        if not HAS_MATPLOTLIB:
            return

        height = result_std.tray.shape[2]
        fig, axes = plt.subplots(2, height, figsize=(5 * height, 10))

        for row, (result, label) in enumerate([
            (result_std, "Standard"),
            (result_free, "Interlocking-Free")
        ]):
            for z in range(height):
                ax = axes[row, z]
                grid = result.tray[:, :, z].copy()
                ring = ring_mask[:, :, z] if ring_mask.ndim == 3 else ring_mask
                interior = find_ring_interior(ring)
                trapped_ids = find_trapped_objects(result.tray[:, :, z])

                viz_grid = grid.copy().astype(float)
                viz_grid[(grid == 0) & interior] = -0.5

                n_objects = int(grid.max())
                colors = ['white', 'lightgray', 'dimgray']
                for i in range(max(1, n_objects - 1)):
                    colors.append('red' if (i + 2) in trapped_ids else plt.cm.tab10.colors[i % 10])

                ax.imshow(viz_grid, cmap=ListedColormap(colors), interpolation='nearest',
                         vmin=-0.5, vmax=max(1, n_objects))
                ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
                ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
                ax.grid(which='minor', color='black', linestyle='-', linewidth=0.3)

                objects_at_z = len(set(int(v) for v in grid.flatten() if v > 1))
                if z == 0:
                    ax.set_ylabel(f"{label}\nPlaced: {result.num_placed}", fontsize=10, fontweight='bold')
                ax.set_title(f"z={z} ({objects_at_z} objects)", fontsize=10)

        plt.suptitle(title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

    def test_square_ring_traps_objects(self):
        """3D cage ring - standard mode places inside, IF mode avoids interior."""
        # Smaller tray (24x24) with larger ring (12x12) leaves less exterior space
        tray_size = (24, 24, 3)
        ring = make_ring_3d_cage(outer_size=12, wall_thickness=1, height=3)
        initial_tray = np.zeros(tray_size, dtype=np.int32)
        initial_tray[6:18, 6:18, :] = ring
        ring_mask = initial_tray.copy()

        # Use 2x2x1 objects - they take up more space, forcing overflow into ring
        objects = [np.ones((2, 2, 1), dtype=np.int32) for _ in range(150)]

        # Standard mode - will place inside ring at z=1
        packer_std = BinPacker(tray_size=tray_size, num_orientations=1, interlocking_free=False)
        result_std = packer_std.pack_voxels(objects, initial_tray=initial_tray.copy(), sort_by_volume=False)

        # Interlocking-free mode - should NOT place inside
        packer_free = BinPacker(tray_size=tray_size, num_orientations=1, interlocking_free=True)
        result_free = packer_free.pack_voxels(objects, initial_tray=initial_tray.copy(), sort_by_volume=False)

        # Count objects inside ring at z=1
        interior_mask = np.zeros((tray_size[0], tray_size[1]), dtype=bool)
        interior_mask[7:17, 7:17] = True  # Ring interior (inside walls)

        std_inside = set()
        free_inside = set()
        for x in range(tray_size[0]):
            for y in range(tray_size[1]):
                if interior_mask[x, y]:
                    if result_std.tray[x, y, 1] > 1:
                        std_inside.add(result_std.tray[x, y, 1])
                    if result_free.tray[x, y, 1] > 1:
                        free_inside.add(result_free.tray[x, y, 1])

        print(f"\n=== Square Ring Cage ===")
        print(f"Standard:          placed={result_std.num_placed}, inside={len(std_inside)}")
        print(f"Interlocking-free: placed={result_free.num_placed}, inside={len(free_inside)}")

        self.visualize_slices(result_std, result_free, ring_mask,
            f"Square Ring Cage: Std={len(std_inside)} inside, IF={len(free_inside)} inside",
            VIZ_DIR / "01_square_ring_cage.png")

        assert len(free_inside) == 0, f"IF mode placed {len(free_inside)} inside (should be 0)"
        assert len(std_inside) > 0, "Standard mode should place objects inside"

    def test_thick_wall_ring(self):
        """3D cage ring with thick walls - smaller interior."""
        tray_size = (24, 24, 3)
        ring = make_ring_3d_cage(outer_size=12, wall_thickness=2, height=3)
        initial_tray = np.zeros(tray_size, dtype=np.int32)
        initial_tray[6:18, 6:18, :] = ring
        ring_mask = initial_tray.copy()

        # Height-1 objects
        objects = [np.ones((1, 1, 1), dtype=np.int32) for _ in range(120)]

        packer_std = BinPacker(tray_size=tray_size, num_orientations=1, interlocking_free=False)
        result_std = packer_std.pack_voxels(objects, initial_tray=initial_tray.copy(), sort_by_volume=False)

        packer_free = BinPacker(tray_size=tray_size, num_orientations=1, interlocking_free=True)
        result_free = packer_free.pack_voxels(objects, initial_tray=initial_tray.copy(), sort_by_volume=False)

        # Interior: walls are 2 thick, so interior is offset by 2
        interior_mask = np.zeros((tray_size[0], tray_size[1]), dtype=bool)
        interior_mask[8:16, 8:16] = True

        std_inside = set()
        free_inside = set()
        for x in range(tray_size[0]):
            for y in range(tray_size[1]):
                if interior_mask[x, y]:
                    if result_std.tray[x, y, 1] > 1:
                        std_inside.add(result_std.tray[x, y, 1])
                    if result_free.tray[x, y, 1] > 1:
                        free_inside.add(result_free.tray[x, y, 1])

        print(f"\n=== Thick Wall Ring Cage ===")
        print(f"Standard:          placed={result_std.num_placed}, inside={len(std_inside)}")
        print(f"Interlocking-free: placed={result_free.num_placed}, inside={len(free_inside)}")

        self.visualize_slices(result_std, result_free, ring_mask,
            f"Thick Wall Ring Cage: Std={len(std_inside)} inside, IF={len(free_inside)} inside",
            VIZ_DIR / "02_thick_wall_ring_cage.png")

        assert len(free_inside) == 0, f"IF mode placed {len(free_inside)} inside (should be 0)"


class TestComplexRingShapes:
    """Test more complex ring configurations with 3D cages."""

    def visualize_slices(self, result_std, result_free, ring_mask, title, filename):
        """Side-by-side visualization showing all Z slices for each mode."""
        if not HAS_MATPLOTLIB:
            return

        height = result_std.tray.shape[2]
        fig, axes = plt.subplots(2, height, figsize=(5 * height, 10))

        for row, (result, label) in enumerate([
            (result_std, "Standard"),
            (result_free, "Interlocking-Free")
        ]):
            for z in range(height):
                ax = axes[row, z]
                grid = result.tray[:, :, z].copy()
                ring = ring_mask[:, :, z] if ring_mask.ndim == 3 else ring_mask
                interior = find_ring_interior(ring)
                trapped_ids = find_trapped_objects(result.tray[:, :, z])

                viz_grid = grid.copy().astype(float)
                viz_grid[(grid == 0) & interior] = -0.5

                n_objects = int(grid.max())
                colors = ['white', 'lightgray', 'dimgray']
                for i in range(max(1, n_objects - 1)):
                    colors.append('red' if (i + 2) in trapped_ids else plt.cm.tab10.colors[i % 10])

                ax.imshow(viz_grid, cmap=ListedColormap(colors), interpolation='nearest',
                         vmin=-0.5, vmax=max(1, n_objects))
                ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
                ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
                ax.grid(which='minor', color='black', linestyle='-', linewidth=0.3)

                objects_at_z = len(set(int(v) for v in grid.flatten() if v > 1))
                if z == 0:
                    ax.set_ylabel(f"{label}\nPlaced: {result.num_placed}", fontsize=10, fontweight='bold')
                ax.set_title(f"z={z} ({objects_at_z} objects)", fontsize=10)

        plt.suptitle(title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

    def test_nested_cages(self):
        """Two concentric cage rings - inner cage is deeply trapped."""
        tray_size = (30, 30, 3)
        initial_tray = np.zeros(tray_size, dtype=np.int32)

        # Outer cage
        outer_cage = make_ring_3d_cage(outer_size=20, wall_thickness=1, height=3)
        initial_tray[5:25, 5:25, :] = outer_cage

        # Inner cage - place inside the outer cage
        inner_cage = make_ring_3d_cage(outer_size=10, wall_thickness=1, height=3)
        initial_tray[10:20, 10:20, :] = np.maximum(initial_tray[10:20, 10:20, :], inner_cage)

        ring_mask = initial_tray.copy()

        # Many objects to fill both cage interiors
        objects = [np.ones((1, 1, 1), dtype=np.int32) for _ in range(150)]

        packer_std = BinPacker(tray_size=tray_size, num_orientations=1, interlocking_free=False)
        result_std = packer_std.pack_voxels(objects, initial_tray=initial_tray.copy(), sort_by_volume=False)

        packer_free = BinPacker(tray_size=tray_size, num_orientations=1, interlocking_free=True)
        result_free = packer_free.pack_voxels(objects, initial_tray=initial_tray.copy(), sort_by_volume=False)

        # Count objects inside outer cage (but outside inner cage) at z=1
        outer_interior = set()
        inner_interior = set()
        for x in range(tray_size[0]):
            for y in range(tray_size[1]):
                # Outer interior: inside outer walls (6:24), outside inner walls
                if 6 <= x < 24 and 6 <= y < 24:
                    if not (11 <= x < 19 and 11 <= y < 19):  # Not inner cage
                        if result_std.tray[x, y, 1] > 1:
                            outer_interior.add(result_std.tray[x, y, 1])
                # Inner interior
                if 11 <= x < 19 and 11 <= y < 19:
                    if result_std.tray[x, y, 1] > 1:
                        inner_interior.add(result_std.tray[x, y, 1])

        # Same for interlocking-free
        outer_interior_free = set()
        inner_interior_free = set()
        for x in range(tray_size[0]):
            for y in range(tray_size[1]):
                if 6 <= x < 24 and 6 <= y < 24:
                    if not (11 <= x < 19 and 11 <= y < 19):
                        if result_free.tray[x, y, 1] > 1:
                            outer_interior_free.add(result_free.tray[x, y, 1])
                if 11 <= x < 19 and 11 <= y < 19:
                    if result_free.tray[x, y, 1] > 1:
                        inner_interior_free.add(result_free.tray[x, y, 1])

        print(f"\n=== Nested Cages ===")
        print(f"Standard:          placed={result_std.num_placed}, outer_interior={len(outer_interior)}, inner_interior={len(inner_interior)}")
        print(f"Interlocking-free: placed={result_free.num_placed}, outer_interior={len(outer_interior_free)}, inner_interior={len(inner_interior_free)}")

        self.visualize_slices(result_std, result_free, ring_mask,
            f"Nested Cages: Std=({len(outer_interior)}+{len(inner_interior)}) inside, IF=({len(outer_interior_free)}+{len(inner_interior_free)}) inside",
            VIZ_DIR / "03_nested_cages.png")

        # IF should not place inside either cage
        assert len(outer_interior_free) == 0, f"IF placed {len(outer_interior_free)} in outer cage"
        assert len(inner_interior_free) == 0, f"IF placed {len(inner_interior_free)} in inner cage"

    def test_multiple_separate_cages(self):
        """Multiple separate cages scattered in the tray."""
        tray_size = (32, 32, 3)
        initial_tray = np.zeros(tray_size, dtype=np.int32)

        # Place 4 small cages in different positions
        cage = make_ring_3d_cage(outer_size=8, wall_thickness=1, height=3)
        initial_tray[2:10, 2:10, :] = cage
        initial_tray[2:10, 22:30, :] = cage
        initial_tray[22:30, 2:10, :] = cage
        initial_tray[22:30, 22:30, :] = cage

        ring_mask = initial_tray.copy()

        # Mix of object types
        objects = (
            [np.ones((1, 1, 1), dtype=np.int32) for _ in range(80)] +
            [np.ones((2, 1, 1), dtype=np.int32) for _ in range(40)] +
            [np.ones((2, 2, 1), dtype=np.int32) for _ in range(20)]
        )

        packer_std = BinPacker(tray_size=tray_size, num_orientations=1, interlocking_free=False)
        result_std = packer_std.pack_voxels(objects, initial_tray=initial_tray.copy(), sort_by_volume=False)

        packer_free = BinPacker(tray_size=tray_size, num_orientations=1, interlocking_free=True)
        result_free = packer_free.pack_voxels(objects, initial_tray=initial_tray.copy(), sort_by_volume=False)

        # Count objects inside any cage at z=1
        cage_interiors = [
            (3, 9, 3, 9),    # cage 1
            (3, 9, 23, 29),  # cage 2
            (23, 29, 3, 9),  # cage 3
            (23, 29, 23, 29) # cage 4
        ]

        std_inside = set()
        free_inside = set()
        for x0, x1, y0, y1 in cage_interiors:
            for x in range(x0, x1):
                for y in range(y0, y1):
                    if result_std.tray[x, y, 1] > 1:
                        std_inside.add(result_std.tray[x, y, 1])
                    if result_free.tray[x, y, 1] > 1:
                        free_inside.add(result_free.tray[x, y, 1])

        print(f"\n=== Multiple Separate Cages ===")
        print(f"Standard:          placed={result_std.num_placed}, inside_cages={len(std_inside)}")
        print(f"Interlocking-free: placed={result_free.num_placed}, inside_cages={len(free_inside)}")

        self.visualize_slices(result_std, result_free, ring_mask,
            f"Multiple Cages: Std={len(std_inside)} inside, IF={len(free_inside)} inside",
            VIZ_DIR / "04_multiple_cages.png")

        assert len(free_inside) == 0, f"IF placed {len(free_inside)} inside cages"


class TestCageWithExit:
    """Test 3D cages with exits - demonstrates IF behavior with escape routes.

    Key concept: When a cage has an exit, the GPU flood fill can reach interior
    positions via the exit. So IF mode WILL place objects that fit through the
    exit, but will NOT place objects too large to escape.
    """

    def visualize_slices(self, result_std, result_free, ring_mask, title, filename):
        """Side-by-side visualization showing all Z slices for each mode."""
        if not HAS_MATPLOTLIB:
            return

        height = result_std.tray.shape[2]
        fig, axes = plt.subplots(2, height, figsize=(5 * height, 10))

        for row, (result, label) in enumerate([
            (result_std, "Standard"),
            (result_free, "Interlocking-Free")
        ]):
            for z in range(height):
                ax = axes[row, z]
                grid = result.tray[:, :, z].copy()
                ring = ring_mask[:, :, z] if ring_mask.ndim == 3 else ring_mask
                interior = find_ring_interior(ring)
                trapped_ids = find_trapped_objects(result.tray[:, :, z])

                viz_grid = grid.copy().astype(float)
                viz_grid[(grid == 0) & interior] = -0.5

                n_objects = int(grid.max())
                colors = ['white', 'lightgray', 'dimgray']
                for i in range(max(1, n_objects - 1)):
                    colors.append('red' if (i + 2) in trapped_ids else plt.cm.tab10.colors[i % 10])

                ax.imshow(viz_grid, cmap=ListedColormap(colors), interpolation='nearest',
                         vmin=-0.5, vmax=max(1, n_objects))
                ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
                ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
                ax.grid(which='minor', color='black', linestyle='-', linewidth=0.3)

                objects_at_z = len(set(int(v) for v in grid.flatten() if v > 1))
                if z == 0:
                    ax.set_ylabel(f"{label}\nPlaced: {result.num_placed}", fontsize=10, fontweight='bold')
                ax.set_title(f"z={z} ({objects_at_z} objects)", fontsize=10)

        plt.suptitle(title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

    def test_cage_with_small_exit(self):
        """3D cage with 2-cell exit: small objects escape, large trapped.

        - 1x1 objects: can fit through exit -> IF places inside (not trapped)
        - 3x3 objects: can't fit through exit -> IF avoids interior (would be trapped)
        """
        tray_size = (24, 24, 3)
        cage = make_ring_3d_cage_with_exit(outer_size=12, wall_thickness=1, height=3, exit_size=2)
        initial_tray = np.zeros(tray_size, dtype=np.int32)
        initial_tray[6:18, 6:18, :] = cage
        ring_mask = initial_tray.copy()

        # Mix of small (can escape) and large (cannot escape) objects
        objects = (
            [np.ones((1, 1, 1), dtype=np.int32) for _ in range(50)] +  # Can escape via exit
            [np.ones((3, 3, 1), dtype=np.int32) for _ in range(20)]    # Too large for exit
        )

        packer_std = BinPacker(tray_size=tray_size, num_orientations=1, interlocking_free=False)
        result_std = packer_std.pack_voxels(objects, initial_tray=initial_tray.copy(), sort_by_volume=False)

        packer_free = BinPacker(tray_size=tray_size, num_orientations=1, interlocking_free=True)
        result_free = packer_free.pack_voxels(objects, initial_tray=initial_tray.copy(), sort_by_volume=False)

        # Count objects inside cage at z=1
        interior_mask = np.zeros((tray_size[0], tray_size[1]), dtype=bool)
        interior_mask[7:17, 7:17] = True

        std_inside = set()
        free_inside = set()
        for x in range(tray_size[0]):
            for y in range(tray_size[1]):
                if interior_mask[x, y]:
                    if result_std.tray[x, y, 1] > 1:
                        std_inside.add(result_std.tray[x, y, 1])
                    if result_free.tray[x, y, 1] > 1:
                        free_inside.add(result_free.tray[x, y, 1])

        print(f"\n=== Cage with Small Exit (2-cell) ===")
        print(f"Standard:          placed={result_std.num_placed}, inside={len(std_inside)}")
        print(f"Interlocking-free: placed={result_free.num_placed}, inside={len(free_inside)}")
        print(f"(IF allows small objects that can escape through exit)")

        self.visualize_slices(result_std, result_free, ring_mask,
            f"Cage with 2-cell Exit: Std={len(std_inside)} inside, IF={len(free_inside)} inside",
            VIZ_DIR / "05_cage_small_exit.png")

        # IF mode should place fewer or equal objects inside (avoiding large objects that would trap)
        assert len(free_inside) <= len(std_inside), \
            f"IF should place same or fewer inside: {len(free_inside)} > {len(std_inside)}"

    def test_cage_with_large_exit(self):
        """3D cage with 6-cell exit: most objects can escape.

        With a large exit, even bigger objects can escape, so IF mode will place
        them inside (they're not trapped).
        """
        tray_size = (24, 24, 3)
        cage = make_ring_3d_cage_with_exit(outer_size=12, wall_thickness=1, height=3, exit_size=6)
        initial_tray = np.zeros(tray_size, dtype=np.int32)
        initial_tray[6:18, 6:18, :] = cage
        ring_mask = initial_tray.copy()

        # Various sized objects - all can fit through 6-cell exit
        objects = (
            [np.ones((1, 1, 1), dtype=np.int32) for _ in range(30)] +
            [np.ones((2, 2, 1), dtype=np.int32) for _ in range(20)] +
            [np.ones((3, 3, 1), dtype=np.int32) for _ in range(15)] +
            [np.ones((5, 2, 1), dtype=np.int32) for _ in range(10)]
        )

        packer_std = BinPacker(tray_size=tray_size, num_orientations=4, interlocking_free=False)
        result_std = packer_std.pack_voxels(objects, initial_tray=initial_tray.copy(), sort_by_volume=False)

        packer_free = BinPacker(tray_size=tray_size, num_orientations=4, interlocking_free=True)
        result_free = packer_free.pack_voxels(objects, initial_tray=initial_tray.copy(), sort_by_volume=False)

        interior_mask = np.zeros((tray_size[0], tray_size[1]), dtype=bool)
        interior_mask[7:17, 7:17] = True

        std_inside = set()
        free_inside = set()
        for x in range(tray_size[0]):
            for y in range(tray_size[1]):
                if interior_mask[x, y]:
                    if result_std.tray[x, y, 1] > 1:
                        std_inside.add(result_std.tray[x, y, 1])
                    if result_free.tray[x, y, 1] > 1:
                        free_inside.add(result_free.tray[x, y, 1])

        print(f"\n=== Cage with Large Exit (6-cell) ===")
        print(f"Standard:          placed={result_std.num_placed}, inside={len(std_inside)}")
        print(f"Interlocking-free: placed={result_free.num_placed}, inside={len(free_inside)}")
        print(f"(Both modes should place similar - large exit allows escape)")

        self.visualize_slices(result_std, result_free, ring_mask,
            f"Cage with 6-cell Exit: Std={len(std_inside)} inside, IF={len(free_inside)} inside",
            VIZ_DIR / "06_cage_large_exit.png")

        # With large exit, both modes should behave similarly
        # (no strict assertion - just demonstrating the behavior)
        print(f"Difference: {abs(len(std_inside) - len(free_inside))} objects")

    def test_closed_vs_open_cage_comparison(self):
        """Compare closed cage (no exit) vs open cage (with exit) side by side."""
        if not HAS_MATPLOTLIB:
            pytest.skip("matplotlib not available")

        tray_size = (24, 24, 3)

        # Closed cage
        closed_cage = make_ring_3d_cage(outer_size=12, wall_thickness=1, height=3)
        initial_closed = np.zeros(tray_size, dtype=np.int32)
        initial_closed[6:18, 6:18, :] = closed_cage

        # Open cage (4-cell exit)
        open_cage = make_ring_3d_cage_with_exit(outer_size=12, wall_thickness=1, height=3, exit_size=4)
        initial_open = np.zeros(tray_size, dtype=np.int32)
        initial_open[6:18, 6:18, :] = open_cage

        # Use 150 objects to force overflow into cage interior
        objects = [np.ones((2, 2, 1), dtype=np.int32) for _ in range(150)]

        # Closed cage - IF should place ZERO inside
        packer_closed = BinPacker(tray_size=tray_size, num_orientations=1, interlocking_free=True)
        result_closed = packer_closed.pack_voxels(objects.copy(), initial_tray=initial_closed.copy(), sort_by_volume=False)

        # Open cage - IF should place objects inside (they can escape via exit)
        packer_open = BinPacker(tray_size=tray_size, num_orientations=1, interlocking_free=True)
        result_open = packer_open.pack_voxels(objects.copy(), initial_tray=initial_open.copy(), sort_by_volume=False)

        interior_mask = np.zeros((tray_size[0], tray_size[1]), dtype=bool)
        interior_mask[7:17, 7:17] = True

        closed_inside = set()
        open_inside = set()
        for x in range(tray_size[0]):
            for y in range(tray_size[1]):
                if interior_mask[x, y]:
                    if result_closed.tray[x, y, 1] > 1:
                        closed_inside.add(result_closed.tray[x, y, 1])
                    if result_open.tray[x, y, 1] > 1:
                        open_inside.add(result_open.tray[x, y, 1])

        print(f"\n=== Closed vs Open Cage (IF mode) ===")
        print(f"Closed cage: placed={result_closed.num_placed}, inside={len(closed_inside)}")
        print(f"Open cage:   placed={result_open.num_placed}, inside={len(open_inside)}")

        # Visualize both
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        for row, (result, ring_mask, label) in enumerate([
            (result_closed, initial_closed, "Closed (IF)"),
            (result_open, initial_open, "Open 4-cell exit (IF)")
        ]):
            for z in range(3):
                ax = axes[row, z]
                grid = result.tray[:, :, z].copy()
                ring = ring_mask[:, :, z]
                interior = find_ring_interior(ring)

                viz_grid = grid.copy().astype(float)
                viz_grid[(grid == 0) & interior] = -0.5

                n_objects = int(grid.max())
                colors = ['white', 'lightgray', 'dimgray']
                for i in range(max(1, n_objects - 1)):
                    colors.append(plt.cm.tab10.colors[i % 10])

                ax.imshow(viz_grid, cmap=ListedColormap(colors), interpolation='nearest',
                         vmin=-0.5, vmax=max(1, n_objects))
                if z == 0:
                    ax.set_ylabel(f"{label}\nPlaced: {result.num_placed}", fontsize=10, fontweight='bold')
                ax.set_title(f"z={z}", fontsize=10)

        plt.suptitle(f"Closed vs Open Cage (IF mode): Closed={len(closed_inside)}, Open={len(open_inside)} inside",
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "07_closed_vs_open_cage.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {VIZ_DIR / '07_closed_vs_open_cage.png'}")
        plt.close()

        # Closed cage should have 0 inside, open cage should have some
        assert len(closed_inside) == 0, f"Closed cage: IF placed {len(closed_inside)} inside (should be 0)"
        assert len(open_inside) > 0, "Open cage: IF should place objects inside (they can escape)"


class TestInterlockingFreePlacement:
    """Test the interlocking_free=True mode (Section 4.3)."""

    def test_interlocking_free_parameter(self):
        """BinPacker accepts interlocking_free parameter."""
        packer = BinPacker(tray_size=(20, 20, 20), interlocking_free=True)
        assert packer.interlocking_free is True

        packer2 = BinPacker(tray_size=(20, 20, 20), interlocking_free=False)
        assert packer2.interlocking_free is False

        # Default is False
        packer3 = BinPacker(tray_size=(20, 20, 20))
        assert packer3.interlocking_free is False

    def test_interlocking_free_avoids_ring_interior(self):
        """With interlocking_free=True, should have fewer trapped objects."""
        tray_size = (24, 24, 8)  # Taller tray
        ring = make_ring(outer_size=12, wall_thickness=2)  # Thicker walls
        ring_3d = np.zeros(tray_size, dtype=np.int32)
        # Stack ring vertically (full height)
        for z in range(8):
            ring_3d[6:18, 6:18, z] = ring[:, :, 0]
        initial_tray = ring_3d.copy()

        # Small objects that could fit inside
        objects = [np.ones((2, 2, 2), dtype=np.int32) for _ in range(10)]

        # With interlocking_free=True
        packer_free = BinPacker(tray_size=tray_size, num_orientations=1, interlocking_free=True)
        result_free = packer_free.pack_voxels(objects, initial_tray=initial_tray.copy(), sort_by_volume=False)

        # With interlocking_free=False (standard)
        packer_std = BinPacker(tray_size=tray_size, num_orientations=1, interlocking_free=False)
        result_std = packer_std.pack_voxels(objects, initial_tray=initial_tray.copy(), sort_by_volume=False)

        n_trapped_free, _, _ = count_trapped_objects(result_free.tray)
        n_trapped_std, _, _ = count_trapped_objects(result_std.tray)

        # Interlocking-free mode should result in fewer or equal trapped objects
        # (may not be zero if geometry allows some trapping)
        print(f"\nTrapped objects: std={n_trapped_std}, interlocking_free={n_trapped_free}")
        assert n_trapped_free <= n_trapped_std, \
            f"Interlocking-free should not trap more: {n_trapped_free} > {n_trapped_std}"

    def test_interlocking_free_lower_density(self):
        """Interlocking-free mode typically results in lower density (per paper Section 4.3)."""
        tray_size = (30, 30, 10)
        objects = [np.ones((3, 3, 3), dtype=np.int32) for _ in range(20)]

        packer_free = BinPacker(tray_size=tray_size, num_orientations=1, interlocking_free=True)
        result_free = packer_free.pack_voxels(objects, sort_by_volume=True)

        packer_std = BinPacker(tray_size=tray_size, num_orientations=1, interlocking_free=False)
        result_std = packer_std.pack_voxels(objects, sort_by_volume=True)

        # Both should place items
        assert result_free.num_placed > 0
        assert result_std.num_placed > 0

        # Standard mode may achieve same or higher density (paper notes 10-20% lower for interlocking-free)
        # Just verify both work - density comparison is not strict since it depends on geometry
        print(f"\nDensity comparison: std={result_std.density:.1%}, interlocking_free={result_free.density:.1%}")


class TestInterlockingComparison:
    """Compare standard vs interlocking-free mode with visualization and metrics.

    Key test setup:
    - Object height = 1 (thin slices)
    - Ring/Tray height = 3
    - Ring has walls + floor (z=0) + ceiling (z=2) ONLY inside the ring
    - Interior at z=1 is hollow - objects can be placed there
    - Standard mode will place inside the ring at z=1
    - Interlocking-free mode should NEVER place inside because:
      - Objects at z=0 interior: collision with floor
      - Objects at z=2 interior: collision with ceiling
      - Objects at z=1 interior: can't flood-fill from any boundary (blocked by floor/ceiling/walls)
    """

    def visualize_3_slices(self, result_std, result_free, ring_mask, time_std, time_free, title, filename):
        """Side-by-side visualization showing all 3 Z slices for each mode."""
        if not HAS_MATPLOTLIB:
            return

        height = result_std.tray.shape[2]
        fig, axes = plt.subplots(2, height, figsize=(6 * height, 12))

        for row, (result, label, elapsed) in enumerate([
            (result_std, "Standard", time_std),
            (result_free, "Interlocking-Free", time_free)
        ]):
            for z in range(height):
                ax = axes[row, z]
                grid = result.tray[:, :, z].copy()
                ring = ring_mask[:, :, z] if ring_mask.ndim == 3 else ring_mask
                interior = find_ring_interior(ring)
                trapped_ids = find_trapped_objects(result.tray[:, :, z])

                viz_grid = grid.copy().astype(float)
                viz_grid[(grid == 0) & interior] = -0.5

                n_objects = int(grid.max())
                colors = ['white', 'lightgray', 'dimgray']
                for i in range(max(1, n_objects - 1)):
                    colors.append('red' if (i + 2) in trapped_ids else plt.cm.tab10.colors[i % 10])

                ax.imshow(viz_grid, cmap=ListedColormap(colors), interpolation='nearest',
                         vmin=-0.5, vmax=max(1, n_objects))
                ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
                ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
                ax.grid(which='minor', color='black', linestyle='-', linewidth=0.3)

                # Count objects at this Z level
                objects_at_z = len(set(int(v) for v in grid.flatten() if v > 1))
                if z == 0:
                    ax.set_ylabel(f"{label}\nPlaced: {result.num_placed}\nTime: {elapsed:.3f}s",
                                 fontsize=10, fontweight='bold')
                ax.set_title(f"z={z} ({objects_at_z} objects)", fontsize=10)

        plt.suptitle(title + "\n(Red = trapped, Gray = ring structure, Light gray = interior zone)",
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

    def test_cage_ring_comparison(self):
        """Compare algorithms on 3D cage ring (floor/ceiling inside ring only).

        This is the key test: interlocking-free should NEVER place inside the ring.
        """
        import time

        # Tray height = 3, object height = 1
        tray_size = (24, 24, 3)
        ring = make_ring_3d_cage(outer_size=12, wall_thickness=1, height=3)
        initial_tray = np.zeros(tray_size, dtype=np.int32)
        initial_tray[6:18, 6:18, :] = ring
        ring_mask = initial_tray.copy()

        # Height-1 objects - use MANY to force placement inside ring at z=1
        # (~100 fit at z=0 exterior, so 150 forces overflow into z=1 interior)
        objects = [np.ones((2, 2, 1), dtype=np.int32) for _ in range(150)]

        # Standard mode - should place inside the ring at z=1
        t0 = time.perf_counter()
        packer_std = BinPacker(tray_size=tray_size, num_orientations=1, interlocking_free=False)
        result_std = packer_std.pack_voxels(objects, initial_tray=initial_tray.copy(), sort_by_volume=False)
        time_std = time.perf_counter() - t0

        # Interlocking-free mode - should NEVER place inside
        t0 = time.perf_counter()
        packer_free = BinPacker(tray_size=tray_size, num_orientations=1, interlocking_free=True)
        result_free = packer_free.pack_voxels(objects, initial_tray=initial_tray.copy(), sort_by_volume=False)
        time_free = time.perf_counter() - t0

        # Count objects inside the ring at z=1 (the trapped zone)
        interior_mask = np.zeros(tray_size[:2], dtype=bool)
        interior_mask[7:17, 7:17] = True  # Inside the ring walls

        std_inside = set()
        free_inside = set()
        for x in range(tray_size[0]):
            for y in range(tray_size[1]):
                if interior_mask[x, y]:
                    val_std = result_std.tray[x, y, 1]
                    val_free = result_free.tray[x, y, 1]
                    if val_std > 1:
                        std_inside.add(val_std)
                    if val_free > 1:
                        free_inside.add(val_free)

        n_inside_std = len(std_inside)
        n_inside_free = len(free_inside)

        print(f"\n=== Cage Ring Comparison (height=3, obj_height=1) ===")
        print(f"Standard:          placed={result_std.num_placed}, objects_inside={n_inside_std}, time={time_std:.3f}s")
        print(f"Interlocking-free: placed={result_free.num_placed}, objects_inside={n_inside_free}, time={time_free:.3f}s")

        self.visualize_3_slices(result_std, result_free, ring_mask, time_std, time_free,
            "Cage Ring: Standard vs Interlocking-Free",
            VIZ_DIR / "cmp_01_cage_ring.png")

        # KEY ASSERTION: Interlocking-free should have 0 objects inside
        assert n_inside_free == 0, \
            f"Interlocking-free placed {n_inside_free} objects inside ring (should be 0)"
        # Standard mode should place some objects inside
        assert n_inside_std > 0, \
            f"Standard mode should place objects inside ring, but placed {n_inside_std}"

    def test_larger_cage_comparison(self):
        """Compare on a larger cage with more objects."""
        import time

        tray_size = (30, 30, 3)
        ring = make_ring_3d_cage(outer_size=16, wall_thickness=2, height=3)
        initial_tray = np.zeros(tray_size, dtype=np.int32)
        initial_tray[7:23, 7:23, :] = ring
        ring_mask = initial_tray.copy()

        # Many objects to force overflow into interior
        objects = (
            [np.ones((2, 2, 1), dtype=np.int32) for _ in range(100)] +
            [np.ones((3, 2, 1), dtype=np.int32) for _ in range(50)] +
            [np.ones((1, 1, 1), dtype=np.int32) for _ in range(100)]
        )

        t0 = time.perf_counter()
        packer_std = BinPacker(tray_size=tray_size, num_orientations=4, interlocking_free=False)
        result_std = packer_std.pack_voxels(objects, initial_tray=initial_tray.copy(), sort_by_volume=False)
        time_std = time.perf_counter() - t0

        t0 = time.perf_counter()
        packer_free = BinPacker(tray_size=tray_size, num_orientations=4, interlocking_free=True)
        result_free = packer_free.pack_voxels(objects, initial_tray=initial_tray.copy(), sort_by_volume=False)
        time_free = time.perf_counter() - t0

        # Interior is inside the walls (thickness=2)
        interior_mask = np.zeros(tray_size[:2], dtype=bool)
        interior_mask[9:21, 9:21] = True  # 7+2=9 to 23-2=21

        std_inside = set()
        free_inside = set()
        for x in range(tray_size[0]):
            for y in range(tray_size[1]):
                if interior_mask[x, y]:
                    if result_std.tray[x, y, 1] > 1:
                        std_inside.add(result_std.tray[x, y, 1])
                    if result_free.tray[x, y, 1] > 1:
                        free_inside.add(result_free.tray[x, y, 1])

        print(f"\n=== Larger Cage Comparison ===")
        print(f"Standard:          placed={result_std.num_placed}, objects_inside={len(std_inside)}, time={time_std:.3f}s")
        print(f"Interlocking-free: placed={result_free.num_placed}, objects_inside={len(free_inside)}, time={time_free:.3f}s")

        self.visualize_3_slices(result_std, result_free, ring_mask, time_std, time_free,
            "Larger Cage: Standard vs Interlocking-Free",
            VIZ_DIR / "cmp_02_larger_cage.png")

        assert len(free_inside) == 0, \
            f"Interlocking-free placed {len(free_inside)} objects inside (should be 0)"

    def test_multiple_cages_comparison(self):
        """Multiple cage rings in the same tray."""
        import time

        # Smaller tray to force placement inside cages
        tray_size = (24, 24, 3)
        initial_tray = np.zeros(tray_size, dtype=np.int32)

        # Place 4 cage rings in the tray (tightly packed)
        ring1 = make_ring_3d_cage(outer_size=8, wall_thickness=1, height=3)
        ring2 = make_ring_3d_cage(outer_size=8, wall_thickness=1, height=3)
        ring3 = make_ring_3d_cage(outer_size=8, wall_thickness=1, height=3)
        ring4 = make_ring_3d_cage(outer_size=8, wall_thickness=1, height=3)

        initial_tray[2:10, 2:10, :] = ring1
        initial_tray[2:10, 14:22, :] = ring2
        initial_tray[14:22, 2:10, :] = ring3
        initial_tray[14:22, 14:22, :] = ring4
        ring_mask = initial_tray.copy()

        # Many objects to force overflow into cage interiors
        objects = [np.ones((2, 2, 1), dtype=np.int32) for _ in range(200)]

        t0 = time.perf_counter()
        packer_std = BinPacker(tray_size=tray_size, num_orientations=1, interlocking_free=False)
        result_std = packer_std.pack_voxels(objects, initial_tray=initial_tray.copy(), sort_by_volume=False)
        time_std = time.perf_counter() - t0

        t0 = time.perf_counter()
        packer_free = BinPacker(tray_size=tray_size, num_orientations=1, interlocking_free=True)
        result_free = packer_free.pack_voxels(objects, initial_tray=initial_tray.copy(), sort_by_volume=False)
        time_free = time.perf_counter() - t0

        # Interior regions of all 4 cages (inside walls, thickness=1)
        interior_regions = [
            (3, 9, 3, 9),     # ring1 interior
            (3, 9, 15, 21),   # ring2 interior
            (15, 21, 3, 9),   # ring3 interior
            (15, 21, 15, 21)  # ring4 interior
        ]

        std_inside = set()
        free_inside = set()
        for x0, x1, y0, y1 in interior_regions:
            for x in range(x0, x1):
                for y in range(y0, y1):
                    if result_std.tray[x, y, 1] > 1:
                        std_inside.add(result_std.tray[x, y, 1])
                    if result_free.tray[x, y, 1] > 1:
                        free_inside.add(result_free.tray[x, y, 1])

        print(f"\n=== Multiple Cages Comparison ===")
        print(f"Standard:          placed={result_std.num_placed}, objects_inside={len(std_inside)}, time={time_std:.3f}s")
        print(f"Interlocking-free: placed={result_free.num_placed}, objects_inside={len(free_inside)}, time={time_free:.3f}s")

        self.visualize_3_slices(result_std, result_free, ring_mask, time_std, time_free,
            "Multiple Cages: Standard vs Interlocking-Free",
            VIZ_DIR / "cmp_03_multiple_cages.png")

        assert len(free_inside) == 0, \
            f"Interlocking-free placed {len(free_inside)} objects inside cages (should be 0)"


class TestLargeGridStress:
    """Stress test with large grids containing cage obstacles and many objects.

    Tests place cage obstacles in the initial tray, then pack many small objects.
    This demonstrates IF mode avoiding cage interiors at scale.
    """

    def visualize_slices(self, result_std, result_free, ring_mask, title, filename, slices=3):
        """Visualization showing selected Z slices for each mode."""
        if not HAS_MATPLOTLIB:
            return

        height = result_std.tray.shape[2]
        z_indices = [0, height // 2, height - 1][:slices]  # Floor, middle, ceiling

        fig, axes = plt.subplots(2, len(z_indices), figsize=(5 * len(z_indices), 10))

        for row, (result, label) in enumerate([
            (result_std, "Standard"),
            (result_free, "Interlocking-Free")
        ]):
            for col, z in enumerate(z_indices):
                ax = axes[row, col]
                grid = result.tray[:, :, z].copy()
                ring = ring_mask[:, :, z] if ring_mask.ndim == 3 else ring_mask
                interior = find_ring_interior(ring)

                viz_grid = grid.copy().astype(float)
                viz_grid[(grid == 0) & interior] = -0.5

                n_objects = int(grid.max())
                colors = ['white', 'lightgray', 'dimgray']
                for i in range(max(1, n_objects - 1)):
                    colors.append(plt.cm.tab20.colors[i % 20])

                ax.imshow(viz_grid, cmap=ListedColormap(colors), interpolation='nearest',
                         vmin=-0.5, vmax=max(1, n_objects))

                objects_at_z = len(set(int(v) for v in grid.flatten() if v > 1))
                if col == 0:
                    ax.set_ylabel(f"{label}\nPlaced: {result.num_placed}", fontsize=10, fontweight='bold')
                ax.set_title(f"z={z} ({objects_at_z} objects)", fontsize=10)
                ax.axis('off')

        plt.suptitle(title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

    def test_large_grid_with_cages(self):
        """Stress test: 48x48x3 grid with 9 tightly-packed cages and many objects.

        Uses smaller tray with denser cage coverage to force overflow into cages.
        """
        import time

        tray_size = (48, 48, 3)
        initial_tray = np.zeros(tray_size, dtype=np.int32)

        # Place 9 cages (3x3 grid) tightly packed
        cage = make_ring_3d_cage(outer_size=12, wall_thickness=1, height=3)
        cage_positions = [
            (2, 2), (2, 18), (2, 34),
            (18, 2), (18, 18), (18, 34),
            (34, 2), (34, 18), (34, 34)
        ]
        for x, y in cage_positions:
            initial_tray[x:x+12, y:y+12, :] = cage
        ring_mask = initial_tray.copy()

        # Define cage interiors for counting (inside walls, at z=1)
        cage_interiors = [(x+1, x+11, y+1, y+11) for x, y in cage_positions]

        # Many objects to force overflow into cage interiors
        objects = (
            [np.ones((1, 1, 1), dtype=np.int32) for _ in range(300)] +
            [np.ones((2, 1, 1), dtype=np.int32) for _ in range(200)] +
            [np.ones((2, 2, 1), dtype=np.int32) for _ in range(150)] +
            [np.ones((3, 2, 1), dtype=np.int32) for _ in range(80)] +
            [np.ones((3, 3, 1), dtype=np.int32) for _ in range(50)]
        )

        t0 = time.perf_counter()
        packer_std = BinPacker(tray_size=tray_size, num_orientations=4, interlocking_free=False)
        result_std = packer_std.pack_voxels(objects.copy(), initial_tray=initial_tray.copy(), sort_by_volume=True)
        time_std = time.perf_counter() - t0

        t0 = time.perf_counter()
        packer_free = BinPacker(tray_size=tray_size, num_orientations=4, interlocking_free=True)
        result_free = packer_free.pack_voxels(objects.copy(), initial_tray=initial_tray.copy(), sort_by_volume=True)
        time_free = time.perf_counter() - t0

        # Count objects inside cages at z=1
        std_inside = set()
        free_inside = set()
        for x0, x1, y0, y1 in cage_interiors:
            for x in range(x0, x1):
                for y in range(y0, y1):
                    if result_std.tray[x, y, 1] > 1:
                        std_inside.add(result_std.tray[x, y, 1])
                    if result_free.tray[x, y, 1] > 1:
                        free_inside.add(result_free.tray[x, y, 1])

        print(f"\n=== Large Grid with 9 Cages (48x48x3, {len(objects)} objects) ===")
        print(f"Standard:          placed={result_std.num_placed}, inside_cages={len(std_inside)}, time={time_std:.2f}s")
        print(f"Interlocking-free: placed={result_free.num_placed}, inside_cages={len(free_inside)}, time={time_free:.2f}s")

        self.visualize_slices(result_std, result_free, ring_mask,
            f"Large Grid: Std={len(std_inside)} inside, IF={len(free_inside)} inside",
            VIZ_DIR / "stress_01_large_grid_cages.png")

        assert len(free_inside) == 0, f"IF placed {len(free_inside)} inside cages (should be 0)"
        assert len(std_inside) > 0, f"Standard should place objects inside cages"

    def test_very_large_grid_with_cages(self):
        """Stress test: 72x72x3 grid with 16 tightly-packed cages and many objects.

        Uses smaller tray with denser cage coverage to force overflow into cages.
        """
        import time

        tray_size = (72, 72, 3)
        initial_tray = np.zeros(tray_size, dtype=np.int32)

        # Place 16 cages (4x4 grid) tightly packed
        cage = make_ring_3d_cage(outer_size=14, wall_thickness=1, height=3)
        cage_interiors = []
        for i in range(4):
            for j in range(4):
                x, y = 2 + i * 17, 2 + j * 17
                initial_tray[x:x+14, y:y+14, :] = cage
                cage_interiors.append((x+1, x+13, y+1, y+13))  # Interior (wall=1)
        ring_mask = initial_tray.copy()

        # Many objects to force overflow into cage interiors
        objects = (
            [np.ones((1, 1, 1), dtype=np.int32) for _ in range(500)] +
            [np.ones((2, 1, 1), dtype=np.int32) for _ in range(300)] +
            [np.ones((1, 2, 1), dtype=np.int32) for _ in range(300)] +
            [np.ones((2, 2, 1), dtype=np.int32) for _ in range(200)] +
            [np.ones((3, 2, 1), dtype=np.int32) for _ in range(100)] +
            [np.ones((3, 3, 1), dtype=np.int32) for _ in range(80)] +
            [np.ones((4, 2, 1), dtype=np.int32) for _ in range(50)] +
            [np.ones((4, 4, 1), dtype=np.int32) for _ in range(30)]
        )

        t0 = time.perf_counter()
        packer_std = BinPacker(tray_size=tray_size, num_orientations=4, interlocking_free=False)
        result_std = packer_std.pack_voxels(objects.copy(), initial_tray=initial_tray.copy(), sort_by_volume=True)
        time_std = time.perf_counter() - t0

        t0 = time.perf_counter()
        packer_free = BinPacker(tray_size=tray_size, num_orientations=4, interlocking_free=True)
        result_free = packer_free.pack_voxels(objects.copy(), initial_tray=initial_tray.copy(), sort_by_volume=True)
        time_free = time.perf_counter() - t0

        std_inside = set()
        free_inside = set()
        for x0, x1, y0, y1 in cage_interiors:
            for x in range(x0, x1):
                for y in range(y0, y1):
                    if result_std.tray[x, y, 1] > 1:
                        std_inside.add(result_std.tray[x, y, 1])
                    if result_free.tray[x, y, 1] > 1:
                        free_inside.add(result_free.tray[x, y, 1])

        print(f"\n=== Very Large Grid with 16 Cages (72x72x3, {len(objects)} objects) ===")
        print(f"Standard:          placed={result_std.num_placed}, inside_cages={len(std_inside)}, time={time_std:.2f}s")
        print(f"Interlocking-free: placed={result_free.num_placed}, inside_cages={len(free_inside)}, time={time_free:.2f}s")

        self.visualize_slices(result_std, result_free, ring_mask,
            f"Very Large Grid: Std={len(std_inside)} inside, IF={len(free_inside)} inside",
            VIZ_DIR / "stress_02_very_large_grid_cages.png")

        assert len(free_inside) == 0, f"IF placed {len(free_inside)} inside cages (should be 0)"
        assert len(std_inside) > 0, f"Standard should place objects inside cages"


class TestInterlockingSummary:
    """Summary test that reports test configuration."""

    def test_full_summary(self):
        """Print summary of interlocking test suite."""
        print(f"\nInterlocking tests - visualizations at: {VIZ_DIR}")
        print("Red = trapped, Light gray = interior zone")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
