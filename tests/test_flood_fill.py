"""
GPU Flood Fill Tests - Verifies GPU implementation matches CPU reference.

Implements Algorithm 3 from the paper: Flood-fill disassembly to determine
interlocking-free placement positions for an object.

Key insight: The flood fill operates on the COLLISION METRIC (not occupancy).
The collision metric ζ_A,Ω(q) = number of overlapping voxels if object A
is placed at position q. Zero means valid placement, positive means collision.
"""
import numpy as np
import pytest
from collections import deque
from pathlib import Path

from spectral_packer import interlocking_free_positions, dft_corr3

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


VIZ_DIR = Path("/tmp/flood_fill_tests")


def setup_module(module):
    """Create output directory for visualizations."""
    VIZ_DIR.mkdir(exist_ok=True)


# =============================================================================
# CPU Reference Implementation (Algorithm 3 from paper)
# =============================================================================

def cpu_collision_metric(tray, obj):
    """
    Compute collision metric ζ_A,Ω via correlation.

    ζ(q) = number of overlapping voxels if object is placed at position q.
    Zero means no collision, positive means collision.

    Uses the same FFT-based correlation as the GPU implementation.

    Parameters
    ----------
    tray : np.ndarray
        3D int array of current tray state (0=empty, non-zero=occupied)
    obj : np.ndarray
        3D int array of object to place (0=empty, non-zero=occupied)

    Returns
    -------
    np.ndarray
        Collision metric, same size as tray
    """
    # Pad object to tray size
    padded_obj = np.zeros(tray.shape, dtype=np.int32)
    padded_obj[:obj.shape[0], :obj.shape[1], :obj.shape[2]] = obj
    return dft_corr3(tray, padded_obj)


def cpu_interlocking_free_positions(tray, obj):
    """
    CPU reference: Algorithm 3 - Flood fill on collision metric.

    Determines which positions allow interlocking-free placement of the object.
    A position is interlocking-free if the object can be placed there AND
    moved out to the boundary without colliding with existing objects.

    Parameters
    ----------
    tray : np.ndarray
        3D int array of current tray state (0=empty, non-zero=occupied)
    obj : np.ndarray
        3D int array of object to place (0=empty, non-zero=occupied)

    Returns
    -------
    np.ndarray
        3D int array where 1=interlocking-free position, 0=blocked/collision
    """
    # Get object bounds (non-zero extent)
    obj_coords = np.argwhere(obj != 0)
    if len(obj_coords) == 0:
        # Empty object - all positions are valid
        return np.ones(tray.shape, dtype=np.int32)

    obj_max = obj_coords.max(axis=0)  # (max_i, max_j, max_k)

    # Compute collision metric
    collision_metric = cpu_collision_metric(tray, obj)

    # Grid dimensions
    nx, ny, nz = tray.shape

    # Initialize labels (Algorithm 3 lines 4-6):
    # 2 = colliding position (ζ ≠ 0)
    # 1 = non-colliding position (ζ = 0), not yet reached
    # 0 = feasible/reachable (boundary or flood-filled)
    labels = np.ones((nx, ny, nz), dtype=np.int32)
    labels[collision_metric != 0] = 2  # Colliding positions

    # Also mark out-of-bounds positions as colliding
    # Object placed at (i,j,k) occupies up to (i+obj_max[0], j+obj_max[1], k+obj_max[2])
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if (i + obj_max[0] >= nx or
                    j + obj_max[1] >= ny or
                    k + obj_max[2] >= nz):
                    labels[i, j, k] = 2  # Would clip out of tray

    # Seed boundary positions where object would be fully outside
    # These are positions where the object's bounding box doesn't intersect tray
    # For simplicity, use the 6 faces of the grid minus object extent
    queue = deque()

    # The boundary X consists of positions where object is "outside"
    # Since we're working with the tray grid, boundary is where placing
    # the object at that position would put it at the edge of the tray
    # Actually, per the paper: "feasible starting locations X are the voxels
    # on the six boundary faces of the grid"
    # But we need the object to be able to EXIT, so we seed from positions
    # where the object could slide out

    # Seed from face i=0 (can exit in -x direction)
    for j in range(ny):
        for k in range(nz):
            if labels[0, j, k] == 1:  # Non-colliding
                labels[0, j, k] = 0
                queue.append((0, j, k))

    # Seed from face j=0 (can exit in -y direction)
    for i in range(nx):
        for k in range(nz):
            if labels[i, 0, k] == 1:
                labels[i, 0, k] = 0
                queue.append((i, 0, k))

    # Seed from face k=0 (can exit in -z direction)
    for i in range(nx):
        for j in range(ny):
            if labels[i, j, 0] == 1:
                labels[i, j, 0] = 0
                queue.append((i, j, 0))

    # Seed from face i=nx-1-obj_max[0] (can exit in +x direction)
    exit_i = max(0, nx - 1 - obj_max[0])
    for j in range(ny):
        for k in range(nz):
            if exit_i < nx and labels[exit_i, j, k] == 1:
                labels[exit_i, j, k] = 0
                queue.append((exit_i, j, k))

    # Seed from face j=ny-1-obj_max[1] (can exit in +y direction)
    exit_j = max(0, ny - 1 - obj_max[1])
    for i in range(nx):
        for k in range(nz):
            if exit_j < ny and labels[i, exit_j, k] == 1:
                labels[i, exit_j, k] = 0
                queue.append((i, exit_j, k))

    # Seed from face k=nz-1-obj_max[2] (can exit in +z direction)
    exit_k = max(0, nz - 1 - obj_max[2])
    for i in range(nx):
        for j in range(ny):
            if exit_k < nz and labels[i, j, exit_k] == 1:
                labels[i, j, exit_k] = 0
                queue.append((i, j, exit_k))

    # BFS flood fill (Algorithm 3 lines 7-9)
    # Propagate 0 to neighbors that are 1 (non-colliding but not yet reached)
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    while queue:
        x, y, z = queue.popleft()
        for dx, dy, dz in directions:
            nx_, ny_, nz_ = x + dx, y + dy, z + dz
            if (0 <= nx_ < nx and 0 <= ny_ < ny and 0 <= nz_ < nz
                    and labels[nx_, ny_, nz_] == 1):
                labels[nx_, ny_, nz_] = 0
                queue.append((nx_, ny_, nz_))

    # Return mask: 1 where interlocking-free (label=0), 0 otherwise
    return (labels == 0).astype(np.int32)


# =============================================================================
# Test Fixtures
# =============================================================================

def make_3d_box(outer_size, wall_thickness=1):
    """Create a 3D hollow box (enclosed on all 6 faces)."""
    grid = np.ones((outer_size, outer_size, outer_size), dtype=np.int32)
    t = wall_thickness
    if outer_size > 2 * t:
        grid[t:-t, t:-t, t:-t] = 0  # Hollow interior
    return grid


def make_3d_box_with_exit(outer_size, wall_thickness=1, exit_size=1):
    """Create a 3D hollow box with an exit gap on the right face."""
    grid = make_3d_box(outer_size, wall_thickness)
    t = wall_thickness
    mid = outer_size // 2
    half_exit = exit_size // 2
    start, end = mid - half_exit, mid - half_exit + exit_size
    # Remove right face wall in a square region
    grid[start:end, -t:, start:end] = 0
    return grid


def make_l_shape(size=3):
    """
    Create an L-shaped 3D object:
    x..
    xx.
    xxx
    """
    grid = np.zeros((size, size, size), dtype=np.int32)
    for i in range(size):
        for j in range(i + 1):
            grid[i, j, :] = 1
    return grid


# =============================================================================
# Visualization
# =============================================================================

def visualize_interlocking_free(tray, obj, reachable, title, filename, slice_z=None):
    """
    Visualize 2D slice showing interlocking-free positions for an object.

    Parameters
    ----------
    tray : np.ndarray
        3D tray occupancy
    obj : np.ndarray
        3D object being tested
    reachable : np.ndarray
        3D mask of interlocking-free positions
    title : str
        Plot title
    filename : Path
        Output file path
    slice_z : int, optional
        Z-slice to visualize. If None, uses middle slice.
    """
    if not HAS_MATPLOTLIB:
        return

    if slice_z is None:
        slice_z = tray.shape[2] // 2

    tray_2d = tray[:, :, slice_z]
    reach_2d = reachable[:, :, slice_z]

    # Get object 2D slice for display
    obj_slice_z = min(slice_z, obj.shape[2] - 1) if obj.shape[2] > 0 else 0
    obj_2d = obj[:, :, obj_slice_z] if obj.shape[2] > obj_slice_z else obj[:, :, 0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Tray state
    ax1 = axes[0]
    ax1.imshow(tray_2d.T, cmap='gray_r', interpolation='nearest', origin='lower')
    ax1.set_title(f'Tray (z={slice_z})', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # Panel 2: Object shape
    ax2 = axes[1]
    obj_display = np.zeros((max(obj.shape[0], 10), max(obj.shape[1], 10)))
    obj_display[:obj_2d.shape[0], :obj_2d.shape[1]] = obj_2d
    ax2.imshow(obj_display.T, cmap='Blues', interpolation='nearest', origin='lower')
    ax2.set_title(f'Object (z={obj_slice_z})\nSize: {obj.shape}', fontsize=12)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    # Panel 3: Interlocking-free positions
    ax3 = axes[2]
    # Build visualization:
    # 0 = blocked/collision (red)
    # 1 = wall in tray (dark gray)
    # 2 = interlocking-free (green)
    viz = np.zeros_like(tray_2d, dtype=int)
    viz[tray_2d != 0] = 1  # Walls/obstacles in tray
    viz[reach_2d == 1] = 2  # Interlocking-free positions
    viz[(tray_2d == 0) & (reach_2d == 0)] = 0  # Blocked positions

    cmap = ListedColormap(['lightcoral', 'dimgray', 'lightgreen'])
    ax3.imshow(viz.T, cmap=cmap, interpolation='nearest', origin='lower')
    ax3.set_title(f'Interlocking-Free Positions (z={slice_z})', fontsize=12)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')

    # Add grid lines
    for ax in axes:
        ax.set_xticks(np.arange(-0.5, ax.images[0].get_array().shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, ax.images[0].get_array().shape[0], 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.3, alpha=0.5)

    # Stats
    blocked_count = np.sum((tray_2d == 0) & (reach_2d == 0))
    free_count = np.sum(reach_2d == 1)
    total_free_3d = np.sum(reachable == 1)
    stats_text = f"Slice stats:\n  Blocked: {blocked_count}\n  Free: {free_count}\n\n3D total free: {total_free_3d}"
    fig.text(0.98, 0.5, stats_text, transform=fig.transFigure,
             verticalalignment='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


# =============================================================================
# CPU-only Tests (always run)
# =============================================================================

class TestCPUInterlockingFree:
    """Test CPU reference implementation of Algorithm 3."""

    def test_empty_tray_small_object(self):
        """Empty tray - all positions should be interlocking-free for small object."""
        tray = np.zeros((20, 20, 20), dtype=np.int32)
        obj = np.ones((3, 3, 3), dtype=np.int32)  # 3x3x3 cube

        result = cpu_interlocking_free_positions(tray, obj)

        # Most positions should be valid (except near edges where object clips)
        # Object can be placed at positions [0..17, 0..17, 0..17]
        assert result[0, 0, 0] == 1, "Corner should be valid"
        assert result[10, 10, 10] == 1, "Center should be valid"
        # Near edge where object would clip
        assert result[19, 19, 19] == 0, "Edge position should be invalid (object clips)"

    def test_filled_tray(self):
        """Fully occupied tray - no valid positions."""
        tray = np.ones((20, 20, 20), dtype=np.int32)
        obj = np.ones((3, 3, 3), dtype=np.int32)

        result = cpu_interlocking_free_positions(tray, obj)

        # No positions should be valid (everything collides)
        assert result.sum() == 0, "No valid positions in filled tray"

    def test_closed_box_small_object(self):
        """Closed 3D box - small object cannot be placed inside."""
        box = make_3d_box(12, wall_thickness=1)
        tray = np.zeros((20, 20, 20), dtype=np.int32)
        tray[4:16, 4:16, 4:16] = box

        # Small 2x2x2 object
        obj = np.ones((2, 2, 2), dtype=np.int32)

        result = cpu_interlocking_free_positions(tray, obj)

        # Interior of box should NOT be interlocking-free
        # (object could fit but couldn't be extracted)
        assert result[9, 9, 9] == 0, "Interior should be blocked (interlocking)"
        # Exterior should be valid
        assert result[0, 0, 0] == 1, "Exterior corner should be valid"

        visualize_interlocking_free(tray, obj, result, "Closed Box + 2x2x2 Object (CPU)",
                                     VIZ_DIR / "cpu_closed_box_obj.png", slice_z=10)

    def test_box_with_exit_fitting_object(self):
        """Box with 3x3 exit - 2x2x2 object CAN be placed inside (can exit)."""
        box = make_3d_box_with_exit(12, wall_thickness=1, exit_size=3)
        tray = np.zeros((20, 20, 20), dtype=np.int32)
        tray[4:16, 4:16, 4:16] = box

        # 2x2x2 object fits through 3x3 exit
        obj = np.ones((2, 2, 2), dtype=np.int32)

        result = cpu_interlocking_free_positions(tray, obj)

        # Interior SHOULD be interlocking-free (object can exit through hole)
        assert result[9, 9, 9] == 1, "Interior should be valid (object fits through exit)"

        visualize_interlocking_free(tray, obj, result, "Box with Exit + 2x2x2 Object (CPU)",
                                     VIZ_DIR / "cpu_box_exit_obj.png", slice_z=10)

    def test_box_with_exit_large_object(self):
        """Box with 3x3 exit - 4x4x4 object CANNOT fit through."""
        box = make_3d_box_with_exit(12, wall_thickness=1, exit_size=3)
        tray = np.zeros((20, 20, 20), dtype=np.int32)
        tray[4:16, 4:16, 4:16] = box

        # 4x4x4 object does NOT fit through 3x3 exit
        obj = np.ones((4, 4, 4), dtype=np.int32)

        result = cpu_interlocking_free_positions(tray, obj)

        # Interior should NOT be interlocking-free (object too big for exit)
        assert result[8, 8, 8] == 0, "Interior should be blocked (object too big for exit)"

        visualize_interlocking_free(tray, obj, result, "Box with Exit + 4x4x4 Object (CPU)",
                                     VIZ_DIR / "cpu_box_exit_large_obj.png", slice_z=10)

    def test_l_shape_object(self):
        """L-shaped object in tray with obstacles."""
        tray = np.zeros((24, 24, 24), dtype=np.int32)

        # Add a closed box
        box = make_3d_box(10, wall_thickness=1)
        tray[7:17, 7:17, 7:17] = box

        # L-shaped object
        obj = make_l_shape(size=3)

        result = cpu_interlocking_free_positions(tray, obj)

        # Exterior positions should be valid
        assert result[0, 0, 0] == 1, "Exterior should be valid"
        # Interior of closed box should be blocked
        assert result[11, 11, 11] == 0, "Interior of closed box should be blocked"

        visualize_interlocking_free(tray, obj, result, "Tray + L-shaped Object (CPU)",
                                     VIZ_DIR / "cpu_l_shape_obj.png", slice_z=12)

    def test_narrow_corridor(self):
        """Object must navigate narrow corridor to reach interior."""
        tray = np.zeros((30, 30, 30), dtype=np.int32)

        # Create walls with a narrow corridor
        # Left wall
        tray[5:25, 10:12, 5:25] = 1
        # Right wall
        tray[5:25, 18:20, 5:25] = 1
        # Back wall with 3x3 opening
        tray[5:25, 12:18, 22:24] = 1
        tray[5:12, 12:18, 22:24] = 1  # Close top part
        tray[17:25, 12:18, 22:24] = 1  # Close bottom part
        # Opening is at [12:17, 12:18, 22:24] - 5x6x2

        # 3x3x3 object should fit through
        obj = np.ones((3, 3, 3), dtype=np.int32)

        result = cpu_interlocking_free_positions(tray, obj)

        # Position inside corridor should be valid (can exit through opening)
        assert result[14, 14, 10] == 1, "Inside corridor should be valid"
        # Position at entrance should be valid
        assert result[14, 14, 0] == 1, "Corridor entrance should be valid"

        visualize_interlocking_free(tray, obj, result, "Narrow Corridor + 3x3x3 Object (CPU)",
                                     VIZ_DIR / "cpu_corridor_obj.png", slice_z=15)


# =============================================================================
# GPU vs CPU Comparison Tests
# =============================================================================

class TestGPUInterlockingFree:
    """Test GPU implementation matches CPU reference."""

    def test_empty_tray_small_object(self):
        """Empty tray - compare GPU vs CPU."""
        tray = np.zeros((20, 20, 20), dtype=np.int32)
        obj = np.ones((3, 3, 3), dtype=np.int32)

        gpu_result = interlocking_free_positions(tray, obj)
        cpu_result = cpu_interlocking_free_positions(tray, obj)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_filled_tray(self):
        """Fully occupied tray."""
        tray = np.ones((20, 20, 20), dtype=np.int32)
        obj = np.ones((3, 3, 3), dtype=np.int32)

        gpu_result = interlocking_free_positions(tray, obj)
        cpu_result = cpu_interlocking_free_positions(tray, obj)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_closed_box_small_object(self):
        """Closed box - small object blocked inside."""
        box = make_3d_box(12, wall_thickness=1)
        tray = np.zeros((20, 20, 20), dtype=np.int32)
        tray[4:16, 4:16, 4:16] = box

        obj = np.ones((2, 2, 2), dtype=np.int32)

        gpu_result = interlocking_free_positions(tray, obj)
        cpu_result = cpu_interlocking_free_positions(tray, obj)
        np.testing.assert_array_equal(gpu_result, cpu_result)

        visualize_interlocking_free(tray, obj, gpu_result, "Closed Box + 2x2x2 Object (GPU)",
                                     VIZ_DIR / "gpu_closed_box_obj.png", slice_z=10)

    def test_box_with_exit_fitting_object(self):
        """Box with exit - object fits through."""
        box = make_3d_box_with_exit(12, wall_thickness=1, exit_size=3)
        tray = np.zeros((20, 20, 20), dtype=np.int32)
        tray[4:16, 4:16, 4:16] = box

        obj = np.ones((2, 2, 2), dtype=np.int32)

        gpu_result = interlocking_free_positions(tray, obj)
        cpu_result = cpu_interlocking_free_positions(tray, obj)
        np.testing.assert_array_equal(gpu_result, cpu_result)

        visualize_interlocking_free(tray, obj, gpu_result, "Box with Exit + 2x2x2 Object (GPU)",
                                     VIZ_DIR / "gpu_box_exit_obj.png", slice_z=10)

    def test_box_with_exit_large_object(self):
        """Box with exit - object too large to fit through."""
        box = make_3d_box_with_exit(12, wall_thickness=1, exit_size=3)
        tray = np.zeros((20, 20, 20), dtype=np.int32)
        tray[4:16, 4:16, 4:16] = box

        obj = np.ones((4, 4, 4), dtype=np.int32)

        gpu_result = interlocking_free_positions(tray, obj)
        cpu_result = cpu_interlocking_free_positions(tray, obj)
        np.testing.assert_array_equal(gpu_result, cpu_result)

        visualize_interlocking_free(tray, obj, gpu_result, "Box with Exit + 4x4x4 Object (GPU)",
                                     VIZ_DIR / "gpu_box_exit_large_obj.png", slice_z=10)

    def test_l_shape_object(self):
        """L-shaped object."""
        tray = np.zeros((24, 24, 24), dtype=np.int32)
        box = make_3d_box(10, wall_thickness=1)
        tray[7:17, 7:17, 7:17] = box

        obj = make_l_shape(size=3)

        gpu_result = interlocking_free_positions(tray, obj)
        cpu_result = cpu_interlocking_free_positions(tray, obj)
        np.testing.assert_array_equal(gpu_result, cpu_result)

        visualize_interlocking_free(tray, obj, gpu_result, "Tray + L-shaped Object (GPU)",
                                     VIZ_DIR / "gpu_l_shape_obj.png", slice_z=12)

    def test_narrow_corridor(self):
        """Object must navigate narrow corridor to reach interior."""
        tray = np.zeros((30, 30, 30), dtype=np.int32)

        # Create walls with a narrow corridor
        tray[5:25, 10:12, 5:25] = 1  # Left wall
        tray[5:25, 18:20, 5:25] = 1  # Right wall
        tray[5:25, 12:18, 22:24] = 1  # Back wall with opening
        tray[5:12, 12:18, 22:24] = 1  # Close top part
        tray[17:25, 12:18, 22:24] = 1  # Close bottom part

        obj = np.ones((3, 3, 3), dtype=np.int32)

        gpu_result = interlocking_free_positions(tray, obj)
        cpu_result = cpu_interlocking_free_positions(tray, obj)
        np.testing.assert_array_equal(gpu_result, cpu_result)

        visualize_interlocking_free(tray, obj, gpu_result, "Narrow Corridor + 3x3x3 Object (GPU)",
                                     VIZ_DIR / "gpu_corridor_obj.png", slice_z=15)

    def test_large_grid_64(self):
        """64x64x64 grid performance test."""
        tray = np.zeros((64, 64, 64), dtype=np.int32)
        tray[20:44, 20:44, 20:44] = 1
        tray[25:39, 25:39, 25:39] = 0  # Hollow interior

        obj = np.ones((3, 3, 3), dtype=np.int32)

        gpu_result = interlocking_free_positions(tray, obj)
        cpu_result = cpu_interlocking_free_positions(tray, obj)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_large_grid_100(self):
        """100x100x100 grid."""
        tray = np.zeros((100, 100, 100), dtype=np.int32)
        tray[30:70, 30:70, 30:70] = 1
        tray[35:65, 35:65, 35:65] = 0  # Hollow

        obj = np.ones((5, 5, 5), dtype=np.int32)

        gpu_result = interlocking_free_positions(tray, obj)
        cpu_result = cpu_interlocking_free_positions(tray, obj)

        # Verify key properties match
        np.testing.assert_array_equal(gpu_result, cpu_result)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
