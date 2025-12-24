"""
3D voxel grid rotation utilities.

This module provides functions to rotate 3D voxel grids for orientation
sampling during placement search. The paper samples Euler angles at 90°
intervals, which produces 24 unique orientations (the rotation group of a cube).
"""

from typing import List, Tuple, Callable
import numpy as np


# Base 90-degree rotation matrices (matching np.rot90 behavior)
# RX: 90° around X axis, np.rot90(grid, k=1, axes=(1, 2))
# Transforms: Y → Z, Z → -Y
_RX = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0],
], dtype=np.float64)

# RY: 90° around Y axis, np.rot90(grid, k=1, axes=(2, 0))
# Transforms: Z → X, X → -Z
_RY = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0],
], dtype=np.float64)

# RZ: 90° around Z axis, np.rot90(grid, k=1, axes=(0, 1))
# Transforms: X → Y, Y → -X
_RZ = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1],
], dtype=np.float64)

_I = np.eye(3, dtype=np.float64)


def _generate_rotation_matrices() -> List[np.ndarray]:
    """Generate all 24 rotation matrices matching get_24_orientations() order.

    The matrices are generated following the exact same pattern as
    get_24_orientations() to ensure orientation_index maps correctly.
    """
    matrices = []

    # Powers of base rotations
    RX = _RX
    RX2 = RX @ RX
    RX3 = RX2 @ RX

    RY = _RY
    RY3 = RY @ RY @ RY

    RZ = _RZ
    RZ2 = RZ @ RZ
    RZ3 = RZ2 @ RZ

    # Face 1: Identity (top face up) + Z rotations
    for Rz in [_I, RZ, RZ2, RZ3]:
        matrices.append(Rz.copy())

    # Face 2: RX (front face up) + Z rotations
    for Rz in [_I, RZ, RZ2, RZ3]:
        matrices.append(Rz @ RX)

    # Face 3: RX² (bottom face up) + Z rotations
    for Rz in [_I, RZ, RZ2, RZ3]:
        matrices.append(Rz @ RX2)

    # Face 4: RX³ (back face up) + Z rotations
    for Rz in [_I, RZ, RZ2, RZ3]:
        matrices.append(Rz @ RX3)

    # Face 5: RY (right face up) + Z rotations
    for Rz in [_I, RZ, RZ2, RZ3]:
        matrices.append(Rz @ RY)

    # Face 6: RY³ (left face up) + Z rotations
    for Rz in [_I, RZ, RZ2, RZ3]:
        matrices.append(Rz @ RY3)

    return matrices


# Pre-computed 3x3 rotation matrices for all 24 orientations
ROTATION_MATRICES_3x3: List[np.ndarray] = _generate_rotation_matrices()


def get_rotation_matrix_3x3(orientation_index: int) -> np.ndarray:
    """Get the 3x3 rotation matrix for a given orientation index.

    Parameters
    ----------
    orientation_index : int
        Index from 0 to 23, matching the order returned by get_24_orientations().

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.

    Raises
    ------
    IndexError
        If orientation_index is out of range [0, 23].
    """
    if not 0 <= orientation_index < 24:
        raise IndexError(f"orientation_index must be 0-23, got {orientation_index}")
    return ROTATION_MATRICES_3x3[orientation_index].copy()


def get_rotation_matrix_4x4(orientation_index: int) -> np.ndarray:
    """Get the 4x4 homogeneous rotation matrix for a given orientation index.

    This is useful for Blender export where 4x4 transformation matrices
    are required.

    Parameters
    ----------
    orientation_index : int
        Index from 0 to 23, matching the order returned by get_24_orientations().

    Returns
    -------
    np.ndarray
        4x4 homogeneous rotation matrix (rotation only, no translation).

    Raises
    ------
    IndexError
        If orientation_index is out of range [0, 23].
    """
    R = np.eye(4, dtype=np.float64)
    R[:3, :3] = get_rotation_matrix_3x3(orientation_index)
    return R


def rotate_90_x(grid: np.ndarray) -> np.ndarray:
    """Rotate grid 90° around X axis."""
    # X stays, Y->Z, Z->-Y
    return np.rot90(grid, k=1, axes=(1, 2))


def rotate_90_y(grid: np.ndarray) -> np.ndarray:
    """Rotate grid 90° around Y axis."""
    # Y stays, X->-Z, Z->X
    return np.rot90(grid, k=1, axes=(2, 0))


def rotate_90_z(grid: np.ndarray) -> np.ndarray:
    """Rotate grid 90° around Z axis."""
    # Z stays, X->Y, Y->-X
    return np.rot90(grid, k=1, axes=(0, 1))


def get_24_orientations(grid: np.ndarray) -> List[np.ndarray]:
    """
    Generate all 24 unique orientations of a 3D grid.

    These correspond to the 24 rotational symmetries of a cube.
    The rotations are generated systematically to cover all unique orientations.

    Parameters
    ----------
    grid : np.ndarray
        3D voxel grid to rotate.

    Returns
    -------
    list of np.ndarray
        List of 24 rotated grids (first one is the original).
    """
    orientations = []

    # Start with identity
    g = grid.copy()

    # 6 faces can be "up" (pointing in +Z direction)
    # For each face up, 4 rotations around Z axis

    # Face 1: Original orientation (top face up)
    for _ in range(4):
        orientations.append(g.copy())
        g = rotate_90_z(g)

    # Face 2: Rotate around X to get front face up
    g = rotate_90_x(grid)
    for _ in range(4):
        orientations.append(g.copy())
        g = rotate_90_z(g)

    # Face 3: Rotate around X twice to get bottom face up
    g = rotate_90_x(rotate_90_x(grid))
    for _ in range(4):
        orientations.append(g.copy())
        g = rotate_90_z(g)

    # Face 4: Rotate around X three times to get back face up
    g = rotate_90_x(rotate_90_x(rotate_90_x(grid)))
    for _ in range(4):
        orientations.append(g.copy())
        g = rotate_90_z(g)

    # Face 5: Rotate around Y to get right face up
    g = rotate_90_y(grid)
    for _ in range(4):
        orientations.append(g.copy())
        g = rotate_90_z(g)

    # Face 6: Rotate around Y three times to get left face up
    g = rotate_90_y(rotate_90_y(rotate_90_y(grid)))
    for _ in range(4):
        orientations.append(g.copy())
        g = rotate_90_z(g)

    return orientations


def get_orientations(grid: np.ndarray, num_orientations: int = 1) -> List[np.ndarray]:
    """
    Get a specified number of orientations for a grid.

    Parameters
    ----------
    grid : np.ndarray
        3D voxel grid to rotate.
    num_orientations : int
        Number of orientations to return:
        - 1: Original only
        - 4: Original + 3 rotations around Z axis
        - 6: Original + rotations to put each face "forward"
        - 24: All 24 unique orientations (full cube symmetry group)

    Returns
    -------
    list of np.ndarray
        List of rotated grids.

    Raises
    ------
    ValueError
        If num_orientations is not 1, 4, 6, or 24.
    """
    if num_orientations == 1:
        return [grid]
    elif num_orientations == 4:
        # Just Z-axis rotations (useful for "upright" objects)
        orientations = []
        g = grid.copy()
        for _ in range(4):
            orientations.append(g.copy())
            g = rotate_90_z(g)
        return orientations
    elif num_orientations == 6:
        # 6 face orientations (one rotation per face)
        return [
            grid,                                          # Top up
            rotate_90_x(grid),                             # Front up
            rotate_90_x(rotate_90_x(grid)),                # Bottom up
            rotate_90_x(rotate_90_x(rotate_90_x(grid))),   # Back up
            rotate_90_y(grid),                             # Right up
            rotate_90_y(rotate_90_y(rotate_90_y(grid))),   # Left up
        ]
    elif num_orientations == 24:
        return get_24_orientations(grid)
    else:
        raise ValueError(
            f"num_orientations must be 1, 4, 6, or 24, got {num_orientations}"
        )


def make_contiguous(grid: np.ndarray) -> np.ndarray:
    """Ensure array is C-contiguous (required for C++ bindings)."""
    if not grid.flags['C_CONTIGUOUS']:
        return np.ascontiguousarray(grid)
    return grid
