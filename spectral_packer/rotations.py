"""
3D voxel grid rotation utilities.

This module provides functions to rotate 3D voxel grids for orientation
sampling during placement search. The paper samples Euler angles at 90°
intervals, which produces 24 unique orientations (the rotation group of a cube).
"""

from typing import List, Tuple, Callable
import numpy as np


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
