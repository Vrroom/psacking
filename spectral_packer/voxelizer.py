"""
Voxelization utilities for converting meshes to voxel grids.

This module provides utilities for converting 3D triangle meshes to
voxel grids suitable for the spectral packing algorithm.

Examples
--------
>>> from spectral_packer import Voxelizer
>>> vox = Voxelizer(resolution=128)
>>> grid = vox.voxelize_file("model.stl")
>>> print(f"Voxel grid shape: {grid.shape}")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from .mesh_io import load_mesh


class Voxelizer:
    """
    Mesh to voxel grid converter.

    This class provides methods for converting 3D meshes (from files or
    vertex/face arrays) into voxel grids for use with the packing algorithm.

    Parameters
    ----------
    resolution : int, default 128
        Maximum resolution along the longest axis. The mesh will be
        voxelized to fit within a grid of at most this size.

    Attributes
    ----------
    resolution : int
        The voxelization resolution.

    Examples
    --------
    >>> vox = Voxelizer(resolution=64)
    >>> grid = vox.voxelize_file("model.stl")
    >>> print(f"Grid shape: {grid.shape}")

    >>> vertices, faces = load_mesh("model.obj")
    >>> grid = vox.voxelize_mesh(vertices, faces)
    """

    def __init__(self, resolution: int = 128):
        if resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")
        self.resolution = resolution

    def voxelize_file(
        self,
        path: Union[str, Path],
        validate: bool = True,
        repair: bool = True,
    ) -> np.ndarray:
        """
        Voxelize a mesh from a file.

        Parameters
        ----------
        path : str or Path
            Path to the mesh file.
        validate : bool, default True
            Validate the mesh during loading.
        repair : bool, default True
            Attempt to repair invalid meshes.

        Returns
        -------
        np.ndarray
            3D int32 array with 1 for occupied voxels, 0 for empty.
        """
        path = Path(path)
        suffix = path.suffix.lower()

        # For STL files, try to use the C++ voxelizer (faster)
        if suffix == ".stl":
            try:
                from . import _CORE_AVAILABLE, voxelize_stl
                if _CORE_AVAILABLE:
                    return voxelize_stl(str(path), self.resolution)
            except Exception:
                pass  # Fall back to Python voxelization

        # Load mesh and voxelize
        vertices, faces = load_mesh(path, validate=validate, repair=repair)
        return self.voxelize_mesh(vertices, faces)

    def voxelize_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        fill: bool = True,
    ) -> np.ndarray:
        """
        Voxelize a mesh from vertex and face arrays.

        Parameters
        ----------
        vertices : np.ndarray
            Vertex positions, shape (N, 3).
        faces : np.ndarray
            Triangle indices, shape (M, 3).
        fill : bool, default True
            Fill the interior of the mesh.

        Returns
        -------
        np.ndarray
            3D int32 array with 1 for occupied voxels, 0 for empty.
        """
        try:
            import trimesh
        except ImportError:
            raise ImportError(
                "trimesh required for mesh voxelization. "
                "Install with: pip install trimesh"
            )

        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Calculate voxel pitch based on resolution
        extents = mesh.extents
        max_extent = extents.max()
        pitch = max_extent / (self.resolution - 1)

        # Voxelize
        try:
            voxelized = mesh.voxelized(pitch=pitch)
            if fill:
                voxelized = voxelized.fill()
            grid = voxelized.matrix.astype(np.int32)
        except Exception as e:
            # Fallback: simple point sampling
            import warnings
            warnings.warn(f"Trimesh voxelization failed: {e}. Using fallback method.")
            grid = self._fallback_voxelize(mesh, pitch)

        return grid

    def _fallback_voxelize(
        self,
        mesh,
        pitch: float,
    ) -> np.ndarray:
        """Fallback voxelization using point containment."""
        import trimesh

        # Calculate grid size
        bounds = mesh.bounds
        grid_size = np.ceil((bounds[1] - bounds[0]) / pitch).astype(int) + 1

        # Limit grid size
        grid_size = np.minimum(grid_size, self.resolution)

        # Create grid of sample points
        x = np.linspace(bounds[0, 0], bounds[1, 0], grid_size[0])
        y = np.linspace(bounds[0, 1], bounds[1, 1], grid_size[1])
        z = np.linspace(bounds[0, 2], bounds[1, 2], grid_size[2])
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

        # Check containment
        try:
            inside = mesh.contains(points)
        except Exception:
            # If contains fails, just use the surface
            inside = np.zeros(len(points), dtype=bool)

        # Reshape to grid
        grid = inside.reshape(grid_size).astype(np.int32)

        return grid

    def voxelize_numpy(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> np.ndarray:
        """
        Simple NumPy-based voxelization (no external dependencies).

        This is a simpler but less accurate voxelization method that
        doesn't require trimesh. It works by sampling points inside
        the mesh's bounding box.

        Parameters
        ----------
        vertices : np.ndarray
            Vertex positions, shape (N, 3).
        faces : np.ndarray
            Triangle indices, shape (M, 3).

        Returns
        -------
        np.ndarray
            3D int32 array with 1 for surface voxels.
        """
        # Get bounds
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        extents = max_coords - min_coords
        max_extent = extents.max()

        if max_extent == 0:
            return np.zeros((1, 1, 1), dtype=np.int32)

        # Calculate grid size proportional to extents
        scale = (self.resolution - 1) / max_extent
        grid_size = np.ceil(extents * scale).astype(int) + 1

        # Transform vertices to grid coordinates
        grid_vertices = (vertices - min_coords) * scale

        # Create empty grid
        grid = np.zeros(tuple(grid_size), dtype=np.int32)

        # Rasterize triangles (surface only)
        for face in faces:
            v0, v1, v2 = grid_vertices[face]
            self._rasterize_triangle(grid, v0, v1, v2)

        return grid

    def _rasterize_triangle(
        self,
        grid: np.ndarray,
        v0: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
    ) -> None:
        """Rasterize a single triangle into the grid."""
        # Simple edge sampling
        n_samples = max(
            int(np.linalg.norm(v1 - v0)) + 1,
            int(np.linalg.norm(v2 - v1)) + 1,
            int(np.linalg.norm(v0 - v2)) + 1,
        )
        n_samples = max(n_samples, 3)

        # Sample points on edges and interior
        for i in range(n_samples + 1):
            t = i / n_samples
            for j in range(n_samples + 1 - i):
                s = j / n_samples
                # Barycentric interpolation
                p = (1 - t - s) * v0 + t * v1 + s * v2
                idx = np.round(p).astype(int)
                # Check bounds
                if all(0 <= idx[k] < grid.shape[k] for k in range(3)):
                    grid[idx[0], idx[1], idx[2]] = 1

    def __repr__(self) -> str:
        return f"Voxelizer(resolution={self.resolution})"
