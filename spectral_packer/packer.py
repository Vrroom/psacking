"""
High-level bin packing interface.

This module provides the main BinPacker class for 3D bin packing operations,
along with the PackingResult dataclass for results.

Examples
--------
>>> from spectral_packer import BinPacker
>>> packer = BinPacker(tray_size=(100, 100, 100))
>>> result = packer.pack_files(["item1.stl", "item2.obj"])
>>> print(f"Packed {result.num_placed}/{result.num_placed + result.num_failed} items")
>>> print(f"Density: {result.density:.1%}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .mesh_io import load_mesh
from .voxelizer import Voxelizer


@dataclass
class PlacementInfo:
    """Information about a single item placement.

    Attributes
    ----------
    item_index : int
        Original index of the item in the input list.
    position : tuple or None
        (x, y, z) placement position, or None if placement failed.
    score : float or None
        Placement score (lower is better), or None if placement failed.
    success : bool
        Whether the item was successfully placed.
    volume : int
        Number of voxels in the item.
    """

    item_index: int
    position: Optional[Tuple[int, int, int]]
    score: Optional[float]
    success: bool
    volume: int = 0


@dataclass
class PackingResult:
    """Results from a packing operation.

    Attributes
    ----------
    tray : np.ndarray
        Final voxel grid with all placed items. Each item's voxels are
        marked with a unique ID (1, 2, 3, ...).
    placements : list of PlacementInfo
        List of placement information for each item.
    num_placed : int
        Number of items successfully placed.
    num_failed : int
        Number of items that could not be placed.
    density : float
        Packing density (occupied volume / bounding box volume).
    total_volume : int
        Total number of occupied voxels.
    bounding_box : tuple
        ((min_x, min_y, min_z), (max_x, max_y, max_z)) of occupied region.
    """

    tray: np.ndarray
    placements: List[PlacementInfo] = field(default_factory=list)
    num_placed: int = 0
    num_failed: int = 0
    density: float = 0.0
    total_volume: int = 0
    bounding_box: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None

    def get_item_mask(self, item_id: int) -> np.ndarray:
        """Get a binary mask for a specific item.

        Parameters
        ----------
        item_id : int
            The ID of the item (1-indexed).

        Returns
        -------
        np.ndarray
            Boolean mask where True indicates voxels belonging to the item.
        """
        return self.tray == item_id

    def save_vox(self, path: Union[str, Path]) -> None:
        """Save result to MagicaVoxel .vox format.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        from . import save_vox
        save_vox(self.tray, str(path))

    def summary(self) -> str:
        """Get a human-readable summary of the packing result.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [
            f"Packing Result Summary",
            f"=" * 40,
            f"Items placed:    {self.num_placed}",
            f"Items failed:    {self.num_failed}",
            f"Success rate:    {self.num_placed / max(1, self.num_placed + self.num_failed):.1%}",
            f"Packing density: {self.density:.1%}",
            f"Total volume:    {self.total_volume} voxels",
            f"Tray size:       {self.tray.shape}",
        ]
        if self.bounding_box:
            bbox_min, bbox_max = self.bounding_box
            lines.append(f"Bounding box:    {bbox_min} to {bbox_max}")
        return "\n".join(lines)


class BinPacker:
    """
    GPU-accelerated 3D bin packing using spectral (FFT) collision detection.

    This class provides the main interface for packing 3D items into a
    rectangular tray. It uses FFT-based algorithms for efficient collision
    detection and optimal placement finding.

    Parameters
    ----------
    tray_size : tuple of int
        Size of the packing tray as (x, y, z).
    voxel_resolution : int, default 128
        Resolution for mesh voxelization.
    height_penalty : float, default 1e8
        Penalty factor for height in placement scoring.
        Higher values encourage items to be placed lower.

    Attributes
    ----------
    tray_size : tuple
        The tray dimensions.
    voxel_resolution : int
        Voxelization resolution.
    height_penalty : float
        Height penalty factor.

    Examples
    --------
    Basic usage with file paths:

    >>> packer = BinPacker(tray_size=(100, 100, 100))
    >>> result = packer.pack_files(["item1.stl", "item2.obj"])
    >>> print(f"Density: {result.density:.1%}")

    With pre-voxelized items:

    >>> voxelizer = Voxelizer(resolution=64)
    >>> items = [voxelizer.voxelize_file(f) for f in files]
    >>> result = packer.pack_voxels(items)

    Accessing placement details:

    >>> for p in result.placements:
    ...     if p.success:
    ...         print(f"Item {p.item_index} at {p.position}, score={p.score:.2f}")
    """

    def __init__(
        self,
        tray_size: Tuple[int, int, int],
        voxel_resolution: int = 128,
        height_penalty: float = 1e8,
    ):
        if len(tray_size) != 3:
            raise ValueError(f"tray_size must be a 3-tuple, got {len(tray_size)} elements")
        if any(s <= 0 for s in tray_size):
            raise ValueError(f"tray_size dimensions must be positive, got {tray_size}")
        if voxel_resolution <= 0:
            raise ValueError(f"voxel_resolution must be positive, got {voxel_resolution}")

        self.tray_size = tuple(tray_size)
        self.voxel_resolution = voxel_resolution
        self.height_penalty = height_penalty
        self._voxelizer = Voxelizer(resolution=voxel_resolution)

    def pack_files(
        self,
        paths: Sequence[Union[str, Path]],
        sort_by_volume: bool = True,
        validate_meshes: bool = True,
        repair_meshes: bool = True,
    ) -> PackingResult:
        """
        Pack meshes from file paths.

        Parameters
        ----------
        paths : sequence of str or Path
            Paths to mesh files (STL, OBJ, PLY, etc.).
        sort_by_volume : bool, default True
            Sort items by volume (largest first) before packing.
        validate_meshes : bool, default True
            Validate meshes during loading.
        repair_meshes : bool, default True
            Attempt to repair invalid meshes.

        Returns
        -------
        PackingResult
            Packing results including final tray and statistics.

        Raises
        ------
        FileNotFoundError
            If any mesh file does not exist.
        MeshLoadError
            If any mesh fails to load.
        """
        voxels = []
        for path in paths:
            voxel = self._voxelizer.voxelize_file(
                path,
                validate=validate_meshes,
                repair=repair_meshes,
            )
            voxels.append(voxel)

        return self.pack_voxels(voxels, sort_by_volume=sort_by_volume)

    def pack_voxels(
        self,
        items: Sequence[np.ndarray],
        sort_by_volume: bool = True,
    ) -> PackingResult:
        """
        Pack pre-voxelized items.

        Parameters
        ----------
        items : sequence of np.ndarray
            List of 3D int/bool arrays representing voxelized items.
            Non-zero values indicate occupied voxels.
        sort_by_volume : bool, default True
            Sort items by volume (largest first) before packing.

        Returns
        -------
        PackingResult
            Packing results including final tray and statistics.

        Raises
        ------
        ValueError
            If items list is empty or contains invalid arrays.
        RuntimeError
            If the C++ core module is not available.
        """
        from . import _CORE_AVAILABLE
        if not _CORE_AVAILABLE:
            raise RuntimeError(
                "C++ core module not available. "
                "Make sure the package was built with CUDA support."
            )

        from . import fft_search_placement, place_in_tray

        if len(items) == 0:
            raise ValueError("items list cannot be empty")

        # Validate and convert items
        processed_items = []
        volumes = []
        for i, item in enumerate(items):
            if not isinstance(item, np.ndarray):
                raise ValueError(f"Item {i} is not a numpy array")
            if item.ndim != 3:
                raise ValueError(f"Item {i} must be 3D, got {item.ndim}D")
            item_int = item.astype(np.int32)
            processed_items.append(item_int)
            volumes.append(int(np.sum(item_int > 0)))

        # Sort by volume if requested
        if sort_by_volume:
            sorted_indices = np.argsort(volumes)[::-1]  # Largest first
            processed_items = [processed_items[i] for i in sorted_indices]
            volumes = [volumes[i] for i in sorted_indices]
            original_indices = list(sorted_indices)
        else:
            original_indices = list(range(len(items)))

        # Initialize tray
        tray = np.zeros(self.tray_size, dtype=np.int32)

        placements = []
        num_placed = 0

        for idx, (item, orig_idx, volume) in enumerate(
            zip(processed_items, original_indices, volumes)
        ):
            # Find placement
            position, found, score = fft_search_placement(item, tray)

            if found:
                # Place item (item_id is 1-indexed)
                item_id = num_placed + 1
                tray = place_in_tray(item, tray, position, item_id)
                num_placed += 1
                placements.append(PlacementInfo(
                    item_index=orig_idx,
                    position=position,
                    score=score,
                    success=True,
                    volume=volume,
                ))
            else:
                placements.append(PlacementInfo(
                    item_index=orig_idx,
                    position=None,
                    score=None,
                    success=False,
                    volume=volume,
                ))

        # Sort placements by original index for consistent ordering
        placements.sort(key=lambda p: p.item_index)

        # Calculate statistics
        density, total_volume, bbox = self._calculate_stats(tray)

        return PackingResult(
            tray=tray,
            placements=placements,
            num_placed=num_placed,
            num_failed=len(items) - num_placed,
            density=density,
            total_volume=total_volume,
            bounding_box=bbox,
        )

    def pack_single(
        self,
        item: np.ndarray,
        tray: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[Tuple[int, int, int]], bool, float]:
        """
        Find optimal placement for a single item.

        Parameters
        ----------
        item : np.ndarray
            3D int/bool array representing the item.
        tray : np.ndarray, optional
            Current tray state. If None, uses an empty tray.

        Returns
        -------
        position : tuple or None
            (x, y, z) placement position, or None if no valid placement.
        found : bool
            Whether a valid placement was found.
        score : float
            Placement score (lower is better), or 0.0 if not found.
        """
        from . import _CORE_AVAILABLE
        if not _CORE_AVAILABLE:
            raise RuntimeError("C++ core module not available")

        from . import fft_search_placement

        if tray is None:
            tray = np.zeros(self.tray_size, dtype=np.int32)

        item_int = item.astype(np.int32)
        position, found, score = fft_search_placement(item_int, tray)

        if found:
            return tuple(position), True, score
        return None, False, 0.0

    def _calculate_stats(
        self, tray: np.ndarray
    ) -> Tuple[float, int, Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]]:
        """Calculate packing statistics."""
        occupied = np.sum(tray > 0)
        total_volume = int(occupied)

        if occupied == 0:
            return 0.0, 0, None

        # Find bounding box of occupied voxels
        occupied_indices = np.argwhere(tray > 0)
        bbox_min = tuple(occupied_indices.min(axis=0).tolist())
        bbox_max = tuple(occupied_indices.max(axis=0).tolist())
        bbox_dims = tuple(mx - mn + 1 for mn, mx in zip(bbox_min, bbox_max))
        bbox_volume = np.prod(bbox_dims)

        density = float(occupied / bbox_volume) if bbox_volume > 0 else 0.0

        return density, total_volume, (bbox_min, bbox_max)

    def __repr__(self) -> str:
        return (
            f"BinPacker(tray_size={self.tray_size}, "
            f"voxel_resolution={self.voxel_resolution})"
        )
