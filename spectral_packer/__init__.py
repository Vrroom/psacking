"""
Spectral 3D Bin Packing
=======================

GPU-accelerated 3D bin packing using FFT-based collision detection.

This package provides efficient 3D bin packing algorithms that leverage
spectral methods (Fast Fourier Transform) for rapid collision detection
and optimal placement finding.

Quick Start
-----------
>>> from spectral_packer import BinPacker
>>> packer = BinPacker(tray_size=(100, 100, 100))
>>> result = packer.pack_files(["item1.stl", "item2.obj"])
>>> print(f"Packed {result.num_placed} items with {result.density:.1%} density")

Features
--------
- GPU-accelerated FFT operations via CUDA
- Multiple mesh format support (STL, OBJ, PLY, etc.)
- High-level Python API with NumPy integration
- Automatic mesh validation and repair

Modules
-------
packer
    High-level BinPacker class for packing operations
mesh_io
    Multi-format mesh loading with validation
voxelizer
    Mesh-to-voxel conversion utilities

Low-Level Functions
-------------------
The _core module provides direct access to C++ functions:

- fft_search_placement: Find optimal placement using FFT
- place_in_tray: Place item at given position
- voxelize_stl: Convert STL to voxel grid
- dft_conv3: 3D FFT convolution
- dft_corr3: 3D FFT cross-correlation
- calculate_distance: Compute distance field
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Spectral Packing Authors"

# Import core C++ bindings (required)
from ._core import (
    VOXEL_RESOLUTION,
    HEIGHT_PENALTY,
    fft_search_placement,
    fft_search_placement_with_cache,
    fft_search_batch,
    place_in_tray,
    voxelize_stl,
    dft_conv3,
    dft_corr3,
    calculate_distance,
    collision_grid,
    make_tight,
    get_bounds,
    save_vox,
    # GPU-resident tray functions
    gpu_tray_init,
    gpu_tray_search,
    gpu_tray_cleanup,
    # GPU interlocking-free positions (Algorithm 3)
    interlocking_free_positions,
)

# Import Python modules
from .mesh_io import (
    load_mesh,
    get_mesh_info,
    MeshLoadError,
    MeshValidationError,
    SUPPORTED_FORMATS,
)
from .packer import BinPacker, PackingResult, PlacementInfo, MeshPlacementInfo
from .voxelizer import Voxelizer, VoxelizationInfo
from .rotations import (
    get_orientations,
    get_24_orientations,
    rotate_90_x,
    rotate_90_y,
    rotate_90_z,
    get_rotation_matrix_3x3,
    get_rotation_matrix_4x4,
    ROTATION_MATRICES_3x3,
)
from .blender_export import (
    export_to_blend,
    is_blender_available,
    compute_mesh_transform,
    BlenderExportError,
    NoMeshMetadataError,
    UnsupportedFormatError,
)

__all__ = [
    # Version info
    "__version__",
    # High-level API
    "BinPacker",
    "PackingResult",
    "PlacementInfo",
    "MeshPlacementInfo",
    "Voxelizer",
    "VoxelizationInfo",
    # Blender export
    "export_to_blend",
    "is_blender_available",
    "compute_mesh_transform",
    "BlenderExportError",
    "NoMeshMetadataError",
    "UnsupportedFormatError",
    # Rotation utilities
    "get_orientations",
    "get_24_orientations",
    "rotate_90_x",
    "rotate_90_y",
    "rotate_90_z",
    "get_rotation_matrix_3x3",
    "get_rotation_matrix_4x4",
    "ROTATION_MATRICES_3x3",
    # Mesh I/O
    "load_mesh",
    "get_mesh_info",
    "MeshLoadError",
    "MeshValidationError",
    "SUPPORTED_FORMATS",
    # Constants
    "VOXEL_RESOLUTION",
    "HEIGHT_PENALTY",
    # Core functions (if available)
    "fft_search_placement",
    "fft_search_placement_with_cache",
    "fft_search_batch",
    "place_in_tray",
    "voxelize_stl",
    "dft_conv3",
    "dft_corr3",
    "calculate_distance",
    "collision_grid",
    "make_tight",
    "get_bounds",
    "save_vox",
    # GPU-resident tray functions
    "gpu_tray_init",
    "gpu_tray_search",
    "gpu_tray_cleanup",
    # GPU interlocking-free positions (Algorithm 3)
    "interlocking_free_positions",
]


def is_cuda_available() -> bool:
    """Check if CUDA core module is available.

    Returns
    -------
    bool
        True if the CUDA-accelerated core module loaded successfully.
        Always True since the module import is required.
    """
    return True
