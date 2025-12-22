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

# Import core C++ bindings
try:
    from ._core import (
        VOXEL_RESOLUTION,
        HEIGHT_PENALTY,
        fft_search_placement,
        place_in_tray,
        voxelize_stl,
        dft_conv3,
        dft_corr3,
        calculate_distance,
        collision_grid,
        make_tight,
        get_bounds,
        save_vox,
    )
    _CORE_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(
        f"Failed to import C++ core module: {e}. "
        "Some functionality will be unavailable. "
        "Make sure the package was built correctly with CUDA support."
    )
    _CORE_AVAILABLE = False
    VOXEL_RESOLUTION = 128
    HEIGHT_PENALTY = 1e8

# Import Python modules
from .mesh_io import (
    load_mesh,
    get_mesh_info,
    MeshLoadError,
    MeshValidationError,
    SUPPORTED_FORMATS,
)
from .packer import BinPacker, PackingResult
from .voxelizer import Voxelizer

__all__ = [
    # Version info
    "__version__",
    # High-level API
    "BinPacker",
    "PackingResult",
    "Voxelizer",
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
    "place_in_tray",
    "voxelize_stl",
    "dft_conv3",
    "dft_corr3",
    "calculate_distance",
    "collision_grid",
    "make_tight",
    "get_bounds",
    "save_vox",
]


def is_cuda_available() -> bool:
    """Check if CUDA core module is available.

    Returns
    -------
    bool
        True if the CUDA-accelerated core module loaded successfully.
    """
    return _CORE_AVAILABLE
