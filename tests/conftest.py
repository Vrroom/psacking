"""
Pytest fixtures for spectral_packer tests.

This module provides reusable test fixtures including:
- Sample voxel grids (cubes, L-shapes, etc.)
- Empty trays of various sizes
- Sample mesh files (if available)
- GPU availability checks
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile


# Directory containing test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"


# =============================================================================
# Voxel Grid Fixtures
# =============================================================================

@pytest.fixture
def cube_3x3x3() -> np.ndarray:
    """A solid 3x3x3 cube voxel grid."""
    return np.ones((3, 3, 3), dtype=np.int32)


@pytest.fixture
def cube_5x5x5() -> np.ndarray:
    """A solid 5x5x5 cube voxel grid."""
    return np.ones((5, 5, 5), dtype=np.int32)


@pytest.fixture
def cube_with_padding() -> np.ndarray:
    """A 5x5x5 cube centered in a 10x10x10 grid."""
    grid = np.zeros((10, 10, 10), dtype=np.int32)
    grid[2:7, 2:7, 2:7] = 1
    return grid


@pytest.fixture
def l_shape() -> np.ndarray:
    """An L-shaped voxel grid for testing non-cubic shapes."""
    shape = np.zeros((6, 6, 6), dtype=np.int32)
    # Vertical part
    shape[0:4, 0:2, 0:2] = 1
    # Horizontal part
    shape[0:2, 2:4, 0:2] = 1
    return shape


@pytest.fixture
def small_item() -> np.ndarray:
    """A small 2x2x2 cube."""
    return np.ones((2, 2, 2), dtype=np.int32)


@pytest.fixture
def single_voxel() -> np.ndarray:
    """A single voxel (1x1x1)."""
    return np.ones((1, 1, 1), dtype=np.int32)


# =============================================================================
# Tray Fixtures
# =============================================================================

@pytest.fixture
def empty_tray_small() -> np.ndarray:
    """Empty 20x20x20 tray."""
    return np.zeros((20, 20, 20), dtype=np.int32)


@pytest.fixture
def empty_tray_medium() -> np.ndarray:
    """Empty 50x50x50 tray."""
    return np.zeros((50, 50, 50), dtype=np.int32)


@pytest.fixture
def empty_tray_large() -> np.ndarray:
    """Empty 100x100x100 tray."""
    return np.zeros((100, 100, 100), dtype=np.int32)


@pytest.fixture
def partially_filled_tray() -> np.ndarray:
    """A 20x20x20 tray with one corner occupied."""
    tray = np.zeros((20, 20, 20), dtype=np.int32)
    tray[0:5, 0:5, 0:5] = 1
    return tray


# =============================================================================
# File Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_stl_path():
    """Path to a sample STL file if available."""
    # Check for existing test STL files
    possible_paths = [
        FIXTURES_DIR / "cube.stl",
        Path(__file__).parent.parent / "spectral_packing" / "VoxSurf" / "bunny.stl",
    ]
    for path in possible_paths:
        if path.exists():
            return path
    pytest.skip("No sample STL file found")


@pytest.fixture
def sample_stl_content() -> bytes:
    """Binary content of a simple STL cube."""
    # A simple binary STL cube (minimal valid STL)
    # This is a unit cube with 12 triangles
    import struct

    triangles = [
        # Front face
        ((0, 0, 1), ((0, 0, 0), (1, 0, 0), (1, 1, 0))),
        ((0, 0, 1), ((0, 0, 0), (1, 1, 0), (0, 1, 0))),
        # Back face
        ((0, 0, -1), ((0, 0, 1), (1, 1, 1), (1, 0, 1))),
        ((0, 0, -1), ((0, 0, 1), (0, 1, 1), (1, 1, 1))),
        # Top face
        ((0, 1, 0), ((0, 1, 0), (1, 1, 0), (1, 1, 1))),
        ((0, 1, 0), ((0, 1, 0), (1, 1, 1), (0, 1, 1))),
        # Bottom face
        ((0, -1, 0), ((0, 0, 0), (1, 0, 1), (1, 0, 0))),
        ((0, -1, 0), ((0, 0, 0), (0, 0, 1), (1, 0, 1))),
        # Right face
        ((1, 0, 0), ((1, 0, 0), (1, 0, 1), (1, 1, 1))),
        ((1, 0, 0), ((1, 0, 0), (1, 1, 1), (1, 1, 0))),
        # Left face
        ((-1, 0, 0), ((0, 0, 0), (0, 1, 1), (0, 0, 1))),
        ((-1, 0, 0), ((0, 0, 0), (0, 1, 0), (0, 1, 1))),
    ]

    # Header (80 bytes) + num triangles (4 bytes)
    data = b'\x00' * 80 + struct.pack('<I', len(triangles))

    for normal, (v1, v2, v3) in triangles:
        # Normal vector (3 floats)
        data += struct.pack('<3f', *normal)
        # Three vertices (9 floats)
        data += struct.pack('<3f', *v1)
        data += struct.pack('<3f', *v2)
        data += struct.pack('<3f', *v3)
        # Attribute byte count (2 bytes)
        data += struct.pack('<H', 0)

    return data


@pytest.fixture
def temp_stl_file(temp_dir, sample_stl_content) -> Path:
    """Create a temporary STL file for testing."""
    stl_path = temp_dir / "test_cube.stl"
    stl_path.write_bytes(sample_stl_content)
    return stl_path


# =============================================================================
# GPU/CUDA Fixtures
# =============================================================================

@pytest.fixture
def requires_cuda():
    """Skip test if CUDA core module is not available."""
    # CUDA core is now a required dependency
    pass


@pytest.fixture
def cuda_available() -> bool:
    """Check if CUDA core module is available."""
    # CUDA core is now a required dependency
    return True


# =============================================================================
# BinPacker Fixtures
# =============================================================================

@pytest.fixture
def packer_small():
    """BinPacker with a small 20x20x20 tray."""
    from spectral_packer import BinPacker
    return BinPacker(tray_size=(20, 20, 20), voxel_resolution=32)


@pytest.fixture
def packer_medium():
    """BinPacker with a medium 50x50x50 tray."""
    from spectral_packer import BinPacker
    return BinPacker(tray_size=(50, 50, 50), voxel_resolution=64)


@pytest.fixture
def packer_large():
    """BinPacker with a large 100x100x100 tray."""
    from spectral_packer import BinPacker
    return BinPacker(tray_size=(100, 100, 100), voxel_resolution=128)


# =============================================================================
# Voxelizer Fixtures
# =============================================================================

@pytest.fixture
def voxelizer_low_res():
    """Voxelizer with low resolution (32)."""
    from spectral_packer import Voxelizer
    return Voxelizer(resolution=32)


@pytest.fixture
def voxelizer_medium_res():
    """Voxelizer with medium resolution (64)."""
    from spectral_packer import Voxelizer
    return Voxelizer(resolution=64)


@pytest.fixture
def voxelizer_high_res():
    """Voxelizer with high resolution (128)."""
    from spectral_packer import Voxelizer
    return Voxelizer(resolution=128)
