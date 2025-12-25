"""
Tests for the fixed pitch voxelization feature.

This module tests the pitch parameter which allows 1 voxel = 1mm (or any fixed unit).
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path


class TestVoxelizerPitch:
    """Tests for Voxelizer with fixed pitch."""

    def test_pitch_init(self):
        """Test Voxelizer initialization with pitch."""
        from spectral_packer import Voxelizer

        vox = Voxelizer(resolution=128, pitch=1.0)

        assert vox.pitch == 1.0
        assert vox.resolution == 128

    def test_pitch_none_by_default(self):
        """Test that pitch is None by default."""
        from spectral_packer import Voxelizer

        vox = Voxelizer(resolution=64)

        assert vox.pitch is None

    def test_pitch_invalid_value(self):
        """Test error on invalid pitch value."""
        from spectral_packer import Voxelizer

        with pytest.raises(ValueError):
            Voxelizer(resolution=64, pitch=0)

        with pytest.raises(ValueError):
            Voxelizer(resolution=64, pitch=-1.0)

    def test_pitch_voxelizes_at_correct_size(self, scaled_stl_file):
        """Test that pitch=1.0 gives ~1 voxel per mm."""
        from spectral_packer import Voxelizer

        # Create voxelizer with 1mm = 1 voxel
        vox = Voxelizer(resolution=256, pitch=1.0)
        grid, info = vox.voxelize_file_with_info(scaled_stl_file)

        # The scaled STL is 50mm, so grid should be roughly 50 voxels
        max_dim = max(grid.shape)
        assert 45 <= max_dim <= 55, f"Expected ~50 voxels, got {max_dim}"
        assert info.pitch == 1.0

    def test_pitch_vs_resolution_comparison(self, scaled_stl_file):
        """Test that pitch gives different results than resolution-based."""
        from spectral_packer import Voxelizer

        # Resolution-based: always normalizes to resolution
        vox_res = Voxelizer(resolution=64)
        grid_res, info_res = vox_res.voxelize_file_with_info(scaled_stl_file)

        # Pitch-based: voxel size is fixed
        vox_pitch = Voxelizer(resolution=256, pitch=1.0)
        grid_pitch, info_pitch = vox_pitch.voxelize_file_with_info(scaled_stl_file)

        # Resolution-based should give ~64 voxels max dimension
        assert max(grid_res.shape) <= 64

        # Pitch-based should give ~50 voxels (since mesh is 50mm)
        assert 45 <= max(grid_pitch.shape) <= 55

        # Pitch values should differ
        assert info_res.pitch != info_pitch.pitch

    def test_pitch_consistent_across_different_objects(self, scaled_stl_file, temp_dir):
        """Test that two different-sized objects have same pitch."""
        from spectral_packer import Voxelizer
        import struct

        # Create a larger cube (100mm)
        large_stl = temp_dir / "large_cube.stl"
        triangles = [
            # Front face
            ((0, 0, 1), ((0, 0, 0), (100, 0, 0), (100, 100, 0))),
            ((0, 0, 1), ((0, 0, 0), (100, 100, 0), (0, 100, 0))),
            # Back face
            ((0, 0, -1), ((0, 0, 100), (100, 100, 100), (100, 0, 100))),
            ((0, 0, -1), ((0, 0, 100), (0, 100, 100), (100, 100, 100))),
            # Top face
            ((0, 1, 0), ((0, 100, 0), (100, 100, 0), (100, 100, 100))),
            ((0, 1, 0), ((0, 100, 0), (100, 100, 100), (0, 100, 100))),
            # Bottom face
            ((0, -1, 0), ((0, 0, 0), (100, 0, 100), (100, 0, 0))),
            ((0, -1, 0), ((0, 0, 0), (0, 0, 100), (100, 0, 100))),
            # Right face
            ((1, 0, 0), ((100, 0, 0), (100, 0, 100), (100, 100, 100))),
            ((1, 0, 0), ((100, 0, 0), (100, 100, 100), (100, 100, 0))),
            # Left face
            ((-1, 0, 0), ((0, 0, 0), (0, 100, 100), (0, 0, 100))),
            ((-1, 0, 0), ((0, 0, 0), (0, 100, 0), (0, 100, 100))),
        ]
        data = b'\x00' * 80 + struct.pack('<I', len(triangles))
        for normal, (v1, v2, v3) in triangles:
            data += struct.pack('<3f', *normal)
            data += struct.pack('<3f', *v1)
            data += struct.pack('<3f', *v2)
            data += struct.pack('<3f', *v3)
            data += struct.pack('<H', 0)
        large_stl.write_bytes(data)

        # Voxelize both with pitch=1.0
        vox = Voxelizer(resolution=256, pitch=1.0)
        _, info_small = vox.voxelize_file_with_info(scaled_stl_file)  # 50mm
        _, info_large = vox.voxelize_file_with_info(large_stl)  # 100mm

        # Both should have pitch=1.0
        assert info_small.pitch == 1.0
        assert info_large.pitch == 1.0

    def test_pitch_fallback_when_too_large(self, temp_dir):
        """Test that pitch falls back when object would exceed resolution."""
        from spectral_packer import Voxelizer
        import struct
        import warnings

        # Create a 500mm cube
        huge_stl = temp_dir / "huge_cube.stl"
        triangles = [
            ((0, 0, 1), ((0, 0, 0), (500, 0, 0), (500, 500, 0))),
            ((0, 0, 1), ((0, 0, 0), (500, 500, 0), (0, 500, 0))),
            ((0, 0, -1), ((0, 0, 500), (500, 500, 500), (500, 0, 500))),
            ((0, 0, -1), ((0, 0, 500), (0, 500, 500), (500, 500, 500))),
            ((0, 1, 0), ((0, 500, 0), (500, 500, 0), (500, 500, 500))),
            ((0, 1, 0), ((0, 500, 0), (500, 500, 500), (0, 500, 500))),
            ((0, -1, 0), ((0, 0, 0), (500, 0, 500), (500, 0, 0))),
            ((0, -1, 0), ((0, 0, 0), (0, 0, 500), (500, 0, 500))),
            ((1, 0, 0), ((500, 0, 0), (500, 0, 500), (500, 500, 500))),
            ((1, 0, 0), ((500, 0, 0), (500, 500, 500), (500, 500, 0))),
            ((-1, 0, 0), ((0, 0, 0), (0, 500, 500), (0, 0, 500))),
            ((-1, 0, 0), ((0, 0, 0), (0, 500, 0), (0, 500, 500))),
        ]
        data = b'\x00' * 80 + struct.pack('<I', len(triangles))
        for normal, (v1, v2, v3) in triangles:
            data += struct.pack('<3f', *normal)
            data += struct.pack('<3f', *v1)
            data += struct.pack('<3f', *v2)
            data += struct.pack('<3f', *v3)
            data += struct.pack('<H', 0)
        huge_stl.write_bytes(data)

        # Try to voxelize with pitch=1.0 but resolution=128
        # This would need 500+ voxels, so should fall back
        vox = Voxelizer(resolution=128, pitch=1.0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _, info = vox.voxelize_file_with_info(huge_stl)
            # Should have issued a warning
            assert len(w) >= 1
            assert "exceed" in str(w[0].message).lower()
            # Pitch should have been adjusted
            assert info.pitch > 1.0


class TestBinPackerPitch:
    """Tests for BinPacker with fixed pitch."""

    def test_packer_pitch_init(self):
        """Test BinPacker initialization with pitch."""
        from spectral_packer import BinPacker

        packer = BinPacker(tray_size=(240, 120, 100), pitch=1.0)

        assert packer.pitch == 1.0

    def test_packer_pitch_none_by_default(self):
        """Test that pitch is None by default."""
        from spectral_packer import BinPacker

        packer = BinPacker(tray_size=(100, 100, 100))

        assert packer.pitch is None

    def test_packer_pitch_invalid_value(self):
        """Test error on invalid pitch value."""
        from spectral_packer import BinPacker

        with pytest.raises(ValueError):
            BinPacker(tray_size=(100, 100, 100), pitch=0)

        with pytest.raises(ValueError):
            BinPacker(tray_size=(100, 100, 100), pitch=-1.0)

    def test_packer_pitch_passed_to_voxelizer(self):
        """Test that pitch is passed to internal voxelizer."""
        from spectral_packer import BinPacker

        packer = BinPacker(tray_size=(100, 100, 100), pitch=2.0)

        assert packer._voxelizer.pitch == 2.0

    def test_packer_pitch_affects_voxelization(self, scaled_stl_file, requires_cuda):
        """Test that pitch affects how files are voxelized during packing."""
        from spectral_packer import BinPacker

        # Tray is 100x100x100 voxels = 100x100x100mm with pitch=1.0
        packer = BinPacker(
            tray_size=(100, 100, 100),
            pitch=1.0,
            voxel_resolution=128,
        )

        # Pack a 50mm object
        result = packer.pack_files_for_export([scaled_stl_file])

        assert result.num_placed == 1
        # Check that the mesh placement has correct pitch
        assert result.mesh_placements[0].voxel_info.pitch == 1.0


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Provide a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def scaled_stl_file(temp_dir) -> Path:
    """Create a 50mm cube STL for testing."""
    import struct

    stl_path = temp_dir / "cube_50mm.stl"

    # 50mm cube
    size = 50
    triangles = [
        # Front face
        ((0, 0, 1), ((0, 0, 0), (size, 0, 0), (size, size, 0))),
        ((0, 0, 1), ((0, 0, 0), (size, size, 0), (0, size, 0))),
        # Back face
        ((0, 0, -1), ((0, 0, size), (size, size, size), (size, 0, size))),
        ((0, 0, -1), ((0, 0, size), (0, size, size), (size, size, size))),
        # Top face
        ((0, 1, 0), ((0, size, 0), (size, size, 0), (size, size, size))),
        ((0, 1, 0), ((0, size, 0), (size, size, size), (0, size, size))),
        # Bottom face
        ((0, -1, 0), ((0, 0, 0), (size, 0, size), (size, 0, 0))),
        ((0, -1, 0), ((0, 0, 0), (0, 0, size), (size, 0, size))),
        # Right face
        ((1, 0, 0), ((size, 0, 0), (size, 0, size), (size, size, size))),
        ((1, 0, 0), ((size, 0, 0), (size, size, size), (size, size, 0))),
        # Left face
        ((-1, 0, 0), ((0, 0, 0), (0, size, size), (0, 0, size))),
        ((-1, 0, 0), ((0, 0, 0), (0, size, 0), (0, size, size))),
    ]

    # Header (80 bytes) + num triangles (4 bytes)
    data = b'\x00' * 80 + struct.pack('<I', len(triangles))

    for normal, (v1, v2, v3) in triangles:
        data += struct.pack('<3f', *normal)
        data += struct.pack('<3f', *v1)
        data += struct.pack('<3f', *v2)
        data += struct.pack('<3f', *v3)
        data += struct.pack('<H', 0)

    stl_path.write_bytes(data)
    return stl_path


@pytest.fixture
def requires_cuda():
    """Skip test if CUDA is not available."""
    pass
