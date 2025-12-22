"""
Tests for mesh loading and validation.
"""

import pytest
import numpy as np
from pathlib import Path


class TestLoadMesh:
    """Tests for load_mesh function."""

    def test_load_stl(self, temp_stl_file):
        """Test loading an STL file."""
        pytest.importorskip("trimesh")
        from spectral_packer import load_mesh

        vertices, faces = load_mesh(temp_stl_file)

        assert isinstance(vertices, np.ndarray)
        assert isinstance(faces, np.ndarray)
        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        assert faces.ndim == 2
        assert faces.shape[1] == 3

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        from spectral_packer import load_mesh

        with pytest.raises(FileNotFoundError):
            load_mesh("/nonexistent/path/model.stl")

    def test_center_option(self, temp_stl_file):
        """Test centering the mesh."""
        pytest.importorskip("trimesh")
        from spectral_packer import load_mesh

        vertices, _ = load_mesh(temp_stl_file, center=True)

        # Center should be approximately at origin
        centroid = vertices.mean(axis=0)
        assert np.allclose(centroid, 0, atol=1e-5)

    def test_scale_option(self, temp_stl_file):
        """Test scaling the mesh."""
        pytest.importorskip("trimesh")
        from spectral_packer import load_mesh

        vertices, _ = load_mesh(temp_stl_file, scale=1.0)

        # Max extent should be approximately 1.0
        extents = vertices.max(axis=0) - vertices.min(axis=0)
        assert np.isclose(extents.max(), 1.0, atol=1e-5)

    def test_vertex_dtype(self, temp_stl_file):
        """Test that vertices are float32."""
        pytest.importorskip("trimesh")
        from spectral_packer import load_mesh

        vertices, _ = load_mesh(temp_stl_file)

        assert vertices.dtype == np.float32

    def test_face_dtype(self, temp_stl_file):
        """Test that faces are int32."""
        pytest.importorskip("trimesh")
        from spectral_packer import load_mesh

        _, faces = load_mesh(temp_stl_file)

        assert faces.dtype == np.int32


class TestGetMeshInfo:
    """Tests for get_mesh_info function."""

    def test_info_keys(self, temp_stl_file):
        """Test that info contains expected keys."""
        pytest.importorskip("trimesh")
        from spectral_packer import get_mesh_info

        info = get_mesh_info(temp_stl_file)

        expected_keys = [
            "format",
            "num_vertices",
            "num_faces",
            "bounding_box",
            "is_watertight",
            "surface_area",
            "file_size_bytes",
        ]
        for key in expected_keys:
            assert key in info

    def test_bounding_box_structure(self, temp_stl_file):
        """Test bounding box has correct structure."""
        pytest.importorskip("trimesh")
        from spectral_packer import get_mesh_info

        info = get_mesh_info(temp_stl_file)

        bbox = info["bounding_box"]
        assert "min" in bbox
        assert "max" in bbox
        assert "extents" in bbox
        assert len(bbox["min"]) == 3
        assert len(bbox["max"]) == 3
        assert len(bbox["extents"]) == 3

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        from spectral_packer import get_mesh_info

        with pytest.raises(FileNotFoundError):
            get_mesh_info("/nonexistent/path/model.stl")


class TestMeshValidation:
    """Tests for mesh validation and repair."""

    def test_validate_flag(self, temp_stl_file):
        """Test that validation can be disabled."""
        pytest.importorskip("trimesh")
        from spectral_packer import load_mesh

        # Should not raise even with validation disabled
        vertices, faces = load_mesh(temp_stl_file, validate=False)
        assert vertices is not None


class TestSupportedFormats:
    """Tests for format support."""

    def test_supported_formats_exists(self):
        """Test that SUPPORTED_FORMATS is available."""
        from spectral_packer import SUPPORTED_FORMATS

        assert isinstance(SUPPORTED_FORMATS, dict)
        assert len(SUPPORTED_FORMATS) > 0

    def test_stl_supported(self):
        """Test that STL is a supported format."""
        from spectral_packer import SUPPORTED_FORMATS

        assert ".stl" in SUPPORTED_FORMATS

    def test_obj_supported(self):
        """Test that OBJ is a supported format."""
        from spectral_packer import SUPPORTED_FORMATS

        assert ".obj" in SUPPORTED_FORMATS

    def test_ply_supported(self):
        """Test that PLY is a supported format."""
        from spectral_packer import SUPPORTED_FORMATS

        assert ".ply" in SUPPORTED_FORMATS


class TestMeshExceptions:
    """Tests for mesh-related exceptions."""

    def test_mesh_load_error_importable(self):
        """Test that MeshLoadError is importable."""
        from spectral_packer import MeshLoadError

        assert issubclass(MeshLoadError, Exception)

    def test_mesh_validation_error_importable(self):
        """Test that MeshValidationError is importable."""
        from spectral_packer import MeshValidationError

        assert issubclass(MeshValidationError, Exception)
