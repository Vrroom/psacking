"""
Multi-format mesh loading with validation and repair.

This module provides utilities for loading 3D meshes from various file formats,
with optional validation and automatic repair of common mesh issues.

Supported Formats
-----------------
- STL (Stereolithography) - .stl
- Wavefront OBJ - .obj
- Stanford PLY - .ply
- OFF (Object File Format) - .off
- GLTF/GLB - .gltf, .glb
- 3MF (3D Manufacturing Format) - .3mf
- COLLADA - .dae

Examples
--------
>>> from spectral_packer import load_mesh, get_mesh_info
>>> vertices, faces = load_mesh("model.stl")
>>> print(f"Loaded mesh with {len(vertices)} vertices")

>>> info = get_mesh_info("model.stl")
>>> print(f"Watertight: {info['is_watertight']}")
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any

import numpy as np

# Try to import mesh loading libraries
try:
    import trimesh
    _HAS_TRIMESH = True
except ImportError:
    _HAS_TRIMESH = False

try:
    import meshio
    _HAS_MESHIO = True
except ImportError:
    _HAS_MESHIO = False


class MeshLoadError(Exception):
    """Raised when mesh loading fails."""
    pass


class MeshValidationError(Exception):
    """Raised when mesh validation fails and repair is disabled or unsuccessful."""
    pass


# Supported file formats with descriptions
SUPPORTED_FORMATS: Dict[str, str] = {
    ".stl": "STL (Stereolithography)",
    ".obj": "Wavefront OBJ",
    ".ply": "Stanford PLY",
    ".off": "Object File Format",
    ".gltf": "GL Transmission Format",
    ".glb": "GL Transmission Format (Binary)",
    ".3mf": "3D Manufacturing Format",
    ".dae": "COLLADA",
}


def load_mesh(
    path: Union[str, Path],
    validate: bool = True,
    repair: bool = True,
    center: bool = False,
    scale: Optional[float] = None,
    force_watertight: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a 3D mesh from various file formats.

    Parameters
    ----------
    path : str or Path
        Path to the mesh file.
    validate : bool, default True
        Check mesh for watertightness and manifoldness.
    repair : bool, default True
        Attempt to repair non-watertight meshes.
    center : bool, default False
        Center the mesh at origin.
    scale : float, optional
        Scale the mesh to fit in a cube of this size.
    force_watertight : bool, default False
        Raise an error if the mesh is not watertight after repair.

    Returns
    -------
    vertices : np.ndarray
        Vertex positions, shape (N, 3), dtype float32.
    faces : np.ndarray
        Triangle indices, shape (M, 3), dtype int32.

    Raises
    ------
    MeshLoadError
        If the file cannot be loaded.
    MeshValidationError
        If validation fails and repair is disabled or fails.
    FileNotFoundError
        If the file does not exist.
    ImportError
        If no mesh loading library is available.

    Examples
    --------
    >>> vertices, faces = load_mesh("model.stl")
    >>> print(f"Vertices: {vertices.shape}, Faces: {faces.shape}")
    Vertices: (1000, 3), Faces: (2000, 3)

    >>> # Center and scale the mesh
    >>> verts, faces = load_mesh("model.obj", center=True, scale=1.0)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        if _HAS_TRIMESH:
            # trimesh might still handle it
            warnings.warn(
                f"Format '{suffix}' not officially supported, attempting to load anyway"
            )
        else:
            raise MeshLoadError(f"Unsupported format: {suffix}")

    # Try trimesh first (preferred)
    if _HAS_TRIMESH:
        return _load_with_trimesh(
            path, validate, repair, center, scale, force_watertight
        )
    elif _HAS_MESHIO:
        return _load_with_meshio(path, center, scale)
    else:
        raise ImportError(
            "Neither trimesh nor meshio is installed. "
            "Install with: pip install trimesh"
        )


def _load_with_trimesh(
    path: Path,
    validate: bool,
    repair: bool,
    center: bool,
    scale: Optional[float],
    force_watertight: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load mesh using trimesh library."""
    try:
        mesh = trimesh.load(str(path), force="mesh")
    except Exception as e:
        raise MeshLoadError(f"Failed to load mesh '{path}': {e}")

    # Handle Scene objects (multiple meshes)
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise MeshLoadError(f"Scene '{path}' contains no geometry")
        # Combine all meshes into one
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    # Validate mesh
    if validate:
        issues = []
        if not mesh.is_watertight:
            issues.append("not watertight")
        if not mesh.is_winding_consistent:
            issues.append("inconsistent winding")

        if issues:
            if repair:
                # Attempt repairs
                trimesh.repair.fix_normals(mesh)
                trimesh.repair.fix_winding(mesh)
                trimesh.repair.fill_holes(mesh)

                # Re-check after repair
                if force_watertight and not mesh.is_watertight:
                    raise MeshValidationError(
                        f"Mesh '{path}' repair failed. Still not watertight."
                    )
            else:
                if force_watertight:
                    raise MeshValidationError(
                        f"Mesh '{path}' validation failed: {', '.join(issues)}"
                    )
                else:
                    warnings.warn(
                        f"Mesh '{path}' has issues: {', '.join(issues)}. "
                        "Set repair=True to attempt automatic fix."
                    )

    # Apply transformations
    if center:
        mesh.vertices -= mesh.centroid

    if scale is not None:
        current_scale = mesh.extents.max()
        if current_scale > 0:
            mesh.vertices *= scale / current_scale

    return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)


def _load_with_meshio(
    path: Path,
    center: bool,
    scale: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load mesh using meshio (fallback)."""
    try:
        mesh = meshio.read(str(path))
    except Exception as e:
        raise MeshLoadError(f"Failed to load mesh '{path}': {e}")

    vertices = mesh.points.astype(np.float32)

    # Find triangle cells
    faces = None
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            faces = cell_block.data.astype(np.int32)
            break

    if faces is None:
        raise MeshLoadError(f"No triangle faces found in mesh '{path}'")

    # Apply transformations
    if center:
        vertices -= vertices.mean(axis=0)

    if scale is not None:
        current_scale = (vertices.max(axis=0) - vertices.min(axis=0)).max()
        if current_scale > 0:
            vertices *= scale / current_scale

    return vertices, faces


def get_mesh_info(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a mesh file.

    Parameters
    ----------
    path : str or Path
        Path to the mesh file.

    Returns
    -------
    dict
        Dictionary containing:
        - format: File extension
        - num_vertices: Number of vertices
        - num_faces: Number of triangular faces
        - bounding_box: Dict with 'min', 'max', 'extents'
        - is_watertight: Whether mesh is watertight
        - volume: Mesh volume (if watertight)
        - surface_area: Total surface area
        - file_size_bytes: File size in bytes

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ImportError
        If trimesh is not installed.

    Examples
    --------
    >>> info = get_mesh_info("model.stl")
    >>> print(f"Vertices: {info['num_vertices']}")
    >>> print(f"Watertight: {info['is_watertight']}")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    if not _HAS_TRIMESH:
        raise ImportError("trimesh required for mesh info. Install with: pip install trimesh")

    mesh = trimesh.load(str(path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise MeshLoadError(f"Scene '{path}' contains no geometry")
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    info = {
        "format": path.suffix.lower(),
        "num_vertices": len(mesh.vertices),
        "num_faces": len(mesh.faces),
        "bounding_box": {
            "min": mesh.bounds[0].tolist(),
            "max": mesh.bounds[1].tolist(),
            "extents": mesh.extents.tolist(),
        },
        "is_watertight": mesh.is_watertight,
        "is_winding_consistent": mesh.is_winding_consistent,
        "surface_area": float(mesh.area),
        "file_size_bytes": path.stat().st_size,
    }

    # Volume only makes sense for watertight meshes
    if mesh.is_watertight:
        info["volume"] = float(mesh.volume)
    else:
        info["volume"] = None

    return info


def list_supported_formats() -> Dict[str, str]:
    """
    List all supported mesh file formats.

    Returns
    -------
    dict
        Dictionary mapping file extensions to format descriptions.
    """
    return SUPPORTED_FORMATS.copy()
