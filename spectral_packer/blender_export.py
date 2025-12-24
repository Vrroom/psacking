"""
Blender .blend file export for packed objects.

This module provides functions to export packing results to Blender's
native .blend format using the original mesh files with correct transforms.

Requires the bpy module (Blender Python API). This can be used either:
- From within Blender (File > Run Script)
- From command line: blender --python script.py
- With bpy installed as a standalone module

Examples
--------
>>> from spectral_packer import BinPacker, export_to_blend, is_blender_available
>>> packer = BinPacker(tray_size=(100, 100, 100))
>>> result = packer.pack_files_for_export(["part1.stl", "part2.obj"])
>>> if is_blender_available():
...     export_to_blend(result, "packed.blend")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np

from .rotations import get_rotation_matrix_3x3

if TYPE_CHECKING:
    from .packer import MeshPlacementInfo, PackingResult


# Try to import bpy once at module load
try:
    import bpy as _bpy
    _BPY_AVAILABLE = True
except ImportError:
    _bpy = None
    _BPY_AVAILABLE = False


class BlenderExportError(Exception):
    """Base exception for Blender export errors."""
    pass


class NoMeshMetadataError(BlenderExportError):
    """Raised when PackingResult lacks mesh placement metadata."""
    pass


class UnsupportedFormatError(BlenderExportError):
    """Raised when mesh format is not supported by Blender."""
    pass


def is_blender_available() -> bool:
    """Check if the bpy (Blender Python) module is available.

    Returns
    -------
    bool
        True if bpy can be imported, False otherwise.
    """
    return _BPY_AVAILABLE


def _get_bpy():
    """Get the bpy module, raising ImportError with helpful message if unavailable."""
    if not _BPY_AVAILABLE:
        raise ImportError(
            "bpy module not available. To use Blender export:\n"
            "  1. Run this script from within Blender, or\n"
            "  2. Install bpy as a standalone module (pip install bpy)"
        )
    return _bpy


def compute_mesh_transform(
    placement: "MeshPlacementInfo",
    tray_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Compute the 4x4 transformation matrix for a placed mesh.

    This computes the transform needed to move and rotate an original mesh
    to its packed position, accounting for:
    1. Mesh centering during voxelization
    2. Rotation around mesh center
    3. Translation to final voxel position

    Parameters
    ----------
    placement : MeshPlacementInfo
        Placement information including voxel position and orientation.
    tray_origin : tuple of float, optional
        Origin of the tray in world coordinates. Defaults to (0, 0, 0).

    Returns
    -------
    np.ndarray
        4x4 homogeneous transformation matrix.

    Raises
    ------
    ValueError
        If placement was not successful (no valid position).
    """
    if not placement.success or placement.voxel_position is None:
        raise ValueError("Cannot compute transform for failed placement")

    vox = placement.voxel_info

    # Mesh center and half-extents in original mesh coordinates
    mesh_center = (vox.mesh_bounds_min + vox.mesh_bounds_max) / 2
    mesh_half_extents = (vox.mesh_bounds_max - vox.mesh_bounds_min) / 2

    # Get 3x3 rotation matrix
    R = get_rotation_matrix_3x3(placement.orientation_index)

    # After rotation, the bounding box changes size along each axis
    # The rotated half-extents determine the new bounding box
    rotated_half_extents = np.abs(R) @ mesh_half_extents

    # The voxel position is the corner of the bounding box in voxel space
    # Convert to mesh coordinates and find the center
    voxel_pos = np.array(placement.voxel_position, dtype=np.float64)
    final_center = (
        np.array(tray_origin) +
        voxel_pos * vox.pitch +
        rotated_half_extents
    )

    # Build the 4x4 transform matrix
    # Transform: translate mesh center to origin, rotate, translate to final position
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = final_center - R @ mesh_center

    return T


def _import_mesh(bpy, mesh_path: Path, name: str):
    """Import a mesh file into the current Blender scene.

    Parameters
    ----------
    bpy : module
        The bpy module.
    mesh_path : Path
        Path to the mesh file.
    name : str
        Name to assign to the imported object.

    Returns
    -------
    bpy.types.Object
        The imported Blender object.

    Raises
    ------
    FileNotFoundError
        If the mesh file does not exist.
    UnsupportedFormatError
        If the mesh format is not supported.
    """
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    suffix = mesh_path.suffix.lower()

    # Clear selection before import
    bpy.ops.object.select_all(action='DESELECT')

    if suffix == '.stl':
        # Blender 4.0+ uses wm.stl_import, older versions use import_mesh.stl
        if hasattr(bpy.ops.wm, 'stl_import'):
            bpy.ops.wm.stl_import(filepath=str(mesh_path))
        else:
            bpy.ops.import_mesh.stl(filepath=str(mesh_path))
    elif suffix == '.obj':
        # Blender 4.0+ uses wm.obj_import, older versions use import_scene.obj
        if hasattr(bpy.ops.wm, 'obj_import'):
            bpy.ops.wm.obj_import(filepath=str(mesh_path))
        else:
            bpy.ops.import_scene.obj(filepath=str(mesh_path))
    elif suffix == '.ply':
        # Blender 4.0+ uses wm.ply_import, older versions use import_mesh.ply
        if hasattr(bpy.ops.wm, 'ply_import'):
            bpy.ops.wm.ply_import(filepath=str(mesh_path))
        else:
            bpy.ops.import_mesh.ply(filepath=str(mesh_path))
    elif suffix in ('.gltf', '.glb'):
        bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    elif suffix == '.dae':
        bpy.ops.wm.collada_import(filepath=str(mesh_path))
    elif suffix == '.3mf':
        # 3MF import may not be available in all Blender versions
        if hasattr(bpy.ops.import_mesh, 'threemf'):
            bpy.ops.import_mesh.threemf(filepath=str(mesh_path))
        else:
            raise UnsupportedFormatError(
                f"3MF import not available in this Blender version"
            )
    else:
        raise UnsupportedFormatError(
            f"Unsupported mesh format for Blender import: {suffix}"
        )

    # Get the imported object(s)
    imported_objects = bpy.context.selected_objects
    if not imported_objects:
        raise BlenderExportError(f"No objects imported from {mesh_path}")

    # If multiple objects were imported (e.g., from OBJ with groups),
    # join them into one
    if len(imported_objects) > 1:
        bpy.context.view_layer.objects.active = imported_objects[0]
        bpy.ops.object.join()

    obj = bpy.context.active_object
    obj.name = name
    return obj


def _apply_transform(bpy, obj, matrix: np.ndarray) -> None:
    """Apply a 4x4 transformation matrix to a Blender object.

    Parameters
    ----------
    bpy : module
        The bpy module.
    obj : bpy.types.Object
        The Blender object to transform.
    matrix : np.ndarray
        4x4 transformation matrix.
    """
    import mathutils

    # Convert numpy matrix to Blender Matrix
    # Blender's Matrix takes row-major input
    mat = mathutils.Matrix(matrix.tolist())
    obj.matrix_world = mat


def _create_tray_boundary(
    bpy,
    tray_shape: Tuple[int, int, int],
    pitch: float,
    tray_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Create a wireframe cube representing the tray boundary.

    Parameters
    ----------
    bpy : module
        The bpy module.
    tray_shape : tuple of int
        Shape of the tray (x, y, z) in voxels.
    pitch : float
        Size of each voxel in world units.
    tray_origin : tuple of float
        Origin of the tray in world coordinates.
    """
    # Calculate tray dimensions in world units
    dims = np.array(tray_shape) * pitch

    # Create cube
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.active_object
    obj.name = "Tray_Boundary"

    # Scale to tray dimensions (cube default is 2x2x2 centered at origin)
    obj.scale = (dims[0] / 2, dims[1] / 2, dims[2] / 2)

    # Move to correct position (center of tray)
    obj.location = (
        tray_origin[0] + dims[0] / 2,
        tray_origin[1] + dims[1] / 2,
        tray_origin[2] + dims[2] / 2,
    )

    # Set to wireframe display
    obj.display_type = 'WIRE'

    # Apply scale to mesh data
    bpy.ops.object.transform_apply(scale=True)


def export_to_blend(
    result: "PackingResult",
    output_path: Union[str, Path],
    scale: float = 1.0,
    include_tray_boundary: bool = True,
    tray_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Export packed objects to a Blender .blend file.

    This function imports the original mesh files and applies the transforms
    computed from the packing result to position each mesh correctly.

    Parameters
    ----------
    result : PackingResult
        Packing result from `BinPacker.pack_files_for_export()`.
        Must have `mesh_placements` populated.
    output_path : str or Path
        Path for the output .blend file.
    scale : float, optional
        Scale factor applied to all objects. Defaults to 1.0.
    include_tray_boundary : bool, optional
        Whether to include a wireframe cube showing tray boundaries.
        Defaults to True.
    tray_origin : tuple of float, optional
        Origin of the tray in world coordinates. Defaults to (0, 0, 0).

    Raises
    ------
    ImportError
        If the bpy module is not available.
    NoMeshMetadataError
        If `result.mesh_placements` is None.
    FileNotFoundError
        If any mesh file is not found.
    """
    bpy = _get_bpy()

    if result.mesh_placements is None:
        raise NoMeshMetadataError(
            "PackingResult has no mesh_placements. "
            "Use BinPacker.pack_files_for_export() instead of pack_files()."
        )

    output_path = Path(output_path)

    # Clear the default scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Determine pitch from first successful placement
    pitch = None
    for mp in result.mesh_placements:
        if mp.success:
            pitch = mp.voxel_info.pitch
            break

    if pitch is None:
        # No successful placements - nothing to export
        bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
        return

    # Apply scale to pitch
    scaled_pitch = pitch * scale
    scaled_origin = tuple(o * scale for o in tray_origin)

    # Import and transform each successfully placed mesh
    for i, placement in enumerate(result.mesh_placements):
        if not placement.success:
            continue

        # Import mesh
        mesh_name = f"Object_{i:03d}_{placement.mesh_path.stem}"
        obj = _import_mesh(bpy, placement.mesh_path, mesh_name)

        # Compute and apply transform
        # We need to scale the voxel info to account for the scale factor
        scaled_placement = placement
        if scale != 1.0:
            # Create a modified voxel info with scaled pitch
            from .voxelizer import VoxelizationInfo
            scaled_vox_info = VoxelizationInfo(
                mesh_path=placement.voxel_info.mesh_path,
                mesh_bounds_min=placement.voxel_info.mesh_bounds_min * scale,
                mesh_bounds_max=placement.voxel_info.mesh_bounds_max * scale,
                pitch=scaled_pitch,
                voxel_shape=placement.voxel_info.voxel_shape,
            )
            from .packer import MeshPlacementInfo
            scaled_placement = MeshPlacementInfo(
                mesh_path=placement.mesh_path,
                voxel_info=scaled_vox_info,
                voxel_position=placement.voxel_position,
                orientation_index=placement.orientation_index,
                success=placement.success,
            )

        transform = compute_mesh_transform(scaled_placement, scaled_origin)
        _apply_transform(bpy, obj, transform)

        # If scale != 1.0, also scale the object
        if scale != 1.0:
            obj.scale = (scale, scale, scale)
            bpy.ops.object.transform_apply(scale=True)

    # Add tray boundary if requested
    if include_tray_boundary:
        _create_tray_boundary(
            bpy,
            result.tray.shape,
            scaled_pitch,
            scaled_origin,
        )

    # Save the file
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
