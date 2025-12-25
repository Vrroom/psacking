# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spectral 3D Bin Packing: A GPU-accelerated 3D bin packing library using FFT-based collision detection. Implements the [Scalable Spectral Packing](https://inkbit3d.com/packing/) algorithm where collision detection is formulated as cross-correlation computed via FFT in O(n log n) time.

**Paper**: `3592126.txt` in project root (ACM paper text)

## Build & Development Commands

```bash
# Install package in development mode
pip install -e .

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_core.py -v

# Run a specific test
pytest tests/test_core.py::test_fft_search_placement -v

# Run tests with coverage
pytest tests/ --cov=spectral_packer

# Run benchmarks
python tests/benchmarks/benchmark_thingi.py

# Manual CMake build (alternative)
mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON
make -j$(nproc)
```

## Architecture

### Layer Structure

```
Python API (spectral_packer/)
        ↓
pybind11 bindings (spectral_packing/bindings.cpp)
        ↓
C++/CUDA core (spectral_packing/)
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `BinPacker` | `spectral_packer/packer.py` | High-level packing API |
| `_core` | `spectral_packing/bindings.cpp` | pybind11 Python bindings |
| `fft3.cu` | `spectral_packing/fft3.cu` | CUDA FFT kernels, `GPUTrayContext` class |
| `packing.cpp` | `spectral_packing/packing.cpp` | Core placement algorithm |
| `VoxSurf/` | `spectral_packing/VoxSurf/` | CPU mesh voxelizer (LibSL) |

### Data Flow

1. **Input**: Mesh files or numpy voxel grids
2. **Voxelization**: Mesh → binary occupancy grid (via VoxSurf C++)
3. **Orientation sampling**: Generate 24 rotations of each item
4. **GPU search**: FFT correlation finds collision-free positions with minimal proximity score
5. **Placement**: Item placed in tray, GPU context invalidated via generation counter

### GPU Context Caching

The C++ layer uses a **generation counter** to cache GPU FFT plans and tray transforms:

```python
# Python side (packer.py)
generation = 0
for item in items:
    search(item, tray, phi, generation)  # Cache reused if generation unchanged
    tray = place(item, tray, pos)
    generation += 1  # Invalidates cache
```

Cache invalidation is O(1) - only compares `generation != g_cached_generation` in `bindings.cpp`.

### Key Data Structures

| Type | Location | Use |
|------|----------|-----|
| `VoxelGrid` | C++ | Nested vectors, slower GPU transfer |
| `FlatVoxelGrid` | C++ | Contiguous memory, fast GPU transfer |
| `np.ndarray (int32)` | Python | Voxel grids passed to/from C++ |

## Important Implementation Details

### FFT Placement Search

Core algorithm in `fft_search_placement()`:
- Cross-correlates item with tray to find collision counts
- Cross-correlates item with distance field for proximity scoring
- Returns `(position, found, score)` tuple

### Interlocking Detection

`interlocking_free_positions()` implements Algorithm 3 from the paper - finds positions where items can be removed without obstruction along any axis direction.

### Orientation Sampling

`spectral_packer/rotations.py` defines 24 orientation matrices (cube rotation group). Items are tested in all 24 orientations during packing.

## Blender Export

Export packing results to `.blend` files with original meshes positioned correctly.

### Installing for Blender

Blender uses its own bundled Python. To use `spectral_packer` from within Blender:

```bash
# Find Blender's Python (adjust path for your Blender version)
BLENDER_PYTHON=~/blender-4.5.0-linux-x64/4.5/python/bin/python3.11

# Install dependencies into Blender's Python
$BLENDER_PYTHON -m pip install numpy trimesh

# Install spectral_packer into Blender's Python
$BLENDER_PYTHON -m pip install -e /path/to/sc3344_final_cpsc424
```

Alternative: set `PYTHONPATH` when invoking Blender:
```bash
PYTHONPATH=/path/to/sc3344_final_cpsc424 blender --background --python script.py
```

### Usage

```python
from spectral_packer import BinPacker, export_to_blend, is_blender_available

packer = BinPacker(tray_size=(120, 120, 80), num_orientations=6)

# Use pack_files_for_export() (not pack_files()) to retain mesh metadata
result = packer.pack_files_for_export(["part1.stl", "part2.obj"])

if is_blender_available():
    export_to_blend(result, "packed.blend")
```

### Running Scripts in Blender

```bash
# Run a script with Blender's Python (headless)
blender --background --python examples/pack_and_export_blender.py

# Run Blender export tests
blender --background --python tests/run_blender_tests.py
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `is_blender_available()` | Check if `bpy` module is available |
| `export_to_blend(result, path)` | Export PackingResult to .blend file |
| `compute_mesh_transform(placement)` | Get 4x4 transform matrix for a placed mesh |

### Important Notes

- Use `pack_files_for_export()` instead of `pack_files()` - the former preserves `mesh_placements` metadata needed for Blender export
- The export applies correct rotation and translation transforms computed from voxel positions
- Tray boundary is rendered as a wireframe cube by default (`include_tray_boundary=True`)

## Test Fixtures

Defined in `tests/conftest.py`:
- `cube_3x3x3`, `cube_5x5x5`, `l_shape` - Sample voxel grids
- `empty_tray_small` (20³), `empty_tray_medium` (50³), `empty_tray_large` (100³)
- `packer_small`, `packer_medium`, `packer_large` - Pre-configured BinPacker instances
- `temp_stl_file` - Creates temporary STL for testing
