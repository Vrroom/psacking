/**
 * @file bindings.cpp
 * @brief pybind11 bindings for the spectral packing C++/CUDA library
 *
 * This file exposes the core spectral packing functions to Python,
 * enabling GPU-accelerated 3D bin packing from Python code.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>

#include "types.h"
#include "voxelGrid.h"
#include "indexOps.h"
#include "error.h"
#include "constants.h"

// Forward declarations from packing.cpp
void dft_conv3(const VoxelGrid &a, const VoxelGrid &b, VoxelGrid &result);
void dft_corr3(const VoxelGrid &a, const VoxelGrid &b, VoxelGrid &result);
Index3 fft_search_placement(const VoxelGrid &A, const VoxelGrid &tray, bool &found, double &score);
Index3 fft_search_placement_with_cache(const VoxelGrid &A, const VoxelGrid &tray,
                                        const VoxelGrid &tray_phi, bool &found, double &score);
Index3 fft_search_placement_with_cache_flat(const FlatVoxelGrid &item_flat,
                                             const FlatVoxelGrid &tray_flat,
                                             const FlatVoxelGrid &tray_phi_flat,
                                             bool &found, double &score);
void place_in_tray(const VoxelGrid &item, VoxelGrid &tray, Index3 st_id, int val);
void collision_grid(const VoxelGrid &tray, const VoxelGrid &item, VoxelGrid &corr);

// GPU-resident tray context functions (from fft3.cu)
void gpu_tray_context_init(const FlatVoxelGrid& tray, const FlatVoxelGrid& tray_phi);
void gpu_tray_context_cleanup();
bool gpu_tray_context_is_initialized();
Index3 fft_search_with_gpu_context(const FlatVoxelGrid& item, bool& found, double& score);

// Phase 4: Batch orientation processing (from fft3.cu)
void fft_search_batch(const std::vector<FlatVoxelGrid>& orientations,
                      Index3& best_position, bool& found, double& best_score);

// Forward declarations from LibSL (mesh loading)
#include <LibSL/LibSL.h>
void voxelize(TriangleMesh_Ptr &mesh, VoxelGrid &vg, int voxel_resolution);
void saveVoxelGrid(const char *fname, const VoxelGrid& vg);

namespace py = pybind11;

// ============================================================================
// Type Conversion Functions
// ============================================================================

/**
 * Convert a 3D NumPy array to a VoxelGrid (nested C++ vectors)
 *
 * @param arr NumPy array of shape (X, Y, Z) with int32 dtype
 * @return VoxelGrid containing the same data
 * @throws std::runtime_error if array is not 3D or has wrong dtype
 */
VoxelGrid numpy_to_voxel_grid(py::array_t<int, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info buf = arr.request();

    if (buf.ndim != 3) {
        throw std::runtime_error("Input array must be 3-dimensional, got " +
                                 std::to_string(buf.ndim) + "D");
    }

    int nx = static_cast<int>(buf.shape[0]);
    int ny = static_cast<int>(buf.shape[1]);
    int nz = static_cast<int>(buf.shape[2]);

    VoxelGrid grid;
    resize3d(grid, Index3(nx, ny, nz));

    int* ptr = static_cast<int*>(buf.ptr);
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                grid[i][j][k] = ptr[(i * ny + j) * nz + k];
            }
        }
    }

    return grid;
}

/**
 * Convert a VoxelGrid to a 3D NumPy array
 *
 * @param grid VoxelGrid to convert
 * @return NumPy array of shape (X, Y, Z) with int32 dtype
 */
py::array_t<int> voxel_grid_to_numpy(const VoxelGrid& grid) {
    Index3 sz = get_size(grid);
    int nx = std::get<0>(sz);
    int ny = std::get<1>(sz);
    int nz = std::get<2>(sz);

    // Create output array
    py::array_t<int> arr({nx, ny, nz});
    auto buf = arr.mutable_unchecked<3>();

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                buf(i, j, k) = grid[i][j][k];
            }
        }
    }

    return arr;
}

/**
 * Convert a 3D NumPy array directly to a FlatVoxelGrid (single memcpy)
 */
FlatVoxelGrid numpy_to_flat_grid(py::array_t<int, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info buf = arr.request();

    if (buf.ndim != 3) {
        throw std::runtime_error("Input array must be 3-dimensional, got " +
                                 std::to_string(buf.ndim) + "D");
    }

    int nx = static_cast<int>(buf.shape[0]);
    int ny = static_cast<int>(buf.shape[1]);
    int nz = static_cast<int>(buf.shape[2]);

    FlatVoxelGrid grid(nx, ny, nz);

    // Single memcpy instead of triple loop - numpy is already row-major
    int* ptr = static_cast<int*>(buf.ptr);
    std::memcpy(grid.ptr(), ptr, grid.size_bytes());

    return grid;
}

/**
 * Convert a FlatVoxelGrid to a 3D NumPy array (single memcpy)
 */
py::array_t<int> flat_grid_to_numpy(const FlatVoxelGrid& grid) {
    py::array_t<int> arr({grid.nx, grid.ny, grid.nz});
    auto buf = arr.request();

    // Single memcpy - both are row-major contiguous
    std::memcpy(buf.ptr, grid.ptr(), grid.size_bytes());

    return arr;
}

// ============================================================================
// Python Wrapper Functions
// ============================================================================

/**
 * Find optimal placement for an item in a tray using FFT-based collision detection.
 *
 * This function uses spectral methods (FFT) to efficiently find a non-colliding
 * placement position that minimizes a cost function combining:
 * - Proximity to existing items (encourages tight packing)
 * - Height penalty (prefers lower placements)
 *
 * @param item 3D int32 array representing the item to place (1 = occupied, 0 = empty)
 * @param tray 3D int32 array representing current tray state
 * @return tuple of (position, found, score) where:
 *         - position: (x, y, z) tuple of placement coordinates
 *         - found: bool indicating if valid placement was found
 *         - score: double, the placement score (lower is better)
 */
py::tuple py_fft_search_placement(py::array_t<int> item, py::array_t<int> tray) {
    VoxelGrid item_grid = numpy_to_voxel_grid(item);
    VoxelGrid tray_grid = numpy_to_voxel_grid(tray);

    // Check if item is larger than tray (would cause segfault)
    Index3 item_size = get_size(item_grid);
    Index3 tray_size = get_size(tray_grid);
    if (std::get<0>(item_size) > std::get<0>(tray_size) ||
        std::get<1>(item_size) > std::get<1>(tray_size) ||
        std::get<2>(item_size) > std::get<2>(tray_size)) {
        // Item too large - no valid placement
        return py::make_tuple(py::make_tuple(-1, -1, -1), false, 0.0);
    }

    bool found = false;
    double score = 0.0;

    Index3 position = fft_search_placement(item_grid, tray_grid, found, score);

    auto [x, y, z] = position;
    return py::make_tuple(py::make_tuple(x, y, z), found, score);
}

// ============================================================================
// GPU Context Cache Management
// ============================================================================

static uint64_t g_cached_generation = 0;

/**
 * Find optimal placement using a pre-computed distance field.
 *
 * @param item 3D int32 array representing the item to place
 * @param tray 3D int32 array representing current tray state
 * @param tray_distance Pre-computed distance field from calculate_distance(tray)
 * @param generation Tray version number (increment after each placement)
 * @return tuple of (position, found, score)
 */
py::tuple py_fft_search_placement_with_cache(
    py::array_t<int> item,
    py::array_t<int> tray,
    py::array_t<int> tray_distance,
    uint64_t generation
) {
    FlatVoxelGrid item_flat = numpy_to_flat_grid(item);
    FlatVoxelGrid tray_flat = numpy_to_flat_grid(tray);
    FlatVoxelGrid tray_phi_flat = numpy_to_flat_grid(tray_distance);

    if (item_flat.nx > tray_flat.nx ||
        item_flat.ny > tray_flat.ny ||
        item_flat.nz > tray_flat.nz) {
        return py::make_tuple(py::make_tuple(-1, -1, -1), false, 0.0);
    }

    if (!gpu_tray_context_is_initialized() || generation != g_cached_generation) {
        gpu_tray_context_init(tray_flat, tray_phi_flat);
        g_cached_generation = generation;
    }

    bool found = false;
    double score = 0.0;
    Index3 position = fft_search_with_gpu_context(item_flat, found, score);

    auto [x, y, z] = position;
    return py::make_tuple(py::make_tuple(x, y, z), found, score);
}

/**
 * Place an item in the tray at a specified position.
 *
 * @param item 3D int32 array of the item to place
 * @param tray 3D int32 array of the current tray (modified in place)
 * @param position tuple (x, y, z) specifying placement position
 * @param item_id integer ID to mark the placed item's voxels
 * @return Modified tray as a new NumPy array
 * @throws RuntimeError if placement would cause collision
 */
py::array_t<int> py_place_in_tray(
    py::array_t<int> item,
    py::array_t<int> tray,
    py::tuple position,
    int item_id
) {
    VoxelGrid item_grid = numpy_to_voxel_grid(item);
    VoxelGrid tray_grid = numpy_to_voxel_grid(tray);

    int x = position[0].cast<int>();
    int y = position[1].cast<int>();
    int z = position[2].cast<int>();
    Index3 pos(x, y, z);

    place_in_tray(item_grid, tray_grid, pos, item_id);

    return voxel_grid_to_numpy(tray_grid);
}

/**
 * Voxelize an STL file at a given resolution.
 *
 * @param path Path to the STL file
 * @param resolution Voxel grid resolution (default 128)
 * @return 3D int32 NumPy array with 1 for occupied voxels, 0 for empty
 */
py::array_t<int> py_voxelize_stl(const std::string& path, int resolution) {
    try {
        TriangleMesh_Ptr mesh(loadTriangleMesh(path.c_str()));
        if (!mesh) {
            throw std::runtime_error("Failed to load mesh from: " + path);
        }

        VoxelGrid vg;
        voxelize(mesh, vg, resolution);

        // Make the grid tight (remove empty borders)
        make_voxel_grid_tight(vg);

        return voxel_grid_to_numpy(vg);
    } catch (const std::exception& e) {
        throw std::runtime_error("Voxelization failed for " + path + ": " + e.what());
    }
}

/**
 * Compute 3D FFT convolution of two grids.
 *
 * @param a First input grid
 * @param b Second input grid
 * @return Convolution result
 */
py::array_t<int> py_dft_conv3(py::array_t<int> a, py::array_t<int> b) {
    VoxelGrid va = numpy_to_voxel_grid(a);
    VoxelGrid vb = numpy_to_voxel_grid(b);
    VoxelGrid result;

    dft_conv3(va, vb, result);

    return voxel_grid_to_numpy(result);
}

/**
 * Compute 3D FFT cross-correlation of two grids.
 *
 * @param a First input grid
 * @param b Second input grid
 * @return Cross-correlation result
 */
py::array_t<int> py_dft_corr3(py::array_t<int> a, py::array_t<int> b) {
    VoxelGrid va = numpy_to_voxel_grid(a);
    VoxelGrid vb = numpy_to_voxel_grid(b);
    VoxelGrid result;

    dft_corr3(va, vb, result);

    return voxel_grid_to_numpy(result);
}

/**
 * Calculate Euclidean distance field from occupied voxels.
 *
 * @param grid Input occupancy grid (1 = occupied, 0 = empty)
 * @return Distance field where each cell contains distance to nearest occupied voxel
 */
py::array_t<int> py_calculate_distance(py::array_t<int> grid) {
    VoxelGrid vg = numpy_to_voxel_grid(grid);
    VoxelGrid dist;

    calculate_distance(vg, dist);

    return voxel_grid_to_numpy(dist);
}

/**
 * Compute collision grid between tray and item.
 *
 * @param tray Current tray state
 * @param item Item to test for collisions
 * @return Collision metric grid (0 = no collision at that position)
 */
py::array_t<int> py_collision_grid(py::array_t<int> tray, py::array_t<int> item) {
    VoxelGrid tray_grid = numpy_to_voxel_grid(tray);
    VoxelGrid item_grid = numpy_to_voxel_grid(item);
    VoxelGrid collision;

    // Pad item to match tray size (same as fft_search_placement does internally)
    Index3 tray_size = get_size(tray_grid);
    padto3d(item_grid, tray_size);

    collision_grid(tray_grid, item_grid, collision);

    return voxel_grid_to_numpy(collision);
}

/**
 * Make a voxel grid tight by removing empty borders.
 *
 * @param grid Input grid with potential empty borders
 * @return Tight grid with no empty borders
 */
py::array_t<int> py_make_tight(py::array_t<int> grid) {
    VoxelGrid vg = numpy_to_voxel_grid(grid);
    make_voxel_grid_tight(vg);
    return voxel_grid_to_numpy(vg);
}

/**
 * Get bounds of non-zero voxels in a grid.
 *
 * @param grid Input grid
 * @return tuple of (lo, hi) where lo and hi are (x,y,z) tuples
 */
py::tuple py_get_bounds(py::array_t<int> grid) {
    VoxelGrid vg = numpy_to_voxel_grid(grid);
    Index3 lo, hi;
    get_voxel_grid_bounds(vg, lo, hi);

    auto [lx, ly, lz] = lo;
    auto [hx, hy, hz] = hi;

    return py::make_tuple(
        py::make_tuple(lx, ly, lz),
        py::make_tuple(hx, hy, hz)
    );
}

/**
 * Save a voxel grid to a MagicaVoxel .vox file.
 *
 * @param grid Grid to save
 * @param path Output file path
 */
void py_save_vox(py::array_t<int> grid, const std::string& path) {
    VoxelGrid vg = numpy_to_voxel_grid(grid);
    saveVoxelGrid(path.c_str(), vg);
}

// ============================================================================
// GPU-Resident Tray Context Functions
// ============================================================================

/**
 * Initialize GPU-resident tray context.
 * Pre-computes FFT of tray and distance field, keeping them on GPU.
 */
void py_gpu_tray_init(py::array_t<int> tray, py::array_t<int> tray_distance) {
    FlatVoxelGrid tray_flat = numpy_to_flat_grid(tray);
    FlatVoxelGrid tray_phi_flat = numpy_to_flat_grid(tray_distance);
    gpu_tray_context_init(tray_flat, tray_phi_flat);
}

/**
 * Cleanup GPU-resident tray context and free GPU memory.
 */
void py_gpu_tray_cleanup() {
    gpu_tray_context_cleanup();
}

/**
 * Check if GPU tray context is initialized.
 */
bool py_gpu_tray_is_initialized() {
    return gpu_tray_context_is_initialized();
}

/**
 * Find optimal placement using GPU-resident tray context.
 * Much faster than fft_search_placement_with_cache when testing multiple
 * orientations, as tray data stays on GPU.
 */
py::tuple py_gpu_tray_search(py::array_t<int> item) {
    if (!gpu_tray_context_is_initialized()) {
        throw std::runtime_error("GPU tray context not initialized. Call gpu_tray_init() first.");
    }

    FlatVoxelGrid item_flat = numpy_to_flat_grid(item);

    bool found = false;
    double score = 0.0;
    Index3 position = fft_search_with_gpu_context(item_flat, found, score);

    auto [x, y, z] = position;
    return py::make_tuple(py::make_tuple(x, y, z), found, score);
}

/**
 * Phase 4: Batch search over multiple orientations in a single call.
 *
 * @param orientations List of 3D int32 arrays, one per orientation
 * @param tray 3D int32 array representing current tray state
 * @param tray_distance Pre-computed distance field
 * @param generation Tray version number
 * @return tuple of (position, found, score) for the best orientation
 */
py::tuple py_fft_search_batch(
    py::list orientations,
    py::array_t<int> tray,
    py::array_t<int> tray_distance,
    uint64_t generation
) {
    FlatVoxelGrid tray_flat = numpy_to_flat_grid(tray);
    FlatVoxelGrid tray_phi_flat = numpy_to_flat_grid(tray_distance);

    if (!gpu_tray_context_is_initialized() || generation != g_cached_generation) {
        gpu_tray_context_init(tray_flat, tray_phi_flat);
        g_cached_generation = generation;
    }

    std::vector<FlatVoxelGrid> orientation_grids;
    orientation_grids.reserve(py::len(orientations));
    for (auto item : orientations) {
        auto arr = item.cast<py::array_t<int>>();
        orientation_grids.push_back(numpy_to_flat_grid(arr));
    }

    Index3 best_position;
    bool found = false;
    double best_score = 0.0;
    fft_search_batch(orientation_grids, best_position, found, best_score);

    auto [x, y, z] = best_position;
    return py::make_tuple(py::make_tuple(x, y, z), found, best_score);
}

// ============================================================================
// Module Definition
// ============================================================================

PYBIND11_MODULE(_core, m) {
    m.doc() = R"pbdoc(
        Spectral 3D Bin Packing - Core C++/CUDA Functions
        ==================================================

        This module provides GPU-accelerated 3D bin packing using FFT-based
        collision detection. The algorithm uses spectral methods to efficiently
        find optimal placements for items in a packing tray.

        Core Functions:
            fft_search_placement: Find optimal placement using FFT correlation
            place_in_tray: Place an item at a given position
            voxelize_stl: Convert STL mesh to voxel grid

        FFT Operations:
            dft_conv3: 3D FFT convolution
            dft_corr3: 3D FFT cross-correlation
            calculate_distance: Compute distance field
            collision_grid: Compute collision metric

        Utilities:
            make_tight: Remove empty borders from grid
            get_bounds: Get bounding box of occupied voxels
            save_vox: Save grid to MagicaVoxel format
    )pbdoc";

    // Core packing functions
    m.def("fft_search_placement", &py_fft_search_placement,
          py::arg("item"), py::arg("tray"),
          R"pbdoc(
              Find optimal placement for an item in a tray using FFT-based collision detection.

              Parameters
              ----------
              item : numpy.ndarray
                  3D int32 array representing the item (1=occupied, 0=empty)
              tray : numpy.ndarray
                  3D int32 array representing current tray state

              Returns
              -------
              tuple
                  (position, found, score) where:
                  - position: (x, y, z) placement coordinates
                  - found: bool, whether valid placement exists
                  - score: float, placement quality (lower is better)
          )pbdoc");

    m.def("fft_search_placement_with_cache", &py_fft_search_placement_with_cache,
          py::arg("item"), py::arg("tray"), py::arg("tray_distance"), py::arg("generation"),
          R"pbdoc(
              Find optimal placement using a pre-computed distance field.

              Parameters
              ----------
              item : numpy.ndarray
                  3D int32 array representing the item (1=occupied, 0=empty)
              tray : numpy.ndarray
                  3D int32 array representing current tray state
              tray_distance : numpy.ndarray
                  Pre-computed distance field from calculate_distance(tray)
              generation : int
                  Tray version number (increment after each placement)

              Returns
              -------
              tuple
                  (position, found, score) where:
                  - position: (x, y, z) placement coordinates
                  - found: bool, whether valid placement exists
                  - score: float, placement quality (lower is better)
          )pbdoc");

    m.def("place_in_tray", &py_place_in_tray,
          py::arg("item"), py::arg("tray"), py::arg("position"), py::arg("item_id") = 1,
          R"pbdoc(
              Place an item in the tray at a specified position.

              Parameters
              ----------
              item : numpy.ndarray
                  3D int32 array of the item to place
              tray : numpy.ndarray
                  3D int32 array of the current tray
              position : tuple
                  (x, y, z) placement position
              item_id : int, optional
                  ID to mark placed voxels (default: 1)

              Returns
              -------
              numpy.ndarray
                  Modified tray with item placed

              Raises
              ------
              RuntimeError
                  If placement causes collision with existing items
          )pbdoc");

    m.def("voxelize_stl", &py_voxelize_stl,
          py::arg("path"), py::arg("resolution") = VOXEL_RESOLUTION,
          R"pbdoc(
              Voxelize an STL mesh file.

              Parameters
              ----------
              path : str
                  Path to STL file
              resolution : int, optional
                  Voxel grid resolution (default: 128)

              Returns
              -------
              numpy.ndarray
                  3D int32 array with 1 for occupied voxels
          )pbdoc");

    // FFT operations
    m.def("dft_conv3", &py_dft_conv3,
          py::arg("a"), py::arg("b"),
          "Compute 3D FFT convolution of two grids.");

    m.def("dft_corr3", &py_dft_corr3,
          py::arg("a"), py::arg("b"),
          "Compute 3D FFT cross-correlation of two grids.");

    m.def("calculate_distance", &py_calculate_distance,
          py::arg("grid"),
          "Calculate Euclidean distance field from occupied voxels.");

    m.def("collision_grid", &py_collision_grid,
          py::arg("tray"), py::arg("item"),
          "Compute collision metric between tray and item.");

    // Utility functions
    m.def("make_tight", &py_make_tight,
          py::arg("grid"),
          "Remove empty borders from a voxel grid.");

    m.def("get_bounds", &py_get_bounds,
          py::arg("grid"),
          "Get bounding box of non-zero voxels as ((lo_x, lo_y, lo_z), (hi_x, hi_y, hi_z)).");

    m.def("save_vox", &py_save_vox,
          py::arg("grid"), py::arg("path"),
          "Save voxel grid to MagicaVoxel .vox format.");

    // GPU-Resident Tray Context functions
    m.def("gpu_tray_init", &py_gpu_tray_init,
          py::arg("tray"), py::arg("tray_distance"),
          R"pbdoc(
              Initialize GPU-resident tray context for fast multi-orientation search.

              Pre-computes FFT of tray and distance field, keeping them on GPU.
              This avoids repeated tray transfers when testing multiple item orientations.

              Parameters
              ----------
              tray : numpy.ndarray
                  3D int32 array representing current tray state
              tray_distance : numpy.ndarray
                  Pre-computed distance field from calculate_distance(tray)

              Note
              ----
              Call gpu_tray_cleanup() when done to free GPU memory.
          )pbdoc");

    m.def("gpu_tray_cleanup", &py_gpu_tray_cleanup,
          "Free GPU memory used by the tray context.");

    m.def("gpu_tray_is_initialized", &py_gpu_tray_is_initialized,
          "Check if GPU tray context is initialized.");

    m.def("gpu_tray_search", &py_gpu_tray_search,
          py::arg("item"),
          R"pbdoc(
              Find optimal placement using GPU-resident tray context.

              Much faster than fft_search_placement_with_cache when testing multiple
              orientations of the same item, as tray data stays on GPU.

              Parameters
              ----------
              item : numpy.ndarray
                  3D int32 array representing the item (1=occupied, 0=empty)

              Returns
              -------
              tuple
                  (position, found, score) where:
                  - position: (x, y, z) placement coordinates
                  - found: bool, whether valid placement exists
                  - score: float, placement quality (lower is better)

              Raises
              ------
              RuntimeError
                  If gpu_tray_init() was not called first
          )pbdoc");

    m.def("fft_search_batch", &py_fft_search_batch,
          py::arg("orientations"), py::arg("tray"), py::arg("tray_distance"), py::arg("generation"),
          R"pbdoc(
              Batch search over multiple item orientations in a single call.

              Parameters
              ----------
              orientations : list of numpy.ndarray
                  List of 3D int32 arrays, one per orientation
              tray : numpy.ndarray
                  3D int32 array representing current tray state
              tray_distance : numpy.ndarray
                  Pre-computed distance field from calculate_distance(tray)
              generation : int
                  Tray version number (increment after each placement)

              Returns
              -------
              tuple
                  (position, found, score) for the best orientation
          )pbdoc");

    // Module-level constants
    m.attr("VOXEL_RESOLUTION") = VOXEL_RESOLUTION;
    m.attr("HEIGHT_PENALTY") = P;
    m.attr("__version__") = "0.1.0";
}
