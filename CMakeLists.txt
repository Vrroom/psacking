cmake_minimum_required(VERSION 3.18)
project(SpectralPacker LANGUAGES CXX CUDA)

# ============================================================================
# Project Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architectures
if(CMAKE_CUDA_ARCHITECTURES LESS 60)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
endif()

# ============================================================================
# Dependencies
# ============================================================================
find_package(CUDAToolkit REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Find pybind11
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_RESULT
  )
  if(pybind11_RESULT EQUAL 0)
    list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
    find_package(pybind11 CONFIG REQUIRED)
  else()
    message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
  endif()
endif()

# ============================================================================
# Include Directories
# ============================================================================
set(SPECTRAL_PACKING_DIR ${CMAKE_SOURCE_DIR}/spectral_packing)

include_directories(
  ${SPECTRAL_PACKING_DIR}
  ${SPECTRAL_PACKING_DIR}/VoxSurf/LibSL-small
  ${SPECTRAL_PACKING_DIR}/VoxSurf/LibSL-small/src/
  ${SPECTRAL_PACKING_DIR}/VoxSurf/LibSL-small/src/LibSL
)

# Configure path.h
configure_file(
  "${SPECTRAL_PACKING_DIR}/VoxSurf/path.h.in"
  "${SPECTRAL_PACKING_DIR}/VoxSurf/path.h"
)

# ============================================================================
# Source Files
# ============================================================================
set(CORE_SOURCES
  ${SPECTRAL_PACKING_DIR}/packing.cpp
  ${SPECTRAL_PACKING_DIR}/fft3.cu
  ${SPECTRAL_PACKING_DIR}/indexOps.cpp
  ${SPECTRAL_PACKING_DIR}/error.cpp
  ${SPECTRAL_PACKING_DIR}/voxelGrid.cpp
  ${SPECTRAL_PACKING_DIR}/VoxSurf/LibSL-small/src/LibSL/Math/Math.cpp
  ${SPECTRAL_PACKING_DIR}/VoxSurf/LibSL-small/src/LibSL/Math/Vertex.cpp
  ${SPECTRAL_PACKING_DIR}/VoxSurf/LibSL-small/src/LibSL/Mesh/Mesh.cpp
  ${SPECTRAL_PACKING_DIR}/VoxSurf/LibSL-small/src/LibSL/Mesh/MeshFormat_stl.cpp
  ${SPECTRAL_PACKING_DIR}/VoxSurf/LibSL-small/src/LibSL/Mesh/VertexFormat_dynamic.cpp
  ${SPECTRAL_PACKING_DIR}/VoxSurf/LibSL-small/src/LibSL/System/System.cpp
  ${SPECTRAL_PACKING_DIR}/VoxSurf/LibSL-small/src/LibSL/CppHelpers/CppHelpers.cpp
)

# ============================================================================
# Python Extension Module
# ============================================================================
pybind11_add_module(_core MODULE
  ${SPECTRAL_PACKING_DIR}/bindings.cpp
  ${CORE_SOURCES}
)

target_include_directories(_core PRIVATE
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${SPECTRAL_PACKING_DIR}
)

target_link_libraries(_core PRIVATE cufft fftw3 m)

# Compiler flags
target_compile_options(_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -O3>
)

# ============================================================================
# Installation
# ============================================================================
# Install to spectral_packer package directory
install(TARGETS _core LIBRARY DESTINATION spectral_packer)
