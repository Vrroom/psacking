cmake_minimum_required(VERSION 3.18)
project(Packing LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON)

if(CMAKE_CUDA_ARCHITECTURES LESS 60)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
endif()

INCLUDE_DIRECTORIES(
  ${PROJECT_SOURCE_DIR}/VoxSurf/LibSL-small
  ${PROJECT_SOURCE_DIR}/VoxSurf/LibSL-small/src/
  ${PROJECT_SOURCE_DIR}/VoxSurf/LibSL-small/src/LibSL
)

CONFIGURE_FILE(
  "${CMAKE_CURRENT_SOURCE_DIR}/VoxSurf/path.h.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/VoxSurf/path.h"
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Common source files (shared between executable and Python module)
set(COMMON_SOURCES
  packing.cpp
  fft3.cu
  indexOps.cpp
  error.cpp
  voxelGrid.cpp
  VoxSurf/LibSL-small/src/LibSL/Math/Math.cpp
  VoxSurf/LibSL-small/src/LibSL/Math/Vertex.cpp
  VoxSurf/LibSL-small/src/LibSL/Mesh/Mesh.cpp
  VoxSurf/LibSL-small/src/LibSL/Mesh/MeshFormat_stl.cpp
  VoxSurf/LibSL-small/src/LibSL/Mesh/VertexFormat_dynamic.cpp
  VoxSurf/LibSL-small/src/LibSL/System/System.cpp
  VoxSurf/LibSL-small/src/LibSL/CppHelpers/CppHelpers.cpp
)

# ============================================================================
# Standalone Executable (original target)
# ============================================================================
add_executable(packing ${COMMON_SOURCES})

target_include_directories(packing PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(packing cufft fftw3 m)

# ============================================================================
# Python Module (pybind11)
# ============================================================================
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

if(BUILD_PYTHON_BINDINGS)
  # Find Python and pybind11
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG QUIET)

  if(NOT pybind11_FOUND)
    # Try to find pybind11 via Python
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
      message(WARNING "pybind11 not found. Python bindings will not be built.")
      set(BUILD_PYTHON_BINDINGS OFF)
    endif()
  endif()
endif()

if(BUILD_PYTHON_BINDINGS)
  message(STATUS "Building Python bindings with pybind11")

  # Create the Python module
  pybind11_add_module(_core MODULE
    bindings.cpp
    ${COMMON_SOURCES}
  )

  target_include_directories(_core PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${PROJECT_SOURCE_DIR}
  )

  target_link_libraries(_core PRIVATE cufft fftw3 m)

  # Set output directory for the module
  set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/spectral_packer"
  )

  # Install the Python module
  install(TARGETS _core
    LIBRARY DESTINATION spectral_packer
  )
endif()
