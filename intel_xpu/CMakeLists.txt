cmake_minimum_required(VERSION 3.20)
project(deep_ep_xpu LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(USE_ISHMEM "Enable Intel SHMEM support for multi-node communication" ON)
option(BUILD_TESTS "Build test executables" OFF)

# Find Intel oneAPI DPC++/SYCL compiler
if(NOT CMAKE_CXX_COMPILER MATCHES "icpx" AND NOT CMAKE_CXX_COMPILER MATCHES "dpcpp")
    message(WARNING "Intel DPC++/SYCL compiler (icpx/dpcpp) recommended. Current: ${CMAKE_CXX_COMPILER}")
    message(WARNING "To use Intel compiler, set: export CXX=icpx")
endif()

# Find Python and PyTorch
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Try to find PyTorch
find_package(Torch QUIET)
if(NOT Torch_FOUND)
    message(STATUS "PyTorch not found via CMake, trying Python import...")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(TORCH_CMAKE_PATH)
        set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PATH})
        find_package(Torch REQUIRED)
    else()
        message(FATAL_ERROR "PyTorch not found. Please install PyTorch first.")
    endif()
endif()

# Find pybind11
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found via CMake, trying Python import...")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(PYBIND11_CMAKE_PATH)
        set(CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_PATH})
        find_package(pybind11 REQUIRED)
    else()
        message(FATAL_ERROR "pybind11 not found. Please install: pip install pybind11")
    endif()
endif()

# Find Intel SHMEM (optional)
set(ISHMEM_FOUND FALSE)
if(USE_ISHMEM)
    set(ISHMEM_HOME $ENV{ISHMEM_HOME})
    if(NOT ISHMEM_HOME)
        message(STATUS "ISHMEM_HOME not set. Trying default locations...")
        # Try common installation paths
        set(ISHMEM_SEARCH_PATHS
            "/opt/intel/ishmem"
            "/opt/intel/oneapi/ishmem/latest"
            "$ENV{HOME}/ishmem"
        )
        foreach(path ${ISHMEM_SEARCH_PATHS})
            if(EXISTS ${path})
                set(ISHMEM_HOME ${path})
                message(STATUS "Found Intel SHMEM at: ${path}")
                break()
            endif()
        endforeach()
    endif()

    if(ISHMEM_HOME)
        find_library(ISHMEM_LIBRARY
            NAMES ishmem
            PATHS ${ISHMEM_HOME}/lib ${ISHMEM_HOME}/lib64
            NO_DEFAULT_PATH
        )

        find_path(ISHMEM_INCLUDE_DIR
            NAMES ishmem.h
            PATHS ${ISHMEM_HOME}/include
            NO_DEFAULT_PATH
        )

        if(ISHMEM_LIBRARY AND ISHMEM_INCLUDE_DIR)
            set(ISHMEM_FOUND TRUE)
            message(STATUS "Found Intel SHMEM library: ${ISHMEM_LIBRARY}")
            message(STATUS "Found Intel SHMEM include: ${ISHMEM_INCLUDE_DIR}")
            add_compile_definitions(USE_ISHMEM)
        else()
            message(WARNING "Intel SHMEM not found at ${ISHMEM_HOME}")
            message(WARNING "Multi-node communication will be disabled.")
            message(WARNING "To enable, install Intel SHMEM and set ISHMEM_HOME environment variable.")
        endif()
    else()
        message(WARNING "Intel SHMEM not found. Multi-node communication will be disabled.")
        message(WARNING "To enable, install Intel SHMEM and set ISHMEM_HOME environment variable.")
    endif()
else()
    message(STATUS "Intel SHMEM support disabled (USE_ISHMEM=OFF)")
endif()

# SYCL flags
# -fsycl enables SYCL compilation and should automatically link required libraries
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsycl")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsycl")

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    # Additional optimizations for Intel compiler
    if(CMAKE_CXX_COMPILER MATCHES "icpx" OR CMAKE_CXX_COMPILER MATCHES "dpcpp")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
endif()

# Warning flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

# Add Intel SHMEM include directory if found
if(ISHMEM_FOUND)
    include_directories(${ISHMEM_INCLUDE_DIR})
endif()

# Source files
set(SOURCES
    csrc/deep_ep_xpu.cpp
    csrc/kernels/internode_ll.cpp
    csrc/python_bindings.cpp
)

# Header files (for IDE support)
set(HEADERS
    csrc/deep_ep_xpu.hpp
    csrc/kernels/internode_ll.hpp
    csrc/kernels/ishmem_utils.hpp
    csrc/kernels/utils.hpp
    csrc/kernels/config.hpp
)

# Create Python extension module
pybind11_add_module(_C ${SOURCES} ${HEADERS})

# Define the module name for pybind11
# This must match the name used in PYBIND11_MODULE() macro
target_compile_definitions(_C PRIVATE MODULE_NAME=_C)

# Set target properties
set_target_properties(_C PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Find SYCL runtime libraries (required for Intel XPU)
# These are needed because we compile with -fsycl

# Find Level Zero (runtime for Intel GPUs)
find_library(LEVEL_ZERO_LIBRARY
    NAMES ze_loader
    PATHS /lib/x86_64-linux-gnu
)

if(LEVEL_ZERO_LIBRARY)
    message(STATUS "Found Level Zero library: ${LEVEL_ZERO_LIBRARY}")
else()
    message(WARNING "Level Zero library not found.")
endif()
# Link libraries
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES})

# Find torch_python library
# Strategy:
# 1. If user provides TORCH_LIB_DIR, use it
# 2. Otherwise, query from conda/Python environment
# 3. If not found, FATAL ERROR (don't continue with broken build)

# Method 1: Check if user provided TORCH_LIB_DIR via command line
# Usage: cmake .. -DTORCH_LIB_DIR=/path/to/torch/lib
if(DEFINED TORCH_LIB_DIR)
    if(EXISTS ${TORCH_LIB_DIR})
        message(STATUS "PyTorch library directory (user-provided): ${TORCH_LIB_DIR}")
    else()
        message(FATAL_ERROR "User-provided TORCH_LIB_DIR does not exist: ${TORCH_LIB_DIR}")
    endif()
else()
    # Method 2: Query from Python (conda environment or system Python)
    message(FATAL_ERROR "Querying PyTorch library directory from Python environment...")
endif()

# Now search for torch_python in the library directory
find_library(TORCH_PYTHON_LIBRARY
    NAMES torch_python
    PATHS ${TORCH_LIB_DIR}
    NO_DEFAULT_PATH
)

if(TORCH_PYTHON_LIBRARY)
    message(STATUS "✓ Found torch_python library: ${TORCH_PYTHON_LIBRARY}")
    target_link_libraries(_C PRIVATE ${TORCH_PYTHON_LIBRARY})
else()
    # torch_python not found - this is a critical error
    message(FATAL_ERROR "torch_python library not found in ${TORCH_LIB_DIR}")
endif()
if(LEVEL_ZERO_LIBRARY)
    target_link_libraries(_C PRIVATE ${LEVEL_ZERO_LIBRARY})
endif()
# Link Intel SHMEM if found
if(ISHMEM_FOUND)
    target_link_libraries(_C PRIVATE ${ISHMEM_LIBRARY})
    target_compile_definitions(_C PRIVATE USE_ISHMEM)
endif()

# Set output directory
# For development builds, output directly to the package directory
if(NOT DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(_C PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/deep_ep_xpu
    )
else()
    set_target_properties(_C PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    )
endif()

# Set proper extension suffix - pybind11 handles this automatically
# Just ensure we don't override it
set_target_properties(_C PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_C"
)

# Installation
install(TARGETS _C
    LIBRARY DESTINATION deep_ep_xpu
)

# Print configuration summary
message(STATUS "")
message(STATUS "========================================")
message(STATUS "Deep EP XPU Build Configuration")
message(STATUS "========================================")
message(STATUS "  Build Type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Compiler:     ${CMAKE_CXX_COMPILER}")
message(STATUS "  C++ Standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "  C++ Flags:        ${CMAKE_CXX_FLAGS}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  Python:           ${Python3_EXECUTABLE} (${Python3_VERSION})")
message(STATUS "  PyTorch:          ${Torch_VERSION}")
message(STATUS "  pybind11:         ${pybind11_VERSION}")
if(ISHMEM_FOUND)
    message(STATUS "  Intel SHMEM:      ${ISHMEM_LIBRARY}")
else()
    message(STATUS "  Intel SHMEM:      Not found (multi-node disabled)")
endif()
message(STATUS "")
message(STATUS "Install:")
message(STATUS "  Prefix:           ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Module output:    ${CMAKE_CURRENT_SOURCE_DIR}/deep_ep_xpu")
message(STATUS "========================================")
message(STATUS "")

