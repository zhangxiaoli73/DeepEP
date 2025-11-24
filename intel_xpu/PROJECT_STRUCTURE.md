# Intel XPU Deep EP Project Structure

This document provides an overview of the project structure and components.

## Directory Tree

```
intel_xpu/
├── README.md                          # Main documentation
├── INSTALL.md                         # Installation guide
├── QUICKSTART.md                      # Quick start guide
├── PROJECT_STRUCTURE.md               # This file
├── requirements.txt                   # Python dependencies
├── setup.py                          # Python package setup
├── build.sh                          # Build script
├── CMakeLists.txt                    # CMake build configuration
│
├── csrc/                             # C++/SYCL source code
│   ├── deep_ep_xpu.hpp               # Main header file
│   ├── deep_ep_xpu.cpp               # Main implementation
│   ├── python_bindings.cpp           # PyBind11 bindings
│   │
│   └── kernels/                      # SYCL kernel implementations
│       ├── config.hpp                # Configuration and data structures
│       ├── utils.hpp                 # Utility functions
│       ├── low_latency_dispatch.cpp  # Dispatch kernel
│       └── low_latency_combine.cpp   # Combine kernel
│
├── deep_ep_xpu/                      # Python package
│   ├── __init__.py                   # Package initialization
│   ├── buffer.py                     # Buffer management
│   └── utils.py                      # Utility functions
│
├── tests/                            # Test suite
│   ├── test_low_latency_xpu.py       # Main test file
│   └── utils.py                      # Test utilities
│
└── examples/                         # Example code
    └── simple_moe.py                 # Simple MoE example
```

## Component Overview

### Core C++/SYCL Components

#### 1. **deep_ep_xpu.hpp/cpp** - Main Buffer Class
- Manages device memory and communication buffers
- Provides high-level API for dispatch and combine operations
- Handles buffer lifecycle and synchronization

#### 2. **kernels/low_latency_dispatch.cpp** - Dispatch Kernel
- Distributes tokens to experts across devices
- Supports FP8 quantization for reduced latency
- Implements efficient packing and RDMA communication

#### 3. **kernels/low_latency_combine.cpp** - Combine Kernel
- Aggregates expert outputs with weighted reduction
- Supports log-format weights
- Optimized with SYCL sub-groups

#### 4. **kernels/config.hpp** - Configuration
- Data structures (LowLatencyLayout, LowLatencyBuffer)
- Constants and alignment settings
- Atomic operation helpers

#### 5. **kernels/utils.hpp** - Utilities
- BF16/FP8 conversion functions
- Quantization/dequantization helpers
- Sub-group reduction utilities

#### 6. **python_bindings.cpp** - Python Interface
- PyBind11 bindings for C++ classes
- Tensor conversion between PyTorch and SYCL
- Python-friendly API wrappers

### Python Components

#### 1. **deep_ep_xpu/__init__.py** - Package Entry Point
- Imports C++ extension
- Exports main Buffer class
- Version information

#### 2. **deep_ep_xpu/buffer.py** - Buffer Management
- High-level wrapper functions
- EventOverlap class for async operations
- Helper functions for dispatch/combine

#### 3. **deep_ep_xpu/utils.py** - Utilities
- Distributed initialization
- Tensor operations (diff, hash, quantization)
- Benchmarking utilities
- Device information

### Test Components

#### 1. **tests/test_low_latency_xpu.py** - Main Tests
- Dispatch and combine correctness tests
- Multi-precision tests (BF16, FP8)
- Multi-device tests
- Performance verification

#### 2. **tests/utils.py** - Test Utilities
- Distributed setup helpers
- Comparison functions
- Benchmarking tools

### Build System

#### 1. **CMakeLists.txt** - CMake Configuration
- Finds Intel DPC++/SYCL compiler
- Configures PyTorch integration
- Sets up PyBind11
- Defines build targets

#### 2. **setup.py** - Python Package Setup
- CMake integration with setuptools
- Package metadata
- Dependency specification

#### 3. **build.sh** - Build Script
- Environment checking
- Automated build process
- Installation

## Key Features Implementation

### 1. Low Latency Dispatch
**Files**: `kernels/low_latency_dispatch.cpp`, `buffer.py`

- Token counting and layout computation
- FP8 quantization with per-token scaling
- RDMA-based communication
- Double buffering for overlap

### 2. Low Latency Combine
**Files**: `kernels/low_latency_combine.cpp`, `buffer.py`

- Weighted reduction across experts
- Support for log-format weights
- Zero-copy mode
- Async completion

### 3. Multi-Precision Support
**Files**: `kernels/utils.hpp`, `utils.py`

- BF16 (default precision)
- FP8 E4M3 (low latency)
- Automatic quantization/dequantization
- Per-token and per-block scaling

### 4. Distributed Communication
**Files**: `deep_ep_xpu.cpp`, `utils.py`

- Intel CCL backend integration
- Multi-device support
- RDMA-like operations with USM
- Fault tolerance (mask buffer)

## API Layers

### Layer 1: C++ Core (SYCL)
- Direct SYCL kernel implementations
- Memory management
- Device operations

### Layer 2: C++ Interface
- Buffer class
- High-level dispatch/combine methods
- Event and hook management

### Layer 3: Python Bindings
- PyBind11 wrappers
- Tensor conversions
- Python-friendly API

### Layer 4: Python Wrappers
- Simplified wrapper functions
- Utility helpers
- Distributed initialization

## Data Flow

```
Input Tensors (PyTorch)
    ↓
Python Wrapper Functions
    ↓
PyBind11 Bindings
    ↓
C++ Buffer Class
    ↓
SYCL Kernels
    ↓
Intel XPU Device
    ↓
SYCL Kernels
    ↓
C++ Buffer Class
    ↓
PyBind11 Bindings
    ↓
Python Wrapper Functions
    ↓
Output Tensors (PyTorch)
```

## Extension Points

### Adding New Kernels
1. Create kernel in `csrc/kernels/`
2. Add method to Buffer class in `deep_ep_xpu.hpp/cpp`
3. Add Python binding in `python_bindings.cpp`
4. Add wrapper in `buffer.py`
5. Add tests in `tests/`

### Supporting New Precisions
1. Add conversion functions in `kernels/utils.hpp`
2. Update dispatch/combine kernels
3. Add Python helpers in `utils.py`
4. Add tests

### Optimizations
1. Tune buffer layouts in `kernels/config.hpp`
2. Optimize kernels with sub-groups
3. Add async operations
4. Implement communication overlap

## Building and Testing

```bash
# Build
./build.sh

# Test
cd tests
python test_low_latency_xpu.py --num-devices 2

# Example
cd examples
python simple_moe.py
```

## Documentation

- **README.md**: Overview and features
- **INSTALL.md**: Detailed installation instructions
- **QUICKSTART.md**: Quick start guide with examples
- **PROJECT_STRUCTURE.md**: This file

