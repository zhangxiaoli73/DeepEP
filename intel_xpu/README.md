# Intel XPU Low Latency Implementation

## ✅ Status: Complete

All core kernels have been successfully ported from CUDA/NVSHMEM to SYCL/Intel SHMEM.

This directory contains the Intel XPU implementation of low-latency MoE (Mixture-of-Experts) dispatch and combine operations using SYCL and Intel SHMEM.

## Overview

This is a **faithful port** of DeepEP's CUDA/NVSHMEM implementation to Intel XPU. The implementation preserves all original logic and algorithms, only replacing CUDA APIs with SYCL and NVSHMEM APIs with Intel SHMEM.

### Completed Components

- ✅ **Dispatch Kernel** (`internode_ll.cpp` lines 303-709) - Low-latency token dispatch with FP8 quantization
- ✅ **Combine Kernel** (`internode_ll.cpp` lines 813-1123) - Weighted reduction of expert outputs
- ✅ **Intel SHMEM Integration** (`ishmem_utils.hpp`) - Complete API wrapper layer
- ✅ **Utility Functions** (`utils.hpp`) - BF16/FP8 conversion, atomic operations
- ✅ **Build System** (`CMakeLists.txt`) - CMake configuration with Intel SHMEM support
- ✅ **Documentation** - Comprehensive API mapping and implementation notes

The Intel XPU implementation provides optimized kernels for:
- **Low-latency dispatch**: Distributes tokens to experts across XPU devices
- **Low-latency combine**: Aggregates expert outputs with weighted reduction
- **RDMA communication**: Uses Intel SHMEM for device-initiated communication

## Features

- **SYCL-based kernels**: Portable across Intel GPUs and accelerators
- **Multi-precision support**: BF16 and FP8 operations
- **Asynchronous execution**: Event-based synchronization for overlapping computation and communication
- **Scalable**: Supports multi-device and multi-node configurations
- **Intel SHMEM Integration**: Uses Intel SHMEM (ishmem) for RDMA communication, providing API compatibility with NVSHMEM from the CUDA version

## Directory Structure

```
intel_xpu/
├── README.md                 # This file
├── csrc/                     # C++/SYCL source code
│   ├── kernels/              # SYCL kernel implementations
│   │   ├── low_latency_dispatch.cpp
│   │   ├── low_latency_combine.cpp
│   │   ├── utils.hpp
│   │   └── config.hpp
│   ├── deep_ep_xpu.cpp       # Main C++ interface
│   └── deep_ep_xpu.hpp       # Header file
├── deep_ep_xpu/              # Python package
│   ├── __init__.py
│   ├── buffer.py             # Buffer management
│   └── utils.py              # Utility functions
├── tests/                    # Python tests
│   ├── test_low_latency_xpu.py
│   └── utils.py
├── CMakeLists.txt            # Build configuration
└── setup.py                  # Python package setup

```

## Requirements

- Intel oneAPI Base Toolkit (2024.0 or later)
- Intel Extension for PyTorch
- Python 3.8+
- CMake 3.20+

## Building

```bash
# Source Intel oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Build the extension
cd intel_xpu
mkdir build && cd build
cmake ..
make -j

# Install Python package
cd ..
pip install -e .
```

## Usage

```python
import torch
import intel_extension_for_pytorch as ipex
import deep_ep_xpu

# Initialize distributed group
group = ...  # Your distributed process group

# Create buffer
buffer = deep_ep_xpu.Buffer(
    group=group,
    buffer_size=int(2e9),
    rdma_buffer_size=int(1e9),
    low_latency_mode=True
)

# Dispatch tokens to experts
hidden_states = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device='xpu')
topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk), device='xpu')

recv_hidden_states, recv_count, handle, event, hook = buffer.low_latency_dispatch(
    hidden_states, topk_idx, num_max_dispatch_tokens_per_rank, num_experts
)

# Process with experts...
expert_output = process_experts(recv_hidden_states)

# Combine expert outputs
topk_weights = torch.randn(num_tokens, num_topk, dtype=torch.bfloat16, device='xpu')
combined_output, event, hook = buffer.low_latency_combine(
    expert_output, topk_idx, topk_weights, handle
)
```

## Testing

```bash
# Run tests
cd tests
python test_low_latency_xpu.py --num-devices 2
```

## Performance

The Intel XPU implementation is optimized for:
- Intel Data Center GPU Max Series (PVC)
- Intel Arc GPUs
- Future Intel Xe architectures

Expected performance characteristics:
- Intra-device: High bandwidth via shared memory
- Inter-device: Optimized via Intel Xe Link or PCIe
- Multi-node: Scalable via Intel MPI and CCL

## License

Same as the parent DeepEP project.

