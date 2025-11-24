"""
Intel XPU Low Latency MoE Communication Library

This package provides optimized SYCL-based kernels for Mixture-of-Experts (MoE)
dispatch and combine operations on Intel XPU devices.
"""

import torch

# Import the C++ extension
try:
    from . import _C
    Buffer = _C.Buffer
except ImportError as e:
    print(f"Warning: Failed to import C++ extension: {e}")
    print("Please build the extension first using: python setup.py install")
    Buffer = None

__version__ = "0.1.0"

__all__ = ["Buffer"]

