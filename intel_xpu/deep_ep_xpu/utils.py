"""
Utility functions for Intel XPU MoE operations
"""

import torch
from typing import Any, Optional, Tuple

class EventOverlap:
    """
    A wrapper class to manage CUDA events, also for better overlapping convenience.

    Attributes:
        event: the CUDA event captured.
        extra_tensors: an easier way to simulate PyTorch tensor `record_stream`, may be useful with CUDA graph.
    """

    def __init__(self, event = None, extra_tensors: Optional[Tuple[torch.Tensor]] = None) -> None:
        """
        Initialize the class.

        Arguments:
            event: the CUDA event captured.
            extra_tensors: an easier way to simulate PyTorch tensor `record_stream`, may be useful with CUDA graph.
        """
        self.event = event

        # NOTES: we use extra tensors to achieve stream recording, otherwise,
        # stream recording will be incompatible with CUDA graph.
        self.extra_tensors = extra_tensors

    def current_stream_wait(self) -> None:
        """
        The current stream `torch.cuda.current_stream()` waits for the event to be finished.
        """
        assert self.event is not None
        self.event.current_stream_wait()

    def __enter__(self) -> Any:
        """
        Utility for overlapping and Python `with` syntax.

        You can overlap the kernels on the current stream with the following example:
        ```python
        event_overlap = event_after_all_to_all_kernels()
        with event_overlap():
            do_something_on_current_stream()
        # After exiting the `with` scope, the current stream with wait the event to be finished.
        ```
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Utility for overlapping and Python `with` syntax.

        Please follow the example in the `__enter__` function.
        """
        if self.event is not None:
            self.event.current_stream_wait()

def init_xpu_distributed(local_rank: int, num_local_ranks: int, backend: str = 'xccl'):
    """
    Initialize distributed training for Intel XPU
    
    Args:
        local_rank: Local rank ID
        num_local_ranks: Number of local ranks
        backend: Distributed backend ('ccl' for Intel)
    
    Returns:
        Tuple of (rank, world_size, process_group)
    """
    import torch.distributed as dist
    import os
    
    # Set XPU device
    torch.xpu.set_device(local_rank)
    
    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Create process group
    group = dist.new_group(list(range(world_size)))
    
    # Set default device and dtype
    torch.set_default_device('xpu:' + str(rank))
    torch.set_default_dtype(torch.bfloat16)
    
    return rank, world_size, group


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Calculate relative difference between two tensors
    
    Args:
        x: First tensor
        y: Second tensor
    
    Returns:
        Relative difference (0 = identical, 1 = completely different)
    """
    x_double = x.double() + 1
    y_double = y.double() + 1
    denominator = (x_double * x_double + y_double * y_double).sum()
    similarity = 2 * (x_double * y_double).sum() / denominator
    return (1 - similarity).item()


def hash_tensor(tensor: torch.Tensor) -> int:
    """
    Compute hash of tensor for verification
    
    Args:
        tensor: Input tensor
    
    Returns:
        Hash value
    """
    # Convert to bytes and compute hash
    tensor_bytes = tensor.cpu().numpy().tobytes()
    return hash(tensor_bytes)


def align_up(x: int, alignment: int) -> int:
    """
    Align value up to alignment boundary
    
    Args:
        x: Value to align
        alignment: Alignment boundary
    
    Returns:
        Aligned value
    """
    return (x + alignment - 1) // alignment * alignment


def per_token_cast_to_fp8(x: torch.Tensor, round_scale: bool = False):
    """
    Cast tensor to FP8 with per-token scaling
    
    Args:
        x: Input tensor [num_tokens, hidden]
        round_scale: Round scales to integers
    
    Returns:
        Tuple of (fp8_tensor, scales)
    """
    num_tokens, hidden = x.shape
    
    # Compute per-token scales
    max_vals = x.abs().max(dim=1, keepdim=True)[0]
    scales = 448.0 / (max_vals + 1e-12)  # FP8 E4M3 range
    
    if round_scale:
        scales = scales.round()
    
    # Quantize
    x_scaled = x * scales
    x_fp8 = x_scaled.clamp(-448, 448).to(torch.int8)
    
    return x_fp8, scales.squeeze()


def per_token_cast_back(x_fp8: torch.Tensor, scales: torch.Tensor, dtype=torch.bfloat16):
    """
    Cast FP8 tensor back to original dtype
    
    Args:
        x_fp8: FP8 tensor
        scales: Scales from quantization
        dtype: Target dtype
    
    Returns:
        Dequantized tensor
    """
    x_float = x_fp8.float() / scales.unsqueeze(1)
    return x_float.to(dtype)


def benchmark_kernel(fn, num_warmup: int = 10, num_iters: int = 100):
    """
    Benchmark a kernel function
    
    Args:
        fn: Function to benchmark
        num_warmup: Number of warmup iterations
        num_iters: Number of benchmark iterations
    
    Returns:
        Average time in milliseconds
    """
    import time
    
    # Warmup
    for _ in range(num_warmup):
        fn()
    torch.xpu.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        fn()
    torch.xpu.synchronize()
    end = time.perf_counter()
    
    avg_time_ms = (end - start) * 1000 / num_iters
    return avg_time_ms


def get_xpu_device_info():
    """
    Get Intel XPU device information
    
    Returns:
        Dictionary with device information
    """
    if not torch.xpu.is_available():
        return {"available": False}
    
    device_count = torch.xpu.device_count()
    current_device = torch.xpu.current_device()
    
    info = {
        "available": True,
        "device_count": device_count,
        "current_device": current_device,
        "device_name": torch.xpu.get_device_name(current_device),
    }
    
    return info

