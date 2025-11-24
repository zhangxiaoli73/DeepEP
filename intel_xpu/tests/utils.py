"""
Utility functions for testing Intel XPU MoE operations
"""
import inspect
import torch
import torch.distributed as dist
import os


def init_dist(local_rank: int, num_local_ranks: int, backend: str = 'xccl'):
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))

    sig = inspect.signature(dist.init_process_group)
    params = {
        'backend': 'xccl',
        'init_method': f'tcp://{ip}:{port}',
        'world_size': num_nodes * num_local_ranks,
        'rank': node_rank * num_local_ranks + local_rank,
    }
    if 'device_id' in sig.parameters:
        # noinspection PyTypeChecker
        params['device_id'] = torch.device(f'xpu:{local_rank}')
    dist.init_process_group(**params)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('xpu')
    torch.xpu.set_device(local_rank)

    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """Calculate relative difference between tensors"""
    x_double = x.double() + 1
    y_double = y.double() + 1
    denominator = (x_double * x_double + y_double * y_double).sum()
    similarity = 2 * (x_double * y_double).sum() / denominator
    return (1 - similarity).item()


def hash_tensor(tensor: torch.Tensor) -> int:
    """Compute hash of tensor"""
    # Simple hash based on sum and mean
    return int((tensor.sum().item() * 1e6 + tensor.mean().item() * 1e9) % (2**31))


def align_up(x: int, alignment: int) -> int:
    """Align value up to boundary"""
    return (x + alignment - 1) // alignment * alignment


def per_token_cast_to_fp8(x: torch.Tensor, round_scale: bool = False):
    """Cast to FP8 with per-token scaling"""
    num_tokens, hidden = x.shape
    
    # Compute scales
    max_vals = x.abs().max(dim=1, keepdim=True)[0]
    scales = 448.0 / (max_vals + 1e-12)
    
    if round_scale:
        scales = scales.round()
    
    # Quantize
    x_scaled = x * scales
    x_fp8 = x_scaled.clamp(-448, 448).to(torch.int8)
    
    return x_fp8, scales.squeeze()


def per_token_cast_back(x_fp8: torch.Tensor, scales: torch.Tensor, dtype=torch.bfloat16):
    """Cast FP8 back to original dtype"""
    x_float = x_fp8.float() / scales.unsqueeze(1)
    return x_float.to(dtype)


def benchmark(fn, num_warmup: int = 10, num_iters: int = 100, sync: bool = True):
    """
    Benchmark a function
    
    Args:
        fn: Function to benchmark
        num_warmup: Warmup iterations
        num_iters: Benchmark iterations
        sync: Synchronize after each iteration
    
    Returns:
        Average time in milliseconds
    """
    import time
    
    # Warmup
    for _ in range(num_warmup):
        fn()
        if sync:
            torch.xpu.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        fn()
        if sync:
            torch.xpu.synchronize()
    end = time.perf_counter()
    
    return (end - start) * 1000 / num_iters


def print_device_info():
    """Print Intel XPU device information"""
    if not torch.xpu.is_available():
        print("Intel XPU not available")
        return
    
    device_count = torch.xpu.device_count()
    print(f"Number of XPU devices: {device_count}")
    
    for i in range(device_count):
        print(f"  Device {i}: {torch.xpu.get_device_name(i)}")
    
    current = torch.xpu.current_device()
    print(f"Current device: {current}")


def verify_distributed_setup():
    """Verify distributed setup is correct"""
    if not dist.is_initialized():
        raise RuntimeError("Distributed not initialized")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"Rank {rank}/{world_size} on device {torch.xpu.current_device()}")
    
    # Test all-reduce
    tensor = torch.ones(1, device='xpu') * rank
    dist.all_reduce(tensor)
    expected = sum(range(world_size))
    
    assert tensor.item() == expected, f"All-reduce failed: {tensor.item()} != {expected}"
    print(f"Rank {rank}: Distributed setup verified")

