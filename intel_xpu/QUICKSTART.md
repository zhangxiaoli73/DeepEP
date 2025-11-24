# Quick Start Guide

Get started with Intel XPU Deep EP in 5 minutes!

## Prerequisites

- Intel XPU device (Data Center GPU Max, Arc GPU, etc.)
- Intel oneAPI Base Toolkit installed
- Python 3.8+

## Installation

```bash
# 1. Source Intel oneAPI environment
source /opt/intel/oneapi/setvars.sh

# 2. Install dependencies
pip install torch intel-extension-for-pytorch

# 3. Build and install
cd intel_xpu
./build.sh
```

## Basic Usage

### Example 1: Simple MoE Dispatch and Combine

```python
import torch
import deep_ep_xpu
from deep_ep_xpu.buffer import create_buffer

# Initialize distributed (if using multiple devices)
import torch.distributed as dist
dist.init_process_group(backend='ccl')

# Create buffer
buffer = create_buffer(
    process_group=dist.group.WORLD,
    buffer_size=int(2e9),
    rdma_buffer_size=int(1e9),
    low_latency_mode=True
)

# Prepare data
num_tokens, hidden, num_experts, num_topk = 32, 4096, 64, 6
hidden_states = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device='xpu')
topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk), device='xpu')
topk_weights = torch.softmax(torch.randn(num_tokens, num_topk, device='xpu'), dim=-1)

# Dispatch tokens to experts
recv_hidden_states, recv_count, handle, event, hook = buffer.low_latency_dispatch(
    hidden_states, topk_idx, None, None,
    num_max_dispatch_tokens_per_rank=num_tokens,
    num_experts=num_experts,
    use_fp8=True
)

# Process with experts (your expert network here)
expert_output = process_experts(recv_hidden_states)

# Combine expert outputs
combined_output, event, hook = buffer.low_latency_combine(
    expert_output, topk_idx, topk_weights,
    handle[0], handle[1], None,
    num_max_dispatch_tokens_per_rank=num_tokens,
    num_experts=num_experts
)

print(f"Output shape: {combined_output.shape}")
```

### Example 2: Using Wrapper Functions

```python
from deep_ep_xpu.buffer import (
    create_buffer,
    low_latency_dispatch_wrapper,
    low_latency_combine_wrapper
)

# Create buffer
buffer = create_buffer(
    process_group=dist.group.WORLD,
    low_latency_mode=True
)

# Dispatch (simplified API)
recv_hidden_states, recv_count, handle, event, hook = \
    low_latency_dispatch_wrapper(
        buffer, hidden_states, topk_idx,
        num_max_dispatch_tokens_per_rank=num_tokens,
        num_experts=num_experts,
        use_fp8=True
    )

# Combine (simplified API)
combined_output, event, hook = \
    low_latency_combine_wrapper(
        buffer, expert_output, topk_idx, topk_weights, handle
    )
```

### Example 3: Multi-Device Setup

```bash
# Launch with 2 XPU devices
mpirun -n 2 python your_script.py

# Or with torchrun
torchrun --nproc_per_node=2 your_script.py
```

```python
# In your_script.py
from deep_ep_xpu.utils import init_xpu_distributed

# Initialize
rank, world_size, group = init_xpu_distributed(
    local_rank=int(os.environ.get('LOCAL_RANK', 0)),
    num_local_ranks=int(os.environ.get('WORLD_SIZE', 1))
)

print(f"Rank {rank}/{world_size} on device {torch.xpu.current_device()}")

# Rest of your code...
```

## Running Tests

```bash
# Single device
cd tests
python test_low_latency_xpu.py --num-devices 1

# Multiple devices
mpirun -n 2 python test_low_latency_xpu.py --num-devices 2
```

## Running Examples

```bash
cd examples
python simple_moe.py
```

## Key Features

### Low Latency Mode
- Optimized for inference decoding
- Minimal communication overhead
- FP8 quantization support

### High Throughput Mode
- Optimized for training and prefilling
- Batch processing
- Efficient memory usage

### Multi-Precision Support
- BF16 (default)
- FP8 E4M3 (for low latency)
- Automatic quantization/dequantization

## Performance Tips

1. **Use FP8 for inference**: `use_fp8=True` reduces latency
2. **Batch tokens**: Larger batches improve throughput
3. **Tune buffer sizes**: Match your workload
4. **Use multiple devices**: Scale horizontally

## Common Patterns

### Pattern 1: Inference Loop

```python
for batch in dataloader:
    # Dispatch
    recv_data, recv_count, handle, _, _ = dispatch_wrapper(
        buffer, batch, topk_idx, num_tokens, num_experts
    )
    
    # Process
    expert_out = expert_network(recv_data)
    
    # Combine
    output, _, _ = combine_wrapper(
        buffer, expert_out, topk_idx, topk_weights, handle
    )
```

### Pattern 2: Async Processing

```python
# Dispatch with async
recv_data, recv_count, handle, event, hook = dispatch_wrapper(
    buffer, batch, topk_idx, num_tokens, num_experts,
    async_finish=True, return_recv_hook=True
)

# Do other work...

# Wait for completion
if hook:
    hook()
```

## Next Steps

- Read the full [README.md](README.md)
- Check [INSTALL.md](INSTALL.md) for detailed installation
- Explore [examples/](examples/) for more use cases
- Run [tests/](tests/) to verify your setup

## Troubleshooting

**Q: Import error?**
```bash
# Make sure oneAPI is sourced
source /opt/intel/oneapi/setvars.sh
```

**Q: No XPU device?**
```bash
# Check devices
python -c "import torch; print(torch.xpu.device_count())"
```

**Q: Build failed?**
```bash
# Clean and rebuild
rm -rf build
./build.sh
```

For more help, see [INSTALL.md](INSTALL.md#troubleshooting).

