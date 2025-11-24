"""
Simple example of using Intel XPU Deep EP for MoE inference

This example demonstrates low-latency dispatch and combine operations
for a Mixture-of-Experts model on Intel XPU.
"""

import torch
import torch.distributed as dist
import intel_extension_for_pytorch as ipex

import sys
sys.path.insert(0, '..')

import deep_ep_xpu
from deep_ep_xpu.buffer import create_buffer, low_latency_dispatch_wrapper, low_latency_combine_wrapper
from deep_ep_xpu.utils import init_xpu_distributed, get_xpu_device_info


def simple_expert(x: torch.Tensor, expert_id: int) -> torch.Tensor:
    """
    Simple expert network (linear transformation for demonstration)
    
    Args:
        x: Input tensor [num_tokens, hidden]
        expert_id: Expert ID
    
    Returns:
        Output tensor [num_tokens, hidden]
    """
    # In real implementation, this would be a neural network
    # For demo, just apply a simple transformation
    return x * (1.0 + expert_id * 0.1)


def run_moe_inference(
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    buffer: deep_ep_xpu.Buffer,
    num_experts: int,
    rank: int
):
    """
    Run MoE inference with low latency dispatch and combine
    
    Args:
        hidden_states: Input hidden states [num_tokens, hidden]
        topk_idx: Top-K expert indices [num_tokens, num_topk]
        topk_weights: Top-K weights [num_tokens, num_topk]
        buffer: Deep EP buffer
        num_experts: Total number of experts
        rank: Current rank
    
    Returns:
        Combined output [num_tokens, hidden]
    """
    num_tokens, hidden = hidden_states.shape
    num_topk = topk_idx.size(1)
    
    print(f"[Rank {rank}] Starting MoE inference...")
    print(f"  Input shape: {hidden_states.shape}")
    print(f"  Top-K: {num_topk}, Experts: {num_experts}")
    
    # Step 1: Dispatch tokens to experts
    print(f"[Rank {rank}] Dispatching tokens to experts...")
    recv_hidden_states, recv_count, handle, event, hook = low_latency_dispatch_wrapper(
        buffer, hidden_states, topk_idx, num_tokens, num_experts,
        use_fp8=True,  # Use FP8 for lower latency
        async_finish=False,
        return_recv_hook=False
    )
    
    print(f"[Rank {rank}] Received {recv_count.sum().item()} tokens")
    
    # Step 2: Process tokens with local experts
    print(f"[Rank {rank}] Processing with local experts...")
    
    if isinstance(recv_hidden_states, tuple):
        # FP8 format - need to dequantize
        from deep_ep_xpu.utils import per_token_cast_back
        recv_x, recv_scales = recv_hidden_states
        expert_input = per_token_cast_back(recv_x, recv_scales)
    else:
        expert_input = recv_hidden_states
    
    # Apply experts (simplified - in real code, would batch by expert)
    expert_output = expert_input.clone()
    num_local_experts = num_experts // dist.get_world_size()
    
    for local_expert_id in range(num_local_experts):
        global_expert_id = rank * num_local_experts + local_expert_id
        # Apply expert to corresponding tokens
        expert_output = simple_expert(expert_output, global_expert_id)
    
    # Step 3: Combine expert outputs
    print(f"[Rank {rank}] Combining expert outputs...")
    combined_output, event, hook = low_latency_combine_wrapper(
        buffer, expert_output, topk_idx, topk_weights, handle,
        use_logfmt=False,
        async_finish=False,
        return_recv_hook=False
    )
    
    print(f"[Rank {rank}] MoE inference completed!")
    print(f"  Output shape: {combined_output.shape}")
    
    return combined_output


def main():
    """Main function"""
    
    # Configuration
    num_devices = 2
    num_tokens = 32
    hidden = 4096
    num_experts = 64
    num_topk = 6
    
    # Initialize distributed
    local_rank = 0  # Would be set by launcher in multi-device setup
    rank, world_size, group = init_xpu_distributed(local_rank, num_devices)
    
    print(f"\n{'='*60}")
    print(f"Intel XPU MoE Inference Example")
    print(f"{'='*60}")
    
    # Print device info
    device_info = get_xpu_device_info()
    print(f"\nDevice Info:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    print(f"\nConfiguration:")
    print(f"  Tokens: {num_tokens}")
    print(f"  Hidden: {hidden}")
    print(f"  Experts: {num_experts}")
    print(f"  Top-K: {num_topk}")
    print(f"  Ranks: {world_size}")
    
    # Create buffer
    rdma_buffer_size = deep_ep_xpu.Buffer.get_low_latency_rdma_size_hint(
        num_tokens, hidden, world_size, num_experts
    )
    
    buffer = create_buffer(
        group,
        buffer_size=int(1e9),
        rdma_buffer_size=rdma_buffer_size,
        low_latency_mode=True,
        num_qps_per_rank=num_experts // world_size,
        explicitly_destroy=True
    )
    
    print(f"\nBuffer created (RDMA size: {rdma_buffer_size / 1e6:.1f} MB)")
    
    # Create sample input
    torch.manual_seed(42 + rank)
    hidden_states = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device='xpu')
    topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk), dtype=torch.int32, device='xpu')
    topk_weights = torch.softmax(torch.randn(num_tokens, num_topk, device='xpu'), dim=-1)
    
    # Run inference
    output = run_moe_inference(hidden_states, topk_idx, topk_weights, buffer, num_experts, rank)
    
    # Verify output
    print(f"\n[Rank {rank}] Output statistics:")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")
    print(f"  Min: {output.min().item():.4f}")
    print(f"  Max: {output.max().item():.4f}")
    
    # Cleanup
    buffer.destroy()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("✓ Example completed successfully!")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

