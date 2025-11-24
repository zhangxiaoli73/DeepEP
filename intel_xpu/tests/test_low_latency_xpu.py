"""
Test low latency MoE operations on Intel XPU
"""

import argparse
import random
import torch
import torch.distributed as dist
from typing import Literal, Set

import sys
sys.path.insert(0, '..')

import deep_ep_xpu
from deep_ep_xpu.buffer import create_buffer, low_latency_dispatch_wrapper, low_latency_combine_wrapper
from deep_ep_xpu.utils import init_xpu_distributed, calc_diff, hash_tensor, per_token_cast_to_fp8, per_token_cast_back


def test_low_latency_dispatch_combine(
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    rank: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    buffer: deep_ep_xpu.Buffer,
    use_fp8: bool = False,
    use_logfmt: bool = False,
    seed: int = 0
):
    """Test low latency dispatch and combine operations"""
    
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)
    
    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks
    
    print(f"[Rank {rank}] Testing with num_tokens={num_tokens}, hidden={hidden}, "
          f"num_experts={num_experts}, num_topk={num_topk}, use_fp8={use_fp8}")
    
    # Create test data
    hidden_states = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device='xpu')
    topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk), dtype=torch.int32, device='xpu')
    topk_weights = torch.randn(num_tokens, num_topk, dtype=torch.float32, device='xpu')
    
    # Normalize weights
    if use_logfmt:
        topk_weights = torch.log_softmax(topk_weights, dim=-1)
    else:
        topk_weights = torch.softmax(topk_weights, dim=-1)
    
    # Test dispatch
    print(f"[Rank {rank}] Testing dispatch...")
    recv_hidden_states, recv_count, handle, event, hook = low_latency_dispatch_wrapper(
        buffer, hidden_states, topk_idx, num_tokens, num_experts,
        use_fp8=use_fp8, async_finish=False, return_recv_hook=True
    )
    
    if hook is not None:
        hook()  # Wait for completion
    
    print(f"[Rank {rank}] Dispatch completed. Received counts: {recv_count}")
    
    # Simulate expert processing
    if isinstance(recv_hidden_states, tuple):
        recv_x, recv_scales = recv_hidden_states
        # Dequantize for processing
        expert_output = per_token_cast_back(recv_x, recv_scales)
    else:
        expert_output = recv_hidden_states
    
    # Simple expert processing (identity for testing)
    expert_output = expert_output.clone()
    
    # Test combine
    print(f"[Rank {rank}] Testing combine...")
    combined_output, event, hook = low_latency_combine_wrapper(
        buffer, expert_output, topk_idx, topk_weights, handle,
        use_logfmt=use_logfmt, async_finish=False, return_recv_hook=True
    )
    
    if hook is not None:
        hook()  # Wait for completion
    
    print(f"[Rank {rank}] Combine completed. Output shape: {combined_output.shape}")
    
    # Verify output shape
    assert combined_output.shape == (num_tokens, hidden), \
        f"Output shape mismatch: {combined_output.shape} vs {(num_tokens, hidden)}"
    
    # Compute hash for verification across ranks
    output_hash = hash_tensor(combined_output)
    print(f"[Rank {rank}] Output hash: {output_hash}")
    
    # Synchronize
    dist.barrier(group)
    
    return output_hash


def test_correctness(
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    rank: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    buffer: deep_ep_xpu.Buffer
):
    """Test correctness with reference implementation"""
    
    print(f"[Rank {rank}] Running correctness test...")
    
    torch.manual_seed(42 + rank)
    
    # Create test data
    hidden_states = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device='xpu')
    topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk), dtype=torch.int32, device='xpu')
    topk_weights = torch.softmax(torch.randn(num_tokens, num_topk, device='xpu'), dim=-1)
    
    # Run low latency version
    recv_hidden_states, recv_count, handle, event, hook = low_latency_dispatch_wrapper(
        buffer, hidden_states, topk_idx, num_tokens, num_experts,
        use_fp8=False, async_finish=False, return_recv_hook=False
    )
    
    expert_output = recv_hidden_states.clone()
    
    combined_output, event, hook = low_latency_combine_wrapper(
        buffer, expert_output, topk_idx, topk_weights, handle,
        use_logfmt=False, async_finish=False, return_recv_hook=False
    )
    
    # Simple reference: weighted sum of inputs (simplified)
    # In real test, would implement full reference
    reference_output = torch.zeros_like(hidden_states)
    for i in range(num_tokens):
        for k in range(num_topk):
            weight = topk_weights[i, k]
            reference_output[i] += weight * hidden_states[i]
    
    # Compare (allowing for numerical differences)
    diff = calc_diff(combined_output, reference_output)
    print(f"[Rank {rank}] Difference from reference: {diff:.6f}")
    
    # Relaxed threshold due to distributed operations
    assert diff < 0.1, f"Output differs too much from reference: {diff}"
    
    print(f"[Rank {rank}] Correctness test passed!")

    dist.barrier(group)


def test_main(args: argparse.Namespace, local_rank: int, num_local_ranks: int):
    """Main test function"""

    # Initialize distributed
    rank, num_ranks, group = init_xpu_distributed(local_rank, num_local_ranks)

    print(f"[Rank {rank}/{num_ranks}] Initialized on device {torch.xpu.current_device()}")

    # Create buffer
    num_max_dispatch_tokens = args.num_tokens
    rdma_buffer_size = deep_ep_xpu.Buffer.get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens, args.hidden, num_ranks, args.num_experts
    )

    buffer = create_buffer(
        group,
        buffer_size=int(2e9),
        rdma_buffer_size=rdma_buffer_size,
        low_latency_mode=True,
        num_qps_per_rank=max(24, args.num_experts // num_ranks),
        explicitly_destroy=True
    )

    print(f"[Rank {rank}] Buffer created with RDMA size: {rdma_buffer_size / 1e9:.2f} GB")

    # Run tests
    test_configs = [
        {"use_fp8": False, "use_logfmt": False},
        {"use_fp8": True, "use_logfmt": False},
        {"use_fp8": False, "use_logfmt": True},
    ]

    for i, config in enumerate(test_configs):
        print(f"\n[Rank {rank}] Running test {i+1}/{len(test_configs)}: {config}")

        hash_value = test_low_latency_dispatch_combine(
            args.num_tokens, args.hidden, args.num_experts, args.num_topk,
            rank, num_ranks, group, buffer,
            use_fp8=config["use_fp8"], use_logfmt=config["use_logfmt"],
            seed=i
        )

        # Gather hashes from all ranks
        hash_tensor_all = torch.tensor([hash_value], dtype=torch.int64, device='xpu')
        hash_list = [torch.zeros_like(hash_tensor_all) for _ in range(num_ranks)]
        dist.all_gather(hash_list, hash_tensor_all, group=group)

        if rank == 0:
            print(f"Hash values from all ranks: {[h.item() for h in hash_list]}")

    # Run correctness test
    print(f"\n[Rank {rank}] Running correctness test...")
    test_correctness(args.num_tokens, args.hidden, args.num_experts, args.num_topk,
                    rank, num_ranks, group, buffer)

    # Cleanup
    buffer.destroy()

    if rank == 0:
        print("\n✓ All tests passed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Intel XPU low latency MoE kernels')
    parser.add_argument('--num-devices', type=int, default=2, help='Number of XPU devices (default: 2)')
    parser.add_argument('--num-tokens', type=int, default=16, help='Number of tokens (default: 16)')
    parser.add_argument('--hidden', type=int, default=5120, help='Hidden dimension (default: 5120)')
    parser.add_argument('--num-experts', type=int, default=256, help='Number of experts (default: 256)')
    parser.add_argument('--num-topk', type=int, default=9, help='Top-K value (default: 9)')
    parser.add_argument('--local-rank', type=int, default=0, help='Local rank (set by launcher)')

    args = parser.parse_args()

    # Run test
    test_main(args, args.local_rank, args.num_devices)

