/*
 * Low Latency Internode Communication Kernels for Intel XPU - Header
 * 
 * This file is a faithful port of csrc/kernels/api.cuh (internode_ll section)
 * Only API replacements are made (CUDA -> SYCL, NVSHMEM -> Intel SHMEM)
 * All function signatures are preserved from the original implementation
 * 
 * Original file: csrc/kernels/api.cuh
 */

#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

namespace deep_ep_xpu {
namespace internode_ll {

// Phase constants for low latency communication
constexpr int LOW_LATENCY_SEND_PHASE = 1;
constexpr int LOW_LATENCY_RECV_PHASE = 2;

// Clean low latency buffer
// Port of: clean_low_latency_buffer from api.cuh
void clean_low_latency_buffer(
    sycl::queue& q,
    int* clean_0,
    int num_clean_int_0,
    int* clean_1,
    int num_clean_int_1,
    int rank,
    int num_ranks,
    int* mask_buffer_ptr,
    int* sync_buffer_ptr
);

// Dispatch kernel - send tokens to experts
// Port of: dispatch from api.cuh (lines 285-313)
void dispatch(
    sycl::queue& q,
    void* packed_recv_x,
    void* packed_recv_x_scales,
    int* packed_recv_src_info,
    int64_t* packed_recv_layout_range,
    int* packed_recv_count,
    int* mask_buffer_ptr,
    int* cumulative_local_expert_recv_stats,
    int64_t* dispatch_wait_recv_cost_stats,
    void* rdma_recv_x,
    int* rdma_recv_count,
    void* rdma_x,
    const void* x,
    const int* topk_idx,
    int* next_clean,
    int num_next_clean_int,
    int num_tokens,
    int hidden,
    int num_max_dispatch_tokens_per_rank,
    int num_topk,
    int num_experts,
    int rank,
    int num_ranks,
    bool use_fp8,
    bool round_scale,
    bool use_ue8m0,
    void* workspace,
    int num_device_sms,
    int phases
);

// Combine kernel - aggregate expert outputs
// Port of: combine from api.cuh (lines 315-343)
void combine(
    sycl::queue& q,
    void* combined_x,
    void* rdma_recv_x,
    int* rdma_recv_flag,
    void* rdma_send_x,
    const void* x,
    const int* topk_idx,
    const float* topk_weights,
    const int* src_info,
    const int64_t* layout_range,
    int* mask_buffer_ptr,
    int64_t* combine_wait_recv_cost_stats,
    int* next_clean,
    int num_next_clean_int,
    int num_combined_tokens,
    int hidden,
    int num_max_dispatch_tokens_per_rank,
    int num_topk,
    int num_experts,
    int rank,
    int num_ranks,
    bool use_logfmt,
    void* workspace,
    int num_device_sms,
    int phases,
    bool zero_copy
);

// Query mask buffer - read mask status
// Port of: query_mask_buffer from api.cuh
void query_mask_buffer(
    sycl::queue& q,
    int* mask_buffer_ptr,
    int num_ranks,
    int* mask_tensor
);

// Update mask buffer - set mask for a specific rank
// Port of: update_mask_buffer from api.cuh
void update_mask_buffer(
    sycl::queue& q,
    int* mask_buffer_ptr,
    int rank,
    bool mask
);

// Clean mask buffer - reset all masks
// Port of: clean_mask_buffer from api.cuh
void clean_mask_buffer(
    sycl::queue& q,
    int* mask_buffer_ptr,
    int num_ranks
);

} // namespace internode_ll
} // namespace deep_ep_xpu

