#pragma once

#include <sycl/sycl.hpp>
#include <memory>
#include <vector>
#include <functional>
#include <optional>
#include "kernels/config.hpp"

namespace deep_ep_xpu {

// Forward declarations
class Buffer;

// Event handle for async operations
using EventHandle = sycl::event;

// Buffer class for managing MoE communication
class Buffer {
public:
    // Constructor
    Buffer(int rank,                       // Rank in the distributed group
           int num_ranks,                  // Total number of ranks
           size_t buffer_size,             // Main buffer size
           size_t rdma_buffer_size,        // RDMA buffer size
           bool low_latency_mode = false,  // Enable low latency mode
           int num_qps_per_rank = 1,       // Number of queue pairs per rank
           bool explicitly_destroy = false);

    // Destructor
    ~Buffer();

    // Disable copy
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    
    // Low latency dispatch
    std::tuple<void*,                              // packed_recv_x
               std::optional<void*>,               // packed_recv_x_scales
               void*,                              // packed_recv_count
               void*,                              // packed_recv_src_info
               void*,                              // packed_recv_layout_range
               std::optional<EventHandle>,         // event
               std::optional<std::function<void()>>> // hook
    low_latency_dispatch(
        const void* x,                             // Input [num_tokens, hidden]
        const void* topk_idx,                      // Top-K indices [num_tokens, num_topk]
        const std::optional<void*>& cumulative_local_expert_recv_stats,
        const std::optional<void*>& dispatch_wait_recv_cost_stats,
        int num_max_dispatch_tokens_per_rank,
        int num_experts,
        bool use_fp8 = true,
        bool round_scale = false,
        bool use_ue8m0 = false,
        bool async_finish = false,
        bool return_recv_hook = false);
    
    // Low latency combine
    std::tuple<void*,                              // combined_x
               std::optional<EventHandle>,         // event
               std::optional<std::function<void()>>> // hook
    low_latency_combine(
        const void* x,                             // Expert outputs
        const void* topk_idx,                      // Top-K indices
        const void* topk_weights,                  // Top-K weights
        const void* src_info,                      // Source info from dispatch
        const void* layout_range,                  // Layout range from dispatch
        const std::optional<void*>& combine_wait_recv_cost_stats,
        int num_max_dispatch_tokens_per_rank,
        int num_experts,
        bool use_logfmt = false,
        bool zero_copy = false,
        bool async_finish = false,
        bool return_recv_hook = false,
        const std::optional<void*>& out = std::nullopt);
    
    // Clean low latency buffer
    void clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, 
                                  int hidden, 
                                  int num_experts);
    
    // Query mask buffer (for fault tolerance)
    void low_latency_query_mask_buffer(void* mask_status);
    
    // Get RDMA size hint
    static size_t get_low_latency_rdma_size_hint(int num_max_dispatch_tokens_per_rank,
                                                 int hidden,
                                                 int num_ranks,
                                                 int num_experts);
    
    // Synchronize
    void synchronize();
    
    // Destroy (if explicitly_destroy is true)
    void destroy();

private:
    // SYCL queue
    sycl::queue queue_;

    // Distributed info
    int rank_;
    int num_ranks_;

    // Buffers
    void* main_buffer_ = nullptr;
    void* rdma_buffer_ = nullptr;
    size_t buffer_size_;
    size_t rdma_buffer_size_;
    
    // Low latency mode
    bool low_latency_mode_;
    int low_latency_buffer_idx_ = 0;  // Double buffering index
    
    // Workspace
    void* workspace_ = nullptr;
    
    // Flags
    bool available_ = false;
    bool explicitly_destroy_;
    bool destroyed_ = false;
    
    // Communication parameters
    int num_qps_per_rank_;
    
    // Device info
    int num_device_compute_units_;
    
    // Helper methods
    void initialize_buffers();
    void initialize_communication();
    void cleanup();
};

}  // namespace deep_ep_xpu

