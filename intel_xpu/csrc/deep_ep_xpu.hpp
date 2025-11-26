#pragma once

#include <sycl/sycl.hpp>
#include <torch/types.h>
#include <ATen/xpu/XPUEvent.h>
#include <memory>
#include <vector>
#include <functional>
#include <optional>
#include "kernels/config.hpp"

namespace deep_ep_xpu {

// Forward declarations
class Buffer;

// Event handle for async operations - wrapping XPU event similar to CUDA's EventHandle
class EventHandle {
public:
    EventHandle() : event_() {}
    explicit EventHandle(sycl::queue& queue) {
        // Record event on the queue
        event_ = queue.ext_oneapi_submit_barrier();
    }

    void current_stream_wait() {
        // Wait for the event on current stream
        event_.wait();
    }

    sycl::event get_event() { return event_; }

private:
    sycl::event event_;
};

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

    // Low latency dispatch - API matching CUDA version (using torch::Tensor)
    std::tuple<torch::Tensor,                      // packed_recv_x
               std::optional<torch::Tensor>,       // packed_recv_x_scales
               torch::Tensor,                      // packed_recv_count
               torch::Tensor,                      // packed_recv_src_info
               torch::Tensor,                      // packed_recv_layout_range
               std::optional<EventHandle>,         // event
               std::optional<std::function<void()>>> // hook
    low_latency_dispatch(
        const torch::Tensor& x,                    // Input [num_tokens, hidden]
        const torch::Tensor& topk_idx,             // Top-K indices [num_tokens, num_topk]
        const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
        const std::optional<torch::Tensor>& dispatch_wait_recv_cost_stats,
        int num_max_dispatch_tokens_per_rank,
        int num_experts,
        bool use_fp8 = true,
        bool round_scale = false,
        bool use_ue8m0 = false,
        bool async_finish = false,
        bool return_recv_hook = false);

    // Low latency combine - API matching CUDA version (using torch::Tensor)
    std::tuple<torch::Tensor,                      // combined_x
               std::optional<EventHandle>,         // event
               std::optional<std::function<void()>>> // hook
    low_latency_combine(
        const torch::Tensor& x,                    // Expert outputs [num_local_experts, num_tokens, hidden]
        const torch::Tensor& topk_idx,             // Top-K indices [num_combined_tokens, num_topk]
        const torch::Tensor& topk_weights,         // Top-K weights [num_combined_tokens, num_topk]
        const torch::Tensor& src_info,             // Source info from dispatch
        const torch::Tensor& layout_range,         // Layout range from dispatch
        const std::optional<torch::Tensor>& combine_wait_recv_cost_stats,
        int num_max_dispatch_tokens_per_rank,
        int num_experts,
        bool use_logfmt = false,
        bool zero_copy = false,
        bool async_finish = false,
        bool return_recv_hook = false,
        const std::optional<torch::Tensor>& out = std::nullopt);

    // Clean low latency buffer
    void clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank,
                                  int hidden,
                                  int num_experts);

    // Query mask buffer (for fault tolerance) - using torch::Tensor
    void low_latency_query_mask_buffer(const torch::Tensor& mask_status);

    // Update mask buffer
    void low_latency_update_mask_buffer(int rank_to_mask, bool mask);

    // Clean mask buffer
    void low_latency_clean_mask_buffer();

    // Get next low latency combine buffer
    torch::Tensor get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank,
                                                       int hidden,
                                                       int num_experts) const;

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
    mutable int low_latency_buffer_idx_ = 0;  // Double buffering index (mutable for const methods)

    // Shrink mode buffers (for fault tolerance)
    int* mask_buffer_ptr_ = nullptr;
    int* sync_buffer_ptr_ = nullptr;

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

