#include "deep_ep_xpu.hpp"
#include "kernels/internode_ll.hpp"
#include <ATen/xpu/XPUContext.h>
#include <ATen/xpu/XPUEvent.h>
#include <c10/core/StreamGuard.h>
#include <torch/torch.h>

#include <iostream>
#include <stdexcept>
#include <ishmem.h>
#include <ishmemx.h>
// #include <mpi.h>

#define NUM_MAX_NVL_PEERS 8
#define NUM_MAX_RDMA_PEERS 20
#define NUM_WORKSPACE_BYTES (32 * 1024 * 1024)
#define NUM_MAX_LOCAL_EXPERTS 1024
#define NUM_BUFFER_ALIGNMENT_BYTES 128

#define FINISHED_SUM_TAG 1024
#define NUM_WAIT_NANOSECONDS 500

// Helper macro for assertions similar to CUDA version
#define EP_HOST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            throw std::runtime_error("DeepEP XPU assertion failed: " #cond); \
        } \
    } while (0)

namespace deep_ep_xpu {

Buffer::Buffer(int rank,
               int num_ranks,
               size_t buffer_size,
               size_t rdma_buffer_size,
               bool low_latency_mode,
               int num_qps_per_rank,
               bool explicitly_destroy)
    : rank_(rank),
      num_ranks_(num_ranks),
      buffer_size_(buffer_size),
      rdma_buffer_size_(rdma_buffer_size),
      low_latency_mode_(low_latency_mode),
      num_qps_per_rank_(num_qps_per_rank),
      explicitly_destroy_(explicitly_destroy) {

       // Metadata memory
    int64_t barrier_signal_bytes = NUM_MAX_NVL_PEERS * sizeof(int);
    int64_t buffer_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(void*);
    int64_t barrier_signal_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(int*);

    // Get ranks
    auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    auto num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS);
    auto num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);

     auto torch_stream = at::xpu::getCurrentXPUStream();
     auto queue_ = torch_stream.queue();

    // Get device info
    auto device = queue_.get_device();
    num_device_compute_units_ = device.get_info<sycl::info::device::max_compute_units>();
    
    std::cout << "Intel XPU Device: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Compute Units: " << num_device_compute_units_ << std::endl;

    // Allocate buffers
    initialize_buffers();
    
    // Initialize communication
    if (low_latency_mode_) {
        initialize_communication();
    }
    
    available_ = true;
}

Buffer::~Buffer() {
    if (!destroyed_ && !explicitly_destroy_) {
        cleanup();
    }
}

void Buffer::initialize_buffers() {
    // Allocate main buffer (skip if size is 0)
    if (buffer_size_ > 0) {
        main_buffer_ = sycl::malloc_device(buffer_size_, queue_);
        if (!main_buffer_) {
            throw std::runtime_error("Failed to allocate main buffer");
        }
        queue_.memset(main_buffer_, 0, buffer_size_);
    }
    
    // Allocate RDMA buffer
    if (rdma_buffer_size_ > 0) {
        rdma_buffer_ = sycl::malloc_device(rdma_buffer_size_, queue_);
        if (!rdma_buffer_) {
            throw std::runtime_error("Failed to allocate RDMA buffer");
        }
    }
    
    // Allocate workspace
    workspace_ = sycl::malloc_device(NUM_WORKSPACE_BYTES, queue_);
    if (!workspace_) {
        throw std::runtime_error("Failed to allocate workspace");
    }
    
    // Initialize remaining buffers to zero
    if (rdma_buffer_) {
        queue_.memset(rdma_buffer_, 0, rdma_buffer_size_);
    }
    queue_.memset(workspace_, 0, NUM_WORKSPACE_BYTES);
    queue_.wait();
}

void Buffer::initialize_communication() {
    ishmem_init();
    ishmem_barrier_all();
}

void Buffer::cleanup() {
    if (main_buffer_) {
        sycl::free(main_buffer_, queue_);
        main_buffer_ = nullptr;
    }
    if (rdma_buffer_) {
        sycl::free(rdma_buffer_, queue_);
        rdma_buffer_ = nullptr;
    }
    if (workspace_) {
        sycl::free(workspace_, queue_);
        workspace_ = nullptr;
    }
    if (low_latency_mode_) {
        std::cout << "Finalizing ISHMEM..." << std::endl;
        ishmem_finalize();
    }
    destroyed_ = true;
}

void Buffer::destroy() {
    if (!destroyed_) {
        cleanup();
    }
}

void Buffer::synchronize() {
    queue_.wait();
}

size_t Buffer::get_low_latency_rdma_size_hint(int num_max_dispatch_tokens_per_rank,
                                              int hidden,
                                              int num_ranks,
                                              int num_experts) {
    LowLatencyLayout layout(nullptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    return layout.total_bytes;
}

void Buffer::clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank,
                                     int hidden,
                                     int num_experts) {
    EP_HOST_ASSERT(low_latency_mode_);

    LowLatencyLayout layout(rdma_buffer_, num_max_dispatch_tokens_per_rank, hidden, num_ranks_, num_experts);
    auto clean_meta_0 = layout.buffers[0].clean_meta();
    auto clean_meta_1 = layout.buffers[1].clean_meta();

    // Call the clean kernel
    internode_ll::clean_low_latency_buffer(
        queue_,
        clean_meta_0.first,
        clean_meta_0.second,
        clean_meta_1.first,
        clean_meta_1.second,
        rank_,
        num_ranks_,
        mask_buffer_ptr_,
        sync_buffer_ptr_
    );
}

void Buffer::low_latency_query_mask_buffer(const torch::Tensor& mask_status) {
    // Query mask buffer for fault tolerance
    EP_HOST_ASSERT(mask_buffer_ptr_ != nullptr && "Shrink mode must be enabled");
    EP_HOST_ASSERT(mask_status.numel() == num_ranks_ && mask_status.scalar_type() == torch::kInt32);

    internode_ll::query_mask_buffer(
        queue_,
        mask_buffer_ptr_,
        num_ranks_,
        reinterpret_cast<int*>(mask_status.data_ptr())
    );
}

void Buffer::low_latency_update_mask_buffer(int rank_to_mask, bool mask) {
    EP_HOST_ASSERT(mask_buffer_ptr_ != nullptr && "Shrink mode must be enabled");
    EP_HOST_ASSERT(rank_to_mask >= 0 && rank_to_mask < num_ranks_);

    internode_ll::update_mask_buffer(
        queue_,
        mask_buffer_ptr_,
        rank_to_mask,
        mask
    );
}

void Buffer::low_latency_clean_mask_buffer() {
    EP_HOST_ASSERT(mask_buffer_ptr_ != nullptr && "Shrink mode must be enabled");

    internode_ll::clean_mask_buffer(
        queue_,
        mask_buffer_ptr_,
        num_ranks_
    );
}

torch::Tensor Buffer::get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank,
                                                           int hidden,
                                                           int num_experts) const {
    EP_HOST_ASSERT(low_latency_mode_);

    LowLatencyLayout layout(rdma_buffer_, num_max_dispatch_tokens_per_rank, hidden, num_ranks_, num_experts);
    auto& buffer = layout.buffers[low_latency_buffer_idx_];

    auto dtype = torch::kBFloat16;
    auto num_msg_elems = static_cast<int>(buffer.num_bytes_per_combine_msg / sizeof(c10::BFloat16));

    EP_HOST_ASSERT(buffer.num_bytes_per_combine_msg % sizeof(c10::BFloat16) == 0);

    return torch::from_blob(
        buffer.combine_rdma_send_buffer_data_start,
        {num_experts / num_ranks_, num_ranks_ * num_max_dispatch_tokens_per_rank, hidden},
        {num_ranks_ * num_max_dispatch_tokens_per_rank * num_msg_elems, num_msg_elems, 1},
        torch::TensorOptions().dtype(dtype).device(torch::kXPU)
    );
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor,
           std::optional<EventHandle>, std::optional<std::function<void()>>>
Buffer::low_latency_dispatch(
    const torch::Tensor& x,
    const torch::Tensor& topk_idx,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const std::optional<torch::Tensor>& dispatch_wait_recv_cost_stats,
    int num_max_dispatch_tokens_per_rank,
    int num_experts,
    bool use_fp8,
    bool round_scale,
    bool use_ue8m0,
    bool async_finish,
    bool return_recv_hook) {

    EP_HOST_ASSERT(low_latency_mode_);

    // Tensor checks - matching CUDA implementation
    EP_HOST_ASSERT(x.dim() == 2 && x.is_contiguous() && x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(x.size(1) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 && topk_idx.is_contiguous());
    EP_HOST_ASSERT(x.size(0) == topk_idx.size(0) && x.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_idx.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(num_experts % num_ranks_ == 0);

    // Diagnosis tensors validation
    if (cumulative_local_expert_recv_stats.has_value()) {
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->dim() == 1 && cumulative_local_expert_recv_stats->is_contiguous());
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->size(0) == num_experts / num_ranks_);
    }
    if (dispatch_wait_recv_cost_stats.has_value()) {
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->dim() == 1 && dispatch_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->size(0) == num_ranks_);
    }

    auto num_tokens = static_cast<int>(x.size(0));
    auto hidden = static_cast<int>(x.size(1));
    auto num_topk = static_cast<int>(topk_idx.size(1));
    auto num_local_experts = num_experts / num_ranks_;

    // Buffer control
    LowLatencyLayout layout(rdma_buffer_, num_max_dispatch_tokens_per_rank, hidden, num_ranks_, num_experts);
    auto& buffer = layout.buffers[low_latency_buffer_idx_];
    auto& next_buffer = layout.buffers[low_latency_buffer_idx_ ^= 1];

    // Allocate packed tensors - matching CUDA implementation
    auto packed_recv_x = torch::empty(
        {num_local_experts, num_ranks_ * num_max_dispatch_tokens_per_rank, hidden},
        x.options().dtype(use_fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16)
    );
    auto packed_recv_src_info = torch::empty(
        {num_local_experts, num_ranks_ * num_max_dispatch_tokens_per_rank},
        torch::dtype(torch::kInt32).device(torch::kXPU)
    );
    auto packed_recv_layout_range = torch::empty(
        {num_local_experts, num_ranks_},
        torch::dtype(torch::kInt64).device(torch::kXPU)
    );
    auto packed_recv_count = torch::empty(
        {num_local_experts},
        torch::dtype(torch::kInt32).device(torch::kXPU)
    );

    // Allocate column-majored scales
    std::optional<torch::Tensor> packed_recv_x_scales = std::nullopt;
    void* packed_recv_x_scales_ptr = nullptr;

    if (use_fp8) {
        EP_HOST_ASSERT(hidden % 512 == 0);
        if (!use_ue8m0) {
            packed_recv_x_scales = torch::empty(
                {num_local_experts, hidden / 128, num_ranks_ * num_max_dispatch_tokens_per_rank},
                torch::dtype(torch::kFloat32).device(torch::kXPU)
            );
        } else {
            EP_HOST_ASSERT(round_scale);
            packed_recv_x_scales = torch::empty(
                {num_local_experts, hidden / 512, num_ranks_ * num_max_dispatch_tokens_per_rank},
                torch::dtype(torch::kInt).device(torch::kXPU)
            );
        }
        packed_recv_x_scales = torch::transpose(packed_recv_x_scales.value(), 1, 2);
        packed_recv_x_scales_ptr = packed_recv_x_scales->data_ptr();
    }

    // Get next buffer clean metadata
    auto next_clean_meta = next_buffer.clean_meta();

    // Kernel launcher
    auto launcher = [=, &queue = queue_](int phases) {
        internode_ll::dispatch(
            queue,
            packed_recv_x.data_ptr(),
            packed_recv_x_scales_ptr,
            packed_recv_src_info.data_ptr<int>(),
            packed_recv_layout_range.data_ptr<int64_t>(),
            packed_recv_count.data_ptr<int>(),
            mask_buffer_ptr_,
            cumulative_local_expert_recv_stats.has_value() ? cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr,
            dispatch_wait_recv_cost_stats.has_value() ? dispatch_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
            buffer.dispatch_rdma_recv_data_buffer,
            static_cast<int*>(buffer.dispatch_rdma_recv_count_buffer),
            buffer.dispatch_rdma_send_buffer,
            x.data_ptr(),
            topk_idx.data_ptr<int>(),
            next_clean_meta.first,
            next_clean_meta.second,
            num_tokens,
            hidden,
            num_max_dispatch_tokens_per_rank,
            num_topk,
            num_experts,
            rank_,
            num_ranks_,
            use_fp8,
            round_scale,
            use_ue8m0,
            workspace_,
            num_device_compute_units_,
            phases
        );
    };

    // Launch kernel
    launcher(return_recv_hook ? internode_ll::LOW_LATENCY_SEND_PHASE :
             (internode_ll::LOW_LATENCY_SEND_PHASE | internode_ll::LOW_LATENCY_RECV_PHASE));

    // Handle async event
    std::optional<EventHandle> event = std::nullopt;
    if (async_finish) {
        event = EventHandle(queue_);
    }

    // Receiver callback
    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook) {
        recv_hook = [=, &queue = queue_]() {
            internode_ll::dispatch(
                queue,
                packed_recv_x.data_ptr(),
                packed_recv_x_scales_ptr,
                packed_recv_src_info.data_ptr<int>(),
                packed_recv_layout_range.data_ptr<int64_t>(),
                packed_recv_count.data_ptr<int>(),
                mask_buffer_ptr_,
                cumulative_local_expert_recv_stats.has_value() ? cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr,
                dispatch_wait_recv_cost_stats.has_value() ? dispatch_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
                buffer.dispatch_rdma_recv_data_buffer,
                static_cast<int*>(buffer.dispatch_rdma_recv_count_buffer),
                buffer.dispatch_rdma_send_buffer,
                x.data_ptr(),
                topk_idx.data_ptr<int>(),
                next_clean_meta.first,
                next_clean_meta.second,
                num_tokens,
                hidden,
                num_max_dispatch_tokens_per_rank,
                num_topk,
                num_experts,
                rank_,
                num_ranks_,
                use_fp8,
                round_scale,
                use_ue8m0,
                workspace_,
                num_device_compute_units_,
                internode_ll::LOW_LATENCY_RECV_PHASE
            );
        };
    }

    return {packed_recv_x, packed_recv_x_scales, packed_recv_count,
            packed_recv_src_info, packed_recv_layout_range, event, recv_hook};
}

std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
Buffer::low_latency_combine(
    const torch::Tensor& x,
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weights,
    const torch::Tensor& src_info,
    const torch::Tensor& layout_range,
    const std::optional<torch::Tensor>& combine_wait_recv_cost_stats,
    int num_max_dispatch_tokens_per_rank,
    int num_experts,
    bool use_logfmt,
    bool zero_copy,
    bool async_finish,
    bool return_recv_hook,
    const std::optional<torch::Tensor>& out) {

    EP_HOST_ASSERT(low_latency_mode_);

    // Tensor checks - matching CUDA implementation
    EP_HOST_ASSERT(x.dim() == 3 && x.is_contiguous() && x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(x.size(0) == num_experts / num_ranks_);
    EP_HOST_ASSERT(x.size(1) == num_ranks_ * num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(x.size(2) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 && topk_idx.is_contiguous());
    EP_HOST_ASSERT(topk_idx.size(0) == topk_weights.size(0) && topk_idx.size(1) == topk_weights.size(1));
    EP_HOST_ASSERT(topk_idx.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(topk_weights.dim() == 2 && topk_weights.is_contiguous());
    EP_HOST_ASSERT(topk_weights.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32);
    EP_HOST_ASSERT(src_info.dim() == 2 && src_info.is_contiguous());
    EP_HOST_ASSERT(src_info.scalar_type() == torch::kInt32 && x.size(0) == src_info.size(0));
    EP_HOST_ASSERT(layout_range.dim() == 2 && layout_range.is_contiguous());
    EP_HOST_ASSERT(layout_range.scalar_type() == torch::kInt64);
    EP_HOST_ASSERT(layout_range.size(0) == num_experts / num_ranks_ && layout_range.size(1) == num_ranks_);

    if (combine_wait_recv_cost_stats.has_value()) {
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->dim() == 1 && combine_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->size(0) == num_ranks_);
    }

    auto hidden = static_cast<int>(x.size(2));
    auto num_topk = static_cast<int>(topk_weights.size(1));
    auto num_combined_tokens = static_cast<int>(topk_weights.size(0));

    // Buffer control
    LowLatencyLayout layout(rdma_buffer_, num_max_dispatch_tokens_per_rank, hidden, num_ranks_, num_experts);
    auto& buffer = layout.buffers[low_latency_buffer_idx_];
    auto& next_buffer = layout.buffers[low_latency_buffer_idx_ ^= 1];

    // Allocate output tensor
    torch::Tensor combined_x;
    if (out.has_value()) {
        EP_HOST_ASSERT(out->dim() == 2 && out->is_contiguous());
        EP_HOST_ASSERT(out->size(0) == num_combined_tokens && out->size(1) == hidden);
        EP_HOST_ASSERT(out->scalar_type() == x.scalar_type());
        combined_x = out.value();
    } else {
        combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
    }

    // Get next buffer clean metadata
    auto next_clean_meta = next_buffer.clean_meta();

    // Kernel launcher
    auto launcher = [=, &queue = queue_](int phases) {
        internode_ll::combine(
            queue,
            combined_x.data_ptr(),
            buffer.combine_rdma_recv_data_buffer,
            static_cast<int*>(buffer.combine_rdma_recv_flag_buffer),
            buffer.combine_rdma_send_buffer,
            x.data_ptr(),
            topk_idx.data_ptr<int>(),
            topk_weights.data_ptr<float>(),
            src_info.data_ptr<int>(),
            layout_range.data_ptr<int64_t>(),
            mask_buffer_ptr_,
            combine_wait_recv_cost_stats.has_value() ? combine_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
            next_clean_meta.first,
            next_clean_meta.second,
            num_combined_tokens,
            hidden,
            num_max_dispatch_tokens_per_rank,
            num_topk,
            num_experts,
            rank_,
            num_ranks_,
            use_logfmt,
            workspace_,
            num_device_compute_units_,
            phases,
            zero_copy
        );
    };

    // Launch kernel
    launcher(return_recv_hook ? internode_ll::LOW_LATENCY_SEND_PHASE :
             (internode_ll::LOW_LATENCY_SEND_PHASE | internode_ll::LOW_LATENCY_RECV_PHASE));

    // Handle async event
    std::optional<EventHandle> event = std::nullopt;
    if (async_finish) {
        event = EventHandle(queue_);
    }

    // Receiver callback
    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook) {
        recv_hook = [=, &queue = queue_]() {
            internode_ll::combine(
                queue,
                combined_x.data_ptr(),
                buffer.combine_rdma_recv_data_buffer,
                static_cast<int*>(buffer.combine_rdma_recv_flag_buffer),
                buffer.combine_rdma_send_buffer,
                x.data_ptr(),
                topk_idx.data_ptr<int>(),
                topk_weights.data_ptr<float>(),
                src_info.data_ptr<int>(),
                layout_range.data_ptr<int64_t>(),
                mask_buffer_ptr_,
                combine_wait_recv_cost_stats.has_value() ? combine_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
                next_clean_meta.first,
                next_clean_meta.second,
                num_combined_tokens,
                hidden,
                num_max_dispatch_tokens_per_rank,
                num_topk,
                num_experts,
                rank_,
                num_ranks_,
                use_logfmt,
                workspace_,
                num_device_compute_units_,
                internode_ll::LOW_LATENCY_RECV_PHASE,
                zero_copy
            );
        };
    }

    return {combined_x, event, recv_hook};
}

}  // namespace deep_ep_xpu

