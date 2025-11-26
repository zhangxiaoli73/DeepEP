#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <sycl/sycl.hpp>

namespace deep_ep_xpu {

// Maximum number of devices
constexpr int NUM_MAX_DEVICES = 16;

// Maximum number of local experts
constexpr int NUM_MAX_LOCAL_EXPERTS = 256;

// Workspace size (32 MiB)
constexpr size_t NUM_WORKSPACE_BYTES = 32 * 1024 * 1024;

// Buffer alignment
constexpr int BUFFER_ALIGNMENT = 128;

// FP8 configuration
constexpr int FP8_SCALE_BLOCK_SIZE = 128;

// Low latency buffer layout - matching CUDA version structure
struct LowLatencyBuffer {
    int num_clean_int = 0;

    // Dispatch buffers
    void* dispatch_rdma_send_buffer = nullptr;
    void* dispatch_rdma_recv_data_buffer = nullptr;
    int* dispatch_rdma_recv_count_buffer = nullptr;

    // Combine buffers
    void* combine_rdma_send_buffer = nullptr;
    void* combine_rdma_recv_data_buffer = nullptr;
    int* combine_rdma_recv_flag_buffer = nullptr;

    void* combine_rdma_send_buffer_data_start = nullptr;
    size_t num_bytes_per_combine_msg = 0;

    std::pair<int*, int> clean_meta() {
        // dispatch_rdma_recv_count_buffer and combine_rdma_recv_flag_buffer share the same memory
        return {dispatch_rdma_recv_count_buffer, num_clean_int};
    }
};

struct LowLatencyLayout {
    size_t total_bytes = 0;
    LowLatencyBuffer buffers[2];  // Double buffering

    template <typename out_ptr_t = void*, typename count_ptr_t = uint8_t*, typename in_ptr_t = void*>
    out_ptr_t advance(const in_ptr_t& ptr, size_t count) {
        return reinterpret_cast<out_ptr_t>(reinterpret_cast<count_ptr_t>(ptr) + count);
    }

    LowLatencyLayout(void* rdma_buffer, int num_max_dispatch_tokens_per_rank,
                     int hidden, int num_ranks, int num_experts) {
        const int num_scales = hidden / FP8_SCALE_BLOCK_SIZE;
        const int num_local_experts = num_experts / num_ranks;

        // Dispatch and combine layout (matching CUDA version):
        //  - 2 symmetric odd/even send buffer
        //  - 2 symmetric odd/even receive buffers
        //  - 2 symmetric odd/even signaling buffers

        // Dispatch send buffer: [num_max_dispatch_tokens_per_rank, hidden] FP8 + scales
        size_t dispatch_send_x_size = num_max_dispatch_tokens_per_rank * hidden * sizeof(uint8_t);
        size_t dispatch_send_scales_size = num_max_dispatch_tokens_per_rank * num_scales * sizeof(float);
        size_t dispatch_send_size = dispatch_send_x_size + dispatch_send_scales_size;

        // Dispatch recv buffer: [num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden] FP8
        size_t dispatch_recv_x_size = num_local_experts * num_ranks * num_max_dispatch_tokens_per_rank * hidden * sizeof(uint8_t);

        // Dispatch recv count: [num_local_experts] int
        size_t dispatch_recv_count_size = num_local_experts * sizeof(int);

        // Combine send buffer: [num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden] BF16 + flag
        size_t combine_send_x_size = num_local_experts * num_ranks * num_max_dispatch_tokens_per_rank * hidden * sizeof(uint16_t);
        size_t combine_send_flag_size = num_local_experts * num_ranks * num_max_dispatch_tokens_per_rank * sizeof(int);
        size_t combine_send_size = combine_send_x_size + combine_send_flag_size;
        size_t num_bytes_per_combine_msg = hidden * sizeof(uint16_t) + sizeof(int);

        // Combine recv buffer: [num_max_dispatch_tokens_per_rank, hidden] BF16
        size_t combine_recv_x_size = num_max_dispatch_tokens_per_rank * hidden * sizeof(uint16_t);

        // Combine recv flag: [num_max_dispatch_tokens_per_rank * num_topk] int (use max topk = 16)
        size_t combine_recv_flag_size = num_max_dispatch_tokens_per_rank * 16 * sizeof(int);

        // Align sizes
        auto align = [](size_t size) { return (size + BUFFER_ALIGNMENT - 1) / BUFFER_ALIGNMENT * BUFFER_ALIGNMENT; };

        dispatch_send_size = align(dispatch_send_size);
        dispatch_recv_x_size = align(dispatch_recv_x_size);
        dispatch_recv_count_size = align(dispatch_recv_count_size);
        combine_send_size = align(combine_send_size);
        combine_recv_x_size = align(combine_recv_x_size);
        combine_recv_flag_size = align(combine_recv_flag_size);

        // Total size per buffer
        size_t buffer_size = dispatch_send_size + dispatch_recv_x_size + dispatch_recv_count_size +
                            combine_send_size + combine_recv_x_size + combine_recv_flag_size;

        total_bytes = 2 * buffer_size;

        // Setup buffer pointers
        for (int i = 0; i < 2; i++) {
            void* base = rdma_buffer ? advance<void*>(rdma_buffer, i * buffer_size) : nullptr;

            // Dispatch buffers
            buffers[i].dispatch_rdma_send_buffer = base;
            buffers[i].dispatch_rdma_recv_data_buffer = base ? advance<void*>(base, dispatch_send_size) : nullptr;
            buffers[i].dispatch_rdma_recv_count_buffer = base ? advance<int*>(buffers[i].dispatch_rdma_recv_data_buffer, dispatch_recv_x_size) : nullptr;

            // Combine buffers
            buffers[i].combine_rdma_send_buffer = base ? advance<void*>(buffers[i].dispatch_rdma_recv_count_buffer, dispatch_recv_count_size) : nullptr;
            buffers[i].combine_rdma_send_buffer_data_start = buffers[i].combine_rdma_send_buffer;
            buffers[i].combine_rdma_recv_data_buffer = base ? advance<void*>(buffers[i].combine_rdma_send_buffer, combine_send_size) : nullptr;
            buffers[i].combine_rdma_recv_flag_buffer = base ? advance<int*>(buffers[i].combine_rdma_recv_data_buffer, combine_recv_x_size) : nullptr;

            // Metadata
            buffers[i].num_bytes_per_combine_msg = num_bytes_per_combine_msg;
            buffers[i].num_clean_int = static_cast<int>((dispatch_recv_count_size + combine_recv_flag_size) / sizeof(int));
        }
    }
};

// Helper functions
inline size_t align_up(size_t x, size_t alignment) {
    return (x + alignment - 1) / alignment * alignment;
}

inline int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

// Device-side atomic operations
template<typename T>
inline T atomic_load(T* ptr, sycl::memory_order order = sycl::memory_order::seq_cst) {
    sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::device> atomic(*ptr);
    return atomic.load(order);
}

template<typename T>
inline void atomic_store(T* ptr, T value, sycl::memory_order order = sycl::memory_order::seq_cst) {
    sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::device> atomic(*ptr);
    atomic.store(value, order);
}

template<typename T>
inline T atomic_fetch_add(T* ptr, T value, sycl::memory_order order = sycl::memory_order::seq_cst) {
    sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::device> atomic(*ptr);
    return atomic.fetch_add(value, order);
}

template<typename T>
inline bool atomic_compare_exchange(T* ptr, T& expected, T desired, 
                                   sycl::memory_order order = sycl::memory_order::seq_cst) {
    sycl::atomic_ref<T, sycl::memory_order::seq_cst, sycl::memory_scope::device> atomic(*ptr);
    return atomic.compare_exchange_strong(expected, desired, order);
}

}  // namespace deep_ep_xpu

