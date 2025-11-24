#pragma once

#include <cstddef>
#include <cstdint>
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

// Low latency buffer layout
struct LowLatencyBuffer {
    void* send_x = nullptr;           // Send buffer for hidden states
    void* send_x_scales = nullptr;    // Send buffer for FP8 scales
    void* recv_x = nullptr;           // Receive buffer for hidden states
    void* recv_x_scales = nullptr;    // Receive buffer for FP8 scales
    int* recv_count = nullptr;        // Receive count buffer
    int* send_flag = nullptr;         // Send completion flag
    int* recv_flag = nullptr;         // Receive completion flag
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

        // Calculate buffer sizes
        size_t send_x_size = num_max_dispatch_tokens_per_rank * hidden * sizeof(uint8_t);  // FP8
        size_t send_scales_size = num_max_dispatch_tokens_per_rank * num_scales * sizeof(float);
        size_t recv_x_size = num_max_dispatch_tokens_per_rank * num_local_experts * hidden * sizeof(uint8_t);
        size_t recv_scales_size = num_max_dispatch_tokens_per_rank * num_local_experts * num_scales * sizeof(float);
        size_t recv_count_size = num_local_experts * sizeof(int);
        size_t flag_size = sizeof(int);

        // Align sizes
        auto align = [](size_t size) { return (size + BUFFER_ALIGNMENT - 1) / BUFFER_ALIGNMENT * BUFFER_ALIGNMENT; };

        send_x_size = align(send_x_size);
        send_scales_size = align(send_scales_size);
        recv_x_size = align(recv_x_size);
        recv_scales_size = align(recv_scales_size);
        recv_count_size = align(recv_count_size);
        flag_size = align(flag_size);

        // Layout for each buffer (double buffering)
        size_t buffer_size = send_x_size + send_scales_size + recv_x_size + 
                            recv_scales_size + recv_count_size + 2 * flag_size;
        
        total_bytes = 2 * buffer_size;

        // Setup buffer pointers
        for (int i = 0; i < 2; i++) {
            void* base = advance<void*>(rdma_buffer, i * buffer_size);
            
            buffers[i].send_x = base;
            buffers[i].send_x_scales = advance<void*>(base, send_x_size);
            buffers[i].recv_x = advance<void*>(buffers[i].send_x_scales, send_scales_size);
            buffers[i].recv_x_scales = advance<void*>(buffers[i].recv_x, recv_x_size);
            buffers[i].recv_count = advance<int*>(buffers[i].recv_x_scales, recv_scales_size);
            buffers[i].send_flag = advance<int*>(buffers[i].recv_count, recv_count_size);
            buffers[i].recv_flag = advance<int*>(buffers[i].send_flag, flag_size);
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

