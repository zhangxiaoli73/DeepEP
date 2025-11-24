/*
 * Intel SHMEM Utility Functions for Intel XPU
 * 
 * This file provides wrapper functions for Intel SHMEM (ishmem) operations
 * to replace NVSHMEM API calls in the original CUDA implementation.
 * 
 * API Mapping:
 * - nvshmemi_ibgda_put_nbi_warp -> ishmemx_putmem_nbi_work_group
 * - nvshmemi_ibgda_amo_nonfetch_add -> ishmemx_int_atomic_add_work_group
 * - nvshmemi_ibgda_rma_p -> ishmemx_int_p_work_group
 * - nvshmemi_get_p2p_ptr -> ishmem_ptr
 * - nvshmemi_ibgda_quiet -> ishmemx_quiet_work_group
 * - nvshmemx_barrier_all_block -> ishmemx_barrier_all_work_group
 */

#pragma once

#include <sycl/sycl.hpp>
#include <ishmem.h>
#include <ishmemx.h>

namespace deep_ep_xpu {
namespace kernels {

// Get P2P pointer if accessible, otherwise return nullptr
// Equivalent to nvshmemi_get_p2p_ptr
SYCL_EXTERNAL inline void* ishmem_get_p2p_ptr(void* ptr, int src_pe, int dst_pe) {
    if (src_pe == dst_pe) {
        return ptr;
    }
    // ishmem_ptr returns local pointer if accessible via P2P, nullptr otherwise
    return ishmem_ptr(ptr, dst_pe);
}

// Non-blocking put operation for work group
// Equivalent to nvshmemi_ibgda_put_nbi_warp
template<typename Group>
SYCL_EXTERNAL inline void ishmem_put_nbi_work_group(
    Group& g,
    void* dst,
    const void* src,
    size_t bytes,
    int dst_pe
) {
    // Use ishmemx_putmem_nbi_work_group for device-initiated non-blocking put
    ishmemx_putmem_nbi_work_group(dst, src, bytes, dst_pe, g);
}

// Atomic add operation (non-fetching) for work group
// Equivalent to nvshmemi_ibgda_amo_nonfetch_add
template<typename Group>
SYCL_EXTERNAL inline void ishmem_atomic_add_nonfetch_work_group(
    Group& g,
    int* dst,
    int value,
    int dst_pe
) {
    // Use ishmemx_int_atomic_add_work_group for device-initiated atomic add
    ishmemx_int_atomic_add_work_group(dst, value, dst_pe, g);
}

// Remote put operation (single value) for work group
// Equivalent to nvshmemi_ibgda_rma_p
template<typename Group>
SYCL_EXTERNAL inline void ishmem_int_p_work_group(
    Group& g,
    int* dst,
    int value,
    int dst_pe
) {
    // Use ishmemx_int_p_work_group for device-initiated put
    ishmemx_int_p_work_group(dst, value, dst_pe, g);
}

// Quiet operation - wait for all outstanding operations to complete
// Equivalent to nvshmemi_ibgda_quiet
template<typename Group>
SYCL_EXTERNAL inline void ishmem_quiet_work_group(Group& g) {
    // Use ishmemx_quiet_work_group to wait for all operations
    ishmemx_quiet_work_group(g);
}

// Barrier for all PEs in work group
// Equivalent to nvshmemx_barrier_all_block
template<typename Group>
SYCL_EXTERNAL inline void ishmem_barrier_all_work_group(Group& g) {
    // Use ishmemx_barrier_all_work_group for barrier
    ishmemx_barrier_all_work_group(g);
}

// Fence operation for work group
template<typename Group>
SYCL_EXTERNAL inline void ishmem_fence_work_group(Group& g) {
    // Use ishmemx_fence_work_group for memory ordering
    ishmemx_fence_work_group(g);
}

// Sync operation for work group
template<typename Group>
SYCL_EXTERNAL inline void ishmem_sync_all_work_group(Group& g) {
    // Use ishmemx_sync_all_work_group for synchronization
    ishmemx_sync_all_work_group(g);
}

// Helper function to check if rank is masked (for fault tolerance)
SYCL_EXTERNAL inline bool is_rank_masked(int* mask_buffer_ptr, int rank, sycl::sub_group& sg) {
    if (mask_buffer_ptr == nullptr) {
        return false;
    }

    // Use sub-group shuffle to broadcast mask value from lane 0
    int mask_value = mask_buffer_ptr[rank];
    mask_value = sycl::group_broadcast(sg, mask_value, 0);
    return mask_value != 0;
}

// Atomic load with acquire semantics
template<typename T>
inline T atomic_load_acquire(T* ptr) {
    sycl::atomic_ref<T, sycl::memory_order::acquire, sycl::memory_scope::device> atomic_ptr(*ptr);
    return atomic_ptr.load();
}

// Atomic store with release semantics
template<typename T>
inline void atomic_store_release(T* ptr, T value) {
    sycl::atomic_ref<T, sycl::memory_order::release, sycl::memory_scope::device> atomic_ptr(*ptr);
    atomic_ptr.store(value);
}

// Atomic add with release semantics
template<typename T>
inline T atomic_add_release(T* ptr, T value) {
    sycl::atomic_ref<T, sycl::memory_order::release, sycl::memory_scope::device> atomic_ptr(*ptr);
    return atomic_ptr.fetch_add(value);
}

// Atomic exchange
template<typename T>
inline T atomic_exchange(T* ptr, T value) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_ptr(*ptr);
    return atomic_ptr.exchange(value);
}

} // namespace kernels
} // namespace deep_ep_xpu

