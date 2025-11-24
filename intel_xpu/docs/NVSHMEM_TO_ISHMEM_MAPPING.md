# NVSHMEM to Intel SHMEM API Mapping

This document describes the mapping between NVSHMEM APIs used in the original CUDA implementation and Intel SHMEM (ishmem) APIs used in the Intel XPU implementation.

## Overview

Intel SHMEM (ishmem) is Intel's implementation of OpenSHMEM for Intel GPUs and accelerators. It provides similar functionality to NVSHMEM but with different API names and conventions.

## Key Differences

1. **Naming Convention**:
   - NVSHMEM: `nvshmem*` / `nvshmemi_*` / `nvshmemx_*`
   - Intel SHMEM: `ishmem*` / `ishmemx_*`

2. **Device-Initiated Operations**:
   - NVSHMEM: Uses `_warp` suffix for warp-level operations
   - Intel SHMEM: Uses `_work_group` suffix for work-group operations

3. **IBGDA (InfiniBand GPU Direct Async)**:
   - NVSHMEM: Has explicit IBGDA functions (`nvshmemi_ibgda_*`)
   - Intel SHMEM: RDMA operations are implicit in standard functions

## API Mapping Table

| NVSHMEM API | Intel SHMEM API | Description |
|-------------|-----------------|-------------|
| `nvshmemi_get_p2p_ptr(ptr, src_pe, dst_pe)` | `ishmem_ptr(ptr, dst_pe)` | Get P2P accessible pointer |
| `nvshmemi_ibgda_put_nbi_warp(dst, src, bytes, pe, qp, lane, idx)` | `ishmemx_putmem_nbi_work_group(dst, src, bytes, pe, group)` | Non-blocking put |
| `nvshmemi_ibgda_amo_nonfetch_add(dst, val, pe, qp)` | `ishmemx_int_atomic_add_work_group(dst, val, pe, group)` | Atomic add (non-fetching) |
| `nvshmemi_ibgda_rma_p(dst, val, pe, qp)` | `ishmemx_int_p_work_group(dst, val, pe, group)` | Remote put (single value) |
| `nvshmemi_ibgda_quiet(pe, qp)` | `ishmemx_quiet_work_group(group)` | Wait for operations to complete |
| `nvshmemx_barrier_all_block()` | `ishmemx_barrier_all_work_group(group)` | Barrier across all PEs |
| `nvshmem_fence()` | `ishmemx_fence_work_group(group)` | Memory fence |
| `nvshmem_quiet()` | `ishmemx_quiet_work_group(group)` | Quiet all operations |
| `nvshmem_barrier_all()` | `ishmem_barrier_all()` | Host-side barrier |
| `nvshmem_sync_all()` | `ishmem_sync_all()` | Host-side sync |

## Detailed Mapping

### 1. P2P Pointer Access

**NVSHMEM:**
```cuda
uint64_t dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
if (dst_p2p_ptr == 0) {
    // Use RDMA
} else {
    // Use P2P direct access
}
```

**Intel SHMEM:**
```cpp
void* dst_p2p_ptr = ishmem_ptr(dst_ptr, dst_rank);
if (dst_p2p_ptr == nullptr) {
    // Use RDMA
} else {
    // Use P2P direct access
}
```

### 2. Non-Blocking Put

**NVSHMEM:**
```cuda
nvshmemi_ibgda_put_nbi_warp(
    dst_ptr,      // destination address
    src_ptr,      // source address
    num_bytes,    // number of bytes
    dst_rank,     // destination PE
    qp_id,        // queue pair ID
    lane_id,      // lane ID in warp
    message_idx   // message index
);
```

**Intel SHMEM:**
```cpp
auto sg = item.get_sub_group();
ishmemx_putmem_nbi_work_group(
    dst_ptr,      // destination address
    src_ptr,      // source address
    num_bytes,    // number of bytes
    dst_rank,     // destination PE
    sg            // sub-group (equivalent to warp)
);
```

### 3. Atomic Add (Non-Fetching)

**NVSHMEM:**
```cuda
nvshmemi_ibgda_amo_nonfetch_add(
    dst_ptr,      // destination address
    value,        // value to add
    dst_rank,     // destination PE
    qp_id         // queue pair ID
);
```

**Intel SHMEM:**
```cpp
auto sg = item.get_sub_group();
ishmemx_int_atomic_add_work_group(
    dst_ptr,      // destination address
    value,        // value to add
    dst_rank,     // destination PE
    sg            // sub-group
);
```

### 4. Remote Put (Single Value)

**NVSHMEM:**
```cuda
nvshmemi_ibgda_rma_p(
    reinterpret_cast<int*>(dst_ptr),
    value,
    dst_rank,
    qp_id
);
```

**Intel SHMEM:**
```cpp
auto sg = item.get_sub_group();
ishmemx_int_p_work_group(
    reinterpret_cast<int*>(dst_ptr),
    value,
    dst_rank,
    sg
);
```

### 5. Quiet Operation

**NVSHMEM:**
```cuda
// Quiet specific QP to specific PE
nvshmemi_ibgda_quiet(dst_rank, qp_id);
```

**Intel SHMEM:**
```cpp
// Quiet all operations in work group
auto sg = item.get_sub_group();
ishmemx_quiet_work_group(sg);
```

### 6. Barrier

**NVSHMEM:**
```cuda
// Block-level barrier
nvshmemx_barrier_all_block();
```

**Intel SHMEM:**
```cpp
// Work-group barrier
auto wg = item.get_group();
ishmemx_barrier_all_work_group(wg);
```

## Important Notes

### Queue Pair (QP) Management

- **NVSHMEM**: Explicitly manages multiple queue pairs per PE for parallelism
- **Intel SHMEM**: QP management is handled internally by the library

### Work Group vs Warp

- **CUDA Warp**: 32 threads executing in lockstep
- **SYCL Sub-Group**: Similar concept, typically 16 or 32 work-items
- **SYCL Work-Group**: Larger group of work-items (like CUDA thread block)

Use `item.get_sub_group()` for warp-equivalent operations.

### Memory Ordering

Both NVSHMEM and Intel SHMEM provide similar memory ordering guarantees:
- Fence operations ensure ordering
- Quiet operations ensure completion
- Atomic operations have acquire/release semantics

### Initialization

**NVSHMEM:**
```cpp
nvshmem_init();
```

**Intel SHMEM:**
```cpp
ishmem_init();
// Or with attributes:
ishmemx_attr_t attr;
attr.initialize_runtime = true;
ishmemx_init_attr(&attr);
```

### Finalization

**NVSHMEM:**
```cpp
nvshmem_finalize();
```

**Intel SHMEM:**
```cpp
ishmem_finalize();
```

## Performance Considerations

1. **Batching**: Both libraries benefit from batching small operations
2. **Alignment**: Ensure data is properly aligned for optimal performance
3. **Overlap**: Use non-blocking operations to overlap communication and computation
4. **P2P**: Use P2P pointers when available for lower latency

## References

- [Intel SHMEM Documentation](https://oneapi-src.github.io/ishmem/)
- [NVSHMEM Documentation](https://docs.nvidia.com/nvshmem/)
- [OpenSHMEM Specification](http://openshmem.org/)

