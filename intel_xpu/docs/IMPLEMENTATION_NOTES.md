# Intel XPU Implementation Notes

## Overview

This document describes the implementation details of the Intel XPU port of DeepEP, focusing on the API replacements and design decisions.

## Key Design Principles

1. **Preserve Original Logic**: The implementation maintains the same algorithmic logic as the CUDA version
2. **API Replacement Only**: Only CUDA/NVSHMEM APIs are replaced with SYCL/Intel SHMEM equivalents
3. **Performance Parity**: Aim for similar performance characteristics on equivalent hardware

## CUDA to SYCL Mapping

### Execution Model

| CUDA Concept | SYCL Equivalent | Notes |
|--------------|-----------------|-------|
| Thread | Work-item | Individual execution unit |
| Warp (32 threads) | Sub-group (16-32 items) | Hardware-dependent size |
| Thread Block | Work-group | Group of work-items |
| Grid | ND-range | Multi-dimensional execution space |
| `__global__` | `parallel_for` | Kernel launch |
| `__device__` | Regular function | Device-side function |
| `__syncthreads()` | `group_barrier()` | Work-group synchronization |
| `__syncwarp()` | `group_barrier(sub_group)` | Sub-group synchronization |

### Memory Model

| CUDA | SYCL | Notes |
|------|------|-------|
| `__shared__` | `local_accessor` | Shared/local memory |
| `cudaMalloc` | `sycl::malloc_device` | Device memory allocation |
| `cudaMemcpy` | `queue.memcpy` | Memory copy |
| `cudaFree` | `sycl::free` | Memory deallocation |

### Atomic Operations

| CUDA | SYCL | Notes |
|------|------|-------|
| `atomicAdd` | `atomic_ref::fetch_add` | Atomic addition |
| `atomicExch` | `atomic_ref::exchange` | Atomic exchange |
| `atomicCAS` | `atomic_ref::compare_exchange` | Compare-and-swap |

### Intrinsics

| CUDA | SYCL | Notes |
|------|------|-------|
| `__shfl_sync` | `shuffle` | Warp/sub-group shuffle |
| `__ballot_sync` | `ballot` | Warp/sub-group ballot |
| `__any_sync` | `any_of` | Warp/sub-group any |
| `__all_sync` | `all_of` | Warp/sub-group all |

## NVSHMEM to Intel SHMEM Mapping

### Core Communication APIs

The implementation uses Intel SHMEM (ishmem) to replace NVSHMEM functionality:

1. **Initialization**:
   - NVSHMEM: `nvshmem_init()`
   - Intel SHMEM: `ishmem_init()` or `ishmemx_init_attr()`

2. **Memory Management**:
   - NVSHMEM: `nvshmem_malloc()` / `nvshmem_free()`
   - Intel SHMEM: `ishmem_malloc()` / `ishmem_free()`

3. **RMA Operations**:
   - NVSHMEM: `nvshmemi_ibgda_put_nbi_warp()`
   - Intel SHMEM: `ishmemx_putmem_nbi_work_group()`

4. **Atomic Operations**:
   - NVSHMEM: `nvshmemi_ibgda_amo_nonfetch_add()`
   - Intel SHMEM: `ishmemx_int_atomic_add_work_group()`

5. **Synchronization**:
   - NVSHMEM: `nvshmemx_barrier_all_block()`
   - Intel SHMEM: `ishmemx_barrier_all_work_group()`

See [NVSHMEM_TO_ISHMEM_MAPPING.md](NVSHMEM_TO_ISHMEM_MAPPING.md) for complete API mapping.

## Implementation Differences

### Queue Pair (QP) Management

**NVSHMEM**:
- Explicitly manages multiple queue pairs per PE
- QP ID passed to each communication function
- Allows fine-grained control over communication channels

**Intel SHMEM**:
- QP management is internal to the library
- No explicit QP ID in API calls
- Simplified programming model

**Impact**: The Intel XPU implementation removes explicit QP management, relying on the library's internal optimization.

### P2P Pointer Access

**NVSHMEM**:
```cuda
uint64_t p2p_ptr = nvshmemi_get_p2p_ptr(ptr, src_pe, dst_pe);
if (p2p_ptr == 0) {
    // Use RDMA
} else {
    // Use direct P2P access
}
```

**Intel SHMEM**:
```cpp
void* p2p_ptr = ishmem_ptr(ptr, dst_pe);
if (p2p_ptr == nullptr) {
    // Use RDMA
} else {
    // Use direct P2P access
}
```

**Impact**: Same logic, different return type (uint64_t vs void*).

### Work Group vs Warp

**CUDA Warp**:
- Fixed size: 32 threads
- Implicit synchronization within warp
- Warp-level primitives

**SYCL Sub-Group**:
- Variable size: typically 16 or 32 work-items
- Query size with `get_max_local_range()`
- Similar primitives but may require explicit barriers

**Impact**: Code must be more flexible with sub-group sizes.

## Performance Considerations

### Memory Coalescing

Both CUDA and SYCL benefit from coalesced memory access:
- Access consecutive memory locations in adjacent work-items
- Align data structures to cache line boundaries
- Use vectorized loads/stores where possible

### Communication Optimization

1. **Batching**: Group small messages to reduce overhead
2. **Overlap**: Use non-blocking operations to overlap communication and computation
3. **P2P**: Prefer P2P access when available (lower latency)
4. **Alignment**: Ensure buffers are properly aligned for DMA

### Kernel Launch Configuration

**CUDA**:
```cuda
kernel<<<num_blocks, threads_per_block, shared_mem, stream>>>(args);
```

**SYCL**:
```cpp
queue.submit([&](handler& h) {
    h.parallel_for(nd_range<1>(global_size, local_size), [=](nd_item<1> item) {
        // kernel code
    });
});
```

**Best Practices**:
- Choose work-group size as multiple of sub-group size
- Maximize occupancy while respecting resource limits
- Profile to find optimal configuration

## Testing Strategy

1. **Unit Tests**: Test individual kernels with known inputs
2. **Integration Tests**: Test full dispatch-combine pipeline
3. **Correctness Tests**: Compare outputs with reference implementation
4. **Performance Tests**: Benchmark against CUDA version on equivalent hardware
5. **Multi-Device Tests**: Verify communication across devices

## Known Limitations

1. **Intel SHMEM Maturity**: Intel SHMEM is newer than NVSHMEM, may have fewer optimizations
2. **Hardware Support**: Requires Intel Data Center GPU Max Series or newer
3. **Driver Requirements**: Needs recent Intel GPU drivers with Level Zero support
4. **Documentation**: Intel SHMEM documentation is less extensive than NVSHMEM

## Future Improvements

1. **Kernel Fusion**: Combine multiple small kernels to reduce launch overhead
2. **Persistent Kernels**: Use persistent kernels for lower latency
3. **Advanced Atomics**: Leverage Intel GPU atomic capabilities
4. **Profiling**: Add detailed profiling hooks for performance analysis
5. **Auto-tuning**: Implement auto-tuning for optimal kernel configurations

## References

- [Intel SHMEM Documentation](https://oneapi-src.github.io/ishmem/)
- [SYCL Specification](https://www.khronos.org/sycl/)
- [Intel oneAPI Programming Guide](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/)
- [DeepEP Original Implementation](../../csrc/)

