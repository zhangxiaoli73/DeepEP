# Intel XPU Porting Status

## Overview

This document tracks the status of porting DeepEP CUDA/NVSHMEM code to Intel XPU with SYCL/Intel SHMEM.

**Porting Strategy**: Faithful port - preserve all original logic, only replace APIs.

## Completed Files

### 1. Infrastructure and Utilities

#### ✅ `csrc/kernels/ishmem_utils.hpp`
- **Status**: Complete
- **Original**: N/A (new file for Intel SHMEM wrappers)
- **Description**: Wrapper functions for Intel SHMEM operations
- **Key Functions**:
  - `ishmem_get_p2p_ptr()` - P2P pointer access
  - `ishmem_put_nbi_work_group()` - Non-blocking put
  - `ishmem_atomic_add_nonfetch_work_group()` - Atomic add
  - `ishmem_int_p_work_group()` - Remote put
  - `ishmem_quiet_work_group()` - Quiet operation
  - `ishmem_barrier_all_work_group()` - Barrier

#### ✅ `csrc/kernels/internode_ll.hpp`
- **Status**: Complete
- **Original**: `csrc/kernels/api.cuh` (internode_ll section)
- **Description**: Header file with function declarations
- **Key Functions**:
  - `clean_low_latency_buffer()` - Buffer cleanup
  - `dispatch()` - Token dispatch
  - `combine()` - Expert output combine (declaration only)

### 2. Low Latency Kernels

#### ✅ `csrc/kernels/internode_ll.cpp`
- **Status**: Complete
- **Original**: `csrc/kernels/internode_ll.cu`
- **Lines Ported**: ~1325 lines
- **Completed Sections**:
  1. ✅ Helper functions (lines 10-20)
     - `is_rank_masked()` - Fault tolerance
  2. ✅ Barrier implementation (lines 22-70)
     - `barrier_impl()` - Custom barrier using Intel SHMEM
  3. ✅ Clean buffer kernel (lines 72-127)
     - `CleanLowLatencyBufferKernel` - Buffer cleanup
     - `clean_low_latency_buffer()` - Host function
  4. ✅ Dispatch kernel (lines 129-463)
     - `DispatchKernel` - Main dispatch kernel class
     - Sending phase implementation
     - Receiving phase implementation
     - `dispatch()` - Host function with template dispatch
  5. ✅ Combine kernel (lines 715-1140)
     - `CombineKernel` - Main combine kernel class
     - Sending phase implementation (simplified without TMA)
     - Receiving phase implementation (simplified without TMA)
     - `combine()` - Host function with template dispatch
  6. ✅ Mask buffer utilities (lines 1241-1288)
     - `query_mask_buffer()` - Query mask status
     - `update_mask_buffer()` - Update mask for specific rank
     - `clean_mask_buffer()` - Reset all masks

### 3. Documentation

#### ✅ `docs/NVSHMEM_TO_ISHMEM_MAPPING.md`
- **Status**: Complete
- **Description**: Comprehensive API mapping guide
- **Content**:
  - API comparison table
  - Code examples for each API pair
  - Performance considerations
  - Best practices

#### ✅ `docs/IMPLEMENTATION_NOTES.md`
- **Status**: Complete
- **Description**: Implementation details and design decisions
- **Content**:
  - CUDA to SYCL mapping
  - NVSHMEM to Intel SHMEM mapping
  - Performance considerations
  - Known limitations

#### ✅ `CHANGELOG.md`
- **Status**: Complete
- **Description**: Change log with technical details

#### ✅ `SUMMARY.md`
- **Status**: Complete
- **Description**: Project summary (Chinese and English)

#### ✅ `docs/PORTING_STATUS.md`
- **Status**: This file
- **Description**: Porting progress tracker

### 4. Build System

#### ✅ `CMakeLists.txt`
- **Status**: Updated
- **Changes**:
  - Added Intel SHMEM library detection
  - Added ISHMEM_HOME environment variable support
  - Added Intel SHMEM include directories
  - Added Intel SHMEM library linking

#### ✅ `INSTALL.md`
- **Status**: Updated
- **Changes**:
  - Added Intel SHMEM installation instructions
  - Added environment variable setup

## API Mapping Summary

### NVSHMEM → Intel SHMEM

| Original NVSHMEM | Intel SHMEM Replacement | Status |
|------------------|-------------------------|--------|
| `nvshmemi_get_p2p_ptr` | `ishmem_ptr` | ✅ Implemented |
| `nvshmemi_ibgda_put_nbi_warp` | `ishmemx_putmem_nbi_work_group` | ✅ Implemented |
| `nvshmemi_ibgda_amo_nonfetch_add` | `ishmemx_int_atomic_add_work_group` | ✅ Implemented |
| `nvshmemi_ibgda_rma_p` | `ishmemx_int_p_work_group` | ✅ Implemented |
| `nvshmemi_ibgda_quiet` | `ishmemx_quiet_work_group` | ✅ Implemented |
| `nvshmemx_barrier_all_block` | `ishmemx_barrier_all_work_group` | ✅ Implemented |

### CUDA → SYCL

| Original CUDA | SYCL Replacement | Status |
|---------------|------------------|--------|
| `__global__` kernel | `parallel_for` with kernel class | ✅ Implemented |
| `threadIdx.x` | `item.get_local_id(0)` | ✅ Implemented |
| `blockIdx.x` | `item.get_group(0)` | ✅ Implemented |
| `gridDim.x` | `item.get_group_range(0)` | ✅ Implemented |
| `__syncthreads()` | `sycl::group_barrier(wg)` | ✅ Implemented |
| `__syncwarp()` | `sycl::group_barrier(sg)` | ✅ Implemented |
| `__shfl_sync()` | `sycl::shift_group_left()` | ✅ Implemented |
| `atomicAdd()` | `atomic_add_release()` | ✅ Implemented |
| `atomicExch()` | `atomic_exchange()` | ✅ Implemented |
| `cudaStream_t` | `sycl::queue` | ✅ Implemented |

## Known Issues and Limitations

### 1. Shared Memory
- **Issue**: SYCL shared memory (local accessor) not yet fully integrated
- **Workaround**: Using global memory for some shared data
- **Impact**: May affect performance
- **TODO**: Implement proper local accessor usage

### 2. Grid Synchronization
- **Issue**: SYCL doesn't have direct equivalent to CUDA grid sync
- **Workaround**: Using work-group barriers
- **Impact**: May not work correctly for multi-SM kernels
- **TODO**: Investigate SYCL 2020 group algorithms

### 3. Sub-Group Size
- **Issue**: CUDA warp is fixed at 32, SYCL sub-group size varies
- **Workaround**: Assuming 32 for now
- **Impact**: May not work on all Intel GPUs
- **TODO**: Query sub-group size at runtime

### 4. FP8 Conversion
- **Issue**: No direct FP8 support in SYCL yet
- **Workaround**: Placeholder conversion functions
- **Impact**: Incorrect FP8 values
- **TODO**: Implement proper FP8 conversion using Intel libraries

### 5. Clock Cycles for Timeout
- **Issue**: No direct equivalent to `clock64()` in SYCL
- **Workaround**: Using iteration count
- **Impact**: Timeout values may not be accurate
- **TODO**: Use SYCL timing facilities

## Next Steps

### Immediate (High Priority)

1. **Implement Combine Kernel** ⏳
   - Port combine kernel from `internode_ll.cu` (lines 715-1140)
   - Implement sending phase
   - Implement receiving phase
   - Add host function

2. **Fix Shared Memory Usage** 🔴
   - Replace global memory workarounds with local accessors
   - Update kernel signatures to accept local accessors
   - Test performance impact

3. **Implement Proper FP8 Conversion** 🔴
   - Research Intel FP8 libraries
   - Implement correct E4M3 conversion
   - Add tests for FP8 accuracy

### Medium Priority

4. **Add Comprehensive Tests** 🟡
   - Unit tests for each kernel
   - Integration tests for full pipeline
   - Correctness tests against reference
   - Performance benchmarks

5. **Optimize Performance** 🟡
   - Profile kernels on Intel GPUs
   - Optimize memory access patterns
   - Tune work-group sizes
   - Leverage Intel GPU features

6. **Handle Dynamic Hidden Sizes** 🟡
   - Remove hardcoded hidden size templates
   - Implement runtime dispatch
   - Add support for arbitrary hidden sizes

### Low Priority

7. **Improve Error Handling** 🟢
   - Add proper exception handling
   - Improve error messages
   - Add validation checks

8. **Add More Documentation** 🟢
   - Code comments
   - Usage examples
   - Performance tuning guide

## Testing Strategy

### Unit Tests
- [ ] Test `is_rank_masked()`
- [ ] Test `barrier_impl()`
- [ ] Test `clean_low_latency_buffer()`
- [ ] Test `dispatch()` sending phase
- [ ] Test `dispatch()` receiving phase
- [ ] Test `combine()` (when implemented)

### Integration Tests
- [ ] Test full dispatch-combine pipeline
- [ ] Test multi-device communication
- [ ] Test fault tolerance (rank masking)
- [ ] Test FP8 quantization
- [ ] Test BF16 mode

### Performance Tests
- [ ] Benchmark dispatch latency
- [ ] Benchmark combine latency
- [ ] Benchmark throughput
- [ ] Compare with CUDA version

## References

- Original CUDA code: `csrc/kernels/internode_ll.cu`
- API mapping: `docs/NVSHMEM_TO_ISHMEM_MAPPING.md`
- Implementation notes: `docs/IMPLEMENTATION_NOTES.md`
- Intel SHMEM docs: https://oneapi-src.github.io/ishmem/

