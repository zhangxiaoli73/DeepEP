# Changelog - Intel XPU Implementation

## 2024-11-20 - Intel SHMEM Integration

### Added

#### Intel SHMEM Support
- **ishmem_utils.hpp**: New header file with Intel SHMEM wrapper functions
  - `ishmem_get_p2p_ptr()`: P2P pointer access (replaces `nvshmemi_get_p2p_ptr`)
  - `ishmem_put_nbi_work_group()`: Non-blocking put (replaces `nvshmemi_ibgda_put_nbi_warp`)
  - `ishmem_atomic_add_nonfetch_work_group()`: Atomic add (replaces `nvshmemi_ibgda_amo_nonfetch_add`)
  - `ishmem_int_p_work_group()`: Remote put (replaces `nvshmemi_ibgda_rma_p`)
  - `ishmem_quiet_work_group()`: Quiet operation (replaces `nvshmemi_ibgda_quiet`)
  - `ishmem_barrier_all_work_group()`: Barrier (replaces `nvshmemx_barrier_all_block`)
  - Helper functions for atomic operations and rank masking

#### Documentation
- **NVSHMEM_TO_ISHMEM_MAPPING.md**: Complete API mapping guide
  - Detailed comparison of NVSHMEM and Intel SHMEM APIs
  - Code examples for each API pair
  - Performance considerations
  - Best practices

- **IMPLEMENTATION_NOTES.md**: Implementation details
  - CUDA to SYCL mapping
  - NVSHMEM to Intel SHMEM mapping
  - Design decisions and rationale
  - Performance considerations
  - Known limitations and future improvements

### Modified

#### Build System
- **CMakeLists.txt**:
  - Added Intel SHMEM library detection
  - Added ISHMEM_HOME environment variable support
  - Added Intel SHMEM include directories
  - Added Intel SHMEM library linking

#### Installation
- **INSTALL.md**:
  - Added Intel SHMEM installation instructions
  - Added environment variable setup for Intel SHMEM
  - Updated system requirements

#### Documentation
- **README.md**:
  - Added Intel SHMEM integration to features list
  - Updated description to mention NVSHMEM API compatibility

- **README_CN.md**:
  - Added Intel SHMEM integration description (Chinese)
  - Updated system requirements (Chinese)

### Technical Details

#### API Replacement Strategy

The implementation follows a **conservative replacement** approach:

1. **Preserve Original Logic**: All algorithmic logic from the CUDA version is preserved
2. **API-Only Changes**: Only CUDA/NVSHMEM APIs are replaced with SYCL/Intel SHMEM equivalents
3. **No Algorithmic Changes**: No changes to dispatch/combine algorithms, buffer layouts, or communication patterns

#### Key Mappings

| Original (NVSHMEM) | Replacement (Intel SHMEM) | Purpose |
|--------------------|---------------------------|---------|
| `nvshmemi_get_p2p_ptr` | `ishmem_ptr` | P2P pointer access |
| `nvshmemi_ibgda_put_nbi_warp` | `ishmemx_putmem_nbi_work_group` | Non-blocking put |
| `nvshmemi_ibgda_amo_nonfetch_add` | `ishmemx_int_atomic_add_work_group` | Atomic add |
| `nvshmemi_ibgda_rma_p` | `ishmemx_int_p_work_group` | Remote put |
| `nvshmemi_ibgda_quiet` | `ishmemx_quiet_work_group` | Wait for completion |
| `nvshmemx_barrier_all_block` | `ishmemx_barrier_all_work_group` | Barrier |

#### Execution Model Mapping

| CUDA | SYCL | Notes |
|------|------|-------|
| Warp (32 threads) | Sub-group (16-32 items) | Hardware-dependent |
| Thread Block | Work-group | Group of work-items |
| `__syncwarp()` | `group_barrier(sub_group)` | Sub-group sync |
| `__syncthreads()` | `group_barrier(work_group)` | Work-group sync |

### Dependencies

#### New Dependencies
- **Intel SHMEM (ishmem)**: Required for multi-device communication
  - Repository: https://github.com/oneapi-src/ishmem
  - Version: Latest stable release
  - License: BSD-3-Clause

#### Existing Dependencies (Unchanged)
- Intel oneAPI Base Toolkit (2024.0+)
- Intel Extension for PyTorch (2.0+)
- PyTorch (2.0+)
- Python (3.8+)
- CMake (3.20+)

### Compatibility

#### Hardware Compatibility
- Intel Data Center GPU Max Series (PVC)
- Intel Arc GPUs
- Future Intel Xe architecture GPUs

#### Software Compatibility
- Linux (Ubuntu 20.04+, CentOS 8+)
- Intel oneAPI 2024.0 or later
- Level Zero drivers

### Testing

#### Test Coverage
- Unit tests for individual kernels
- Integration tests for dispatch-combine pipeline
- Multi-device communication tests
- Correctness verification against reference implementation

#### Known Issues
- Intel SHMEM is newer than NVSHMEM, may have different performance characteristics
- Requires recent Intel GPU drivers with Level Zero support
- Multi-node testing requires proper InfiniBand/network configuration

### Migration Guide

For users migrating from the CUDA version:

1. **Install Intel SHMEM**: Follow instructions in INSTALL.md
2. **Update Environment**: Set ISHMEM_HOME and LD_LIBRARY_PATH
3. **Build**: Use provided build.sh script
4. **Test**: Run test suite to verify functionality
5. **Benchmark**: Compare performance with CUDA version

### Performance Notes

- Intel SHMEM provides similar abstractions to NVSHMEM
- Performance depends on hardware (NVLink vs Xe Link)
- P2P access preferred when available
- Non-blocking operations enable communication-computation overlap

### Future Work

1. **Performance Optimization**: Profile and optimize hot paths
2. **Advanced Features**: Implement additional NVSHMEM features as needed
3. **Multi-Node**: Test and optimize multi-node configurations
4. **Documentation**: Expand examples and tutorials
5. **CI/CD**: Set up continuous integration on Intel hardware

### Contributors

- Intel XPU Team
- Based on original DeepEP implementation

### License

Same as parent DeepEP project.

