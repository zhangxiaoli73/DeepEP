# Intel XPU Implementation Summary

## 项目概述 (Project Overview)

本项目是 DeepEP 的 Intel XPU 移植版本，使用 SYCL 和 Intel SHMEM 实现了与原始 CUDA/NVSHMEM 版本相同的功能。

This project is an Intel XPU port of DeepEP, implementing the same functionality as the original CUDA/NVSHMEM version using SYCL and Intel SHMEM.

## 核心原则 (Core Principles)

### 1. 保持原有逻辑 (Preserve Original Logic)
- ✅ 不改变 CUDA 代码的算法逻辑
- ✅ 保持相同的数据布局和通信模式
- ✅ 维护相同的性能特征

### 2. 仅替换 API (API Replacement Only)
- ✅ CUDA → SYCL
- ✅ NVSHMEM → Intel SHMEM
- ✅ 保持函数签名和语义一致

### 3. 完整的文档 (Complete Documentation)
- ✅ API 映射文档
- ✅ 实现说明
- ✅ 安装指南
- ✅ 快速开始指南

## 主要组件 (Main Components)

### 1. SYCL 内核 (SYCL Kernels)
```
csrc/kernels/
├── config.hpp                  # 配置和数据结构
├── utils.hpp                   # 工具函数 (BF16/FP8 转换)
├── ishmem_utils.hpp           # Intel SHMEM 包装函数 (新增)
├── low_latency_dispatch.cpp   # Dispatch 内核
└── low_latency_combine.cpp    # Combine 内核
```

### 2. Intel SHMEM 集成 (Intel SHMEM Integration)
```cpp
// 新增的 Intel SHMEM 包装函数
ishmem_get_p2p_ptr()                    // P2P 指针访问
ishmem_put_nbi_work_group()             // 非阻塞 put
ishmem_atomic_add_nonfetch_work_group() // 原子加法
ishmem_int_p_work_group()               // 远程 put
ishmem_quiet_work_group()               // 等待完成
ishmem_barrier_all_work_group()         // 屏障同步
```

### 3. 文档 (Documentation)
```
docs/
├── NVSHMEM_TO_ISHMEM_MAPPING.md  # API 映射指南
└── IMPLEMENTATION_NOTES.md        # 实现细节
```

## API 映射 (API Mapping)

### NVSHMEM → Intel SHMEM

| NVSHMEM | Intel SHMEM | 用途 |
|---------|-------------|------|
| `nvshmemi_get_p2p_ptr` | `ishmem_ptr` | P2P 指针 |
| `nvshmemi_ibgda_put_nbi_warp` | `ishmemx_putmem_nbi_work_group` | 非阻塞 put |
| `nvshmemi_ibgda_amo_nonfetch_add` | `ishmemx_int_atomic_add_work_group` | 原子加法 |
| `nvshmemi_ibgda_rma_p` | `ishmemx_int_p_work_group` | 远程 put |
| `nvshmemi_ibgda_quiet` | `ishmemx_quiet_work_group` | 等待完成 |
| `nvshmemx_barrier_all_block` | `ishmemx_barrier_all_work_group` | 屏障 |

### CUDA → SYCL

| CUDA | SYCL | 说明 |
|------|------|------|
| Warp | Sub-group | 硬件相关大小 |
| Thread Block | Work-group | 工作组 |
| `__syncwarp()` | `group_barrier(sg)` | 子组同步 |
| `__syncthreads()` | `group_barrier(wg)` | 工作组同步 |
| `cudaMalloc` | `sycl::malloc_device` | 设备内存分配 |

## 文件清单 (File List)

### 新增文件 (New Files)
1. **csrc/kernels/ishmem_utils.hpp** - Intel SHMEM 包装函数
2. **docs/NVSHMEM_TO_ISHMEM_MAPPING.md** - API 映射文档
3. **docs/IMPLEMENTATION_NOTES.md** - 实现说明
4. **CHANGELOG.md** - 变更日志
5. **SUMMARY.md** - 本文件

### 修改文件 (Modified Files)
1. **CMakeLists.txt** - 添加 Intel SHMEM 支持
2. **INSTALL.md** - 添加 Intel SHMEM 安装说明
3. **README.md** - 添加 Intel SHMEM 功能说明
4. **README_CN.md** - 添加中文说明

### 现有文件 (Existing Files - Unchanged)
- csrc/deep_ep_xpu.hpp/cpp
- csrc/python_bindings.cpp
- csrc/kernels/config.hpp
- csrc/kernels/utils.hpp
- csrc/kernels/low_latency_dispatch.cpp
- csrc/kernels/low_latency_combine.cpp
- deep_ep_xpu/*.py
- tests/*.py
- examples/*.py

## 依赖项 (Dependencies)

### 新增依赖 (New Dependencies)
- **Intel SHMEM (ishmem)** - RDMA 通信库
  - 仓库: https://github.com/oneapi-src/ishmem
  - 许可: BSD-3-Clause

### 现有依赖 (Existing Dependencies)
- Intel oneAPI Base Toolkit (2024.0+)
- Intel Extension for PyTorch (2.0+)
- PyTorch (2.0+)
- Python (3.8+)
- CMake (3.20+)

## 安装步骤 (Installation Steps)

```bash
# 1. 安装 Intel SHMEM
git clone https://github.com/oneapi-src/ishmem.git
cd ishmem && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/intel/ishmem
make -j$(nproc) && sudo make install

# 2. 设置环境变量
export ISHMEM_HOME=/opt/intel/ishmem
export LD_LIBRARY_PATH=$ISHMEM_HOME/lib:$LD_LIBRARY_PATH
source /opt/intel/oneapi/setvars.sh

# 3. 构建项目
cd intel_xpu
./build.sh

# 4. 测试
cd tests
python test_low_latency_xpu.py --num-devices 2
```

## 使用示例 (Usage Example)

```python
import deep_ep_xpu
from deep_ep_xpu.buffer import create_buffer

# 创建缓冲区 (使用 Intel SHMEM)
buffer = create_buffer(
    process_group=dist.group.WORLD,
    low_latency_mode=True
)

# Dispatch (使用 Intel SHMEM 进行通信)
recv_data, recv_count, handle, event, hook = \
    buffer.low_latency_dispatch(
        hidden_states, topk_idx, None, None,
        num_max_dispatch_tokens_per_rank=num_tokens,
        num_experts=num_experts,
        use_fp8=True
    )

# Combine (使用 Intel SHMEM 进行通信)
output, event, hook = buffer.low_latency_combine(
    expert_output, topk_idx, topk_weights,
    handle[0], handle[1], None,
    num_max_dispatch_tokens_per_rank=num_tokens,
    num_experts=num_experts
)
```

## 性能特点 (Performance Characteristics)

- ✅ 低延迟通信 (使用 Intel SHMEM)
- ✅ P2P 直接访问 (当可用时)
- ✅ 非阻塞操作 (重叠计算和通信)
- ✅ FP8 量化 (降低延迟)
- ✅ 双缓冲 (提高吞吐量)

## 测试覆盖 (Test Coverage)

- ✅ 单元测试 (内核正确性)
- ✅ 集成测试 (端到端流程)
- ✅ 多设备测试 (通信验证)
- ✅ 多精度测试 (BF16, FP8)
- ✅ 性能测试 (基准测试)

## 已知限制 (Known Limitations)

1. Intel SHMEM 相对较新，可能有不同的性能特征
2. 需要 Intel Data Center GPU Max Series 或更新的硬件
3. 需要最新的 Intel GPU 驱动和 Level Zero 支持
4. 多节点配置需要正确的网络设置

## 未来改进 (Future Improvements)

1. 性能优化和调优
2. 更多的 NVSHMEM 功能支持
3. 多节点配置的测试和优化
4. 更详细的性能分析工具
5. 自动调优功能

## 参考资料 (References)

- [Intel SHMEM 文档](https://oneapi-src.github.io/ishmem/)
- [SYCL 规范](https://www.khronos.org/sycl/)
- [Intel oneAPI 编程指南](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/)
- [DeepEP 原始实现](../csrc/)

## 联系方式 (Contact)

如有问题或建议，请参考主 DeepEP 仓库或 Intel oneAPI 论坛。

For questions or suggestions, please refer to the main DeepEP repository or Intel oneAPI forums.

