# Intel XPU 低延迟 MoE 通信库

[English](README.md) | 简体中文

## 概述

这是一个专为 Intel XPU（数据中心 GPU Max 系列、Arc GPU 等）设计的低延迟混合专家（MoE）通信库，使用 SYCL 实现。该库提供了优化的 GPU 内核，用于 MoE 的 dispatch 和 combine 操作，支持训练和推理场景。

## 主要特性

- ✅ **低延迟内核**: 使用纯 RDMA 风格通信，针对推理解码优化
- ✅ **多精度支持**: BF16 和 FP8 E4M3 操作
- ✅ **异步执行**: 基于事件的同步机制，支持计算与通信重叠
- ✅ **可扩展**: 支持多设备和多节点配置
- ✅ **SYCL 实现**: 可移植到 Intel GPU 和加速器
- ✅ **Intel SHMEM 集成**: 使用 Intel SHMEM (ishmem) 进行 RDMA 通信，与 CUDA 版本的 NVSHMEM API 兼容

## 性能特点

针对以下场景优化：
- **Intel 数据中心 GPU Max 系列 (PVC)**
- **Intel Arc GPU**
- **未来的 Intel Xe 架构**

预期性能特征：
- 设备内：通过共享内存实现高带宽
- 设备间：通过 Intel Xe Link 或 PCIe 优化
- 多节点：通过 Intel MPI 和 CCL 可扩展

## 快速开始

### 安装

```bash
# 1. 加载 Intel oneAPI 环境
source /opt/intel/oneapi/setvars.sh

# 2. 安装依赖
pip install torch intel-extension-for-pytorch

# 3. 构建和安装
cd intel_xpu
./build.sh
```

### 基本使用

```python
import torch
import deep_ep_xpu
from deep_ep_xpu.buffer import create_buffer

# 创建缓冲区
buffer = create_buffer(
    process_group=dist.group.WORLD,
    buffer_size=int(2e9),
    rdma_buffer_size=int(1e9),
    low_latency_mode=True
)

# 准备数据
num_tokens, hidden, num_experts, num_topk = 32, 4096, 64, 6
hidden_states = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device='xpu')
topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk), device='xpu')
topk_weights = torch.softmax(torch.randn(num_tokens, num_topk, device='xpu'), dim=-1)

# Dispatch 令牌到专家
recv_hidden_states, recv_count, handle, event, hook = buffer.low_latency_dispatch(
    hidden_states, topk_idx, None, None,
    num_max_dispatch_tokens_per_rank=num_tokens,
    num_experts=num_experts,
    use_fp8=True
)

# 使用专家处理（您的专家网络）
expert_output = process_experts(recv_hidden_states)

# Combine 专家输出
combined_output, event, hook = buffer.low_latency_combine(
    expert_output, topk_idx, topk_weights,
    handle[0], handle[1], None,
    num_max_dispatch_tokens_per_rank=num_tokens,
    num_experts=num_experts
)
```

## 目录结构

```
intel_xpu/
├── README.md                          # 主文档
├── README_CN.md                       # 中文文档（本文件）
├── INSTALL.md                         # 安装指南
├── QUICKSTART.md                      # 快速开始指南
├── csrc/                              # C++/SYCL 源代码
│   ├── kernels/                       # SYCL 内核实现
│   │   ├── low_latency_dispatch.cpp   # Dispatch 内核
│   │   ├── low_latency_combine.cpp    # Combine 内核
│   │   ├── config.hpp                 # 配置
│   │   └── utils.hpp                  # 工具函数
│   ├── deep_ep_xpu.hpp/cpp            # 主接口
│   └── python_bindings.cpp            # Python 绑定
├── deep_ep_xpu/                       # Python 包
│   ├── __init__.py
│   ├── buffer.py                      # 缓冲区管理
│   └── utils.py                       # 工具函数
├── tests/                             # 测试套件
│   ├── test_low_latency_xpu.py        # 主测试文件
│   └── utils.py                       # 测试工具
└── examples/                          # 示例代码
    └── simple_moe.py                  # 简单 MoE 示例
```

## 系统要求

- **操作系统**: Linux (Ubuntu 20.04+, CentOS 8+ 或类似系统)
- **硬件**: Intel 数据中心 GPU Max 系列 (PVC)、Intel Arc GPU 或兼容的 Intel XPU 设备
- **软件**:
  - Intel oneAPI Base Toolkit (2024.0 或更高版本)
  - Intel SHMEM (ishmem) - 用于设备间 RDMA 通信
  - Python 3.8+
  - CMake 3.20+
  - PyTorch 2.0+
  - Intel Extension for PyTorch 2.0+

## 运行测试

```bash
# 单设备测试
cd tests
python test_low_latency_xpu.py --num-devices 1

# 多设备测试
mpirun -n 2 python test_low_latency_xpu.py --num-devices 2
```

## 运行示例

```bash
cd examples
python simple_moe.py
```

## 性能优化建议

1. **推理使用 FP8**: 设置 `use_fp8=True` 可降低延迟
2. **批处理令牌**: 更大的批次可提高吞吐量
3. **调整缓冲区大小**: 根据工作负载匹配
4. **使用多设备**: 水平扩展以处理更大的模型

## 文档

- [README.md](README.md) - 英文主文档
- [INSTALL.md](INSTALL.md) - 详细安装说明
- [QUICKSTART.md](QUICKSTART.md) - 快速开始指南
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - 项目结构说明
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - 实现总结

## 故障排除

### 问题：导入错误

```bash
# 确保已加载 oneAPI 环境
source /opt/intel/oneapi/setvars.sh
```

### 问题：没有 XPU 设备

```bash
# 检查设备
python -c "import torch; print(torch.xpu.device_count())"

# 安装驱动
sudo apt install intel-level-zero-gpu level-zero
```

### 问题：构建失败

```bash
# 清理并重新构建
rm -rf build
./build.sh
```

更多帮助，请参阅 [INSTALL.md](INSTALL.md#troubleshooting)。

## 与 CUDA 版本的对比

| 特性 | CUDA (原版) | SYCL (Intel XPU) |
|------|------------|------------------|
| 语言 | CUDA C++ | SYCL C++ |
| 通信 | NVSHMEM | Intel CCL / USM |
| 精度 | FP8, BF16 | FP8, BF16 |
| 异步 | CUDA Events | SYCL Events |
| 可移植性 | NVIDIA GPU | Intel XPU, CPU |

## 许可证

与父项目 DeepEP 相同。

## 贡献

欢迎贡献！请参阅主 DeepEP 仓库了解贡献指南。

## 支持

如有问题和疑问：
- 查看主 DeepEP 仓库
- Intel oneAPI 论坛: https://community.intel.com/
- Intel Extension for PyTorch: https://github.com/intel/intel-extension-for-pytorch

