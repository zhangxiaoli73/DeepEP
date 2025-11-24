# 构建系统总结

## 概述

为 `intel_xpu` 文件夹创建了完整的构建系统，支持 Linux 和 Windows 平台。

## 📁 新增文件

### 1. **CMakeLists.txt** (增强版)
完整的 CMake 配置文件，包含：

**主要功能**:
- ✅ 自动检测 Intel DPC++/SYCL 编译器
- ✅ 自动查找 PyTorch 和 pybind11
- ✅ 可选的 Intel SHMEM 支持
- ✅ 多路径搜索 Intel SHMEM
- ✅ Release/Debug 构建模式
- ✅ 详细的配置信息输出
- ✅ 编译命令导出 (compile_commands.json)

**CMake 选项**:
```cmake
-DCMAKE_BUILD_TYPE=Release    # 或 Debug
-DUSE_ISHMEM=ON               # 启用/禁用 Intel SHMEM
-DBUILD_TESTS=OFF             # 构建测试
```

**关键改进**:
- 智能查找 PyTorch（通过 CMake 或 Python）
- 智能查找 pybind11（通过 CMake 或 Python）
- 多路径搜索 Intel SHMEM（/opt/intel/ishmem, oneapi/ishmem, ~/ishmem）
- 更好的错误消息和警告
- 支持 SOABI 后缀（Python 扩展模块）

### 2. **build.sh** (增强版)
Linux 构建脚本，支持命令行参数：

**用法**:
```bash
./build.sh [OPTIONS]

Options:
  --debug       Debug 模式构建
  --no-ishmem   禁用 Intel SHMEM
  --clean       清理后构建
  --help        显示帮助
```

**功能**:
- ✅ 命令行参数解析
- ✅ Intel oneAPI 环境检测
- ✅ 工具链检测（CMake, Python, icpx）
- ✅ 自动并行构建（使用 nproc）
- ✅ 构建验证（导入测试）
- ✅ 详细的输出信息
- ✅ VERBOSE 构建模式

### 3. **build.ps1** (新增)
Windows PowerShell 构建脚本：

**用法**:
```powershell
.\build.ps1 [OPTIONS]

Options:
  -Debug       Debug 模式构建
  -NoIshmem    禁用 Intel SHMEM
  -Clean       清理后构建
  -Help        显示帮助
```

**功能**:
- ✅ PowerShell 参数支持
- ✅ 彩色输出
- ✅ Intel oneAPI 环境检测
- ✅ 工具链检测
- ✅ 自动并行构建
- ✅ 构建验证
- ✅ 使用 Ninja 构建系统

### 4. **Makefile** (新增)
便捷的 Makefile 包装器：

**用法**:
```bash
make [target]

Targets:
  all         默认构建（Release）
  build       Release 构建
  release     Release 构建
  debug       Debug 构建
  clean       清理构建产物
  rebuild     清理并重新构建
  install     安装 Python 包
  test        运行基础测试
  test-all    运行所有测试
  no-ishmem   不使用 Intel SHMEM 构建
  config      显示构建配置
  help        显示帮助
```

### 5. **BUILD.md** (新增)
详细的构建文档：

**内容**:
- ✅ 前置条件列表
- ✅ 快速开始指南（Linux/Windows）
- ✅ 构建选项说明
- ✅ 手动构建步骤
- ✅ CMake 选项表格
- ✅ 验证步骤
- ✅ 故障排除指南
- ✅ 构建输出说明

### 6. **.gitignore** (新增)
Git 忽略文件：

**忽略内容**:
- 构建目录（build/, dist/）
- Python 缓存（__pycache__/, *.pyc）
- CMake 产物
- IDE 文件
- 扩展模块（*.so, *.pyd）
- 临时文件

### 7. **BUILD_SYSTEM_SUMMARY.md** (本文件)
构建系统总结文档

## 🚀 使用方法

### 最简单的方式（推荐）

**Linux**:
```bash
cd intel_xpu
source /opt/intel/oneapi/setvars.sh
./build.sh
```

**Windows**:
```powershell
cd intel_xpu
& "C:\Program Files (x86)\Intel\oneAPI\setvars.ps1"
.\build.ps1
```

### 使用 Makefile（Linux）

```bash
cd intel_xpu
source /opt/intel/oneapi/setvars.sh
make              # Release 构建
make debug        # Debug 构建
make test         # 运行测试
make clean build  # 清理并重新构建
```

### 手动使用 CMake

```bash
cd intel_xpu
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_ISHMEM=ON
make -j$(nproc)
cd ..
pip install -e .
```

## 📊 构建系统特性

### 跨平台支持
- ✅ Linux (bash script)
- ✅ Windows (PowerShell script)
- ✅ macOS (bash script, 未测试)

### 智能依赖检测
- ✅ 自动查找 PyTorch
- ✅ 自动查找 pybind11
- ✅ 多路径搜索 Intel SHMEM
- ✅ 编译器检测（icpx/dpcpp）

### 构建选项
- ✅ Release/Debug 模式
- ✅ 可选 Intel SHMEM
- ✅ 清理构建
- ✅ 并行构建

### 用户友好
- ✅ 彩色输出（Windows）
- ✅ 详细的错误消息
- ✅ 构建验证
- ✅ 帮助信息
- ✅ 配置摘要

### 开发者友好
- ✅ compile_commands.json 导出
- ✅ VERBOSE 构建模式
- ✅ 头文件包含（IDE 支持）
- ✅ .gitignore 配置

## 🔧 构建流程

1. **环境检测**
   - 检查 Intel oneAPI 环境
   - 检查必需工具（CMake, Python）
   - 检查编译器（icpx/dpcpp）

2. **依赖查找**
   - 查找 PyTorch
   - 查找 pybind11
   - 查找 Intel SHMEM（可选）

3. **CMake 配置**
   - 设置构建类型
   - 设置编译器标志
   - 配置包含目录
   - 配置链接库

4. **编译**
   - 并行编译源文件
   - 链接扩展模块

5. **安装**
   - 安装 Python 包（editable mode）
   - 验证导入

6. **验证**
   - 测试模块导入
   - 显示构建产物位置

## 📝 构建输出

成功构建后的文件结构：

```
intel_xpu/
├── build/
│   ├── _C.cpython-*.so          # 扩展模块（构建目录）
│   ├── compile_commands.json    # 编译命令数据库
│   └── ...
├── deep_ep_xpu/
│   ├── _C.cpython-*.so          # 扩展模块（安装位置）
│   └── ...
└── ...
```

## ⚠️ 常见问题

### 问题 1: "Intel oneAPI environment not detected"
**解决**: 运行 setvars 脚本
```bash
source /opt/intel/oneapi/setvars.sh
```

### 问题 2: "PyTorch not found"
**解决**: 安装 PyTorch
```bash
pip install torch --index-url https://download.pytorch.org/whl/xpu
```

### 问题 3: "Intel SHMEM not found"
**解决**: 
- 选项 1: 安装 Intel SHMEM 并设置 `ISHMEM_HOME`
- 选项 2: 禁用 Intel SHMEM: `./build.sh --no-ishmem`

## ✅ 验证构建

```bash
# 测试导入
python -c "import deep_ep_xpu; print('Success!')"

# 运行测试
cd tests
python test_compilation.py
```

## 🎯 下一步

1. 运行构建脚本
2. 验证安装
3. 运行测试
4. 查看示例代码

详细信息请参阅 [BUILD.md](BUILD.md)

