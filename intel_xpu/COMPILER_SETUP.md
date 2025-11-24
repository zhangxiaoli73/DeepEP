# 编译器设置指南

## 问题：CMake 使用了错误的编译器

如果您看到类似这样的警告：
```
Intel DPC++/SYCL compiler (icpx/dpcpp) recommended. Current: /usr/bin/c++
To use Intel compiler, set: export CXX=icpx
```

这是因为 CMake 缓存了之前的编译器选择。

## 解决方法

### 方法 1: 清理构建后重新构建（最简单）

```bash
cd intel_xpu

# 清理构建
./build.sh --clean

# 构建脚本会自动检测并使用 icpx
```

**说明**: 更新后的 `build.sh` 脚本会自动检测 `icpx` 或 `dpcpp` 编译器，无需手动设置。

### 方法 2: 手动删除构建目录

```bash
cd intel_xpu

# 删除构建目录
rm -rf build

# 重新构建（脚本会自动使用 icpx）
./build.sh
```

### 方法 3: 显式设置编译器环境变量

```bash
cd intel_xpu

# 设置编译器
export CXX=icpx

# 清理并构建
rm -rf build
./build.sh
```

### 方法 4: 手动使用 CMake（高级）

```bash
cd intel_xpu

# 删除旧的构建
rm -rf build
mkdir build
cd build

# 显式指定编译器
cmake .. \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_ISHMEM=ON

make -j$(nproc)
cd ..
pip install -e .
```

## Windows (PowerShell)

### 方法 1: 清理构建（推荐）

```powershell
cd intel_xpu

# 清理构建
.\build.ps1 -Clean

# 脚本会自动检测并使用 icpx
```

### 方法 2: 手动删除构建目录

```powershell
cd intel_xpu

# 删除构建目录
Remove-Item -Recurse -Force build

# 重新构建
.\build.ps1
```

## 编译器检测优先级

更新后的构建脚本会按以下优先级自动检测编译器：

1. **icpx** - Intel DPC++ 编译器（首选）
2. **dpcpp** - Intel DPC++ 编译器（旧版本）
3. **$CXX** - 环境变量指定的编译器
4. **c++** - 系统默认编译器（不推荐）

## 验证编译器

构建时，脚本会显示使用的编译器：

```
✓ Using Intel DPC++ compiler: icpx
```

或者手动检查：

```bash
# Linux/macOS
which icpx
icpx --version

# Windows
where icpx
icpx --version
```

## 确保 Intel oneAPI 环境已加载

在构建之前，确保已经加载 Intel oneAPI 环境：

### Linux/macOS
```bash
source /opt/intel/oneapi/setvars.sh
```

### Windows
```powershell
& "C:\Program Files (x86)\Intel\oneAPI\setvars.ps1"
```

## 常见问题

### Q: 为什么设置了 `export CXX=icpx` 还是用 `/usr/bin/c++`？

**A**: CMake 会缓存编译器选择。解决方法：
1. 删除 `build` 目录
2. 重新运行构建脚本

### Q: 如何确认使用了正确的编译器？

**A**: 查看 CMake 配置输出：
```
-- The CXX compiler identification is IntelLLVM 2024.0.0
-- Check for working CXX compiler: /opt/intel/oneapi/compiler/latest/bin/icpx
```

或者查看构建脚本输出：
```
✓ Using Intel DPC++ compiler: icpx
```

### Q: 可以使用其他编译器吗？

**A**: 理论上可以，但**强烈推荐使用 Intel DPC++ 编译器 (icpx)**，因为：
- 需要 SYCL 支持
- 需要 Intel XPU 运行时支持
- 其他编译器可能无法正确编译 SYCL 代码

### Q: 构建脚本更新后有什么变化？

**A**: 
- ✅ 自动检测 `icpx` 或 `dpcpp` 编译器
- ✅ 自动传递 `-DCMAKE_CXX_COMPILER` 给 CMake
- ✅ 显示使用的编译器
- ✅ 无需手动设置 `CXX` 环境变量

## 总结

**最简单的方法**：
```bash
cd intel_xpu
./build.sh --clean
```

构建脚本会自动处理一切！

