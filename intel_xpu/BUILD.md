# Build Instructions

This document provides detailed instructions for building the Intel XPU Deep EP extension.

## Prerequisites

### Required

1. **Intel oneAPI Base Toolkit** (2024.0 or later)
   - Includes Intel DPC++/SYCL compiler (icpx)
   - Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html

2. **Python** (3.8 or later)
   ```bash
   python --version
   ```

3. **CMake** (3.20 or later)
   ```bash
   cmake --version
   ```

4. **PyTorch** (with Intel XPU support)
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
   ```

5. **pybind11**
   ```bash
   pip install pybind11
   ```

### Optional

6. **Intel SHMEM** (for multi-node communication)
   - Set `ISHMEM_HOME` environment variable to installation path
   - Or install to default location: `/opt/intel/ishmem`

## Quick Start

### Linux

```bash
# 1. Source Intel oneAPI environment
source /opt/intel/oneapi/setvars.sh

# 2. Navigate to intel_xpu directory
cd intel_xpu

# 3. Run build script
./build.sh
```

### Windows

```powershell
# 1. Source Intel oneAPI environment
& "C:\Program Files (x86)\Intel\oneAPI\setvars.ps1"

# 2. Navigate to intel_xpu directory
cd intel_xpu

# 3. Run build script
.\build.ps1
```

## Build Options

### Linux (build.sh)

```bash
./build.sh [OPTIONS]

Options:
  --debug       Build in Debug mode (default: Release)
  --no-ishmem   Disable Intel SHMEM support
  --clean       Clean build directory before building
  --help        Show help message
```

Examples:
```bash
# Debug build
./build.sh --debug

# Release build without Intel SHMEM
./build.sh --no-ishmem

# Clean build
./build.sh --clean
```

### Windows (build.ps1)

```powershell
.\build.ps1 [OPTIONS]

Options:
  -Debug       Build in Debug mode (default: Release)
  -NoIshmem    Disable Intel SHMEM support
  -Clean       Clean build directory before building
  -Help        Show help message
```

Examples:
```powershell
# Debug build
.\build.ps1 -Debug

# Release build without Intel SHMEM
.\build.ps1 -NoIshmem

# Clean build
.\build.ps1 -Clean
```

## Manual Build (Advanced)

If you prefer to build manually:

```bash
# 1. Source Intel oneAPI environment
source /opt/intel/oneapi/setvars.sh

# 2. Create build directory
mkdir -p build
cd build

# 3. Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_ISHMEM=ON \
    -DCMAKE_CXX_COMPILER=icpx

# 4. Build
make -j$(nproc)

# 5. Install Python package
cd ..
pip install -e .
```

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Release | Build type (Release/Debug) |
| `USE_ISHMEM` | ON | Enable Intel SHMEM support |
| `BUILD_TESTS` | OFF | Build test executables |
| `CMAKE_CXX_COMPILER` | icpx | C++ compiler to use |

## Verification

After building, verify the installation:

```bash
# Test module import
python -c "import deep_ep_xpu; print('Success!')"

# Run compilation test
cd tests
python test_compilation.py
```

## Troubleshooting

### Issue: "Intel oneAPI environment not detected"

**Solution**: Source the oneAPI environment:
```bash
# Linux
source /opt/intel/oneapi/setvars.sh

# Windows
& "C:\Program Files (x86)\Intel\oneAPI\setvars.ps1"
```

### Issue: "PyTorch not found"

**Solution**: Install PyTorch with Intel XPU support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/xpu
```

### Issue: "pybind11 not found"

**Solution**: Install pybind11:
```bash
pip install pybind11
```

### Issue: "Intel SHMEM not found"

**Solution**: Either:
1. Install Intel SHMEM and set `ISHMEM_HOME` environment variable
2. Build without Intel SHMEM: `./build.sh --no-ishmem`

### Issue: Build fails with SYCL errors

**Solution**: Make sure you're using the Intel DPC++ compiler:
```bash
export CXX=icpx
./build.sh --clean
```

## Build Output

After a successful build, you should see:

```
intel_xpu/
├── build/                      # CMake build directory
│   ├── compile_commands.json  # Compilation database
│   └── ...
├── deep_ep_xpu/
│   ├── _C.cpython-*.so        # Extension module (Linux)
│   └── _C.*.pyd               # Extension module (Windows)
└── ...
```

## Next Steps

After building successfully:

1. **Run tests**: See [tests/README.md](tests/README.md)
2. **Try examples**: See [examples/README.md](examples/README.md)
3. **Read documentation**: See [README.md](README.md)

