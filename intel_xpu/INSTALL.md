# Installation Guide for Intel XPU Deep EP

This guide provides detailed instructions for building and installing the Intel XPU Deep EP library.

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+, CentOS 8+, or similar)
- **Hardware**: Intel Data Center GPU Max Series (PVC), Intel Arc GPUs, or compatible Intel XPU devices
- **Memory**: At least 16GB RAM recommended
- **Storage**: At least 10GB free disk space

### Software Requirements

1. **Intel oneAPI Base Toolkit** (2024.0 or later)
   - Includes Intel DPC++/SYCL compiler
   - Includes Intel MKL and other libraries

2. **Intel SHMEM** (ishmem)
   - Required for RDMA-like communication between devices
   - Provides OpenSHMEM API for Intel GPUs
   - See: https://github.com/oneapi-src/ishmem

3. **Python** 3.8 or later

4. **CMake** 3.20 or later

5. **PyTorch** 2.0 or later

6. **Intel Extension for PyTorch** 2.0 or later

## Installation Steps

### Step 1: Install Intel oneAPI Base Toolkit

Download and install from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html

```bash
# Example for Ubuntu/Debian
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/...
sudo sh ./l_BaseKit_*.sh

# Or use APT repository
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-basekit
```

### Step 2: Install Intel SHMEM

```bash
# Clone Intel SHMEM repository
git clone https://github.com/oneapi-src/ishmem.git
cd ishmem

# Build and install
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/intel/ishmem
make -j$(nproc)
sudo make install

# Add to environment
export ISHMEM_HOME=/opt/intel/ishmem
export LD_LIBRARY_PATH=$ISHMEM_HOME/lib:$LD_LIBRARY_PATH
```

### Step 3: Source oneAPI Environment

```bash
source /opt/intel/oneapi/setvars.sh
```

Add this to your `~/.bashrc` for automatic setup:
```bash
echo "source /opt/intel/oneapi/setvars.sh" >> ~/.bashrc
echo "export ISHMEM_HOME=/opt/intel/ishmem" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$ISHMEM_HOME/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
```

### Step 3: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with XPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# Install Intel Extension for PyTorch
pip install intel-extension-for-pytorch

# Install other dependencies
pip install -r requirements.txt
```

### Step 4: Build the Extension

#### Option A: Using the build script (recommended)

```bash
chmod +x build.sh
./build.sh
```

#### Option B: Manual build with CMake

```bash
# Clean previous builds
rm -rf build dist *.egg-info

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx

# Build
make -j$(nproc)

# Install Python package
cd ..
pip install -e .
```

#### Option C: Using setup.py

```bash
python setup.py install
```

### Step 5: Verify Installation

```bash
# Test import
python -c "import deep_ep_xpu; print('✓ Installation successful!')"

# Check device availability
python -c "import torch; print(f'XPU available: {torch.xpu.is_available()}')"
python -c "import torch; print(f'XPU devices: {torch.xpu.device_count()}')"
```

## Running Tests

### Single Device Test

```bash
cd tests
python test_low_latency_xpu.py --num-devices 1 --num-tokens 16 --hidden 4096
```

### Multi-Device Test

```bash
# Using Intel MPI or torchrun
mpirun -n 2 python test_low_latency_xpu.py --num-devices 2

# Or with torchrun
torchrun --nproc_per_node=2 test_low_latency_xpu.py --num-devices 2
```

### Run Example

```bash
cd examples
python simple_moe.py
```

## Troubleshooting

### Issue: "Intel DPC++ compiler not found"

**Solution**: Make sure you've sourced the oneAPI environment:
```bash
source /opt/intel/oneapi/setvars.sh
```

### Issue: "XPU device not available"

**Solution**: 
1. Check if Intel GPU drivers are installed:
   ```bash
   sudo apt install intel-level-zero-gpu level-zero
   ```

2. Verify device is detected:
   ```bash
   sycl-ls
   ```

### Issue: "CMake version too old"

**Solution**: Install newer CMake:
```bash
pip install cmake --upgrade
```

### Issue: "PyTorch XPU support not found"

**Solution**: Install PyTorch with XPU support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/xpu
pip install intel-extension-for-pytorch
```

### Issue: Build fails with linking errors

**Solution**: Make sure all oneAPI components are properly sourced:
```bash
source /opt/intel/oneapi/setvars.sh --force
```

## Environment Variables

Useful environment variables for debugging and optimization:

```bash
# Enable SYCL debugging
export SYCL_PI_TRACE=1

# Set device selection
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

# Enable verbose output
export SYCL_DEVICE_FILTER=level_zero:gpu

# Set number of threads
export OMP_NUM_THREADS=8
```

## Performance Tuning

For optimal performance:

1. **Use appropriate batch sizes**: Larger batches typically achieve better throughput
2. **Enable FP8 quantization**: Use `use_fp8=True` for lower latency
3. **Tune buffer sizes**: Adjust based on your workload
4. **Use multiple devices**: Scale across multiple XPUs for larger models

## Uninstallation

```bash
pip uninstall deep_ep_xpu
rm -rf build dist *.egg-info
```

## Support

For issues and questions:
- Check the main DeepEP repository
- Intel oneAPI forums: https://community.intel.com/
- Intel Extension for PyTorch: https://github.com/intel/intel-extension-for-pytorch

