#!/bin/bash

# Build script for Intel XPU Deep EP extension

set -e

echo "========================================="
echo "Building Intel XPU Deep EP Extension"
echo "========================================="
echo ""

# Parse command line arguments
BUILD_TYPE="Release"
USE_ISHMEM="ON"
CLEAN_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --no-ishmem)
            USE_ISHMEM="OFF"
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug       Build in Debug mode (default: Release)"
            echo "  --no-ishmem   Disable Intel SHMEM support"
            echo "  --clean       Clean build directory before building"
            echo "  --help        Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Build configuration:"
echo "  Build type: $BUILD_TYPE"
echo "  Intel SHMEM: $USE_ISHMEM"
echo ""

# Check if Intel oneAPI is sourced
if [ -z "$ONEAPI_ROOT" ]; then
    echo "⚠️  Warning: Intel oneAPI environment not detected."
    echo "Please source the oneAPI environment first:"
    echo "  source /opt/intel/oneapi/setvars.sh"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ Intel oneAPI environment detected: $ONEAPI_ROOT"
fi

# Check for required tools
echo "Checking for required tools..."

if ! command -v cmake &> /dev/null; then
    echo "Error: CMake not found. Please install CMake 3.20 or later."
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found."
    exit 1
fi

echo "✓ CMake found: $(cmake --version | head -n1)"
echo "✓ Python found: $(python3 --version)"

# Check for Intel compiler
if command -v icpx &> /dev/null; then
    echo "✓ Intel DPC++ compiler found: $(icpx --version | head -n1)"
    export CXX=icpx
elif command -v dpcpp &> /dev/null; then
    echo "✓ Intel DPC++ compiler found: $(dpcpp --version | head -n1)"
    export CXX=dpcpp
else
    echo "Warning: Intel DPC++ compiler (icpx/dpcpp) not found."
    echo "Using default compiler: $CXX"
fi

# Clean previous build
if [ "$CLEAN_BUILD" = true ]; then
    echo ""
    echo "Cleaning previous build..."
    rm -rf build dist *.egg-info
    rm -f deep_ep_xpu/_C*.so deep_ep_xpu/*.pyd
fi

# Create build directory
echo ""
echo "Creating build directory..."
mkdir -p build

# Determine number of parallel jobs
if command -v nproc &> /dev/null; then
    NPROC=$(nproc)
elif command -v sysctl &> /dev/null; then
    NPROC=$(sysctl -n hw.ncpu)
else
    NPROC=4
fi

# Determine compiler to use
if command -v icpx &> /dev/null; then
    CMAKE_CXX_COMPILER="icpx"
    echo "✓ Using Intel DPC++ compiler: icpx"
elif command -v dpcpp &> /dev/null; then
    CMAKE_CXX_COMPILER="dpcpp"
    echo "✓ Using Intel DPC++ compiler: dpcpp"
elif [ -n "$CXX" ]; then
    CMAKE_CXX_COMPILER="$CXX"
    echo "Using compiler from CXX environment variable: $CXX"
else
    CMAKE_CXX_COMPILER="c++"
    echo "⚠️  Warning: Intel DPC++ compiler not found, using default: c++"
fi

# Build with CMake
echo ""
echo "Configuring with CMake..."
cd build
cmake .. \
    -DCMAKE_CXX_COMPILER=$CMAKE_CXX_COMPILER \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DUSE_ISHMEM=$USE_ISHMEM \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo ""
echo "Building with make (using $NPROC parallel jobs)..."
make -j$NPROC VERBOSE=1
cd ..

# Install Python package
echo ""
echo "Installing Python package..."
pip install -e . --no-build-isolation

# Verify installation
echo ""
echo "Verifying installation..."
if python3 -c "import deep_ep_xpu; print('✓ Module imported successfully')" 2>/dev/null; then
    echo "✓ Installation verified"
else
    echo "⚠️  Warning: Module import failed. Check the build output for errors."
fi

echo ""
echo "========================================="
echo "Build completed successfully!"
echo "========================================="
echo ""
echo "Build artifacts:"
echo "  Extension module: deep_ep_xpu/_C*.so"
if [ -f "build/compile_commands.json" ]; then
    echo "  Compile commands: build/compile_commands.json"
fi
echo ""
echo "To test the installation, run:"
echo "  cd tests"
echo "  python test_compilation.py"
echo "  python test_low_latency_xpu.py --num-devices 2"
echo ""

