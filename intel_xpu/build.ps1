# Build script for Intel XPU Deep EP extension (Windows PowerShell)

param(
    [switch]$Debug,
    [switch]$NoIshmem,
    [switch]$Clean,
    [switch]$Help
)

if ($Help) {
    Write-Host "Usage: .\build.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Debug       Build in Debug mode (default: Release)"
    Write-Host "  -NoIshmem    Disable Intel SHMEM support"
    Write-Host "  -Clean       Clean build directory before building"
    Write-Host "  -Help        Show this help message"
    Write-Host ""
    exit 0
}

$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Building Intel XPU Deep EP Extension" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Determine build configuration
$BuildType = if ($Debug) { "Debug" } else { "Release" }
$UseIshmem = if ($NoIshmem) { "OFF" } else { "ON" }

Write-Host "Build configuration:"
Write-Host "  Build type: $BuildType"
Write-Host "  Intel SHMEM: $UseIshmem"
Write-Host ""

# Check if Intel oneAPI is sourced
if (-not $env:ONEAPI_ROOT) {
    Write-Host "Warning: Intel oneAPI environment not detected." -ForegroundColor Yellow
    Write-Host "Please run the oneAPI environment setup first:" -ForegroundColor Yellow
    Write-Host '  & "C:\Program Files (x86)\Intel\oneAPI\setvars.ps1"' -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Continue anyway? (y/n)"
    if ($response -ne 'y' -and $response -ne 'Y') {
        exit 1
    }
} else {
    Write-Host "✓ Intel oneAPI environment detected: $env:ONEAPI_ROOT" -ForegroundColor Green
}

# Check for required tools
Write-Host ""
Write-Host "Checking for required tools..."

try {
    $cmakeVersion = cmake --version | Select-Object -First 1
    Write-Host "✓ CMake found: $cmakeVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Error: CMake not found. Please install CMake 3.20 or later." -ForegroundColor Red
    exit 1
}

try {
    $pythonVersion = python --version
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Error: Python not found." -ForegroundColor Red
    exit 1
}

# Determine compiler to use
$CmakeCxxCompiler = $null
try {
    $icpxVersion = icpx --version | Select-Object -First 1
    Write-Host "✓ Intel DPC++ compiler found: $icpxVersion" -ForegroundColor Green
    $CmakeCxxCompiler = "icpx"
} catch {
    try {
        $dpcppVersion = dpcpp --version | Select-Object -First 1
        Write-Host "✓ Intel DPC++ compiler found: $dpcppVersion" -ForegroundColor Green
        $CmakeCxxCompiler = "dpcpp"
    } catch {
        if ($env:CXX) {
            Write-Host "⚠ Warning: Intel DPC++ compiler not found." -ForegroundColor Yellow
            Write-Host "Using compiler from CXX environment variable: $env:CXX" -ForegroundColor Yellow
            $CmakeCxxCompiler = $env:CXX
        } else {
            Write-Host "⚠ Warning: Intel DPC++ compiler not found." -ForegroundColor Yellow
            Write-Host "Using default compiler" -ForegroundColor Yellow
        }
    }
}

# Clean previous build
if ($Clean) {
    Write-Host ""
    Write-Host "Cleaning previous build..."
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
    if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
    if (Test-Path "*.egg-info") { Remove-Item -Recurse -Force "*.egg-info" }
    Get-ChildItem -Path "deep_ep_xpu" -Filter "_C*.pyd" | Remove-Item -Force
}

# Create build directory
Write-Host ""
Write-Host "Creating build directory..."
New-Item -ItemType Directory -Force -Path "build" | Out-Null

# Determine number of parallel jobs
$NumProcs = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
if (-not $NumProcs) { $NumProcs = 4 }

# Build with CMake
Write-Host ""
Write-Host "Configuring with CMake..."
Push-Location build
try {
    $cmakeArgs = @(
        ".."
        "-DCMAKE_BUILD_TYPE=$BuildType"
        "-DUSE_ISHMEM=$UseIshmem"
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
        "-G", "Ninja"
    )

    # Add compiler specification if found
    if ($CmakeCxxCompiler) {
        $cmakeArgs += "-DCMAKE_CXX_COMPILER=$CmakeCxxCompiler"
    }

    cmake @cmakeArgs
    
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed"
    }

    Write-Host ""
    Write-Host "Building with Ninja (using $NumProcs parallel jobs)..."
    cmake --build . --parallel $NumProcs --verbose
    
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed"
    }
} finally {
    Pop-Location
}

# Install Python package
Write-Host ""
Write-Host "Installing Python package..."
pip install -e . --no-build-isolation

# Verify installation
Write-Host ""
Write-Host "Verifying installation..."
try {
    python -c "import deep_ep_xpu; print('✓ Module imported successfully')"
    Write-Host "✓ Installation verified" -ForegroundColor Green
} catch {
    Write-Host "⚠ Warning: Module import failed. Check the build output for errors." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Build completed successfully!" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Build artifacts:"
Write-Host "  Extension module: deep_ep_xpu\_C*.pyd"
if (Test-Path "build\compile_commands.json") {
    Write-Host "  Compile commands: build\compile_commands.json"
}
Write-Host ""
Write-Host "To test the installation, run:"
Write-Host "  cd tests"
Write-Host "  python test_compilation.py"
Write-Host "  python test_low_latency_xpu.py --num-devices 2"
Write-Host ""

