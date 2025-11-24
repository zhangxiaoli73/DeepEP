# Makefile for Intel XPU Deep EP extension
# This is a convenience wrapper around CMake

.PHONY: all build clean install test help debug release

# Default target
all: build

# Build in release mode (default)
build: release

# Release build
release:
	@echo "Building in Release mode..."
	@bash build.sh

# Debug build
debug:
	@echo "Building in Debug mode..."
	@bash build.sh --debug

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf build dist *.egg-info
	@rm -f deep_ep_xpu/_C*.so deep_ep_xpu/*.pyd
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete."

# Clean and rebuild
rebuild: clean build

# Install Python package
install:
	@echo "Installing Python package..."
	@pip install -e .

# Run tests
test:
	@echo "Running tests..."
	@cd tests && python test_compilation.py

# Run all tests
test-all:
	@echo "Running all tests..."
	@cd tests && python test_compilation.py
	@cd tests && python test_low_latency_xpu.py --num-devices 2

# Build without Intel SHMEM
no-ishmem:
	@echo "Building without Intel SHMEM..."
	@bash build.sh --no-ishmem

# Show build configuration
config:
	@echo "Build Configuration:"
	@echo "  CMake version:    $$(cmake --version | head -n1)"
	@echo "  Python version:   $$(python3 --version)"
	@echo "  C++ compiler:     $${CXX:-g++}"
	@if command -v icpx &> /dev/null; then \
		echo "  Intel compiler:   $$(icpx --version | head -n1)"; \
	else \
		echo "  Intel compiler:   Not found"; \
	fi
	@if [ -n "$${ONEAPI_ROOT}" ]; then \
		echo "  oneAPI root:      $${ONEAPI_ROOT}"; \
	else \
		echo "  oneAPI root:      Not set"; \
	fi
	@if [ -n "$${ISHMEM_HOME}" ]; then \
		echo "  ISHMEM home:      $${ISHMEM_HOME}"; \
	else \
		echo "  ISHMEM home:      Not set"; \
	fi

# Help message
help:
	@echo "Intel XPU Deep EP Extension - Build System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all         Build in release mode (default)"
	@echo "  build       Build in release mode"
	@echo "  release     Build in release mode"
	@echo "  debug       Build in debug mode"
	@echo "  clean       Remove build artifacts"
	@echo "  rebuild     Clean and rebuild"
	@echo "  install     Install Python package"
	@echo "  test        Run basic tests"
	@echo "  test-all    Run all tests"
	@echo "  no-ishmem   Build without Intel SHMEM support"
	@echo "  config      Show build configuration"
	@echo "  help        Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make              # Build in release mode"
	@echo "  make debug        # Build in debug mode"
	@echo "  make clean build  # Clean and rebuild"
	@echo "  make test         # Run tests"
	@echo ""
	@echo "For more options, use the build script directly:"
	@echo "  ./build.sh --help"

