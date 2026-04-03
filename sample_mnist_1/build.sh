#!/bin/bash
# Build script for MNIST GRID K1 sample

set -e

echo "Building MNIST classification for GRID K1..."
echo "=============================================="

# Check for CUDA toolkit
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA toolkit not found. Please install CUDA."
    exit 1
fi

echo "CUDA version: $(nvcc --version | tail -n1)"
echo ""

# Compile CUDA code
echo "Compiling CUDA code..."
mkdir -p bin

nvcc -arch=sm_35 -std=c++11 -o bin/classify_mnist classify_mnist.cu -lcuda -lcudart

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo ""
    echo "Executable: bin/classify_mnist"
    echo "Run with: ./bin/classify_mnist"
else
    echo "✗ Build failed!"
    exit 1
fi
