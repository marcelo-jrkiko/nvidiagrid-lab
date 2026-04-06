#!/bin/bash

# FFT Multi-GPU Build and Test Script
# This script builds and tests the CUDA FFT implementations

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/build.log"

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}CUDA FFT Multi-GPU Build & Test${NC}"
echo -e "${BLUE}=====================================${NC}\n"

# Function to print section headers
print_header() {
    echo -e "\n${BLUE}>> $1${NC}"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error messages
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to print info messages
print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check prerequisites
print_header "Checking Prerequisites"

if ! command -v nvcc &> /dev/null; then
    print_error "CUDA nvcc compiler not found"
    exit 1
fi
print_success "CUDA nvcc found: $(nvcc --version | tail -1)"

if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found"
    exit 1
fi
print_success "NVIDIA drivers found"

if ! command -v make &> /dev/null; then
    print_error "make not found"
    exit 1
fi
print_success "make found"

# Display GPU information
print_header "Available GPU Devices"

gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
print_info "Found $gpu_count GPU(s)"

nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader

echo ""

# Build the project
print_header "Building CUDA FFT Programs"

cd "$SCRIPT_DIR"

# Clean previous builds
print_info "Cleaning previous builds..."
make clean > "$LOG_FILE" 2>&1 || true

# Build all targets
print_info "Building fft_2d_multi_gpu..."
if make fft_2d_multi_gpu >> "$LOG_FILE" 2>&1; then
    print_success "fft_2d_multi_gpu built successfully"
else
    print_error "Failed to build fft_2d_multi_gpu"
    echo "Build log:"
    cat "$LOG_FILE"
    exit 1
fi

print_info "Building fft_batch_processing..."
if make fft_batch_processing >> "$LOG_FILE" 2>&1; then
    print_success "fft_batch_processing built successfully"
else
    print_error "Failed to build fft_batch_processing"
    echo "Build log:"
    tail -20 "$LOG_FILE"
    exit 1
fi

# Verify executables
print_header "Verifying Executables"

for exe in fft_2d_multi_gpu fft_batch_processing; do
    if [ -f "$SCRIPT_DIR/$exe" ]; then
        size=$(ls -lh "$SCRIPT_DIR/$exe" | awk '{print $5}')
        print_success "$exe exists (size: $size)"
    else
        print_error "$exe not found"
        exit 1
    fi
done

# Test single GPU
print_header "Testing Single-GPU FFT"

print_info "Running: ./fft_2d_multi_gpu -o test_single.ppm"
if ./fft_2d_multi_gpu -o test_single.ppm > /tmp/fft_single.log 2>&1; then
    if [ -f test_single.ppm ]; then
        size=$(ls -lh test_single.ppm | awk '{print $5}')
        print_success "Single-GPU test passed (output: $size)"
    else
        print_error "Output PPM file not created"
    fi
else
    print_error "Single-GPU test failed"
    cat /tmp/fft_single.log
fi

# Test multi-GPU if available
if [ "$gpu_count" -ge 2 ]; then
    print_header "Testing Multi-GPU FFT (using $gpu_count GPUs)"
    
    print_info "Running: ./fft_2d_multi_gpu -m -o test_multi.ppm"
    if ./fft_2d_multi_gpu -m -o test_multi.ppm > /tmp/fft_multi.log 2>&1; then
        if [ -f test_multi.ppm ]; then
            size=$(ls -lh test_multi.ppm | awk '{print $5}')
            print_success "Multi-GPU test passed (output: $size)"
        else
            print_error "Output PPM file not created"
        fi
    else
        print_error "Multi-GPU test failed"
        cat /tmp/fft_multi.log
    fi
else
    print_info "Skipping multi-GPU test (only 1 GPU available)"
fi

# Test batch processing
print_header "Testing Batch Processing"

batch_size=4
img_size=256

print_info "Running: ./fft_batch_processing $batch_size $img_size $img_size"
if ./fft_batch_processing $batch_size $img_size $img_size > /tmp/fft_batch.log 2>&1; then
    print_success "Batch processing test completed"
    
    # Extract timing info
    if grep -q "Total time:" /tmp/fft_batch.log; then
        timing=$(grep "Total time:" /tmp/fft_batch.log)
        print_info "$timing"
    fi
else
    print_error "Batch processing test failed"
    cat /tmp/fft_batch.log
fi

# Summary
print_header "Build Summary"

echo ""
print_success "All builds completed successfully!"
print_success "Executables ready:"
echo "  - $SCRIPT_DIR/fft_2d_multi_gpu"
echo "  - $SCRIPT_DIR/fft_batch_processing"

echo ""
print_info "Output files generated:"
ls -lh *.ppm 2>/dev/null | awk '{print "  -", $9, "(" $5 ")"}' || print_info "  (No PPM files yet - run executables to generate)"

echo ""
print_header "Next Steps"

echo "1. Run single-GPU FFT:"
echo "   $SCRIPT_DIR/fft_2d_multi_gpu -o output.ppm"
echo ""
echo "2. Run multi-GPU FFT:"
echo "   $SCRIPT_DIR/fft_2d_multi_gpu -m -o output_multi.ppm"
echo ""
echo "3. Run batch processing:"
echo "   $SCRIPT_DIR/fft_batch_processing 8 512 512"
echo ""
echo "For more information, see:"
echo "  - README.md"
echo "  - IMPLEMENTATION_GUIDE.md"
echo ""

# Clean up temp logs
rm -f "$LOG_FILE" /tmp/fft_*.log

print_header "Build Complete"
echo ""
