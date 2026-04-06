# CUDA 2D FFT Image Reconstruction - Multi-GPU Implementation

This project provides a comprehensive CUDA implementation for computing 2D Fast Fourier Transforms (FFT) on images with multi-GPU support. It uses NVIDIA's cuFFT library for optimal performance and demonstrates advanced memory management and GPU resource utilization.

## Features

- **2D FFT Computation**: Forward and inverse FFT transforms using cuFFT
- **Multi-GPU Support**: Distributed computation across multiple NVIDIA GPUs using OpenMP
- **Image I/O**: PPM image format support for simple I/O operations
- **Memory Management**: Efficient GPU memory allocation and CPU-GPU data transfers
- **Test Image Generation**: Built-in test image generator for demonstrations
- **Channel Processing**: Independent processing of RGB channels for realistic color image handling

## Requirements

- CUDA Toolkit (version 6.0 or later)
- cuFFT library (included with CUDA)
- GCC/G++ compiler with C++11 support
- OpenMP support

## Building

The project includes a comprehensive Makefile with multiple build targets:

```bash
# Build the project
make

# Build and run single-GPU FFT
make run

# Build and run multi-GPU FFT (requires 2+ GPUs)
make run-multi

# Clean build artifacts
make clean

# Show help
make help
```

## Usage

### Command Line Options

```bash
./fft_2d_multi_gpu [options]
  -m, --multi-gpu        Enable multi-GPU computation
  -o, --output FILE      Output filename (default: fft_output.ppm)
  -i, --input FILE       Input image file (optional)
```

### Examples

**Single GPU - Generate and process test image:**
```bash
./fft_2d_multi_gpu -o result_single.ppm
```

**Multi-GPU - Distributed computation:**
```bash
./fft_2d_multi_gpu -m -o result_multi.ppm
```

**Multi-GPU with input image:**
```bash
./fft_2d_multi_gpu -m -i input.ppm -o result_multi.ppm
```

## Code Structure

### Main Components

#### 1. **GPU Device Management** (`printDeviceInfo`, `getAvailableGPUs`)
   - Detects available CUDA devices
   - Displays device properties and capabilities
   - Enables multi-GPU resource allocation

#### 2. **Image Data Structures** (`ImageData`, PPM I/O functions)
   - RGB image representation in unsigned char format
   - PPM file format support for simple image I/O
   - Automatic test image generation with patterns and shapes

#### 3. **GPU Context Management** (`GPUContext`)
   - Per-GPU resource allocation (FFT plans, memory buffers)
   - Maintains separate FFT plans for each GPU
   - Manages separate float buffers for R, G, B channels
   - Handles complex number buffers for FFT computation

#### 4. **CUDA Kernels**
   - `rgbToFloat`: Convert unsigned char RGB data to float
   - `floatToRGB`: Convert float data back to unsigned char RGB
   - `floatToComplex`: Prepare float data for FFT (set imaginary part to 0)
   - `complexToFloat`: Extract results from FFT output
   - `complexToFloatReal`: Extract real part from complex FFT results

#### 5. **FFT Computation** (`computeFFTOnGPU`)
   - Per-channel FFT processing (R, G, B channels handled independently)
   - Forward FFT transform
   - Inverse FFT transform (reconstruction)
   - Normalization by (width × height)

#### 6. **Multi-GPU Execution** (`computeFFTMultiGPU`)
   - Distributes image rows across multiple GPUs
   - Uses OpenMP for parallel GPU resource management
   - Demonstrates domain decomposition patterns

### Algorithm Flow

```
Input Image (RGB)
    ↓
[Per-GPU Context]: Convert RGB → Float channels
    ↓
[For each channel (R, G, B)]:
    │ 
    ├─→ Float → Complex (imaginary = 0)
    ├─→ Forward FFT (Frequency Domain)
    ├─→ Inverse FFT (Time Domain)
    ├─→ Extract Real Part + Normalize
    │
[Reconstruct]: Reassemble RGB from channels
    ↓
Output Image (PPM)
```

## Kernel Specifications

### `rgbToFloat` Kernel
**Purpose**: Convert RGB unsigned char data to float (0-1 range)
**Grid**: 2D blocks matching image dimensions
**Thread Block**: 16×16 threads

### `floatToComplex` Kernel
**Purpose**: Prepare float data for FFT by setting real part and zeroing imaginary
**Input**: Float channel data
**Output**: Complex array ready for FFT

### `complexToFloatReal` Kernel
**Purpose**: Extract real component from FFT output after IFFT
**Parameters**: 
- `scale = 1/(width × height)` for proper normalization

## Multi-GPU Execution

The implementation distributes image processing across GPUs by:

1. **GPU Detection**: Queries available CUDA devices
2. **Context Creation**: Allocates separate FFT plans and memory on each GPU
3. **Workload Distribution**: Divides image rows among available GPUs using OpenMP
4. **Parallel Processing**: Each GPU independently computes FFT for its assigned data
5. **Result Assembly**: Combines results from all GPUs

**OpenMP Thread Mapping:**
```cpp
#pragma omp parallel for num_threads(num_gpus)
for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
    cudaSetDevice(gpu_id);
    // Process assigned image rows
}
```

## Memory Considerations

For an image of size W × H:
- **Complex FFT Buffer**: W × H × sizeof(cufftComplex) = W × H × 8 bytes
- **Float Channels (3×)**: 3 × W × H × sizeof(float) = 3 × W × H × 4 bytes
- **Total per GPU**: ~20 bytes per pixel

Example for 256×256 image:
- Per-channel size: 256 × 256 × 4 = 262 KB
- Total requirements: ~3.3 MB per GPU

## Performance Notes

1. **Optimal Block Size**: 16×16 threads per block balances occupancy and shared memory usage
2. **FFT Efficiency**: cuFFT is optimized for power-of-2 dimensions; performance degrades for arbitrary sizes
3. **Multi-GPU Scalability**: Actual speedup depends on:
   - PCI-E bandwidth between GPUs
   - GPU memory transfer speed
   - Compute capability of devices

## Building on Different Systems

### NVIDIA GRID Systems
```bash
make NVCC=/path/to/nvcc CUDA_PATH=/path/to/cuda
```

### Custom Architectures
Update `NVCCFLAGS` in Makefile:
```makefile
# For Kepler (K1/K2):
-arch=sm_30

# For Maxwell (M6/M60):
-arch=sm_52

# For Pascal (P4/P40):
-arch=sm_61

# For multiple architectures:
-arch=sm_30 -arch=sm_52 -arch=sm_61
```

## PPM to JPG Conversion

After running the FFT computation, you can convert the PPM output files to JPG format using the included Python script.

### Setup

Install Python dependencies:
```bash
pip install -r requirements-python.txt
```

### Usage

**Basic conversion:**
```bash
python3 convert_ppm_to_jpg.py
```

**With quality setting (1-100):**
```bash
python3 convert_ppm_to_jpg.py --quality=90
```

**Keep original PPM files:**
```bash
python3 convert_ppm_to_jpg.py --keep-ppm
```

### Environment Variables

The conversion script uses the following environment variables:

- `FFT_RESULTS_FOLDER` - Path to results folder (default: `./results`)
- `JPG_QUALITY` - JPEG quality level 1-100 (default: 85)
- `DELETE_PPM` - Delete original PPM files after conversion (default: false)

### Example Workflow

```bash
# 1. Run FFT processing
export FFT_RESULTS_FOLDER=./output
export FFT_OUTPUT_PREFIX=exp1_
export FFT_OUTPUT_FILE=reconstruction.ppm
./fft_2d_multi_gpu

# 2. Convert results to JPG
export JPG_QUALITY=90
python3 convert_ppm_to_jpg.py

# Results will be in ./output/:
# - exp1_original_reconstruction.jpg
# - exp1_reconstruction.jpg
```

## Troubleshooting

### "No CUDA devices found"
- Check NVIDIA driver: `nvidia-smi`
- Verify CUDA installation: `nvcc --version`

### cuFFT link errors
Ensure cuFFT is installed:
```bash
apt-get install libcufft-dev  # Ubuntu/Debian
yum install cuda-cufft-devel  # CentOS/RHEL
```

### Multi-GPU not working
- Verify multiple GPUs: `nvidia-smi -L`
- Check OpenMP: `gcc -fopenmp -v`

## Future Enhancements

1. **Batched FFT Processing**: Process multiple images simultaneously
2. **Custom FFT Kernels**: Implement Cooley-Tukey FFT for learning purposes
3. **Filtering in Frequency Domain**: Add low-pass/high-pass filters
4. **Real Input FFT**: Use CUFFT_R2C for real-valued inputs
5. **CUDA Graphs**: Optimize multi-GPU execution with CUDA graphs
6. **Advanced Load Balancing**: Dynamic workload distribution based on GPU capabilities

## References

- [NVIDIA cuFFT Documentation](https://docs.nvidia.com/cuda/cufft/)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [FFT Algorithms and Properties](https://en.wikipedia.org/wiki/Fast_Fourier_transform)

## License

This code is provided as a demonstration and educational material for NVIDIA GRID systems.
