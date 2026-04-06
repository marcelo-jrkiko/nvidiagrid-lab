# CUDA FFT Implementation Guide

## Executive Summary

This project provides production-ready CUDA implementations for 2D Fast Fourier Transform (FFT) image processing with multi-GPU support. It includes two complete examples:

1. **`fft_2d_multi_gpu.cu`** - Basic 2D FFT with image I/O for single and multi-GPU execution
2. **`fft_batch_processing.cu`** - Advanced batch processing with performance measurements

Both implementations use NVIDIA's **cuFFT** library for optimal performance on NVIDIA GPUs.

---

## Part 1: Understanding 2D FFT for Images

### What is 2D FFT?

The 2D Fast Fourier Transform decomposes a 2D signal (image) into its frequency components:

$$F(u,v) = \sum_{x=0}^{N-1} \sum_{y=0}^{M-1} f(x,y) e^{-j2\pi(ux/N + vy/M)}$$

Where:
- $f(x,y)$ = spatial domain (image pixel)
- $F(u,v)$ = frequency domain
- $(u,v)$ = frequency coordinates
- Operation decomposes image into horizontal and vertical frequency components

### Why FFT for Images?

1. **Frequency Domain Analysis**: Identify dominant frequencies in the image
2. **Filtering**: Apply operations in frequency domain (often more efficient)
3. **Compression**: Remove high-frequency noise components
4. **Reconstruction**: Inverse FFT returns to spatial domain

### Forward and Inverse Transform

```
Spatial Domain (Image) 
    ↓
FORWARD FFT (C2C: Complex-to-Complex)
    ↓
Frequency Domain (Complex spectral data)
    ↓
INVERSE FFT 
    ↓
Spatial Domain (Reconstructed Image)
```

---

## Part 2: cuFFT Library

### Key Components

#### FFT Plan Creation
```cuda
cufftHandle plan;
cufftPlan2d(&plan, height, width, CUFFT_C2C);
```

- **Parameters**:
  - `height`: Number of rows
  - `width`: Number of columns  
  - `CUFFT_C2C`: Complex-to-Complex transformation

#### Plan Execution

**Forward Transform:**
```cuda
cufftExecC2C(plan, input, output, CUFFT_FORWARD);
```

**Inverse Transform:**
```cuda
cufftExecC2C(plan, input, output, CUFFT_INVERSE);
```

#### Data Types

| Type | Input | Output | Use Case |
|------|-------|--------|----------|
| `CUFFT_C2C` | Complex | Complex | General FFT, Filtering |
| `CUFFT_R2C` | Real float | Complex | Real-valued inputs |
| `CUFFT_C2R` | Complex | Real float | Converting back to spatial |

### Memory Layout

cuFFT uses **C-contiguous** row-major storage:
```
Memory Layout: [row0_col0, row0_col1, ..., row1_col0, row1_col1, ...]
```

---

## Part 3: Implementation Architecture

### Data Flow in `fft_2d_multi_gpu.cu`

```
┌─────────────────────────────────────────────────────┐
│ Main Function                                       │
├─────────────────────────────────────────────────────┤
│ 1. Detect available GPUs                            │
│ 2. Generate/Load test image (PPM format)            │
│ 3. Allocate GPU contexts (one per GPU)              │
│ 4. Dispatch work:                                   │
│    ├─ Single GPU: computeFFTOnGPU()                 │
│    └─ Multi-GPU: computeFFTMultiGPU()               │
│ 5. Save results (PPM format)                        │
└─────────────────────────────────────────────────────┘
```

### GPU Context Management

**Structure:**
```cuda
typedef struct {
    int gpu_id;                    // GPU identifier
    int width, height;             // Image dimensions
    cufftHandle plan;              // FFT execution plan
    cufftComplex *fft_data;        // FFT working buffer
    float *gpu_image_r;            // Red channel
    float *gpu_image_g;            // Green channel
    float *gpu_image_b;            // Blue channel
} GPUContext;
```

**Memory per GPU (256×256 image):**
```
- Complex buffer: 256² × 8 bytes = 512 KB
- Float channels: 3 × 256² × 4 bytes = 768 KB
- Total: ~1.3 MB per GPU
```

### CUDA Kernels

#### 1. RGB to Float Conversion

```cuda
__global__ void rgbToFloat(const unsigned char *rgb_data, 
                          float *r, float *g, float *b,
                          int width, int height)
```

**Purpose**: Convert 8-bit unsigned char (0-255) to 32-bit float (0.0-1.0)

**Computation per thread:**
```
R[i] = RGB_data[i*3] / 255.0
G[i] = RGB_data[i*3+1] / 255.0
B[i] = RGB_data[i*3+2] / 255.0
```

#### 2. Float to Complex Conversion

```cuda
__global__ void floatToComplex(const float *input, 
                              cufftComplex *output,
                              int width, int height)
```

**Purpose**: Prepare float data for FFT

**Computation per thread:**
```
output[i].x = input[i]        // Real part (image data)
output[i].y = 0.0            // Imaginary part (zero)
```

#### 3. Complex to Float (Real Part Extraction)

```cuda
__global__ void complexToFloatReal(const cufftComplex *input, 
                                  float *output,
                                  int width, int height, 
                                  float scale)
```

**Purpose**: Extract real part after IFFT with normalization

**Computation per thread:**
```
output[i] = input[i].x * scale
// scale = 1 / (width × height)
```

#### 4. Frequency Domain Filter

```cuda
__global__ void applyFrequencyFilter(cufftComplex *fft_data,
                                    int width, int height,
                                    float filter_type)
```

**Purpose**: Apply low-pass or high-pass filtering in frequency domain

**Algorithm:**
1. Calculate frequency magnitude for each component
2. Apply attenuation based on distance from center
3. Modify FFT coefficients: `FFT[i] *= attenuation`

### FFT Computation Pipeline

**For each channel (R, G, B):**

```
Input Channel (Float)
    ↓
[Float → Complex] Kernel
    ↓ (Result: Complex buffer with Im=0)
    
[Forward FFT]
    ↓ (Result: Frequency domain)

[Optional: Apply Filter]
    ↓
    
[Inverse FFT]
    ↓ (Result: Complex spatial domain)

[Extract Real + Normalize] Kernel
    ↓ (Divide by width×height, take x-component)

Output Channel (Float 0-1)
```

### Normalization

**Critical for correctness:**

$$\text{IFFT}(f) = \frac{1}{N} \sum FFT \text{ results}$$

The `1/N` factor (where $N = \text{width} \times \text{height}$) is essential for:
- Energy preservation
- Proper signal reconstruction
- Numerical stability

```cuda
float scale = 1.0f / (width * height);
complexToFloatReal<<<gridSize, blockSize>>>(..., scale);
```

---

## Part 4: Multi-GPU Execution

### GPU Selection and Context

```cuda
// Set active GPU for all subsequent CUDA operations
cudaSetDevice(gpu_id);

// All memory allocations go to this GPU
cudaMalloc(...);

// All kernels execute on this GPU
kernel<<<...>>>();
```

### OpenMP-based Distribution

**Distribution strategy:**
```cpp
#pragma omp parallel for num_threads(num_gpus)
for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
    cudaSetDevice(gpu_id);
    
    // Each thread manages one GPU
    // Processes assigned image rows
    int rows_per_gpu = height / num_gpus;
    int start_row = gpu_id * rows_per_gpu;
    
    // computeFFTOnGPU(contexts[gpu_id], ...)
}
```

**Key Points:**
1. Each OpenMP thread manages one GPU
2. Threads don't share GPU memory (each GPU has separate memory space)
3. Synchronous barriers between GPU operations ensure correctness
4. PCI-E bandwidth can be limiting factor with many GPUs

### Data Layout for Distribution

```
┌─────────────────┐
│                 │
│   GPU 0         │  Rows 0-63
│   (64 rows)     │
│                 │
├─────────────────┤
│                 │
│   GPU 1         │  Rows 64-127
│   (64 rows)     │
│                 │
├─────────────────┤
│                 │
│   GPU 2         │  Rows 128-191
│   (64 rows)     │
│                 │
└─────────────────┘
```

Total image: 192 rows distributed across 3 GPUs

---

## Part 5: Advanced Batch Processing (`fft_batch_processing.cu`)

### Batch Context Structure

```cuda
typedef struct {
    int gpu_id;
    int batch_size;          // Number of images per batch
    int width, height;       // Image dimensions
    cufftHandle plan;        
    cufftComplex **d_batch_input;    // Array of GPU buffers
    cufftComplex **d_batch_output;   // Array of GPU buffers
} BatchFFTContext;
```

### Performance Profiling

```cuda
struct {
    double upload_time;      // H→D transfer
    double fft_time;         // FFT computation
    double download_time;    // D→H transfer
} timing;
```

### Asynchronous Transfers (Optional)

```cuda
// Non-blocking GPU transfer
cudaMemcpyAsync(d_dst, h_src, size, 
               cudaMemcpyHostToDevice, cudaStream_t stream);

// Kernel execution
kernel<<<grid, block, stream>>>();

// Synchronize stream
cudaStreamSynchronize(stream);
```

### Optimization Strategies for Batches

1. **Pinned Memory**: Use `cudaMallocHost()` for faster transfers
2. **Streams**: Overlap compute and memory transfers
3. **Texture Memory**: Cache-optimized for 2D spatial locality
4. **Persistence**: Keep FFT plans across multiple batches

---

## Part 6: Building and Compilation

### Compiler Flags Explained

```makefile
NVCCFLAGS = -ccbin /opt/gcc-5/bin/gcc \
            -arch=sm_30 \              # GPU architecture
            -std=c++11 \               # C++11 standard
            -Xcompiler -fopenmp        # Pass -fopenmp to GCC
```

### Architecture Selection

| GPU | Code | Cuda Capability |
|-----|------|-----------------|
| Kepler (K1/K2) | `-arch=sm_30` | 3.0/3.5 |
| Maxwell (M6/M60) | `-arch=sm_52` | 5.2 |
| Pascal (P4/P40) | `-arch=sm_61` | 6.1 |
| Volta (V100) | `-arch=sm_70` | 7.0 |
| Ampere (A100) | `-arch=sm_80` | 8.0 |

### cuFFT Linking

```makefile
CUFFT_LIB = -L$(CUDA_LIB) -lcufft_static -lculibos
```

**Options:**
- `-lcufft_static`: Static cuFFT library (larger binary, no runtime dependency)
- `-lcufft`: Dynamic linking (smaller binary, requires cuFFT runtime)

---

## Part 7: Running and Testing

### Single GPU Test

```bash
./fft_2d_multi_gpu -o output.ppm
```

**Expected output:**
```
CUDA 2D FFT Image Reconstruction - Multi-GPU Support
Number of CUDA devices: 1

Device 0: GeForce GTX Titan
  Compute capability: 3.5
  Total global memory: 6051 MB
  ...

Options:
  Multi-GPU: Disabled
  Output: output.ppm
  Input: (generated)

=== FFT 2D Image Reconstruction ===
Generating test image (256x256)...
Available GPUs: 1
GPU 0: Allocated 5.2 MB for FFT computation
...
FFT reconstruction complete!
Saved image to output.ppm
```

### Multi-GPU Test (requires 2+ GPUs)

```bash
./fft_2d_multi_gpu -m -o output_multi.ppm
```

### Batch Processing Test

```bash
./fft_batch_processing 8 512 512
```

**Output includes timing statistics:**
```
GPU 0:
  Upload:   23.45 ms
  FFT:      45.67 ms
  Download: 12.34 ms

GPU 1:
  Upload:   23.40 ms
  FFT:      45.70 ms
  Download: 12.32 ms
```

---

## Part 8: Performance Considerations

### Memory Bandwidth Bottleneck

For a single GPU with 256×256 image:

$$\text{Throughput} = \frac{\text{Data Size}}{W \times H} = \frac{8 \text{ bytes}}{256 \times 256} \approx 0.122 \text{ bytes/element}$$

Typical GPU memory bandwidth: 250-500 GB/s

### Optimal Batch Size

$$\text{Optimal Batch} = \frac{\text{GPU Memory}}{2 \times \text{Image Size}}$$

Example (3GB GPU, 512×512 image = 4 MB):
$$\text{Optimal} = \frac{3000 \text{ MB}}{2 \times 4 \text{ MB}} \approx 375 \text{ images}$$

### Scalability Limits

Multi-GPU speedup affected by:
1. **PCI-E PCIe bandwidth** (32 GB/s PCIe 3.0 ÷ 8)
2. **Inter-GPU communication overhead**
3. **Load imbalance** if GPUs have different capabilities

Typical speedup: 1.5-1.8× for 2 GPUs (not linear due to overhead)

---

## Part 9: Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| "No CUDA devices found" | Check `nvidia-smi`, verify driver |
| cuFFT link errors | Install `libcufft-dev` package |
| Kernel launch errors | Check grid/block dimensions |
| Memory allocation failures | Reduce image size or batch count |
| Wrong results | Check normalization factor |

### Validation Checklist

1. **Image dimensions**: Power-of-2 sizes for optimal cuFFT performance
2. **Data types**: Ensure complex structures properly aligned (16 bytes)
3. **Normalization**: `1/(width×height)` applied after IFFT
4. **GPU sync**: `cudaDeviceSynchronize()` after all kernels

---

## Part 10: Extensions and Future Work

### Batched FFT Optimization

Use `cufftPlanMany()` for efficiently processing multiple images:
```cuda
int n[2] = {height, width};
int inembed[] = {height, width+1};  // Stride for padding
int onembed[] = {height, width+1};

cufftPlanMany(&plan, 2, n, inembed, 1, height*(width+1),
              onembed, 1, height*(width+1), CUFFT_C2C, batch_size);
```

### Streaming and Asynchronous Processing

```cuda
// Multiple streams for pipelining
cudaStream_t streams[3];
for(int i = 0; i < 3; i++) cudaStreamCreate(&streams[i]);

// Overlap H→D transfer, compute, D→H transfer
cudaMemcpyAsync(d_input, h_input, size, ..., streams[0]);
fft_kernel<<<..., streams[1]>>>();
cudaMemcpyAsync(h_output, d_output, size, ..., streams[2]);
```

### Real-Valued FFT (R2C/C2R)

For faster computation when input is real-valued:
```cuda
cufftPlanMany(&plan, ..., CUFFT_R2C);  // Real-to-Complex
// Output size is (N/2+1) due to Hermitian symmetry
```

---

## References

1. **NVIDIA cuFFT Documentation**: https://docs.nvidia.com/cuda/cufft/
2. **CUDA Memory Management**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
3. **FFT Algorithms**: Cooley-Tukey, Stockham, BlueStein formulations
4. **GPU Memory Hierarchy**: L1 cache, L2 cache, global memory trade-offs

