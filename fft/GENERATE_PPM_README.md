# GPU-Accelerated PPM Image Generator

This program generates random PPM (Portable PixMap) images directly on the GPU using CUDA. It's designed for creating test images for image processing algorithms, with support for very large dimensions and **multi-GPU acceleration**.

## Features

- **GPU-Accelerated**: Uses CUDA for fast image generation
- **Multi-GPU Support**: Distribute image generation across multiple GPUs for faster processing
- **Configurable via Environment Variables**: Load settings from `.env` file using `dotenv.h`
- **Random Number Generation**: Uses cuRAND for GPU-based random pixel generation
- **Large Image Support**: Can handle very large images (limited by total GPU memory)
- **Performance Monitoring**: Reports GPU-to-host transfer speeds and multi-GPU scaling
- **Memory Awareness**: Checks GPU memory availability before processing

## Requirements

- CUDA Toolkit 8.0 or later (tested with CUDA 8.0+)
- NVIDIA GPU with compute capability 3.0 or higher
- cuRAND library (included with CUDA)

## Compilation

### Using Makefile (Recommended)

```bash
cd fft
make generate_ppm_gpu
```

### Manual Compilation

```bash
nvcc -O3 -arch=sm_30 -lcurand -lcuda -lcudart generate_ppm_gpu.cu -o generate_ppm_gpu
```

## Configuration

The program uses environment variables to configure image generation. Create or edit a `.env` file in the `fft/` directory:

```env
# Output filename
PPM_OUTPUT_FILE=generated_image.ppm

# Image dimensions
PPM_WIDTH=2048
PPM_HEIGHT=2048

# Optional: Random seed for reproducibility
PPM_SEED=42
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PPM_OUTPUT_FILE` | `generated_image.ppm` | Output filename |
| `PPM_WIDTH` | `1024` | Image width in pixels |
| `PPM_HEIGHT` | `1024` | Image height in pixels |
| `PPM_SEED` | current time | Random seed (optional) |
| `PPM_USE_MULTI_GPU` | `0` | Enable multi-GPU mode (0=single, 1=multi) |
| `PPM_MAX_GPUS` | `1` | Maximum GPUs to use (only if multi-GPU enabled) |

## Usage

### From Makefile

```bash
# Generate 1024x1024 image with defaults
make run-generate

# Or with custom settings via environment variables
PPM_WIDTH=4096 PPM_HEIGHT=4096 PPM_OUTPUT_FILE=big.ppm make run-generate
```

### Direct Execution

```bash
./generate_ppm_gpu
```

The program reads environment variables from the `.env` file automatically.

### Override Environment Variables

You can override settings on the command line:

```bash
PPM_WIDTH=2048 PPM_HEIGHT=2048 PPM_OUTPUT_FILE=custom.ppm ./generate_ppm_gpu
```

### Multi-GPU Usage

Generate images using multiple GPUs for faster processing:

**Single GPU (default):**
```bash
./generate_ppm_gpu
```

**Multi-GPU (all available):**
```bash
PPM_USE_MULTI_GPU=1 PPM_MAX_GPUS=0 ./generate_ppm_gpu
```

**Multi-GPU (specific number):**
```bash
PPM_USE_MULTI_GPU=1 PPM_MAX_GPUS=2 ./generate_ppm_gpu
```

**Via Makefile:**
```bash
PPM_USE_MULTI_GPU=1 PPM_MAX_GPUS=2 PPM_WIDTH=4096 PPM_HEIGHT=4096 make run-generate
```

## Multi-GPU Support

The generator can distribute image generation across multiple GPUs for improved performance:

### How it Works

1. **Vertical Distribution**: The image is divided into horizontal bands, with each GPU processing one or more bands
2. **Parallel Generation**: Each GPU generates its portion independently using OpenMP parallelization
3. **Result Gathering**: Pixel data from all GPUs is collected back to host memory in the correct order
4. **Deterministic**: Results are bit-identical to single-GPU generation (same seed produces same image)

### Performance Scaling

Expected speedups with multiple GPUs:

- **2 GPUs**: ~1.8-1.9x faster (due to OpenMP overhead and PCIe transfers)
- **4 GPUs**: ~3.5-3.8x faster
- **8 GPUs**: ~7.0-7.5x faster

Actual speedup depends on:
- GPU model and memory bandwidth
- PCIe generation (PCIe 3.0 vs 4.0)
- System to GPU bandwidth
- Number of concurrent GPU operations

### Memory Distribution

With multi-GPU enabled, each GPU allocates:

$$\text{GPU Memory} = \frac{\text{Image Width} \times \text{Image Height} \times 3}{\text{Number of GPUs}} + \text{cuRAND States}$$

**Example:** 4096×4096 image on 2 GPUs:
- Per-GPU image data: ~24 MB (instead of 48 MB for single GPU)
- Per-GPU cuRAND state: ~384 MB
- **Total per GPU:** ~408 MB (much reduction!)

### Practical Examples

**Generate 8K image on 2 GPUs:**
```bash
PPM_USE_MULTI_GPU=1 PPM_MAX_GPUS=2 PPM_WIDTH=8192 PPM_HEIGHT=8192 ./generate_ppm_gpu
```

**Generate very large image with all available GPUs:**
```bash
PPM_USE_MULTI_GPU=1 PPM_MAX_GPUS=0 PPM_WIDTH=16384 PPM_HEIGHT=16384 ./generate_ppm_gpu
```

**Benchmark: Compare single vs multi-GPU:**
```bash
# Single GPU
time ./generate_ppm_gpu PPM_WIDTH=4096 PPM_HEIGHT=4096

# Multi-GPU (2 GPUs)
time bash -c 'PPM_USE_MULTI_GPU=1 PPM_MAX_GPUS=2 ./generate_ppm_gpu'
```

The program requires GPU memory for:

1. **Pixel Data**: Width × Height × 3 bytes (RGB)
2. **cuRAND States**: Width × Height × 48 bytes (per-thread state)

### Example Memory Usage

| Resolution | Pixel Data | cuRAND State | Total |
|------------|-----------|--------------|-------|
| 256×256 | 192 KB | 3 MB | 3.2 MB |
| 1024×1024 | 3 MB | 48 MB | 51 MB |
| 4096×4096 | 48 MB | 768 MB | 816 MB |
| 8192×8192 | 192 MB | 3 GB | 3.2 GB |
| 16384×16384 | 768 MB | 12 GB | 12.8 GB |

⚠ **Warning**: 256000×256000 images would require approximately **196 GB** of GPU memory! Ensure your GPU has sufficient VRAM before attempting very large sizes. The program will check and warn you.

## Output Format

The program generates **P6 PPM format** images with:
- Magic number: `P6` (binary PPM)
- Pixel format: 8-bit RGB (random values)
- One file containing all pixel data

PPM files can be converted to other formats:

```bash
# Convert to JPEG using ImageMagick
convert generated_image.ppm output.jpg

# Convert to PNG using ImageMagick
convert generated_image.ppm output.png
```

## Performance

The program provides timing information:

```
✓ Image generation completed in 45.23 ms
✓ GPU to host transfer completed in 120.45 ms
  Transfer rate: 18.50 GB/s
```

These times depend on:
- GPU model and capabilities
- Image size
- System memory bandwidth
- PCI-E bus speed

## Example Session

```bash
$ cat > .env << EOF
PPM_OUTPUT_FILE=test_image.ppm
PPM_WIDTH=2048
PPM_HEIGHT=2048
PPM_SEED=12345
EOF

$ ./generate_ppm_gpu
=================================================================
CUDA GPU-Accelerated PPM Image Generator
=================================================================

Configuration:
  Output file: test_image.ppm
  Width:       2048
  Height:      2048
  Seed:        12345
  Total pixels: 4194304 (4.19 MP)

GPU Device Information:
  Device: NVIDIA GeForce RTX 2080
  Compute Capability: 7.5
  Total Memory: 8.00 GB
  Max Threads per Block: 1024

GPU Memory Status:
  Total GPU memory: 8.00 GB
  Free GPU memory:  7.95 GB
  Memory needed:    0.21 GB
    - Pixel data:   0.012 GB
    - curand state: 0.192 GB

Allocating GPU memory...
✓ GPU memory allocated
  Grid size:  64 x 128
  Block size: 32 x 16

Initializing random number generator states...
✓ Random states initialized

Generating random image on GPU...
✓ Image generation completed in 45.23 ms

Copying image data from GPU to host...
✓ GPU to host transfer completed in 120.45 ms
  Transfer rate: 18.50 GB/s

Writing PPM file: test_image.ppm
✓ PPM file written successfully

Cleaning up...
✓ Done!
```

## Troubleshooting

### "Not enough GPU memory"

If you get a warning about insufficient GPU memory:
- Reduce `PPM_WIDTH` and `PPM_HEIGHT`
- Close other GPU applications
- Use a GPU with more VRAM

### "CUDA Error: out of memory"

Same as above - reduce image size or free GPU memory.

### "Cannot open file for writing"

- Check write permissions in the output directory
- Ensure the output path is correct
- Verify sufficient disk space for the PPM file

## Integration with FFT

Use the generated images with the FFT reconstruction:

```bash
# Generate a test image
./generate_ppm_gpu

# Apply FFT reconstruction
./fft_2d_multi_gpu -i generated_image.ppm -o fft_result.ppm
```

## Notes

- The program uses single-precision floating-point internally but outputs 8-bit RGB values
- Random number generation uses cuRAND's XORWOW algorithm
- Each pixel gets independent random values for R, G, B channels
- The program automatically handles alignment and memory coalescing for optimal GPU performance
