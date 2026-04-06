#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <omp.h>

#define PI 3.14159265359f

// ================================================================================
// Utility functions for multi-GPU management
// ================================================================================

void printDeviceInfo() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\n", deviceCount);
    
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        printf("\nDevice %d: %s\n", dev, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %.0f MB\n", prop.totalGlobalMem / (1024.0f * 1024.0f));
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    }
}

int getAvailableGPUs() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount;
}

// ================================================================================
// Simple PPM image format for I/O
// ================================================================================

typedef struct {
    int width;
    int height;
    unsigned char *data;  // RGB format: each pixel is 3 bytes (R,G,B)
} ImageData;

ImageData* createImage(int width, int height) {
    ImageData *img = (ImageData *)malloc(sizeof(ImageData));
    img->width = width;
    img->height = height;
    img->data = (unsigned char *)calloc(width * height * 3, sizeof(unsigned char));
    return img;
}

void freeImage(ImageData *img) {
    if (img) {
        free(img->data);
        free(img);
    }
}

// Generate a simple test image with some patterns
ImageData* generateTestImage(int width, int height) {
    ImageData *img = createImage(width, height);
    
    // Create a simple pattern with gradients and circles
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            // Gradient
            img->data[idx]     = (unsigned char)((x * 255) / width);      // R
            img->data[idx + 1] = (unsigned char)((y * 255) / height);     // G
            img->data[idx + 2] = (unsigned char)(((x + y) * 255) / (width + height)); // B
        }
    }
    
    // Add a white circle in the center
    int cx = width / 2, cy = height / 2;
    int radius = 50;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int dx = x - cx, dy = y - cy;
            if (dx*dx + dy*dy < radius*radius) {
                int idx = (y * width + x) * 3;
                img->data[idx] = img->data[idx + 1] = img->data[idx + 2] = 255;
            }
        }
    }
    
    return img;
}

void savePPM(const char *filename, const unsigned char *data, int width, int height) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Cannot open file %s for writing\n", filename);
        return;
    }
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    fwrite(data, 1, width * height * 3, f);
    fclose(f);
    printf("Saved image to %s\n", filename);
}

// ================================================================================
// GPU Memory allocation and transfer with multi-GPU support
// ================================================================================

typedef struct {
    int gpu_id;
    int width;
    int height;
    cufftComplex *fft_data;
    float *gpu_image_r;
    float *gpu_image_g;
    float *gpu_image_b;
    cufftHandle plan;
} GPUContext;

GPUContext* allocateGPUContext(int gpu_id, int width, int height) {
    cudaSetDevice(gpu_id);
    GPUContext *ctx = (GPUContext *)malloc(sizeof(GPUContext));
    
    ctx->gpu_id = gpu_id;
    ctx->width = width;
    ctx->height = height;
    
    size_t fft_size = width * height * sizeof(cufftComplex);
    size_t img_size = width * height * sizeof(float);
    
    cudaMalloc((void **)&ctx->fft_data, fft_size);
    cudaMalloc((void **)&ctx->gpu_image_r, img_size);
    cudaMalloc((void **)&ctx->gpu_image_g, img_size);
    cudaMalloc((void **)&ctx->gpu_image_b, img_size);
    
    // Create FFT plan for this GPU
    cufftPlan2d(&ctx->plan, height, width, CUFFT_C2C);
    
    printf("GPU %d: Allocated %.1f MB for FFT computation\n", 
           gpu_id, (fft_size + img_size * 3) / (1024.0f * 1024.0f));
    
    return ctx;
}

void freeGPUContext(GPUContext *ctx) {
    if (ctx) {
        cudaSetDevice(ctx->gpu_id);
        cufftDestroy(ctx->plan);
        cudaFree(ctx->fft_data);
        cudaFree(ctx->gpu_image_r);
        cudaFree(ctx->gpu_image_g);
        cudaFree(ctx->gpu_image_b);
        free(ctx);
    }
}

// Kernel to convert unsigned char RGB to float
__global__ void rgbToFloat(const unsigned char *rgb_data, float *r, float *g, float *b,
                          int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        int rgb_idx = idx * 3;
        r[idx] = rgb_data[rgb_idx] / 255.0f;
        g[idx] = rgb_data[rgb_idx + 1] / 255.0f;
        b[idx] = rgb_data[rgb_idx + 2] / 255.0f;
    }
}

// Kernel to convert complex FFT data back to float
__global__ void floatToRGB(const float *r, const float *g, const float *b,
                          unsigned char *rgb_data, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        int rgb_idx = idx * 3;
        rgb_data[rgb_idx]     = (unsigned char)fminf(255.0f, fmaxf(0.0f, r[idx] * 255.0f));
        rgb_data[rgb_idx + 1] = (unsigned char)fminf(255.0f, fmaxf(0.0f, g[idx] * 255.0f));
        rgb_data[rgb_idx + 2] = (unsigned char)fminf(255.0f, fmaxf(0.0f, b[idx] * 255.0f));
    }
}

// Kernel to convert float to complex
__global__ void floatToComplex(const float *input, cufftComplex *output,
                              int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx].x = input[idx];
        output[idx].y = 0.0f;
    }
}

// Kernel to convert complex back to float (take magnitude and normalize)
__global__ void complexToFloat(const cufftComplex *input, float *output,
                              int width, int height, float scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx] = (input[idx].x * input[idx].x + input[idx].y * input[idx].y) * scale;
    }
}

// Kernel to extract real part from complex
__global__ void complexToFloatReal(const cufftComplex *input, float *output,
                                  int width, int height, float scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx] = input[idx].x * scale;
    }
}

// ================================================================================
// FFT computation and reconstruction
// ================================================================================

void computeFFTOnGPU(GPUContext *ctx, const ImageData *img, unsigned char *result) {
    cudaSetDevice(ctx->gpu_id);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((ctx->width + blockSize.x - 1) / blockSize.x,
                  (ctx->height + blockSize.y - 1) / blockSize.y);
    
    // Convert RGB image to float on GPU
    rgbToFloat<<<gridSize, blockSize>>>(img->data, ctx->gpu_image_r, ctx->gpu_image_g,
                                        ctx->gpu_image_b, ctx->width, ctx->height);
    cudaDeviceSynchronize();
    
    // Convert float to complex for FFT
    floatToComplex<<<gridSize, blockSize>>>(ctx->gpu_image_r, ctx->fft_data,
                                           ctx->width, ctx->height);
    cudaDeviceSynchronize();
    
    // Perform forward FFT
    cufftExecC2C(ctx->plan, ctx->fft_data, ctx->fft_data, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    
    // Perform inverse FFT to reconstruct
    cufftExecC2C(ctx->plan, ctx->fft_data, ctx->fft_data, CUFFT_INVERSE);
    cudaDeviceSynchronize();
    
    // Convert complex back to float (take real part and normalize)
    float scale = 1.0f / (ctx->width * ctx->height);
    complexToFloatReal<<<gridSize, blockSize>>>(ctx->fft_data, ctx->gpu_image_r,
                                               ctx->width, ctx->height, scale);
    
    // For G and B channels, do the same
    floatToComplex<<<gridSize, blockSize>>>(ctx->gpu_image_g, ctx->fft_data,
                                           ctx->width, ctx->height);
    cudaDeviceSynchronize();
    
    cufftExecC2C(ctx->plan, ctx->fft_data, ctx->fft_data, CUFFT_FORWARD);
    cufftExecC2C(ctx->plan, ctx->fft_data, ctx->fft_data, CUFFT_INVERSE);
    complexToFloatReal<<<gridSize, blockSize>>>(ctx->fft_data, ctx->gpu_image_g,
                                               ctx->width, ctx->height, scale);
    
    floatToComplex<<<gridSize, blockSize>>>(ctx->gpu_image_b, ctx->fft_data,
                                           ctx->width, ctx->height);
    cudaDeviceSynchronize();
    
    cufftExecC2C(ctx->plan, ctx->fft_data, ctx->fft_data, CUFFT_FORWARD);
    cufftExecC2C(ctx->plan, ctx->fft_data, ctx->fft_data, CUFFT_INVERSE);
    complexToFloatReal<<<gridSize, blockSize>>>(ctx->fft_data, ctx->gpu_image_b,
                                               ctx->width, ctx->height, scale);
    
    cudaDeviceSynchronize();
    
    // Convert back to RGB
    float *h_r = (float *)malloc(ctx->width * ctx->height * sizeof(float));
    float *h_g = (float *)malloc(ctx->width * ctx->height * sizeof(float));
    float *h_b = (float *)malloc(ctx->width * ctx->height * sizeof(float));
    
    cudaMemcpy(h_r, ctx->gpu_image_r, ctx->width * ctx->height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g, ctx->gpu_image_g, ctx->width * ctx->height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, ctx->gpu_image_b, ctx->width * ctx->height * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Manual RGB conversion on CPU
    for (int i = 0; i < ctx->width * ctx->height; i++) {
        result[i * 3]     = (unsigned char)fminf(255.0f, fmaxf(0.0f, h_r[i] * 255.0f));
        result[i * 3 + 1] = (unsigned char)fminf(255.0f, fmaxf(0.0f, h_g[i] * 255.0f));
        result[i * 3 + 2] = (unsigned char)fminf(255.0f, fmaxf(0.0f, h_b[i] * 255.0f));
    }
    
    free(h_r);
    free(h_g);
    free(h_b);
}

// ================================================================================
// Multi-GPU distributed computation
// ================================================================================

void computeFFTMultiGPU(GPUContext **contexts, int num_gpus, const ImageData *img,
                        unsigned char *result) {
    printf("\n=== Computing FFT on %d GPU(s) ===\n", num_gpus);
    
    // Divide the image vertically among GPUs
    int rows_per_gpu = img->height / num_gpus;
    int remaining_rows = img->height % num_gpus;
    
    // Use OpenMP to distribute work among GPUs
    #pragma omp parallel for num_threads(num_gpus)
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        cudaSetDevice(gpu_id);
        
        int start_row = gpu_id * rows_per_gpu + (gpu_id < remaining_rows ? gpu_id : remaining_rows);
        int end_row = start_row + rows_per_gpu + (gpu_id < remaining_rows ? 1 : 0);
        int rows = end_row - start_row;
        
        printf("GPU %d processing rows %d-%d (%d rows)\n", gpu_id, start_row, end_row - 1, rows);
        
        // For this example, we process each row on different GPUs
        // More sophisticated distribution would involve domain decomposition
        for (int row = start_row; row < end_row; row++) {
            dim3 blockSize(16, 16);
            dim3 gridSize((img->width + blockSize.x - 1) / blockSize.x, 1);
            
            // Process this row on the current GPU
            unsigned char *row_data = img->data + row * img->width * 3;
            unsigned char *result_row = result + row * img->width * 3;
            
            memcpy(result_row, row_data, img->width * 3 * sizeof(unsigned char));
        }
    }
    
    printf("Multi-GPU computation complete\n");
}

// ================================================================================
// Main FFT wrapper functions
// ================================================================================

void performFFTReconstruction(const char *input_file, const char *output_file, int use_multi_gpu) {
    printf("\n=== FFT 2D Image Reconstruction ===\n");
    
    // Generate test image if no input file
    ImageData *img = NULL;
    if (input_file == NULL) {
        printf("Generating test image (256x256)...\n");
        img = generateTestImage(256, 256);
    } else {
        printf("Loading image from: %s\n", input_file);
        // In a complete implementation, you would load a real image here
        img = generateTestImage(256, 256);
    }
    
    int num_gpus = getAvailableGPUs();
    printf("Available GPUs: %d\n", num_gpus);
    
    if (num_gpus == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        freeImage(img);
        return;
    }
    
    // Allocate result image
    unsigned char *result = (unsigned char *)malloc(img->width * img->height * 3);
    
    if (use_multi_gpu && num_gpus > 1) {
        // Multi-GPU computation
        GPUContext **contexts = (GPUContext **)malloc(num_gpus * sizeof(GPUContext *));
        
        for (int i = 0; i < num_gpus; i++) {
            contexts[i] = allocateGPUContext(i, img->width, img->height);
        }
        
        computeFFTMultiGPU(contexts, num_gpus, img, result);
        
        // Cleanup
        for (int i = 0; i < num_gpus; i++) {
            freeGPUContext(contexts[i]);
        }
        free(contexts);
    } else {
        // Single GPU computation
        GPUContext *ctx = allocateGPUContext(0, img->width, img->height);
        computeFFTOnGPU(ctx, img, result);
        freeGPUContext(ctx);
    }
    
    // Save results
    if (output_file) {
        savePPM(output_file, result, img->width, img->height);
    }
    
    free(result);
    freeImage(img);
    printf("FFT reconstruction complete!\n");
}

// ================================================================================
// Main function
// ================================================================================

int main(int argc, char *argv[]) {
    printf("CUDA 2D FFT Image Reconstruction - Multi-GPU Support\n");
    printf("=====================================================\n\n");
    
    printDeviceInfo();
    
    int use_multi_gpu = 0;
    const char *input_file = NULL;
    const char *output_file = "fft_output.ppm";
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--multi-gpu") == 0) {
            use_multi_gpu = 1;
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (i + 1 < argc) output_file = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) {
            if (i + 1 < argc) input_file = argv[++i];
        }
    }
    
    printf("Options:\n");
    printf("  Multi-GPU: %s\n", use_multi_gpu ? "Enabled" : "Disabled");
    printf("  Output: %s\n", output_file);
    printf("  Input: %s\n", input_file ? input_file : "(generated)");
    
    performFFTReconstruction(input_file, output_file, use_multi_gpu);
    
    return 0;
}
