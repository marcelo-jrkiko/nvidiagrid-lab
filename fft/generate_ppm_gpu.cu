/*
 * CUDA GPU-accelerated PPM Image Generator
 * Generates random PPM images using GPU acceleration with environment variable configuration
 * 
 * Environment variables (from .env file):
 *   - PPM_OUTPUT_FILE: Output filename (default: generated_image.ppm)
 *   - PPM_WIDTH: Image width in pixels (default: 1024)
 *   - PPM_HEIGHT: Image height in pixels (default: 1024)
 *   - PPM_SEED: Random seed for reproducibility (optional)
 * 
 * Compilation:
 *   nvcc -O3 -arch=sm_30 generate_ppm_gpu.cu -o generate_ppm_gpu
 * 
 * Usage:
 *   ./generate_ppm_gpu
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "dotenv.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 16

// ==================================================================================
// CUDA Kernels
// ==================================================================================

/**
 * Initialize curand states for random number generation
 */
__global__ void init_curand_states(curandState *states, unsigned int seed, 
                                   int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        curand_init(seed, idx, 0, &states[idx]);
    }
}

/**
 * Generate random RGB pixel data on GPU
 */
__global__ void generate_random_image(unsigned char *output, 
                                      curandState *states,
                                      int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        curandState localState = states[idx];
        
        // Generate 3 random bytes (RGB)
        unsigned char r = (unsigned char)(curand_uniform(&localState) * 255.0f);
        unsigned char g = (unsigned char)(curand_uniform(&localState) * 255.0f);
        unsigned char b = (unsigned char)(curand_uniform(&localState) * 255.0f);
        
        // Store in output buffer
        int pixel_idx = idx * 3;
        output[pixel_idx]     = r;
        output[pixel_idx + 1] = g;
        output[pixel_idx + 2] = b;
        
        states[idx] = localState;
    }
}

/**
 * Generate pattern-based image (for debugging/testing)
 */
__global__ void generate_pattern_image(unsigned char *output, 
                                       int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        int pixel_idx = idx * 3;
        
        // Create a gradient pattern
        unsigned char r = (unsigned char)((float)x / width * 255.0f);
        unsigned char g = (unsigned char)((float)y / height * 255.0f);
        unsigned char b = (unsigned char)(((float)x + y) / (width + height) * 255.0f);
        
        output[pixel_idx]     = r;
        output[pixel_idx + 1] = g;
        output[pixel_idx + 2] = b;
    }
}

// ==================================================================================
// CPU Helper Functions
// ==================================================================================

/**
 * Get environment variable as integer
 */
int get_env_int(const char *name, int default_value)
{
    const char *value = getenv(name);
    if (value == NULL) {
        return default_value;
    }
    return atoi(value);
}

/**
 * Get environment variable as string
 */
const char* get_env_string(const char *name, const char *default_value)
{
    const char *value = getenv(name);
    return (value != NULL) ? value : default_value;
}

/**
 * Write PPM file header and pixel data
 */
void write_ppm_file(const char *filename, int width, int height, 
                    const unsigned char *pixel_data, long long total_pixels)
{
    printf("Writing PPM file: %s\n", filename);
    
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open file '%s' for writing\n", filename);
        exit(1);
    }
    
    // Write header
    fprintf(f, "P6\n");
    fprintf(f, "%d %d\n", width, height);
    fprintf(f, "255\n");
    
    // Write pixel data
    size_t bytes_written = fwrite(pixel_data, 1, (size_t)(total_pixels * 3), f);
    
    if (bytes_written != (size_t)(total_pixels * 3)) {
        fprintf(stderr, "Warning: Incomplete write. Expected %lld bytes, wrote %zu bytes\n", 
                total_pixels * 3, bytes_written);
    }
    
    fclose(f);
    printf("✓ PPM file written successfully\n");
}

/**
 * Check GPU memory availability
 */
void check_gpu_memory(long long total_pixels)
{
    size_t free_memory, total_memory;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
    
    // Memory needed: pixel data (RGB) + curand states
    long long pixel_memory = total_pixels * 3;  // RGB bytes
    long long state_memory = total_pixels * sizeof(curandState);
    long long total_needed = pixel_memory + state_memory;
    
    printf("\nGPU Memory Status:\n");
    printf("  Total GPU memory: %.2f GB\n", total_memory / (1024.0 * 1024.0 * 1024.0));
    printf("  Free GPU memory:  %.2f GB\n", free_memory / (1024.0 * 1024.0 * 1024.0));
    printf("  Memory needed:    %.2f GB\n", total_needed / (1024.0 * 1024.0 * 1024.0));
    printf("    - Pixel data:   %.2f GB\n", pixel_memory / (1024.0 * 1024.0 * 1024.0));
    printf("    - curand state: %.2f GB\n", state_memory / (1024.0 * 1024.0 * 1024.0));
    
    if ((long long)free_memory < total_needed) {
        fprintf(stderr, "\n⚠ WARNING: Not enough GPU memory!\n");
        fprintf(stderr, "  Need: %.2f GB, Available: %.2f GB\n",
                total_needed / (1024.0 * 1024.0 * 1024.0),
                free_memory / (1024.0 * 1024.0 * 1024.0));
        fprintf(stderr, "  Consider reducing width or height.\n");
    }
}

/**
 * Print device information
 */
void print_device_info()
{
    int device;
    struct cudaDeviceProp props;
    
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    
    printf("\nGPU Device Information:\n");
    printf("  Device: %s\n", props.name);
    printf("  Compute Capability: %d.%d\n", props.major, props.minor);
    printf("  Total Memory: %.2f GB\n", props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Max Threads per Block: %d\n", props.maxThreadsPerBlock);
}

// ==================================================================================
// Main Function
// ==================================================================================

int main()
{
    printf("=================================================================\n");
    printf("CUDA GPU-Accelerated PPM Image Generator\n");
    printf("=================================================================\n\n");
    
    // Load environment variables from .env file
    dotenv::init();
    
    // Get parameters from environment variables
    const char *output_file = get_env_string("PPM_OUTPUT_FILE", "generated_image.ppm");
    int width = get_env_int("PPM_WIDTH", 1024);
    int height = get_env_int("PPM_HEIGHT", 1024);
    unsigned int seed = (unsigned int)get_env_int("PPM_SEED", (int)time(NULL));
    
    printf("Configuration:\n");
    printf("  Output file: %s\n", output_file);
    printf("  Width:       %d\n", width);
    printf("  Height:      %d\n", height);
    printf("  Seed:        %u\n", seed);
    
    // Validate dimensions
    if (width <= 0 || height <= 0) {
        fprintf(stderr, "Error: Invalid dimensions (width=%d, height=%d)\n", width, height);
        return 1;
    }
    
    long long total_pixels = (long long)width * height;
    
    printf("  Total pixels: %lld (%.2f MP)\n", total_pixels, total_pixels / 1000000.0);
    
    // Print device info
    print_device_info();
    
    // Check GPU memory
    check_gpu_memory(total_pixels);
    
    // Allocate GPU memory
    printf("\nAllocating GPU memory...\n");
    unsigned char *d_output;
    curandState *d_states;
    
    size_t pixel_bytes = total_pixels * 3;
    size_t state_bytes = total_pixels * sizeof(curandState);
    
    CUDA_CHECK(cudaMalloc(&d_output, pixel_bytes));
    CUDA_CHECK(cudaMalloc(&d_states, state_bytes));
    printf("✓ GPU memory allocated\n");
    
    // Calculate grid and block dimensions
    dim3 blockSize(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    printf("  Grid size:  %d x %d\n", gridSize.x, gridSize.y);
    printf("  Block size: %d x %d\n", blockSize.x, blockSize.y);
    
    // Initialize random states
    printf("\nInitializing random number generator states...\n");
    init_curand_states<<<gridSize, blockSize>>>(d_states, seed, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("✓ Random states initialized\n");
    
    // Generate random image on GPU
    printf("Generating random image on GPU...\n");
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    generate_random_image<<<gridSize, blockSize>>>(d_output, d_states, width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("✓ Image generation completed in %.2f ms\n", milliseconds);
    
    // Copy result back to host
    printf("Copying image data from GPU to host...\n");
    unsigned char *h_output = (unsigned char *)malloc(pixel_bytes);
    if (!h_output) {
        fprintf(stderr, "Error: Cannot allocate host memory\n");
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_states));
        return 1;
    }
    
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, pixel_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("✓ GPU to host transfer completed in %.2f ms\n", milliseconds);
    printf("  Transfer rate: %.2f GB/s\n", 
           (pixel_bytes / (1024.0 * 1024.0 * 1024.0)) / (milliseconds / 1000.0));
    
    // Write PPM file
    write_ppm_file(output_file, width, height, h_output, total_pixels);
    
    // Cleanup
    printf("\nCleaning up...\n");
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_states));
    free(h_output);
    
    printf("✓ Done!\n\n");
    
    return 0;
}
