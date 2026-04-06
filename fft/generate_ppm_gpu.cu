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
#include <sys/time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <omp.h>
#include "dotenv.h"
#include "fft_common.h"


#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 16

// GPU context structure for multi-GPU processing
typedef struct {
    int gpu_id;
    int start_row;
    int end_row;
    int num_rows;
    int width;
    unsigned char *d_output;
    curandState *d_states;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    float execution_time;
} GPUWork;

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
 * Single GPU generation wrapper
 */
void generate_on_single_gpu(const char *output_file, int width, int height, 
                            unsigned int seed, unsigned char *h_output)
{
    printf("\n=== Single GPU Generation ===\n");
    
    long long total_pixels = (long long)width * height;
    size_t pixel_bytes = total_pixels * 3;
    size_t state_bytes = total_pixels * sizeof(curandState);
    
    printDeviceInfo();
    check_gpu_memory(total_pixels);
    
    // Allocate GPU memory
    printf("\nAllocating GPU memory...\n");
    unsigned char *d_output;
    curandState *d_states;
    
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
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, pixel_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("✓ GPU to host transfer completed in %.2f ms\n", milliseconds);
    printf("  Transfer rate: %.2f GB/s\n", 
           (pixel_bytes / (1024.0 * 1024.0 * 1024.0)) / (milliseconds / 1000.0));
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_states));
}

/**
 * Multi-GPU generation function
 */
void generate_on_multi_gpu(const char *output_file, int width, int height, 
                           unsigned int seed, int num_gpus, unsigned char *h_output)
{
    printf("\n=== Multi-GPU Generation (using %d GPUs) ===\n", num_gpus);
    
    long long total_pixels = (long long)width * height;
    size_t pixel_bytes = total_pixels * 3;
    
    printDeviceInfo();
    check_gpu_memory_multi(num_gpus, total_pixels);
    
    // Allocate GPU work descriptors
    GPUWork **gpu_works = (GPUWork **)malloc(num_gpus * sizeof(GPUWork *));
    
    // Divide work among GPUs (vertical strips)
    int rows_per_gpu = height / num_gpus;
    int remaining_rows = height % num_gpus;
    
    printf("\nAllocating GPU memory...\n");
    printf("  Rows per GPU: %d (with %d remainder rows)\n", rows_per_gpu, remaining_rows);
    
    // OpenMP parallel allocation
    #pragma omp parallel for num_threads(num_gpus)
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        CUDA_CHECK(cudaSetDevice(gpu_id));
        
        // Calculate row range for this GPU
        int start_row = gpu_id * rows_per_gpu + (gpu_id < remaining_rows ? gpu_id : remaining_rows);
        int end_row = start_row + rows_per_gpu + (gpu_id < remaining_rows ? 1 : 0);
        int num_rows = end_row - start_row;
        
        long long gpu_total_pixels = (long long)width * num_rows;
        size_t gpu_pixel_bytes = gpu_total_pixels * 3;
        size_t gpu_state_bytes = gpu_total_pixels * sizeof(curandState);
        
        // Allocate work descriptor
        gpu_works[gpu_id] = (GPUWork *)malloc(sizeof(GPUWork));
        gpu_works[gpu_id]->gpu_id = gpu_id;
        gpu_works[gpu_id]->start_row = start_row;
        gpu_works[gpu_id]->end_row = end_row;
        gpu_works[gpu_id]->num_rows = num_rows;
        gpu_works[gpu_id]->width = width;
        
        // Allocate GPU memory for this work unit
        CUDA_CHECK(cudaMalloc(&gpu_works[gpu_id]->d_output, gpu_pixel_bytes));
        CUDA_CHECK(cudaMalloc(&gpu_works[gpu_id]->d_states, gpu_state_bytes));
        CUDA_CHECK(cudaEventCreate(&gpu_works[gpu_id]->start_event));
        CUDA_CHECK(cudaEventCreate(&gpu_works[gpu_id]->stop_event));
        
        printf("  GPU %d: rows %d-%d (%d rows, %.2f MB)\n", 
               gpu_id, start_row, end_row - 1, num_rows, 
               gpu_pixel_bytes / (1024.0 * 1024.0));
    }
    
    printf("✓ GPU memory allocated\n");
    
    // Initialize and generate on all GPUs in parallel
    double total_gen_time = 0, total_transfer_time = 0;
    
    #pragma omp parallel for num_threads(num_gpus) reduction(+:total_gen_time, total_transfer_time)
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        CUDA_CHECK(cudaSetDevice(gpu_id));
        GPUWork *work = gpu_works[gpu_id];
        
        long long gpu_total_pixels = (long long)work->width * work->num_rows;
        
        // Calculate grid and block dimensions
        dim3 blockSize(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
        dim3 gridSize((work->width + blockSize.x - 1) / blockSize.x,
                      (work->num_rows + blockSize.y - 1) / blockSize.y);
        
        // Initialize random states
        printf("GPU %d: Initializing random states...\n", gpu_id);
        init_curand_states<<<gridSize, blockSize>>>(work->d_states, seed + gpu_id, 
                                                    work->width, work->num_rows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Generate image
        printf("GPU %d: Generating %lld pixels...\n", gpu_id, gpu_total_pixels);
        CUDA_CHECK(cudaEventRecord(work->start_event));
        generate_random_image<<<gridSize, blockSize>>>(work->d_output, work->d_states, 
                                                       work->width, work->num_rows);
        CUDA_CHECK(cudaEventRecord(work->stop_event));
        CUDA_CHECK(cudaEventSynchronize(work->stop_event));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, work->start_event, work->stop_event));
        work->execution_time = milliseconds;
        total_gen_time += milliseconds;
        printf("GPU %d: Generation completed in %.2f ms\n", gpu_id, milliseconds);
    }
    
    // Gather results from all GPUs
    printf("\nGathering results from GPUs...\n");
    
    #pragma omp parallel for num_threads(num_gpus) reduction(+:total_transfer_time)
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        CUDA_CHECK(cudaSetDevice(gpu_id));
        GPUWork *work = gpu_works[gpu_id];
        
        long long gpu_total_pixels = (long long)work->width * work->num_rows;
        size_t gpu_pixel_bytes = gpu_total_pixels * 3;
        
        // Calculate output offset
        long long output_offset = (long long)work->start_row * work->width * 3;
        
        printf("GPU %d: Transferring %lld bytes from device...\n", gpu_id, gpu_pixel_bytes);
        
        CUDA_CHECK(cudaEventRecord(work->start_event));
        CUDA_CHECK(cudaMemcpy(h_output + output_offset, work->d_output, 
                             gpu_pixel_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(work->stop_event));
        CUDA_CHECK(cudaEventSynchronize(work->stop_event));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, work->start_event, work->stop_event));
        total_transfer_time += milliseconds;
        printf("GPU %d: Transfer completed in %.2f ms\n", gpu_id, milliseconds);
    }
    
    // Print performance summary
    printf("\n=== Multi-GPU Performance Summary ===\n");
    printf("  Total generation time: %.2f ms\n", total_gen_time);
    printf("  Total transfer time:   %.2f ms\n", total_transfer_time);
    printf("  Average gen time per GPU: %.2f ms\n", total_gen_time / num_gpus);
    printf("  Average transfer rate per GPU: %.2f GB/s\n", 
           (pixel_bytes / (1024.0 * 1024.0 * 1024.0 * num_gpus)) / (total_transfer_time / num_gpus / 1000.0));
    
    // Cleanup
    printf("\nCleaning up GPU resources...\n");
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        CUDA_CHECK(cudaSetDevice(gpu_id));
        CUDA_CHECK(cudaFree(gpu_works[gpu_id]->d_output));
        CUDA_CHECK(cudaFree(gpu_works[gpu_id]->d_states));
        CUDA_CHECK(cudaEventDestroy(gpu_works[gpu_id]->start_event));
        CUDA_CHECK(cudaEventDestroy(gpu_works[gpu_id]->stop_event));
        free(gpu_works[gpu_id]);
    }
    free(gpu_works);
}

// ==================================================================================
// Main Function
// ==================================================================================

int main()
{
    printf("=================================================================\n");
    printf("CUDA GPU-Accelerated PPM Image Generator (Multi-GPU Support)\n");
    printf("=================================================================\n\n");
    
    // Load environment variables from .env file
    dotenv::init();
    
    // Get parameters from environment variables
    const char *output_file = get_env_string("PPM_OUTPUT_FILE", "generated_image.ppm");
    int width = get_env_int("PPM_WIDTH", 1024);
    int height = get_env_int("PPM_HEIGHT", 1024);
    unsigned int seed = (unsigned int)get_env_int("PPM_SEED", (int)time(NULL));
    int use_multi_gpu = get_env_int("PPM_USE_MULTI_GPU", 0);
    int max_gpus = get_env_int("PPM_MAX_GPUS", 1);
    
    // Validate dimensions
    if (width <= 0 || height <= 0) {
        fprintf(stderr, "Error: Invalid dimensions (width=%d, height=%d)\n", width, height);
        return 1;
    }
    
    // Get actual number of available GPUs
    int available_gpus = getAvailableGPUs();
    
    if (available_gpus == 0) {
        fprintf(stderr, "Error: No CUDA devices found!\n");
        return 1;
    }
    
    // Limit to available GPUs or requested maximum
    int num_gpus = (use_multi_gpu) ? ((max_gpus < available_gpus) ? max_gpus : available_gpus) : 1;
    
    printf("Configuration:\n");
    printf("  Output file:      %s\n", output_file);
    printf("  Width:            %d\n", width);
    printf("  Height:           %d\n", height);
    printf("  Seed:             %u\n", seed);
    printf("  Multi-GPU mode:   %s\n", use_multi_gpu ? "Enabled" : "Disabled");
    printf("  GPUs to use:      %d / %d available\n", num_gpus, available_gpus);
    
    long long total_pixels = (long long)width * height;
    printf("  Total pixels:     %lld (%.2f MP)\n", total_pixels, total_pixels / 1000000.0);
    
    // Start overall timing
    double overall_start = get_current_time_ms();
    printf("\n=== Starting Generation Process ===\n");
    
    // Allocate unified output buffer
    size_t pixel_bytes = total_pixels * 3;
    double alloc_start = get_current_time_ms();
    unsigned char *h_output = (unsigned char *)malloc(pixel_bytes);
    double alloc_time = elapsed_time_ms(alloc_start, get_current_time_ms());
    
    if (!h_output) {
        fprintf(stderr, "Error: Cannot allocate host memory (%.2f GB)\n", 
                pixel_bytes / (1024.0 * 1024.0 * 1024.0));
        return 1;
    }
    printf("  Host memory allocation: %.2f ms\n", alloc_time);
    
    // Generate image using single or multi-GPU mode
    double gen_start = get_current_time_ms();
    if (use_multi_gpu && num_gpus > 1 && available_gpus > 1) {
        generate_on_multi_gpu(output_file, width, height, seed, num_gpus, h_output);
    } else {
        if (use_multi_gpu && available_gpus == 1) {
            printf("⚠ Warning: Multi-GPU mode requested but only 1 GPU available. Using single GPU.\n");
        }
        generate_on_single_gpu(output_file, width, height, seed, h_output);
    }
    double gen_time = elapsed_time_ms(gen_start, get_current_time_ms());
    
    // Write PPM file
    double file_start = get_current_time_ms();
    write_ppm_file(output_file, width, height, h_output, total_pixels);
    double file_time = elapsed_time_ms(file_start, get_current_time_ms());
    
    // Calculate total time
    double overall_end = get_current_time_ms();
    double total_elapsed = elapsed_time_ms(overall_start, overall_end);
    
    // Print timing summary
    printf("\n=== Execution Summary ===\n");
    printf("  Host allocation time: %.2f ms\n", alloc_time);
    printf("  GPU generation time:  %.2f ms\n", gen_time);
    printf("  File write time:      %.2f ms\n", file_time);
    printf("  Total elapsed time:   %.2f ms (%.2f seconds)\n", total_elapsed, total_elapsed / 1000.0);
    printf("  Generation speed:     %.2f MP/s\n", total_pixels / 1000000.0 / (gen_time / 1000.0));
    
    // Cleanup
    printf("\nCleaning up...\n");
    free(h_output);
    
    printf("✓ Done!\n\n");
    
    return 0;
}
