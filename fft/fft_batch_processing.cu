/*
 * Advanced Multi-GPU FFT Example with Batch Processing
 * 
 * This example demonstates:
 * 1. Processing multiple images in batch
 * 2. Efficient multi-GPU resource management
 * 3. Asynchronous GPU transfers
 * 4. Performance measurement
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define PI 3.14159265359f

// Utility: Get current time in milliseconds
double getCurrentMs() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// Kernel to create a test pattern
__global__ void createTestPattern(cufftComplex *data, int width, int height, 
                                  int pattern_id) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        float fx = (float)x / width;
        float fy = (float)y / height;
        
        // Different patterns for different pattern IDs
        float value = 0.0f;
        switch(pattern_id) {
            case 0: // Checkerboard
                value = ((int)(fx * 8) + (int)(fy * 8)) % 2 ? 1.0f : 0.0f;
                break;
            case 1: // Sinusoidal
                value = 0.5f + 0.5f * sinf(2 * PI * fx * 3) * cosf(2 * PI * fy * 3);
                break;
            case 2: // Circular gradient
                value = sqrtf(fx*fx + fy*fy) < 0.5f ? 1.0f : 0.0f;
                break;
            case 3: // Random-like
                value = fabsf(sinf(fx * 13.0f) * cosf(fy * 17.0f));
                break;
        }
        
        data[idx].x = value;
        data[idx].y = 0.0f;
    }
}

// Kernel to apply a frequency domain filter
__global__ void applyFrequencyFilter(cufftComplex *fft_data, int width, int height,
                                     float filter_type) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        // Calculate frequency distance from center
        float fx = (x < width/2) ? x : (x - width);
        float fy = (y < height/2) ? y : (y - height);
        float freq = sqrtf(fx*fx + fy*fy);
        
        // Apply different filters
        float attenuation = 1.0f;
        if (filter_type < 1.0f) {  // Low-pass
            attenuation = 1.0f - freq / (sqrtf(width*width + height*height) * 0.3f);
            attenuation = fmaxf(0.0f, attenuation);
        } else if (filter_type < 2.0f) {  // High-pass
            attenuation = 1.0f - attenuation;
        }
        
        fft_data[idx].x *= attenuation;
        fft_data[idx].y *= attenuation;
    }
}

// Structure to hold batch FFT computation data
typedef struct {
    int gpu_id;
    int batch_size;
    int width;
    int height;
    cufftHandle plan;
    cufftComplex **d_batch_input;
    cufftComplex **d_batch_output;
    float *h_input_buffer;
    float *h_output_buffer;
    struct {
        double upload_time;
        double fft_time;
        double download_time;
    } timing;
} BatchFFTContext;

BatchFFTContext* createBatchFFTContext(int gpu_id, int batch_size, 
                                       int width, int height) {
    cudaSetDevice(gpu_id);
    
    BatchFFTContext *ctx = (BatchFFTContext *)malloc(sizeof(BatchFFTContext));
    ctx->gpu_id = gpu_id;
    ctx->batch_size = batch_size;
    ctx->width = width;
    ctx->height = height;
    
    // Create FFT plan
    cufftPlan2d(&ctx->plan, height, width, CUFFT_C2C);
    
    // Allocate GPU memory for batch
    ctx->d_batch_input = (cufftComplex **)malloc(batch_size * sizeof(cufftComplex *));
    ctx->d_batch_output = (cufftComplex **)malloc(batch_size * sizeof(cufftComplex *));
    
    size_t buffer_size = width * height * sizeof(cufftComplex);
    
    for (int i = 0; i < batch_size; i++) {
        cudaMalloc((void **)&ctx->d_batch_input[i], buffer_size);
        cudaMalloc((void **)&ctx->d_batch_output[i], buffer_size);
    }
    
    // CPU buffers for transfers
    ctx->h_input_buffer = (float *)malloc(batch_size * width * height * sizeof(float));
    ctx->h_output_buffer = (float *)malloc(batch_size * width * height * sizeof(float));
    
    printf("GPU %d: Created batch context for %d images (%dx%d)\n", 
           gpu_id, batch_size, width, height);
    
    return ctx;
}

void destroyBatchFFTContext(BatchFFTContext *ctx) {
    if (ctx) {
        cudaSetDevice(ctx->gpu_id);
        cufftDestroy(ctx->plan);
        
        for (int i = 0; i < ctx->batch_size; i++) {
            cudaFree(ctx->d_batch_input[i]);
            cudaFree(ctx->d_batch_output[i]);
        }
        
        free(ctx->d_batch_input);
        free(ctx->d_batch_output);
        free(ctx->h_input_buffer);
        free(ctx->h_output_buffer);
        free(ctx);
    }
}

void processBatchFFT(BatchFFTContext *ctx, int num_images, 
                     float *input_data, float *output_data) {
    cudaSetDevice(ctx->gpu_id);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((ctx->width + blockSize.x - 1) / blockSize.x,
                  (ctx->height + blockSize.y - 1) / blockSize.y);
    
    double t_start = getCurrentMs();
    
    // Upload to GPU
    for (int i = 0; i < num_images; i++) {
        int offset = i * ctx->width * ctx->height;
        
        // Create complex data from float
        cufftComplex *d_temp;
        cudaMalloc((void **)&d_temp, ctx->width * ctx->height * sizeof(cufftComplex));
        
        // Copy data to temp buffer and convert to complex
        cudaMemcpyAsync(ctx->d_batch_input[i], 
                       (void*)(input_data + offset),
                       ctx->width * ctx->height * sizeof(float),
                       cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
    
    double t_upload = getCurrentMs();
    ctx->timing.upload_time = t_upload - t_start;
    
    // Process FFT for each image
    for (int i = 0; i < num_images; i++) {
        // Forward FFT
        cufftExecC2C(ctx->plan, ctx->d_batch_input[i], ctx->d_batch_output[i], 
                    CUFFT_FORWARD);
        
        // Apply filter in frequency domain
        applyFrequencyFilter<<<gridSize, blockSize>>>(ctx->d_batch_output[i],
                                                      ctx->width, ctx->height, 0.5f);
        
        // Inverse FFT
        cufftExecC2C(ctx->plan, ctx->d_batch_output[i], ctx->d_batch_output[i],
                    CUFFT_INVERSE);
    }
    cudaDeviceSynchronize();
    
    double t_fft = getCurrentMs();
    ctx->timing.fft_time = t_fft - t_upload;
    
    // Download results
    for (int i = 0; i < num_images; i++) {
        int offset = i * ctx->width * ctx->height;
        cudaMemcpyAsync((void*)(output_data + offset),
                       ctx->d_batch_output[i],
                       ctx->width * ctx->height * sizeof(cufftComplex),
                       cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    
    double t_download = getCurrentMs();
    ctx->timing.download_time = t_download - t_fft;
}

// ============================================================================
// Multi-GPU Batch Processing
// ============================================================================

void multiGPUBatchProcessing(int num_gpus, int batch_per_gpu, 
                             int width, int height) {
    printf("\n=== Multi-GPU Batch FFT Processing ===\n");
    printf("GPUs: %d | Batch per GPU: %d | Image size: %dx%d\n",
           num_gpus, batch_per_gpu, width, height);
    
    // Create contexts for each GPU
    BatchFFTContext **contexts = (BatchFFTContext **)malloc(num_gpus * sizeof(BatchFFTContext *));
    
    for (int i = 0; i < num_gpus; i++) {
        contexts[i] = createBatchFFTContext(i, batch_per_gpu, width, height);
    }
    
    // Allocate input/output buffers
    int total_images = num_gpus * batch_per_gpu;
    size_t image_size = width * height * sizeof(float);
    
    float *h_input = (float *)malloc(total_images * image_size);
    float *h_output = (float *)malloc(total_images * image_size);
    
    // Generate test patterns
    printf("Generating %d test images...\n", total_images);
    for (int i = 0; i < total_images; i++) {
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int idx = i * width * height + y * width + x;
                float fx = (float)x / width;
                float fy = (float)y / height;
                h_input[idx] = 0.5f + 0.5f * sinf(2 * PI * fx * (i + 1)) * 
                                            cosf(2 * PI * fy * (i + 1));
            }
        }
    }
    
    // Process batches on GPUs
    double t_total_start = getCurrentMs();
    
    for (int gpu_idx = 0; gpu_idx < num_gpus; gpu_idx++) {
        int start_image = gpu_idx * batch_per_gpu;
        float *batch_input = h_input + start_image * width * height;
        float *batch_output = h_output + start_image * width * height;
        
        processBatchFFT(contexts[gpu_idx], batch_per_gpu, batch_input, batch_output);
    }
    
    double t_total_end = getCurrentMs();
    
    // Print timing statistics
    printf("\n--- Timing Statistics ---\n");
    double total_upload = 0, total_fft = 0, total_download = 0;
    
    for (int i = 0; i < num_gpus; i++) {
        printf("GPU %d:\n", i);
        printf("  Upload:   %.2f ms\n", contexts[i]->timing.upload_time);
        printf("  FFT:      %.2f ms\n", contexts[i]->timing.fft_time);
        printf("  Download: %.2f ms\n", contexts[i]->timing.download_time);
        
        total_upload += contexts[i]->timing.upload_time;
        total_fft += contexts[i]->timing.fft_time;
        total_download += contexts[i]->timing.download_time;
    }
    
    printf("\nTotal time: %.2f ms\n", t_total_end - t_total_start);
    printf("Average per GPU:\n");
    printf("  Upload:   %.2f ms\n", total_upload / num_gpus);
    printf("  FFT:      %.2f ms\n", total_fft / num_gpus);
    printf("  Download: %.2f ms\n", total_download / num_gpus);
    
    // Cleanup
    for (int i = 0; i < num_gpus; i++) {
        destroyBatchFFTContext(contexts[i]);
    }
    free(contexts);
    free(h_input);
    free(h_output);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char *argv[]) {
    printf("Advanced Multi-GPU FFT Batch Processing\n");
    printf("========================================\n\n");
    
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    
    if (num_gpus == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }
    
    // Print device information
    printf("Available devices:\n");
    for (int i = 0; i < num_gpus; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("  GPU %d: %s (Compute %d.%d)\n", i, prop.name, prop.major, prop.minor);
    }
    
    // Parse arguments
    int batch_size = 4;
    int image_width = 512;
    int image_height = 512;
    
    if (argc > 1) batch_size = atoi(argv[1]);
    if (argc > 2) image_width = atoi(argv[2]);
    if (argc > 3) image_height = atoi(argv[3]);
    
    // Run multi-GPU batch processing
    multiGPUBatchProcessing(num_gpus, batch_size, image_width, image_height);
    
    printf("\nProcessing complete!\n");
    
    return 0;
}
