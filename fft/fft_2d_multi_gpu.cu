#include "fft_common.h"
#include <stdio.h>
#include <omp.h>
#include "dotenv.h"

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

void performFFTReconstruction(const FFTConfig *config) {
    printf("\n=== FFT 2D Image Reconstruction ===\n");
    
    // Create results folder if it doesn't exist
    if (createDirectoryIfNotExists(config->results_folder) != 0) {
        fprintf(stderr, "Warning: Could not create results folder %s\n", config->results_folder);
    }
    
    // Generate test image if no input file
    ImageData *img = NULL;
    bool generated_image = false;

    if (config->input_file == NULL) {
        generated_image = true;
    } else {
        printf("Loading image from: %s\n", config->input_file);
        img = loadPPM(config->input_file);
        if (img == NULL) {
            fprintf(stderr, "Failed to load image from %s, generating test image instead\n", config->input_file);
            generated_image = true;
        }
    }

    if (generated_image)
    {        
        const int size = config->test_image_size;
        printf("Generating test image (%dx%d)...\n", size, size);
        img = generateTestImage(size, size);
    }
    
    
    int num_gpus = config->max_gpus;
    printf("Available GPUs: %d\n", num_gpus);
    
    if (num_gpus == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        freeImage(img);
        return;
    }
    
    // Allocate result image
    unsigned char *result = (unsigned char *)malloc(img->width * img->height * 3);
    
    if (config->use_multi_gpu && num_gpus > 1) {
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
    if (config->output_file) {
        char original_output[512];
        char reconstructed_output[512];
        
        // Build output file paths
        buildOutputPath(config->results_folder, config->output_prefix, config->output_file, reconstructed_output, sizeof(reconstructed_output));
        
        // Build original file path: original_{prefix}{output_file}
        char original_filename[256];
        snprintf(original_filename, sizeof(original_filename), "original_%s", config->output_file);
        buildOutputPath(config->results_folder, config->output_prefix, original_filename, original_output, sizeof(original_output));
        
        // Save the original image for comparison
        savePPM(original_output, img->data, img->width, img->height);
        
        // Save the reconstructed image
        savePPM(reconstructed_output, result, img->width, img->height);
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
    
    dotenv::init(); // Load environment variables from .env file if present
    printDeviceInfo();
    
    // Load configuration from environment variables
    FFTConfig config = loadFFTConfig();
    
    printf("\nConfiguration (from environment variables):\n");
    printf("  Multi-GPU: %s (FFT_USE_MULTI_GPU=%d)\n", 
           config.use_multi_gpu ? "Enabled" : "Disabled", config.use_multi_gpu);
    printf("  Output: %s (FFT_OUTPUT_FILE=%s)\n", config.output_file, config.output_file ? config.output_file : "not set");
    printf("  Input: %s (FFT_INPUT_FILE=%s)\n", 
           config.input_file ? config.input_file : "(generated)", 
           config.input_file ? config.input_file : "not set");
    printf("  Test Image Size: %dx%d (FFT_TEST_IMAGE_SIZE=%d)\n", 
           config.test_image_size, config.test_image_size, config.test_image_size);
    printf("  Results Folder: %s (FFT_RESULTS_FOLDER=%s)\n", 
           config.results_folder, config.results_folder);
    printf("  Output Prefix: %s (FFT_OUTPUT_PREFIX=%s)\n", 
           strlen(config.output_prefix) > 0 ? config.output_prefix : "(none)", 
           strlen(config.output_prefix) > 0 ? config.output_prefix : "not set");
    
    performFFTReconstruction(&config);
    
    return 0;
}
