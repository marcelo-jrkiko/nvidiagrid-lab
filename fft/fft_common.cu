#include "fft_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <errno.h>

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

double getCurrentMs() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// Create directory if it doesn't exist
int createDirectoryIfNotExists(const char *path) {
    if (!path) return -1;
    
    struct stat st = {0};
    if (stat(path, &st) == 0) {
        // Path exists
        if (S_ISDIR(st.st_mode)) {
            // It's a directory, all good
            return 0;
        } else {
            // It exists but is not a directory
            fprintf(stderr, "Error: %s exists but is not a directory\n", path);
            return -1;
        }
    }
    
    // Directory doesn't exist, create it
    #ifdef _WIN32
        if (_mkdir(path) != 0) {
            fprintf(stderr, "Failed to create directory: %s\n", path);
            return -1;
        }
    #else
        if (mkdir(path, 0755) != 0) {
            fprintf(stderr, "Failed to create directory: %s\n", path);
            perror("mkdir");
            return -1;
        }
    #endif
    
    return 0;
}

// Build full output path with folder and prefix
void buildOutputPath(const char *results_folder, const char *prefix, const char *filename, char *full_path, size_t max_len) {
    if (!results_folder || !filename || !full_path || max_len == 0) {
        return;
    }
    
    if (prefix && strlen(prefix) > 0) {
        snprintf(full_path, max_len, "%s/%s%s", results_folder, prefix, filename);
    } else {
        snprintf(full_path, max_len, "%s/%s", results_folder, filename);
    }
}

// ================================================================================
// Simple PPM image format for I/O
// ================================================================================

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

ImageData* loadPPM(const char *filename) {
    if (!filename) {
        fprintf(stderr, "Error: filename is NULL\n");
        return NULL;
    }
    
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open file %s for reading\n", filename);
        return NULL;
    }
    
    // Read PPM header
    char magic[3];
    if (fread(magic, 1, 2, f) != 2) {
        fprintf(stderr, "Error: Cannot read PPM magic number from %s\n", filename);
        fclose(f);
        return NULL;
    }
    magic[2] = '\0';
    
    // Skip whitespace and comments
    int c;
    while ((c = fgetc(f)) != EOF && (isspace(c) || c == '#')) {
        if (c == '#') {
            // Skip comment line
            while ((c = fgetc(f)) != EOF && c != '\n');
        }
    }
    ungetc(c, f);
    
    int width, height, max_val;
    if (fscanf(f, "%d %d %d", &width, &height, &max_val) != 3) {
        fprintf(stderr, "Error: Cannot read PPM dimensions from %s\n", filename);
        fclose(f);
        return NULL;
    }
    
    // Read one whitespace character after max_val
    fgetc(f);
    
    if (strcmp(magic, "P6") != 0) {
        fprintf(stderr, "Error: File %s is not a valid P6 PPM file\n", filename);
        fclose(f);
        return NULL;
    }
    
    if (max_val != 255) {
        fprintf(stderr, "Warning: PPM file %s has max_val=%d (expected 255), attempting to load anyway\n", 
                filename, max_val);
    }
    
    if (width <= 0 || height <= 0) {
        fprintf(stderr, "Error: Invalid image dimensions %dx%d in %s\n", width, height, filename);
        fclose(f);
        return NULL;
    }
    
    // Create image structure
    ImageData *img = createImage(width, height);
    
    // Read image data
    size_t expected_bytes = (size_t)width * height * 3;
    size_t bytes_read = fread(img->data, 1, expected_bytes, f);
    
    if (bytes_read != expected_bytes) {
        fprintf(stderr, "Warning: Expected %zu bytes but read %zu bytes from %s\n", 
                expected_bytes, bytes_read, filename);
    }
    
    fclose(f);
    printf("Loaded image from %s (%dx%d)\n", filename, width, height);
    
    return img;
}

// ================================================================================
// GPU Memory allocation and transfer with multi-GPU support
// ================================================================================

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

// ================================================================================
// GPU Kernels
// ================================================================================

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
// Configuration loading from environment variables
// ================================================================================

FFTConfig loadFFTConfig() {
    FFTConfig config;
    config.use_multi_gpu = 0;
    config.input_file = NULL;
    config.output_file = "fft_output.ppm";
    config.test_image_size = 256;
    config.max_gpus = 1; // Default max GPUs to use
    config.results_folder = "./results"; // Default results folder
    config.output_prefix = ""; // Default: no prefix
    
    // Load use_multi_gpu from environment
    const char *env_multi_gpu = getenv("FFT_USE_MULTI_GPU");
    if (env_multi_gpu != NULL) {
        config.use_multi_gpu = atoi(env_multi_gpu);
    }
    
    // Load input file from environment
    const char *env_input = getenv("FFT_INPUT_FILE");
    if (env_input != NULL) {
        config.input_file = env_input;
    }
    
    // Load test image size from environment
    const char *env_test_image_size = getenv("FFT_TEST_IMAGE_SIZE");
    if (env_test_image_size != NULL) {
        config.test_image_size = atoi(env_test_image_size);
    }

    // Load output file from environment
    const char *env_output = getenv("FFT_OUTPUT_FILE");
    if (env_output != NULL) {
        config.output_file = env_output;
    }

    // Load max GPUs to use from environment
    const char *env_max_gpus = getenv("FFT_MAX_GPUS");
    if (env_max_gpus != NULL) {
        config.max_gpus = atoi(env_max_gpus);
    }
    else {
        config.max_gpus = getAvailableGPUs(); // Default to all available GPUs if not set
    }
    
    // Load results folder from environment
    const char *env_results_folder = getenv("FFT_RESULTS_FOLDER");
    if (env_results_folder != NULL) {
        config.results_folder = env_results_folder;
    }
    
    // Load output prefix from environment
    const char *env_output_prefix = getenv("FFT_OUTPUT_PREFIX");
    if (env_output_prefix != NULL) {
        config.output_prefix = env_output_prefix;
    }
    
    return config;
}

BatchConfig loadBatchConfig() {
    BatchConfig config;
    config.batch_size = 4;
    config.image_width = 512;
    config.image_height = 512;
    
    // Load batch size from environment
    const char *env_batch = getenv("FFT_BATCH_SIZE");
    if (env_batch != NULL) {
        config.batch_size = atoi(env_batch);
    }
    
    // Load image width from environment
    const char *env_width = getenv("FFT_IMAGE_WIDTH");
    if (env_width != NULL) {
        config.image_width = atoi(env_width);
    }
    
    // Load image height from environment
    const char *env_height = getenv("FFT_IMAGE_HEIGHT");
    if (env_height != NULL) {
        config.image_height = atoi(env_height);
    }
    
    return config;
}
