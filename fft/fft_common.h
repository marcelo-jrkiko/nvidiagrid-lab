#ifndef FFT_COMMON_H
#define FFT_COMMON_H

#include <cuda_runtime.h>
#include <cufft.h>

#define PI 3.14159265359f

// Image data structure for PPM format
typedef struct {
    int width;
    int height;
    unsigned char *data;  // RGB format: each pixel is 3 bytes (R,G,B)
} ImageData;

// GPU context for single GPU FFT computation
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

// Configuration struct for fft_2d_multi_gpu
typedef struct {
    int use_multi_gpu;
    const char *input_file;
    const char *output_file;
    int test_image_size; // Added to specify size of generated test image
    int max_gpus; // Maximum number of GPUs to use in multi-GPU mode
    const char *results_folder; // Folder where results will be saved
    const char *output_prefix; // Prefix for output filenames
} FFTConfig;

// Configuration struct for batch processing
typedef struct {
    int batch_size;
    int image_width;
    int image_height;
} BatchConfig;

// ================================================================================
// Image utility functions
// ================================================================================

ImageData* createImage(int width, int height);
void freeImage(ImageData *img);
ImageData* generateTestImage(int width, int height);
ImageData* loadPPM(const char *filename);
void savePPM(const char *filename, const unsigned char *data, int width, int height);

// ================================================================================
// GPU management utility functions
// ================================================================================

void printDeviceInfo();
int getAvailableGPUs();
double getCurrentMs();
int createDirectoryIfNotExists(const char *path);
void buildOutputPath(const char *results_folder, const char *prefix, const char *filename, char *full_path, size_t max_len);
void check_gpu_memory_multi(int num_gpus, long long total_pixels);
void check_gpu_memory(long long total_pixels);

// ================================================================================
// GPU context management
// ================================================================================

GPUContext* allocateGPUContext(int gpu_id, int width, int height);
void freeGPUContext(GPUContext *ctx);

// ================================================================================
// GPU kernels
// ================================================================================

__global__ void rgbToFloat(const unsigned char *rgb_data, float *r, float *g, float *b,
                          int width, int height);
__global__ void floatToRGB(const float *r, const float *g, const float *b,
                          unsigned char *rgb_data, int width, int height);
__global__ void floatToComplex(const float *input, cufftComplex *output,
                              int width, int height);
__global__ void complexToFloat(const cufftComplex *input, float *output,
                              int width, int height, float scale);
__global__ void complexToFloatReal(const cufftComplex *input, float *output,
                                  int width, int height, float scale);

// ================================================================================
// Configuration loading functions
// ================================================================================

FFTConfig loadFFTConfig();
BatchConfig loadBatchConfig();


#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

#endif // FFT_COMMON_H
