/*
 * MNIST Classification with CUDA on GRID K1
 * Performs GPU-accelerated inference on MNIST images
 * Medium batch size: 128 samples per inference
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstring>
#include <string>
#include <cstdlib>

// Configuration struct for batch sizes and presets
struct Config {
    int batch_size;
    int input_size;
    int hidden_size;
    int output_size;
    std::string preset;
};

// Get configuration from environment variables or presets
Config getConfig() {
    Config config;
    config.input_size = 784;   // 28x28 flattened
    config.output_size = 10;   // 10 digits
    
    // Get preset from environment variable (default: TINY)
    const char* preset_env = std::getenv("BATCH_PRESET");
    std::string preset = preset_env ? std::string(preset_env) : "TINY";
    
    // Convert to uppercase for comparison
    for (auto& c : preset) c = std::toupper(c);
    config.preset = preset;
    
    // Set batch_size and hidden_size based on preset
    if (preset == "TINY") {
        config.batch_size = 8;
        config.hidden_size = 32;
    } else if (preset == "SMALL") {
        config.batch_size = 32;
        config.hidden_size = 64;
    } else if (preset == "MEDIUM") {
        config.batch_size = 64;
        config.hidden_size = 96;
    } else if (preset == "LARGE") {
        config.batch_size = 128;
        config.hidden_size = 128;
    } else {
        config.batch_size = 8;
        config.hidden_size = 32;
        std::cerr << "Unknown preset '" << preset << "'. Defaulting to TINY." << std::endl;
    }
    
    return config;
}
__global__ void softmax_kernel(float *data, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        float max_val = -1e10f;
        
        // Find max in case for numerical stability
        for (int i = 0; i < num_classes; i++) {
            max_val = fmaxf(max_val, data[idx * num_classes + i]);
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            data[idx * num_classes + i] = expf(data[idx * num_classes + i] - max_val);
            sum += data[idx * num_classes + i];
        }
        
        // Normalize
        for (int i = 0; i < num_classes; i++) {
            data[idx * num_classes + i] /= sum;
        }
    }
}

// Simple fully connected layer kernel
__global__ void fc_kernel(
    float *input, float *weights, float *bias,
    float *output, int batch_size, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * output_size) {
        int batch = idx / output_size;
        int out = idx % output_size;
        
        float sum = bias[out];
        for (int i = 0; i < input_size; i++) {
            sum += input[batch * input_size + i] * 
                   weights[out * input_size + i];
        }
        output[idx] = sum;
    }
}

// ReLU activation kernel
__global__ void relu_kernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// Argmax kernel to get predictions
__global__ void argmax_kernel(
    float *data, int *predictions, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        float max_val = -1e10f;
        int max_idx = 0;
        
        for (int i = 0; i < num_classes; i++) {
            float val = data[idx * num_classes + i];
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        predictions[idx] = max_idx;
    }
}

void print_device_info() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        exit(1);
    }
    
    std::cout << "Number of CUDA devices: " << device_count << std::endl;
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "\nDevice " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." 
                  << prop.minor << std::endl;
        std::cout << "  Global memory: " 
                  << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Max block size: " << prop.maxThreadsPerBlock << std::endl;
    }
}

int main() {
    // Get configuration from environment variables or presets
    Config config = getConfig();
    
    std::cout << "=== MNIST Classification on GRID K1 ===" << std::endl;
    std::cout << "Preset: " << config.preset << std::endl;
    std::cout << "Batch size: " << config.batch_size << std::endl;
    std::cout << "Hidden layer: " << config.hidden_size << std::endl;
    std::cout << "==========================================\n" << std::endl;
    
    // Print GPU information
    print_device_info();
    
    // Set device
    cudaSetDevice(0);
    
    // Use configuration values
    const int batch_size = config.batch_size;
    const int input_size = config.input_size;
    const int hidden_size = config.hidden_size;
    const int output_size = config.output_size;
    
    std::cout << "\nNetwork Configuration:" << std::endl;
    std::cout << "  Input size: " << input_size << std::endl;
    std::cout << "  Hidden layer: " << hidden_size << std::endl;
    std::cout << "  Output size: " << output_size << std::endl;
    
    // Allocate device memory
    float *d_input, *d_hidden, *d_output;
    float *d_w1, *d_b1, *d_w2, *d_b2;
    int *d_predictions;
    
    cudaMalloc(&d_input, batch_size * input_size * sizeof(float));
    cudaMalloc(&d_hidden, batch_size * hidden_size * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_size * sizeof(float));
    
    cudaMalloc(&d_w1, hidden_size * input_size * sizeof(float));
    cudaMalloc(&d_b1, hidden_size * sizeof(float));
    cudaMalloc(&d_w2, output_size * hidden_size * sizeof(float));
    cudaMalloc(&d_b2, output_size * sizeof(float));
    
    cudaMalloc(&d_predictions, batch_size * sizeof(int));
    
    // Initialize with random weights
    std::cout << "\nInitializing weights..." << std::endl;
    
    float *h_w1 = new float[hidden_size * input_size];
    float *h_w2 = new float[output_size * hidden_size];
    
    for (int i = 0; i < hidden_size * input_size; i++) {
        h_w1[i] = static_cast<float>(rand()) / RAND_MAX * 0.1f;
    }
    for (int i = 0; i < output_size * hidden_size; i++) {
        h_w2[i] = static_cast<float>(rand()) / RAND_MAX * 0.1f;
    }
    
    cudaMemcpy(d_w1, h_w1, hidden_size * input_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, h_w2, output_size * hidden_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // Initialize input with random MNIST-like data
    std::cout << "Creating synthetic MNIST batch..." << std::endl;
    float *h_input = new float[batch_size * input_size];
    for (int i = 0; i < batch_size * input_size; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    cudaMemcpy(d_input, h_input, batch_size * input_size * sizeof(float),
               cudaMemcpyHostToDevice);
    
    // Warm-up
    std::cout << "GPU warm-up..." << std::endl;
    for (int i = 0; i < 10; i++) {
        fc_kernel<<<(batch_size * hidden_size + 255) / 256, 256>>>(
            d_input, d_w1, d_b1, d_hidden, batch_size, input_size, hidden_size
        );
    }
    cudaDeviceSynchronize();
    
    // Benchmark inference
    std::cout << "\nRunning inference benchmark (1000 iterations)..." << std::endl;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    for (int iter = 0; iter < 1000; iter++) {
        // Layer 1: FC + ReLU
        fc_kernel<<<(batch_size * hidden_size + 255) / 256, 256>>>(
            d_input, d_w1, d_b1, d_hidden, batch_size, input_size, hidden_size
        );
        relu_kernel<<<(batch_size * hidden_size + 255) / 256, 256>>>(
            d_hidden, batch_size * hidden_size
        );
        
        // Layer 2: FC
        fc_kernel<<<(batch_size * output_size + 255) / 256, 256>>>(
            d_hidden, d_w2, d_b2, d_output, batch_size, hidden_size, output_size
        );
        
        // Softmax and predictions
        softmax_kernel<<<(batch_size + 255) / 256, 256>>>(
            d_output, batch_size, output_size
        );
        argmax_kernel<<<(batch_size + 255) / 256, 256>>>(
            d_output, d_predictions, batch_size, output_size
        );
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    int total_samples = 1000 * batch_size;
    float throughput = total_samples / (milliseconds / 1000.0f);
    
    std::cout << "\nBenchmark Results:" << std::endl;
    std::cout << "  Total time: " << milliseconds << " ms" << std::endl;
    std::cout << "  Throughput: " << throughput << " samples/sec" << std::endl;
    std::cout << "  Batch throughput: " << (throughput / batch_size) 
              << " batches/sec" << std::endl;
    
    // Get predictions
    int *h_predictions = new int[batch_size];
    cudaMemcpy(h_predictions, d_predictions, batch_size * sizeof(int),
               cudaMemcpyDeviceToHost);
    
    std::cout << "\nSample predictions (first 10 samples):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "  Sample " << i << ": predicted digit = " 
                  << h_predictions[i] << std::endl;
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_w1);
    cudaFree(d_b1);
    cudaFree(d_w2);
    cudaFree(d_b2);
    cudaFree(d_predictions);
    
    delete[] h_input;
    delete[] h_w1;
    delete[] h_w2;
    delete[] h_predictions;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    std::cout << "\n✓ Classification complete!" << std::endl;
    
    return 0;
}
