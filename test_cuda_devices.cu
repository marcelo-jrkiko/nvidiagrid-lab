#include <iostream>
#include <cuda_runtime.h>

// Simple CUDA kernel: each thread writes its thread ID into an array
__global__ void testKernel(int *data) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    data[tid] = tid;
}

int main() {
    int deviceCount = 0;
    int maxThreads = 0;
    int maxMemory = 0;

    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "Failed to get device count: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        std::cout << "\nDevice " << dev << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;

        maxThreads = maxThreads + prop.maxThreadsPerBlock;
        maxMemory = maxMemory + prop.totalGlobalMem;

        // Set current device
        cudaSetDevice(dev);

        // Allocate memory for test
        const int N = 256;
        int *d_data, *h_data;
        h_data = new int[N];
        cudaMalloc((void**)&d_data, N * sizeof(int));

        // Launch kernel
        testKernel<<<N/64, 64>>>(d_data);

        // Copy results back
        cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

        // Verify results
        bool success = true;
        for (int i = 0; i < N; ++i) {
            if (h_data[i] != i) {
                success = false;
                break;
            }
        }

        std::cout << "  Test kernel " << (success ? "PASSED" : "FAILED") << std::endl;

        // Cleanup
        cudaFree(d_data);
        delete[] h_data;
    }

    std::cout << "\nTotal max memory across all devices: " << maxMemory / (1024 * 1024) << " MB" << std::endl;
    std::cout << "\nTotal max threads across all devices: " << maxThreads << std::endl;

    return 0;
}
