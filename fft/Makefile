# Makefile for CUDA 2D FFT Multi-GPU Implementation

NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++11 -Wall -fopenmp
NVCCFLAGS = -ccbin /opt/gcc-5/bin/gcc -arch=sm_30 -std=c++11 -Xcompiler -fopenmp

# CUDA paths
CUDA_PATH ?= /usr/local/cuda
CUDA_LIB = $(CUDA_PATH)/lib64
CUDA_INCLUDE = $(CUDA_PATH)/include

# cuFFT library
CUFFT_LIB = -L$(CUDA_LIB) -lcufft_static -lculibos
CUFFT_INCLUDE = -I$(CUDA_INCLUDE)

# Build targets
TARGETS = fft_2d_multi_gpu fft_batch_processing

all: $(TARGETS)

fft_2d_multi_gpu: fft_2d_multi_gpu.cu
	$(NVCC) $(NVCCFLAGS) $(CUFFT_INCLUDE) \
	        fft_2d_multi_gpu.cu -o fft_2d_multi_gpu \
	        $(CUFFT_LIB) -lcuda -lcudart

fft_batch_processing: fft_batch_processing.cu
	$(NVCC) $(NVCCFLAGS) $(CUFFT_INCLUDE) \
	        fft_batch_processing.cu -o fft_batch_processing \
	        $(CUFFT_LIB) -lcuda -lcudart

clean:
	rm -f $(TARGETS) *.o *.ppm

run: fft_2d_multi_gpu
	./fft_2d_multi_gpu -o reconstruction.ppm

run-multi: fft_2d_multi_gpu
	./fft_2d_multi_gpu -m -o reconstruction_multi.ppm

run-batch: fft_batch_processing
	./fft_batch_processing 4 256 256

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build all FFT programs"
	@echo "  clean          - Remove build artifacts"
	@echo "  run            - Build and run single GPU FFT"
	@echo "  run-multi      - Build and run multi-GPU FFT"
	@echo "  run-batch      - Build and run batch FFT processing"
	@echo ""
	@echo "Program 1: fft_2d_multi_gpu"
	@echo "  ./fft_2d_multi_gpu [options]"
	@echo "    -m, --multi-gpu   Enable multi-GPU computation"
	@echo "    -o, --output FILE Output filename (default: fft_output.ppm)"
	@echo "    -i, --input FILE  Input image file"
	@echo ""
	@echo "Program 2: fft_batch_processing"
	@echo "  ./fft_batch_processing [batch_size] [width] [height]"
	@echo "    batch_size: images per GPU (default: 4)"
	@echo "    width:      image width (default: 512)"
	@echo "    height:     image height (default: 512)"

.PHONY: all clean run run-multi help
