# Makefile for CUDA 2D FFT Multi-GPU Implementation

NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++11 -Wall -fPIC -fopenmp
NVCCFLAGS = -ccbin /opt/gcc-5/bin/gcc -arch=sm_30 -std=c++11 -Xcompiler -fopenmp

# CUDA paths
CUDA_PATH ?= /usr/local/cuda
CUDA_LIB = $(CUDA_PATH)/lib64
CUDA_INCLUDE = $(CUDA_PATH)/include

# cuFFT library
CUFFT_LIB = -L$(CUDA_LIB) -lcufft_static -lculibos
CUFFT_INCLUDE = -I$(CUDA_INCLUDE)

# Common source files
COMMON_SRC = fft_common.cu
COMMON_OBJ = fft_common.o

# Build targets
TARGETS = fft_2d_multi_gpu fft_batch_processing generate_ppm_gpu

all: $(TARGETS)

# Compile common functionality with CUDA compiler
$(COMMON_OBJ): $(COMMON_SRC)
	$(NVCC) $(NVCCFLAGS) $(CUFFT_INCLUDE) -c $< -o $@

fft_2d_multi_gpu: $(COMMON_OBJ) fft_2d_multi_gpu.cu
	$(NVCC) $(NVCCFLAGS) $(CUFFT_INCLUDE) \
	        fft_2d_multi_gpu.cu $(COMMON_OBJ) -o fft_2d_multi_gpu \
	        $(CUFFT_LIB) -lcuda -lcudart

fft_batch_processing: $(COMMON_OBJ) fft_batch_processing.cu
	$(NVCC) $(NVCCFLAGS) $(CUFFT_INCLUDE) \
	        fft_batch_processing.cu $(COMMON_OBJ) -o fft_batch_processing \
	        $(CUFFT_LIB) -lcuda -lcudart

generate_ppm_gpu: generate_ppm_gpu.cu
	$(NVCC) $(NVCCFLAGS) -lcurand -lcuda -lcudart \
	        generate_ppm_gpu.cu -o generate_ppm_gpu

clean:
	rm -f $(TARGETS) *.o *.ppm

run: fft_2d_multi_gpu
	FFT_OUTPUT_FILE=reconstruction.ppm ./fft_2d_multi_gpu

run-multi: fft_2d_multi_gpu
	FFT_USE_MULTI_GPU=1 FFT_OUTPUT_FILE=reconstruction_multi.ppm ./fft_2d_multi_gpu

run-batch: fft_batch_processing
	FFT_BATCH_SIZE=4 FFT_IMAGE_WIDTH=256 FFT_IMAGE_HEIGHT=256 ./fft_batch_processing

run-generate: generate_ppm_gpu
	PPM_WIDTH=1024 PPM_HEIGHT=1024 PPM_OUTPUT_FILE=generated.ppm ./generate_ppm_gpu

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build all programs"
	@echo "  clean          - Remove build artifacts"
	@echo "  run            - Build and run single GPU FFT (environment variable: FFT_OUTPUT_FILE)"
	@echo "  run-multi      - Build and run multi-GPU FFT (environment variables: FFT_USE_MULTI_GPU, FFT_OUTPUT_FILE)"
	@echo "  run-batch      - Build and run batch processing (environment variables: FFT_BATCH_SIZE, FFT_IMAGE_WIDTH, FFT_IMAGE_HEIGHT)"
	@echo "  run-generate   - Build and run GPU PPM image generator (environment variables: PPM_WIDTH, PPM_HEIGHT, PPM_OUTPUT_FILE)"
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
	@echo ""
	@echo "Program 3: generate_ppm_gpu"
	@echo "  ./generate_ppm_gpu (reads environment variables from .env)"
	@echo "    PPM_OUTPUT_FILE: Output filename (default: generated_image.ppm)"
	@echo "    PPM_WIDTH:       Image width (default: 1024)"
	@echo "    PPM_HEIGHT:      Image height (default: 1024)"
	@echo "    PPM_SEED:        Random seed (default: current time)

.PHONY: all clean run run-multi run-batch run-generate help
