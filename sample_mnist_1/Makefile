# Makefile for MNIST GRID K1 Sample

NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++11 -Wall
NVCCFLAGS = -arch=sm_35 -std=c++11

# CUDA paths
CUDA_PATH ?= /usr/local/cuda
CUDA_LIB = $(CUDA_PATH)/lib64
CUDA_INCLUDE = $(CUDA_PATH)/include

# cuBLAS
CUBLAS_LIB = -L$(CUDA_LIB) -lcublas_static -lculibos
CUBLAS_INCLUDE = -I$(CUDA_INCLUDE)

# Build targets
TARGETS = classify_mnist

all: $(TARGETS)

classify_mnist: classify_mnist.cu
	$(NVCC) $(NVCCFLAGS) $(CUBLAS_INCLUDE) -L$(CUDA_LIB) \
	        classify_mnist.cu -o classify_mnist \
	        -lcuda -lcudart

clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean
