# MNIST Sample for NVIDIA GRID K1

This sample demonstrates neural network training and inference for MNIST digit recognition on NVIDIA GRID K1 GPU.

## Files

- `mnist_lenet.prototxt` - Caffe network definition
- `mnist_solver.prototxt` - Caffe solver configuration
- `train_mnist.py` - Python training script
- `classify_mnist.cu` - CUDA C++ inference implementation
- `download_mnist.sh` - Script to download MNIST dataset

## Prerequisites

```bash
# On Debian 9
sudo apt-get install caffe-cuda python-caffe
sudo apt-get install cuda-toolkit
```

## Quick Start

### 1. Download MNIST Dataset
```bash
bash download_mnist.sh
```

### 2. Train Model (Python + Caffe)
```bash
python train_mnist.py
```

### 3. Run Inference (CUDA)
```bash
nvcc classify_mnist.cu -o classify_mnist
export BATCH_PRESET=MEDIUM
./classify_mnist
```

Presets can be:
 - TINY , SMALL, MEDIUM
