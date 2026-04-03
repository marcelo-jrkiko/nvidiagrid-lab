# MNIST Sample for NVIDIA GRID K1

This sample demonstrates neural network training and inference for MNIST digit recognition on NVIDIA GRID K1 GPU.

## Files

- `mnist_lenet.prototxt` - Caffe network definition
- `mnist_solver.prototxt` - Caffe solver configuration
- `train_mnist.py` - Python training script
- `classify_mnist.cu` - CUDA C++ inference implementation
- `download_mnist.sh` - Script to download MNIST dataset

## Prerequisites

## Caffe, CUDA Toolkit and cuDNN
Check the respective documents in the project tree to install the CUDA dependencies.


### Create Virtual Environment with Python 3.7 and Install Libs
```
python3.7 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Quick Start

### 1. Download MNIST Dataset
```bash
bash download_mnist.sh
```

### 2. Train Model (Python + Caffe)
```bash
BATCH_PRESET=MEDIUM python train_mnist.py
```

### 3. Run Inference (CUDA)
```bash
make clean
make all
export BATCH_PRESET=MEDIUM
./classify_mnist
```

Presets can be:
 - TINY , SMALL, MEDIUM
