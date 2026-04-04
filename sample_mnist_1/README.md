# MNIST Sample for NVIDIA GRID K1

This sample demonstrates neural network training and inference for MNIST digit recognition on NVIDIA GRID K1 GPU.

## Files

- `mnist_lenet.prototxt` - Caffe network definition
- `mnist_solver.prototxt` - Caffe solver configuration
- `train_mnist.py` - Python training script with multi-GPU support
- `convert_mnist_to_lmdb.sh` - Convert MNIST binary files to LMDB format
- `classify_mnist.cu` - CUDA C++ inference implementation
- `download_mnist.sh` - Script to download MNIST dataset
- `setup_caffe_env.sh` - Helper to configure Caffe environment
- `REMOTE_SERVER_SETUP.md` - Remote server setup instructions

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
BATCH_PRESET=TINY python3.7 train_mnist.py
```

### 3. Run Inference (CUDA)
```bash
make clean
make all
export BATCH_PRESET=MEDIUM
./classify_mnist
```

Presets can be:
 - TINY, SMALL, MEDIUM, LARGE

## GPU Configuration

Use the `GPU_IDS` environment variable to select which GPU(s) to use:

```bash
# Single GPU (default GPU 0)
BATCH_PRESET=TINY python3.7 train_mnist.py

# Specific GPU
GPU_IDS=1 BATCH_PRESET=SMALL python3.7 train_mnist.py

# Multiple GPUs
GPU_IDS=0,1 BATCH_PRESET=MEDIUM python3.7 train_mnist.py

# Non-consecutive GPUs
GPU_IDS=0,2,3 BATCH_PRESET=LARGE python3.7 train_mnist.py
```

## Environment Variables

- `BATCH_PRESET` - Batch size preset: TINY (8), SMALL (32), MEDIUM (64), LARGE (128)
- `GPU_IDS` - Comma-separated GPU IDs to use (default: "0")
