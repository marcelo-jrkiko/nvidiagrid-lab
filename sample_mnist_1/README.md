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

## How the Training Script Works

The `train_mnist.py` script automates MNIST neural network training with batch size presets, hyperparameter scaling, and multi-GPU support. Here's a step-by-step breakdown:

### 1. **Import and Setup**
```python
# Remove current directory from sys.path before importing caffe
if '' in sys.path:
    sys.path.remove('')
```
- Removes the current directory from Python path to avoid protobuf module conflicts
- Imports required libraries: `caffe`, `numpy`, `re` (regex), `os`, `sys`

### 2. **Configuration Management** (`get_config()`)
```python
preset = os.getenv("BATCH_PRESET", "MEDIUM").upper()
```
- Reads `BATCH_PRESET` environment variable (default: "MEDIUM")
- Maps preset to batch sizes:
  - **TINY**: batch_size=8 (for testing on low-memory systems)
  - **SMALL**: batch_size=32
  - **MEDIUM**: batch_size=64
  - **LARGE**: batch_size=128
- Returns a `Config` object with `batch_size` and other hyperparameters

### 3. **Network Configuration Patching** (`patch_prototxt()`)
The script modifies the network definition file to use the correct batch sizes:
- **Reads** `mnist_lenet.prototxt` (original network definition)
- **Finds** TRAIN and TEST phase sections
- **Replaces** batch sizes:
  - TRAIN phase: uses full batch size (e.g., 64)
  - TEST phase: uses half batch size (e.g., 32) to save memory
- **Writes** patched file as `mnist_lenet.prototxt.patched`

Example modification:
```
Original:  batch_size: 128
Patched:   batch_size: 64    # (for MEDIUM preset)
```

### 4. **Solver Configuration Patching** (`patch_solver()`)
The script adjusts training hyperparameters based on batch size:
- **Scales iterations**: Smaller batches get more iterations
  - Formula: `max_iter = 50000 * (128 / batch_size)`
  - TINY (8): 626,000 iterations | LARGE (128): 50,000 iterations
- **Scales test interval**: When to evaluate accuracy
  - Formula: `test_interval = 500 * (128 / batch_size)`
- **Adjusts learning rate**: Inverse scaling with batch size
  - Formula: `base_lr = 0.01 * (batch_size / 128)`
  - TINY: 0.00078 | LARGE: 0.01
- **Updates network path**: Points to patched network file
- **Writes** patched file as `mnist_solver.prototxt.patched`

### 5. **GPU Configuration** (new multi-GPU support)
```python
gpu_ids_str = os.getenv("GPU_IDS", "0")
gpu_list = [int(x.strip()) for x in gpu_ids_str.split(',')]
```
- Parses `GPU_IDS` environment variable (comma-separated)
- Examples: "0" → [0], "0,1" → [0,1], "0,2,3" → [0,2,3]
- Sets primary GPU to first in list
- Initializes Caffe in GPU mode: `caffe.set_mode_gpu()`

### 6. **Solver Creation**
```python
solver = caffe.SGDSolver(patched_solver)
```
- Creates a Caffe SGDSolver (Stochastic Gradient Descent)
- Loads network and solver configurations
- Initializes network weights and connections

### 7. **Main Training Loop**
```python
for iteration in range(niter):
    solver.step(1)
```
- **`solver.step(1)`**: Performs one training iteration (forward + backward pass)
  - Reads batch from LMDB training data
  - Computes predictions
  - Calculates loss
  - Computes gradients
  - Updates weights

- **Loss reporting** (every 100 iterations):
  ```python
  print("Iteration {}, Loss: {:.6f}".format(iteration, solver.net.blobs['loss'].data))
  ```
  - Shows training loss to monitor convergence
  - Lower loss = model learning better

### 8. **Testing Loop** (periodic evaluation)
```python
if iteration % test_interval == 0 and iteration > 0:
    for test_it in range(100):
        solver.test_nets[0].forward()
        correct += ...
    accuracy = 100.0 * correct / 10000
```
- Runs every `test_interval` iterations
- **Forward pass only** (no weight updates): Evaluates on test set
- Compares model predictions to ground truth labels
- Calculates accuracy: (correct predictions / total) × 100
- Example: "Iteration 5000, Test Accuracy: 98.50%"

### 9. **Cleanup**
```python
if os.path.exists(patched_network):
    os.remove(patched_network)
if os.path.exists(patched_solver):
    os.remove(patched_solver)
```
- Removes temporary patched configuration files
- Keeps workspace clean

## Typical Training Output Example

```
GPU Configuration:
  - GPU IDs: [0]
  - Primary GPU: 0
  - Total GPUs: 1

Patched configuration files for preset: TINY
  - Network: mnist_lenet.prototxt -> mnist_lenet.prototxt.patched
  - Solver: mnist_solver.prototxt -> mnist_solver.prototxt.patched

Training MNIST on GRID K1
Preset: TINY
Batch size: 8 (TRAIN), 8 (TEST)
Max iterations: 626000 iterations
Device: GPU 0
------------------------------------------------------------
Iteration 0, Loss: 2.310000
Iteration 100, Loss: 0.850000
Iteration 200, Loss: 0.420000
...
Iteration 5000, Test Accuracy: 97.50%
Iteration 10000, Test Accuracy: 98.20%
...
Training complete!
Cleaned up temporary patched files
```

## Parameter Scaling Reference

| Preset | Batch Size | Iterations | Test Interval | Learning Rate | Use Case |
|--------|-----------|------------|---------------|---------------|----------|
| TINY | 8 | 626,000 | 4,000 | 0.00078 | Testing, low memory |
| SMALL | 32 | 156,500 | 1,000 | 0.0025 | Validation runs |
| MEDIUM | 64 | 78,250 | 500 | 0.005 | Balanced training |
| LARGE | 128 | 50,000 | 500 | 0.01 | Production (requires more VRAM) |

## Multi-GPU Training

### Important Limitation

The Caffe Python API does **not reliably support synchronous data parallelism** due to GPU memory management complexities. Attempting to manually synchronize gradients and parameters across multiple GPU contexts causes deadlocks and memory access errors.

### Recommended Approaches

#### 1. **Use Caffe's Command-Line Tool** (Recommended - True Multi-GPU Support)

For true multi-GPU training with automatic gradient synchronization, use Caffe's command-line interface:

```bash
# First, setup dataset
cd sample_mnist_1
bash download_mnist.sh
bash convert_mnist_to_lmdb.sh

# Train on multiple GPUs (Caffe handles synchronization)
BATCH_PRESET=TINY caffe train -solver mnist_solver.prototxt -gpu 0,1,2

# Or with environment variable
BATCH_PRESET=MEDIUM caffe train \
    -solver mnist_solver.prototxt \
    -gpu 0,1,2
```

**Advantages:**
- Automatic gradient averaging and synchronization
- True data parallelism with linear speedup
- No Python memory management issues
- Battle-tested in production

#### 2. **Run Sequential Training on Different GPUs** (Python Script)

Run independent training sessions on different GPUs:

```bash
# Terminal 1: Train on GPU 0
GPU_IDS=0 BATCH_PRESET=TINY python3.7 train_mnist.py &

# Terminal 2: Train on GPU 1
GPU_IDS=1 BATCH_PRESET=TINY python3.7 train_mnist.py &

# Terminal 3: Train on GPU 2
GPU_IDS=2 BATCH_PRESET=TINY python3.7 train_mnist.py &

# Wait for all to complete
wait
```

**Advantages:**
- No synchronization overhead
- Trains multiple independent models in parallel
- Simple and reliable
- Easy to run different presets per GPU

#### 3. **Single GPU Training** (Simplest)

```bash
# Use specific GPU
GPU_IDS=0 BATCH_PRESET=MEDIUM python3.7 train_mnist.py

# Use different GPU if GPU 0 is busy
GPU_IDS=1 BATCH_PRESET=MEDIUM python3.7 train_mnist.py
```

### Why Not Manual Data Parallelism in Python?

The script previously attempted manual gradient averaging, but this fails due to:

1. **GPU Context Isolation**: Each Caffe solver runs in its own GPU context. Cross-context access causes synchronization deadlocks.
2. **Memory Pinning**: GPU memory buffers can't be safely accessed from multiple GPU contexts simultaneously.
3. **Hidden Synchronization**: Caffe internally uses CUDA streams that may conflict when managing multiple solvers.
4. **Performance**: Copying large gradient tensors between GPUs through CPU memory is extremely slow.

**Result:** The`average_net_diffs()` function would hang indefinitely trying to synchronize across GPU contexts.

### Performance Comparison

| Method | Speedup | Overhead | Reliability | Ease of Use |
|--------|---------|----------|-------------|-------------|
| Caffe CLI (-gpu) | ~1.8-3.5x | Minimal (~15%) | Excellent | Medium (CLI-specific) |
| Python Sequential | ~1.0x (parallel jobs) | None | Excellent | High (simple loop) |
| Python Data Parallelism | N/A (deadlock) | High | Poor | High (but broken) |
| Single GPU | 1.0x | None | Excellent | Very High |

### Checking Available GPUs

Before running multi-GPU training:

```bash
# List all GPUs
nvidia-smi

# Check GPU memory
nvidia-smi -l 1

# Count GPUs
python3.7 -c "import caffe; caffe.set_mode_gpu(); print('GPUs:', caffe.get_device_count())"
```

### Code Organization

The code is now organized for maintainability:

- **`train_mnist.py`** - Main training script with single and multi-GPU functions
- **`train_utils.py`** - Utility functions:
  - `get_config()` - Batch preset configuration
  - `parse_gpu_ids()` - GPU parsing
  - `patch_prototxt()` / `patch_solver()` - File patching
  - `average_net_diffs()` / `average_net_params()` - Data parallelism core
  - `print_*()` - Output formatting
  - `cleanup_temp_files()` - Cleanup
