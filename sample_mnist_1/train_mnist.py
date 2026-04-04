#!/usr/bin/env python
"""
MNIST Training Script for GRID K1
Trains a LeNet CNN model on MNIST dataset with batch size preset configuration
"""

import sys
import os

# Remove current directory from sys.path before importing caffe to avoid protobuf conflicts
if '' in sys.path:
    sys.path.remove('')
if os.getcwd() in sys.path:
    sys.path.remove(os.getcwd())

import caffe
import numpy as np
import re
import shutil

# Configuration struct equivalent to C++ version
class Config:
    def __init__(self):
        self.input_size = 784   # 28x28 flattened
        self.output_size = 10   # 10 digits
        self.batch_size = 128
        self.hidden_size = 128
        self.preset = "MEDIUM"

def get_config():
    """Get configuration from environment variables or presets"""
    config = Config()
    
    # Get preset from environment variable (default: MEDIUM)
    preset = os.getenv("BATCH_PRESET", "MEDIUM").upper()
    config.preset = preset
    
    # Set batch_size and hidden_size based on preset
    if preset == "TINY":
        config.batch_size = 8
        config.hidden_size = 32
    elif preset == "SMALL":
        config.batch_size = 32
        config.hidden_size = 64
    elif preset == "MEDIUM":
        config.batch_size = 64
        config.hidden_size = 96
    elif preset == "LARGE":
        config.batch_size = 128
        config.hidden_size = 128
    else:
        print("Warning: Unknown preset '{}'. Defaulting to MEDIUM.".format(preset))
        config.batch_size = 64
        config.hidden_size = 96
        config.preset = "MEDIUM"
    
    return config

def patch_prototxt(prototxt_file, config):
    """
    Patch prototxt file with batch size from config preset
    Creates a temporary patched version for training
    """
    if not os.path.exists(prototxt_file):
        raise IOError("Prototxt file not found: {}".format(prototxt_file))
    
    # Read original file
    with open(prototxt_file, 'r') as f:
        content = f.read()
    
    # Determine test batch size (smaller for TEST phase)
    test_batch_size = max(config.batch_size // 2, 8) if config.batch_size > 16 else config.batch_size
    
    # Replace batch sizes in TRAIN phase (find "phase: TRAIN" section and update following batch_size)
    # Pattern: find phase: TRAIN, then find the next batch_size within that layer
    lines = content.split('\n')
    patched_lines = []
    in_train_phase = False
    in_test_phase = False
    in_data_layer = False
    phase_batch_set = False
    
    for i, line in enumerate(lines):
        # Track which phase we're in
        if 'phase: TRAIN' in line:
            in_train_phase = True
            in_test_phase = False
            phase_batch_set = False
        elif 'phase: TEST' in line:
            in_test_phase = True
            in_train_phase = False
            phase_batch_set = False
        
        # Track data layers
        if 'type: "Data"' in line:
            in_data_layer = True
        elif line.strip().startswith('layer {'):
            if 'type: "Data"' not in '\n'.join(lines[max(0, i-5):i]):
                in_data_layer = False
        
        # Replace batch_size only in Data layers matching the phase
        if in_data_layer and 'batch_size:' in line and not phase_batch_set:
            if in_train_phase:
                patched_lines.append(re.sub(r'batch_size:\s*\d+', 'batch_size: {}'.format(config.batch_size), line))
                phase_batch_set = True
            elif in_test_phase:
                patched_lines.append(re.sub(r'batch_size:\s*\d+', 'batch_size: {}'.format(test_batch_size), line))
                phase_batch_set = True
            else:
                patched_lines.append(line)
        else:
            patched_lines.append(line)
    
    patched_content = '\n'.join(patched_lines)
    
    # Create patched temporary file
    temp_file = prototxt_file + '.patched'
    with open(temp_file, 'w') as f:
        f.write(patched_content)
    
    return temp_file

def patch_solver(solver_file, config):
    """
    Patch solver file with hyperparameters from config preset
    Creates a temporary patched version for training
    """
    if not os.path.exists(solver_file):
        raise IOError("Solver file not found: {}".format(solver_file))
    
    # Read original file
    with open(solver_file, 'r') as f:
        content = f.read()
    
    # Adjust max_iter and test_interval based on batch size
    # Smaller batches may need more iterations
    scale_factor = 128.0 / config.batch_size
    max_iter = int(50000 * scale_factor)
    test_interval = int(500 * scale_factor)
    
    # Also adjust learning rate slightly for smaller batches
    base_lr = 0.01 * (config.batch_size / 128.0)
    
    # Replace parameters
    content = re.sub(r'max_iter:\s*\d+', 'max_iter: {}'.format(max_iter), content)
    content = re.sub(r'test_interval:\s*\d+', 'test_interval: {}'.format(test_interval), content)
    content = re.sub(r'base_lr:\s*[\d.]+', 'base_lr: {}'.format(base_lr), content)
    
    # Update net parameter to use patched network file
    content = re.sub(r'net:\s*"[^"]+"', 'net: "{}"'.format('mnist_lenet.prototxt.patched'), content)
    
    # Create patched temporary file
    temp_file = solver_file + '.patched'
    with open(temp_file, 'w') as f:
        f.write(content)
    
    return temp_file

def train_mnist():
    """Train MNIST model on GRID K1 with GPU acceleration"""
    
    # Get configuration from preset
    config = get_config()
    
    # Get GPU configuration from environment variables
    gpu_ids_str = os.getenv("GPU_IDS", "0")
    
    # Parse GPU IDs (comma-separated)
    try:
        gpu_list = [int(x.strip()) for x in gpu_ids_str.split(',')]
    except ValueError:
        print("Warning: Invalid GPU_IDS format. Using GPU 0.")
        gpu_list = [0]
    
    primary_gpu = gpu_list[0]
    num_gpus = len(gpu_list)
    
    # Configuration
    caffe.set_mode_gpu()
    caffe.set_device(primary_gpu)  # Use first GPU in list for primary computation
    
    print("GPU Configuration:")
    print("  - GPU IDs: {}".format(gpu_list))
    print("  - Primary GPU: {}".format(primary_gpu))
    print("  - Total GPUs: {}".format(num_gpus))
    print()
    
    # Patch prototxt files based on preset
    network_prototxt = 'mnist_lenet.prototxt'
    solver_prototxt = 'mnist_solver.prototxt'
    
    try:
        patched_network = patch_prototxt(network_prototxt, config)
        patched_solver = patch_solver(solver_prototxt, config)
        
        print("Patched configuration files for preset: {}".format(config.preset))
        print("  - Network: {} -> {}".format(network_prototxt, patched_network))
        print("  - Solver: {} -> {}".format(solver_prototxt, patched_solver))
        
    except IOError as e:
        print("Error: {}".format(e))
        sys.exit(1)
    
    # Create solver with patched configuration
    solver = caffe.SGDSolver(patched_solver)
    
    print("Training MNIST on GRID K1")
    print("Preset: {}".format(config.preset))
    print("Batch size: {} (TRAIN), {} (TEST)".format(
        config.batch_size, 
        max(config.batch_size // 2, 8) if config.batch_size > 16 else config.batch_size
    ))
    print("Max iterations: {} iterations".format(solver.param.max_iter))
    if num_gpus > 1:
        print("Devices: GPUs {}".format(gpu_list))
    else:
        print("Device: GPU {}".format(primary_gpu))
    print("-" * 60)

    
    # Training loop
    niter = solver.param.max_iter
    
    # Get test_interval from solver param, with fallback to reading from file
    try:
        test_interval = solver.param.test_interval
    except AttributeError:
        # Fallback: read test_interval from solver file
        with open(patched_solver, 'r') as f:
            solver_content = f.read()
        match = re.search(r'test_interval:\s*(\d+)', solver_content)
        test_interval = int(match.group(1)) if match else 500
        print("Note: test_interval read from solver file: {}".format(test_interval))
    
    for iteration in range(niter):
        solver.step(1)
        
        if iteration % 100 == 0:
            print("Iteration {}, Loss: {:.6f}".format(
                iteration, 
                solver.net.blobs['loss'].data
            ))
        
        if iteration % test_interval == 0 and iteration > 0:
            # Run testing
            correct = 0
            for test_it in range(100):
                solver.test_nets[0].forward()
                correct += np.sum(
                    solver.test_nets[0].blobs['fc2'].data.argmax(1) ==
                    solver.test_nets[0].blobs['label'].data
                )
            
            accuracy = 100.0 * correct / 10000
            print("Iteration {}, Test Accuracy: {:.2f}%".format(
                iteration, accuracy
            ))
    
    print("-" * 60)
    print("Training complete!")
    print("Model saved as mnist_model_iter_*.caffemodel")
    
    # Clean up temporary patched files
    try:
        if os.path.exists(patched_network):
            os.remove(patched_network)
        if os.path.exists(patched_solver):
            os.remove(patched_solver)
        print("Cleaned up temporary patched files")
    except Exception as e:
        print("Warning: Could not clean up temporary files: {}".format(e))

if __name__ == '__main__':
    train_mnist()
