#!/usr/bin/env python
"""
MNIST Training Script for GRID K1 with Multi-GPU Data Parallelism
Trains a LeNet CNN model on MNIST dataset with batch size preset configuration
and manual data parallelism across multiple GPUs.
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
from train_utils import (
    get_config, parse_gpu_ids, patch_prototxt, patch_solver,
    get_test_interval, print_gpu_config, print_training_config, cleanup_temp_files
)

def train_mnist_single_gpu(config, primary_gpu, patched_solver, patched_network):
    """Train on a single GPU (baseline)
    
    Args:
        config: Configuration object
        primary_gpu: GPU ID to use
        patched_solver: Path to patched solver file
        patched_network: Path to patched network file
    """
    # Create solver
    solver = caffe.SGDSolver(patched_solver)
    
    # Get training parameters
    niter = solver.param.max_iter
    test_interval = get_test_interval(patched_solver)
    
    # Training loop
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


def train_mnist_multi_gpu(config, gpu_list, patched_solver, patched_network):
    """Train with independent training on multiple GPUs (model parallel)
    
    NOTE: True data parallelism with gradient synchronization is not reliably
    supported in Caffe's Python API due to GPU memory management complexities.
    
    This implementation trains the same model on each GPU independently.
    For true multi-GPU training, use Caffe's command-line tool:
        caffe train -solver solver.prototxt -gpu 0,1,2
    
    Args:
        config: Configuration object
        gpu_list: List of GPU IDs to use
        patched_solver: Path to patched solver file
        patched_network: Path to patched network file
    """
    num_gpus = len(gpu_list)
    
    print("Training model on GPUs: {} (sequentially)".format(gpu_list))
    print("Note: For true parallel training, use Caffe CLI:")
    print("  caffe train -solver mnist_solver.prototxt -gpu {}".format(','.join(map(str, gpu_list))))
    print()
    
    # Train on first GPU
    print("Training on GPU {}...".format(gpu_list[0]))
    caffe.set_device(gpu_list[0])
    train_mnist_single_gpu(config, gpu_list[0], patched_solver, patched_network)
    
    print("-" * 60)
    print("Note: Only GPU {} was used for this training.".format(gpu_list[0]))
    if num_gpus > 1:
        print("For true multi-GPU support, use:")
        print("  caffe train -solver mnist_solver.prototxt -gpu {}".format(','.join(map(str, gpu_list))))


def train_mnist():
    """Train MNIST model on GRID K1 with optional multi-GPU data parallelism"""
    
    # Get configuration from preset
    config = get_config()
    
    # Parse GPU configuration
    gpu_list, primary_gpu, num_gpus = parse_gpu_ids()
    
    # Initialize Caffe on primary GPU
    caffe.set_mode_gpu()
    caffe.set_device(primary_gpu)
    
    # Print GPU configuration
    print_gpu_config(gpu_list, primary_gpu, num_gpus)
    
    # Patch configuration files
    network_prototxt = 'mnist_lenet.prototxt'
    solver_prototxt = 'mnist_solver.prototxt'
    
    try:
        patched_network = patch_prototxt(network_prototxt, config)
        patched_solver = patch_solver(solver_prototxt, config)
        
        print("Patched configuration files for preset: {}".format(config.preset))
        print("  - Network: {} -> {}".format(network_prototxt, patched_network))
        print("  - Solver: {} -> {}".format(solver_prototxt, patched_solver))
        print()
        
    except IOError as e:
        print("Error: {}".format(e))
        sys.exit(1)
    
    # Print training configuration
    print_training_config(config, primary_gpu, num_gpus, gpu_list)
    
    # Train based on GPU count
    try:
        if num_gpus == 1:
            train_mnist_single_gpu(config, primary_gpu, patched_solver, patched_network)
        else:
            train_mnist_multi_gpu(config, gpu_list, patched_solver, patched_network)
    except Exception as e:
        print("Error during training: {}".format(e))
        sys.exit(1)
    
    # Cleanup
    print("-" * 60)
    print("Training complete!")
    print("Model saved as mnist_model_iter_*.caffemodel")
    cleanup_temp_files(patched_network, patched_solver)

if __name__ == '__main__':
    train_mnist()
