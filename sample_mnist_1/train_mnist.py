#!/usr/bin/env python
"""
MNIST Training Script for GRID K1 with Multi-GPU Data Parallelism
Trains a LeNet CNN model on MNIST dataset with batch size preset configuration
and manual data parallelism across multiple GPUs.
"""

import sys
import os
import time
import shutil
import glob
import subprocess

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

# Global variable to hold the caffe utility path
CAFFE_BIN = os.environ.get('CAFFE_BIN', '/usr/local/caffe/build/tools/caffe')

def train_mnist_single_gpu(config, primary_gpu, patched_solver, patched_network):
    """Train on a single GPU (baseline)
    
    Args:
        config: Configuration object
        primary_gpu: GPU ID to use
        patched_solver: Path to patched solver file
        patched_network: Path to patched network file
        
    Returns:
        Tuple of (training_time_seconds, num_iterations)
    """
    start_time = time.time()
    
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
    
    elapsed_time = time.time() - start_time
    return elapsed_time, niter


def train_mnist_multi_gpu(config, gpu_list, patched_solver, patched_network):
    """Train with parallel training on multiple GPUs using Caffe CLI
    
    This implementation uses the Caffe command-line tool to achieve true
    multi-GPU training with parallel execution across all specified GPUs.
    
    Args:
        config: Configuration object
        gpu_list: List of GPU IDs to use
        patched_solver: Path to patched solver file
        patched_network: Path to patched network file
        
    Returns:
        Tuple of (training_time_seconds, num_iterations)
    """
    num_gpus = len(gpu_list)
    gpu_ids = ','.join(map(str, gpu_list))
    
    print("Training model on GPUs: {} (parallel execution)".format(gpu_list))
    print("Using Caffe CLI for true multi-GPU training")
    print()
    
    # Get max iterations from solver
    # Parse solver file to extract max_iter
    num_iterations = None
    with open(patched_solver, 'r') as f:
        for line in f:
            if 'max_iter' in line:
                num_iterations = int(line.split(':')[1].strip())
                break
    
    if num_iterations is None:
        num_iterations = 50000  # Default fallback
    
    # Build caffe train command
    cmd = [
        CAFFE_BIN,
        'train',
        '-solver', patched_solver,
        '-gpu', gpu_ids
    ]
    
    print("Executing: {}".format(' '.join(cmd)))
    print("-" * 60)
    
    # Execute caffe training command
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        elapsed_time = time.time() - start_time
        
        # Print the output from caffe
        if result.stdout:
            print(result.stdout)
        
        # Check for errors
        if result.returncode != 0:
            print("Error: Caffe training failed with return code {}".format(result.returncode))
            sys.exit(1)
        
        print("-" * 60)
        print("Multi-GPU training completed successfully")
        
    except FileNotFoundError:
        print("Error: Caffe binary not found at '{}'".format(CAFFE_BIN))
        print("Please ensure Caffe is installed and CAFFE_BIN environment variable is set correctly")
        sys.exit(1)
    except Exception as e:
        print("Error executing caffe training: {}".format(e))
        sys.exit(1)
    
    return elapsed_time, num_iterations


def save_training_results(results_dir, elapsed_time, num_iterations, config, 
                          primary_gpu, patched_solver, snapshot_prefix):
    """Save training results and model snapshots to results directory
    
    Args:
        results_dir: Path to results directory
        elapsed_time: Training time in seconds
        num_iterations: Number of training iterations
        config: Configuration object
        primary_gpu: Primary GPU used
        patched_solver: Path to patched solver file
        snapshot_prefix: Prefix for model snapshots
    """
    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print("Created results directory: {}".format(results_dir))
    
    # Find and copy model snapshots and solverstates
    model_files = glob.glob(snapshot_prefix + "_iter_*.caffemodel")
    state_files = glob.glob(snapshot_prefix + "_iter_*.solverstate")
    
    for file_path in model_files + state_files:
        if os.path.exists(file_path):
            dest_path = os.path.join(results_dir, os.path.basename(file_path))
            shutil.copy2(file_path, dest_path)
            print("Copied: {} -> {}".format(file_path, dest_path))
    
    # Create training summary file
    summary_file = os.path.join(results_dir, "training_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("MNIST Training Results Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write("Training Configuration\n")
        f.write("-" * 60 + "\n")
        f.write("Preset: {}\n".format(config.preset))
        f.write("Primary GPU: {}\n".format(primary_gpu))
        f.write("Max Iterations: {}\n".format(num_iterations))
        f.write("\n")
        f.write("Training Results\n")
        f.write("-" * 60 + "\n")
        f.write("Total Training Time: {:.2f} seconds\n".format(elapsed_time))
        f.write("Total Training Time: {:.2f} minutes\n".format(elapsed_time / 60.0))
        f.write("Total Training Time: {:.2f} hours\n".format(elapsed_time / 3600.0))
        f.write("Iterations per second: {:.2f}\n".format(num_iterations / elapsed_time))
        f.write("Seconds per iteration: {:.4f}\n".format(elapsed_time / num_iterations))
        f.write("\n")
        f.write("Output Files\n")
        f.write("-" * 60 + "\n")
        f.write("Model Files: {}\n".format(len(model_files)))
        f.write("Solver State Files: {}\n".format(len(state_files)))
        f.write("Results Directory: {}\n".format(results_dir))
    
    print("\nTraining summary saved to: {}".format(summary_file))
    print("\nTraining Statistics:")
    print("  Total Time: {:.2f} seconds ({:.2f} minutes, {:.2f} hours)".format(
        elapsed_time, elapsed_time / 60.0, elapsed_time / 3600.0))
    print("  Iterations: {}".format(num_iterations))
    print("  Speed: {:.2f} iter/sec ({:.4f} sec/iter)".format(
        num_iterations / elapsed_time, elapsed_time / num_iterations))


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
            elapsed_time, num_iterations = train_mnist_single_gpu(config, primary_gpu, patched_solver, patched_network)
        else:
            elapsed_time, num_iterations = train_mnist_multi_gpu(config, gpu_list, patched_solver, patched_network)
    except Exception as e:
        print("Error during training: {}".format(e))
        sys.exit(1)
    
    # Save results to results folder
    results_dir = os.path.join(os.getcwd(), 'results')
    snapshot_prefix = 'mnist_model'
    
    try:
        save_training_results(
            results_dir, 
            elapsed_time, 
            num_iterations, 
            config, 
            primary_gpu, 
            patched_solver, 
            snapshot_prefix
        )
    except Exception as e:
        print("Error saving training results: {}".format(e))
        sys.exit(1)
    
    # Cleanup
    print("-" * 60)
    print("Training complete!")
    print("Results saved to: {}".format(results_dir))
    cleanup_temp_files(patched_network, patched_solver)

if __name__ == '__main__':
    train_mnist()
