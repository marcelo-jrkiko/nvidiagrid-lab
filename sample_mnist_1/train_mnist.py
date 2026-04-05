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
import re

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

from dotenv import load_dotenv
load_dotenv()

import logging  

LOG_NAME = os.getenv('TRAINING_LOG', 'training.log')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_NAME),
        logging.StreamHandler()
    ]
)

# Global variable to hold the caffe utility path
CAFFE_BIN = os.getenv('CAFFE_BIN', '/usr/local/caffe/build/tools/caffe')

# Patch configuration files
network_prototxt = 'mnist_lenet.prototxt'
solver_prototxt = 'mnist_solver.prototxt'

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
            logging.info("Iteration {}, Loss: {:.6f}".format(
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
            logging.info("Iteration {}, Test Accuracy: {:.2f}%".format(
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
    
    logging.info("Training model on GPUs: {} (parallel execution)".format(gpu_list))
    logging.info("Using Caffe CLI for true multi-GPU training")
    logging.info("NOTE: Testing disabled during multi-GPU training to avoid GPU memory/sync issues")
    logging.info("")
    
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
    
    # Disable testing during multi-GPU training to avoid memory/synchronization issues
    # Read solver and temporarily disable testing
    with open(patched_solver, 'r') as f:
        solver_content = f.read()
    
    # Create a no-test version of the solver
    no_test_solver = patched_solver.replace('.patched', '.no_test.patched')
    solver_no_test = re.sub(r'test_interval:\s*\d+', 'test_interval: 10000000', solver_content)
    
    with open(no_test_solver, 'w') as f:
        f.write(solver_no_test)
    
    # Build caffe train command using no-test solver
    cmd = [
        CAFFE_BIN,
        'train',
        '-solver', no_test_solver,
        '-gpu', gpu_ids
    ]
    
    logging.info("Executing: {}".format(' '.join(cmd)))
    logging.info("-" * 60)
    
    # Execute caffe training command
    try:
        start_time = time.time()
        
        # Start process asynchronously
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Read output in real-time and log it
        for line in iter(process.stdout.readline, ''):
            if line:
                logging.info(line.rstrip('\n'))
        
        # Wait for process to complete
        result = process.wait()
        
        elapsed_time = time.time() - start_time
        
        # Check for errors
        if result != 0:
            # Log error output if training failed
            error = process.stderr.read()            
            logging.error("Caffe training failed with return code {}".format(result))
            logging.error("Error output: {}".format(error))
            sys.exit(1)
        
        logging.info("-" * 60)
        logging.info("Multi-GPU training completed successfully")
        
    except FileNotFoundError:
        logging.error("Error: Caffe binary not found at '{}'".format(CAFFE_BIN))
        logging.error("Please ensure Caffe is installed and CAFFE_BIN environment variable is set correctly")
        sys.exit(1)
    except Exception as e:
        logging.error("Error executing caffe training: {}".format(e))
        sys.exit(1)
    
    return elapsed_time, num_iterations, no_test_solver


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
        logging.info("Created results directory: {}".format(results_dir))
    
    # Find and move model snapshots and solverstates
    model_files = glob.glob(snapshot_prefix + "_iter_*.caffemodel")
    state_files = glob.glob(snapshot_prefix + "_iter_*.solverstate")
    
    for file_path in model_files + state_files:
        if os.path.exists(file_path):
            dest_path = os.path.join(results_dir, os.path.basename(file_path))
            shutil.move(file_path, dest_path)
            logging.info("Moved: {} -> {}".format(file_path, dest_path))
    
    # Copy lenet prototxt for each model file with matching filename
    lenet_prototxt = os.path.join(os.getcwd(), network_prototxt)
    if os.path.exists(lenet_prototxt):
        for model_file in model_files:
            if os.path.exists(model_file):
                # Create prototxt copy with model's base name
                model_basename = os.path.basename(model_file)
                prototxt_name = os.path.splitext(model_basename)[0] + '.prototxt'
                dest_prototxt_path = os.path.join(results_dir, prototxt_name)
                shutil.copy2(lenet_prototxt, dest_prototxt_path)
                logging.info("Copied lenet prototxt: {} -> {}".format(lenet_prototxt, dest_prototxt_path))
    
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
    
    logging.info("\nTraining summary saved to: {}".format(summary_file))
    logging.info("\nTraining Statistics:")
    logging.info("  Total Time: {:.2f} seconds ({:.2f} minutes, {:.2f} hours)".format(
        elapsed_time, elapsed_time / 60.0, elapsed_time / 3600.0))
    logging.info("  Iterations: {}".format(num_iterations))
    logging.info("  Speed: {:.2f} iter/sec ({:.4f} sec/iter)".format(
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
            no_test_solver = None
        else:
            elapsed_time, num_iterations, no_test_solver = train_mnist_multi_gpu(config, gpu_list, patched_solver, patched_network)
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
        logging.error("Error saving training results: {}".format(e))
        sys.exit(1)
    
    # Cleanup
    logging.info("-" * 60)
    logging.info("Training complete!")
    logging.info("Results saved to: {}".format(results_dir))
    cleanup_temp_files(patched_network, patched_solver, no_test_solver)

if __name__ == '__main__':
    logging.info("Starting MNIST training script")
    train_mnist()
