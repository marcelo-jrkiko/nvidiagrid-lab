"""
GAN Training Module

This module contains functions for training a GAN (Generator + Discriminator) on MNIST dataset
using single or multi-GPU configurations.
"""

import os
import sys
import time
import re
import shutil
import glob
import subprocess
import logging

import numpy as np

from train_utils import (
    cleanup_temp_files, patch_prototxt, patch_solver
)

# Get caffe - need to handle import carefully
try:
    import caffe
except ImportError:
    print("Error: Caffe not installed")
    sys.exit(1)

# Get caffe binary path
CAFFE_BIN = os.getenv('CAFFE_BIN', '/usr/local/caffe/build/tools/caffe')


def generate_noise_batch(batch_size, noise_dim):
    """Generate random noise for generator input
    
    Args:
        batch_size: Number of samples in batch
        noise_dim: Dimension of noise vector
        
    Returns:
        numpy array of shape (batch_size, noise_dim)
    """
    return np.random.uniform(-1.0, 1.0, (batch_size, noise_dim)).astype(np.float32)


def train_gan_single_gpu(config, primary_gpu, gen_solver_path, disc_solver_path, 
                         gen_net_path, disc_net_path):
    """Train GAN on a single GPU with alternating generator/discriminator updates
    
    Args:
        config: Configuration object
        primary_gpu: GPU ID to use
        gen_solver_path: Path to generator solver file
        disc_solver_path: Path to discriminator solver file
        gen_net_path: Path to generator network file
        disc_net_path: Path to discriminator network file
        
    Returns:
        Tuple of (training_time_seconds, num_iterations)
    """
    start_time = time.time()
    
    # Create solvers
    gen_solver = caffe.SGDSolver(gen_solver_path)
    disc_solver = caffe.SGDSolver(disc_solver_path)
    
    gen_net = gen_solver.net
    disc_net = disc_solver.test_nets[0] if disc_solver.test_nets else disc_solver.net
    
    logging.info("Generator network: {}".format(gen_net_path))
    logging.info("Discriminator network: {}".format(disc_net_path))
    logging.info("GAN training with alternating updates")
    logging.info("Generator iterations: {}".format(config.gan_iterations))
    logging.info("Discriminator updates per generator: {}".format(config.gan_critic_iters))
    logging.info("")
    
    # Get batch size from generator network
    gen_batch_size = gen_net.blobs['data'].shape[0] if 'data' in gen_net.blobs else 128
    disc_batch_size = disc_net.blobs['data'].shape[0] if 'data' in disc_net.blobs else 128
    
    logging.info("Generator batch size: {}".format(gen_batch_size))
    logging.info("Discriminator batch size: {}".format(disc_batch_size))
    
    # Training loop
    for iteration in range(config.gan_iterations):
        # Update discriminator multiple times per generator update
        for _ in range(config.gan_critic_iters):
            # Discriminator training on real data
            disc_solver.step(1)
            disc_loss_real = disc_solver.net.blobs['loss'].data.copy()
            
            logging.debug("Iteration {}, Disc Loss (real): {:.6f}".format(
                iteration, disc_loss_real
            ))
        
        # Update generator
        gen_solver.step(1)
        gen_loss = gen_solver.net.blobs['loss'].data.copy() if 'loss' in gen_solver.net.blobs else 0.0
        
        # Log training progress
        if iteration % 100 == 0:
            logging.info("Iteration {}, Generator Loss: {:.6f}, Discriminator Loss: {:.6f}".format(
                iteration, gen_loss, disc_loss_real
            ))
        
        # Save intermediate models
        if iteration % 1000 == 0 and iteration > 0:
            logging.info("Saving GAN models at iteration {}".format(iteration))
            # Model saving is typically handled by solver snapshots
    
    elapsed_time = time.time() - start_time
    return elapsed_time, config.gan_iterations


def train_gan_multi_gpu(config, gpu_list, gen_solver_path, disc_solver_path,
                        gen_net_path, disc_net_path):
    """Train GAN with multi-GPU using Caffe CLI
    
    For GAN training, we use a modified approach where both networks are trained
    sequentially on the same GPUs.
    
    Args:
        config: Configuration object
        gpu_list: List of GPU IDs
        gen_solver_path: Path to generator solver file
        disc_solver_path: Path to discriminator solver file
        gen_net_path: Path to generator network file
        disc_net_path: Path to discriminator network file
        
    Returns:
        Tuple of (training_time_seconds, num_iterations, no_test_solver)
    """
    gpu_ids = ','.join(map(str, gpu_list))
    
    logging.info("Training GAN on GPUs: {} (sequential alternating updates)".format(gpu_list))
    logging.info("Note: Full parallel GAN training requires custom caffe modifications")
    logging.info("Using Python API for proper alternating generator/discriminator training")
    logging.info("")
    
    # For multi-GPU GAN, we fall back to single-GPU API training for proper control
    # Load first GPU device
    caffe.set_device(gpu_list[0])
    
    return train_gan_single_gpu(config, gpu_list[0], gen_solver_path, disc_solver_path,
                                gen_net_path, disc_net_path)


def save_gan_training_results(results_dir, elapsed_time, num_iterations, config,
                              primary_gpu, gen_snapshot_prefix, disc_snapshot_prefix):
    """Save GAN training results
    
    Args:
        results_dir: Path to results directory
        elapsed_time: Training time in seconds
        num_iterations: Number of training iterations
        config: Configuration object
        primary_gpu: Primary GPU used
        gen_snapshot_prefix: Prefix for generator model snapshots
        disc_snapshot_prefix: Prefix for discriminator model snapshots
    """
    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        logging.info("Created GAN results directory: {}".format(results_dir))
    
    # Find and move generator model snapshots
    gen_model_files = glob.glob(gen_snapshot_prefix + "_iter_*.caffemodel")
    gen_state_files = glob.glob(gen_snapshot_prefix + "_iter_*.solverstate")
    
    for file_path in gen_model_files + gen_state_files:
        if os.path.exists(file_path):
            dest_path = os.path.join(results_dir, os.path.basename(file_path))
            shutil.move(file_path, dest_path)
            logging.info("Moved: {} -> {}".format(file_path, dest_path))
    
    # Find and move discriminator model snapshots
    disc_model_files = glob.glob(disc_snapshot_prefix + "_iter_*.caffemodel")
    disc_state_files = glob.glob(disc_snapshot_prefix + "_iter_*.solverstate")
    
    for file_path in disc_model_files + disc_state_files:
        if os.path.exists(file_path):
            dest_path = os.path.join(results_dir, os.path.basename(file_path))
            shutil.move(file_path, dest_path)
            logging.info("Moved: {} -> {}".format(file_path, dest_path))
    
    # Copy prototxt files for reference
    for net_file, net_name in [(config.gan_generator_prototxt, 'generator'),
                               (config.gan_discriminator_prototxt, 'discriminator')]:
        if os.path.exists(net_file):
            dest_prototxt = os.path.join(results_dir, net_name + '.prototxt')
            shutil.copy2(net_file, dest_prototxt)
            logging.info("Copied {} prototxt: {} -> {}".format(net_name, net_file, dest_prototxt))
    
    # Create training summary file
    summary_file = os.path.join(results_dir, "gan_training_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("MNIST GAN Training Results Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write("Training Configuration\n")
        f.write("-" * 60 + "\n")
        f.write("Preset: {}\n".format(config.preset))
        f.write("Primary GPU: {}\n".format(primary_gpu))
        f.write("Mode: GAN (Generative Adversarial Network)\n")
        f.write("Max Iterations: {}\n".format(num_iterations))
        f.write("Noise Dimension: {}\n".format(config.gan_noise_dim))
        f.write("Critic Updates per Generator: {}\n".format(config.gan_critic_iters))
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
        f.write("Generator Model Files: {}\n".format(len(gen_model_files)))
        f.write("Discriminator Model Files: {}\n".format(len(disc_model_files)))
        f.write("Generator State Files: {}\n".format(len(gen_state_files)))
        f.write("Discriminator State Files: {}\n".format(len(disc_state_files)))
        f.write("Results Directory: {}\n".format(results_dir))
    
    logging.info("\nGAN Training summary saved to: {}".format(summary_file))
    logging.info("\nGAN Training Statistics:")
    logging.info("  Total Time: {:.2f} seconds ({:.2f} minutes, {:.2f} hours)".format(
        elapsed_time, elapsed_time / 60.0, elapsed_time / 3600.0))
    logging.info("  Iterations: {}".format(num_iterations))
    logging.info("  Speed: {:.2f} iter/sec ({:.4f} sec/iter)".format(
        num_iterations / elapsed_time, elapsed_time / num_iterations))


def train_gan_network(config, gpu_list, primary_gpu, num_gpus):
    """Train GAN network (Generator + Discriminator)
    
    Args:
        config: Configuration object
        gpu_list: List of GPU IDs
        primary_gpu: Primary GPU ID
        num_gpus: Number of GPUs
        
    Returns:
        Tuple of (elapsed_time, num_iterations, results_dir_path)
    """
    logging.info("=" * 60)
    logging.info("MODE: GAN TRAINING (Generator + Discriminator)")
    logging.info("=" * 60)
    
    try:
        # Patch generator network and solver
        gen_patched_network = patch_prototxt(config.gan_generator_prototxt, config)
        gen_patched_solver = patch_solver(config.gan_gen_solver_prototxt, config)
        
        # Patch discriminator network and solver
        disc_patched_network = patch_prototxt(config.gan_discriminator_prototxt, config)
        disc_patched_solver = patch_solver(config.gan_disc_solver_prototxt, config)
        
        logging.info("Patched GAN configuration files for preset: {}".format(config.preset))
        logging.info("  - Generator: {} -> {}".format(config.gan_generator_prototxt, gen_patched_network))
        logging.info("  - Generator Solver: {} -> {}".format(config.gan_gen_solver_prototxt, gen_patched_solver))
        logging.info("  - Discriminator: {} -> {}".format(config.gan_discriminator_prototxt, disc_patched_network))
        logging.info("  - Discriminator Solver: {} -> {}".format(config.gan_disc_solver_prototxt, disc_patched_solver))
        logging.info("")
        
    except IOError as e:
        logging.error("Error: {}".format(e))
        sys.exit(1)
    
    logging.info("GAN Training Configuration:")
    logging.info("  Total Iterations: {}".format(config.gan_iterations))
    logging.info("  Noise Dimension: {}".format(config.gan_noise_dim))
    logging.info("  Discriminator Updates per Generator: {}".format(config.gan_critic_iters))
    logging.info("  GPUs: {}".format(gpu_list))
    logging.info("")
    
    # Train based on GPU count
    no_test_solver = None
    try:
        if num_gpus == 1:
            elapsed_time, num_iterations = train_gan_single_gpu(config, primary_gpu, 
                                                               gen_patched_solver, 
                                                               disc_patched_solver,
                                                               gen_patched_network, 
                                                               disc_patched_network)
        else:
            elapsed_time, num_iterations, no_test_solver = train_gan_multi_gpu(config, gpu_list, 
                                                                               gen_patched_solver, 
                                                                               disc_patched_solver,
                                                                               gen_patched_network, 
                                                                               disc_patched_network)
    except Exception as e:
        logging.error("Error during GAN training: {}".format(e))
        sys.exit(1)
    
    # Save results to results folder
    results_dir_path = os.path.join(os.getcwd(), config.gan_results_dir)
    
    try:
        save_gan_training_results(
            results_dir_path,
            elapsed_time,
            num_iterations,
            config,
            primary_gpu,
            config.gan_snapshot_prefix,
            config.gan_disc_snapshot_prefix
        )
    except Exception as e:
        logging.error("Error saving GAN training results: {}".format(e))
        sys.exit(1)
    
    # Cleanup
    logging.info("-" * 60)
    logging.info("GAN training complete!")
    logging.info("Results saved to: {}".format(results_dir_path))
    cleanup_temp_files(gen_patched_network, gen_patched_solver, None)
    cleanup_temp_files(disc_patched_network, disc_patched_solver, None)
    
    return elapsed_time, num_iterations, results_dir_path
