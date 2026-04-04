"""
Utility functions for MNIST training with multi-GPU support
Handles configuration, file patching, and GPU management
"""

import os
import re
import caffe
import numpy as np


class Config:
    """Configuration struct for training hyperparameters"""
    def __init__(self):
        self.input_size = 784   # 28x28 flattened
        self.output_size = 10   # 10 digits
        self.batch_size = 128
        self.hidden_size = 128
        self.preset = "MEDIUM"


def get_config():
    """Get configuration from environment variables or presets
    
    Returns:
        Config: Configuration object with batch_size and other hyperparameters
    """
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


def parse_gpu_ids():
    """Parse GPU IDs from environment variable
    
    Returns:
        tuple: (gpu_list, primary_gpu, num_gpus)
    """
    gpu_ids_str = os.getenv("GPU_IDS", "0")
    
    try:
        gpu_list = [int(x.strip()) for x in gpu_ids_str.split(',')]
    except ValueError:
        print("Warning: Invalid GPU_IDS format. Using GPU 0.")
        gpu_list = [0]
    
    primary_gpu = gpu_list[0]
    num_gpus = len(gpu_list)
    
    return gpu_list, primary_gpu, num_gpus


def patch_prototxt(prototxt_file, config):
    """Patch prototxt file with batch size from config preset
    
    Creates a temporary patched version for training with modified batch sizes
    for TRAIN and TEST phases.
    
    Args:
        prototxt_file (str): Path to network definition file
        config (Config): Configuration object with batch_size
        
    Returns:
        str: Path to temporary patched file
    """
    if not os.path.exists(prototxt_file):
        raise IOError("Prototxt file not found: {}".format(prototxt_file))
    
    # Read original file
    with open(prototxt_file, 'r') as f:
        content = f.read()
    
    # Determine test batch size (smaller for TEST phase)
    test_batch_size = max(config.batch_size // 2, 8) if config.batch_size > 16 else config.batch_size
    
    # Replace batch sizes in TRAIN and TEST phases
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
    """Patch solver file with hyperparameters from config preset
    
    Adjusts max_iter, test_interval, and learning rate based on batch size.
    Smaller batches get more iterations and lower learning rates.
    
    Args:
        solver_file (str): Path to solver configuration file
        config (Config): Configuration object with batch_size
        
    Returns:
        str: Path to temporary patched file
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


def get_test_interval(patched_solver):
    """Get test_interval from solver file with fallback
    
    Attempts to read from solver.param, falls back to reading file directly.
    
    Args:
        patched_solver (str): Path to patched solver file
        
    Returns:
        int: test_interval value
    """
    try:
        # Try to get from solver param first (may not always work)
        return None
    except:
        pass
    
    # Fallback: read from file
    with open(patched_solver, 'r') as f:
        solver_content = f.read()
    match = re.search(r'test_interval:\s*(\d+)', solver_content)
    return int(match.group(1)) if match else 500


def share_params_to_primary_gpu(solvers, primary_gpu_idx):
    """Copy parameters from primary GPU to other GPUs for synchronization
    
    WARNING: This is only for testing/validation. Not recommended for production
    as it creates heavy GPU synchronization overhead.
    
    Args:
        solvers (list): List of Caffe solvers
        primary_gpu_idx (int): Index of primary GPU in solvers list
    """
    # This approach is not efficient and causes deadlocks with Caffe's GPU memory management
    # It's kept here for reference only
    pass


def print_gpu_config(gpu_list, primary_gpu, num_gpus):
    """Print GPU configuration information
    
    Args:
        gpu_list (list): List of GPU IDs
        primary_gpu (int): Primary GPU ID
        num_gpus (int): Total number of GPUs
    """
    print("GPU Configuration:")
    print("  - GPU IDs: {}".format(gpu_list))
    print("  - Primary GPU: {}".format(primary_gpu))
    print("  - Total GPUs: {}".format(num_gpus))
    if num_gpus > 1:
        print("  - Mode: Data Parallelism (batch split across GPUs)")
    print()


def print_training_config(config, primary_gpu, num_gpus, gpu_list):
    """Print training configuration summary
    
    Args:
        config (Config): Configuration object
        primary_gpu (int): Primary GPU ID
        num_gpus (int): Total number of GPUs
        gpu_list (list): List of GPU IDs
    """
    print("Training MNIST on GRID K1")
    print("Preset: {}".format(config.preset))
    test_batch_size = max(config.batch_size // 2, 8) if config.batch_size > 16 else config.batch_size
    print("Batch size per GPU: {} (TRAIN), {} (TEST)".format(config.batch_size, test_batch_size))
    if num_gpus > 1:
        total_train_batch = config.batch_size * num_gpus
        total_test_batch = test_batch_size * num_gpus
        print("Total batch size (all GPUs): {} (TRAIN), {} (TEST)".format(total_train_batch, total_test_batch))
    print("Devices: {}".format("GPUs {}".format(gpu_list) if num_gpus > 1 else "GPU {}".format(primary_gpu)))
    print("-" * 60)


def cleanup_temp_files(patched_network, patched_solver):
    """Clean up temporary patched configuration files
    
    Args:
        patched_network (str): Path to patched network file
        patched_solver (str): Path to patched solver file
    """
    try:
        if os.path.exists(patched_network):
            os.remove(patched_network)
        if os.path.exists(patched_solver):
            os.remove(patched_solver)
        print("Cleaned up temporary patched files")
    except Exception as e:
        print("Warning: Could not clean up temporary files: {}".format(e))
