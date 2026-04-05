#!/usr/bin/env python
"""
MNIST Training Script for GRID K1 with Multi-GPU Data Parallelism
Trains a LeNet CNN model on MNIST dataset or a GAN model with batch size preset 
configuration and manual data parallelism across multiple GPUs.

Usage:
    TRAINING_MODE=single python train_mnist.py
    TRAINING_MODE=gan python train_mnist.py
"""

import sys
import os
import logging

# Remove current directory from sys.path before importing caffe to avoid protobuf conflicts
if '' in sys.path:
    sys.path.remove('')
if os.getcwd() in sys.path:
    sys.path.remove(os.getcwd())

import caffe
from train_utils import (
    get_config, parse_gpu_ids, print_gpu_config
)
from train_single_network import train_single_network
from train_gan_network import train_gan_network

from dotenv import load_dotenv
load_dotenv()

LOG_NAME = os.getenv('TRAINING_LOG', 'training.log')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_NAME),
        logging.StreamHandler()
    ]
)



def train_main():
    """Main training entry point"""
    
    # Get training mode from environment variable (default: single)
    training_mode = os.getenv('TRAINING_MODE', 'single').lower()
    
    if training_mode not in ['single', 'gan']:
        logging.error("Invalid TRAINING_MODE. Must be 'single' or 'gan'")
        sys.exit(1)
    
    logging.info("")
    logging.info("MNIST Training Script - {0}".format(training_mode.upper()))
    logging.info("=" * 60)
    
    # Get configuration from preset
    config = get_config()
    
    # Parse GPU configuration
    gpu_list, primary_gpu, num_gpus = parse_gpu_ids()
    
    # Initialize Caffe on primary GPU
    caffe.set_mode_gpu()
    caffe.set_device(primary_gpu)
    
    # Print GPU configuration
    print_gpu_config(gpu_list, primary_gpu, num_gpus)
    
    # Route to appropriate training function
    if training_mode == 'gan':
        train_gan_network(config, gpu_list, primary_gpu, num_gpus)
    else:
        train_single_network(config, gpu_list, primary_gpu, num_gpus)

if __name__ == '__main__':
    logging.info("Starting MNIST training script")
    train_main()
