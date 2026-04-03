#!/usr/bin/env python
"""
MNIST Training Script for GRID K1
Trains a LeNet CNN model on MNIST dataset with medium batch size (128)
"""

import caffe
import numpy as np
import os
import sys

def train_mnist():
    """Train MNIST model on GRID K1 with GPU acceleration"""
    
    # Configuration
    caffe.set_mode_gpu()
    caffe.set_device(0)  # Use first GPU
    
    solver_prototxt = 'mnist_solver.prototxt'
    
    if not os.path.exists(solver_prototxt):
        print("Error: {} not found".format(solver_prototxt))
        sys.exit(1)
    
    # Create solver
    solver = caffe.SGDSolver(solver_prototxt)
    
    print("Training MNIST on GRID K1")
    print("Batch size: 128 (medium impact)")
    print("Max iterations: 50000")
    print("Device: GPU 0")
    print("-" * 60)
    
    # Training loop
    niter = 50000
    test_interval = 500
    
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

if __name__ == '__main__':
    train_mnist()
