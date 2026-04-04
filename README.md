# KIKO RETRO SPACE - GRID LAB 
I've always been very curious about neural networks, artificial intelligence, and exploring the hardware that runs these algorithms.

So recently (04/2026) I got my hands on some NVIDIA GRID K boards. Although they aren't capable of running modern LLMs, they are capable of running the beginnings of it all.

This project is a collection of small projects I created during this homelab with these GRID K1 and K2 boards.

Here I explore running small algorithms within the GPUs, as well as training small neural networks and visualizing the models obtained from these trainings, exploring MNIST a bit, and other approaches.

*PS:* Yes, I did use AI to help me develop these projects and, most importantly, to document and organize the entire homelab setup process, testing, and how I ran each of the algorithms and SNNs.

## Project Overview
NVIDIA Grid Lab is a comprehensive laboratory environment designed for deep learning experimentation and visualization. This project provides tools and utilities for training, analyzing, and visualizing neural network models, with a particular focus on image classification tasks using popular deep learning frameworks.

## Project Structure

### Sub-Documentation References
- **tools**: Core utility functions and helper modules for model training, data processing, and general operations
- **sample_mnist_1**: A demonstration project implementing MNIST digit classification, showcasing practical deep learning workflows
- **caffe_visualizer**: Advanced visualization suite for analyzing and interpreting Caffe neural network models and their internal representations

## Key Components

### Tools
The `tools` module provides essential utilities including:
- Data preprocessing and loading functions
- Model training utilities and callbacks
- Helper functions for common deep learning operations
- Image processing and augmentation utilities

### Sample MNIST 1
A practical implementation demonstrating:
- Loading and preparing the MNIST handwritten digit dataset
- Building and training a convolutional neural network
- Model evaluation and performance metrics
- Example workflow for digit classification tasks

### Caffe Visualizer
A specialized component for deep learning model analysis:
- Visualization of network architecture and layer configurations
- Feature map visualization from intermediate layers
- Weight and activation analysis
- Model inference and prediction visualization