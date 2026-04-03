#!/bin/bash
# Download and prepare MNIST dataset for Caffe

set -e

echo "Downloading MNIST dataset..."

# Create data directory
mkdir -p mnist_data
cd mnist_data

# Download MNIST dataset
if [ ! -f train-images-idx3-ubyte.gz ]; then
    echo "Downloading MNIST training images..."
    wget -q http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
fi

if [ ! -f train-labels-idx1-ubyte.gz ]; then
    echo "Downloading MNIST training labels..."
    wget -q http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
fi

if [ ! -f t10k-images-idx3-ubyte.gz ]; then
    echo "Downloading MNIST test images..."
    wget -q http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
fi

if [ ! -f t10k-labels-idx1-ubyte.gz ]; then
    echo "Downloading MNIST test labels..."
    wget -q http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
fi

# Extract
echo "Extracting..."
gunzip -f train-images-idx3-ubyte.gz
gunzip -f train-labels-idx1-ubyte.gz
gunzip -f t10k-images-idx3-ubyte.gz
gunzip -f t10k-labels-idx1-ubyte.gz

cd ..

echo "✓ MNIST dataset download complete!"
echo "Data location: ./mnist_data/"
