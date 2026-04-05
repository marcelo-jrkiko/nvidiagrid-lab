#!/bin/bash
# Download and prepare MNIST dataset for Caffe

set -e

echo "Downloading MNIST dataset..."
git clone https://github.com/marcelo-jrkiko/mnist_repo.git mnist_data
cd mnist_data

# Extract
echo "Extracting..."
gunzip -f train-images-idx3-ubyte.gz
gunzip -f train-labels-idx1-ubyte.gz
gunzip -f t10k-images-idx3-ubyte.gz
gunzip -f t10k-labels-idx1-ubyte.gz

cd ..

echo "✓ MNIST dataset download complete!"
echo "Data location: ./mnist_data/"
