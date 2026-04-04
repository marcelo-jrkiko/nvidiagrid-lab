#!/bin/bash
# Convert MNIST binary files to LMDB format for Caffe

set -e

MNIST_DIR="./mnist_data"
LMDB_DIR="./mnist_lmdb"

if [ ! -d "$MNIST_DIR" ]; then
    echo "Error: mnist_data directory not found. Run download_mnist.sh first."
    exit 1
fi

# Check if Caffe convert_mnist_data binary exists
if ! command -v convert_mnist_data &> /dev/null; then
    echo "Error: convert_mnist_data utility not found."
    echo "Make sure Caffe is built and convert_mnist_data is in your PATH"
    echo "It's typically in: /path/to/caffe/build/tools/convert_mnist_data"
    exit 1
fi

# Create LMDB directories
mkdir -p "$LMDB_DIR"

echo "Converting MNIST training data to LMDB format..."
convert_mnist_data \
    "$MNIST_DIR/train-images-idx3-ubyte" \
    "$MNIST_DIR/train-labels-idx1-ubyte" \
    "$LMDB_DIR/mnist_train_lmdb"

echo "Converting MNIST test data to LMDB format..."
convert_mnist_data \
    "$MNIST_DIR/t10k-images-idx3-ubyte" \
    "$MNIST_DIR/t10k-labels-idx1-ubyte" \
    "$LMDB_DIR/mnist_test_lmdb"

echo "✓ MNIST LMDB conversion complete!"
echo "LMDB location: $LMDB_DIR/"
echo "  - mnist_train_lmdb/"
echo "  - mnist_test_lmdb/"
