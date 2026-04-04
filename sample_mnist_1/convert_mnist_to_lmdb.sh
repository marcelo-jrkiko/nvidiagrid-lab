#!/bin/bash
# Convert MNIST binary files to LMDB format for Caffe

set -e

MNIST_DIR="./mnist_data"
LMDB_DIR="./mnist_lmdb"

if [ ! -d "$MNIST_DIR" ]; then
    echo "Error: mnist_data directory not found. Run download_mnist.sh first."
    exit 1
fi

# Find convert_mnist_data binary
# It's an example program that needs to be built from caffe/examples/mnist/convert_mnist_data.cpp
CAFFE_ROOT="/usr/local/caffe"  # Change this to your Caffe root directory if different
CONVERT_TOOL="$CAFFE_ROOT/build/examples/mnist/convert_mnist_data"

if [ ! -f "$CONVERT_TOOL" ]; then
    echo "Error: convert_mnist_data binary not found at $CONVERT_TOOL"
    echo "Please build it first:"
    echo "  cd $CAFFE_ROOT/build"
    echo "  make examples"
    echo "Or specify CAFFE_ROOT:"
    echo "  CAFFE_ROOT=/path/to/caffe ./convert_mnist_to_lmdb.sh"
    exit 1
fi

# Create LMDB directories
mkdir -p "$LMDB_DIR"

echo "Converting MNIST training data to LMDB format..."
"$CONVERT_TOOL" \
    "$MNIST_DIR/train-images-idx3-ubyte" \
    "$MNIST_DIR/train-labels-idx1-ubyte" \
    "$LMDB_DIR/mnist_train_lmdb"

echo "Converting MNIST test data to LMDB format..."
"$CONVERT_TOOL" \
    "$MNIST_DIR/t10k-images-idx3-ubyte" \
    "$MNIST_DIR/t10k-labels-idx1-ubyte" \
    "$LMDB_DIR/mnist_test_lmdb"

echo "✓ MNIST LMDB conversion complete!"
echo "LMDB location: $LMDB_DIR/"
echo "  - mnist_train_lmdb/"
echo "  - mnist_test_lmdb/"
