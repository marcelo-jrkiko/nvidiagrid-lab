
# Caffe Installation on Debian 9

## Prerequisites

Install build tools and dependencies:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git
```

Install protobuf:

```bash
sudo apt-get install -y libprotobuf-dev protobuf-compiler
```

Install ATLAS, Boost, and linear algebra libraries:

```bash
sudo apt-get install -y libopenblas-dev liblapack-dev libatlas-base-dev gfortran
sudo apt-get install -y libboost-all-dev
```

Install logging and flags libraries:

```bash
sudo apt-get install -y libglog-dev libgflags-dev
```

Install HDF5 and database libraries:

```bash
sudo apt-get install -y libhdf5-dev
sudo apt-get install -y libleveldb-dev liblmdb-dev
```

Install Python libraries:

```bash
pip3.7 install numpy scipy scikit-image pillow protobuf six
```

Install OpenCV2:

```bash
sudo apt-get install -y libopencv-dev python-opencv
```

## Install NCCL: Build from Source (Archived Download)

This method downloads NCCL 2.4.8 from an archived source and builds it locally.
```bash
# Create a working directory
mkdir -p ~/nccl-build
cd ~/nccl-build

# Download NCCL 2.4.8 from archived repository
wget http://192.172.1.50/downloads/Downloads/Retro/GPU/CUDA/nccl-2.4.8-1.zip

unzip nccl-2.4.8-1.zip

# If CUDA_HOME is not set, add it
export CUDA_HOME=/usr/local/cuda-8.0

cd ~/nccl-build/nccl-2.4.8-1

# Set environment variables for the build
export CUDA_HOME=/usr/local/cuda-8.0
export CC=/opt/gcc-5/bin/gcc
export CXX=/opt/gcc-5/bin/g++
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Build with parallel make jobs
make -j$(nproc) CUDA_HOME=$CUDA_HOME CCBIN=$CC

# The build should complete with output showing compilation
# Verify build output
ls -lh build/lib/
ls -lh build/include/
```

#### nstall NCCL to System Paths

```bash
cd ~/nccl-build/nccl-2.4.8-1

# Install headers
sudo cp -r include/nccl.h /usr/local/cuda/include/
sudo cp -r include/nccl_net.h /usr/local/cuda/include/ 2>/dev/null || true

# Install libraries
sudo cp build/lib/libnccl.so* /usr/local/cuda/lib64/
sudo cp build/lib/libnccl_static.a /usr/local/cuda/lib64/

# Update library cache
sudo ldconfig

# Set proper permissions
sudo chmod 644 /usr/local/cuda/include/nccl*.h
sudo chmod 755 /usr/local/cuda/lib64/libnccl*
```

#### Step 7: Verify NCCL Installation

```bash
# Check header files
ls -la /usr/local/cuda/include/nccl*.h

# Check library files
ls -la /usr/local/cuda/lib64/libnccl*

# Test library loading
ldd /usr/local/cuda/lib64/libnccl.so.2

# Compile a simple test (optional)
cat > /tmp/test_nccl.cu << 'EOF'
#include <nccl.h>
#include <stdio.h>

int main() {
    printf("NCCL version: %d\n", NCCL_VERSION_CODE);
    return 0;
}
EOF

nvcc -ccbin /opt/gcc-5/bin/gcc -I/usr/local/cuda/include /tmp/test_nccl.cu -o /tmp/test_nccl -L/usr/local/cuda/lib64 -lnccl
./tmp/test_nccl
```

## Caffe: Clone and Build

```bash
git clone https://github.com/BVLC/caffe.git
cd caffe
cp Makefile.config.example Makefile.config
```

## Configure Makefile

Edit `Makefile.config`:
- Uncomment `USE_CUDNN := 1` for using CUDA cores
- Uncomment `USE_NCCL := 1` for multi-GPU support
- Uncomment `USE_OPENCV := 1` to enable OpenCV2 support
- Configure to use Python3.7

```makefile
# Multi-GPU execution
USE_NCCL := 1
USE_CUDNN := 1
USE_OPENCV := 1

PYTHON_INCLUDE :=  /usr/local/include/python3.7m /usr/local/lib/python3.7/site-packages/numpy/core/include
PYTHON_LIB := /usr/local/lib/python3.7

WITH_PYTHON_LAYER := 1

PYTHON_LIBRARIES := boost_python37 boost_thread python3.7m
```

- Set include and library paths for HDF5, OpenCV, and NCCL:

```makefile
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/lib/boost_1_72_0/include /usr/include/hdf5/serial /usr/include/opencv /usr/local/cuda/include
LIBRARY_DIRS := /usr/lib/x86_64-linux-gnu/hdf5/serial /usr/lib/x86_64-linux-gnu /usr/local/lib/boost_1_72_0/lib /usr/local/cuda/lib64

# Link against boost_thread, boost_system, and NCCL
LDFLAGS += -L/usr/local/lib/boost_1_72_0/lib -lboost_thread -lboost_system -L/usr/local/cuda/lib64 -lnccl
```

Alternatively, set environment variables on .bashrc 

```bash

export CPATH=/usr/include/hdf5/serial:/usr/local/lib/boost_1_72_0/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/lib/boost_1_72_0/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/hdf5/serial:/usr/local/lib/boost_1_72_0/lib:$LIBRARY_PATH

source ~/.bashrc
```

## Compile

```bash
mkdir build && cd build
CC=/opt/gcc-5/bin/gcc CXX=/opt/gcc-5/bin/g++ make all -j$(nproc)
CC=/opt/gcc-5/bin/gcc CXX=/opt/gcc-5/bin/g++ make pycaffe
CC=/opt/gcc-5/bin/gcc CXX=/opt/gcc-5/bin/g++ make examples
```

Note: `make examples` is needed to build example tools like `convert_mnist_data` used for MNIST data conversion.

## Add to Python Path

```bash
echo 'export PYTHONPATH=/path/to/caffe/python:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

## Validate the Version of Compiled _CAFFE.SO
Check if the _caffe.so is compiled with python 3.7

```
ldd /usr/local/caffe/python/caffe/_caffe.so | grep python
```

## Troubleshooting

### NCCL Not Found During Caffe Build

If you encounter errors like "nccl.h not found" or "cannot find -lnccl":

1. **Verify NCCL installation:**
```bash
ls /usr/local/cuda/include/nccl.h
ls /usr/local/cuda/lib64/libnccl*
```

2. **Update library cache:**
```bash
sudo ldconfig
```

3. **Create symlinks if needed:**
```bash
sudo ln -sf /usr/local/cuda/lib64/libnccl* /usr/lib/x86_64-linux-gnu/
sudo ldconfig
```

4. **Set environment variables before building:**
```bash
export CPATH=/usr/local/cuda/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
```

5. **Rebuild Caffe:**
```bash
cd /usr/local/caffe
CC=/opt/gcc-5/bin/gcc CXX=/opt/gcc-5/bin/g++ make clean
CC=/opt/gcc-5/bin/gcc CXX=/opt/gcc-5/bin/g++ make all -j$(nproc)
CC=/opt/gcc-5/bin/gcc CXX=/opt/gcc-5/bin/g++ make pycaffe
```

### Multi-GPU Execution Not Available Error

If you get "Multi-GPU execution not available - rebuild with USE_NCCL":

1. **Verify Caffe was built with USE_NCCL:**
```bash
grep USE_NCCL /usr/local/caffe/Makefile.config
```

2. **Check if pyCaffe is linked against NCCL:**
```bash
ldd /usr/local/caffe/python/caffe/_caffe.so | grep nccl
```

3. **If empty, rebuild with proper NCCL paths:**
```bash
cd /usr/local/caffe
CC=/opt/gcc-5/bin/gcc CXX=/opt/gcc-5/bin/g++ make distclean
CC=/opt/gcc-5/bin/gcc CXX=/opt/gcc-5/bin/g++ make all -j$(nproc)
CC=/opt/gcc-5/bin/gcc CXX=/opt/gcc-5/bin/g++ make pycaffe
```

### NCCL Version Compatibility

- **NCCL 2.4.8** is compatible with CUDA 8.0
- **NCCL 2.6+** requires CUDA 9.0+
- **NCCL 2.8+** requires CUDA 10.0+

For CUDA 8, always use NCCL 2.4.8 or earlier.

### MNIST Data Conversion: convert_mnist_data not found

The `convert_mnist_data` tool is an example program that must be compiled. Make sure to build it:

```bash
cd /path/to/caffe/build
make examples
```

The compiled binary will be located at: `/path/to/caffe/build/examples/mnist/convert_mnist_data`

When running the conversion script, specify the Caffe root directory:
```bash
CAFFE_ROOT=/path/to/caffe ./convert_mnist_to_lmdb.sh
```

### TypeError: Couldn't build proto file into descriptor pool: duplicate file name caffe/proto/caffe.proto

This error occurs when protobuf modules are imported multiple times or there's a conflict between different protobuf versions. Solutions:

1. **Clean rebuild Caffe:**
```bash
cd /usr/local/caffe
CC=/opt/gcc-5/bin/gcc CXX=/opt/gcc-5/bin/g++ make clean
CC=/opt/gcc-5/bin/gcc CXX=/opt/gcc-5/bin/g++ make all -j$(nproc)
CC=/opt/gcc-5/bin/gcc CXX=/opt/gcc-5/bin/g++ make pycaffe
```

2. **Verify Python can find Caffe modules:**
```bash
python3.7 -c "import sys; print(sys.path)" | grep caffe
```

3. **If using Caffe in interactive Python, restart the interpreter** after rebuilding, as protobuf descriptors can persist in memory.