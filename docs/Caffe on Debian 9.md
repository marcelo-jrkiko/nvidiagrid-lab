
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

## Clone and Build

```bash
git clone https://github.com/BVLC/caffe.git
cd caffe
cp Makefile.config.example Makefile.config
```

## Configure Makefile

Edit `Makefile.config`:
- Uncomment `USE_CUDNN := 1` for using CUDA cores
- Uncomment `USE_OPENCV := 1` to enable OpenCV2 support
- Configure to use Python3.7

```
PYTHON_INCLUDE :=  /usr/local/include/python3.7m /usr/local/lib/python3.7/site-packages/numpy/core/include
PYTHON_LIB := /usr/local/lib/python3.7

WITH_PYTHON_LAYER := 1

PYTHON_LIBRARIES := boost_python37 boost_thread python3.7m
```

- Set include and library paths for HDF5 and OpenCV:

```makefile
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/lib/boost_1_72_0/include /usr/include/hdf5/serial /usr/include/opencv
LIBRARY_DIRS := /usr/lib/x86_64-linux-gnu/hdf5/serial /usr/lib/x86_64-linux-gnu /usr/local/lib/boost_1_72_0/lib

# Link against boost_thread and boost_system
LDFLAGS += -L/usr/local/lib/boost_1_72_0/lib -lboost_thread -lboost_system
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
```

## Add to Python Path

```bash
echo 'export PYTHONPATH=/path/to/caffe/python:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

## Validate the Version of Compiled _CAFFE.SO
Check if the _caffe.so is compiled with python 3.7

```
ldd /usr/local/caffe/python/caffe/_caffe.so | grep python
``