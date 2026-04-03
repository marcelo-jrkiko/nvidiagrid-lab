
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
sudo apt-get install -y python-dev python-pip
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
- Set `PYTHON_INCLUDE := /usr/include/python2.7`
- Set include and library paths for HDF5 and OpenCV:

```makefile
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/include/hdf5/serial /usr/include/opencv
LD_LIBRARY_PATH := /usr/lib/x86_64-linux-gnu/hdf5/serial:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

## Compile

```bash
mkdir build && cd build
CC=/opt/gcc-5/bin/gcc CXX=/opt/gcc-5/bin/g++ make all -j$(nproc)
CC=/opt/gcc-5/bin/gcc CXX=/opt/gcc-5/bin/g++ make test
CC=/opt/gcc-5/bin/gcc CXX=/opt/gcc-5/bin/g++ make runtest
CC=/opt/gcc-5/bin/gcc CXX=/opt/gcc-5/bin/g++ make pycaffe
```

## Add to Python Path

```bash
echo 'export PYTHONPATH=/path/to/caffe/python:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```
