# CUDA 8.0 Tips to Install on Debian 9

## Driver and Cuda Install
For GRID K1 and K2 , first download and install the NVIDIA DRIVER from the archived site:

[Archive 367.134](https://us.download.nvidia.com/XFree86/Linux-x86_64/367.134/NVIDIA-Linux-x86_64-367.134.run)

Run the default installation process:

```
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/367.134/NVIDIA-Linux-x86_64-367.134.run
chmod +x NVIDIA-Linux-x86_64-367.134.run
./NVIDIA-Linux-x86_64-367.134.run
```

### Cuda Install
Download the CUDA Toolkit from the archives:

[Archive CUDA 8.0.61](https://developer.download.nvidia.com/compute/cuda/8.0/secure/Prod2/local_installers/cuda_8.0.61_375.26_linux.run)

The best way is to install only the toolkit

```
./cuda_8.0.61_375.26_linux.run --extract=/tmp/cuda
cd /tmp/cuda

./cuda-linux64-rel-8.0.61-21551265.run
```

#### Can’t locate InstallUtils.pm in @INC
ref: https://forums.developer.nvidia.com/t/cant-locate-installutils-pm-in-inc/46952

```
./cuda_8.0.61_375.26_linux.run --extract=/tmp/cuda
cd /tmp/cuda
./cuda*.run --tar mxvf
cp InstallUtils.pm /usr/lib/x86_64-linux-gnu/perl-base
export $PERL5LIB
```

#### Add Cuda Path 
Add the following to the .bashrc:

```
#
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH

```

## GCC 5
If you are running on Debian 9 maybe you have the GCC 6 pre installed, but NVCC uses GCC 5. So we need to build and install GCC 5 from the sources:

```
sudo apt-get install build-essential libgmp-dev libmpfr-dev libmpc-dev
wget http://ftp.gnu.org/gnu/gcc/gcc-5.4.0/gcc-5.4.0.tar.gz
tar -xzf gcc-5.4.0.tar.gz
cd gcc-5.4.0
./configure --prefix=/opt/gcc-5 --enable-languages=c,c++
make -j$(nproc)
sudo make install
```

*X64*: Maybe need to disable multiarch on configure if you dont have 32bits libs installed: --disable-multi

GCC 5 will installed on /opt/gcc-5, now we can combine the cuda programs with NVCC using the correct gcc version:

```
nvcc -ccbin /opt/gcc-5/bin/gcc my_cuda_program.cu -o my_cuda_program
```

## Python 3.7
Debian 9 comes with Python 3.5 by default. To install Python 3.7, build it from source:

### Install Build Dependencies
```
sudo apt-get install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget libboost-python-dev
```

### Download and Build Python 3.7
```
wget https://www.python.org/ftp/python/3.7.17/Python-3.7.17.tgz
tar -xzf Python-3.7.17.tgz
cd Python-3.7.17
CFLAGS=-fPIC ./configure --enable-optimizations
make -i -j$(nproc) PROFILE_TASK=
sudo make altinstall
```

### Verify Installation
```
source ~/.bashrc
python3.7 --version
pip3.7 --version
```

## Boost.Python 3.7
Since Python 3.7 is installed as an alternative, you need to build Boost.Python from source for that specific Python installation:

### Download and Extract Boost
```bash
cd /tmp
wget https://archives.boost.io/release/1.72.0/source/boost_1_72_0.tar.gz
tar -xzf boost_1_72_0.tar.gz
cd boost_1_72_0
```

### Bootstrap and Build Boost.Python
```bash
# Set the GCC 5 compiler from previous steps
export CC=/opt/gcc-5/bin/gcc
export CXX=/opt/gcc-5/bin/g++

# Create user-config.jam to specify Python configuration
cat > ~/user-config.jam << 'EOF'
using python : 3.7 : /usr/bin/python3.7 : /usr/local/include/python3.7m : /usr/local/lib/python3.7 ;
using gcc : 5 : /opt/gcc-5/bin/g++ ;
EOF

./bootstrap.sh --prefix=/usr/local/lib/boost_1_72_0 --with-libraries=python,thread

./b2 --with-python --with-thread toolset=gcc-5 variant=release link=shared runtime-link=shared install
```

### Verify Boost.Python Installation
```bashnani
ls /usr/local/lib/boost_1_72_0/lib/libboost_python3*
```