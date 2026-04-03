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