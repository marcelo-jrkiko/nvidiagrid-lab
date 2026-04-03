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