# Installing cuDNN v6 on Debian 9

## Prerequisites

- CUDA Toolkit installed and configured
- Root or sudo access
- Internet connection

## Installation Steps

1. **Download cuDNN**
    - Visit [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
    - Sign in or create an account
    - Download cuDNN v6 for Linux

2. **Extract the archive**
    ```bash
    tar -xzvf cudnn-9.0-linux-x64-v6.0.tgz
    ```

3. **Copy files to CUDA installation**
    ```bash
    sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
    sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
    sudo chmod a+r /usr/local/cuda/include/cudnn.h
    sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
    ```

4. **Update library cache**
    ```bash
    sudo ldconfig /usr/local/cuda/lib64
    ```

5. **Verify installation**
    ```bash
    cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
    ```

## Troubleshooting

- Ensure CUDA paths are properly set in `PATH` and `LD_LIBRARY_PATH`
- Check file permissions if you encounter access errors