# MNIST GAN Example for Caffe

This folder contains a Generative Adversarial Network (GAN) implementation for MNIST using Caffe, inspired by DCGAN architecture.

## Architecture Overview

### Generator Network (`generator.prototxt`)
The generator takes random noise (100-dimensional vector) and generates synthetic MNIST images.

**Architecture:**
1. **FC1**: 100 → 1024 neurons
   - Gaussian weight initialization (std=0.02)
   - Batch normalization
   - ReLU activation

2. **FC2**: 1024 → 6272 neurons (128×7×7)
   - Gaussian weight initialization
   - Batch normalization
   - ReLU activation

3. **Reshape**: 6272 → (128, 7, 7)
   - Reshapes fully connected output to spatial dimensions

4. **Deconv1**: 128 filters (7×7) → 64 filters (14×14)
   - Kernel: 4×4, Stride: 2, Padding: 1
   - Batch normalization
   - ReLU activation

5. **Deconv2**: 64 filters (14×14) → 1 filter (28×28)
   - Kernel: 4×4, Stride: 2, Padding: 1
   - **Output:** Tanh activation (range [-1, 1])

### Discriminator Network (`discriminator.prototxt`)
The discriminator receives images (real or fake) and predicts the probability they are real.

**Architecture:**
1. **Conv1**: (28×28) → 64 filters (14×14)
   - Kernel: 4×4, Stride: 2, Padding: 1
   - LeakyReLU activation (negative_slope: 0.2)

2. **Conv2**: 64 filters (14×14) → 128 filters (7×7)
   - Kernel: 4×4, Stride: 2, Padding: 1
   - Batch normalization
   - LeakyReLU activation

3. **Flatten**: 128×7×7 = 6272 neurons

4. **FC1**: 6272 → 1024 neurons
   - Batch normalization
   - LeakyReLU activation

5. **FC_out**: 1024 → 1 neuron
   - Sigmoid activation (probability output)

6. **Loss**: Sigmoid Cross-Entropy Loss

## File Descriptions

- **`generator.prototxt`**: Generator network architecture
- **`gan_solver.prototxt`**: Solver configuration for generator training
- **`discriminator.prototxt`**: Discriminator network architecture  
- **`discriminator_solver.prototxt`**: Solver configuration for discriminator training

## Solver Configuration

Both `gan_solver.prototxt` (Generator) and `discriminator_solver.prototxt` (Discriminator):

- **Optimizer**: SGD (Stochastic Gradient Descent)
  - Base learning rate: 0.0002
  - Momentum: 0.5
  - Weight decay: 0.0005
  - Policy: fixed (constant learning rate)

- **Training**:
  - Max iterations: 10,000
  - Display interval: Every 100 iterations
  - Snapshot interval: Every 1,000 iterations
  - GPU acceleration enabled

## Implementation Notes

### Key Design Choices

1. **Batch Normalization**: Applied in generator and discriminator (except first layer) to stabilize training
2. **LeakyReLU in Discriminator**: Uses negative slope of 0.2 to allow gradient flow
3. **Tanh Output**: Generator outputs in [-1, 1] range, matching images scaled to [-1, 1]
4. **Gaussian Weight Initialization**: Uses standard deviation of 0.02 following DCGAN specifications
5. **Adam Optimizer**: More stable than vanilla SGD for GAN training

### Training Procedure

For proper GAN training, you would need to:
1. Create alternating training loops for generator and discriminator
2. Label real images as 1 and generated images as 0
3. Alternate updates to minimize distribution shift
4. Monitor loss curves to detect mode collapse

### Data Preparation

Ensure MNIST data is preprocessed:
```bash
# Normalize pixel values to [-1, 1] or [0, 1]
# Create LMDB database
./convert_mnist_to_lmdb.sh
```

## Training Tips

1. **Learning Rate**: 0.0002 is recommended; adjust if training is unstable
2. **Batch Size**: 128 (adjust based on GPU memory)
3. **Early Stopping**: Monitor generated sample quality
4. **Checkpoint Management**: Snapshots saved every 10,000 iterations

## References

- Goodfellow et al., 2014: "Generative Adversarial Networks"
- Radford et al., 2016: "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" (DCGAN)

## Future Enhancements

- Add conditional GAN (CGAN) for class-specific image generation
- Implement Wasserstein GAN (WGAN) for more stable training
- Add gradient penalty techniques
- Multi-scale discriminator
