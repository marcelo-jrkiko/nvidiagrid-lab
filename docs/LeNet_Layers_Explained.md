# LeNet-MNIST Architecture: Layer-by-Layer Explanation

This document explains each layer in the LeNet convolutional neural network as defined in `mnist_lenet.prototxt`.

## Overview

LeNet is a classic convolutional neural network designed for handwritten digit recognition (MNIST dataset). It consists of convolutional layers, pooling layers, fully connected layers, and activation functions.

---

## Data Layers

### Training Data Layer
```
Layer: mnist (TRAIN)
Type: Data
Inputs: MNIST training LMDB database
Outputs: data, label
```

**Purpose**: Loads training data from the MNIST LMDB database.

**Key Parameters**:
- `source`: Points to `mnist_lmdb/mnist_train_lmdb` (training dataset)
- `batch_size`: 128 (processes 128 images per training iteration)
- `backend`: LMDB (efficient data format)
- `scale`: 0.00390625 (normalizes pixel values: 1/256, converts 0-255 range to 0-1)

**Output**:
- `data`: Batch of 128 training images (normalized pixel values)
- `label`: Corresponding digit labels (0-9)

---

### Testing Data Layer
```
Layer: mnist (TEST)
Type: Data
Inputs: MNIST testing LMDB database
Outputs: data, label
```

**Purpose**: Loads test data from the MNIST LMDB database during the testing/validation phase.

**Key Parameters**:
- `source`: Points to `mnist_lmdb/mnist_test_lmdb` (testing dataset)
- `batch_size`: 100 (processes 100 images per test iteration)
- `scale`: 0.00390625 (same normalization as training)

**Output**:
- `data`: Batch of 100 test images (normalized)
- `label`: Corresponding digit labels

---

## First Convolutional Block

### Conv1 Layer
```
Layer: conv1
Type: Convolution
Input: data (28×28×1 images)
Output: conv1 (24×24×32 feature maps)
```

**Purpose**: Extracts low-level features (edges, corners, shapes) from input images.

**Parameters**:
- `num_output`: 32 (creates 32 different feature maps)
- `kernel_size`: 5 (5×5 filters/kernels)
- `stride`: 1 (move filter 1 pixel at a time)
- `weight_filler`: xavier (smart random initialization)
- `bias_filler`: constant (initialized to 0)

**Learnable Parameters**:
- Weights: 32 kernels × 25 values = 800 weights
- Biases: 32 (one per filter)

**Output Size**: (28 - 5 + 1) × (28 - 5 + 1) = 24×24×32

---

### ReLU1 Layer
```
Layer: relu1
Type: ReLU (Rectified Linear Unit)
Input: conv1 (24×24×32)
Output: conv1 (24×24×32, modified in-place)
```

**Purpose**: Introduces non-linearity by removing negative values.

**Operation**: For each value x: output = max(0, x)

**Effect**:
- Helps network learn complex patterns
- Negative values → 0 (sparse representation)
- Positive values → unchanged

---

### Pool1 Layer
```
Layer: pool1
Type: Pooling (MAX)
Input: conv1 (24×24×32)
Output: pool1 (12×12×32)
```

**Purpose**: Reduces spatial dimensions while retaining important features.

**Parameters**:
- `pool`: MAX (keeps maximum value in each region)
- `kernel_size`: 2 (2×2 pooling windows)
- `stride`: 2 (non-overlapping, moves 2 pixels each step)

**Operation**: Divides 24×24 into 2×2 windows, takes max value from each

**Output Size**: 24 ÷ 2 = 12×12×32

**Benefits**:
- Reduces computation and memory
- Provides translation invariance
- Prevents overfitting

---

## Second Convolutional Block

### Conv2 Layer
```
Layer: conv2
Type: Convolution
Input: pool1 (12×12×32)
Output: conv2 (8×8×64)
```

**Purpose**: Learns higher-level features from the pooled first layer.

**Parameters**:
- `num_output`: 64 (creates 64 feature maps)
- `kernel_size`: 5 (5×5 filters)
- `stride`: 1

**Learnable Parameters**:
- Weights: 64 kernels × (32 × 25) = 51,200 weights
- Biases: 64

**Output Size**: (12 - 5 + 1) × (12 - 5 + 1) = 8×8×64

---

### ReLU2 Layer
```
Layer: relu2
Type: ReLU
Input: conv2 (8×8×64)
Output: conv2 (8×8×64)
```

**Purpose**: Again introduces non-linearity to the second convolutional layer.

**Operation**: Same as ReLU1 - output = max(0, x)

---

### Pool2 Layer
```
Layer: pool2
Type: Pooling (MAX)
Input: conv2 (8×8×64)
Output: pool2 (4×4×64)
```

**Purpose**: Further reduces spatial dimensions.

**Parameters**:
- `pool`: MAX
- `kernel_size`: 2
- `stride`: 2

**Output Size**: 8 ÷ 2 = 4×4×64

**Result**: 1,024 values (4 × 4 × 64)

---

## Fully Connected Layers

### FC1 Layer (InnerProduct)
```
Layer: fc1
Type: InnerProduct (Dense/Fully Connected)
Input: pool2 (1,024 flattened values)
Output: fc1 (128 neurons)
```

**Purpose**: Learns non-linear combinations of features for classification.

**Parameters**:
- `num_output`: 128 (output neurons)
- `weight_filler`: xavier
- `bias_filler`: constant

**Learnable Parameters**:
- Weights: 1,024 × 128 = 131,072
- Biases: 128

**Operation**: Fully connected - every input connects to every output

---

### ReLU3 Layer
```
Layer: relu3
Type: ReLU
Input: fc1 (128)
Output: fc1 (128)
```

**Purpose**: Non-linearity for first fully connected layer.

---

### Dropout Layer
```
Layer: drop1
Type: Dropout
Input: fc1 (128)
Output: fc1 (128)
```

**Purpose**: Regularization to prevent overfitting.

**Operation**:
- During training: Randomly sets ~50% of values to 0
- During testing: All values pass through (scaled appropriately)
- `dropout_ratio`: 0.5 (50% dropout rate)

**Effect**:
- Prevents co-adaptation of neurons
- Forces network to learn redundant representations
- Reduces overfitting on training data

---

### FC2 Layer (InnerProduct)
```
Layer: fc2
Type: InnerProduct
Input: fc1 (128)
Output: fc2 (10 output values)
```

**Purpose**: Produces 10 outputs, one for each digit class (0-9).

**Parameters**:
- `num_output`: 10 (one output per digit 0-9)

**Learnable Parameters**:
- Weights: 128 × 10 = 1,280
- Biases: 10

**Output**: 10 logits (raw scores before probability conversion)

---

## Loss & Evaluation Layers

### Accuracy Layer
```
Layer: accuracy
Type: Accuracy
Inputs: fc2 (predictions), label (ground truth)
Output: accuracy metric
```

**Purpose**: Measures classification accuracy during testing phase.

**Operation**:
- Compares predicted class (argmax of fc2) with true label
- Calculates percentage of correct predictions
- Only runs during `TEST` phase

**Metric**: Percentage of correctly classified images

---

### Loss Layer
```
Layer: loss
Type: SoftmaxWithLoss
Inputs: fc2 (logits), label (ground truth)
Output: loss value
```

**Purpose**: Combines softmax activation and cross-entropy loss.

**Operations**:
1. **Softmax**: Converts 10 logits to probability distribution (sums to 1)
2. **Cross-Entropy**: Measures difference between predicted and true distribution

**Role**: 
- Directs training (gradients flow backward)
- Penalizes confident wrong predictions more than uncertain ones
- Used for backpropagation and weight updates

---

## Network Summary

| Layer | Type | Input Size | Output Size | Parameters |
|-------|------|-----------|------------|-----------|
| Data (Train) | Data | - | 128×28×28×1 | 0 |
| Conv1 | Conv | 28×28×1 | 24×24×32 | 832 |
| ReLU1 | ReLU | 24×24×32 | 24×24×32 | 0 |
| Pool1 | Pooling | 24×24×32 | 12×12×32 | 0 |
| Conv2 | Conv | 12×12×32 | 8×8×64 | 51,264 |
| ReLU2 | ReLU | 8×8×64 | 8×8×64 | 0 |
| Pool2 | Pooling | 8×8×64 | 4×4×64 | 0 |
| FC1 | InnerProduct | 1,024 | 128 | 131,200 |
| ReLU3 | ReLU | 128 | 128 | 0 |
| Dropout | Dropout | 128 | 128 | 0 |
| FC2 | InnerProduct | 128 | 10 | 1,290 |
| Accuracy | Accuracy | - | scalar | 0 |
| Loss | SoftmaxWithLoss | - | scalar | 0 |

**Total Trainable Parameters**: ~184,586

---

## Data Flow Visualization

```
Input Images (28×28×1)
        ↓
    [DATA LAYER] → 128 images per batch
        ↓
    [CONV1 32 filters, 5×5] → 24×24×32
        ↓
    [RELU1] → Apply activation
        ↓
    [POOL1 2×2] → 12×12×32 (reduced dimensions)
        ↓
    [CONV2 64 filters, 5×5] → 8×8×64
        ↓
    [RELU2] → Apply activation
        ↓
    [POOL2 2×2] → 4×4×64 (1,024 features)
        ↓
    [FC1 128 neurons] → 128 hidden units
        ↓
    [RELU3] → Apply activation
        ↓
    [DROPOUT 50%] → Regularization
        ↓
    [FC2 10 neurons] → 10 class scores
        ↓
    [SOFTMAX + CROSS-ENTROPY LOSS]
        ↓
    Output: Probability for each digit (0-9)
```

---

## Key Concepts

### Feature Extraction
- **Conv layers**: Learn hierarchical features (edges → shapes → patterns)
- **Pool layers**: Reduce redundancy and provide translation invariance

### Classification
- **FC layers**: Convert features into class probabilities
- **Dropout**: Prevents overfitting by creating ensemble effect

### Training
- **Loss function**: Guides learning through backpropagation
- **Accuracy**: Evaluates model performance on test set

