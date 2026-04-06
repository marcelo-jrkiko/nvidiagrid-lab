# GAN Visualization Enhancement

This document describes the enhancements made to the Caffe Visualizer to support GAN (Generative Adversarial Network) model visualization.

## Overview

The visualizer has been enhanced to automatically detect and visualize GAN models, which consist of two networks:
- **Generator**: Creates fake images from random noise
- **Discriminator**: Distinguishes between real and fake images

## New Features

### 1. Automatic GAN Model Detection
The `model_manager.py` now includes:
- `GANModelInfo` dataclass to store information about GAN models
- `_discover_gan_models()` method to automatically detect generator and discriminator file pairs
- Methods to distinguish between standard and GAN models

**Detection Logic:**
- Scans for files containing "generator" and "discriminator" in their names
- Creates model pairs from matching prototxt and caffemodel files
- Extracts training iteration numbers from filenames (e.g., `generator_iter_1000.caffemodel`)

### 2. Enhanced Visualizer
The `visualizer.py` now includes GAN-specific methods:

```python
# Parse both generator and discriminator networks
parse_gan_prototxt(generator_path, discriminator_path)

# Generate side-by-side HTML comparison
generate_gan_html_visualization(gan_data)

# Generate ASCII diagrams for both networks
generate_gan_ascii_diagram(gan_data)

# Create combined comparison view
generate_gan_comparison_html(gan_data)

# Get detailed statistics about both networks
get_gan_statistics(gan_data)
```

### 3. New Flask API Endpoints

#### Get GAN Models List
```
GET /api/models/gan
```
Returns list of all discovered GAN models.

#### Visualize GAN Model
```
GET /api/visualize/gan/<gan_model_name>
```
Returns comprehensive visualization data including:
- Generator and discriminator layer information
- Side-by-side HTML visualizations
- Combined architecture comparison
- Statistical analysis of both networks

#### GAN Visualization Page
```
GET /visualization/gan/<gan_model_name>
```
Serves the GAN visualization HTML page.

### 4. New GAN Visualization UI

Created `gan-visualization.html` with:

#### Comparison View Tab
- Generator and Discriminator side-by-side
- Visual data flow diagram showing GAN interaction
- Layer count badges and statistics
- Scrollable layer preview for each network

#### Statistics Tab
- Layer type distribution
- Total layer counts
- Input/output blob information

#### Generator Details Tab
- Full generator network architecture
- All layers with connections

#### Discriminator Details Tab
- Full discriminator network architecture
- All layers with connections

#### Combined Layers Tab
- Full architecture comparison in single view
- Shows how both networks work together

### 5. Enhanced Model Discovery

The visualizer now:
- Lists both standard and GAN models
- Distinguishes GAN models with special styling and badge
- Shows model-specific metadata (training iteration, file sizes)
- Groups models by type in the UI

## File Structure

### Modified Files

1. **caffe_visualizer/model_manager.py**
   - Added `GANModelInfo` dataclass
   - Enhanced `CaffeModelManager` with GAN detection
   - New methods: `list_gan_models()`, `get_gan_model_info()`, `is_gan_model()`

2. **caffe_visualizer/visualizer.py**
   - Added GAN parsing methods
   - Added GAN-specific visualization generators
   - Statistics computation for GAN models

3. **caffe_visualizer/app.py**
   - New endpoint: `/api/models/gan`
   - New endpoint: `/api/visualize/gan/<name>`
   - New route: `/visualization/gan/<name>`
   - Updated `/api/visualize/<model_name>` to handle GANs

4. **caffe_visualizer/static/script.js**
   - Updated `displayModels()` to show GAN models separately
   - Special styling and links for GAN models

5. **caffe_visualizer/static/style.css**
   - Added `.gan-model-card` styles
   - Added gradient styling for GAN model display

### New Files

1. **caffe_visualizer/templates/gan-visualization.html**
   - Complete GAN visualization interface
   - Multiple tabs for different views
   - Real-time data loading and display
   - Responsive design

## Usage

### Discovering GAN Models

The visualizer will automatically discover GAN models when:

1. Files are named with patterns like:
   - `generator.prototxt` / `discriminator.prototxt`
   - `generator_iter_1000.caffemodel` / `discriminator_iter_1000.caffemodel`
   - Files stored in the results folder (default: `./results`)

2. The models folder is set through:
   - Environment variable: `RESULTS_FOLDER`
   - UI Setting: "Set Folder" button

### Viewing GAN Models

1. **From Home Page:**
   - GAN models appear in a separate section at the top
   - Click on any GAN model to view detailed visualization

2. **GAN Visualization Page:**
   - Provides multiple views for exploration:
     - Comparison View: Side-by-side generator and discriminator
     - Statistics: Layer counts and distribution
     - Details: Full layer information for each network
     - Combined: Integrated architecture view

## API Response Format

### /api/visualize/gan/<name>

```json
{
  "success": true,
  "model_name": "MNIST_GAN",
  "model_info": {
    "name": "MNIST_GAN",
    "generator_proto": "/path/to/generator.prototxt",
    "generator_model": "/path/to/generator_iter_1000.caffemodel",
    "discriminator_proto": "/path/to/discriminator.prototxt",
    "discriminator_model": "/path/to/discriminator_iter_1000.caffemodel",
    "generator_size_mb": 12.5,
    "discriminator_size_mb": 8.3,
    "total_size_mb": 20.8,
    "iteration": 1000
  },
  "generator_layers": [...],
  "discriminator_layers": [...],
  "generator_html": "...",
  "discriminator_html": "...",
  "comparison_html": "...",
  "statistics": {
    "generator": {
      "total_layers": 15,
      "layer_types": {"InnerProduct": 2, "Convolution": 4, ...},
      "input_blobs": ["noise"],
      "output_blobs": ["generated_images"]
    },
    "discriminator": {
      "total_layers": 12,
      "layer_types": {"Convolution": 5, "ReLU": 5, ...},
      "input_blobs": ["data"],
      "output_blobs": ["classification"]
    }
  }
}
```

## Example Project Structure

For a trained MNIST GAN project:

```
mnist/
├── gan1/
│   ├── generator.prototxt
│   ├── discriminator.prototxt
│   ├── generator_solver.prototxt
│   └── discriminator_solver.prototxt
└── results/
    ├── generator.prototxt
    ├── generator_iter_1000.caffemodel
    ├── discriminator.prototxt
    ├── discriminator_iter_1000.caffemodel
    ├── generator_iter_2000.caffemodel
    ├── discriminator_iter_2000.caffemodel
    └── gan_training_summary.txt
```

All GAN model pairs in the results folder will be automatically discovered and displayed.

## Technical Implementation

### GAN Detection Algorithm

1. **Scan for Generator/Discriminator Files:**
   - Search all proto and model files
   - Identify files matching "generator" or "discriminator" patterns

2. **Extract Base Names:**
   - Remove suffixes like `_generator`, `_discriminator`, `_iter_XXXX`
   - Extract common base name (e.g., "MNIST_GAN")

3. **Pair Matching:**
   - Find generator prototxt and model files
   - Find discriminator prototxt and model files
   - Create GAN model entry when both networks are found

4. **Store Metadata:**
   - Extract training iteration from filename
   - Calculate file sizes
   - Store paths for visualization

### Visualization Strategy

**Comparison View:**
- Uses CSS Grid for side-by-side layout
- Color-coded layer types
- Animated data flow diagram

**Statistics:**
- Counts layers by type
- Identifies input/output blobs
- Calculates network complexity metrics

**Layer Rendering:**
- Recursive HTML generation
- Preserves network topology
- Links show data connections

## Future Enhancements

Potential features for future versions:

1. **3D Network Visualization**
   - Extend Three.js visualization to GAN models
   - Side-by-side 3D models with flow animation

2. **Training Metrics Display**
   - Load and display training loss curves
   - Show generator vs discriminator loss over iterations

3. **Model Comparison**
   - Compare multiple GAN checkpoints
   - Visualize network evolution during training

4. **Interactive Architecture Editor**
   - Modify network definitions visually
   - Generate updated prototxt files

5. **Export Functionality**
   - Export visualizations as SVG/PNG
   - Generate detailed network reports

## Troubleshooting

### GAN Models Not Detected

1. Check file naming:
   - Use "generator" and "discriminator" in filenames
   - Ensure prototxt and caffemodel pairs exist

2. Set Models Folder:
   - Use UI to set folder containing trained models
   - Or set `RESULTS_FOLDER` environment variable

3. Check Logs:
   - Enable debug logging in app.py
   - Look for discovery messages

### Visualization Issues

1. Check Network Files:
   - Verify prototxt files are valid Caffe syntax
   - Check file permissions

2. Browser Cache:
   - Clear browser cache
   - Try incognito/private mode

3. Check API Response:
   - Use browser DevTools to inspect API responses
   - Look for error messages in response JSON

## References

- [Caffe Documentation](http://caffe.berkeleyvision.org/)
- [GAN Papers](https://arxiv.org/abs/1406.2661)
- [Caffe Prototxt Format](https://github.com/BVLC/caffe/blob/master/docs/tutorial/layers.md)
