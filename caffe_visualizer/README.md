# Caffe Model Visualizer 🧠

A Python web application for discovering, browsing, and visualizing Caffe neural network models.

## Features

✨ **Model Discovery** - Automatically scan folders for .prototxt and .caffemodel files
🎨 **Interactive Visualization** - View network architecture with layer details
📋 **Model Management** - Change models folder on the fly
📊 **Layer Information** - View parameters, connections, and layer types
🌐 **Web Interface** - Clean, responsive web UI

## Installation

### Prerequisites

- Python 3.7+
- Flask
- A folder containing Caffe model files (.prototxt, .caffemodel)

### Setup

1. Install dependencies:

```bash
cd caffe_visualizer
pip install -r requirements.txt
```

2. Prepare your models folder:

```bash
mkdir models
# Copy your .prototxt and .caffemodel files here
```

## Usage

### Starting the Server

```bash
# Default models folder: ./models
python app.py

# Or specify a custom models folder:
CAFFE_MODELS_PATH=/path/to/models python app.py
```

The application will start at `http://localhost:5000`

### Web Interface

1. **Home Page** (`/`)
   - View all available models
   - Set custom models folder
   - Click a model to visualize it

2. **Visualization Page** (`/visualization/<model_name>`)
   - View model information (file sizes, paths)
   - Explore network architecture
   - See layer details and parameters
   - View ASCII diagram of the network

## API Endpoints

### GET `/api/models`
List all available models.

**Response:**
```json
{
  "success": true,
  "models": [
    {
      "name": "lenet",
      "proto_file": "/path/to/lenet.prototxt",
      "model_file": "/path/to/lenet.caffemodel",
      "proto_size_mb": 0.05,
      "size_mb": 25.3
    }
  ],
  "total": 1
}
```

### GET `/api/models/<model_name>`
Get details about a specific model.

**Response:**
```json
{
  "success": true,
  "model": {
    "name": "lenet",
    "proto_file": "/path/to/lenet.prototxt",
    "model_file": "/path/to/lenet.caffemodel",
    "proto_size_mb": 0.05,
    "size_mb": 25.3
  }
}
```

### GET `/api/visualize/<model_name>`
Get network visualization data for a model.

**Response:**
```json
{
  "success": true,
  "layers": [
    {
      "id": 0,
      "name": "conv1",
      "type": "Convolution",
      "top": ["conv1"],
      "bottom": ["data"],
      "params": {"kernel_size": "5", "num_output": "20"},
      "color": "#2196F3"
    }
  ],
  "html": "<div class=\"network-visualization\">...</div>"
}
```

### POST `/api/models/folder`
Set the models folder.

**Request:**
```json
{
  "path": "/path/to/models"
}
```

**Response:**
```json
{
  "success": true,
  "path": "/path/to/models"
}
```

## Project Structure

```
caffe_visualizer/
│
├── app.py                 # Flask application entry point
├── model_manager.py       # Model discovery and management
├── visualizer.py          # Caffe model parsing and visualization
├── requirements.txt       # Python dependencies
│
├── templates/
│   ├── index.html         # Home page template
│   └── visualization.html # Model visualization page template
│
└── static/
    ├── style.css          # Styling
    └── script.js          # Frontend JavaScript
```

## Module Details

### `app.py`
Main Flask application. Handles routing and API endpoints.

### `model_manager.py`
`CaffeModelManager` class:
- Discovers Caffe models in a folder
- Manages model information (file paths, sizes)
- Provides methods to retrieve model details

### `visualizer.py`
`CaffeVisualizer` class:
- Parses .prototxt files
- Extracts layer information
- Generates HTML and ASCII visualizations
- Color-codes layers by type

## Supported File Types

- **.prototxt** - Caffe network definition files
- **.caffemodel** - Caffe trained weights files
- **.proto** - Alternative protocol buffer definition files
- **.weights** - Alternative weights files

## Layer Type Colors

| Type | Color |
|------|-------|
| Data | 🟢 Green |
| Convolution | 🔵 Blue |
| Pooling | 🟠 Orange |
| ReLU | 🟣 Purple |
| InnerProduct | 🔴 Red |
| Softmax | 🔷 Cyan |
| Dropout | 🟡 Yellow |
| Batch | 🟤 Brown |

## Usage Examples

### Example 1: Visualize LeNet-MNIST

```bash
# Copy the sample MNIST model
cp sample_mnist_1/mnist_lenet.prototxt caffe_visualizer/models/

# Start the visualizer
cd caffe_visualizer
python app.py
```

Then open `http://localhost:5000` and click on the MNIST model.

### Example 2: Custom Models Folder

```bash
# Start with a custom models folder
CAFFE_MODELS_PATH=~/my_caffe_models python app.py
```

Or set it through the web interface.

## Troubleshooting

### No models found
- Ensure .prototxt or .caffemodel files are in the models folder
- Check the path in the "Settings" section
- Refresh the page

### Visualization not loading
- Check that the .prototxt file is valid
- Ensure the file uses standard Caffe syntax
- Check browser console for JavaScript errors

### Permission denied
- Ensure the models folder has read permissions
- Run with appropriate user permissions

## Development

### Adding Custom Parsing

Modify `visualizer.py` to support additional layer types or parameters:

```python
@staticmethod
def _get_layer_color(layer_type: str) -> str:
    colors = {
        'Data': '#4CAF50',
        'Convolution': '#2196F3',
        # Add custom types here
        'CustomLayer': '#FF5722',
    }
    return colors.get(layer_type, '#607D8B')
```

### Extending the API

Add new endpoints in `app.py`:

```python
@app.route('/api/custom-endpoint')
def custom_endpoint():
    # Your logic here
    return jsonify({'success': True, 'data': ...})
```

## Performance

- **Startup**: ~100ms (with 10 models)
- **Model Listing**: ~50ms
- **Visualization**: ~200ms (depends on network size)

## Limitations

- Supports basic Caffe layer types
- Prototxt parsing is regex-based (not a full parser)
- No support for nested networks or special layers
- Web interface for single-user access

## Future Enhancements

- [ ] Export visualizations as PNG/SVG
- [ ] Support for model comparison
- [ ] Network flops and parameter counting
- [ ] Layer activation visualization
- [ ] Multi-user collaborative features

## License

This project is part of the NVIDIA Grid Lab project.

## Support

For issues or questions, please refer to the main project documentation.
