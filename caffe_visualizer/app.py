#!/usr/bin/env python3
"""
Caffe Model Visualizer Web Application
Allows users to browse, select, and visualize Caffe models from a folder.
"""

import os
import json
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from model_manager import CaffeModelManager
from visualizer import CaffeVisualizer
from dotenv import load_dotenv
import logging

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()  # Load environment variables from .env file if it exists

# Initialize model manager with default models folder
MODELS_FOLDER = os.environ.get('RESULTS_FOLDER', './results')
model_manager = CaffeModelManager(MODELS_FOLDER)
visualizer = CaffeVisualizer()


@app.route('/')
def index():
    """Main page with model selection."""
    return render_template('index.html')


@app.route('/api/models')
def get_models():
    """Get list of available Caffe models."""
    models = model_manager.list_models()
    return jsonify({
        'success': True,
        'models': models,
        'total': len(models)
    })


@app.route('/api/models/<model_name>')
def get_model_details(model_name):
    """Get detailed information about a specific model."""
    try:
        model_info = model_manager.get_model_info(model_name)
        if not model_info:
            return jsonify({'success': False, 'error': 'Model not found'}), 404
        return jsonify({'success': True, 'model': model_info})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/visualize/<model_name>')
def visualize_model(model_name):
    """Generate visualization for a model."""
    try:
        proto_file = model_manager.get_proto_file(model_name)
        if not proto_file:
            return jsonify({'success': False, 'error': 'Model prototxt not found'}), 404
        
        # Parse and visualize the model
        layers_data = visualizer.parse_prototxt(proto_file)
        graph_html = visualizer.generate_html_visualization(layers_data)
        
        return jsonify({
            'success': True,
            'layers': layers_data,
            'html': graph_html
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/models/folder', methods=['POST'])
def set_models_folder():
    """Set the folder where models are stored."""
    try:
        data = request.get_json()
        folder = data.get('path')
        
        if not os.path.isdir(folder):
            return jsonify({'success': False, 'error': 'Folder does not exist'}), 400
        
        model_manager.set_models_folder(folder)
        return jsonify({'success': True, 'path': folder})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/visualization/<model_name>')
def visualization_page(model_name):
    """Visualization page for a specific model."""
    try:
        model_info = model_manager.get_model_info(model_name)
        if not model_info:
            return "Model not found", 404
        return render_template('visualization.html', model_name=model_name, model_info=model_info)
    except Exception as e:
        return f"Error: {str(e)}", 500


if __name__ == '__main__':
    # Create models folder if it doesn't exist
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    
    print(f"Caffe Model Visualizer starting...")
    print(f"Models folder: {MODELS_FOLDER}")
    print(f"Open your browser at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
