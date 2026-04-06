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
# Support both RESULTS_FOLDER and CAFFE_MODELS_PATH for flexibility
MODELS_FOLDER = os.environ.get('RESULTS_FOLDER') or os.environ.get('CAFFE_MODELS_PATH', './results')
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


@app.route('/api/models/gan')
def get_gan_models():
    """Get list of available GAN models."""
    gan_models = model_manager.list_gan_models()
    return jsonify({
        'success': True,
        'gan_models': gan_models,
        'total': len(gan_models)
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
        # Check if it's a GAN model
        if model_manager.is_gan_model(model_name):
            return visualize_gan_model(model_name)
        
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


@app.route('/api/visualize/gan/<gan_model_name>')
def visualize_gan_model(gan_model_name):
    """Generate visualization for a GAN model."""
    try:
        gan_info = model_manager.get_gan_model_info(gan_model_name)
        if not gan_info:
            return jsonify({'success': False, 'error': 'GAN model not found'}), 404
        
        gen_proto = gan_info.get('generator_proto')
        disc_proto = gan_info.get('discriminator_proto')
        
        if not gen_proto or not disc_proto:
            return jsonify({'success': False, 'error': 'Generator or Discriminator prototxt not found'}), 404
        
        # Parse both networks
        gan_data = visualizer.parse_gan_prototxt(gen_proto, disc_proto)
        
        # Generate visualizations
        html_vis = visualizer.generate_gan_html_visualization(gan_data)
        comparison_html = visualizer.generate_gan_comparison_html(gan_data)
        stats = visualizer.get_gan_statistics(gan_data)
        
        return jsonify({
            'success': True,
            'generator_layers': gan_data['generator'],
            'discriminator_layers': gan_data['discriminator'],
            'generator_html': html_vis['generator'],
            'discriminator_html': html_vis['discriminator'],
            'comparison_html': comparison_html,
            'statistics': stats,
            'model_name': gan_model_name,
            'model_info': gan_info
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
        # Check if it's a GAN model
        if model_manager.is_gan_model(model_name):
            model_info = model_manager.get_gan_model_info(model_name)
            return render_template('gan-visualization.html', model_name=model_name, model_info=model_info, is_gan=True)
        
        model_info = model_manager.get_model_info(model_name)
        if not model_info:
            return "Model not found", 404
        return render_template('visualization.html', model_name=model_name, model_info=model_info, is_gan=False)
    except Exception as e:
        return f"Error: {str(e)}", 500


@app.route('/visualization/gan/<gan_model_name>')
def gan_visualization_page(gan_model_name):
    """Visualization page for a GAN model."""
    try:
        model_info = model_manager.get_gan_model_info(gan_model_name)
        if not model_info:
            return "GAN model not found", 404
        return render_template('gan-visualization.html', model_name=gan_model_name, model_info=model_info, is_gan=True)
    except Exception as e:
        return f"Error: {str(e)}", 500


if __name__ == '__main__':
    # Create models folder if it doesn't exist
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    
    print(f"Caffe Model Visualizer starting...")
    print(f"Models folder: {MODELS_FOLDER}")
    print(f"Open your browser at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
