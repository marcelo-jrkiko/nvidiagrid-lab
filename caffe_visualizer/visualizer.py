#!/usr/bin/env python3
"""
Caffe Model Visualizer
Parses and visualizes Caffe model architecture.
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Layer:
    """Represents a Caffe layer."""
    name: str
    type: str
    top: List[str]
    bottom: List[str]
    params: Dict[str, Any]


class CaffeVisualizer:
    """Visualizes Caffe neural network models."""
    
    def __init__(self):
        """Initialize the visualizer."""
        self.layers: List[Layer] = []
    
    def parse_prototxt(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Parse a Caffe prototxt file and extract network structure.
        
        Args:
            filepath: Path to the .prototxt file
            
        Returns:
            List of layer dictionaries with type, name, inputs, and outputs
        """
        self.layers = []
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
        except IOError as e:
            raise IOError(f"Cannot read file: {filepath}") from e
        
        # Remove comments
        content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
        
        # Split by layer blocks
        layer_pattern = r'layer\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}'
        layer_matches = re.finditer(layer_pattern, content)
        
        for match in layer_matches:
            layer_block = match.group(1)
            layer_info = self._parse_layer_block(layer_block)
            if layer_info:
                self.layers.append(layer_info)
        
        return self._format_layers_for_visualization()
    
    def _parse_layer_block(self, block: str) -> Dict[str, Any]:
        """Parse a single layer block."""
        layer_dict = {
            'name': self._extract_value(block, 'name'),
            'type': self._extract_value(block, 'type'),
            'top': self._extract_list(block, 'top'),
            'bottom': self._extract_list(block, 'bottom'),
            'params': self._extract_params(block)
        }
        
        # Validate layer
        if not layer_dict['type']:
            return None
        
        return layer_dict
    
    @staticmethod
    def _extract_value(text: str, key: str) -> str:
        """Extract a single value from prototxt."""
        pattern = f'{key}\\s*:\\s*["\']?([^"\'\\n]*)[\"\']?'
        match = re.search(pattern, text)
        return match.group(1).strip() if match else ''
    
    @staticmethod
    def _extract_list(text: str, key: str) -> List[str]:
        """Extract a list of values from prototxt."""
        pattern = f'{key}\\s*:\\s*["\']([^"\']+)["\']'
        matches = re.findall(pattern, text)
        return [m.strip() for m in matches]
    
    @staticmethod
    def _extract_params(text: str) -> Dict[str, str]:
        """Extract layer parameters."""
        params = {}
        # Common parameters to extract
        param_keys = ['kernel_size', 'stride', 'pad', 'num_output', 'pool', 'alpha', 'beta']
        
        for key in param_keys:
            value = CaffeVisualizer._extract_value(text, key)
            if value:
                params[key] = value
        
        return params
    
    def _format_layers_for_visualization(self) -> List[Dict[str, Any]]:
        """Format layers for visualization."""
        formatted = []
        
        for i, layer in enumerate(self.layers):
            layer_data = {
                'id': i,
                'name': layer['name'],
                'type': layer['type'],
                'top': layer['top'],
                'bottom': layer['bottom'],
                'params': layer['params'],
                'color': self._get_layer_color(layer['type'])
            }
            formatted.append(layer_data)
        
        return formatted
    
    @staticmethod
    def _get_layer_color(layer_type: str) -> str:
        """Get color for layer type."""
        colors = {
            'Data': '#4CAF50',      # Green
            'Convolution': '#2196F3', # Blue
            'Pooling': '#FF9800',   # Orange
            'ReLU': '#9C27B0',      # Purple
            'InnerProduct': '#F44336', # Red
            'Softmax': '#00BCD4',   # Cyan
            'Dropout': '#FFC107',   # Yellow
            'Batch': '#795548',     # Brown
            'Concat': '#E91E63',    # Pink
            'Eltwise': '#3F51B5',   # Indigo
        }
        return colors.get(layer_type, '#607D8B')  # Default: Blue Grey
    
    def generate_html_visualization(self, layers_data: List[Dict]) -> str:
        """Generate HTML visualization of the network."""
        if not layers_data:
            return "<p>No layers found in model.</p>"
        
        html = '<div class="network-visualization">\n'
        html += '<div class="layers-container">\n'
        
        for layer in layers_data:
            html += self._generate_layer_html(layer)
        
        html += '</div>\n'
        html += '</div>\n'
        
        return html
    
    @staticmethod
    def _generate_layer_html(layer: Dict) -> str:
        """Generate HTML for a single layer."""
        html = f'''
        <div class="layer-card" style="border-left: 4px solid {layer['color']}">
            <div class="layer-header">
                <span class="layer-type" style="background-color: {layer['color']}">{layer['type']}</span>
                <span class="layer-name">{layer['name']}</span>
            </div>
            <div class="layer-body">
        '''
        
        if layer['bottom']:
            html += f"<p><strong>Inputs:</strong> {', '.join(layer['bottom'])}</p>\n"
        
        if layer['top']:
            html += f"<p><strong>Outputs:</strong> {', '.join(layer['top'])}</p>\n"
        
        if layer['params']:
            html += "<p><strong>Parameters:</strong></p>\n"
            html += "<ul>\n"
            for key, value in layer['params'].items():
                html += f"<li>{key}: {value}</li>\n"
            html += "</ul>\n"
        
        html += '''
            </div>
        </div>
        '''
        
        return html
    
    def generate_ascii_diagram(self, layers_data: List[Dict]) -> str:
        """Generate ASCII representation of the network."""
        ascii_diagram = "Network Architecture:\n"
        ascii_diagram += "=" * 50 + "\n\n"
        
        for i, layer in enumerate(layers_data):
            indent = "  " if i > 0 else ""
            ascii_diagram += f"{indent}[{layer['type']}] {layer['name']}\n"
            
            if layer['bottom']:
                ascii_diagram += f"{indent}  ← {', '.join(layer['bottom'])}\n"
            
            if layer['top']:
                ascii_diagram += f"{indent}  → {', '.join(layer['top'])}\n"
            
            if layer['params']:
                ascii_diagram += f"{indent}  params: {layer['params']}\n"
            
            ascii_diagram += "\n"
        
        return ascii_diagram
