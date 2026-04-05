#!/usr/bin/env python3
"""
Caffe Model Manager
Handles listing, discovering, and retrieving information about Caffe models.
"""

import logging
import os
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional


@dataclass
class ModelInfo:
    """Information about a Caffe model."""
    name: str
    proto_file: Optional[str] = None
    model_file: Optional[str] = None
    size_mb: float = 0.0
    proto_size_mb: float = 0.0
    
    def to_dict(self):
        return asdict(self)


class CaffeModelManager:
    """Manages discovery and retrieval of Caffe models."""
    
    # Common extensions for Caffe model files
    PROTO_EXTENSIONS = {'.prototxt', '.proto'}
    MODEL_EXTENSIONS = {'.caffemodel', '.weights'}
    
    def __init__(self, models_folder: str):
        """Initialize model manager with a folder path."""
        self.models_folder = models_folder
        self.models: Dict[str, ModelInfo] = {}
        self.discover_models()
    
    def set_models_folder(self, folder: str):
        """Change the models folder and rediscover models."""
        if not os.path.isdir(folder):
            raise ValueError(f"Folder does not exist: {folder}")
        self.models_folder = folder
        self.discover_models()
    
    def discover_models(self) -> Dict[str, ModelInfo]:
        """Discover all Caffe models in the models folder."""
        self.models = {}
        
        if not os.path.isdir(self.models_folder):
            os.makedirs(self.models_folder, exist_ok=True)
            return self.models
        
        # Find all proto files first
        proto_files = {}
        for root, dirs, files in os.walk(self.models_folder):
            for file in files:
                if any(file.endswith(ext) for ext in self.PROTO_EXTENSIONS):
                    filepath = os.path.join(root, file)
                    logging.debug(f"Found proto file: {filepath}")
                    model_name = self._extract_model_name(file)
                    logging.debug(f"Extracted model name: {model_name} from file: {file}")
                    if model_name not in proto_files:
                        proto_files[model_name] = filepath
        
        # Find all model files
        model_files = {}
        for root, dirs, files in os.walk(self.models_folder):
            for file in files:
                if any(file.endswith(ext) for ext in self.MODEL_EXTENSIONS):
                    filepath = os.path.join(root, file)
                    logging.debug(f"Found model file: {filepath}")
                    model_name = self._extract_model_name(file)
                    logging.debug(f"Extracted model name: {model_name} from file: {file}")
                    if model_name not in model_files:
                        model_files[model_name] = filepath
        
        # Combine proto and model files
        all_model_names = set(proto_files.keys()) | set(model_files.keys())
        
        for model_name in all_model_names:
            proto_path = proto_files.get(model_name)
            model_path = model_files.get(model_name)
                       
            logging.debug(f"Processing model: {model_name}, proto: {proto_path}, model: {model_path}")
            
            # Calculate file sizes
            proto_size = os.path.getsize(proto_path) / (1024 * 1024) if proto_path else 0.0
            model_size = os.path.getsize(model_path) / (1024 * 1024) if model_path else 0.0
            
            model_info = ModelInfo(
                name=model_name,
                proto_file=proto_path,
                model_file=model_path,
                proto_size_mb=round(proto_size, 2),
                size_mb=round(model_size, 2)
            )
            self.models[model_name] = model_info
        
        return self.models
    
    def list_models(self) -> List[Dict]:
        """Get list of all available models."""
        return [model.to_dict() for model in self.models.values()]
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a specific model."""
        model = self.models.get(model_name)
        return model.to_dict() if model else None
    
    def get_proto_file(self, model_name: str) -> Optional[str]:
        """Get the path to a model's prototxt file."""
        model = self.models.get(model_name)
        return model.proto_file if model else None
    
    def get_model_file(self, model_name: str) -> Optional[str]:
        """Get the path to a model's weights file."""
        model = self.models.get(model_name)
        return model.model_file if model else None
    
    @staticmethod
    def _extract_model_name(filename: str) -> str:
        """Extract model name from filename."""
        # Remove extension and common suffixes
        name = Path(filename).stem
        # Remove common suffixes like _train, _test, _solver, etc.
        name = re.sub(r'_(train|test|solver|deploy|inference)', '', name)
        return name
