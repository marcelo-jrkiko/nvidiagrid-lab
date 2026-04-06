#!/usr/bin/env python3
"""
Caffe Model Manager
Handles listing, discovering, and retrieving information about Caffe models and GANs.
"""

import logging
import os
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple


@dataclass
class ModelInfo:
    """Information about a Caffe model."""
    name: str
    proto_file: Optional[str] = None
    model_file: Optional[str] = None
    size_mb: float = 0.0
    proto_size_mb: float = 0.0
    model_type: str = "standard"  # "standard" or "gan"
    
    def to_dict(self):
        return asdict(self)


@dataclass
class GANModelInfo:
    """Information about a Caffe GAN model (Generator + Discriminator)."""
    name: str
    generator_proto: Optional[str] = None
    generator_model: Optional[str] = None
    discriminator_proto: Optional[str] = None
    discriminator_model: Optional[str] = None
    generator_size_mb: float = 0.0
    discriminator_size_mb: float = 0.0
    total_size_mb: float = 0.0
    iteration: int = 0  # Training iteration if applicable
    
    def to_dict(self):
        return asdict(self)


class CaffeModelManager:
    """Manages discovery and retrieval of Caffe models and GANs."""
    
    # Common extensions for Caffe model files
    PROTO_EXTENSIONS = {'.prototxt', '.proto'}
    MODEL_EXTENSIONS = {'.caffemodel', '.weights'}
    
    def __init__(self, models_folder: str):
        """Initialize model manager with a folder path."""
        self.models_folder = models_folder
        self.models: Dict[str, ModelInfo] = {}
        self.gan_models: Dict[str, GANModelInfo] = {}
        self.discover_models()
    
    def set_models_folder(self, folder: str):
        """Change the models folder and rediscover models."""
        if not os.path.isdir(folder):
            raise ValueError(f"Folder does not exist: {folder}")
        self.models_folder = folder
        self.discover_models()
    
    def discover_models(self) -> Tuple[Dict[str, ModelInfo], Dict[str, GANModelInfo]]:
        """Discover all Caffe models and GANs in the models folder."""
        self.models = {}
        self.gan_models = {}
        
        if not os.path.isdir(self.models_folder):
            os.makedirs(self.models_folder, exist_ok=True)
            return self.models, self.gan_models
        
        # Find all proto files first
        proto_files = {}
        for root, dirs, files in os.walk(self.models_folder):
            for file in files:
                if any(file.endswith(ext) for ext in self.PROTO_EXTENSIONS):
                    filepath = os.path.join(root, file)
                    logging.debug(f"Found proto file: {filepath}")
                    # Store with the actual filename to detect generator/discriminator
                    proto_files[file] = filepath
        
        # Find all model files
        model_files = {}
        for root, dirs, files in os.walk(self.models_folder):
            for file in files:
                if any(file.endswith(ext) for ext in self.MODEL_EXTENSIONS):
                    filepath = os.path.join(root, file)
                    logging.debug(f"Found model file: {filepath}")
                    model_files[file] = filepath
        
        # First, detect and process GAN models
        self._discover_gan_models(proto_files, model_files)
        
        # Then process remaining standard models
        self._discover_standard_models(proto_files, model_files)
        
        return self.models, self.gan_models
    
    def _discover_gan_models(self, proto_files: Dict, model_files: Dict):
        """Discover GAN model pairs (generator + discriminator)."""
        gan_pattern = re.compile(r'(generator|discriminator)', re.IGNORECASE)
        processed_proto_files = set()
        
        # First pass: Look for direct generator.prototxt + discriminator.prototxt pairs in same directory
        # This handles the common case where GAN training outputs are directly named
        for gen_filename, gen_filepath in proto_files.items():
            if re.search(r'generator.*\.prototxt', gen_filename, re.IGNORECASE):
                gen_dir = os.path.dirname(gen_filepath)
                
                # Look for matching discriminator in the same directory
                for disc_filename, disc_filepath in proto_files.items():
                    if re.search(r'discriminator.*\.prototxt', disc_filename, re.IGNORECASE):
                        disc_dir = os.path.dirname(disc_filepath)
                        
                        if gen_dir == disc_dir:  # Same directory - this is a GAN pair!
                            gan_name = "GAN"
                            gen_model = None
                            disc_model = None
                            iteration = 0
                            
                            # Try to find matching model files with flexible patterns
                            for model_filename, model_filepath in model_files.items():
                                # Look for generator model files (any .caffemodel with generator or simple gan_model_iter_ pattern)
                                if (re.search(r'generator', model_filename, re.IGNORECASE) and 
                                    model_filename.endswith('.caffemodel')):
                                    gen_model = model_filepath
                                    match = re.search(r'iter_(\d+)', model_filename)
                                    if match:
                                        iteration = max(iteration, int(match.group(1)))
                                
                                # For model files that don't have generator/discriminator but have gan_model and iter
                                elif (re.search(r'gan_model', model_filename, re.IGNORECASE) and 
                                      model_filename.endswith('.caffemodel') and
                                      not re.search(r'discriminator|disc', model_filename, re.IGNORECASE)):
                                    # This is likely the generator model
                                    if gen_model is None or 'generator' not in gen_model.lower():
                                        gen_model = model_filepath
                                        match = re.search(r'iter_(\d+)', model_filename)
                                        if match:
                                            iteration = max(iteration, int(match.group(1)))
                                
                                # Look for discriminator model files
                                if ((re.search(r'discriminator', model_filename, re.IGNORECASE) or
                                     re.search(r'disc', model_filename, re.IGNORECASE)) and 
                                    model_filename.endswith('.caffemodel')):
                                    disc_model = model_filepath
                            
                            gen_size = os.path.getsize(gen_model) / (1024 * 1024) if gen_model else 0.0
                            disc_size = os.path.getsize(disc_model) / (1024 * 1024) if disc_model else 0.0
                            
                            gan_info = GANModelInfo(
                                name=gan_name,
                                generator_proto=gen_filepath,
                                generator_model=gen_model,
                                discriminator_proto=disc_filepath,
                                discriminator_model=disc_model,
                                generator_size_mb=round(gen_size, 2),
                                discriminator_size_mb=round(disc_size, 2),
                                total_size_mb=round(gen_size + disc_size, 2),
                                iteration=iteration
                            )
                            self.gan_models[gan_name] = gan_info
                            logging.info(f"Discovered GAN model: {gan_name} at {gen_dir}")
                            
                            # Mark these proto files as processed
                            processed_proto_files.add(gen_filepath)
                            processed_proto_files.add(disc_filepath)
        
        # Second pass: Look for models with generator/discriminator patterns
        # This handles naming patterns where generator/discriminator appear in the filename
        gan_candidates = set()
        
        # Collect candidates from remaining proto files
        for filename in proto_files.keys():
            filepath = proto_files[filename]
            if filepath not in processed_proto_files and gan_pattern.search(filename):
                # Extract base name (e.g., from "my_gan_generator.prototxt")
                base_name = re.sub(r'_(generator|discriminator)', '', filename, flags=re.IGNORECASE)
                base_name = re.sub(r'(\.prototxt|\.caffemodel)', '', base_name)
                if base_name and base_name.lower() not in ['generator', 'discriminator']:
                    gan_candidates.add(base_name)
        
        # For each candidate, find matching generator and discriminator files
        for base_name in gan_candidates:
            gen_proto = None
            gen_model = None
            disc_proto = None
            disc_model = None
            iteration = 0
            
            # Find generator proto
            for filename, filepath in proto_files.items():
                if filepath in processed_proto_files:
                    continue
                if (re.search(r'generator', filename, re.IGNORECASE) and 
                    base_name.lower() in filename.lower() and 
                    filename.endswith('.prototxt')):
                    gen_proto = filepath
            
            # Find discriminator proto
            for filename, filepath in proto_files.items():
                if filepath in processed_proto_files:
                    continue
                if ((re.search(r'discriminator', filename, re.IGNORECASE) or
                     re.search(r'disc_', filename, re.IGNORECASE)) and 
                    base_name.lower() in filename.lower() and 
                    filename.endswith('.prototxt')):
                    disc_proto = filepath
            
            # Find generator model
            for filename, filepath in model_files.items():
                if (re.search(r'generator', filename, re.IGNORECASE) and 
                    base_name.lower() in filename.lower() and 
                    filename.endswith('.caffemodel')):
                    gen_model = filepath
                    match = re.search(r'iter_(\d+)', filename)
                    if match:
                        iteration = max(iteration, int(match.group(1)))
            
            # Find discriminator model
            for filename, filepath in model_files.items():
                if ((re.search(r'discriminator', filename, re.IGNORECASE) or
                     re.search(r'disc_', filename, re.IGNORECASE)) and 
                    base_name.lower() in filename.lower() and 
                    filename.endswith('.caffemodel')):
                    disc_model = filepath
            
            # Create GAN model info only if we have both prototxt files
            if gen_proto and disc_proto:
                gen_size = os.path.getsize(gen_model) / (1024 * 1024) if gen_model else 0.0
                disc_size = os.path.getsize(disc_model) / (1024 * 1024) if disc_model else 0.0
                
                gan_info = GANModelInfo(
                    name=base_name,
                    generator_proto=gen_proto,
                    generator_model=gen_model,
                    discriminator_proto=disc_proto,
                    discriminator_model=disc_model,
                    generator_size_mb=round(gen_size, 2),
                    discriminator_size_mb=round(disc_size, 2),
                    total_size_mb=round(gen_size + disc_size, 2),
                    iteration=iteration
                )
                self.gan_models[base_name] = gan_info
                logging.info(f"Discovered GAN model: {base_name}")
                
                # Mark as processed
                processed_proto_files.add(gen_proto)
                processed_proto_files.add(disc_proto)
        
        # Remove discovered GAN proto files from the proto_files dict for standard model discovery
        for filepath in processed_proto_files:
            proto_files_to_remove = [k for k, v in proto_files.items() if v == filepath]
            for k in proto_files_to_remove:
                del proto_files[k]
    
    def _discover_standard_models(self, proto_files: Dict, model_files: Dict):
        """Discover standard (non-GAN) models."""
        # Pair proto and model files by extracted name
        model_pairs = {}
        
        for filename, filepath in proto_files.items():
            model_name = self._extract_model_name(filename)
            if model_name not in model_pairs:
                model_pairs[model_name] = {}
            model_pairs[model_name]['proto'] = filepath
        
        for filename, filepath in model_files.items():
            model_name = self._extract_model_name(filename)
            if model_name not in model_pairs:
                model_pairs[model_name] = {}
            model_pairs[model_name]['model'] = filepath
        
        # Create ModelInfo objects
        for model_name, files in model_pairs.items():
            proto_path = files.get('proto')
            model_path = files.get('model')
            
            logging.debug(f"Processing model: {model_name}, proto: {proto_path}, model: {model_path}")
            
            proto_size = os.path.getsize(proto_path) / (1024 * 1024) if proto_path else 0.0
            model_size = os.path.getsize(model_path) / (1024 * 1024) if model_path else 0.0
            
            model_info = ModelInfo(
                name=model_name,
                proto_file=proto_path,
                model_file=model_path,
                proto_size_mb=round(proto_size, 2),
                size_mb=round(model_size, 2),
                model_type="standard"
            )
            self.models[model_name] = model_info
    
    def list_models(self) -> List[Dict]:
        """Get list of all available models (both standard and GAN)."""
        models_list = [model.to_dict() for model in self.models.values()]
        gan_list = [
            {**gan.to_dict(), "model_type": "gan"} 
            for gan in self.gan_models.values()
        ]
        return models_list + gan_list
    
    def list_gan_models(self) -> List[Dict]:
        """Get list of all available GAN models."""
        return [gan.to_dict() for gan in self.gan_models.values()]
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a specific model."""
        model = self.models.get(model_name)
        if model:
            return model.to_dict()
        
        # Check if it's a GAN model
        gan = self.gan_models.get(model_name)
        if gan:
            gan_dict = gan.to_dict()
            gan_dict['model_type'] = 'gan'
            return gan_dict
        
        return None
    
    def get_gan_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a GAN model."""
        gan = self.gan_models.get(model_name)
        return gan.to_dict() if gan else None
    
    def is_gan_model(self, model_name: str) -> bool:
        """Check if a model is a GAN model."""
        return model_name in self.gan_models
    
    def get_proto_file(self, model_name: str) -> Optional[str]:
        """Get the path to a model's prototxt file."""
        model = self.models.get(model_name)
        if model:
            return model.proto_file
        
        gan = self.gan_models.get(model_name)
        if gan:
            # Return both generator and discriminator paths
            return {"generator": gan.generator_proto, "discriminator": gan.discriminator_proto}
        
        return None
    
    def get_model_file(self, model_name: str) -> Optional[str]:
        """Get the path to a model's weights file."""
        model = self.models.get(model_name)
        if model:
            return model.model_file
        
        gan = self.gan_models.get(model_name)
        if gan:
            # Return both generator and discriminator paths
            return {"generator": gan.generator_model, "discriminator": gan.discriminator_model}
        
        return None
    
    @staticmethod
    def _extract_model_name(filename: str) -> str:
        """Extract model name from filename."""
        # Remove extension and common suffixes
        name = Path(filename).stem
        # Remove common suffixes like _train, _test, _solver, etc.
        name = re.sub(r'_(train|test|solver|deploy|inference)', '', name)
        return name
