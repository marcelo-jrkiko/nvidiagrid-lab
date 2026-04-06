#!/usr/bin/env python3
"""
Randomly corrupt a PPM image and save it with a new name.
Supports various corruption types: noise, pixel flipping, blocks, streaks.
"""

import sys
import argparse
import random
import numpy as np
from pathlib import Path


def read_ppm(filename):
    """Read a PPM image file and return width, height, max_val, and pixel data."""
    with open(filename, 'rb') as f:
        # Read magic number
        magic = f.readline().strip()
        if magic != b'P6':
            raise ValueError(f"Expected P6 PPM format, got {magic}")
        
        # Skip comments
        while True:
            line = f.readline().strip()
            if not line.startswith(b'#'):
                break
        
        # Parse width and height
        width, height = map(int, line.split())
        
        # Parse max value
        max_val = int(f.readline().strip())
        
        # Read pixel data
        pixel_data = f.read()
        
    return width, height, max_val, pixel_data


def write_ppm(filename, width, height, max_val, pixel_data):
    """Write pixel data to a PPM file."""
    with open(filename, 'wb') as f:
        f.write(b'P6\n')
        f.write(f'{width} {height}\n'.encode())
        f.write(f'{max_val}\n'.encode())
        f.write(pixel_data)


def add_gaussian_noise(pixel_data, intensity=0.1):
    """Add Gaussian noise to the image."""
    data_array = np.frombuffer(pixel_data, dtype=np.uint8).copy()
    noise = np.random.normal(0, intensity * 255, data_array.shape)
    data_array = np.clip(data_array.astype(float) + noise, 0, 255).astype(np.uint8)
    return data_array.tobytes()


def add_salt_pepper_noise(pixel_data, amount=0.1):
    """Add salt and pepper noise to the image."""
    data_array = np.frombuffer(pixel_data, dtype=np.uint8).copy()
    num_pixels = len(data_array)
    num_corrupted = int(num_pixels * amount)
    
    # Salt (white noise - 255)
    salt_indices = np.random.choice(num_pixels, num_corrupted // 2, replace=False)
    data_array[salt_indices] = 255
    
    # Pepper (black noise - 0)
    remaining_pixels = np.setdiff1d(np.arange(num_pixels), salt_indices)
    pepper_indices = np.random.choice(remaining_pixels, num_corrupted // 2, replace=False)
    data_array[pepper_indices] = 0
    
    return data_array.tobytes()


def add_random_bit_flips(pixel_data, amount=0.05):
    """Randomly flip bits in the pixel data."""
    data_array = np.frombuffer(pixel_data, dtype=np.uint8).copy()
    num_pixels = len(data_array)
    num_flipped = int(num_pixels * amount)
    
    flip_indices = np.random.choice(num_pixels, num_flipped, replace=False)
    flip_bits = np.random.randint(0, 8, num_flipped)
    
    for idx, bit in zip(flip_indices, flip_bits):
        data_array[idx] ^= (1 << bit)
    
    return data_array.tobytes()


def add_random_blocks(pixel_data, width, height, num_blocks=10, block_size=10):
    """Add random corrupted blocks to the image."""
    data_array = np.frombuffer(pixel_data, dtype=np.uint8).copy()
    data_array = data_array.reshape((height, width, 3))
    
    for _ in range(num_blocks):
        x = random.randint(0, max(0, width - block_size))
        y = random.randint(0, max(0, height - block_size))
        
        # Random corruption: noise or solid color
        if random.random() > 0.5:
            # Gaussian noise
            corruption = np.random.randint(0, 256, (min(block_size, height - y), min(block_size, width - x), 3), dtype=np.uint8)
        else:
            # Solid color
            corruption = np.full((min(block_size, height - y), min(block_size, width - x), 3), 
                               random.randint(0, 256), dtype=np.uint8)
        
        data_array[y:y+block_size, x:x+block_size] = corruption
    
    return data_array.tobytes()


def add_horizontal_streaks(pixel_data, width, height, num_streaks=5, streak_height=5):
    """Add horizontal corruption streaks to the image."""
    data_array = np.frombuffer(pixel_data, dtype=np.uint8).copy()
    data_array = data_array.reshape((height, width, 3))
    
    for _ in range(num_streaks):
        y = random.randint(0, max(0, height - streak_height))
        corruption = np.random.randint(0, 256, (streak_height, width, 3), dtype=np.uint8)
        data_array[y:y+streak_height, :] = corruption
    
    return data_array.tobytes()


def corrupt_image(input_file, output_file, corruption_type, intensity=0.1):
    """Corrupt the image and save it."""
    print(f"Reading PPM image: {input_file}")
    width, height, max_val, pixel_data = read_ppm(input_file)
    print(f"  Dimensions: {width}x{height}, Max value: {max_val}")
    
    print(f"Applying {corruption_type} corruption (intensity: {intensity})...")
    
    if corruption_type == 'gaussian':
        corrupted_data = add_gaussian_noise(pixel_data, intensity)
    elif corruption_type == 'salt_pepper':
        corrupted_data = add_salt_pepper_noise(pixel_data, intensity)
    elif corruption_type == 'bit_flip':
        corrupted_data = add_random_bit_flips(pixel_data, intensity)
    elif corruption_type == 'blocks':
        num_blocks = int(intensity * 100)
        corrupted_data = add_random_blocks(pixel_data, width, height, num_blocks=num_blocks)
    elif corruption_type == 'streaks':
        num_streaks = int(intensity * 50)
        corrupted_data = add_horizontal_streaks(pixel_data, width, height, num_streaks=num_streaks)
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")
    
    print(f"Saving corrupted image: {output_file}")
    write_ppm(output_file, width, height, max_val, corrupted_data)
    print(f"✓ Done!")


def main():
    parser = argparse.ArgumentParser(
        description='Randomly corrupt a PPM image and save it with a new name',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 corrupt_ppm_image.py input.ppm output.ppm
  python3 corrupt_ppm_image.py input.ppm output.ppm --type salt_pepper --intensity 0.2
  python3 corrupt_ppm_image.py input.ppm output.ppm --type blocks --intensity 0.15
  python3 corrupt_ppm_image.py input.ppm output.ppm --type streaks --intensity 0.1

Corruption types:
  gaussian      - Add Gaussian noise
  salt_pepper   - Add salt and pepper noise (random black/white pixels)
  bit_flip      - Randomly flip bits in pixel data
  blocks        - Add random corrupted blocks
  streaks       - Add horizontal corruption streaks
        """
    )
    
    parser.add_argument('input', help='Input PPM image file')
    parser.add_argument('output', help='Output PPM image file')
    parser.add_argument('--type', '-t', dest='corruption_type', 
                       default='gaussian',
                       choices=['gaussian', 'salt_pepper', 'bit_flip', 'blocks', 'streaks'],
                       help='Type of corruption to apply (default: gaussian)')
    parser.add_argument('--intensity', '-i', type=float, default=0.1,
                       help='Intensity of corruption 0.0-1.0 (default: 0.1)')
    parser.add_argument('--seed', '-s', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Validate intensity
    if not 0.0 <= args.intensity <= 1.0:
        print("Error: Intensity must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    try:
        corrupt_image(args.input, args.output, args.corruption_type, args.intensity)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
