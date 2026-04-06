#!/usr/bin/env python3
"""
Convert PPM images to JPG format from the results folder.

This script:
1. Loads environment variables from .env file
2. Reads the results folder path from FFT_RESULTS_FOLDER
3. Finds all PPM files in that folder
4. Converts them to JPG format
5. Optionally deletes the original PPM files

Usage:
    python3 convert_ppm_to_jpg.py [--keep-ppm] [--quality 85]

Environment Variables:
    FFT_RESULTS_FOLDER - Path to results folder (default: ./results)
    JPG_QUALITY - JPEG quality (1-100, default: 85)
    DELETE_PPM - Delete original PPM files after conversion (default: false)
"""

import os
import sys
import glob
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv


def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get configuration from environment
    results_folder = os.getenv('FFT_RESULTS_FOLDER', './results')
    jpg_quality = int(os.getenv('JPG_QUALITY', '85'))
    delete_ppm = os.getenv('DELETE_PPM', 'false').lower() in ('true', '1', 'yes')
    
    # Parse command line arguments (override environment variables)
    keep_ppm = False
    for arg in sys.argv[1:]:
        if arg == '--keep-ppm':
            keep_ppm = True
            delete_ppm = False
        elif arg.startswith('--quality'):
            if '=' in arg:
                jpg_quality = int(arg.split('=')[1])
            elif len(sys.argv) > sys.argv.index(arg) + 1:
                jpg_quality = int(sys.argv[sys.argv.index(arg) + 1])
    
    # Validate quality
    if jpg_quality < 1 or jpg_quality > 100:
        print(f"Error: JPG quality must be between 1 and 100, got {jpg_quality}")
        sys.exit(1)
    
    # Check if results folder exists
    if not os.path.isdir(results_folder):
        print(f"Error: Results folder '{results_folder}' does not exist")
        sys.exit(1)
    
    print(f"Converting PPM files from: {results_folder}")
    print(f"JPG Quality: {jpg_quality}")
    print(f"Delete original PPM: {delete_ppm}")
    print()
    
    # Find all PPM files
    ppm_pattern = os.path.join(results_folder, '*.ppm')
    ppm_files = glob.glob(ppm_pattern)
    
    if not ppm_files:
        print(f"No PPM files found in {results_folder}")
        return 0
    
    print(f"Found {len(ppm_files)} PPM file(s)")
    print()
    
    successful_conversions = 0
    failed_conversions = 0
    
    # Convert each PPM file to JPG
    for ppm_file in sorted(ppm_files):
        try:
            # Get base filename without extension
            base_name = os.path.splitext(os.path.basename(ppm_file))[0]
            jpg_file = os.path.join(results_folder, f'{base_name}.jpg')
            
            # Open and convert image
            print(f"Converting: {os.path.basename(ppm_file)}", end=' ... ')
            
            # Open PPM image
            img = Image.open(ppm_file)
            
            # Convert to RGB if necessary (in case of mode issues)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as JPG
            img.save(jpg_file, 'JPEG', quality=jpg_quality, optimize=True)
            print(f"✓ ({os.path.getsize(jpg_file) / 1024:.1f} KB)")
            
            # Delete original PPM if requested
            if delete_ppm:
                os.remove(ppm_file)
                print(f"  Deleted original PPM")
            
            successful_conversions += 1
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            failed_conversions += 1
    
    # Print summary
    print()
    print("=" * 60)
    print(f"Conversion Summary:")
    print(f"  Successful: {successful_conversions}")
    print(f"  Failed: {failed_conversions}")
    print(f"  Total: {len(ppm_files)}")
    print("=" * 60)
    
    return 0 if failed_conversions == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
