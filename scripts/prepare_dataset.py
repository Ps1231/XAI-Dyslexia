"""
Download and prepare datasets for training.

Usage:
    python scripts/prepare_dataset.py --dataset all --output data/processed
"""
import os
import sys
import argparse
import zipfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DATASETS = {
    'dysgraphia_mendeley': {
        'name': 'Potential Dysgraphia Handwriting Dataset',
        'url': 'https://data.mendeley.com/datasets/39hr8dx76p/1',
        'doi': '10.17632/39hr8dx76p.1',
        'type': 'images',
    },
    'dyslexia_handwriting': {
        'name': 'Dyslexia Handwriting Dataset',
        'url': 'https://www.kaggle.com/datasets/drizasazanitaisa/dyslexia-handwriting-dataset',
        'type': 'images',
    },
    'dyslexia_risk': {
        'name': 'Predicting Risk of Dyslexia',
        'url': 'https://www.kaggle.com/datasets/luzrello/dyslexia',
        'type': 'tabular',
    },
    'dyslexia_synthetic': {
        'name': 'Synthetic Dyslexia Handwriting Dataset',
        'url': 'https://www.kaggle.com/datasets/michaelfink0923/synthetic-dyslexia-handwriting-dataset',
        'type': 'images',
    },
}


def extract_zip(zip_path, dest_dir):
    """Extract a zip file to destination directory."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest_dir)
    print(f"  Extracted to {dest_dir}")


def organize_dysgraphia_images(source_dir, output_dir):
    """Organize downloaded Mendeley dataset into class folders."""
    target_dir = os.path.join(output_dir, 'dysgraphia')
    os.makedirs(target_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                folder_name = os.path.basename(root).lower().replace(' ', '_')
                class_dir = os.path.join(target_dir, folder_name)
                os.makedirs(class_dir, exist_ok=True)
                shutil.copy2(os.path.join(root, f), os.path.join(class_dir, f))
    
    print(f"  Organized images into {target_dir}")


def organize_dyslexia_images(source_dir, output_dir):
    """Organize Kaggle handwriting datasets into class folders."""
    target_dir = os.path.join(output_dir, 'dyslexia_handwriting')
    os.makedirs(target_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                rel_path = os.path.relpath(root, source_dir)
                folder_name = rel_path.replace(os.sep, '_').lower().replace(' ', '_')
                class_dir = os.path.join(target_dir, folder_name)
                os.makedirs(class_dir, exist_ok=True)
                shutil.copy2(os.path.join(root, f), os.path.join(class_dir, f))
    
    print(f"  Organized images into {target_dir}")


def main():
    parser = argparse.ArgumentParser(description='Prepare datasets for training')
    parser.add_argument('--dataset', choices=['dysgraphia_mendeley', 'dyslexia_handwriting',
                                               'dyslexia_risk', 'dyslexia_synthetic', 'all'],
                        default='all')
    parser.add_argument('--raw-dir', default='data/raw', help='Directory with downloaded files')
    parser.add_argument('--output', default='data/processed', help='Output directory')
    parser.add_argument('--zip-path', default=None, help='Path to downloaded zip file')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.raw_dir, exist_ok=True)
    
    print("=== Dataset Preparation ===\n")
    print("Manual download instructions:")
    print("  1. Download datasets from the URLs below")
    print("  2. Place zip files in data/raw/")
    print("  3. Run this script to organize them\n")
    
    for key, info in DATASETS.items():
        print(f"  {key}:")
        print(f"    Name: {info['name']}")
        print(f"    URL:  {info['url']}")
        print(f"    Type: {info['type']}")
        if info.get('doi'):
            print(f"    DOI:  {info['doi']}")
        print()
    
    if args.zip_path and os.path.exists(args.zip_path):
        print(f"\nProcessing: {args.zip_path}")
        temp_dir = os.path.join(args.raw_dir, 'temp_extract')
        extract_zip(args.zip_path, temp_dir)
        
        if args.dataset == 'dysgraphia_mendeley':
            organize_dysgraphia_images(temp_dir, args.output)
        elif args.dataset in ('dyslexia_handwriting', 'dyslexia_synthetic'):
            organize_dyslexia_images(temp_dir, args.output)
        
        shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        print("\nNo zip file provided. Place downloaded files in data/raw/ and re-run.")
        print("Example: python scripts/prepare_dataset.py --zip-path data/raw/dataset.zip")


if __name__ == '__main__':
    main()
