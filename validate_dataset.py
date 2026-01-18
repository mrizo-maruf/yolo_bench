#!/usr/bin/env python3
"""
Utility script to validate and prepare datasets for YOLO benchmarking.
"""

import argparse
import yaml
from pathlib import Path
from collections import defaultdict


def validate_dataset(data_yaml_path: str):
    """Validate a YOLO format dataset."""
    print(f"Validating dataset: {data_yaml_path}")
    
    # Load data.yaml
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Check required fields
    required_fields = ['path', 'train', 'val', 'names', 'nc']
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        print(f"❌ Missing required fields: {', '.join(missing_fields)}")
        return False
    
    print("✓ All required fields present")
    
    # Validate paths
    dataset_path = Path(data['path'])
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return False
    
    print(f"✓ Dataset path exists: {dataset_path}")
    
    # Check train and val directories
    for split in ['train', 'val']:
        if split in data:
            images_path = dataset_path / data[split]
            # Use pathlib for cross-platform compatibility
            split_name = Path(data[split]).name
            labels_path = dataset_path / 'labels' / split_name
            
            if not images_path.exists():
                print(f"❌ {split} images path does not exist: {images_path}")
                return False
            
            # Count images
            image_files = list(images_path.glob('*.jpg')) + \
                         list(images_path.glob('*.png')) + \
                         list(images_path.glob('*.jpeg'))
            
            print(f"✓ {split} images: {len(image_files)} files in {images_path}")
            
            if labels_path.exists():
                label_files = list(labels_path.glob('*.txt'))
                print(f"✓ {split} labels: {len(label_files)} files in {labels_path}")
                
                if len(label_files) != len(image_files):
                    print(f"⚠️  Warning: Mismatch between images ({len(image_files)}) and labels ({len(label_files)})")
            else:
                print(f"⚠️  Warning: Labels directory not found: {labels_path}")
    
    # Validate classes
    num_classes = data['nc']
    class_names = data['names']
    
    if isinstance(class_names, dict):
        if len(class_names) != num_classes:
            print(f"⚠️  Warning: Number of class names ({len(class_names)}) doesn't match nc ({num_classes})")
    elif isinstance(class_names, list):
        if len(class_names) != num_classes:
            print(f"⚠️  Warning: Number of class names ({len(class_names)}) doesn't match nc ({num_classes})")
    
    print(f"✓ Number of classes: {num_classes}")
    print(f"✓ Class names: {class_names}")
    
    print("\n✅ Dataset validation completed successfully!")
    return True


def generate_dataset_stats(data_yaml_path: str):
    """Generate statistics for a dataset."""
    print(f"\nGenerating dataset statistics...")
    
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    dataset_path = Path(data['path'])
    stats = defaultdict(lambda: defaultdict(int))
    
    # Analyze each split
    for split in ['train', 'val', 'test']:
        if split not in data:
            continue
        
        # Use pathlib for cross-platform compatibility
        split_name = Path(data[split]).name if isinstance(data[split], str) else split
        labels_path = dataset_path / 'labels' / split_name
        if not labels_path.exists():
            continue
        
        label_files = list(labels_path.glob('*.txt'))
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip():
                        class_id = int(line.split()[0])
                        stats[split][class_id] += 1
    
    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    
    class_names = data['names']
    if isinstance(class_names, dict):
        class_list = [class_names[i] for i in range(len(class_names))]
    else:
        class_list = class_names
    
    for split in stats:
        print(f"\n{split.upper()} Split:")
        print("-" * 40)
        total = sum(stats[split].values())
        for class_id in sorted(stats[split].keys()):
            count = stats[split][class_id]
            class_name = class_list[class_id] if class_id < len(class_list) else f"class_{class_id}"
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {class_name:20s}: {count:6d} ({percentage:5.1f}%)")
        print(f"  {'TOTAL':20s}: {total:6d}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Validate and analyze YOLO format datasets'
    )
    parser.add_argument(
        'data_yaml',
        type=str,
        help='Path to data.yaml file'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Generate detailed statistics'
    )
    
    args = parser.parse_args()
    
    # Validate dataset
    if validate_dataset(args.data_yaml):
        # Generate statistics if requested
        if args.stats:
            generate_dataset_stats(args.data_yaml)
    else:
        print("\n❌ Dataset validation failed!")
        exit(1)


if __name__ == '__main__':
    main()
