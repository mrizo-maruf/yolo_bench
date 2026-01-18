#!/usr/bin/env python3
"""
Quick start example for YOLO benchmarking.
This script demonstrates how to use the benchmarking tool with the COCO dataset.
"""

import yaml
from pathlib import Path


def create_example_config():
    """Create an example configuration for testing with COCO dataset."""
    
    # Example configuration using COCO128 dataset (small subset of COCO)
    config = {
        'dataset': {
            'data_yaml': 'coco128.yaml'  # Will be downloaded automatically by ultralytics
        },
        'models': {
            'yolov8': [
                'yolov8n.pt',  # Nano - fastest
                'yolov8s.pt',  # Small
            ],
            'yolov11': [
                'yolo11n.pt',  # Nano
                'yolo11s.pt',  # Small
            ]
        },
        'benchmark': {
            'imgsz': 640,
            'batch': 16,
            'conf': 0.25,
            'iou': 0.45,
            'device': 0,  # Use GPU 0, change to 'cpu' if no GPU
            'warmup': 5,
            'iterations': 50,
            'save_images': True,
            'max_images': 10
        },
        'output': {
            'results_dir': 'results',
            'generate_plots': True,
            'save_csv': True,
            'verbose': True
        }
    }
    
    # Save configuration
    config_path = Path('example_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Example configuration created: {config_path}")
    print("\nTo run the benchmark with this configuration:")
    print(f"  python benchmark.py --config {config_path}")
    print("\nNote: This will use the COCO128 dataset, which will be downloaded automatically.")
    print("For your custom dataset, modify the 'data_yaml' path in the config file.")
    
    return config_path


def main():
    """Main function."""
    print("="*60)
    print("YOLO Benchmark - Quick Start Example")
    print("="*60)
    print()
    
    # Create example config
    config_path = create_example_config()
    
    print("\n" + "="*60)
    print("Quick Start Steps:")
    print("="*60)
    print()
    print("1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("2. Run the example benchmark:")
    print(f"   python benchmark.py --config {config_path}")
    print()
    print("3. Check results in the 'results' directory:")
    print("   - benchmark_results.csv")
    print("   - benchmark_summary.txt")
    print("   - plots/benchmark_comparison.png")
    print("   - plots/speed_accuracy_tradeoff.png")
    print()
    print("4. For custom datasets:")
    print("   - Prepare your dataset in YOLO format")
    print("   - Create a data.yaml file (see data.yaml.example)")
    print("   - Update config.yaml with your data.yaml path")
    print("   - Run: python benchmark.py --config config.yaml")
    print()
    print("="*60)


if __name__ == '__main__':
    main()
