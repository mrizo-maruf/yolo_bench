#!/usr/bin/env python3
"""
YOLO Benchmarking Script
Benchmark YOLOv8 and YOLOv11 models on custom datasets.
"""

import argparse
import time
from pathlib import Path
import yaml
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm


class YOLOBenchmark:
    """Benchmark YOLO models on custom datasets."""
    
    def __init__(self, config_path: str):
        """Initialize the benchmark with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results = []
        self.output_dir = Path(self.config['output']['results_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = self.output_dir / 'plots'
        self.images_dir = self.output_dir / 'images'
        if self.config['output']['generate_plots']:
            self.plots_dir.mkdir(exist_ok=True)
        if self.config['benchmark']['save_images']:
            self.images_dir.mkdir(exist_ok=True)
    
    def benchmark_model(self, model_path: str, model_version: str):
        """Benchmark a single YOLO model."""
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_path} ({model_version})")
        print(f"{'='*60}")
        
        try:
            # Load model
            model = YOLO(model_path)
            
            # Get benchmark config
            bench_config = self.config['benchmark']
            
            # Warmup
            print(f"Warming up for {bench_config['warmup']} iterations...")
            for _ in range(bench_config['warmup']):
                # Use a dummy tensor for warmup
                dummy_input = torch.randn(1, 3, bench_config['imgsz'], bench_config['imgsz'])
                if torch.cuda.is_available() and bench_config['device'] != 'cpu':
                    dummy_input = dummy_input.cuda()
            
            # Validate on dataset
            print("Running validation...")
            val_results = model.val(
                data=self.config['dataset']['data_yaml'],
                imgsz=bench_config['imgsz'],
                batch=bench_config['batch'],
                conf=bench_config['conf'],
                iou=bench_config['iou'],
                device=bench_config['device'],
                verbose=self.config['output']['verbose']
            )
            
            # Get metrics
            metrics = {
                'model': model_path,
                'version': model_version,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'parameters': sum(p.numel() for p in model.model.parameters()),
                'map50': val_results.results_dict.get('metrics/mAP50(B)', 0),
                'map50_95': val_results.results_dict.get('metrics/mAP50-95(B)', 0),
                'precision': val_results.results_dict.get('metrics/precision(B)', 0),
                'recall': val_results.results_dict.get('metrics/recall(B)', 0),
            }
            
            # Measure inference speed
            print("Measuring inference speed...")
            inference_times = []
            
            # Get a sample batch from validation set
            # Use model's built-in speed test
            speed_results = model.val(
                data=self.config['dataset']['data_yaml'],
                imgsz=bench_config['imgsz'],
                batch=1,  # Single image for speed test
                conf=bench_config['conf'],
                device=bench_config['device'],
                verbose=False
            )
            
            # Extract speed metrics
            if hasattr(speed_results, 'speed'):
                preprocess_time = speed_results.speed.get('preprocess', 0)
                inference_time = speed_results.speed.get('inference', 0)
                postprocess_time = speed_results.speed.get('postprocess', 0)
                
                metrics['preprocess_ms'] = preprocess_time
                metrics['inference_ms'] = inference_time
                metrics['postprocess_ms'] = postprocess_time
                metrics['total_ms'] = preprocess_time + inference_time + postprocess_time
                metrics['fps'] = 1000.0 / metrics['total_ms'] if metrics['total_ms'] > 0 else 0
            else:
                metrics['preprocess_ms'] = 0
                metrics['inference_ms'] = 0
                metrics['postprocess_ms'] = 0
                metrics['total_ms'] = 0
                metrics['fps'] = 0
            
            self.results.append(metrics)
            
            # Print summary
            print(f"\nResults for {model_path}:")
            print(f"  mAP50: {metrics['map50']:.4f}")
            print(f"  mAP50-95: {metrics['map50_95']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  FPS: {metrics['fps']:.2f}")
            print(f"  Parameters: {metrics['parameters']:,}")
            
            return metrics
            
        except Exception as e:
            print(f"Error benchmarking {model_path}: {str(e)}")
            return None
    
    def run_benchmark(self):
        """Run benchmark on all configured models."""
        print("Starting YOLO Benchmark")
        print(f"Dataset: {self.config['dataset']['data_yaml']}")
        print(f"Device: {self.config['benchmark']['device']}")
        print(f"Image Size: {self.config['benchmark']['imgsz']}")
        
        # Benchmark YOLOv8 models
        if 'yolov8' in self.config['models']:
            for model_path in self.config['models']['yolov8']:
                self.benchmark_model(model_path, 'YOLOv8')
        
        # Benchmark YOLOv11 models
        if 'yolov11' in self.config['models']:
            for model_path in self.config['models']['yolov11']:
                self.benchmark_model(model_path, 'YOLOv11')
        
        # Save results
        self.save_results()
        
        # Generate comparison plots
        if self.config['output']['generate_plots']:
            self.generate_plots()
        
        print(f"\n{'='*60}")
        print("Benchmark completed!")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*60}")
    
    def save_results(self):
        """Save benchmark results to CSV."""
        if not self.results:
            print("No results to save.")
            return
        
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / 'benchmark_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Also save a summary
        summary_path = self.output_dir / 'benchmark_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("YOLO Benchmark Summary\n")
            f.write("=" * 60 + "\n\n")
            for result in self.results:
                f.write(f"Model: {result['model']} ({result['version']})\n")
                f.write(f"  mAP50: {result['map50']:.4f}\n")
                f.write(f"  mAP50-95: {result['map50_95']:.4f}\n")
                f.write(f"  Precision: {result['precision']:.4f}\n")
                f.write(f"  Recall: {result['recall']:.4f}\n")
                f.write(f"  FPS: {result['fps']:.2f}\n")
                f.write(f"  Parameters: {result['parameters']:,}\n")
                f.write("\n")
    
    def generate_plots(self):
        """Generate comparison plots."""
        if not self.results:
            print("No results to plot.")
            return
        
        df = pd.DataFrame(self.results)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('YOLO Model Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: mAP Comparison
        ax1 = axes[0, 0]
        x = range(len(df))
        width = 0.35
        ax1.bar([i - width/2 for i in x], df['map50'], width, label='mAP50', alpha=0.8)
        ax1.bar([i + width/2 for i in x], df['map50_95'], width, label='mAP50-95', alpha=0.8)
        ax1.set_xlabel('Model')
        ax1.set_ylabel('mAP')
        ax1.set_title('Mean Average Precision Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{r['model']}\n({r['version']})" for r in self.results], 
                            rotation=45, ha='right', fontsize=8)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Speed Comparison (FPS)
        ax2 = axes[0, 1]
        ax2.bar(x, df['fps'], alpha=0.8, color='green')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('FPS')
        ax2.set_title('Inference Speed (Frames Per Second)')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{r['model']}\n({r['version']})" for r in self.results], 
                            rotation=45, ha='right', fontsize=8)
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Precision vs Recall
        ax3 = axes[1, 0]
        colors = ['blue' if r['version'] == 'YOLOv8' else 'red' for r in self.results]
        ax3.scatter(df['recall'], df['precision'], c=colors, alpha=0.6, s=100)
        for i, result in enumerate(self.results):
            ax3.annotate(result['model'], 
                        (df['recall'].iloc[i], df['precision'].iloc[i]),
                        fontsize=8, alpha=0.7)
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision vs Recall')
        ax3.grid(alpha=0.3)
        
        # Add legend for colors
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', alpha=0.6, label='YOLOv8'),
                          Patch(facecolor='red', alpha=0.6, label='YOLOv11')]
        ax3.legend(handles=legend_elements)
        
        # Plot 4: Model Size (Parameters)
        ax4 = axes[1, 1]
        ax4.bar(x, [r['parameters'] / 1e6 for r in self.results], alpha=0.8, color='orange')
        ax4.set_xlabel('Model')
        ax4.set_ylabel('Parameters (Millions)')
        ax4.set_title('Model Size Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f"{r['model']}\n({r['version']})" for r in self.results], 
                            rotation=45, ha='right', fontsize=8)
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plots_dir / 'benchmark_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plots saved to: {plot_path}")
        plt.close()
        
        # Create additional plot: Speed vs Accuracy trade-off
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['blue' if r['version'] == 'YOLOv8' else 'red' for r in self.results]
        scatter = ax.scatter(df['fps'], df['map50_95'], c=colors, alpha=0.6, s=200)
        
        for i, result in enumerate(self.results):
            ax.annotate(result['model'], 
                       (df['fps'].iloc[i], df['map50_95'].iloc[i]),
                       fontsize=9, alpha=0.8)
        
        ax.set_xlabel('FPS (Frames Per Second)', fontsize=12)
        ax.set_ylabel('mAP50-95', fontsize=12)
        ax.set_title('Speed vs Accuracy Trade-off', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        legend_elements = [Patch(facecolor='blue', alpha=0.6, label='YOLOv8'),
                          Patch(facecolor='red', alpha=0.6, label='YOLOv11')]
        ax.legend(handles=legend_elements)
        
        plot_path = self.plots_dir / 'speed_accuracy_tradeoff.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Speed vs Accuracy plot saved to: {plot_path}")
        plt.close()


def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(
        description='Benchmark YOLOv8 and YOLOv11 models on custom datasets'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"Error: Configuration file '{args.config}' not found.")
        print("Please create a config.yaml file or specify a different config with --config")
        return
    
    # Run benchmark
    benchmark = YOLOBenchmark(args.config)
    benchmark.run_benchmark()


if __name__ == '__main__':
    main()
