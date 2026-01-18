# yolo_bench

A comprehensive benchmarking tool for comparing YOLOv8 and YOLOv11 models on custom datasets.

## Features

- 🚀 Benchmark multiple YOLO models (YOLOv8 and YOLOv11) simultaneously
- 📊 Comprehensive metrics: mAP, Precision, Recall, FPS, and more
- 📈 Automatic visualization of comparison results
- ⚙️ Flexible configuration via YAML files
- 🎯 Support for custom datasets in YOLO format
- 💾 Export results to CSV and generate detailed reports

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mrizo-maruf/yolo_bench.git
cd yolo_bench
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Dataset

Your dataset should be in YOLO format with the following structure:

```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
│   └── val/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
└── data.yaml
```

Create a `data.yaml` file (see `data.yaml.example` for reference):

```yaml
path: /path/to/your/dataset
train: images/train
val: images/val
names:
  0: class1
  1: class2
  # ... add your classes
nc: 2  # number of classes
```

### 2. Configure Benchmark

Edit `config.yaml` to customize your benchmark:

```yaml
dataset:
  data_yaml: "path/to/your/dataset/data.yaml"

models:
  yolov8:
    - yolov8n.pt  # nano
    - yolov8s.pt  # small
    - yolov8m.pt  # medium
  
  yolov11:
    - yolo11n.pt  # nano
    - yolo11s.pt  # small
    - yolo11m.pt  # medium

benchmark:
  imgsz: 640
  batch: 16
  device: 0  # GPU device or 'cpu'
```

### 3. Run Benchmark

```bash
python benchmark.py --config config.yaml
```

## Usage

### Basic Usage

```bash
python benchmark.py
```

This will use the default `config.yaml` configuration file.

### Custom Configuration

```bash
python benchmark.py --config my_custom_config.yaml
```

### Configuration Options

#### Dataset Configuration

```yaml
dataset:
  data_yaml: "path/to/your/dataset/data.yaml"
```

#### Model Selection

You can benchmark any combination of YOLOv8 and YOLOv11 models:

```yaml
models:
  yolov8:
    - yolov8n.pt   # Nano - fastest, least accurate
    - yolov8s.pt   # Small
    - yolov8m.pt   # Medium
    - yolov8l.pt   # Large
    - yolov8x.pt   # Extra Large - slowest, most accurate
  
  yolov11:
    - yolo11n.pt   # Nano
    - yolo11s.pt   # Small
    - yolo11m.pt   # Medium
    - yolo11l.pt   # Large
    - yolo11x.pt   # Extra Large
```

#### Benchmark Settings

```yaml
benchmark:
  imgsz: 640           # Image size for inference
  batch: 16            # Batch size
  conf: 0.25           # Confidence threshold
  iou: 0.45            # IoU threshold for NMS
  device: 0            # GPU device (0, 1, ...) or 'cpu'
  warmup: 10           # Number of warmup iterations
  iterations: 100      # Number of benchmark iterations
  save_images: true    # Save prediction visualizations
  max_images: 10       # Maximum images to save
```

#### Output Settings

```yaml
output:
  results_dir: "results"     # Directory to save results
  generate_plots: true       # Generate comparison plots
  save_csv: true             # Save metrics to CSV
  verbose: true              # Verbose output
```

## Output

After running the benchmark, you'll find the following in the `results` directory:

### Files Generated

- `benchmark_results.csv` - Detailed metrics for all models
- `benchmark_summary.txt` - Human-readable summary
- `plots/benchmark_comparison.png` - Comprehensive comparison charts
- `plots/speed_accuracy_tradeoff.png` - Speed vs accuracy scatter plot
- `images/` - Sample prediction visualizations (if enabled)

### Metrics Collected

For each model, the following metrics are collected:

- **mAP50** - Mean Average Precision at IoU threshold 0.5
- **mAP50-95** - Mean Average Precision averaged over IoU thresholds 0.5 to 0.95
- **Precision** - Precision score
- **Recall** - Recall score
- **FPS** - Frames per second (inference speed)
- **Parameters** - Total number of model parameters
- **Inference Time** - Time for inference in milliseconds
- **Preprocessing Time** - Time for preprocessing in milliseconds
- **Postprocessing Time** - Time for postprocessing in milliseconds

## Example Results

```
Model: yolov8n.pt (YOLOv8)
  mAP50: 0.6523
  mAP50-95: 0.4512
  Precision: 0.7123
  Recall: 0.6234
  FPS: 145.23
  Parameters: 3,157,200

Model: yolo11n.pt (YOLOv11)
  mAP50: 0.6789
  mAP50-95: 0.4723
  Precision: 0.7345
  Recall: 0.6456
  FPS: 138.45
  Parameters: 2,632,000
```

## Visualization

The tool generates several comparison plots:

1. **mAP Comparison** - Bar chart comparing mAP50 and mAP50-95
2. **Speed Comparison** - Bar chart showing FPS for each model
3. **Precision vs Recall** - Scatter plot of precision vs recall
4. **Model Size** - Bar chart of model parameters
5. **Speed vs Accuracy Trade-off** - Scatter plot showing the relationship between speed and accuracy

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Ultralytics (for YOLO models)
- CUDA-capable GPU (recommended for better performance)

## Tips for Best Results

1. **Use a representative validation set** - Ensure your validation set accurately represents your use case
2. **Consistent environment** - Run all benchmarks on the same hardware for fair comparison
3. **Warmup iterations** - Allow sufficient warmup iterations for stable measurements
4. **Batch size** - Adjust batch size based on your GPU memory
5. **Image size** - Test with the image size you'll use in production

## Troubleshooting

### CUDA Out of Memory

Reduce the batch size in `config.yaml`:
```yaml
benchmark:
  batch: 8  # or smaller
```

### Models Not Found

The first time you run the benchmark, YOLO models will be automatically downloaded. Ensure you have internet connectivity.

### Custom Dataset Issues

Verify your `data.yaml` file paths are correct and the dataset structure follows YOLO format.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO implementations
- YOLOv8 and YOLOv11 model architectures

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{yolo_bench,
  author = {mrizo-maruf},
  title = {yolo_bench: Benchmarking Tool for YOLO Models},
  year = {2026},
  url = {https://github.com/mrizo-maruf/yolo_bench}
}
```