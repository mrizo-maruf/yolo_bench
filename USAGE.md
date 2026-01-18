# YOLO Benchmark - Usage Guide

## Overview
This tool benchmarks YOLOv8 and YOLOv11 models on custom datasets, providing comprehensive performance comparisons.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Example with COCO128
```bash
python quick_start.py
python benchmark.py --config example_config.yaml
```

### 3. Benchmark Custom Dataset

#### Prepare Your Dataset
Ensure your dataset follows YOLO format:
```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

#### Create data.yaml
```yaml
path: /path/to/dataset
train: images/train
val: images/val
names:
  0: class1
  1: class2
nc: 2
```

#### Validate Dataset (Optional)
```bash
python validate_dataset.py /path/to/data.yaml --stats
```

#### Configure Benchmark
Edit `config.yaml`:
```yaml
dataset:
  data_yaml: "/path/to/data.yaml"

models:
  yolov8:
    - yolov8n.pt
    - yolov8s.pt
  yolov11:
    - yolo11n.pt
    - yolo11s.pt

benchmark:
  imgsz: 640
  batch: 16
  device: 0  # or 'cpu'
```

#### Run Benchmark
```bash
python benchmark.py --config config.yaml
```

## Output Files

After running, check the `results/` directory:
- `benchmark_results.csv` - Detailed metrics table
- `benchmark_summary.txt` - Human-readable summary
- `plots/benchmark_comparison.png` - Comparison charts
- `plots/speed_accuracy_tradeoff.png` - Speed vs accuracy plot

## Metrics Explained

- **mAP50**: Mean Average Precision at IoU=0.5
- **mAP50-95**: Mean Average Precision averaged over IoU 0.5-0.95
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **FPS**: Frames Per Second (inference speed)
- **Parameters**: Total model parameters

## Tips

1. **GPU Recommended**: Benchmarking is much faster on GPU
2. **Batch Size**: Adjust based on GPU memory (reduce if OOM errors)
3. **Image Size**: 640 is standard, but you can use 320, 480, 800, 1280
4. **Model Selection**: Start with nano/small models, add larger ones as needed
5. **Warmup**: Keep warmup iterations ≥10 for stable measurements

## Common Issues

### CUDA Out of Memory
```yaml
benchmark:
  batch: 8  # Reduce batch size
```

### Models Not Downloading
- Ensure internet connectivity
- Models download automatically on first use

### Dataset Not Found
- Verify `data.yaml` path is absolute or relative to config.yaml
- Check paths in data.yaml are relative to the yaml file location

## Examples

### Compare All YOLOv8 Sizes
```yaml
models:
  yolov8:
    - yolov8n.pt
    - yolov8s.pt
    - yolov8m.pt
    - yolov8l.pt
    - yolov8x.pt
```

### CPU-Only Benchmarking
```yaml
benchmark:
  device: cpu
  batch: 1
```

### High-Resolution Images
```yaml
benchmark:
  imgsz: 1280
  batch: 4
```

## Support

For issues or questions:
1. Check the README.md for detailed documentation
2. Validate your dataset with `validate_dataset.py`
3. Review the example config files
4. Open an issue on GitHub

## License
MIT License
