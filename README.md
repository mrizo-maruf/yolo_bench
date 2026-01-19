# YOLO Tracking Benchmarking Suite

A comprehensive toolkit for evaluating YOLO object tracking performance across multiple scenes and configurations.

## Features

- **Multi-Scene Evaluation**: Evaluate entire datasets with multiple scenes
- **Per-Scene and Overall Metrics**: Get detailed metrics for each scene plus dataset-wide aggregates
- **Multiple Tracking Metrics**:
  - MOTA/MOTP (CLEAR MOT metrics)
  - IDF1 (Identity F1 Score)
  - HOTA (Higher Order Tracking Accuracy)
  - DetA, AssA, LocA (Detection, Association, Localization Accuracy)
- **Configuration Tracking**: Save model weights, tracker configs, and thresholds with results
- **Batch Evaluation**: Test multiple model/tracker combinations automatically
- **Result Comparison**: Compare results across different configurations

## Installation

```bash
pip install ultralytics opencv-python numpy scipy matplotlib pandas
```

## Dataset Structure

Your dataset should be organized as follows:

```
dataset/
├── scene1/
│   ├── rgb/           # RGB frames (jpg, png)
│   └── seg/           # Semantic segmentation
│       ├── *_semantic*.png      # Segmentation masks
│       └── *_semantic*_info.json # Metadata
├── scene2/
│   ├── rgb/
│   └── seg/
└── ...
```

## Usage

### 1. Single Evaluation

Evaluate one model/tracker configuration on a dataset:

```bash
python yolo_metrics.py \
    --dataset /path/to/dataset \
    --weights yolov8n.pt \
    --tracker bytetrack.yaml \
    --conf 0.25 \
    --iou 0.5 \
    --match_iou 0.5
```

**Arguments:**
- `--dataset`: Path to dataset directory (contains scene subdirectories)
- `--weights`: YOLO model weights (e.g., yolov8n.pt, yolov8s.pt)
- `--tracker`: Tracker config (bytetrack.yaml, botsort.yaml)
- `--conf`: Detection confidence threshold (default: 0.25)
- `--iou`: NMS IoU threshold (default: 0.5)
- `--match_iou`: IoU threshold for GT-Pred matching (default: 0.5)
- `--min_area`: Minimum GT mask area in pixels (default: 120)
- `--max_frames`: Max frames per scene, -1 for all (default: -1)

**Output:**
- Results saved to `dataset/results/<model>_<tracker>_<timestamp>.json`
- Plots saved to each `scene/plottings/` directory

### 2. Batch Evaluation

Test multiple configurations automatically:

```bash
python batch_evaluate.py \
    --dataset /path/to/dataset \
    --config batch_config.json
```

**Example config file** (`batch_config.json`):
```json
{
  "models": ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
  "trackers": ["bytetrack.yaml", "botsort.yaml"],
  "conf_thresholds": [0.25, 0.3, 0.35],
  "iou_thresholds": [0.5, 0.6],
  "match_iou": 0.5,
  "min_area": 120
}
```

See `batch_config_example.json` for a template.

### 3. Compare Results

Compare all evaluation results:

```bash
python compare_results.py \
    --results_dir /path/to/dataset/results \
    --sort_by HOTA \
    --output comparison.csv
```

**Arguments:**
- `--results_dir`: Directory containing result JSON files
- `--sort_by`: Metric to sort by (HOTA, MOTA, IDF1, DetA, AssA)
- `--output`: Optional CSV output file

**Output:**
- Console table showing all results sorted by chosen metric
- Best configurations for each metric
- Optional CSV export

## Output Format

### Results JSON Structure

```json
{
  "evaluation_info": {
    "timestamp": "20260119_143022",
    "dataset_path": "/path/to/dataset",
    "model_weights": "yolov8n.pt",
    "tracker_config": "bytetrack.yaml",
    "conf_threshold": 0.25,
    "nms_iou_threshold": 0.5,
    "match_iou_threshold": 0.5,
    "min_area": 120,
    "max_frames_per_scene": -1
  },
  "overall_metrics": {
    "total_scenes": 3,
    "total_frames": 450,
    "MOTA": 0.856234,
    "MOTP": 0.723456,
    "IDF1": 0.789012,
    "HOTA": 0.678901,
    "DetA": 0.745678,
    "AssA": 0.612345,
    "LocA": 0.823456,
    "TP": 1234,
    "FP": 56,
    "FN": 78,
    "IDSW": 12
  },
  "per_scene_metrics": [
    {
      "scene_name": "scene1",
      "num_frames": 150,
      "MOTA": 0.850000,
      "MOTP": 0.720000,
      "IDF1": 0.780000,
      ...
    },
    ...
  ]
}
```

## Metrics Explained

- **MOTA** (Multiple Object Tracking Accuracy): Overall tracking accuracy considering FP, FN, and ID switches
- **MOTP** (Multiple Object Tracking Precision): Average IoU of matched detections
- **IDF1**: Identity F1 score, measures identity preservation across frames
- **HOTA**: Higher Order Tracking Accuracy, balanced metric combining detection and association
- **DetA**: Detection Accuracy component of HOTA
- **AssA**: Association Accuracy component of HOTA
- **LocA**: Localization Accuracy (average IoU)

## Workflow Example

```bash
# 1. Evaluate all configurations
python batch_evaluate.py --dataset ./my_dataset --config batch_config.json

# 2. Compare results
python compare_results.py --results_dir ./my_dataset/results --output comparison.csv

# 3. Review best configurations
cat comparison.csv | head -n 5
```

## Module Overview

- `yolo_metrics.py`: Main evaluation script
- `batch_evaluate.py`: Batch evaluation runner
- `compare_results.py`: Results comparison tool
- `io_utils.py`: Dataset loading utilities
- `metrics.py`: Tracking metrics accumulators
- `matching.py`: IoU computation and Hungarian matching
- `yolo_utils.py`: YOLO result extraction
- `visualization.py`: Plot generation

## Tips

1. **Start with small batches**: Test a few configurations first before running full batch
2. **Use max_frames for quick tests**: Set `--max_frames 50` for rapid iteration
3. **Monitor GPU memory**: Large models may require more VRAM
4. **Compare same match_iou**: Keep match_iou constant across runs for fair comparison
5. **Check plots**: Review HOTA curves in scene/plottings/ directories

## Troubleshooting

- **No scenes found**: Ensure each scene has rgb/ and seg/ directories
- **Model not found**: Provide full path or ensure model is in current directory
- **Tracker not found**: Use full path or ensure tracker yaml is in Ultralytics trackers directory
- **Memory errors**: Reduce batch size or use smaller model

## License

MIT