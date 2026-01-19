# YOLO Tracking Benchmark - Quick Reference

## Quick Start Commands

### Basic Evaluation
```bash
# Evaluate dataset with default settings
python yolo_metrics.py --dataset ./my_dataset --weights yolov8n.pt --tracker bytetrack.yaml
```

### Batch Evaluation
```bash
# Create config file (batch_config.json)
# Then run batch evaluation
python batch_evaluate.py --dataset ./my_dataset --config batch_config.json

# Compare all results
python compare_results.py --results_dir ./my_dataset/results
```

### Quick Test (50 frames per scene)
```bash
python yolo_metrics.py --dataset ./my_dataset --weights yolov8n.pt --max_frames 50
```

## Command Line Arguments Reference

### yolo_metrics.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | **required** | Dataset directory path |
| `--weights` | str | **required** | YOLO model weights file |
| `--tracker` | str | `bytetrack.yaml` | Tracker configuration |
| `--conf` | float | `0.25` | Detection confidence threshold |
| `--iou` | float | `0.5` | NMS IoU threshold |
| `--match_iou` | float | `0.5` | GT-Pred matching IoU threshold |
| `--min_area` | int | `120` | Minimum GT mask area (pixels) |
| `--max_frames` | int | `-1` | Max frames per scene (-1 = all) |

### batch_evaluate.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | **required** | Dataset directory path |
| `--config` | str | **required** | JSON config file path |
| `--max_frames` | int | `-1` | Max frames per scene (-1 = all) |

### compare_results.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--results_dir` | str | **required** | Results directory path |
| `--sort_by` | str | `HOTA` | Metric to sort by (HOTA/MOTA/IDF1/DetA/AssA) |
| `--output` | str | `None` | Optional CSV output file |

## Batch Config Template

```json
{
  "models": [
    "yolov8n.pt",     // Nano - fastest
    "yolov8s.pt",     // Small
    "yolov8m.pt",     // Medium
    "yolov8l.pt",     // Large
    "yolov8x.pt"      // Extra large - most accurate
  ],
  "trackers": [
    "bytetrack.yaml",  // Fast, simple
    "botsort.yaml"     // More accurate, slower
  ],
  "conf_thresholds": [0.25, 0.3, 0.35],
  "iou_thresholds": [0.5, 0.6],
  "match_iou": 0.5,
  "min_area": 120
}
```

## Typical Workflows

### 1. Find Best Model
```bash
# Test multiple models with same tracker
python batch_evaluate.py --dataset ./data --config models_comparison.json
python compare_results.py --results_dir ./data/results --sort_by HOTA
```

**models_comparison.json:**
```json
{
  "models": ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
  "trackers": ["bytetrack.yaml"],
  "conf_thresholds": [0.25],
  "iou_thresholds": [0.5],
  "match_iou": 0.5,
  "min_area": 120
}
```

### 2. Find Best Tracker
```bash
# Test multiple trackers with same model
python batch_evaluate.py --dataset ./data --config trackers_comparison.json
python compare_results.py --results_dir ./data/results --sort_by HOTA
```

**trackers_comparison.json:**
```json
{
  "models": ["yolov8m.pt"],
  "trackers": ["bytetrack.yaml", "botsort.yaml"],
  "conf_thresholds": [0.25],
  "iou_thresholds": [0.5],
  "match_iou": 0.5,
  "min_area": 120
}
```

### 3. Optimize Thresholds
```bash
# Fine-tune confidence and IoU thresholds
python batch_evaluate.py --dataset ./data --config threshold_tuning.json
python compare_results.py --results_dir ./data/results --sort_by HOTA
```

**threshold_tuning.json:**
```json
{
  "models": ["yolov8m.pt"],
  "trackers": ["bytetrack.yaml"],
  "conf_thresholds": [0.2, 0.25, 0.3, 0.35, 0.4],
  "iou_thresholds": [0.4, 0.5, 0.6],
  "match_iou": 0.5,
  "min_area": 120
}
```

### 4. Full Sweep
```bash
# Test everything (this will take a while!)
python batch_evaluate.py --dataset ./data --config full_sweep.json
python compare_results.py --results_dir ./data/results --output full_results.csv
```

## Output Files

```
dataset/
├── scene1/
│   └── plottings/
│       ├── yolov8n_bytetrack_HOTA_curve.png
│       ├── yolov8n_bytetrack_DetA_AssA_curves.png
│       ├── yolov8n_bytetrack_LocA_curve.png
│       └── yolov8n_bytetrack_ALL_curves.png
├── scene2/
│   └── plottings/
│       └── ...
└── results/
    ├── yolov8n_bytetrack_20260119_143022.json
    ├── yolov8s_bytetrack_20260119_144530.json
    └── yolov8m_botsort_20260119_150145.json
```

## Metrics Quick Reference

| Metric | Range | Higher is Better | Description |
|--------|-------|------------------|-------------|
| **MOTA** | -∞ to 1.0 | ✓ | Overall tracking accuracy |
| **MOTP** | 0.0 to 1.0 | ✓ | Average IoU of matches |
| **IDF1** | 0.0 to 1.0 | ✓ | Identity preservation |
| **HOTA** | 0.0 to 1.0 | ✓ | Balanced tracking metric |
| **DetA** | 0.0 to 1.0 | ✓ | Detection accuracy |
| **AssA** | 0.0 to 1.0 | ✓ | Association accuracy |
| **LocA** | 0.0 to 1.0 | ✓ | Localization accuracy |
| **IDSW** | 0 to ∞ | ✗ | ID switch count (lower is better) |

## Performance Tips

1. **For speed**: Use yolov8n.pt with bytetrack.yaml
2. **For accuracy**: Use yolov8l.pt or yolov8x.pt with botsort.yaml
3. **For balance**: Use yolov8m.pt with bytetrack.yaml
4. **Low false positives**: Increase `--conf` threshold (0.3-0.4)
5. **Catch more objects**: Decrease `--conf` threshold (0.15-0.2)
6. **Better localization**: Use larger models (m/l/x)
7. **Better ID consistency**: Use botsort.yaml tracker

## Common Issues

### "No valid scene directories found"
- Ensure each scene has `rgb/` and `seg/` subdirectories
- Check scene directory names don't contain special characters

### "No matching RGB/SEG frame triplets found"
- Verify semantic files have consistent naming
- Check for both `.png` and `_info.json` files in seg/

### YOLO tracker not found
- Use full path to tracker config
- Or place in Ultralytics default tracker directory

### Out of memory
- Use smaller model (n/s instead of m/l/x)
- Reduce batch size in YOLO config
- Process fewer scenes at once

### Low metrics across all models
- Check ground truth quality
- Adjust `--match_iou` threshold
- Verify `--min_area` is appropriate
- Review semantic segmentation masks

## Example Session

```bash
# 1. Quick test with one model
python yolo_metrics.py --dataset ./traffic_dataset --weights yolov8n.pt --max_frames 50

# 2. Run full evaluation on promising models
python batch_evaluate.py --dataset ./traffic_dataset --config selected_models.json

# 3. Compare and find winner
python compare_results.py --results_dir ./traffic_dataset/results --output winners.csv

# 4. Review top result
cat winners.csv | head -n 2

# 5. Check plots for top performer
ls ./traffic_dataset/scene1/plottings/
```

## Getting Help

- Check README.md for detailed documentation
- Review example configs in batch_config_example.json
- Inspect result JSON files for structure
- Use `--help` flag on any script

```bash
python yolo_metrics.py --help
python batch_evaluate.py --help
python compare_results.py --help
```
