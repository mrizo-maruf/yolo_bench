# Dataset Structure Example

This document shows the expected dataset structure for the YOLO tracking benchmark.

## Directory Layout

```
my_dataset/
│
├── scene_001/
│   ├── rgb/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   ├── 000003.jpg
│   │   └── ...
│   │
│   └── seg/
│       ├── 000001_semantic.png
│       ├── 000001_semantic_info.json
│       ├── 000002_semantic.png
│       ├── 000002_semantic_info.json
│       ├── 000003_semantic.png
│       ├── 000003_semantic_info.json
│       └── ...
│
├── scene_002/
│   ├── rgb/
│   │   └── ...
│   └── seg/
│       └── ...
│
└── scene_003/
    ├── rgb/
    │   └── ...
    └── seg/
        └── ...
```

## File Naming Requirements

### RGB Images
- Must be in `rgb/` subdirectory
- Supported formats: `.jpg`, `.jpeg`, `.png`
- Naming should have numeric component for sorting (e.g., `000001.jpg`, `frame_001.png`)

### Semantic Segmentation
- Must be in `seg/` subdirectory
- **PNG file**: Contains pixel-wise segmentation mask
  - Must contain `semantic` in filename (e.g., `000001_semantic.png`)
  - Each object has unique BGR color
- **JSON file**: Contains metadata about objects
  - Must end with `_info.json` (e.g., `000001_semantic_info.json`)
  - Contains color-to-ID mappings and class labels

## JSON Metadata Format

Example `000001_semantic_info.json`:

```json
{
  "1": {
    "color_bgr": [255, 0, 0],
    "label": {
      "class": "car"
    }
  },
  "2": {
    "color_bgr": [0, 255, 0],
    "label": {
      "class": "person"
    }
  },
  "3": {
    "color_bgr": [0, 0, 255],
    "label": {
      "class": "bicycle"
    }
  }
}
```

### JSON Field Explanations

- **Top-level keys**: Object IDs (must be integers as strings)
- **color_bgr**: BGR color tuple `[B, G, R]` used in the PNG mask
  - Must be unique for each object
  - Must match exactly with pixel values in PNG
- **label**: Object label information
  - Can contain `"class"` field with class name
  - Or any other string field will be used as label

## Semantic PNG Requirements

- **Format**: PNG image, same dimensions as RGB
- **Color space**: BGR (Blue-Green-Red)
- **Object encoding**: Each object has a unique solid color
  - Background: typically `[0, 0, 0]` or marked as "BACKGROUND" in JSON
  - Object 1: e.g., `[255, 0, 0]`
  - Object 2: e.g., `[0, 255, 0]`
  - Object 3: e.g., `[0, 0, 255]`

### Creating Semantic Masks

Example Python code to create a semantic mask:

```python
import numpy as np
import cv2

# Create blank mask (same size as RGB)
h, w = 1080, 1920
semantic_mask = np.zeros((h, w, 3), dtype=np.uint8)

# Add object 1 (car) - blue
semantic_mask[100:200, 100:300] = [255, 0, 0]  # BGR

# Add object 2 (person) - green
semantic_mask[150:250, 400:500] = [0, 255, 0]  # BGR

# Add object 3 (bicycle) - red
semantic_mask[200:300, 700:800] = [0, 0, 255]  # BGR

# Save
cv2.imwrite("000001_semantic.png", semantic_mask)
```

## Frame Matching

The system matches frames by numeric indices:
1. Extract numbers from filenames: `000001.jpg` → `1`
2. Match RGB `000001.jpg` with `000001_semantic.png` and `000001_semantic_info.json`
3. If numeric indices don't match, those frames are skipped

## Ignored Classes

By default, these class names are ignored (case-insensitive):
- `BACKGROUND`
- `UNLABELLED`

You can modify this in the code by adjusting the `ignore_names` parameter.

## Minimum Object Size

Objects smaller than `--min_area` pixels (default: 120) are filtered out.
This prevents evaluating tiny or occluded objects that may not be visible.

## Example Minimal Scene

Smallest valid scene structure:

```
simple_scene/
├── rgb/
│   └── 001.jpg          # At least one RGB frame
└── seg/
    ├── 001_semantic.png       # Matching semantic mask
    └── 001_semantic_info.json # Matching metadata
```

## Results Storage

After evaluation, results are stored in:

```
my_dataset/
├── scene_001/
│   ├── rgb/
│   ├── seg/
│   └── plottings/              # ← Created during evaluation
│       ├── model_tracker_HOTA_curve.png
│       └── ...
├── scene_002/
│   └── ...
└── results/                    # ← Created during evaluation
    ├── yolov8n_bytetrack_20260119_143022.json
    └── ...
```

## Validation Checklist

Before running evaluation, verify:

- [ ] Each scene has `rgb/` and `seg/` subdirectories
- [ ] RGB images have numeric components in filenames
- [ ] Each semantic PNG has matching `_info.json` file
- [ ] JSON contains `color_bgr` and `label` for each object
- [ ] Colors in PNG match colors in JSON exactly
- [ ] At least one valid object per frame (after min_area filtering)
- [ ] All images are readable (not corrupted)

## Testing Your Dataset

Quick test with one frame:

```bash
python yolo_metrics.py \
    --dataset ./my_dataset \
    --weights yolov8n.pt \
    --tracker bytetrack.yaml \
    --max_frames 1
```

If this succeeds, your dataset structure is correct!
