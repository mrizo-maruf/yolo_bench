"""
Dataset I/O utilities for loading frames and semantic information.
"""
import os
import re
import json
import cv2
import numpy as np


def natural_key(path: str) -> int:
    """Extract numeric value from filename for natural sorting."""
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def list_frames(scene_dir: str):
    """
    List all RGB and semantic segmentation frame pairs in a scene directory.
    
    Args:
        scene_dir: Path to scene directory containing rgb/ and seg/ subdirectories
        
    Returns:
        List of tuples (rgb_path, seg_png_path, seg_json_path)
    """
    rgb_dir = os.path.join(scene_dir, "rgb")
    seg_dir = os.path.join(scene_dir, "seg")

    rgb_paths = sorted(
        [os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir)
         if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=natural_key
    )

    seg_pngs = sorted(
        [os.path.join(seg_dir, f) for f in os.listdir(seg_dir)
         if f.lower().endswith(".png") and "semantic" in f.lower() and "info" not in f.lower()],
        key=natural_key
    )
    seg_jsons = sorted(
        [os.path.join(seg_dir, f) for f in os.listdir(seg_dir)
         if f.lower().endswith("_info.json") and "semantic" in f.lower()],
        key=natural_key
    )

    if not rgb_paths:
        raise FileNotFoundError(f"No RGB images found in: {rgb_dir}")
    if not seg_pngs or not seg_jsons:
        raise FileNotFoundError(f"No semantic png/json found in: {seg_dir}")

    seg_png_map = {natural_key(p): p for p in seg_pngs}
    seg_js_map = {natural_key(p): p for p in seg_jsons}

    pairs = []
    for rgb in rgb_paths:
        k = natural_key(rgb)
        if k in seg_png_map and k in seg_js_map:
            pairs.append((rgb, seg_png_map[k], seg_js_map[k]))

    if not pairs:
        raise RuntimeError("No matching RGB/SEG frame triplets found. Check naming consistency.")
    return pairs


def load_semantic_info(info_json_path: str):
    """
    Load semantic segmentation metadata from JSON file.
    
    Args:
        info_json_path: Path to semantic info JSON file
        
    Returns:
        Tuple of (id_to_color_bgr dict, id_to_classname dict)
    """
    with open(info_json_path, "r") as f:
        info = json.load(f)

    id_to_color_bgr = {}
    id_to_classname = {}
    
    for k, v in info.items():
        try:
            idx = int(k)
        except Exception:
            continue

        color = v.get("color_bgr", None)
        label = v.get("label", {})

        # Extract name from label
        name = None
        if isinstance(label, dict):
            if "class" in label:
                name = str(label["class"])
            else:
                for _, vv in label.items():
                    if vv is None:
                        continue
                    s = str(vv).strip()
                    if s:
                        name = s
                        break
        if name is None:
            name = f"ID_{idx}"

        if color and len(color) == 3:
            id_to_color_bgr[idx] = tuple(int(c) for c in color)
            id_to_classname[idx] = name

    return id_to_color_bgr, id_to_classname


def compute_gt_from_semantic(
    sem_bgr: np.ndarray,
    id_to_color_bgr: dict,
    id_to_classname: dict,
    min_area: int = 120,
    ignore_names=("BACKGROUND", "UNLABELLED"),
):
    """
    Extract ground truth bounding boxes from semantic segmentation masks.
    
    Args:
        sem_bgr: Semantic segmentation image (H, W, 3) in BGR
        id_to_color_bgr: Mapping from object ID to BGR color
        id_to_classname: Mapping from object ID to class name
        min_area: Minimum mask area in pixels to consider
        ignore_names: Class names to ignore
        
    Returns:
        Tuple of (gt_ids array, gt_boxes_xyxy array)
    """
    H, W = sem_bgr.shape[:2]
    ids = []
    boxes = []

    for obj_id, color in id_to_color_bgr.items():
        name = id_to_classname.get(obj_id, f"ID_{obj_id}")
        if any(name.upper() == ig.upper() for ig in ignore_names):
            continue

        mask = (
            (sem_bgr[:, :, 0] == color[0]) &
            (sem_bgr[:, :, 1] == color[1]) &
            (sem_bgr[:, :, 2] == color[2])
        )

        area = int(mask.sum())
        if area < min_area:
            continue

        ys, xs = np.where(mask)
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())

        # Clip to image bounds
        x1 = max(0.0, min(W - 1.0, x1))
        y1 = max(0.0, min(H - 1.0, y1))
        x2 = max(0.0, min(W - 1.0, x2))
        y2 = max(0.0, min(H - 1.0, y2))

        ids.append(int(obj_id))
        boxes.append([x1, y1, x2, y2])

    if len(ids) == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    return np.array(ids, dtype=np.int64), np.array(boxes, dtype=np.float32)
