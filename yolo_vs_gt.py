#!/usr/bin/env python3
"""
Side-by-side GT vs YOLO tracking visualization for Isaac-Sim dataset.

LEFT  panel: GT (IDs from semantic_info.json key, class name, bbox, mask overlay, optional trails)
RIGHT panel: YOLO (track IDs, predicted class, bbox, mask overlay if available, optional trails)

Keys:
  d  -> next frame (manual stepping)
  q  -> quit
Options:
  SAVE_MODE=True -> save stepped frames into MP4 on exit
"""

import os
import re
import json
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO


# -------------------------
# Helpers (dataset)
# -------------------------
def natural_key(path: str) -> int:
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def list_frames(scene_dir: str):
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
    with open(info_json_path, "r") as f:
        info = json.load(f)

    id_to_color_bgr = {}
    id_to_name = {}

    for k, v in info.items():
        try:
            idx = int(k)
        except Exception:
            continue

        color = v.get("color_bgr", None)
        label = v.get("label", {})

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
            id_to_name[idx] = name

    return id_to_color_bgr, id_to_name


def compute_gt_objects_and_overlay(
    sem_bgr: np.ndarray,
    id_to_color_bgr: dict,
    id_to_name: dict,
    min_area: int = 120,
    ignore_names=("BACKGROUND", "UNLABELLED"),
):
    """
    Returns:
      objects: list of {id,name,color,bbox,center,mask}
      overlay_bgr: semantic overlay (same shape as rgb)
    """
    objects = []
    overlay = sem_bgr.copy()  # already in BGR colors

    for obj_id, color in id_to_color_bgr.items():
        name = id_to_name.get(obj_id, f"ID_{obj_id}")
        if any(name.upper() == ig.upper() for ig in ignore_names):
            continue

        mask = (
            (sem_bgr[:, :, 0] == color[0]) &
            (sem_bgr[:, :, 1] == color[1]) &
            (sem_bgr[:, :, 2] == color[2])
        ).astype(np.uint8)

        area = int(mask.sum())
        if area < min_area:
            continue

        ys, xs = np.where(mask > 0)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        cx, cy = int(xs.mean()), int(ys.mean())

        objects.append({
            "id": obj_id,
            "name": name,
            "color": color,
            "bbox": (x1, y1, x2, y2),
            "center": (cx, cy),
            "mask": mask
        })

    return objects, overlay


# -------------------------
# Drawing helpers
# -------------------------
def alpha_blend(base_bgr: np.ndarray, overlay_bgr: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.addWeighted(base_bgr, 1.0 - alpha, overlay_bgr, alpha, 0)


def put_header(panel: np.ndarray, title: str):
    h = panel.shape[0]
    bar_h = max(36, int(0.06 * h))
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (0, 0, 0), -1)
    cv2.putText(out, title, (12, int(bar_h * 0.72)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_trails(img: np.ndarray, trails: dict, color=(255, 255, 255), thickness=2):
    out = img
    for _, pts in trails.items():
        if len(pts) < 2:
            continue
        for i in range(1, len(pts)):
            cv2.line(out, pts[i - 1], pts[i], color, thickness)
    return out


def draw_boxes_labels(img: np.ndarray, items, show_id=True):
    """
    items: list of dicts with:
      - bbox (x1,y1,x2,y2)
      - color (b,g,r)
      - name (class)
      - id (optional)
    """
    out = img
    for it in items:
        x1, y1, x2, y2 = it["bbox"]
        color = it.get("color", (0, 255, 0))
        name = str(it.get("name", "obj"))
        tid = it.get("id", None)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = name
        if show_id and tid is not None:
            label = f"ID {tid}: {name}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 0, 0), -1)
        cv2.putText(out, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def masks_to_overlay(masks, colors, shape_hw):
    """
    Build colored overlay for a set of masks.
    masks: list of HxW float/bool/0-1
    colors: list of (b,g,r)
    """
    H, W = shape_hw
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    for m, c in zip(masks, colors):
        if m is None:
            continue
        mm = (m > 0.5).astype(np.uint8)
        overlay[mm == 1] = c
    return overlay


# -------------------------
# YOLO extraction helpers
# -------------------------
def get_yolo_items(result, frame_shape, conf_thr=0.25):
    """
    Returns:
      items: list dict {id,name,color,bbox,center}
      overlay: BGR overlay from predicted masks if available, else None
    """
    H, W = frame_shape[:2]
    items = []
    overlay = None

    if result.boxes is None or len(result.boxes) == 0:
        return items, overlay

    names = result.names if hasattr(result, "names") else None

    # Boxes in xyxy
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones((len(xyxy),), dtype=np.float32)
    cls = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else np.zeros((len(xyxy),), dtype=int)

    # Track IDs (if tracking is enabled)
    track_ids = None
    if getattr(result.boxes, "is_track", False) and result.boxes.id is not None:
        track_ids = result.boxes.id.int().cpu().numpy().tolist()

    # Masks (if model returns them)
    pred_masks = None
    if hasattr(result, "masks") and result.masks is not None and result.masks.data is not None:
        # Ultralytics masks are (N,H,W) in model output resolution; .data is torch
        pred_masks = result.masks.data.float().cpu().numpy()  # N x Hm x Wm
        # They might already match frame size; if not, resize per mask
        # We'll handle below.

    chosen_masks = []
    chosen_colors = []

    for i in range(len(xyxy)):
        if conf[i] < conf_thr:
            continue

        x1, y1, x2, y2 = xyxy[i]
        x1 = int(max(0, min(W - 1, x1)))
        y1 = int(max(0, min(H - 1, y1)))
        x2 = int(max(0, min(W - 1, x2)))
        y2 = int(max(0, min(H - 1, y2)))

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        class_id = int(cls[i])
        class_name = str(class_id) if names is None else str(names.get(class_id, class_id))

        tid = None
        if track_ids is not None and i < len(track_ids):
            tid = int(track_ids[i])

        # deterministic color by track id or class id
        seed = tid if tid is not None else (1000 + class_id)
        color = (int((seed * 37) % 255), int((seed * 17) % 255), int((seed * 29) % 255))

        items.append({
            "id": tid,
            "name": class_name,
            "color": color,
            "bbox": (x1, y1, x2, y2),
            "center": (cx, cy),
        })

        if pred_masks is not None:
            m = pred_masks[i]
            if m.shape[0] != H or m.shape[1] != W:
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            chosen_masks.append(m)
            chosen_colors.append(color)

    if pred_masks is not None and len(chosen_masks) > 0:
        overlay = masks_to_overlay(chosen_masks, chosen_colors, (H, W))

    return items, overlay


# -------------------------
# Main
# -------------------------
def main():
    # -------------------------
    # USER CONFIG
    # -------------------------
    SCENE_DIR = "/home/maribjonov_mr/IsaacSim_bench/cabinet_simple"
    YOLO_WEIGHTS = "yoloe-11l-seg-pf.pt"  # change to yoloe / yolov11 weights

    FPS = 10
    MIN_GT_AREA = 120
    GT_TRAIL_LEN = 60
    YOLO_TRAIL_LEN = 60

    OVERLAY_ALPHA = 0.40

    # YOLO.track params (edit freely)
    TRACK_CONF = 0.25
    TRACK_IOU = 0.5
    TRACK_PERSIST = True
    TRACKER_CFG = "bytetrack.yaml"  # or "botsort.yaml" or custom yaml

    # Save stepped frames to mp4?
    SAVE_MODE = True
    OUT_MP4 = os.path.join(SCENE_DIR, "gt_vs_yolo_selected.mp4")

    # -------------------------
    pairs = list_frames(SCENE_DIR)
    model = YOLO(YOLO_WEIGHTS)

    gt_trails = defaultdict(lambda: deque(maxlen=GT_TRAIL_LEN))
    yolo_trails = defaultdict(lambda: deque(maxlen=YOLO_TRAIL_LEN))

    saved_frames = []

    win = "GT vs YOLO (LEFT=GT | RIGHT=YOLO)   [d=next  q=quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    delay = max(1, int(1000.0 / FPS))

    idx = 0
    while idx < len(pairs):
        rgb_path, sem_path, info_path = pairs[idx]

        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        sem = cv2.imread(sem_path, cv2.IMREAD_COLOR)
        if rgb is None or sem is None:
            print(f"Skipping unreadable frame: {rgb_path}")
            idx += 1
            continue

        H, W = rgb.shape[:2]

        # -------------------------
        # GT side
        # -------------------------
        id_to_color_bgr, id_to_name = load_semantic_info(info_path)
        gt_objects, gt_overlay = compute_gt_objects_and_overlay(
            sem, id_to_color_bgr, id_to_name,
            min_area=MIN_GT_AREA
        )

        for obj in gt_objects:
            gt_trails[obj["id"]].append(obj["center"])

        gt_panel = alpha_blend(rgb, gt_overlay, OVERLAY_ALPHA)
        gt_panel = draw_trails(gt_panel, gt_trails, color=(255, 255, 255), thickness=2)
        gt_panel = draw_boxes_labels(gt_panel, gt_objects, show_id=True)
        gt_panel = put_header(gt_panel, "GT: bbox + class + ID + mask")

        # -------------------------
        # YOLO side
        # -------------------------
        # Important: model.track returns list of Results, we take [0]
        res = model.track(
            rgb,
            persist=TRACK_PERSIST,
            conf=TRACK_CONF,
            iou=TRACK_IOU,
            tracker=TRACKER_CFG,
            verbose=False
        )[0]

        yolo_items, yolo_mask_overlay = get_yolo_items(res, rgb.shape, conf_thr=TRACK_CONF)

        # Update YOLO trails only for items that have an ID
        for it in yolo_items:
            if it["id"] is None:
                continue
            yolo_trails[it["id"]].append(it["center"])

        yolo_panel = rgb.copy()
        if yolo_mask_overlay is not None:
            yolo_panel = alpha_blend(yolo_panel, yolo_mask_overlay, OVERLAY_ALPHA)
        yolo_panel = draw_trails(yolo_panel, yolo_trails, color=(255, 255, 255), thickness=2)
        yolo_panel = draw_boxes_labels(yolo_panel, yolo_items, show_id=True)
        yolo_panel = put_header(yolo_panel, f"YOLO: {os.path.basename(YOLO_WEIGHTS)} + {TRACKER_CFG}")

        # Frame text
        frame_id = natural_key(rgb_path)
        cv2.putText(gt_panel, f"Frame {frame_id}", (12, gt_panel.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(yolo_panel, f"Frame {frame_id}", (12, yolo_panel.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Combine (one window, two parts)
        if yolo_panel.shape != gt_panel.shape:
            yolo_panel = cv2.resize(yolo_panel, (gt_panel.shape[1], gt_panel.shape[0]), interpolation=cv2.INTER_NEAREST)
        combined = np.concatenate([gt_panel, yolo_panel], axis=1)

        cv2.putText(combined, "Keys: d=next (save if SAVE_MODE)   q=quit",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(win, combined)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        if key == ord('d'):
            if SAVE_MODE:
                saved_frames.append(combined.copy())
            idx += 1
        else:
            # stay on current frame until 'd' or 'q'
            continue

    cv2.destroyAllWindows()

    if SAVE_MODE and len(saved_frames) > 0:
        h, w = saved_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUT_MP4, fourcc, float(FPS), (w, h))
        for fr in saved_frames:
            if fr.shape[0] != h or fr.shape[1] != w:
                fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_NEAREST)
            writer.write(fr)
        writer.release()
        print(f"[OK] Saved {len(saved_frames)} frames to: {OUT_MP4}")
    else:
        print("[OK] No video saved (SAVE_MODE=False or no frames collected).")


if __name__ == "__main__":
    main()
