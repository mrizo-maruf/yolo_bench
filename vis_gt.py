#!/usr/bin/env python3
import os
import re
import json
from collections import defaultdict, deque

import cv2
import numpy as np


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
    id_to_classname = {}
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
            id_to_classname[idx] = name

    return id_to_color_bgr, id_to_classname


def compute_objects_from_semantic(
    sem_bgr: np.ndarray,
    id_to_color_bgr: dict,
    id_to_classname: dict,
    min_area: int = 80,
    ignore_names=("BACKGROUND", "UNLABELLED"),
):
    objects = []
    for obj_id, color in id_to_color_bgr.items():
        name = id_to_classname.get(obj_id, f"ID_{obj_id}")
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
            "area": area
        })

    return objects


def draw_objects(
    img_bgr: np.ndarray,
    objects,
    show_id=False,
    show_trails=False,
    trails=None,
    overlay_seg=None,
    alpha=0.35
):
    out = img_bgr.copy()

    if overlay_seg is not None:
        out = cv2.addWeighted(out, 1.0 - alpha, overlay_seg, alpha, 0)

    if show_trails and trails is not None:
        for _, pts in trails.items():
            if len(pts) < 2:
                continue
            for i in range(1, len(pts)):
                cv2.line(out, pts[i - 1], pts[i], (255, 255, 255), 2)

    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        color = obj["color"]
        name = obj["name"]
        obj_id = obj["id"]

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = name if not show_id else f"ID {obj_id}: {name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 0, 0), -1)
        cv2.putText(out, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.circle(out, obj["center"], 3, (255, 255, 255), -1)

    return out


def put_header(panel: np.ndarray, title: str):
    h = panel.shape[0]
    bar_h = max(36, int(0.06 * h))
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (0, 0, 0), -1)
    cv2.putText(out, title, (12, int(bar_h * 0.72)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    # -------------------------
    # USER CONFIG
    # -------------------------
    scene = "/home/maribjonov_mr/IsaacSim_bench/cabinet_simple"
    fps = 10
    min_area = 120
    trail_len = 60
    overlay_seg = True
    alpha = 0.4

    # If True: save selected frames (pressed with 'd') to MP4 on quit ('q')
    save_mode = Trued
    out_mp4 = os.path.join("/home/maribjonov_mr/yolo_bench/", "gt_det_track_selected.mp4")

    # -------------------------
    pairs = list_frames(scene)
    trails = defaultdict(lambda: deque(maxlen=trail_len))

    win = "GT Visualizer (LEFT: Detection | RIGHT: Tracking)  [d=next/save  q=quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    delay = max(1, int(1000.0 / fps))

    saved_frames = []  # store combined frames only when save_mode=True

    idx = 0
    while idx < len(pairs):
        rgb_path, sem_path, info_path = pairs[idx]

        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        sem = cv2.imread(sem_path, cv2.IMREAD_COLOR)  # BGR

        if rgb is None or sem is None:
            print(f"Skipping unreadable frame: {rgb_path}")
            idx += 1
            continue

        id_to_color_bgr, id_to_classname = load_semantic_info(info_path)
        objects = compute_objects_from_semantic(
            sem, id_to_color_bgr, id_to_classname,
            min_area=min_area
        )

        # Update trails (persistent across frames)
        for obj in objects:
            trails[obj["id"]].append(obj["center"])

        overlay = sem if overlay_seg else None

        det_img = draw_objects(
            rgb, objects,
            show_id=False,
            show_trails=False,
            trails=None,
            overlay_seg=overlay,
            alpha=alpha
        )
        trk_img = draw_objects(
            rgb, objects,
            show_id=True,
            show_trails=True,
            trails=trails,
            overlay_seg=overlay,
            alpha=alpha
        )

        det_img = put_header(det_img, "DETECTION (bbox + class)")
        trk_img = put_header(trk_img, "TRACKING (bbox + ID + trails)")

        frame_id = natural_key(rgb_path)
        cv2.putText(det_img, f"Frame {frame_id}", (12, det_img.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(trk_img, f"Frame {frame_id}", (12, trk_img.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Combine into one window (2 panels)
        if det_img.shape != trk_img.shape:
            trk_img = cv2.resize(trk_img, (det_img.shape[1], det_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        combined = np.concatenate([det_img, trk_img], axis=1)

        # UI hint text
        cv2.putText(combined, "Keys: d=next (and save frame if save_mode)   q=quit",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(win, combined)

        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q'):
            break

        # Only advance when user presses 'd'
        if key == ord('d'):
            if save_mode:
                saved_frames.append(combined.copy())
            idx += 1
        else:
            # keep showing the same frame until 'd' or 'q'
            continue

    cv2.destroyAllWindows()

    # Save MP4 if requested
    if save_mode and len(saved_frames) > 0:
        h, w = saved_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_mp4, fourcc, float(fps), (w, h))
        for fr in saved_frames:
            if fr.shape[0] != h or fr.shape[1] != w:
                fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_NEAREST)
            writer.write(fr)
        writer.release()
        print(f"[OK] Saved {len(saved_frames)} frames to: {out_mp4}")
    else:
        print("[OK] No video saved (save_mode=False or no frames collected).")


if __name__ == "__main__":
    main()
