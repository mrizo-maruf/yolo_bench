#!/usr/bin/env python3
import os
import re
import json
import argparse
from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

# We use Hungarian assignment for optimal matching
try:
    from scipy.optimize import linear_sum_assignment
except Exception as e:
    raise ImportError("Please install scipy: pip install scipy") from e


# -------------------------
# IO helpers (your dataset)
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
    id_to_classname = {}
    for k, v in info.items():
        try:
            idx = int(k)
        except Exception:
            continue

        color = v.get("color_bgr", None)
        label = v.get("label", {})

        # name selection
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
    Returns:
      gt_ids: (Ng,) int
      gt_boxes_xyxy: (Ng,4) float in pixels [x1,y1,x2,y2]
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

        # clip
        x1 = max(0.0, min(W - 1.0, x1))
        y1 = max(0.0, min(H - 1.0, y1))
        x2 = max(0.0, min(W - 1.0, x2))
        y2 = max(0.0, min(H - 1.0, y2))

        ids.append(int(obj_id))
        boxes.append([x1, y1, x2, y2])

    if len(ids) == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    return np.array(ids, dtype=np.int64), np.array(boxes, dtype=np.float32)


# -------------------------
# Geometry / Matching
# -------------------------
def iou_matrix_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: (N,4), b: (M,4) in xyxy
    returns IoU (N,M)
    """
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0][None, :], b[:, 1][None, :], b[:, 2][None, :], b[:, 3][None, :]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1 + 1.0)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1 + 1.0)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1 + 1.0) * (ay2 - ay1 + 1.0)
    area_b = (bx2 - bx1 + 1.0) * (by2 - by1 + 1.0)

    union = area_a + area_b - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def match_frame(gt_boxes, pred_boxes, iou_thr: float):
    """
    Returns matches (list of (gi, pi, iou)), unmatched_gt_idx, unmatched_pred_idx
    """
    N, M = gt_boxes.shape[0], pred_boxes.shape[0]
    if N == 0 and M == 0:
        return [], list(range(N)), list(range(M))
    if N == 0:
        return [], [], list(range(M))
    if M == 0:
        return [], list(range(N)), []

    ious = iou_matrix_xyxy(gt_boxes, pred_boxes)

    # Hungarian wants a cost matrix; we maximize IoU => minimize (1 - IoU)
    cost = 1.0 - ious
    gi, pi = linear_sum_assignment(cost)

    matches = []
    matched_g = set()
    matched_p = set()
    for g, p in zip(gi.tolist(), pi.tolist()):
        if ious[g, p] >= iou_thr:
            matches.append((g, p, float(ious[g, p])))
            matched_g.add(g)
            matched_p.add(p)

    unmatched_g = [g for g in range(N) if g not in matched_g]
    unmatched_p = [p for p in range(M) if p not in matched_p]
    return matches, unmatched_g, unmatched_p


# -------------------------
# Metrics (custom implementation)
# -------------------------
class MOTAccumulator:
    def __init__(self, iou_thr: float):
        self.iou_thr = iou_thr

        # For MOTA/MOTP
        self.total_gt = 0
        self.fp = 0
        self.fn = 0
        self.idsw = 0
        self.sum_iou = 0.0
        self.tp = 0  # detection true positives (matches)

        # For IDF1 (global identity assignment)
        # counts[gt_id][pred_id] = #matched frames
        self.pair_counts = defaultdict(lambda: defaultdict(int))
        self.total_gt_dets = 0
        self.total_pred_dets = 0

        # For ID switches: keep last matched pred id per gt id
        self.last_pred_for_gt = {}  # gt_id -> pred_id

        # For HOTA components at one threshold
        # We'll compute DetA from tp/fp/fn, and AssA via pair_counts
        # AssA needs per matched detection association accuracy
        self.matched_events = []  # list of (gt_id, pred_id) for each matched detection occurrence

    def update(self, gt_ids, gt_boxes, pred_ids, pred_boxes):
        matches, un_g, un_p = match_frame(gt_boxes, pred_boxes, self.iou_thr)

        self.total_gt += len(gt_ids)
        self.total_gt_dets += len(gt_ids)
        self.total_pred_dets += len(pred_ids)

        # FP/FN
        self.fn += len(un_g)
        self.fp += len(un_p)

        # Matches
        self.tp += len(matches)
        for g_i, p_i, iou in matches:
            gt_id = int(gt_ids[g_i])
            pred_id = int(pred_ids[p_i])

            self.sum_iou += float(iou)

            # IDSW (only if GT id was previously matched to a different pred id)
            if gt_id in self.last_pred_for_gt and self.last_pred_for_gt[gt_id] != pred_id:
                self.idsw += 1
            self.last_pred_for_gt[gt_id] = pred_id

            # IDF1 counts
            self.pair_counts[gt_id][pred_id] += 1

            # HOTA matched events
            self.matched_events.append((gt_id, pred_id))

    def compute_mota_motp(self):
        mota = 1.0
        if self.total_gt > 0:
            mota = 1.0 - (self.fn + self.fp + self.idsw) / float(self.total_gt)

        motp = 0.0
        if self.tp > 0:
            motp = self.sum_iou / float(self.tp)  # average IoU over matches

        return mota, motp

    def compute_idf1(self):
        # Build GT ids and Pred ids sets
        gt_ids = sorted(self.pair_counts.keys())
        pred_ids = sorted({pid for g in self.pair_counts for pid in self.pair_counts[g].keys()})

        if len(gt_ids) == 0 or len(pred_ids) == 0:
            # No matches at all
            idtp = 0
            idfp = self.total_pred_dets
            idfn = self.total_gt_dets
            return 0.0, 0.0, 0.0, idtp, idfp, idfn

        # Build matrix of match counts
        mat = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
        for i, gid in enumerate(gt_ids):
            for j, pid in enumerate(pred_ids):
                mat[i, j] = int(self.pair_counts[gid].get(pid, 0))

        # Maximize total matches => minimize negative counts
        cost = -mat.astype(np.int64)
        gi, pj = linear_sum_assignment(cost)

        idtp = int(mat[gi, pj].sum())
        idfp = int(self.total_pred_dets - idtp)
        idfn = int(self.total_gt_dets - idtp)

        idp = idtp / float(idtp + idfp) if (idtp + idfp) > 0 else 0.0
        idr = idtp / float(idtp + idfn) if (idtp + idfn) > 0 else 0.0
        idf1 = (2 * idp * idr / (idp + idr)) if (idp + idr) > 0 else 0.0
        return idf1, idp, idr, idtp, idfp, idfn

    def compute_hota(self):
        """
        Single-threshold HOTA (at self.iou_thr):
          DetA = TP / (TP + FP + FN)
          AssA = average over matched detections of AssAcc(gt_id, pred_id)
          where AssAcc(g,p) = TPA / (TPA + FPA + FNA)
          with:
            TPA = pair_counts[g][p]
            FPA = sum_{g'!=g} pair_counts[g'][p]
            FNA = sum_{p'!=p} pair_counts[g][p']
          HOTA = sqrt(DetA * AssA)
        """
        denom = (self.tp + self.fp + self.fn)
        det_a = (self.tp / float(denom)) if denom > 0 else 0.0

        if len(self.matched_events) == 0:
            return 0.0, det_a, 0.0

        # Precompute totals per pred and per gt
        total_for_pred = defaultdict(int)
        total_for_gt = defaultdict(int)
        for g in self.pair_counts:
            for p, c in self.pair_counts[g].items():
                total_for_gt[g] += c
                total_for_pred[p] += c

        ass_sum = 0.0
        for (g, p) in self.matched_events:
            tpa = self.pair_counts[g].get(p, 0)
            fpa = total_for_pred[p] - tpa
            fna = total_for_gt[g] - tpa
            denom_a = tpa + fpa + fna
            ass_acc = (tpa / float(denom_a)) if denom_a > 0 else 0.0
            ass_sum += ass_acc

        ass_a = ass_sum / float(len(self.matched_events))
        hota = float(np.sqrt(det_a * ass_a))
        return hota, det_a, ass_a


# -------------------------
# YOLO extraction
# -------------------------
def get_yolo_tracks(result, frame_shape, conf_thr: float):
    """
    Returns:
      pred_ids (Np,) int
      pred_boxes (Np,4) float xyxy
    Only keeps detections with track IDs (required for tracking metrics).
    """
    H, W = frame_shape[:2]

    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    if not getattr(result.boxes, "is_track", False) or result.boxes.id is None:
        # tracking IDs missing => cannot compute ID metrics
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    conf = result.boxes.conf.cpu().numpy().astype(np.float32) if result.boxes.conf is not None else np.ones((len(xyxy),), dtype=np.float32)
    tids = result.boxes.id.int().cpu().numpy().astype(np.int64)

    keep = conf >= conf_thr
    xyxy = xyxy[keep]
    tids = tids[keep]

    # clip
    if xyxy.shape[0] > 0:
        xyxy[:, 0] = np.clip(xyxy[:, 0], 0, W - 1)
        xyxy[:, 1] = np.clip(xyxy[:, 1], 0, H - 1)
        xyxy[:, 2] = np.clip(xyxy[:, 2], 0, W - 1)
        xyxy[:, 3] = np.clip(xyxy[:, 3], 0, H - 1)

    return tids, xyxy
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

def iou_matrix_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0][None, :], b[:, 1][None, :], b[:, 2][None, :], b[:, 3][None, :]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1 + 1.0)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1 + 1.0)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1 + 1.0) * (ay2 - ay1 + 1.0)
    area_b = (bx2 - bx1 + 1.0) * (by2 - by1 + 1.0)
    union = area_a + area_b - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def hungarian_matches(ious: np.ndarray, thr: float):
    """
    ious: (Ng, Np)
    Returns matches list: (g, p, iou)
    plus unmatched indices for gt and pred.
    """
    Ng, Np = ious.shape
    if Ng == 0 and Np == 0:
        return [], [], []
    if Ng == 0:
        return [], [], list(range(Np))
    if Np == 0:
        return [], list(range(Ng)), []

    cost = 1.0 - ious
    gi, pi = linear_sum_assignment(cost)

    matches = []
    mg, mp = set(), set()
    for g, p in zip(gi.tolist(), pi.tolist()):
        if ious[g, p] >= thr:
            matches.append((g, p, float(ious[g, p])))
            mg.add(g); mp.add(p)

    un_g = [g for g in range(Ng) if g not in mg]
    un_p = [p for p in range(Np) if p not in mp]
    return matches, un_g, un_p


class HOTAAccumulator:
    """
    Collects events for multiple alpha thresholds at once.
    """
    def __init__(self, alphas):
        self.alphas = list(alphas)

        # Per alpha: TP, FP, FN, sum Loc-IoU over matched pairs
        self.tp = {a: 0 for a in self.alphas}
        self.fp = {a: 0 for a in self.alphas}
        self.fn = {a: 0 for a in self.alphas}
        self.sum_lociou = {a: 0.0 for a in self.alphas}

        # Per alpha: pair counts for association IoU
        # pair_counts[a][gt_id][pred_id] = #matches at this alpha
        self.pair_counts = {a: defaultdict(lambda: defaultdict(int)) for a in self.alphas}

        # Per alpha: list of matched (gt_id, pred_id) occurrences (for averaging AssIoU)
        self.matched_events = {a: [] for a in self.alphas}

    def update(self, gt_ids, gt_boxes, pred_ids, pred_boxes):
        # Compute IoU matrix once
        ious = iou_matrix_xyxy(gt_boxes, pred_boxes)

        for a in self.alphas:
            matches, un_g, un_p = hungarian_matches(ious, thr=a)
            self.tp[a] += len(matches)
            self.fn[a] += len(un_g)
            self.fp[a] += len(un_p)

            for g_i, p_i, iou in matches:
                gid = int(gt_ids[g_i])
                pid = int(pred_ids[p_i])
                self.sum_lociou[a] += float(iou)
                self.pair_counts[a][gid][pid] += 1
                self.matched_events[a].append((gid, pid))

    def compute(self):
        """
        Returns dict with:
          HOTA, DetA, AssA, LocA averaged over alphas
          plus per-alpha curves
        """
        hota_curve = []
        deta_curve = []
        assa_curve = []
        loca_curve = []

        for a in self.alphas:
            tp = self.tp[a]
            fp = self.fp[a]
            fn = self.fn[a]

            denom_det = tp + fp + fn
            det_a = tp / denom_det if denom_det > 0 else 0.0

            loc_a = (self.sum_lociou[a] / tp) if tp > 0 else 0.0

            # Association accuracy (AssA): average over matched detections of AssIoU
            if len(self.matched_events[a]) == 0:
                ass_a = 0.0
            else:
                total_for_pred = defaultdict(int)
                total_for_gt = defaultdict(int)
                pc = self.pair_counts[a]

                for g in pc:
                    for p, c in pc[g].items():
                        total_for_gt[g] += c
                        total_for_pred[p] += c

                ass_sum = 0.0
                for (g, p) in self.matched_events[a]:
                    tpa = pc[g].get(p, 0)
                    fpa = total_for_pred[p] - tpa
                    fna = total_for_gt[g] - tpa
                    denom_ass = tpa + fpa + fna
                    ass_iou = (tpa / denom_ass) if denom_ass > 0 else 0.0
                    ass_sum += ass_iou

                ass_a = ass_sum / float(len(self.matched_events[a]))

            hota_a = float(np.sqrt(det_a * ass_a))

            hota_curve.append(hota_a)
            deta_curve.append(det_a)
            assa_curve.append(ass_a)
            loca_curve.append(loc_a)

        # Average over alphas (integration approximation)
        HOTA = float(np.mean(hota_curve)) if len(hota_curve) else 0.0
        DetA = float(np.mean(deta_curve)) if len(deta_curve) else 0.0
        AssA = float(np.mean(assa_curve)) if len(assa_curve) else 0.0
        LocA = float(np.mean(loca_curve)) if len(loca_curve) else 0.0

        return {
            "HOTA": HOTA,
            "DetA": DetA,
            "AssA": AssA,
            "LocA": LocA,
            "alphas": self.alphas,
            "HOTA_curve": hota_curve,
            "DetA_curve": deta_curve,
            "AssA_curve": assa_curve,
            "LocA_curve": loca_curve,
        }


import os
import matplotlib.pyplot as plt


def plot_hota_curves(hota_results: dict, out_dir: str, prefix: str = "tracker"):
    """
    hota_results: output dict from HOTAAccumulator.compute()
    out_dir: directory to save plots
    prefix: filename prefix (e.g. tracker name)
    """

    os.makedirs(out_dir, exist_ok=True)

    alphas = hota_results["alphas"]
    HOTA_c = hota_results["HOTA_curve"]
    DetA_c = hota_results["DetA_curve"]
    AssA_c = hota_results["AssA_curve"]
    LocA_c = hota_results["LocA_curve"]

    # -------------------------
    # 1) HOTA vs alpha
    # -------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(alphas, HOTA_c, marker="o")
    plt.xlabel("IoU threshold α")
    plt.ylabel("HOTA")
    plt.title("HOTA(α)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_HOTA_curve.png"), dpi=200)
    plt.close()

    # -------------------------
    # 2) Detection & Association
    # -------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(alphas, DetA_c, marker="o", label="DetA")
    plt.plot(alphas, AssA_c, marker="o", label="AssA")
    plt.xlabel("IoU threshold α")
    plt.ylabel("Score")
    plt.title("Detection & Association Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_DetA_AssA_curves.png"), dpi=200)
    plt.close()

    # -------------------------
    # 3) Localization accuracy
    # -------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(alphas, LocA_c, marker="o")
    plt.xlabel("IoU threshold α")
    plt.ylabel("LocA (mean IoU)")
    plt.title("Localization Accuracy (LocA)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_LocA_curve.png"), dpi=200)
    plt.close()

    # -------------------------
    # 4) All curves together (overview)
    # -------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(alphas, HOTA_c, marker="o", label="HOTA")
    plt.plot(alphas, DetA_c, marker="o", label="DetA")
    plt.plot(alphas, AssA_c, marker="o", label="AssA")
    plt.plot(alphas, LocA_c, marker="o", label="LocA")
    plt.xlabel("IoU threshold α")
    plt.ylabel("Score")
    plt.title("HOTA Metrics Overview")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_ALL_curves.png"), dpi=200)
    plt.close()

    print(f"[OK] Plots saved to: {out_dir}")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="Path to scene_name/ (contains rgb/, seg/)")
    ap.add_argument("--weights", required=True, help="YOLO weights (yoloe/yolov11/...)")
    ap.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml / botsort.yaml / custom")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5, help="YOLO NMS IoU (not matching IoU)")
    ap.add_argument("--match_iou", type=float, default=0.5, help="IoU threshold for GT-Pred matching")
    ap.add_argument("--min_area", type=int, default=120, help="Min GT mask pixels to form a bbox")
    ap.add_argument("--max_frames", type=int, default=-1, help="Limit frames for quick test (-1 = all)")
    args = ap.parse_args()

    pairs = list_frames(args.scene)
    if args.max_frames > 0:
        pairs = pairs[:args.max_frames]

    model = YOLO(args.weights)

    acc = MOTAccumulator(iou_thr=args.match_iou)

    alphas = np.round(np.arange(0.05, 0.96, 0.05), 2)  # 0.05..0.95
    hota_acc = HOTAAccumulator(alphas)

    for idx, (rgb_path, sem_path, info_path) in enumerate(pairs):
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        sem = cv2.imread(sem_path, cv2.IMREAD_COLOR)
        if rgb is None or sem is None:
            print(f"Skipping unreadable frame: {rgb_path}")
            continue

        # GT
        id_to_color_bgr, id_to_classname = load_semantic_info(info_path)
        gt_ids, gt_boxes = compute_gt_from_semantic(
            sem, id_to_color_bgr, id_to_classname, min_area=args.min_area
        )

        # YOLO tracking
        res = model.track(
            rgb,
            persist=True,
            tracker=args.tracker,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
            agnostic_nms=True
        )[0]

        pred_ids, pred_boxes = get_yolo_tracks(res, rgb.shape, conf_thr=args.conf)

        # Update metrics
        acc.update(gt_ids, gt_boxes, pred_ids, pred_boxes)
        
        hota_acc.update(gt_ids, gt_boxes, pred_ids, pred_boxes)


        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1}/{len(pairs)} frames...")

    mota, motp = acc.compute_mota_motp()
    idf1, idp, idr, idtp, idfp, idfn = acc.compute_idf1()
    hota, det_a, ass_a = acc.compute_hota()

    res = hota_acc.compute()
    print(res["HOTA"], res["DetA"], res["AssA"], res["LocA"])

    # Plot
    plot_dir = os.path.join(args.scene, "plottings")
    plot_hota_curves(
        hota_results=res,
        out_dir=plot_dir,
        prefix=f"{os.path.basename(args.weights)}_{args.tracker}"
    )

    print("\n==================== RESULTS ====================")
    print(f"Frames evaluated: {len(pairs)}")
    print(f"Match IoU thr:    {args.match_iou:.2f}")
    print("\n--- Detection+Tracking (CLEAR) ---")
    print(f"TP:   {acc.tp}")
    print(f"FP:   {acc.fp}")
    print(f"FN:   {acc.fn}")
    print(f"IDSW: {acc.idsw}")
    print(f"MOTA: {mota:.6f}")
    print(f"MOTP (avg IoU on matches): {motp:.6f}")

    print("\n--- Identity (IDF1) ---")
    print(f"IDTP: {idtp}  IDFP: {idfp}  IDFN: {idfn}")
    print(f"IDP:  {idp:.6f}")
    print(f"IDR:  {idr:.6f}")
    print(f"IDF1: {idf1:.6f}")

    print("\n--- HOTA (single-threshold) ---")
    print(f"DetA: {det_a:.6f}")
    print(f"AssA: {ass_a:.6f}")
    print(f"HOTA: {hota:.6f}")
    print("=================================================\n")


if __name__ == "__main__":
    main()
