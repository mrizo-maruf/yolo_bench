#!/usr/bin/env python3
"""
YOLO Tracking Metrics Evaluation Script

Evaluates YOLO tracking performance using multiple metrics:
- MOTA/MOTP (CLEAR MOT metrics)
- IDF1 (Identity F1 Score)
- HOTA (Higher Order Tracking Accuracy)

Usage:
    python yolo_metrics.py --scene <scene_dir> --weights <model.pt> --tracker <tracker.yaml>
"""
import os
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

# Import modularized components
from io_utils import list_frames, load_semantic_info, compute_gt_from_semantic
from metrics import MOTAccumulator, HOTAAccumulator
from yolo_utils import get_yolo_tracks
from visualization import plot_hota_curves


def parse_args():
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(description="Evaluate YOLO tracking metrics")
    ap.add_argument("--scene", required=True, 
                    help="Path to scene directory (contains rgb/, seg/)")
    ap.add_argument("--weights", required=True, 
                    help="YOLO model weights (e.g., yolov8n.pt)")
    ap.add_argument("--tracker", default="bytetrack.yaml", 
                    help="Tracker config (bytetrack.yaml / botsort.yaml)")
    ap.add_argument("--conf", type=float, default=0.25, 
                    help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, 
                    help="YOLO NMS IoU threshold")
    ap.add_argument("--match_iou", type=float, default=0.5, 
                    help="IoU threshold for GT-Pred matching")
    ap.add_argument("--min_area", type=int, default=120, 
                    help="Minimum GT mask area in pixels")
    ap.add_argument("--max_frames", type=int, default=-1, 
                    help="Limit number of frames (-1 = all)")
    return ap.parse_args()


def print_results(acc, hota_results, num_frames, match_iou):
    """Print evaluation results in a formatted manner."""
    mota, motp = acc.compute_mota_motp()
    idf1, idp, idr, idtp, idfp, idfn = acc.compute_idf1()
    hota, det_a, ass_a = acc.compute_hota()

    print("\n" + "="*50)
    print("TRACKING EVALUATION RESULTS")
    print("="*50)
    print(f"Frames evaluated: {num_frames}")
    print(f"Match IoU threshold: {match_iou:.2f}")
    
    print("\n--- Detection+Tracking (CLEAR MOT) ---")
    print(f"TP:   {acc.tp}")
    print(f"FP:   {acc.fp}")
    print(f"FN:   {acc.fn}")
    print(f"IDSW: {acc.idsw}")
    print(f"MOTA: {mota:.6f}")
    print(f"MOTP: {motp:.6f} (avg IoU on matches)")

    print("\n--- Identity Metrics (IDF1) ---")
    print(f"IDTP: {idtp}  IDFP: {idfp}  IDFN: {idfn}")
    print(f"IDP:  {idp:.6f}")
    print(f"IDR:  {idr:.6f}")
    print(f"IDF1: {idf1:.6f}")

    print("\n--- HOTA (single-threshold) ---")
    print(f"DetA: {det_a:.6f}")
    print(f"AssA: {ass_a:.6f}")
    print(f"HOTA: {hota:.6f}")

    print("\n--- HOTA (multi-threshold) ---")
    print(f"HOTA: {hota_results['HOTA']:.6f}")
    print(f"DetA: {hota_results['DetA']:.6f}")
    print(f"AssA: {hota_results['AssA']:.6f}")
    print(f"LocA: {hota_results['LocA']:.6f}")
    print("="*50 + "\n")


# -------------------------
# Main
# -------------------------
def main():
    """Main evaluation pipeline."""
    args = parse_args()

    # Load dataset
    print(f"Loading frames from: {args.scene}")
    pairs = list_frames(args.scene)
    if args.max_frames > 0:
        pairs = pairs[:args.max_frames]
    print(f"Found {len(pairs)} frame triplets")

    # Initialize YOLO model
    print(f"Loading YOLO model: {args.weights}")
    model = YOLO(args.weights)

    # Initialize metrics accumulators
    acc = MOTAccumulator(iou_thr=args.match_iou)
    alphas = np.round(np.arange(0.05, 0.96, 0.05), 2)  # 0.05..0.95
    hota_acc = HOTAAccumulator(alphas)

    # Process each frame
    print("Starting evaluation...")
    for idx, (rgb_path, sem_path, info_path) in enumerate(pairs):
        # Load images
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        sem = cv2.imread(sem_path, cv2.IMREAD_COLOR)
        if rgb is None or sem is None:
            print(f"Skipping unreadable frame: {rgb_path}")
            continue

        # Extract ground truth
        id_to_color_bgr, id_to_classname = load_semantic_info(info_path)
        gt_ids, gt_boxes = compute_gt_from_semantic(
            sem, id_to_color_bgr, id_to_classname, min_area=args.min_area
        )

        # Run YOLO tracking
        res = model.track(
            rgb,
            persist=True,
            tracker=args.tracker,
            conf=args.conf,
            iou=args.iou,
            verbose=False
        )[0]

        pred_ids, pred_boxes = get_yolo_tracks(res, rgb.shape, conf_thr=args.conf)

        # Update metrics
        acc.update(gt_ids, gt_boxes, pred_ids, pred_boxes)
        hota_acc.update(gt_ids, gt_boxes, pred_ids, pred_boxes)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1}/{len(pairs)} frames...")

    # Compute multi-threshold HOTA
    hota_results = hota_acc.compute()

    # Generate plots
    plot_dir = os.path.join(args.scene, "plottings")
    tracker_name = os.path.splitext(args.tracker)[0]  # Remove .yaml
    model_name = os.path.splitext(os.path.basename(args.weights))[0]
    plot_hota_curves(
        hota_results=hota_results,
        out_dir=plot_dir,
        prefix=f"{model_name}_{tracker_name}"
    )

    # Print results
    print_results(acc, hota_results, len(pairs), args.match_iou)


if __name__ == "__main__":
    main()
