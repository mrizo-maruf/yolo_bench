#!/usr/bin/env python3
"""
YOLO Tracking Metrics Evaluation Script

Evaluates YOLO tracking performance using multiple metrics:
- MOTA/MOTP (CLEAR MOT metrics)
- IDF1 (Identity F1 Score)
- HOTA (Higher Order Tracking Accuracy)

Supports:
- Single scene evaluation
- Multi-scene dataset evaluation with per-scene and overall metrics
- Results saved with model/tracker configuration for comparison

Usage:
    python yolo_metrics.py --dataset <dataset_dir> --weights <model.pt> --tracker <tracker.yaml>
"""
import os
import argparse
import cv2
import numpy as np
import json
from datetime import datetime
from ultralytics import YOLO

# Import modularized components
from io_utils import list_frames, load_semantic_info, compute_gt_from_semantic
from metrics import MOTAccumulator, HOTAAccumulator
from yolo_utils import get_yolo_tracks
from visualization import plot_hota_curves


def parse_args():
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(description="Evaluate YOLO tracking metrics")
    ap.add_argument("--dataset", required=True, 
                    help="Path to dataset directory (contains scene subdirectories)")
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
                    help="Limit number of frames per scene (-1 = all)")
    return ap.parse_args()


def get_scene_dirs(dataset_path):
    """
    Find all scene directories in the dataset.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        List of scene directory paths
    """
    scenes = []
    for item in os.listdir(dataset_path):
        scene_path = os.path.join(dataset_path, item)
        if not os.path.isdir(scene_path):
            continue
        # Check if it has rgb/ and seg/ subdirectories
        rgb_dir = os.path.join(scene_path, "rgb")
        seg_dir = os.path.join(scene_path, "seg")
        if os.path.isdir(rgb_dir) and os.path.isdir(seg_dir):
            scenes.append(scene_path)
    return sorted(scenes)


def evaluate_scene(scene_dir, model, args, scene_name):
    """
    Evaluate tracking metrics for a single scene.
    
    Args:
        scene_dir: Path to scene directory
        model: YOLO model instance
        args: Parsed arguments
        scene_name: Name of the scene for reporting
        
    Returns:
        Dictionary containing all metrics for the scene
    """
    print(f"\n{'='*60}")
    print(f"Evaluating scene: {scene_name}")
    print(f"{'='*60}")
    
    # Load frames
    pairs = list_frames(scene_dir)
    if args.max_frames > 0:
        pairs = pairs[:args.max_frames]
    print(f"Found {len(pairs)} frame triplets")

    # Initialize metrics accumulators
    acc = MOTAccumulator(iou_thr=args.match_iou)
    alphas = np.round(np.arange(0.05, 0.96, 0.05), 2)
    hota_acc = HOTAAccumulator(alphas)

    # Process each frame
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
            print(f"  Processed {idx+1}/{len(pairs)} frames...")

    # Compute metrics
    mota, motp = acc.compute_mota_motp()
    idf1, idp, idr, idtp, idfp, idfn = acc.compute_idf1()
    hota_single, det_a_single, ass_a_single = acc.compute_hota()
    hota_results = hota_acc.compute()

    # Generate plots
    plot_dir = os.path.join(scene_dir, "plottings")
    tracker_name = os.path.splitext(args.tracker)[0]
    model_name = os.path.splitext(os.path.basename(args.weights))[0]
    plot_hota_curves(
        hota_results=hota_results,
        out_dir=plot_dir,
        prefix=f"{model_name}_{tracker_name}"
    )

    return {
        "scene_name": scene_name,
        "num_frames": len(pairs),
        "MOTA": float(mota),
        "MOTP": float(motp),
        "IDF1": float(idf1),
        "IDP": float(idp),
        "IDR": float(idr),
        "HOTA": float(hota_results["HOTA"]),
        "DetA": float(hota_results["DetA"]),
        "AssA": float(hota_results["AssA"]),
        "LocA": float(hota_results["LocA"]),
        "TP": int(acc.tp),
        "FP": int(acc.fp),
        "FN": int(acc.fn),
        "IDSW": int(acc.idsw),
        "IDTP": int(idtp),
        "IDFP": int(idfp),
        "IDFN": int(idfn),
    }


def print_scene_results(results):
    """Print results for a single scene."""
    print(f"\n{'='*60}")
    print(f"Scene: {results['scene_name']}")
    print(f"{'='*60}")
    print(f"Frames: {results['num_frames']}")
    print(f"\n--- CLEAR MOT ---")
    print(f"TP: {results['TP']}  FP: {results['FP']}  FN: {results['FN']}  IDSW: {results['IDSW']}")
    print(f"MOTA: {results['MOTA']:.6f}")
    print(f"MOTP: {results['MOTP']:.6f}")
    print(f"\n--- IDF1 ---")
    print(f"IDTP: {results['IDTP']}  IDFP: {results['IDFP']}  IDFN: {results['IDFN']}")
    print(f"IDP: {results['IDP']:.6f}")
    print(f"IDR: {results['IDR']:.6f}")
    print(f"IDF1: {results['IDF1']:.6f}")
    print(f"\n--- HOTA ---")
    print(f"HOTA: {results['HOTA']:.6f}")
    print(f"DetA: {results['DetA']:.6f}")
    print(f"AssA: {results['AssA']:.6f}")
    print(f"LocA: {results['LocA']:.6f}")


def compute_overall_metrics(scene_results):
    """
    Compute overall metrics by averaging across scenes.
    
    Args:
        scene_results: List of per-scene result dictionaries
        
    Returns:
        Dictionary with overall metrics
    """
    if not scene_results:
        return {}
    
    total_frames = sum(r["num_frames"] for r in scene_results)
    
    # Weighted averages by number of frames
    overall = {
        "total_scenes": len(scene_results),
        "total_frames": total_frames,
        "MOTA": sum(r["MOTA"] * r["num_frames"] for r in scene_results) / total_frames,
        "MOTP": sum(r["MOTP"] * r["num_frames"] for r in scene_results) / total_frames,
        "IDF1": sum(r["IDF1"] * r["num_frames"] for r in scene_results) / total_frames,
        "IDP": sum(r["IDP"] * r["num_frames"] for r in scene_results) / total_frames,
        "IDR": sum(r["IDR"] * r["num_frames"] for r in scene_results) / total_frames,
        "HOTA": sum(r["HOTA"] * r["num_frames"] for r in scene_results) / total_frames,
        "DetA": sum(r["DetA"] * r["num_frames"] for r in scene_results) / total_frames,
        "AssA": sum(r["AssA"] * r["num_frames"] for r in scene_results) / total_frames,
        "LocA": sum(r["LocA"] * r["num_frames"] for r in scene_results) / total_frames,
        "TP": sum(r["TP"] for r in scene_results),
        "FP": sum(r["FP"] for r in scene_results),
        "FN": sum(r["FN"] for r in scene_results),
        "IDSW": sum(r["IDSW"] for r in scene_results),
        "IDTP": sum(r["IDTP"] for r in scene_results),
        "IDFP": sum(r["IDFP"] for r in scene_results),
        "IDFN": sum(r["IDFN"] for r in scene_results),
    }
    
    return overall


def print_overall_results(overall):
    """Print overall dataset results."""
    print(f"\n{'='*60}")
    print("OVERALL DATASET RESULTS")
    print(f"{'='*60}")
    print(f"Total scenes: {overall['total_scenes']}")
    print(f"Total frames: {overall['total_frames']}")
    print(f"\n--- CLEAR MOT ---")
    print(f"TP: {overall['TP']}  FP: {overall['FP']}  FN: {overall['FN']}  IDSW: {overall['IDSW']}")
    print(f"MOTA: {overall['MOTA']:.6f}")
    print(f"MOTP: {overall['MOTP']:.6f}")
    print(f"\n--- IDF1 ---")
    print(f"IDTP: {overall['IDTP']}  IDFP: {overall['IDFP']}  IDFN: {overall['IDFN']}")
    print(f"IDP: {overall['IDP']:.6f}")
    print(f"IDR: {overall['IDR']:.6f}")
    print(f"IDF1: {overall['IDF1']:.6f}")
    print(f"\n--- HOTA ---")
    print(f"HOTA: {overall['HOTA']:.6f}")
    print(f"DetA: {overall['DetA']:.6f}")
    print(f"AssA: {overall['AssA']:.6f}")
    print(f"LocA: {overall['LocA']:.6f}")
    print(f"{'='*60}\n")


def save_results(dataset_path, args, scene_results, overall_results):
    """
    Save evaluation results to JSON file.
    
    Args:
        dataset_path: Path to dataset directory
        args: Parsed arguments
        scene_results: List of per-scene result dictionaries
        overall_results: Overall dataset results dictionary
    """
    # Create results directory
    results_dir = os.path.join(dataset_path, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.splitext(os.path.basename(args.weights))[0]
    tracker_name = os.path.splitext(args.tracker)[0]
    filename = f"{model_name}_{tracker_name}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    # Prepare results data
    results_data = {
        "evaluation_info": {
            "timestamp": timestamp,
            "dataset_path": dataset_path,
            "model_weights": args.weights,
            "tracker_config": args.tracker,
            "conf_threshold": args.conf,
            "nms_iou_threshold": args.iou,
            "match_iou_threshold": args.match_iou,
            "min_area": args.min_area,
            "max_frames_per_scene": args.max_frames,
        },
        "overall_metrics": overall_results,
        "per_scene_metrics": scene_results,
    }
    
    # Save to JSON
    with open(filepath, "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"[OK] Results saved to: {filepath}")
    return filepath


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

    # Find all scenes in dataset
    print(f"Scanning dataset: {args.dataset}")
    scene_dirs = get_scene_dirs(args.dataset)
    
    if not scene_dirs:
        print(f"ERROR: No valid scene directories found in {args.dataset}")
        print("Each scene must have rgb/ and seg/ subdirectories")
        return
    
    print(f"Found {len(scene_dirs)} scenes")
    for scene_dir in scene_dirs:
        print(f"  - {os.path.basename(scene_dir)}")

    # Initialize YOLO model
    print(f"\nLoading YOLO model: {args.weights}")
    model = YOLO(args.weights)

    # Evaluate each scene
    scene_results = []
    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        
        # Reset model tracker state between scenes
        model.predictor = None
        
        try:
            result = evaluate_scene(scene_dir, model, args, scene_name)
            scene_results.append(result)
            print_scene_results(result)
        except Exception as e:
            print(f"ERROR evaluating scene {scene_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Compute and display overall metrics
    if scene_results:
        overall_results = compute_overall_metrics(scene_results)
        print_overall_results(overall_results)
        
        # Save results to JSON
        save_results(args.dataset, args, scene_results, overall_results)
    else:
        print("No scenes were successfully evaluated.")


if __name__ == "__main__":
    main()
