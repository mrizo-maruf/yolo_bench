#!/usr/bin/env python3
"""
Batch YOLO Tracking Evaluation

Evaluate multiple YOLO models and tracker configurations in batch.

Usage:
    python batch_evaluate.py --dataset <dataset_path> --config batch_config.json
"""
import os
import json
import argparse
import subprocess
from datetime import datetime


def parse_args():
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(description="Batch evaluate YOLO tracking")
    ap.add_argument("--dataset", required=True,
                    help="Path to dataset directory")
    ap.add_argument("--config", required=True,
                    help="JSON config file with models and trackers to test")
    ap.add_argument("--max_frames", type=int, default=-1,
                    help="Max frames per scene (-1 = all)")
    return ap.parse_args()


def load_config(config_path):
    """
    Load batch evaluation configuration.
    
    Expected JSON format:
    {
        "models": ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        "trackers": ["bytetrack.yaml", "botsort.yaml"],
        "conf_thresholds": [0.25, 0.3],
        "iou_thresholds": [0.5],
        "match_iou": 0.5,
        "min_area": 120
    }
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def run_evaluation(dataset, model, tracker, conf, iou, match_iou, min_area, max_frames):
    """
    Run a single evaluation configuration.
    
    Args:
        dataset: Dataset path
        model: Model weights path
        tracker: Tracker config file
        conf: Confidence threshold
        iou: NMS IoU threshold
        match_iou: Matching IoU threshold
        min_area: Minimum GT area
        max_frames: Max frames per scene
        
    Returns:
        True if successful, False otherwise
    """
    cmd = [
        "python", "yolo_metrics.py",
        "--dataset", dataset,
        "--weights", model,
        "--tracker", tracker,
        "--conf", str(conf),
        "--iou", str(iou),
        "--match_iou", str(match_iou),
        "--min_area", str(min_area),
        "--max_frames", str(max_frames)
    ]
    
    model_name = os.path.splitext(os.path.basename(model))[0]
    tracker_name = os.path.splitext(tracker)[0]
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name} + {tracker_name}")
    print(f"Config: conf={conf}, iou={iou}, match_iou={match_iou}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n[OK] Completed: {model_name} + {tracker_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed: {model_name} + {tracker_name}")
        print(f"Error: {e}")
        return False


def main():
    """Main batch evaluation pipeline."""
    args = parse_args()
    
    # Load configuration
    print(f"Loading batch config: {args.config}")
    config = load_config(args.config)
    
    models = config.get("models", [])
    trackers = config.get("trackers", ["bytetrack.yaml"])
    conf_thresholds = config.get("conf_thresholds", [0.25])
    iou_thresholds = config.get("iou_thresholds", [0.5])
    match_iou = config.get("match_iou", 0.5)
    min_area = config.get("min_area", 120)
    
    if not models:
        print("ERROR: No models specified in config")
        return
    
    # Calculate total number of evaluations
    total = len(models) * len(trackers) * len(conf_thresholds) * len(iou_thresholds)
    print(f"\nTotal evaluations to run: {total}")
    print(f"  Models: {len(models)}")
    print(f"  Trackers: {len(trackers)}")
    print(f"  Conf thresholds: {len(conf_thresholds)}")
    print(f"  IoU thresholds: {len(iou_thresholds)}")
    
    # Run all combinations
    success_count = 0
    failure_count = 0
    start_time = datetime.now()
    
    eval_num = 0
    for model in models:
        for tracker in trackers:
            for conf in conf_thresholds:
                for iou in iou_thresholds:
                    eval_num += 1
                    print(f"\n[{eval_num}/{total}] Starting evaluation...")
                    
                    success = run_evaluation(
                        dataset=args.dataset,
                        model=model,
                        tracker=tracker,
                        conf=conf,
                        iou=iou,
                        match_iou=match_iou,
                        min_area=min_area,
                        max_frames=args.max_frames
                    )
                    
                    if success:
                        success_count += 1
                    else:
                        failure_count += 1
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print("BATCH EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total evaluations: {total}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failure_count}")
    print(f"Duration: {duration}")
    print(f"{'='*80}\n")
    
    print(f"Results saved in: {os.path.join(args.dataset, 'results')}")
    print(f"Use compare_results.py to compare all results.")


if __name__ == "__main__":
    main()
