#!/usr/bin/env python3
"""
Compare YOLO Tracking Results

Utility to compare tracking metrics across different YOLO models and tracker configurations.

Usage:
    python compare_results.py --results_dir <dataset/results>
    python compare_results.py --results_dir <dataset/results> --output comparison.csv
"""
import os
import json
import argparse
import pandas as pd
from datetime import datetime


def parse_args():
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(description="Compare YOLO tracking results")
    ap.add_argument("--results_dir", required=True,
                    help="Directory containing result JSON files")
    ap.add_argument("--output", default=None,
                    help="Output CSV file path (optional)")
    ap.add_argument("--sort_by", default="HOTA",
                    choices=["HOTA", "MOTA", "IDF1", "DetA", "AssA"],
                    help="Metric to sort results by")
    return ap.parse_args()


def load_all_results(results_dir):
    """
    Load all result JSON files from directory.
    
    Args:
        results_dir: Directory containing JSON result files
        
    Returns:
        List of result dictionaries
    """
    results = []
    
    if not os.path.exists(results_dir):
        print(f"ERROR: Results directory not found: {results_dir}")
        return results
    
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON result files found in {results_dir}")
        return results
    
    for filename in json_files:
        filepath = os.path.join(results_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    return results


def create_comparison_table(results, sort_by="HOTA"):
    """
    Create a comparison table from results.
    
    Args:
        results: List of result dictionaries
        sort_by: Metric name to sort by
        
    Returns:
        pandas DataFrame with comparison
    """
    rows = []
    
    for result in results:
        eval_info = result.get("evaluation_info", {})
        overall = result.get("overall_metrics", {})
        
        # Extract model and tracker names
        model_path = eval_info.get("model_weights", "")
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        tracker_config = eval_info.get("tracker_config", "")
        tracker_name = os.path.splitext(tracker_config)[0]
        
        row = {
            "Model": model_name,
            "Tracker": tracker_name,
            "Conf": eval_info.get("conf_threshold", 0.0),
            "NMS_IoU": eval_info.get("nms_iou_threshold", 0.0),
            "Match_IoU": eval_info.get("match_iou_threshold", 0.0),
            "Scenes": overall.get("total_scenes", 0),
            "Frames": overall.get("total_frames", 0),
            "HOTA": overall.get("HOTA", 0.0),
            "MOTA": overall.get("MOTA", 0.0),
            "MOTP": overall.get("MOTP", 0.0),
            "IDF1": overall.get("IDF1", 0.0),
            "DetA": overall.get("DetA", 0.0),
            "AssA": overall.get("AssA", 0.0),
            "LocA": overall.get("LocA", 0.0),
            "TP": overall.get("TP", 0),
            "FP": overall.get("FP", 0),
            "FN": overall.get("FN", 0),
            "IDSW": overall.get("IDSW", 0),
            "Timestamp": eval_info.get("timestamp", ""),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by specified metric (descending)
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=False)
    
    return df


def print_comparison_table(df):
    """Print comparison table in a readable format."""
    print("\n" + "="*120)
    print("YOLO TRACKING RESULTS COMPARISON")
    print("="*120)
    
    # Main metrics table
    print("\n--- Model Performance (sorted by HOTA) ---")
    display_cols = ["Model", "Tracker", "HOTA", "MOTA", "IDF1", "DetA", "AssA", "LocA", "Scenes", "Frames"]
    print(df[display_cols].to_string(index=False))
    
    # Configuration details
    print("\n--- Configuration Details ---")
    config_cols = ["Model", "Tracker", "Conf", "NMS_IoU", "Match_IoU", "Timestamp"]
    print(df[config_cols].to_string(index=False))
    
    # Detection statistics
    print("\n--- Detection Statistics ---")
    stats_cols = ["Model", "Tracker", "TP", "FP", "FN", "IDSW"]
    print(df[stats_cols].to_string(index=False))
    
    print("="*120 + "\n")


def print_best_configurations(df):
    """Print best performing configurations for each metric."""
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS PER METRIC")
    print("="*80)
    
    metrics = ["HOTA", "MOTA", "IDF1", "DetA", "AssA", "LocA"]
    
    for metric in metrics:
        if metric in df.columns and len(df) > 0:
            best = df.loc[df[metric].idxmax()]
            print(f"\n{metric}: {best[metric]:.6f}")
            print(f"  Model: {best['Model']}")
            print(f"  Tracker: {best['Tracker']}")
            print(f"  Conf: {best['Conf']:.2f}, NMS_IoU: {best['NMS_IoU']:.2f}")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main comparison pipeline."""
    args = parse_args()
    
    # Load all results
    print(f"Loading results from: {args.results_dir}")
    results = load_all_results(args.results_dir)
    
    if not results:
        print("No results to compare.")
        return
    
    print(f"Loaded {len(results)} result files")
    
    # Create comparison table
    df = create_comparison_table(results, sort_by=args.sort_by)
    
    # Print comparison
    print_comparison_table(df)
    print_best_configurations(df)
    
    # Save to CSV if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"[OK] Comparison saved to: {args.output}")


if __name__ == "__main__":
    main()
