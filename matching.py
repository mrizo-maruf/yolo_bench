"""
Geometry and matching utilities for object detection/tracking.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between two sets of bounding boxes.
    
    Args:
        a: Array of shape (N, 4) in xyxy format
        b: Array of shape (M, 4) in xyxy format
        
    Returns:
        IoU matrix of shape (N, M)
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


def hungarian_matches(ious: np.ndarray, thr: float):
    """
    Perform Hungarian matching on IoU matrix.
    
    Args:
        ious: IoU matrix of shape (Ng, Np)
        thr: IoU threshold for valid matches
        
    Returns:
        Tuple of (matches, unmatched_gt, unmatched_pred)
        - matches: List of (gt_idx, pred_idx, iou) tuples
        - unmatched_gt: List of unmatched ground truth indices
        - unmatched_pred: List of unmatched prediction indices
    """
    Ng, Np = ious.shape
    
    if Ng == 0 and Np == 0:
        return [], [], []
    if Ng == 0:
        return [], [], list(range(Np))
    if Np == 0:
        return [], list(range(Ng)), []

    # Hungarian algorithm minimizes cost
    cost = 1.0 - ious
    gi, pi = linear_sum_assignment(cost)

    matches = []
    matched_g = set()
    matched_p = set()
    
    for g, p in zip(gi.tolist(), pi.tolist()):
        if ious[g, p] >= thr:
            matches.append((g, p, float(ious[g, p])))
            matched_g.add(g)
            matched_p.add(p)

    unmatched_g = [g for g in range(Ng) if g not in matched_g]
    unmatched_p = [p for p in range(Np) if p not in matched_p]
    
    return matches, unmatched_g, unmatched_p


def match_frame(gt_boxes, pred_boxes, iou_thr: float):
    """
    Match ground truth and prediction boxes for a single frame.
    
    Args:
        gt_boxes: Ground truth boxes (N, 4) in xyxy format
        pred_boxes: Predicted boxes (M, 4) in xyxy format
        iou_thr: IoU threshold for matching
        
    Returns:
        Tuple of (matches, unmatched_gt_idx, unmatched_pred_idx)
    """
    ious = iou_matrix_xyxy(gt_boxes, pred_boxes)
    return hungarian_matches(ious, iou_thr)
