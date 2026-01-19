"""
YOLO-specific utilities for extracting tracking results.
"""
import numpy as np


def get_yolo_tracks(result, frame_shape, conf_thr: float):
    """
    Extract tracking IDs and bounding boxes from YOLO tracking result.
    
    Args:
        result: YOLO result object from model.track()
        frame_shape: Shape of the input frame (H, W, C)
        conf_thr: Confidence threshold for filtering detections
        
    Returns:
        Tuple of (pred_ids, pred_boxes):
        - pred_ids: Array of shape (Np,) with track IDs
        - pred_boxes: Array of shape (Np, 4) with boxes in xyxy format
        
    Note:
        Only returns detections with valid track IDs. Detections without
        track IDs are filtered out as they cannot be used for tracking metrics.
    """
    H, W = frame_shape[:2]

    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    # Check if tracking IDs are available
    if not getattr(result.boxes, "is_track", False) or result.boxes.id is None:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    # Extract boxes, confidences, and track IDs
    xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    conf = result.boxes.conf.cpu().numpy().astype(np.float32) if result.boxes.conf is not None else np.ones((len(xyxy),), dtype=np.float32)
    tids = result.boxes.id.int().cpu().numpy().astype(np.int64)

    # Filter by confidence threshold
    keep = conf >= conf_thr
    xyxy = xyxy[keep]
    tids = tids[keep]

    # Clip boxes to image boundaries
    if xyxy.shape[0] > 0:
        xyxy[:, 0] = np.clip(xyxy[:, 0], 0, W - 1)
        xyxy[:, 1] = np.clip(xyxy[:, 1], 0, H - 1)
        xyxy[:, 2] = np.clip(xyxy[:, 2], 0, W - 1)
        xyxy[:, 3] = np.clip(xyxy[:, 3], 0, H - 1)

    return tids, xyxy
