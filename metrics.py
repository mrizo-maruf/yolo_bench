"""
Metrics accumulators for tracking evaluation (MOTA, MOTP, IDF1, HOTA).
"""
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from matching import match_frame, iou_matrix_xyxy, hungarian_matches


class MOTAccumulator:
    """
    Accumulator for Multiple Object Tracking metrics.
    
    Computes:
    - MOTA (Multiple Object Tracking Accuracy)
    - MOTP (Multiple Object Tracking Precision)
    - IDF1 (ID F1 Score)
    - HOTA (Higher Order Tracking Accuracy) at single threshold
    """
    
    def __init__(self, iou_thr: float):
        """
        Args:
            iou_thr: IoU threshold for matching ground truth and predictions
        """
        self.iou_thr = iou_thr

        # For MOTA/MOTP
        self.total_gt = 0
        self.fp = 0
        self.fn = 0
        self.idsw = 0
        self.sum_iou = 0.0
        self.tp = 0

        # For IDF1 (global identity assignment)
        self.pair_counts = defaultdict(lambda: defaultdict(int))
        self.total_gt_dets = 0
        self.total_pred_dets = 0

        # For ID switches
        self.last_pred_for_gt = {}

        # For HOTA components at one threshold
        self.matched_events = []

    def update(self, gt_ids, gt_boxes, pred_ids, pred_boxes):
        """
        Update metrics with one frame of data.
        
        Args:
            gt_ids: Ground truth IDs (Ng,)
            gt_boxes: Ground truth boxes (Ng, 4) in xyxy
            pred_ids: Predicted IDs (Np,)
            pred_boxes: Predicted boxes (Np, 4) in xyxy
        """
        matches, un_g, un_p = match_frame(gt_boxes, pred_boxes, self.iou_thr)

        self.total_gt += len(gt_ids)
        self.total_gt_dets += len(gt_ids)
        self.total_pred_dets += len(pred_ids)

        # Count false positives and false negatives
        self.fn += len(un_g)
        self.fp += len(un_p)

        # Process matches
        self.tp += len(matches)
        for g_i, p_i, iou in matches:
            gt_id = int(gt_ids[g_i])
            pred_id = int(pred_ids[p_i])

            self.sum_iou += float(iou)

            # ID switch detection
            if gt_id in self.last_pred_for_gt and self.last_pred_for_gt[gt_id] != pred_id:
                self.idsw += 1
            self.last_pred_for_gt[gt_id] = pred_id

            # IDF1 counts
            self.pair_counts[gt_id][pred_id] += 1

            # HOTA matched events
            self.matched_events.append((gt_id, pred_id))

    def compute_mota_motp(self):
        """
        Compute MOTA and MOTP metrics.
        
        Returns:
            Tuple of (mota, motp)
        """
        mota = 1.0
        if self.total_gt > 0:
            mota = 1.0 - (self.fn + self.fp + self.idsw) / float(self.total_gt)

        motp = 0.0
        if self.tp > 0:
            motp = self.sum_iou / float(self.tp)

        return mota, motp

    def compute_idf1(self):
        """
        Compute IDF1 (ID F1 Score) metric.
        
        Returns:
            Tuple of (idf1, idp, idr, idtp, idfp, idfn)
        """
        gt_ids = sorted(self.pair_counts.keys())
        pred_ids = sorted({pid for g in self.pair_counts for pid in self.pair_counts[g].keys()})

        if len(gt_ids) == 0 or len(pred_ids) == 0:
            idtp = 0
            idfp = self.total_pred_dets
            idfn = self.total_gt_dets
            return 0.0, 0.0, 0.0, idtp, idfp, idfn

        # Build matrix of match counts
        mat = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
        for i, gid in enumerate(gt_ids):
            for j, pid in enumerate(pred_ids):
                mat[i, j] = int(self.pair_counts[gid].get(pid, 0))

        # Maximize total matches
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
        Compute single-threshold HOTA metric.
        
        Returns:
            Tuple of (hota, det_a, ass_a)
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


class HOTAAccumulator:
    """
    Accumulator for HOTA metric across multiple IoU thresholds.
    
    Computes HOTA, DetA, AssA, and LocA averaged over alpha thresholds.
    """
    
    def __init__(self, alphas):
        """
        Args:
            alphas: List or array of IoU thresholds (alpha values)
        """
        self.alphas = list(alphas)

        # Per alpha: TP, FP, FN, sum of localization IoU
        self.tp = {a: 0 for a in self.alphas}
        self.fp = {a: 0 for a in self.alphas}
        self.fn = {a: 0 for a in self.alphas}
        self.sum_lociou = {a: 0.0 for a in self.alphas}

        # Per alpha: pair counts for association IoU
        self.pair_counts = {a: defaultdict(lambda: defaultdict(int)) for a in self.alphas}

        # Per alpha: list of matched (gt_id, pred_id) occurrences
        self.matched_events = {a: [] for a in self.alphas}

    def update(self, gt_ids, gt_boxes, pred_ids, pred_boxes):
        """
        Update HOTA metrics with one frame of data.
        
        Args:
            gt_ids: Ground truth IDs (Ng,)
            gt_boxes: Ground truth boxes (Ng, 4) in xyxy
            pred_ids: Predicted IDs (Np,)
            pred_boxes: Predicted boxes (Np, 4) in xyxy
        """
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
        Compute HOTA metrics averaged over all alpha thresholds.
        
        Returns:
            Dictionary with HOTA, DetA, AssA, LocA and per-threshold curves
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

            # Association accuracy
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

        # Average over alphas
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
