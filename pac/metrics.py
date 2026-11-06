"""Matching and weighted recall/precision/AP. With weights == 1 these reduce to standard recall/precision/AP."""
import numpy as np


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def inside(box, region, thr=0.5):
    ix = max(0, min(box[2], region[2]) - max(box[0], region[0]))
    iy = max(0, min(box[3], region[3]) - max(box[1], region[1]))
    area = (box[2] - box[0]) * (box[3] - box[1])
    return area > 0 and ix * iy / area >= thr


def match_image(gts, ignores, dets, iou_thr=0.5):
    """Greedy per class by descending score. gts: [{cls,box}], dets: [{cls,box,score}].
    Returns per-det status ('tp','fp','ign') and matched gt index (or -1)."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i]["score"])
    used = set()
    status, gt_of = {}, {}
    for i in order:
        d = dets[i]
        best, bj = iou_thr, -1
        for j, g in enumerate(gts):
            if g["cls"] != d["cls"] or j in used:
                continue
            o = iou(d["box"], g["box"])
            if o >= best:
                best, bj = o, j
        if bj >= 0:
            used.add(bj); status[i] = "tp"; gt_of[i] = bj
        elif any(r["cls"] == d["cls"] and inside(d["box"], r["box"]) for r in ignores):
            status[i] = "ign"; gt_of[i] = -1
        else:
            status[i] = "fp"; gt_of[i] = -1
    return status, gt_of


def weighted_pr_curve(records, total_gt_weight, n_points=101):
    """records: list of (score, status, w_tp_or_fp). Returns recall/precision arrays sorted by score and PAC-AP.
    status 'tp' weight = GT criticality; 'fp' weight = FP criticality; 'ign' skipped."""
    recs = sorted([r for r in records if r[1] != "ign"], key=lambda r: -r[0])
    tp = np.cumsum([r[2] if r[1] == "tp" else 0.0 for r in recs])
    fp = np.cumsum([r[2] if r[1] == "fp" else 0.0 for r in recs])
    rec = tp / total_gt_weight if total_gt_weight > 0 else np.zeros_like(tp)
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    return rec, prec


def interpolated_ap(rec, prec, n_points=101):
    if len(rec) == 0:
        return 0.0
    ap = 0.0
    for t in np.linspace(0, 1, n_points):
        m = prec[rec >= t]
        ap += (m.max() if m.size else 0.0)
    return ap / n_points


def critical_miss_rate(per_image_missed_kappas, thr=0.8):
    """Share of images with >=1 missed GT whose kappa >= thr."""
    if not per_image_missed_kappas:
        return 0.0
    return float(np.mean([any(k >= thr for k in ks) for ks in per_image_missed_kappas]))
