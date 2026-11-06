"""Cityscapes ground-truth preparation: instance masks -> boxes, ignore regions, depth, surface."""
import json
import numpy as np

CLASS_IDS = {24: "person", 25: "rider", 26: "car", 27: "truck", 28: "bus",
             31: "train", 32: "motorcycle", 33: "bicycle"}
CLASS_NAMES = list(CLASS_IDS.values())
ROAD_ID, SIDEWALK_ID, PARKING_ID, RAIL_ID = 7, 8, 9, 10


def instances_to_boxes(instance_ids: np.ndarray):
    """Return (objects, ignore_boxes). Objects: dict(cls, box xyxy, mask_id). Values >=1000 are instances;
    class-pixels with value <1000 are group/crowd regions (ignore)."""
    objs, ignore = [], []
    for v in np.unique(instance_ids):
        cid = int(v // 1000) if v >= 1000 else int(v)
        if cid not in CLASS_IDS:
            continue
        ys, xs = np.nonzero(instance_ids == v)
        box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        rec = dict(cls=CLASS_IDS[cid], box=box, mask_id=int(v))
        (objs if v >= 1000 else ignore).append(rec)
    return objs, ignore


def load_camera(path):
    c = json.load(open(path))
    return dict(fx=c["intrinsic"]["fx"], fy=c["intrinsic"]["fy"], cx=c["intrinsic"]["u0"],
                cy=c["intrinsic"]["v0"], baseline=c["extrinsic"]["baseline"],
                h_cam=c["extrinsic"]["z"], pitch=c["extrinsic"].get("pitch", 0.0))


def decode_disparity(png: np.ndarray):
    """Cityscapes disparity PNG: d = (p-1)/256 for p>0; 0 = invalid. Returns float array with NaN invalid."""
    p = png.astype(np.float32)
    d = (p - 1.0) / 256.0
    d[png == 0] = np.nan
    return d


def object_depth(disp, mask, cam, box, min_pixels=20, erode=3):
    """Median stereo depth inside the eroded instance mask; ground-plane fallback. Returns (Z, source)."""
    import cv2
    m = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=erode).astype(bool) if erode else mask
    vals = disp[m]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size >= min_pixels:
        return cam["baseline"] * cam["fx"] / float(np.median(vals)), "stereo"
    vb = box[3]
    v_h = cam["cy"] - cam["fy"] * np.tan(cam["pitch"])  # horizon row (sign checked against stereo: 14% median error)
    dv = vb - v_h
    if dv <= 1:
        return float("inf"), "none"
    return cam["fy"] * cam["h_cam"] / dv, "ground_plane"


GROUND_IDS = (6, ROAD_ID, SIDEWALK_ID, PARKING_ID, RAIL_ID, 22)  # ground, road, sidewalk, parking, rail, terrain
UNKNOWN = -1


def contact_surface(label_ids, box):
    """Ground class under the object: majority over ground-class pixels in a band around the bottom edge
    (central half of the box width). Occluders (cars, people, poles) are ignored; UNKNOWN if no ground found."""
    h, w = label_ids.shape
    x0, x1 = box[0] + (box[2] - box[0]) // 4, box[2] - (box[2] - box[0]) // 4
    x0, x1 = max(x0, 0), max(min(max(x1, x0 + 1), w), 1)
    for dy in (3, 8, 20):  # widen the search band if the first is fully occluded
        win = label_ids[max(box[3] - 2, 0):min(box[3] + dy, h), x0:x1].ravel()
        win = win[np.isin(win, GROUND_IDS)]
        if win.size:
            return int(np.bincount(win).argmax())
    return UNKNOWN


def is_road(label_id):
    return label_id in (ROAD_ID, RAIL_ID)
