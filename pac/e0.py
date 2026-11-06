"""E0 failure demos on val. F1: OCM-no-velocity weights == 1. F2: height-proxy vs stereo depth. F3: off-road share within d_stop."""
import json, sys
import numpy as np
from .criticality import stopping_distance, ocm_no_velocity

PERSON_H = 1.7


def run(attrs_path="data/boxes/val_attrs.json"):
    A = json.load(open(attrs_path))
    objs = [(a["cam"], o) for a in A.values() for o in a["objects"]]
    valid = [(c, o) for c, o in objs if o["Z"] is not None]
    dstop = stopping_distance()
    res = {"n_objects": len(objs), "n_with_depth": len(valid), "d_stop_m": round(dstop, 1),
           "depth_source": {s: sum(o["depth_source"] == s for _, o in objs) for s in ("stereo", "ground_plane", "none")}}
    # F1
    Z = np.array([o["Z"] for _, o in valid])
    w = ocm_no_velocity(Z)
    res["F1_ocm_weights_all_one"] = bool(np.all(w == 1.0))
    # F2: persons, stereo-only. height-proxy distance = f_y * 1.7 / h_px ; compare to stereo Z
    pers = [(c, o) for c, o in valid if o["cls"] == "person" and o["depth_source"] == "stereo" and o["h_px"] > 0]
    zh = np.array([c["fy"] * PERSON_H / o["h_px"] for c, o in pers]); zs = np.array([o["Z"] for _, o in pers])
    rel = np.abs(zh - zs) / zs
    fg_h = np.array([o["h_px"] >= 190 for _, o in pers]); fg_s = zs <= 22.0
    res["F2"] = {"n_persons": len(pers), "median_rel_error": round(float(np.median(rel)), 3),
                 "share_rel_error_gt_25pct": round(float(np.mean(rel > .25)), 3),
                 "share_foreground_misassigned_vs_22m": round(float(np.mean(fg_h != fg_s)), 3),
                 "share_misassigned_excluding_truncated": None}
    # F3: objects within stopping distance that are off-road
    near = [(c, o) for c, o in valid if o["Z"] <= dstop]
    known = [(c, o) for c, o in near if o["surface"] != -1]  # surface unknown = fully occluded ground, excluded
    off = [o for _, o in known if not o["road"]]
    res["F3"] = {"n_within_dstop": len(near), "n_surface_unknown": len(near) - len(known),
                 "share_offroad_of_known": round(len(off) / max(len(known), 1), 3),
                 "by_class": {c: round(float(np.mean([not o["road"] for _, o in known if o["cls"] == c])), 3)
                              for c in sorted({o["cls"] for _, o in known})}}
    return res


if __name__ == "__main__":
    r = run(*sys.argv[1:]); print(json.dumps(r, indent=2)); json.dump(r, open("data/boxes/e0_val.json", "w"), indent=2)
