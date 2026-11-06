"""Score detector predictions with mAP@0.5 and criticality-weighted variants (PAC + baselines + ablations).
  python -m pac.evaluate data/cityscapes data/preds/yolo11m_coco.json data/preds/yoloe11m_zs.json [--min-h 10]
Per-class 101-point AP, mean over classes with >=1 GT. FP weight: --fp unit|crit (crit = depth+surface at predicted box)."""
import json, sys, argparse, pathlib
import numpy as np, cv2
from . import criticality as C
from .gt import decode_disparity, contact_surface, is_road, UNKNOWN, CLASS_NAMES
from .metrics import match_image, weighted_pr_curve, interpolated_ap

SCHEMES = ["unweighted", "ocm_no_velocity", "ocm_distance", "inverse_distance", "pac_distance_only", "pac_path", "pac"]
OP = 0.25  # operating-point score threshold for recall / precision / critical miss


def weights(scheme, Z, road, X, cls, d_max):
    Z = np.where(np.isfinite(Z), Z, 1e6)
    if scheme in ("unweighted", "ocm_no_velocity"): return np.ones(len(Z))
    if scheme == "ocm_distance": return C.ocm_distance_only(Z, d_max)
    if scheme == "inverse_distance": return C.inverse_distance(Z)
    if scheme == "pac_distance_only": return C.kappa_d(Z, d_max)
    if scheme == "pac_path": return C.kappa_d(Z, d_max) * C.path_g(road, X)
    if scheme == "pac": return C.pac(Z, road, X, cls, d_max)
    raise ValueError(scheme)


def det_geometry(det, disp, lab, cam):
    """Depth + surface + lateral offset for a predicted box: median disparity in the lower half, else ground plane."""
    x0, y0, x1, y1 = [int(round(v)) for v in det["box"]]
    h, w = disp.shape
    x0, x1, y1 = max(x0, 0), min(x1, w), min(y1, h); ym = max((y0 + y1) // 2, 0)
    v = disp[ym:y1, x0:x1].ravel() if y1 > ym and x1 > x0 else np.array([])
    v = v[np.isfinite(v) & (v > 0)]
    if v.size >= 20: Z = cam["baseline"] * cam["fx"] / float(np.median(v))
    else:
        dv = det["box"][3] - (cam["cy"] - cam["fy"] * np.tan(cam["pitch"]))
        Z = cam["fy"] * cam["h_cam"] / dv if dv > 1 else np.inf
    s = contact_surface(lab, [x0, y0, x1, y1])
    X = ((det["box"][0] + det["box"][2]) / 2 - cam["cx"]) * Z / cam["fx"] if np.isfinite(Z) else 0.0
    return Z, bool(is_road(s)), float(X)


def evaluate(root, attrs, preds_list, min_h=0, fp_mode="crit", d_max=None, merge_rider=False):
    d_max = d_max or C.stopping_distance()
    names = list(preds_list)
    rec = {n: {s: {c: [] for c in CLASS_NAMES} for s in SCHEMES} for n in names}   # (score,status,w)
    gtw = {s: {c: 0.0 for c in CLASS_NAMES} for s in SCHEMES}
    op = {n: {s: dict(tp=0.0, gt=0.0, fp=0.0) for s in SCHEMES} for n in names}
    cmiss = {n: [] for n in names}; fg = {n: [0, 0] for n in names}
    for k, (stem, A) in enumerate(sorted(attrs.items())):
        city = stem.split("_")[0]
        gts = [dict(o, cls="person") if (merge_rider and o["cls"] == "rider") else o for o in A["objects"] if o["h_px"] >= min_h]
        ign = A["ignore"]
        Zg = np.array([o["Z"] if o["Z"] is not None else np.inf for o in gts]); roadg = np.array([o["road"] for o in gts], bool)
        Xg = np.array([o["X"] if o["X"] is not None else 0.0 for o in gts]); clsg = [o["cls"] for o in gts]
        W = {s: (weights(s, Zg, roadg, Xg, clsg, d_max) if gts else np.zeros(0)) for s in SCHEMES}
        for s in SCHEMES:
            for j, g in enumerate(gts): gtw[s][g["cls"]] += W[s][j]
        disp = lab = None
        for n in names:
            dets = [d for d in preds_list[n].get(stem, []) if (d["box"][3] - d["box"][1]) >= min_h
                    and not (merge_rider and d["cls"] == "rider")]  # riders are scored as persons; rider detections dropped
            st, gof = match_image(gts, ign, dets)
            fpi = [i for i in range(len(dets)) if st[i] == "fp"]
            Wf = {s: np.ones(len(fpi)) for s in SCHEMES}
            if fp_mode == "crit" and fpi:
                if disp is None:
                    disp = decode_disparity(cv2.imread(str(root / "disparity" / "val" / city / f"{stem}_disparity.png"), cv2.IMREAD_UNCHANGED))
                    lab = cv2.imread(str(root / "gtFine" / "val" / city / f"{stem}_gtFine_labelIds.png"), cv2.IMREAD_UNCHANGED)
                geo = [det_geometry(dets[i], disp, lab, A["cam"]) for i in fpi]
                Zf, rf, Xf = np.array([g[0] for g in geo]), np.array([g[1] for g in geo], bool), np.array([g[2] for g in geo])
                cf = [dets[i]["cls"] for i in fpi]
                Wf = {s: weights(s, Zf, rf, Xf, cf, d_max) for s in SCHEMES}
            fpos = {i: p for p, i in enumerate(fpi)}
            missed = set(range(len(gts))) - {gof[i] for i in range(len(dets)) if st[i] == "tp" and dets[i]["score"] >= OP}
            for s in SCHEMES:
                for i, d in enumerate(dets):
                    if st[i] == "tp": rec[n][s][d["cls"]].append((d["score"], "tp", W[s][gof[i]]))
                    elif st[i] == "fp": rec[n][s][d["cls"]].append((d["score"], "fp", Wf[s][fpos[i]]))
                    if d["score"] >= OP:
                        if st[i] == "tp": op[n][s]["tp"] += W[s][gof[i]]
                        elif st[i] == "fp": op[n][s]["fp"] += Wf[s][fpos[i]]
                op[n][s]["gt"] += float(W[s].sum())
            cmiss[n].append([float(W["pac"][j]) for j in missed])
            for j, g in enumerate(gts):
                if g["cls"] == "person" and g["h_px"] >= 190:
                    fg[n][1] += 1; fg[n][0] += int(j in missed)
    res = {}
    for n in names:
        res[n] = {}
        for s in SCHEMES:
            aps = {}
            for c in CLASS_NAMES:
                if gtw[s][c] <= 0: continue
                r, p = weighted_pr_curve(rec[n][s][c], gtw[s][c])
                aps[c] = interpolated_ap(r, p)
            o = op[n][s]
            allr = [r for c in CLASS_NAMES for r in rec[n][s][c]]; tot = sum(gtw[s].values())
            rp, pp = weighted_pr_curve(allr, tot) if tot > 0 else (np.zeros(0), np.zeros(0))
            res[n][s] = dict(AP=float(np.mean(list(aps.values()))), pooled_AP=interpolated_ap(rp, pp), per_class=aps,
                             recall=o["tp"] / o["gt"] if o["gt"] else 0.0,
                             precision=o["tp"] / (o["tp"] + o["fp"]) if o["tp"] + o["fp"] else 0.0)
        res[n]["critical_miss_rate"] = float(np.mean([any(k >= 0.8 for k in ks) for ks in cmiss[n]]))
        res[n]["feifel_fg_person_miss"] = fg[n][0] / fg[n][1] if fg[n][1] else None
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("preds", nargs="+")
    ap.add_argument("--min-h", type=int, default=0); ap.add_argument("--fp", default="crit"); ap.add_argument("--out", default=None)
    ap.add_argument("--merge-rider", action="store_true")
    a = ap.parse_args()
    attrs = json.load(open("data/boxes/val_attrs.json"))
    preds = {pathlib.Path(p).stem: json.load(open(p)) for p in a.preds}
    res = evaluate(pathlib.Path(a.root), attrs, preds, a.min_h, a.fp, merge_rider=a.merge_rider)
    out = a.out or f"data/results_minh{a.min_h}_fp{a.fp}{'_mr' if a.merge_rider else ''}.json"
    json.dump(res, open(out, "w"), indent=1)
    names = list(preds)
    print(f"min_h={a.min_h} fp={a.fp}\n{'scheme':20s}" + "".join(f"{n[:16]:>18s}" for n in names) + "   gap(A-B)")
    for s in SCHEMES:
        v = [res[n][s]["AP"] for n in names]
        print(f"{s:20s}" + "".join(f"{x:18.3f}" for x in v) + (f"{(v[0]-v[1])*100:11.1f}" if len(v) == 2 else ""))
    print("pooled AP (all classes together):")
    for s in SCHEMES:
        v = [res[n][s]["pooled_AP"] for n in names]
        print(f"{s:20s}" + "".join(f"{x:18.3f}" for x in v) + (f"{(v[0]-v[1])*100:11.1f}" if len(v) == 2 else ""))
    for k in ("critical_miss_rate", "feifel_fg_person_miss"):
        print(f"{k:20s}" + "".join(f"{res[n][k]:18.3f}" for n in names))
    print("saved", out)


if __name__ == "__main__":
    main()
