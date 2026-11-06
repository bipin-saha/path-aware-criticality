"""Bootstrap over images (E7): 95% CIs for AP per model, YOLO11m-YOLOE gap per scheme, and (PAC gap - mAP gap).
Resampling image i m_i times == scaling its record weights by m_i, so each resample is a re-weighted cumsum.
  python -m pac.bootstrap data/cityscapes A.json B.json [--merge-rider] [--n 1000] [--seed 0] [--out results/boot.json]"""
import json, argparse, pathlib
import numpy as np
from .evaluate import evaluate, SCHEMES
from .gt import CLASS_NAMES


def _prep(recs):
    """recs: list of (score, status, w, img). Sorted by score desc; arrays for fast reweighting."""
    r = sorted([x for x in recs if x[1] != "ign"], key=lambda x: -x[0])
    return (np.array([x[3] for x in r], int), np.array([x[2] if x[1] == "tp" else 0.0 for x in r]),
            np.array([x[2] if x[1] == "fp" else 0.0 for x in r]))


def _ap(img, wtp, wfp, m, total):
    if total <= 0 or len(img) == 0: return 0.0
    ctp = np.cumsum(wtp * m[img]); cfp = np.cumsum(wfp * m[img])
    rec = ctp / total; den = ctp + cfp
    prec = np.divide(ctp, den, out=np.zeros_like(ctp), where=den > 0)
    env = np.maximum.accumulate(prec[::-1])[::-1]               # precision envelope: max prec at recall >= t
    idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
    return float(np.where(idx < len(rec), env[np.minimum(idx, len(rec) - 1)], 0.0).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("preds", nargs=2)
    ap.add_argument("--merge-rider", action="store_true"); ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/bootstrap.json")
    a = ap.parse_args()
    attrs = json.load(open("data/boxes/val_attrs.json")); N = len(attrs)
    preds = {pathlib.Path(p).stem: json.load(open(p)) for p in a.preds}; names = list(preds)
    res, raw = evaluate(pathlib.Path(a.root), attrs, preds, merge_rider=a.merge_rider, return_raw=True)
    prep = {n: {s: {c: _prep(raw["rec"][n][s][c]) for c in CLASS_NAMES} for s in SCHEMES} for n in names}
    gi = raw["gimg"]
    def stat(n, s, m):
        aps = []
        for c in CLASS_NAMES:
            tot = float((gi[s][c] * m).sum())
            if tot > 0: aps.append(_ap(*prep[n][s][c], m, tot))
        return float(np.mean(aps))
    ones = np.ones(N)
    for n in names:                      # sanity: bootstrap machinery at multiplicity 1 reproduces evaluate()
        for s in SCHEMES: assert abs(stat(n, s, ones) - res[n][s]["AP"]) < 1e-9, (n, s)
    rng = np.random.default_rng(a.seed); B = {n: {s: [] for s in SCHEMES} for n in names}
    for b in range(a.n):
        m = np.bincount(rng.integers(0, N, N), minlength=N).astype(float)
        for n in names:
            for s in SCHEMES: B[n][s].append(stat(n, s, m))
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    out = {"n_resamples": a.n, "seed": a.seed, "merge_rider": a.merge_rider, "models": names, "schemes": {}}
    A, Bm = names
    for s in SCHEMES:
        gap = (np.array(B[A][s]) - np.array(B[Bm][s])) * 100
        d = gap - (np.array(B[A]["unweighted"]) - np.array(B[Bm]["unweighted"])) * 100   # this scheme's gap minus the mAP gap
        out["schemes"][s] = {"AP": {n: dict(point=res[n][s]["AP"], ci95=ci(B[n][s])) for n in names},
                             "gap_points": dict(point=(res[A][s]["AP"] - res[Bm][s]["AP"]) * 100, ci95=ci(gap)),
                             "gap_minus_mAP_gap_points": dict(point=(res[A][s]["AP"] - res[Bm][s]["AP"] - res[A]["unweighted"]["AP"] + res[Bm]["unweighted"]["AP"]) * 100, ci95=ci(d),
                                                              share_abs_ge_5=float(np.mean(np.abs(d) >= 5)))}
    pathlib.Path(a.out).parent.mkdir(exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
    print(f"{A} vs {Bm}, merge_rider={a.merge_rider}, {a.n} resamples")
    print(f"{'scheme':20s}{'gap (pts)':>12s}{'95% CI':>20s}{'gap-mAPgap':>13s}{'95% CI':>20s}")
    for s in SCHEMES:
        g, d = out["schemes"][s]["gap_points"], out["schemes"][s]["gap_minus_mAP_gap_points"]
        print(f"{s:20s}{g['point']:12.1f}{str([round(x,1) for x in g['ci95']]):>20s}{d['point']:13.1f}{str([round(x,1) for x in d['ci95']]):>20s}")
    print("saved", a.out)


if __name__ == "__main__":
    main()
