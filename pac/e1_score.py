"""E1 step 2: score criticality metrics against human labels.
  python -m pac.e1_score data/e1_key.json labels_A.json labels_B.json [--final final.json] [--n 1000]
Labels: {"labels": {"img/o000.jpg": 0|1, ...}}. Without --final, objects where both annotators agree are used and
the disagreements are listed for discussion (resolve them, write final.json = {"img/o000.jpg": 0|1}, rerun).
Metrics (same inputs as the paper): PAC, PAC path-only (no class), PAC distance-only, OCM no velocity (constant),
OCM distance-only, inverse distance, height rule (h>=190 px), height (continuous, generous)."""
import json, argparse
import numpy as np
from .stats import auroc
from . import criticality as C

NAMES = ["pac", "pac_path", "pac_distance_only", "ocm_no_velocity", "ocm_distance", "inverse_distance", "height_rule_190", "height_px"]


def scores(key, ids):
    Z = np.array([key[i]["Z"] for i in ids]); road = np.array([key[i]["road"] for i in ids], bool)
    X = np.array([key[i]["X"] if key[i]["X"] is not None else 0.0 for i in ids]); cls = [key[i]["cls"] for i in ids]
    h = np.array([key[i]["h_px"] for i in ids], float); d = C.stopping_distance()
    return {"pac": C.pac(Z, road, X, cls, d), "pac_path": C.pac(Z, road, X, cls, d, use_class=False),
            "pac_distance_only": C.kappa_d(Z, d), "ocm_no_velocity": C.ocm_no_velocity(Z), "ocm_distance": C.ocm_distance_only(Z, d),
            "inverse_distance": C.inverse_distance(Z), "height_rule_190": C.feifel_foreground(h), "height_px": h}


def cohen_kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int); po = float(np.mean(a == b))
    pe = float(np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b)))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def best_acc(s, y):
    return float(max(np.mean((s >= t) == y) for t in np.unique(np.r_[s, s.max() + 1])))


def evaluate(key, ids, y, n=1000, seed=0):
    S = scores(key, ids); y = np.asarray(y, bool); rng = np.random.default_rng(seed)
    out = {k: dict(auroc=auroc(S[k], y), acc=best_acc(S[k], y)) for k in NAMES}
    boot = {k: [] for k in NAMES}
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        if y[ix].all() or (~y[ix]).all(): continue
        for k in NAMES: boot[k].append(auroc(S[k][ix], y[ix]))
    for k in NAMES: out[k]["ci95"] = [float(np.percentile(boot[k], 2.5)), float(np.percentile(boot[k], 97.5))]
    base = max((k for k in NAMES if k != "pac"), key=lambda k: out[k]["auroc"])   # strongest baseline on these labels
    m = min(len(boot["pac"]), len(boot[base])); dd = np.array(boot["pac"][:m]) - np.array(boot[base][:m])
    out["pac_minus_best_baseline"] = dict(baseline=base, point=out["pac"]["auroc"] - out[base]["auroc"], ci95=[float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("key"); ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--final"); ap.add_argument("--n", type=int, default=1000); a = ap.parse_args()
    key = json.load(open(a.key)); LA = json.load(open(a.a))["labels"]; LB = json.load(open(a.b))["labels"]
    ids = [i for i in key if i in LA and i in LB]
    ka = cohen_kappa([LA[i] for i in ids], [LB[i] for i in ids]); agree = [i for i in ids if LA[i] == LB[i]]
    print(f"both labelled {len(ids)}/{len(key)}; raw agreement {len(agree)/max(len(ids),1):.3f}; Cohen's kappa {ka:.3f}; disagreements {len(ids)-len(agree)}")
    if a.final:
        F = json.load(open(a.final)); use = [i for i in ids if i in F]; y = [F[i] for i in use]; print("using final labels:", len(use))
    else:
        use = agree; y = [LA[i] for i in use]; print("no --final: using only agreed objects:", len(use))
        print("disagreements to discuss:", [i for i in ids if LA[i] != LB[i]][:60])
    r = evaluate(key, use, y, a.n); print(f"positives (critical): {int(np.sum(y))}/{len(y)}")
    print(f"{'metric':20s}{'AUROC':>8s}{'95% CI':>18s}{'best-thr acc':>14s}")
    for k in NAMES: print(f"{k:20s}{r[k]['auroc']:8.3f}{str([round(x,3) for x in r[k]['ci95']]):>18s}{r[k]['acc']:14.3f}")
    d = r["pac_minus_best_baseline"]; print(f"PAC minus best baseline ({d['baseline']}): {d['point']:+.3f}, 95% CI {[round(x,3) for x in d['ci95']]}")
    json.dump(r | dict(kappa=ka, n=len(use)), open("results/e1_scores.json", "w"), indent=1)


if __name__ == "__main__":
    main()
