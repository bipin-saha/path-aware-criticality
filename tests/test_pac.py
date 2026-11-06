import numpy as np
from pac.metrics import match_image, weighted_pr_curve, interpolated_ap
from pac import criticality as C


def _toy():
    gts = [dict(cls="car", box=[0, 0, 10, 10]), dict(cls="person", box=[20, 0, 30, 10]),
           dict(cls="car", box=[40, 0, 50, 10])]
    dets = [dict(cls="car", box=[0, 0, 10, 10], score=.9), dict(cls="person", box=[21, 0, 30, 10], score=.8),
            dict(cls="car", box=[100, 0, 110, 10], score=.7)]
    return gts, dets


def test_unit_weights_reduce_to_standard_ap():
    gts, dets = _toy()
    st, _ = match_image(gts, [], dets)
    recs = [(d["score"], st[i], 1.0) for i, d in enumerate(dets)]
    rec, prec = weighted_pr_curve(recs, len(gts))
    # standard: tp,tp,fp -> rec .333,.667,.667 ; prec 1,1,.667
    assert np.allclose(rec, [1/3, 2/3, 2/3]) and np.allclose(prec, [1, 1, 2/3])
    assert abs(interpolated_ap(rec, prec) - (np.sum(np.linspace(0, 1, 101) <= 2/3) / 101)) < 1e-9


def test_ocm_no_velocity_collapses_to_recall():
    Z = np.array([5., 30., 80.])
    assert np.all(C.ocm_no_velocity(Z) == 1.0)


def test_pac_offroad_lower_than_inlane():
    k = C.pac([10, 10], [True, False], [0.0, 0.0], ["person", "person"])
    assert k[0] > k[1] and np.isclose(k[1] / k[0], 0.2)


def test_kappa_d_monotone_and_floor():
    k = C.kappa_d([1, 20, 35, 100], C.stopping_distance())
    assert np.all(np.diff(k) <= 0) and k[-1] == 0.05


def test_stopping_distance():
    assert 34 < C.stopping_distance() < 37


def test_auroc():
    from pac.stats import auroc
    assert auroc([.1, .2, .8, .9], [0, 0, 1, 1]) == 1.0
    assert auroc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5


def test_e1_scorer_on_synthetic_labels():
    """Synthetic key and labels (NOT annotations): checks the scorer's mechanics only."""
    from pac.e1_score import evaluate, cohen_kappa
    rng = np.random.default_rng(0); key = {}
    for i in range(120):
        key[f"img/o{i:03d}.jpg"] = dict(Z=float(rng.uniform(3, 60)), road=bool(rng.random() < .5), X=0.0, cls="person", h_px=float(rng.uniform(20, 300)))
    ids = list(key); y = [(key[i]["Z"] < 30) and key[i]["road"] for i in ids]   # labels follow distance+road exactly
    r = evaluate(key, ids, y, n=50)
    assert r["pac"]["auroc"] > 0.95 and abs(r["ocm_no_velocity"]["auroc"] - 0.5) < 1e-9
    assert r["pac"]["auroc"] > r["inverse_distance"]["auroc"]
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0 and abs(cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-9
