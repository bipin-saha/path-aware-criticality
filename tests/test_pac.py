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
