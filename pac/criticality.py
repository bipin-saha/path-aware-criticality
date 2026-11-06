"""Criticality functions: PAC (starting form) and baselines. All take arrays so they vectorise."""
import numpy as np

VRU = {"person", "rider", "bicycle", "motorcycle"}


def stopping_distance(v_kmh=50.0, t_r=1.5, a=7.0):
    v = v_kmh / 3.6
    return v * t_r + v * v / (2 * a)


def kappa_d(Z, d_max, eps=0.05):
    Z = np.asarray(Z, float)
    return np.maximum(eps, 1.0 - (Z / d_max) ** 2)


def path_g(on_road, lateral, corridor=2.0, wide=5.0, g_off=0.2, g_far=0.5):
    on_road = np.asarray(on_road, bool)
    x = np.abs(np.asarray(lateral, float))
    ramp = 1.0 - (1.0 - g_far) * np.clip((x - corridor) / (wide - corridor), 0, 1)
    return np.where(on_road, ramp, g_off)


def class_h(classes, h_vru=1.0, h_veh=0.6):
    return np.array([h_vru if c in VRU else h_veh for c in classes], float)


def lateral_offset(box, Z, cam):
    return ((box[0] + box[2]) / 2.0 - cam["cx"]) * Z / cam["fx"]


def pac(Z, on_road, lateral, classes, d_max=None, use_path=True, use_class=True):
    d_max = d_max or stopping_distance()
    k = kappa_d(Z, d_max)
    if use_path:
        k = k * path_g(on_road, lateral)
    if use_class:
        k = k * class_h(classes)
    return k


# ---- baselines (same inputs) ----
def ocm_no_velocity(Z, d_max=None):
    """OCM with velocity-dependent terms at fallback 1: every object ends up maximally critical (F1)."""
    return np.ones_like(np.asarray(Z, float))


def ocm_distance_only(Z, d_max=None):
    return np.maximum(0.0, 1.0 - (np.asarray(Z, float) / (d_max or stopping_distance())) ** 2)


def inverse_distance(Z):
    return 1.0 / np.maximum(np.asarray(Z, float), 1.0)


def feifel_foreground(heights_px, min_h=190):
    return (np.asarray(heights_px) >= min_h).astype(float)
