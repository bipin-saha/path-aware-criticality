"""Small stats helpers (no scipy/sklearn: they do not load on this macOS)."""
import numpy as np


def auroc(scores, labels):
    """Rank-based AUROC with average ranks for ties."""
    s = np.asarray(scores, float); y = np.asarray(labels, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
