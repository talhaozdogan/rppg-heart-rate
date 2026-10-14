"""POS (Plane-Orthogonal-to-Skin) rPPG algorithm.

Wang et al. (2017), "Algorithmic Principles of Remote PPG", IEEE TBME 64(7).

POS projects the temporally-normalised RGB onto a plane orthogonal to the skin-tone
direction. Motion/lighting mostly shift the colour vector along skin tone, while the
pulse sits on a roughly fixed chrominance direction, so this cancels much of the
intensity noise while keeping the pulse. Steps per overlapping window (paper Alg. 1):

    1. normalise:   Cn = C_win / mean(C_win)
    2. project:     S  = P @ Cn,  P = [[0, 1, -1], [-2, 1, 1]]
    3. alpha-tune:  h  = S1 + (std(S1)/std(S2)) * S2
    4. overlap-add: H[win] += (h - mean(h))
"""
from __future__ import annotations

import numpy as np

_P = np.array([[0.0, 1.0, -1.0],
               [-2.0, 1.0, 1.0]])


def pos(rgb: np.ndarray, fs: float, win_seconds: float = 1.6) -> np.ndarray:
    """POS pulse signal from uniformly-sampled (N, 3) RGB. Band-pass separately."""
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("rgb must be (N, 3)")
    n = rgb.shape[0]
    l = max(2, int(round(win_seconds * fs)))
    out = np.zeros(n, dtype=float)

    for end in range(l - 1, n):
        start = end - l + 1
        window = rgb[start:end + 1]                 # (l, 3)

        mu = window.mean(axis=0)                    # per-channel temporal mean
        mu[mu == 0] = 1e-9
        cn = window / mu                            # 1. temporal normalisation

        s = cn @ _P.T                               # 2. projection -> (l, 2)
        s1, s2 = s[:, 0], s[:, 1]

        std2 = s2.std()
        alpha = s1.std() / std2 if std2 > 1e-12 else 0.0
        h = s1 + alpha * s2                         # 3. alpha tuning

        out[start:end + 1] += h - h.mean()          # 4. overlap-add (mean-removed)

    return out
