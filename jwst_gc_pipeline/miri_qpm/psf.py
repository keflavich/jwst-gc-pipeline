"""Richardson-Lucy de-smearing of the pane ghost series (unit-sum 5 x 21 kernels), edge trim, variance scaling.

The pane PSF = imager PSF (x) K (a ghost series toward detector -x; strong in pane 3).  RL with
u <- u * K^T (d / K u) is invariant to the kernel scale and conserves the total, so it is never divided by sum K.
It changes diffuse surface brightness by < 0.1 %, amplifies white noise (variance x 1.3 / 1.3 / 3.1, anti-correlated
along x) and cannot restore the ghosts of sources beyond a pane edge or a large NaN structure: the PSF study measured
2-8 % deficits within ~3 px of them.  Those pixels form the edge margin (EDGE_TRIM, a round 3-px distance):
flagged by default (kept), dropped only with edge_mode = 'drop'.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve

RL_ITER = 30
KX = 10                                         # kernel half-width in x (5 x 21 kernels)
RL_VAR_FACTOR = {1: 1.3, 2: 1.3, 3: 3.1}        # white-noise variance amplification after RL (psf track 4.3)
EDGE_TRIM = 3                                   # edge margin [px] at pane edges / large NaN structures
NAN_CLUSTER = 20                                # NaN clusters >= this many px count as edges
PAD = 2 * KX + 4                                # edge-value padding of each filled pane box before RL


def fill_nan(a, valid, n_iter=12):
    """Fill invalid pixels by repeated normalised 5x5 averaging, then the median.

    The minimum denominator is 0.02 to reject floating-point residue from uniform_filter.
    One valid sample in the 5x5 window gives a denominator of 1/25."""
    a = np.where(valid, a, 0.0).astype(np.float64)
    w = valid.astype(np.float64)
    for _ in range(n_iter):
        if w.all():
            break
        num = ndimage.uniform_filter(a * w, 5)
        den = ndimage.uniform_filter(w, 5)
        new = (w == 0) & (den > 0.02)
        a[new] = num[new] / den[new]
        w[new] = 1.0
    a[w == 0] = np.median(a[valid]) if valid.any() else 0.0
    return a


def richardson_lucy(d, K, n_iter=RL_ITER):
    u = d.copy()
    Kf = K[::-1, ::-1]
    for _ in range(n_iter):
        u *= fftconvolve(d / np.clip(fftconvolve(u, K, mode="same"), 1e-9, None), Kf, mode="same")
    return u


def deconvolve(Q, region, kernels, n_iter=RL_ITER):
    """De-smear each pane of Q (crop frame); NaN in = NaN out (a pane with < 1000 valid pixels is set to NaN).

    Edge-value padding preserves a flat sky to float precision at pane boundaries."""
    out = Q.copy()
    for g in (1, 2, 3):
        m = region == g
        ys, xs = np.nonzero(m)
        y0, y1 = max(ys.min() - KX - 2, 0), min(ys.max() + KX + 3, Q.shape[0])
        x0, x1 = max(xs.min() - KX - 2, 0), min(xs.max() + KX + 3, Q.shape[1])
        sub = Q[y0:y1, x0:x1]
        valid = m[y0:y1, x0:x1] & np.isfinite(sub)
        if valid.sum() < 1000:
            out[m] = np.nan
            continue
        d = fill_nan(sub, valid)
        floor = max(1e-3, 1e-3 * np.median(d[valid]))
        # pad with edge values: fftconvolve's zero boundary (the box meets detector column 0 on the -x side) would
        # otherwise leak into the pane over the iterations (flat sky: up to 1.4 % at the -x edge of pane 3)
        dp = np.pad(np.clip(d, floor, None), PAD, mode="edge")
        u = richardson_lucy(dp, kernels[g], n_iter)[PAD:-PAD, PAD:-PAD]
        blk = out[y0:y1, x0:x1]
        blk[valid] = u[valid]
    return out


def edge_margin(Q, region, n=EDGE_TRIM, cluster=NAN_CLUSTER, round_=True):
    """Finite pane pixels within n px of their pane's outer edge or of a NaN cluster of >= cluster px inside it:
    Euclidean distance <= n to the nearest such pixel (round_; the image margin), or chessboard n x 3x3
    dilation (round_=False; the QA exclusion margin). Isolated
    holes (static bad pixels, which the region map and the bad-pixel mask leave as 1-10 px holes) are filled before
    RL and do not count as edges."""
    out = np.zeros(region.shape, bool)
    if n <= 0:
        return out
    for g in (1, 2, 3):
        pane = region == g
        filled = ndimage.binary_fill_holes(pane)
        hole = filled & ~(pane & np.isfinite(Q))
        lab, _ = ndimage.label(hole)
        size = np.bincount(lab.ravel())
        size[0] = 0
        edge = ~filled | np.isin(lab, np.flatnonzero(size >= cluster))
        near = (ndimage.distance_transform_edt(~edge) <= n) if round_ else \
            ndimage.binary_dilation(edge, np.ones((3, 3), bool), iterations=n)
        out |= near & pane & np.isfinite(Q)
    return out


def scale_variance(var, region, factors=None):
    v = var.copy()
    for g, f in (RL_VAR_FACTOR if factors is None else factors).items():
        v[region == int(g)] *= f
    return v
