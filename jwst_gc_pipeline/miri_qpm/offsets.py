"""Sky offsets and the optional overlap terms.

* Science sky offset (always): median(cal - reference) over the exposure's imager pixels (nearest reference pixel,
  4-px stride, Lyot excluded).  Pane pixels inherit it, exactly like the exposure's science pixels.
* Anchoring (L3g, optional, default off): one additive level per exposure-pane, the median over its >= 8 clean 8 x 8
  overlap blocks (>= 32 good px, reference 30-450 MJy/sr) of (Q_b - MK_b), MK = reference convolved with the pane
  kernel.  A local skymatch of the pane to the reference, not a blind-validated calibration term.
* Dataset gain (Lg, optional, external pipeline only, default off): one gain per pane from all overlap blocks,
  adopted only if >= 5 pointing groups have overlap AND leave-one-group-out cross-validation lowers the RMS of the
  per-bin bias.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

BLK, BLK_MIN, MIN_ANCHOR = 8, 32, 8
ANCHOR_MK = (30.0, 450.0)
MIN_GAIN_GROUPS = 5
BINS = (30, 100, 300, 450, 1000)
SAMPLE = 4


def sky_offset(S, to_ref, cut, cx0, cy0, sample=SAMPLE):
    """median(S - reference) over S's finite pixels on a stride grid (nearest reference pixel).

    to_ref: callable, detector (x, y) arrays -> reference pixel (x, y); cut: reference cutout with origin (cx0, cy0).
    Returns (offset, n); offset = nan if fewer than 1000 overlapping pixels."""
    yy, xx = np.mgrid[0:S.shape[0]:sample, 0:S.shape[1]:sample]
    yy, xx = yy.ravel(), xx.ravel()
    sv = S[yy, xx]
    f = np.isfinite(sv)
    yy, xx, sv = yy[f], xx[f], sv[f]
    ox, oy = to_ref(xx.astype(float), yy.astype(float))
    ix, iy = np.rint(ox).astype(int) - cx0, np.rint(oy).astype(int) - cy0
    inside = (ix >= 0) & (ix < cut.shape[1]) & (iy >= 0) & (iy < cut.shape[0])
    rv = np.full(len(sv), np.nan, np.float32)
    rv[inside] = cut[iy[inside], ix[inside]]
    d = sv - rv
    g = np.isfinite(d)
    return (float(np.median(d[g])) if g.sum() > 1000 else np.nan), int(g.sum())


def convolve_mk(M, region, kernels):
    """Reference sampled at the pane pixels, convolved per pane with the (unit-sum) pane kernel (normalised
    convolution; NaN where the kernel footprint is < 99 % covered)."""
    MK = np.full(M.shape, np.nan)
    for g in (1, 2, 3):
        m = region == g
        w = (m & np.isfinite(M)).astype(np.float64)
        K = kernels[g] / kernels[g].sum()
        num = fftconvolve(np.where(w > 0, M, 0.0), K, mode="same")
        den = fftconvolve(w, K, mode="same")
        ok = m & (w > 0) & (den > 0.99)
        MK[ok] = num[ok] / den[ok]
    return MK


def blocks(V, MK, region, y0, x0, valid=None):
    """8 x 8-block means of V and MK where both are finite (>= 32 px).  Blocks are on the full-frame 8-px grid
    (crop origin y0, x0) and take the pane of their centre pixel, as in the calibration study."""
    ok = (region > 0) & np.isfinite(V) & np.isfinite(MK) & (MK > 0)
    if valid is not None:
        ok &= valid
    yy, xx = np.nonzero(ok)
    if not len(yy):
        z = np.zeros(0)
        return z, z, z.astype(np.int8)
    key = ((yy + y0) // BLK) * 1000 + (xx + x0) // BLK
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    vb = np.bincount(inv, weights=V[yy, xx]) / cnt
    mb = np.bincount(inv, weights=MK[yy, xx]) / cnt
    gb = region[(u // 1000) * BLK + BLK // 2 - y0, (u % 1000) * BLK + BLK // 2 - x0]
    keep = (cnt >= BLK_MIN) & (gb >= 1) & (gb <= 3)
    return vb[keep], mb[keep], gb[keep].astype(np.int8)


def anchor_offsets(Qb, MKb, g, gain=None):
    """{pane: (n_blocks, offset)}; offset = median(Qb / gain - MKb) over blocks with MK in 30-450 MJy/sr, NaN if
    fewer than 8 such blocks."""
    gain = gain or {1: 1.0, 2: 1.0, 3: 1.0}
    out = {}
    for p in (1, 2, 3):
        s = (g == p) & (MKb >= ANCHOR_MK[0]) & (MKb < ANCHOR_MK[1])
        n = int(s.sum())
        out[p] = (n, float(np.median(Qb[s] / gain[p] - MKb[s])) if n >= MIN_ANCHOR else np.nan)
    return out


def groups_of(stems):
    """Pointing groups: visits if >= 5 distinct, else visit + activity."""
    vis = np.array([s[:13] for s in stems])
    if len(set(vis)) >= 5:
        return vis, "visit"
    return np.array([s[:19] for s in stems]), "visit+activity"


def bin_bias(Qb, MKb, g):
    r = Qb / MKb - 1
    out = {}
    for p in (1, 2, 3):
        row = {}
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            s = (g == p) & (MKb >= lo) & (MKb < hi)
            if s.sum() >= 20:
                row[f"{lo}-{hi}"] = dict(n=int(s.sum()), bias=float(np.median(r[s])),
                                         abias=float(np.median((Qb - MKb)[s])))
        out[p] = row
    return out


def cell_rms(Qb, MKb, g):
    bb = bin_bias(Qb, MKb, g)
    v = [x["bias"] for p in bb.values() for k, x in p.items() if not k.startswith("450")]
    return float(np.sqrt(np.mean(np.square(v)))) if v else np.nan


def dataset_gain(Qb, MKb, g, grp):
    """Lg with leave-one-group-out CV.  Qb, MKb, g: pooled blocks; grp: pointing group of each block.
    Returns ({pane: gain}, report); gains are 1 unless adopted."""
    one = {p: 1.0 for p in (1, 2, 3)}
    sel = (MKb >= ANCHOR_MK[0]) & (MKb < ANCHOR_MK[1])
    Qb, MKb, g, gb = Qb[sel], MKb[sel], g[sel], np.asarray(grp)[sel]
    ug = np.unique(gb)
    gain = {p: float(np.median(Qb[g == p] / MKb[g == p])) if (g == p).sum() >= 30 else 1.0 for p in (1, 2, 3)}
    rep = dict(n_groups_overlap=int(len(ug)), gain_all=gain, adopted=False)
    if len(ug) < MIN_GAIN_GROUPS:
        rep["reason"] = f"{len(ug)} pointing groups with overlap < {MIN_GAIN_GROUPS}"
        return one, rep
    Qh = np.full(len(Qb), np.nan)
    for u in ug:
        te = gb == u
        for p in (1, 2, 3):
            tr = (~te) & (g == p)
            if tr.sum() >= 30:
                s = te & (g == p)
                Qh[s] = Qb[s] / np.median(Qb[tr] / MKb[tr])
    ok = np.isfinite(Qh)
    r0, r1 = cell_rms(Qb[ok], MKb[ok], g[ok]), cell_rms(Qh[ok], MKb[ok], g[ok])
    adopt = bool(np.isfinite(r1) and r1 < r0)
    rep.update(cv_cell_rms_L0=r0, cv_cell_rms_Lg=r1, adopted=adopt,
               reason="LOGO CV improves" if adopt else "LOGO CV does not improve")
    return (gain if adopt else one), rep
