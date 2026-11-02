"""Shared arithmetic of the satstar wing self-calibration (#1041).

A satstar's catalog flux is divided by C(r_mask), the ratio of a masked-core
fit to an unmasked fit, measured on bright unsaturated stars in buckets of
integer mask radius (3..30 px).  Two places apply it: the per-frame
calibration (``saturated_star_finding.apply_wing_selfcal``) and the pooled
cross-frame fallback (``merge_catalogs.apply_pooled_wingcal``).  Both used
to apply every bucket, however noisy, and clip a star's r_mask to the
smallest bucket, so a star with one masked pixel (r_mask = 0.564) received
C(3 px).  Measured across /orange: pooled buckets reached C = 12-20 at large
r_mask (n = 2-30 stars) and shifted catalog stars by up to 3 mag; on wd2 the
1-px-core rows moved 0.03-0.07 mag away from dolphot.

This module holds what both callers share: the bucket standard error with
its scatter floor, the precision gate and the interpolation anchored at
C(0) = 1.  It is
import-light so that both ``saturated_star_finding`` and ``merge_catalogs``
can use it.
"""
import os

import numpy as np

# Standard error of the median of n Gaussian draws: sqrt(pi/2) sigma/sqrt(n).
MEDIAN_SE_FACTOR = float(np.sqrt(np.pi / 2.0))
DEFAULT_MAX_SE = 0.05
# Fewest stars whose scatter may set the floor.
FLOOR_MIN_STARS = 5
# Smallest bucket ratio that is applied.  Measured buckets sit at 0.93-1.5
# where they are precise; a median near or below zero means the masked
# fits are noise, and dividing a flux by it is meaningless.
MIN_RATIO = 0.5


def wingcal_max_se():
    """Largest bucket standard error that is applied.

    Env ``SATSTAR_WINGCAL_MAX_SE`` (default 0.05, i.e. 5% in flux); a value
    <= 0 disables the gate.
    """
    return float(os.environ.get('SATSTAR_WINGCAL_MAX_SE', DEFAULT_MAX_SE))


def relative_scatter_floor(rs, ratios, madstds, n_stars,
                           min_stars=FLOOR_MIN_STARS):
    """Fractional scatter ``madstd / |ratio|`` at the smallest r_mask, or 0.

    A bucket's ``madstd`` comes from as few as 2 stars and can be near zero
    by chance, which gives it a tiny standard error.  The fractional scatter
    grows with r_mask (median over frames 0.04-0.13 at 3 px and 0.3-1.1
    beyond 10 px on ngc6334, wd1, brick, cloudef and wd2), so the value at
    the smallest bucket is a lower bound for every bucket.  On ngc6334
    F444W a one-frame, 3-star r=18 bucket at C = 3.2 had a fractional
    scatter of 0.005 and se = 0.012; floored at 3 px's 0.13 it has se = 0.3.

    ``rs``, ``ratios``, ``madstds`` and ``n_stars`` describe bucket
    measurements (one per bucket for a frame, one per frame and bucket for
    the pool).  Only measurements from >= ``min_stars`` stars count; the
    floor is their median at the smallest such r_mask.
    """
    r = np.asarray(rs, dtype=float)
    v = np.asarray(ratios, dtype=float)
    s = np.asarray(madstds, dtype=float)
    n = np.asarray(n_stars, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = s / np.abs(v)
    ok = np.isfinite(r) & np.isfinite(rel) & (rel > 0) & (n >= min_stars)
    if not ok.any():
        return 0.0
    return float(np.median(rel[ok & (r == r[ok].min())]))


def bucket_se(madstd, n, ratio=None, rel_floor=0.0):
    """Standard error of a bucket median from its per-star ``madstd`` and
    count ``n``.  With ``ratio`` given, ``madstd`` is floored at
    ``rel_floor * |ratio|`` (see :func:`relative_scatter_floor`).  NaN when
    the inputs are unusable."""
    madstd = np.asarray(madstd, dtype=float)
    n = np.asarray(n, dtype=float)
    floor = (0.0 if ratio is None
             else rel_floor * np.abs(np.asarray(ratio, dtype=float)))
    with np.errstate(divide='ignore', invalid='ignore'):
        se = MEDIAN_SE_FACTOR * np.maximum(madstd, floor) / np.sqrt(n)
    return np.where((n > 0) & np.isfinite(madstd), se, np.nan)


def passes_se_gate(se, max_se=None, ratio=None):
    """True where a bucket's standard error is precise enough to apply.  A
    NaN standard error fails unless the gate is disabled.  With ``ratio``
    given, a bucket below :data:`MIN_RATIO` also fails (a noise-dominated
    median can sit near zero with a small standard error)."""
    if max_se is None:
        max_se = wingcal_max_se()
    se = np.asarray(se, dtype=float)
    if max_se <= 0:
        return np.ones(se.shape, dtype=bool)
    ok = np.isfinite(se) & (se <= max_se)
    if ratio is not None:
        ok &= np.asarray(ratio, dtype=float) >= MIN_RATIO
    return ok


def pool_bucket(ratios, n_stars, madstds, rel_floor=0.0):
    """Combine one r_mask bucket's per-frame medians into a pooled ratio.

    Inverse-variance mean, each frame's variance ``(1.2533 madstd)^2 / n``.
    Each frame's ``madstd`` is floored at the larger of the bucket's median
    ``madstd`` across frames and ``rel_floor`` times that frame's ratio
    (the band's :func:`relative_scatter_floor`).  The standard error is
    inflated by sqrt(chi2_red) when the frames disagree by more than their
    errors.  With no usable ``madstd`` the n-weighted mean is returned with
    a NaN standard error, which the gate rejects.

    Returns ``(ratio, ratio_se)``.
    """
    v = np.asarray(ratios, dtype=float)
    n = np.asarray(n_stars, dtype=float)
    s = np.asarray(madstds, dtype=float)
    ok = np.isfinite(v) & (n > 0)
    v, n, s = v[ok], n[ok], s[ok]
    if v.size == 0:
        return np.nan, np.nan
    usable = np.isfinite(s) & (s > 0)
    if not usable.any():
        return float(np.average(v, weights=n)), np.nan
    lo = np.maximum(float(np.median(s[usable])), rel_floor * np.abs(v))
    s_eff = np.where(usable & (s >= lo), s, lo)
    w = n / (MEDIAN_SE_FACTOR * s_eff) ** 2
    mean = float(np.sum(w * v) / np.sum(w))
    se = float(1.0 / np.sqrt(np.sum(w)))
    if v.size > 1:
        chi2_red = float(np.sum(w * (v - mean) ** 2) / (v.size - 1))
        se *= np.sqrt(max(1.0, chi2_red))
    return mean, se


def interp_wingcal_ratio(rmask, rs, vs):
    """C(r_mask) interpolated from bucket radii ``rs`` and ratios ``vs``.

    Anchored at C(0) = 1: an unmasked fit has no wing-fit bias, and a star
    whose fit masked one pixel (r_mask = sqrt(1/pi) = 0.564) is nearly
    unmasked.  Between 0 and the smallest bucket the ratio is linear; beyond
    the largest bucket it is held at the largest bucket's value.  Rows with
    a non-finite r_mask get 1.  Buckets with a ratio <= 0 are ignored even
    when the gate is disabled, since the flux is divided by C.
    """
    rs = np.asarray(rs, dtype=float)
    vs = np.asarray(vs, dtype=float)
    keep = np.isfinite(rs) & np.isfinite(vs) & (rs > 0) & (vs > 0)
    rs, vs = rs[keep], vs[keep]
    r = np.asarray(rmask, dtype=float)
    if rs.size == 0:
        return np.ones(r.shape)
    order = np.argsort(rs)
    xp = np.concatenate([[0.0], rs[order]])
    fp = np.concatenate([[1.0], vs[order]])
    with np.errstate(invalid='ignore'):
        out = np.interp(np.clip(r, 0.0, xp[-1]), xp, fp)
    return np.where(np.isfinite(r), out, 1.0)
