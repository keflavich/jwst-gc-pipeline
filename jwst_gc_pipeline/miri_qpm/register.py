"""Optional registration of a frame to the reference mosaic by star voting (as fullframe_preview.register), and
pane-star offsets for the geometry check (as fullframe_preview.fit_regions).

Imager stars are matched to the mosaic's stars: offset voting within 25, then 80, then 300 px, a similarity fit
with 3-sigma clipping, accepted only if matches >= max(50, 0.3 x frame stars), rms < 0.8 px and |scale - 1| < 1e-3
(a crowded field can elect a wrong offset with a few dozen chance matches).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .geometry import apply_sim

SEARCH_PX = (25, 80, 300)
MIN_MATCH, MIN_MATCH_FRAC, MAX_RMS = 50, 0.3, 0.8


def peaks(img, nmax=400, box=15, nsig=6.0):
    """Local maxima of the high-passed image with an all-finite 5x5 core (sub-pixel by parabola)."""
    fill = np.nanmedian(img)
    d = img - ndimage.uniform_filter(np.where(np.isfinite(img), img, fill), box)
    ok = np.isfinite(d)
    if ok.sum() < 100:
        return np.zeros(0), np.zeros(0)
    d0 = np.where(ok, d, -np.inf)
    v = d[ok]
    mad = 1.4826 * np.median(np.abs(v - np.median(v)))
    fin = ndimage.minimum_filter(ok.astype(np.uint8), 5) > 0
    fin[:3] = fin[-3:] = False
    fin[:, :3] = fin[:, -3:] = False
    m = (ndimage.maximum_filter(d0, 5) == d0) & (d0 > nsig * mad) & fin
    y, x = np.nonzero(m)
    o = np.argsort(-d0[y, x])[:nmax]
    y, x = y[o], x[o]

    def q(a, b, c):
        den = a - 2 * b + c
        with np.errstate(all="ignore"):
            return np.where(den < 0, 0.5 * (a - c) / den, 0.0)
    dx = q(d0[y, x - 1], d0[y, x], d0[y, x + 1])
    dy = q(d0[y - 1, x], d0[y, x], d0[y + 1, x])
    return x + np.clip(dx, -1, 1), y + np.clip(dy, -1, 1)


def vote(fx, fy, tree, rx, ry, search):
    """Most common (ref - frame) offset among all pairs within ``search`` px."""
    dd = [(rx[j] - fx[i], ry[j] - fy[i])
          for i, js in enumerate(tree.query_ball_point(np.c_[fx, fy], search)) for j in js]
    if len(dd) < 5:
        return None
    dd = np.array(dd)
    edges = np.arange(-search, search + 0.5, 0.5)
    H, xe, ye = np.histogram2d(dd[:, 0], dd[:, 1], bins=edges)
    H = ndimage.uniform_filter(H, 3)
    i, j = np.unravel_index(np.argmax(H), H.shape)
    return xe[i] + 0.25, ye[j] + 0.25


def similarity(x, y, X, Y, c):
    """Fit X = a(x-cx) - b(y-cy) + cx + tx, Y = b(x-cx) + a(y-cy) + cy + ty."""
    u, v = x - c[0], y - c[1]
    A = np.zeros((2 * len(x), 4))
    A[0::2] = np.c_[u, -v, np.ones_like(u), np.zeros_like(u)]
    A[1::2] = np.c_[v, u, np.zeros_like(u), np.ones_like(u)]
    rhs = np.empty(2 * len(x))
    rhs[0::2] = X - c[0]
    rhs[1::2] = Y - c[1]
    return np.linalg.lstsq(A, rhs, rcond=None)[0]          # a, b, tx, ty


def ref_stars(cut, cx0, cy0):
    rx, ry = peaks(cut, nmax=4000)
    return rx + cx0, ry + cy0


def register_frame(S, wcs, rw, rx, ry, c, search_px=SEARCH_PX, min_match=MIN_MATCH,
                   min_match_frac=MIN_MATCH_FRAC, max_rms=MAX_RMS):
    """Similarity (p) of the frame's imager stars S onto the reference stars (rx, ry) about c.
    Returns dict(p, nmatch, rms, ok)."""
    out = dict(p=np.array([1.0, 0, 0, 0]), nmatch=0, rms=np.nan, ok=False, nstars=0)
    if len(rx) < 20:
        return out
    tree = cKDTree(np.c_[rx, ry])
    sx, sy = peaks(S)
    out["nstars"] = int(len(sx))
    if len(sx) < 10:
        return out
    ox, oy = rw.world_to_pixel_values(*wcs(sx, sy, with_bounding_box=False))
    for search in search_px:
        v = vote(ox, oy, tree, rx, ry, search)
        if v is None:
            continue
        d, j = tree.query(np.c_[ox + v[0], oy + v[1]], distance_upper_bound=2.0)
        ok = np.isfinite(d)
        if ok.sum() < 10:
            continue
        xa, ya, Xa, Ya = ox[ok], oy[ok], rx[j[ok]], ry[j[ok]]
        keep = np.ones(len(xa), bool)
        for _ in range(4):
            p = similarity(xa[keep], ya[keep], Xa[keep], Ya[keep], c)
            mx, my = apply_sim(p, c, xa, ya)
            r = np.hypot(Xa - mx, Ya - my)
            sig = max(0.15, 1.4826 * np.median(r[keep]))
            keep = r < 3 * sig
        rms = float(np.sqrt(np.mean(r[keep] ** 2)))
        out.update(nmatch=int(keep.sum()), rms=rms)
        need = max(min_match, min_match_frac * len(sx))
        if keep.sum() >= need and rms < max_rms and abs(np.hypot(p[0], p[1]) - 1) < 1e-3:
            out.update(p=p, ok=True)
            break
    return out


def pane_candidates(Qrl, ref, wcs, rw, p, c, rx, ry, nsig=8.0):
    """(region, x, y, dx, dy) candidate offsets [detector px] of pane stars (de-smeared panes, crop frame) to the
    reference stars within 10 ref px; the static pane model is NOT applied here (fit_regions measures it)."""
    out = []
    qx, qy = peaks(Qrl, nsig=nsig)
    if not len(qx) or len(rx) < 2:
        return out
    tree = cKDTree(np.c_[rx, ry])
    X, Y = qx + ref.x0, qy + ref.y0
    T0 = np.array(apply_sim(p, c, *rw.world_to_pixel_values(*wcs(X, Y, with_bounding_box=False))))
    T1 = np.array(apply_sim(p, c, *rw.world_to_pixel_values(*wcs(X + 1, Y, with_bounding_box=False)))) - T0
    T2 = np.array(apply_sim(p, c, *rw.world_to_pixel_values(*wcs(X, Y + 1, with_bounding_box=False)))) - T0
    rr = ref.region[np.rint(qy).astype(int), np.rint(qx).astype(int)]
    for n, js in enumerate(tree.query_ball_point(T0.T, 10.0)):
        if rr[n] == 0 or not js:
            continue
        Ji = np.linalg.inv(np.array([[T1[0, n], T2[0, n]], [T1[1, n], T2[1, n]]]))
        for jj in js:
            dd = Ji @ np.array([rx[jj] - T0[0, n], ry[jj] - T0[1, n]])
            out.append((int(rr[n]), X[n], Y[n], dd[0], dd[1]))
    return out


def fit_regions(cands):
    """Per pane, affine (x, y) -> detector offset from candidate rows (frame, region, x, y, dx, dy)."""
    models, stats = {}, {}
    for r in (1, 2, 3):
        c = cands[cands[:, 1] == r]
        if len(c) < 30:
            continue
        H, xe, ye = np.histogram2d(c[:, 4], c[:, 5], bins=np.arange(-12, 12.25, 0.25))
        H = ndimage.uniform_filter(H, 3)
        a, b = np.unravel_index(np.argmax(H), H.shape)
        pred = np.tile([xe[a] + 0.125, ye[b] + 0.125], (len(c), 1))
        A = np.c_[np.ones(len(c)), (c[:, 2] - 500) / 500, (c[:, 3] - 500) / 500]
        for tol in (1.5, 1.0, 0.7, 0.7, 0.7):
            r_ = np.hypot(c[:, 4] - pred[:, 0], c[:, 5] - pred[:, 1])
            key = np.unique(c[:, [0, 2, 3]], axis=0, return_inverse=True)[1].ravel()
            order = np.lexsort((r_, key))
            first = np.r_[True, key[order][1:] != key[order][:-1]]
            sel = np.zeros(len(c), bool)
            sel[order[first]] = True
            sel &= r_ < tol
            if sel.sum() < 10:
                break
            cx = np.linalg.lstsq(A[sel], c[sel, 4], rcond=None)[0]
            cy = np.linalg.lstsq(A[sel], c[sel, 5], rcond=None)[0]
            pred = np.c_[A @ cx, A @ cy]
        if sel.sum() < 10:
            continue
        res = c[sel, 4:6] - pred[sel]
        models[r] = (cx, cy)
        stats[r] = dict(n=int(sel.sum()), frames=int(len(np.unique(c[sel, 0]))),
                        rms_after=np.sqrt(np.mean(res ** 2, 0)).tolist())
    return models, stats
