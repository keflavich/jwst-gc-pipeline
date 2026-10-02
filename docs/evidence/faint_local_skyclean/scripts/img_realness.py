"""Image-based realness of the #1019 additions: does the independent-visit
reference IMAGE show a peak at the source, independent of the reference
catalog's own vetting?

At each source position (reference-image pixels): peak = a local maximum of
the lightly smoothed reference image (Gaussian sigma SMOOTH_PX, 3x3 maximum
filter) within R_PEAK px.  Chance = the same test at control positions: the
source's offset from its nearest brighter base-kept star rotated by 60, 120,
180, 240 and 300 degrees about that star (same distance from the same star,
and PSF features of 6-fold symmetry map onto themselves), or for a source with
no brighter kept star within 20 px, shifts of 2" in four directions.
Image realness = (peak - chance) / (1 - chance), per set; reported for the
additions and for base-kept stars of the same per-frame S/N range and the same
distance to the nearest brighter kept star.

usage: python img_realness.py <field> [variant=lsky]
"""
import json
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare import FIELDS, sky  # noqa: E402
from added_gallery import CFG  # noqa: E402

SMOOTH_PX = 0.7
R_PEAK = 1.5
SNR_BINS = ((0, 5), (5, 10), (10, np.inf))
DIST_EDGES = [0, 3, 5, 8, 12, 20, np.inf]
ROT_DEG = (60, 120, 180, 240, 300)
SHIFT_AS = 2.0
MAX_BASE = 3000


def main(field, v='lsky'):
    cfg = FIELDS[field]
    band = cfg['band']
    base = Table.read(f'{HERE}/out/{field}_{band}_seed.fits')
    var = Table.read(f'{HERE}/out/{field}_{band}_{v}.fits')
    assert np.array_equal(np.asarray(var['rowid']), np.asarray(base['rowid']))
    kb = np.asarray(base['kept'], bool)
    add = np.asarray(var['kept'], bool) & ~kb
    sc = sky(base)
    flux = np.asarray(base['flux'], float)
    snr = flux / np.asarray(base['flux_err'], float)

    with fits.open(CFG[field]['refimg']) as fh:
        hdu = fh['SCI'] if 'SCI' in [h.name for h in fh] else fh[0]
        img = np.asarray(hdu.data, np.float32)
        w = WCS(hdu.header)
    sm = gaussian_filter(np.nan_to_num(img, nan=-1e30), SMOOTH_PX)
    ismax = (sm == maximum_filter(sm, size=3)) & np.isfinite(img)
    del sm
    ny, nx = img.shape
    pixas = float(np.sqrt(np.abs(np.linalg.det(w.pixel_scale_matrix))) * 3600)

    x, y = w.world_to_pixel(sc)
    x, y = np.asarray(x, float), np.asarray(y, float)
    # nearest brighter kept star in the reference-image pixel grid
    ik = np.flatnonzero(kb)
    good = np.isfinite(x) & np.isfinite(y)
    tree = cKDTree(np.c_[x[ik], y[ik]])
    want = np.flatnonzero((add | kb) & good)
    dd, jj = tree.query(np.c_[x[want], y[want]], k=16, distance_upper_bound=20 * 0.0311 / pixas)
    d_b = np.full(len(base), np.inf)
    j_b = np.full(len(base), -1)
    for n, i in enumerate(want):
        for d, j in zip(dd[n], jj[n]):
            if not np.isfinite(d):
                break
            if ik[j] != i and flux[ik[j]] > flux[i]:
                d_b[i], j_b[i] = d * pixas / 0.0311, ik[j]     # in 0.0311" px, as lsky_snr_diag
                break

    oy, ox = np.mgrid[-2:3, -2:3]

    def peak_at(px, py):
        out = np.full(px.shape, np.nan)
        cx, cy = np.rint(px).astype(int), np.rint(py).astype(int)
        ok = (cx >= 2) & (cx < nx - 2) & (cy >= 2) & (cy < ny - 2)
        for m in np.flatnonzero(ok):
            box = img[cy[m] - 2:cy[m] + 3, cx[m] - 2:cx[m] + 3]
            if not np.isfinite(box).all():
                continue
            # distance from the true (sub-pixel) position to each box pixel
            dx = (cx[m] + ox) - px[m]
            dy = (cy[m] + oy) - py[m]
            near = (dx ** 2 + dy ** 2) <= R_PEAK ** 2
            out[m] = float(np.any(ismax[cy[m] - 2:cy[m] + 3, cx[m] - 2:cx[m] + 3] & near))
        return out

    def controls(rows):
        cxs, cys = [], []
        has = j_b[rows] >= 0
        for a in np.deg2rad(ROT_DEG):
            ca, sa = np.cos(a), np.sin(a)
            px = np.where(has, 0.0, np.nan)
            py = np.where(has, 0.0, np.nan)
            jr = j_b[rows[has]]
            dx, dy = x[rows[has]] - x[jr], y[rows[has]] - y[jr]
            px[has] = x[jr] + ca * dx - sa * dy
            py[has] = y[jr] + sa * dx + ca * dy
            cxs.append(px)
            cys.append(py)
        sh = SHIFT_AS / pixas
        for k, (sx, sy) in enumerate(((sh, 0), (-sh, 0), (0, sh), (0, -sh))):
            if k < len(cxs):
                cxs[k] = np.where(has, cxs[k], x[rows] + sx)
                cys[k] = np.where(has, cys[k], y[rows] + sy)
        return cxs, cys

    def score(rows):
        if rows.size == 0:
            return dict(n=0)
        p = peak_at(x[rows], y[rows])
        cx, cy = controls(rows)
        ch = np.nanmean(np.vstack([peak_at(a, b) for a, b in zip(cx, cy)]), axis=0)
        ok = np.isfinite(p) & np.isfinite(ch)
        if ok.sum() < 10:
            return dict(n=int(ok.sum()))
        pk, chance = float(np.mean(p[ok])), float(np.mean(ch[ok]))
        return dict(n=int(ok.sum()), peak=pk, chance=chance,
                    img_rel=(pk - chance) / (1 - chance) if chance < 1 else None)

    rng = np.random.default_rng(0)
    res = {}
    print(f'== {field} {v}: image realness against {os.path.basename(CFG[field]["refimg"])} '
          f'(local max within {R_PEAK} px, smoothing {SMOOTH_PX} px)')
    for lo, hi in SNR_BINS:
        rows_out = []
        print(f'-- S/N [{lo:g},{hi:g}) by distance (px) to nearest brighter kept star: '
              f'added n peak chance img_rel | base-kept')
        for a, b in zip(DIST_EDGES[:-1], DIST_EDGES[1:]):
            sel = good & (snr >= lo) & (snr < hi) & (d_b >= a) & ((d_b < b) | ~np.isfinite(b))
            ra = score(np.flatnonzero(sel & add))
            kc = np.flatnonzero(sel & kb)
            if kc.size > MAX_BASE:
                kc = rng.choice(kc, MAX_BASE, replace=False)
            rc = score(kc)
            rows_out.append(dict(d_lo=a, d_hi=b, added=ra, basekept=rc))

            def f(r):
                if 'peak' not in r:
                    return f'n {r["n"]:5d}' + ' ' * 23
                return f'n {r["n"]:5d} {r["peak"]:.2f} {r["chance"]:.2f} {r["img_rel"]:5.2f}'
            print(f'   [{a:g},{b:g}) added {f(ra)} | base-kept {f(rc)}', flush=True)
        res[f'{lo:g}-{hi:g}'] = rows_out
    with open(f'{HERE}/img_realness_{field}.json', 'w') as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == '__main__':
    main(*sys.argv[1:])
