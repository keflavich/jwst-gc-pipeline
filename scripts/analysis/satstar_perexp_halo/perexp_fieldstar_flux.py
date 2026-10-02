"""Do UNSATURATED field stars change flux with detector x the way the saturated
halos do (#1013 follow-up)?

    python perexp_fieldstar_flux.py <out prefix> visit_dir_glob...

Arguments after the prefix are globs of the ``_cal`` frames of one star-visit
each (e.g. ``data/jw10678061001_02101_*_nrcblong_cal.fits``).  Bright,
unsaturated, isolated stars are found in the first dither and carried to the
others through the GWCS (``frame_wcs``; no catalog matching), with a
centroid refinement.  As the dither moves a star by +-385 px in detector x,
its aperture flux (r = 3 px, annulus 6-9 px) divided by its median over the
dithers is binned by detector x.  A throughput error (flat field) shows here;
a change confined to saturated stars does not.
"""
import glob
import json
import os
import sys

import numpy as np
from astropy.io import fits
from scipy import ndimage
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats, aperture_photometry
from photutils.centroids import centroid_sources, centroid_com
from photutils.detection import DAOStarFinder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from jwst_gc_pipeline.frame_wcs import frame_wcs

EDGES = np.arange(0, 2049, 128)


def read(fn):
    with fits.open(fn) as h:
        d = h['SCI'].data.astype(float)
        dq = h['DQ'].data.astype(np.int64)
    bad = ((dq & 3) > 0) | ~np.isfinite(d)
    return d, bad, (dq & 2) > 0, frame_wcs(fn)


def phot(d, bad, xy):
    # isolated DO_NOT_USE pixels (~5% of the frame) are filled from their 3x3
    # neighbourhood; an aperture with more than 3 of them, or one within 1 px of
    # the centre, is dropped
    fill = ndimage.median_filter(np.where(bad, np.nanmedian(d[~bad]), d), size=3)
    dd = np.where(bad, fill, d)
    ap, an = CircularAperture(xy, 3.0), CircularAnnulus(xy, 6.0, 9.0)
    bkg = np.asarray(ApertureStats(dd, an).median)
    f = np.asarray(aperture_photometry(dd, ap)['aperture_sum']) - bkg * ap.area
    nbad = np.asarray(aperture_photometry(bad.astype(float), ap)['aperture_sum'])
    ncen = np.asarray(aperture_photometry(bad.astype(float), CircularAperture(xy, 1.5))['aperture_sum'])
    f[(nbad > 3.5) | (ncen > 0)] = np.nan
    return f


def visit(fns):
    d0, bad0, sat0, w0 = read(fns[0])
    # saturated cores (and their bright wings), not the scattered DO_NOT_USE pixels
    sat = ndimage.binary_dilation(sat0, iterations=10)
    bk = np.nanmedian(d0[~bad0]); sd = 1.4826 * np.nanmedian(np.abs(d0[~bad0] - bk))
    cat = DAOStarFinder(threshold=20 * sd, fwhm=2.3, exclude_border=True)(np.nan_to_num(d0 - bk), mask=bad0)
    xk = 'xcentroid' if 'xcentroid' in cat.colnames else 'x_centroid'
    yk = 'ycentroid' if 'ycentroid' in cat.colnames else 'y_centroid'
    x, y, pk = np.asarray(cat[xk]), np.asarray(cat[yk]), np.asarray(cat['peak'])
    ix, iy = np.clip(x.round().astype(int), 0, 2047), np.clip(y.round().astype(int), 0, 2047)
    ok = ~sat[iy, ix] & (x > 20) & (x < 2028) & (y > 20) & (y < 2028) & (pk < 0.3 * np.nanmax(pk))
    # isolation: no detection brighter than 10% of the star within 10 px
    tree_ok = np.ones(len(x), bool)
    order = np.argsort(x)
    for i in np.flatnonzero(ok):
        j = order[np.searchsorted(x[order], x[i] - 10):np.searchsorted(x[order], x[i] + 10)]
        near = (np.hypot(x[j] - x[i], y[j] - y[i]) < 10) & (j != i) & (pk[j] > 0.1 * pk[i])
        tree_ok[i] = not near.any()
    ok &= tree_ok
    sky = w0.pixel_to_world(x[ok], y[ok])
    F = np.full((len(fns), ok.sum()), np.nan); X = np.full_like(F, np.nan)
    for e, fn in enumerate(fns):
        d, bad, _, w = (d0, bad0, sat0, w0) if e == 0 else read(fn)
        px, py = w.world_to_pixel(sky)
        inn = np.isfinite(px) & (px > 12) & (px < 2035) & (py > 12) & (py < 2035)
        if not inn.any():
            continue
        cx, cy = centroid_sources(np.nan_to_num(np.where(bad, 0, d) - bk), px[inn], py[inn], box_size=7,
                                  centroid_func=centroid_com)
        xy = np.array([cx, cy]).T
        f = np.full(ok.sum(), np.nan)
        f[inn] = phot(d, bad, xy)
        F[e] = f; X[e, inn] = cx
    return F, X


def main():
    outp = sys.argv[1]
    allx, allr = [], []
    for g in sys.argv[2:]:
        fns = sorted(glob.glob(g))
        if len(fns) < 4:
            continue
        F, X = visit(fns)
        good = (np.isfinite(F).sum(0) >= 4) & (np.nanmedian(F, 0) > 0)
        R = F[:, good] / np.nanmedian(F[:, good], 0)
        allx.append(X[:, good].ravel()); allr.append(R.ravel())
        print(os.path.basename(fns[0])[:26], 'stars', good.sum(), flush=True)
    x = np.concatenate(allx); r = np.concatenate(allr)
    m = np.isfinite(x) & np.isfinite(r) & (r > 0.5) & (r < 2)
    k = np.digitize(x[m], EDGES) - 1
    med, err, n = [], [], []
    for i in range(len(EDGES) - 1):
        v = r[m][k == i]
        n.append(int(v.size))
        med.append(float(np.median(v)) if v.size >= 30 else np.nan)
        err.append(float(1.2533 * 1.4826 * np.median(np.abs(v - np.median(v))) / np.sqrt(v.size)) if v.size >= 30 else np.nan)
    json.dump(dict(edges=EDGES.tolist(), median_ratio=med, err=err, n=n), open(outp + '.json', 'w'), indent=1)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar(0.5 * (EDGES[1:] + EDGES[:-1]), med, err, fmt='o-', label=f'unsaturated field stars ({m.sum()} measurements)')
    ax.axhline(1, color='k', lw=0.5); ax.set_ylim(0.9, 1.1)
    ax.set_xlabel('NRCBLONG detector x [px]'); ax.set_ylabel('flux / the star\'s median over the dithers')
    ax.legend(fontsize=8); ax.set_title('field-star aperture flux vs detector x (F480M, 10678)')
    fig.tight_layout(); fig.savefig(outp + '.png', dpi=int(os.environ.get('FIG_DPI', 130)))
    print(' '.join(f'{v:.3f}' for v in med)); print(n)


if __name__ == '__main__':
    main()
