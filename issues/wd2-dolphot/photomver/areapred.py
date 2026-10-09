"""Pixel-area prediction for every matched star of an arm: pred = 2.5 log10(mean over exposures of PIXAR_SR * AREA(x, y) /
proj_plane_pixel_area), the amount by which a constant-pixel-area calibration reads faint (see areatest.py).
Saves areapred_<arm>.npz (one array per band, length of the matched table) and prints robust std / median of dm and
dm - pred by dolphot magnitude bin, for unsaturated and for satstar-replaced rows.
usage: python areapred.py [arm] > areapred_<arm>.txt"""
import os
import sys
import glob
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

an.ZPWIN.update(an.zp_windows())
arm = sys.argv[1] if len(sys.argv) > 1 else 'main2'
A = an.Arm(arm)
tree = f'{an.Q}/tree_main2'
out = f'{an.Q}/photomver/areapred_{arm}.npz'
sky = A.sky[A.idx]
P = dict(np.load(out)) if os.path.exists(out) else {}
for band in an.BANDS:
    if band in P:
        continue
    frames = sorted(glob.glob(f'{tree}/F{band}/pipeline/jw03523005001_*_align_o005_crf.fits'))
    fsum = np.zeros(A.n)
    nexp = np.zeros(A.n)
    for fn in frames:
        with fits.open(fn, memmap=True) as fh:
            h = fh['SCI'].header
            area = np.asarray(fh['AREA'].data, float)
        w = WCS(h)
        ny, nx = area.shape
        x, y = w.world_to_pixel(sky)
        inn = A.matched & (x > 0) & (x < nx - 1) & (y > 0) & (y < ny - 1)
        a = np.full(A.n, np.nan)
        a[inn] = area[np.round(y[inn]).astype(int), np.round(x[inn]).astype(int)] * h['PIXAR_SR'] / w.proj_plane_pixel_area().to(u.sr).value
        good = np.isfinite(a) & (a > 0)
        fsum[good] += a[good]
        nexp[good] += 1
    pred = np.full(A.n, np.nan)
    pred[nexp > 0] = 2.5 * np.log10(fsum[nexp > 0] / nexp[nexp > 0])
    P[band] = pred
    np.savez(out, **P)


def rs(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


BINS = [(11, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23)]
for band in an.BANDS:
    dm = A.dm(band)
    pred = P[band]
    base = A.matched & np.isfinite(dm) & np.isfinite(pred) & np.isfinite(A.ref[band])
    print(f'\n### F{band}: robust std of dm -> dm - pred [N] (median dm -> median dm - pred), by dolphot bin')
    print('| bin | unsaturated | satstar-replaced |')
    print('|---|---|---|')
    for lo, hi in BINS:
        cells = []
        for s in (base & ~A.rep[band] & ~A.sat[band], base & A.rep[band]):
            q = s & (A.ref[band] >= lo) & (A.ref[band] < hi)
            cells.append(f'{rs(dm[q]):.4f} -> {rs((dm - pred)[q]):.4f} [{q.sum()}] ({np.median(dm[q]):+.3f} -> {np.median((dm - pred)[q]):+.3f})'
                         if q.sum() >= 15 else '')
        if any(cells):
            print(f'| {lo}-{hi} | ' + ' | '.join(cells) + ' |')
