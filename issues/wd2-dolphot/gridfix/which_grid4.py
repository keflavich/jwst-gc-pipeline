"""Does the F150W main2 / old-grid forced-fit ratio depend on masked pixels in the fit box?

For each closure frame row (src == 1), count non-finite or DQ-flagged (DO_NOT_USE or
SATURATED) pixels of the crf in the 5x5 box at the rounded position, then compare the
per-frame slope of do = -2.5 log10(flux_fit_main2 / fold5) against lA for stars with
and without masked pixels.
"""
import glob
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import cKDTree

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an  # noqa: E402

D = f'{an.Q}/gridfix'
DNU, SAT = 1, 2


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def slope(lA, d):
    k = np.isfinite(d) & np.isfinite(lA)
    if k.sum() < 15:
        return np.nan
    lA, d = lA[k], d[k]
    g = np.abs(d - np.median(d)) <= max(5 * mad(d), 1e-6)
    return np.polyfit(lA[g], d[g], 1)[0]


for b in sys.argv[1:]:
    F = Table.read(f'{D}/frames_{b}.ecsv')
    F = F[F['src'] == 1]
    cats = {}
    for fn in glob.glob(f'{an.Q}/tree_main2/F{b}/f{b.lower()}_*_daophot_basic.fits'):
        t = Table.read(fn)
        cats[os.path.basename(t.meta['FILENAME'])] = t
    out = {}
    for fr in np.unique(F['frame']):
        G = F[F['frame'] == fr]
        cat = cats[fr]
        ct = cKDTree(np.c_[np.asarray(cat['x_fit'], float), np.asarray(cat['y_fit'], float)])
        d, j = ct.query(np.c_[G['x'], G['y']])
        ok = d < 1e-3
        G, j = G[ok], j[ok]
        fl = np.asarray(cat['flux_fit'], float)[j]
        with fits.open(f'{an.Q}/tree_main2/F{b}/pipeline/{fr}') as h:
            sci = h['SCI'].data
            dq = h['DQ'].data
        bad = ~np.isfinite(sci) | ((dq & (DNU | SAT)) != 0)
        nb = np.zeros(len(G), int)
        pk = np.zeros(len(G))
        for i, (x, y) in enumerate(zip(G['x'], G['y'])):
            xi, yi = int(round(x)), int(round(y))
            sl = (slice(max(yi - 2, 0), yi + 3), slice(max(xi - 2, 0), xi + 3))
            nb[i] = bad[sl].sum()
            pk[i] = np.nanmax(sci[sl]) if np.isfinite(sci[sl]).any() else np.nan
        o = out.setdefault(G['det'][0], [])
        o.append(dict(lA=np.asarray(G['lA']), do=-2.5 * np.log10(fl / np.asarray(G['fold5'])), nb=nb, pk=pk, fl=fl))
    print(f'F{b}: det  frac_masked  slope(no mask)  slope(masked)  median do(masked) - do(clean)   N')
    for det, R in sorted(out.items()):
        frac = np.mean(np.concatenate([r['nb'] > 0 for r in R]))
        s0 = np.nanmedian([slope(r['lA'][r['nb'] == 0], r['do'][r['nb'] == 0]) for r in R])
        s1 = np.nanmedian([slope(r['lA'][r['nb'] > 0], r['do'][r['nb'] > 0]) for r in R])
        dd = np.nanmedian([np.nanmedian(r['do'][r['nb'] > 0]) - np.nanmedian(r['do'][r['nb'] == 0])
                           for r in R if (r['nb'] > 0).sum() > 3])
        n = sum(len(r['nb']) for r in R)
        print(f'  {det}  {frac:.3f}  {s0:+.2f}  {s1:+.2f}  {dd:+.4f}  {n}')
