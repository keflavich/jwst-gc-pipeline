"""Per detector: robust scatter of main2 flux_fit against the old- and new-grid forced fits,
and slope of dmain on x, y and lA jointly (to separate a position trend from lA)."""
import glob
import os
import sys

import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an  # noqa: E402

D = f'{an.Q}/gridfix'


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


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
        fl = np.asarray(cat['flux_fit'], float)[j]
        ok = d < 1e-3
        det = G['det'][0]
        o = out.setdefault(det, dict(do=[], dn=[], x=[], y=[], lA=[], gs=[]))
        o['do'] += list(-2.5 * np.log10(fl[ok] / G['fold5'][ok]))
        o['dn'] += list(-2.5 * np.log10(fl[ok] / G['fnew5'][ok]))
        o['x'] += list(G['x'][ok]); o['y'] += list(G['y'][ok]); o['lA'] += list(G['lA'][ok])
        o['gs'] += list(np.asarray(cat['group_size'])[j][ok])
    print(f'F{b}: det  N  rstd(main/old) rstd(main/new)  coef lA, x/1000, y/1000 (joint fit)  median group_size')
    for det, o in sorted(out.items()):
        do, dn = np.array(o['do']), np.array(o['dn'])
        k = np.isfinite(do) & np.isfinite(dn) & (np.abs(do - np.median(do)) < 5 * mad(do))
        X = np.c_[np.array(o['lA'])[k], np.array(o['x'])[k] / 1e3, np.array(o['y'])[k] / 1e3, np.ones(k.sum())]
        c = np.linalg.lstsq(X, do[k], rcond=None)[0]
        print(f'  {det} {k.sum():5d}  {mad(do[k]):.4f}  {mad(dn[k]):.4f}   {c[0]:+.2f} {c[1]:+.4f} {c[2]:+.4f}   '
              f'{np.median(np.array(o["gs"])[k]):.0f}')
