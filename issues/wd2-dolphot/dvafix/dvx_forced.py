"""Flux of forced-refit rows, dvx1 / dvx0, matched by fitted pixel position per frame (#1128).

Row ids differ between the arms (the seed lists differ), so rows are matched by nearest
(x_fit, y_fit) within RMAX px (env, default 0.5), mutual nearest only.  Classes: forced in both arms, forced
only in dvx0, forced only in dvx1, neither.  Reports percentiles of -2.5 log10(f1 / f0).
"""
import glob
import os

import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree

RMAX = float(os.environ.get("RMAX", 0.5))
H = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/dvafix'
acc = {}
for _k in ('both', 'only0', 'only1', 'neither'):
    acc[_k] = []
nun = [0, 0]
for f0 in sorted(glob.glob(f'{H}/tree_dvx0/F150W/*_resbgsub_m7_daophot_basic.fits')):
    f1 = f'{H}/tree_dvx1/F150W/{os.path.basename(f0)}'
    if not os.path.exists(f1):
        continue
    t0, t1 = Table.read(f0), Table.read(f1)
    p0 = np.c_[t0['x_fit'], t0['y_fit']]
    p1 = np.c_[t1['x_fit'], t1['y_fit']]
    g0, g1 = np.all(np.isfinite(p0), 1), np.all(np.isfinite(p1), 1)
    k1 = cKDTree(p1[g1])
    k0 = cKDTree(p0[g0])
    d01, j01 = k1.query(p0[g0])
    d10, j10 = k0.query(p1[g1])
    ia = np.flatnonzero(g0)
    ib = np.flatnonzero(g1)
    mutual = (j10[j01] == np.arange(len(j01))) & (d01 < RMAX)
    a, b = t0[ia[mutual]], t1[ib[j01[mutual]]]
    nun[0] += len(t0) - mutual.sum()
    nun[1] += len(t1) - mutual.sum()
    F0 = np.asarray(a['forced_refit'], bool)
    F1 = np.asarray(b['forced_refit'], bool)
    ok = (a['flux_fit'] > 0) & (b['flux_fit'] > 0)
    dm = np.full(len(a), np.nan)
    dm[ok] = -2.5 * np.log10(b['flux_fit'][ok] / a['flux_fit'][ok])
    sh = np.hypot(a['x_init'] - b['x_init'], a['y_init'] - b['y_init']) > 0.2
    for k, m in (('both', F0 & F1), ('only0', F0 & ~F1), ('only1', ~F0 & F1), ('neither', ~F0 & ~F1)):
        acc[k].append(dm[m & ok])
        acc.setdefault(k + '_seedmoved', []).append(dm[m & ok & sh])
print('unmatched rows dvx0 %d dvx1 %d' % tuple(nun))
for k, v in acc.items():
    v = np.concatenate(v)
    p = np.nanpercentile(v, [10, 50, 90]) if len(v) else [np.nan] * 3
    print(f'{k:8s} N={len(v):6d}  dm(1-0) p10 {p[0]:+.4f} p50 {p[1]:+.4f} p90 {p[2]:+.4f}')
