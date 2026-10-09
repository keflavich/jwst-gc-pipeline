"""Per detector: slope of main2 flux_fit against the old-grid forced fit, versus lA,
split by forced_refit and by flux tercile, plus local_bkg statistics.

do = -2.5 log10(flux_fit_main2 / fold5)   (fold5: closure.py forced fit, old loader, 5x5)
"""
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


def slope(lA, d):
    k = np.isfinite(d) & np.isfinite(lA)
    if k.sum() < 20:
        return np.nan, k.sum()
    lA, d = lA[k], d[k]
    g = np.abs(d - np.median(d)) <= max(5 * mad(d), 1e-6)
    return np.polyfit(lA[g], d[g], 1)[0], g.sum()


for b in sys.argv[1:]:
    F = Table.read(f'{D}/frames_{b}.ecsv')
    F = F[F['src'] == 1]
    cats = {}
    for fn in glob.glob(f'{an.Q}/tree_main2/F{b}/f{b.lower()}_*_daophot_basic.fits'):
        t = Table.read(fn)
        cats[os.path.basename(t.meta['FILENAME'])] = t
    rows = []
    for fr in np.unique(F['frame']):
        G = F[F['frame'] == fr]
        cat = cats[fr]
        ct = cKDTree(np.c_[np.asarray(cat['x_fit'], float), np.asarray(cat['y_fit'], float)])
        d, j = ct.query(np.c_[G['x'], G['y']])
        ok = d < 1e-3
        G, j = G[ok], j[ok]
        fl = np.asarray(cat['flux_fit'], float)[j]
        rows.append(dict(fr=fr, det=G['det'][0], lA=np.asarray(G['lA']),
                         do=-2.5 * np.log10(fl / np.asarray(G['fold5'])),
                         fr_flag=np.asarray(cat['forced_refit'])[j].astype(bool),
                         flux=fl, lbkg=np.asarray(cat['local_bkg'], float)[j],
                         lbkg_r=np.asarray(cat['local_bkg_resbgsub'], float)[j],
                         msb=np.asarray(cat['modelsub_bkg'], float)[j],
                         x=np.asarray(G['x']), y=np.asarray(G['y'])))
    print(f'F{b}: per-frame slope of do on lA (median over frames per detector)')
    print('  det    all   refit=0 refit=1  frac_refit  faint  mid   bright   med lbkg  med lbkg_resbgsub  med modelsub_bkg')
    for det in sorted({r['det'] for r in rows}):
        R = [r for r in rows if r['det'] == det]
        s_all = [slope(r['lA'], r['do'])[0] for r in R]
        s0 = [slope(r['lA'][~r['fr_flag']], r['do'][~r['fr_flag']])[0] for r in R]
        s1 = [slope(r['lA'][r['fr_flag']], r['do'][r['fr_flag']])[0] for r in R]
        frac = np.mean(np.concatenate([r['fr_flag'] for r in R]))
        terc = []
        for q in range(3):
            ss = []
            for r in R:
                lo, hi = np.nanpercentile(r['flux'], [100 * q / 3, 100 * (q + 1) / 3])
                k = (r['flux'] >= lo) & (r['flux'] <= hi)
                ss.append(slope(r['lA'][k], r['do'][k])[0])
            terc.append(np.nanmedian(ss))
        lb = np.concatenate([r['lbkg'] for r in R])
        lbr = np.concatenate([r['lbkg_r'] for r in R])
        msb = np.concatenate([r['msb'] for r in R])
        print(f'  {det}  {np.nanmedian(s_all):+.2f}  {np.nanmedian(s0):+.2f}  {np.nanmedian(s1):+.2f}   {frac:.3f}'
              f'   {terc[0]:+.2f} {terc[1]:+.2f} {terc[2]:+.2f}   {np.nanmedian(lb):+.4f}  {np.nanmedian(lbr):+.4f}  {np.nanmedian(msb):+.4f}')
