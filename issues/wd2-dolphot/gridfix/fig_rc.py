"""Benchmark score against lA = pred - predT for the recentre_ab variants.

Rows: F150W, F200W.  Columns: main2 fit (old grid, seed box), new grid with seed box,
new grid with recentred box.  Score = dm + d_X - pred, minus the median of the star's
detector.  Grey points: stars; black: binned medians; red: least-squares slope k.
"""
import glob
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.table import Table, vstack  # noqa: E402

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an  # noqa: E402

D = f'{an.Q}/gridfix'
VAR = [('oi', 'main2: old grid, seed box'), ('ni', 'new grid, seed box'), ('nf', 'new grid, recentred box')]


def mad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x)))


fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True, constrained_layout=True)
for row, b in enumerate(['150W', '200W']):
    R = vstack([Table.read(f) for f in sorted(glob.glob(f'{D}/recentre/rc_{b}_*.ecsv'))])
    S = Table.read(f'{D}/stars_{b}.ecsv')
    S = S[np.isin(S['i'], R['i'])]
    idx = {k: n for n, k in enumerate(S['i'])}
    rr = np.array([idx[k] for k in R['i']])
    dets = np.unique(R['det'])
    cnt_det = np.array([np.bincount(rr[R['det'] == d], minlength=len(S)) for d in dets])
    sdet = dets[np.argmax(cnt_det, axis=0)]
    lA = np.asarray(S['pred'] - S['predT'], float)
    for col, (v, lab) in enumerate(VAR):
        num = np.bincount(rr, weights=R[f'f_{v}'] / R['fmain'], minlength=len(S))
        cnt = np.bincount(rr, minlength=len(S))
        sc = np.asarray(S['dm']) - 2.5 * np.log10(num / cnt) - np.asarray(S['pred'])
        res = sc - np.array([np.nanmedian(sc[sdet == d]) for d in sdet])
        k = np.isfinite(res) & np.isfinite(lA)
        g = k & (np.abs(res - np.nanmedian(res)) < 5 * mad(res))
        slope, icpt = np.polyfit(lA[g], res[g], 1)
        ax = axes[row, col]
        ax.scatter(lA[k], res[k], s=3, c='0.65', lw=0)
        edges = np.linspace(-0.03, 0.03, 13)
        cen = 0.5 * (edges[1:] + edges[:-1])
        med = [np.nanmedian(res[k & (lA >= lo) & (lA < hi)]) if np.sum(k & (lA >= lo) & (lA < hi)) > 10 else np.nan
               for lo, hi in zip(edges[:-1], edges[1:])]
        ax.plot(cen, med, 'ko-', ms=4, lw=1.5, label='binned median')
        xx = np.array([-0.035, 0.035])
        ax.plot(xx, icpt + slope * xx, 'r-', lw=1.5, label=f'k = {slope:+.2f}')
        ax.axhline(0, c='k', lw=0.5)
        ax.set_xlim(-0.035, 0.035)
        ax.set_ylim(-0.06, 0.06)
        ax.set_title(f'F{b}  {lab}\nrstd {mad(res):.4f} (detector medians removed)', fontsize=9)
        ax.legend(fontsize=8, loc='upper left')
        if row == 1:
            ax.set_xlabel('lA = pred - predT [mag]')
        if col == 0:
            ax.set_ylabel('dm + d - pred - detector median [mag]')
fig.savefig(f'{D}/rc_score.png', dpi=100)
print('wrote rc_score.png')
