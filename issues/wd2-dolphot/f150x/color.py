"""Colour dependence of the star-level residuals (F150W excess scatter).

Per closure star: O-D' (benchmark score, new loader m7 replay 'ni' for F150W/F200W,
closure dfix5 otherwise), D'-A and O-A (r = 3 px aperture, apclosure), each with
per-detector medians removed, against the dolphot colour F115W - F200W.
Reports the least-squares slope [mag/mag], the rstd before and after removing a
quadratic in colour, and binned medians.
usage: python color.py -> color.txt, color.png
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
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/f150x')

Q = an.Q
GF = f'{Q}/gridfix'
AP = f'{Q}/apclosure'
BANDS = ['150W', '200W', '212N', '182M']


def rs(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def demed(x, det):
    out = np.array(x, float)
    for d in np.unique(det):
        m = det == d
        out[m] -= np.nanmedian(out[m])
    return out


def ap_mag(band, r):
    T = Table.read(f'{AP}/frames_{band}.ecsv')
    T = T[(T['src'] == 1) & (T['psf'] > 0) & (T[f'area{r}'] > 0)]
    ids, inv = np.unique(T['i'], return_inverse=True)
    s = np.bincount(inv, weights=np.asarray(T[f'area{r}'], float))
    n = np.bincount(inv)
    m = -2.5 * np.log10(s / n)
    return dict(zip(ids[n >= 2], m[n >= 2]))


an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')
col_all = np.asarray(A.ref['115W'], float) - np.asarray(A.ref['200W'], float)
L = ['| band | diff | N | slope [mag/mag] | rstd | rstd after quad(colour) |', '|---|---|---|---|---|---|']
fig, axes = plt.subplots(1, len(BANDS), figsize=(4 * len(BANDS), 4), sharey=True, constrained_layout=True)
for ax, b in zip(axes, BANDS):
    S = Table.read(f'{GF}/stars_{b}.ecsv')
    i = np.asarray(S['i'])
    d = np.asarray(S['dfix5'], float)
    rc = sorted(glob.glob(f'{GF}/recentre/rc_{b}_*.ecsv'))
    lab = 'closure dfix5'
    if rc:
        R = vstack([Table.read(f) for f in rc])
        R = R[np.isin(R['i'], i)]
        idx = {k: n for n, k in enumerate(i)}
        row = np.array([idx[k] for k in R['i']])
        num = np.bincount(row, weights=R['f_ni'] / R['fmain'], minlength=len(S))
        cnt = np.bincount(row, minlength=len(S))
        with np.errstate(divide='ignore', invalid='ignore'):
            d = -2.5 * np.log10(num / cnt)
        lab = 'm7 replay ni'
    am = ap_mag(b, 3)
    apm = np.array([am.get(k, np.nan) for k in i])
    ref = np.asarray(S['ref'], float)
    det = np.asarray(S['det'])
    col = col_all[i]
    OD = np.asarray(S['dm']) + d - np.asarray(S['pred'])
    DA = ref - apm + np.asarray(S['pred'])
    ok = np.isfinite(OD) & np.isfinite(DA) & np.isfinite(col)
    series = {"O-D'": demed(OD[ok], det[ok]), "D'-A": demed(DA[ok], det[ok])}
    series['O-A'] = series["O-D'"] + series["D'-A"]
    c = col[ok]
    for k, y in series.items():
        g = np.abs(y) < 5 * rs(y)
        s1 = np.polyfit(c[g], y[g], 1)[0]
        p2 = np.polyfit(c[g], y[g], 2)
        r2 = y[g] - np.polyval(p2, c[g])
        L.append(f'| F{b} ({lab}) | {k} | {g.sum()} | {s1:+.4f} | {rs(y[g]):.4f} | {rs(r2):.4f} |')
        edges = np.nanpercentile(c[g], np.linspace(0, 100, 9))
        cen = [np.median(c[g][(c[g] >= lo) & (c[g] <= hi)]) for lo, hi in zip(edges[:-1], edges[1:])]
        med = [np.median(y[g][(c[g] >= lo) & (c[g] <= hi)]) for lo, hi in zip(edges[:-1], edges[1:])]
        ax.plot(cen, med, 'o-', ms=4, label=f'{k}  slope {s1:+.4f}')
    ax.axhline(0, c='k', lw=0.5)
    ax.set_title(f'F{b} ({lab})', fontsize=10)
    ax.set_xlabel('dolphot F115W - F200W [mag]')
    ax.legend(fontsize=7)
axes[0].set_ylabel('binned median residual, detector medians removed [mag]')
axes[0].set_ylim(-0.03, 0.03)
fig.savefig(f'{Q}/f150x/color.png', dpi=100)
open(f'{Q}/f150x/color.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
