"""Maps of the 8-neighbour-mean residuals (coherent part) for F150W and F200W, and their
relation to the m6 smoothed background level at each star.

Rows: F150W, F200W.  Columns: O-D' (score), D'-A (dolphot - aperture), O-A (ours -
aperture); colour = mean of the 8 nearest closure neighbours of each star (same
definitions as spatial2.py).  The background test samples the main2 m6 smoothed
background i2d at each star.
usage: python fig_map.py -> map.png, map.txt
"""
import glob
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.table import Table, vstack  # noqa: E402
from astropy.wcs import WCS  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an  # noqa: E402

Q = an.Q
GF = f'{Q}/gridfix'
AP = f'{Q}/apclosure'


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
    return dict(zip(ids[n >= 2], (-2.5 * np.log10(s / n))[n >= 2]))


an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')
ra = np.asarray(A.m['RA'], float)
dec = np.asarray(A.m['DEC'], float)
fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharex=True, sharey=True, constrained_layout=True)
L = []
for row, b in enumerate(['150W', '200W']):
    S = Table.read(f'{GF}/stars_{b}.ecsv')
    i = np.asarray(S['i'])
    R = vstack([Table.read(f) for f in sorted(glob.glob(f'{GF}/recentre/rc_{b}_*.ecsv'))])
    R = R[np.isin(R['i'], i)]
    idx = {k: n for n, k in enumerate(i)}
    rr = np.array([idx[k] for k in R['i']])
    num = np.bincount(rr, weights=R['f_ni'] / R['fmain'], minlength=len(S))
    cnt = np.bincount(rr, minlength=len(S))
    forced = np.bincount(rr, weights=R['forced'].astype(float), minlength=len(S)) > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        d = -2.5 * np.log10(num / cnt)
    am = ap_mag(b, 3)
    apm = np.array([am.get(k, np.nan) for k in i])
    OD = np.asarray(S['dm']) + d - np.asarray(S['pred'])
    DA = np.asarray(S['ref']) - apm + np.asarray(S['pred'])
    ok = np.isfinite(OD) & np.isfinite(DA) & ~forced
    det = np.asarray(S['det'])[ok]
    ser = {"O-D' (ours - dolphot)": demed(OD[ok], det), "D'-A (dolphot - aperture)": demed(DA[ok], det)}
    ser['O-A (ours - aperture)'] = ser["O-D' (ours - dolphot)"] + ser["D'-A (dolphot - aperture)"]
    g = np.ones(ok.sum(), bool)
    for y in ser.values():
        g &= np.abs(y) < 5 * rs(y)
    ii = i[ok][g]
    x = (ra[ii] - 156.0) * np.cos(np.deg2rad(-57.757)) * 3600
    yv = (dec[ii] + 57.757) * 3600
    _, jj = cKDTree(np.c_[x, yv]).query(np.c_[x, yv], k=9)
    bgfn = glob.glob(f'{Q}/tree_main2/F{b}/pipeline/*-f{b.lower()}-merged_resbgsub_m6_daophot_basic_'
                     'mergedcat_residual_smoothed_bg_i2d.fits')[0]
    with fits.open(bgfn) as bh:
        hdu = bh['SCI'] if 'SCI' in [h.name for h in bh] else bh[0]
        bg, w = hdu.data.astype(float), WCS(hdu.header)
    px, py = w.world_to_pixel_values(ra[ii], dec[ii])
    bgv = bg[np.clip(np.round(py).astype(int), 0, bg.shape[0] - 1), np.clip(np.round(px).astype(int), 0, bg.shape[1] - 1)]
    for col, (k, y) in enumerate(ser.items()):
        y = y[g]
        nb = y[jj[:, 1:]].mean(axis=1)
        ax = axes[row, col]
        sc = ax.scatter(x, yv, c=nb, s=10, cmap='RdBu_r', vmin=-0.012, vmax=0.012, lw=0)
        ax.set_title(f'F{b}  {k}\n8-nbr mean, rstd {rs(nb):.4f}', fontsize=9)
        ax.set_aspect('equal')
        kk = np.isfinite(bgv)
        L.append(f'F{b} {k}: rstd nbr mean {rs(nb):.4f}; Spearman-like corr(nbr mean, rank bg) '
                 f'{np.corrcoef(nb[kk], np.argsort(np.argsort(bgv[kk])))[0, 1]:+.3f}')
        if row == 1:
            ax.set_xlabel('dRA cos(dec) from (156.0, -57.757) ["]')
        if col == 0:
            ax.set_ylabel('dDec ["]')
fig.colorbar(sc, ax=axes, shrink=0.6, label='8-neighbour mean residual [mag]')
fig.savefig(f'{Q}/f150x/map.png', dpi=90)
open(f'{Q}/f150x/map.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
