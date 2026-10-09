"""Which side carries the spatially coherent F150W residual?

For F150W and F200W closure stars without forced-refit frames: O-D' (score), O-A and D'-A
(A = r = 3 px aperture on the crf with the AREA map, apclosure; no satstar model, local
annulus background), detector medians removed, 5-sigma clipped.  For each series: rstd,
correlation with the mean of its k = 8 nearest neighbours, and the rstd of that
neighbour mean (the coherent part).
usage: python spatial2.py -> spatial2.txt
"""
import glob
import sys

import numpy as np
from astropy.table import Table, vstack
from scipy.spatial import cKDTree

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
L = ['| band | series | N | rstd | corr with 8-nbr mean | rstd nbr mean | rstd minus nbr mean |',
     '|---|---|---|---|---|---|---|']
for b in ['150W', '200W']:
    S = Table.read(f'{GF}/stars_{b}.ecsv')
    i = np.asarray(S['i'])
    R = vstack([Table.read(f) for f in sorted(glob.glob(f'{GF}/recentre/rc_{b}_*.ecsv'))])
    R = R[np.isin(R['i'], i)]
    idx = {k: n for n, k in enumerate(i)}
    row = np.array([idx[k] for k in R['i']])
    num = np.bincount(row, weights=R['f_ni'] / R['fmain'], minlength=len(S))
    cnt = np.bincount(row, minlength=len(S))
    forced = np.bincount(row, weights=R['forced'].astype(float), minlength=len(S)) > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        d = -2.5 * np.log10(num / cnt)
    am = ap_mag(b, 3)
    apm = np.array([am.get(k, np.nan) for k in i])
    OD = np.asarray(S['dm']) + d - np.asarray(S['pred'])
    DA = np.asarray(S['ref']) - apm + np.asarray(S['pred'])
    ok = np.isfinite(OD) & np.isfinite(DA) & ~forced
    det = np.asarray(S['det'])[ok]
    ser = {"O-D'": demed(OD[ok], det), "D'-A": demed(DA[ok], det)}
    ser['O-A'] = ser["O-D'"] + ser["D'-A"]
    g = np.ones(ok.sum(), bool)
    for y in ser.values():
        g &= np.abs(y) < 5 * rs(y)
    ii = i[ok][g]
    xy = np.c_[(ra[ii] - 156.0) * np.cos(np.deg2rad(-57.757)) * 3600, (dec[ii] + 57.757) * 3600]
    _, jj = cKDTree(xy).query(xy, k=9)
    for k, y in ser.items():
        y = y[g]
        nb = y[jj[:, 1:]].mean(axis=1)
        L.append(f'| F{b} | {k} | {g.sum()} | {rs(y):.4f} | {np.corrcoef(y, nb)[0, 1]:+.3f} | {rs(nb):.4f} | {rs(y - nb):.4f} |')
open(f'{Q}/f150x/spatial2.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
