"""Do per-image offsets explain the coherent residual?

Model: residual_s = mean over the frames covering star s of c_frame (one offset per crf
image, 32 per SW band), plus noise.  Least-squares fit (ridge 1e-6) on the closure stars
(no forced-refit frames, detector medians NOT removed, 5-sigma clipped), for O-D', D'-A
and O-A.  Reports rstd before and after, the rstd of the fitted offsets, the expected
variance drop for pure noise (32 / N), and the neighbour-mean coherence after the fit.
usage: python perimage.py -> perimage.txt (+ perimage_offsets_<band>.ecsv)
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


an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')
ra = np.asarray(A.m['RA'], float)
dec = np.asarray(A.m['DEC'], float)
L = ['| band | series | N | rstd before | rstd after per-image fit | rstd of offsets | noise-only drop | nbr corr before | after |',
     '|---|---|---|---|---|---|---|---|---|']
for b in ['150W', '200W']:
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
    T = Table.read(f'{AP}/frames_{b}.ecsv')
    T = T[(T['src'] == 1) & (T['psf'] > 0) & (T['area3'] > 0)]
    ids, inv = np.unique(T['i'], return_inverse=True)
    s3 = np.bincount(inv, weights=np.asarray(T['area3'], float))
    n3 = np.bincount(inv)
    am = dict(zip(ids[n3 >= 2], (-2.5 * np.log10(s3 / n3))[n3 >= 2]))
    apm = np.array([am.get(k, np.nan) for k in i])
    OD = np.asarray(S['dm']) + d - np.asarray(S['pred'])
    DA = np.asarray(S['ref']) - apm + np.asarray(S['pred'])
    ok = np.isfinite(OD) & np.isfinite(DA) & ~forced
    ser = {"O-D'": OD[ok] - np.median(OD[ok]), "D'-A": DA[ok] - np.median(DA[ok])}
    ser['O-A'] = ser["O-D'"] + ser["D'-A"]
    g = np.ones(ok.sum(), bool)
    for y in ser.values():
        g &= np.abs(y) < 5 * rs(y)
    ii = i[ok][g]
    frames = np.unique(T['frame'])
    fidx = {f: k for k, f in enumerate(frames)}
    sidx = {k: n for n, k in enumerate(ii)}
    W = np.zeros((len(ii), len(frames)))
    for r in T:
        if r['i'] in sidx:
            W[sidx[r['i']], fidx[r['frame']]] = 1.0
    W /= np.maximum(W.sum(axis=1, keepdims=True), 1)
    keep = W.sum(axis=1) > 0
    xy = np.c_[(ra[ii] - 156.0) * np.cos(np.deg2rad(-57.757)) * 3600, (dec[ii] + 57.757) * 3600][keep]
    _, jj = cKDTree(xy).query(xy, k=9)
    offs = Table()
    offs['frame'] = frames
    for k, y in ser.items():
        y = y[g][keep]
        Wk = W[keep]
        c = np.linalg.solve(Wk.T @ Wk + 1e-6 * np.eye(len(frames)), Wk.T @ y)
        res = y - Wk @ c
        nb0 = np.corrcoef(y, y[jj[:, 1:]].mean(axis=1))[0, 1]
        nb1 = np.corrcoef(res, res[jj[:, 1:]].mean(axis=1))[0, 1]
        L.append(f'| F{b} | {k} | {len(y)} | {rs(y):.4f} | {rs(res):.4f} | {np.std(c - np.median(c)):.4f} '
                 f'| {np.sqrt(1 - len(frames) / len(y)) * rs(y):.4f} | {nb0:+.3f} | {nb1:+.3f} |')
        offs[k] = c - np.median(c)
    offs.write(f'{Q}/f150x/perimage_offsets_{b}.ecsv', overwrite=True)
open(f'{Q}/f150x/perimage.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
