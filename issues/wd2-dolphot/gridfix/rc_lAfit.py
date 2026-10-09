"""Does the remaining F150W scatter carry an lA term?  Fit
score_X = c0 + c_det + k (pred - predT) per band and variant, and report k and the rstd
after removing the per-detector medians and k (pred - predT).

usage: python rc_lAfit.py BAND [BAND ...]
"""
import glob
import sys

import numpy as np
from astropy.table import Table, vstack

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an  # noqa: E402

D = f'{an.Q}/gridfix'


def mad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x)))


for b in sys.argv[1:]:
    R = vstack([Table.read(f) for f in sorted(glob.glob(f'{D}/recentre/rc_{b}_*.ecsv'))])
    S = Table.read(f'{D}/stars_{b}.ecsv')
    S = S[np.isin(S['i'], R['i'])]
    idx = {k: n for n, k in enumerate(S['i'])}
    row = np.array([idx[k] for k in R['i']])
    dets = np.unique(R['det'])
    cnt_det = np.array([np.bincount(row[R['det'] == d], minlength=len(S)) for d in dets])
    sdet = dets[np.argmax(cnt_det, axis=0)]
    lA = np.asarray(S['pred'] - S['predT'], float)
    for v in ('oi', 'ni', 'nf'):
        num = np.bincount(row, weights=R[f'f_{v}'] / R['fmain'], minlength=len(S))
        cnt = np.bincount(row, minlength=len(S))
        sc = np.asarray(S['dm']) - 2.5 * np.log10(num / cnt) - np.asarray(S['pred'])
        res = sc - np.array([np.nanmedian(sc[sdet == d]) for d in sdet])
        k = np.isfinite(res) & np.isfinite(lA)
        g = k & (np.abs(res - np.nanmedian(res)) < 5 * mad(res))
        A = np.c_[lA[g], *[(sdet[g] == d).astype(float) for d in dets]]
        coef = np.linalg.lstsq(A, sc[g], rcond=None)[0]
        res2 = sc - coef[0] * lA - np.array([coef[1 + list(dets).index(d)] for d in sdet])
        print(f'F{b} {v}: k = {coef[0]:+.3f}, rstd after det medians {mad(res):.4f}, '
              f'after det + k lA {mad(res2):.4f}, rms lA {np.nanstd(lA):.4f}')
