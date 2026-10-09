"""phase.py / rc_score.py metrics for the samp4 experiment (grids c=cached samp2, a=samp2 control, b=samp4)."""
import sys
import numpy as np
from astropy.table import Table, join
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
SD = f'{Q}/f150x/samp4'
LAB = {'c': 'cached samp2', 'a': 'control samp2', 'b': 'new samp4'}


def rs(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x)))


L = ['band grid  rstd_dq  r2_coef  r2_intercept  N_rows  score_rstd  N_stars  median f/f_cached  median f/f_control(a)']
for b in ('150W', '200W'):
    R = Table.read(f'{SD}/fits/s4_{b}_nrcb1.ecsv')
    P = Table.read(f'{Q}/apclosure/frames_{b}.ecsv')
    P = P[(P['src'] == 1) & (P['area3'] > 0)]['i', 'frame', 'area3']
    J = join(R, P, keys=['i', 'frame'])
    J = J[~np.asarray(J['forced'], bool)]
    rp = np.hypot(J['x_fit'] - np.round(J['x_fit']), J['y_fit'] - np.round(J['y_fit']))
    ids, inv = np.unique(J['i'], return_inverse=True)
    n = np.bincount(inv)
    S = Table.read(f'{Q}/gridfix/stars_{b}.ecsv')
    S = S[np.isin(S['i'], R['i'])]
    idx = {k: m for m, k in enumerate(S['i'])}
    row = np.array([idx[k] for k in R['i']])
    for g in 'cab':
        q = -2.5 * np.log10(np.asarray(J[f'f_{g}i'], float) / np.asarray(J['area3'], float))
        ok = np.isfinite(q)
        qm = np.bincount(inv[ok], weights=q[ok], minlength=len(ids)) / np.maximum(np.bincount(inv[ok], minlength=len(ids)), 1)
        dq = q - qm[inv]
        sel = ok & (n[inv] >= 3)
        sel &= np.abs(dq) < 5 * rs(dq[sel])
        A = np.c_[np.ones(sel.sum()), rp[sel] ** 2]
        a = np.linalg.lstsq(A, dq[sel], rcond=None)[0]
        num = np.bincount(row, weights=R[f'f_{g}i'] / R['fmain'], minlength=len(S))
        cnt = np.bincount(row, minlength=len(S))
        sc = np.asarray(S['dm']) + (-2.5 * np.log10(num / cnt)) - np.asarray(S['pred'])
        rc = np.nanmedian(np.asarray(R[f'f_{g}i']) / np.asarray(R['f_ci']))
        ra = np.nanmedian(np.asarray(R[f'f_{g}i']) / np.asarray(R['f_ai']))
        L.append(f'F{b} {LAB[g]:14s} {rs(dq[sel]):.4f} {a[1]:+.4f} {a[0]:+.4f} {sel.sum()} {rs(sc):.4f} {len(S)} {rc:.5f} {ra:.5f}')
open(f'{SD}/results.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
