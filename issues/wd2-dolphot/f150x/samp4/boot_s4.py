"""Star-bootstrap uncertainty of the r^2 phase coefficient for the samp2 (cached) and samp4
grids, and of their paired difference, on the analyze_s4.py selection (nrcb1)."""
import numpy as np
from astropy.table import Table, join

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
SD = f'{Q}/f150x/samp4'
rng = np.random.default_rng(1)


def rs(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x)))


for b in ('150W', '200W'):
    R = Table.read(f'{SD}/fits/s4_{b}_nrcb1.ecsv')
    P = Table.read(f'{Q}/apclosure/frames_{b}.ecsv')
    P = P[(P['src'] == 1) & (P['area3'] > 0)]['i', 'frame', 'area3']
    J = join(R, P, keys=['i', 'frame'])
    J = J[~np.asarray(J['forced'], bool)]
    ids, inv = np.unique(J['i'], return_inverse=True)
    n = np.bincount(inv)
    out = {}
    for g in 'cb':
        x, y = np.asarray(J[f'x_{g}i'], float), np.asarray(J[f'y_{g}i'], float)
        rp = np.hypot(x - np.round(x), y - np.round(y))
        q = -2.5 * np.log10(np.asarray(J[f'f_{g}i'], float) / np.asarray(J['area3'], float))
        ok = np.isfinite(q)
        qm = np.bincount(inv[ok], weights=q[ok], minlength=len(ids)) / np.maximum(np.bincount(inv[ok], minlength=len(ids)), 1)
        dq = q - qm[inv]
        sel = ok & (n[inv] >= 3)
        sel &= np.abs(dq) < 5 * rs(dq[sel])
        out[g] = (rp, dq, sel)
    sel = out['c'][2] & out['b'][2]

    def coef(m, g):
        rp, dq, _ = out[g]
        A = np.c_[np.ones(m.sum()), rp[m] ** 2]
        return np.linalg.lstsq(A, dq[m], rcond=None)[0][1]

    c0, b0 = coef(sel, 'c'), coef(sel, 'b')
    bs = []
    for _ in range(2000):
        pick = rng.integers(0, len(ids), len(ids))
        w = np.bincount(pick, minlength=len(ids))[inv]
        m = sel & (w > 0)
        m = np.repeat(np.flatnonzero(m), w[m])
        mm = np.zeros(len(J), bool)
        rp_c, dq_c, _ = out['c']
        rp_b, dq_b, _ = out['b']
        Ac = np.c_[np.ones(len(m)), rp_c[m] ** 2]
        Ab = np.c_[np.ones(len(m)), rp_b[m] ** 2]
        kc = np.linalg.lstsq(Ac, dq_c[m], rcond=None)[0][1]
        kb = np.linalg.lstsq(Ab, dq_b[m], rcond=None)[0][1]
        bs.append((kc, kb, kb - kc))
    bs = np.array(bs)
    print(f'F{b}: N rows {sel.sum()}  samp2 {c0:+.4f} +- {bs[:, 0].std():.4f}   samp4 {b0:+.4f} +- {bs[:, 1].std():.4f}'
          f'   samp4 - samp2 {b0 - c0:+.4f} +- {bs[:, 2].std():.4f}')
