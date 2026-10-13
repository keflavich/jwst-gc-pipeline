"""Does the LW residual A in (e) psf - ap_area follow crowding?  Adds log local density (dolphot stars within 2") and
the distance from the cluster centre as covariates, and repeats the fit on the outer half only.  usage: python crowd_gt.py"""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
import importlib.util
spec = importlib.util.spec_from_file_location('ap', f'{Q}/apclosure/analyze_ap.py')
ap = importlib.util.module_from_spec(spec)
exec(compile(open(spec.origin).read().replace('\nmain()\n', '\n'), spec.origin, 'exec'), ap.__dict__)
HERE = os.path.dirname(os.path.abspath(__file__))
A = ap.A
ra, dec = np.asarray(A.m['RA'], float), np.asarray(A.m['DEC'], float)
ok = np.isfinite(ra) & np.isfinite(dec)
d0 = np.nanmedian(dec)
X = (ra - np.nanmedian(ra)) * np.cos(np.deg2rad(d0)) * 3600
Y = (dec - d0) * 3600
tr = cKDTree(np.c_[X[ok], Y[ok]])
dens = np.full(len(ra), np.nan)
dens[ok] = [len(v) for v in tr.query_ball_point(np.c_[X[ok], Y[ok]], 2.0)]
br = ok & (np.nan_to_num(A.ref['200W'], nan=99) < 15)
cx, cy = np.median(X[br]), np.median(Y[br])
rc = np.hypot(X - cx, Y - cy)


def fitk(d, cols):
    M = np.column_stack(cols + [np.ones_like(d)])
    keep = np.all(np.isfinite(M), 1) & np.isfinite(d)
    for _ in range(5):
        c = np.linalg.lstsq(M[keep], d[keep], rcond=None)[0]
        r = d - M @ c
        keep &= np.abs(r - np.median(r[keep])) < 3 * ap.rs(r[keep])
    rng = np.random.default_rng(1)
    idx = np.flatnonzero(keep)
    bs = np.array([np.linalg.lstsq(M[s], d[s], rcond=None)[0] for s in (rng.choice(idx, len(idx)) for _ in range(300))])
    return c, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0), keep.sum()


for band in ('200W', '212N', '150W', '250M', '300M', '323N', '410M'):
    for arm, here in (('gt', f'{HERE}/gt'),):
        ap.HERE = here
        for r in (3, 5):
            D = ap.diffs(band, ap.stars(band), r)
            i = D['i']
            ld = np.log10(dens[i])
            print(f'F{band} r{r}: corr(pred, log dens) {np.corrcoef(D["pred"], ld)[0,1]:+.2f}  corr(pred, rc) {np.corrcoef(D["pred"], rc[i])[0,1]:+.2f}  N {len(i)}')
            for lab, cols, sel in (('pred,predT', [D['pred'], D['predT']], None),
                                   ('+logdens', [D['pred'], D['predT'], ld], None),
                                   ('outer half', [D['pred'], D['predT']], rc[i] > np.median(rc[i])),
                                   ('inner half', [D['pred'], D['predT']], rc[i] <= np.median(rc[i]))):
                if sel is None:
                    c, lo, hi, n = fitk(D['e'], cols)
                else:
                    c, lo, hi, n = fitk(D['e'][sel], [x[sel] for x in cols])
                print(f'    {lab:11s} A {c[0]:+.2f} [{lo[0]:+.2f},{hi[0]:+.2f}]  B {c[1]:+.2f} [{lo[1]:+.2f},{hi[1]:+.2f}]' + (f'  C(logdens) {c[2]:+.3f} [{lo[2]:+.3f},{hi[2]:+.3f}]' if len(cols) == 3 else '') + f'  N {n}')
