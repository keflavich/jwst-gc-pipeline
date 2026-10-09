"""Three-cornered hat: split the per-star scatter of ours - dolphot into an "ours", a
"dolphot" and an "aperture" part, per band.

Per star (closure stars, gridfix/stars_<band>.ecsv; aperture sums from apclosure/,
mean over the star's PSF-matched frames, AREA map applied):
  O - D' = dm + d - pred      ours (new loader) - dolphot (area double count removed)
  D' - A = c + pred           c = dolphot - ap_area (apclosure)
  O - A  = (O - D') + (D' - A)
d = closure dfix5 (forced fit at the main2 position) or, for F150W/F200W, the recentre_ab
'ni' replay of the main2 m7 fit with the new loader (seed box).  Each difference has its
per-detector median removed.  With V = rstd^2:
  s_O^2 = (V(O-D') + V(O-A) - V(D'-A)) / 2,  s_D^2 = (V(O-D') + V(D'-A) - V(O-A)) / 2,
  s_A^2 = (V(D'-A) + V(O-A) - V(O-D')) / 2.
usage: python hat.py  ->  hat.txt
"""
import glob
import sys

import numpy as np
from astropy.table import Table, vstack

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an  # noqa: E402

Q = an.Q
AP = f'{Q}/apclosure'
GF = f'{Q}/gridfix'
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
L = []
L.append('| band | ours flux | r_ap | N | rstd O-D\' | rstd D\'-A | rstd O-A | s_O | s_D | s_A |')
L.append('|---|---|---|---|---|---|---|---|---|---|')
for b in BANDS:
    S = Table.read(f'{GF}/stars_{b}.ecsv')
    ref = np.asarray(A.ref[b])[np.asarray(S['i'])]
    assert np.allclose(ref, S['ref'], equal_nan=True), b
    variants = {'closure dfix5': np.asarray(S['dfix5'], float)}
    rc = sorted(glob.glob(f'{GF}/recentre/rc_{b}_*.ecsv'))
    if rc:
        R = vstack([Table.read(f) for f in rc])
        idx = {k: n for n, k in enumerate(S['i'])}
        keep = np.isin(R['i'], S['i'])
        R = R[keep]
        row = np.array([idx[k] for k in R['i']])
        num = np.bincount(row, weights=R['f_ni'] / R['fmain'], minlength=len(S))
        cnt = np.bincount(row, minlength=len(S))
        with np.errstate(divide='ignore', invalid='ignore'):
            variants['m7 replay ni'] = -2.5 * np.log10(num / cnt)
    det = np.asarray(S['det'])
    for r in (3, 5):
        am = ap_mag(b, r)
        apm = np.array([am.get(k, np.nan) for k in S['i']])
        c = ref - apm
        for vn, d in variants.items():
            OD = np.asarray(S['dm']) + d - np.asarray(S['pred'])
            DA = c + np.asarray(S['pred'])
            ok = np.isfinite(OD) & np.isfinite(DA)
            od, da = demed(OD[ok], det[ok]), demed(DA[ok], det[ok])
            oa = od + da
            # 5-sigma clip on all three
            g = (np.abs(od) < 5 * rs(od)) & (np.abs(da) < 5 * rs(da)) & (np.abs(oa) < 5 * rs(oa))
            v1, v2, v3 = rs(od[g]) ** 2, rs(da[g]) ** 2, rs(oa[g]) ** 2
            sO = np.sqrt(max((v1 + v3 - v2) / 2, 0))
            sD = np.sqrt(max((v1 + v2 - v3) / 2, 0))
            sA = np.sqrt(max((v2 + v3 - v1) / 2, 0))
            L.append(f'| F{b} | {vn} | {r} | {g.sum()} | {np.sqrt(v1):.4f} | {np.sqrt(v2):.4f} | {np.sqrt(v3):.4f} '
                     f'| {sO:.4f} | {sD:.4f} | {sA:.4f} |')
open(f'{Q}/f150x/hat.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
