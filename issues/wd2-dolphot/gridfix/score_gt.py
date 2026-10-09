"""Score the grid-loader fix arm (main2gt) against main2 on the dolphot benchmark.

Per band, unsaturated, unreplaced stars in the ZP window matched in both arms:
  rstd of dm and of dm - pred (pred removes the dolphot pixel-area double count, #1153);
  fit dm = a pred + b predT + c (transposed grid: a ~ 2, b ~ -1; fixed grid: a ~ 1, b ~ 0);
  per-detector median of dm - pred (relative to the band median) and its rms over detectors.
Per frame (if both per-frame catalogue trees exist): -2.5 log10(flux_gt / flux_main2) for sources at
the same position, against lA = 2.5 log10(A(x,y)/A(y,x)); the fix predicts slope -1.
usage: python analyze_g.py gridfix/score_gt.py [arm_old arm_new]
"""
import glob
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import cKDTree

import analyze as an

old, new = (sys.argv[1:3] if len(sys.argv) > 2 else ('main2', 'main2gt'))
an.ZPWIN.update(an.zp_windows())
A, B = an.Arm(old), an.Arm(new)
P = np.load(f'{an.Q}/photomver/areapred_main2.npz')
PT = np.load(f'{an.Q}/psfsum/predT_main2.npz')
mad = an.mad


def fit(d, p, pt):
    X = np.column_stack([p, pt, np.ones_like(p)])
    k = np.ones(len(d), bool)
    for _ in range(3):
        c = np.linalg.lstsq(X[k], d[k], rcond=None)[0]
        r = d - X @ c
        k = np.abs(r) < 3 * mad(r[k])
    return c


print(f'## {old} vs {new}: robust std of ours - dolphot (stars matched in both arms)\n')
print(f'| band | N | {old} dm | {old} dm - pred | {new} dm | **{new} dm - pred** | {new} a (pred) | {new} b (predT) '
      f'| det rms {old} | det rms {new} |')
print('|---|---|---|---|---|---|---|---|---|---|')
dets = {}
for b in an.BANDS:
    lo, hi = an.ZPWIN.get(b, (0, 19))
    p, pt, det = P[b], PT[b], PT[f'{b}_det']
    ok = (A.matched & B.matched & np.isfinite(p) & np.isfinite(pt))
    for X in (A, B):
        ok &= np.isfinite(X.dm(b)) & ~X.rep[b] & ~X.sat[b]
    ok &= (A.ref[b] >= lo) & (A.ref[b] < hi)
    if ok.sum() < 50:
        continue
    da, db = A.dm(b)[ok], B.dm(b)[ok]
    c = fit(db, p[ok], pt[ok])
    rms = []
    for d in (da - p[ok], db - p[ok]):
        meds = [np.median(d[det[ok] == u]) for u in np.unique(det[ok]) if (det[ok] == u).sum() >= 50]
        rms.append(np.std(meds) if len(meds) > 1 else np.nan)
        dets.setdefault(b, []).append({u: np.median(d[det[ok] == u]) - np.median(d) for u in np.unique(det[ok])
                                       if (det[ok] == u).sum() >= 50})
    print(f'| F{b} | {ok.sum()} | {mad(da):.4f} | {mad(da - p[ok]):.4f} | {mad(db):.4f} | **{mad(db - p[ok]):.4f}** '
          f'| {c[0]:+.2f} | {c[1]:+.2f} | {rms[0]:.4f} | {rms[1]:.4f} |')

print(f'\n## Per-detector median of dm - pred, relative to the band median ({old} -> {new})\n')
for b, (da, db) in dets.items():
    print(f'F{b}: ' + ', '.join(f'{u} {da.get(u, np.nan):+.3f}->{db.get(u, np.nan):+.3f}' for u in sorted(db)))

print(f'\n## Per frame: -2.5 log10(flux_{new} / flux_{old}) against lA (model slope -1)\n')
print('| band | frames | median slope | min | max | N sources |')
print('|---|---|---|---|---|---|')
for b in an.BANDS:
    ta, tb = f'{an.Q}/tree_{old}/F{b}', f'{an.Q}/tree_{new}/F{b}'
    fa = sorted(glob.glob(f'{tb}/f{b.lower()}_*_resbgsub_m7_daophot_basic.fits'))
    sl, ntot = [], 0
    for fn in fa:
        fo = f'{ta}/{os.path.basename(fn)}'
        if not os.path.exists(fo):
            continue
        tn, to = Table.read(fn), Table.read(fo)
        crf = tn.meta['FILENAME']
        with fits.open(crf) as fh:
            area = np.asarray(fh['AREA'].data, float)
        xo, yo = np.asarray(to['x_fit'], float), np.asarray(to['y_fit'], float)
        xn, yn = np.asarray(tn['x_fit'], float), np.asarray(tn['y_fit'], float)
        d, j = cKDTree(np.c_[xo, yo]).query(np.c_[xn, yn])
        fo_, fn_ = np.asarray(to['flux_fit'], float)[j], np.asarray(tn['flux_fit'], float)
        snr = fn_ / np.asarray(tn['flux_err'], float)
        k = (d < 0.05) & (fo_ > 0) & (fn_ > 0) & (snr > 50) & (xn > 1) & (yn > 1) & (xn < 2046) & (yn < 2046)
        if k.sum() < 30:
            continue
        ix, iy = np.round(xn[k]).astype(int), np.round(yn[k]).astype(int)
        lA = 2.5 * np.log10(area[iy, ix] / area[ix, iy])
        dd = -2.5 * np.log10(fn_[k] / fo_[k])
        g = np.abs(dd - np.median(dd)) <= max(5 * mad(dd), 1e-6)
        sl.append(np.polyfit(lA[g], dd[g], 1)[0])
        ntot += g.sum()
    if sl:
        print(f'| F{b} | {len(sl)} | {np.median(sl):+.2f} | {np.min(sl):+.2f} | {np.max(sl):+.2f} | {ntot} |')
