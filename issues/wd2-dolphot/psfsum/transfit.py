"""Regress dm = ours - dolphot - ZP (main2, unsaturated stars in each band's ZP
window) on pred = 2.5 log10(AREA(x, y)) and predT = 2.5 log10(AREA(y, x)),
both flux-averaged over the exposures that contain the star.

Model (see psfsum.md): our STPSF grid PSF sums go as 1/AREA at the transposed
position, dolphot's PSF library sums go as 1/AREA at the star's position and
dolphot also multiplies the data by AREA.  With PSF sums only,
dm = 2 pred - predT; a Gaussian width mismatch on our side lowers this toward
1.5 pred - 0.5 predT.
usage: python transfit.py [arm]"""
import glob
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

an.ZPWIN.update(an.zp_windows())
arm = sys.argv[1] if len(sys.argv) > 1 else 'main2'
A = an.Arm(arm)
tree = f'{an.Q}/tree_main2'
P = dict(np.load(f'{an.Q}/photomver/areapred_{arm}.npz'))
out = f'{an.Q}/psfsum/predT_{arm}.npz'
PT = dict(np.load(out)) if os.path.exists(out) else {}
sky = A.sky[A.idx]
for band in an.BANDS:
    if band in PT:
        continue
    frames = sorted(glob.glob(f'{tree}/F{band}/pipeline/jw03523005001_*_align_o005_crf.fits'))
    fsum = np.zeros(A.n)
    nexp = np.zeros(A.n)
    det = np.full(A.n, '', dtype='U8')
    for fn in frames:
        with fits.open(fn, memmap=True) as fh:
            h = fh['SCI'].header
            area = np.asarray(fh['AREA'].data, float)
            dname = fh[0].header['DETECTOR'].lower()
        w = WCS(h)
        ny, nx = area.shape
        assert ny == nx
        x, y = w.world_to_pixel(sky)
        inn = A.matched & (x > 0) & (x < nx - 1) & (y > 0) & (y < ny - 1)
        a = np.full(A.n, np.nan)
        # transposed lookup: row = x, column = y
        a[inn] = area[np.round(x[inn]).astype(int), np.round(y[inn]).astype(int)] * h['PIXAR_SR'] / w.proj_plane_pixel_area().to(u.sr).value
        good = np.isfinite(a) & (a > 0)
        fsum[good] += a[good]
        nexp[good] += 1
        new = good & (det == '')
        det[new] = dname
    pt = np.full(A.n, np.nan)
    pt[nexp > 0] = 2.5 * np.log10(fsum[nexp > 0] / nexp[nexp > 0])
    PT[band] = pt
    PT[band + '_det'] = det
    np.savez(out, **PT)


def rs(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def fit(dm, p, pt, nboot=200, seed=1):
    """3-sigma (MAD) clipped OLS of dm on [p, pt, 1]; bootstrap 95 % CI."""
    X = np.column_stack([p, pt, np.ones_like(p)])
    keep = np.ones(len(dm), bool)
    for _ in range(5):
        c = np.linalg.lstsq(X[keep], dm[keep], rcond=None)[0]
        r = dm - X @ c
        keep = np.abs(r - np.median(r[keep])) < 3 * rs(r[keep])
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(keep)
    bs = np.array([np.linalg.lstsq(X[s], dm[s], rcond=None)[0]
                   for s in (rng.choice(idx, len(idx)) for _ in range(nboot))])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return c, lo, hi, keep


print(f'arm {arm}: dm = a*pred + b*predT + c, unsaturated stars in the ZP window')
print('| band | N | corr(pred, predT) | a [95 % CI] | b [95 % CI] | rstd dm | dm - pred | dm - (2 pred - predT) | dm - (1.5 pred - 0.5 predT) | dm - fit |')
print('|---|---|---|---|---|---|---|---|---|---|')
rows = {}
for band in an.BANDS:
    dm = A.dm(band)
    p, pt = P[band], PT[band]
    lo, hi = an.ZPWIN.get(band, (0, 19))
    s = (A.matched & np.isfinite(dm) & np.isfinite(p) & np.isfinite(pt) & ~A.rep[band] & ~A.sat[band]
         & (A.ref[band] >= lo) & (A.ref[band] < hi))
    if s.sum() < 50:
        continue
    c, clo, chi, keep = fit(dm[s], p[s], pt[s])
    rows[band] = dict(a=c[0], b=c[1])
    r = [rs(dm[s]), rs((dm - p)[s]), rs((dm - 2 * p + pt)[s]), rs((dm - 1.5 * p + 0.5 * pt)[s]),
         rs(dm[s] - c[0] * p[s] - c[1] * pt[s])]
    print(f'| F{band} | {s.sum()} | {np.corrcoef(p[s], pt[s])[0, 1]:+.2f} | {c[0]:.2f} [{clo[0]:.2f}, {chi[0]:.2f}] | '
          f'{c[1]:+.2f} [{clo[1]:+.2f}, {chi[1]:+.2f}] | ' + ' | '.join(f'{v:.4f}' for v in r) + ' |')
    sys.stdout.flush()

print('\nPer detector (first exposure containing the star), dm = a*pred + b*predT + c')
print('| band | det | N | corr(pred, predT) | a | b | rstd dm | dm - pred | dm - fit(band) |')
print('|---|---|---|---|---|---|---|---|---|')
for band in an.BANDS:
    if band not in rows:
        continue
    dm = A.dm(band)
    p, pt, det = P[band], PT[band], PT[band + '_det']
    lo, hi = an.ZPWIN.get(band, (0, 19))
    base = (A.matched & np.isfinite(dm) & np.isfinite(p) & np.isfinite(pt) & ~A.rep[band] & ~A.sat[band]
            & (A.ref[band] >= lo) & (A.ref[band] < hi))
    for d in sorted(set(det[base])):
        s = base & (det == d)
        if s.sum() < 80:
            continue
        c, clo, chi, keep = fit(dm[s], p[s], pt[s], nboot=50)
        ab = rows[band]
        print(f'| F{band} | {d} | {s.sum()} | {np.corrcoef(p[s], pt[s])[0, 1]:+.2f} | {c[0]:.2f} [{clo[0]:.2f}, {chi[0]:.2f}] | '
              f'{c[1]:+.2f} [{clo[1]:+.2f}, {chi[1]:+.2f}] | {rs(dm[s]):.4f} | {rs((dm - p)[s]):.4f} | '
              f'{rs(dm[s] - ab["a"] * p[s] - ab["b"] * pt[s]):.4f} |')
    sys.stdout.flush()
