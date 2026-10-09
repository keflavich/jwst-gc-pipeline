"""Closure test of the STPSF grid-loader fix (psf_grid_io.load_stpsf_grid).

For main2 stars in each band's ZP window (matched, unsaturated, not replaced),
fit the flux on every crf frame at a fixed position with the production grid
loaded two ways: stpsf.utils.to_griddedpsfmodel (old, transposed planes) and
psf_grid_io.load_stpsf_grid (new).  Weighted linear flux solve
f = sum(d p w) / sum(p^2 w), local annulus background, fit boxes 5x5 (the
production fit_shape) and 11x11.  Position: the per-frame catalogue x_fit/y_fit
when one lies within 0.5 px, else skycoord_f<band> projected with the frame WCS.

Per frame:  dfix = -2.5 log10(f_new / f_old), predicted -2.5 log10(A(x,y)/A(y,x))
from the frame's AREA extension.
Per star:   dfix_star = -2.5 log10(mean over frames of f_new / f_old).

usage: python closure.py BAND [NMAX]  ->  frames_<band>.ecsv, stars_<band>.ecsv
"""
import glob
import os
import sys
import warnings

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from scipy.spatial import cKDTree
from stpsf.utils import to_griddedpsfmodel

from jwst_gc_pipeline.photometry.psf_grid_io import load_stpsf_grid

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an  # noqa: E402

warnings.simplefilter('ignore')
band = sys.argv[1]
NMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
OUT = f'{an.Q}/gridfix'
PSFDIR = '/orange/adamginsburg/jwst/wd2/psfs'
BOXES = (5, 11)
RIN, ROUT = 12, 18
HW = ROUT + 3

an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')
P = np.load(f'{an.Q}/photomver/areapred_main2.npz')[band]
PT = np.load(f'{an.Q}/psfsum/predT_main2.npz')[band]
ref = A.ref[band]
lo, hi = an.ZPWIN[band]
ok = (A.matched & np.isfinite(P) & np.isfinite(PT) & np.isfinite(A.dm(band)) & ~A.rep[band]
      & ~A.sat[band] & (ref >= lo) & (ref < hi))
sel = np.nonzero(ok)[0]
if len(sel) > NMAX:
    sel = np.sort(np.random.default_rng(1).choice(sel, NMAX, replace=False))
print(band, 'ok', ok.sum(), 'used', len(sel), flush=True)
psky = SkyCoord(A.cat[f'skycoord_f{band.lower()}'])[np.where(A.idx >= 0, A.idx, 0)]

tree_dir = f'{an.Q}/tree_main2/F{band}'
cats = {}
for fn in glob.glob(f'{tree_dir}/f{band.lower()}_*_daophot_basic.fits'):
    t = Table.read(fn)
    cats[os.path.basename(t.meta['FILENAME'])] = t
frames = sorted(glob.glob(f'{tree_dir}/pipeline/jw03523005001_*_align_o005_crf.fits'))
yy, xx = np.mgrid[-HW:HW + 1, -HW:HW + 1]
grids = {}


def grid_pair(det):
    d = det.lower().replace('along', 'a5').replace('blong', 'b5')
    if d not in grids:
        fn = f'{PSFDIR}/nircam_{d}_f{band.lower()}_fovp101_samp2_npsf16.fits'
        grids[d] = (to_griddedpsfmodel(fn), load_stpsf_grid(fn))
    return grids[d]


def bkg(img, good, cx, cy):
    r = np.hypot(xx - cx, yy - cy)
    m = (r >= RIN) & (r <= ROUT) & good
    if m.sum() < 50:
        return np.nan
    return sigma_clipped_stats(img[m], sigma=3, maxiters=5)[1]


rows = []
for fn in frames:
    with fits.open(fn) as fh:
        sci = np.asarray(fh['SCI'].data, float)
        err = np.asarray(fh['ERR'].data, float)
        dq = np.asarray(fh['DQ'].data)
        area = np.asarray(fh['AREA'].data, float)
        h = fh['SCI'].header
        det = fh[0].header['DETECTOR']
    old, new = grid_pair(det)
    w = WCS(h)
    ny, nx = sci.shape
    x, y = w.world_to_pixel(psky[sel])
    cat = cats.get(os.path.basename(fn))
    if cat is not None:
        cx_, cy_ = np.asarray(cat['x_fit'], float), np.asarray(cat['y_fit'], float)
        ct = cKDTree(np.c_[cx_, cy_])
    bad = ((dq & 1) > 0) | ~np.isfinite(sci) | ~np.isfinite(err) | (err <= 0)
    for k, i in enumerate(sel):
        if not (np.isfinite(x[k]) and HW + 1 < x[k] < nx - HW - 1 and HW + 1 < y[k] < ny - HW - 1):
            continue
        px, py, src = x[k], y[k], 0
        if cat is not None:
            d, j = ct.query([x[k], y[k]])
            if d < 0.5:
                px, py, src = cx_[j], cy_[j], 1
        ix, iy = int(round(px)), int(round(py))
        sl = (slice(iy - HW, iy + HW + 1), slice(ix - HW, ix + HW + 1))
        g = ~bad[sl]
        cxl, cyl = px - ix, py - iy
        bk = bkg(np.where(g, sci[sl], 0.0), g, cxl, cyl)
        if not np.isfinite(bk):
            continue
        rec = dict(i=int(i), frame=os.path.basename(fn), det=det.lower(), src=src, x=px, y=py,
                   lA=2.5 * np.log10(area[iy, ix] / area[ix, iy]))
        good = True
        for box in BOXES:
            hb = box // 2
            m = (np.abs(xx) <= hb) & (np.abs(yy) <= hb)
            mm = m & g
            if mm.sum() < 0.8 * m.sum():
                good = False
                break
            dd = (sci[sl] - bk)[mm]
            ww = err[sl][mm] ** -2
            for tag, gr in (('old', old), ('new', new)):
                p = gr.evaluate(xx[mm] + ix, yy[mm] + iy, 1.0, px, py)
                rec[f'f{tag}{box}'] = np.sum(dd * p * ww) / np.sum(p * p * ww)
        if good:
            rows.append(rec)
F = Table(rows)
for box in BOXES:
    F[f'dfix{box}'] = -2.5 * np.log10(F[f'fnew{box}'] / F[f'fold{box}'])
F.write(f'{OUT}/frames_{band}.ecsv', overwrite=True)

ii = np.unique(F['i'])
S = Table({'i': ii})
for box in BOXES:
    r = F[f'fnew{box}'] / F[f'fold{box}']
    S[f'dfix{box}'] = [-2.5 * np.log10(np.mean(r[F['i'] == j])) for j in ii]
S['nfr'] = [np.sum(F['i'] == j) for j in ii]
S['dm'] = A.dm(band)[ii]
S['pred'] = P[ii]
S['predT'] = PT[ii]
S['det'] = np.load(f'{an.Q}/psfsum/predT_main2.npz')[band + '_det'][ii]
S['ref'] = ref[ii]
S.write(f'{OUT}/stars_{band}.ecsv', overwrite=True)
print(band, 'frame rows', len(F), 'stars', len(S), flush=True)
