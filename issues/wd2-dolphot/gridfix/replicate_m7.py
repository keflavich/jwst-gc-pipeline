"""Rebuild the main2 m7 per-frame fit input step by step and find which step makes the
F150W main2 flux depart from the closure forced fit.

Images (all on the crf grid):
  d0 = crf SCI                                    (closure.py input)
  d1 = d0 - m7 satstar model                      (satstar removal)
  d2 = d1 - reprojected m6 smoothed bg            (main2 m7 input, up to zero-pixel handling)
Backgrounds:
  ann = sigma-clipped median in 12-18 px          (closure.py)
  loc = photutils LocalBackground(6, 10)          (main2)
Flux: weighted linear solve with the old (transposed) grid at the main2 x_fit, y_fit, 5x5
box, weights ERR^-2 (closure.py and PSFPhotometry use the same ERR).

For each variant: robust std of -2.5 log10(flux_variant / flux_fit_main2) and the slope of
that quantity on lA.

usage: python replicate_m7.py BAND DET [EXP ...]
"""
import glob
import os
import sys
import warnings

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from photutils.background import LocalBackground
from photutils.psf import PSFPhotometry
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.table import QTable
from reproject import reproject_interp
from scipy.spatial import cKDTree
from stpsf.utils import to_griddedpsfmodel

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an  # noqa: E402

warnings.simplefilter('ignore')
band, det = sys.argv[1], sys.argv[2]
exps = sys.argv[3:] or ['00001']
D = f'{an.Q}/gridfix'
T = f'{an.Q}/tree_main2/F{band}'
PSFDIR = '/orange/adamginsburg/jwst/wd2/psfs'
HW = 21
yy, xx = np.mgrid[-HW:HW + 1, -HW:HW + 1]


def mad(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def slope(lA, d):
    k = np.isfinite(d) & np.isfinite(lA)
    lA, d = lA[k], d[k]
    g = np.abs(d - np.median(d)) <= max(5 * mad(d), 1e-6)
    return np.polyfit(lA[g], d[g], 1)[0]


F = Table.read(f'{D}/frames_{band}.ecsv')
F = F[(F['src'] == 1) & (F['det'] == det)]
grid = to_griddedpsfmodel(f'{PSFDIR}/nircam_{det}_f{band.lower()}_fovp101_samp2_npsf16.fits')
bgfn = glob.glob(f'{T}/pipeline/jw03523-o005_t001_nircam_clear-f{band.lower()}-merged_resbgsub_m6_'
                 'daophot_basic_mergedcat_residual_smoothed_bg_i2d.fits')[0]
with fits.open(bgfn) as bh:
    bhdu = bh['SCI'] if 'SCI' in [h.name for h in bh] else bh[0]
    bg_all, bg_wcs = bhdu.data.astype(float), WCS(bhdu.header)
lb = LocalBackground(6, 10)

res = {}
for exp in exps:
    fr = [f for f in np.unique(F['frame']) if f'_{exp}_{det}_' in f]
    if not fr:
        continue
    fr = fr[0]
    G = F[F['frame'] == fr]
    cat = [t for t in (Table.read(fn) for fn in glob.glob(f'{T}/f{band.lower()}_{det}_*_daophot_basic.fits'))
           if os.path.basename(t.meta['FILENAME']) == fr][0]
    ct = cKDTree(np.c_[np.asarray(cat['x_fit'], float), np.asarray(cat['y_fit'], float)])
    d, j = ct.query(np.c_[G['x'], G['y']])
    ok = d < 1e-3
    G, j = G[ok], j[ok]
    fmain = np.asarray(cat['flux_fit'], float)[j]
    lbk_main = np.asarray(cat['local_bkg'], float)[j]
    with fits.open(f'{T}/pipeline/{fr}') as fh:
        sci = np.asarray(fh['SCI'].data, float)
        err = np.asarray(fh['ERR'].data, float)
        dq = np.asarray(fh['DQ'].data)
        h = fh['SCI'].header
    ssm = fits.getdata(f'{T}/pipeline/{fr.replace(".fits", "")}_resbgsub_m7_satstar_model.fits').astype(float)
    bgr, _ = reproject_interp((bg_all, bg_wcs), WCS(h), shape_out=sci.shape)
    bgr = np.where(np.isfinite(bgr), bgr, 0.0)
    zeros = sci == 0
    d1 = sci - ssm
    d2 = d1 - bgr
    d1[zeros] = 0
    d2[zeros] = 0
    imgs = {'d0': sci, 'd1': d1, 'd2': d2}
    bad = ((dq & 1) > 0) | ~np.isfinite(sci) | ~np.isfinite(err) | (err <= 0) | zeros
    xy = np.c_[G['x'], G['y']]
    out = {}
    for name, img in imgs.items():
        im_m = np.where(bad, np.nan, img)
        loc = lb(np.where(bad, 0.0, img), xy[:, 0], xy[:, 1], mask=bad)
        for bk in ('ann', 'loc'):
            fl = np.full(len(G), np.nan)
            for i, (px, py) in enumerate(xy):
                ix, iy = int(round(px)), int(round(py))
                sl = (slice(iy - HW, iy + HW + 1), slice(ix - HW, ix + HW + 1))
                g = ~bad[sl]
                if bk == 'ann':
                    r = np.hypot(xx - (px - ix), yy - (py - iy))
                    m = (r >= 12) & (r <= 18) & g
                    b0 = sigma_clipped_stats(im_m[sl][m], sigma=3, maxiters=5)[1]
                else:
                    b0 = loc[i]
                mm = (np.abs(xx) <= 2) & (np.abs(yy) <= 2) & g
                p = grid.evaluate(xx[mm] + ix, yy[mm] + iy, 1.0, px, py)
                w = err[sl][mm] ** -2
                fl[i] = np.sum((img[sl][mm] - b0) * p * w) / np.sum(p * p * w)
            out[f'{name}_{bk}'] = fl
        if name == 'd2':
            out['loc_d2'] = loc
            ip = QTable()
            ip['x'] = np.asarray(cat['x_init'], float)[j]
            ip['y'] = np.asarray(cat['y_init'], float)[j]
            ip['flux'] = np.asarray(cat['flux_init'], float)[j]
            gm = grid.copy()
            gm.flux.min = 0
            phot = PSFPhotometry(gm, (5, 5), finder=None, grouper=None, fitter=LevMarLSQFitter(),
                                 localbkg_estimator=LocalBackground(6, 10),
                                 aperture_radius=float(cat.meta.get('APERTURE', 3.256)))
            r = phot(img.astype('float32'), mask=bad, init_params=ip, error=np.where(bad, 1e10, err))
            out['lm_d2'] = np.asarray(r['flux_fit'], float)
            out['lm_dx'] = np.asarray(r['x_fit'], float) - xy[:, 0]
            out['lm_dy'] = np.asarray(r['y_fit'], float) - xy[:, 1]
    o = res.setdefault('all', dict(lA=[], fold=[], fmain=[], lbk_main=[], **{k: [] for k in out}))
    o['lA'] += list(G['lA']); o['fold'] += list(G['fold5']); o['fmain'] += list(fmain)
    o['lbk_main'] += list(lbk_main)
    for k, v in out.items():
        o[k] += list(v)
    print(f'{fr}: {len(G)} stars, smoothed bg median {np.median(bgr):.3f}, '
          f'satstar model sum {np.nansum(ssm):.3e}, max {np.nanmax(ssm):.3g}', flush=True)

o = {k: np.array(v) for k, v in res['all'].items()}
print(f'F{band} {det}: variant vs main2 flux_fit, -2.5 log10(f_variant / f_main2)  (N={len(o["lA"])})')
print('  variant      rstd     slope on lA   median')
for k in ['fold', 'd0_ann', 'd0_loc', 'd1_ann', 'd1_loc', 'd2_ann', 'd2_loc', 'lm_d2']:
    v = -2.5 * np.log10(o[k] / o['fmain'])
    print(f'  {k:10s}  {mad(v):.4f}   {slope(o["lA"], v):+.2f}   {np.nanmedian(v):+.4f}')
dl = o['loc_d2'] - o['lbk_main']
print(f'  local_bkg(d2) - main2 local_bkg: median {np.nanmedian(dl):+.4f}, rstd {mad(dl):.4f}')
dx, dy = o['lm_dx'], o['lm_dy']
print(f'  LevMar refit position - main2 x_fit,y_fit: median |dr| {np.nanmedian(np.hypot(dx, dy)):.4f} px, '
      f'p90 {np.nanpercentile(np.hypot(dx, dy), 90):.4f} px')
