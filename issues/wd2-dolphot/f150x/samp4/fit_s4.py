"""samp4 experiment variant of gridfix/recentre_ab.py: grids c=cached samp2, a=new samp2 control, b=new samp4; seed box only.

Original doc: Per-frame A/B of the main2 m7 PSF fit: PSF grid loader x fit-box centre.

The fit reproduces main2's m7 call (replicate_m7.py variant lm_d2, which matches
main2 flux_fit exactly): PSFPhotometry, 5x5 box, finder/grouper None, LevMarLSQFitter,
LocalBackground(6, 10), flux >= 0, on d2 = crf - m7 satstar model - reprojected m6
smoothed background.  photutils extracts the fit box around the initial position, so
the init position sets the box centre.

Variants (grid, init):
  oi  old loader (to_griddedpsfmodel), init x_init/y_init        (= main2)
  ni  new loader (load_stpsf_grid),    init x_init/y_init        (grid fix only)
  of  old loader,                      init main2 x_fit/y_fit    (recentred box only)
  nf  new loader,                      init main2 x_fit/y_fit    (both)

Stars: closure rows (frames_<band>.ecsv, src == 1) on detector DET.

usage: python recentre_ab.py BAND DET   ->  recentre/rc_<band>_<det>.ecsv
"""
import glob
import os
import sys
import warnings

import numpy as np
from astropy.io import fits
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.table import QTable, Table, vstack
from astropy.wcs import WCS
from photutils.background import LocalBackground
from photutils.psf import PSFPhotometry
from reproject import reproject_interp
from scipy.spatial import cKDTree

from jwst_gc_pipeline.photometry.psf_grid_io import load_stpsf_grid

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an  # noqa: E402

warnings.simplefilter('ignore')
band, det = sys.argv[1], sys.argv[2]
D = f'{an.Q}/gridfix'
SD = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/f150x/samp4'
T = f'{an.Q}/tree_main2/F{band}'
PSFDIR = '/orange/adamginsburg/jwst/wd2/psfs'
os.makedirs(f'{SD}/fits', exist_ok=True)

F = Table.read(f'{D}/frames_{band}.ecsv')
F = F[(F['src'] == 1) & (F['det'] == det)]
psf_fn = f'{PSFDIR}/nircam_{det}_f{band.lower()}_fovp101_samp2_npsf16.fits'
gfn = lambda ov: f'{SD}/grids/nircam_{det}_f{band.lower()}_fovp101_samp{ov}_npsf16.fits'
grids = {'c': load_stpsf_grid(psf_fn), 'a': load_stpsf_grid(gfn(2)), 'b': load_stpsf_grid(gfn(4))}
bgfn = glob.glob(f'{T}/pipeline/jw03523-o005_t001_nircam_clear-f{band.lower()}-merged_resbgsub_m6_'
                 'daophot_basic_mergedcat_residual_smoothed_bg_i2d.fits')[0]
with fits.open(bgfn) as bh:
    bhdu = bh['SCI'] if 'SCI' in [h.name for h in bh] else bh[0]
    bg_all, bg_wcs = bhdu.data.astype(float), WCS(bhdu.header)
cats = {}
for fn in glob.glob(f'{T}/f{band.lower()}_{det}_*_daophot_basic.fits'):
    t = Table.read(fn)
    cats[os.path.basename(t.meta['FILENAME'])] = t

out = []
for fr in np.unique(F['frame']):
    G = F[F['frame'] == fr]
    cat = cats[fr]
    ct = cKDTree(np.c_[np.asarray(cat['x_fit'], float), np.asarray(cat['y_fit'], float)])
    d, j = ct.query(np.c_[G['x'], G['y']])
    ok = d < 1e-3
    G, j = G[ok], j[ok]
    with fits.open(f'{T}/pipeline/{fr}') as fh:
        sci = np.asarray(fh['SCI'].data, float)
        err = np.asarray(fh['ERR'].data, float)
        dq = np.asarray(fh['DQ'].data)
        h = fh['SCI'].header
    ssm = fits.getdata(f'{T}/pipeline/{fr.replace(".fits", "")}_resbgsub_m7_satstar_model.fits').astype(float)
    bgr, _ = reproject_interp((bg_all, bg_wcs), WCS(h), shape_out=sci.shape)
    bgr = np.where(np.isfinite(bgr), bgr, 0.0)
    zeros = sci == 0
    d2 = sci - ssm - bgr
    d2[zeros] = 0
    bad = ((dq & 1) > 0) | ~np.isfinite(sci) | ~np.isfinite(err) | (err <= 0) | zeros
    R = Table()
    R['i'] = G['i']
    R['frame'] = G['frame']
    R['det'] = G['det']
    R['lA'] = G['lA']
    R['x_init'] = np.asarray(cat['x_init'], float)[j]
    R['y_init'] = np.asarray(cat['y_init'], float)[j]
    R['x_fit'] = np.asarray(cat['x_fit'], float)[j]
    R['y_fit'] = np.asarray(cat['y_fit'], float)[j]
    R['fmain'] = np.asarray(cat['flux_fit'], float)[j]
    R['forced'] = np.asarray(cat['forced_refit']).astype(bool)[j] if 'forced_refit' in cat.colnames \
        else np.zeros(len(G), bool)
    for g in ('c', 'a', 'b'):
        for p in ('i',):
            ip = QTable()
            ip['x'] = R['x_init'] if p == 'i' else R['x_fit']
            ip['y'] = R['y_init'] if p == 'i' else R['y_fit']
            ip['flux'] = np.asarray(cat['flux_init'], float)[j]
            gm = grids[g].copy()
            gm.flux.min = 0
            phot = PSFPhotometry(gm, (5, 5), finder=None, grouper=None, fitter=LevMarLSQFitter(),
                                 localbkg_estimator=LocalBackground(6, 10),
                                 aperture_radius=float(cat.meta.get('APERTURE', 3.256)))
            r = phot(d2.astype('float32'), mask=bad, init_params=ip, error=np.where(bad, 1e10, err))
            R[f'f_{g}{p}'] = np.asarray(r['flux_fit'], float)
            R[f'x_{g}{p}'] = np.asarray(r['x_fit'], float)
            R[f'y_{g}{p}'] = np.asarray(r['y_fit'], float)
    out.append(R)
    v = -2.5 * np.log10(R['f_ci'] / R['fmain'])
    print(f'{fr}: {len(R)} stars, ci vs main2 median {np.nanmedian(v):+.5f}, '
          f'max |d| {np.nanmax(np.abs(v)):.5f}', flush=True)

vstack(out).write(f'{SD}/fits/s4_{band}_{det}.ecsv', overwrite=True)
print('done', band, det)
