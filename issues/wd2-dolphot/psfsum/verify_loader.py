"""Per-plane PSF sum vs SIAF pixel area for the stpsf loader and the
psf_grid_io loader, on every 4x4 fovp101 grid in the wd2 PSF store.  A plane
assigned to the position it was computed at has sum proportional to
1/area(x, y): slope of log(sum) on log(area) near -1."""
import glob
import os
import sys
import warnings

import numpy as np
from stpsf.utils import to_griddedpsfmodel

from jwst_gc_pipeline.photometry.psf_grid_io import load_stpsf_grid
from siafarea import siaf_area

warnings.simplefilter('ignore')
P = '/orange/adamginsburg/jwst/wd2/psfs'


def slope(grid, det):
    xy = np.asarray(grid.grid_xypos)
    s = grid.data.sum(axis=(1, 2)) / float(np.atleast_1d(grid.oversampling)[0]) ** 2
    a = siaf_area(det, xy[:, 0], xy[:, 1])
    return np.polyfit(np.log(a), np.log(s), 1)[0], np.corrcoef(np.log(a), np.log(s))[0, 1]


rows = []
for fn in sorted(glob.glob(f'{P}/nircam_nrc*_f*_fovp101_samp2_npsf16.fits')):
    det = os.path.basename(fn).split('_')[1]
    filt = os.path.basename(fn).split('_')[2]
    old, new = to_griddedpsfmodel(fn), load_stpsf_grid(fn)
    so, co = slope(old, det)
    sn, cn = slope(new, det)
    rows.append((filt, det, so, co, sn, cn))
print('| filter | det | slope stpsf loader | corr | slope psf_grid_io | corr |')
print('|---|---|---|---|---|---|')
for r in rows:
    print(f'| {r[0].upper()} | {r[1]} | {r[2]:+.2f} | {r[3]:+.3f} | {r[4]:+.2f} | {r[5]:+.3f} |')
r = np.array([x[2:] for x in rows])
print(f'\nN={len(rows)}  median slope stpsf loader {np.median(r[:,0]):+.2f}  '
      f'psf_grid_io {np.median(r[:,2]):+.2f}; median corr {np.median(r[:,1]):+.3f} / {np.median(r[:,3]):+.3f}')
