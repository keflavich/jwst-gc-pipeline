"""Median (x_fit - x_init, y_fit - y_init) per detector for bright free fits
(snr > 50, forced_refit False) in m7 per-frame catalogs, exposure 00001.
Forced-refit rows stay at (x_init, y_init), so this is the seed error they
inherit.  Radial component: projection on the detector-centre -> module-centre
direction is not computed here; the per-detector vector is printed."""
import glob
import os
import sys

import numpy as np
from astropy.table import Table

ROOT = sys.argv[1] if len(sys.argv) > 1 else '/orange/adamginsburg/jwst/wd2'
BANDS = ['F150W', 'F162M', 'F164N', 'F182M', 'F187N', 'F200W', 'F212N']
DETS = ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
print(f'root {ROOT}')
for b in BANDS:
    cells = []
    for d in DETS:
        fs = sorted(glob.glob(f'{ROOT}/{b}/{b.lower()}_{d}_visit001_vgroup*_exp00001_resbgsub_m7_daophot_basic.fits'))
        if not fs:
            cells.append(f'{d[3:]}   n/a        ')
            continue
        t = Table.read(fs[0])
        snr = np.asarray(t['flux_fit'], float) / np.asarray(t['flux_err'], float)
        k = np.isfinite(snr) & (snr > 50) & ~np.asarray(t['forced_refit'], bool)
        dx = np.asarray(t['x_fit'], float)[k] - np.asarray(t['x_init'], float)[k]
        dy = np.asarray(t['y_fit'], float)[k] - np.asarray(t['y_init'], float)[k]
        cells.append(f'{d[3:]} ({np.nanmedian(dx):+.3f},{np.nanmedian(dy):+.3f}) n={k.sum()}')
    print(f'{b}: ' + '  '.join(cells))
