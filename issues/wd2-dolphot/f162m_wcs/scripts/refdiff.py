"""Predicted per-detector sky offset (filter - F212N) from the CRDS distortion
reference files alone, to compare with the measured residuals in solve_m6.txt.

For each detector, both bands' distortion models map the same detector pixels
to V2/V3; the V2/V3 difference is carried to the sky through the F212N crf
v2v3 -> world transform of exposure 00001, and the per-module mean is removed
(as the solver removes a per-(exposure, module) median)."""
import glob
import os
import sys

import asdf
import numpy as np
from astropy.io import fits
from stdatamodels.jwst import datamodels

R = '/orange/adamginsburg/jwst/wd2'
CRDS = '/orange/adamginsburg/jwst/crds/references/jwst/nircam/'
DETS = ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
ANCHOR = 'F212N'
BANDS = sys.argv[1:] or ['F162M', 'F164N', 'F150W', 'F115W', 'F182M', 'F187N', 'F200W']


def crf(band, det):
    fs = sorted(glob.glob(f'{R}/{band}/pipeline/jw03523005001_*_00001_{det}_align_o005_crf.fits'))
    return fs[0] if fs else None


def dist_model(path):
    ref = fits.getheader(path, 0)['R_DISTOR'].replace('crds://', '')
    with asdf.open(CRDS + os.path.basename(ref), lazy_load=False, memmap=False) as af:
        return os.path.basename(ref), af.tree['model']


# pixel grid: detector centre plus a 5x5 grid for the within-detector spread
g = np.linspace(100, 1947, 5)
GX, GY = np.meshgrid(g, g)
X = np.concatenate([[1023.5], GX.ravel()])
Y = np.concatenate([[1023.5], GY.ravel()])

anchor = {}
for det in DETS:
    p = crf(ANCHOR, det)
    name, mod = dist_model(p)
    with datamodels.open(p) as dm:
        w = dm.meta.wcs
        t = w.get_transform('v2v3', 'world')
        dec0 = w(1023.5, 1023.5)[1]
    anchor[det] = (name, mod, t, np.cos(np.deg2rad(dec0)))

print(f'anchor {ANCHOR}: ' + ' '.join(f'{d}:{anchor[d][0][-9:-5]}' for d in DETS))
for band in BANDS:
    rows = {}
    for det in DETS:
        p = crf(band, det)
        if p is None:
            continue
        name, mod = dist_model(p)
        aname, amod, t, c = anchor[det]
        v2b, v3b = mod(X, Y)
        v2a, v3a = amod(X, Y)
        ra_a, de_a = t(v2a, v3a)
        ra_b, de_b = t(v2b, v3b)
        dra = (np.asarray(ra_b) - np.asarray(ra_a)) * c * 3.6e6
        dde = (np.asarray(de_b) - np.asarray(de_a)) * 3.6e6
        rows[det] = (name, dra, dde)
    print(f'\n### {band} - {ANCHOR}: distortion-ref prediction [mas, sky], per-module mean (at centres) removed')
    print('det     ref            dRA*     dDec   grid-rms(after rigid)')
    for mod_ in 'ab':
        ds = [d for d in rows if d[3] == mod_]
        mra = np.mean([rows[d][1][0] for d in ds])
        mde = np.mean([rows[d][2][0] for d in ds])
        for d in ds:
            name, dra, dde = rows[d]
            spread = np.sqrt(np.mean((dra[1:] - dra[0]) ** 2 + (dde[1:] - dde[0]) ** 2))
            print(f'{d}  {name[-14:-5]}  {dra[0] - mra:+8.2f} {dde[0] - mde:+8.2f}   {spread:6.2f}')
        print(f'  module {mod_.upper()} mean ({mra:+.2f}, {mde:+.2f})')
