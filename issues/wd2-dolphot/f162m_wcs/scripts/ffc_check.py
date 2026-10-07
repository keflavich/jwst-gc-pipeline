"""End-to-end sign check of FILTER_FRAME_CORRECTION on wd2.

For each SW detector of exposure 00001, the correction that
``apply_filter_frame_correction`` would apply is computed with the module's
own ``table_filter_offsets`` -> ``sky_shift_deg`` -> ``adjust_wcs`` chain, and
the resulting on-sky displacement of the detector centre is measured from the
GWCS before and after.  That displacement is added to the measured
per-detector (frame - F212N anchor) residual from datascale.py (explicit
pair arithmetic, module median removed), and the module-mean-removed rms is
compared before and after.

usage: python ffc_check.py BAND RESIDUAL_FILE TABLE [TABLE ...]
"""
import glob
import re
import sys

import numpy as np
import astropy.units as u
from astropy.io import fits
from stdatamodels.jwst import datamodels
from jwst.tweakreg.utils import adjust_wcs

from jwst_gc_pipeline.reduction import filter_frame_correction as ffc

R = '/orange/adamginsburg/jwst/wd2'
band, resid_file, tables = sys.argv[1].upper(), sys.argv[2], sys.argv[3:]


def measured(path, band):
    txt = open(path).read()
    m = re.search(rf'### {band.lower()} - f212n.*?\n.*?\n(.*?)(?:\n###|\Z)', txt, re.S)
    out = {}
    for line in m.group(1).splitlines():
        p = line.split()
        if p and re.fullmatch(r'nrc[ab][1-4]', p[0]):
            out[p[0].upper()] = np.array([float(p[2]), float(p[3])])
    return out


def applied_shift(det, offsets):
    crf = sorted(glob.glob(f'{R}/{band}/pipeline/jw03523005001_*_00001_{det.lower()}_align_o005_crf.fits'))[0]
    h = fits.getheader(crf, 'SCI')
    roll = float(h['ROLL_REF']) if 'ROLL_REF' in h else float(fits.getheader(crf, 0)['ROLL_REF'])
    with datamodels.open(crf) as dm:
        w = dm.meta.wcs
        ra0, de0 = w(1023.5, 1023.5)
        dra, ddec = ffc.sky_shift_deg(offsets, det, 'instrument', roll, float(de0))
        w2 = adjust_wcs(w, delta_ra=dra * u.deg, delta_dec=ddec * u.deg)
        ra1, de1 = w2(1023.5, 1023.5)
    c = np.cos(np.deg2rad(de0))
    return np.array([(ra1 - ra0) * c * 3.6e6, (de1 - de0) * 3.6e6]), roll


meas = measured(resid_file, band)
print(f'{band}: measured (frame - F212N anchor), module median removed, from {resid_file}')
for tpath in tables:
    tbl = ffc.load_filter_frame_table(tpath)
    frame, off = ffc.table_filter_offsets(tbl, band, 'F212N')
    print(f'\n table {tpath}  (frame={frame})')
    print('  det     measured(sky mas)    applied shift(sky mas)   after')
    before, after = [], []
    for mod in ('NRCA', 'NRCB'):
        ds = [d for d in sorted(meas) if d.startswith(mod) and d in off]
        b = np.array([meas[d] for d in ds])
        s = np.array([applied_shift(d, off)[0] for d in ds])
        b0 = b - b.mean(0)
        a0 = (b + s) - (b + s).mean(0)
        for d, bb, ss, aa in zip(ds, b0, s, a0):
            print(f'  {d}  ({bb[0]:+7.2f},{bb[1]:+7.2f})     ({ss[0]:+7.2f},{ss[1]:+7.2f})     ({aa[0]:+7.2f},{aa[1]:+7.2f})')
        before.append(b0)
        after.append(a0)
    rb = np.sqrt((np.concatenate(before) ** 2).sum(1).mean())
    ra = np.sqrt((np.concatenate(after) ** 2).sum(1).mean())
    print(f'  2-D rms before {rb:.2f} mas -> after {ra:.2f} mas  ({ra / rb:.2f}x)')
