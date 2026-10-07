"""F150W - F212N per-detector centre offsets vs the inter-detector DVA shift
that ``reduction/dva_correction.py`` applied to F150W only (DVACORR=True).

Prediction per detector: (VA_SCALE - 1) * (ref_d - V1) on the sky, from the
F150W exposure-00001 crf SCI header (RA_REF/DEC_REF/RA_V1/DEC_V1/VA_SCALE),
module mean removed.  Measured: c0 of datascale(_field).py, module mean
removed.  Also reports the residual after subtracting the prediction."""
import glob
import re
import sys

import numpy as np
from astropy.io import fits

DETS = ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
FIELDS = {
    'wd2': ('/orange/adamginsburg/jwst/wd2/F150W/pipeline/jw03523005001_*_00001_{d}_align_o005_crf.fits',
            'datascale_other.txt'),
    'wd1': ('/orange/adamginsburg/jwst/wd1/F150W/pipeline/jw01905001001_*_00001_{d}_destreak_o001_crf.fits',
            'datascale_wd1_other.txt'),
}


def c0(path, band='f150w'):
    txt = open(path).read()
    m = re.search(rf'### {band} - f212n.*?\n.*?\n(.*?)(?:\n\s*\n|\n###|\Z)', txt, re.S)
    return {p[0]: np.array([float(p[2]), float(p[3])])
            for p in (ln.split() for ln in m.group(1).splitlines()) if p and p[0] in DETS}


def demean(v):
    out = {}
    for mod in 'ab':
        ds = [d for d in DETS if d[3] == mod]
        mu = np.mean([v[d] for d in ds], axis=0)
        out.update({d: v[d] - mu for d in ds})
    return out


def rms(v):
    return float(np.sqrt(np.mean([x @ x for x in v.values()])))


for field, (pat, src) in FIELDS.items():
    pred = {}
    for d in DETS:
        h = fits.getheader(sorted(glob.glob(pat.format(d=d)))[0], 'SCI')
        s = float(h['VA_SCALE']) - 1.0
        cosd = np.cos(np.radians(h['DEC_REF']))
        pred[d] = s * np.array([(h['RA_REF'] - h['RA_V1']) * cosd, h['DEC_REF'] - h['DEC_V1']]) * 3.6e6
    pred = demean(pred)
    meas = demean(c0(src))
    res = {d: meas[d] - pred[d] for d in DETS}
    print(f'### {field}  VA_SCALE-1 = {s * 1e6:+.1f} ppm')
    print('  det     meas (mas)        pred (mas)        meas-pred')
    for d in DETS:
        print(f'  {d}  ({meas[d][0]:+5.2f},{meas[d][1]:+5.2f})  ({pred[d][0]:+5.2f},{pred[d][1]:+5.2f})'
              f'  ({res[d][0]:+5.2f},{res[d][1]:+5.2f})')
    # best-fit multiplier k on the prediction
    P = np.concatenate([pred[d] for d in DETS]); M = np.concatenate([meas[d] for d in DETS])
    k = float(P @ M / (P @ P))
    print(f'  rms meas {rms(meas):.2f}  pred {rms(pred):.2f}  meas-pred {rms(res):.2f} mas;'
          f'  best-fit k = {k:+.2f}')
