"""Within-detector scale and rotation of each band's distortion ref relative
to F212N, and the per-detector centre offset that the same scale and
rotation would give about the module centre.

For each detector a 4-parameter similarity (scale s, rotation r, shift) is
fit to the V2/V3 positions of the band's model against F212N's on a pixel
grid.  If the band's optics magnify/rotate the whole module field by (s, r),
the detector centres move by s*R + r*z(R) about the module centre, where R
is the detector-centre vector from the module centre; the refs keep every
centre at the F210M SIAF V2Ref/V3Ref, so this part is missing from the WCS.
Predictions are printed in sky mas (via the F212N crf v2v3->world transform)
next to the measured residual from solve_m6.txt."""
import glob
import os
import re
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
        return af.tree['model']


def measured(band):
    txt = open('solve_m6.txt').read()
    m = re.search(rf'### wd2 {band.lower()} - f212n.*?\n(.*?)within', txt, re.S)
    out = {}
    if m:
        for line in m.group(1).splitlines():
            p = line.split()
            if p and p[0] in DETS:
                out[p[0]] = (float(p[2]), float(p[3]))
    return out


g = np.linspace(100, 1947, 9)
GX, GY = np.meshgrid(g, g)
X, Y = GX.ravel(), GY.ravel()

anc = {}
for det in DETS:
    p = crf(ANCHOR, det)
    mod = dist_model(p)
    with datamodels.open(p) as dm:
        t = dm.meta.wcs.get_transform('v2v3', 'world')
        dec0 = dm.meta.wcs(1023.5, 1023.5)[1]
    anc[det] = (mod, t, np.cos(np.deg2rad(dec0)), mod(1023.5, 1023.5))


def to_sky(det, v2, v3, dv2, dv3):
    _, t, c, _ = anc[det]
    r0, d0 = t(v2, v3)
    r1, d1 = t(v2 + dv2, v3 + dv3)
    return (r1 - r0) * c * 3.6e6, (d1 - d0) * 3.6e6


for band in BANDS:
    meas = measured(band)
    fits_ = {}
    for det in DETS:
        p = crf(band, det)
        if p is None:
            continue
        mb = dist_model(p)
        ma = anc[det][0]
        v2a, v3a = ma(X, Y)
        v2b, v3b = mb(X, Y)
        # similarity fit: (v2b, v3b) = (1+s) Rot(r) (v2a, v3a) + shift, centred on the anchor centre
        c2, c3 = anc[det][3]
        za = (v2a - c2) + 1j * (v3a - c3)
        zb = (v2b - c2) + 1j * (v3b - c3)
        A = np.vstack([za, np.ones_like(za)]).T
        coef, *_ = np.linalg.lstsq(A, zb, rcond=None)
        k = coef[0]
        s, r = abs(k) - 1, np.angle(k)
        res = zb - A @ coef
        fits_[det] = (s, r, np.sqrt(np.mean(abs(res) ** 2)) * 1e3)
    print(f'\n### {band} - {ANCHOR}')
    print('det    scale(ppm)  rot(arcsec)  resid-after-sim(mas) | predicted centre offset (sky mas) | measured (solve_m6)')
    for mod_ in 'ab':
        ds = [d for d in fits_ if d[3] == mod_]
        cen = np.array([anc[d][3] for d in ds])
        mc = cen.mean(axis=0)
        sm = np.mean([fits_[d][0] for d in ds])
        rm = np.mean([fits_[d][1] for d in ds])
        pred = {}
        for d, cc in zip(ds, cen):
            Rv = (cc[0] - mc[0]) + 1j * (cc[1] - mc[1])
            dz = (sm + 1j * rm) * Rv     # small-angle similarity about the module centre
            pred[d] = to_sky(d, cc[0], cc[1], dz.real, dz.imag)
        pm = np.mean([pred[d] for d in ds], axis=0)
        for d in ds:
            s, r, rr = fits_[d]
            pr = np.array(pred[d]) - pm
            ms = meas.get(d, (np.nan, np.nan))
            print(f'{d}  {s * 1e6:+9.1f}  {np.rad2deg(r) * 3600:+9.3f}   {rr:8.2f}          '
                  f'({pr[0]:+7.2f},{pr[1]:+7.2f})              ({ms[0]:+7.2f},{ms[1]:+7.2f})')
        print(f'  module {mod_.upper()} mean scale {sm * 1e6:+.1f} ppm, rot {np.rad2deg(rm) * 3600:+.3f} arcsec')
