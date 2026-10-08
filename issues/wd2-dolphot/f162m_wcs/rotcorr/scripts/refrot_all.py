"""Rotation and scale of every SW CRDS distortion reference relative to the
F212N reference of the same detector (similarity fit of pixel->V2V3 on a grid
about the detector centre), resolved through jwst_1596.pmap."""
import os

import asdf
import crds
import numpy as np

DETS = ['NRCA1', 'NRCA2', 'NRCA3', 'NRCA4', 'NRCB1', 'NRCB2', 'NRCB3', 'NRCB4']
COMBOS = [(f, 'CLEAR') for f in ['F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F150W2', 'F182M',
                                 'F187N', 'F200W', 'F210M', 'F212N']] + [('F150W2', 'F162M'), ('F150W2', 'F164N')]
CR = '/orange/adamginsburg/jwst/crds/references/jwst/nircam/'
g = np.linspace(100, 1947, 9)
GX, GY = np.meshgrid(g, g)
X, Y = GX.ravel(), GY.ravel()


def ref(det, f, p):
    pars = {'META.EXPOSURE.TYPE': 'NRC_IMAGE', 'META.INSTRUMENT.DETECTOR': det,
            'META.INSTRUMENT.CHANNEL': 'SHORT', 'META.INSTRUMENT.PUPIL': p,
            'META.INSTRUMENT.FILTER': f, 'META.INSTRUMENT.NAME': 'NIRCAM',
            'META.OBSERVATION.DATE': '2023-06-01', 'META.OBSERVATION.TIME': '00:00:00'}
    return crds.getrecommendations(pars, reftypes=['distortion'], context='jwst_1596.pmap',
                                   observatory='jwst')['distortion']


def model(fn):
    with asdf.open(CR + fn, lazy_load=False, memmap=False) as af:
        return af.tree['model']


def sim(ma, mb):
    v2a, v3a = ma(X, Y)
    v2b, v3b = mb(X, Y)
    c2, c3 = ma(1023.5, 1023.5)
    za = (v2a - c2) + 1j * (v3a - c3)
    zb = (v2b - c2) + 1j * (v3b - c3)
    A = np.vstack([za, np.ones_like(za)]).T
    coef, *_ = np.linalg.lstsq(A, zb, rcond=None)
    return np.rad2deg(np.angle(coef[0])) * 3600, (abs(coef[0]) - 1) * 1e6


rows = {}
for det in DETS:
    fa = ref(det, 'F212N', 'CLEAR')
    ma = model(fa)
    for f, p in COMBOS:
        fb = ref(det, f, p)
        if not os.path.exists(CR + fb):
            rows[(det, f, p)] = (fb, np.nan, np.nan)
            continue
        r, s = sim(ma, model(fb))
        rows[(det, f, p)] = (fb, r, s)

print('rotation (arcsec) of each reference relative to F212N, v2+i*v3 sense')
print('filter/pupil    ' + ' '.join(f'{d:>7s}' for d in DETS))
for f, p in COMBOS:
    lab = f if p == 'CLEAR' else p
    print(f'{lab:14s}  ' + ' '.join(f'{rows[(d, f, p)][1]:+7.2f}' for d in DETS))
print('\nscale (ppm) relative to F212N')
for f, p in COMBOS:
    lab = f if p == 'CLEAR' else p
    print(f'{lab:14s}  ' + ' '.join(f'{rows[(d, f, p)][2]:+7.1f}' for d in DETS))
print('\nreference files')
for f, p in COMBOS:
    lab = f if p == 'CLEAR' else p
    print(f'{lab:14s}  ' + ' '.join(f'{rows[(d, f, p)][0][-9:-5]:>7s}' for d in DETS))
