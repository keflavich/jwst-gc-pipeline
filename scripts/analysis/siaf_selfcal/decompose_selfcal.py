"""Decompose per-detector network shifts into rotation+scale (per band) + residual."""
import sys, glob, re, os
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord

B = '/blue/adamginsburg/adamginsburg/jwst/brick'
sol = np.load(sys.argv[1])
detkeys = [eval(k) for k in sol['detkeys']]
dra, dde = sol['det_ra'], sol['det_de']

# detector centers: median skycoord of one exposure's catalog per (band,det)
cen = {}
for band, det in detkeys:
    stage = 'm1' if band in ('f200w', 'f115w') else 'resbgsub_m7'
    g = sorted(glob.glob(f'{B}/{band.upper()}/{band}_{det}_visit*_vgroup*_exp*_{stage}_daophot_basic.fits'))
    t = Table.read(g[0])
    sc = SkyCoord(t['skycoord_centroid'])
    cen[(band, det)] = (np.median(sc.ra.deg), np.median(sc.dec.deg))

bands = sorted({b for b, d in detkeys})
for band in bands:
    idx = [i for i, k in enumerate(detkeys) if k[0] == band]
    if len(idx) < 3:
        print(f'{band}: <3 detectors, skip rotation/scale fit'); continue
    cens = np.array([cen[detkeys[i]] for i in idx])
    c0 = cens.mean(axis=0); cosd = np.cos(np.radians(c0[1]))
    X = (cens[:, 0]-c0[0])*cosd*3600; Y = (cens[:, 1]-c0[1])*3600
    n = len(idx)
    A = np.zeros((2*n, 4))
    A[:n, 0] = -Y; A[:n, 1] = X; A[:n, 2] = 1
    A[n:, 0] = X;  A[n:, 1] = Y; A[n:, 3] = 1
    y = np.concatenate([dra[idx], dde[idx]])
    p, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A@p
    print(f'\n=== {band}: rotation {p[0]*1e-3*206265:.3f} arcsec roll, scale {p[1]*1e-3:.2e}, '
          f'residual rms ({res[:n].std():.2f}, {res[n:].std():.2f}) mas ===')
    for j, i in enumerate(idx):
        print(f'  {detkeys[i][1]:8s}: resid ({res[j]:+6.2f},{res[n+j]:+6.2f})  raw ({dra[i]:+7.2f},{dde[i]:+7.2f})')
