"""Assign each matched_Q_main2 row to a SW detector group (nrcb1 / nrcb3 / other) per W band.
Satstar rows (Q_integ/nrcb3/rows_<band>.npz, S_starm + S_det) take priority (mode over rows); otherwise the exposure-1 crf WCS footprint.
Usage: nice -19 python -u groups.py  -> groups.npz"""
import glob
import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
m = Table.read('/orange/adamginsburg/jwst/wd2/dolphot_benchmark/matched_Q_main2.fits')
ra = np.asarray(m['RA'], float)
dec = np.asarray(m['DEC'], float)
GN = {'nrcb1': 0, 'nrcb3': 1}
out = {}
for band in ('F150W', 'F162M', 'F182M', 'F200W', 'F164N', 'F187N'):
    foot = np.full(len(m), 'none', dtype='<U6')
    nhit = np.zeros(len(m), int)
    first = np.full(len(m), '', dtype='<U6')
    for f in sorted(glob.glob(f'{Q}/tree_main2/{band}/pipeline/jw03523005001_*_00001_nrc*_align_o005_crf.fits')):
        det = f.split('_')[-4]
        if det.startswith('nrca') or det.startswith('nrcb'):
            if det.endswith('long'):
                continue
        with fits.open(f, memmap=False) as h:
            hd = h['SCI'].header
            ny, nx = h['SCI'].data.shape
        w = WCS(hd)
        cra, cdec = w.all_pix2world(nx / 2, ny / 2, 0)
        near = np.hypot((ra - cra) * np.cos(np.radians(cdec)), dec - cdec) * 3600 < 60.0   # detector half-diagonal is 45 arcsec
        ins = np.zeros(len(m), bool)
        x, y = w.all_world2pix(ra[near], dec[near], 0, quiet=True)
        ins[near] = (x >= -0.5) & (x < nx - 0.5) & (y >= -0.5) & (y < ny - 0.5)
        nhit += ins
        first[ins & (first == '')] = det
        print(band, det, ins.sum(), flush=True)
    out['foot_' + band] = first
    out['nhit_' + band] = nhit
    if band in ('F150W', 'F162M', 'F182M', 'F200W'):
        z = np.load(f'{Q}/nrcb3/rows_{band}.npz', allow_pickle=True)
        sm, sd = z['S_starm'], z['S_det']
        sat_det = np.full(len(m), '', dtype='<U6')
        pur = np.zeros(len(m))
        for s in np.unique(sm):
            d = sd[sm == s]
            u, c = np.unique(d, return_counts=True)
            sat_det[s] = u[np.argmax(c)]
            pur[s] = c.max() / c.sum()
        out['sat_' + band] = sat_det
        out['pur_' + band] = pur
        # agreement of footprint with satstar-row detector
        both = (sat_det != '') & (first != '')
        print(band, 'sat stars', (sat_det != '').sum(), 'footprint agrees', (sat_det[both] == first[both]).mean(), 'purity<1:', (pur[sat_det != ''] < 1).sum(), flush=True)
np.savez(f'groups.npz', **out)
