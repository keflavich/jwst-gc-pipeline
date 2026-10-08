"""Distortion reference provenance for NRCA1-NRCA4 per band (wd2 crf headers
-> CRDS cache asdf meta)."""
import glob
import os

import asdf
from astropy.io import fits

R = '/orange/adamginsburg/jwst/wd2'
CR = os.environ.get('CRDS_PATH', os.path.expanduser('~/crds_cache'))
for band in ['F115W', 'F150W', 'F162M', 'F164N', 'F200W', 'F182M', 'F187N', 'F212N']:
    for det in ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb2']:
        fs = sorted(glob.glob(f'{R}/{band}/pipeline/jw03523005001_*_00001_{det}_*_crf.fits'))
        if not fs:
            print(band, det, 'no crf')
            continue
        h = fits.getheader(fs[0], 0)
        ref = h['R_DISTOR'].replace('crds://', '')
        p = glob.glob(f'{CR}/references/jwst/nircam/{ref}')
        if not p:
            print(band, det, ref, 'not in cache')
            continue
        with asdf.open(p[0], lazy_load=True) as af:
            m = af.tree.get('meta', {})
            print(f"{band} {det} {ref} useafter={m.get('useafter')} pedigree={m.get('pedigree')} "
                  f"author={m.get('author')} | {str(m.get('description'))[:110]}")
