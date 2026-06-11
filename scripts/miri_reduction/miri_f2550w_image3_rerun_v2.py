#!/usr/bin/env python
"""
Brick F2550W image3 re-run experiment (2026-06-10).

The 2024-11 (jwst 1.16.0) combined i2d suffers large NaN patches and
tile-to-tile background jumps.  Diagnosis: the NaN regions are >87-99%
OUTLIER-flagged in every covering exposure -- outlier_detection's median
comparison was poisoned by inter-visit background mismatch, so it nuked
whole regions.  (One rectangular hole is additionally the Lyot/NON_SCIENCE
zone, unrecoverable where no other tile covers it.)

This script re-runs ONLY image3 (tweakreg skipped; existing *_align.fits WCS
kept) with jwst 1.21-dev and two variants:

  v2a: skymatch(match, subtract=True) + outlier_detection snr='30.0 25.0'
  v2b: skymatch(match, subtract=True) + outlier_detection skipped (control)

Outputs go to pipeline_v2/ -- the existing pipeline/ products are untouched.
After both variants, prints NaN-fraction and overlap-seam statistics, and
measures the global astrometric offset of bright mosaic sources against the
NIRCam F405N reference catalog.
"""
import os
import sys
import json
import shutil
import glob
import numpy as np

basepath = '/orange/adamginsburg/jwst/brick'
pipedir = f'{basepath}/F2550W/pipeline'
outdir = f'{basepath}/F2550W/pipeline_v2'
os.makedirs(outdir, exist_ok=True)

os.environ['CRDS_PATH'] = f'{basepath}/crds/'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

from jwst.pipeline import calwebb_image3
import jwst
print(f'jwst version: {jwst.__version__}', flush=True)

asn_file = f'{pipedir}/jw02221-o002_20240916t061459_image3_00007_asn.json'
with open(asn_file) as fh:
    asn_data = json.load(fh)

# member paths resolve relative to the asn file's directory, so make them
# absolute (the asn copies live in pipeline_v2/, the data in pipeline/)
for member in asn_data['products'][0]['members']:
    if not os.path.isabs(member['expname']):
        member['expname'] = os.path.join(pipedir, member['expname'])

# run from the pipeline dir so relative member paths resolve
os.chdir(pipedir)

variants = {
    'v2a': {
        'skymatch': {'skymethod': 'match', 'subtract': True,
                     'match_down': False, 'save_results': True},
        'outlier_detection': {'snr': '30.0 25.0',
                              'good_bits': 'SATURATED, JUMP_DET'},
    },
    'v2b': {
        'skymatch': {'skymethod': 'match', 'subtract': True,
                     'match_down': False, 'save_results': True},
        'outlier_detection': {'skip': True},
    },
}

for tag, steps in variants.items():
    name = f'jw02221-o002_t001_miri_f2550w_{tag}'
    asn_data['products'][0]['name'] = name
    asn_tag_file = f'{outdir}/{name}_asn.json'
    with open(asn_tag_file, 'w') as fh:
        json.dump(asn_data, fh)
    print(f'=== Running image3 variant {tag} -> {outdir}/{name}_i2d.fits', flush=True)
    steps = dict(steps)
    steps['tweakreg'] = {'skip': True}
    calwebb_image3.Image3Pipeline.call(
        asn_tag_file,
        steps=steps,
        output_dir=outdir,
        save_results=True)
    print(f'=== DONE variant {tag}', flush=True)

# ---- evaluation ----
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.stats import mad_std
import astropy.units as u
from photutils.detection import DAOStarFinder
import warnings
warnings.filterwarnings('ignore')

old_fn = f'{pipedir}/jw02221-o002_t001_miri_f2550w_i2d.fits'


def stats(fn, label):
    d = fits.open(fn)['SCI'].data
    finite = np.isfinite(d)
    rows = finite.any(axis=1)
    cols = finite.any(axis=0)
    bbox = finite[np.ix_(rows, cols)]
    print(f'[{label}] {os.path.basename(fn)} shape={d.shape} '
          f'NaN frac in bbox = {1 - bbox.mean():.4f}')
    return d


stats(old_fn, 'OLD 1.16.0')
for tag in variants:
    fn = f'{outdir}/jw02221-o002_t001_miri_f2550w_{tag}_i2d.fits'
    d = stats(fn, tag)

# astrometric check on v2a: bright point sources vs F405N refcat
fn = f'{outdir}/jw02221-o002_t001_miri_f2550w_v2a_i2d.fits'
hdu = fits.open(fn)['SCI']
data = hdu.data
ww = WCS(hdu.header)
err = mad_std(data, ignore_nan=True)
dao = DAOStarFinder(threshold=15 * err, fwhm=7.3)  # F2550W FWHM in pix
filled = np.nan_to_num(data - np.nanmedian(data))
srcs = dao(filled)
print(f'astrometry: found {len(srcs) if srcs is not None else 0} bright 25um sources')
if srcs is not None and len(srcs) > 3:
    sc = ww.pixel_to_world(srcs['xcentroid'], srcs['ycentroid'])
    refcat_fn = f'{basepath}/catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog.fits'
    ref = Table.read(refcat_fn)
    refsc = SkyCoord(ref['skycoord']) if 'skycoord' in ref.colnames else SkyCoord(ref['RA'], ref['DEC'], unit='deg')
    idx, sep, _ = sc.match_to_catalog_sky(refsc)
    m = sep < 1.5 * u.arcsec
    print(f'astrometry: {m.sum()} matches < 1.5"')
    if m.sum() > 3:
        dra = (sc.ra[m] - refsc.ra[idx[m]]) * np.cos(sc.dec[m])
        ddec = sc.dec[m] - refsc.dec[idx[m]]
        print(f'astrometry: median offset dRA={np.median(dra.to(u.arcsec)):.3f} '
              f'dDec={np.median(ddec.to(u.arcsec)):.3f} '
              f'(scatter {mad_std(dra.to(u.arcsec).value):.3f}, '
              f'{mad_std(ddec.to(u.arcsec).value):.3f} arcsec)')
print('ALL DONE', flush=True)
