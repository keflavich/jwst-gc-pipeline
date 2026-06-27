#!/usr/bin/env python
"""Generalized per-tile background homogenization + image3 rebuild for MIRI
mosaics (2026-06-25, generalized from miri_f2550w_visitfix_v14.py).

skymatch ('match') matches only OVERLAPPING-region backgrounds pairwise; when
tiles have structured/thermal background differences (MIRI F2550W) it leaves
per-tile JUMPS at the mosaic tile boundaries.  This computes a GLOBAL per-tile
constant -- each tile's median image vs the median of all OTHER tiles -- and
subtracts it, then rebuilds image3 with skymatch SKIPPED (backgrounds already
homogenized).  Operates IN PLACE on the existing *_align.fits frames, so the
re-run crf/i2d carry the homogenized background and downstream cataloging uses
it.

Usage (env): BASEPATH FILT FIELD PROP, e.g.
    BASEPATH=/orange/adamginsburg/jwst/brick FILT=F2550W FIELD=002 PROP=2221 \
        python miri_tile_homogenize.py
"""
import os
import glob
import json
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from reproject import reproject_interp
import warnings
warnings.filterwarnings('ignore')

BASEPATH = os.environ.get('BASEPATH', '/orange/adamginsburg/jwst/brick')
FILT = os.environ.get('FILT', 'F2550W')
FIELD = os.environ.get('FIELD', '002')
PROP = os.environ.get('PROP', '2221')
# how many extra columns to MASK (NaN) on the east edge for the offset estimate
# only (not a DQ trim): the residual glow inside the kept region biases the
# median.  0 = use the frame as-is.
FITMASK_EXTRA_EAST = int(os.environ.get('FITMASK_EXTRA_EAST', 60))

os.environ.setdefault("CRDS_PATH", f"{BASEPATH}/crds/")
os.environ.setdefault("CRDS_SERVER_URL", "https://jwst-crds.stsci.edu")

pipedir = f'{BASEPATH}/{FILT}/pipeline'
inst = 'miri'
prod = f'jw0{PROP}-o{FIELD}_t001_{inst}_{FILT.lower()}'
ref_i2d = f'{pipedir}/{prod}_i2d.fits'

align = sorted(glob.glob(f'{pipedir}/jw0{PROP}{FIELD}*_mirimage_align.fits'))
if not align:
    raise SystemExit(f"no align frames in {pipedir}")
ref_fh = fits.open(ref_i2d)
ref_ext = 'SCI' if 'SCI' in [h.name for h in ref_fh] else 1
ref_wcs = WCS(ref_fh[ref_ext].header)
ny, nx = ref_fh[ref_ext].data.shape

# group by TILE = visit + mosaic-position prefix (jw0PROPVVVVV_MMMMM)
tiles = {}
for fn in align:
    tiles.setdefault(os.path.basename(fn)[:19], []).append(fn)
print(f"{len(align)} frames in {len(tiles)} tiles")

tile_meds = {}
for key, fns in tiles.items():
    stack = []
    for fn in fns:
        f2 = fits.open(fn)
        d = f2['SCI'].data.astype('float32').copy()
        dq = f2['DQ'].data
        bad = (dq & 1) > 0
        colgood = (~bad).any(axis=0)
        sci = np.where(colgood)[0]
        if FITMASK_EXTRA_EAST > 0 and sci.size:
            d[:, max(0, sci.max() - FITMASK_EXTRA_EAST) + 1:] = np.nan
        d[bad] = np.nan
        r, _ = reproject_interp((d, WCS(f2['SCI'].header)), ref_wcs, shape_out=(ny, nx))
        stack.append(r)
        f2.close()
    tile_meds[key] = np.nanmedian(np.array(stack, dtype='float32'), axis=0)

keys = list(tile_meds)
consts = {}
for k in keys:
    others = np.nanmedian(np.array([tile_meds[o] for o in keys if o != k],
                                   dtype='float32'), axis=0)
    diff = tile_meds[k] - others
    fin = np.isfinite(diff)
    if fin.sum() < 50:
        consts[k] = 0.0
        continue
    _, med, _ = sigma_clipped_stats(diff[fin], sigma=3, maxiters=5)
    consts[k] = float(med)
meanc = float(np.mean(list(consts.values())))
print('tile constants (rel):', {k[-11:]: round(v - meanc, 1) for k, v in consts.items()})

for k, fns in tiles.items():
    corr = consts[k] - meanc
    for fn in fns:
        f2 = fits.open(fn)
        prev = f2[1].header.get('SKYCONST', 0.0)
        f2['SCI'].data = (f2['SCI'].data - corr).astype(f2['SCI'].data.dtype)
        f2[1].header['SKYCONST'] = (float(prev) + corr,
                                    'tile sky constant subtracted [MJy/sr]')
        f2.writeto(fn, overwrite=True)
        f2.close()
print("subtracted per-tile constants in place")

# rebuild image3 with skymatch SKIPPED (backgrounds homogenized)
from jwst.pipeline import calwebb_image3
asn = {'asn_type': 'image3', 'asn_rule': 'candidate_Asn_Lv3Image',
       'version_id': None, 'code_version': '1', 'degraded_status': 'No known degraded exposures in association.',
       'program': PROP, 'constraints': 'No constraints', 'asn_id': f'o{FIELD}',
       'target': 't001', 'asn_pool': 'none',
       'products': [{'name': prod,
                     'members': [{'expname': fn, 'exptype': 'science'} for fn in align]}]}
asn_file = f'{pipedir}/{prod}_homogenize_asn.json'
with open(asn_file, 'w') as fh:
    json.dump(asn, fh)
os.chdir(pipedir)
calwebb_image3.Image3Pipeline.call(
    asn_file,
    steps={'tweakreg': {'skip': True},
           'skymatch': {'skip': True},
           'outlier_detection': {'snr': '30.0 25.0',
                                 'good_bits': 'SATURATED, JUMP_DET'}},
    output_dir=pipedir, save_results=True)
print(f"rebuilt {prod}_i2d.fits (tile-homogenized, skymatch skipped)")
