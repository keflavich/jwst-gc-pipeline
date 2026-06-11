#!/usr/bin/env python
"""
Brick F2550W v12: robust per-tile sky constants, skymatch disabled (2026-06-11).

Diagnosis of the user-reported ~10% (additive ~+100 MJy/sr) jump at
x~2185, y~400: the RAW frames of the two visit groups agree to <=10 MJy/sr
in that overlap (leave-group-out sub-mosaic arbitration), but the v11
mosaic shows the step -- jwst skymatch's least-squares constants are biased
by the residual east-edge glow that contaminates OTHER overlaps (x~1565
zone), and the error propagates to clean boundaries as an additive offset
(invisible on bright emission, ~10% on faint).

v12 replaces skymatch with constants we solve robustly:
 1. Reproject each frame (DQ-masked, east 100 columns excluded so glow
    never enters the fit) onto the mosaic grid; per-tile (visit) medians.
 2. For each tile: reference = median of the other 11 tile-medians;
    constant = sigma-clipped median of (tile - reference) over their
    common footprint.
 3. Subtract (constant - mean(constants)) from the tile's 4 members
    (preserves the global level), with the E56/W16/R12 trim applied.
 4. Rebuild image3 with skymatch SKIPPED; outlier snr='30.0 25.0'.
 5. Metrics at all three boundary columns + WCS correction.
"""
import os
import json
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from reproject import reproject_interp
import warnings
warnings.filterwarnings('ignore')

basepath = '/orange/adamginsburg/jwst/brick'
pipedir = f'{basepath}/F2550W/pipeline'
outdir = f'{basepath}/F2550W/pipeline_v2'
os.environ["CRDS_PATH"] = f"{basepath}/crds/"
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

TRIM_EAST, TRIM_WEST, TRIM_ROWS = 56, 16, 12
FITMASK_EAST = 100   # exclude from constant fitting (glow zone)

ref_fh = fits.open(f'{outdir}/jw02221-o002_t001_miri_f2550w_v2a_i2d.fits')
hdr = ref_fh[1].header.copy()
ww = WCS(hdr)
ny, nx = ref_fh[1].data.shape

asn_file = f'{outdir}/jw02221-o002_t001_miri_f2550w_v2a_asn.json'
with open(asn_file) as fh:
    asn_data = json.load(fh)
members = [m['expname'] if os.path.isabs(m['expname'])
           else os.path.join(pipedir, m['expname'])
           for m in asn_data['products'][0]['members']]

tiles = {}
for fn in members:
    tiles.setdefault(os.path.basename(fn)[:19], []).append(fn)

tile_meds = {}
for key, fns in tiles.items():
    stack = []
    for fn in fns:
        f2 = fits.open(fn)
        d = f2['SCI'].data.copy()
        dq = f2['DQ'].data
        bad = (dq & 1) > 0
        colgood = (~bad).any(axis=0)
        sci = np.where(colgood)[0]
        d[:, sci.max() - FITMASK_EAST + 1:] = np.nan
        d[:, :sci.min() + TRIM_WEST] = np.nan
        d[:TRIM_ROWS, :] = np.nan
        d[-TRIM_ROWS:, :] = np.nan
        d[bad] = np.nan
        r, _ = reproject_interp((d, WCS(f2['SCI'].header)), ww, shape_out=(ny, nx))
        stack.append(r)
        f2.close()
    tile_meds[key] = np.nanmedian(np.array(stack, dtype='float32'), axis=0)
    print(f'{key}: tile median built', flush=True)

keys = list(tile_meds)
consts = {}
for k in keys:
    others = np.nanmedian(np.array([tile_meds[o] for o in keys if o != k],
                                   dtype='float32'), axis=0)
    diff = tile_meds[k] - others
    _, med, _ = sigma_clipped_stats(diff[np.isfinite(diff)], sigma=3, maxiters=5)
    consts[k] = med
    print(f'{k}: sky constant {med:+.1f} MJy/sr '
          f'(n_overlap={np.isfinite(diff).sum()})', flush=True)
meanc = np.mean(list(consts.values()))
print(f'mean constant (restored): {meanc:+.1f}')

new_members = []
for k, fns in tiles.items():
    corr = consts[k] - meanc
    for fn in fns:
        f2 = fits.open(fn)
        dq = f2['DQ'].data
        colgood = ((dq & 1) == 0).any(axis=0)
        sci = np.where(colgood)[0]
        lo, hi = sci.min(), sci.max()
        dq[:, hi - TRIM_EAST + 1:] |= 1
        dq[:, :lo + TRIM_WEST] |= 1
        dq[:TRIM_ROWS, :] |= 1
        dq[-TRIM_ROWS:, :] |= 1
        f2['DQ'].data = dq
        f2['SCI'].data = (f2['SCI'].data - corr).astype(f2['SCI'].data.dtype)
        f2[1].header['SKYCONST'] = (corr, 'tile sky constant subtracted [MJy/sr]')
        f2[1].header['EDGETRIM'] = (f'E{TRIM_EAST} W{TRIM_WEST} R{TRIM_ROWS}',
                                    'edge columns/rows DQ-flagged')
        outfn = fn.replace('_align.fits', '_skyconst12.fits')
        f2.writeto(outfn, overwrite=True)
        new_members.append(outfn)
print(f'wrote {len(new_members)} corrected frames')

from jwst.pipeline import calwebb_image3
name = 'jw02221-o002_t001_miri_f2550w_v12'
asn_data['products'][0]['name'] = name
asn_data['products'][0]['members'] = [
    {'expname': fn, 'exptype': 'science'} for fn in new_members]
asn_out = f'{outdir}/{name}_asn.json'
with open(asn_out, 'w') as fh:
    json.dump(asn_data, fh)
os.chdir(pipedir)
calwebb_image3.Image3Pipeline.call(
    asn_out,
    steps={'tweakreg': {'skip': True},
           'skymatch': {'skip': True},
           'outlier_detection': {'snr': '30.0 25.0',
                                 'good_bits': 'SATURATED, JUMP_DET'}},
    output_dir=outdir,
    save_results=True)

v11 = fits.open(f'{pipedir}/jw02221-o002_t001_miri_f2550w_i2d.fits')['SCI'].data
v12 = fits.open(f'{outdir}/{name}_i2d.fits')['SCI'].data
for label, img in [('v11', v11), ('v12', v12)]:
    for xseam in (1556, 1609, 2185):
        jumps = []
        for y0 in range(350, 800, 75):
            left = np.nanmedian(img[y0:y0+75, xseam-22:xseam-6])
            right = np.nanmedian(img[y0:y0+75, xseam+6:xseam+22])
            jumps.append(right - left)
        print(f'{label} @x={xseam}: ' + ' '.join(f'{j:+.0f}' for j in jumps))
    fin = np.isfinite(img)
    rows = fin.any(axis=1)
    cols = fin.any(axis=0)
    print(f'{label}: NaN frac in bbox {1 - fin[np.ix_(rows, cols)].mean():.4f}')

fh = fits.open(f'{outdir}/{name}_i2d.fits')
dra_as, ddec_as = -4.697, -2.469
cosd = np.cos(np.deg2rad(fh[1].header['CRVAL2']))
fh[1].header['OLCRVAL1'] = fh[1].header['CRVAL1']
fh[1].header['OLCRVAL2'] = fh[1].header['CRVAL2']
fh[1].header['CRVAL1'] = fh[1].header['CRVAL1'] - dra_as / 3600. / cosd
fh[1].header['CRVAL2'] = fh[1].header['CRVAL2'] - ddec_as / 3600.
fh[1].header['MIRIDRA'] = (-dra_as, 'applied RA correction [arcsec], 2026-06-11')
fh[1].header['MIRIDDE'] = (-ddec_as, 'applied Dec correction [arcsec], 2026-06-11')
fh[1].header['MIRIWCSN'] = ('FITS WCS corrected; ASDF gwcs NOT corrected',
                            'offset-histogram registration to NIRCam refcat')
fh.writeto(f'{outdir}/{name}_i2d.fits', overwrite=True)
print('applied WCS correction to v12 product')
print('ALL DONE')
