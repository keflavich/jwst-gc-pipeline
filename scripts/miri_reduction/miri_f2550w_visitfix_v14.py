#!/usr/bin/env python
"""
Brick F2550W v14: visit-001 WCS fix (2026-06-12).

The doubled star (user-reported) arises because the two VISITS of obs 002
are mutually misregistered by 4.2": per-visit sub-mosaic centroids of the
star at 17:46:08.63 -28:42:32.5 (NIRCam-confirmed position) give, after
the global anchor correction:
    visit 002001: dRA=+1.01" dDec=-4.12"   <- wrong (the 'ghost')
    visit 002002: dRA=+0.00" dDec=+0.01"   <- correct
In v1/v2a the displaced visit-001 copies were OUTLIER-rejected (silently);
the v4-v12 fixes exposed them.  Note both the refcat offset-histogram
(F2550W has too few refcat counterparts) and frame-vs-mosaic
cross-correlation (reference follows each visit locally -- circular) were
blind to this; per-visit sub-mosaics + a NIRCam-confirmed star were needed.

v14: adjust_wcs(-1.01", +4.12") on the 24 visit-001 frames, visit-002
untouched, then the v12 recipe (tile constants, skymatch skipped, E56
trim), then the global anchor.
"""
import os
import json
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
import astropy.units as u
from reproject import reproject_interp
import warnings
warnings.filterwarnings('ignore')

basepath = '/orange/adamginsburg/jwst/brick'
pipedir = f'{basepath}/F2550W/pipeline'
outdir = f'{basepath}/F2550W/pipeline_v2'
os.environ["CRDS_PATH"] = f"{basepath}/crds/"
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

from jwst.datamodels import ImageModel
from jwst.tweakreg.utils import adjust_wcs

TRIM_EAST, TRIM_WEST, TRIM_ROWS = 56, 16, 12
FITMASK_EAST = 100
VISIT1_DRA, VISIT1_DDEC = -1.01, +4.12   # arcsec, correction for jw02221002001*

asn_file = f'{outdir}/jw02221-o002_t001_miri_f2550w_v2a_asn.json'
with open(asn_file) as fh:
    asn_data = json.load(fh)
members = [m['expname'] if os.path.isabs(m['expname'])
           else os.path.join(pipedir, m['expname'])
           for m in asn_data['products'][0]['members']]

new_members = []
for fn in members:
    is_v1 = '02221002001' in os.path.basename(fn)
    outfn = fn.replace('_align.fits', '_visitfix14.fits')
    fa = ImageModel(fn)
    if is_v1:
        fa.meta.wcs = adjust_wcs(fa.meta.wcs,
                                 delta_ra=VISIT1_DRA * u.arcsec,
                                 delta_dec=VISIT1_DDEC * u.arcsec)
    fa.save(outfn, overwrite=True)
    f3 = fits.open(outfn)
    dq = f3['DQ'].data
    colgood = ((dq & 1) == 0).any(axis=0)
    sci = np.where(colgood)[0]
    dq[:, sci.max() - TRIM_EAST + 1:] |= 1
    dq[:, :sci.min() + TRIM_WEST] |= 1
    dq[:TRIM_ROWS, :] |= 1
    dq[-TRIM_ROWS:, :] |= 1
    f3['DQ'].data = dq
    if is_v1:
        # keep the FITS WCS consistent with the adjusted gwcs for our own
        # reproject-based steps (CRVAL shift; exact enough at this scale)
        cosd = np.cos(np.deg2rad(f3[1].header['CRVAL2']))
        f3[1].header['CRVAL1'] = f3[1].header['CRVAL1'] + VISIT1_DRA / 3600. / cosd
        f3[1].header['CRVAL2'] = f3[1].header['CRVAL2'] + VISIT1_DDEC / 3600.
        f3[1].header['VISFIX'] = (f'{VISIT1_DRA},{VISIT1_DDEC}',
                                  'visit-001 WCS correction [arcsec] 2026-06-12')
    f3.writeto(outfn, overwrite=True)
    new_members.append(outfn)
print(f'wrote {len(new_members)} frames (visit-001 shifted by '
      f'({VISIT1_DRA}", {VISIT1_DDEC}"))')

# tile constants (v12 recipe) on the corrected frames
ref_fh = fits.open(f'{outdir}/jw02221-o002_t001_miri_f2550w_v2a_i2d.fits')
ref_wcs = WCS(ref_fh[1].header)
ny, nx = ref_fh[1].data.shape
tiles = {}
for fn in new_members:
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
        d[:, sci.max() - (FITMASK_EAST - TRIM_EAST) + 1:] = np.nan
        d[bad] = np.nan
        r, _ = reproject_interp((d, WCS(f2['SCI'].header)), ref_wcs, shape_out=(ny, nx))
        stack.append(r)
        f2.close()
    tile_meds[key] = np.nanmedian(np.array(stack, dtype='float32'), axis=0)
keys = list(tile_meds)
consts = {}
for k in keys:
    others = np.nanmedian(np.array([tile_meds[o] for o in keys if o != k], dtype='float32'), axis=0)
    diff = tile_meds[k] - others
    _, med, _ = sigma_clipped_stats(diff[np.isfinite(diff)], sigma=3, maxiters=5)
    consts[k] = med
meanc = np.mean(list(consts.values()))
print('tile constants:', {k: round(v - meanc, 1) for k, v in consts.items()})
for k, fns in tiles.items():
    corr = consts[k] - meanc
    for fn in fns:
        f2 = fits.open(fn)
        f2['SCI'].data = (f2['SCI'].data - corr).astype(f2['SCI'].data.dtype)
        f2[1].header['SKYCONST'] = (corr, 'tile sky constant subtracted [MJy/sr]')
        f2.writeto(fn, overwrite=True)

from jwst.pipeline import calwebb_image3
name = 'jw02221-o002_t001_miri_f2550w_v14'
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

fh = fits.open(f'{outdir}/{name}_i2d.fits')
dra_as, ddec_as = -4.697, -2.469
cosd = np.cos(np.deg2rad(fh[1].header['CRVAL2']))
fh[1].header['OLCRVAL1'] = fh[1].header['CRVAL1']
fh[1].header['OLCRVAL2'] = fh[1].header['CRVAL2']
fh[1].header['CRVAL1'] = fh[1].header['CRVAL1'] - dra_as / 3600. / cosd
fh[1].header['CRVAL2'] = fh[1].header['CRVAL2'] - ddec_as / 3600.
fh[1].header['MIRIDRA'] = (-dra_as, 'applied RA correction [arcsec]')
fh[1].header['MIRIDDE'] = (-ddec_as, 'applied Dec correction [arcsec]')
fh[1].header['MIRIWCSN'] = ('FITS WCS corrected; ASDF gwcs NOT corrected',
                            'offset-histogram registration to NIRCam refcat')
fh[1].header['VISFIX'] = (f'{VISIT1_DRA},{VISIT1_DDEC}',
                          'visit-001 frames pre-shifted [arcsec]')
fh.writeto(f'{outdir}/{name}_i2d.fits', overwrite=True)

# verification: star must be single, at the NIRCam position
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from photutils.centroids import centroid_com
truth = SkyCoord('17:46:08.63 -28:42:32.5', unit=(u.hourangle, u.deg))
ghost = SkyCoord('17:46:08.71 -28:42:36.6', unit=(u.hourangle, u.deg))
d = fh[1].data
wwf = WCS(fh[1].header)
co = Cutout2D(d, truth, 14 * u.arcsec, wcs=wwf, mode='partial')
img = np.nan_to_num(co.data - np.nanmedian(co.data))
cy, cx = np.unravel_index(np.nanargmax(img), img.shape)
box = img[max(0, cy-5):cy+6, max(0, cx-5):cx+6]
ccx, ccy = centroid_com(box)
sc = co.wcs.pixel_to_world(max(0, cx-5)+ccx, max(0, cy-5)+ccy)
print(f'v14 star centroid: {sc.to_string("hmsdms")} '
      f'(offset from truth {sc.separation(truth).arcsec:.2f}")')
xg, yg = co.wcs.world_to_pixel(ghost)
xg, yg = int(xg), int(yg)
peak_t = np.nanmax(img[max(0,cy-4):cy+5, max(0,cx-4):cx+5])
peak_g = np.nanmax(img[max(0,yg-4):yg+5, max(0,xg-4):xg+5]) if 0 <= yg < img.shape[0] and 0 <= xg < img.shape[1] else np.nan
print(f'peak at truth: {peak_t:.0f}; peak at ghost position: {peak_g:.0f} '
      f'(ratio {peak_g/peak_t:.2f}; should be background-level)')
for xseam in (1556, 1609, 2185):
    jumps = []
    for y0 in range(350, 800, 75):
        left = np.nanmedian(d[y0:y0+75, xseam-22:xseam-6])
        right = np.nanmedian(d[y0:y0+75, xseam+6:xseam+22])
        jumps.append(right - left)
    print(f'v14 @x={xseam}: ' + ' '.join(f'{j:+.0f}' for j in jumps))
print('ALL DONE')
