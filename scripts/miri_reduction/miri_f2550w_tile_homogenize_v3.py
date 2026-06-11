#!/usr/bin/env python
"""
Brick F2550W v3: per-tile background-plane homogenization (2026-06-11).

The v2a mosaic still shows seams at tile-coverage edges (user-reported at
x=1609, y=563-627): the MIRI frames carry frame-dependent low-order
background structure (edge brightening up to +1400 MJy/sr in the last ~50
columns, strongly variable frame-to-frame), so the constant-per-frame
skymatch cannot make tiles agree and the boundary shows a y-dependent jump
(-4 to -75 MJy/sr).

Custom step added here:
 1. Group the 48 exposures into 12 visit-tiles (4 dithers each).
 2. Reproject each exposure onto the v2a mosaic grid (DQ-masked); median the
    4 dithers into a tile image.
 3. diff = tile - v2a reference mosaic; sigma-clip (rejects stars/artifacts)
    and fit a plane a + b*x + c*y (mosaic pixel coords) over the tile
    footprint.
 4. Remove the mean of all tile planes (so the global background level and
    gradient are preserved); subtract each tile's residual plane from its 4
    member _align files (evaluated through each frame's own WCS).
 5. Rebuild the mosaic with the v2a image3 configuration (tweakreg skipped,
    skymatch subtract=True, outlier snr='30.0 25.0').
 6. Report seam metrics at the known boundary and write
    ..._f2550w_v3_i2d.fits.  (The measured astrometric CRVAL correction is
    applied to the final product, as in apply_measured_miri_wcs_offsets.py.)

NOTE: a plane per tile can absorb real sky gradients on tile scales; the
mean-plane restoration keeps the mosaic-scale structure.  Good enough for
imaging/photometry; do not use for absolute surface-brightness science
without checking against the v2a product.
"""
import os
import json
import shutil
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clip
from reproject import reproject_interp
import warnings
warnings.filterwarnings('ignore')

basepath = '/orange/adamginsburg/jwst/brick'
pipedir = f'{basepath}/F2550W/pipeline'
outdir = f'{basepath}/F2550W/pipeline_v2'

os.environ["CRDS_PATH"] = f"{basepath}/crds/"
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

asn_file = f'{outdir}/jw02221-o002_t001_miri_f2550w_v2a_asn.json'
with open(asn_file) as fh:
    asn_data = json.load(fh)
members = [m['expname'] if os.path.isabs(m['expname'])
           else os.path.join(pipedir, m['expname'])
           for m in asn_data['products'][0]['members']]

ref_fn = f'{outdir}/jw02221-o002_t001_miri_f2550w_v2a_i2d.fits'
ref_hdu = fits.open(ref_fn)['SCI']
ref = ref_hdu.data
ref_wcs = WCS(ref_hdu.header)
ny, nx = ref.shape

# 12 visit-tiles: key = jw02221002001_02101 etc.
tiles = {}
for fn in members:
    key = os.path.basename(fn)[:19]
    tiles.setdefault(key, []).append(fn)
print(f'{len(tiles)} tiles: ' + ', '.join(f'{k}({len(v)})' for k, v in tiles.items()))

yy, xx = np.mgrid[0:ny, 0:nx]
planes = {}
coeffs = []
for key, fns in tiles.items():
    stack = []
    for fn in fns:
        f2 = fits.open(fn)
        d = np.where((f2['DQ'].data & 1) == 0, f2['SCI'].data, np.nan)
        r, _ = reproject_interp((d, WCS(f2['SCI'].header)), ref_wcs, shape_out=(ny, nx))
        stack.append(r)
    tile_img = np.nanmedian(stack, axis=0)
    diff = tile_img - ref
    good = np.isfinite(diff)
    clipped = sigma_clip(diff[good], sigma=3, maxiters=5)
    use = np.zeros_like(diff, dtype=bool)
    use[good] = ~clipped.mask
    A = np.c_[np.ones(use.sum()), xx[use] / nx, yy[use] / ny]
    coef, *_ = np.linalg.lstsq(A, diff[use], rcond=None)
    planes[key] = coef
    coeffs.append(coef)
    print(f'{key}: plane offset={coef[0]:+.1f} dx={coef[1]:+.1f} dy={coef[2]:+.1f} MJy/sr (n={use.sum()})')

# preserve global background: remove the mean plane from the corrections
meanc = np.mean(coeffs, axis=0)
print(f'mean plane (restored): {meanc}')

new_members = []
for key, fns in tiles.items():
    coef = planes[key] - meanc
    for fn in fns:
        f2 = fits.open(fn)
        w2 = WCS(f2['SCI'].header)
        fny, fnx = f2['SCI'].data.shape
        fyy, fxx = np.mgrid[0:fny, 0:fnx]
        sky = w2.pixel_to_world(fxx.ravel(), fyy.ravel())
        mx, my = ref_wcs.world_to_pixel(sky)
        plane = (coef[0] + coef[1] * mx / nx + coef[2] * my / ny).reshape(fny, fnx)
        f2['SCI'].data = f2['SCI'].data - plane.astype(f2['SCI'].data.dtype)
        f2[1].header['TILEPLNA'] = (coef[0], 'tile plane const subtracted [MJy/sr]')
        outfn = fn.replace('_align.fits', '_planecorr.fits')
        f2.writeto(outfn, overwrite=True)
        # copy the ASDF-carrying original?  writeto preserves all HDUs incl ASDF.
        new_members.append(outfn)
print(f'wrote {len(new_members)} plane-corrected frames')

# rebuild mosaic with v2a configuration
from jwst.pipeline import calwebb_image3
name = 'jw02221-o002_t001_miri_f2550w_v3'
asn_data['products'][0]['name'] = name
asn_data['products'][0]['members'] = [
    {'expname': fn, 'exptype': 'science'} for fn in new_members]
asn_v3 = f'{outdir}/{name}_asn.json'
with open(asn_v3, 'w') as fh:
    json.dump(asn_data, fh)
os.chdir(pipedir)
calwebb_image3.Image3Pipeline.call(
    asn_v3,
    steps={'tweakreg': {'skip': True},
           'skymatch': {'skymethod': 'match', 'subtract': True,
                        'match_down': False, 'save_results': True},
           'outlier_detection': {'snr': '30.0 25.0',
                                 'good_bits': 'SATURATED, JUMP_DET'}},
    output_dir=outdir,
    save_results=True)

# evaluate the reported seam (in v2a pixel coords; v3 grid should match)
v3 = fits.open(f'{outdir}/{name}_i2d.fits')['SCI'].data
for label, img in [('v2a', ref), ('v3', v3)]:
    if img.shape != ref.shape:
        print(f'{label}: shape {img.shape} != ref; skipping metric')
        continue
    jumps = []
    for y0 in range(450, 800, 50):
        left = np.nanmedian(img[y0:y0+50, 1590:1605])
        right = np.nanmedian(img[y0:y0+50, 1613:1628])
        jumps.append(right - left)
    print(f'{label}: seam jumps y=450..800: ' + ' '.join(f'{j:+.0f}' for j in jumps))

# apply the measured astrometric correction to the v3 product
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
print('applied WCS correction to v3 product')
print('ALL DONE')
