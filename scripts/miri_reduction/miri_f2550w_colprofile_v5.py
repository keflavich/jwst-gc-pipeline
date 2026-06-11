#!/usr/bin/env python
"""
Brick F2550W v5: per-frame detector-column profile correction (2026-06-11).

v4 (40-column east edge trim) removed the reported x=1609 seam at most y,
but the boundary simply moved to the new coverage edge (x~1572) where a
~130 MJy/sr step persists at y=750-800: the east-edge glow extends deeper
than 40 columns in some frames, with frame-dependent depth.  Trimming
deeper sacrifices area without bounding the artifact.

v5 measures and removes the artifact itself, per frame: the frame's
deviation from the consensus mosaic, collapsed along detector rows into a
robust per-COLUMN profile (the glow is a function of detector x), smoothed,
and subtracted.  This is the destreak philosophy applied along the MIRI x
axis; it flattens edge glow of any depth and any frame-to-frame variation,
while leaving (sky-fixed) real structure intact because the consensus is
built from many frames at different sky positions.

Procedure:
 1. reference = v4 mosaic (edge-trimmed build; its WCS correction is undone
    in memory so it matches the frames' uncorrected WCS).
 2. For each of the 48 _align frames: reproject the reference onto the
    frame grid; diff = frame - ref; sigma-clip along columns; column-median
    profile; median-smooth (15 px); subtract from the frame.
 3. Keep the v4 edge trim (E40/W16/rows12) as well -- the outermost columns
    have too little overlap to constrain their profile.
 4. Rebuild image3 (v2a configuration), evaluate seam metrics at both the
    original (x=1609) and shifted (x=1572) boundaries, apply the WCS
    correction.
"""
import os
import json
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clip
from scipy.ndimage import median_filter
from reproject import reproject_interp
import warnings
warnings.filterwarnings('ignore')

basepath = '/orange/adamginsburg/jwst/brick'
pipedir = f'{basepath}/F2550W/pipeline'
outdir = f'{basepath}/F2550W/pipeline_v2'

os.environ["CRDS_PATH"] = f"{basepath}/crds/"
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

TRIM_EAST, TRIM_WEST, TRIM_ROWS = 40, 16, 12

# reference: v4 mosaic with its CRVAL correction undone (frames are uncorrected)
ref_fh = fits.open(f'{outdir}/jw02221-o002_t001_miri_f2550w_v4_i2d.fits')
ref_hdr = ref_fh[1].header.copy()
if 'OLCRVAL1' in ref_hdr:
    ref_hdr['CRVAL1'] = ref_hdr['OLCRVAL1']
    ref_hdr['CRVAL2'] = ref_hdr['OLCRVAL2']
ref = ref_fh[1].data
ref_wcs = WCS(ref_hdr)

asn_file = f'{outdir}/jw02221-o002_t001_miri_f2550w_v2a_asn.json'
with open(asn_file) as fh:
    asn_data = json.load(fh)
members = [m['expname'] if os.path.isabs(m['expname'])
           else os.path.join(pipedir, m['expname'])
           for m in asn_data['products'][0]['members']]

new_members = []
for fn in members:
    f2 = fits.open(fn)
    d = f2['SCI'].data
    dq = f2['DQ'].data
    ww = WCS(f2['SCI'].header)
    refproj, _ = reproject_interp((ref, ref_wcs), ww, shape_out=d.shape)
    diff = np.where((dq & 1) == 0, d - refproj, np.nan)
    clipped = sigma_clip(diff, sigma=3, maxiters=3, axis=0)
    colprof = np.ma.median(clipped, axis=0).filled(np.nan)
    # smooth; fill gaps with 0 (no correction where unconstrained)
    cp = np.where(np.isfinite(colprof), colprof, 0.0)
    cp_s = median_filter(cp, size=15)
    cp_s[~np.isfinite(colprof)] = 0.0
    f2['SCI'].data = (d - cp_s[None, :]).astype(d.dtype)
    # edge trim as in v4
    colgood = ((dq & 1) == 0).any(axis=0)
    sci_cols = np.where(colgood)[0]
    lo, hi = sci_cols.min(), sci_cols.max()
    dq[:, hi - TRIM_EAST + 1:] |= 1
    dq[:, :lo + TRIM_WEST] |= 1
    dq[:TRIM_ROWS, :] |= 1
    dq[-TRIM_ROWS:, :] |= 1
    f2['DQ'].data = dq
    f2[1].header['COLPROF'] = (float(np.nanmax(np.abs(cp_s))),
                               'max |column-profile corr| [MJy/sr] 2026-06-11')
    outfn = fn.replace('_align.fits', '_colprofcorr.fits')
    f2.writeto(outfn, overwrite=True)
    new_members.append(outfn)
    print(f'{os.path.basename(fn)}: max|colprof|={np.nanmax(np.abs(cp_s)):.0f} '
          f'east-50-col mean={np.nanmean(cp_s[-90:-40]):+.0f} MJy/sr', flush=True)

from jwst.pipeline import calwebb_image3
name = 'jw02221-o002_t001_miri_f2550w_v5'
asn_data['products'][0]['name'] = name
asn_data['products'][0]['members'] = [
    {'expname': fn, 'exptype': 'science'} for fn in new_members]
asn_v5 = f'{outdir}/{name}_asn.json'
with open(asn_v5, 'w') as fh:
    json.dump(asn_data, fh)
os.chdir(pipedir)
calwebb_image3.Image3Pipeline.call(
    asn_v5,
    steps={'tweakreg': {'skip': True},
           'skymatch': {'skymethod': 'match', 'subtract': True,
                        'match_down': False, 'save_results': True},
           'outlier_detection': {'snr': '30.0 25.0',
                                 'good_bits': 'SATURATED, JUMP_DET'}},
    output_dir=outdir,
    save_results=True)

v2a = fits.open(f'{outdir}/jw02221-o002_t001_miri_f2550w_v2a_i2d.fits')['SCI'].data
v5 = fits.open(f'{outdir}/{name}_i2d.fits')['SCI'].data
for label, img in [('v2a', v2a), ('v5', v5)]:
    for xseam in (1609, 1572):
        jumps = []
        for y0 in range(450, 800, 50):
            left = np.nanmedian(img[y0:y0+50, xseam-19:xseam-4])
            right = np.nanmedian(img[y0:y0+50, xseam+4:xseam+19])
            jumps.append(right - left)
        print(f'{label} @x={xseam}: ' + ' '.join(f'{j:+.0f}' for j in jumps))
    fin = np.isfinite(img)
    rows = fin.any(axis=1); cols = fin.any(axis=0)
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
print('applied WCS correction to v5 product')
print('ALL DONE')
