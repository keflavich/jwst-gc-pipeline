#!/usr/bin/env python
"""
Brick F2550W v4: detector-edge trim (2026-06-11).

v3 (per-tile plane homogenization) did NOT reduce the tile-boundary seam at
x=1609: the artifact is brightening of the last ~40 detector columns that
varies frame-to-frame AND along the edge, so neither constant skymatch nor
per-tile planes can remove it.  v4 instead excludes the contaminated pixels:
DQ-flag (DO_NOT_USE) a margin around each frame -- 40 columns on the east
(high-x) edge where the excess reaches ~+150-1400 MJy/sr, 16 columns on the
west science edge, 12 rows top/bottom -- then rebuild image3 with the v2a
configuration.  In tile-overlap strips the trimmed pixels are covered by
neighboring frames' interiors, so the seam should disappear; the mosaic
outer boundary shrinks by the trim width.
"""
import os
import json
import numpy as np
from astropy.io import fits
import warnings
warnings.filterwarnings('ignore')

basepath = '/orange/adamginsburg/jwst/brick'
pipedir = f'{basepath}/F2550W/pipeline'
outdir = f'{basepath}/F2550W/pipeline_v2'

os.environ["CRDS_PATH"] = f"{basepath}/crds/"
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

TRIM_EAST = 40   # high-x detector columns (worst edge glow)
TRIM_WEST = 16   # beyond the coronagraph/NON_SCIENCE region
TRIM_ROWS = 12

asn_file = f'{outdir}/jw02221-o002_t001_miri_f2550w_v2a_asn.json'
with open(asn_file) as fh:
    asn_data = json.load(fh)
members = [m['expname'] if os.path.isabs(m['expname'])
           else os.path.join(pipedir, m['expname'])
           for m in asn_data['products'][0]['members']]

new_members = []
for fn in members:
    f2 = fits.open(fn)
    dq = f2['DQ'].data
    ny, nx = dq.shape
    # science columns = where not everything is already DO_NOT_USE
    colgood = ((dq & 1) == 0).any(axis=0)
    sci_cols = np.where(colgood)[0]
    lo, hi = sci_cols.min(), sci_cols.max()
    dq[:, hi - TRIM_EAST + 1:] |= 1
    dq[:, :lo + TRIM_WEST] |= 1
    dq[:TRIM_ROWS, :] |= 1
    dq[-TRIM_ROWS:, :] |= 1
    f2['DQ'].data = dq
    f2[1].header['EDGETRIM'] = (f'E{TRIM_EAST} W{TRIM_WEST} R{TRIM_ROWS}',
                                'edge columns/rows DQ-flagged 2026-06-11')
    outfn = fn.replace('_align.fits', '_edgetrim.fits')
    f2.writeto(outfn, overwrite=True)
    new_members.append(outfn)
print(f'wrote {len(new_members)} edge-trimmed frames '
      f'(sci cols {lo}-{hi}, trims E{TRIM_EAST}/W{TRIM_WEST}/rows{TRIM_ROWS})')

from jwst.pipeline import calwebb_image3
name = 'jw02221-o002_t001_miri_f2550w_v4'
asn_data['products'][0]['name'] = name
asn_data['products'][0]['members'] = [
    {'expname': fn, 'exptype': 'science'} for fn in new_members]
asn_v4 = f'{outdir}/{name}_asn.json'
with open(asn_v4, 'w') as fh:
    json.dump(asn_data, fh)
os.chdir(pipedir)
calwebb_image3.Image3Pipeline.call(
    asn_v4,
    steps={'tweakreg': {'skip': True},
           'skymatch': {'skymethod': 'match', 'subtract': True,
                        'match_down': False, 'save_results': True},
           'outlier_detection': {'snr': '30.0 25.0',
                                 'good_bits': 'SATURATED, JUMP_DET'}},
    output_dir=outdir,
    save_results=True)

# seam metric vs v2a (same grid expected; the seam may MOVE west by the trim
# width, so scan a window of candidate boundary columns too)
ref = fits.open(f'{outdir}/jw02221-o002_t001_miri_f2550w_v2a_i2d.fits')['SCI'].data
v4 = fits.open(f'{outdir}/{name}_i2d.fits')['SCI'].data
for label, img in [('v2a', ref), ('v4', v4)]:
    if img.shape != ref.shape:
        print(f'{label}: shape {img.shape} != {ref.shape}; metrics in own frame')
    jumps = []
    for y0 in range(450, 800, 50):
        left = np.nanmedian(img[y0:y0+50, 1590:1605])
        right = np.nanmedian(img[y0:y0+50, 1613:1628])
        jumps.append(right - left)
    print(f'{label}: seam jumps @x=1609 y=450..800: ' + ' '.join(f'{j:+.0f}' for j in jumps))
    # NaN sanity
    fin = np.isfinite(img)
    rows = fin.any(axis=1); cols = fin.any(axis=0)
    print(f'{label}: NaN frac in bbox {1 - fin[np.ix_(rows, cols)].mean():.4f}')
# scan for the strongest residual vertical discontinuity in v4
col_med = np.nanmedian(v4[450:800, :], axis=0)
dcol = np.abs(np.diff(col_med))
top = np.argsort(np.nan_to_num(dcol))[::-1][:5]
print('v4 strongest column-to-column steps (x, step):',
      [(int(x), float(np.round(dcol[x], 1))) for x in sorted(top)])

# apply the measured astrometric correction
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
print('applied WCS correction to v4 product')
print('ALL DONE')
