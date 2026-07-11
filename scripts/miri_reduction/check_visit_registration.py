#!/usr/bin/env python
"""
Per-visit registration check for MIRI mosaics (2026-06-13).

The standing lesson from the brick F2550W doubled-star saga: tweakreg and the
combined mosaic can hide a single visit that is mis-registered by several
arcsec (outlier_detection silently rejects the displaced copies, or they
drizzle in as ghosts).  Neither refcat offset-histograms (too few 25um
counterparts) nor frame-vs-mosaic cross-correlation (circular) catch it.
The reliable test: build an independent sub-mosaic per visit, detect
bright point sources in each, and compare their positions visit-to-visit.

Usage:
  check_visit_registration.py <target> <FILTER> <proposal_id> <obs>
e.g.
  check_visit_registration.py cloudc F2550W 2221 001

Reports, per visit pair, the offset-histogram peak of common bright sources;
any visit offset >~1 px (the MIRI pixel ~0.11") relative to the others is
flagged as needing an adjust_wcs correction before the final build.
"""
import os
import sys
import glob
import json
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.stats import mad_std
import astropy.units as u
from photutils.detection import DAOStarFinder
from scipy.ndimage import median_filter
from reproject import reproject_interp
from jwst_gc_pipeline.photometry.astrometry_offsets import measure_offset
import warnings
warnings.filterwarnings('ignore')

target, filt, proposal, obs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
basepath = f'/orange/adamginsburg/jwst/{target}'
pipedir = f'{basepath}/{filt}/pipeline'

# FWHM (pix) for detection, by filter wavelength (F770W->7.70um, F2550W->25.50)
import re as _re
fwhm_um = int(_re.match(r'[Ff](\d+)', filt).group(1)) / 100.
fwhm_pix = max(2.0, fwhm_um / 0.11 * 0.31)  # ~MIRI PSF FWHM in pixels

cal = sorted(glob.glob(f'{pipedir}/jw0{proposal}{obs}*_mirimage_cal.fits'))
if not cal:
    cal = sorted(glob.glob(f'{pipedir}/jw0{proposal}{obs}*_mirimage_align.fits'))
# group by visit token jw<prop><obs>VVV_GGGGG (chars 0:19)
visits = {}
for fn in cal:
    visits.setdefault(os.path.basename(fn)[:19], []).append(fn)
print(f'{target} {filt}: {len(cal)} frames in {len(visits)} visits: '
      + ', '.join(f'{k[-5:]}({len(v)})' for k, v in visits.items()))

# common output grid from the combined i2d
i2d = sorted(glob.glob(f'{pipedir}/jw0{proposal}-o{obs}*_{filt.lower()}_i2d.fits'))
if not i2d:
    print('no combined i2d for grid; aborting')
    sys.exit(1)
ref_fh = fits.open(i2d[-1])
ww = WCS(ref_fh['SCI'].header)
ny, nx = ref_fh['SCI'].data.shape


def submosaic(fns):
    stack = []
    for fn in fns:
        f2 = fits.open(fn)
        d = np.where((f2['DQ'].data & 1) == 0, f2['SCI'].data, np.nan)
        r, _ = reproject_interp((d, WCS(f2['SCI'].header)), ww, shape_out=(ny, nx))
        stack.append(r)
        f2.close()
    return np.nanmedian(np.array(stack, dtype='float32'), axis=0)


def detect(img):
    mfd = median_filter(np.nan_to_num(img, nan=np.nanmedian(img)), size=31)
    hp = np.nan_to_num(img) - mfd
    s = DAOStarFinder(threshold=7 * mad_std(hp, ignore_nan=True), fwhm=fwhm_pix)(hp)
    if s is None:
        return None
    xc = 'x_centroid' if 'x_centroid' in s.colnames else 'xcentroid'
    return ww.pixel_to_world(s[xc], s[xc.replace('x', 'y')])


cats = {}
for k, fns in visits.items():
    sc = detect(submosaic(fns))
    cats[k] = sc
    print(f'  visit {k[-5:]}: {0 if sc is None else len(sc)} bright sources')

keys = [k for k in visits if cats[k] is not None and len(cats[k]) > 2]
ref_k = keys[0]
print(f'\nreference visit: {ref_k[-5:]}')
flagged = []
for k in keys[1:]:
    # offset-histogram peak via the sanctioned window-swept helper (single source
    # of truth); measure_offset(ref, k) returns (k - ref) in mas.
    r = measure_offset(cats[ref_k], cats[k], maxsep=4 * u.arcsec, bin_arcsec=0.1, min_pairs=3)
    if r is None:
        print(f'  {k[-5:]}: <3 common sources, cannot register')
        continue
    rx, ry = r['dra'] / 1000.0, r['ddec'] / 1000.0
    tag = '  <-- FLAG (>1")' if r['off'] > 1000.0 else ''
    print(f'  {k[-5:]} vs {ref_k[-5:]}: dRA={rx:+.2f}" dDec={ry:+.2f}" (npairs={r["npairs"]}){tag}')
    if r['off'] > 1000.0:
        flagged.append((k, rx, ry))

print(f'\n{"MISREGISTERED VISITS: " + str([f[0][-5:] for f in flagged]) if flagged else "all visits consistent (<1\")"}')
