#!/usr/bin/env python
"""
Create a display/RGB-quality *_i2d_filled.fits with small interior NaN holes
interpolated (2026-06-11).

The brick F2550W mosaic has ~100 small interior NaN clusters that repeat
once per tile at the same detector location (x~370, y~295-325): a
NON_SCIENCE + NO_FLAT_FIELD pixel cluster at the coronagraph-region
boundary.  No flat exists there and the dithers are too small to cover it,
so the pixels are unrecoverable from data; for display products we
interpolate them (Gaussian sigma=3) and record the filled pixels in a
FILLED image extension + NFILLED header keyword.  Science measurements
should use the unfilled canonical i2d.

Usage: make_filled_i2d.py <i2d.fits> [max_hole_px=500]
"""
import sys
import numpy as np
from astropy.io import fits
from astropy.io.fits import ImageHDU
from astropy.convolution import convolve, Gaussian2DKernel
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

fn = sys.argv[1]
maxhole = int(sys.argv[2]) if len(sys.argv) > 2 else 500

fh = fits.open(fn)
d = fh['SCI'].data
fin = np.isfinite(d)
lab, nl = ndimage.label(~fin)
sizes = ndimage.sum(~fin, lab, range(1, nl + 1))
fillmask = np.zeros(d.shape, dtype=bool)
nfill = 0
for i, sl in enumerate(ndimage.find_objects(lab)):
    if sizes[i] > maxhole:
        continue
    y0, y1 = max(0, sl[0].start - 3), min(d.shape[0], sl[0].stop + 3)
    x0, x1 = max(0, sl[1].start - 3), min(d.shape[1], sl[1].stop + 3)
    if fin[y0:y1, x0:x1].mean() > 0.5:   # interior hole, not footprint edge
        fillmask[lab == i + 1] = True
        nfill += 1
print(f'filling {int(fillmask.sum())} px in {nfill} interior holes (<= {maxhole} px)')
sm = convolve(d, Gaussian2DKernel(3), nan_treatment='interpolate', preserve_nan=False)
fh['SCI'].data = np.where(fillmask, sm, d)
fh[1].header['NFILLED'] = (int(fillmask.sum()),
                           'interior NaN px interpolated (Gaussian s=3)')
fh.append(ImageHDU(fillmask.astype(np.uint8), name='FILLED'))
out = fn.replace('_i2d.fits', '_i2d_filled.fits')
fh.writeto(out, overwrite=True)
print('wrote', out)
