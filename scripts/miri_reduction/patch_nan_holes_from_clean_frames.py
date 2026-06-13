#!/usr/bin/env python
"""
Surgical NaN-hole fill from clean exposures (2026-06-13).

The brick F2550W mosaic has small NaN holes (and a ring of partially-flagged
"structured junk") where a fixed-detector-location defect (NON_SCIENCE /
NO_FLAT_FIELD at the coronagraph boundary) landed.  Because the defect is at
a FIXED DETECTOR position, each pointing maps it to a different SKY position,
so most hole sky-positions ARE covered by other exposures with clean data
there -- the clean pixels just didn't reach the mosaic (edge trim / outlier
rejection of the defect-corrupted stack).

This patcher recovers them WITHOUT touching the already-good image:
 1. Find interior NaN clusters in the canonical mosaic.
 2. For each, reproject every exposure onto the mosaic grid over a small
    bounding box, keeping only DQ-clean (DO_NOT_USE==0) pixels.
 3. Median-combine the clean reprojections.
 4. Write the median ONLY into pixels that are currently NaN (default) or,
    with --also-junk, also into a dilated ring flagged as junk.  Good pixels
    are never modified.
 5. Record patched pixels in a PATCHED extension; write *_i2d_patched.fits.

Usage:
  patch_nan_holes_from_clean_frames.py <canonical_i2d> <v14_asn.json> [maxhole]
"""
import os
import sys
import json
import numpy as np
from astropy.io import fits
from astropy.io.fits import ImageHDU
from astropy.wcs import WCS
from scipy import ndimage
from reproject import reproject_interp
import warnings
warnings.filterwarnings('ignore')

canon_fn = sys.argv[1]
asn_fn = sys.argv[2]
maxhole = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
pipedir = os.path.dirname(canon_fn)

canon = fits.open(canon_fn)
d = canon['SCI'].data.astype('float64')
ww = WCS(canon['SCI'].header)
ny, nx = d.shape
fin = np.isfinite(d)

asn = json.load(open(asn_fn))
members = [m['expname'] if os.path.isabs(m['expname'])
           else os.path.join(pipedir, m['expname'])
           for m in asn['products'][0]['members']]

# interior NaN clusters (ring mostly finite -> not a footprint edge)
lab, nl = ndimage.label(~fin)
sizes = ndimage.sum(~fin, lab, range(1, nl + 1))
objs = ndimage.find_objects(lab)
holes = []
for i, sl in enumerate(objs):
    if sizes[i] > maxhole:
        continue
    y0, y1 = max(0, sl[0].start - 3), min(ny, sl[0].stop + 3)
    x0, x1 = max(0, sl[1].start - 3), min(nx, sl[1].stop + 3)
    if fin[y0:y1, x0:x1].mean() > 0.5:
        holes.append((i + 1, sl))
print(f'{len(holes)} interior NaN holes to patch (<= {maxhole} px each)')

patch_mask = np.zeros((ny, nx), dtype=bool)
patched_px = 0
unrecovered = 0
for lab_id, sl in holes:
    pad = 12
    y0, y1 = max(0, sl[0].start - pad), min(ny, sl[0].stop + pad)
    x0, x1 = max(0, sl[1].start - pad), min(nx, sl[1].stop + pad)
    sub_shape = (y1 - y0, x1 - x0)
    sub_wcs = ww[y0:y1, x0:x1]
    # which exposures overlap this box at all?  reproject clean pixels.
    stack = []
    for fn in members:
        f2 = fits.open(fn)
        sci = f2['SCI'].data
        dq = f2['DQ'].data
        clean = np.where((dq & 1) == 0, sci, np.nan)
        r, _ = reproject_interp((clean, WCS(f2['SCI'].header)), sub_wcs,
                                shape_out=sub_shape, order='bilinear')
        if np.isfinite(r).any():
            stack.append(r)
        f2.close()
    holepix = (lab[y0:y1, x0:x1] == lab_id)
    if not stack:
        unrecovered += int(holepix.sum())
        continue
    med = np.nanmedian(np.array(stack), axis=0)
    fillable = holepix & np.isfinite(med)
    d[y0:y1, x0:x1][fillable] = med[fillable]
    patch_mask[y0:y1, x0:x1] |= fillable
    patched_px += int(fillable.sum())
    unrecovered += int((holepix & ~np.isfinite(med)).sum())

print(f'patched {patched_px} px from clean exposures; '
      f'{unrecovered} px still unrecoverable (no clean coverage)')
canon['SCI'].data = d.astype('float32')
canon['SCI'].header['NPATCH'] = (patched_px, 'NaN px filled from clean exposures 2026-06-13')
canon.append(ImageHDU(patch_mask.astype(np.uint8), name='PATCHED'))
out = canon_fn.replace('_i2d.fits', '_i2d_patched.fits')
canon.writeto(out, overwrite=True)
print(f'wrote {out}')
