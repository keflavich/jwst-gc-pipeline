"""Masked blobs near bright-window detections: area of the blob the hole guard sees and the distance to the nearest masked pixel.

usage: python holeguard.py TILE:BAND:MODULE:X,Y [TILE:BAND:MODULE:X,Y ...] [--root /orange/adamginsburg/jwst/gc-treasury]
The mask is the one of ``_build_i2d_augmented_seed``: a pixel of the m6 residual i2d (SCI) that is not finite or has a weight (WHT) of 0 or less.  Per position the output is
the number of masked pixels in the 8-connected blob nearest to it (within 4 px), that distance, and whether the guard (blob of at least 20 pixels within 4 px) drops it.
Reads the gc-treasury tree and writes nothing.
"""
import sys

import numpy as np
from astropy.io import fits
from scipy import ndimage

args = [x for x in sys.argv[1:] if not x.startswith('--')]
ROOT = sys.argv[sys.argv.index('--root') + 1] if '--root' in sys.argv else '/orange/adamginsburg/jwst/gc-treasury'
if '--root' in sys.argv:
    args.remove(ROOT)
cache = {}
for spec in args:
    tile, band, mod, xy = spec.split(':')
    x, y = (int(round(float(v))) for v in xy.split(','))
    key = (tile, band, mod)
    if key not in cache:
        p = (f'{ROOT}/{band}/pipeline/jw10678-o{tile}_t001_nircam_clear-{band.lower()}-{mod}_resbgsub_hybpsf_m6_daophot_basic_mergedcat_residual_i2d.fits')
        with fits.open(p) as h:
            sci = np.asarray(h['SCI'].data, float)
            wht = np.asarray(h['WHT'].data, float) if 'WHT' in [e.name for e in h] else None
        mask = ~np.isfinite(sci)
        if wht is not None:
            mask |= ~np.isfinite(wht) | (wht <= 0)
        lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
        cache[key] = (lab, np.bincount(lab.ravel(), minlength=n + 1))
    lab, area = cache[key]
    r = 4
    cut = lab[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1]
    gy, gx = np.mgrid[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1]
    dist = np.hypot(gx - x, gy - y)
    inside = (cut > 0) & (dist < r)
    if not inside.any():
        print(f'o{tile} {band} {mod} ({x},{y}): no masked pixel within {r} px')
        continue
    ids = np.unique(cut[inside])
    parts = []
    for i in ids:
        d = dist[(cut == i) & inside].min()
        parts.append(f'a blob of {int(area[i])} masked px, nearest pixel {d:.1f} px away')
    big = any(area[i] >= 20 for i in ids)
    print(f'o{tile} {band} {mod} ({x},{y}): ' + '; '.join(parts) + f'; hole guard {"drops" if big else "keeps"} it')
