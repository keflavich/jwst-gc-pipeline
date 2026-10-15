"""Count CR-like sharp pixels in the data mosaic at the before-snapshot positions,
before (saved stamps) vs after (current mosaic).  Sharp pixel: exceeds the max of
its 8 neighbours' median by > 8 robust sigma and holds > 50% of its 3x3 flux
above the local median."""
import json
import os
import sys

import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(S, '..'))
sys.path.insert(0, os.path.join(S, '..', '..', 'phaseloss'))
from phase_loss import _image_hdu  # noqa: E402
from compare import stamps_from  # noqa: E402


def nsharp(st):
    n = 0
    for s in st:
        if not np.isfinite(s).any():
            continue
        s = np.nan_to_num(s, nan=np.nanmedian(s))
        med = np.median(s)
        sig = 1.4826 * np.median(np.abs(s - med)) + 1e-12
        mf = median_filter(s, size=3)
        c = s[1:-1, 1:-1]
        excess = c - mf[1:-1, 1:-1]
        box = sum(np.roll(np.roll(s, dy, 0), dx, 1)[1:-1, 1:-1] for dy in (-1, 0, 1) for dx in (-1, 0, 1)) - 9 * med
        n += int(np.sum((excess > 8 * sig) & ((c - med) > 0.5 * np.maximum(box, 1e-12))))
    return n


for field in sys.argv[1:]:
    snap = json.load(open(os.path.join(S, '..', '..', 'prod_before', field, 'f212n', 'snapshot.json')))
    npz = np.load(os.path.join(S, '..', '..', 'prod_before', field, 'f212n', 'cutouts.npz'))
    with fits.open(snap['files']['data']['path'], memmap=True) as h:
        a = stamps_from(_image_hdu(h), npz['x'], npz['y'])
    print(field, 'stamps', len(a), 'sharp px before', nsharp(npz['data']), 'after', nsharp(a))
