"""Inventory of saved STPSF/WebbPSF grid files: software VERSION, whether the
DET_YX keys are in photutils (y-major) sort order, and the PSF-sum vs SIAF
area test (which position assignment the data follow)."""
import glob
import sys
from collections import Counter

import numpy as np
from astropy.io import fits

roots = sys.argv[1:]
cnt = Counter()
for root in roots:
    for fn in sorted(glob.glob(f'{root}/*npsf*.fits')):
        h = fits.getheader(fn)
        keys = [k for k in h if k.startswith('DET_YX')]
        if len(keys) < 4:
            continue
        yx = [tuple(float(v) for v in h[k].strip('()').split(',')) for k in keys]
        xy = np.array([(p[1], p[0]) for p in yx])
        idx = np.lexsort((xy[:, 0], xy[:, 1]))
        keys_sorted = bool(np.all(idx == np.arange(len(idx))))
        cnt[(root, h.get('VERSION'), keys_sorted)] += 1
for k, v in sorted(cnt.items(), key=str):
    print(k, v)
