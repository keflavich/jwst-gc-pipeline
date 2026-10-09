"""Per-frame comparison of the dvx0 / dvx1 m7 F150W fits (#1128).

For each frame done in both arms: median init - fit offset (dx, dy) [px] for well-fit,
non-forced rows, the number and fraction of forced-refit rows, and the median per-source
flux ratio dvx1 / dvx0 for rows matched by id.

usage: python dvx_frames.py
"""
import glob
import os

import numpy as np
from astropy.table import Table

H = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/dvafix'
P = '*_resbgsub_m7_daophot_basic.fits'


def stats(t):
    ok = ~np.asarray(t['forced_refit'], bool) & np.isfinite(t['x_fit']) & (t['flux_fit'] > 0)
    dx = np.nanmedian((t['x_init'] - t['x_fit'])[ok])
    dy = np.nanmedian((t['y_init'] - t['y_fit'])[ok])
    nf = int(np.sum(np.asarray(t['forced_refit'], bool)))
    return dx, dy, nf, len(t)


print(f'{"frame":40s} {"dx0":>7s} {"dy0":>7s} {"dx1":>7s} {"dy1":>7s} {"nf0":>5s} {"nf1":>5s} {"N0":>6s} {"N1":>6s} {"f1/f0":>7s}')
tot = np.zeros(4, int)
for f0 in sorted(glob.glob(f'{H}/tree_dvx0/F150W/{P}')):
    name = os.path.basename(f0)
    f1 = f'{H}/tree_dvx1/F150W/{name}'
    if not os.path.exists(f1):
        continue
    t0, t1 = Table.read(f0), Table.read(f1)
    s0, s1 = stats(t0), stats(t1)
    common, i0, i1 = np.intersect1d(t0['id'], t1['id'], return_indices=True)
    g = ((t0['flux_fit'][i0] > 0) & (t1['flux_fit'][i1] > 0)
         & ~np.asarray(t0['forced_refit'][i0], bool) & ~np.asarray(t1['forced_refit'][i1], bool))
    r = np.nanmedian(t1['flux_fit'][i1][g] / t0['flux_fit'][i0][g])
    tot += [s0[2], s1[2], s0[3], s1[3]]
    print(f'{name[:40]:40s} {s0[0]:+7.3f} {s0[1]:+7.3f} {s1[0]:+7.3f} {s1[1]:+7.3f} {s0[2]:5d} {s1[2]:5d} {s0[3]:6d} {s1[3]:6d} {r:7.4f}')
print('total forced dvx0 %d dvx1 %d rows dvx0 %d dvx1 %d' % tuple(tot))
