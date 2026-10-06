"""Injected stars found at m6 and kept at m7, per variant, split by neighbour.

Reads the ``inj_track.py`` tables.  For every injected star with an m6 vetted
match, the m6 flux ratio to the brightest brighter m6 vetted source within
2.5 FWHM (``brighter_nb_ratio``) decides its group; "isolated" has no such
neighbour.  Prints the m7 kept counts per group and per field.

usage: python inj_summary.py <inj_variant1.fits> [<inj_variant2.fits> ...]
"""
import sys

import numpy as np
from astropy.table import Table

GROUPS = (('brighter neighbour within 2.5 FWHM', lambda r: np.isfinite(r)),
          ('  ratio < 0.1', lambda r: np.isfinite(r) & (r < 0.1)),
          ('  ratio 0.1-0.3', lambda r: np.isfinite(r) & (r >= 0.1) & (r < 0.3)),
          ('  ratio >= 0.3', lambda r: np.isfinite(r) & (r >= 0.3)),
          ('isolated', lambda r: ~np.isfinite(r)))


def _str(t, c):
    return np.char.strip(np.asarray(t[c]).astype(str))


for path in sys.argv[1:]:
    t = Table.read(path)
    found6 = np.asarray(t['found_resbgsub_m6'], bool)
    kept7 = np.asarray(t['found_resbgsub_m7'], bool)
    ratio = np.asarray(t['brighter_nb_ratio'], float)
    print(f"{path}: variant {_str(t, 'variant')[0]}, "
          f"{found6.sum()} injected stars found at m6 (of {len(t)})")
    for name, sel in GROUPS:
        m = found6 & sel(ratio)
        print(f'  {name:36s} kept at m7 {kept7[m].sum():4d}/{m.sum()}')
    field = _str(t, 'field')
    for f in sorted(set(field)):
        m = found6 & np.isfinite(ratio) & (field == f)
        print(f'    {f:14s} with a brighter neighbour: kept {kept7[m].sum()}/{m.sum()}')
