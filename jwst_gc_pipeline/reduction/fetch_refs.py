#!/usr/bin/env python
"""Fetch reference catalogs over the brick F115W FOV and cache them.

VIRAC2 (II/387): VVV NIR astrometry tied to Gaia DR3 frame, with proper motions.
GSC2.4.2 (I/353): closest Vizier proxy to the JWST operational guide-star catalog.
Gaia DR3 already cached elsewhere.  VVV DR4 (II/376) is 2MASS-tied (kept for the
documented comparison).
"""
import os, warnings
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.vizier import Vizier

warnings.simplefilter('ignore')
OUT = '/orange/adamginsburg/jwst/brick/astrometry_diag/refcache'
os.makedirs(OUT, exist_ok=True)
CTR = SkyCoord(266.5378, -28.7029, unit='deg')
W, H = 11 * u.arcmin, 11 * u.arcmin   # cover full mosaic + margin

def fetch(code, name, cols):
    fn = f'{OUT}/{name}.fits'
    if os.path.exists(fn):
        t = Table.read(fn)
        print(f"{name}: cached {len(t)} rows")
        return t
    v = Vizier(catalog=code, columns=cols, row_limit=-1)
    r = v.query_region(CTR, width=W, height=H)
    if not r:
        print(f"{name}: NO ROWS"); return None
    t = r[0]
    t.write(fn, overwrite=True)
    print(f"{name} ({code}): {len(t)} rows -> {fn}")
    print("  cols:", t.colnames)
    return t

if __name__ == '__main__':
    fetch('II/387', 'virac2',
          ['srcid', 'RAJ2000', 'DEJ2000', 'e_RAJ2000', 'e_DEJ2000',
           'plx', 'e_plx', 'pmRA', 'e_pmRA', 'pmDE', 'e_pmDE', 'Chi2', 'UWE', 'Nep',
           'Jmag', 'e_Jmag', 'Hmag', 'e_Hmag', 'Ksmag', 'e_Ksmag'])
    fetch('I/353/gsc242', 'gsc242', ['all'])
    print("done")
