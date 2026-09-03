#!/usr/bin/env python
"""Fetch reference catalogs over a field's FOV and cache them.

VIRAC2 (II/387): VVV NIR astrometry tied to Gaia DR3 frame, with proper motions.
GSC2.4.2 (I/353): closest Vizier proxy to the JWST operational guide-star catalog.
Gaia DR3 already cached elsewhere.  VVV DR4 (II/376) is 2MASS-tied (kept for the
documented comparison).
"""
import argparse
import os, warnings
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.vizier import Vizier

warnings.simplefilter('ignore')

#: field -> (centre, width, height).  A field needs an entry before its cache can
#: be built; ``build_virac2_offsets`` reads
#: ``{basepath}/astrometry_diag/refcache/virac2.fits`` and cannot run without it.
#: Sizes cover the full mosaic plus margin, so proper-motion propagation to the
#: observation epoch never runs off the edge of the reference.
FIELDS = {
    # brick: the original hard-coded target, values unchanged.
    'brick':  (SkyCoord(266.5378, -28.7029, unit='deg'), 11 * u.arcmin, 11 * u.arcmin),
    # sickle (3958/007): footprint measured from its own o007 i2d mosaics --
    # centre (266.5733, -28.8009), extent 2.07 x 0.89 arcmin (single module,
    # nrcb only).  5 arcmin square is generous margin on a field that small, and
    # -28.8009 agrees with the field's own mosaics; alignment_config no longer
    # carries a dec_ref_deg for 3958 (that key is RECORDED_BULK-only and sickle is
    # now TABLE_LOCKED).
    'sickle': (SkyCoord(266.5733, -28.8009, unit='deg'), 5 * u.arcmin, 5 * u.arcmin),
    # cloud E/F control field (2092/o005, t002): a separate offset pointing from cloudef.
    # Centre + extent from its own o005 F210M i2d (266.466, -28.484; 2.7 x 6.1 arcmin);
    # 10 arcmin square is generous margin for PM propagation.  It had a gaia_virac2 refcat
    # (positions) but no virac2 Ks cache, so stage-3 photometric calibration could not run.
    'cloudef_controlfield': (SkyCoord(266.4658, -28.4842, unit='deg'),
                             10 * u.arcmin, 10 * u.arcmin),
    # JWST 9438 (Schlafly): seven Galactic-plane pointings, centres from their
    # own MAST s_ra/s_dec, observed 2025.71-2025.74.  10 arcmin square each --
    # a NIRCam two-module mosaic is ~5 x 2.2 arcmin, so this is generous margin
    # for dithers and for propagating proper motions to the 2025.72 epoch.
    # Only g007 and crowded_l3 are inside VVV; VIRAC2 will return nothing for
    # the other five, which is the expected answer, not a failure.
    'g028':        (SkyCoord(279.64500,  -3.53500, unit='deg'), 10 * u.arcmin, 10 * u.arcmin),
    'g033':        (SkyCoord(281.87000,   0.59100, unit='deg'), 10 * u.arcmin, 10 * u.arcmin),
    'g041':        (SkyCoord(284.31700,   8.26600, unit='deg'), 10 * u.arcmin, 10 * u.arcmin),
    'g054':        (SkyCoord(291.24400,  19.57500, unit='deg'), 10 * u.arcmin, 10 * u.arcmin),
    'g007':        (SkyCoord(270.54615, -22.47238, unit='deg'), 10 * u.arcmin, 10 * u.arcmin),
    'crowded_l3':  (SkyCoord(268.14900, -26.36400, unit='deg'), 10 * u.arcmin, 10 * u.arcmin),
    'crowded_l20': (SkyCoord(276.88200, -11.48900, unit='deg'), 10 * u.arcmin, 10 * u.arcmin),
}
BASE = '/orange/adamginsburg/jwst'


def fetch(code, name, cols, out, ctr, width, height):
    fn = f'{out}/{name}.fits'
    if os.path.exists(fn):
        t = Table.read(fn)
        print(f"{name}: cached {len(t)} rows")
        return t
    v = Vizier(catalog=code, columns=cols, row_limit=-1)
    r = v.query_region(ctr, width=width, height=height)
    if not r:
        print(f"{name}: NO ROWS"); return None
    t = r[0]
    t.write(fn, overwrite=True)
    print(f"{name} ({code}): {len(t)} rows -> {fn}")
    print("  cols:", t.colnames)
    return t

VIRAC2_COLS = ['srcid', 'RAJ2000', 'DEJ2000', 'e_RAJ2000', 'e_DEJ2000',
               'plx', 'e_plx', 'pmRA', 'e_pmRA', 'pmDE', 'e_pmDE',
               'Chi2', 'UWE', 'Nep',
               'Jmag', 'e_Jmag', 'Hmag', 'e_Hmag', 'Ksmag', 'e_Ksmag']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--field', default='brick', choices=sorted(FIELDS),
                    help='which field FOV to fetch over (default: brick, the '
                         'original hard-coded behaviour)')
    ap.add_argument('--gsc', action='store_true',
                    help='also fetch GSC2.4.2 (I/353); VIRAC2 alone is what '
                         'build_virac2_offsets needs')
    args = ap.parse_args(argv)

    ctr, width, height = FIELDS[args.field]
    out = f'{BASE}/{args.field}/astrometry_diag/refcache'
    os.makedirs(out, exist_ok=True)
    print(f"[fetch_refs] {args.field}: {ctr.to_string('hmsdms')} "
          f"{width.to(u.arcmin):.1f} x {height.to(u.arcmin):.1f} -> {out}")
    # An existing cache is kept, never re-fetched: the tie already measured
    # against it must stay reproducible.
    fetch('II/387', 'virac2', VIRAC2_COLS, out, ctr, width, height)
    if args.gsc:
        fetch('I/353/gsc242', 'gsc242', ['all'], out, ctr, width, height)
    print("done")


if __name__ == '__main__':
    main()
