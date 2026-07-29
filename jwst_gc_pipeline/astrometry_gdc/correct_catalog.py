"""Add GDC-corrected sky positions to a per-exposure (m1) catalog.

Usage::

    python -m jwst_gc_pipeline.astrometry_gdc.correct_catalog \
        --catalog <m1 per-exposure .fits> --cal <cal/crf file> [--inplace-cols]

Reads the catalog's detector pixel positions (``x_fit``/``y_fit`` preferred;
falls back through the pipeline's canonical column conventions via
``jwst_gc_pipeline.photometry.column_utils``), builds the affine-anchored GDC
sky solution from the cal/crf frame, and writes NEW columns
``skycoord_gdc_ra``/``skycoord_gdc_dec`` (deg) plus GDC provenance metadata.
Existing skycoord columns are never overwritten; if the GDC columns already
exist the tool refuses (re-run on a fresh catalog instead).

By default the augmented table is written next to the input as
``<catalog stem>_gdc.fits``; ``--inplace-cols`` adds the columns to the input
file itself (still never touching existing columns).
"""
import argparse
import os

import numpy as np
from astropy.table import Table

from ..photometry.column_utils import _XY_COLUMN_CANDIDATES
from .gdc_wcs import GDCSkySolution, load_frame_wcs
from .stdgdc import STDGDC, detector_filter_from_header

__all__ = ['add_gdc_skycoords', 'main']

GDC_RA_COL = 'skycoord_gdc_ra'
GDC_DEC_COL = 'skycoord_gdc_dec'


def _resolve_xy_columns(tbl):
    """(xname, yname) using the pipeline's canonical preference order."""
    for xname, yname in _XY_COLUMN_CANDIDATES:
        if xname in tbl.colnames and yname in tbl.colnames:
            return xname, yname
    raise KeyError(
        f"No recognized x/y pixel columns in catalog (have {tbl.colnames}); "
        f"expected one of {_XY_COLUMN_CANDIDATES}")


def add_gdc_skycoords(cat, cal_file, root=None, version='auto',
                      fallback_filter=None, grid_n=32):
    """Append ``skycoord_gdc_ra``/``skycoord_gdc_dec`` (deg) to ``cat``.

    Modifies ``cat`` in memory and returns it (plus the solution) -- callers
    decide where to write.  Never overwrites existing columns.
    """
    if GDC_RA_COL in cat.colnames or GDC_DEC_COL in cat.colnames:
        raise ValueError(
            f"Catalog already has {GDC_RA_COL}/{GDC_DEC_COL}; refusing to "
            f"overwrite existing skycoord columns")
    xname, yname = _resolve_xy_columns(cat)
    x = np.asarray(cat[xname], dtype=float)
    y = np.asarray(cat[yname], dtype=float)

    wcs_like, header = load_frame_wcs(cal_file)
    detector, filt = detector_filter_from_header(header)
    gdc = STDGDC.load(detector, filt, root=root, version=version,
                      fallback_filter=fallback_filter)
    sol = GDCSkySolution(wcs_like, gdc, grid_n=grid_n)

    sky = sol.gdc_sky(x, y)
    cat[GDC_RA_COL] = sky.ra.deg
    cat[GDC_DEC_COL] = sky.dec.deg
    cat[GDC_RA_COL].unit = 'deg'
    cat[GDC_DEC_COL].unit = 'deg'

    prov = sol.provenance()
    cat.meta['GDCFILE'] = os.path.basename(str(prov['gdc_file']))
    cat.meta['GDCVERS'] = str(prov['gdc_version_requested'])
    cat.meta['GDCDET'] = str(prov['gdc_detector'])
    cat.meta['GDCFILT'] = str(prov['gdc_filter'])
    if prov['gdc_filter_fallback']:
        cat.meta['GDCFALLB'] = str(prov['gdc_filter_fallback'])
    cat.meta['GDCAFFX'] = ' '.join(f'{c:.9e}' for c in prov['gdc_affine_x'])
    cat.meta['GDCAFFY'] = ' '.join(f'{c:.9e}' for c in prov['gdc_affine_y'])
    cat.meta['GDCRMS'] = float(prov['gdc_affine_rms_mas'])
    cat.meta['GDCXYCOL'] = f'{xname},{yname}'
    cat.meta['GDCCAL'] = os.path.basename(str(cal_file))
    return cat, sol


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Add STDGDC-corrected sky positions to an m1 per-exposure "
                    "catalog (new skycoord_gdc_ra/dec columns; opt-in, never "
                    "overwrites existing skycoords)")
    parser.add_argument('--catalog', required=True,
                        help='m1 per-exposure catalog FITS table')
    parser.add_argument('--cal', required=True,
                        help='matching cal/crf frame (WCS + detector/filter)')
    parser.add_argument('--inplace-cols', action='store_true',
                        help='add the columns to the input catalog file '
                             'instead of writing <stem>_gdc.fits')
    parser.add_argument('--gdc-root', default=None,
                        help='STDGDC library root (default $JWST_GDC_ROOT or '
                             'the /orange mirror)')
    parser.add_argument('--gdc-version', default='auto',
                        choices=['auto', 'v1', 'v2'],
                        help="library version; 'auto' prefers v2/ when present "
                             "(peppar behaviour)")
    parser.add_argument('--fallback-filter', default=None,
                        help='neighbour filter to use if the exact filter has '
                             'no GDC (e.g. F212N for F187N); OFF by default')
    parser.add_argument('--grid-n', type=int, default=32,
                        help='affine anchor grid is N x N (default 32)')
    args = parser.parse_args(argv)

    cat = Table.read(args.catalog)
    cat, sol = add_gdc_skycoords(cat, args.cal, root=args.gdc_root,
                                 version=args.gdc_version,
                                 fallback_filter=args.fallback_filter,
                                 grid_n=args.grid_n)

    if args.inplace_cols:
        outpath = args.catalog
    else:
        stem, ext = os.path.splitext(args.catalog)
        outpath = f'{stem}_gdc{ext or ".fits"}'
    cat.write(outpath, overwrite=args.inplace_cols or not os.path.exists(outpath))
    print(f"wrote {outpath}  (gdc={cat.meta['GDCFILE']}, "
          f"affine rms={sol.affine_rms_mas:.3f} mas, "
          f"delta-field span xi=[{sol.delta_xi_mas.min():.1f},"
          f"{sol.delta_xi_mas.max():.1f}] "
          f"eta=[{sol.delta_eta_mas.min():.1f},{sol.delta_eta_mas.max():.1f}] mas)")
    return outpath


if __name__ == '__main__':
    main()
