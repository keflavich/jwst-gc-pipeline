#!/usr/bin/env python
"""Report where an on-disk product's FITS/SIP header disagrees with its GWCS.

The GWCS in the ASDF extension is authoritative; the SCI-header
``RA---TAN-SIP`` is a fitted approximation.  Frames written before the
tight-SIP fix carry a fit made at gwcs's ``max_pix_error=0.25`` px default and
disagree with their own GWCS by up to ~5-8 mas in a position-dependent way.
Anything that read the FITS header instead of the GWCS -- per-frame catalog
positions, the interframe overlap gate, satstar positions -- inherited that
error.

Usage
-----
    python -m scripts.release.audit_fits_gwcs_agreement \
        '/orange/adamginsburg/jwst/brick/F182M/pipeline/*_crf.fits'
    python scripts/release/audit_fits_gwcs_agreement.py --field brick --tol 0.5

Exit status is 1 if any scanned frame exceeds ``--tol`` mas.
"""
import argparse
import glob
import os
import sys
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def audit_file(fn, npoints=25):
    """``(max_mas, median_mas, a_order)`` for one product, or None if no GWCS."""
    from astropy.io import fits
    from stdatamodels.jwst.datamodels import ImageModel

    from jwst_gc_pipeline.reduction.fits_wcs_sync import fits_gwcs_discrepancy_mas

    with ImageModel(fn) as m:
        gw = getattr(m.meta, 'wcs', None)
        if gw is None:
            return None
        shape = m.data.shape
        hdr = fits.getheader(fn, 'SCI')
        max_mas, med_mas = fits_gwcs_discrepancy_mas(hdr, gw, shape,
                                                     npoints=npoints)
    return max_mas, med_mas, hdr.get('A_ORDER')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('patterns', nargs='*',
                    help='glob patterns of FITS products to audit')
    ap.add_argument('--field', help='shorthand for the standard per-frame '
                                    'products of a field under $JWST_ROOT')
    ap.add_argument('--root', default=os.environ.get(
        'JWST_ROOT', '/orange/adamginsburg/jwst'))
    ap.add_argument('--tol', type=float, default=None,
                    help='mas ceiling (default: fits_wcs_sync.FITS_GWCS_TOL_MAS)')
    ap.add_argument('--npoints', type=int, default=25)
    ap.add_argument('--limit', type=int, default=0,
                    help='audit at most this many files (0 = all)')
    args = ap.parse_args(argv)

    from jwst_gc_pipeline.reduction.fits_wcs_sync import FITS_GWCS_TOL_MAS
    tol = FITS_GWCS_TOL_MAS if args.tol is None else args.tol

    patterns = list(args.patterns)
    if args.field:
        patterns.append(os.path.join(args.root, args.field, '*', 'pipeline',
                                     '*_crf.fits'))
    if not patterns:
        ap.error('give at least one glob pattern or --field')

    files = sorted({f for p in patterns for f in glob.glob(p)})
    if args.limit:
        files = files[:args.limit]
    if not files:
        print('no files matched', file=sys.stderr)
        return 2

    nbad = 0
    print(f"{'max mas':>9} {'med mas':>9} {'ord':>4}  file")
    for fn in files:
        try:
            res = audit_file(fn, npoints=args.npoints)
        except (OSError, ValueError) as ex:
            print(f"{'ERR':>9} {'':>9} {'':>4}  {fn}  ({type(ex).__name__}: {ex})")
            continue
        if res is None:
            print(f"{'no-gwcs':>9} {'':>9} {'':>4}  {fn}")
            continue
        max_mas, med_mas, order = res
        flag = '  <-- OVER TOL' if max_mas > tol else ''
        nbad += max_mas > tol
        print(f"{max_mas:9.3f} {med_mas:9.3f} {str(order):>4}  "
              f"{os.path.basename(fn)}{flag}")

    print(f"\n{nbad}/{len(files)} frames exceed {tol} mas FITS-vs-GWCS "
          f"disagreement.")
    if nbad:
        print("These frames' FITS headers are a degraded approximation of their "
              "own GWCS. Read the GWCS (jwst_gc_pipeline.frame_wcs) or "
              "regenerate the frames.")
    return 1 if nbad else 0


if __name__ == '__main__':
    sys.exit(main())
