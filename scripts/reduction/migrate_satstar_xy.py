#!/usr/bin/env python
"""Rewrite satstar catalogs to the detector-frame x_fit/y_fit vintage (#949).

Satstar catalogs written before #949 hold the fit stamp's coordinates in
``x_fit``/``y_fit``/``x_init``/``y_init`` and an ``id`` of 1 on every row.  The
detector pixels are in ``xcentroid``/``ycentroid``.  This script runs
``satstar_columns.finalize_satstar_columns`` on each such file: ``x_fit`` etc.
become detector pixels, the stamp values move to ``*_cutout`` columns, ``id``
becomes a running index and ``XYFRAME = 'detector'`` is stamped.

No refit is needed.  The pipeline's own read paths normalise old catalogs in
memory (the helper is idempotent), so this migration only tidies the files for
readers outside the pipeline.

Per-exposure ``*_satstar_catalog.fits`` (extended ones included) are rewritten
first and the consolidated ``catalogs/*_consolidated_satstar_catalog.fits``
last, so a consolidated cache never has an older mtime than the per-exposure
files it was built from (``load_satstar_catalog`` rebuilds it otherwise).
Catalogs with ``XYFRAME`` already set are left alone; catalogs without
``xcentroid`` are reported and skipped.  Each file is written to a temporary
sibling and renamed over the original.

Dry run unless ``--apply`` is given.

Usage:
  python migrate_satstar_xy.py --field brick --filter F182M [--filter F187N]
      [--basepath /blue/adamginsburg/adamginsburg/jwst] [--apply]
"""
import argparse
import glob
import os

from astropy.table import Table

from jwst_gc_pipeline.photometry.satstar_columns import (
    finalize_satstar_columns, is_detector_frame)


def satstar_files(basepath, field, filters):
    """Per-exposure files first, consolidated caches last."""
    per_exposure, consolidated = [], []
    for filt in filters:
        per_exposure += glob.glob(
            f'{basepath}/{field}/{filt.upper()}/pipeline/*_satstar_catalog.fits')
        consolidated += glob.glob(
            f'{basepath}/{field}/catalogs/{filt.lower()}_consolidated_satstar_catalog.fits')
        consolidated += glob.glob(
            f'{basepath}/{field}/catalogs/{filt.lower()}_o[0-9][0-9][0-9]_consolidated_satstar_catalog.fits')
    return sorted(set(per_exposure)) + sorted(set(consolidated))


def migrate_file(path, apply=False):
    """Returns 'migrated', 'current' (already stamped) or 'no-xcentroid'."""
    tbl = Table.read(path)
    if is_detector_frame(tbl):
        return 'current'
    if 'xcentroid' not in tbl.colnames or 'ycentroid' not in tbl.colnames:
        return 'no-xcentroid'
    if apply:
        finalize_satstar_columns(tbl)
        tmp = f'{path}.tmp{os.getpid()}'
        tbl.write(tmp, format='fits', overwrite=True)
        os.replace(tmp, path)
    return 'migrated'


def migrate(basepath, field, filters, apply=False):
    counts = {'migrated': 0, 'current': 0, 'no-xcentroid': 0}
    for path in satstar_files(basepath, field, filters):
        status = migrate_file(path, apply=apply)
        counts[status] += 1
        if status == 'migrated':
            print(('MIGRATE ' if apply else 'would migrate ') + path)
        elif status == 'no-xcentroid':
            print(f'SKIP (no xcentroid/ycentroid) {path}')
    print(f"{'migrated' if apply else 'would migrate'} {counts['migrated']} "
          f"file(s); {counts['current']} already current; "
          f"{counts['no-xcentroid']} skipped without xcentroid")
    return counts['migrated']


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--field', required=True)
    ap.add_argument('--filter', action='append', required=True)
    ap.add_argument('--basepath', default='/blue/adamginsburg/adamginsburg/jwst')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    migrate(args.basepath, args.field, args.filter, apply=args.apply)


if __name__ == '__main__':
    main()
