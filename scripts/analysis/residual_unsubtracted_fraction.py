#!/usr/bin/env python
"""Unsubtracted-star fraction per magnitude bin for one residual image (#972).

Reads a pre-subtraction image, the residual image of the same frame and a
table of star positions (0-indexed pixel ``x``/``y`` on that frame) with a
magnitude column, and prints the fraction left above ``res_frac > 0.5`` and
below ``res_frac < -0.2`` per 0.5-mag bin.  The estimator lives in
``jwst_gc_pipeline.photometry.residual_census``.

Both images must be sky-free: no annulus subtraction is done, so residual sky
reads as unsubtracted flux.  Pass isolated stars well above the noise.  Stars
that cannot be measured (masked or off-image position, NaN box, non-positive
pre-subtraction peak) are counted in the ``unmeas`` column and in the summary
line, not dropped silently.

Usage::

    python scripts/analysis/residual_unsubtracted_fraction.py PRE.fits RESID.fits \\
        STARS.ecsv --mag-col mag_f480m [--x-col x] [--y-col y] [--ext SCI] \\
        [--lo 11.5 --hi 14.7 --step 0.5]
"""
import argparse

import numpy as np
from astropy.io import fits
from astropy.table import Table

from jwst_gc_pipeline.photometry.residual_census import (
    bin_edges, column_as_float, format_table, residual_fraction, unsubtracted_fraction_by_bin)


def _read(path, ext):
    with fits.open(path) as hdul:
        return np.asarray(hdul[int(ext) if ext.isdigit() else ext].data,
                          dtype=float)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('pre')
    ap.add_argument('residual')
    ap.add_argument('stars')
    ap.add_argument('--mag-col', required=True)
    ap.add_argument('--x-col', default='x')
    ap.add_argument('--y-col', default='y')
    ap.add_argument('--ext', default='SCI')
    ap.add_argument('--lo', type=float, default=11.5)
    ap.add_argument('--hi', type=float, default=14.7)
    ap.add_argument('--step', type=float, default=0.5)
    args = ap.parse_args(argv)

    pre = _read(args.pre, args.ext)
    resid = _read(args.residual, args.ext)
    if pre.shape != resid.shape:
        ap.error('pre image shape %s differs from residual shape %s'
                 % (pre.shape, resid.shape))
    tbl = Table.read(args.stars)
    mag = column_as_float(tbl[args.mag_col])
    frac = residual_fraction(pre, resid, column_as_float(tbl[args.x_col]),
                             column_as_float(tbl[args.y_col]))
    edges = bin_edges(args.lo, args.hi, args.step)
    print(format_table(unsubtracted_fraction_by_bin(mag, frac, edges)))
    n_nan_mag = int((~np.isfinite(mag)).sum())
    print('stars in table: %d; unmeasurable (non-finite res_frac): %d; '
          'non-finite magnitude: %d'
          % (len(tbl), int((~np.isfinite(frac)).sum()), n_nan_mag))


if __name__ == '__main__':
    main()
