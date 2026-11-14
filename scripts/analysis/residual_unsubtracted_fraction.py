#!/usr/bin/env python
"""Unsubtracted-star fraction per magnitude bin for one residual image (#972).

Reads a pre-subtraction image, the residual image of the same frame and a
table of star positions (0-indexed pixel ``x``/``y`` on that frame) with a
magnitude column, and prints the fraction left above ``res_frac > 0.5`` and
below ``res_frac < -0.2`` per 0.5-mag bin.  The estimator lives in
``jwst_gc_pipeline.photometry.residual_census``.

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
    bin_edges, format_table, residual_fraction, unsubtracted_fraction_by_bin)


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
    tbl = Table.read(args.stars)
    frac = residual_fraction(pre, resid, np.asarray(tbl[args.x_col], float),
                             np.asarray(tbl[args.y_col], float))
    edges = bin_edges(args.lo, args.hi, args.step)
    print(format_table(unsubtracted_fraction_by_bin(
        np.asarray(tbl[args.mag_col], float), frac, edges)))


if __name__ == '__main__':
    main()
