#!/usr/bin/env python
"""Census of TRULY-LOST vs any-group saturation over every delivered band.

Sums the ``DQ`` extension of ``_cal`` products and reports, per
``<field>/<filter>``, how many pixels carry ``SATURATED`` and how many of those
also carry ``DO_NOT_USE``.  The second number is what
``saturated_star_finding.truly_lost_saturated_mask`` would keep; the ratio is
the fraction of the saturation mask that survives the restriction.

This is the input to the #418 decision about making that restriction the
default.  It is a cheap DQ sum -- no ramp files, no photometry -- so it is
worth running over the whole archive rather than over a hand-picked few bands,
which is how the "no-op on NIRCam" claim in the first version of that decision
came to be wrong: it rested on two bands that happened to read 100%.

Measured 2026-08-26 over 140 bands (8 frames each), the split tracks the
``CAL_VER`` the products were reduced with, NOT the instrument:

    CAL_VER                     bands   min%   med%   max%   n at 100%
    1.14.1.dev43+g4641c6a09         5  100.0  100.0  100.0           5
    1.15.1                          1  100.0  100.0  100.0           1
    1.18.0                         13  100.0  100.0  100.0          13
    1.20.2                         20  100.0  100.0  100.0          20
    1.21.0.dev314+g61bd2fe47      101    0.5   15.7  100.0           9

Every band reduced with ``jwst <= 1.20.2`` reads 100% -- ``SATURATED`` and
``DO_NOT_USE`` are coextensive there, so the restriction is a bit-for-bit
no-op.  On current (1.21.0.dev314) products the median band keeps 15.7% of its
saturation mask, on both instruments: NIRCam ranges 4.1% (sgrc F470N) to 100%,
MIRI 0.5% (sickle F1500W) to 100%.  Neither instrument is uniformly one or the
other, so an instrument gate does not separate the two populations.

Usage::

    python scripts/analysis/truly_lost_saturation_census.py [--root DIR]
        [--nframes N] [--csv OUT.csv]
"""
import argparse
import csv
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from astropy.io import fits

# Bit values of jwst.datamodels.dqflags.pixel; hard-coded so the census runs
# without importing the (slow) jwst stack.  Checked against dqflags in the
# unit test that accompanies this script.
SATURATED = 2
DO_NOT_USE = 1

DEFAULT_ROOT = '/orange/adamginsburg/jwst'


def band_census(dirpath, nframes=8):
    """Sum SATURATED / SATURATED&DO_NOT_USE over the first ``nframes`` _cal."""
    parts = dirpath.rstrip('/').split(os.sep)
    field, filt = parts[-3], parts[-2]
    files = sorted(glob.glob(os.path.join(dirpath, '*_cal.fits')))[:nframes]
    tot_sat = tot_both = nfr = 0
    ngroups, readpatt, instrument, cal_ver = set(), set(), set(), set()
    for fn in files:
        with fits.open(fn, memmap=False) as hdul:
            if 'DQ' not in hdul:
                continue
            dq = hdul['DQ'].data
            if dq is None:
                continue
            h0 = hdul[0].header
            ngroups.add(h0.get('NGROUPS'))
            readpatt.add(h0.get('READPATT'))
            instrument.add(h0.get('INSTRUME'))
            cal_ver.add(h0.get('CAL_VER'))
            sat = (dq & SATURATED) > 0
            tot_sat += int(sat.sum())
            tot_both += int((sat & ((dq & DO_NOT_USE) > 0)).sum())
            nfr += 1
    return {
        'field': field,
        'filter': filt,
        'nframes': nfr,
        'instrument': '/'.join(sorted(str(i) for i in instrument)),
        'cal_ver': '/'.join(sorted(str(c) for c in cal_ver)),
        'ngroups': '/'.join(sorted(str(g) for g in ngroups)),
        'readpatt': '/'.join(sorted(str(r) for r in readpatt)),
        'saturated': tot_sat,
        'also_do_not_use': tot_both,
        'truly_lost_pct': (100.0 * tot_both / tot_sat) if tot_sat else float('nan'),
    }


def _sortkey(row):
    pct = row['truly_lost_pct']
    return pct if pct == pct else 1e9   # NaN last


def run(root=DEFAULT_ROOT, nframes=8, workers=16):
    dirs = sorted(glob.glob(os.path.join(root, '*', 'F*', 'pipeline')))
    if not dirs:
        return []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for r in ex.map(band_census, dirs, [nframes] * len(dirs))
                if r['nframes']]
    rows.sort(key=_sortkey)
    return rows


def summarize(rows, stream=sys.stdout):
    print(f"{'field':<22}{'filter':<9}{'inst':<8}{'CAL_VER':<26}"
          f"{'ng':<5}{'nfr':>4}{'SATURATED':>13}{'alsoDNU':>12}{'lost%':>8}",
          file=stream)
    for r in rows:
        print(f"{r['field']:<22}{r['filter']:<9}{r['instrument']:<8}"
              f"{r['cal_ver']:<26}{r['ngroups']:<5}{r['nframes']:>4}"
              f"{r['saturated']:>13,}{r['also_do_not_use']:>12,}"
              f"{r['truly_lost_pct']:>7.1f}%", file=stream)
    for inst in sorted({r['instrument'] for r in rows}):
        grp = [r for r in rows if r['instrument'] == inst and r['saturated']]
        if not grp:
            continue
        pct = np.array([r['truly_lost_pct'] for r in grp])
        print(f"\n{inst}: {len(grp)} bands  truly-lost% min={pct.min():.1f} "
              f"median={np.median(pct):.1f} max={pct.max():.1f}  "
              f"({int((pct >= 99.99).sum())} at 100%, "
              f"{int((pct <= 50).sum())} at or below 50%)", file=stream)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--nframes', type=int, default=8)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--csv', default=None)
    args = ap.parse_args(argv)

    rows = run(root=args.root, nframes=args.nframes, workers=args.workers)
    if not rows:
        print(f'no <field>/<filter>/pipeline/*_cal.fits under {args.root}')
        return 1
    summarize(rows)
    if args.csv:
        with open(args.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'\nwrote {args.csv} ({len(rows)} bands)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
