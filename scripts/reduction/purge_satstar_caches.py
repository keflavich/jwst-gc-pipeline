#!/usr/bin/env python
"""Sideline pre-severity-gate satstar caches so a re-run rebuilds them clean.

The saturation-severity gates (satstar-severity-gate branch) reject the
over-flagged UNSATURATED stars that contaminated the satstar catalogs (45% of
brick F182M).  But the pipeline CACHES satstar results at two levels, so a
re-run on the fixed code would silently reuse the contaminated results:

  * per-exposure ``*_m*_satstar_catalog.fits`` next to each frame
    (``load_or_make_satstar_catalog`` loads instead of re-fitting), and
  * ``catalogs/<filter>_consolidated_satstar_catalog.fits``.

This renames both to ``<name>_preseveritygate`` (recoverable, un-globbable).
Run per field+filter AFTER the severity-gate code is deployed and BEFORE the
re-cataloging job, with no cataloging job for that field/filter in the queue.

Usage:
  python purge_satstar_caches.py --field brick --filter F182M [--filter F187N]
      [--basepath /blue/adamginsburg/adamginsburg/jwst] [--execute]
"""
import argparse
import glob
import os

SUFFIX = '_preseveritygate'


def purge(basepath, field, filters, execute=False):
    n = 0
    for filt in filters:
        pats = [f'{basepath}/{field}/{filt.upper()}/pipeline/*satstar_catalog*.fits',
                f'{basepath}/{field}/catalogs/{filt.lower()}_consolidated_satstar_catalog.fits']
        for pat in pats:
            for f in sorted(glob.glob(pat)):
                if f.endswith(SUFFIX) or SUFFIX in f:
                    continue
                print(('RENAME ' if execute else 'would rename ') + f)
                if execute:
                    os.rename(f, f + SUFFIX)
                n += 1
    print(f"{'renamed' if execute else 'would rename'} {n} cache file(s)")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--field', required=True)
    ap.add_argument('--filter', action='append', required=True)
    ap.add_argument('--basepath', default='/blue/adamginsburg/adamginsburg/jwst')
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    purge(args.basepath, args.field, args.filter, execute=args.execute)


if __name__ == '__main__':
    main()
