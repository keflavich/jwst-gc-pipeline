#!/usr/bin/env python
"""Combine the two brick proposals' merged catalogs into ONE all-band table.

The brick 'target' spans jw02221 (narrows: F182M/F187N/F212N/F405N/F410M/F466N, field
o001) and jw01182 (broadbands: F115W/F200W/F356W/F444W, field o004). Their cross-band
merges share the generic filename and clobber each other, so cataloging.py now also
writes proposal-scoped copies (`..._m8_dedup_o001.fits`, `..._o004.fits`). This script
spatially matches those two into a single 10-band catalog for PM/SED/CMD/selection work.

Match: bidirectional nearest-neighbour on skycoord_ref within --radius (same epoch, same
stars). Output carries every column of both, suffixed _2221 / _1182 on collision, plus a
unified skycoord_ref (from the 2221 side, the PM-validated frame) and a match separation.

Usage:
  python combine_brick_allband.py [--radius-mas 120] [--out <path>]
"""
import argparse, glob, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
from astropy.table import Table, hstack
from astropy.coordinates import SkyCoord
import astropy.units as u

CATDIR = '/orange/adamginsburg/jwst/brick/catalogs'
STEM = 'basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup'

def _newest(pat):
    g = sorted(glob.glob(pat), key=os.path.getmtime)
    return g[-1] if g else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--radius-mas', type=float, default=120.0,
                    help='max match separation on skycoord_ref (same epoch/stars)')
    ap.add_argument('--cat2221', default=f'{CATDIR}/{STEM}_o001.fits')
    ap.add_argument('--cat1182', default=f'{CATDIR}/{STEM}_o004.fits')
    ap.add_argument('--out', default=f'{CATDIR}/brick_allband_{STEM}.fits')
    args = ap.parse_args()

    for p, lab in ((args.cat2221, '2221 (o001)'), (args.cat1182, '1182 (o004)')):
        if not os.path.exists(p):
            raise SystemExit(f"[combine] missing {lab} catalog: {p}\n"
                             f"  regenerate its merge first (the proposal-scoped copy is written by "
                             f"cataloging._maybe_dedup_m8).")
    t22 = Table.read(args.cat2221); t11 = Table.read(args.cat1182)
    print(f"2221: {len(t22)} rows; 1182: {len(t11)} rows", flush=True)
    sc22 = SkyCoord(t22['skycoord_ref']); sc11 = SkyCoord(t11['skycoord_ref'])

    # bidirectional NN match on the shared frame
    idx, sep, _ = sc22.match_to_catalog_sky(sc11)
    idxb, _, _ = sc11.match_to_catalog_sky(sc22)
    keep = (np.arange(len(sc22)) == idxb[idx]) & (sep.to(u.mas).value < args.radius_mas)
    print(f"matched {keep.sum()} / {len(sc22)} (2221) within {args.radius_mas:.0f} mas; "
          f"median sep {np.median(sep.to(u.mas).value[keep]):.1f} mas", flush=True)

    # matched all-band table (suffix collisions), then append the unmatched from each side
    a = t22[keep]; b = t11[idx[keep]]
    matched = hstack([a, b], table_names=['2221', '1182'], uniq_col_name='{col_name}_{table_name}')
    matched['sep_2221_1182_mas'] = sep.to(u.mas).value[keep]
    matched['match_flag'] = 'both'
    only22 = t22[~keep]; only22['match_flag'] = 'only2221'
    um = np.ones(len(t11), bool); um[idx[keep]] = False
    only11 = t11[um]; only11['match_flag'] = 'only1182'
    print(f"unmatched: 2221-only {len(only22)}, 1182-only {len(only11)}", flush=True)

    matched.meta['COMBINE'] = (f'brick all-band: 2221(o001)+1182(o004) matched on skycoord_ref '
                               f'<{args.radius_mas:.0f}mas; matched={keep.sum()} '
                               f'only2221={len(only22)} only1182={len(only11)}')
    matched.write(args.out, overwrite=True)
    print(f"wrote matched all-band core -> {args.out} ({len(matched)} rows, "
          f"{len(matched.colnames)} cols)", flush=True)
    print("NOTE: unmatched-source union not stacked (differing columns); matched core is the "
          "all-band table. Extend to a full outer join if single-proposal sources are needed.", flush=True)

if __name__ == '__main__':
    main()
