"""Recovery benchmark: score catalog completeness against a hand-curated
DS9/CARTA region file of must-detect sources.

Built for the Brick F182M "selected stars" test bed (34 by-eye sources in the
zero-background extinction-wall clump, ``jwst_cat_f182m_selected_stars.reg``)
but generic: any region file whose ``point`` regions (and optionally the
centers of ``box`` regions) mark true stars, scored against any catalog with a
``skycoord`` (or ra/dec) column.

Matching is ONE-TO-ONE greedy by separation: in a sub-FWHM clump a single
catalog row must not be allowed to "recover" two distinct targets, which is
exactly the failure mode (blended pairs merged into one detection) the test
bed exists to measure.

Usage::

    python -m jwst_gc_pipeline.photometry.evaluate_region_recovery \
        --regions /blue/.../jwst_cat_f182m_selected_stars.reg \
        --tolerance-arcsec 0.1 \
        catalog1.fits [catalog2.fits ...]

Prints a per-target table (recovered? separation, flux, qfit, nmatch) and a
summary line per catalog, so strategies can be A/B'd on the same targets.
"""
import argparse
import os
import re

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table


def load_region_targets(reg_path, include_boxes=True):
    """Parse point (and optionally box-center) sky positions from a DS9/CARTA
    region file.  Returns (SkyCoord array, list-of-kind-strings)."""
    ras, decs, kinds = [], [], []
    pat = re.compile(r'^\s*(point|box)\(([-0-9.]+),\s*([-0-9.]+)')
    with open(reg_path) as fh:
        for line in fh:
            m = pat.match(line)
            if not m:
                continue
            kind = m.group(1)
            if kind == 'box' and not include_boxes:
                continue
            ras.append(float(m.group(2)))
            decs.append(float(m.group(3)))
            kinds.append(kind)
    if not ras:
        raise ValueError(f"no point/box regions parsed from {reg_path}")
    return SkyCoord(ras * u.deg, decs * u.deg, frame='icrs'), kinds


def _catalog_skycoord(cat):
    for sc_col in ('skycoord', 'skycoord_ref'):
        if sc_col in cat.colnames:
            return SkyCoord(cat[sc_col])
    for ra_col, dec_col in (('ra', 'dec'), ('RA', 'DEC'), ('ra_deg', 'dec_deg')):
        if ra_col in cat.colnames and dec_col in cat.colnames:
            return SkyCoord(np.asarray(cat[ra_col]) * u.deg,
                            np.asarray(cat[dec_col]) * u.deg)
    raise ValueError(f"no skycoord/ra/dec column in {cat.colnames[:10]}...")


def match_targets(targets, cat_coords, tolerance_arcsec):
    """Greedy one-to-one match: repeatedly take the globally closest
    (target, source) pair under the tolerance; each source recovers at most
    one target.  Returns (matched_source_index or -1, separation_arcsec) per
    target."""
    tol = float(tolerance_arcsec)
    n_t = len(targets)
    # candidate pairs within tolerance
    pairs = []  # (sep, ti, si)
    for ti in range(n_t):
        seps = targets[ti].separation(cat_coords).arcsec
        for si in np.where(seps < tol)[0]:
            pairs.append((seps[si], ti, int(si)))
    pairs.sort()
    match_idx = np.full(n_t, -1, dtype=int)
    match_sep = np.full(n_t, np.nan)
    used_sources = set()
    for sep, ti, si in pairs:
        if match_idx[ti] >= 0 or si in used_sources:
            continue
        match_idx[ti] = si
        match_sep[ti] = sep
        used_sources.add(si)
    return match_idx, match_sep


def evaluate_catalog(cat_path, targets, kinds, tolerance_arcsec, verbose=True):
    cat = Table.read(cat_path)
    coords = _catalog_skycoord(cat)
    match_idx, match_sep = match_targets(targets, coords, tolerance_arcsec)
    n_rec = int(np.sum(match_idx >= 0))

    if verbose:
        print(f"\n=== {os.path.basename(cat_path)} "
              f"({len(cat)} rows, tol {tolerance_arcsec}\") ===")
        hdr = f"{'#':>3} {'kind':5} {'ra':>12} {'dec':>12} {'rec':>3} {'sep\"':>6}"
        extra_cols = [c for c in ('flux', 'qfit', 'nmatch', 'group_size',
                                  'prominence') if c in cat.colnames]
        for c in extra_cols:
            hdr += f" {c:>10}"
        print(hdr)
        for ti in range(len(targets)):
            si = match_idx[ti]
            row = (f"{ti:>3} {kinds[ti]:5} {targets[ti].ra.deg:12.6f} "
                   f"{targets[ti].dec.deg:12.6f} "
                   f"{'Y' if si >= 0 else '.':>3} "
                   f"{match_sep[ti]:6.3f}" if si >= 0 else
                   f"{ti:>3} {kinds[ti]:5} {targets[ti].ra.deg:12.6f} "
                   f"{targets[ti].dec.deg:12.6f} {'.':>3} {'--':>6}")
            if si >= 0:
                for c in extra_cols:
                    val = cat[c][si]
                    try:
                        row += f" {float(val):>10.3g}"
                    except (TypeError, ValueError):
                        row += f" {str(val):>10}"
            print(row)
        n_pt = sum(1 for k in kinds if k == 'point')
        n_pt_rec = sum(1 for ti in range(len(targets))
                       if kinds[ti] == 'point' and match_idx[ti] >= 0)
        print(f"--- recovered {n_rec}/{len(targets)} total "
              f"({n_pt_rec}/{n_pt} points)")
    return match_idx, match_sep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('catalogs', nargs='+')
    ap.add_argument('--regions', required=True)
    ap.add_argument('--tolerance-arcsec', type=float, default=0.1)
    ap.add_argument('--no-boxes', action='store_true',
                    help='score only point regions (skip box centers)')
    args = ap.parse_args()

    targets, kinds = load_region_targets(args.regions,
                                         include_boxes=not args.no_boxes)
    print(f"{len(targets)} targets from {args.regions} "
          f"({sum(1 for k in kinds if k == 'point')} points, "
          f"{sum(1 for k in kinds if k == 'box')} box centers)")
    summary = []
    for cp in args.catalogs:
        mi, _ = evaluate_catalog(cp, targets, kinds, args.tolerance_arcsec)
        summary.append((os.path.basename(cp), int(np.sum(mi >= 0)), len(targets)))
    print("\n=== SUMMARY ===")
    for name, nrec, ntot in summary:
        print(f"{nrec:>3}/{ntot} {name}")


if __name__ == '__main__':
    main()
