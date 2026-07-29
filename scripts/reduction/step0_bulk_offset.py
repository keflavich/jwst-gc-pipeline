#!/usr/bin/env python
"""Run pipeline step 0 -- settle the BULK astrometric offset -- for one field.

Step 0 answers one question before any product is built: *is this field's tie to
the absolute reference frame the one we think it is?*

* When a bulk offset is already on record (every field currently in the
  pipeline), this VERIFIES it and changes nothing.  A disagreement is a hard
  failure: it means either the frames moved to a new reduction generation, or
  the recorded value is wrong.
* When there is no record (new data), this MEASURES it and writes a record for
  ``alignment_config.py`` to adopt.  A measurement that
  ``measure_reference_tie`` will not sign off on is refused rather than recorded.

The expensive re-measure is skipped when nothing that could change the answer
has changed -- see ``bulk_offset_step0.generation_hash``.

ORDERING: this needs source positions, so **m1 per-frame cataloging must have
run**.  It reads catalogs built on the raw-WCS frames; a rigid field shift moves
every source identically, so a pre-alignment catalog measures the bulk offset
perfectly well.

Examples
--------
Verify a field whose bulk offset is recorded::

    python scripts/reduction/step0_bulk_offset.py --target sgrc --proposal 4147 \
        --field 012 --filter F212N

Measure a new field (nothing recorded yet)::

    python scripts/reduction/step0_bulk_offset.py --target newfield --proposal 9999 \
        --field 001 --filter F212N --allow-measure
"""

import argparse
import glob
import os
import sys

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack

from jwst_gc_pipeline.reduction import alignment_config as ac
from jwst_gc_pipeline.reduction.bulk_offset_step0 import (
    BulkOffsetVerificationError, step0_bulk_offset,
)


def _load_catalog_coords(paths):
    """Stack per-frame m1 catalogs into one SkyCoord set."""
    tables = []
    for path in paths:
        tbl = Table.read(path)
        cols = {c.lower(): c for c in tbl.colnames}
        if 'skycoord_ra' in cols and 'skycoord_dec' in cols:
            tables.append(tbl[[cols['skycoord_ra'], cols['skycoord_dec']]])
        elif 'ra' in cols and 'dec' in cols:
            keep = tbl[[cols['ra'], cols['dec']]]
            keep.rename_columns(keep.colnames, ['skycoord_ra', 'skycoord_dec'])
            tables.append(keep)
        else:
            raise ValueError(f"{path}: no RA/DEC (or skycoord_ra/dec) columns; "
                             f"found {tbl.colnames}")
    if not tables:
        raise ValueError("no catalogs loaded")
    stacked = vstack(tables)
    return SkyCoord(ra=np.asarray(stacked['skycoord_ra'], dtype=float) * u.deg,
                    dec=np.asarray(stacked['skycoord_dec'], dtype=float) * u.deg,
                    frame='icrs')


def _recorded_bulk_mas(cfg, visit, filtername, dec_deg):
    """The bulk offset on record for this (visit, filter), as ON-SKY mas.

    ``alignment_config`` stores coordinate-convention arcsec; convert back to
    on-sky mas so it is comparable with a measurement.
    """
    if cfg is None or cfg.source != ac.RECORDED_BULK:
        return None
    dra, ddec, found = ac.lookup_recorded_bulk(cfg, visit, filtername)
    if not found:
        return None
    cosd = np.cos(np.radians(dec_deg))
    return (dra * cosd * 1000.0, ddec * 1000.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target', required=True, help='field directory name, e.g. sgrc')
    ap.add_argument('--proposal', required=True)
    ap.add_argument('--field', required=True, help='observation number, e.g. 012')
    ap.add_argument('--filter', dest='filtername', required=True)
    ap.add_argument('--basepath', default=None,
                    help='default /orange/adamginsburg/jwst/<target>')
    ap.add_argument('--refcat', default=None,
                    help='gaia+virac2 seed refcat (default: newest in <basepath>/catalogs)')
    ap.add_argument('--catalog-glob', default=None,
                    help='m1 per-frame catalogs (default: the standard m1 pattern)')
    ap.add_argument('--frame-glob', default=None,
                    help='frames whose WCS generation is hashed')
    ap.add_argument('--allow-measure', action='store_true',
                    help='permit MEASURE mode when nothing is on record (new data)')
    ap.add_argument('--force', action='store_true',
                    help='re-measure even when the generation hash is unchanged')
    args = ap.parse_args(argv)

    basepath = args.basepath or f'/orange/adamginsburg/jwst/{args.target}'
    filt = args.filtername.upper()

    refcat = args.refcat
    if refcat is None:
        cands = sorted(glob.glob(f'{basepath}/catalogs/gaia_virac2_refcat*.fits'))
        if not cands:
            print(f"ERROR: no gaia_virac2_refcat*.fits in {basepath}/catalogs; "
                  f"pass --refcat", file=sys.stderr)
            return 2
        refcat = cands[-1]

    cat_glob = args.catalog_glob or (
        f'{basepath}/{filt}/pipeline/*_exp?????_{filt.lower()}*_daophot_basic.fits')
    cat_paths = sorted(glob.glob(cat_glob))
    if not cat_paths:
        print(f"ERROR: no m1 catalogs matched {cat_glob}. Step 0 needs m1 to have "
              f"run -- catalog the raw-WCS frames first.", file=sys.stderr)
        return 2

    frame_glob = args.frame_glob or f'{basepath}/{filt}/pipeline/*_destreak.fits'
    frame_paths = sorted(glob.glob(frame_glob))
    if not frame_paths:
        print(f"ERROR: no frames matched {frame_glob}", file=sys.stderr)
        return 2

    from jwst_gc_pipeline.photometry.visit_consensus import load_reference_catalog
    ref = load_reference_catalog(refcat)
    coords = _load_catalog_coords(cat_paths)

    cfg = ac.resolve(args.proposal, args.field)
    if cfg is None:
        print(f"NOTE: {args.proposal}/o{args.field} has no alignment_config entry, "
              f"so there is nothing on record to verify.")
    visit = os.path.basename(frame_paths[0]).split('_')[0]
    recorded = _recorded_bulk_mas(cfg, visit, filt,
                                  float(np.median(coords.dec.deg)))

    if recorded is None and not args.allow_measure:
        print(f"REFUSING to measure: {args.proposal}/o{args.field}/{filt} has no "
              f"recorded bulk offset, and measuring one writes a new tie. Re-run "
              f"with --allow-measure if this really is new data.", file=sys.stderr)
        return 3

    print(f"[step0] {args.target} {args.proposal}/o{args.field}/{filt}: "
          f"{len(cat_paths)} catalogs, {len(frame_paths)} frames, refcat "
          f"{os.path.basename(refcat)}")

    try:
        result = step0_bulk_offset(
            coords, ref['all'], ref['sparse'], frame_paths, basepath,
            args.proposal, args.field, filt, recorded_mas=recorded,
            reference_id=(cfg.reference_frame if cfg else 'unknown'),
            ref_mag=ref.get('mag'), force=args.force)
    except BulkOffsetVerificationError as ex:
        print(f"\nSTEP 0 FAILED\n{ex}", file=sys.stderr)
        return 1

    print(f"\n[step0] {result.mode.upper()} OK -- {result.detail}"
          f"{' (cached)' if result.from_cache else ''}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
