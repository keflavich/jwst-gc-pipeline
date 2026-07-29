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
from astropy.table import Table

from jwst_gc_pipeline.reduction import alignment_config as ac
from jwst_gc_pipeline.reduction.bulk_offset_step0 import (
    BULK_IN_TABLE, BULK_NONE, BulkOffsetVerificationError, bulk_tie_state,
    recorded_bulk_mas, step0_bulk_offset,
)


def _load_catalog_coords(paths):
    """Stack per-frame m1 catalogs into one SkyCoord set.

    The archive's per-frame catalogs carry their sky positions in a
    ``skycoord_centroid`` SkyCoord mixin column, not as separate RA/DEC columns
    -- the plain-column forms are accepted as a fallback for hand-made inputs.
    """
    coords = []
    for path in paths:
        tbl = Table.read(path)
        cols = {c.lower(): c for c in tbl.colnames}
        if 'skycoord_centroid' in cols:
            coords.append(SkyCoord(tbl[cols['skycoord_centroid']]).icrs)
        elif 'skycoord_ra' in cols and 'skycoord_dec' in cols:
            coords.append(SkyCoord(
                ra=np.asarray(tbl[cols['skycoord_ra']], dtype=float) * u.deg,
                dec=np.asarray(tbl[cols['skycoord_dec']], dtype=float) * u.deg,
                frame='icrs'))
        elif 'ra' in cols and 'dec' in cols:
            coords.append(SkyCoord(
                ra=np.asarray(tbl[cols['ra']], dtype=float) * u.deg,
                dec=np.asarray(tbl[cols['dec']], dtype=float) * u.deg,
                frame='icrs'))
        else:
            raise ValueError(
                f"{path}: no sky positions found. Expected a skycoord_centroid "
                f"column (what the pipeline writes) or RA/DEC; got {tbl.colnames}")
    if not coords:
        raise ValueError("no catalogs loaded")
    return SkyCoord(np.concatenate([c.ra.deg for c in coords]) * u.deg,
                    np.concatenate([c.dec.deg for c in coords]) * u.deg,
                    frame='icrs')


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
                    help='per-frame catalogs (default: the standard per-frame pattern)')
    ap.add_argument('--mtag', default='_m1',
                    help='merge-stage tag of the catalogs to read (default _m1; '
                         'a field cataloged only through m2 needs _m2)')
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

    # Same shape build_virac2_offsets._gather uses; the filter token comes FIRST
    # and the catalogs sit directly under <basepath>/<FILT>/.
    cat_glob = args.catalog_glob or (
        f'{basepath}/{filt}/{filt.lower()}_*_visit*_vgroup*_exp*'
        f'{args.mtag}_daophot_basic.fits')
    cat_paths = sorted(glob.glob(cat_glob))
    if not cat_paths:
        print(f"ERROR: no catalogs matched {cat_glob}. Either this field has not "
              f"been cataloged at {args.mtag}, or it reached a different stage -- "
              f"check which stages exist before concluding cataloging has not run.",
              file=sys.stderr)
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
    state = bulk_tie_state(args.proposal, args.field)
    if state == BULK_NONE:
        print(f"NOTE: {args.proposal}/o{args.field} has no alignment_config entry, "
              f"so there is nothing on record to verify.")
    frame_name = os.path.basename(frame_paths[0])
    visit = frame_name.split('_')[0]
    recorded = recorded_bulk_mas(basepath, args.proposal, args.field, filt, visit,
                                 float(np.median(coords.dec.deg)),
                                 frame_name=frame_name)
    if recorded is not None:
        print(f"[step0] bulk tie on record ({state}): "
              f"({recorded[0]:+.1f},{recorded[1]:+.1f}) mas")
    elif state == BULK_IN_TABLE:
        print(f"[step0] {args.proposal}/o{args.field}/{filt} is tied by an offsets "
              f"TABLE, but that table has no bulk entry for visit {visit} / {filt} "
              f"yet -- so there is nothing to verify for this band.")

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

    if not result.passed:
        print(f"\nSTEP 0 FAILED: {result.mode} did not pass "
              f"(sep={result.sep_mas}, apply_ok={result.apply_ok}). "
              f"'no exception' is not the same as 'verified'.", file=sys.stderr)
        return 1
    print(f"\n[step0] {result.mode.upper()} OK -- {result.detail}"
          f"{' (cached)' if result.from_cache else ''}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
