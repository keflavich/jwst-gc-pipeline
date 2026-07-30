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
run**.  A rigid field shift moves every source identically, so a pre-alignment
catalog measures the bulk offset perfectly well.

CATALOG STATE: the per-frame catalogs on this archive are usually built AFTER the
correction was applied -- the frames carry ``RAOFFSET``/``DEOFFSET`` -- so a fresh
measurement is the residual the frames still owe rather than the bulk itself.
Step 0 reads the applied value and verifies ``recorded - applied``, printing all
three numbers; ``--assume-raw-wcs`` compares against the recorded bulk instead.

VISITS: a field whose visits carry different bulk ties (brick 1182 obs004 visit
001 sits ~17" from visit 002) cannot be verified by one measurement over both.
Step 0 refuses and asks for ``--visit``.

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
    BULK_NO_ROW, BULK_NONE, BULK_TABLE_ABSENT, BULK_VISITS_DISAGREE,
    BULK_VISITS_MIXED, REFCAT_PATTERNS, BulkOffsetVerificationError,
    applied_bulk_mas, bulk_tie_state, default_catalog_glob,
    default_frame_glob, duplicate_exposure_catalogs,
    recorded_bulk_over_visits, refcat_for_frame,
    reference_frame_matches_refcat, step0_bulk_offset, verify_tolerance_mas,
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
                    help='reference catalog (default: the one under '
                         '<basepath>/catalogs matching the field\'s frame)')
    ap.add_argument('--catalog-glob', default=None,
                    help='per-frame catalogs (default: the standard per-frame pattern)')
    ap.add_argument('--mtag', default='_m1',
                    help='merge-stage tag of the catalogs to read (default _m1; '
                         'a field cataloged only through m2 needs _m2)')
    ap.add_argument('--frame-glob', default=None,
                    help='frames whose WCS generation is hashed')
    ap.add_argument('--visit', default=None,
                    help='scope the run to one visit, as the 3-digit suffix (e.g. '
                         '002). Required when a field\'s visits carry different '
                         'bulk ties -- one stacked measurement cannot verify two.')
    ap.add_argument('--assume-raw-wcs', action='store_true',
                    help='compare against the recorded bulk even when the frames '
                         'already carry an applied offset (default: verify the '
                         'residual the frames still owe)')
    ap.add_argument('--allow-measure', action='store_true',
                    help='permit MEASURE mode when nothing is on record (new data)')
    ap.add_argument('--force', action='store_true',
                    help='re-measure even when the generation hash is unchanged')
    args = ap.parse_args(argv)

    basepath = args.basepath or f'/orange/adamginsburg/jwst/{args.target}'
    filt = args.filtername.upper()
    cfg = ac.resolve(args.proposal, args.field)
    state = bulk_tie_state(args.proposal, args.field)

    refcat = args.refcat
    if refcat is None:
        refcat, cands = refcat_for_frame(
            basepath, cfg.reference_frame if cfg else '')
        if refcat is None:
            print(f"ERROR: no reference catalog in {basepath}/catalogs matching "
                  f"{list(REFCAT_PATTERNS)}; pass --refcat", file=sys.stderr)
            return 2
        if len(cands) > 1:
            print(f"[step0] {len(cands)} reference catalogs available; chose "
                  f"{os.path.basename(refcat)} for the "
                  f"{cfg.reference_frame if cfg else 'unknown'} frame")

    # See bulk_offset_step0.default_catalog_glob for why the exposure number is
    # pinned to five digits: a greedy `exp*` also matches the `_group_` variant of
    # the same exposure, and on sickle F187N (192 catalogs for 96 frames) that
    # moved the measured tie from (-11.2, -106.8) to (+73.2, -70.6) mas -- ~60 mas
    # against a 100 mas tolerance.
    cat_glob = args.catalog_glob or default_catalog_glob(
        basepath, filt, args.mtag)
    cat_paths = sorted(glob.glob(cat_glob))
    if not cat_paths:
        print(f"ERROR: no catalogs matched {cat_glob}. Either this field has not "
              f"been cataloged at {args.mtag}, or it reached a different stage -- "
              f"check which stages exist before concluding cataloging has not run.",
              file=sys.stderr)
        return 2

    # One catalog per exposure.  Several catalog STAGES exist per exposure and a
    # loose glob can match two of them, feeding the same stars in twice under two
    # different measurements -- which shifted the sickle tie by ~60 mas.
    dupes = duplicate_exposure_catalogs(cat_paths)
    if dupes:
        example = sorted(dupes.items())[0]
        print(f"ERROR: {len(dupes)} exposures matched more than one catalog, so the "
              f"same stars would be measured twice under different stages. Narrow "
              f"--catalog-glob / --mtag to exactly one stage.\n"
              f"  e.g. exposure {example[0]} matched:\n    "
              + "\n    ".join(os.path.basename(p) for p in sorted(example[1])),
              file=sys.stderr)
        return 2

    # Scoped to this observation: a filter directory holds every observation of
    # the proposal, and folding another one's frames in corrupts both the
    # generation hash and the set of visits considered.
    frame_glob = args.frame_glob or default_frame_glob(
        basepath, filt, args.proposal, args.field)
    frame_paths = sorted(glob.glob(frame_glob))
    if not frame_paths:
        print(f"ERROR: no frames matched {frame_glob}", file=sys.stderr)
        return 2

    if args.visit:
        want = str(args.visit).zfill(3)
        cat_paths = [p for p in cat_paths if f'_visit{want}_' in os.path.basename(p)]
        frame_paths = [p for p in frame_paths
                       if os.path.basename(p).split('_')[0].endswith(want)]
        if not cat_paths or not frame_paths:
            print(f"ERROR: --visit {want} left {len(cat_paths)} catalogs and "
                  f"{len(frame_paths)} frames; check the visit number",
                  file=sys.stderr)
            return 2
        print(f"[step0] scoped to visit {want}: {len(cat_paths)} catalogs, "
              f"{len(frame_paths)} frames")

    from jwst_gc_pipeline.photometry.visit_consensus import load_reference_catalog
    ref = load_reference_catalog(refcat)
    coords = _load_catalog_coords(cat_paths)

    if state == BULK_NONE:
        print(f"NOTE: {args.proposal}/o{args.field} has no alignment_config entry, "
              f"so there is nothing on record to verify.")
    # Resolve the tie over EVERY visit in play, not just whichever frame sorts
    # first -- a field spanning two differently-tied visits has no single "the"
    # bulk offset, and one stacked measurement cannot verify two of them.
    recorded, status, per_visit = recorded_bulk_over_visits(
        basepath, args.proposal, args.field, filt, frame_paths,
        float(np.median(coords.dec.deg)))
    visit = os.path.basename(frame_paths[0]).split('_')[0]

    def _per_visit_report():
        lines = []
        for vis in sorted(per_visit):
            val, st = per_visit[vis]
            shown = f"({val[0]:+.1f},{val[1]:+.1f}) mas" if val else "--"
            lines.append(f"    {vis}: {st:<16} {shown}")
        return "\n".join(lines)

    # Only "no config entry" is genuinely new data.  A declared table that is
    # missing, or present without a matching row, means we CANNOT TELL -- and
    # routing that to MEASURE would record a new tie for a field that already
    # has one.
    if status == BULK_TABLE_ABSENT:
        print(f"STEP 0 CANNOT RUN: {args.proposal}/o{args.field}/{filt} is declared "
              f"{state} in alignment_config, but the offsets table it names is not "
              f"on disk under {basepath}/offsets/. This is not new data -- it is a "
              f"missing table. Build it (build_virac2_offsets) or fix the declared "
              f"source; do NOT measure a fresh tie over the top.", file=sys.stderr)
        return 4
    if status == BULK_NO_ROW:
        print(f"STEP 0 CANNOT RUN: {args.proposal}/o{args.field} is tied "
              f"({state}), but no bulk row was found for visit {visit} / {filt}. "
              f"The field is tied; this (visit, band) is not. Rebuild the table for "
              f"this filter rather than recording a new tie.", file=sys.stderr)
        return 4
    if status in (BULK_VISITS_MIXED, BULK_VISITS_DISAGREE):
        why = ("some visits are tied and others are not"
               if status == BULK_VISITS_MIXED
               else "the visits are tied to different values")
        print(f"STEP 0 CANNOT RUN: {args.proposal}/o{args.field}/{filt} spans "
              f"{len(per_visit)} visits and {why}, so a single measurement over all "
              f"of them cannot verify any one of them:\n{_per_visit_report()}\n"
              f"  Re-run per visit with --visit <3-digit suffix>.", file=sys.stderr)
        return 4

    # A recorded tie is only comparable with a measurement in the SAME frame, and
    # a tie MEASURED against the wrong frame would be recorded in the wrong frame,
    # so this is checked on both paths.
    frame_ok, frame_detail = reference_frame_matches_refcat(
        cfg.reference_frame if cfg else '', refcat)
    if recorded is not None:
        print(f"[step0] bulk tie on record ({state}): "
              f"({recorded[0]:+.1f},{recorded[1]:+.1f}) mas")
    if not frame_ok and (recorded is not None or args.allow_measure):
        print(f"\nSTEP 0 CANNOT RUN\n{frame_detail}", file=sys.stderr)
        return 5

    # What the measurement SHOULD come out as depends on whether these catalogs
    # were built before or after the correction was applied.  On this archive it is
    # usually after -- brick F200W visit 001 frames carry an applied
    # (-17.597, +13.453) arcsec, exactly the recorded bulk -- in which case a fresh
    # measurement is the RESIDUAL, near zero.  Comparing that against the recorded
    # bulk fails by the entire size of the offset and reports a correctly-tied
    # field as broken.
    if recorded is not None:
        adra, addec, n_with, n_total = applied_bulk_mas(
            frame_paths, float(np.median(coords.dec.deg)))
        if n_with and np.hypot(adra, addec) > 1.0:
            expected = (recorded[0] - adra, recorded[1] - addec)
            print(f"[step0] {n_with}/{n_total} frames already carry an applied "
                  f"({adra:+.1f},{addec:+.1f}) mas, so these catalogs are not "
                  f"raw-WCS.")
            if args.assume_raw_wcs:
                print("[step0] --assume-raw-wcs: comparing against the "
                      "recorded value anyway.")
            else:
                owed = float(np.hypot(*expected))
                print(f"[step0] verifying the RESIDUAL still owed: recorded - "
                      f"applied = ({expected[0]:+.1f},{expected[1]:+.1f}) mas")
                if owed > verify_tolerance_mas():
                    print(f"[step0] NOTE: the applied offset differs from the "
                          f"recorded one by {owed:.1f} mas, so the frames carry "
                          f"neither the raw state nor the recorded tie. What "
                          f"follows tests the frames as they stand, not the table.")
                recorded = expected

    if recorded is None and not args.allow_measure:
        print(f"REFUSING to measure: {args.proposal}/o{args.field}/{filt} has no "
              f"alignment_config entry, and measuring one writes a new tie. Re-run "
              f"with --allow-measure if this really is new data.", file=sys.stderr)
        return 3

    print(f"[step0] {args.target} {args.proposal}/o{args.field}/{filt}: "
          f"{len(cat_paths)} catalogs, {len(coords)} sources, "
          f"{len(frame_paths)} frames, refcat {os.path.basename(refcat)}")

    try:
        result = step0_bulk_offset(
            coords, ref['all'], ref['sparse'], frame_paths, basepath,
            args.proposal, args.field, filt, recorded_mas=recorded,
            visit=(str(args.visit).zfill(3) if args.visit else None),
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
