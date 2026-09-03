#!/usr/bin/env python
"""Cycle-2 applier: turn m2 checkpoint records into offsets-table corrections.

The m2 visit-consensus checkpoint (run in WARN_ONLY mode during a cycle-1
cataloging pass) writes records whose ``corrections`` lists are ALREADY in
``update_offsets_table`` format: per-exposure exposure-vs-consensus residuals
plus per-visit consensus-vs-reference offsets, each measured with the swept
histogram + multi-check reference ladder.  This script:

1. loads every ``checkpoint_m2_*_latest.json`` in ``--records-dir``;
2. routes each correction to its offsets table BY FILTER (the brick tables do
   not share filters: 1182 = broadbands, 2221 = narrowbands+F410M);
3. drops sub-floor corrections (default 2 mas — the checkpoint tolerance);
4. EXTENDS a per-visit table to per-exposure rows when exposure-level
   corrections target it (a per-visit row cannot express a single-exposure
   fix; ``update_offsets_table`` refuses by design);
5. with ``--apply``: applies via ``update_offsets_table`` (collapse-guard
   validated, timestamped backup, provenance columns) and stale-tags the
   affected filters' im0 ``_i2d`` mosaics.

Then REGENERATE the affected reductions (destreak overwrite -> fix_alignment
with the corrected table -> Image3) and re-run cataloging with the
checkpoints ENFORCING.  Never re-apply on top of the stale shift.
"""
import argparse
import glob
import re
import json
import os
import sys
from collections import defaultdict

import numpy as np
from astropy.table import Table

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    find_i2d_for_filter, mark_i2d_stale, update_offsets_table,
)


def load_corrections(records_dir, obs_token=None):
    """Corrections from the m2 records under ``records_dir``.

    ``obs_token`` restricts to ONE observation.  Checkpoint records are keyed
    on the observation (issue #281), and this glob is obs-blind: before the
    token existed only one record per filter survived, so a union was
    impossible.  With tokened records both survive, and unioning them lands two
    observations' corrections on the same table rows -- their `visit` fields
    are both "1".  Pass the token; omitting it on a multi-observation field is
    refused rather than silently unioned.
    """
    corrections = []
    pattern = (f"checkpoint_m2_*{obs_token}_latest.json" if obs_token
               else "checkpoint_m2_*_latest.json")
    records = sorted(glob.glob(os.path.join(records_dir, pattern)))
    # `not obs_token`, not `is None`: an EMPTY token takes the blind glob above,
    # so testing for None let `--obs-token ''` walk straight past the refusal.
    if not obs_token:
        # An UNTOKENED record is its own identity, not the absence of one.
        # Counting only tokened names meant legacy + one tokened gave
        # len(tokens) == 1 and no refusal -- and that is the state this change
        # creates on every field at its first tokened run, reachable by the
        # default invocation.  Measured on a real cloudef F360M pair:
        # records=2, corrections=42, 3 of 3 targeted rows getting two
        # corrections summed onto them; with --obs-token _o002, records=1,
        # corrections=21.
        tokens = {(m.group(1) if m else '<untokened>') for m in
                  (_TOKEN_RE.search(os.path.basename(p)) for p in records)}
        if len(tokens) > 1:
            raise SystemExit(
                f"{records_dir} holds m2 records for more than one observation "
                f"({sorted(tokens)}) and no --obs-token was given.  Unioning "
                f"them would apply two observations' corrections to the same "
                f"table rows (issue #281).")
    for path in records:
        with open(path) as fh:
            rec = json.load(fh)
        for corr in rec.get("corrections", []):
            corr["_record"] = os.path.basename(path)
            corrections.append(corr)
    return records, corrections


def _table_visit_number(visit_value):
    """Visit NUMBER (int) from an offsets-table ``Visit`` cell.  The table uses
    the full JWST visit id ``jw{prop:05d}{obs:03d}{visit:03d}`` (e.g.
    ``jw01182004002`` -> visit 2); the m2-record key uses the bare number
    (``'2'``).  A bare-number table cell is accepted too."""
    s = str(visit_value).strip()
    tail = s[-3:] if s.lower().startswith("jw") and len(s) >= 3 else s
    return int(tail)


#: Registered obsids include JOINT forms -- sgrb2 `o002-998`, sickle `o001-002`
#: -- so a bare `o\d{3}` misses them, the union goes unrefused, and scan.py's
#: keys collide back to last-wins.
_TOKEN_RE = re.compile(r"checkpoint_m2_[^_]+(_(?:o[\d-]{3,}|j\d{4,5}))_latest")


def load_exposure_universe(records_dir, obs_token=None):
    """The TRUE per-(visit, filter) exposure set of the reduction, read from the
    m2 records' full exposure enumeration -- NOT from the corrections.

    Every m2 record enumerates every exposure the checkpoint measured
    (``visits[].exposures[].key`` = ``[visit, exposure, detector, filter]``),
    whether or not it produced a correction.  Sourcing the extension universe
    from here (rather than from correction-carrying exposures) closes the
    match=0 gap the corrections-only universe leaves open: an exposure that is
    sub-floor in EVERY filter carries no correction anywhere, so it is absent
    from the corrections but still present as a real frame -- fix_alignment
    would then find 0 rows for it and raise (the exact 2026-07-15 F182M
    exp 10/11 crash).

    Keyed by ``(visit_number, filter)`` -- 1182 tiles the mosaic across two
    visits (v001/v002) whose exposure numbers overlap, so a per-filter union
    would over-generate phantom rows for the wrong visit.  Returns
    ``{(visit_int, filter): sorted list of exposure ints}``."""
    universe = defaultdict(set)
    # Token-aware for the same reason load_corrections is: `key[0]` is "1" in
    # every real record, so unioning two observations merges o002's exposures
    # 1-3 with o005's 4-6 under one (visit, filter) key and feeds that to
    # extend_table_to_per_exposure.  Half this script was tokened.
    pattern = (f"checkpoint_m2_*{obs_token}_latest.json" if obs_token
               else "checkpoint_m2_*_latest.json")
    records = sorted(glob.glob(os.path.join(records_dir, pattern)))
    # `not obs_token`, not `is None`: an EMPTY token takes the blind glob above,
    # so testing for None let `--obs-token ''` walk straight past the refusal.
    if not obs_token:
        # An UNTOKENED record is its own identity, not the absence of one.
        # Counting only tokened names meant legacy + one tokened gave
        # len(tokens) == 1 and no refusal -- and that is the state this change
        # creates on every field at its first tokened run, reachable by the
        # default invocation.  Measured on a real cloudef F360M pair:
        # records=2, corrections=42, 3 of 3 targeted rows getting two
        # corrections summed onto them; with --obs-token _o002, records=1,
        # corrections=21.
        tokens = {(m.group(1) if m else '<untokened>') for m in
                  (_TOKEN_RE.search(os.path.basename(p)) for p in records)}
        if len(tokens) > 1:
            raise SystemExit(
                f"{records_dir} holds m2 records for more than one observation "
                f"({sorted(tokens)}) and no --obs-token was given; the exposure "
                f"universe would merge them (issue #281).")
    for path in records:
        with open(path) as fh:
            rec = json.load(fh)
        for v in rec.get("visits", []):
            for e in v.get("exposures", []):
                key = e.get("key")
                if not key or len(key) < 4:
                    continue
                vnum, exp, filt = int(key[0]), int(key[1]), str(key[3])
                universe[(vnum, filt)].add(exp)
    return {k: sorted(exps) for k, exps in universe.items()}


def extend_table_to_per_exposure(table_path, universe_by_visit_filter,
                                 extend_filters):
    """Replicate per-visit rows into per-exposure rows so single-exposure
    corrections are expressible.  Filters without exposure-level corrections
    (not in ``extend_filters``) keep their single per-visit row.

    ``universe_by_visit_filter`` (``{(visit_int, filter): [exposure ints]}``,
    from ``load_exposure_universe``) is the reduction's TRUE frame set, so an
    extended filter gets a row for every real exposure it has -- including those
    whose residual sat under the correction floor and carry no correction
    (fix_alignment still requires exactly one row per real frame; 2026-07-15:
    F182M exposures 10/11 were dropped this way and every 2221 V12 reduction
    died with match=0).  The universe is applied PER (visit, filter), never a
    global or per-filter union, so a filter is never given rows for exposure
    numbers a visit lacks (phantom rows) -- 1182's two visits share exposure
    numbers.  A bulk-only exposure inherits the per-visit row's value verbatim;
    the visit-level correction is applied to every row of the visit afterwards,
    which is exactly right for it."""
    tbl = Table.read(table_path)
    if "Exposure" in tbl.colnames:
        return tbl, False
    rows = []
    extended = False
    for row in tbl:
        filt = str(row["Filter"])
        vnum = _table_visit_number(row["Visit"])
        exps = (universe_by_visit_filter.get((vnum, filt), [])
                if filt in extend_filters else [])
        if exps:
            extended = True
            for e in exps:
                r = dict(zip(tbl.colnames, [row[c] for c in tbl.colnames]))
                r["Exposure"] = int(e)
                rows.append(r)
        else:
            r = dict(zip(tbl.colnames, [row[c] for c in tbl.colnames]))
            r["Exposure"] = -1   # per-visit row: never matched by exposure
            rows.append(r)
    if not extended:
        return tbl, False
    out = Table(rows)
    return out, True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records-dir", required=True)
    p.add_argument("--table", action="append", required=True, metavar="CSV",
                   help="offsets table path (repeatable; corrections routed "
                        "by Filter membership)")
    p.add_argument("--basepath", default=None,
                   help="for --mark-stale: tree holding <FILTER>/pipeline/*_i2d")
    p.add_argument("--min-mas", type=float, default=2.0,
                   help="drop corrections below this magnitude (checkpoint "
                        "tolerance; sub-floor 'corrections' are noise)")
    p.add_argument("--visit-level-only", action="store_true",
                   help="apply only visit-level (reference-tie) corrections")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--pool", action="store_true",
                   help="Pool per-detector corrections to the granularity of the offsets table before applying them (module-family rows cannot express a per-detector shift, and un-pooled corrections are SUMMED onto the shared row).  This is what the one-correction-per-row refusal asks for.")
    p.add_argument("--obs-token", default=None,
                   help="Restrict to ONE observation's m2 records (e.g. _o002, "
                        "or _j6778 on a shared tree -- ngc6334's records are "
                        "named that way, see naming.SHARED_TREE_PROPOSALS). "
                        "Checkpoint records are keyed on the observation "
                        "(issue #281); without this, a directory holding more "
                        "than one observation is REFUSED rather than unioned, "
                        "because their `visit` fields are both \"1\" and the "
                        "corrections would land on the same table rows.")
    p.add_argument("--mark-stale", action="store_true",
                   help="rename the affected filters' im0 _i2d mosaics to "
                        "*_im0_badastrom.fits.  Scoped by --obs-token: without "
                        "it, every observation's mosaics in the shared "
                        "<FILTER>/pipeline directory are tagged, not just the "
                        "corrected one's.  A --obs-token that matches no "
                        "product in a <FILTER>/pipeline directory RAISES "
                        "rather than tagging zero mosaics")
    args = p.parse_args(argv)

    records, corrections = load_corrections(args.records_dir, args.obs_token)
    print(f"{len(records)} m2 records -> {len(corrections)} raw corrections")
    if not corrections:
        print("nothing to do")
        return 0
    # true per-(visit, filter) frame set (all measured exposures) = extension universe
    universe_by_visit_filter = load_exposure_universe(args.records_dir,
                                                     args.obs_token)
    if universe_by_visit_filter:
        print("exposure universe per (visit, filter) from records: "
              + ", ".join(f"v{vnum}/{f}:{len(e)}"
                          for (vnum, f), e in sorted(universe_by_visit_filter.items())))

    kept = []
    for c in corrections:
        mag = float(np.hypot(c["dra_onsky_mas"], c["ddec_onsky_mas"]))
        if mag < args.min_mas:
            continue
        if args.visit_level_only and c.get("exposure") is not None:
            continue
        kept.append(c)
    print(f"{len(kept)} corrections above {args.min_mas} mas"
          + (" (visit-level only)" if args.visit_level_only else ""))

    # route by filter to the table that carries that filter
    tables = {}
    for tp in args.table:
        t = Table.read(tp)
        tables[tp] = {str(f) for f in t["Filter"]}
    routed = defaultdict(list)
    unrouted = []
    for c in kept:
        for tp, filters in tables.items():
            if c["filtername"] in filters:
                routed[tp].append(c)
                break
        else:
            unrouted.append(c)
    if unrouted:
        missing = sorted({c["filtername"] for c in unrouted})
        print(f"ERROR: {len(unrouted)} correction(s) match no table "
              f"(filters {missing})", file=sys.stderr)
        if len(tables) == 1:
            # The single-table case is not a routing mistake -- there is no
            # other table it could have gone to.  The filter is simply absent
            # from this one, and this script UPDATES existing rows
            # (`update_offsets_table`); it cannot create a filter's first row.
            # Say so here rather than letting it run on and die further down on
            # "correction ... matches NO row", which reads like a curation bug.
            # crowded_l3 hit exactly this: its table held only the 11 F070W
            # rows its first m2 pass seeded, so the other 13 filters had no
            # rows to update.
            only = next(iter(tables))
            print(f"  {os.path.basename(only)} carries no row for "
                  f"{', '.join(missing)}.  This script UPDATES rows that "
                  f"already exist; it cannot seed a filter's first row.  Seed "
                  f"those filters with the checkpoint runner, which calls "
                  f"seed_offsets_table_from_consensus:\n"
                  f"    python scripts/reduction/run_astrometry_checkpoint.py "
                  f"--stage m2 --filter <FILT> \\\n"
                  f"        --catalog-glob '<basepath>/<FILT>/<filt>_*_visit*_"
                  f"exp*_m2_daophot_basic.fits' \\\n"
                  f"        --refcat <refcat> --basepath <basepath> --apply "
                  f"--offsets-table {only} --mark-stale\n"
                  f"  Then re-run this script for the cycle-N updates.",
                  file=sys.stderr)
        return 2

    stale_filters = set()
    for tp, corrs in routed.items():
        by_kind = defaultdict(int)
        for c in corrs:
            by_kind["exposure" if c.get("exposure") is not None else "visit"] += 1
        print(f"\n{os.path.basename(tp)}: {len(corrs)} corrections "
              f"({dict(by_kind)})")
        for c in sorted(corrs, key=lambda c: -abs(np.hypot(
                c['dra_onsky_mas'], c['ddec_onsky_mas'])))[:8]:
            print(f"   {c['filtername']} visit {c['visit']} "
                  f"exp {c.get('exposure')}: "
                  f"({c['dra_onsky_mas']:+.2f}, {c['ddec_onsky_mas']:+.2f}) mas "
                  f"[{c.get('source', '')}]")
        stale_filters |= {c["filtername"] for c in corrs}

        if not args.apply:
            continue
        # a filter needs per-exposure rows iff it carries an exposure-level
        # correction; the exposures it then gets are its TRUE frame set from the
        # records (universe_by_filter), not just the corrected ones.
        extend_filters = {c["filtername"] for c in corrs
                          if c.get("exposure") is not None}
        tbl, extended = extend_table_to_per_exposure(
            tp, universe_by_visit_filter, extend_filters)
        if extended:
            backup = tp + ".pre_perexp_extension"
            os.replace(tp, backup)
            tbl.write(tp, overwrite=True)
            print(f"   table extended to per-exposure rows "
                  f"(original -> {os.path.basename(backup)})")
        update_offsets_table(tp, corrs, "m2cycle2", pool=args.pool)
        print(f"   APPLIED -> {tp} (backup + provenance columns written)")

    if args.apply and args.mark_stale and args.basepath:
        for filt in sorted(stale_filters):
            renames = mark_i2d_stale(
                # --obs-token already restricts which records are read; carry
                # the same restriction into the stale-tag, or a run scoped to
                # one observation still quarantines every other observation's
                # mosaics in the shared <FILTER>/pipeline directory.
                find_i2d_for_filter(args.basepath, filt,
                                    observation=args.obs_token),
                reason="m2 cycle-2 checkpoint corrections applied; im0 stale",
                record_dir=os.path.join(args.basepath, "astrometry_checkpoints"))
            for old, new in renames:
                print(f"   stale-tagged: {os.path.basename(new)}")
    if args.apply:
        print("\nNEXT: regenerate the affected reductions from _cal "
              "(destreak -> fix_alignment(corrected table) -> Image3), then "
              "re-run cataloging with checkpoints ENFORCING "
              "(no ASTROM_CHECKPOINT_WARN_ONLY).")
    else:
        print("\nDRY RUN (no --apply): nothing modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
