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
import json
import os
import sys
from collections import defaultdict

import numpy as np
from astropy.table import Table

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    find_i2d_for_filter, mark_i2d_stale, update_offsets_table,
)


def load_corrections(records_dir):
    corrections = []
    records = sorted(glob.glob(os.path.join(records_dir,
                                            "checkpoint_m2_*_latest.json")))
    for path in records:
        with open(path) as fh:
            rec = json.load(fh)
        for corr in rec.get("corrections", []):
            corr["_record"] = os.path.basename(path)
            corrections.append(corr)
    return records, corrections


def extend_table_to_per_exposure(table_path, exposures_by_filter):
    """Replicate per-visit rows into per-exposure rows so single-exposure
    corrections are expressible.  Filters without exposure-level corrections
    keep their per-visit row.

    The exposure UNIVERSE is the union of exposures seen across ALL filters
    in the records, not just the filter's own correction-carrying exposures:
    an exposure whose residual sat under the correction floor has NO
    correction, but fix_alignment still requires exactly one row for it
    (2026-07-15: F182M exposures 10/11 were dropped this way and every 2221
    V12 reduction died with match=0).  A bulk-only exposure inherits the
    per-visit row's value verbatim; the visit-level correction is applied to
    every row of the visit afterwards, which is exactly right for it."""
    tbl = Table.read(table_path)
    if "Exposure" in tbl.colnames:
        return tbl, False
    rows = []
    extended = False
    universe = sorted({e for exps in exposures_by_filter.values() for e in exps})
    for row in tbl:
        filt = str(row["Filter"])
        exps = universe if exposures_by_filter.get(filt) else []
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
    p.add_argument("--mark-stale", action="store_true")
    args = p.parse_args(argv)

    records, corrections = load_corrections(args.records_dir)
    print(f"{len(records)} m2 records -> {len(corrections)} raw corrections")
    if not corrections:
        print("nothing to do")
        return 0

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
        print(f"ERROR: {len(unrouted)} correction(s) match no table "
              f"(filters {sorted({c['filtername'] for c in unrouted})})",
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
        exp_by_filter = defaultdict(set)
        for c in corrs:
            if c.get("exposure") is not None:
                exp_by_filter[c["filtername"]].add(int(c["exposure"]))
        tbl, extended = extend_table_to_per_exposure(tp, exp_by_filter)
        if extended:
            backup = tp + ".pre_perexp_extension"
            os.replace(tp, backup)
            tbl.write(tp, overwrite=True)
            print(f"   table extended to per-exposure rows "
                  f"(original -> {os.path.basename(backup)})")
        update_offsets_table(tp, corrs, "m2cycle2")
        print(f"   APPLIED -> {tp} (backup + provenance columns written)")

    if args.apply and args.mark_stale and args.basepath:
        for filt in sorted(stale_filters):
            renames = mark_i2d_stale(
                find_i2d_for_filter(args.basepath, filt),
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
