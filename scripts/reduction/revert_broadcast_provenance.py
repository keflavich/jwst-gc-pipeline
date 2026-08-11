#!/usr/bin/env python
"""Revert an offsets table whose corrections were broadcast across every visit.

The usual repair direction is the one ``reconcile_offsets_column_pairs.py``
takes: ``dra (arcsec)`` is AUTHORITATIVE because it is what the reduced pixels
were built from, and the plain pair is copied FROM it.  That script says,
correctly, that the direction is not configurable -- "if a table ever needs the
other direction, that is a re-reduction, not a column edit."

This is that case, and it comes with the re-reduction.

## When it applies

``flag_broadcast_provenance`` finds filters whose DISTINCT visits carry an
identical ``prov_*`` correction.  One correction is measured for one visit; five
pointings arcminutes apart cannot need the same one to a thousandth of a mas.
When they all carry it, a per-filter value was smeared over the visits instead of
applied to the one it was measured on -- and since the reducer reads
``dra (arcsec)`` = as-built + ``prov_*``, the smear reached every product while
the as-built pair stayed correct and unread.

gc2211 (#284), measured 2026-08-07:

    F200W  o023 o046 o049 o050        prov (-2470.1,  +2825.4) mas   spread 0.0000
    F277W  o023 o028 o046 o049 o050   prov (-7031.7, +15009.7) mas   spread 0.0000

Its five observations tie to VIRAC2 at 0.05", 0.15", 3.2", 5.6" and 22.4" -- five
different states, one correction.  The as-built pair reproduces an INDEPENDENT
swept offset-histogram measurement of each region (issue #284, two filters per
observation agreeing to ~3 mas) to 21-56 mas in RA and 1-26 mas in Dec, on
offsets spanning 0.02" to 22".  So the good value is the one nothing reads, and
reconciling would copy the bad pair over it -- the sickle #270 mistake.

## What it does

Copies the as-built plain pair ONTO ``dra (arcsec)``, and zeroes the ``prov_*``
accumulators it is discarding, recording the revert in ``prov_stage`` /
``prov_source`` so the audit trail says a revert happened rather than going
quiet.  Refuses unless ``flag_broadcast_provenance`` actually fires, so it cannot
be pointed at a table whose corrections are real.

## The detector cannot tell a broadcast from a deliberate whole-field bulk

READ THIS BEFORE RUNNING IT ON A FIELD OTHER THAN gc2211.

``flag_broadcast_provenance`` sees only that several visits carry one identical
``prov_*``.  A correction that is broad BY DESIGN looks the same --
``_is_bulk_correction`` exists precisely because some corrections are whole-field
-- so the flag is necessary, not sufficient.  What separates them is the
provenance, and this script does not read it.

cloudef is the live example.  It is flagged (F162M/F210M/F360M), and its rows
carry::

    prov_source = "m2 consensus->reference"
    visits jw02092002001, jw02092005001   -> observations 002 and 005, visit 001

which is the same visit-number collision as gc2211 (see the sibling fix), so it
is very likely a true positive.  But cloudef's ``-1775.2 mas`` is what #281's
1.80043" ``RAOFFSET`` change rests on, and ``--field cloudef --apply`` would
revert it.  Establish which it is -- by measuring the field, the way gc2211's
five states were measured -- before pointing this at it.

**The products on disk are now stale by construction.**  They were drizzled from
the pre-revert value, so this MUST be followed by a re-reduction from ``_cal``
(the destreak overwrite resets ``RAOFFSET`` so the new table is applied) --
never by re-applying on top of the frames as they stand.

Usage::

    python revert_broadcast_provenance.py --field gc2211            # dry run
    python revert_broadcast_provenance.py --field gc2211 --apply

Every write leaves a ``.pre_provrevert_<UTC>`` backup beside the table.
"""
import argparse
import glob
import os
import shutil
import sys
from datetime import datetime, timezone

import numpy as np

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    PROV_TEXT_MIN_CHARS, _widen_prov_text_columns)
from astropy.table import Table

from jwst_gc_pipeline.atomic_io import locked, write_table_atomic
from jwst_gc_pipeline.reduction.validate_offsets_table import (
    flag_broadcast_provenance)

BASE = os.environ.get("GC_BASEPATH_OVERRIDE",
                      os.environ.get("JWST_BASE", "/orange/adamginsburg/jwst"))

AS_BUILT = ("dra", "ddec")
APPLIED = ("dra (arcsec)", "ddec (arcsec)")
#: The on-sky provenance columns, under either spelling: the convention was
#: put into the name (``_onsky_``) because right ascension also has a
#: COORDINATE offset differing by cos(declination), and every live table
#: predates that.  Resolved per table rather than assumed.
_PROV_CURRENT = ("prov_dra_onsky_mas", "prov_ddec_onsky_mas")
_PROV_LEGACY = ("prov_dra_added_mas", "prov_ddec_added_mas")


def prov_columns(colnames):
    """Whichever spelling of the on-sky provenance pair this table carries."""
    return (_PROV_CURRENT if _PROV_CURRENT[0] in colnames else _PROV_LEGACY)


PROV = _PROV_LEGACY

SNAPSHOT_MARKERS = ("_backup", ".pre_", ".contaminated", ".old", ".removed_",
                    "_bak", "preclean")


def is_snapshot(path):
    return any(m in os.path.basename(path) for m in SNAPSHOT_MARKERS)


def find_tables(field):
    return sorted(p for p in glob.glob(f"{BASE}/{field}/offsets/Offsets_JWST_*.csv")
                  if not is_snapshot(p))


def revert_rows(tbl):
    """Row indices belonging to a flagged (filter) group, and the groups."""
    groups = flag_broadcast_provenance(tbl)
    if not groups:
        return np.array([], dtype=int), []
    filt = np.asarray([str(f) for f in tbl["Filter"]])
    bad = np.zeros(len(tbl), dtype=bool)
    for g in groups:
        bad |= filt == g["filter"]
    return np.flatnonzero(bad), groups


def report(path):
    tbl = Table.read(path, format="ascii.csv")
    if not all(c in tbl.colnames for c in AS_BUILT + APPLIED):
        print(f"  {os.path.basename(path)}: single-pair table, nothing to revert")
        return 0
    idx, groups = revert_rows(tbl)
    if not len(idx):
        print(f"  {os.path.basename(path)}: clean ({len(tbl)} rows)")
        return 0
    print(f"  {os.path.basename(path)}: {len(idx)} row(s) in "
          f"{len(groups)} broadcast group(s)")
    for g in groups:
        print(f"      {g['filter']}: {g['n_visits']} visits all carry prov "
              f"({g['prov_dra_mas']:+.1f},{g['prov_ddec_mas']:+.1f}) mas, "
              f"spread {g['spread_mas']:.4f} mas")
        for v in g["visits"]:
            m = (np.asarray([str(x) for x in tbl["Visit"]]) == v) & \
                (np.asarray([str(x) for x in tbl["Filter"]]) == g["filter"])
            if not m.any():
                continue
            print(f"        {v}  applied "
                  f"({np.median(np.asarray(tbl[APPLIED[0]][m], float)):+8.4f},"
                  f"{np.median(np.asarray(tbl[APPLIED[1]][m], float)):+8.4f})\""
                  f"  ->  as-built "
                  f"({np.median(np.asarray(tbl[AS_BUILT[0]][m], float)):+8.4f},"
                  f"{np.median(np.asarray(tbl[AS_BUILT[1]][m], float)):+8.4f})\"")
    return len(idx)


def revert(path, apply=False):
    tbl = Table.read(path, format="ascii.csv")
    if not all(c in tbl.colnames for c in AS_BUILT + APPLIED):
        return 0
    idx, groups = revert_rows(tbl)
    if not len(idx) or not apply:
        return len(idx)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = f"{path}.pre_provrevert_{stamp}"

    # Under the same lock update_offsets_table takes, and re-read inside it: a
    # retie loop can be writing this table, and a read-then-write window lets a
    # concurrent correction be overwritten wholesale.
    with locked(path):
        tbl = Table.read(path, format="ascii.csv")
        idx, groups = revert_rows(tbl)
        if not len(idx):
            return 0
        shutil.copy2(path, backup)
        for applied, as_built in zip(APPLIED, AS_BUILT):
            col = np.asarray(tbl[applied], dtype=float)
            col[idx] = np.asarray(tbl[as_built], dtype=float)[idx]
            tbl[applied] = col
        for c in prov_columns(tbl.colnames):
            if c in tbl.colnames:
                col = np.asarray(tbl[c], dtype=float)
                col[idx] = 0.0
                tbl[c] = col
        # Say a revert happened.  Clearing prov_* without saying so would leave
        # the table looking as though it had never been corrected at all.
        # Widen through the shared helper rather than to a hardcoded width.
        # A literal `astype("U64")` here was not merely redundant: it NARROWED
        # any column already wider than 64.  w51's live table carries a
        # 62-character source and cloudc a 70-character pooled one, and the
        # widest form the checkpoint can write is longer than both -- see
        # PROV_TEXT_MAX_CHARS, which no longer quotes a figure because the
        # spread field's width depends on an operator setting.  A U64 write
        # leaves 'm12 visit-consensus [median of 4, ptp 49.99mas: nrcb1,nrcb' of
        # it: silent truncation, on live tables, from the script whose job is to
        # repair provenance.  Not hypothetical: gc2211's table carries 240 rows
        # written by this script, and its `prov_source` is `<U27`.
        _stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        _values = {"prov_stage": "revert",
                   "prov_source": "revert_broadcast_provenance",
                   "prov_date": _stamp}
        _widen_prov_text_columns(
            tbl, max([PROV_TEXT_MIN_CHARS] + [len(v) for v in _values.values()]))
        for _col, _val in _values.items():
            if _col in tbl.colnames:
                tbl[_col][idx] = _val
        write_table_atomic(tbl, path, format="ascii.csv")

    # Re-read and verify rather than trusting the write: this edits the file the
    # reducer consumes.
    check = Table.read(path, format="ascii.csv")
    still, _ = revert_rows(check)
    if len(still):
        raise SystemExit(
            f"FAILED: {path} still shows {len(still)} broadcast row(s) after "
            f"the write. The original is at {backup} -- restore it.")
    for applied, as_built in zip(APPLIED, AS_BUILT):
        a = np.asarray(check[applied], dtype=float)[idx]
        b = np.asarray(check[as_built], dtype=float)[idx]
        if not np.allclose(a, b, atol=1e-9):
            raise SystemExit(
                f"FAILED: {applied} does not match {as_built} on the reverted "
                f"rows of {path}. The original is at {backup} -- restore it.")
    print(f"    reverted {len(idx)} row(s); backup {os.path.basename(backup)}")
    print(f"    NEXT: re-reduce this field from _cal -- every product on disk "
          f"was drizzled from the pre-revert value.")
    return len(idx)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--field", help="one field, e.g. gc2211")
    g.add_argument("--all", action="store_true",
                   help="every field under the base path")
    ap.add_argument("--apply", action="store_true",
                    help="write (default is a dry run that changes nothing)")
    args = ap.parse_args(argv)

    fields = ([args.field] if args.field else
              sorted(os.path.basename(d) for d in glob.glob(f"{BASE}/*")
                     if os.path.isdir(f"{d}/offsets")))
    total = 0
    for field in fields:
        tables = find_tables(field)
        if not tables:
            continue
        print(f"{field}:")
        for p in tables:
            n = report(p)
            if n and args.apply:
                revert(p, apply=True)
            total += n
    if not args.apply:
        print(f"\ndry run: {total} row(s) would be reverted; "
              f"re-run with --apply to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
