#!/usr/bin/env python
"""Reconcile an offsets table's two dra/ddec column pairs.

An offsets table can carry two column conventions for ONE quantity:

    dra / ddec                     generate_offsets_table's names
    dra (arcsec) / ddec (arcsec)   what the VIRAC2locked tables carry

``build_virac2_offsets`` writes the second as a COPY of the first, so a
builder-shaped table starts with them identical.  ``update_offsets_table`` wrote
only the ``(arcsec)`` pair, so every correction froze the plain pair one step
further behind.  On the 2026-08 archive:

    field     rows   diverged rows   worst
    cloudef    128        96          7329.1 mas
    cloudc     192        95          7876.8 mas
    sickle     120        24            95.6 mas
    sgrc        96         7             6.1 mas

The reductions are unaffected -- ``unified_alignment`` reads ``dra (arcsec)`` --
which is exactly why that pair is AUTHORITATIVE here: it is the one the reduced
pixels, the drizzled mosaics and the catalogs on disk all agree with.  This
script copies it onto the plain pair.  It never averages the two and never
prefers the plain pair, because doing either would assert a tie that no product
on disk was built from.

The direction is not configurable for that reason.  If a table ever needs the
other direction, that is a re-reduction, not a column edit.

Usage::

    # look, change nothing (default)
    python reconcile_offsets_column_pairs.py --field cloudef
    python reconcile_offsets_column_pairs.py --all

    # write, after reading the dry run
    python reconcile_offsets_column_pairs.py --field cloudef --apply

Every write leaves a ``.pre_colreconcile_<UTC>`` backup beside the table.
"""
import argparse
import glob
import os
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
from astropy.table import Table

BASE = os.environ.get("JWST_BASE", "/orange/adamginsburg/jwst")

#: (authoritative, stale).  NOT a preference order -- a direction.  The reducer
#: reads the first, so it is what every product on disk was built from.
AUTHORITATIVE = ("dra (arcsec)", "ddec (arcsec)")
STALE = ("dra", "ddec")

#: Below this the pairs are "equal" -- CSV float round-trip, not divergence.
#: 0.1 mas, against a tree whose tightest real gate is 2 mas.
TOL_ARCSEC = 1e-4

#: Fields where "the pair the pixels agree with" is NOT the pair to keep, because
#: the products themselves are known bad.  Reconciling these would launder a bad
#: correction into the one column that still holds the good value.
#:
#: The rule this script follows -- prefer what the reduction read -- is only as
#: good as the reduction.  Anywhere a correction is known to have been computed
#: against stale inputs, the stale plain column is the SURVIVING good value and
#: copying over it destroys it.
SKIP = {
    "sickle": (
        "issue #270: the +78/-77 mas m2 correction in 'dra (arcsec)' was measured "
        "against crf that never received the VIRAC2 tie, so the AUTHORITATIVE pair "
        "is the wrong one here and the plain pair still holds the clean built "
        "values. Rebuild the table (build_virac2_offsets) instead of reconciling."),
}


def find_tables(field):
    """Every VIRAC2locked/consensus offsets table for a field."""
    return sorted(glob.glob(f"{BASE}/{field}/offsets/Offsets_JWST_*.csv"))


def diverged_rows(tbl):
    """Indices where the two pairs disagree, or None if the table has one pair."""
    if not all(c in tbl.colnames for c in AUTHORITATIVE + STALE):
        return None
    d = np.abs(np.asarray(tbl[AUTHORITATIVE[0]], dtype=float)
               - np.asarray(tbl[STALE[0]], dtype=float))
    c = np.abs(np.asarray(tbl[AUTHORITATIVE[1]], dtype=float)
               - np.asarray(tbl[STALE[1]], dtype=float))
    return np.where((d > TOL_ARCSEC) | (c > TOL_ARCSEC))[0], d, c


def report(path, verbose=True):
    """Describe one table.  Returns the number of diverged rows, or -1."""
    tbl = Table.read(path, format="ascii.csv")
    res = diverged_rows(tbl)
    name = os.path.basename(path)
    if res is None:
        if verbose:
            print(f"  {name}: single column pair, nothing to reconcile")
        return -1
    bad, d, c = res
    if not len(bad):
        if verbose:
            print(f"  {name}: {len(tbl)} rows, pairs agree")
        return 0
    worst = int(np.argmax(np.maximum(d, c)))
    print(f"  {name}: {len(bad)} of {len(tbl)} rows diverged, worst "
          f"{1000 * max(d[worst], c[worst]):.1f} mas "
          f"({tbl['Visit'][worst]} {tbl['Filter'][worst]})")
    if verbose:
        # Show the spread, not just the worst: one 8" outlier and 96 rows at 8"
        # are different situations and the fix is the same only by luck.
        mags = 1000 * np.maximum(d[bad], c[bad])
        print(f"      divergence mas: min {mags.min():.1f}  median "
              f"{np.median(mags):.1f}  max {mags.max():.1f}")
        for i in bad[:3]:
            print(f"      {tbl['Visit'][i]} {tbl['Filter'][i]}: "
                  f"authoritative {float(tbl[AUTHORITATIVE[0]][i]):+.5f},"
                  f"{float(tbl[AUTHORITATIVE[1]][i]):+.5f}  "
                  f"stale {float(tbl[STALE[0]][i]):+.5f},"
                  f"{float(tbl[STALE[1]][i]):+.5f}")
        if len(bad) > 3:
            print(f"      ... and {len(bad) - 3} more")
    return len(bad)


def reconcile(path, apply=False):
    """Copy the authoritative pair onto the stale one.  Returns rows changed."""
    tbl = Table.read(path, format="ascii.csv")
    res = diverged_rows(tbl)
    if res is None:
        return 0
    bad, _, _ = res
    if not len(bad):
        return 0
    if not apply:
        return len(bad)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = f"{path}.pre_colreconcile_{stamp}"
    shutil.copy2(path, backup)

    for auth, stale in zip(AUTHORITATIVE, STALE):
        tbl[stale] = np.asarray(tbl[auth], dtype=float)
    tbl.write(path, format="ascii.csv", overwrite=True)

    # Re-read and verify rather than trusting the write: this edits the file the
    # reducer consumes, and a silent partial write here is the failure class the
    # whole exercise is about.
    check = Table.read(path, format="ascii.csv")
    bad2, _, _ = diverged_rows(check)
    if len(bad2):
        raise SystemExit(
            f"FAILED: {path} still has {len(bad2)} diverged rows after the "
            f"write. The original is at {backup} -- restore it.")
    print(f"    wrote {len(bad)} row(s); backup {os.path.basename(backup)}")
    return len(bad)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--field", help="one field, e.g. cloudef")
    g.add_argument("--all", action="store_true",
                   help="every field with an offsets directory")
    ap.add_argument("--apply", action="store_true",
                    help="write (default is a dry run that changes nothing)")
    ap.add_argument("--force-skipped", action="store_true",
                    help="reconcile a field listed in SKIP anyway. Read its "
                         "reason first -- the entry exists because the pair the "
                         "reduction read is the WRONG one there.")
    args = ap.parse_args(argv)

    if args.all:
        fields = sorted(os.path.basename(os.path.dirname(d))
                        for d in glob.glob(f"{BASE}/*/offsets"))
    else:
        fields = [args.field]

    total = 0
    skipped = 0
    for field in fields:
        tables = find_tables(field)
        if not tables:
            continue
        print(f"{field}:")
        if field in SKIP and not args.force_skipped:
            print(f"  SKIPPED -- {SKIP[field]}")
            skipped += 1
            for path in tables:
                report(path, verbose=False)
            continue
        for path in tables:
            n = report(path)
            if n > 0:
                total += reconcile(path, apply=args.apply)

    note = (f" {skipped} field(s) skipped -- see the reasons above."
            if skipped else "")
    if not args.apply:
        print(f"\nDRY RUN -- {total} row(s) across all tables would be "
              f"rewritten. Re-run with --apply to write.{note}")
    else:
        print(f"\nreconciled {total} row(s).{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
