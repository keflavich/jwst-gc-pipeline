#!/usr/bin/env python
"""How many m2 records would the split rule BLOCK, counted by call site.

The blast-radius figure for #312/#341 was first computed by grepping the
message text, and it came out 21 -- wrong, because the one call site that
appends to ``unverified_blocking`` has had three spellings:

    "the tie is not trustworthy"       (current)
    "VIRAC tie is not trustworthy"     (a6d4782)
    "independent checks DISAGREE"      (12e254b)

A grep for any one of them undercounts, and which fields appear depends on
which spelling you picked -- sgra drops out entirely under the current one.
The true figure is the union, and the only way to keep it true is to stop
counting text.

Records written from #341 onward carry ``unverified_blocking`` as its own list,
so for those the count is exact and needs no pattern at all.  Historical records
predate the field, so this falls back to the union of the three spellings and
SAYS SO in the output rather than quietly mixing the two.

Usage::

    python count_blocking_unverified.py                 # latest record per filter
    python count_blocking_unverified.py --all-history   # every record on disk
"""
import argparse
import collections
import glob
import json
import os
import sys

BASE = os.environ.get("GC_BASEPATH_OVERRIDE",
                      os.environ.get("JWST_BASE", "/orange/adamginsburg/jwst"))

#: Every spelling the blocking site has used.  Only needed for records written
#: before ``unverified_blocking`` was persisted; do not add to this list to
#: describe a NEW condition -- add a call site and let the field carry it.
LEGACY_BLOCKING_PATTERNS = (
    "the tie is not trustworthy",
    "VIRAC tie is not trustworthy",
    "independent checks DISAGREE",
)


def blocking_entries(record):
    """(entries, exact).  ``exact`` is False when the count had to use text."""
    ub = record.get("unverified_blocking")
    if ub is not None:
        return list(ub), True
    out = []
    for u in (record.get("unverified") or []):
        s = str(u)
        if any(p in s for p in LEGACY_BLOCKING_PATTERNS):
            out.append(s)
    return out, False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all-history", action="store_true",
                    help="every checkpoint record, not just the latest per filter")
    args = ap.parse_args(argv)

    pattern = (f"{BASE}/*/astrometry_checkpoints/checkpoint_m2_*.json"
               if args.all_history
               else f"{BASE}/*/astrometry_checkpoints/checkpoint_m2_*_latest.json")
    byfield = collections.Counter()
    n_exact = n_text = 0
    for path in sorted(glob.glob(pattern)):
        if args.all_history and path.endswith("_latest.json"):
            continue
        field = path.split("/")[-3]
        try:
            with open(path) as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            continue
        entries, exact = blocking_entries(rec)
        if not entries:
            continue
        n_exact += exact
        n_text += (not exact)
        byfield[field] += len(entries)

    scope = "all history" if args.all_history else "latest per filter"
    total = sum(byfield.values())
    print(f"blocking-unverified entries ({scope}): {total}")
    for f, c in byfield.most_common():
        print(f"   {f:<12} {c}")
    print(f"\n   {n_exact} record(s) counted EXACTLY (unverified_blocking present)")
    print(f"   {n_text} record(s) counted from message text (pre-#341 records; "
          f"union of {len(LEGACY_BLOCKING_PATTERNS)} spellings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
