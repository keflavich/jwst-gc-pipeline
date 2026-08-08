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

The default scope is genuinely latest-per-FILTER, which the glob alone does not
give: ``_latest`` records are tokened now (``..._F162M_o012_latest.json``) and
the untokened predecessor is NOT removed when the tokened one appears, so one
filter can have two live ``_latest`` files a day apart describing the same item
(sgrc F162M: 5.70 mas and 5.72 mas).  Counting both double-counts, which is the
same class of error as counting the message text.
"""
import argparse
import collections
import glob
import json
import os
import re
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


def _tally(record, path, byfield):
    entries, _ = blocking_entries(record)
    if entries:
        byfield[path.split(os.sep)[-3]] += len(entries)


def _all_records(pattern, args):
    for path in sorted(glob.glob(pattern)):
        if args.all_history and path.endswith("_latest.json"):
            continue
        try:
            with open(path) as fh:
                yield json.load(fh)
        except (OSError, ValueError):
            continue


def record_filter_key(path, record):
    """``(field, filter, token)`` for latest-per-filter deduplication.

    The token DOES come from the filename, and must.  Keying on
    ``(field, filter)`` alone -- taking everything from the record -- collapses
    a multi-observation field's records into one, because the observation is
    the only thing that distinguishes them and the record does not carry it:

        checkpoint_m2_F200W_latest.json       entries=0   gc2211 F200W/nrcb
        checkpoint_m2_F200W_o023_latest.json  entries=1   gc2211 F200W/nrca
        checkpoint_m2_F200W_o049_latest.json  entries=0   gc2211 F200W/nrca
        checkpoint_m2_F200W_o050_latest.json  entries=0   gc2211 F200W/nrcb

    All four key to ('gc2211', 'F200W'); `_o050` is newest, so o023's blocking
    entry -- the record #345 was opened about -- was discarded and the count
    read 0 with a blocking record sitting in the directory.
    """
    field = path.split(os.sep)[-3]
    filt = str(record.get("filtername") or "")
    m = re.search(r"_o(\d{3})_latest\.json$", os.path.basename(path))
    return field, filt, (m.group(1) if m else None)


def drop_superseded_untokened(keys):
    """Remove the untokened key for a (field, filter) that has a tokened one.

    The untokened `_latest` is the LEGACY record, written before the token
    existed; where a tokened sibling covers the same filter it is a duplicate,
    not another pointing.  sgrc has one observation, so its untokened
    `F162M_latest` really is a copy of `_o012_latest` and drops out; gc2211's
    untokened `F200W_latest` drops too, while `_o023`/`_o049`/`_o050` each
    stand on their own.

    Counting the live tree three ways:

        (field, filter)                        12   collapses observations
        (field, filter, token)                 14   sgrc F162M counted twice
        (field, filter, token) + this          13   correct
    """
    tokened = {(f, fl) for f, fl, tok in keys if tok is not None}
    return {k for k in keys if k[2] is not None or (k[0], k[1]) not in tokened}


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
    n_scanned = n_unreadable = 0
    n_exact = n_text = 0
    newest = {}                     # (field, filter) -> (mtime, path, record)
    for path in sorted(glob.glob(pattern)):
        if args.all_history and path.endswith("_latest.json"):
            continue
        try:
            with open(path) as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            n_unreadable += 1
            continue
        n_scanned += 1
        if args.all_history:
            records = [(path, rec)]
        else:
            # latest per FILTER, not per record NAME: keep the newest file for
            # each (field, filter) so a tokened and an untokened `_latest` for
            # one filter count once.
            key = record_filter_key(path, rec)
            mtime = os.path.getmtime(path)
            if key not in newest or mtime > newest[key][0]:
                newest[key] = (mtime, path, rec)
            records = []
        for pth, r in records:
            _tally(r, pth, byfield)

    if not args.all_history:
        keep = drop_superseded_untokened(set(newest))
        for k in sorted(keep, key=lambda x: (x[0], x[1], x[2] or "")):
            _, pth, r = newest[k]
            _tally(r, pth, byfield)
        counted = [newest[k][2] for k in keep]
    else:
        counted = None

    # Count the METHOD for every record considered, not only the ones that
    # contributed an entry.  Otherwise "0 counted EXACTLY" cannot tell "no
    # record has been written since #341" from "records have been written and
    # none of them blocked" -- which is the question the line exists to answer.
    for r in (counted if counted is not None else _all_records(pattern, args)):
        if r.get("unverified_blocking") is not None:
            n_exact += 1
        else:
            n_text += 1

    scope = "all history" if args.all_history else "latest per filter"
    total = sum(byfield.values())
    n_considered = n_scanned if args.all_history else len(counted)
    # The denominator, ALWAYS.  Without it a mistyped base, an unmounted
    # /orange or a glob that stopped matching prints byte-identical output to a
    # genuinely clean tree -- for a tool whose job is to make a number
    # trustworthy, "0 of 0" and "0 of 233" must not look the same.
    print(f"scanned {n_scanned} record(s) under {BASE}"
          + (f" ({n_unreadable} unreadable)" if n_unreadable else ""))
    print(f"considered {n_considered} record(s) after latest-per-filter "
          f"deduplication" if not args.all_history else
          f"considered {n_considered} record(s)")
    if not n_considered:
        print("\nNOTHING SCANNED -- check the base path; this is not a clean bill "
              "of health.")
        return 1
    print(f"blocking-unverified entries ({scope}): {total}")
    for f, c in byfield.most_common():
        print(f"   {f:<12} {c}")
    print(f"\n   {n_exact} record(s) counted EXACTLY (unverified_blocking present)")
    print(f"   {n_text} record(s) counted from message text (pre-#341 records; "
          f"union of {len(LEGACY_BLOCKING_PATTERNS)} spellings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
