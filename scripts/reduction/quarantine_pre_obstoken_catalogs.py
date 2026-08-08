#!/usr/bin/env python
"""Rename pre-obs-token per-frame catalogs out of the consensus glob.

A multi-observation proposal writes one per-frame catalog per (observation,
visit, vgroup, exposure, detector).  Before the obs token existed those names
carried no observation:

    f150w_nrca1_visit001_vgroup02201_exp00001_m2_daophot_basic.fits       2026-06-22
    f150w_nrca1_o028_visit001_vgroup02201_exp00001_m2_daophot_basic.fits  2026-08-08

Both describe the SAME (visit, vgroup, exposure, detector, filter) identity, so
the m2 visit-consensus ingests each one twice and refuses to build:

    error_kind: duplicate_exposure
    192 exposure identity/ies ingested more than once

Every gc2211 observation hit this.  The consensus then contained ZERO exposures,
the m12 finalize still exited 0, the retie loop printed "m2 checkpoint PASSED --
converged", and the frozen-stage m3 check failed for want of a baseline that was
never written (#350).  So the field's astrometry gate was skipped silently.

The m2 CORRECTION path already excludes untokened files -- o023's log reads
``excluded 560 of 592 foreign-observation per-frame catalog(s) (['<untokened>',
'o046', 'o049', 'o050'])`` -- so the filter exists; the consensus build does not
use it.  Fixing that is #350 and is the durable half.  This script is the data
half: get the superseded files out of the glob so the current generation can be
measured.

## Safety

Renames, never deletes: ``<name>_pre_obstoken_stale``, the same shape
``rename_stale_mosaics.py`` uses.  Reversible with a rename back.

Refuses to touch a file unless BOTH hold:

  * it carries no observation token, AND
  * a tokened file exists for the same identity

so a field that simply predates the token (and has no replacement) is left
alone.  The second condition is what makes this safe to run anywhere rather than
only where someone has already checked the dates by hand.

Usage::

    python quarantine_pre_obstoken_catalogs.py --field gc2211            # dry run
    python quarantine_pre_obstoken_catalogs.py --field gc2211 --execute
"""
import argparse
import collections
import os
import re
import sys

BASE = os.environ.get("GC_BASEPATH_OVERRIDE",
                      os.environ.get("JWST_BASE", "/orange/adamginsburg/jwst"))

SUFFIX = "_pre_obstoken_stale"

#: <filter>_<detector>[_o<obs>]_visit<NNN>_vgroup<...>_exp<NNNNN>_<rest>
_NAME = re.compile(
    r"^(?P<filt>f\d{3}[a-z]\d?)_"
    r"(?P<det>nrc[ab](?:\d|long)|mirim|nis)_"
    r"(?:o(?P<obs>\d{3})_)?"
    r"(?P<rest>visit\d+_.*\.fits)$",
    re.IGNORECASE)


def identity(name):
    """``(filter, detector, rest)`` -- the same frame regardless of obs token."""
    m = _NAME.match(name)
    if not m:
        return None
    return (m.group("filt").lower(), m.group("det").lower(), m.group("rest"))


def has_obs_token(name):
    m = _NAME.match(name)
    return bool(m and m.group("obs"))


def plan_for_dir(directory):
    """Untokened files in ``directory`` that a tokened file supersedes."""
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    tokened = collections.defaultdict(list)
    untokened = {}
    for n in names:
        if n.endswith(SUFFIX) or not n.endswith(".fits"):
            continue
        ident = identity(n)
        if ident is None:
            continue
        if has_obs_token(n):
            tokened[ident].append(n)
        else:
            untokened[ident] = n
    return [(untokened[i], sorted(tokened[i]))
            for i in sorted(untokened) if i in tokened]


def field_dirs(field):
    root = os.path.join(BASE, field)
    if not os.path.isdir(root):
        return []
    return sorted(os.path.join(root, d) for d in os.listdir(root)
                  if re.fullmatch(r"[Ff]\d{3}[A-Za-z]\d?", d)
                  and os.path.isdir(os.path.join(root, d)))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", required=True, help="field dir, e.g. gc2211")
    ap.add_argument("--execute", action="store_true",
                    help="rename (default is a dry run that changes nothing)")
    args = ap.parse_args(argv)

    dirs = field_dirs(args.field)
    if not dirs:
        print(f"no per-filter directories under {BASE}/{args.field}",
              file=sys.stderr)
        return 2

    total = 0
    for d in dirs:
        plan = plan_for_dir(d)
        if not plan:
            print(f"{os.path.basename(d)}: nothing superseded")
            continue
        print(f"{os.path.basename(d)}: {len(plan)} untokened file(s) superseded "
              f"by a tokened one")
        for old, replacements in plan[:2]:
            print(f"    {old}")
            print(f"      superseded by: {', '.join(replacements[:3])}"
                  + (" ..." if len(replacements) > 3 else ""))
        if len(plan) > 2:
            print(f"    ... and {len(plan) - 2} more")
        total += len(plan)
        if args.execute:
            for old, _ in plan:
                src = os.path.join(d, old)
                os.rename(src, src + SUFFIX)

    if args.execute:
        print(f"\nrenamed {total} file(s) -> *{SUFFIX}")
        print("Reverse with: for f in *"+SUFFIX+"; do mv \"$f\" \"${f%"
              + SUFFIX + "}\"; done")
    else:
        print(f"\ndry run: {total} file(s) would be renamed; "
              f"re-run with --execute")
    return 0


if __name__ == "__main__":
    sys.exit(main())
