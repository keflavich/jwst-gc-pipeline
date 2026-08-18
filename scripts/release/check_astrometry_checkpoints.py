#!/usr/bin/env python
"""Release gate: no field ships carrying a FAILED astrometry checkpoint.

The frozen stages (m3 and later) and the m7 cross-filter check default to
recording a failure rather than raising inside the stage
(``ASTROM_CHECKPOINT_ENFORCE=release``, see
``photometry/astrometry_checkpoint.py``).  That default is only defensible
because this gate exists: the stop moves from the middle of an ``afterok``
chain to the front of the release, and nothing is waived on the way.

Why the stop moved.  m3 and later cannot CHANGE the astrometry -- the solution
is frozen, which is what makes a shift there a defect -- so the check is a
measurement wired up as a control.  Raising inside the stage discarded every
other filter's finished work (the chain is ``afterok``), and it destroyed the
products an investigator needs, so every diagnosis began by re-running the
chain to get them back.  m2 is different and is untouched: it is the one stage
that can still correct, its stop sends the field back for regeneration, and
nothing downstream can do that for it.

Exit codes, matching the sibling gates in this directory:

    0  every checkpoint record in scope passed
    1  at least one record has ``passed: false`` -- REFUSE
    2  records exist but could not be read -- REFUSE (fail closed)
    3  no checkpoint records at all for this field -- REFUSE (fail closed);
       a field that never ran the checkpoint is unverified, not verified

Usage::

    python scripts/release/check_astrometry_checkpoints.py --field brick
    python scripts/release/check_astrometry_checkpoints.py --field brick \\
        --observations 001,004
"""
import argparse
import glob
import json
import os
import re
import sys

BASE = os.environ.get("GC_BASEPATH_OVERRIDE",
                      os.environ.get("JWST_BASE", "/orange/adamginsburg/jwst"))

#: ``checkpoint_m3_F212N_o001_latest.json`` -> stage m3, filter F212N, obs 001.
RECORD_RE = re.compile(
    r"^checkpoint_(?P<stage>m\d+(?:_crossfilter)?)"
    r"(?:_(?P<filt>[A-Za-z0-9]+))?"
    r"(?:_o(?P<obs>\d+))?"
    r"_latest\.json$")

#: m2 raises in place, so an m2 record that reached disk with ``passed: false``
#: is a run that was stopped and re-run.  Reading it here would refuse a field
#: on the strength of an iteration that has since been superseded.  The frozen
#: stages are the ones this gate is for.
CORRECTING_STAGES = ("m1", "m2", "m12")


def records(field, observations=None):
    """``[(path, parsed-name, record)]`` for the field's ``*_latest`` records."""
    d = os.path.join(BASE, field, "astrometry_checkpoints")
    out, unreadable = [], []
    for path in sorted(glob.glob(os.path.join(d, "checkpoint_*_latest.json"))):
        m = RECORD_RE.match(os.path.basename(path))
        if not m:
            continue
        info = m.groupdict()
        if observations and info["obs"] and info["obs"] not in observations:
            continue
        try:
            with open(path) as fh:
                out.append((path, info, json.load(fh)))
        except (OSError, ValueError) as ex:
            unreadable.append((path, f"{type(ex).__name__}: {ex}"))
    return _prefer_tokenised(out), unreadable


def _prefer_tokenised(found):
    """Drop an UNTOKENISED record when a tokenised sibling exists.

    Obs tokens were added partway through, so a field carries both
    ``checkpoint_m3_F187N_latest.json`` (written before, and never overwritten
    again once the tokenised name took over) and
    ``checkpoint_m3_F187N_o007_latest.json``.  Reading both means a superseded
    record from an earlier campaign refuses the field forever, no matter what
    the current run measured -- sickle's untokened m3/F187N is from 2026-08-05
    and its tokenised one from 2026-08-17.

    Keyed on (stage, filter): the untokened file cannot say which observation it
    belonged to, so the only safe reading is that a tokenised record for the same
    stage and filter supersedes it.
    """
    tokenised = {(i["stage"], i["filt"]) for _p, i, _r in found if i["obs"]}
    return [(p, i, r) for p, i, r in found
            if i["obs"] or (i["stage"], i["filt"]) not in tokenised]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", required=True)
    ap.add_argument("--observations", default="",
                    help="comma-separated obsids to scope to (e.g. 001,004)")
    ap.add_argument("--scan", action="store_true",
                    help="accepted for symmetry with the sibling gates; this "
                         "check always scans every record in scope")
    args = ap.parse_args(argv)
    obs = {o.strip().lstrip("o") for o in args.observations.split(",") if o.strip()}

    found, unreadable = records(args.field, obs or None)
    if unreadable:
        for path, why in unreadable:
            print(f"UNREADABLE {os.path.basename(path)}: {why}", file=sys.stderr)
        print(f"\nREFUSING: {len(unreadable)} astrometry checkpoint record(s) for "
              f"'{args.field}' could not be read, so the field's frozen-stage "
              f"astrometry is unconfirmed.", file=sys.stderr)
        return 2
    if not found:
        print(f"REFUSING: no astrometry checkpoint records for '{args.field}' "
              f"under {os.path.join(BASE, args.field, 'astrometry_checkpoints')}"
              f"{' (observations ' + ','.join(sorted(obs)) + ')' if obs else ''}."
              f"\nA field that never ran the checkpoint is UNVERIFIED, not "
              f"verified.  Run the cataloging chain, or scope --observations to "
              f"the ones this release ships.", file=sys.stderr)
        return 3

    failed = []
    for path, info, rec in found:
        if info["stage"] in CORRECTING_STAGES:
            continue
        if rec.get("passed"):
            continue
        failed.append((path, info, rec))

    n_frozen = sum(1 for _p, i, _r in found
                   if i["stage"] not in CORRECTING_STAGES)
    print(f"{args.field}: {len(found)} checkpoint record(s), {n_frozen} at a "
          f"frozen stage, {len(failed)} FAILED")
    for path, info, rec in failed:
        who = "/".join(x for x in (info["stage"], info["filt"],
                                   (f"o{info['obs']}" if info["obs"] else None))
                       if x)
        print(f"\nFAILED {who}  ({os.path.basename(path)}, {rec.get('date')})")
        for line in (rec.get("failures") or []):
            print(f"    {line}")
        for line in (rec.get("unverified_blocking") or []):
            print(f"    [measured and refused] {line}")
    if failed:
        print(f"\nREFUSING TO STAGE '{args.field}': {len(failed)} frozen-stage "
              f"astrometry checkpoint(s) FAILED.  These were recorded rather "
              f"than raised inside the chain (ASTROM_CHECKPOINT_ENFORCE="
              f"release, the default), which is why the products exist -- use "
              f"them to diagnose.  The field does not ship until the "
              f"checkpoint passes.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
