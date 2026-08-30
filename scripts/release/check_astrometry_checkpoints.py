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

    0  every checkpoint record in scope passed, on the current products
    1  at least one record has ``passed: false`` -- REFUSE
    2  records exist but could not be read -- REFUSE (fail closed)
    3  the frozen stages have no verdict on the CURRENT products -- REFUSE
       (fail closed).  Either there are no records at all, or every frozen
       record predates the field's newest m2 and therefore describes products
       that have since been re-reduced.  Neither is a pass: a field whose
       frozen stages have not run is unverified, not verified.

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
#:
#: The filter group must REFUSE an obs token.  ``m7_crossfilter`` records carry
#: no filter (they are the cross-FILTER check), so
#: ``checkpoint_m7_crossfilter_o050_latest.json`` offered its ``o050`` to the
#: optional filter group, which accepted it as ``[A-Za-z0-9]+`` and left
#: ``obs=None``.  Two things broke, on all nine fields that have a crossfilter
#: record:
#:
#: 1. ``--observations`` stopped scoping.  ``records()`` skips a record only
#:    when ``info["obs"]`` is truthy and out of scope, so an obs-less
#:    crossfilter record survived EVERY scope -- gc2211's failing o050 anchor
#:    refused o023, o028 and o049 as well, none of which it describes.
#: 2. ``_prefer_tokenised`` keys on ``(stage, filt)``, so the tokenised record
#:    (``filt='o050'``) and its untokened predecessor (``filt=None``) landed in
#:    different buckets and the stale one was never superseded.  brick and
#:    quintuplet both carry the pair today; both happen to say ``passed: true``,
#:    so nothing is wrongly refused right now, but a stale ``false`` would
#:    refuse those fields forever.
#:
#: Filters are all ``F...`` (F212N, F1130W, F150W2), so excluding a bare
#: ``o<digits>`` costs no real filter name.
RECORD_RE = re.compile(
    r"^checkpoint_(?P<stage>m\d+(?:_crossfilter)?)"
    r"(?:_(?P<filt>(?!o\d+_latest\.json$)[A-Za-z0-9]+))?"
    r"(?:_o(?P<obs>\d+))?"
    r"_latest\.json$")

#: m2 raises in place, so an m2 record that reached disk with ``passed: false``
#: is a run that was stopped and re-run.  Reading it here would refuse a field
#: on the strength of an iteration that has since been superseded.  The frozen
#: stages are the ones this gate is for.
CORRECTING_STAGES = ("m1", "m2", "m12")


def _stage_index(stage):
    """``m5`` -> 5, ``m7_crossfilter`` -> 7.  ``None`` for an unparseable name."""
    m = re.match(r"^m(\d+)", str(stage or ""))
    return int(m.group(1)) if m else None


def _superseded_by_later_stage(info, rec, found, shipped_stages):
    """The later frozen record that answers this failure, or ``None``.

    The frozen ladder asks ONE question at every stage: is the solution still
    the one m2 froze?  A LATER stage on the SAME chain answering "no movement"
    is a measurement of that same property, on the same exposures, against the
    same m2 baseline -- and it is a measurement made after the failing one.  So
    an excursion that a later stage does not see did not reach the products the
    later stage describes.

    brick F200W is the live case (issue #258).  One chain, 22-24 July:

        m2  2/7 = (+1.90, -0.31)   the freeze
        m5  2/7 = (+4.06, -1.10)   MOVED 2.30 mas   <- FAILED, tol 2.0
        m6  2/7 = (+2.56, -0.42)   moved 0.68 mas   -- passed
        m7  2/7 = (+2.41, -0.34)   moved 0.51 mas   -- passed

    and brick ships the m7 products.  Refusing the field on m5 refuses it for a
    transient in an intermediate nobody receives, while the two later
    measurements of the same property, on the shipped stage, both say the
    solution held.

    ``shipped_stages`` is the guard, and it is what keeps this from being a
    blanket "a later pass forgives an earlier failure": if the RELEASE actually
    ships the failing stage's products, the failure is about something a user
    will download and is never superseded.  Empty/None means the caller did not
    say, and nothing is superseded -- fail closed.
    """
    idx = _stage_index(info["stage"])
    if idx is None or not shipped_stages:
        return None
    if info["stage"] in shipped_stages or f"m{idx}" in shipped_stages:
        return None
    date = str(rec.get("date") or "")
    best = None
    for _p, i2, r2 in found:
        if i2["stage"] in CORRECTING_STAGES or not r2.get("passed"):
            continue
        if (i2["filt"], i2["obs"]) != (info["filt"], info["obs"]):
            continue
        j = _stage_index(i2["stage"])
        if j is None or j <= idx:
            continue
        if str(r2.get("date") or "") <= date:
            continue          # not later in the same chain
        if best is None or j > _stage_index(best[1]["stage"]):
            best = (_p, i2, r2)
    return best


def _newest_correcting(found):
    """Newest correcting-stage record date, by ``(filter, obs)`` and overall.

    m2 rewrites the offsets table and the field is re-reduced from it, so every
    frozen-stage verdict older than the newest m2 was measured on products that
    no longer exist.
    """
    by_key, overall = {}, ""
    for _p, i, r in found:
        if i["stage"] not in CORRECTING_STAGES:
            continue
        date = str(r.get("date") or "")
        overall = max(overall, date)
        for key in ((i["filt"], i["obs"]), (i["filt"], None)):
            by_key[key] = max(by_key.get(key, ""), date)
    return by_key, overall


def _stale_against(info, rec, by_key, overall):
    """The m2 date this frozen record is older than, or ``""`` if it is current.

    Most specific baseline first: the same filter and observation, then the same
    filter in any observation, then the field's newest m2 (which is what an m7
    cross-filter record -- it has no filter of its own -- must use).
    """
    date = str(rec.get("date") or "")
    for key in ((info["filt"], info["obs"]), (info["filt"], None)):
        if key in by_key:
            return by_key[key] if date < by_key[key] else ""
    return overall if (overall and date < overall) else ""


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
    ap.add_argument("--shipped-stages", default="",
                    help="comma-separated stages whose products this release "
                         "actually ships (e.g. m7,m8). A failure at a stage "
                         "that is NOT shipped, and that a LATER stage of the "
                         "same chain measured as passing, describes an "
                         "intermediate nobody receives and is reported as "
                         "superseded rather than refused. Omit to supersede "
                         "nothing (fail closed).")
    ap.add_argument("--scan", action="store_true",
                    help="accepted for symmetry with the sibling gates; this "
                         "check always scans every record in scope")
    args = ap.parse_args(argv)
    # Accept BOTH spellings.  This gate keys on the bare obsid a record name
    # carries (`checkpoint_m3_F212N_o007_latest.json` -> "007"), while
    # `stage_release._release_observations` -- shared with the overlap gate --
    # yields PROPOSAL-obs keys ("02221-002").  Parsing those as bare obsids
    # matched nothing, so every TOKENISED record was silently dropped; with the
    # newest m2 among them, a superseded failure came back as live and refused
    # the field.  Measured on 2026-08-20: cloudc read "1 FAILED" (m3/F182M,
    # 2026-08-06) under the proposal-obs spelling and "0 FAILED, superseded by
    # the 2026-08-12 m2" under the bare one; arches the same with m3/F212N.
    # A false FAILED sends someone hunting an astrometry defect that is not there.
    obs = set()
    for token in args.observations.split(","):
        token = token.strip()
        if not token:
            continue
        obs.add(token.rsplit("-", 1)[-1].lstrip("o"))

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

    shipped_stages = {t.strip() for t in args.shipped_stages.split(",")
                      if t.strip()}
    by_key, overall = _newest_correcting(found)
    failed, stale, current, answered = [], [], [], []
    for path, info, rec in found:
        if info["stage"] in CORRECTING_STAGES:
            continue
        older_than = _stale_against(info, rec, by_key, overall)
        if older_than:
            stale.append((path, info, rec, older_than))
            continue
        current.append((path, info, rec))
        if rec.get("passed"):
            continue
        later = _superseded_by_later_stage(info, rec, found, shipped_stages)
        if later is not None:
            answered.append((path, info, rec, later))
            continue
        failed.append((path, info, rec))

    n_frozen = len(current) + len(stale)
    # Summary FIRST.  It goes to stdout and the verdict goes to stderr, so on a
    # terminal the two interleave by order of printing -- and a verdict that
    # cites counts printed after it reads backwards.
    print(f"{args.field}: {len(found)} checkpoint record(s), {n_frozen} at a "
          f"frozen stage ({len(stale)} superseded), {len(failed)} FAILED"
          + (f", {len(answered)} answered by a later stage" if answered else ""))
    sys.stdout.flush()
    for path, info, rec, (_lp, li, lr) in answered:
        who = "/".join(x for x in (info["stage"], info["filt"],
                                   (f"o{info['obs']}" if info["obs"] else None))
                       if x)
        print(f"ANSWERED BY A LATER STAGE {who}: recorded FAILED "
              f"{rec.get('date')}, but {li['stage']} measured the same property "
              f"on the same exposures at {lr.get('date')} and passed. This "
              f"release does not ship {info['stage']} products "
              f"(shipped: {','.join(sorted(shipped_stages)) or 'none declared'}), "
              f"so the excursion did not reach anything a user receives.")
        for line in (rec.get("failures") or [])[:4]:
            print(f"    was: {line}")
    for path, info, rec, older_than in stale:
        who = "/".join(x for x in (info["stage"], info["filt"],
                                   (f"o{info['obs']}" if info["obs"] else None))
                       if x)
        print(f"SUPERSEDED {who}: recorded {rec.get('date')}, but m2 last ran "
              f"{older_than} -- this verdict is about products that have since "
              f"been re-reduced"
              f"{'' if rec.get('passed') else ' (it says FAILED, and that is not a statement about what is on disk now)'}")
    for path, info, rec in failed:
        who = "/".join(x for x in (info["stage"], info["filt"],
                                   (f"o{info['obs']}" if info["obs"] else None))
                       if x)
        print(f"\nFAILED {who}  ({os.path.basename(path)}, {rec.get('date')})")
        for line in (rec.get("failures") or []):
            print(f"    {line}")
        for line in (rec.get("unverified_blocking") or []):
            print(f"    [measured and refused] {line}")
        # Whether the run STOPPED here or walked past it.  Both leave the same
        # `passed: false`, and until the record carried this the difference
        # existed only in a SLURM log: issue #258 is a red m5 record whose run
        # continued, with nothing on disk saying why.  `None` means the record
        # predates the field, which is not the same as "not overridden".
        ov = rec.get("gate_override")
        if ov is None:
            print("    [override] not recorded -- this record predates the "
                  "field, so whether the run stopped here is not knowable "
                  "from it")
        elif ov.get("used"):
            reason = (ov.get("reason") or "").strip()
            print(f"    [override] {ov.get('env')}=1 was set: the run "
                  f"CONTINUED past this failure")
            print(f"    [override] justification: {reason}" if reason else
                  f"    [override] NO JUSTIFICATION RECORDED -- CLAUDE.md "
                  f"requires one; set {ov.get('reason_env')} on the run")
    if not failed and not current:
        # rc 3 covers two states and the remedy is the same for both, but they
        # are different situations for whoever picks the field up, so they get
        # different sentences.  Saying "all 0 frozen record(s) predate the
        # field's newest m2" to a field that has never run a frozen stage --
        # sgrb2, 33 m2 records and no m3 -- states two things that are not true
        # of it.
        if stale:
            print(f"\nREFUSING TO STAGE '{args.field}': the frozen stages have "
                  f"no verdict on the CURRENT products -- all {len(stale)} "
                  f"frozen record(s) predate the field's newest m2, so they "
                  f"describe products that have since been re-reduced.  That is "
                  f"not a pass and it is not a failure either: re-run m3..m7 so "
                  f"the checkpoint judges what is actually on disk.",
                  file=sys.stderr)
        else:
            print(f"\nREFUSING TO STAGE '{args.field}': the frozen stages have "
                  f"NEVER RUN for this field/observation set -- "
                  f"{len(found)} checkpoint record(s) on disk and not one of "
                  f"them is m3 or later.  A field whose frozen stages have not "
                  f"run is unverified, not verified: run the cataloging chain "
                  f"past m2.", file=sys.stderr)
        return 3
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
