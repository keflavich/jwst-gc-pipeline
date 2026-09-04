#!/usr/bin/env python
"""Release gate: no field ships carrying a FAILED astrometry checkpoint.

The frozen stages (m3 and later) and the m7 cross-filter check default to
recording a failure rather than raising inside the stage
(``ASTROM_CHECKPOINT_ENFORCE=release``, see
``photometry/astrometry_checkpoint.py``).  That default is only defensible
because this gate exists: the stop moves from the middle of an ``afterok``
chain to the front of the release, and nothing is waived on the way.

It also refuses a field whose m2 MEASURED per-exposure corrections that nothing
applied.  m2 is the one stage that can still change the astrometry, and its
verdict is a demand: rewrite the offsets table, regenerate the frames, re-run.
A run that walks past that demand -- ``ASTROM_CHECKPOINT_WARN_ONLY=1``, or a
restart that resumes at m3 -- leaves every later stage measuring the solution
against the m2 freeze and finding it unmoved, which is true and beside the
point, because the freeze is the misaligned thing.  Reading only the frozen
records, this gate called that field clean.

Why the stop moved.  m3 and later cannot CHANGE the astrometry -- the solution
is frozen, which is what makes a shift there a defect -- so the check is a
measurement wired up as a control.  Raising inside the stage discarded every
other filter's finished work (the chain is ``afterok``), and it destroyed the
products an investigator needs, so every diagnosis began by re-running the
chain to get them back.  m2 is different: it is the one stage that can still
correct, its stop sends the field back for regeneration, and nothing downstream
can do that for it -- which is why this gate checks that the stop was obeyed
rather than grading m2's own verdict.  What is read from an m2 record here is
what it MEASURED, never its ``passed`` flag: an m2 that reached disk saying
false is an iteration that was stopped and re-run, and refusing on it would
block a field for a pass that has since been superseded.

Exit codes, matching the sibling gates in this directory:

    0  every checkpoint record in scope passed, on the current products
    1  at least one record has ``passed: false``, or a correcting-stage record
       holds corrections nothing applied -- REFUSE
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
import math
import os
import re
import sys

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    REFERENCE_TIE_SOURCE_SUFFIX, parse_record_name)

BASE = os.environ.get("GC_BASEPATH_OVERRIDE",
                      os.environ.get("JWST_BASE", "/orange/adamginsburg/jwst"))

#: m2 raises in place, so an m2 record that reached disk with ``passed: false``
#: is a run that was stopped and re-run.  Grading its VERDICT here would refuse
#: a field on the strength of an iteration that has since been superseded, so
#: these stages stay out of the pass/fail loop below; what is read from them is
#: their date (``_newest_correcting``, which supersedes older frozen records)
#: and their ``corrections`` (``_unapplied_correcting``, which refuses a field
#: whose chain went on without applying them).
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


def _newest_correcting_records(found):
    """Newest correcting-stage RECORD, by ``(filter, obs)`` and overall.

    Same keys as ``_newest_correcting``, which is written in terms of this one;
    the difference is that the record comes back rather than just its date, so a
    caller can also read what m2 MEASURED and not only when it ran.
    """
    def _date(entry):
        return str(entry[2].get("date") or "")

    by_key, overall = {}, None
    for entry in found:
        if entry[1]["stage"] not in CORRECTING_STAGES:
            continue
        if overall is None or _date(entry) > _date(overall):
            overall = entry
        for key in ((entry[1]["filt"], entry[1]["obs"]),
                    (entry[1]["filt"], None)):
            if key not in by_key or _date(entry) > _date(by_key[key]):
                by_key[key] = entry
    return by_key, overall


def _newest_correcting(found):
    """Newest correcting-stage record date, by ``(filter, obs)`` and overall.

    m2 rewrites the offsets table and the field is re-reduced from it, so every
    frozen-stage verdict older than the newest m2 was measured on products that
    no longer exist.
    """
    by_key, overall = _newest_correcting_records(found)
    return ({k: str(v[2].get("date") or "") for k, v in by_key.items()},
            str(overall[2].get("date") or "") if overall is not None else "")


def _governing_correcting(info, by_key, overall):
    """The correcting-stage record whose date makes THIS frozen record current.

    The same baseline ladder as ``_stale_against``, and deliberately so: a
    frozen record is graded current precisely because it is newer than this m2,
    so this m2 is the one whose verdict it is standing on.
    """
    for key in ((info["filt"], info["obs"]), (info["filt"], None)):
        if key in by_key:
            return by_key[key]
    return overall


def _as_float(value):
    """``value`` as a finite float, or ``None`` for anything that is not one."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _magnitude(corr):
    """On-sky size of a correction in mas, or ``None`` if it cannot be read."""
    dra, ddec = (_as_float(corr.get("dra_onsky_mas")),
                 _as_float(corr.get("ddec_onsky_mas")))
    return None if dra is None or ddec is None else math.hypot(dra, ddec)


def _actionable_corrections(rec):
    """The corrections in an m2 record that the run was obliged to APPLY.

    m2 writes its record BEFORE the correction floor is applied
    (``cataloging._run_astrometry_stage_checkpoint``), so a non-empty
    ``corrections`` list is not by itself a demand: a field with a standing
    floor (cloudc and sgrc are at 8.0 mas, `m2_correction_floors`) passes and
    goes on when every residual is below it, and reading the raw list would
    refuse those fields for the state they are supposed to be in.  So apply the
    same filter the caller applied -- ``hypot`` of the on-sky terms against the
    floor the record itself carries, with the whole-consensus tie to the
    reference catalog exempt from the floor exactly as
    ``cataloging._is_whole_consensus_shift`` has it.

    Unreadable is COUNTED, where the caller raises: a correction with no finite
    on-sky magnitude cannot be shown to be sub-floor, and a gate that drops what
    it cannot read reports a clean field on the strength of a broken record.
    A record with no ``tolerances`` block predates the floor being written down,
    and 0.0 -- every correction actionable -- is the reading that does not
    invent a floor the run may never have had.
    """
    floor = _as_float((rec.get("tolerances") or {}).get("correction_floor_mas"))
    floor = 0.0 if floor is None else floor
    out = []
    for corr in rec.get("corrections") or []:
        mag = _magnitude(corr)
        if (REFERENCE_TIE_SOURCE_SUFFIX in str(corr.get("source", ""))
                or mag is None or mag >= floor):
            out.append(corr)
    return out


def _unapplied_correcting(current, by_key, overall):
    """``[(path, info, rec, corrections, witnesses)]`` -- m2 asked, nobody applied.

    m2 measuring a correction means the frames it just cataloged are misaligned
    by that much: the offsets table is rewritten, the im0 mosaics are
    stale-tagged, and the run is supposed to STOP so the frames can be
    regenerated from the corrected table.  When it stops, the frozen stages do
    not run again until after the regenerated frames have been through m2 once
    more -- and that second m2 record, the one this gate reads, carries no
    corrections.

    So a newest-m2 record that still holds actionable corrections, with frozen
    records for the same key recorded AFTER it, is the state where the chain
    went on without the regeneration.  Those frozen stages then measure the
    solution against the m2 freeze and find it unmoved, which is true and beside
    the point: the freeze itself is the misaligned one.  ``witnesses`` are the
    frozen records that ran anyway, which is what makes this different from the
    ordinary stopped-and-waiting field (rc 3).
    """
    witnesses, governing = {}, {}
    for entry in current:
        gov = _governing_correcting(entry[1], by_key, overall)
        if gov is None or not _actionable_corrections(gov[2]):
            continue
        governing[gov[0]] = gov
        witnesses.setdefault(gov[0], []).append(entry)
    return [(gov[0], gov[1], gov[2], _actionable_corrections(gov[2]),
             witnesses[path])
            for path, gov in sorted(governing.items())]


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


def _who(info):
    """``m5/F200W/o001`` -- the record's identity, as the report spells it."""
    return "/".join(x for x in (info["stage"], info["filt"],
                                (f"o{info['obs']}" if info["obs"] else None))
                    if x)


def _print_justification(ov):
    """The operator's written reason, or the fact that there is none."""
    reason = (ov.get("reason") or "").strip()
    if reason:
        print(f"    [override] justification: {reason}")
    else:
        print(f"    [override] NO JUSTIFICATION RECORDED -- CLAUDE.md "
              f"requires one; set {ov.get('reason_env')} on the run")


def _used_override(rec):
    """The record's ``gate_override`` block if a gate was demoted, else None."""
    ov = rec.get("gate_override")
    return ov if isinstance(ov, dict) and ov.get("used") else None


def _obs_in_scope(obs, observations):
    """Is a record's observation token inside the release's scope?

    A joint registration writes both observations into one token
    (``_o002-998``); it describes each of them, so it is in scope when the
    release ships EITHER.  Requiring the whole token to appear in
    ``--observations`` would silently drop the only record covering an
    observation that IS being shipped.
    """
    return any(part in observations for part in str(obs).split("-"))


def records(field, observations=None):
    """``[(path, parsed-name, record)]`` for the field's ``*_latest`` records."""
    d = os.path.join(BASE, field, "astrometry_checkpoints")
    out, unreadable = [], []
    for path in sorted(glob.glob(os.path.join(d, "checkpoint_*_latest.json"))):
        info = parse_record_name(os.path.basename(path))
        if info is None:
            continue
        if observations and info["obs"] and not _obs_in_scope(info["obs"],
                                                              observations):
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

    ngc6334's shared-tree token (``_j7213``) counts as tokenised for the same
    reason an obs token does -- it names which run wrote the record -- so an
    untokened sibling of the same stage and filter is superseded by it.
    """
    def _tokenised(i):
        return bool(i["obs"] or i["proposal"])

    tokenised = {(i["stage"], i["filt"]) for _p, i, _r in found if _tokenised(i)}
    return [(p, i, r) for p, i, r in found
            if _tokenised(i) or (i["stage"], i["filt"]) not in tokenised]


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
    correcting_by_key, correcting_overall = _newest_correcting_records(found)
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

    unapplied = _unapplied_correcting(current, correcting_by_key,
                                      correcting_overall)

    n_frozen = len(current) + len(stale)
    # Summary FIRST.  It goes to stdout and the verdict goes to stderr, so on a
    # terminal the two interleave by order of printing -- and a verdict that
    # cites counts printed after it reads backwards.
    print(f"{args.field}: {len(found)} checkpoint record(s), {n_frozen} at a "
          f"frozen stage ({len(stale)} superseded), {len(failed)} FAILED"
          + (f", {len(answered)} answered by a later stage" if answered else "")
          + (f", {len(unapplied)} m2 record(s) with corrections nothing applied"
             if unapplied else ""))
    sys.stdout.flush()
    for path, info, rec, (_lp, li, lr) in answered:
        who = _who(info)
        print(f"ANSWERED BY A LATER STAGE {who}: recorded FAILED "
              f"{rec.get('date')}, but {li['stage']} measured the same property "
              f"on the same exposures at {lr.get('date')} and passed. This "
              f"release does not ship {info['stage']} products "
              f"(shipped: {','.join(sorted(shipped_stages)) or 'none declared'}), "
              f"so the excursion did not reach anything a user receives.")
        for line in (rec.get("failures") or [])[:4]:
            print(f"    was: {line}")
    for path, info, rec, older_than in stale:
        who = _who(info)
        print(f"SUPERSEDED {who}: recorded {rec.get('date')}, but m2 last ran "
              f"{older_than} -- this verdict is about products that have since "
              f"been re-reduced"
              f"{'' if rec.get('passed') else ' (it says FAILED, and that is not a statement about what is on disk now)'}")
    for path, info, rec in failed:
        who = _who(info)
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
            print(f"    [override] {ov.get('env')}=1 was set: the run "
                  f"CONTINUED past this failure")
            _print_justification(ov)
    # An overridden PASS.  The block above is inside the `failed` loop, and
    # the broadest override never produces a record that reaches it:
    # ASTROM_CHECKPOINT_WARN_ONLY demotes the stage's raise, so the record is
    # written `passed: true`, and a correcting stage is skipped before it
    # enters any list at all.  arches ran its m2 repair pass that way and this
    # gate printed "0 FAILED", exit 0, with the operator's justification alive
    # only in a SLURM log (#581).  Report a demoted gate wherever it sits and
    # whatever its verdict; a superseded record is left to the SUPERSEDED line
    # above, which already says it describes products that no longer exist.
    #
    # Reporting only.  Whether an overridden pass should also REFUSE the field
    # is a policy question this gate does not answer, and the exit codes are
    # unchanged by this block.
    _already_said = {p for p, _i, _r in failed}
    _superseded = {p for p, _i, _r, _o in stale}
    for path, info, rec in found:
        if path in _already_said or path in _superseded:
            continue
        ov = _used_override(rec)
        if ov is None:
            continue
        print(f"\nOVERRIDDEN {_who(info)}  ({os.path.basename(path)}, "
              f"{rec.get('date')}): {ov.get('env')}=1 was set and the record "
              f"reads passed={bool(rec.get('passed'))} -- a blocking check was "
              f"DEMOTED on this run, so the verdict is not an unassisted one.")
        _print_justification(ov)
    for path, info, rec, corrections, witnesses in unapplied:
        ran_on = ", ".join(sorted({_who(i) for _p, i, _r in witnesses}))
        print(f"\nCORRECTIONS NEVER APPLIED {_who(info)}  "
              f"({os.path.basename(path)}, {rec.get('date')})")
        print(f"    {len(corrections)} correction(s) at or above the "
              f"{(rec.get('tolerances') or {}).get('correction_floor_mas')} mas "
              f"floor, and this is the NEWEST {info['stage']} for this key -- "
              f"so no post-regeneration {info['stage']} has run since.")
        print(f"    {len(witnesses)} frozen record(s) were written AFTER it "
              f"({ran_on}), so the chain went on rather than stopping for the "
              f"regeneration {info['stage']} asked for.")
        for corr in sorted(corrections,
                           key=lambda c: -(_magnitude(c) or math.inf))[:4]:
            print(f"    largest: visit {corr.get('visit')} exp "
                  f"{corr.get('exposure')} {corr.get('module')} "
                  f"({corr.get('dra_onsky_mas')}, {corr.get('ddec_onsky_mas')}) "
                  f"mas [{corr.get('source')}]")
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
    if unapplied:
        n = sum(len(c) for _p, _i, _r, c, _w in unapplied)
        print(f"\nREFUSING TO STAGE '{args.field}': {n} astrometry "
              f"correction(s) across {len(unapplied)} correcting-stage record(s) "
              f"were MEASURED and never applied -- the frozen stages ran on top "
              f"of them.  A frozen stage asks only whether the solution still "
              f"matches the m2 freeze, so it passes while the freeze itself is "
              f"the misaligned one.  Apply the corrections "
              f"(scripts/reduction/apply_m2_checkpoint_corrections.py), "
              f"REGENERATE the affected frames from _cal, and re-run cataloging "
              f"from m2 so a post-regeneration record exists.", file=sys.stderr)
    if failed or unapplied:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
