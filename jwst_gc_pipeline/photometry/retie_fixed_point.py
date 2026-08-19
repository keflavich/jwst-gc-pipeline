"""Detect a re-tie loop that is repeating itself rather than converging.

``run_field_retie_loop.sh`` stops on two conditions: the m2 checkpoint passes
(converged), or the offsets table is byte-identical after a failed finalize
(nothing was re-tied, so something else broke).  Neither fires when the loop
reaches a FIXED POINT: the checkpoint keeps measuring the same residual, keeps
"correcting" it, and the table changes in the last decimal place every pass, so
the md5 differs and the loop runs to MAXITER.

sgrc, iterations 3-6 (F162M, o012) -- the same 21 corrections, the same values
to ~0.1 mas, four passes running::

                      iter 3        iter 4        iter 5        iter 6
    nrca2 exp1   -2.41/+0.58   -2.44/+0.62   -2.41/+0.55   -2.43/+0.62
    nrca3 exp1   -2.07/+3.29   -2.10/+3.40   -2.07/+3.26   -2.10/+3.41
    nrca4 exp4   -1.29/+3.63   -1.32/+3.67   -1.29/+3.60   -1.32/+3.67

Applying the corrections that clear the floor does not reduce what the next
pass measures.  That is the diagnosis, not a step towards one: the residual is
not something a per-exposure rigid shift can remove -- intrinsic scatter, or a
distortion-class systematic -- and six more passes at ~7 h each cannot discover
anything the first two did not already show.

Stopping on it is not a substitute for deciding what the residual IS.  It ends
the spend and reports the numbers the decision needs.

``--accept-below-mas`` is that decision, made once and written down.  A fixed
point says the residual REPEATS; it says nothing about how BIG it is, and the
two need different answers:

* a few mas that will not close is a SIAF/DVA-class systematic -- the
  per-exposure offsets table cannot express a per-detector distortion term, so
  no number of further passes removes it, and holding the field costs the whole
  m3-m7 chain to learn nothing;
* twenty-five mas that will not close is a correction that is not REACHING the
  frame, which is a defect, and stopping is right.

Given a ceiling, this exits 4 for the first and 3 for the second, and prints
the ``ASTROM_M2_CORRECTION_FLOOR_MAS`` that lets the frozen m3+ stages run over
a residual that is measured, recorded and left alone.  Without the flag every
fixed point still stops, which is the behaviour this had before.
"""
import collections
import glob
import json
import math
import os
import re

# The one cross-module import in this file.  It is here rather than inlined so
# MAX_ACCEPT_CEILING_MAS cannot drift from the gate it is derived from.
from .astrometry_checkpoint import LOCAL_CELL_TOL_MAS

#: Two iterations agree when every shared correction agrees to better than this.
#: Well under the 2 mas checkpoint tolerance -- the question is not "is the
#: frame aligned" (it is not, or there would be no correction) but "is applying
#: these changing what we measure".  sgrc's pass-to-pass wobble is ~0.1 mas.
DEFAULT_TOL_MAS = 0.5

#: How many passes of history to judge on.  Two identical passes could be a
#: coincidence of rounding; three is a loop.
#:
#: THREE, not four, because `--since` bounds the scan to this run and the loop's
#: own default cap is MAXITER=3 -- at most one record per filter per iteration,
#: so a default of 4 made the check unreachable on any loop that had not been
#: given a larger MAXITER.  It printed "3 pass(es) recorded, need 4 to judge"
#: and exited 0, which is the silence this whole check exists to remove.
#:
#: Period-2 OSCILLATION still needs two lag-2 comparisons to separate from a
#: single chance match, so that branch needs four passes and therefore
#: MAXITER>=4; at three it declines and the field reads as "still moving",
#: which is the conservative direction.  A plain REPEAT is caught at three.
DEFAULT_REPEATS = 3

#: Hardest ceiling `--accept-below-mas` may be given.
#:
#: DERIVED, not chosen.  The quantity being waived is a PER-EXPOSURE residual,
#: and the nearest downstream gate that sees one is the m7 local-cell gate
#: (`LOCAL_CELL_TOL_MAS`).  A ceiling above it admits floors that spend a whole
#: m3-m7 chain and then fail at m7 on the same number, which is the worst of
#: both: the compute is spent and the field is still blocked.  So the hardest
#: ceiling IS that gate, imported rather than copied so the two move together.
#:
#: An earlier revision set 50.0 and justified it against
#: `REFERENCE_CROSSCHECK_GROSS_MAS` (~100).  That gate bounds the WHOLE-VISIT
#: consensus-vs-reference separation -- which this module explicitly exempts
#: from the floor via `_is_whole_consensus_shift` -- so it never bounded the
#: quantity in hand.  At 50 the tool accepted sgra/o001 at a 19.73 mas residual
#: and printed a floor of 20.3, while this module's own docstring calls
#: "twenty-five mas that will not close" a defect.
#:
#: Without any cap, `inf`, `nan` and 1e9 all reached the acceptance branch,
#: caught only by a character class in the shell wrapper that let 100000 and
#: 999999.9 through.
MAX_ACCEPT_CEILING_MAS = float(LOCAL_CELL_TOL_MAS)


def measurements(rec):
    """``{exposure key: (dra, ddec)}`` for every exposure the pass MEASURED.

    Deliberately not the ``corrections`` list.  Which exposures appear there
    depends on which ones happen to cross the correction floor, and that
    membership churns pass to pass even when nothing moves: sgrc F162M went
    ``21 -> 21`` corrections with five keys swapped, purely because exposures
    sitting at 2-4 mas drift across a 4 mas threshold.  Comparing the set would
    read that as "still converging".

    The per-exposure measurement is the quantity the question is actually
    about: does re-tying change what the next pass measures?
    """
    out = {}
    for visit in (rec.get("visits") or []):
        for exp in (visit.get("exposures") or []):
            dra, ddec = exp.get("dra"), exp.get("ddec")
            if dra is None or ddec is None:
                continue
            out[tuple(exp.get("key") or ())] = (float(dra), float(ddec))
    return out


#: ``checkpoint_m2_F162M_o012_20260809T044537Z.json``
_STAMP_RE = re.compile(r"_(\d{8}T\d{6}Z)\.json$")


def record_stamp(path):
    """The record's own timestamp, or None."""
    m = _STAMP_RE.search(os.path.basename(path))
    return m.group(1) if m else None


def load_records(record_dir, stage="m2", filtername=None, obs_token=None,
                 since=None):
    """Timestamped checkpoint records, oldest first.

    ``*_latest.json`` is deliberately excluded: it is a COPY of the newest
    timestamped record, so counting it would compare a pass against itself and
    report a repeat that never happened.

    ``obs_token`` is a PREFERENCE, not a filter.  Only sgrc, gc2211 and
    gc-treasury (10678) write ``_oNNN`` m2 records; on brick, cloudc, cloudef, sgrb2, arches, quintuplet,
    sickle, sgra and ngc6334 the records are untokened, and requiring the token
    made this glob match nothing -- the check exited 0 in silence and the loop
    ran to MAXITER on exactly the fields with the longest histories.  So: use
    the tokened records where they exist, and the untokened ones where they do
    not, which is what the records themselves mean.
    """
    filt = filtername or "*"
    def _scan(tok_glob):
        return [p for p in glob.glob(
            os.path.join(record_dir, f"checkpoint_{stage}_{filt}{tok_glob}.json"))
            if not os.path.basename(p).endswith("_latest.json")]

    paths = _scan(f"_{obs_token}_*") if obs_token else []
    if not paths:
        paths = _scan("_*")
    if since:
        # Records from an EARLIER campaign are not this loop's passes.  brick,
        # cloudc and cloudef all carry repeating July histories; without this the
        # first re-run of any of them would stop at iteration 2 citing passes
        # from a different campaign.  The loop passes its own start time.
        paths = [p for p in paths
                 if (record_stamp(p) or "") >= str(since)]
    out = []
    for path in sorted(paths):
        try:
            with open(path) as fh:
                out.append((path, json.load(fh)))
        except (OSError, ValueError):
            continue
    return out


def _group_by_filter_token(records):
    """Records keyed by the (filter, obs token) they describe.

    A field runs several filters, each with its own iteration history; pooling
    them would compare F162M's pass against F212N's and find a difference that
    means nothing.
    """
    groups = collections.defaultdict(list)
    for path, rec in records:
        base = os.path.basename(path)
        parts = base[len("checkpoint_"):].rsplit("_", 1)[0].split("_")
        # checkpoint_<stage>_<FILTER>[_<token>]
        filt = parts[1] if len(parts) > 1 else ""
        token = parts[2] if len(parts) > 2 else ""
        groups[(filt, token)].append((path, rec))
    # A filter with BOTH histories is one continuous history that was
    # re-tokened mid-campaign (sgrc F115W: untokened to 2026-08-06, _o012 from
    # 2026-08-07).  Judging them as two series reports two contradictory
    # verdicts for one filter, and the stale untokened tail -- which nobody is
    # writing any more -- can be long enough to judge while the live tokened one
    # is not.  Drop the untokened group when a tokened sibling exists, which is
    # what #341 settled on for the same collision.
    tokened_filters = {f for (f, t) in groups if t}
    for filt in tokened_filters:
        groups.pop((filt, ""), None)
    return groups


def compare(rec_a, rec_b, tol_mas=DEFAULT_TOL_MAS):
    """Do two records carry the same corrections, to ``tol_mas``?

    Returns ``(same, detail)``.  ``detail`` names what differs, so a loop that
    is genuinely still moving says so with a number rather than "not converged".
    """
    a, b = measurements(rec_a), measurements(rec_b)
    shared = set(a) & set(b)
    if not shared:
        return False, (f"no exposure measured in both passes "
                       f"({len(a)} vs {len(b)})")
    if not (rec_a.get("corrections") or rec_b.get("corrections")):
        # Nothing was corrected in either pass: that is the checkpoint PASSING,
        # which is the loop's own converged exit.  Not a fixed point to report.
        return False, "no corrections in either pass"
    worst_key, worst = None, 0.0
    for k in shared:
        d = max(abs(a[k][0] - b[k][0]), abs(a[k][1] - b[k][1]))
        if d > worst:
            worst_key, worst = k, d
    detail = (f"{len(shared)} exposure(s) measured in both, largest "
              f"pass-to-pass change {worst:.2f} mas")
    if worst > tol_mas:
        return False, f"{detail} at {worst_key}"
    return True, f"{detail} (tol {tol_mas} mas)"


def _correctable(rec, key):
    """Would m2 act on this exposure's residual, or is it excluded upstream?

    The record holds an entry for EVERY exposure in the visit consensus, not
    only the ones the checkpoint would correct.  Two kinds are excluded there
    and must be excluded here too, because this number sizes the floor that m2
    will then apply:

    * ``alias_suspect`` -- a module-antisymmetric residual, which the checkpoint
      never corrects (it is a detector-naming collision, not a misalignment);
    * anything the pass did not call misaligned.

    Including them cuts both ways.  A 900 mas alias alongside a 3 mas fixed
    point refuses acceptance for the wrong reason; a 12 mas one sets a 12.1 mas
    floor that then suppresses real 10 mas corrections.  The number driving the
    decision has to be the number m2 would act on.
    """
    if rec is None:
        return True
    for visit in (rec.get("visits") or []):
        for exp in (visit.get("exposures") or []):
            if tuple(exp.get("key") or ()) != key:
                continue
            if exp.get("alias_suspect"):
                return False
            if "misaligned" in exp:
                return bool(exp["misaligned"])
            return True
    return True


def largest_measured_residual(record_dir, stage="m2", filtername=None,
                              obs_token=None, since=None, only_groups=None):
    """``(worst_mas, key, label)`` over the NEWEST pass of every filter.

    A fixed point says the residual REPEATS.  It does not say how big it is,
    and those are different decisions: a 3 mas residual that will not close is
    a SIAF/DVA-class systematic the offsets table cannot express, while a 25 mas
    one that will not close means the correction is not reaching the frame.
    The loop stops on both today, so a field is held for a decision that the
    records already answer.

    Measured over the newest record per (filter, token) rather than pooled over
    the history: the older passes are what the loop has already superseded, and
    including them reports a residual that no longer exists.

    ``only_groups`` restricts the scan to the ``(filter, token)`` groups that
    actually reached a fixed point.  Without it the worst residual could come
    from a filter that is still converging, and the floor derived from it would
    be an amnesty that filter never needed -- applied to the whole field.
    """
    groups = _group_by_filter_token(
        load_records(record_dir, stage=stage, filtername=filtername,
                     obs_token=obs_token, since=since))
    worst, worst_key, worst_label = 0.0, None, None
    for (filt, token), recs in sorted(groups.items()):
        if only_groups is not None and (filt, token) not in only_groups:
            continue
        path, rec = recs[-1]
        for key, (dra, ddec) in measurements(rec).items():
            if not _correctable(rec, key):
                continue
            mag = (dra ** 2 + ddec ** 2) ** 0.5
            if mag > worst:
                worst, worst_key = mag, key
                worst_label = f"{filt}{'/' + token if token else ''} " \
                              f"({os.path.basename(path)})"
    return worst, worst_key, worst_label


def group_verdicts(record_dir, stage="m2", tol_mas=DEFAULT_TOL_MAS,
                   repeats=DEFAULT_REPEATS, filtername=None, obs_token=None,
                   since=None):
    """``(stuck, moving, unjudged, report_lines)``, per (filter, token).

    ``find_fixed_point`` below is this with the two sets collapsed to one
    boolean, which is all the STOP decision needs.  The ACCEPT decision needs
    them apart: the residual that sizes the correction floor has to come from a
    group that is actually stuck, and a field where one band is stuck while
    another is still converging is a case a person should see rather than one
    to hand an automatic amnesty.

    A group with too little history to judge is in ``unjudged`` -- not known to
    repeat and not known to be moving.  It is reported separately because the
    correction floor is applied to the whole field, so accepting while a group
    is unjudged waives a residual nothing has looked at.
    """
    groups = _group_by_filter_token(
        load_records(record_dir, stage=stage, filtername=filtername,
                     obs_token=obs_token, since=since))
    lines = []
    stuck, moving, unjudged = set(), set(), set()
    if not groups:
        # Silence here reads as "nothing is wrong".  An empty scan means the
        # check did not apply, which the operator has to be able to tell from a
        # clean one -- the same reason #341's denominator and #351's ngc6334
        # all-clear had to say what they had looked at.
        want = f"checkpoint_{stage}_{filtername or '*'}"
        lines.append(f"no {stage} checkpoint records matched {want}* under "
                     f"{record_dir}"
                     + (f" written at/after {since}" if since else "")
                     + " -- the fixed-point check did NOT run")
        return stuck, moving, unjudged, lines
    for (filt, token), recs in sorted(groups.items()):
        label = f"{filt}{'/' + token if token else ''}"
        if len(recs) < repeats:
            unjudged.add((filt, token))
            lines.append(f"{label}: {len(recs)} pass(es) recorded, need "
                         f"{repeats} to judge")
            continue
        tail = recs[-repeats:]
        verdicts = [compare(tail[i][1], tail[i + 1][1], tol_mas=tol_mas)
                    for i in range(len(tail) - 1)]
        if all(v[0] for v in verdicts):
            stuck.add((filt, token))
            lines.append(f"{label}: REPEATING over the last {repeats} pass(es) "
                         f"-- {verdicts[-1][1]}")
        else:
            # Period-2: pass N reproduces pass N-2 while differing from N-1.
            # This is sgrc F162M -- corrections push a set of exposures one way,
            # the next pass pushes them back, and the loop alternates forever.
            # Judged on the same evidence and just as unresolvable by more
            # iterations, but it is a different thing to report: the residual is
            # being MOVED, not merely re-measured.
            lagged = [compare(tail[i][1], tail[i + 2][1], tol_mas=tol_mas)
                      for i in range(len(tail) - 2)]
            if len(lagged) >= 2 and all(v[0] for v in lagged):
                stuck.add((filt, token))
                lines.append(f"{label}: OSCILLATING with period 2 over the last "
                             f"{repeats} pass(es) -- {lagged[-1][1]}; "
                             f"consecutive passes differ by "
                             f"{verdicts[-1][1].split('change ')[-1]}")
            else:
                moving.add((filt, token))
                detail = next(v[1] for v in verdicts if not v[0])
                lines.append(f"{label}: still moving -- {detail}")
                continue
        for path, _ in tail:
            lines.append(f"    {os.path.basename(path)}")
    return stuck, moving, unjudged, lines


def find_fixed_point(record_dir, stage="m2", tol_mas=DEFAULT_TOL_MAS,
                     repeats=DEFAULT_REPEATS, filtername=None, obs_token=None,
                     since=None):
    """``(is_fixed_point, report_lines)`` for a field's checkpoint history.

    A fixed point is declared only when SOME (filter, token) has repeated
    ``repeats`` times running.  One filter looping is enough: the loop
    re-reduces the whole field every pass, so the cost is the same whether one
    filter or all of them are stuck.

    Returns the set of stuck groups, which is truthy exactly when the boolean
    this used to return was True; ``group_verdicts`` is the same scan with the
    still-moving groups kept as well.
    """
    stuck, _moving, _unjudged, lines = group_verdicts(
        record_dir, stage=stage, tol_mas=tol_mas, repeats=repeats,
        filtername=filtername, obs_token=obs_token, since=since)
    return stuck, lines


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--record-dir", required=True,
                    help="{basepath}/astrometry_checkpoints")
    ap.add_argument("--stage", default="m2")
    ap.add_argument("--filter", default=None, dest="filtername")
    ap.add_argument("--obs-token", default=None)
    ap.add_argument("--tol-mas", type=float, default=DEFAULT_TOL_MAS)
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--since", default=None,
                    help="ignore records stamped before this (YYYYMMDDTHHMMSSZ);"
                         " the retie loop passes its own start time so an"
                         " earlier campaign's passes are not counted as this"
                         " loop's")
    # `None` (omitted) and `""` (given, empty) have to stay distinguishable:
    # an explicit empty declaration silently disabled the coverage gate.
    ap.add_argument("--expect-filters", default=None,
                    help="space-separated filters THIS OBSERVATION "
                         "is running -- an operator declaration scoped to the "
                         "scan, not the field's filter list.  gc2211 o046 "
                         "images F200W and F277W but not F150W, so passing the "
                         "field list there refuses forever.  Any declared "
                         "filter the scan did not cover refuses acceptance: "
                         "the scan is scoped by --obs-token/--since, the "
                         "correction floor is not.")
    ap.add_argument("--accept-below-mas", type=float, default=0.0,
                    help="when a fixed point is reached and the largest"
                         " residual still measured is below this, exit 4"
                         " (BOUNDED) instead of 3 (STOP), and print the m2"
                         " correction floor that lets the frozen stages run."
                         " 0 (default) disables it: every fixed point stops.")
    args = ap.parse_args(argv)
    # WHITESPACE only, which is what `run_field_retie_loop.sh:138` does with
    # the same string (`read -r -a`).  An earlier revision also split on commas,
    # citing docs/HIPERGATOR.md:118 -- that line documents a comma as a MODE
    # SWITCH for the cataloging monolith, on a different script, and the reduce
    # this loop submits reads `FILTERS="F115W,F162M"` as ONE filter (NF=1).
    # Accepting the comma here would have the gate see three declared filters
    # while the loop submits a one-task reduce array from the same string.
    _declared = args.expect_filters
    args.expect_filters = str(_declared or '').split()
    if _declared is not None and not args.expect_filters:
        ap.error("--expect-filters was given but contains no filter names "
                 "(whitespace only).  An empty declaration silently disables "
                 "the coverage check, which is the regression it exists to "
                 "stop; omit the flag to run without it.")
    if args.accept_below_mas and not math.isfinite(args.accept_below_mas):
        ap.error(f"--accept-below-mas {args.accept_below_mas} is not a finite "
                 f"number of milliarcseconds")
    if args.accept_below_mas > MAX_ACCEPT_CEILING_MAS:
        ap.error(f"--accept-below-mas {args.accept_below_mas} exceeds "
                 f"{MAX_ACCEPT_CEILING_MAS} mas.  A ceiling that large accepts "
                 f"the gross-misalignment class this check exists to stop; if "
                 f"that is really wanted, raise MAX_ACCEPT_CEILING_MAS "
                 f"deliberately and say why.")

    stuck, moving, unjudged, lines = group_verdicts(
        args.record_dir, stage=args.stage, tol_mas=args.tol_mas,
        repeats=args.repeats, filtername=args.filtername,
        obs_token=args.obs_token, since=args.since)
    for line in lines:
        print(line)
    # The scan is SCOPED by --obs-token and --since; the floor is not -- it is
    # applied to every filter in the field.  A filter whose only history is
    # untokened, or predates --since, disappears from `groups` entirely: not
    # stuck, not moving, not even unjudged.  On sgrc `--obs-token o012` hid six
    # groups the bare scan calls unjudged, with residuals up to 4.99 mas.  So
    # the caller declares what the field is running and the scan has to have
    # covered it.
    unscanned = sorted(
        f for f in (args.expect_filters or [])
        if f.upper() not in {g[0].upper() for g in (stuck | moving | unjudged)})
    # The other half of the declaration contract, and the half that matters.
    #
    # `unscanned` catches OVER-declaration -- a filter named that the scan
    # cannot see.  Its mirror is UNDER-declaration, and the dangerous case is
    # NOT a filter the scan judged and the operator forgot to name: that one is
    # already covered by the moving refusal, the unjudged refusal and
    # `largest_measured_residual`.  It is a filter whose records lie OUTSIDE
    # this scan -- untokened while a tokened sibling exists, or older than
    # `--since` -- and which is not declared either.  That filter is in no
    # group, so nothing above sees it, and the correction floor still reaches
    # it.  Measured: F162M/o012 stuck beside an untokened F200W history at
    # 8602 mas accepted at rc=4 without F200W being named anywhere.
    #
    # So the comparison is against an UNSCOPED read of the record directory --
    # every filter with a record at this stage, regardless of token or date --
    # which is exactly the population the floor reaches.  Reported rather than
    # refused: an operator narrowing the scope deliberately is legitimate, and
    # nothing here can tell that from a forgotten filter.
    declared = {f.upper() for f in (args.expect_filters or [])}
    undeclared = []
    if declared:
        every = {str(f).upper() for f, _tok in _group_by_filter_token(
            load_records(args.record_dir, stage=args.stage))}
        undeclared = sorted(every - declared)
    if not stuck:
        return 0
    print("\nFIXED POINT: applying these corrections does not change what "
          "the next pass measures.\nThe residual is not something a "
          "per-exposure rigid shift removes -- inspect the records above "
          "(intrinsic scatter? distortion? centroid bias?) rather than "
          "running more iterations.")
    if args.accept_below_mas <= 0:
        return 3
    if undeclared:
        print(f"\nNOTE: this scan also holds records for "
              f"{', '.join(undeclared)}, which --expect-filters did not declare."
              f"  The correction floor reaches every filter this run catalogs,"
              f" so a filter left out of the declaration is floored without its"
              f" coverage ever being checked.  Declare every filter the run"
              f" catalogs, or run them in separate scopes.")
    if unscanned:
        # Report the measured residual BEFORE returning.  This branch and the
        # NOT-BOUNDED one below share a return code, so ordering them this way
        # costs no safety -- but returning first threw away the number that
        # identifies the problem: gc2211/o023 under the production invocation
        # printed the coverage refusal and lost `9340.44 mas at
        # ('1',4,'nrcb1','F200W','02201')`, which is the whole diagnosis.
        _worst, _key, _label = largest_measured_residual(
            args.record_dir, stage=args.stage, filtername=args.filtername,
            obs_token=args.obs_token, since=args.since)
        if _key is not None:
            print(f"\nlargest measured residual in scope: {_worst:.2f} mas at "
                  f"{_key} [{_label}] (the filters named below were NOT "
                  f"scanned and are not in this number)")
        print(f"\nNOT ACCEPTED: {', '.join(unscanned)} produced no checkpoint "
              f"record this scan could see.  --expect-filters is an OPERATOR "
              f"DECLARATION scoped to the observation being scanned, not the "
              f"field's filter list: gc2211 o046 images F200W and F277W but not "
              f"F150W, so declaring the field list there refuses forever.  If "
              f"those filters are genuinely absent from this observation, drop "
              f"them from --expect-filters; if they are present, they have no "
              f"record this scan could see and the correction floor -- which is "
              f"applied to every filter in the field, unlike the scan -- would "
              f"waive whatever they are doing.  STOPPING.")
        return 3
    if unjudged:
        # A group with too little history is not "converging" and not "stuck" --
        # it is unknown, and the floor is applied to the WHOLE field.  w51 has
        # an unjudged group whose newest residual is 29 arcseconds sitting
        # beside three stuck ones reading zero.
        names = ", ".join(f"{f}{'/' + t if t else ''}" for f, t in sorted(unjudged))
        print(f"\nNOT ACCEPTED: {names} {'has' if len(unjudged) == 1 else 'have'} "
              f"too little history to judge, and the correction floor is applied "
              f"to every filter in the field.  Accepting now would waive a "
              f"residual nothing has looked at.  STOPPING.")
        return 3
    if moving:
        # One band stuck while another is still halving its residual is not a
        # field to hand an amnesty to: the floor is applied to the WHOLE field,
        # so the converging band would ship under a ceiling it never needed and
        # was one pass from clearing.  Stop and let a person look.
        names = ", ".join(f"{f}{'/' + t if t else ''}" for f, t in sorted(moving))
        print(f"\nNOT ACCEPTED: {names} {'is' if len(moving) == 1 else 'are'} "
              f"still converging while "
              + ", ".join(f"{f}{'/' + t if t else ''}" for f, t in sorted(stuck))
              + f" repeat(s).  The correction floor is applied to every filter "
                f"in the field, so accepting here would waive a residual the "
                f"converging filter(s) had not finished removing.  STOPPING.")
        return 3
    # `only_groups=stuck` was a no-op and is gone: `main` returns 3 on
    # `unjudged` and on `moving` above, and groups = stuck | moving | unjudged,
    # so by here `stuck` IS every group in scope.  An argument that cannot
    # change the result cannot be pinned by a test, and claiming it was pinned
    # was wrong.
    worst, key, label = largest_measured_residual(
        args.record_dir, stage=args.stage, filtername=args.filtername,
        obs_token=args.obs_token, since=args.since)
    if key is None:
        # Nothing correctable was measured, so "the largest residual is 0.00 mas"
        # is the absence of a measurement rather than a small one -- and the
        # floor derived from it would be 0.5 mas, LOWER than the 4.0 the campaign
        # runs at, which makes the next pass correct everything and never
        # converge.  w51's three stuck groups read exactly this today.
        print(f"\nNOT ACCEPTED: the stuck group(s) "
              + ", ".join(f"{f}{'/' + t if t else ''}" for f, t in sorted(stuck))
              + " have no correctable residual in their newest record, so there "
                "is no measurement to bound.  That is not a small residual, it "
                "is the absence of one.  STOPPING.")
        return 3
    if worst >= args.accept_below_mas:
        print(f"\nNOT BOUNDED: the largest residual still measured is "
              f"{worst:.2f} mas at {key} in {label}, at or above the "
              f"{args.accept_below_mas:.1f} mas acceptance ceiling.  A residual "
              f"this large that does not close is not a systematic the table "
              f"cannot express -- it is a correction that is not reaching the "
              f"frame.  STOPPING.")
        return 3
    # The margin has to be the tolerance that DEFINED the fixed point, not a
    # token 0.05.  Consecutive passes are allowed to differ by up to --tol-mas
    # per axis and still count as repeating, so the residual the frozen stages
    # re-measure can legitimately come back that much larger than the newest
    # pass.  A margin smaller than the tolerance puts the field one ordinary
    # wobble away from re-correcting at m2 and raising in the frozen stages --
    # having spent the whole m3-m7 chain to get there, which is the outcome
    # this is supposed to prevent.
    floor = math.ceil((worst + args.tol_mas) * 10) / 10
    if floor > args.accept_below_mas:
        # The margin can push the floor past the ceiling the operator set (a
        # 14.6 mas residual under a 15 mas ceiling gives 15.1).  Running at a
        # floor above the ceiling waives more than was agreed to.
        print(f"\nNOT ACCEPTED: the residual is {worst:.2f} mas, but the floor "
              f"needed to run over it ({floor} mas) exceeds the "
              f"{args.accept_below_mas:.1f} mas ceiling once the {args.tol_mas} "
              f"mas fixed-point tolerance is allowed for.  STOPPING.")
        return 3
    print(f"\nBOUNDED: the largest residual still measured is {worst:.2f} mas "
          f"at {key} in {label}, below the {args.accept_below_mas:.1f} mas "
          f"acceptance ceiling.\nResiduals below the printed floor are still "
          f"measured and recorded in the checkpoint records above; they are "
          f"not applied.  Residuals at or above it are applied as before, and "
          f"the consensus->reference tie is never floored.\n"
          f"ASTROM_M2_CORRECTION_FLOOR_MAS={floor}")
    return 4


if __name__ == "__main__":
    import sys
    sys.exit(main())
