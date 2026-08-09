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
"""
import collections
import glob
import json
import os

#: Two iterations agree when every shared correction agrees to better than this.
#: Well under the 2 mas checkpoint tolerance -- the question is not "is the
#: frame aligned" (it is not, or there would be no correction) but "is applying
#: these changing what we measure".  sgrc's pass-to-pass wobble is ~0.1 mas.
DEFAULT_TOL_MAS = 0.5

#: How many passes of history to judge on.  Two identical passes could be a
#: coincidence of rounding; three is a loop.  Four, because period-2 oscillation
#: needs TWO lag-2 comparisons to separate from a single chance match -- with
#: fewer, that branch declines to fire and the field reads as "still moving".
DEFAULT_REPEATS = 4


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


def load_records(record_dir, stage="m2", filtername=None, obs_token=None):
    """Timestamped checkpoint records, oldest first.

    ``*_latest.json`` is deliberately excluded: it is a COPY of the newest
    timestamped record, so counting it would compare a pass against itself and
    report a repeat that never happened.
    """
    filt = filtername or "*"
    tok = f"_{obs_token}" if obs_token else "*"
    pat = os.path.join(record_dir, f"checkpoint_{stage}_{filt}{tok}_*.json")
    paths = [p for p in glob.glob(pat)
             if not os.path.basename(p).endswith("_latest.json")]
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


def find_fixed_point(record_dir, stage="m2", tol_mas=DEFAULT_TOL_MAS,
                     repeats=DEFAULT_REPEATS, filtername=None, obs_token=None):
    """``(is_fixed_point, report_lines)`` for a field's checkpoint history.

    A fixed point is declared only when SOME (filter, token) has repeated
    ``repeats`` times running.  One filter looping is enough: the loop
    re-reduces the whole field every pass, so the cost is the same whether one
    filter or all of them are stuck.
    """
    groups = _group_by_filter_token(
        load_records(record_dir, stage=stage, filtername=filtername,
                     obs_token=obs_token))
    lines = []
    stuck = False
    for (filt, token), recs in sorted(groups.items()):
        label = f"{filt}{'/' + token if token else ''}"
        if len(recs) < repeats:
            lines.append(f"{label}: {len(recs)} pass(es) recorded, need "
                         f"{repeats} to judge")
            continue
        tail = recs[-repeats:]
        verdicts = [compare(tail[i][1], tail[i + 1][1], tol_mas=tol_mas)
                    for i in range(len(tail) - 1)]
        if all(v[0] for v in verdicts):
            stuck = True
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
                stuck = True
                lines.append(f"{label}: OSCILLATING with period 2 over the last "
                             f"{repeats} pass(es) -- {lagged[-1][1]}; "
                             f"consecutive passes differ by "
                             f"{verdicts[-1][1].split('change ')[-1]}")
            else:
                moving = next(v[1] for v in verdicts if not v[0])
                lines.append(f"{label}: still moving -- {moving}")
                continue
        for path, _ in tail:
            lines.append(f"    {os.path.basename(path)}")
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
    args = ap.parse_args(argv)

    stuck, lines = find_fixed_point(
        args.record_dir, stage=args.stage, tol_mas=args.tol_mas,
        repeats=args.repeats, filtername=args.filtername,
        obs_token=args.obs_token)
    for line in lines:
        print(line)
    if stuck:
        print("\nFIXED POINT: applying these corrections does not change what "
              "the next pass measures.\nThe residual is not something a "
              "per-exposure rigid shift removes -- inspect the records above "
              "(intrinsic scatter? distortion? centroid bias?) rather than "
              "running more iterations.")
        return 3
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
