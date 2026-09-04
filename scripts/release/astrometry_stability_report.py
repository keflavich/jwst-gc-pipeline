#!/usr/bin/env python
"""Per-release astrometric stability report for one field.

Every release ships the positions the pipeline measured.  The frozen-stage
checkpoints (m3-m7) re-measure those positions against the m2 freeze, and the
release gate refuses a field whose *shipped* stage moved.  A field can still
ship with movement the gate correctly tolerated -- either inside the budget, or
in an intermediate stage the release does not distribute -- and a tolerated
movement is still a measurement.  For a field whose point-to-point precision is
a few tenths of a mas, a coherent 2-3 mas stage-to-stage shift is a many-sigma
systematic in POSITIONS, and an astrometric user learns of it only if the
release itself carries the number.

This module states it, per release, from that release's own checkpoint records.

Two obvious shortcuts are wrong, so they are not taken:

* It does NOT scrape the ``failures`` text.  A movement inside tolerance never
  becomes a failure line, so a failure-only reader goes blind exactly when the
  tolerance is generous, and reports "nothing moved" for a field that moved.
  Every exposure is re-derived here by differencing the stage record's own
  per-exposure (dra, ddec) against the m2 record's.
* It therefore reports the FULL distribution, not the tail.  Characterising
  only the exposures that crossed a threshold and quoting the tail's tightness
  as evidence of a systematic is a selection effect: brick's m3 nrcb2 group
  reads 1.16 mas over all 12 exposures and 2.26 mas over the 4 above 2.0.

Written into the release tree as ``ASTROMETRY.md`` by ``stage_release.py``;
runnable standalone:

    python scripts/release/astrometry_stability_report.py --field brick --obs-token o004
"""
import argparse
import glob
import json
import math
import os
import statistics
from collections import defaultdict

FROZEN_STAGES = ("m3", "m4", "m5", "m6", "m7")

#: Two group means are "the same offset" only if they agree to this.  A pure
#: sign test calls (+2.50,+0.01) and (+2.52,-0.01) opposed (they differ by
#: 0.02 mas) and calls (+1,0) and (+9,0) agreed (they differ by 8).
GROUP_AGREEMENT_MAS = 1.0


def _record_name(path):
    """(stage, filter, obs) from a checkpoint filename, obs '' when untokened."""
    base = os.path.basename(path)
    stem = base[len("checkpoint_"):-len("_latest.json")]
    bits = stem.split("_")
    return (bits[0] if bits else "",
            bits[1] if len(bits) > 1 else "",
            bits[2] if len(bits) > 2 else "")


def _load(path):
    """Parsed record, or None -- a record that is not a dict is not a record."""
    try:
        with open(path) as fh:
            obj = json.load(fh)
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _records(basepath, obs_token=None):
    """Newest record per (stage, filter, obs), plus the paths that would not load.

    ``obs_token`` filtering is EXACT, including the untokened case: a legacy
    record carrying no observation token is not part of a tokened release, and
    admitting it silently pools another observation's -- or another
    reduction's -- numbers into this report.
    """
    pat = os.path.join(basepath, "astrometry_checkpoints",
                       "checkpoint_*_latest.json")
    out, unreadable = {}, []
    for path in sorted(glob.glob(pat)):
        stage, filt, obs = _record_name(path)
        if obs_token is not None and obs != obs_token:
            continue
        rec = _load(path)
        if rec is None:
            unreadable.append(path)
            continue
        out[(stage, filt, obs)] = rec
    return out, unreadable


def _exposure_positions(record):
    """{exposure key tuple: (dra, ddec)} from a record's per-visit entries."""
    out = {}
    for vis in (record.get("visits") or []):
        if not isinstance(vis, dict):
            continue
        for e in (vis.get("exposures") or []):
            if not isinstance(e, dict):
                continue
            key = tuple(e.get("key") or ())
            dra, ddec = e.get("dra"), e.get("ddec")
            if not key or dra is None or ddec is None:
                continue
            try:
                dra, ddec = float(dra), float(ddec)
            except (TypeError, ValueError):
                continue
            if math.isfinite(dra) and math.isfinite(ddec):
                out[key] = (dra, ddec)
    return out


def _movements(m2_record, stage_record):
    """Every exposure both records measured, with its stage-to-m2 movement.

    The whole population, not the exposures that failed a threshold.
    """
    base = _exposure_positions(m2_record)
    now = _exposure_positions(stage_record)
    rows = []
    for key, (nra, ndec) in sorted(now.items(), key=lambda kv: str(kv[0])):
        if key not in base:
            continue  # no frozen value to have moved away from
        bra, bdec = base[key]
        rows.append(dict(
            key=key,
            visit=str(key[0]) if len(key) > 0 else "",
            exposure=key[1] if len(key) > 1 else "",
            detector=str(key[2]) if len(key) > 2 else "",
            filtername=str(key[3]) if len(key) > 3 else "",
            m2=(bra, bdec), now=(nra, ndec),
            moved_mas=math.hypot(nra - bra, ndec - bdec)))
    return rows


def _stats(rows):
    """Mean shift, scatter and worst case for a set of exposures."""
    if not rows:
        return None
    dra = [r["now"][0] - r["m2"][0] for r in rows]
    ddec = [r["now"][1] - r["m2"][1] for r in rows]
    mean = (statistics.fmean(dra), statistics.fmean(ddec))
    # Sample (n-1) standard deviation: these are measurements, not a
    # population, and the population form understates the spread at small n.
    scat = ((statistics.stdev(dra) if len(dra) > 1 else 0.0),
            (statistics.stdev(ddec) if len(ddec) > 1 else 0.0))
    return dict(n=len(rows), mean=mean, scatter=scat,
                magnitude=math.hypot(*mean),
                worst=max(r["moved_mas"] for r in rows))


def _group_key(row):
    """(filter, visit, detector) -- the grouping the mechanism acts on.

    Deliberately NOT including the dither group: within one filter and
    observation it is constant (brick F200W/o004 is '04101' throughout), so
    adding it splits nothing while implying the report can resolve dither
    groups.  Sub-structure within a detector shows up as scatter instead.
    """
    return (row["filtername"], row["visit"], row["detector"])


def collect(basepath, obs_token=None):
    """Movement of every exposure at every frozen stage, from one field's records."""
    recs, unreadable = _records(basepath, obs_token)
    per_stage, tolerance = {}, None
    for (stage, filt, obs), rec in sorted(recs.items()):
        if stage not in FROZEN_STAGES:
            continue
        m2 = recs.get(("m2", filt, obs))
        if m2 is None:
            continue
        rows = _movements(m2, rec)
        if not rows:
            continue
        tol = (rec.get("tolerances") or {}).get("stage_stability_tol_mas")
        if isinstance(tol, (int, float)) and tolerance is None:
            tolerance = float(tol)
        per_stage.setdefault(stage, []).extend(rows)
    return dict(per_stage=per_stage, tolerance_mas=tolerance,
                unreadable=unreadable, n_records=len(recs))


def _table(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[_group_key(r)].append(r)
    lines = ["| filter | visit | detector | n | mean dRA | mean dDec "
             "| scatter dRA/dDec | worst |",
             "|---|---|---|---|---|---|---|---|"]
    for key in sorted(groups):
        s = _stats(groups[key])
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {s['n']} | "
            f"{s['mean'][0]:+.2f} | {s['mean'][1]:+.2f} | "
            f"{s['scatter'][0]:.2f}/{s['scatter'][1]:.2f} | {s['worst']:.2f} |")
    return "\n".join(lines)


def _group_spread_mas(gstats):
    """Largest separation between any two group mean shifts."""
    spread = 0.0
    means = list(gstats.values())
    for a in means:
        for b in means:
            spread = max(spread, math.hypot(a["mean"][0] - b["mean"][0],
                                            a["mean"][1] - b["mean"][1]))
    return spread


def _describe_stage(stage, rows, out):
    groups = defaultdict(list)
    for r in rows:
        groups[_group_key(r)].append(r)
    gstats = {k: _stats(v) for k, v in groups.items()}
    overall = _stats(rows)
    spread = _group_spread_mas(gstats)

    out.append(f"### Stage {stage}\n")
    out.append(f"{overall['n']} exposure(s) measured at both m2 and {stage}, in "
               f"{len(groups)} group(s) of (filter, visit, detector).  "
               f"Worst single exposure **{overall['worst']:.2f} mas**.")

    if len(groups) > 1 and spread > GROUP_AGREEMENT_MAS:
        out.append(
            f"\n**The groups do not share one offset.**  Their mean shifts span "
            f"{spread:.2f} mas, so there is no single number for this stage: the "
            f"movement is *differential* between detectors.  A pooled average "
            f"would describe none of them, and the per-group table is the "
            f"measurement.")
    else:
        out.append(
            f"\nThe groups agree to within {spread:.2f} mas.  Pooled mean shift "
            f"**({overall['mean'][0]:+.2f}, {overall['mean'][1]:+.2f}) mas**, "
            f"magnitude **{overall['magnitude']:.2f} mas**.")

    tight = [k for k, s in gstats.items()
             if s["n"] > 1 and math.hypot(*s["scatter"]) < 0.5 * s["magnitude"]]
    if tight:
        out.append(
            f"\n{len(tight)} of {len(groups)} group(s) have a scatter well below "
            f"their own mean shift: within those, the exposures moved together "
            f"rather than independently.")
    out.append("\n" + _table(rows) + "\n")


def render(field, data, version=None):
    """The report, as Markdown."""
    per_stage = data["per_stage"]
    out = [f"# Astrometric stability — {field}"]
    if version:
        out.append(f"\nRelease `{version}`.")
    out.append("""
## What this file is

The pipeline measures each star's position, freezes that answer at the m2
stage, and re-measures it at every later stage (m3–m7).  This file reports what
those re-measurements found, for the products in *this* release.  It is rebuilt
for every release, because the numbers describe the files shipped beside it.

Every exposure measured at both stages is included, whether or not it crossed
the gate's tolerance — a movement the gate tolerated is still a movement, and
reporting only the ones that failed would both hide the rest and overstate the
ones shown.

**This does not affect photometry.**  Brightnesses are fitted at the position
the star occupies in each frame; a shift in the *reported* position does not
change the flux measured there.  What it affects is anything that uses the
positions themselves — proper motions, parallaxes, cross-matching against an
external catalog at a different epoch, or differential astrometry between
filters or detectors.
""")

    if data.get("unreadable"):
        out.append("> **Incomplete:** these checkpoint record(s) could not be "
                   "read, so this report does not cover them:\n>\n"
                   + "\n".join(f"> - `{os.path.basename(p)}`"
                               for p in data["unreadable"]) + "\n")

    if not per_stage:
        out.append(f"""## Result

No frozen-stage movement could be measured for this field: of
{data.get('n_records', 0)} checkpoint record(s), none paired a frozen stage with
an m2 baseline over the same exposures.  That is the expected result for a field
whose frozen stages have not run, and it is NOT a statement that the positions
are stable.
""")
        return "\n".join(out)

    out.append("## What moved\n")
    tol = data.get("tolerance_mas")
    if tol is not None:
        out.append(f"The gate's frozen-stage tolerance is **{tol:.1f} mas** per "
                   f"exposure.  Movements below it are reported here but did not "
                   f"block the release.\n")
    for stage in FROZEN_STAGES:
        rows = per_stage.get(stage)
        if rows:
            _describe_stage(stage, rows, out)

    out.append("""## How to read this

The checkpoint records carry what is needed to tell a reference-population
change from actual motion of the frames, and the two have different
consequences:

1. **The baked corrections** (`raoffset_meta`/`deoffset_meta`, mirroring each
   frame's `RAOFFSET`/`DEOFFSET`).  Identical between m2 and the later stage
   means the frame did not move and the change is in what it was compared
   against.  Different means the frame itself was re-corrected.
2. **The visit consensus** (`n_stars`, `median_scatter_mas`).  Each stage
   rebuilds it from that stage's detections, so a later stage measures against
   a different — usually smaller and tighter — reference population.
3. **The grouping.**  Movement confined to one detector, or differing between
   detectors, is a position-dependent term rather than a rigid field offset.

This report does not assert which applies; it reports the measurement.

## What it implies

* **Photometry: unaffected.**
* **Absolute astrometry:** the tie to the reference catalog is a separate,
  larger-scale measurement, gated separately (`check_astrometry_checkpoints`,
  and the cross-filter check at m7).  Nothing here speaks to it.
* **Differential astrometry: use the numbers above as a floor.**  Where groups
  disagree, position differences between those detectors carry the quoted
  spread as a systematic, and it does not average down with more stars.
* **Proper motions:** a programme differencing this release against another
  epoch inherits these terms in the difference.
""")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", required=True)
    ap.add_argument("--basepath", default=None,
                    help="default /orange/adamginsburg/jwst/<field>")
    ap.add_argument("--obs-token", default=None,
                    help="restrict to records carrying exactly this token, "
                         "e.g. o004; omit to use every record")
    ap.add_argument("--version", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    basepath = args.basepath or f"/orange/adamginsburg/jwst/{args.field}"
    text = render(args.field, collect(basepath, args.obs_token), args.version)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
