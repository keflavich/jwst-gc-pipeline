#!/usr/bin/env python
"""Give an offsets table Module and Vgroup rows WITHOUT changing any offset.

gc2211's ``Offsets_JWST_Brick2211_VIRAC2locked.csv`` is keyed
``(Visit, Exposure, Filter)`` only.  Its astrometry is solved and recorded --
o023 (+3.112, -1.818)", o028 (-10.289, +20.549)", o050 (-2.521, +5.136)" -- but
the m2 checkpoint cannot write a correction back to it, because a correction is
per (visit, filter, exposure, MODULE, VGROUP) and the table has neither column::

    OffsetsTableUpdateError: 32 corrections spanning module families
    ['nrca', 'nrcb'] land on the same row(s) (19,) ... rebuild it per-module
    (build_virac2_offsets --per-module) so corrections map 1:1.

    OffsetsTableUpdateError: module(s) ['nrcb1'..'nrcb4'] contribute MORE THAN
    ONE correction to the same row(s) (10,) ... typically two visit groups
    against a Vgroup-less table.

Both refusals are correct: pooling module A against module B, or two visit
groups against one row, would average away a real difference.
``alignment_config`` already records the gap ("Rebuild is pending --per-module +
Vgroup"), and ``unified_alignment`` already narrows on both columns when they
exist -- only the table is behind.

## Why expand rather than rebuild

``build_virac2_offsets --region gc2211_0NN --per-module`` re-SOLVES the tie from
the current catalogs.  Those catalogs were produced from frames the existing
table had already been applied to, so a rebuild measures the RESIDUAL, not the
recorded arcsecond-scale solution, and reconciling the two is a separate
question with its own failure modes.

This does something narrower and reversible: each row is replicated once per
(module, vgroup) that actually exists on disk for its (visit, filter,
exposure), with **every offset column copied unchanged**.  The value any given
frame receives is therefore bit-identical to what it receives today -- the same
single row, now reachable by a key that names the module and visit group.  What
changes is only that a correction can address one of them.

Verified by ``--verify``: for every frame on disk, the row ``unified_alignment``
selects from the expanded table carries the same (dra, ddec) as the row it
selects from the original.

## Usage

    python expand_offsets_granularity.py --field gc2211              # dry run
    python expand_offsets_granularity.py --field gc2211 --verify     # + check
    python expand_offsets_granularity.py --field gc2211 --execute
"""
import argparse
import collections
import glob
import os
import re
import shutil
import sys
import time

import numpy as np
from astropy.table import Table

BASE = os.environ.get("GC_BASEPATH_OVERRIDE",
                      os.environ.get("JWST_BASE", "/orange/adamginsburg/jwst"))

#: jw<prop 5><obs 3><visit 3>_<vgroup>_<exp>_<detector>_...
_FRAME_RE = re.compile(
    r"^jw(?P<prop>\d{5})(?P<obs>\d{3})(?P<visit>\d{3})_"
    r"(?P<vgroup>[^_]+)_(?P<exp>\d+)_(?P<det>nrc[ab](?:\d|long)|mirim|nis)_",
    re.IGNORECASE)

#: Short-wave NIRCam detectors collapse to their module family; LONG-WAVE ones
#: keep their own name.  This is not a choice -- it is the convention every
#: other locked table on the tree already uses (cloudc, cloudef, quintuplet,
#: sgrc, sickle all store ``['nrca', 'nrcalong', 'nrcb', 'nrcblong']``) and it
#: is what the reader can match.  `_shift_from_locked` takes the module from the
#: FILENAME (``jw..._00001_nrcalong_cal.fits`` -> ``nrcalong``) and its family
#: fallback is ``thismodule.strip('1234')``, which does nothing to an LW name::
#:
#:     'nrcb1'.strip('1234')    -> 'nrcb'       matches a 'nrcb' row
#:     'nrcalong'.strip('1234') -> 'nrcalong'   matches a 'nrca' row NOT AT ALL
#:
#: Writing LW frames into an ``nrca``/``nrcb`` row therefore took gc2211's 148
#: long-wave frames (68 nrcalong + 80 nrcblong) from resolving to raising
#: ``match.sum() != 1`` -- the table previously had no Module column at all, so
#: the narrowing was skipped and they resolved fine.
_SW_DETECTOR = re.compile(r"^(nrc[ab])[1-4]$", re.IGNORECASE)


def module_label(detector):
    """The ``Module`` cell a detector's row must carry."""
    det = str(detector).lower()
    m = _SW_DETECTOR.match(det)
    return m.group(1) if m else det


def frames_for(field, filt):
    """``{(visit_id, exposure): {(module, vgroup, detector)}}`` from disk.

    The DETECTOR is carried alongside the module label because it is what the
    reader is given: `verify` must ask `locked_row_match` the question a real
    frame asks (``module='nrcalong'``), not the question this tool happens to
    have written into the table (``'nrca'``).  Enumerating frames by their row
    label instead made `--verify` structurally unable to see the LW breakage --
    the same hazard as reimplementing the matching rules, one level up.
    """
    out = collections.defaultdict(set)
    pat = os.path.join(BASE, field, filt, "pipeline", "jw*_cal.fits")
    for path in glob.glob(pat):
        m = _FRAME_RE.match(os.path.basename(path))
        if not m:
            continue
        visit = f"jw{m.group('prop')}{m.group('obs')}{m.group('visit')}"
        det = m.group("det").lower()
        out[(visit, int(m.group("exp")))].add(
            (module_label(det), m.group("vgroup"), det))
    return out


def expand(tbl, field):
    """The expanded table, plus a per-row note of what happened to it."""
    by_filter = {}
    notes = []
    rows = []
    for row in tbl:
        filt = str(row["Filter"])
        if filt not in by_filter:
            by_filter[filt] = frames_for(field, filt)
        found = by_filter[filt].get((str(row["Visit"]), int(row["Exposure"])))
        if not found:
            # No frame on disk for this row.  Keep it EXACTLY as it is: a row
            # this tool cannot resolve is a row it must not touch, and dropping
            # it would delete a recorded solution for data that is merely not
            # staged right now.
            rows.append({c: row[c] for c in tbl.colnames})
            notes.append((str(row["Visit"]), filt, int(row["Exposure"]),
                          "kept as-is (no frames on disk)"))
            continue
        # Several detectors share one module label (nrca1..nrca4 -> 'nrca'), so
        # the ROW set is the distinct (module, vgroup) pairs.
        pairs = sorted({(m, g) for m, g, _det in found})
        for module, vgroup in pairs:
            new = {c: row[c] for c in tbl.colnames}
            new["Module"] = module
            new["Vgroup"] = str(vgroup)
            rows.append(new)
        notes.append((str(row["Visit"]), filt, int(row["Exposure"]),
                      f"-> {len(pairs)} rows: "
                      + ", ".join(f"{m}/{g}" for m, g in pairs)))
    cols = list(tbl.colnames)
    for extra in ("Module", "Vgroup"):
        if extra not in cols:
            cols.append(extra)
    out = Table([[r.get(c, "") for r in rows] for c in cols], names=cols)
    return out, notes


def verify(original, expanded, field):
    """Every frame must still land on its own offset, unchanged.

    Uses `unified_alignment`'s own row selection rather than reimplementing it,
    because a private copy of the matching rules is exactly the thing that would
    stay green while the real reader picked a different row.
    """
    from jwst_gc_pipeline.reduction.unified_alignment import locked_row_match
    problems = []
    checked = 0
    for filt in sorted(set(str(f) for f in original["Filter"])):
        for (visit, exp), mods in sorted(frames_for(field, filt).items()):
            for _module, vgroup, detector in sorted(mods):
                # `detector`, not the row label: this must ask the question a
                # real frame asks.
                kw = dict(visit=visit, exposure=exp, filtername=filt,
                          module=detector, vgroup=vgroup)
                before = original[locked_row_match(original, **kw)]
                after = expanded[locked_row_match(expanded, **kw)]
                checked += 1
                if len(after) != 1:
                    problems.append(
                        f"{visit} {filt} exp{exp} {detector}/{vgroup}: the "
                        f"expanded table matches {len(after)} rows, not 1")
                    continue
                if len(before) != 1:
                    # Unresolvable BEFORE too: this frame is already broken and
                    # the expansion neither caused nor must hide it, but it also
                    # cannot be used as evidence that nothing changed.
                    problems.append(
                        f"{visit} {filt} exp{exp} {detector}/{vgroup}: the "
                        f"ORIGINAL table matches {len(before)} rows, not 1 "
                        f"(pre-existing, not caused by this expansion)")
                    continue
                for col in ("dra", "ddec", "dra (arcsec)", "ddec (arcsec)"):
                    if col not in original.colnames:
                        continue
                    a, b = float(before[col][0]), float(after[col][0])
                    if not (np.isnan(a) and np.isnan(b)) and a != b:
                        problems.append(
                            f"{visit} {filt} exp{exp} {detector}/{vgroup}: "
                            f"{col} {a!r} -> {b!r}")
    return checked, problems


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", required=True)
    ap.add_argument("--table", default=None,
                    help="offsets csv (default: the field's only one)")
    ap.add_argument("--execute", action="store_true",
                    help="write (default is a dry run that changes nothing)")
    # Verifying is the DEFAULT.  The tool's whole claim is that no frame changes
    # value; making the check that establishes it optional -- and then letting
    # --execute fall through to a write without it -- meant --execute alone
    # rewrote the table with nothing checked at all.
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the check that every frame still resolves to the "
                         "same offset (deliberate opt-out; the check is the "
                         "tool's only safety property)")
    ap.add_argument("--verify", action="store_true",
                    help=argparse.SUPPRESS)   # accepted, now redundant
    args = ap.parse_args(argv)

    path = args.table
    if path is None:
        found = sorted(glob.glob(os.path.join(BASE, args.field, "offsets",
                                              "Offsets_*.csv")))
        if len(found) != 1:
            print(f"expected exactly one offsets table under "
                  f"{BASE}/{args.field}/offsets, found {len(found)}; "
                  f"pass --table", file=sys.stderr)
            return 2
        path = found[0]

    tbl = Table.read(path)
    if "Module" in tbl.colnames and "Vgroup" in tbl.colnames:
        print(f"{os.path.basename(path)} already carries Module and Vgroup; "
              f"nothing to do")
        return 0

    out, notes = expand(tbl, args.field)
    print(f"{os.path.basename(path)}: {len(tbl)} rows -> {len(out)}")
    for visit, filt, exp, note in notes:
        print(f"    {visit} {filt} exp{exp}: {note}")

    if not args.no_verify:
        checked, problems = verify(tbl, out, args.field)
        print(f"\nverify: {checked} frame(s) checked")
        if problems:
            print(f"REFUSING: {len(problems)} frame(s) would change offset:")
            for p in problems[:20]:
                print(f"    {p}")
            return 1
        print("every frame resolves to the same offset as before")

    if not args.execute:
        print(f"\ndry run: nothing written; re-run with --execute")
        return 0

    backup = f"{path}.pre_granularity_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    shutil.copy2(path, backup)
    out.write(path, overwrite=True)
    print(f"\nwrote {path}\nbackup {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
