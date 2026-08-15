"""Regenerate the population figures quoted in ``pool_corrections_to_table_granularity``.

Four sets of figures for this have been quoted and three were wrong, each in a
way that came from measuring a DIFFERENT population than the one the argument is
about.  So this states the population explicitly and prints it alongside:

* every group carrying a ``pooled_from`` in a checkpoint record -- the widest
  count, and the wrong one, because most of what it adds is refused before the
  statistic is ever applied;
* the groups that survive ``_assert_poolable`` and the spread refusal, i.e. the
  ones this function actually pools.  Those are the figures in the docstring:
  the choice between mean and median has no effect anywhere else.

Members are reconstructed from the same record's per-exposure measurements,
matched on the group's own (exposure, vgroup) and its ``pooled_from`` modules --
the recorded group holds only the pooled result.

Read-only.  Usage::

    python reports/measure_pooling_population.py [--root /orange/adamginsburg/jwst]
"""
import argparse
import glob
import itertools
import json
import os
from collections import Counter

import numpy as np
from astropy.table import Table

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    MAX_POOL_SPREAD_MAS, OffsetsTableUpdateError, _assert_poolable)


def _records(root):
    return [p for p in sorted(glob.glob(
        os.path.join(root, '*', 'astrometry_checkpoints', '*.json')))
        if not p.endswith('_latest.json')]


#: A NIRCam module has four detectors, so a reconstructed group larger than
#: that did not come from one module -- it came from concatenating two.  Kept as
#: a tripwire rather than a filter: the first version of this script reported
#: N=6 and N=8 groups, which is impossible, and the figures were quoted anyway.
MAX_MEMBERS = 4


def _members_of(by_key, group, mods):
    """The per-exposure measurements this pooled group was built from, or None.

    The exposure key is ``(visit, exposure, module, filter, vgroup)``.  Leaving
    the VISIT unconstrained -- what the first version of this script did --
    concatenates two visits that share an exposure number and vgroup into one
    "group", which is how it produced six- and eight-member groups from a
    four-detector module.  101 of 285 groups had the wrong membership.

    Every reconstruction is then CHECKED against the record's own pooled value.
    The live records were written with the median, so a correct reconstruction
    reproduces it exactly; one that does not is discarded rather than counted.
    """
    want_visit = str(group.get('visit'))
    members = [by_key[k] for k in by_key
               if k[2] in mods
               and k[1] == group.get('exposure')
               and str(k[4]) == str(group.get('vgroup'))
               and (want_visit in ('None', '') or str(k[0]) == want_visit)]
    if not 2 <= len(members) <= MAX_MEMBERS:
        return None
    if group.get('n') is not None and len(members) != int(group['n']):
        return None
    recorded = (group.get('dra_onsky_mas'), group.get('ddec_onsky_mas'))
    if None not in recorded:
        got = (float(np.median([m[0] for m in members])),
               float(np.median([m[1] for m in members])))
        if not (abs(got[0] - recorded[0]) < 1e-6
                and abs(got[1] - recorded[1]) < 1e-6):
            return None
    return members


def group_sizes(root):
    """``{N: count}`` over unique pooled groups, straight from the records.

    The one figure that needs no reconstruction and therefore no discarding:
    ``len(pooled_from)`` is what the pooler collapsed.  Reported beside the
    reconstructed population because the argument for the mean rests on which
    group sizes are common, and a reconstruction that drops what it cannot
    verify would bias exactly that.
    """
    seen, sizes = set(), {}
    for path in _records(root):
        try:
            with open(path) as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            continue
        field = path.split(os.sep + 'astrometry_checkpoints')[0].rsplit(os.sep, 1)[-1]
        filt = rec.get('filtername')
        for g in ((rec.get('pooling') or {}).get('groups') or []):
            mods = tuple(g.get('pooled_from') or ())
            if len(mods) < 2:
                continue
            sig = (field, filt, g.get('visit'), g.get('exposure'),
                   g.get('vgroup'), mods)
            if sig in seen:
                continue
            seen.add(sig)
            sizes[len(mods)] = sizes.get(len(mods), 0) + 1
    return sizes


def collect(root):
    """``(all_groups, pooled_groups, counts)``.

    Each group is a list of member ``(dra, ddec)`` pairs; ``counts`` reports how
    many were discarded and why, so the blast-radius numbers quoted in the
    pooler's comment come from the same run as the population figures.
    """
    seen, everything, pooled = set(), [], []
    unresolved = refused_poolable = refused_spread = 0
    for path in _records(root):
        try:
            with open(path) as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            continue
        field = path.split(os.sep + 'astrometry_checkpoints')[0].rsplit(os.sep, 1)[-1]
        filt = rec.get('filtername')
        by_key = {}
        for visit in (rec.get('visits') or []):
            for exp in (visit.get('exposures') or []):
                key = tuple(exp.get('key') or ())
                if len(key) >= 5 and exp.get('dra') is not None:
                    by_key[key] = (float(exp['dra']), float(exp['ddec']))
        for g in ((rec.get('pooling') or {}).get('groups') or []):
            mods = tuple(g.get('pooled_from') or ())
            if len(mods) < 2:
                continue
            sig = (field, filt, g.get('visit'), g.get('exposure'),
                   g.get('vgroup'), mods)
            if sig in seen:
                continue
            seen.add(sig)
            members = _members_of(by_key, g, mods)
            if members is None:
                unresolved += 1
                continue
            everything.append(members)
            try:
                _assert_poolable([{}] * len(mods), list(mods), 'row', Table(), 't.csv')
            except OffsetsTableUpdateError:
                refused_poolable += 1
                continue
            sep = max(float(np.hypot(a[0] - b[0], a[1] - b[1]))
                      for a, b in itertools.combinations(members, 2))
            if sep > MAX_POOL_SPREAD_MAS:
                refused_spread += 1
                continue
            pooled.append(members)
    return everything, pooled, dict(unresolved=unresolved,
                                    refused_poolable=refused_poolable,
                                    refused_spread=refused_spread)


def _shift(members):
    """How far the pooled value moves between the two statistics, in mas."""
    dra = [m[0] for m in members]
    ddec = [m[1] for m in members]
    return float(np.hypot(np.mean(dra) - np.median(dra),
                          np.mean(ddec) - np.median(ddec)))


def _magnitude(members):
    return float(np.hypot(np.mean([m[0] for m in members]),
                          np.mean([m[1] for m in members])))


def report(groups, label):
    n = Counter(len(g) for g in groups)
    total = sum(n.values())
    print(f"\n{label}: {total} groups")
    if not total:
        return
    for size in sorted(n):
        sub = [g for g in groups if len(g) == size]
        shifts = [_shift(g) for g in sub]
        mags = [_magnitude(g) for g in sub]
        print(f"  N={size}  {n[size]:4d}  {100 * n[size] / total:4.1f}%   "
              f"|mean-median| p50 {np.percentile(shifts, 50):.3f} "
              f"p90 {np.percentile(shifts, 90):.3f} max {max(shifts):.3f}   "
              f"|pooled| p50 {np.percentile(mags, 50):.2f} mas")
    shifts = [_shift(g) for g in groups]
    changed = sum(1 for s in shifts if s > 0.5)
    print(f"  all: |mean-median| p50 {np.percentile(shifts, 50):.3f} mas, "
          f"max {max(shifts):.3f} mas")
    print(f"       typical pooled |correction| p50 "
          f"{np.percentile([_magnitude(g) for g in groups], 50):.2f} mas")
    print(f"       groups where the two differ by > 0.5 mas: {changed} "
          f"({100 * changed / total:.1f}%)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', default='/orange/adamginsburg/jwst')
    args = ap.parse_args(argv)

    everything, pooled, counts = collect(args.root)
    print(f"checkpoint records read: {len(_records(args.root))}")
    # Group SIZE needs no reconstruction: it is `pooled_from` in the record.
    # Reported separately because the reconstruction below discards a group it
    # cannot verify, and the size distribution should not inherit that bias --
    # it is the figure the argument for the mean rests on.
    sizes = group_sizes(args.root)
    total = sum(sizes.values())
    print("\ngroup size, counted directly from `pooled_from` (no reconstruction):")
    for k in sorted(sizes):
        print(f"  N={k}  {sizes[k]:4d}  {100 * sizes[k] / total:4.1f}%")
    print(f"groups discarded as unreconstructable: {counts['unresolved']}")
    print(f"refused by _assert_poolable: {counts['refused_poolable']}   "
          f"refused by the spread limit: {counts['refused_spread']}")
    report(everything, 'EVERY group with a pooled_from (the wrong population)')
    report(pooled, 'Groups this function POOLS (the docstring figures)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
