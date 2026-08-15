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


def collect(root):
    """``(all_groups, pooled_groups)`` -- each a list of member (dra, ddec) lists."""
    seen, everything, pooled = set(), [], []
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
            sig = (field, filt, g.get('exposure'), g.get('vgroup'), mods)
            if sig in seen:
                continue
            seen.add(sig)
            members = [by_key[k] for k in by_key
                       if k[2] in mods and k[1] == g.get('exposure')
                       and str(k[4]) == str(g.get('vgroup'))]
            if len(members) < 2:
                continue
            everything.append(members)
            try:
                _assert_poolable([{}] * len(mods), list(mods), 'row', Table(), 't.csv')
            except OffsetsTableUpdateError:
                continue
            sep = max(float(np.hypot(a[0] - b[0], a[1] - b[1]))
                      for a, b in itertools.combinations(members, 2))
            if sep > MAX_POOL_SPREAD_MAS:
                continue
            pooled.append(members)
    return everything, pooled


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

    everything, pooled = collect(args.root)
    print(f"checkpoint records read: {len(_records(args.root))}")
    report(everything, 'EVERY group with a pooled_from (the wrong population)')
    report(pooled, 'Groups this function POOLS (the docstring figures)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
