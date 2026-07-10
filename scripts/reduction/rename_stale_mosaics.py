#!/usr/bin/env python
"""Rename STALE realigned/reprojected mosaics so they stop matching *.fits globs.

Generalisation of the brick-only 2026-07-03 scratch pass (repo-root
``_stale_rename.py``, executed: 192 renamed, log
``<field>/_stale_rename_2026-07-03.log``).  A "merged"/"reproject"/"realigned"
mosaic left over from a superseded reduction carries that reduction's (often
bad) astrometry; keeping it around under a live ``*.fits`` name invites
downstream tools to read it.  STALE files are renamed to
``<name>_badastrometry_stale`` (self-documenting, un-globbable).

STALE = BOTH of:
  * older than its band's current ``*-merged_data_i2d.fits`` by > 1 day
    (same-run files made hours apart are never flagged), AND
  * older than the field's reduction-campaign floor = (newest data_i2d across
    all bands) - 21 days (so a whole in-progress campaign is never flagged
    just because one band finished later).

Patterns covered (the 07-03 pass missed ``merged-reproject-vvv_i2d`` -- the
canonical known-bad 2023 brick files -- because its globs were too narrow;
this version matches any 'reproject' or 'realigned' mosaic variant):
  *reproject*i2d*.fits, *realigned-to-*.fits, *-merged-reproject-*.fits

Band token is parsed from either NIRCam ('clear-f182m-', 'f405n-f444w-'
pupil forms) or MIRI ('_f2550w') filename conventions.

Usage:
  python rename_stale_mosaics.py --field brick [--field cloudc ...] [--execute]

Dry-run by default; --execute renames and appends to
``<fieldpath>/_stale_rename_<date>.log``.
"""
import argparse
import glob
import os
import re
import time

DAY = 86400
BASE = '/orange/adamginsburg/jwst'
# band token: 'clear-f182m-', '-f405n-' (pupil form), or MIRI '_f2550w'
BAND_RES = (re.compile(r'clear-([a-z0-9]+)[-_]', re.I),
            re.compile(r'[-_](f\d{3,4}[mnw])[-_.]', re.I))
PATTERNS = ('*reproject*i2d*.fits', '*realigned-to-*.fits',
            '*-merged-reproject-*.fits')
SUFFIX = '_badastrometry_stale'


def mt(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return None


def fmt(t):
    return time.strftime('%Y-%m-%d', time.localtime(t)) if t else 'NONE'


def band_of(name):
    for rex in BAND_RES:
        m = rex.search(name)
        if m:
            return m.group(1).lower()
    return None


def rename_stale_for_field(field, execute=False, campaign_days=21):
    pipe = f'{BASE}/{field}'
    banddirs = glob.glob(f'{pipe}/*/pipeline')
    if not banddirs:
        print(f"[{field}] no */pipeline dirs under {pipe}; skipping")
        return []

    dref = {}
    for d in banddirs:
        band = os.path.basename(d.rsplit('/pipeline', 1)[0]).lower()
        g = [x for x in glob.glob(f'{d}/*-merged_data_i2d.fits')
             if 'nrca' not in os.path.basename(x)
             and 'nrcb' not in os.path.basename(x)]
        dref[band] = max((mt(x) for x in g), default=None)
    campaign = max((v for v in dref.values() if v), default=None)
    campaign = (campaign - campaign_days * DAY) if campaign else None

    cands = set()
    for d in banddirs:
        for pat in PATTERNS:
            cands.update(glob.glob(f'{d}/{pat}'))

    plan, kept = [], 0
    for f in sorted(cands):
        base = os.path.basename(f)
        if 'stale' in base:
            continue
        band = band_of(base)
        d = dref.get(band)
        fm = mt(f)
        if band is None or d is None or fm is None:
            print(f"  SKIP [{field}] {base} "
                  f"[{'no-band' if band is None else 'no-data_i2d'}]")
            continue
        if fm < d - DAY and (campaign is None or fm < campaign):
            plan.append((f, fm, d))
        else:
            kept += 1

    print(f"[{field}] {'EXECUTE' if execute else 'DRY RUN'}: "
          f"{len(plan)} stale, {kept} fresh kept "
          f"(campaign floor {fmt(campaign)})")
    log = None
    if execute and plan:
        log = open(f'{pipe}/_stale_rename_{time.strftime("%Y-%m-%d")}.log', 'a')
        log.write(f"# rename_stale_mosaics.py {time.strftime('%Y-%m-%d %H:%M')}\n")
    for f, fm, d in plan:
        line = f"  {fmt(fm)} (band data_i2d {fmt(d)}) {os.path.basename(f)}"
        print(('RENAME' if execute else 'would rename') + line)
        if execute:
            os.rename(f, f + SUFFIX)
            log.write(f"RENAME {f} -> {f}{SUFFIX}\n")
    if log:
        log.close()
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--field', action='append', required=True,
                    help='field dir name under /orange/adamginsburg/jwst '
                         '(repeatable)')
    ap.add_argument('--execute', action='store_true',
                    help='actually rename (default: dry run)')
    ap.add_argument('--campaign-days', type=int, default=21)
    args = ap.parse_args()
    for field in args.field:
        rename_stale_for_field(field, execute=args.execute,
                               campaign_days=args.campaign_days)


if __name__ == '__main__':
    main()
