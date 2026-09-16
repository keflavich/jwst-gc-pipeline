#!/usr/bin/env python
"""Refresh a release's astrometric offsets table and summarise it for the page.

The offsets table is the correction from each exposure's `assign_wcs` pointing
to this pipeline's reference frame. It is NOT frozen at release time: the merge
stages keep re-measuring, and a table that was current when the release was cut
is stale a day later. This script re-copies the live table into the staged
release and writes the summary the web page renders.

It also answers the question the page cannot answer from the table alone:
**which released frames already carry a correction.** `fix_alignment` bakes the
applied shift into `RAOFFSET`/`DEOFFSET` (extension 1) and is idempotent on a
frame that has one, so a release cut while the pipeline was mid-pass holds both
corrected and uncorrected frames. Telling a user to apply the table to all of
them would double-correct most of them.
"""
import argparse
import collections
import json
import os
import shutil
import sys
import warnings

SUMMARY_FILE = 'offsets_summary.json'
DEFAULT_RELEASE = '/orange/adamginsburg/jwst/releases/v1.8-2026.09/gc-treasury'
DEFAULT_TABLE = ('/orange/adamginsburg/jwst/gc-treasury/offsets/'
                 'Offsets_JWST_Brick10678_consensus.csv')


def _has_row(tbl, path):
    """True if the offsets table describes this exposure."""
    from astropy.io import fits
    from jwst_gc_pipeline.reduction.unified_alignment import locked_row_match
    try:
        hdr = fits.getheader(path, 0)
        parts = hdr['FILENAME'].split('_')
        match = locked_row_match(tbl, visit=parts[0], exposure=int(parts[2]),
                                 filtername=hdr['FILTER'],
                                 module=hdr['DETECTOR'].lower(),
                                 vgroup=parts[1])
    except (OSError, KeyError, IndexError, ValueError):
        return False
    return bool(match.sum() == 1)


def scan_frames(exposures_dir, tbl=None):
    """``(per_obs, totals)`` -- how many frames already carry a baked shift.

    Reads extension 1, where `fix_alignment` writes `RAOFFSET`/`DEOFFSET`. A
    frame with a non-zero value has already been corrected by the amount it
    records; one with 0.0 (or no keyword) has not been corrected at all.
    """
    from astropy.io import fits
    per_obs = collections.defaultdict(lambda: collections.Counter())
    unreadable = []
    for root, _, names in os.walk(exposures_dir):
        for name in sorted(names):
            if not name.endswith('.fits'):
                continue
            path = os.path.join(root, name)
            obs = os.path.relpath(path, exposures_dir).split(os.sep)[0]
            try:
                header = fits.getheader(path, 1)
            except (OSError, IndexError, ValueError) as err:
                unreadable.append(f'{path}: {type(err).__name__}: {err}')
                continue
            ra = header.get('RAOFFSET')
            dec = header.get('DEOFFSET')
            baked = bool((ra not in (None, 0.0)) or (dec not in (None, 0.0)))
            per_obs[obs]['corrected' if baked else 'uncorrected'] += 1
            if tbl is not None:
                # Does the table have a row for this exposure YET? The merge
                # stages fill these in as they measure, so a release routinely
                # holds frames the table does not describe. Counted with the
                # pipeline's own matcher rather than a private lookup: the
                # narrowing has cases (Vgroup always, Module/Exposure only when
                # they disambiguate) that a second implementation gets wrong.
                per_obs[obs]['has_row' if _has_row(tbl, path) else 'no_row'] += 1
    totals = collections.Counter()
    for counts in per_obs.values():
        totals.update(counts)
    return dict(per_obs), dict(totals), unreadable


def summarise_table(path):
    """Per-(observation, filter) medians, plus the table's own provenance."""
    import numpy as np
    from astropy.table import Table

    tbl = Table.read(path)
    rows = []
    # `Visit` is jw<prop><obs><vis>; the observation is the tile number the
    # rest of the release is keyed on.
    by = collections.defaultdict(list)
    for row in tbl:
        visit = str(row['Visit'])
        obs = f'o{visit[7:10]}' if len(visit) >= 10 else visit
        by[(obs, str(row['Filter']))].append(
            (float(row['dra (arcsec)']), float(row['ddec (arcsec)']),
             str(row['prov_stage']) if 'prov_stage' in tbl.colnames else ''))
    for (obs, filt), vals in sorted(by.items()):
        dra = np.array([v[0] for v in vals])
        ddec = np.array([v[1] for v in vals])
        stages = sorted({v[2] for v in vals if v[2]})
        rows.append({
            'observation': obs, 'filter': filt, 'n': len(vals),
            'dra_median_arcsec': float(np.median(dra)),
            'ddec_median_arcsec': float(np.median(ddec)),
            'dra_span_mas': float((dra.max() - dra.min()) * 1000),
            'ddec_span_mas': float((ddec.max() - ddec.min()) * 1000),
            'stages': stages,
        })
    return rows, len(tbl)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--release-dir', default=DEFAULT_RELEASE)
    ap.add_argument('--table', default=DEFAULT_TABLE,
                    help='the LIVE offsets table the pipeline is updating')
    ap.add_argument('--no-copy', action='store_true',
                    help='summarise without refreshing the staged copy')
    args = ap.parse_args(argv)

    release = args.release_dir
    astro_dir = os.path.join(release, 'astrometry')
    if not os.path.isdir(release):
        print(f'no staged release at {release}', file=sys.stderr)
        return 1
    if not os.path.isfile(args.table):
        print(f'no offsets table at {args.table}', file=sys.stderr)
        return 1

    os.makedirs(astro_dir, exist_ok=True)
    staged = os.path.join(astro_dir, os.path.basename(args.table))
    if not args.no_copy:
        shutil.copy2(args.table, staged)
        print(f'refreshed {staged}')

    from astropy.table import Table
    tbl = Table.read(staged)
    rows, n_rows = summarise_table(staged)
    per_obs, totals, unreadable = scan_frames(
        os.path.join(release, 'exposures'), tbl=tbl)
    if unreadable:
        print(f'WARNING: {len(unreadable)} frame(s) could not be read for their '
              f'applied-shift state; they are counted in neither column:')
        for line in unreadable[:5]:
            print(f'    {line}')

    summary = {
        'table_file': os.path.basename(staged),
        'table_rows': n_rows,
        'table_mtime': __import__('datetime').datetime.fromtimestamp(
            os.path.getmtime(staged),
            __import__('datetime').timezone.utc).strftime('%Y-%m-%dT%H:%MZ'),
        'per_filter': rows,
        'frames': {'totals': totals, 'per_obs': per_obs,
                   'unreadable': len(unreadable)},
    }
    out = os.path.join(release, SUMMARY_FILE)
    with open(out, 'w') as fh:
        json.dump(summary, fh, indent=1)
    print(f'{n_rows} offset rows over {len(rows)} (observation, filter) pairs')
    print(f'frames: {totals.get("corrected", 0)} already corrected, '
          f'{totals.get("uncorrected", 0)} not; '
          f'{totals.get("has_row", 0)} have a table row, '
          f'{totals.get("no_row", 0)} not measured yet')
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    sys.exit(main())
