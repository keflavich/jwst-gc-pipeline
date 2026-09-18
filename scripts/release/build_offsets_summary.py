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
import re
import sys
import warnings

SUMMARY_FILE = 'offsets_summary.json'

#: Below this, a scatter is not a scatter. o114 F212N has ONE per-exposure row,
#: whose standard deviation is 0.0 -- which the page rendered as "0.0 mas
#: frame-to-frame", i.e. perfect agreement, from a single measurement.
MIN_ROWS_FOR_RMS = 3
DEFAULT_RELEASE = '/orange/adamginsburg/jwst/releases/v1.8-2026.09/gc-treasury'
DEFAULT_TABLE = ('/orange/adamginsburg/jwst/gc-treasury/offsets/'
                 'Offsets_JWST_Brick10678_consensus.csv')


def _row_state(tbl, path):
    """``'has_row'`` / ``'no_row'`` / ``'no_row_not_nircam'`` for one frame.

    The third is its own answer rather than a flavour of the second. A NIRCam
    frame with no row gains one when the merge stages reach it; a MIRI frame
    never will, because this table is NIRCam-only. A page that lumps them
    together promises 198 frames a correction that is not coming.
    """
    from astropy.io import fits
    from jwst_gc_pipeline.reduction.unified_alignment import locked_row_match
    try:
        hdr = fits.getheader(path, 0)
        parts = hdr['FILENAME'].split('_')
        if len(parts) < 4 or not parts[2].isdigit():
            # Association-style FILENAME (jw10678-o132_t001_miri_f770w_2_...),
            # which is what every MIRI frame here carries.
            return 'no_row_not_nircam'
        match = locked_row_match(tbl, visit=parts[0], exposure=int(parts[2]),
                                 filtername=hdr['FILTER'],
                                 module=hdr['DETECTOR'].lower(),
                                 vgroup=parts[1])
    except (OSError, KeyError, IndexError, ValueError):
        return 'no_row'
    return 'has_row' if match.sum() == 1 else 'no_row'


#: `o123`, as the release stages its directories.
_OBS_DIR_RE = re.compile(r'^o\d+$')


def _obs_of_path(path, exposures_dir):
    """The observation a staged frame belongs to, from its path.

    NIRCam stages as ``exposures/o127/F212N/`` and MIRI as
    ``exposures/MIRI/o132/F770W/``, so the first path segment is the
    observation for one instrument and the instrument name for the other.
    Taking the first segment reported every MIRI frame under an observation
    called ``MIRI``, which matches no row in the offsets table and no
    observation in the release: the frames counted, and then had no
    denominator to be counted against.
    """
    parts = os.path.relpath(path, exposures_dir).split(os.sep)
    for part in parts[:-1]:
        if _OBS_DIR_RE.match(part):
            return part
    return parts[0]


def scan_frames(exposures_dir, tbl=None):
    """``(per_obs, totals)`` -- how many frames already carry a baked shift.

    Reads extension 1, where `fix_alignment` writes `RAOFFSET`/`DEOFFSET`. A
    frame with a non-zero value has already been corrected by the amount it
    records; one with 0.0 (or no keyword) has not been corrected at all.
    """
    from astropy.io import fits
    per_obs = collections.defaultdict(lambda: collections.Counter())
    per_key = {}
    unreadable = []
    for root, _, names in os.walk(exposures_dir):
        for name in sorted(names):
            if not name.endswith('.fits'):
                continue
            path = os.path.join(root, name)
            obs = _obs_of_path(path, exposures_dir)
            try:
                header = fits.getheader(path, 1)
            except (OSError, IndexError, ValueError) as err:
                unreadable.append(f'{path}: {type(err).__name__}: {err}')
                continue
            ra = header.get('RAOFFSET')
            dec = header.get('DEOFFSET')
            baked = bool((ra not in (None, 0.0)) or (dec not in (None, 0.0)))
            per_obs[obs]['corrected' if baked else 'uncorrected'] += 1
            filt = os.path.basename(os.path.dirname(path))
            per_key[(obs, filt)] = per_key.get((obs, filt), 0) + 1
            if tbl is not None:
                # Does the table have a row for this exposure YET? The merge
                # stages fill these in as they measure, so a release routinely
                # holds frames the table does not describe. Counted with the
                # pipeline's own matcher rather than a private lookup: the
                # narrowing has cases (Vgroup always, Module/Exposure only when
                # they disambiguate) that a second implementation gets wrong.
                state = _row_state(tbl, path)
                per_obs[obs][state] += 1
                if state == 'no_row_not_nircam':
                    # Counted as no_row too, so has_row + no_row still covers
                    # every frame; the finer key says WHY, and the page needs
                    # the difference: a NIRCam frame without a row gains one
                    # when the stages reach it, a MIRI frame never does.
                    per_obs[obs]['no_row'] += 1
    totals = collections.Counter()
    for counts in per_obs.values():
        totals.update(counts)
    return dict(per_obs), dict(totals), unreadable, per_key


def summarise_table(path, frames=None):
    """Split the table into the two quantities it holds, which are not the same
    measurement and must never be pooled.

    * ``m2 consensus->reference`` rows (``Module='all'``, ``Exposure=-1``) are
      the BULK TIE: how far the visit's whole pointing sits from the reference
      frame. This is "the offset" -- what a user asking how wrong the
      astrometry is wants. 70 to 500 mas here.
    * ``m2 visit-consensus`` rows (one per detector per exposure) are RESIDUALS
      about that visit consensus: frame-to-frame scatter, a few mas.

    Taking a median over both reports ~5 mas, because the 878 residuals swamp
    the 20 bulk rows -- a number two orders of magnitude below the real offset,
    on a page telling people how to correct their astrometry. That is what this
    function did until 2026-09-17.
    """
    import numpy as np
    from astropy.table import Table

    tbl = Table.read(path)
    frames = frames or {}
    # The offsets table covers the PROGRAMME; a release stages part of it. An
    # observation with no staged frames gets no denominator, and saying so is
    # different from reporting it as zero coverage.
    staged_obs = {obs for obs, _filt in frames}
    have_source = 'prov_source' in tbl.colnames

    def obs_of(visit):
        visit = str(visit)
        return f'o{visit[7:10]}' if len(visit) >= 10 else visit

    bulk, resid = {}, collections.defaultdict(list)
    for row in tbl:
        key = (obs_of(row['Visit']), str(row['Filter']))
        source = str(row['prov_source']) if have_source else ''
        dra, ddec = float(row['dra (arcsec)']), float(row['ddec (arcsec)'])
        if source == 'm2 consensus->reference' or str(row['Module']) == 'all':
            bulk[key] = (dra, ddec, source)
        else:
            resid[key].append((dra, ddec))

    rows = []
    for key in sorted(set(bulk) | set(resid)):
        obs, filt = key
        scatter = resid.get(key, [])
        entry = {'observation': obs, 'filter': filt,
                 'n_exposure_rows': len(scatter),
                 'n_frames': frames.get(key),
                 'in_release': (obs in staged_obs) if staged_obs else None}
        if len(scatter) >= MIN_ROWS_FOR_RMS:
            dra = np.array([s[0] for s in scatter])
            ddec = np.array([s[1] for s in scatter])
            # ddof=1: these are a SAMPLE of the frames of one visit, and at
            # the three-row floor the population form reads ~18% low -- in the
            # direction the floor exists to guard against.
            entry['residual_rms_mas'] = float(
                np.hypot(np.std(dra, ddof=1), np.std(ddec, ddof=1)) * 1000)
        if key in bulk:
            dra, ddec, source = bulk[key]
            entry.update({
                'dra_arcsec': dra, 'ddec_arcsec': ddec,
                'total_mas': float(np.hypot(dra, ddec) * 1000),
                'source': source or 'consensus->reference'})
        rows.append(entry)
    measured = sum(1 for r in rows if 'total_mas' in r)
    return rows, len(tbl), measured


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
    per_obs, totals, unreadable, per_key = scan_frames(
        os.path.join(release, 'exposures'), tbl=tbl)
    rows, n_rows, n_measured = summarise_table(staged, frames=per_key)
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
        'n_measured_ties': n_measured,
        'frames': {'totals': totals, 'per_obs': per_obs,
                   'unreadable': len(unreadable)},
    }
    out = os.path.join(release, SUMMARY_FILE)
    with open(out, 'w') as fh:
        json.dump(summary, fh, indent=1)
    print(f'{n_rows} offset rows over {len(rows)} (observation, filter) pairs; '
          f'{n_measured} have a measured tie to the reference frame, '
          f'{len(rows) - n_measured} have only per-exposure residuals')
    print(f'frames: {totals.get("corrected", 0)} already corrected, '
          f'{totals.get("uncorrected", 0)} not; '
          f'{totals.get("has_row", 0)} have a table row, '
          f'{totals.get("no_row", 0)} not ('
          f'{totals.get("no_row_not_nircam", 0)} of those are MIRI, which this '
          f'NIRCam-only table will never describe)')
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    sys.exit(main())
