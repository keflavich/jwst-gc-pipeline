#!/usr/bin/env python3
"""Add ``oksep_pass`` / ``cmd_default`` to existing m7, m8 and m8_dedup tables (#1283).

New runs write the two columns themselves (``cmd_default.annotate_cmd_default``,
called from ``cataloging._maybe_dedup_m8``, the m7 post-annotation block and
``m8_merge_partials.py``).  This script brings tables that are already on disk
up to the same state without rerunning any photometry.  It reads and rewrites
tables only.

Selected from each ``--catdir``: ``basic_*_m7[_o<obs>]``, ``basic_*_m8[_o<obs>]``
and ``basic_*_m8_dedup[_o<obs>]`` tables, plus the ``_qualcuts*`` subsets of
the m7 tables.  Partial (``_m8_partial_<band>``) and dated-copy files are
skipped.  Each rewrite keeps the primary header and the table header cards
(``GCTAG``, ``ROLLC*``), so a roll-corrected snapshot keeps its correction
stamp.  When the table has a ``.prov.json`` sidecar, its output facets are
refreshed in place with the recorded tag, stage and inputs unchanged.

Dry-run by default: reports what would change.  ``--apply`` writes.  Run it on
every directory a downstream reader uses, e.g. the field ``catalogs/`` and any
``catalogs_rollcorr/<tag>/`` snapshot the joint catalogue is built from.

    python scripts/reduction/add_cmd_default_flag.py --catdir <dir> [--apply]
"""
import argparse
import glob
import os
import re
import sys

from astropy.io import fits

TABLE_RE = re.compile(
    r'^basic_.*_(?P<stage>m7|m8|m8_dedup)(?:_o\d+(?:-\d+)?)?'
    r'(?:_qualcuts[A-Za-z0-9_-]*)?\.fits$')


def select_tables(catdirs):
    """Paths of the tables in ``catdirs`` that should carry the flag."""
    out = []
    for d in catdirs:
        for p in sorted(glob.glob(os.path.join(d, 'basic_*.fits'))):
            if TABLE_RE.match(os.path.basename(p)):
                out.append(p)
    return out


def _has_flag(path):
    """``(has both columns, CMDDEF value or None)`` from the table header."""
    with fits.open(path, memmap=True) as hdul:
        names = set(hdul[1].columns.names)
        card = hdul[1].header.get('CMDDEF')
    return {'oksep_pass', 'cmd_default'} <= names, card


def refresh_sidecar(path):
    """Recompute the output facets in ``<path>.prov.json``; keep everything else.

    Returns True when a sidecar existed and was rewritten.  The record's tag,
    stage and inputs stay as they were: the flag columns are an annotation of
    the same run, not a new run.
    """
    from jwst_gc_pipeline.versioning import fingerprint as _fp
    from jwst_gc_pipeline.versioning import prov_sidecar
    from jwst_gc_pipeline.versioning import stamping
    rec = prov_sidecar.read_sidecar(path)
    if rec is None:
        return False
    with fits.open(path, memmap=True) as hdul:
        meta = _fp.meta_hash(hdul[0].header)
    facets = {'data': _fp.table_hash(path, exclude_cols=(
        'ra', 'dec', 'skycoord_ra', 'skycoord_dec')),
        'wcs': None, 'meta': meta}
    inputs = rec.get('inputs') or {}
    new = prov_sidecar.build_record(
        rec['stage'], rec['tag'], facets, env=inputs.get('env'),
        code=inputs.get('code'), params=inputs.get('params'),
        upstream=inputs.get('upstream'), extra=dict(
            rec.get('extra') or {}, cmd_default_backfill=True))
    prov_sidecar.write_sidecar(path, new)
    stamping._mirror_keys(path, rec['stage'], facets)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--catdir', action='append', required=True,
                    help='catalogue directory (repeatable)')
    ap.add_argument('--apply', action='store_true',
                    help='write the columns (default: dry-run)')
    ap.add_argument('--force', action='store_true',
                    help='also rewrite tables that already carry both columns')
    args = ap.parse_args(argv)

    from jwst_gc_pipeline.photometry.cmd_default import annotate_cmd_default

    tables = select_tables(args.catdir)
    if not tables:
        print('no m7/m8/m8_dedup tables found', file=sys.stderr)
        return 1
    todo = 0
    for p in tables:
        done, card = _has_flag(p)
        if done and not args.force:
            print(f'SKIP (already flagged, CMDDEF={card}): {os.path.basename(p)}')
            continue
        todo += 1
        if not args.apply:
            print(f'[DRY] would flag {os.path.basename(p)}')
            continue
        annotate_cmd_default(p)
        if refresh_sidecar(p):
            print(f'  refreshed {os.path.basename(p)}.prov.json')
    if not args.apply:
        print(f'\n[DRY] {todo} of {len(tables)} table(s) need the flag; '
              f're-run with --apply to write')
    return 0


if __name__ == '__main__':
    sys.exit(main())
