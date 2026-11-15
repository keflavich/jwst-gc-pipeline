#!/usr/bin/env python
"""Cross-exposure audit of existing m2 merged catalogs (read-only).

At the m12 finalize the pipeline adds ``ncover``, ``nfound`` and
``single_exposure`` to the m2 merged catalog (``photometry/xexp_reconcile.py``):
for every merged source, how many of the merge's exposures cover its position,
how many carry it, and whether it is a bright or saturated source that two or
more exposures cover and exactly one carries.  This script gives the same
accounting for catalogs that exist already, from the catalog's own ``fn<i>``
cards, the per-frame catalogs, the crfs' DQ planes and the satstar catalogs, so
fields and filters can be compared without rerunning a phase.

Nothing under a product tree is written.  Catalogs are read; the audit columns
are added to an in-memory copy; the reports go to ``--out``, which must not be a
catalog directory, and an existing report is not replaced unless ``--overwrite``
is given.  A catalog that already carries the three columns is summarised from
them (``source = column``, using the S/N and rank cuts recorded in its header, not
``--single-snr`` / ``--single-top-pct``); ``--recompute`` ignores them.

Usage::

    python xexp_audit.py --out DIR CATALOG [CATALOG ...]
    python xexp_audit.py --out DIR --catalog-list LIST --shard 3/40
    python xexp_audit.py --out DIR --collect

``CATALOG`` is ``<root>/catalogs/<filt>_..._indivexp_merged_m2_dao_basic.fits``;
its per-frame catalogs are looked up in ``<root>/<FILT>`` (``--catalog-dir``
overrides).  Each catalog costs one WCS load per exposure (about 16 s cold, 48
exposures for an eight-detector module), so a broad run is one catalog per task
of a SLURM array (``--account=astronomy-dept --qos=astronomy-dept-b``), each
task passing ``--shard $SLURM_ARRAY_TASK_ID/N``; ``--collect`` then merges the
per-shard ``summary.<tag>.tsv`` files into ``summary.tsv`` and ``report.md``.

Reading the numbers.  ``multi`` counts the rows that two or more exposures
cover; those are split by how many of the covering exposures carry the row:
``all``, ``some`` (more than one, not all), ``one`` and ``none``.  A row in
``one`` is not necessarily wrong: an exposure can lack a real star (a masked
neighbourhood, a cosmic-ray snowball beside it, a NaN core), and the merge
keeps it.  ``nfound_eq_nmatch`` compares ``nfound`` with the merge's own
``nmatch`` over the multi-covered rows; they are not the same quantity (the
merge matches on position within its own tolerance and does not look at the
other exposures' coverage).
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io.registry import IORegistryError
from astropy.table import Table, vstack

from jwst_gc_pipeline.photometry import xexp_reconcile as xexp
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS

AUDIT_COLUMNS = ('ncover', 'nfound', 'single_exposure')
SUMMARY_COLUMNS = (
    'catalog', 'filter', 'source', 'rows', 'frames_read', 'frames_skipped',
    'frames_newer',
    'multi', 'all', 'some', 'one', 'none', 'frac_one',
    'bright_multi', 'bright_all', 'bright_some', 'bright_one', 'bright_none',
    'bright_frac_one', 'single_exposure',
    'nfound_eq_nmatch', 'nfound_gt_nmatch', 'nfound_lt_nmatch', 'snr_min',
    'top_pct', 'path')
TEXT_COLUMNS = ('catalog', 'filter', 'source', 'path')
RATIO_COLUMNS = ('frac_one', 'bright_frac_one', 'nfound_eq_nmatch',
                 'nfound_gt_nmatch', 'nfound_lt_nmatch', 'snr_min', 'top_pct')
_FILTER = re.compile(r'(f\d{3,4}[wmn])_', re.IGNORECASE)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def filter_from_name(path):
    """``F212N`` from ``f212n_merged_o078_indivexp_merged_m2_dao_basic.fits``."""
    m = _FILTER.match(os.path.basename(path))
    if m is None:
        raise ValueError(f'{path}: the catalog name does not start with a filter')
    return m.group(1).upper()


def default_catalog_dirs(path, filt):
    """Per-frame catalogs sit in ``<root>/<FILT>``, the merged one in
    ``<root>/catalogs``."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    return [os.path.join(root, filt)]


def select_shard(items, spec):
    """``items[i::n]`` for ``spec`` = ``'i/n'`` (0-based); all items for None."""
    if spec is None:
        return list(items)
    m = re.fullmatch(r'(\d+)/(\d+)', spec)
    if m is None or not 0 <= int(m.group(1)) < int(m.group(2)):
        raise ValueError(f'--shard {spec!r}: expected I/N with 0 <= I < N')
    return list(items)[int(m.group(1))::int(m.group(2))]


def read_merged(path):
    """The merged catalog as a Table.  A file that is not FITS raises
    ``IORegistryError``; it becomes ValueError, as a table cut short in its data
    already is, so a broad run can record the catalog and go on."""
    try:
        return Table.read(path)
    except IORegistryError as exc:
        raise ValueError(f'{path}: cannot read ({type(exc).__name__}: {exc})') from exc


def flagged_name(path):
    return 'flagged.' + os.path.splitext(os.path.basename(path))[0] + '.tsv'


def refuse_existing(targets, overwrite):
    """Fail before any work if a report would be replaced."""
    clash = [t for t in targets if os.path.exists(t)]
    if clash and not overwrite:
        raise FileExistsError(
            f'{len(clash)} report(s) exist, e.g. {clash[0]}; pass --overwrite to '
            'replace them')


def _ratio(num, den):
    return float(num) / float(den) if den else float('nan')


def _float_column(table, name):
    return np.ma.filled(table[name].astype(float), np.nan)


def flagged_table(table, single, snr_min, n):
    """The ``n`` brightest ``single_exposure`` rows."""
    cols = ('ra', 'dec', 'flux', 'flux_err', 'snr', 'ncover', 'nfound', 'nmatch',
            'saturated')
    idx = np.flatnonzero(single)
    flux = _float_column(table, 'flux')
    idx = idx[np.argsort(-flux[idx])][:n]
    if idx.size == 0:
        return Table(names=cols, dtype=[float] * 5 + [int] * 3 + [bool])
    sc = SkyCoord(table['skycoord'][idx])
    err = _float_column(table, 'flux_err')[idx]
    out = Table({'ra': sc.ra.deg, 'dec': sc.dec.deg, 'flux': flux[idx],
                 'flux_err': err, 'snr': flux[idx] / err,
                 'ncover': np.asarray(table['ncover'])[idx],
                 'nfound': np.asarray(table['nfound'])[idx]})
    out['nmatch'] = (np.asarray(table['nmatch'])[idx] if 'nmatch' in table.colnames
                     else -1)
    out['saturated'] = xexp.saturated_rows(table)[idx]
    return out


def audit_catalog_file(path, *, wcs_loader, bad_dq_mask, catalog_dirs=None,
                       satstar_suffix='_m12', params=None, recompute=False,
                       n_flagged=0, log=print):
    """``(summary, frames, flagged)`` for one merged m2 catalog.

    ``summary`` is a dict keyed by ``SUMMARY_COLUMNS``; ``frames`` is the
    per-frame accounting (None when the numbers come from stored columns);
    ``flagged`` holds the ``n_flagged`` brightest ``single_exposure`` rows.
    Reads ``path`` and the per-frame products; writes nothing.
    """
    params = params or xexp.XexpParams()
    filt = filter_from_name(path)
    table = read_merged(path)
    frames = None
    newer = -1
    stored = all(c in table.colnames for c in AUDIT_COLUMNS)
    if stored and not recompute:
        source = 'column'
        snr_min = float(table.meta.get('XEXPSNR', params.single_snr_min))
        # a table audited before the rank cut existed was flagged without one
        top_pct = float(table.meta.get('XEXPTOP', 100.0))
        read = int(table.meta.get('XEXPFRM', -1))
        skipped = int(table.meta.get('XEXPSKP', -1))
    else:
        dirs = list(catalog_dirs) if catalog_dirs else default_catalog_dirs(path, filt)
        result = xexp.audit_merged_catalog(
            table, filt=filt, catalog_dirs=dirs, satstar_suffix=satstar_suffix,
            params=params, wcs_loader=wcs_loader, bad_dq_mask=bad_dq_mask, log=log)
        snr_min, top_pct = params.single_snr_min, params.single_top_pct
        if result is None:
            source, read, skipped = 'no-fn-cards', 0, 0
            table['ncover'] = np.zeros(len(table), np.int16)
            table['nfound'] = np.zeros(len(table), np.int16)
            table['single_exposure'] = np.zeros(len(table), bool)
        else:
            source = 'recomputed'
            read = int(table.meta['XEXPFRM'])
            skipped = int(table.meta['XEXPSKP'])
            frames = result.frames.copy()
            frames.add_column(os.path.basename(path), name='catalog', index=0)
            newer = xexp.frames_newer_than(
                xexp.resolve_frames(xexp.frame_catalog_names(table.meta),
                                    catalog_dirs=dirs, filt=filt,
                                    satstar_suffix=satstar_suffix), path)

    ncover = np.asarray(table['ncover'])
    nfound = np.asarray(table['nfound'])
    single = np.asarray(table['single_exposure'], dtype=bool)
    bright = xexp.bright_rows(_float_column(table, 'flux'),
                              _float_column(table, 'flux_err'),
                              xexp.saturated_rows(table), snr_min, top_pct,
                              multi=ncover >= 2)
    summary = xexp.audit_summary(ncover, nfound, single, bright)
    multi = ncover >= 2
    nan = float('nan')
    summary.update(
        catalog=os.path.basename(path), filter=filt, source=source,
        frames_read=read, frames_skipped=skipped, frames_newer=newer,
        frac_one=_ratio(summary['one'], summary['multi']),
        bright_frac_one=_ratio(summary['bright_one'], summary['bright_multi']),
        nfound_eq_nmatch=nan, nfound_gt_nmatch=nan, nfound_lt_nmatch=nan,
        snr_min=snr_min, top_pct=top_pct, path=os.path.abspath(path))
    if 'nmatch' in table.colnames:
        nmatch = np.asarray(table['nmatch'])
        for key, sel in (('nfound_eq_nmatch', nfound == nmatch),
                         ('nfound_gt_nmatch', nfound > nmatch),
                         ('nfound_lt_nmatch', nfound < nmatch)):
            summary[key] = _ratio((multi & sel).sum(), summary['multi'])
    return summary, frames, flagged_table(table, single, snr_min, n_flagged)


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------

def write_table(table, path, overwrite):
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f'{path} exists; pass --overwrite to replace it')
    table.write(path, format='ascii.tab', overwrite=overwrite)


def write_text(text, path, overwrite):
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f'{path} exists; pass --overwrite to replace it')
    with open(path, 'w') as fh:
        fh.write(text)


def summary_table(rows):
    """One row per catalog; a catalog that could not be read has counts of -1
    and fractions of NaN."""
    cols = {}
    for name in SUMMARY_COLUMNS:
        vals = [r.get(name) for r in rows]
        if name in TEXT_COLUMNS:
            cols[name] = [str(v) if v is not None else '' for v in vals]
        elif name in RATIO_COLUMNS:
            cols[name] = np.array([np.nan if v is None else float(v) for v in vals])
        else:
            cols[name] = np.array([-1 if v is None else int(v) for v in vals],
                                  dtype=np.int64)
    return Table(cols)


def _pct(x):
    return 'n/a' if not np.isfinite(x) else f'{100.0 * x:.1f}%'


def render_report(summary):
    """Markdown table of a summary table."""
    lines = [
        '# Cross-exposure audit of m2 merged catalogs', '',
        'One row per merged catalog. `multi` = rows covered by two or more '
        'exposures; `all` / `some` / `one` / `none` split them by how many of '
        'the covering exposures carry the row. `bright` = saturated, or S/N at '
        'or above `snr_min` and merged flux in the brightest `top_pct` percent '
        'of the multi rows. `flagged` = bright rows carried by exactly one of '
        'two or more covering exposures (`single_exposure`).', '',
        '| catalog | source | rows | frames read/skipped | frames newer '
        '| multi | all | some | one | none | bright multi | bright one '
        '| flagged |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if str(r['source']).startswith('unreadable'):
            lines.append(f"| {r['catalog']} | {r['source']} |" + ' |' * 11)
            continue
        multi = int(r['multi'])

        def share(key):
            return f'{int(r[key])} ({_pct(_ratio(r[key], multi))})'

        lines.append(
            f"| {r['catalog']} | {r['source']} | {int(r['rows'])} "
            f"| {int(r['frames_read'])}/{int(r['frames_skipped'])} "
            f"| {'n/a' if int(r['frames_newer']) < 0 else int(r['frames_newer'])} "
            f"| {multi} "
            f"| {share('all')} | {share('some')} | {share('one')} | {share('none')} "
            f"| {int(r['bright_multi'])} "
            f"| {int(r['bright_one'])} ({_pct(r['bright_frac_one'])}) "
            f"| {int(r['single_exposure'])} |")
    stale = [r['catalog'] for r in summary if int(r['frames_newer']) > 0]
    if stale:
        lines += ['', f'**Different runs:** {len(stale)} catalog(s) were recomputed '
                  'against per-frame catalogs written after the merged catalog '
                  '(`frames newer` above zero), so the counts compare two runs of '
                  'the pipeline and `none` reads high: '
                  + ', '.join(f'`{c}`' for c in stale) + '.']
    return '\n'.join(lines) + '\n'


def collect(out, overwrite):
    """Merge ``summary.<tag>.tsv`` files of ``out`` into ``summary.tsv`` and
    ``report.md``."""
    parts = sorted(glob.glob(os.path.join(out, 'summary.*.tsv')))
    if not parts:
        raise FileNotFoundError(f'no summary.<tag>.tsv files in {out}')
    refuse_existing([os.path.join(out, 'summary.tsv'), os.path.join(out, 'report.md')],
                    overwrite)
    merged = vstack([Table.read(p, format='ascii.tab') for p in parts])
    merged.sort(['filter', 'catalog'])
    write_table(merged, os.path.join(out, 'summary.tsv'), overwrite)
    write_text(render_report(merged), os.path.join(out, 'report.md'), overwrite)
    return merged


def refuse_product_dirs(out, paths, catalog_dirs):
    """``--out`` must not be a directory that holds the catalogs it reads."""
    here = os.path.realpath(out)
    dirs = {os.path.realpath(os.path.dirname(os.path.abspath(p))) for p in paths}
    for p in paths:
        dirs.update(os.path.realpath(d) for d in (
            catalog_dirs or default_catalog_dirs(p, filter_from_name(p))))
    if here in dirs:
        raise ValueError(f'--out {out} is a product directory; use a separate '
                         'report directory')


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        description='Read-only cross-exposure audit of m2 merged catalogs.')
    ap.add_argument('catalogs', nargs='*',
                    help='merged m2 catalogs (…_indivexp_merged_m2_dao_basic.fits)')
    ap.add_argument('--catalog-list', help='text file with one catalog per line')
    ap.add_argument('--shard', help='I/N: every N-th catalog from the sorted '
                    'list, starting at I (0-based); for SLURM arrays')
    ap.add_argument('--out', required=True,
                    help='report directory (never a product directory)')
    ap.add_argument('--catalog-dir', action='append',
                    help='directory of the per-frame catalogs (repeatable; '
                    'default <root>/<FILTER>)')
    ap.add_argument('--satstar-suffix', default='_m12',
                    help='suffix of the satstar catalogs (default %(default)s)')
    ap.add_argument('--single-snr', type=float,
                    default=MANUAL_DEFAULTS['manual_xexp_single_snr'],
                    help='S/N above which a row is bright (default %(default)s)')
    ap.add_argument('--single-top-pct', type=float,
                    default=MANUAL_DEFAULTS['manual_xexp_single_top_pct'],
                    help='a bright row is also in the brightest this-many percent '
                    'of the multi-covered rows by merged flux; 100 = no rank cut '
                    '(default %(default)s)')
    ap.add_argument('--recompute', action='store_true',
                    help='ignore audit columns the catalog already carries')
    ap.add_argument('--flagged', type=int, default=0, metavar='N',
                    help='also write the N brightest single_exposure rows of '
                    'each catalog')
    ap.add_argument('--overwrite', action='store_true',
                    help='replace reports that exist in --out')
    ap.add_argument('--collect', action='store_true',
                    help='merge summary.<tag>.tsv in --out into summary.tsv and '
                    'report.md, then exit')
    return ap


def main(argv=None, *, wcs_loader=None, bad_dq_mask=None):
    """``wcs_loader`` and ``bad_dq_mask`` are for tests; the command line uses
    the pipeline's ``frame_wcs`` and the instrument's bad-pixel bitmask."""
    args = build_parser().parse_args(argv)
    out = os.path.abspath(args.out)
    if args.collect:
        collect(out, args.overwrite)
        print(f'wrote {out}/summary.tsv and report.md')
        return 0
    params = xexp.XexpParams(single_snr_min=args.single_snr,
                             single_top_pct=args.single_top_pct)
    paths = list(args.catalogs)
    if args.catalog_list:
        with open(args.catalog_list) as fh:
            paths += [ln.strip() for ln in fh
                      if ln.strip() and not ln.lstrip().startswith('#')]
    paths = select_shard(sorted(set(paths)), args.shard)
    if not paths:
        print('no catalogs to audit', file=sys.stderr)
        return 1
    filters = {p: filter_from_name(p) for p in paths}
    refuse_product_dirs(out, paths, args.catalog_dir)
    if args.shard is None:
        tag = 'all'
    else:
        i, n = (int(x) for x in args.shard.split('/'))
        tag = f'shard{i:03d}of{n:03d}'
    targets = [os.path.join(out, f'summary.{tag}.tsv'),
               os.path.join(out, f'frames.{tag}.tsv')]
    if args.flagged:
        targets += [os.path.join(out, flagged_name(p)) for p in paths]
    refuse_existing(targets, args.overwrite)
    os.makedirs(out, exist_ok=True)
    if wcs_loader is None:
        from jwst_gc_pipeline.frame_wcs import frame_wcs as wcs_loader

    rows, frame_tables = [], []
    for path in paths:
        t0 = time.time()
        filt = filters[path]
        mask = bad_dq_mask if bad_dq_mask is not None else xexp.bad_dq_mask_for(filt)
        try:
            summary, frames, flagged = audit_catalog_file(
                path, wcs_loader=wcs_loader, bad_dq_mask=mask,
                catalog_dirs=args.catalog_dir, satstar_suffix=args.satstar_suffix,
                params=params, recompute=args.recompute, n_flagged=args.flagged)
        except (OSError, ValueError, KeyError) as exc:
            print(f'{path}: UNREADABLE ({type(exc).__name__}: {exc})', flush=True)
            rows.append({'catalog': os.path.basename(path), 'filter': filt,
                         'source': f'unreadable: {type(exc).__name__}',
                         'path': os.path.abspath(path)})
            continue
        rows.append(summary)
        if frames is not None and len(frames):
            frame_tables.append(frames)
        if args.flagged:
            write_table(flagged, os.path.join(out, flagged_name(path)), args.overwrite)
        print(f"{summary['catalog']}: {summary['source']}, {summary['rows']} rows, "
              f"multi {summary['multi']}, one {summary['one']}, flagged "
              f"{summary['single_exposure']} ({time.time() - t0:.0f} s)", flush=True)

    write_table(summary_table(rows), os.path.join(out, f'summary.{tag}.tsv'),
                args.overwrite)
    if frame_tables:
        write_table(vstack(frame_tables), os.path.join(out, f'frames.{tag}.tsv'),
                    args.overwrite)
    return 0


if __name__ == '__main__':
    sys.exit(main())
