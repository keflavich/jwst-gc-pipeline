"""Assemble a CMZ-wide catalog from per-field cross-band catalogs.

All catalog merging in the pipeline is WITHIN a field; there is no CMZ-wide
table.  This builds one: vstack the per-field combined (``resbgsub_m7``) catalogs
with a column union, add per-source provenance (which field / program / obsid /
pipeline tag it came from), de-duplicate sources that fall in a field-overlap
region, and record a per-source filter-coverage count.

Deliberately schema-flexible: the per-filter column names are discovered at
runtime (they differ field to field).  A field table only needs RA/Dec columns;
every other column is carried through, and the column union fills gaps with
masked values.  ``coverage_cols`` (or the auto-detected flux/mag columns) drive the
coverage count and the dedup "keep the better detection" rule.
"""
import glob as _glob
import os

import numpy as np
import scipy.sparse as _sparse
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from scipy.sparse.csgraph import connected_components

# Provenance columns added to every source.
PROV_COLS = ('cmz_field', 'cmz_program', 'cmz_obsid', 'cmz_src_tag')
_RA_CANDIDATES = ('ra', 'RA', 'skycoord_ra', 'skycoord_centroid_ra', 'ra_deg')
_DEC_CANDIDATES = ('dec', 'DEC', 'skycoord_dec', 'skycoord_centroid_dec', 'dec_deg')


def _find_col(table, candidates):
    lower = {c.lower(): c for c in table.colnames}
    for c in candidates:
        if c in table.colnames:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _skycoord(table, ra_col=None, dec_col=None):
    """Build a SkyCoord from a table, preferring a ``skycoord`` mixin column."""
    for c in table.colnames:
        if table[c].__class__.__name__ == 'SkyCoord':
            return SkyCoord(table[c])
    ra_col = ra_col or _find_col(table, _RA_CANDIDATES)
    dec_col = dec_col or _find_col(table, _DEC_CANDIDATES)
    if ra_col is None or dec_col is None:
        raise KeyError(f'no RA/Dec columns found (looked for {_RA_CANDIDATES} / '
                       f'{_DEC_CANDIDATES}); pass ra_col/dec_col')
    return SkyCoord(np.asarray(table[ra_col], float) * u.deg,
                    np.asarray(table[dec_col], float) * u.deg)


def _coverage_cols(table, coverage_cols=None):
    """Columns whose finiteness counts as 'measured in a band'."""
    if coverage_cols is not None:
        return [c for c in coverage_cols if c in table.colnames]
    # auto: flux/mag columns, excluding error columns.
    out = []
    for c in table.colnames:
        cl = c.lower()
        if ('flux' in cl or 'mag' in cl) and not (
                cl.startswith('e') or 'err' in cl or cl.startswith('d')):
            if np.issubdtype(np.asarray(table[c]).dtype, np.number):
                out.append(c)
    return out


def _coverage_count(table, cols):
    if not cols:
        return np.zeros(len(table), dtype=int)
    n = np.zeros(len(table), dtype=int)
    for c in cols:
        col = table[c]
        mask = np.ma.getmaskarray(col)          # outer-join gaps are masked
        vals = np.asarray(np.ma.getdata(col), float)
        n += (np.isfinite(vals) & ~mask).astype(int)
    return n


def load_field_catalog(path, field, program='', obsid='', tag=None,
                       ra_col=None, dec_col=None):
    """Load one field's combined catalog and stamp provenance columns.

    ``tag`` defaults to the catalog's recorded ``GCTAG`` meta (the pipeline tag
    that produced it), else ``'unknown'`` -- so the CMZ table records which
    tagged run each field came from.
    """
    t = Table.read(path)
    # validate coordinates are resolvable (raises early with a clear message)
    _skycoord(t, ra_col=ra_col, dec_col=dec_col)
    n = len(t)
    t['cmz_field'] = np.array([field] * n)
    t['cmz_program'] = np.array([str(program)] * n)
    t['cmz_obsid'] = np.array([str(obsid)] * n)
    src_tag = tag if tag is not None else str(t.meta.get('GCTAG', 'unknown'))
    t['cmz_src_tag'] = np.array([src_tag] * n)
    return t


def _dedup_cross_field(table, radius_arcsec, coverage_cols, field_col='cmz_field',
                       rank_cols=None):
    """Collapse cross-FIELD duplicate sources; keep + record the dropped fields.

    Only pairs from DIFFERENT groups (``field_col``, default ``'cmz_field'`` --
    pass e.g. ``'cmz_obsid'`` to group by TILE instead) within ``radius_arcsec``
    are merged.  A close pair within the SAME group is a real blend the
    per-group pipeline already resolved and is never merged.

    Vectorized (2026-10, issue: the original pure-Python union-find + ``for root
    in np.unique(roots): np.where(roots == root)`` scan was O(N^2) -- at 10678's
    ~9.2M-row scale that never finished).  The pipeline is now:
    ``search_around_sky`` -> keep only cross-group pairs -> build a sparse graph
    on those pairs only -> ``scipy.sparse.csgraph.connected_components`` ->
    choose each component's winner with ONE global ``np.lexsort`` (no python
    loop over all N components).  Only clusters that actually merge (a small
    fraction of N) are then visited, once each, to assemble the ``also_in``
    string.

    ``rank_cols`` : tuple of column names in ``table``, most-important FIRST,
    each **ascending == more preferred** (precompute e.g. ``-n_bands`` if you
    want "most bands wins").  Defaults to ``None``, which ranks by (most finite
    coverage bands, lowest row index) -- the original behaviour.  Row index is
    always appended as the final, deterministic tie-break.

    Returns ``(keep, also_in)``: ``keep`` is the boolean survivor mask; ``also_in``
    is a per-row string listing the OTHER groups the kept source was also detected
    in (comma-joined, sorted) -- so an overlap-region source stays recoverable
    (which groups saw it) instead of the duplicate detection being silently lost.
    """
    n = len(table)
    keep = np.ones(n, dtype=bool)
    also_in = np.array([''] * n, dtype=object)
    if n < 2 or radius_arcsec <= 0:
        return keep, also_in
    sc = _skycoord(table)
    fields = np.asarray(table[field_col]).astype(str)
    idx = np.arange(n)

    i1, i2, _, _ = sc.search_around_sky(sc, radius_arcsec * u.arcsec)
    i1 = np.asarray(i1, dtype=np.int64)
    i2 = np.asarray(i2, dtype=np.int64)
    cross = (i1 < i2) & (fields[i1] != fields[i2])   # cross-group only
    if not np.any(cross):
        return keep, also_in
    r1, r2 = i1[cross], i2[cross]

    graph = _sparse.coo_matrix((np.ones(len(r1), dtype=np.int8), (r1, r2)),
                               shape=(n, n))
    n_comp, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels, minlength=n_comp)
    if not np.any(sizes > 1):
        return keep, also_in

    if rank_cols:
        rank_arrays = [np.asarray(table[c]) for c in rank_cols]
    else:
        cover = _coverage_count(table, coverage_cols)
        rank_arrays = [-cover]
    # np.lexsort's LAST key is primary. We need the result grouped by `labels`
    # (most significant) and, within each label, ranked by rank_arrays[0]
    # (next-most-significant) down to `idx` (least significant, final
    # tie-break) -- so the lexsort argument order is (idx, ..., rank_arrays[0],
    # labels), i.e. idx first / labels last.
    lexsort_keys = [idx] + list(reversed(rank_arrays)) + [labels]
    order = np.lexsort(tuple(lexsort_keys))
    sorted_labels = labels[order]
    first_mask = np.empty(len(order), dtype=bool)
    first_mask[0] = True
    first_mask[1:] = sorted_labels[1:] != sorted_labels[:-1]
    winner_for_label = order[first_mask]        # length n_comp, label-ordered asc
    winner_per_row = winner_for_label[labels]   # length n, vectorized gather
    keep = (idx == winner_per_row)

    loser_mask = (sizes[labels] > 1) & ~keep
    losers_idx = idx[loser_mask]
    losers_winner = winner_per_row[loser_mask]
    sort2 = np.argsort(losers_winner, kind='stable')
    losers_idx_s = losers_idx[sort2]
    losers_winner_s = losers_winner[sort2]
    if losers_idx_s.size:
        boundaries = np.flatnonzero(np.diff(losers_winner_s)) + 1
        groups = np.split(losers_idx_s, boundaries)
        starts = np.concatenate(([0], boundaries))
        winner_vals = losers_winner_s[starts]
        for w, members in zip(winner_vals, groups):
            other = sorted({fields[m] for m in members})
            also_in[w] = ','.join(other)
    return keep, also_in


def assemble(field_tables, dedup_radius_arcsec=0.2, coverage_cols=None,
            field_col='cmz_field', rank_cols=None):
    """Assemble per-field tables into one CMZ-wide catalog.

    Returns the combined ``Table`` with cross-field duplicates removed and two new
    columns: ``cmz_n_bands`` (finite-band coverage) and ``cmz_also_in`` (other
    fields an overlap source was also detected in).  ``field_tables`` are the
    outputs of :func:`load_field_catalog` (or any tables carrying ``cmz_field`` +
    RA/Dec).

    ``coverage_cols`` names the photometry columns whose finiteness defines "a
    band measured this source" (it drives both the coverage count and the
    keep-the-better-detection dedup rule).  **For production, pass it explicitly**
    -- the auto-detect fallback (:func:`_coverage_cols`) is a loose
    name-heuristic (``'flux'``/``'mag'`` substring, crude error-column exclusion)
    that can miss or over-include columns in an unfamiliar schema.

    ``field_col`` names the grouping column for cross-group dedup (default
    ``'cmz_field'``; pass ``'cmz_obsid'`` or another column to group by TILE
    instead of by named field -- see :func:`_dedup_cross_field`).

    ``rank_cols`` overrides the winner-selection rule inside a merged cluster
    (default: most coverage bands, then lowest row index); see
    :func:`_dedup_cross_field` for the column-ranking convention.
    """
    if not field_tables:
        raise ValueError('no field tables to assemble')
    combined = vstack(list(field_tables), join_type='outer',
                      metadata_conflicts='silent')
    cov = _coverage_cols(combined, coverage_cols)
    keep, also_in = _dedup_cross_field(combined, dedup_radius_arcsec, cov,
                                       field_col=field_col, rank_cols=rank_cols)
    combined['cmz_also_in'] = also_in.astype(str)   # other fields that saw it
    combined = combined[keep]
    combined['cmz_n_bands'] = _coverage_count(combined, cov)
    # FITS-safe (<=8 char) meta keywords
    combined.meta['CMZNFLD'] = len(set(np.asarray(combined[field_col])))
    combined.meta['CMZDEDR'] = float(dedup_radius_arcsec)
    combined.meta['CMZNSRC'] = len(combined)
    return combined


def assemble_from_paths(specs, dedup_radius_arcsec=0.2, coverage_cols=None):
    """``specs`` = iterable of dicts {path, field, program?, obsid?, tag?}.

    ``path`` may be a glob (each match loaded and tagged with the same field
    metadata -- e.g. per-observation tables of one field).
    """
    tables = []
    for s in specs:
        paths = (sorted(_glob.glob(s['path']))
                 if any(ch in s['path'] for ch in '*?[') else [s['path']])
        if not paths:
            raise FileNotFoundError(f"no catalog matched {s['path']!r}")
        for p in paths:
            tables.append(load_field_catalog(
                p, s['field'], program=s.get('program', ''),
                obsid=s.get('obsid', ''), tag=s.get('tag'),
                ra_col=s.get('ra_col'), dec_col=s.get('dec_col')))
    return assemble(tables, dedup_radius_arcsec=dedup_radius_arcsec,
                    coverage_cols=coverage_cols)


def write_outputs(table, out_stem, formats=('fits', 'ecsv', 'parquet')):
    """Write the assembled catalog to ``out_stem.{fits,ecsv,parquet}``.

    FITS/ECSV are the familiar deliverables; Parquet is the flat columnar form
    (a stepping stone to the partitioned HATS product -- see ``hats_export``).
    """
    written = []
    for fmt in formats:
        if fmt == 'fits':
            path = out_stem + '.fits'
            table.write(path, overwrite=True)
        elif fmt == 'ecsv':
            path = out_stem + '.ecsv'
            table.write(path, format='ascii.ecsv', overwrite=True)
        elif fmt == 'parquet':
            try:
                import pyarrow  # noqa: F401
            except ImportError:
                import warnings
                warnings.warn("parquet output skipped: 'pyarrow' not installed "
                              "(pip install pyarrow). FITS/ECSV still written.")
                continue
            path = out_stem + '.parquet'
            # SkyCoord/mixin columns don't serialize to parquet; drop to a plain
            # table of the storable columns first.
            flat = _to_storable(table)
            flat.write(path, format='parquet', overwrite=True)
        else:
            raise ValueError(f'unknown format {fmt!r}')
        written.append(path)
    return written


def _to_storable(table):
    """A copy with mixin (SkyCoord) columns expanded to plain float columns."""
    out = Table()
    for c in table.colnames:
        col = table[c]
        if col.__class__.__name__ == 'SkyCoord':
            out[c + '_ra'] = col.ra.deg
            out[c + '_dec'] = col.dec.deg
        else:
            out[c] = col
    out.meta = dict(table.meta)
    return out


def build_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog='python -m jwst_gc_pipeline.cmz.catalog_assembly',
        description='Assemble a CMZ-wide catalog from per-field combined catalogs.')
    p.add_argument('--spec', required=True,
                   help='JSON file: list of {path, field, program?, obsid?, tag?}')
    p.add_argument('--out', required=True, help='output stem (no extension)')
    p.add_argument('--dedup-radius-arcsec', type=float, default=0.2)
    p.add_argument('--formats', default='fits,ecsv,parquet')
    return p


def main(argv=None):
    import json
    args = build_parser().parse_args(argv)
    with open(args.spec) as fh:
        specs = json.load(fh)
    table = assemble_from_paths(specs, dedup_radius_arcsec=args.dedup_radius_arcsec)
    written = write_outputs(table, args.out,
                            formats=tuple(args.formats.split(',')))
    print(f"assembled {len(table)} sources from "
          f"{table.meta['CMZNFLD']} field(s) -> {', '.join(written)}")


if __name__ == '__main__':
    main()
