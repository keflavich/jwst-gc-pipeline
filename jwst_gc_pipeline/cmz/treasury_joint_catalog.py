"""Build the all-TILE JOINT NIRCam catalog for JWST program 10678 (GC Treasury).

There is no all-tile joint catalog for 10678 today: every one of the 66
per-pointing ``..._qualcuts_oksep<proposal>.fits`` tables stands alone, so a star
seen in two adjacent, overlapping pointings appears twice with two independent
positions and two independent photometry measurements.  This module assembles
the single joint table, with cross-TILE duplicates collapsed and every
surviving row carrying which OTHER tile(s) also saw it.

INPUT FILE COUNT CAVEAT: the input directory actually holds 198 files, not 66 --
each of the 66 pointings (``obsid``) has THREE ``..._qualcuts_oksep<proposal>.fits``
siblings: ``basic_merged_...`` (both NIRCam modules combined -- the one this
module wants) and ``basic_nrca_...`` / ``basic_nrcb_...`` (single-module
SUBSETS of that same pointing, not independent pointings).  A naive glob on
``*_qualcuts_oksep<proposal>.fits`` matches all three per obsid and -- keyed into a
``{obsid: path}`` dict -- silently keeps whichever sorts last
(``nrcb`` > ``nrca`` > ``merged`` alphabetically), handing the joint build a
single-module PARTIAL catalog instead of the full pointing (caught during this
PR's real-data smoke test: o040 read back only 35631 rows under the naive glob,
not the 62933 the ``basic_merged_`` file actually has).  ``discover_tile_paths``
therefore globs ``basic_merged_*_qualcuts_oksep<proposal>.fits`` specifically; see
``test_discover_tile_paths_excludes_module_subset_siblings``.

This reuses :mod:`jwst_gc_pipeline.cmz.catalog_assembly` (in particular its
vectorized, graph-based ``_dedup_cross_field``), but does NOT call
``catalog_assembly.assemble()`` directly: that function ``vstack``s the FULL
input tables up front, which at 10678 scale (~4.54M rows x ~101 columns after
qualcuts, across the 66 ``basic_merged_`` pointings) means holding every column
of every pointing in memory just to decide which rows survive.  Instead this
module runs the match/dedup decision on a LIGHTWEIGHT per-row table (RA/Dec,
tile id, band-coverage count, edge distance, F212N flux error -- the columns
the winner rule actually needs), then gathers the FULL rows of the survivors
tile-by-tile from disk.  See ``build_joint_catalog`` and the module README
section "Memory" below.

Grouping is by TILE (``obsid``), not by named field: each of the 66 10678
pointings is its own dedup group, so only CROSS-TILE pairs are ever merged and
a close pair inside one pointing is left exactly as the per-pointing m7
pipeline resolved it
(``catalog_assembly._dedup_cross_field(..., field_col='obsid')``).

Schema (from inspecting a real tile,
``basic_merged_indivexp_photometry_tables_merged_resbgsub_m7_o040_qualcuts_oksep<proposal>.fits``):

* position: the ``skycoord_ref`` mixin column (``skycoord_ref.ra`` /
  ``skycoord_ref.dec`` on disk) -- the position the per-tile m7 cross-band merge
  already chose as each source's reference position.  NOT ``skycoord_f212n`` /
  ``skycoord_f480m`` (the per-band centroids), which can be masked per-row when
  a band has no detection.
* bands: two NIRCam filters are present, ``f212n`` and ``f480m``, as
  ``flux_jy_<band>`` / ``eflux_jy_<band>`` (physical Jy units, comparable across
  tiles and filters -- unlike the raw ``flux_<band>`` / ``flux_err_<band>``
  DN-like columns).  Bands are discovered at runtime from the column names
  (``discover_bands``), not hardcoded, so a schema change is visible rather than
  silently mismatched.
* roll correction: detected via the ``ROLLCCAT`` FITS header card (PR #1006,
  not yet merged at the time this module was written, so no on-disk tile
  carries it yet -- every tile here reads ``roll_corrected=False``, which is
  recorded in the provenance so a later re-run against the rolled copies is
  distinguishable).  As of 2026-10-02, 50 of the 66 ``basic_merged`` pointings
  have a roll-corrected copy; the maintainer is holding the PRODUCTION run
  until all 66 do (do not run it before then -- see ``--obsids`` for a smoke
  test on a handful of real pointings instead).

Astrometric gate (REQUIRED, before any dedup decision): every pair of tiles
whose footprints overlap is checked reference-free, pairwise, with
``jwst_gc_pipeline.photometry.interframe_overlap.pairwise_overlap_offsets``
(offset-histogram stacking, window-swept -- never a nearest-neighbour median
against a dense catalog; CLAUDE.md Astrometry Rule #1).  A pair that overlaps
and is not cleanly tied within ``DEFAULT_OVERLAP_TOL_MAS`` (30 mas) REFUSES the
build, naming the pair, unless overridden with a written reason
(``--allow-overlap-fail-reason``), which is stored in the provenance's
``gate_override`` block.

MIRI F770W is explicitly OUT OF SCOPE here -- see ``_miri_crossmatch_todo`` and
issue #956: a per-tile MIRI tie must be independently verified (swept
``measure_offset`` vs the joint F480M position list, contrast >= 5,
``confirm_windows=True``) before any MIRI source could be matched in; this
module only ever reads NIRCam ``..._qualcuts_oksep<proposal>.fits`` tiles.

Winner rule inside a cross-tile duplicate cluster (most-important first, each
criterion only breaks a tie left by the previous one):

1. most finite bands (``f212n``/``f480m`` ``flux_jy_*`` both finite beats one);
2. larger distance from its OWN tile's footprint edge -- computed from the
   convex hull of that tile's own source positions in a local tangent-plane
   projection centered on the tile (``edge_distance_arcsec``); a source near a
   tile's edge is covered by fewer exposures and is the less reliable of a
   duplicate pair even at equal band coverage;
3. lower F212N flux error (``eflux_jy_f212n``, Jy -- missing F212N sorts last);
4. lowest row index (deterministic final tie-break).

Output: ``<out-dir>/<version-tag>/gctreasury10678_joint_nircam.{fits,parquet}``
plus a ``.prov.json`` sidecar recording input sha1s, the pipeline tag, the
per-tile ``roll_corrected`` flag, the overlap-gate results, the dedup
configuration, rows in/out per tile, and ``n_clusters``.  Refuses to overwrite
an existing output, and refuses to write under
``/blue/adamginsburg/adamginsburg/jwst/`` (issue #937).

Memory (~4.54M rows in / ~3.6M rows out x ~101 columns, full 66-pointing run --
a peer measurement at 0.1" dedup radius read 3,606,678 surviving rows; this
module defaults to 0.2"):
* match/dedup phase: one lightweight ~8-column float/str/int table for ALL
  66 pointings' rows at once (~4.54M rows x 8 columns x 8 bytes ~= 300 MB) plus
  the ``search_around_sky`` pair lists at the dedup radius (duplicates are a
  sizable fraction here -- ~20% of rows collapse -- but still far below O(N^2)
  pairs).
* gather phase: the final joint table itself, ~3.6M rows x ~101 columns,
  materialized once before writing -- this is the unavoidable floor, and the
  dominant cost: roughly 3.6M x 101 x 8 bytes ~= 2.9 GB of raw column payload,
  plus astropy/FITS overhead and the transient per-pointing full-table reads
  during gathering (each pointing read-and-discarded, so this does not
  accumulate across pointings, but the largest single pointing's full table is
  briefly held alongside the growing output list before the final ``vstack``).
  Budget ~6-8 GB peak RAM for the full 66-pointing run; the 4-tile smoke test
  in this PR used well under 1 GB.
"""
import argparse
import glob
import functools
import hashlib
import json
import os
import re

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, vstack

from jwst_gc_pipeline.cmz import catalog_assembly as CA
from jwst_gc_pipeline.photometry.interframe_overlap import (
    DEFAULT_OVERLAP_TOL_MAS, pairwise_overlap_offsets)
from jwst_gc_pipeline.versioning.tags import get_pipeline_tag

DEFAULT_INPUT_DIR = '/orange/adamginsburg/jwst/gc-treasury/catalogs'
DEFAULT_OUTPUT_ROOT = '/orange/adamginsburg/jwst/gc-treasury/catalogs_joint'
OUT_STEM = 'gctreasury10678_joint_nircam'

# issue #937: this directory tree is forbidden for any pipeline OUTPUT (it is
# not the data root -- see memory note pipe-root-is-the-repo-not-the-data-root).
FORBIDDEN_OUTPUT_PREFIX = '/blue/adamginsburg/adamginsburg/jwst/'

DEFAULT_DEDUP_RADIUS_ARCSEC = 0.2

# NOTE: the input directory also carries 'basic_nrca_...'/'basic_nrcb_...'
# per-MODULE subset catalogs alongside the module-combined 'basic_merged_...'
# one, all three sharing the same obsid and the same qualcuts suffix (e.g.
# 'basic_merged_indivexp_photometry_tables_merged_resbgsub_m7_o040<suffix>.fits'
# next to 'basic_nrca_...' and 'basic_nrcb_...' siblings for the SAME o040).
# A glob on '*<suffix>.fits' matches all three; keying a dict by obsid then
# silently keeps whichever sorts LAST (alphabetically 'nrcb' > 'merged'),
# handing the module-only partial catalog to the joint build instead of the
# full tile.  Match 'basic_merged_' specifically.
#
# The suffix comes from the per-field registry
# (merge_catalogs._qualcuts_oksep_suffix), never a literal; see
# jwst_gc_pipeline/tests/test_no_hardcoded_qualcuts_token.py.
TREASURY_TARGET = 'gc-treasury'


@functools.lru_cache(maxsize=None)
def qualcuts_suffix(target=TREASURY_TARGET):
    """The quality-cut table suffix for ``target`` (10678 for gc-treasury).

    Imported lazily: merge_catalogs is a slow import, and only tile discovery
    needs it."""
    from jwst_gc_pipeline.photometry.merge_catalogs import (
        _qualcuts_oksep_suffix)
    return _qualcuts_oksep_suffix(target)


def _tile_re(suffix):
    return re.compile(
        r'^basic_merged_.*_o(?P<obsid>\d+)' + re.escape(suffix) + r'\.fits$')


_FLUX_JY_RE = re.compile(r'^flux_jy_(?P<band>.+)$')

REF_RA_COL = 'skycoord_ref.ra'
REF_DEC_COL = 'skycoord_ref.dec'


class MixedRollCorrectionError(ValueError):
    """Raised when the input tiles mix roll-corrected and raw catalogs."""


class OverlapGateError(RuntimeError):
    """Raised when a cross-tile overlap pair fails the astrometric gate."""


class OutputRefusedError(RuntimeError):
    """Raised when the output path is forbidden or already exists."""


# ---------------------------------------------------------------------------
# Tile discovery and per-tile metadata
# ---------------------------------------------------------------------------
def discover_tile_paths(input_dir, obsids=None):
    """Return ``{obsid: path}`` for every ``..._qualcuts_oksep<proposal>.fits`` tile
    in ``input_dir``, optionally restricted to ``obsids`` (an iterable of
    3-digit-or-wider obsid strings, e.g. ``['040', '041']`` -- the smoke-test
    subsetting hook; a full production run omits it)."""
    suffix = qualcuts_suffix()
    tile_re = _tile_re(suffix)
    paths = sorted(glob.glob(os.path.join(
        input_dir, f'basic_merged_*{suffix}.fits')))
    out = {}
    for p in paths:
        m = tile_re.search(os.path.basename(p))
        if not m:
            continue
        obsid = m.group('obsid')
        if obsids is not None and obsid not in set(obsids):
            continue
        out[obsid] = p
    if not out:
        raise FileNotFoundError(
            f'no *{suffix}.fits tiles found in {input_dir!r} '
            f'(obsids filter={obsids!r})')
    return dict(sorted(out.items()))


def _sha1_file(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def _header_roll_corrected(path):
    """Read the ``ROLLCCAT`` card (PR #1006) from a tile's PRIMARY or table
    HDU header. Returns ``False`` (not an error) if the card is absent -- every
    tile written before that PR merges has no such card, and that absence IS
    the correct ``roll_corrected=False`` answer, not a missing-data condition."""
    with fits.open(path, memmap=True) as hdul:
        for hdu in hdul:
            if 'ROLLCCAT' in hdu.header:
                return bool(hdu.header['ROLLCCAT'])
    return False


def _tile_pipeline_tag(path):
    """The per-tile GCTAG stamped by the pipeline run that produced it (its own
    provenance), distinct from this module's own ``get_pipeline_tag()``."""
    with fits.open(path, memmap=True) as hdul:
        for hdu in hdul:
            if 'GCTAG' in hdu.header:
                return str(hdu.header['GCTAG'])
    return 'unknown'


def discover_bands(colnames):
    """Bands present in a tile's schema, discovered from ``flux_jy_<band>``
    column names (NOT hardcoded -- a schema change becomes visible instead of
    silently mismatched)."""
    bands = sorted(m.group('band') for c in colnames
                   for m in [_FLUX_JY_RE.match(c)] if m)
    if not bands:
        raise KeyError(
            "no 'flux_jy_<band>' columns found; cannot discover bands")
    return bands


def assert_no_mixed_roll_correction(tile_roll_flags):
    """``tile_roll_flags``: ``{obsid: bool}``. Raise if the set mixes True/False
    -- either every tile in this run comes from the roll-corrected directory,
    or none does; a silent mix would merge a corrected tile against a raw one
    as though they shared a frame."""
    values = set(tile_roll_flags.values())
    if len(values) > 1:
        corrected = sorted(o for o, v in tile_roll_flags.items() if v)
        raw = sorted(o for o, v in tile_roll_flags.items() if not v)
        raise MixedRollCorrectionError(
            f'mixed roll-corrected input: {len(corrected)} tile(s) '
            f'roll-corrected ({corrected[:10]}{"..." if len(corrected) > 10 else ""}), '
            f'{len(raw)} tile(s) raw ({raw[:10]}{"..." if len(raw) > 10 else ""}). '
            f'Point --input-dir at either an all-corrected or an all-raw '
            f'directory, never a mix.')


# ---------------------------------------------------------------------------
# Edge distance (winner-rule criterion 2)
# ---------------------------------------------------------------------------
def edge_distance_arcsec(ra_deg, dec_deg):
    """Distance (arcsec) of each of a tile's OWN sources from the tile's own
    footprint edge, via the convex hull of its own positions in a local
    tangent-plane projection centered on the tile's centroid.

    Chosen over a bounding-box because the 10678 tiles are not axis-aligned on
    the sky (``footprints_gc-treasury_*.json`` footprints are rotated
    quadrilaterals): a bounding box computed in RA/Dec would call a source near
    a ROTATED edge "deep interior". The convex hull of the tile's own star
    positions tracks the actual (possibly rotated) footprint boundary that
    matters for "how many overlapping exposures cover this point" -- which is
    the reliability signal this criterion is a proxy for.

    Positive = inside the hull, by that many arcsec from the nearest facet;
    negative = outside (can occur for a source right at the convex hull
    boundary due to floating point, or if the hull degenerates -- see fallback
    below). Only the RELATIVE ordering of this value is used by the winner
    rule, so a a few-mas edge case does not need to be exactly right, just
    directionally consistent.
    """
    import scipy.spatial

    ra_deg = np.asarray(ra_deg, dtype=float)
    dec_deg = np.asarray(dec_deg, dtype=float)
    n = len(ra_deg)
    if n < 3:
        return np.zeros(n)
    dec0 = float(np.mean(dec_deg))
    ra0 = float(np.mean(ra_deg))
    cosd = max(np.cos(np.radians(dec0)), 1e-6)
    x = (ra_deg - ra0) * cosd * 3600.0   # arcsec, tangent-plane
    y = (dec_deg - dec0) * 3600.0
    pts = np.column_stack([x, y])
    try:
        hull = scipy.spatial.ConvexHull(pts)
    except scipy.spatial.QhullError:
        # degenerate (collinear / near-collinear) point set: fall back to a
        # bounding-box margin in the same tangent frame, which is still a
        # valid (if cruder) "distance from the footprint edge".
        margin = np.minimum.reduce([
            x - x.min(), x.max() - x, y - y.min(), y.max() - y])
        return margin
    A = hull.equations[:, :-1]   # unit facet normals
    b = hull.equations[:, -1]
    # Qhull convention: A.x + b <= 0 for interior points: signed distance to
    # facet i (positive = inside) is -(A_i . x + b_i); the nearest facet sets
    # the distance-from-edge.
    signed = -(pts @ A.T + b[None, :])
    return signed.min(axis=1)


# ---------------------------------------------------------------------------
# Lightweight per-row match table (RA/Dec + the winner-rule inputs only)
# ---------------------------------------------------------------------------
def _load_match_rows(path, obsid, bands):
    """Read ONLY the columns the match/dedup decision needs, via a memmapped
    FITS open so untouched columns are never paged into RAM. Returns a plain
    dict of numpy arrays (length = number of rows in this tile)."""
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        ra = np.asarray(data[REF_RA_COL], dtype=float)
        dec = np.asarray(data[REF_DEC_COL], dtype=float)
        n = len(ra)
        finite = np.zeros((len(bands), n), dtype=bool)
        for i, band in enumerate(bands):
            col = f'flux_jy_{band}'
            if col in data.columns.names:
                finite[i] = np.isfinite(np.asarray(data[col], dtype=float))
        n_bands = finite.sum(axis=0)
        f212_err_col = 'eflux_jy_f212n'
        if f212_err_col in data.columns.names:
            f212_err = np.asarray(data[f212_err_col], dtype=float)
            f212_err = np.where(np.isfinite(f212_err), f212_err, np.inf)
        else:
            f212_err = np.full(n, np.inf)
    edge_dist = edge_distance_arcsec(ra, dec)
    return dict(ra=ra, dec=dec, n_bands=n_bands, edge_dist=edge_dist,
               f212_err=f212_err, obsid=np.full(n, obsid, dtype=object),
               tile_row=np.arange(n, dtype=np.int64))


def build_match_table(tile_paths, bands):
    """Stack every tile's lightweight match rows into one table, carrying
    ``obsid`` + ``tile_row`` so a surviving GLOBAL row can be traced back to
    its exact (file, row) for the gather phase."""
    chunks = [_load_match_rows(p, obsid, bands)
             for obsid, p in tile_paths.items()]
    t = Table()
    for key in ('ra', 'dec', 'n_bands', 'edge_dist', 'f212_err', 'obsid',
               'tile_row'):
        t[key] = np.concatenate([c[key] for c in chunks])
    return t


# ---------------------------------------------------------------------------
# Pre-merge astrometric gate (REQUIRED)
# ---------------------------------------------------------------------------
def run_overlap_gate(match_table, tol_mas=DEFAULT_OVERLAP_TOL_MAS,
                     allow_overlap_fail_reason=None):
    """Reference-free, pairwise overlap tie for every pair of tiles whose
    footprints intersect (``pairwise_overlap_offsets`` -- offset-histogram
    stacking, window-swept; never a nearest-neighbour median against a dense
    catalog, per CLAUDE.md Astrometry Rule #1).

    Raises ``OverlapGateError`` naming every pair that overlaps and is not
    cleanly tied, unless ``allow_overlap_fail_reason`` is a non-empty string
    (the override, stored by the caller in the provenance).

    Returns the full list of per-pair result dicts (for the provenance), even
    on a tolerated override.
    """
    obsid_arr = np.asarray(match_table['obsid'])
    groups = {}
    for obsid in np.unique(obsid_arr):
        m = obsid_arr == obsid
        groups[str(obsid)] = SkyCoord(
            ra=np.asarray(match_table['ra'])[m] * u.deg,
            dec=np.asarray(match_table['dec'])[m] * u.deg, frame='icrs')
    results = pairwise_overlap_offsets(groups, tol_mas=tol_mas,
                                       context='10678 joint catalog')
    bad = [r for r in results if r['overlap'] and not r['ok']]
    if bad and not allow_overlap_fail_reason:
        names = ', '.join(f"{r['a']}|{r['b']} ({r['off_mas']} mas)"
                          for r in bad)
        raise OverlapGateError(
            f'{len(bad)} overlapping tile pair(s) failed the {tol_mas} mas '
            f'astrometric gate: {names}. Fix the tiles, or override with '
            f'--allow-overlap-fail-reason (a written justification, stored '
            f'in the provenance) -- never a silent re-run.')
    return results


# ---------------------------------------------------------------------------
# Dedup (reuses catalog_assembly._dedup_cross_field, grouped by TILE)
# ---------------------------------------------------------------------------
def dedup_match_table(match_table, dedup_radius_arcsec=DEFAULT_DEDUP_RADIUS_ARCSEC):
    """Run the vectorized cross-TILE dedup on the lightweight match table.

    Winner rule (most-important first): most finite bands; larger edge
    distance; lower F212N flux error; lowest row index (the automatic final
    tie-break inside ``_dedup_cross_field``).

    Returns ``(keep, also_in_tiles)`` aligned to ``match_table``'s rows.
    """
    t = match_table.copy()
    t['__neg_n_bands'] = -np.asarray(t['n_bands'], dtype=float)
    t['__neg_edge_dist'] = -np.asarray(t['edge_dist'], dtype=float)
    t['__f212_err'] = np.asarray(t['f212_err'], dtype=float)
    keep, also_in = CA._dedup_cross_field(
        t, dedup_radius_arcsec, coverage_cols=None, field_col='obsid',
        rank_cols=('__neg_n_bands', '__neg_edge_dist', '__f212_err'))
    return keep, also_in


# ---------------------------------------------------------------------------
# Gather phase: pull the FULL rows of the survivors, tile by tile
# ---------------------------------------------------------------------------
def gather_survivor_rows(tile_paths, match_table, keep, also_in_tiles):
    """Re-read each tile's FULL table once, select only its surviving rows
    (by the ``tile_row`` indices recorded in ``match_table``), stamp
    ``also_in_tiles`` / ``joint_tile``, and vstack. Each tile's full table is
    read once and then dropped, so peak RAM never holds more than one tile's
    full columns plus the accumulating output.
    """
    obsid_arr = np.asarray(match_table['obsid'])
    tile_row_arr = np.asarray(match_table['tile_row'])
    out_chunks = []
    rows_in_out = {}
    for obsid, path in tile_paths.items():
        m = obsid_arr == obsid
        n_in = int(m.sum())
        m_keep = m & keep
        n_out = int(m_keep.sum())
        rows_in_out[obsid] = dict(rows_in=n_in, rows_out=n_out)
        if n_out == 0:
            continue
        rows = tile_row_arr[m_keep]
        order = np.argsort(rows)
        full = Table.read(path)
        sel = full[rows[order]]
        sel['joint_tile'] = obsid
        sel['also_in_tiles'] = np.asarray(also_in_tiles)[m_keep][order].astype(str)
        out_chunks.append(sel)
        del full
    if not out_chunks:
        raise RuntimeError('no surviving rows in any tile -- nothing to assemble')
    combined = vstack(out_chunks, join_type='outer', metadata_conflicts='silent')
    return combined, rows_in_out


# ---------------------------------------------------------------------------
# Top-level build
# ---------------------------------------------------------------------------
def build_joint_catalog(input_dir, obsids=None,
                        dedup_radius_arcsec=DEFAULT_DEDUP_RADIUS_ARCSEC,
                        tol_mas=DEFAULT_OVERLAP_TOL_MAS,
                        allow_overlap_fail_reason=None):
    """End to end: discover tiles -> roll-correction consistency check ->
    pre-merge overlap gate -> lightweight match/dedup -> gather survivor rows.

    Returns ``(table, provenance)``.
    """
    tile_paths = discover_tile_paths(input_dir, obsids=obsids)
    roll_flags = {o: _header_roll_corrected(p) for o, p in tile_paths.items()}
    assert_no_mixed_roll_correction(roll_flags)
    roll_corrected = next(iter(roll_flags.values())) if roll_flags else False

    sample = Table.read(next(iter(tile_paths.values())), memmap=True)
    bands = discover_bands(sample.colnames)
    del sample

    match_table = build_match_table(tile_paths, bands)
    gate_results = run_overlap_gate(
        match_table, tol_mas=tol_mas,
        allow_overlap_fail_reason=allow_overlap_fail_reason)

    keep, also_in_tiles = dedup_match_table(
        match_table, dedup_radius_arcsec=dedup_radius_arcsec)
    n_clusters = int(np.sum(np.asarray(also_in_tiles) != ''))

    combined, rows_in_out = gather_survivor_rows(
        tile_paths, match_table, keep, also_in_tiles)

    prov = dict(
        pipeline_tag=get_pipeline_tag(),
        bands=bands,
        dedup_radius_arcsec=float(dedup_radius_arcsec),
        winner_rule=['most_finite_bands', 'largest_edge_distance_arcsec',
                    'lowest_f212n_flux_err_jy', 'lowest_row_index'],
        n_clusters=n_clusters,
        n_rows_total=len(combined),
        n_tiles=len(tile_paths),
        roll_corrected=roll_corrected,
        overlap_gate=dict(
            tol_mas=float(tol_mas),
            n_pairs_checked=len(gate_results),
            n_overlapping=sum(1 for r in gate_results if r['overlap']),
            n_failed=sum(1 for r in gate_results
                        if r['overlap'] and not r['ok']),
            results=gate_results,
        ),
        gate_override=(
            {'allow_overlap_fail_reason': allow_overlap_fail_reason}
            if allow_overlap_fail_reason else None),
        tiles={
            o: dict(
                path=p,
                sha1=_sha1_file(p),
                roll_corrected=roll_flags[o],
                tile_pipeline_tag=_tile_pipeline_tag(p),
                **rows_in_out[o],
            )
            for o, p in tile_paths.items()
        },
        miri_crossmatch='OUT OF SCOPE -- see issue #956 and _miri_crossmatch_todo()',
    )
    combined.meta['CMZNFLD'] = len(tile_paths)
    combined.meta['CMZDEDR'] = float(dedup_radius_arcsec)
    combined.meta['CMZNSRC'] = len(combined)
    combined.meta['GCTAG'] = prov['pipeline_tag']
    return combined, prov


def _miri_crossmatch_todo():
    """TODO (issue #956): MIRI F770W is OUT OF SCOPE for this joint catalog.

    Before any MIRI source could be matched into ``gctreasury10678_joint_nircam``,
    a per-tile MIRI tie must be independently verified: a swept
    ``astrometry_offsets.measure_offset`` of the MIRI tile's own source list
    against THIS joint catalog's F480M positions, requiring contrast >= 5 and
    ``confirm_windows=True`` (never trusting the tile's own NIRCam-derived
    offset table as a MIRI proxy -- MIRI parallels have shown independent
    residuals up to tens of arcsec; see memory note
    treasury-miri-parallels-uncorrected). Not implemented here.
    """
    raise NotImplementedError(
        'MIRI F770W joint crossmatch is out of scope for this module; see '
        'issue #956. This NIRCam-only joint catalog must not be extended to '
        'include MIRI sources without that per-tile verification.')


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _assert_output_allowed(out_dir, stem_paths):
    if os.path.abspath(out_dir).startswith(
            os.path.abspath(FORBIDDEN_OUTPUT_PREFIX)):
        raise OutputRefusedError(
            f'refusing to write under {FORBIDDEN_OUTPUT_PREFIX!r} (issue #937) '
            f'-- that is the repo tree, not the data root; pass --out-dir '
            f'under /orange/.../gc-treasury/catalogs_joint/')
    existing = [p for p in stem_paths if os.path.exists(p)]
    if existing:
        raise OutputRefusedError(
            f'refusing to overwrite existing output: {existing}. Use a new '
            f'--version-tag directory instead.')


def write_joint_catalog(table, prov, out_dir, version_tag,
                        formats=('fits', 'parquet')):
    """Write ``<out_dir>/<version_tag>/gctreasury10678_joint_nircam.{fmt}``
    plus the ``.prov.json`` sidecar. Refuses to overwrite, and refuses the
    forbidden ``/blue/.../jwst/`` output prefix (issue #937)."""
    version_dir = os.path.join(out_dir, version_tag)
    stem = os.path.join(version_dir, OUT_STEM)
    candidate_paths = [stem + '.fits', stem + '.parquet',
                       stem + '.prov.json']
    _assert_output_allowed(out_dir, candidate_paths)
    os.makedirs(version_dir, exist_ok=True)
    written = CA.write_outputs(table, stem, formats=formats)
    prov_path = stem + '.prov.json'
    with open(prov_path + '.tmp', 'w') as fh:
        json.dump(prov, fh, indent=2, sort_keys=True, default=str)
        fh.write('\n')
    os.replace(prov_path + '.tmp', prov_path)
    written.append(prov_path)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog='python -m jwst_gc_pipeline.cmz.treasury_joint_catalog',
        description='Build the all-tile JOINT NIRCam catalog for 10678 '
                    '(GC Treasury).')
    p.add_argument('--input-dir', default=DEFAULT_INPUT_DIR,
                   help='directory of *_qualcuts_oksep<proposal>.fits tiles '
                        '(default: the raw per-tile catalogs; point at a '
                        'catalogs_rollcorr/<tag>/ directory once roll '
                        'correction has landed for every tile -- never mix)')
    p.add_argument('--out-dir', default=DEFAULT_OUTPUT_ROOT)
    p.add_argument('--version-tag', required=True,
                   help='output subdirectory, e.g. a date tag or PR-derived '
                        'tag; the output refuses to overwrite an existing one')
    p.add_argument('--obsids', default=None,
                   help='comma-separated obsid subset for a smoke test '
                        '(e.g. "040,041,042,043"); omit for the full run')
    p.add_argument('--dedup-radius-arcsec', type=float,
                   default=DEFAULT_DEDUP_RADIUS_ARCSEC)
    p.add_argument('--overlap-tol-mas', type=float,
                   default=DEFAULT_OVERLAP_TOL_MAS)
    p.add_argument('--allow-overlap-fail-reason', default=None,
                   help='written justification to override a failed overlap '
                        'gate; stored in the provenance. Omit to fail closed.')
    p.add_argument('--formats', default='fits,parquet')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    obsids = args.obsids.split(',') if args.obsids else None
    table, prov = build_joint_catalog(
        args.input_dir, obsids=obsids,
        dedup_radius_arcsec=args.dedup_radius_arcsec,
        tol_mas=args.overlap_tol_mas,
        allow_overlap_fail_reason=args.allow_overlap_fail_reason)
    written = write_joint_catalog(
        table, prov, args.out_dir, args.version_tag,
        formats=tuple(args.formats.split(',')))
    print(f'assembled {len(table)} sources from {prov["n_tiles"]} tile(s), '
         f'{prov["n_clusters"]} cross-tile duplicate cluster(s) collapsed -> '
         f'{", ".join(written)}')


if __name__ == '__main__':
    main()
