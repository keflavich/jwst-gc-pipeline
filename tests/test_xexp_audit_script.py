"""Tests for scripts/analysis/xexp_audit.py (read-only cross-exposure QA report).

Each test builds a small field on disk::

    <tmp>/field/catalogs/f212n_merged_o078_indivexp_merged_m2_dao_basic.fits
    <tmp>/field/F212N/<per-frame catalogs, crfs>

and runs the script's ``main`` with a flat stand-in WCS, so no pipeline data are
needed.  The report directory is ``<tmp>/reports``.
"""
import hashlib
import importlib.util
import os

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from jwst_gc_pipeline.photometry import xexp_reconcile as X

_SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'analysis',
                       'xexp_audit.py')
_spec = importlib.util.spec_from_file_location('xexp_audit_script', _SCRIPT)
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

SCALE = 0.031                      # arcsec / pixel
SHAPE = (200, 200)
MERGED = 'f212n_merged_o078_indivexp_merged_m2_dao_basic.fits'


class _LinearWCS:
    """sky = pixel * 1e-5 deg about (RA0, DEC0), flat; enough for the audit."""
    RA0, DEC0, DEG = 266.0, -29.0, 1e-5

    def world_to_pixel_values(self, ra, dec):
        return ((np.asarray(ra) - self.RA0) / self.DEG,
                (np.asarray(dec) - self.DEC0) / self.DEG)

    def pixel_to_world_values(self, x, y):
        return (self.RA0 + np.asarray(x, float) * self.DEG,
                self.DEC0 + np.asarray(y, float) * self.DEG)


def _wcs(crf):
    return _LinearWCS()


def _sky(x, y):
    return _LinearWCS.RA0 + x * _LinearWCS.DEG, _LinearWCS.DEC0 + y * _LinearWCS.DEG


def _write_frame(d, tag, rows, sat_rows=(), sat_suffix='_m12', shape=SHAPE):
    """crf with a clean DQ plane and a per-frame catalog; returns its basename."""
    crf = str(d / f'{tag}_crf.fits')
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(np.zeros(shape, np.float32), name='SCI'),
                  fits.ImageHDU(np.zeros(shape, np.uint32), name='DQ')]).writeto(crf)
    arr = np.asarray(rows, float).reshape(-1, 3)
    cat = Table({'x_fit': arr[:, 0], 'y_fit': arr[:, 1], 'flux_fit': arr[:, 2]})
    cat.meta['FILENAME'] = crf
    cat.meta['PIXSCALE'] = SCALE
    name = f'f212n_nrcb1_o078_{tag}_m2_daophot_basic.fits'
    cat.write(str(d / name))
    if len(sat_rows):
        sarr = np.asarray(sat_rows, float).reshape(-1, 3)
        # as the pipeline writes it: x_fit/y_fit are cutout-frame coordinates,
        # the detector pixels are xcentroid/ycentroid
        Table({'x_fit': np.full(len(sarr), 40.0), 'y_fit': np.full(len(sarr), 40.0),
               'flux_fit': sarr[:, 2], 'xcentroid': sarr[:, 0],
               'ycentroid': sarr[:, 1]}).write(
            crf.replace('.fits', f'{sat_suffix}_satstar_catalog.fits'))
    return name


# six merged sources on a three-exposure field:
#   R0 (50, 50)    bright, in all three            -> all
#   R1 (100, 100)  bright, in exposure 0 only      -> one, flagged
#   R2 (150, 50)   faint (S/N 2), in exposure 1    -> one, not bright
#   R3 (50, 150)   bright, in exposures 0 and 1    -> some
#   R4 (150, 150)  bright, in none                 -> none
#   R5 (300, 300)  off every array                 -> not covered at all
ROWS = [(50, 50, 1e5), (100, 100, 1e5), (150, 50, 1e3), (50, 150, 1e5),
        (150, 150, 1e5), (300, 300, 1e5)]
ERR = [1e3, 1e3, 5e2, 1e3, 1e3, 1e3]
NMATCH = [3, 2, 1, 1, 1, 1]
FRAMES = [[ROWS[0], ROWS[1], ROWS[3]],
          [ROWS[0], ROWS[2], ROWS[3]],
          [ROWS[0]]]


def _merged(names, rows=ROWS, err=ERR, nmatch=None, with_fn=True, saturated=None):
    xy = np.asarray([(x, y) for x, y, _ in rows], float).reshape(-1, 2)
    ra, dec = _sky(xy[:, 0], xy[:, 1])
    tab = Table({'skycoord': SkyCoord(ra, dec, unit='deg'),
                 'flux': np.asarray([f for _, _, f in rows], float),
                 'flux_err': np.asarray(err, float),
                 'is_saturated': (np.zeros(len(rows), bool) if saturated is None
                                  else np.asarray(saturated, bool))})
    if nmatch is not None:
        tab['nmatch'] = np.asarray(nmatch, int)
    if with_fn:
        for i, name in enumerate(names):
            tab.meta[f'fn{i}'] = name
    return tab


def _field(tmp_path, name=MERGED, frames=FRAMES, sat=None, shapes=None, **kw):
    """``sat`` = {frame index: satstar rows}; ``shapes`` = {frame index: array shape}."""
    root = tmp_path / 'field'
    (root / 'F212N').mkdir(parents=True, exist_ok=True)
    (root / 'catalogs').mkdir(exist_ok=True)
    sat, shapes = sat or {}, shapes or {}
    names = [_write_frame(root / 'F212N', f'e{i}', rows, sat_rows=sat.get(i, ()),
                          shape=shapes.get(i, SHAPE))
             for i, rows in enumerate(frames)]
    path = root / 'catalogs' / name
    _merged(names, **kw).write(str(path))
    return str(path), root


def _run(out, *args, **kw):
    # no rank cut unless the caller passes --single-top-pct (the last one wins)
    return A.main(['--out', str(out), '--single-top-pct', '100', *map(str, args)],
                  wcs_loader=_wcs, bad_dq_mask=3, **kw)


def _tree(root):
    """name -> sha256 of every file under root."""
    found = {}
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            with open(p, 'rb') as fh:
                found[os.path.relpath(p, root)] = hashlib.sha256(fh.read()).hexdigest()
    return found


def _age(root, frames=1000, merged=2000):
    """Per-frame products written at ``frames``, merged catalogs at ``merged``
    (seconds since the epoch), so which is newer does not depend on how fast
    the test wrote them."""
    for sub, mtime in (('F212N', frames), ('catalogs', merged)):
        for name in os.listdir(root / sub):
            os.utime(root / sub / name, (mtime, mtime))


def _report(out, *catalogs):
    """Audit ``catalogs`` into ``out``, collect, and return the report text."""
    assert _run(out, *catalogs) == 0
    assert A.main(['--out', str(out), '--collect']) == 0
    return (out / 'report.md').read_text()


def _summary(out, tag='all'):
    return Table.read(str(out / f'summary.{tag}.tsv'), format='ascii.tab')


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name, filt', [
    ('f212n_merged_o078_indivexp_merged_m2_dao_basic.fits', 'F212N'),
    ('F480M_x.fits', 'F480M'),
    ('f1000w_merged.fits', 'F1000W'),
    ('/a/b/catalogs/f150w_merged.fits', 'F150W'),
])
def test_filter_from_name(name, filt):
    assert A.filter_from_name(name) == filt


@pytest.mark.parametrize('name', ['merged_f212n.fits', 'f212x_merged.fits',
                                  'catalog.fits', 'f21n_merged.fits',
                                  'merged_f212n_x.fits'])
def test_filter_from_name_rejects_names_without_a_leading_filter(name):
    with pytest.raises(ValueError, match='filter'):
        A.filter_from_name(name)


def test_default_catalog_dirs_is_the_filter_directory_beside_catalogs():
    assert A.default_catalog_dirs('/a/b/catalogs/f212n_x.fits', 'F212N') == ['/a/b/F212N']


def test_select_shard_partitions_the_list():
    items = list(range(10))
    parts = [A.select_shard(items, f'{i}/3') for i in range(3)]
    assert sorted(sum(parts, [])) == items
    assert all(not set(a) & set(b) for i, a in enumerate(parts) for b in parts[i + 1:])
    assert parts[1] == [1, 4, 7]
    assert A.select_shard(items, None) == items
    assert A.select_shard(items, '0/1') == items
    assert A.select_shard(items, '7/8') == [7]


@pytest.mark.parametrize('spec', ['3/3', '1/0', 'x', '-1/3', '1/3/4', '', '1', '1/'])
def test_select_shard_rejects_bad_specs(spec):
    with pytest.raises(ValueError, match='shard'):
        A.select_shard(list(range(5)), spec)


# ---------------------------------------------------------------------------
# one catalog
# ---------------------------------------------------------------------------

def _file_audit(path, **kw):
    # the rank cut keeps the brightest few percent of a handful of rows, so these
    # fixtures run without it unless a test is about it
    kw.setdefault('params', X.XexpParams(single_top_pct=100.0))
    kw.setdefault('wcs_loader', _wcs)
    kw.setdefault('bad_dq_mask', 3)
    kw.setdefault('log', lambda *a, **k: None)
    return A.audit_catalog_file(path, **kw)


def test_audit_of_a_catalog_without_audit_columns_is_recomputed(tmp_path):
    path, _ = _field(tmp_path)
    s, frames, flagged = _file_audit(path, n_flagged=5)
    assert s['source'] == 'recomputed' and s['filter'] == 'F212N'
    assert (s['rows'], s['frames_read'], s['frames_skipped']) == (6, 3, 0)
    assert (s['multi'], s['all'], s['some'], s['one'], s['none']) == (5, 1, 1, 2, 1)
    assert (s['bright_multi'], s['bright_one'], s['single_exposure']) == (4, 1, 1)
    assert s['multi'] == 5 and s['frac_one'] == pytest.approx(2 / 5)
    assert s['bright_frac_one'] == pytest.approx(1 / 4)
    assert s['snr_min'] == 50.0
    assert len(frames) == 3 and frames['catalog'][0] == MERGED
    assert len(flagged) == 1
    assert flagged['flux'][0] == 1e5 and flagged['ncover'][0] == 3
    assert flagged['nfound'][0] == 1
    assert flagged['ra'][0] == pytest.approx(_sky(100, 100)[0])
    assert flagged['dec'][0] == pytest.approx(_sky(100, 100)[1])
    assert flagged['snr'][0] == pytest.approx(100.0)
    assert flagged['nmatch'][0] == -1 and not flagged['saturated'][0]


def test_flagged_rows_are_the_brightest_first_and_capped(tmp_path):
    rows = [(40, 40, 2e5), (80, 80, 9e5), (120, 120, 5e5)]
    path, _ = _field(tmp_path, frames=[[rows[0], rows[1], rows[2]], [], []], rows=rows,
                     err=[1e3, 1e3, 1e3])
    s, _, flagged = _file_audit(path, n_flagged=2)
    assert s['single_exposure'] == 3
    assert flagged['flux'].tolist() == [9e5, 5e5]


def test_a_faint_saturated_row_is_flagged_and_marked_saturated(tmp_path):
    rows = [(60, 60, 1e3), (120, 120, 1e5)]
    path, _ = _field(tmp_path, frames=[rows, [rows[1]], [rows[1]]], rows=rows,
                     err=[5e2, 1e3], saturated=[True, False])
    s, _, flagged = _file_audit(path, n_flagged=5)
    assert s['single_exposure'] == 1
    assert flagged['flux'].tolist() == [1e3] and flagged['saturated'].tolist() == [True]

def test_flagged_is_empty_but_has_columns_when_nothing_is_flagged(tmp_path):
    path, _ = _field(tmp_path, frames=[[ROWS[0]]] * 3, rows=[ROWS[0]], err=[1e3])
    s, _, flagged = _file_audit(path, n_flagged=5)
    assert s['single_exposure'] == 0 and len(flagged) == 0
    assert 'ncover' in flagged.colnames and 'saturated' in flagged.colnames


def test_audit_uses_stored_columns_unless_told_to_recompute(tmp_path):
    path, root = _field(tmp_path)
    first = _file_audit(path)[0]
    # write the merged catalog back with the columns the pipeline adds
    tab = Table.read(path)
    X.audit_merged_catalog(
        tab, filt='F212N', catalog_dirs=[str(root / 'F212N')], satstar_suffix='_m12',
        params=X.XexpParams(single_top_pct=100.0), wcs_loader=_wcs, bad_dq_mask=3,
        log=lambda m: None)
    stored = tmp_path / 'field' / 'catalogs' / 'f212n_stored_o078_indivexp_merged_m2_dao_basic.fits'
    tab.write(str(stored))

    def boom(crf):
        raise AssertionError('the WCS must not be loaded when the columns are stored')

    s = A.audit_catalog_file(str(stored), wcs_loader=boom, bad_dq_mask=3)[0]
    assert s['source'] == 'column' and s['frames_newer'] == -1
    for key in ('rows', 'multi', 'all', 'some', 'one', 'none', 'bright_multi',
                'bright_one', 'single_exposure', 'frames_read', 'frames_skipped'):
        assert s[key] == first[key], key
    r = _file_audit(str(stored), recompute=True)[0]
    assert r['source'] == 'recomputed'
    for key in ('multi', 'all', 'some', 'one', 'none', 'single_exposure'):
        assert r[key] == first[key], key


def test_stored_columns_keep_the_snr_cut_they_were_made_with(tmp_path):
    path, root = _field(tmp_path)
    tab = Table.read(path)
    X.audit_merged_catalog(
        tab, filt='F212N', catalog_dirs=[str(root / 'F212N')], satstar_suffix='_m12',
        params=X.XexpParams(single_snr_min=500.0, single_top_pct=100.0), wcs_loader=_wcs,
        bad_dq_mask=3, log=lambda m: None)
    stored = str(root / 'catalogs' / 'f212n_stored_o078_indivexp_merged_m2_dao_basic.fits')
    tab.write(stored)
    s = _file_audit(stored, params=X.XexpParams(single_snr_min=50.0, single_top_pct=100.0))[0]
    assert s['source'] == 'column' and s['snr_min'] == 500.0
    assert s['bright_multi'] == 0 and s['single_exposure'] == 0     # S/N is 100


def test_single_snr_changes_the_flag(tmp_path):
    path, _ = _field(tmp_path)
    cut = dict(single_top_pct=100.0)
    assert _file_audit(path, params=X.XexpParams(single_snr_min=50.0, **cut))[0]['single_exposure'] == 1
    s = _file_audit(path, params=X.XexpParams(single_snr_min=500.0, **cut))[0]
    assert s['single_exposure'] == 0 and s['snr_min'] == 500.0


# two rows of S/N 100: the 1e5 one is carried by exposure 0 only, the 9e5 one by all three
RANK_ROWS = [(60, 60, 1e5), (120, 120, 9e5)]
RANK_ERR = [1e3, 9e3]


def _rank_field(tmp_path):
    return _field(tmp_path, frames=[RANK_ROWS, [RANK_ROWS[1]], [RANK_ROWS[1]]],
                  rows=RANK_ROWS, err=RANK_ERR)


def _stored(tmp_path, params, drop=()):
    """A merged catalog that carries the audit columns made with ``params``."""
    path, root = _rank_field(tmp_path)
    tab = Table.read(path)
    X.audit_merged_catalog(
        tab, filt='F212N', catalog_dirs=[str(root / 'F212N')], satstar_suffix='_m12',
        params=params, wcs_loader=_wcs, bad_dq_mask=3, log=lambda m: None)
    for key in drop:
        del tab.meta[key]
    stored = str(root / 'catalogs' / 'f212n_stored_o078_indivexp_merged_m2_dao_basic.fits')
    tab.write(stored)
    return stored


def _pool_field(tmp_path):
    """A (60, 60) 9e5, carried by exposure 0 of three; B (120, 120) 1e5, carried
    by all three; C (300, 300) 1e7, inside exposure 0's larger array only."""
    a, b, c = (60, 60, 9e5), (120, 120, 1e5), (300, 300, 1e7)
    return _field(tmp_path, frames=[[a, b, c], [b], [b]], rows=[a, b, c],
                  err=[9e3, 1e3, 1e5], shapes={0: (400, 400)})


def test_the_rank_is_taken_among_the_multi_covered_rows(tmp_path):
    path, root = _pool_field(tmp_path)
    s = _file_audit(path, params=X.XexpParams(single_top_pct=40.0))[0]
    assert s['multi'] == 2 and s['bright_multi'] == 1 and s['single_exposure'] == 1
    # the same through the stored columns: A is in the top 40% of A and B, not of A, B and C
    tab = Table.read(path)
    X.audit_merged_catalog(
        tab, filt='F212N', catalog_dirs=[str(root / 'F212N')], satstar_suffix='_m12',
        params=X.XexpParams(single_top_pct=40.0), wcs_loader=_wcs, bad_dq_mask=3,
        log=lambda m: None)
    stored = str(root / 'catalogs' / 'f212n_stored_o078_indivexp_merged_m2_dao_basic.fits')
    tab.write(stored)
    s = _file_audit(stored)[0]
    assert s['source'] == 'column' and s['top_pct'] == 40.0
    assert s['multi'] == 2 and s['bright_multi'] == 1 and s['single_exposure'] == 1


def test_single_top_pct_changes_the_flag(tmp_path):
    path, _ = _rank_field(tmp_path)
    whole = _file_audit(path, params=X.XexpParams(single_top_pct=100.0))[0]
    assert whole['single_exposure'] == 1 and whole['bright_multi'] == 2
    assert whole['top_pct'] == 100.0
    top = _file_audit(path, params=X.XexpParams(single_top_pct=5.0))[0]
    assert top['single_exposure'] == 0 and top['bright_multi'] == 1
    assert top['bright_one'] == 0 and top['top_pct'] == 5.0


def test_stored_columns_keep_the_rank_cut_they_were_made_with(tmp_path):
    stored = _stored(tmp_path, X.XexpParams(single_top_pct=5.0))
    s = _file_audit(stored, params=X.XexpParams(single_top_pct=100.0))[0]
    assert s['source'] == 'column' and s['top_pct'] == 5.0
    assert s['bright_multi'] == 1 and s['single_exposure'] == 0


def test_a_stored_table_from_before_the_rank_cut_reads_as_having_none(tmp_path):
    stored = _stored(tmp_path, X.XexpParams(single_top_pct=100.0), drop=('XEXPTOP',))
    s = _file_audit(stored, params=X.XexpParams(single_top_pct=5.0))[0]
    assert s['source'] == 'column' and s['top_pct'] == 100.0
    assert s['bright_multi'] == 2 and s['single_exposure'] == 1


def test_recompute_replaces_the_stored_rank_cut_with_the_given_one(tmp_path):
    stored = _stored(tmp_path, X.XexpParams(single_top_pct=5.0))
    s = _file_audit(stored, params=X.XexpParams(single_top_pct=100.0), recompute=True)[0]
    assert s['source'] == 'recomputed' and s['top_pct'] == 100.0
    assert s['bright_multi'] == 2 and s['single_exposure'] == 1


def test_single_top_pct_option_reaches_the_summary(tmp_path):
    path, _ = _rank_field(tmp_path)
    top = tmp_path / 'top'
    assert _run(top, path, '--single-top-pct', 5) == 0
    s = _summary(top)
    assert s['top_pct'][0] == 5.0 and s['single_exposure'][0] == 0
    whole = tmp_path / 'whole'
    assert _run(whole, path) == 0                     # _run's own default: no rank cut
    s = _summary(whole)
    assert s['top_pct'][0] == 100.0 and s['single_exposure'][0] == 1


def test_command_line_defaults_are_the_pipelines(tmp_path):
    ns = A.build_parser().parse_args(['--out', str(tmp_path)])
    assert ns.single_snr == A.MANUAL_DEFAULTS['manual_xexp_single_snr'] == 50.0
    assert ns.single_top_pct == A.MANUAL_DEFAULTS['manual_xexp_single_top_pct'] == 5.0


@pytest.mark.parametrize('pct', ['0', '-1', '100.5'])
def test_a_rank_cut_outside_zero_to_100_stops_the_run_before_anything_is_written(tmp_path, pct):
    path, _ = _rank_field(tmp_path)
    out = tmp_path / 'reports'
    with pytest.raises(ValueError, match='single_top_pct'):
        _run(out, path, '--single-top-pct', pct)
    assert not out.exists()


def test_nfound_is_compared_with_the_merges_own_nmatch(tmp_path):
    # nfound = [3, 1, 1, 2, 0] against nmatch = [3, 2, 1, 1, 1]
    path, _ = _field(tmp_path, nmatch=NMATCH)
    s = _file_audit(path, n_flagged=1)[0]
    assert s['nfound_eq_nmatch'] == pytest.approx(2 / 5)
    assert s['nfound_gt_nmatch'] == pytest.approx(1 / 5)
    assert s['nfound_lt_nmatch'] == pytest.approx(2 / 5)
    assert _file_audit(path, n_flagged=1)[2]['nmatch'][0] == 2


def test_no_nmatch_column_leaves_the_comparison_blank(tmp_path):
    path, _ = _field(tmp_path)
    s = _file_audit(path)[0]
    assert np.isnan([s['nfound_eq_nmatch'], s['nfound_gt_nmatch'],
                     s['nfound_lt_nmatch']]).all()


def test_a_catalog_without_fn_cards_is_reported_not_failed(tmp_path):
    path, _ = _field(tmp_path, with_fn=False)
    s, frames, flagged = _file_audit(path, n_flagged=3)
    assert s['source'] == 'no-fn-cards' and frames is None and len(flagged) == 0
    assert s['frames_newer'] == -1
    assert (s['rows'], s['multi'], s['single_exposure']) == (6, 0, 0)


def test_a_missing_frame_is_counted_as_skipped(tmp_path):
    path, root = _field(tmp_path)
    os.remove(root / 'F212N' / 'f212n_nrcb1_o078_e2_m2_daophot_basic.fits')
    s = _file_audit(path)[0]
    assert (s['frames_read'], s['frames_skipped']) == (2, 1)


def test_frames_written_after_the_merge_are_counted(tmp_path):
    path, root = _field(tmp_path)
    _age(root)
    assert _file_audit(path)[0]['frames_newer'] == 0
    cat = root / 'F212N' / 'f212n_nrcb1_o078_e1_m2_daophot_basic.fits'
    os.utime(cat, (3000, 3000))
    assert _file_audit(path)[0]['frames_newer'] == 1
    _age(root, frames=3000)
    assert _file_audit(path)[0]['frames_newer'] == 3


def test_a_satstar_catalog_written_after_the_merge_counts_too(tmp_path):
    path, root = _field(tmp_path, sat={1: [(60, 60, 1e5)]})
    _age(root)
    assert _file_audit(path)[0]['frames_newer'] == 0
    sat = root / 'F212N' / 'e1_crf_m12_satstar_catalog.fits'
    os.utime(sat, (3000, 3000))
    assert _file_audit(path)[0]['frames_newer'] == 1


def test_a_missing_frame_is_not_a_newer_frame(tmp_path):
    path, root = _field(tmp_path)
    os.remove(root / 'F212N' / 'f212n_nrcb1_o078_e2_m2_daophot_basic.fits')
    _age(root, frames=3000)
    assert _file_audit(path)[0]['frames_newer'] == 2


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def test_run_writes_reports_and_nothing_under_the_product_tree(tmp_path):
    path, root = _field(tmp_path)
    before = _tree(root)
    out = tmp_path / 'reports'
    assert _run(out, path, '--flagged', 3) == 0
    assert _tree(root) == before                    # same files, same bytes
    assert sorted(os.listdir(out)) == [
        'flagged.' + MERGED[:-5] + '.tsv', 'frames.all.tsv', 'summary.all.tsv']
    s = _summary(out)
    assert len(s) == 1 and s['catalog'][0] == MERGED and s['source'][0] == 'recomputed'
    assert (s['multi'][0], s['one'][0], s['single_exposure'][0]) == (5, 2, 1)
    assert s['rows'].dtype.kind == 'i' and s['frac_one'].dtype.kind == 'f'
    f = Table.read(str(out / 'frames.all.tsv'), format='ascii.tab')
    assert len(f) == 3 and set(f['catalog']) == {MERGED}
    fl = Table.read(str(out / ('flagged.' + MERGED[:-5] + '.tsv')), format='ascii.tab')
    assert len(fl) == 1 and fl['flux'][0] == 1e5


def test_no_flagged_files_unless_asked(tmp_path):
    path, _ = _field(tmp_path)
    out = tmp_path / 'reports'
    _run(out, path)
    assert not [f for f in os.listdir(out) if f.startswith('flagged.')]


def test_reports_are_not_replaced_without_overwrite(tmp_path):
    path, _ = _field(tmp_path)
    out = tmp_path / 'reports'
    _run(out, path)
    stamp = (out / 'summary.all.tsv').read_bytes()
    (out / 'summary.all.tsv').write_text('kept\n')


    def boom(crf):
        raise AssertionError('no exposure may be read once the reports are known to clash')

    with pytest.raises(FileExistsError):
        A.main(['--out', str(out), path], wcs_loader=boom, bad_dq_mask=3)
    assert (out / 'summary.all.tsv').read_text() == 'kept\n'
    assert _run(out, path, '--overwrite') == 0
    assert (out / 'summary.all.tsv').read_bytes() == stamp


@pytest.mark.parametrize('where', ['catalogs', 'F212N'])
def test_a_product_directory_is_refused_as_the_report_directory(tmp_path, where):
    path, root = _field(tmp_path)
    before = _tree(root)
    with pytest.raises(ValueError, match='product directory'):
        _run(root / where, path)
    assert _tree(root) == before


def test_a_symlink_to_a_product_directory_is_refused_too(tmp_path):
    path, root = _field(tmp_path)
    link = tmp_path / 'looks_harmless'
    link.symlink_to(root / 'catalogs')
    with pytest.raises(ValueError, match='product directory'):
        _run(link, path)


def test_catalog_dir_option_replaces_the_default_location(tmp_path):
    path, root = _field(tmp_path)
    moved = tmp_path / 'elsewhere'
    moved.mkdir()
    for f in (root / 'F212N').glob('*_daophot_basic.fits'):
        os.rename(f, moved / f.name)              # the crfs stay where the headers say
    out = tmp_path / 'reports'
    assert _run(out, path, '--catalog-dir', moved) == 0
    assert _summary(out)['frames_read'][0] == 3
    out2 = tmp_path / 'reports2'
    _run(out2, path)                              # the default location holds none now
    assert _summary(out2)['frames_read'][0] == 0


def test_a_misnamed_catalog_stops_the_run_before_anything_is_written(tmp_path):
    path, root = _field(tmp_path)
    bad = root / 'catalogs' / 'merged_f212n.fits'
    bad.write_bytes(b'')
    out = tmp_path / 'reports'
    with pytest.raises(ValueError, match='filter'):
        _run(out, path, bad)
    assert not out.exists()


def test_an_unreadable_catalog_is_recorded_and_the_run_goes_on(tmp_path):
    path, root = _field(tmp_path)
    junk = root / 'catalogs' / 'f212n_junk_o078_indivexp_merged_m2_dao_basic.fits'
    junk.write_text('not a fits file\n')
    empty = root / 'catalogs' / 'f212n_empty_o078_indivexp_merged_m2_dao_basic.fits'
    empty.write_bytes(b'')
    gone = root / 'catalogs' / 'f212n_gone_o078_indivexp_merged_m2_dao_basic.fits'
    cut = root / 'catalogs' / 'f212n_cut_o078_indivexp_merged_m2_dao_basic.fits'
    many = [(20 + i % 150, 20 + 10 * (i // 150), 1e5) for i in range(400)]
    _merged([], rows=many, err=[1e3] * 400).write(str(cut))
    cut.write_bytes(cut.read_bytes()[:-2880])                # cut short in its data
    out = tmp_path / 'reports'
    assert _run(out, path, junk, empty, gone, cut) == 0
    s = _summary(out)
    by = {r['catalog']: r for r in s}
    assert by[MERGED]['source'] == 'recomputed' and by[MERGED]['multi'] == 5
    for bad in (junk, empty, gone, cut):
        row = by[bad.name]
        assert row['source'].startswith('unreadable') and row['rows'] == -1
    assert len(s) == 5


def test_satstar_catalogs_use_the_m12_suffix_unless_told_otherwise(tmp_path):
    # R4 (bright, no daophot row anywhere) is carried by exposure 2's satstar fit
    path, root = _field(tmp_path, sat={2: [ROWS[4]]})
    out = tmp_path / 'reports'
    _run(out, path)
    s = _summary(out)
    assert (s['none'][0], s['one'][0], s['single_exposure'][0]) == (0, 3, 2)
    out2 = tmp_path / 'reports2'
    _run(out2, path, '--satstar-suffix', '_other')
    s2 = _summary(out2)
    assert (s2['none'][0], s2['one'][0], s2['single_exposure'][0]) == (1, 2, 1)


def test_a_catalog_given_twice_is_audited_once(tmp_path):
    path, _ = _field(tmp_path)
    out = tmp_path / 'reports'
    _run(out, path, path)
    assert len(_summary(out)) == 1


def test_the_bad_pixel_mask_defaults_to_the_instruments(tmp_path, monkeypatch):
    path, _ = _field(tmp_path)
    asked = []
    monkeypatch.setattr(X, 'bad_dq_mask_for', lambda filt: asked.append(filt) or 3)
    assert A.main(['--out', str(tmp_path / 'reports'), path], wcs_loader=_wcs) == 0
    assert asked == ['F212N']


def test_a_flagged_report_that_exists_stops_the_run_before_any_work(tmp_path):
    path, _ = _field(tmp_path)
    out = tmp_path / 'reports'
    _run(out, path, '--flagged', 2)
    (out / 'summary.all.tsv').unlink()
    (out / 'frames.all.tsv').unlink()

    def boom(crf):
        raise AssertionError('no exposure may be read')

    with pytest.raises(FileExistsError, match='flagged|exist'):
        A.main(['--out', str(out), path, '--flagged', '2'], wcs_loader=boom, bad_dq_mask=3)


def test_write_helpers_do_not_replace_files_unless_asked(tmp_path):
    t = Table({'a': [1, 2]})
    p = str(tmp_path / 'x.tsv')
    A.write_table(t, p, overwrite=False)
    with pytest.raises(FileExistsError):
        A.write_table(t, p, overwrite=False)
    A.write_table(Table({'a': [3]}), p, overwrite=True)
    assert Table.read(p, format='ascii.tab')['a'].tolist() == [3]
    q = str(tmp_path / 'x.md')
    A.write_text('one\n', q, overwrite=False)
    with pytest.raises(FileExistsError):
        A.write_text('two\n', q, overwrite=False)
    assert open(q).read() == 'one\n'
    A.write_text('two\n', q, overwrite=True)
    assert open(q).read() == 'two\n'


def test_a_catalog_without_a_position_column_is_recorded_not_fatal(tmp_path):
    path, root = _field(tmp_path)
    odd = root / 'catalogs' / 'f212n_odd_o078_indivexp_merged_m2_dao_basic.fits'
    tab = Table({'flux': [1.0, 2.0]})
    tab.meta['fn0'] = 'f212n_nrcb1_o078_e0_m2_daophot_basic.fits'
    tab.write(str(odd))
    out = tmp_path / 'reports'
    assert _run(out, path, odd) == 0
    by = {r['catalog']: r for r in _summary(out)}
    assert by[odd.name]['source'] == 'unreadable: KeyError'
    assert by[MERGED]['source'] == 'recomputed'


def test_catalog_list_skips_comments_and_blank_lines(tmp_path):
    path, root = _field(tmp_path)
    lst = tmp_path / 'list.txt'
    lst.write_text(f'# fields\n\n{path}\n   \n  # more\n')
    out = tmp_path / 'reports'
    assert _run(out, '--catalog-list', lst) == 0
    assert len(_summary(out)) == 1


def test_nothing_to_audit_is_an_error_exit(tmp_path, capsys):
    assert _run(tmp_path / 'reports') == 1
    assert 'no catalogs' in capsys.readouterr().err


def test_shards_split_the_sorted_list_and_are_tagged(tmp_path):
    path, root = _field(tmp_path)
    others = []
    for tag in ('b', 'c'):
        p = root / 'catalogs' / f'f212n_{tag}_o078_indivexp_merged_m2_dao_basic.fits'
        _merged([f'f212n_nrcb1_o078_e{i}_m2_daophot_basic.fits' for i in range(3)]).write(str(p))
        others.append(str(p))
    everything = sorted([path] + others)
    out = tmp_path / 'reports'
    for i in range(2):
        assert _run(out, *everything, '--shard', f'{i}/2') == 0
    assert sorted(os.listdir(out)) == ['frames.shard000of002.tsv',
                                       'frames.shard001of002.tsv',
                                       'summary.shard000of002.tsv',
                                       'summary.shard001of002.tsv']
    a = _summary(out, 'shard000of002')
    b = _summary(out, 'shard001of002')
    assert [r['path'] for r in a] == everything[0::2]
    assert [r['path'] for r in b] == everything[1::2]


def test_report_warns_when_frames_are_newer_than_the_merge(tmp_path):
    path, root = _field(tmp_path)
    _age(root, frames=3000)
    out = tmp_path / 'reports'
    text = _report(out, path)
    assert '| frames newer |' in text and '| 3/0 | 3 | 5 |' in text
    assert '**Different runs:** 1 catalog(s)' in text
    assert text.split('**Different runs:**')[1].strip().endswith(f'`{MERGED}`.')
    assert int(_summary(out)['frames_newer'][0]) == 3


def test_report_has_no_warning_when_the_merge_is_the_latest_write(tmp_path):
    path, root = _field(tmp_path)
    _age(root)
    out = tmp_path / 'reports'
    text = _report(out, path)
    assert '| 3/0 | 0 | 5 |' in text and 'Different runs' not in text


def test_report_warning_names_only_the_catalogs_with_newer_frames(tmp_path):
    path, root = _field(tmp_path)
    fresh = root / 'catalogs' / 'f212n_fresh_o078_indivexp_merged_m2_dao_basic.fits'
    _merged([f'f212n_nrcb1_o078_e{i}_m2_daophot_basic.fits' for i in range(3)]).write(str(fresh))
    _age(root, frames=3000, merged=2000)
    os.utime(fresh, (4000, 4000))
    out = tmp_path / 'reports'
    text = _report(out, path, fresh)
    warning = text.split('**Different runs:**')[1]
    assert warning.startswith(' 1 catalog(s)') and MERGED in warning
    assert 'f212n_fresh' not in warning
    news = {str(c): int(n) for c, n in zip(_summary(out)['catalog'],
                                           _summary(out)['frames_newer'])}
    assert news == {MERGED: 3, fresh.name: 0}


def test_report_shows_na_for_stored_columns_and_unreadable_catalogs(tmp_path):
    path, root = _field(tmp_path)
    tab = Table.read(path)
    X.audit_merged_catalog(
        tab, filt='F212N', catalog_dirs=[str(root / 'F212N')], satstar_suffix='_m12',
        params=X.XexpParams(), wcs_loader=_wcs, bad_dq_mask=3, log=lambda m: None)
    stored = root / 'catalogs' / 'f212n_stored_o078_indivexp_merged_m2_dao_basic.fits'
    tab.write(str(stored))
    junk = root / 'catalogs' / 'f212n_junk_o078_indivexp_merged_m2_dao_basic.fits'
    junk.write_text('not a fits file\n')
    _age(root, frames=3000)
    out = tmp_path / 'reports'
    text = _report(out, stored, junk)
    assert 'Different runs' not in text and '| 3/0 | n/a |' in text
    news = {str(c): int(n) for c, n in zip(_summary(out)['catalog'],
                                           _summary(out)['frames_newer'])}
    assert news == {stored.name: -1, junk.name: -1}


def test_collect_merges_the_shards_into_summary_and_report(tmp_path):
    path, root = _field(tmp_path)
    p480 = root / 'catalogs' / 'f480m_merged_o078_indivexp_merged_m2_dao_basic.fits'
    _merged([f'f212n_nrcb1_o078_e{i}_m2_daophot_basic.fits' for i in range(3)]).write(str(p480))
    junk = root / 'catalogs' / 'f212n_junk_o078_indivexp_merged_m2_dao_basic.fits'
    junk.write_text('not a fits file\n')
    _age(root)
    out = tmp_path / 'reports'
    everything = sorted([path, str(p480), str(junk)])
    for i in range(2):                                   # what the array tasks write
        _run(out, *everything, '--shard', f'{i}/2')
    assert A.main(['--out', str(out), '--collect']) == 0
    s = Table.read(str(out / 'summary.tsv'), format='ascii.tab')
    assert len(s) == 3
    assert [str(f) for f in s['filter']] == ['F212N', 'F212N', 'F480M']
    text = (out / 'report.md').read_text()
    assert MERGED in text and 'f480m_merged' in text
    assert text.count('| catalog |') == 1
    assert ('| f212n_junk_o078_indivexp_merged_m2_dao_basic.fits | unreadable: ValueError |'
            + ' |' * 11) in text
    assert ('| 6 | 3/0 | 0 | 5 | 1 (20.0%) | 1 (20.0%) | 2 (40.0%) | 1 (20.0%) |'
            in text)


def test_collect_without_shard_files_is_an_error(tmp_path):
    out = tmp_path / 'reports'
    out.mkdir()
    with pytest.raises(FileNotFoundError):
        A.main(['--out', str(out), '--collect'])


def test_collect_does_not_replace_reports_without_overwrite(tmp_path):
    path, _ = _field(tmp_path)
    out = tmp_path / 'reports'
    _run(out, path)
    assert A.main(['--out', str(out), '--collect']) == 0
    with pytest.raises(FileExistsError):
        A.main(['--out', str(out), '--collect'])
    assert A.main(['--out', str(out), '--collect', '--overwrite']) == 0


def test_collect_writes_nothing_when_only_the_report_would_be_replaced(tmp_path):
    path, _ = _field(tmp_path)
    out = tmp_path / 'reports'
    _run(out, path)
    (out / 'report.md').write_text('kept\n')
    with pytest.raises(FileExistsError):
        A.main(['--out', str(out), '--collect'])
    assert not (out / 'summary.tsv').exists()              # refused before any write
    assert (out / 'report.md').read_text() == 'kept\n'
