"""Tests for the 10678 all-tile JOINT catalog builder (synthetic, fast)."""
import os

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table

from jwst_gc_pipeline.cmz import treasury_joint_catalog as TJC


def _write_tile(path, ra, dec, f212=None, f480=None, ef212=None,
                roll_corrected=None, gctag='2026-09-24_PR959'):
    n = len(ra)
    t = Table()
    t['skycoord_ref'] = SkyCoord(ra=np.asarray(ra) * u.deg,
                                 dec=np.asarray(dec) * u.deg)
    t['flux_jy_f212n'] = np.asarray(f212 if f212 is not None else [1.0] * n,
                                    dtype=float)
    t['flux_jy_f480m'] = np.asarray(f480 if f480 is not None else [1.0] * n,
                                    dtype=float)
    t['eflux_jy_f212n'] = np.asarray(ef212 if ef212 is not None else [0.1] * n,
                                     dtype=float)
    t['some_other_col'] = np.arange(n)
    t.meta['GCTAG'] = gctag
    if roll_corrected is not None:
        t.meta['ROLLCCAT'] = bool(roll_corrected)
    t.write(path, overwrite=True)


def _grid_tile_coords(ra0, dec0, nx=6, ny=6, step_arcsec=5.0):
    step_deg = step_arcsec / 3600.0
    ra, dec = np.meshgrid(ra0 + np.arange(nx) * step_deg,
                          dec0 + np.arange(ny) * step_deg)
    return ra.ravel(), dec.ravel()


# ---------------------------------------------------------------------------
# discovery / schema helpers
# ---------------------------------------------------------------------------
def test_discover_tile_paths(tmp_path):
    for o in ('040', '041', '099'):
        _write_tile(tmp_path / f'basic_merged_m7_o{o}_qualcuts_oksep10678.fits',
                   *_grid_tile_coords(266.3 + int(o) * 0.001, -29.1))
    paths = TJC.discover_tile_paths(str(tmp_path))
    assert set(paths) == {'040', '041', '099'}
    sub = TJC.discover_tile_paths(str(tmp_path), obsids=['040', '041'])
    assert set(sub) == {'040', '041'}


def test_discover_tile_paths_empty_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TJC.discover_tile_paths(str(tmp_path))


def test_discover_tile_paths_excludes_module_subset_siblings(tmp_path):
    # Real layout: each obsid has THREE '..._qualcuts_oksep10678.fits' files --
    # 'basic_merged_...' (both modules) plus 'basic_nrca_...'/'basic_nrcb_...'
    # (single-module SUBSETS of the SAME pointing, not separate pointings).
    # A naive glob matches all three per obsid and, keyed by obsid, silently
    # keeps whichever sorts last ('nrcb' > 'nrca' > 'merged'), so the smoke
    # test against real 10678 data read back a single-module partial catalog
    # (35631 rows) instead of the full pointing (62933 rows) for o040.
    ra, dec = _grid_tile_coords(266.30, -29.10, nx=6, ny=6)
    for stem in ('basic_merged_indivexp_photometry_tables_merged_resbgsub_m7',
                'basic_nrca_indivexp_photometry_tables_merged_resbgsub_m7',
                'basic_nrcb_indivexp_photometry_tables_merged_resbgsub_m7'):
        _write_tile(tmp_path / f'{stem}_o040_qualcuts_oksep10678.fits', ra, dec)
    paths = TJC.discover_tile_paths(str(tmp_path))
    assert set(paths) == {'040'}   # one pointing, not double/triple-counted
    assert os.path.basename(paths['040']).startswith('basic_merged_')


def test_discover_tile_paths_module_subset_alone_is_invisible(tmp_path):
    # If ONLY the module-subset files exist for an obsid (no 'basic_merged_'
    # sibling), that obsid must not appear at all -- it is not a usable
    # pointing for this module, not a fallback to a partial catalog.
    ra, dec = _grid_tile_coords(266.30, -29.10, nx=6, ny=6)
    _write_tile(
        tmp_path / 'basic_nrca_indivexp_photometry_tables_merged_resbgsub_m7_o099_qualcuts_oksep10678.fits',
        ra, dec)
    with pytest.raises(FileNotFoundError):
        TJC.discover_tile_paths(str(tmp_path))


def test_discover_bands():
    assert TJC.discover_bands(
        ['flux_jy_f212n', 'flux_jy_f480m', 'eflux_jy_f212n', 'qfit']
    ) == ['f212n', 'f480m']
    with pytest.raises(KeyError):
        TJC.discover_bands(['qfit', 'ra', 'dec'])


def test_header_roll_corrected(tmp_path):
    p_raw = tmp_path / 'a.fits'
    p_rolled = tmp_path / 'b.fits'
    _write_tile(p_raw, *_grid_tile_coords(266.3, -29.1))
    _write_tile(p_rolled, *_grid_tile_coords(266.4, -29.1), roll_corrected=True)
    assert TJC._header_roll_corrected(str(p_raw)) is False
    assert TJC._header_roll_corrected(str(p_rolled)) is True


def test_assert_no_mixed_roll_correction_raises():
    with pytest.raises(TJC.MixedRollCorrectionError):
        TJC.assert_no_mixed_roll_correction({'040': True, '041': False})
    TJC.assert_no_mixed_roll_correction({'040': True, '041': True})  # no raise
    TJC.assert_no_mixed_roll_correction({'040': False, '041': False})  # no raise


# ---------------------------------------------------------------------------
# edge distance
# ---------------------------------------------------------------------------
def test_edge_distance_interior_beats_corner():
    ra = np.array([0.0, 0.0, 0.002, 0.002, 0.001])
    dec = np.array([0.0, 0.002, 0.0, 0.002, 0.001])
    d = TJC.edge_distance_arcsec(ra, dec)
    assert d[-1] > d[0]          # the interior point beats a corner
    assert d[-1] > 0


def test_edge_distance_degenerate_collinear_falls_back():
    ra = np.array([0.0, 0.001, 0.002, 0.003])
    dec = np.array([0.0, 0.0, 0.0, 0.0])
    d = TJC.edge_distance_arcsec(ra, dec)   # must not raise
    assert len(d) == 4


# ---------------------------------------------------------------------------
# dedup winner rule (via dedup_match_table, on a synthetic match table)
# ---------------------------------------------------------------------------
def _match_table(ra, dec, obsid, n_bands, edge_dist, f212_err):
    t = Table()
    n = len(ra)
    t['ra'] = np.asarray(ra, float)
    t['dec'] = np.asarray(dec, float)
    t['obsid'] = np.asarray(obsid, dtype=object)
    t['n_bands'] = np.asarray(n_bands, dtype=int)
    t['edge_dist'] = np.asarray(edge_dist, dtype=float)
    t['f212_err'] = np.asarray(f212_err, dtype=float)
    t['tile_row'] = np.arange(n, dtype=np.int64)
    return t


def test_winner_rule_most_bands_wins():
    mt = _match_table(
        ra=[266.5, 266.50001], dec=[-29.0, -29.0], obsid=['040', '041'],
        n_bands=[1, 2], edge_dist=[10.0, 10.0], f212_err=[0.01, 0.01])
    keep, also_in = TJC.dedup_match_table(mt, dedup_radius_arcsec=0.2)
    assert list(keep) == [False, True]
    assert also_in[1] == '040'


def test_winner_rule_edge_distance_breaks_band_tie():
    mt = _match_table(
        ra=[266.5, 266.50001], dec=[-29.0, -29.0], obsid=['040', '041'],
        n_bands=[2, 2], edge_dist=[3.0, 30.0], f212_err=[0.01, 0.01])
    keep, also_in = TJC.dedup_match_table(mt, dedup_radius_arcsec=0.2)
    assert list(keep) == [False, True]   # row 1: larger edge distance wins


def test_winner_rule_f212_err_breaks_band_and_edge_tie():
    mt = _match_table(
        ra=[266.5, 266.50001], dec=[-29.0, -29.0], obsid=['040', '041'],
        n_bands=[2, 2], edge_dist=[10.0, 10.0], f212_err=[0.05, 0.01])
    keep, also_in = TJC.dedup_match_table(mt, dedup_radius_arcsec=0.2)
    assert list(keep) == [False, True]   # row 1: lower F212N flux err wins


def test_winner_rule_row_index_final_tiebreak():
    mt = _match_table(
        ra=[266.5, 266.50001], dec=[-29.0, -29.0], obsid=['040', '041'],
        n_bands=[2, 2], edge_dist=[10.0, 10.0], f212_err=[0.01, 0.01])
    keep, also_in = TJC.dedup_match_table(mt, dedup_radius_arcsec=0.2)
    assert list(keep) == [True, False]   # all tied -> lowest row index wins


def test_within_tile_pairs_never_merge():
    mt = _match_table(
        ra=[266.5, 266.50001], dec=[-29.0, -29.0], obsid=['040', '040'],
        n_bands=[2, 1], edge_dist=[10.0, 1.0], f212_err=[0.01, 0.05])
    keep, also_in = TJC.dedup_match_table(mt, dedup_radius_arcsec=0.2)
    assert list(keep) == [True, True]
    assert list(also_in) == ['', '']


# ---------------------------------------------------------------------------
# overlap gate refusal (mocked measurement -- do not run the real estimator
# here, just the refuse/override wiring)
# ---------------------------------------------------------------------------
def test_overlap_gate_refuses_on_bad_pair(monkeypatch):
    def fake_pairwise(groups, tol_mas, context=''):
        return [dict(a='040', b='041', overlap=True, off_mas=55.0, ok=False)]
    monkeypatch.setattr(TJC, 'pairwise_overlap_offsets', fake_pairwise)
    mt = Table()
    mt['ra'] = [266.5, 266.6]
    mt['dec'] = [-29.0, -29.0]
    mt['obsid'] = np.array(['040', '041'], dtype=object)
    with pytest.raises(TJC.OverlapGateError, match='040'):
        TJC.run_overlap_gate(mt, tol_mas=30.0)


def test_overlap_gate_override_with_reason_does_not_raise(monkeypatch):
    def fake_pairwise(groups, tol_mas, context=''):
        return [dict(a='040', b='041', overlap=True, off_mas=55.0, ok=False)]
    monkeypatch.setattr(TJC, 'pairwise_overlap_offsets', fake_pairwise)
    mt = Table()
    mt['ra'] = [266.5, 266.6]
    mt['dec'] = [-29.0, -29.0]
    mt['obsid'] = np.array(['040', '041'], dtype=object)
    results = TJC.run_overlap_gate(
        mt, tol_mas=30.0,
        allow_overlap_fail_reason='known MIRI parallel, tracked in #956')
    assert results[0]['off_mas'] == 55.0


def test_overlap_gate_passes_clean_pair(monkeypatch):
    def fake_pairwise(groups, tol_mas, context=''):
        return [dict(a='040', b='041', overlap=True, off_mas=5.0, ok=True)]
    monkeypatch.setattr(TJC, 'pairwise_overlap_offsets', fake_pairwise)
    mt = Table()
    mt['ra'] = [266.5, 266.6]
    mt['dec'] = [-29.0, -29.0]
    mt['obsid'] = np.array(['040', '041'], dtype=object)
    results = TJC.run_overlap_gate(mt, tol_mas=30.0)
    assert results[0]['ok']


# ---------------------------------------------------------------------------
# output refusal
# ---------------------------------------------------------------------------
def test_refuses_forbidden_output_prefix():
    with pytest.raises(TJC.OutputRefusedError, match='937'):
        TJC._assert_output_allowed(
            TJC.FORBIDDEN_OUTPUT_PREFIX + 'gc-treasury/catalogs_joint',
            [TJC.FORBIDDEN_OUTPUT_PREFIX + 'x.fits'])


def test_refuses_to_overwrite_existing_output(tmp_path):
    existing = tmp_path / 'gctreasury10678_joint_nircam.fits'
    existing.write_text('placeholder')
    with pytest.raises(TJC.OutputRefusedError, match='overwrite'):
        TJC._assert_output_allowed(str(tmp_path), [str(existing)])


def test_write_joint_catalog_roundtrip(tmp_path):
    t = Table()
    t['ra'] = [266.5, 266.6]
    t['dec'] = [-29.0, -29.0]
    t['flux_jy_f212n'] = [1.0, 2.0]
    prov = dict(n_rows_total=2, n_tiles=1)
    out_dir = str(tmp_path / 'catalogs_joint')
    written = TJC.write_joint_catalog(t, prov, out_dir, 'smoketest',
                                      formats=('fits',))
    assert any(w.endswith('.fits') for w in written)
    assert any(w.endswith('.prov.json') for w in written)
    import json
    with open([w for w in written if w.endswith('.prov.json')][0]) as fh:
        back = json.load(fh)
    assert back['n_rows_total'] == 2
    # a second call to the SAME version-tag must refuse
    with pytest.raises(TJC.OutputRefusedError):
        TJC.write_joint_catalog(t, prov, out_dir, 'smoketest', formats=('fits',))


# ---------------------------------------------------------------------------
# end-to-end on tiny synthetic tiles (no real data; exercises the whole path
# except the real `measure_offset` astrometric gate, which is unit-tested in
# jwst_gc_pipeline/photometry/tests/ already -- here we just check the wiring
# by using tiles far enough apart that the gate's footprint-intersection step
# itself finds nothing to measure, i.e. gate passes trivially).
# ---------------------------------------------------------------------------
def test_build_joint_catalog_end_to_end_no_overlap(tmp_path):
    in_dir = tmp_path / 'tiles'
    in_dir.mkdir()
    ra0, dec0 = _grid_tile_coords(266.30, -29.10, nx=8, ny=8, step_arcsec=3.0)
    ra1, dec1 = _grid_tile_coords(266.50, -29.30, nx=8, ny=8, step_arcsec=3.0)
    _write_tile(in_dir / 'basic_merged_m7_o040_qualcuts_oksep10678.fits',
               ra0, dec0, f212=[1.0] * 64, f480=[1.0] * 64)
    _write_tile(in_dir / 'basic_merged_m7_o041_qualcuts_oksep10678.fits',
               ra1, dec1, f212=[1.0] * 64, f480=[1.0] * 64)
    table, prov = TJC.build_joint_catalog(str(in_dir))
    assert prov['n_tiles'] == 2
    assert len(table) == 128   # disjoint tiles -> nothing deduped
    assert prov['n_clusters'] == 0
    assert prov['roll_corrected'] is False
    assert 'joint_tile' in table.colnames
    assert 'also_in_tiles' in table.colnames


def test_build_joint_catalog_refuses_mixed_roll_correction(tmp_path):
    in_dir = tmp_path / 'tiles'
    in_dir.mkdir()
    ra0, dec0 = _grid_tile_coords(266.30, -29.10, nx=4, ny=4, step_arcsec=3.0)
    ra1, dec1 = _grid_tile_coords(266.50, -29.30, nx=4, ny=4, step_arcsec=3.0)
    _write_tile(in_dir / 'basic_merged_m7_o040_qualcuts_oksep10678.fits',
               ra0, dec0, roll_corrected=False)
    _write_tile(in_dir / 'basic_merged_m7_o041_qualcuts_oksep10678.fits',
               ra1, dec1, roll_corrected=True)
    with pytest.raises(TJC.MixedRollCorrectionError):
        TJC.build_joint_catalog(str(in_dir))
