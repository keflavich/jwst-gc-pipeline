"""The default-CMD-selection flag (#1283): the tests, the writer and its hooks.

``cmd_default`` is the oksep separation test AND ``n_filt_independent >= 2``,
carried as a column on the full m7 / m8 / m8_dedup tables.
"""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from jwst_gc_pipeline.photometry import cataloging
from jwst_gc_pipeline.photometry import cmd_default as CD
from jwst_gc_pipeline.photometry import dedup_catalog
from jwst_gc_pipeline.photometry import merge_catalogs as MC
from jwst_gc_pipeline.photometry import spike_flag

REPO = Path(__file__).resolve().parents[3]


def _table(sep_unit=u.arcsec, n_indep=True):
    """Six rows: (sep_f212n, sep_f480m, sep_f182m) in arcsec, n_filt_independent."""
    seps = np.array([
        [0.01, 0.02, 0.50],   # oksep (2 bands < 0.1), independent in 2
        [0.01, 0.02, 0.50],   # oksep, independent in 1 -> not cmd_default
        [0.30, 0.40, 0.50],   # sep too large, independent in 2 -> not cmd_default
        [0.01, 0.50, 0.50],   # only one band matched
        [np.nan, 0.02, 0.01],  # NaN counts as unmatched; two others pass
        [0.01, np.nan, np.nan],  # one pass
    ])
    t = Table()
    t['sep_f212n'] = (seps[:, 0] * u.arcsec).to(sep_unit)
    t['sep_f480m'] = (seps[:, 1] * u.arcsec).to(sep_unit)
    t['sep_f182m'] = (seps[:, 2] * u.arcsec).to(sep_unit)
    t['sep_f200w'] = np.zeros(len(t)) * sep_unit    # wide band: ignored
    t['flux'] = np.arange(len(t), dtype=float)
    if n_indep:
        t['n_filt_independent'] = np.array([2, 1, 2, 2, 2, 2], dtype=np.int16)
    return t


@pytest.mark.parametrize('unit', [u.arcsec, u.deg])
def test_compute_cmd_default_intersection_in_either_unit(unit):
    oksep, cmd = MC.compute_cmd_default(_table(unit))
    assert list(oksep) == [True, True, False, False, True, False]
    assert list(cmd) == [True, False, False, False, True, False]


def test_wide_band_separation_is_ignored():
    t = _table()
    # the wide band is zero everywhere: counted, row 3 and 5 would pass
    oksep, _ = MC.compute_cmd_default(t)
    assert not oksep[3] and not oksep[5]


def test_masked_separation_counts_as_unmatched():
    t = Table(masked=True)
    t['sep_f212n'] = np.ma.array([0.01, 0.01], mask=[True, False]) * 1.0
    t['sep_f212n'].unit = u.arcsec
    t['sep_f480m'] = np.array([0.02, 0.02])
    t['sep_f480m'].unit = u.arcsec
    t['n_filt_independent'] = [2, 2]
    oksep, cmd = MC.compute_cmd_default(t)
    assert list(oksep) == [False, True] and list(cmd) == [False, True]


def test_without_n_filt_independent_the_oksep_mask_is_returned():
    t = _table(n_indep=False)
    oksep, cmd = MC.compute_cmd_default(t)
    assert list(cmd) == list(oksep)
    assert MC.cmd_default_rule(t) == 'oksep'
    assert MC.cmd_default_rule(_table()) == 'oksep&n_filt_independent>=2'


def test_oksep_matches_the_mask_merge_catalogs_wrote():
    """compute_oksep is the cut ``merge_catalogs`` applied inline before."""
    t = _table()
    old = (np.array([t[c] < 0.1 * u.arcsec for c in MC._oksep_sep_cols(t.colnames)])
           .sum(axis=0) > 1)
    assert list(MC.compute_oksep(t)) == list(old)


def test_fewer_than_two_sep_columns_gives_an_all_true_oksep():
    t = Table({'sep_f212n': [0.5, 0.6] * u.arcsec})
    assert list(MC.compute_oksep(t)) == [True, True]


# ---------------------------------------------------------------------------
# the writer
# ---------------------------------------------------------------------------
def _write_fits(path, table, primary=None):
    table.write(path, overwrite=True)
    if primary:
        with fits.open(path, mode='update') as hdul:
            for k, v in primary.items():
                hdul[0].header[k] = v


def test_annotate_adds_columns_and_card_and_keeps_headers(tmp_path):
    t = _table(u.deg)
    t['skycoord_ref'] = SkyCoord(ra=np.arange(6) * 0.001 * u.deg,
                                 dec=np.zeros(6) * u.deg)
    t.meta['ROLLCTAB'] = 'sha1:aaaaaaaaaaaa'
    p = tmp_path / 'basic_merged_x_m8_dedup_o001.fits'
    _write_fits(p, t, primary={'GCTAG': 'tag-1'})
    n, n_ok, n_cmd = CD.annotate_cmd_default(str(p))
    assert (n, n_ok, n_cmd) == (6, 3, 2)
    with fits.open(p) as hdul:
        assert hdul[0].header['GCTAG'] == 'tag-1'
        assert hdul[1].header['ROLLCTAB'] == 'sha1:aaaaaaaaaaaa'
        assert hdul[1].header['CMDDEF'] == 'oksep&n_filt_independent>=2'
    out = Table.read(p)
    assert len(out) == 6
    assert list(out['cmd_default']) == [True, False, False, False, True, False]
    assert list(out['oksep_pass']) == [True, True, False, False, True, False]
    assert out['sep_f212n'].unit == u.deg
    assert 'skycoord_ref' in out.colnames
    assert not [f for f in os.listdir(tmp_path) if f.startswith('.')]


def test_annotate_is_idempotent_and_recomputes(tmp_path):
    p = tmp_path / 'basic_merged_x_m8_o001.fits'
    _write_fits(p, _table())
    CD.annotate_cmd_default(str(p))
    first = Table.read(p)
    CD.annotate_cmd_default(str(p))
    second = Table.read(p)
    assert first.colnames == second.colnames
    assert list(first['cmd_default']) == list(second['cmd_default'])
    # a changed n_filt_independent is followed by one more call
    second['n_filt_independent'][1] = 2
    _write_fits(p, second)
    CD.annotate_cmd_default(str(p))
    assert Table.read(p)['cmd_default'][1]


def test_annotate_with_subsets_covers_every_qualcuts_file(tmp_path):
    stem = tmp_path / 'basic_merged_x_m7_o001'
    _write_fits(f'{stem}.fits', _table())
    _write_fits(f'{stem}_qualcuts.fits', _table())
    _write_fits(f'{stem}_qualcuts_oksep10678.fits', _table()[[0, 1, 4]])
    done = CD.annotate_cmd_default_with_subsets(f'{stem}.fits')
    assert sorted(os.path.basename(p) for p in done) == [
        'basic_merged_x_m7_o001.fits', 'basic_merged_x_m7_o001_qualcuts.fits',
        'basic_merged_x_m7_o001_qualcuts_oksep10678.fits']
    sub = Table.read(f'{stem}_qualcuts_oksep10678.fits')
    assert list(sub['cmd_default']) == [True, False, True]


# ---------------------------------------------------------------------------
# hooks
# ---------------------------------------------------------------------------
@pytest.fixture
def m8_pair(tmp_path, monkeypatch):
    """A real m8 FITS table; the fake dedup copies it to the dedup sibling."""
    cat = tmp_path / 'catalogs'
    cat.mkdir()
    path = cat / 'basic_merged_indivexp_photometry_tables_merged_resbgsub_m8.fits'
    _write_fits(path, _table())

    def fake_dedup(src, dst, **kw):
        _write_fits(dst, _table()[[0, 1, 2, 4]])
    monkeypatch.setattr(dedup_catalog, 'dedup_merged_catalog', fake_dedup)
    monkeypatch.setattr(spike_flag, 'flag_m8_spike_artifacts',
                        lambda path, basepath, bands=None: None)
    monkeypatch.delenv('M8_CMD_DEFAULT', raising=False)
    return str(path)


def test_maybe_dedup_m8_flags_both_tables_and_keeps_rows(m8_pair):
    out = cataloging._maybe_dedup_m8(m8_pair, SimpleNamespace(), label='t')
    assert out.endswith('_m8_dedup.fits')
    dedup = Table.read(out)
    assert len(dedup) == 4
    assert list(dedup['cmd_default']) == [True, False, False, True]
    raw = Table.read(m8_pair)
    assert len(raw) == 6 and 'cmd_default' in raw.colnames


def test_env_switch_leaves_the_tables_unflagged(m8_pair, monkeypatch):
    monkeypatch.setenv('M8_CMD_DEFAULT', '0')
    out = cataloging._maybe_dedup_m8(m8_pair, SimpleNamespace(), label='t')
    assert 'cmd_default' not in Table.read(out).colnames
    assert 'cmd_default' not in Table.read(m8_pair).colnames


def test_unreadable_table_is_logged_and_does_not_raise(tmp_path, capsys):
    bad = tmp_path / 'bad_m8.fits'
    bad.write_bytes(b'not fits')
    assert cataloging._maybe_annotate_cmd_default([str(bad)], 't') == []
    assert 'cmd_default flag FAILED' in capsys.readouterr().out


def test_m8_merge_partials_flags_the_merged_and_dedup_tables():
    """Source pin: the partial-merge script calls dedup_merged_catalog itself, so it needs its own hook."""
    src = (REPO / 'scripts' / 'reduction' / 'm8_merge_partials.py').read_text()
    assert src.index('annotate_cmd_default(out_path)') < src.index('dedup_merged_catalog(out_path')
    assert (src.index('flag_m8_spike_artifacts(dedup_path, basepath)')
            < src.index('annotate_cmd_default(dedup_path)'))


def test_m7_block_flags_after_propagation_and_before_the_ecsv():
    src = Path(cataloging.__file__).read_text()
    prop = src.index('propagate_independent_detection(_xb)')
    flag = src.index('_maybe_annotate_cmd_default(\n                [_xb]')
    ecsv = src.index('write_crossband_ecsv(_xb)')
    assert prop < flag < ecsv


# ---------------------------------------------------------------------------
# the backfill script
# ---------------------------------------------------------------------------
def _backfill_module():
    spec = importlib.util.spec_from_file_location(
        'add_cmd_default_flag', REPO / 'scripts' / 'reduction' / 'add_cmd_default_flag.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_backfill_selects_tables_and_skips_partials_and_other_files(tmp_path):
    mod = _backfill_module()
    base = 'basic_merged_indivexp_photometry_tables_merged_resbgsub'
    wanted = [f'{base}_hybpsf_m7_o107.fits',
              f'{base}_hybpsf_m7_o107_qualcuts_oksep10678.fits',
              f'{base}_m8_o107.fits', f'{base}_m8_dedup_o107.fits',
              f'{base}_m8.fits', f'{base}_m8_dedup.fits']
    skipped = [f'{base}_m8_partial_F115W.fits', f'{base}_m8_dedup_20260713.fits',
               f'{base}_m8_o107.fits.prov.json', 'crossband_seed_manual_o107.fits',
               f'{base}_m7_o107.ecsv']
    for name in wanted + skipped:
        (tmp_path / name).write_bytes(b'')
    got = [os.path.basename(p) for p in mod.select_tables([str(tmp_path)])]
    assert sorted(got) == sorted(wanted)


def test_backfill_dry_run_writes_nothing_and_apply_flags(tmp_path, capsys):
    mod = _backfill_module()
    p = tmp_path / 'basic_merged_x_resbgsub_m8_dedup_o107.fits'
    _write_fits(p, _table(u.deg), primary={'GCTAG': 'tag-1'})
    before = p.read_bytes()
    assert mod.main(['--catdir', str(tmp_path)]) == 0
    assert p.read_bytes() == before
    assert '[DRY]' in capsys.readouterr().out
    assert mod.main(['--catdir', str(tmp_path), '--apply']) == 0
    out = Table.read(p)
    assert int(out['cmd_default'].sum()) == 2
    # a second run skips it
    mod.main(['--catdir', str(tmp_path), '--apply'])
    assert 'SKIP (already flagged' in capsys.readouterr().out


def test_backfill_refreshes_an_existing_sidecar_keeping_tag_and_stage(tmp_path):
    from jwst_gc_pipeline.versioning import prov_sidecar
    mod = _backfill_module()
    p = tmp_path / 'basic_merged_x_resbgsub_m8_o107.fits'
    _write_fits(p, _table())
    rec = prov_sidecar.build_record(
        'm8', 'tag-old', {'data': 'old', 'wcs': None, 'meta': 'old'},
        env={'a': 1}, code='c', params='p', upstream={'m7': {'data': 'u'}})
    prov_sidecar.write_sidecar(str(p), rec)
    mod.main(['--catdir', str(tmp_path), '--apply'])
    new = prov_sidecar.read_sidecar(str(p))
    assert new['tag'] == 'tag-old' and new['stage'] == 'm8'
    assert new['inputs'] == rec['inputs']
    assert new['outputs']['data'] != 'old'
    assert new['extra']['cmd_default_backfill'] is True


# ---------------------------------------------------------------------------
# CMD script
# ---------------------------------------------------------------------------
def test_cmd_script_catalog_path_stage_choice():
    cmds = pytest.importorskip('jwst_gc_pipeline.plotting.make_all_cmds_m7')
    m8 = cmds.catalog_path('/b', 'qualcuts', field='brick', stage='m8_dedup')
    assert m8 == ('/b/catalogs/basic_merged_indivexp_photometry_tables_'
                  'merged_resbgsub_m8_dedup.fits')
    # default stage still reads the m7 table
    assert cmds.catalog_path('/b', 'basic', field='brick').endswith('resbgsub_m7.fits')
    with pytest.raises(ValueError, match='stage'):
        cmds.catalog_path('/b', 'basic', stage='m9')
