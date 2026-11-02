"""The cross-band table's ECSV twin is written for the merged module only.

ECSV is a slow format for these tables: the sgrb2 m7 finalize (2026-10) spent
2.3 h writing the nrca, nrcb and merged cross-band ECSVs against 4.6 min for
the same tables as FITS.  A field whose modules overlap ships only the merged
module's table (``stage_release.catalog_modules``), so the nrca/nrcb ECSVs are
written only with ``--manual-crossband-ecsv-all-modules`` (#1227).  A run with
no merged module (a disjoint field, a MIRI run) writes every module's.  The
FITS table is written for every module either way.
"""
import importlib.util
import inspect
import pathlib
import sys

import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry import cataloging, merge_catalogs
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS

ROOT = pathlib.Path(__file__).resolve().parents[3]


def _stage_release():
    spec = importlib.util.spec_from_file_location(
        '_sr_crossband_ecsv', ROOT / 'scripts' / 'release' / 'stage_release.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['_sr_crossband_ecsv'] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('modules, all_modules, want', [
    (['nrca', 'nrcb', 'merged'], False, ['merged']),
    (('nrca', 'nrcb', 'merged'), False, ['merged']),
    (['merged'], False, ['merged']),
    (['nrca', 'nrcb', 'merged'], True, ['nrca', 'nrcb', 'merged']),
    (['nrca', 'nrcb'], False, ['nrca', 'nrcb']),     # disjoint field
    (['nrcb'], False, ['nrcb']),                     # one-module field
    (['mirimage'], False, ['mirimage']),             # MIRI
])
def test_which_modules_write_the_ecsv_twin(modules, all_modules, want):
    assert cataloging.crossband_ecsv_modules(modules, all_modules) == want


def test_the_nrca_nrcb_ecsv_is_opt_in():
    assert MANUAL_DEFAULTS['manual_crossband_ecsv_all_modules'] is False


def _table():
    return Table({'flux': np.arange(3.0), 'name': ['a', 'b', 'c']})


def test_the_twin_is_written(tmp_path):
    stem = str(tmp_path / 'basic_merged_tbl')
    merge_catalogs._write_ecsv_twin(_table(), stem, True)
    back = Table.read(f'{stem}.ecsv')
    assert list(back['flux']) == [0.0, 1.0, 2.0]
    assert list(back['name']) == ['a', 'b', 'c']


def test_a_skipped_twin_removes_an_earlier_runs_ecsv(tmp_path):
    stem = str(tmp_path / 'basic_nrca_tbl')
    _table().write(f'{stem}.fits')
    (tmp_path / 'basic_nrca_tbl.ecsv').write_text('an earlier run')
    merge_catalogs._write_ecsv_twin(_table(), stem, False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['basic_nrca_tbl.fits']


def test_a_skipped_twin_with_nothing_to_remove(tmp_path):
    merge_catalogs._write_ecsv_twin(_table(), str(tmp_path / 'basic_nrcb_tbl'),
                                    False)
    assert list(tmp_path.iterdir()) == []


def test_every_other_caller_still_writes_the_ecsv():
    for func in (merge_catalogs.merge_catalogs, merge_catalogs.merge_daophot):
        assert inspect.signature(func).parameters['write_ecsv'].default is True


def test_the_choice_reaches_the_writer():
    """run_manual_pipeline -> merge_daophot -> merge_catalogs -> _write_ecsv_twin."""
    def src(func):
        return ' '.join(inspect.getsource(func).split())
    run = src(cataloging.run_manual_pipeline)
    assert ("_xb_ecsv_modules = crossband_ecsv_modules( modules, "
            "getattr(options, 'manual_crossband_ecsv_all_modules', False))") in run
    assert "vetted=True, write_ecsv=(module in _xb_ecsv_modules))" in run
    assert "basepath=basepath, write_ecsv=write_ecsv)" in src(
        merge_catalogs.merge_daophot)
    merge = src(merge_catalogs.merge_catalogs)
    assert "_write_ecsv_twin(basetable, tablename, write_ecsv)" in merge
    assert '.ecsv"' not in merge


def test_a_disjoint_field_ships_its_m8_fits_and_no_m7_ecsv(tmp_path):
    """The default rests on this for arches and quintuplet: their release
    stages the nrca/nrcb tables, but only the highest iteration's, and m8 is
    written as FITS only.  If the release starts shipping a lower iteration's
    ECSV beside m8, the nrca/nrcb ECSVs must be written again by default."""
    sr = _stage_release()
    field = 'arches'
    assert sr.catalog_modules(field) == ['nrca', 'nrcb']
    bases = sr.cat_base(field)
    cat_dir = tmp_path / 'catalogs'
    cat_dir.mkdir()
    for base in bases:
        for tail in ('_resbgsub_m7.fits', '_resbgsub_m7.ecsv',
                     '_resbgsub_m8.fits'):
            (cat_dir / f'{base}{tail}').write_text('x')
    shipped = sorted(pathlib.Path(it['src']).name
                     for it in sr.discover_catalogs({'data_dir': tmp_path}, field)
                     if it.get('kind') == 'catalog_full')
    assert shipped == sorted(f'{base}_resbgsub_m8.fits' for base in bases)
