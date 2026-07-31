"""The FWHM table is an instrument constant, so a target tree need not carry one.

Before 2026-07-31 all three reduction drivers read
``{basepath}/reduction/fwhm_table.ecsv`` and stopped when it was absent, which
made copying that file a mandatory setup step for every new target -- while the
photometry side had always read the packaged copy.
"""
import os

import pytest

from jwst_gc_pipeline.reduction.fwhm import (PACKAGED, PACKAGED_NIRISS,
                                             fwhm_table_path)


def test_the_packaged_table_is_used_when_the_target_tree_has_none(tmp_path):
    assert fwhm_table_path(str(tmp_path)) == str(PACKAGED)


def test_a_target_copy_wins(tmp_path):
    local = tmp_path / 'reduction' / 'fwhm_table.ecsv'
    local.parent.mkdir()
    local.write_text('placeholder')
    assert fwhm_table_path(str(tmp_path)) == str(local)


def test_no_basepath_gives_the_packaged_table():
    assert fwhm_table_path(None) == str(PACKAGED)


def test_niriss_always_gets_the_niriss_table(tmp_path):
    local = tmp_path / 'reduction' / 'fwhm_table.ecsv'
    local.parent.mkdir()
    local.write_text('placeholder')
    assert fwhm_table_path(str(tmp_path), 'NIRISS') == str(PACKAGED_NIRISS)
    assert fwhm_table_path(str(tmp_path), 'niriss') == str(PACKAGED_NIRISS)


def test_both_packaged_tables_ship():
    assert os.path.exists(PACKAGED)
    assert os.path.exists(PACKAGED_NIRISS)


def test_the_packaged_table_covers_nircam_and_miri():
    Table = pytest.importorskip('astropy.table').Table
    filters = set(Table.read(PACKAGED)['Filter'])
    for filtername in ('F115W', 'F212N', 'F410M', 'F480M', 'F770W', 'F2550W'):
        assert filtername in filters


def test_the_drivers_call_the_helper_rather_than_building_the_path():
    """A driver that rebuilds ``{basepath}/reduction/fwhm_table.ecsv`` itself
    re-imposes the setup step this helper removes."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    drivers = ['jwst_gc_pipeline/reduction/PipelineRerunNIRCAM-LONG.py',
               'jwst_gc_pipeline/reduction/PipelineMIRI.py',
               'jwst_gc_pipeline/reduction/filtering.py']
    for driver in drivers:
        with open(os.path.join(os.path.dirname(here), driver)) as fh:
            source = fh.read()
        assert 'reduction/fwhm_table.ecsv' not in source, driver
        assert 'fwhm_table_path' in source, driver
