"""sgra's release ships its catalogs (#602).

sgra carried ``skip_catalogs: True`` from the period when it sat ~14.8" off for
want of an ``ALIGNMENT_CONFIG`` entry and its F115W mosaics were stale-tagged by
the m2 astrometry checkpoint.  The field has since been re-drizzled, re-tied and
re-catalogued, so the flag stopped protecting anything and started suppressing
nine deliverables while the staging run still reported success -- the failure
mode is silent, which is why it is pinned by a test rather than by the comment
alone.

The assertion is on BEHAVIOUR (``build_manifest`` emits catalog items for sgra),
not only on the config key: a later refactor that reintroduces the suppression
by another route should fail here too.
"""
import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _stage_release():
    spec = importlib.util.spec_from_file_location(
        '_sr_sgra', ROOT / 'scripts' / 'release' / 'stage_release.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['_sr_sgra'] = mod
    spec.loader.exec_module(mod)
    return mod


# The names sgra's cataloging actually wrote (2026-08-30/31), one filter's worth.
M8 = 'basic_merged_indivexp_photometry_tables_merged_resbgsub_m8.fits'
VETTED = 'f115w_merged_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits'
MOSAIC = 'jw01939-o001_t001_nircam_clear-f115w-merged_i2d.fits'


@pytest.fixture
def sgra_on_disk(tmp_path, monkeypatch):
    """A miniature sgra data dir, wired into the real FIELDS entry."""
    sr = _stage_release()
    cats = tmp_path / 'catalogs'
    cats.mkdir()
    (cats / M8).write_bytes(b'')
    (cats / VETTED).write_bytes(b'')
    pipe = tmp_path / 'F115W' / 'pipeline'
    pipe.mkdir(parents=True)
    (pipe / MOSAIC).write_bytes(b'')
    monkeypatch.setitem(sr.FIELDS['sgra'], 'data_dir', tmp_path)
    return sr


def test_sgra_does_not_declare_skip_catalogs(sgra_on_disk):
    assert sgra_on_disk.FIELDS['sgra'].get('skip_catalogs') is not True


def test_sgra_manifest_carries_its_catalogs(sgra_on_disk):
    items = sgra_on_disk.build_manifest('sgra', 'v0.0-test', exposures=False)
    names = {pathlib.Path(i['src']).name
             for i in items if i.get('category') == 'catalog'}
    assert M8 in names, 'the combined merged table must ship'
    assert VETTED in names, 'the per-filter vetted catalog must ship'


def test_sgra_ships_merged_tables_beside_its_merged_mosaics(sgra_on_disk):
    """The default ``catalog_modules`` is the one that matches what sgra images.

    sgra's modules are disjoint, but it drizzles and catalogs ``merged`` too, so
    the merged tables are the products that pair with the merged mosaics it
    ships.  A per-module key here would stage two non-overlapping halves with no
    image beside either.
    """
    sr = sgra_on_disk
    assert sr.catalog_modules('sgra') == ['merged']
    items = sr.build_manifest('sgra', 'v0.0-test', exposures=False)
    pairs, unpaired = sr.same_run_pairs(items)
    assert unpaired == [], f'every shipped mosaic needs a catalog: {unpaired}'
    assert pairs, 'the merged mosaic must pair with the merged catalog'
