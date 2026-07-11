"""Unit tests for the operational scripts in scripts/reduction/ (not part of
the package; imported by path)."""
import importlib.util
import os
import time

SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'reduction')


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(SCRIPTS, f'{name}.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rename_stale_band_token():
    m = _load('rename_stale_mosaics')
    assert m.band_of('jw02221-o001_t001_nircam_clear-f182m-merged-reproject-vvv_i2d.fits') == 'f182m'
    assert m.band_of('jw02221-o002_t001_miri_f2550w_realigned-to-vvv.fits') == 'f2550w'
    assert m.band_of('jw02221-o001_t001_nircam_clear-F405N-merged_realigned-to-vvv.fits') == 'f405n'
    assert m.band_of('no_band_here.fits') is None


def test_rename_stale_staleness_logic(tmp_path):
    """A pre-campaign realigned mosaic is renamed; a same-campaign one is kept."""
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    pipe = tmp_path / 'myfield' / 'F182M' / 'pipeline'
    pipe.mkdir(parents=True)
    ref = pipe / 'jw1-o001_t001_nircam_clear-f182m-merged_data_i2d.fits'
    stale = pipe / 'jw1-o001_t001_nircam_clear-f182m-merged-reproject-vvv_i2d.fits'
    fresh = pipe / 'jw1-o001_t001_nircam_clear-f182m-merged_realigned-to-refcat.fits'
    now = time.time()
    for p, age_days in ((ref, 0), (stale, 400), (fresh, 0.5)):
        p.write_bytes(b'x')
        os.utime(p, (now - age_days * 86400,) * 2)
    plan = m.rename_stale_for_field('myfield', execute=True)
    assert len(plan) == 1
    assert not stale.exists()
    assert (str(stale) + m.SUFFIX) == str(stale) + '_badastrometry_stale'
    assert os.path.exists(str(stale) + m.SUFFIX)
    assert fresh.exists()


def test_purge_satstar_caches(tmp_path):
    m = _load('purge_satstar_caches')
    pipe = tmp_path / 'brick' / 'F182M' / 'pipeline'
    cats = tmp_path / 'brick' / 'catalogs'
    pipe.mkdir(parents=True)
    cats.mkdir(parents=True)
    a = pipe / 'exp1_m12_satstar_catalog.fits'
    b = cats / 'f182m_consolidated_satstar_catalog.fits'
    other = pipe / 'exp1_m12_daophot_basic.fits'
    for p in (a, b, other):
        p.write_bytes(b'x')
    # dry run: nothing moves
    n = m.purge(str(tmp_path), 'brick', ['F182M'], execute=False)
    assert n == 2 and a.exists() and b.exists()
    # execute: both cache levels sidelined, unrelated file untouched
    n = m.purge(str(tmp_path), 'brick', ['F182M'], execute=True)
    assert n == 2
    assert not a.exists() and not b.exists()
    assert os.path.exists(str(a) + m.SUFFIX) and os.path.exists(str(b) + m.SUFFIX)
    assert other.exists()
    # idempotent: second execute finds nothing
    assert m.purge(str(tmp_path), 'brick', ['F182M'], execute=True) == 0
