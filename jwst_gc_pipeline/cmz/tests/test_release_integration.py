"""Tests for the CMZ release orchestrator + webpage Aladin pane.

These live in scripts/release (not importable as a package), so import them from
their file paths.
"""
import importlib.util
import os

import pytest

# .../jwst_gc_pipeline/cmz/tests/test_release_integration.py -> repo root (4 up)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
_REL = os.path.join(_REPO, 'scripts', 'release')


def _load(name, path):
    import sys
    if _REL not in sys.path:            # scripts/release siblings import each other
        sys.path.insert(0, _REL)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_cmz():
    return _load('build_cmz_products',
                 os.path.join(_REL, 'build_cmz_products.py'))


def _make_webpage():
    return _load('make_webpage', os.path.join(_REL, 'make_webpage.py'))


# ---- orchestrator ----
def test_dry_run_all_steps_no_deps(tmp_path, capsys):
    bcp = _build_cmz()
    spec = {
        'version': 'v1.2', 'out_dir': str(tmp_path / 'cmz'),
        'hats': False,
        'fields': [
            {'field': 'brick', 'program': '2221', 'obsid': '001',
             'catalog': '/data/brick_m7.fits',
             'f212n_i2d': '/data/brick_f212n_i2d.fits',
             'long_i2d': '/data/brick_f405n_i2d.fits', 'long_band': 'F405N'},
            {'field': 'sgrc', 'program': '4147', 'obsid': '012',
             'catalog': '/data/sgrc_m7.fits',
             'f212n_i2d': '/data/sgrc_f212n_i2d.fits',
             'long_i2d': '/data/sgrc_f480m_i2d.fits', 'long_band': 'F480M'},
        ],
    }
    out = bcp.run(spec, dry_run=True)
    assert out == str(tmp_path / 'cmz')
    log = capsys.readouterr().out
    assert 'assembling catalog from 2 field(s)' in log
    assert 'HiPS F212N += brick' in log and 'HiPS F480M += sgrc' in log
    # two-color prefers F480M as red
    assert 'R=' in log and 'F480M' in log


def test_only_subset_and_bad_step(tmp_path):
    bcp = _build_cmz()
    spec = {'out_dir': str(tmp_path), 'fields': []}
    bcp.run(spec, only=('hips',), dry_run=True)   # empty fields -> no-op, no raise
    with pytest.raises(SystemExit):
        bcp.main(['--spec', os.devnull, '--only', 'bogus'])


def test_hats_skipped_when_not_requested(tmp_path, capsys):
    bcp = _build_cmz()
    spec = {'out_dir': str(tmp_path), 'hats': False, 'fields': []}
    bcp.run(spec, only=('hats',), dry_run=True)
    assert 'not requested' in capsys.readouterr().out


# ---- webpage Aladin pane ----
def test_cmz_explorer_html():
    mw = _make_webpage()
    html = mw.render_cmz_explorer('cmz/hips/CMZ_color',
                                  cat_hips_url='cmz/hips/cmz_cat',
                                  moc_url='cmz/cmz_f212n_coverage.fits')
    assert 'aladin-lite-div' in html
    assert 'aladin.cds.unistra.fr/AladinLite/api/v3' in html
    assert 'cmz/hips/CMZ_color' in html          # color HiPS wired
    assert 'A.catalogHiPS' in html and 'cmz/hips/cmz_cat' in html
    assert 'A.MOCFromURL' in html and 'coverage.fits' in html
    assert html.startswith('<!doctype html>')


def test_cmz_explorer_optional_layers_omitted():
    mw = _make_webpage()
    html = mw.render_cmz_explorer('cmz/hips/CMZ_color')  # no cat, no moc
    assert 'A.catalogHiPS' not in html
    assert 'A.MOCFromURL' not in html
    assert 'cmz/hips/CMZ_color' in html
