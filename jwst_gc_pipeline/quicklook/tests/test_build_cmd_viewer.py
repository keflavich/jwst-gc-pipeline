"""End to end on synthetic catalogs: the payload contract and the cache."""
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

astropy = pytest.importorskip('astropy')
from astropy.coordinates import SkyCoord            # noqa: E402
from astropy.table import Table                     # noqa: E402
import astropy.units as u                           # noqa: E402

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    'build_cmd_viewer', REPO / 'scripts' / 'quicklook' / 'build_cmd_viewer.py')
build_cmd_viewer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_cmd_viewer)


def _catalogs(cat_dir, obsids=('127', '132'), n=400, seed=0):
    rng = np.random.default_rng(seed)
    cat_dir.mkdir(parents=True, exist_ok=True)
    for obs in obsids:
        ra = 266.5 + rng.random(n) * 0.01
        dec = -28.7 + rng.random(n) * 0.01
        for band, scale in ((  'f212n', 300.0), ('f480m', 900.0)):
            tbl = Table({'flux': rng.lognormal(np.log(scale), 0.8, n)})
            tbl['skycoord'] = SkyCoord(ra * u.deg, dec * u.deg)
            tbl.meta['PIXSCALE'] = 0.0312229602876695
            tbl.meta['FILTER'] = band.upper()
            tbl.write(cat_dir /
                      f'{band}_merged_o{obs}_indivexp_merged_m2_dao_basic_vetted.fits',
                      overwrite=True)


def _footprints(path, obsids=('127', '132')):
    observed = [{'target': f'GC_{int(o)}', 'number': str(int(o)),
                 'ra': 266.505, 'dec': -28.695,
                 'nircam': [[[266.50, -28.70], [266.51, -28.70],
                             [266.51, -28.69], [266.50, -28.69]]]}
                for o in obsids]
    path.write_text(json.dumps({'observed': observed, 'planned': []}))


def _build(tmp_path, extra=()):
    """Run the builder against catalogs already on disk.

    Separate from ``_run`` because the cache is keyed on the inputs' mtimes:
    re-writing the FITS files between two builds invalidates it, so a test that
    prepared them twice would only pass when both writes landed in the same
    whole second.
    """
    out = tmp_path / 'site'
    rc = build_cmd_viewer.main([
        '--catalogs', str(tmp_path / 'catalogs'),
        '--mast', str(tmp_path / 'nomast'),
        '--footprints', str(tmp_path / 'footprints.json'), '--out', str(out),
        '--cache', str(tmp_path / 'cache'), '--offline', *extra])
    assert rc == 0
    return out, json.loads((out / 'cmd_explorer_data.json').read_text())


def _run(tmp_path, extra=()):
    _catalogs(tmp_path / 'catalogs')
    _footprints(tmp_path / 'footprints.json')
    return _build(tmp_path, extra)


def test_builder_output_has_what_the_page_reads(tmp_path):
    """Every field the browser dereferences. The page is a separate program
    from the builder; nothing else checks that they agree."""
    out, data = _run(tmp_path)
    assert (out / 'cmd_explorer.html').exists()
    for key in ('grid', 'all', 'fields', 'labels', 'centre', 'fov', 'incomplete'):
        assert key in data, key
    for key in ('extent', 'nx', 'radius'):
        assert key in data['grid'], key
    for key in ('cells', 'max', 'n'):
        assert key in data['all'], key
    assert set(data['labels']) == {'x', 'y'}
    for field in data['fields']:
        for key in ('id', 'label', 'source', 'n', 'max', 'cells', 'polys', 'centre'):
            assert key in field, key
    assert [f['id'] for f in data['fields']] == ['o127', 'o132']


def test_the_axis_labels_and_the_page_both_say_vega(tmp_path):
    _, data = _run(tmp_path)
    assert data['mag_system'] == 'vega'
    assert 'Vega' in data['labels']['x'] and 'Vega' in data['labels']['y']


def test_every_field_is_binned_on_the_shared_grid(tmp_path):
    """A per-field overlay drawn on its own grid would sit somewhere else on
    the canvas than the background it is compared against."""
    _, data = _run(tmp_path)
    background = {(c[0], c[1]) for c in data['all']['cells']}
    for field in data['fields']:
        assert {(c[0], c[1]) for c in field['cells']} <= background


def test_the_whole_sample_is_the_sum_of_the_pointings(tmp_path):
    _, data = _run(tmp_path)
    assert data['all']['n'] == sum(f['n'] for f in data['fields'])
    assert (sum(c[2] for c in data['all']['cells'])
            == sum(sum(c[2] for c in f['cells']) for f in data['fields']))


def test_the_second_build_reuses_the_cache(tmp_path, capsys):
    """The cross-match is the whole cost, and it is repeated every time the
    page is regenerated for a handful of new tiles."""
    _run(tmp_path)
    capsys.readouterr()
    _build(tmp_path)
    assert '2 from cache' in capsys.readouterr().out


def test_a_rewritten_catalog_invalidates_its_cache_entry(tmp_path, capsys):
    """A re-reduction usually lands at a new filename, but not always -- an
    overwrite at the same path has to be picked up or the page would keep
    serving the superseded photometry."""
    _run(tmp_path)
    capsys.readouterr()
    _catalogs(tmp_path / 'catalogs', obsids=('132',), seed=7)
    os.utime(tmp_path / 'catalogs' /
             'f212n_merged_o132_indivexp_merged_m2_dao_basic_vetted.fits',
             (time.time() + 10, time.time() + 10))
    _build(tmp_path)
    assert '1 from cache' in capsys.readouterr().out


def test_no_cache_forces_the_work(tmp_path, capsys):
    _run(tmp_path)
    capsys.readouterr()
    _build(tmp_path, extra=('--no-cache',))
    assert '0 from cache' in capsys.readouterr().out


def test_a_pinned_extent_is_honoured(tmp_path):
    """Successive builds have to be comparable; the derived extent moves as
    tiles are added."""
    _, data = _run(tmp_path, extra=('--extent', '0', '4', '10', '20'))
    assert data['grid']['extent'] == [0.0, 4.0, 10.0, 20.0]


def test_a_build_with_no_complete_pointing_says_so_rather_than_writing_a_page(tmp_path):
    cat = tmp_path / 'catalogs'
    _catalogs(cat, obsids=('133',))
    (cat / 'f480m_merged_o133_indivexp_merged_m2_dao_basic_vetted.fits').unlink()
    out = tmp_path / 'site'
    with pytest.raises(SystemExit) as excinfo:
        build_cmd_viewer.main([
            '--catalogs', str(cat), '--mast', str(tmp_path / 'nomast'),
            '--footprints', str(tmp_path / 'nofoot.json'), '--out', str(out),
            '--cache', str(tmp_path / 'cache'), '--offline'])
    assert 'o133' in str(excinfo.value)
    assert not out.exists()


def test_a_pointing_missing_from_the_footprints_is_listed_without_an_outline(tmp_path):
    """Drawing a guessed rectangle would put a wrong footprint on a page whose
    subject is which tile you are looking at."""
    cat = tmp_path / 'catalogs'
    _catalogs(cat)
    foot = tmp_path / 'footprints.json'
    _footprints(foot, obsids=('127',))
    out = tmp_path / 'site'
    build_cmd_viewer.main([
        '--catalogs', str(cat), '--mast', str(tmp_path / 'nomast'),
        '--footprints', str(foot), '--out', str(out),
        '--cache', str(tmp_path / 'cache'), '--offline'])
    data = json.loads((out / 'cmd_explorer_data.json').read_text())
    by_id = {f['id']: f for f in data['fields']}
    assert by_id['o127']['polys']
    assert by_id['o132']['polys'] == []
    assert by_id['o132']['n'] > 0
