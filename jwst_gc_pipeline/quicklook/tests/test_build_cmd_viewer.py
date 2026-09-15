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

from jwst_gc_pipeline.quicklook import catalogs as C
from jwst_gc_pipeline.quicklook import cmdview

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


def _offset_catalogs(cat_dir, sep_arcsec, obsids=('132',), n=400, seed=1):
    """Two bands whose sources sit `sep_arcsec` apart, so the match radius
    decides how many pairs come out."""
    rng = np.random.default_rng(seed)
    cat_dir.mkdir(parents=True, exist_ok=True)
    for obs in obsids:
        ra = 266.5 + rng.random(n) * 0.01
        dec = -28.7 + rng.random(n) * 0.01
        for band, shift in ((C.BLUE_BAND, 0.0), (C.RED_BAND, sep_arcsec / 3600.0)):
            tbl = Table({'flux': rng.lognormal(np.log(300.0), 0.8, n)})
            tbl['skycoord'] = SkyCoord(ra * u.deg, (dec + shift) * u.deg)
            tbl.meta['PIXSCALE'] = 0.0312229602876695
            tbl.meta['FILTER'] = band.upper()
            tbl.write(cat_dir /
                      f'{band}_merged_o{obs}_indivexp_merged_m2_dao_basic_vetted.fits',
                      overwrite=True)


def test_the_match_radius_is_part_of_the_cache_key(tmp_path):
    """The cache holds MATCHED PAIRS, not catalogs. Keyed on the inputs alone,
    a rebuild at a different radius reused pairs made at the old one, and the
    page then printed the new radius in its provenance over old data."""
    _offset_catalogs(tmp_path / 'catalogs', 0.15)
    _footprints(tmp_path / 'footprints.json', obsids=('132',))
    tight = _build(tmp_path, extra=('--match-arcsec', '0.10'))[1]
    wide = _build(tmp_path, extra=('--match-arcsec', '0.50'))[1]
    n_tight = sum(f['n'] for f in tight['fields'])
    n_wide = sum(f['n'] for f in wide['fields'])
    assert n_tight < n_wide, (
        f'0.5" reused the 0.1" pairs: {n_tight} then {n_wide}')
    assert wide['match_arcsec'] == 0.5


def test_the_cache_key_changes_with_the_radius():
    from pathlib import Path as _P
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        f = _P(d) / 'x.fits'
        f.write_bytes(b'0')
        paths = {C.BLUE_BAND: f, C.RED_BAND: f}
        assert (build_cmd_viewer.cache_key(paths, 0.10)
                != build_cmd_viewer.cache_key(paths, 0.50))
        assert (build_cmd_viewer.cache_key(paths, 0.10)
                == build_cmd_viewer.cache_key(paths, 0.10))


def test_a_single_module_pointing_is_flagged_as_part_of_the_tile(tmp_path):
    """One NIRCam module is ~60% of a tile. Four of the ten live pointings
    resolve to a single-module catalog because no `merged` file exists for
    them, which ranking cannot fix -- the wider file is absent, not
    out-ranked -- so the page has to say so."""
    cat = tmp_path / 'catalogs'
    _catalogs(cat, obsids=('127',))                      # merged, both bands
    _catalogs(cat, obsids=('128',))
    for band in C.BANDS:                                  # o128: nrca only
        src = cat / f'{band}_merged_o128_indivexp_merged_m2_dao_basic_vetted.fits'
        src.rename(cat / f'{band}_nrca_o128_indivexp_merged_m2_dao_basic_vetted.fits')
    _footprints(tmp_path / 'footprints.json', obsids=('127', '128'))
    _, data = _build(tmp_path)
    by_id = {f['id']: f for f in data['fields']}
    assert by_id['o127']['partial'] is False
    assert by_id['o127']['modules'] == ['merged']
    assert by_id['o128']['partial'] is True
    assert by_id['o128']['modules'] == ['nrca']


def test_the_page_says_which_tiles_are_only_part_of_a_tile(tmp_path):
    from jwst_gc_pipeline.quicklook import cmdview
    cat = tmp_path / 'catalogs'
    _catalogs(cat, obsids=('128',))
    for band in C.BANDS:
        src = cat / f'{band}_merged_o128_indivexp_merged_m2_dao_basic_vetted.fits'
        src.rename(cat / f'{band}_nrca_o128_indivexp_merged_m2_dao_basic_vetted.fits')
    _footprints(tmp_path / 'footprints.json', obsids=('128',))
    out, data = _build(tmp_path)
    html = (out / 'cmd_explorer.html').read_text()
    assert 'Part of the tile' in html
    assert 'nrca' in html
    # and the table marks it rather than leaving it to the caveat block
    assert "f.partial ? ' \\u26a0' : ''" in cmdview._SCRIPT


def test_a_pointing_that_matches_nothing_is_named_not_dropped(tmp_path):
    """Both bands present and zero pairs is a real condition -- a failed
    cross-match, or an astrometric offset between the filters -- and not the
    same thing as a missing band."""
    _offset_catalogs(tmp_path / 'catalogs', 30.0)      # far beyond any radius
    _footprints(tmp_path / 'footprints.json', obsids=('132',))
    with pytest.raises(SystemExit):
        _build(tmp_path)


def test_a_mast_pointing_short_a_band_is_reported(tmp_path):
    """`incomplete_pointings` iterated `find_mast`, which pre-filters to
    complete pointings -- so the one function whose job is naming incomplete
    pointings could never name a MAST one."""
    mast = tmp_path / 'mast'; mast.mkdir()
    tbl = Table({'aper_total_abmag': np.full(3, 18.0)})
    tbl['sky_centroid'] = SkyCoord([266.5] * 3 * u.deg, [-28.7] * 3 * u.deg)
    tbl.write(mast / 'jw10678-o140_t001_nircam_clear-f212n_cat.ecsv', overwrite=True)
    assert C.find_mast(mast) == {}
    assert C.incomplete_pointings(tmp_path / 'nocat', mast) == {'o140': ['f480m']}


def test_selecting_a_field_outlines_it_on_the_sky():
    """Clicking a row highlights the tile, and one code path decides that.

    The ledger, the sky, the hover and the unpin all change which field is
    current; if each set the outline itself they would drift apart.  `drawCMD`
    is the single place that already knows the answer, so the outline is drawn
    from there.
    """
    page = cmdview._SCRIPT
    assert 'function highlight(field)' in page
    # driven from drawCMD, not from the click listener
    drawcmd = page.split('function drawCMD(')[1].split('\nfunction ')[0]
    assert 'highlight(field);' in drawcmd
    # its own overlay, added after the base one so it draws over a shared edge
    foot = page.split('function drawFootprints(')[1].split('\nfunction ')[0]
    assert foot.index("name: 'Treasury pointings") < foot.index("name: 'Selected tile'")


def test_the_outline_is_drawn_once_aladin_exists():
    """`drawCMD(null)` runs before `A.init` resolves, so the first highlight
    call has no overlay to draw into.  Boot has to re-assert it, or a page
    restored with a selection shows the diagram without the tile."""
    boot = cmdview._SCRIPT.split('function boot(')[1]
    assert boot.index('drawFootprints();') < boot.index('highlight(hovered || pinned);')


def test_every_javascript_name_the_page_uses_is_defined():
    """A ReferenceError in the Aladin callback kills the rest of boot silently
    (the survey switcher on the overview page died this way, #885)."""
    import re
    src = cmdview._SCRIPT
    defined = set(re.findall(r'function\s+(\w+)\s*\(', src))
    for name in ('highlight', 'drawFootprints', 'drawCMD', 'show', 'fieldAt'):
        assert name in defined, f'{name} is called but never defined'
    for var in ('hiOv', 'footOv', 'HILITE'):
        assert re.search(rf'\bvar\s+[^;]*\b{var}\b', src), \
            f'{var} is used but never declared with var'
