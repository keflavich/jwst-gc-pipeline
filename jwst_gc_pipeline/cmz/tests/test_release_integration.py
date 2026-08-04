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


# ---- webpage preview selection ----
def _fake_release(root, field, version, with_preview):
    """Minimal on-disk release: MANIFEST.json (+ an optional preview jpg)."""
    import json
    d = os.path.join(root, version, field)
    os.makedirs(d, exist_ok=True)
    json.dump({'version': version, 'built': '2026-08-04T00:00:00Z', 'files': [],
               'globus_https_base': 'https://example.invalid',
               'globus_collection_id': '00000000-0000-0000-0000-000000000000',
               'release_path': f'/releases/{version}/{field}'},
              open(os.path.join(d, 'MANIFEST.json'), 'w'))
    if with_preview:
        p = os.path.join(d, 'preview')
        os.makedirs(p, exist_ok=True)
        # 1x1 JPEG is enough: make_webpage only copies the file
        open(os.path.join(p, f'{field}_rgb_f212n_f187n_f182m.jpg'), 'wb').write(b'\xff\xd8\xff\xd9')


def test_preview_falls_back_to_an_older_version(tmp_path):
    """A re-stage that ships no preview/ of its own must not blank the card:
    the newest version that HAS a preview supplies it."""
    mw = _make_webpage()
    root, out = str(tmp_path / 'rel'), str(tmp_path / 'site')
    _fake_release(root, 'testfield', 'v1.0-2026.06', with_preview=True)
    _fake_release(root, 'testfield', 'v1.1-2026.07', with_preview=False)
    mw.main(['--fields', 'testfield', '--release-root', root, '--out', out])
    index = open(os.path.join(out, 'index.html')).read()
    assert 'assets/testfield.jpg' in index
    assert os.path.isfile(os.path.join(out, 'assets', 'testfield.jpg'))
    # the channels are still parsed from the preview filename
    assert 'R=F212N' in open(os.path.join(out, 'testfield.html')).read()


def test_no_preview_anywhere_leaves_the_card_thumbless(tmp_path):
    mw = _make_webpage()
    root, out = str(tmp_path / 'rel'), str(tmp_path / 'site')
    _fake_release(root, 'testfield', 'v1.0-2026.06', with_preview=False)
    mw.main(['--fields', 'testfield', '--release-root', root, '--out', out])
    assert 'assets/testfield.jpg' not in open(os.path.join(out, 'index.html')).read()


# ---- RGB preview: module-split fields ----
def test_science_paths_returns_every_module_mosaic(tmp_path):
    """arches/quintuplet stage one mosaic per module; all of them must be
    returned so load_science can coadd them (picking one would show half the
    field)."""
    mpr = _load('make_preview_rgb', os.path.join(_REL, 'make_preview_rgb.py'))
    import pathlib
    d = tmp_path / 'images' / 'F212N'
    d.mkdir(parents=True)
    for mod in ('nrca', 'nrcb'):
        (d / f'jw02045-o001_t001_nircam_clear-f212n-{mod}_i2d.fits').touch()
    got = mpr.science_paths(pathlib.Path(str(tmp_path)), 'F212N', None)
    assert len(got) == 2 and got == sorted(got)
    # a field WITH a merged mosaic still prefers it, alone
    (d / 'jw02045-o001_t001_nircam_clear-f212n-merged_i2d.fits').touch()
    assert len(mpr.science_paths(pathlib.Path(str(tmp_path)), 'F212N', None)) == 1


# NOTE: the gate-honesty checks that used to live here (a check with no PASS key
# is not a pass; a field where nothing could be compared is not green) were
# superseded on main by the per-module/view rework in
# registration_failsafes.scan_field, which makes could-not-verify BLOCK rather
# than warn.  They are covered by
# jwst_gc_pipeline/photometry/tests/test_registration_per_module_gate.py
# (test_errored_check_is_not_counted_as_a_pass, test_no_mosaics_is_unverified_not_pass,
# test_sole_band_with_passing_own_catalog_is_not_blocked).


# ---- README describes what is actually staged ----
def _sr():
    return _load('stage_release', os.path.join(_REL, 'stage_release.py'))


def _readme_images_section(tmp_path, srcs, kinds=()):
    sr = _sr()
    items = [{'category': 'image', 'kind': k or 'science', 'src': s, 'filter': 'F212N'}
             for s, k in zip(srcs, list(kinds) + ['science'] * len(srcs))]
    sr.write_readme(tmp_path, 'testfield', 'v9.9', items, 'copy')
    text = (tmp_path / 'README.md').read_text()
    return text.split('## Images')[1].split('##')[0]


def test_readme_calls_out_a_module_split_field(tmp_path):
    section = _readme_images_section(tmp_path, [
        '/x/jw02045-o001_t001_nircam_clear-f212n-nrca_i2d.fits',
        '/x/jw02045-o001_t001_nircam_clear-f212n-nrcb_i2d.fits'])
    assert 'nrca' in section and 'nrcb' in section
    assert 'per-module' in section


def test_readme_does_not_call_miri_a_module_split(tmp_path):
    """sgrb2 has a merged mosaic in all ten NIRCam filters PLUS MIRI; keying off
    'anything but merged' told its readers it had no full-field mosaic."""
    section = _readme_images_section(tmp_path, [
        '/x/jw05365-o001_t001_nircam_clear-f182m-merged_i2d.fits',
        '/x/jw05365-o002-998_t001_miri_clear-f770w-mirimage_data_i2d.fits'])
    assert 'per-module' not in section
    assert 'MIRI science mosaic' in section


def test_readme_omits_residual_model_lines_when_none_staged(tmp_path):
    section = _readme_images_section(
        tmp_path, ['/x/jw05365-o001_t001_nircam_clear-f182m-merged_i2d.fits'])
    assert 'residual' not in section and 'PSF model' not in section


def test_readme_says_image_only_when_no_catalogs(tmp_path):
    sr = _sr()
    sr.write_readme(tmp_path, 'testfield', 'v9.9',
                    [{'category': 'image', 'kind': 'science', 'filter': 'F182M',
                      'src': '/x/jw05365-o001_t001_nircam_clear-f182m-merged_i2d.fits'}],
                    'copy')
    assert 'image-only release: no catalogs' in (tmp_path / 'README.md').read_text()


# ---- staging guards ----
def test_image_only_fields_declare_skip_catalogs():
    """An image-only field's status must not depend on the operator remembering
    --images-only: sgra's uncertified catalogs include the held F115W band."""
    sr = _sr()
    for field in ('sgra', 'arches', 'quintuplet', 'sickle'):
        assert sr.FIELDS[field].get('skip_catalogs') is True, field


def test_version_is_required():
    sr = _sr()
    with pytest.raises(SystemExit):
        sr.main(['--field', 'sgra'])


def test_refuses_to_stage_into_an_older_version(tmp_path, capsys, monkeypatch):
    sr = _sr()
    for name in ('v1.0-2026.06', 'v2.0-2027.01'):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(sr, 'build_manifest',
                        lambda *a, **k: [{'category': 'image', 'kind': 'science',
                                          'filter': 'F212N', 'iteration': None,
                                          'observation': None, 'instrument': 'NIRCam',
                                          'src': '/x/a.fits', 'dest': 'images/F212N/a.fits',
                                          'size_bytes': 1}])
    rc = sr.main(['--field', 'sgra', '--stage', '--release-root', str(tmp_path),
                  '--version', 'v1.0-2026.06'])
    assert rc == 2
    assert 'REFUSING TO STAGE' in capsys.readouterr().err


# ---- preview grids ----
def test_grids_differ_detects_a_mixed_scale_pair(tmp_path):
    from astropy.io import fits
    import numpy as np
    mpr = _load('make_preview_rgb', os.path.join(_REL, 'make_preview_rgb.py'))
    paths = []
    for name, cdelt in (('sw.fits', 1e-5), ('lw.fits', 2e-5)):
        hdu = fits.ImageHDU(np.zeros((4, 4), dtype='float32'), name='SCI')
        hdu.header.update({'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
                           'CRVAL1': 266.4, 'CRVAL2': -28.9, 'CRPIX1': 2, 'CRPIX2': 2,
                           'CDELT1': -cdelt, 'CDELT2': cdelt})
        path = tmp_path / name
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
        paths.append(str(path))
    assert mpr.grids_differ(paths) is True
    assert mpr.grids_differ(paths[:1]) is False


# ---- on-sky overview panel ----
def _fo():
    return _load('field_overview', os.path.join(_REL, 'field_overview.py'))


def _geom(name, lon0, lat0, size=0.05):
    """One square footprint near the Galactic Centre, in ICRS degrees."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    corners = [(lon0, lat0), (lon0 + size, lat0), (lon0 + size, lat0 + size),
               (lon0, lat0 + size)]
    icrs = [SkyCoord(l * u.deg, b * u.deg, frame='galactic').icrs for l, b in corners]
    return {'field': name, 'href': f'{name}.html',
            'polys': [[(float(c.ra.deg), float(c.dec.deg)) for c in icrs]]}


def test_section_is_empty_without_geometry():
    """No footprints -> the index must look exactly as it always did, not like a
    broken widget."""
    assert _fo().section([]) == ''


def test_every_field_is_a_link_in_the_static_map():
    fo = _fo()
    html_out = fo.section([_geom('brick', 0.2, 0.0), _geom('sgrc', -0.5, -0.1)])
    assert html_out.count('class="ov-field"') == 2
    # once in the map, once in the legend -- the legend is what keeps the
    # click-through usable if the SVG cannot render
    assert html_out.count('href="brick.html"') == 2
    assert html_out.count('href="sgrc.html"') == 2


def test_static_map_needs_no_script_to_navigate():
    """The <a> elements are plain links: JS off, strict CSP, file:// all work."""
    fo = _fo()
    svg = fo.static_svg([_geom('brick', 0.2, 0.0)], 'galactic')
    assert svg.startswith('<svg') and '<a class="ov-field" href="brick.html"' in svg
    assert 'script' not in svg


def test_galactic_longitudes_wrap_at_180():
    """The CMZ straddles l = 0; on a 0..360 axis it would split into two clumps
    at opposite ends of the map."""
    fo = _fo()
    framed, frame = fo.to_galactic([_geom('x', -0.4, 0.0)['polys'][0]])
    assert frame == 'galactic'
    assert all(-1.0 < lon < 1.0 for lon, _ in framed[0])


def test_labels_are_pushed_apart_when_they_would_overlap():
    fo = _fo()
    out = fo._spread_labels([[100.0, 50.0, 40.0], [110.0, 50.0, 40.0],
                             [900.0, 50.0, 40.0]])
    assert out[0][1] == 50.0
    assert out[1][1] >= 50.0 + fo.LABEL_DY        # collides -> moved down
    assert out[2][1] == 50.0                      # far away -> untouched


def test_collect_drops_a_field_whose_mosaics_cannot_be_read(tmp_path):
    fo = _fo()
    (tmp_path / 'images' / 'F212N').mkdir(parents=True)
    (tmp_path / 'images' / 'F212N' / 'not-a-mosaic_i2d.fits').write_text('garbage')
    assert fo.collect([('broken', tmp_path, 'broken.html')]) == []


def test_overview_sits_between_the_gc_section_and_the_next_group():
    mw = _make_webpage()
    fields = [{'field': 'brick', 'version': 'v1', 'group': None, 'preview': None,
               'n_images': 1, 'n_catalogs': 0},
              {'field': 'w51', 'version': 'v1', 'group': 'galactic_plane',
               'preview': None, 'n_images': 1, 'n_catalogs': 0}]
    page = mw.render_index(fields, overview_html='<section class=overview>MAP</section>')
    gc = page.index('>Galactic Center<')
    panel = page.index('class=overview')
    plane = page.index('>Galactic Plane<')
    assert gc < panel < plane


def test_overview_is_omitted_when_absent():
    mw = _make_webpage()
    page = mw.render_index([{'field': 'brick', 'version': 'v1', 'group': None,
                             'preview': None, 'n_images': 1, 'n_catalogs': 0}])
    assert 'class=overview' not in page
