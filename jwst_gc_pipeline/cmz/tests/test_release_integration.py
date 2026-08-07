"""Tests for the CMZ release orchestrator + webpage Aladin pane.

These live in scripts/release (not importable as a package), so import them from
their file paths.
"""
import importlib.util
import math
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


# ---- overview geometry, against real FITS rather than synthetic dicts ----
def _write_mosaic(path, crval=(266.4, -28.94), cdelt=1.0e-5, shape=(64, 64)):
    """A staged-looking i2d: rectified plain TAN, SCI extension, no SIP."""
    from astropy.io import fits
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.ImageHDU(np.zeros(shape, dtype='float32'), name='SCI')
    hdu.header.update({'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
                       'CRVAL1': crval[0], 'CRVAL2': crval[1],
                       'CRPIX1': shape[1] / 2, 'CRPIX2': shape[0] / 2,
                       'CDELT1': -cdelt, 'CDELT2': cdelt})
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


def test_footprint_polys_reads_the_real_corners(tmp_path):
    """The synthetic-geometry tests all pass with a broken glob or a wrong HDU
    name; this one does not."""
    fo = _fo()
    _write_mosaic(tmp_path / 'images' / 'F212N' /
                  'jw01939-o001_t001_nircam_clear-f212n-merged_i2d.fits')
    polys = fo.footprint_polys(tmp_path)
    assert len(polys) == 1 and len(polys[0]) == 4
    ras = [p[0] for p in polys[0]]
    decs = [p[1] for p in polys[0]]
    assert min(ras) < 266.4 < max(ras) and min(decs) < -28.94 < max(decs)
    # calc_footprint spans pixel centres 0.5..N+0.5, i.e. 63 px at 1e-5 deg/px
    assert abs((max(decs) - min(decs)) - 63 * 1.0e-5) < 1e-7


def test_residual_and_model_images_are_not_drawn(tmp_path):
    """A field stages residual/model i2d beside the science mosaic; including
    them triples the polygons and draws each field three times over."""
    fo = _fo()
    base = tmp_path / 'images' / 'F212N'
    _write_mosaic(base / 'x-f212n-merged_i2d.fits')
    _write_mosaic(base / 'x-f212n-merged_resbgsub_m7_daophot_basic_mergedcat_residual_i2d.fits')
    _write_mosaic(base / 'x-f212n-merged_resbgsub_m7_daophot_basic_mergedcat_model_i2d.fits')
    assert len(fo.footprint_polys(tmp_path)) == 1


def test_a_broken_wcs_does_not_shrink_every_other_field(tmp_path):
    """A CDELT typo parses as a valid TAN and yields a ~190 deg footprint;
    `_projection` mins/maxes over every corner, so one such polygon rescales the
    map until the real fields are sub-pixel."""
    fo = _fo()
    good, bad = tmp_path / 'good', tmp_path / 'bad'
    _write_mosaic(good / 'images' / 'F212N' / 'a_i2d.fits')
    _write_mosaic(bad / 'images' / 'F212N' / 'b_i2d.fits', cdelt=10.0)
    assert fo.footprint_polys(bad) == []          # dropped, with a printed reason
    geoms = fo.collect([('good', good, 'good.html'), ('bad', bad, 'bad.html')])
    assert [g['field'] for g in geoms] == ['good']
    _, _, (width, height) = fo._projection(
        [dict(g, polys=fo.to_galactic(g['polys'])[0]) for g in geoms], 'galactic')
    assert width > 100 and height > 10            # the good field still has area


def test_usable_poly_rejects_what_would_break_the_build():
    """math.ceil(nan) in the graticule raises and nothing above main() catches
    it, so one non-finite corner would cost the whole index.html."""
    fo = _fo()
    ok = [(266.40, -28.94), (266.41, -28.94), (266.41, -28.93), (266.40, -28.93)]
    assert fo.usable_poly(ok) is True
    assert fo.usable_poly([(float('nan'), -28.94)] + ok[1:]) is False
    assert fo.usable_poly([(float('inf'), -28.94)] + ok[1:]) is False
    assert fo.usable_poly([(0.0, -80.0), (190.0, -80.0),
                           (190.0, 80.0), (0.0, 80.0)]) is False
    # and the section survives the degenerate single-footprint case
    assert '<svg' in fo.section([{'field': 'x', 'href': 'x.html', 'polys': [ok]}])


def test_longitude_runs_right_to_left():
    """Galactic maps put increasing l to the LEFT; a mirrored map is wrong and
    looks entirely plausible."""
    fo = _fo()
    geoms = [dict(g, polys=fo.to_galactic(g['polys'])[0])
             for g in (_geom('left', 0.5, 0.0), _geom('right', -0.5, 0.0))]
    project, _, _ = fo._projection(geoms, 'galactic')
    x_high_l, _ = project(0.5, 0.0)
    x_low_l, _ = project(-0.5, 0.0)
    assert x_high_l < x_low_l


def test_projection_compresses_longitude_by_cos_lat():
    """Tested AT HIGH LATITUDE on purpose: at the CMZ's |b| < 0.2 deg,
    cos(lat) ~ 1, so dropping the term entirely changes nothing measurable and
    an equal-scale assertion there passes either way."""
    fo = _fo()
    geoms = [{'field': 'a', 'href': 'a.html',
              'polys': [[(0.0, 59.9), (0.4, 59.9), (0.4, 60.1), (0.0, 60.1)]]}]
    project, _, _ = fo._projection(geoms, 'galactic')
    dx = abs(project(0.0, 60.0)[0] - project(0.1, 60.0)[0])
    dy = abs(project(0.0, 60.0)[1] - project(0.0, 60.1)[1])
    # a degree of longitude at b = 60 subtends cos(60) = 0.5 of a degree of arc
    assert abs((dx / 0.1) / (dy / 0.1) - math.cos(math.radians(60.0))) < 0.02


def test_aladin_payload_carries_icrs_not_galactic():
    """Aladin works in ICRS; handing it the Galactic polygons puts every
    footprint tens of degrees from where the HiPS shows the field."""
    import json as _json
    import re as _re
    fo = _fo()
    geom = _geom('brick', 0.2, 0.0)
    out = fo.section([geom])
    payload = _json.loads(_re.search(
        r'<script id=ov-data type="application/json">(.*?)</script>', out, _re.S).group(1))
    assert payload['fields'][0]['polys'] == [[list(pt) for pt in geom['polys'][0]]]
    assert payload['fields'][0]['polys'][0][0][0] > 200            # an RA, not an l


def test_a_field_name_cannot_break_out_of_the_json_payload():
    fo = _fo()
    out = fo.section([{'field': '</script><img src=x onerror=alert(1)>',
                       'href': 'x.html',
                       'polys': [_geom('x', 0.2, 0.0)['polys'][0]]}])
    assert '</script><img' not in out
    assert '\\u003c/script' in out or '\\u003cscript' in out
    # exactly the two script elements the panel emits, both properly closed
    assert out.count('<script') == 2 and out.count('</script>') == 2


def test_aladin_failure_removes_the_overlay_it_added():
    """`.ov-aladin` is position:absolute;inset:0 over the map. Left behind after
    a failure it covers every field link while the status says the map is fine."""
    fo = _fo()
    out = fo.section([_geom('brick', 0.2, 0.0)])
    assert 'function teardown()' in out
    # fail() must tear down; and the host must not be created before the loader
    # can catch a throw from A.init
    fail_body = out.split('function fail(msg) {')[1].split('}')[0]
    assert 'teardown();' in fail_body
    assert 'Promise.resolve(A && A.init)' in out


def test_survey_switcher_is_wired_to_every_listed_survey():
    """SURVEYS was serialised into the page with only entry 0 ever read."""
    fo = _fo()
    out = fo.section([_geom('brick', 0.2, 0.0)])
    assert 'data.surveys.forEach' in out and 'setImageSurvey' in out
    for name, _ in fo.SURVEYS:
        assert name in out


def test_aladin_source_is_pinned():
    """`latest` lets a third party change the API under a published page."""
    fo = _fo()
    assert '/latest/' not in fo.ALADIN_JS


def test_emitted_javascript_parses(tmp_path):
    """The JS is built by an f-string with doubled braces and is only ever
    asserted as substrings, so a syntax error in the template would ship a page
    whose interactive panel silently does nothing.  Parse it."""
    import re
    import shutil as _shutil
    import subprocess
    node = _shutil.which('node') or \
        '/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/node'
    if not os.path.isfile(node):
        pytest.skip('node not available to parse the emitted script')
    fo = _fo()
    out = fo.section([_geom('brick', 0.2, 0.0), _geom('sgrc', -0.5, -0.1)])
    script = re.search(r'<script>\n(.*?)\n</script>', out, re.S).group(1)
    path = tmp_path / 'emitted.js'
    path.write_text(script)
    proc = subprocess.run([node, '--check', str(path)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_success_is_not_reported_until_the_layer_is_visible():
    """setOverlayImageLayer returns the layer object, not a promise: a HiPS that
    never resolves does not throw, so writing 'background: X' on the next line
    claims a background that never arrived -- the reported symptom exactly."""
    fo = _fo()
    out = fo.section([_geom('brick', 0.2, 0.0)])
    body = out.split("b.addEventListener('click'")[1]
    setter = body.index('setBaseImageLayer')
    success = body.index("'background: '")
    assert body.index('function poll()') < success, \
        'success must be reported from the poll, not straight after the setter'
    assert setter < body.index('currentLayerId()')
    assert 'getBaseImageLayer' in out          # verified against what is displayed
    assert 'switchSeq' in out                  # a later click supersedes an older poll


def test_verification_is_gated_on_the_getter_being_available():
    """The SETTER is feature-detected, so an Aladin without `getBaseImageLayer`
    is explicitly anticipated. Polling there would return null forever and roll
    every switch back after 8 s -- reporting failure on a background that loaded
    fine, the same lie pointing the other way."""
    fo = _fo()
    out = fo.section([_geom('brick', 0.2, 0.0)])
    assert "typeof aladin.getBaseImageLayer === 'function'" in out
    assert 'cannot confirm it loaded' in out
    # and the id comparison tolerates protocol / trailing-slash differences
    assert 'function normaliseId' in out
# ---- one preview per pointing, and every band in some preview ----
def _pp():
    return _load('preview_plan', os.path.join(_REL, 'preview_plan.py'))


def _stage_filters(root, layout):
    """layout: {subdir or '': [FILTER, ...]} under <root>/images/."""
    for sub, filters in layout.items():
        for filt in filters:
            d = os.path.join(root, 'images', sub, filt) if sub else \
                os.path.join(root, 'images', filt)
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, f'x-{filt.lower()}-merged_i2d.fits'), 'w').close()


def test_every_staged_band_appears_in_some_preview(tmp_path):
    """sgrb2 ships 14 bands; one RGB shows three and silently drops eleven."""
    pp = _pp()
    bands = ['F150W', 'F182M', 'F187N', 'F210M', 'F212N', 'F300M', 'F360M',
             'F405N', 'F410M', 'F466N', 'F480M']
    _stage_filters(str(tmp_path), {'': bands, 'MIRI': ['F770W', 'F1280W', 'F2550W']})
    specs = pp.plan(tmp_path)
    covered = {f for s in specs for f in s['filters']}
    assert covered == set(bands) | {'F770W', 'F1280W', 'F2550W'}
    assert len(specs) == 5
    assert all(2 <= len(s['filters']) <= 3 for s in specs)


def test_miri_is_pooled_not_treated_as_a_separate_pointing(tmp_path):
    """MIRI is the same sky at a longer wavelength.  Left as its own group,
    brick's single F2550W would be a group of one and never get an image."""
    pp = _pp()
    _stage_filters(str(tmp_path), {'': ['F405N', 'F444W', 'F466N'],
                                   'MIRI': ['F2550W']})
    specs = pp.plan(tmp_path)
    assert all(s['pointing'] is None for s in specs)
    assert 'F2550W' in {f for s in specs for f in s['filters']}
    assert any('MIRI' in s['subdirs'] for s in specs)


def test_each_pointing_gets_its_own_preview(tmp_path):
    """gc2211 is four separate pointings; one image can only show one of them."""
    pp = _pp()
    _stage_filters(str(tmp_path), {'o023': ['F200W', 'F277W'],
                                   'o028': ['F150W', 'F277W'],
                                   'o046': ['F200W', 'F277W']})
    specs = pp.plan(tmp_path)
    assert sorted(s['pointing'] for s in specs) == ['o023', 'o028', 'o046']
    # two bands -> the existing R/(G=mean)/B, only expanded spatially
    assert all(len(s['filters']) == 2 for s in specs)


def test_plan_assigns_the_reddest_band_to_red(tmp_path):
    """The name said reddest-first while the assertion exercised chunk_filters,
    which sorts BLUEST-first -- so inverting the sort inside plan(), the channel
    assignment this whole change is about, passed the suite untouched."""
    pp = _pp()
    _stage_filters(str(tmp_path), {'': ['F115W', 'F212N', 'F405N', 'F444W']})
    for spec in pp.plan(tmp_path):
        waves = [pp.wavelength_um(f) for f in spec['filters']]
        assert waves == sorted(waves, reverse=True), spec['filters']
    spec = {'pointing': None, 'filters': ['F444W', 'F410M', 'F405N'], 'subdirs': []}
    assert pp.describe(spec) == 'F444W/F410M/F405N'


def test_a_trailing_singleton_borrows_instead_of_being_dropped():
    """3+3+1 would leave the last band in a chunk that cannot make an image."""
    pp = _pp()
    seven = ['F115W', 'F150W', 'F182M', 'F212N', 'F300M', 'F405N', 'F444W']
    chunks = pp.chunk_filters(seven)
    assert [len(c) for c in chunks] == [3, 2, 2]
    assert sorted(f for c in chunks for f in c) == sorted(seven)


def test_field_page_shows_every_preview(tmp_path):
    mw = _make_webpage()
    manifest = {'version': 'v1', 'built': '2026-08-05T00:00:00Z', 'files': [],
                'globus_https_base': 'https://example.invalid',
                'globus_collection_id': '0', 'release_path': '/r'}
    page = mw.render_field_page(
        'brick', manifest, 'assets/brick.jpg', None, preview_version='v1',
        previews=[('assets/brick_rgb_f187n_f182m_f115w.jpg', 'brick_rgb_f187n_f182m_f115w'),
                  ('assets/brick_rgb_f2550w_mean_f466n.jpg', 'brick_rgb_f2550w_mean_f466n')])
    assert page.count('class=preview ') == 2
    assert 'R=F187N, G=F182M, B=F115W' in page
    assert 'R=F2550W, G=mean(F2550W,F466N), B=F466N' in page


def test_single_preview_field_page_is_unchanged(tmp_path):
    mw = _make_webpage()
    manifest = {'version': 'v1', 'built': '2026-08-05T00:00:00Z', 'files': [],
                'globus_https_base': 'https://example.invalid',
                'globus_collection_id': '0', 'release_path': '/r'}
    page = mw.render_field_page('sgra', manifest, 'assets/sgra.jpg',
                                ['F405N', 'MEAN', 'F212N'], preview_version='v1',
                                previews=[('assets/sgra.jpg', 'sgra_rgb_f405n_mean_f212n')])
    assert 'class=previews' not in page
    assert 'RGB preview - R=F405N, G=mean(F405N,F212N), B=F212N' in page


def test_auto_supersedes_previews_left_over_from_an_older_plan(tmp_path):
    """Re-running --auto after the plan changes must not leave both sets on the
    page: wd1 ended up with seven files for a four-preview plan."""
    mpr = _load('make_preview_rgb', os.path.join(_REL, 'make_preview_rgb.py'))
    import pathlib
    d = pathlib.Path(tmp_path)
    for stem in ('wd1_rgb_f164n_f150w_f115w',      # in plan
                 'wd1_rgb_f187n_f150w_f115w'):     # left over
        (d / f'{stem}.jpg').touch()
        (d / f'{stem}.png').touch()
    moved = mpr.supersede_unplanned(d, {'wd1_rgb_f164n_f150w_f115w'})
    assert sorted(moved) == ['wd1_rgb_f187n_f150w_f115w.jpg',
                             'wd1_rgb_f187n_f150w_f115w.png']
    assert (d / 'wd1_rgb_f164n_f150w_f115w.jpg').is_file()      # kept
    # renamed, not deleted -- these live in a published tree
    assert (d / 'wd1_rgb_f187n_f150w_f115w.jpg.superseded').is_file()
    assert not (d / 'wd1_rgb_f187n_f150w_f115w.jpg').exists()


def test_planned_stems_match_what_main_writes():
    """The prune is only safe if it computes the SAME name the renderer does."""
    mpr = _load('make_preview_rgb', os.path.join(_REL, 'make_preview_rgb.py'))
    assert mpr.preview_stem('gc2211', 'o023', 'F277W', 'mean', 'F200W') == \
        'gc2211_o023_rgb_f277w_mean_f200w'
    assert mpr.preview_stem('brick', None, 'F444W', 'F410M', 'F405N') == \
        'brick_rgb_f444w_f410m_f405n'
    specs = [{'pointing': 'o023', 'filters': ['F277W', 'F200W'], 'subdirs': []},
             {'pointing': None, 'filters': ['F444W', 'F410M', 'F405N'], 'subdirs': []}]
    assert mpr.planned_stems('gc2211', specs) == {
        'gc2211_o023_rgb_f277w_mean_f200w', 'gc2211_rgb_f444w_f410m_f405n'}


def test_every_filter_dir_in_the_release_tree_has_a_derivable_wavelength():
    """The previous guard read `stage_release.FIELDS` -- but F164N and F250M are
    not declared there; they reach a page through the STAGED TREE, which is what
    plan() reads. Deleting them from the table left that guard green. Assert
    against the same source plan() uses."""
    pp = _pp()
    root = '/orange/adamginsburg/jwst/releases'
    if not os.path.isdir(root):
        pytest.skip('release tree not present')
    seen = set()
    for version in os.listdir(root):
        for dirpath, dirnames, _ in os.walk(os.path.join(root, version)):
            if os.path.basename(dirpath) != 'images':
                continue
            for sub in dirnames:
                if pp.FILTER_DIR_RE.match(sub):
                    seen.add(sub)
                else:
                    inner = os.path.join(dirpath, sub)
                    if os.path.isdir(inner):
                        seen.update(n for n in os.listdir(inner)
                                    if pp.FILTER_DIR_RE.match(n))
    assert seen, 'no staged filters found -- the walk is wrong, not the tree'
    for filt in sorted(seen):
        pp.wavelength_um(filt)          # raises on anything it cannot derive


def test_wavelength_is_derived_not_defaulted():
    pp = _pp()
    assert pp.wavelength_um('F164N') == 1.64      # was silently 99.0 -> into R
    assert pp.wavelength_um('F250M') == 2.50
    assert pp.wavelength_um('F2550W') == 25.50
    assert pp.wavelength_um('F150W2') == 1.50
    with pytest.raises(ValueError):
        pp.wavelength_um('NOTAFILTER')


def test_extra_subdirs_are_actually_searched(tmp_path):
    """Deleting the extra-subdir search from science_paths passed the suite:
    the MIRI test only checked that 'MIRI' appeared in spec['subdirs'], never
    that a filter living there can be found."""
    mpr = _load('make_preview_rgb', os.path.join(_REL, 'make_preview_rgb.py'))
    import pathlib
    d = tmp_path / 'images' / 'MIRI' / 'F2550W'
    d.mkdir(parents=True)
    (d / 'jw02221-o002_t001_miri_f2550w_i2d.fits').touch()
    found = mpr.science_paths(pathlib.Path(str(tmp_path)), 'F2550W', None, ['MIRI'])
    assert len(found) == 1 and found[0].endswith('f2550w_i2d.fits')
    with pytest.raises(FileNotFoundError):
        mpr.science_paths(pathlib.Path(str(tmp_path)), 'F2550W', None, [])


def test_a_one_band_group_is_skipped_not_fatal(tmp_path, capsys):
    """A singleton reached `--filters takes 2 or 3` inside the recursive main()
    and exited 2 -- killing the whole run mid-way and leaving a partial gallery
    under an 'every band appears' caption."""
    pp = _pp()
    _stage_filters(str(tmp_path), {'o023': ['F200W', 'F277W'],
                                   'MIRI': ['F2550W']})
    specs = pp.plan(tmp_path)
    assert [s['pointing'] for s in specs] == ['o023']
    assert 'cannot make a colour image' in capsys.readouterr().out


def test_a_quarantined_band_does_not_enter_the_plan(tmp_path):
    """A band whose mosaics were stale-tagged leaves the directory behind; on
    isdir alone it entered the plan and died at render time with
    FileNotFoundError -- after earlier previews had been written."""
    pp = _pp()
    _stage_filters(str(tmp_path), {'': ['F200W', 'F277W']})
    bad = tmp_path / 'images' / 'F405N'
    bad.mkdir(parents=True)
    (bad / 'x-f405n-merged_i2d_im0_badastrom.fits').touch()
    covered = {f for s in pp.plan(tmp_path) for f in s['filters']}
    assert covered == {'F200W', 'F277W'}


def test_aladin_must_prove_it_rendered():
    """Construction returning is not evidence the view works. Aladin can fail in
    its own async setup -- WebGL2 unavailable, where aladin.js throws a BARE
    STRING no try can catch -- leaving an opaque inset:0 host over the static
    map. That is a black rectangle where the map was: 'it's just not there'."""
    fo = _fo()
    out = fo.section([_geom('brick', 0.2, 0.0)])
    assert 'awaitCanvas' in out
    assert "querySelector('canvas')" in out
    assert 'needs WebGL2' in out
    # the bare-string throw is captured globally, since nothing wraps it
    assert "window.addEventListener('error'" in out
    assert 'function describeError' in out
    # and the host is torn down so the static map comes back
    body = out.split('function awaitCanvas')[1]
    assert 'fail(' in body.split('READY_TRIES')[1][:600]


def test_a_one_pixel_fallback_canvas_is_not_accepted():
    """`canvas.width > 0` was satisfied by the 1-pixel canvas Aladin builds when
    it measures the container as zero -- so the panel reported success for a view
    0 px tall.  The canvas has to be laid out at a usable size."""
    fo = _fo()
    out = fo.section([_geom('brick', 0.2, 0.0)])
    check = out.split('function awaitCanvas')[1].split('readyTries')[0]
    assert 'canvas.width > 0' not in check, 'a 1-pixel fallback would pass'
    assert 'MIN_CANVAS_PX' in check and 'getBoundingClientRect' in check, \
        'the LAID-OUT size is what matters, not the drawing-buffer attribute'
    assert 'rendered with no usable size' in out


def test_the_aladin_host_is_sized_inline_because_aladin_outranks_the_class():
    """aladin.js ships `.aladin-container{position:relative}` and puts that class
    on the host.  It loads after this page's <style>, equal specificity, so it
    WINS: `.ov-aladin{position:absolute}` is overridden, `inset:0` no longer
    sizes anything, and the div collapses to height 0 with no `height` rule
    visible anywhere.  Inline style is the only thing a later sheet cannot beat."""
    fo = _fo()
    out = fo.section([_geom('brick', 0.2, 0.0)])
    build = out.split('function build()')[1].split('A.aladin(')[0]
    assert "host.style.position = 'absolute'" in build
    assert "host.style.height = '100%'" in build
    # ... which needs a definite height on the stage, or 100% is itself auto
    assert "stage.style.height = Math.max(measured, MIN_STAGE_PX) + 'px'" in build
    # and layout must be flushed before Aladin measures the container
    assert 'void host.offsetHeight;' in build
    assert build.index('stage.style.height') < build.index('void host.offsetHeight')
    # the pin is released on teardown, or a dead gap outlives the map
    assert "stage.style.height = '';" in out.split('function teardown')[1]


def _node():
    import shutil as _shutil
    node = _shutil.which('node') or \
        '/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/node'
    return node if os.path.isfile(node) else None


_DOM_HARNESS = r"""
// Minimal DOM: enough to run the panel's IIFE through one click and into
// build().  Asserting on the emitted SOURCE cannot tell whether the sizing
// actually reaches the element; running it can.
const fs = require('fs');
const emitted = fs.readFileSync(process.argv[2], 'utf8');
const payload = fs.readFileSync(process.argv[3], 'utf8');
function El(tag) {
  return {
    tagName: tag, style: {}, children: [], parentNode: null, className: '',
    textContent: '', innerHTML: '', hidden: false, disabled: false,
    _w: 0, _h: 0,
    appendChild(c) { c.parentNode = this; this.children.push(c); return c; },
    removeChild(c) {
      this.children = this.children.filter(x => x !== c); c.parentNode = null;
      return c;
    },
    querySelector(sel) { return this._q ? this._q(sel) : null; },
    getBoundingClientRect() { return {width: this._w, height: this._h}; },
    addEventListener() {}, setAttribute() {},
  };
}
const btn = El('button'), status = El('span'), stage = El('div');
const surveys = El('span'), dataEl = El('script'), head = El('head');
dataEl.textContent = payload;
const svg = El('svg'); svg._w = 1068; svg._h = 322;
// the stage is height:auto off the SVG's intrinsic ratio -- a real height
stage._w = 1068; stage._h = 322;
stage._q = (sel) => (sel === 'svg' ? svg : null);
const byId = {'ov-load': btn, 'ov-status': status, 'ov-stage': stage,
              'ov-surveys': surveys, 'ov-data': dataEl};
let clickHandler = null;
btn.addEventListener = (evt, fn) => { if (evt === 'click') clickHandler = fn; };
let injected = null;
head.appendChild = (el) => { injected = el; return el; };
globalThis.document = {
  head, getElementById: (id) => byId[id] || null,
  createElement: (tag) => El(tag),
};
globalThis.window = {addEventListener() {}, console: null, location: {}};
globalThis.setTimeout = () => 0;          // do not spin the 10 s canvas poll
globalThis.A = {
  init: Promise.resolve(),
  aladin(host) { globalThis.__host = host; return {}; },
  catalog: () => ({addSources() {}}), graphicOverlay: () => ({add() {}}),
  polygon: () => ({}), source: () => ({}), HiPS: (u) => u,
};
eval(emitted);
clickHandler();
injected.onload();
setImmediate(() => setImmediate(() => {
  const host = globalThis.__host;
  console.log(JSON.stringify({
    stageHeight: stage.style.height || null,
    hostPosition: (host && host.style.position) || null,
    hostHeight: (host && host.style.height) || null,
    hostWidth: (host && host.style.width) || null,
    attached: !!(host && host.parentNode === stage),
  }));
}));
"""


def test_running_the_panel_actually_sizes_the_aladin_host(tmp_path):
    """The bug shipped past every source-level assertion: the CSS said
    `position:absolute;inset:0` and looked right.  Run the emitted script against
    a DOM and read the geometry off the element Aladin is handed."""
    import json
    import re
    import subprocess
    node = _node()
    if node is None:
        pytest.skip('node not available to run the emitted script')
    fo = _fo()
    out = fo.section([_geom('brick', 0.2, 0.0)])
    script = re.search(r'<script>\n(.*?)\n</script>', out, re.S).group(1)
    payload = re.search(r'<script id=ov-data[^>]*>(.*?)</script>', out, re.S).group(1)
    (tmp_path / 'emitted.js').write_text(script)
    (tmp_path / 'payload.json').write_text(payload)
    (tmp_path / 'harness.js').write_text(_DOM_HARNESS)
    proc = subprocess.run(
        [node, str(tmp_path / 'harness.js'), str(tmp_path / 'emitted.js'),
         str(tmp_path / 'payload.json')], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got['attached'], 'the host never reached the stage'
    assert got['hostPosition'] == 'absolute'
    assert got['hostHeight'] == '100%' and got['hostWidth'] == '100%'
    # the stage carries a definite pixel height for that 100% to resolve against
    assert got['stageHeight'] and got['stageHeight'].endswith('px')
    assert int(got['stageHeight'][:-2]) >= 240


def test_a_prior_page_error_is_not_blamed_on_aladin():
    """`lastGlobalError` is set by a window.onerror listener installed at load
    and was never cleared, so an unrelated page error -- another script's bug,
    an ad blocker, a failed analytics fetch -- occurring BEFORE the button was
    pressed got named on a PUBLIC page as the reason the interactive view
    failed.  Reviewer's reproduction:

        status: "The interactive view could not start in this browser (Cannot
                 read properties of null (reading 'appendChild') at
                 analytics.js:12). Aladin Lite needs WebGL2. ..."

    Source-level, and deliberately so: the DOM harness does not fire a page
    error before the click, and extending it is a bigger change than the
    one-line reset it would be checking.
    """
    fo = _fo()
    out = fo.section([_geom('brick', 0.2, 0.0)])
    click = out.split("btn.addEventListener('click'")[1].split('script.src')[0]
    assert 'lastGlobalError = null' in click, \
        'the attempt must start from a clean error slate'


def test_the_promise_path_describes_its_error_like_the_others():
    """`describeError` is used in the synchronous catch and the global
    listener; the promise catch concatenated the raw value, which prints
    "Error: ..." for a real Error and "[object Object]" for anything without a
    useful toString."""
    fo = _fo()
    out = fo.section([_geom('brick', 0.2, 0.0)])
    assert "'Aladin Lite failed to start (' + err +" not in out
    assert "'Aladin Lite failed to start (' + describeError(err)" in out


def test_a_silent_timeout_does_not_assert_webgl2():
    """No canvas AND no observed error is not evidence of a missing WebGL2 --
    a slow machine or a stalled fetch reads exactly the same.  The
    canvas-present branch already worded this carefully; this one did not."""
    fo = _fo()
    out = fo.section([_geom('brick', 0.2, 0.0)])
    assert 'did not start within' in out
    tail = out.split('The interactive view did not start within')[1][:200]
    assert 'WebGL2' not in tail, tail
# ---- a superseded product must never be published ----
def _rf_fresh():
    return _load('release_freshness', os.path.join(_REL, 'release_freshness.py'))


def test_quarantined_source_is_detected(tmp_path):
    """The m2 checkpoint renames a mosaic it supersedes to *_im0_badastrom.fits.
    Cloud C shipped six such images for weeks, presented as evidence that the
    astrometry is sound."""
    rf = _rf_fresh()
    live = tmp_path / 'a-f212n-merged_i2d.fits'
    live.touch()
    assert rf.source_state(str(live)) == rf.LIVE
    gone = tmp_path / 'b-f182m-merged_i2d.fits'
    (tmp_path / 'b-f182m-merged_i2d_im0_badastrom.fits').touch()
    assert rf.source_state(str(gone)) == rf.QUARANTINED
    assert rf.source_state(str(tmp_path / 'never-existed_i2d.fits')) == rf.MISSING
    assert rf.source_state(None) == rf.MISSING


def test_there_is_no_single_SUPERSEDED_name_to_compare_against():
    """An alias pointing at one of the two states is worse than no alias.
    `SUPERSEDED` used to mean EITHER stale state; re-adding it pointed at
    QUARANTINED silently stopped every comparison matching the REBUILT case --
    the majority (54 of 114) -- and shipped the branch red, because the two
    regression tests pinning the overwrite fix compare against it.  Removing the
    name fails loudly at import instead of quietly changing what it matches."""
    rf = _rf_fresh()
    assert not hasattr(rf, 'SUPERSEDED'), \
        'use is_superseded() for "either", or the two states for "which"'
    assert rf.is_superseded(rf.QUARANTINED) and rf.is_superseded(rf.REBUILT)
    assert not rf.is_superseded(rf.LIVE) and not rf.is_superseded(rf.MISSING)


def test_a_quarantine_that_was_later_re_drizzled_is_still_a_quarantine(tmp_path):
    """The commonest shape in the tree, and the one `isfile`-first got wrong:
    m2 quarantines a mosaic, the field is corrected and RE-DRIZZLED under the
    same name, and the `*_im0_badastrom.fits` twin stays in the directory.  The
    source exists again, so a presence-first test calls it `rebuilt` -- while
    the twin is the only on-disk evidence that the bytes staged before the
    correction are the repudiated ones.

    49 of the 54 rebuilt entries in the release tree are this, including all 23
    of brick v1.0 and all 6 of cloudc."""
    rf = _rf_fresh()
    src = tmp_path / 'jw01182-o001_t001_nircam_f200w-merged_i2d.fits'
    src.write_bytes(b'\0' * 200)                       # re-drizzled, present
    (tmp_path / 'jw01182-o001_t001_nircam_f200w-merged_i2d_im0_badastrom.fits'
     ).write_bytes(b'\0' * 100)                        # the repudiated twin
    assert rf.has_quarantine_twin(str(src))
    assert rf.source_state(str(src), recorded_size=100) == rf.QUARANTINED, \
        'a re-drizzled quarantine was reported as a plain rebuild'
    # ... but a staged copy whose bytes ARE the current bytes is LIVE: the
    # field was corrected and re-staged, which is what the quarantine was for.
    # Condemning it on the twin alone withholds 65 correctly-staged images.
    assert rf.source_state(str(src), recorded_size=200) == rf.LIVE
    # and a genuine rebuild with no twin is still REBUILT
    plain = tmp_path / 'jw01182-o001_t001_nircam_f356w-merged_i2d.fits'
    plain.write_bytes(b'\0' * 200)
    assert not rf.has_quarantine_twin(str(plain))
    assert rf.source_state(str(plain), recorded_size=100) == rf.REBUILT


def test_a_rebuild_in_place_is_not_reported_as_a_quarantine(tmp_path):
    """Both states stop publication, but only one is knowable.  A renamed source
    is the pipeline repudiating the file; a size mismatch is a re-run, a
    re-chunk or a new stage, and nothing in a `stat` says which.  Collapsing
    them put "the m2 checkpoint quarantined this" on 23 of brick's 31 frozen
    v1.0 images, where no such thing had happened."""
    rf = _rf_fresh()
    rebuilt = tmp_path / 'c-f405n-merged_i2d.fits'
    rebuilt.write_bytes(b'12345')
    assert rf.source_state(str(rebuilt), recorded_size=5) == rf.LIVE
    assert rf.source_state(str(rebuilt), recorded_size=4096) == rf.REBUILT
    assert rf.REBUILT != rf.QUARANTINED

    quarantined = tmp_path / 'd-f405n-merged_i2d.fits'
    (tmp_path / 'd-f405n-merged_i2d_im0_badastrom.fits').touch()
    assert rf.source_state(str(quarantined), recorded_size=4096) == rf.QUARANTINED

    # both are withheld, and the reason travels with each one
    manifest = {'files': [
        {'category': 'image', 'dest': 'r', 'src': str(rebuilt), 'size_bytes': 4096},
        {'category': 'image', 'dest': 'q', 'src': str(quarantined), 'size_bytes': 4096},
    ]}
    assert rf.superseded_files(manifest) == ['q', 'r']
    assert rf.superseded_reasons(manifest) == {'r': rf.REBUILT,
                                               'q': rf.QUARANTINED}


def test_superseded_files_reads_the_manifest(tmp_path):
    rf = _rf_fresh()
    (tmp_path / 'x_i2d.fits').touch()
    (tmp_path / 'y_i2d_im0_badastrom.fits').touch()
    manifest = {'files': [
        {'category': 'image', 'dest': 'images/F212N/x.fits',
         'src': str(tmp_path / 'x_i2d.fits')},
        {'category': 'image', 'dest': 'images/F182M/y.fits',
         'src': str(tmp_path / 'y_i2d.fits')},
        {'category': 'catalog', 'dest': 'catalogs/c.fits',
         'src': str(tmp_path / 'gone.fits')},
    ]}
    assert rf.superseded_files(manifest) == ['images/F182M/y.fits']


def test_page_withholds_a_superseded_image_and_says_so():
    mw = _make_webpage()
    manifest = {'version': 'v1', 'built': '2026-08-05T00:00:00Z',
                'globus_https_base': 'https://example.invalid',
                'globus_collection_id': '0', 'release_path': '/r',
                'files': [
                    {'category': 'image', 'kind': 'science', 'filter': 'F212N',
                     'iteration': None, 'observation': None,
                     'dest': 'images/F212N/good.fits', 'url': 'u1', 'size_bytes': 1},
                    {'category': 'image', 'kind': 'science', 'filter': 'F182M',
                     'iteration': None, 'observation': None,
                     'dest': 'images/F182M/bad.fits', 'url': 'u2', 'size_bytes': 1},
                ]}
    rf = _load('release_freshness', os.path.join(_REL, 'release_freshness.py'))
    page = mw.render_field_page(
        'cloudc', manifest, None, superseded=['images/F182M/bad.fits'],
        reasons={'images/F182M/bad.fits': rf.QUARANTINED})
    assert 'withheld as bad astrometry' in page
    assert 'F182M' in page.split('withheld')[1][:500]     # named in the notice
    assert 'u2' not in page                               # no download offered
    assert 'u1' in page                                   # the current one stays


def test_the_notice_only_blames_the_astrometry_checkpoint_where_it_acted():
    """The notice used to assert -- for every withheld image -- that the m2
    checkpoint quarantined the mosaic.  That is knowable only for a RENAMED
    source; a rebuild in place is one `stat` disagreeing with a recorded size,
    and it is the majority (52 of 116).  Under the old wording, 23 of brick's
    31 frozen v1.0 images carried a quarantine that never happened."""
    mw = _make_webpage()
    rf = _load('release_freshness', os.path.join(_REL, 'release_freshness.py'))

    def _page(state):
        manifest = {'version': 'v1', 'built': '2026-08-05T00:00:00Z',
                    'globus_https_base': 'https://example.invalid',
                    'globus_collection_id': '0', 'release_path': '/r',
                    'files': [{'category': 'image', 'kind': 'science',
                               'filter': 'F182M', 'iteration': None,
                               'observation': None, 'dest': 'images/F182M/b.fits',
                               'url': 'u', 'size_bytes': 1}]}
        return mw.render_field_page('brick', manifest, None,
                                    superseded=['images/F182M/b.fits'],
                                    reasons={'images/F182M/b.fits': state})

    quarantined = _page(rf.QUARANTINED)
    assert 'astrometry checkpoint' in quarantined
    assert 'bad astrometry' in quarantined

    rebuilt = _page(rf.REBUILT)
    assert 'astrometry checkpoint' not in rebuilt, \
        'a rebuild in place is not evidence the checkpoint acted'
    assert 'quarantin' not in rebuilt.lower()
    assert 'rebuilt or replaced' in rebuilt
    assert 'no claim is made about their astrometry' in rebuilt

    # a caller that supplies no reason gets the sentence that asserts less
    unknown = mw.render_field_page(
        'brick', {'version': 'v1', 'built': '2026-08-05T00:00:00Z',
                  'globus_https_base': 'https://example.invalid',
                  'globus_collection_id': '0', 'release_path': '/r',
                  'files': [{'category': 'image', 'kind': 'science',
                             'filter': 'F182M', 'iteration': None,
                             'observation': None, 'dest': 'd', 'url': 'u',
                             'size_bytes': 1}]},
        None, superseded=['d'])
    assert 'astrometry checkpoint' not in unknown


def test_a_preview_built_from_a_withheld_band_is_not_shown():
    """Withholding the download row and leaving the picture up publishes the bad
    astrometry anyway -- the preview is what a reader actually looks at."""
    mw = _make_webpage()
    manifest = {'version': 'v1', 'built': '2026-08-05T00:00:00Z',
                'globus_https_base': 'https://example.invalid',
                'globus_collection_id': '0', 'release_path': '/r',
                'files': [{'category': 'image', 'kind': 'science', 'filter': 'F182M',
                           'iteration': None, 'observation': None,
                           'dest': 'images/F182M/bad.fits', 'url': 'u',
                           'size_bytes': 1}]}
    page = mw.render_field_page(
        'cloudc', manifest, 'assets/cloudc.jpg',
        previews=[('assets/cloudc_rgb_f212n_f187n_f182m.jpg',
                   'cloudc_rgb_f212n_f187n_f182m')],
        superseded=['images/F182M/bad.fits'])
    assert 'cloudc_rgb_f212n_f187n_f182m' not in page
    assert 'class=preview ' not in page


def _gc2211_manifest():
    """gc2211: two observations, both shipping F200W and F277W."""
    files = []
    for obs in ('o023', 'o050'):
        for band in ('F200W', 'F277W'):
            files.append({'category': 'image', 'kind': 'science', 'filter': band,
                          'iteration': None, 'observation': obs,
                          'dest': f'images/{obs}/{band}/m.fits',
                          'url': f'u-{obs}-{band}', 'size_bytes': 1})
    return {'version': 'v1', 'built': '2026-08-05T00:00:00Z',
            'globus_https_base': 'https://example.invalid',
            'globus_collection_id': '0', 'release_path': '/r', 'files': files}


def test_previews_are_withheld_by_observation_not_by_band():
    """gc2211's five observations all ship F200W/F277W, so a band-keyed rule
    drops EVERY preview when one observation is superseded -- including ones
    rendered entirely from o023's live mosaics -- while the same page still
    offers o023's F200W and F277W for download.  The manifest carries
    `observation` and the stems carry the obs token; both were ignored."""
    mw = _make_webpage()
    previews = [('a/gc2211_o023_rgb_f277w_mean_f200w.jpg',
                 'gc2211_o023_rgb_f277w_mean_f200w'),
                ('a/gc2211_o050_rgb_f277w_mean_f200w.jpg',
                 'gc2211_o050_rgb_f277w_mean_f200w')]
    page = mw.render_field_page(
        'gc2211', _gc2211_manifest(), 'assets/gc2211.jpg', previews=previews,
        superseded=['images/o050/F200W/m.fits', 'images/o050/F277W/m.fits'])
    assert 'gc2211_o023_rgb_f277w_mean_f200w' in page, \
        "the live pointing's preview was dropped for a sibling's supersession"
    assert 'gc2211_o050_rgb_f277w_mean_f200w' not in page
    # and the downloads agree with the pictures, which is what broke before
    assert 'u-o023-F200W' in page and 'u-o050-F200W' not in page

    # With the withheld render FIRST, the headline image must move to the
    # survivor: `assets/<field>.jpg` is a byte copy of previews[0], so dropping
    # it from the gallery alone leaves it as the page's main picture.
    page = mw.render_field_page(
        'gc2211', _gc2211_manifest(), 'assets/gc2211.jpg',
        previews=list(reversed(previews)),
        superseded=['images/o050/F200W/m.fits', 'images/o050/F277W/m.fits'])
    assert 'gc2211_o050_rgb_f277w_mean_f200w' not in page
    assert 'gc2211_o023_rgb_f277w_mean_f200w' in page
    assert 'assets/gc2211.jpg' not in page, \
        'the headline image is still the copy of the withheld render'


def test_an_off_convention_preview_stem_fails_closed():
    """`stem.partition("_rgb_")[2]` is '' for a stem without `_rgb_`, so the
    band intersection was empty and the preview was SHOWN.  The picture is what
    a reader looks at; one that cannot be tied to anything must not survive a
    withholding round."""
    mw = _make_webpage()
    manifest = {'version': 'v1', 'built': '2026-08-05T00:00:00Z',
                'globus_https_base': 'https://example.invalid',
                'globus_collection_id': '0', 'release_path': '/r',
                'files': [{'category': 'image', 'kind': 'science', 'filter': 'F182M',
                           'iteration': None, 'observation': None,
                           'dest': 'images/F182M/bad.fits', 'url': 'u',
                           'size_bytes': 1}]}
    page = mw.render_field_page(
        'cloudc', manifest, 'assets/cloudc.jpg',
        previews=[('assets/cloudc_pretty.jpg', 'cloudc_pretty')],
        superseded=['images/F182M/bad.fits'])
    assert 'cloudc_pretty' not in page
    # ... but with nothing withheld it is published as before
    page = mw.render_field_page(
        'cloudc', manifest, 'assets/cloudc.jpg',
        previews=[('assets/cloudc_pretty.jpg', 'cloudc_pretty')])
    assert 'cloudc_pretty' in page
