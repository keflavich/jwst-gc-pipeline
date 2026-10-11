"""Field pages list the HiPS layers made from their data; the site has a HiPS index."""
import importlib.util
import json
import os
import sys

import pytest

_REL = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', 'scripts', 'release'))


def _load(name):
    sys.path.insert(0, _REL)
    spec = importlib.util.spec_from_file_location(name, os.path.join(_REL, f'{name}.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def hi():
    return _load('hips_index')


@pytest.fixture(scope='module')
def mw():
    return _load('make_webpage')


def _layer(name, title='', order='12'):
    return {'name': name, 'obs_title': title, 'hips_order': order,
            'hips_initial_ra': '266.5', 'hips_initial_dec': '-28.7',
            'hips_initial_fov': '0.2', 'hips_release_date': '2026-09-15T10:00Z'}


INVENTORY = {'collected': '2026-10-10T12:00Z', 'source': 'test',
             'layers': [_layer('SgrB2_RGB_187-182-150_hips', 'JWST Sgr B2 RGB'),
                        _layer('SgrB2_DS_alma_hips', 'ALMA'),
                        _layer('SgrB2_RGB_old_stale_hips', 'old'),
                        _layer('Cloudef_RGB_2100_hips', '/orange/x/y.png'),
                        _layer('jwst_nir_hips', 'CMZ NIRCam coadd'),
                        _layer('MeerKAT_hips', 'radio')]}


def test_field_selection_and_exclusions(hi):
    names = [la['name'] for la in hi.field_layers('sgrb2', INVENTORY)]
    assert names == ['SgrB2_RGB_187-182-150_hips']
    assert hi.field_layers('m92', INVENTORY) == []
    assert hi.field_layers('sgrb2', None) == []


def test_exclusion_matches_whole_tokens(hi):
    assert hi.EXCLUDE.search('jwst_gc_treasury_hips_stale_20260912b')
    assert hi.EXCLUDE.search('Brick_RGB_test_hips')
    assert not hi.EXCLUDE.search('Brick_RGB_checkerboard_hips')
    assert not hi.EXCLUDE.search('Sickle_RGB_contest_hips')


def test_label_falls_back_to_directory_name(hi):
    cloudef = hi.field_layers('cloudef_controlfield', INVENTORY)
    assert [hi.layer_label(la) for la in cloudef] == ['Cloudef_RGB_2100_hips']


def test_aladin_link_points_at_served_layer(hi):
    url = hi.aladin_url(INVENTORY['layers'][0])
    assert url.startswith(hi.ALADIN_LITE)
    assert 'avm_images%2FSgrB2_RGB_187-182-150_hips' in url or \
        'avm_images/SgrB2_RGB_187-182-150_hips' in url


def test_section_omitted_without_inventory(hi):
    assert hi.field_section_html('sgrb2', None) == ''


def test_section_lists_layers_tours_and_aggregates(hi):
    sec = hi.field_section_html('sgrb2', INVENTORY)
    assert 'HiPS sky maps' in sec
    assert 'avm_images/SgrB2_RGB_187-182-150_hips' in sec
    assert 'SgrB2_DS_alma_hips' not in sec
    assert 'sgrb2_wavelength_tour_linear.html' in sec
    assert 'jwst_nir_hips' in sec           # CMZ field -> survey-wide coadd
    assert hi.INDEX_PAGE in sec


def test_field_without_layers_still_says_so(hi):
    sec = hi.field_section_html('wd1', INVENTORY)
    assert 'No HiPS layer has been published' in sec
    assert 'jwst_nir_hips' not in sec       # wd1 is outside the CMZ


def test_cloudef_tour_uses_its_page_name(hi):
    links = hi.tour_links('cloudef_controlfield')
    assert any('cloudef_wavelength_tour_linear.html' in href for href, _, _ in links)


def test_parse_properties_dump(hi):
    text = ('### b_hips\nobs_title = B\nhips_order = 11\nunrelated = x\n'
            '### a_hips\nobs_title = A = first\n')
    layers = hi.parse_properties_dump(text)
    assert [la['name'] for la in layers] == ['a_hips', 'b_hips']
    assert layers[0]['obs_title'] == 'A = first'
    assert 'unrelated' not in layers[1]


def test_collect_local_and_roundtrip(hi, tmp_path):
    for name in ('x_hips', 'y_hips'):
        (tmp_path / name).mkdir()
        (tmp_path / name / 'properties').write_text(f'obs_title = {name}\nhips_order = 9\n')
    out = tmp_path / 'inv.json'
    assert hi.main(['--out', str(out), '--local-root', str(tmp_path)]) == 0
    inv = hi.load_inventory(out)
    assert [la['name'] for la in inv['layers']] == ['x_hips', 'y_hips']
    out.write_text('{not json')
    assert hi.load_inventory(out) is None
    assert hi.load_inventory(tmp_path / 'missing.json') is None


def test_index_page_groups_by_field(hi):
    page = hi.render_index_page(['m92', 'sgrb2', 'wd2'], INVENTORY,
                                lambda title: f'<html><title>{title}</title>',
                                lambda: '<footer></footer>')
    assert "href='sgrb2.html'" in page
    assert 'SgrB2_RGB_187-182-150_hips' in page
    assert "href='m92.html'" not in page    # no layers and no tours


def _manifest(field):
    return {'field': field, 'version': 'v1.7-2026.09', 'group': None,
            'release_path': f'/releases/v1.7/{field}',
            'built': '2026-09-17T12:00:00', 'mode': 'copy',
            'globus_collection_id': 'x',
            'globus_https_base': 'https://example.invalid', 'files': []}


def test_field_page_carries_the_section(mw):
    page = mw.render_field_page('sgrb2', _manifest('sgrb2'), '',
                                hips_inventory=INVENTORY)
    assert 'HiPS sky maps' in page
    assert 'SgrB2_RGB_187-182-150_hips' in page
    page = mw.render_field_page('sgrb2', _manifest('sgrb2'), '')
    assert 'HiPS sky maps' not in page


def test_inventory_json_is_plain(hi, tmp_path):
    doc = hi.write_inventory([_layer('a_hips')], tmp_path / 'i.json', 'test')
    assert json.loads((tmp_path / 'i.json').read_text()) == doc
