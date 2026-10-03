"""Field pages link their catalog manifold viewer only when it is built."""
import importlib.util
import json
import os

import pytest

_REL = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', 'scripts', 'release'))


@pytest.fixture(scope='module')
def mw():
    import sys
    sys.path.insert(0, _REL)
    spec = importlib.util.spec_from_file_location(
        'make_webpage', os.path.join(_REL, 'make_webpage.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(field):
    return {'field': field, 'version': 'v1.7-2026.09', 'group': None,
            'release_path': f'/releases/v1.7/{field}',
            'built': '2026-09-17T12:00:00', 'mode': 'copy',
            'globus_collection_id': 'x',
            'globus_https_base': 'https://example.invalid', 'files': []}


def _viewer(root, name, **manifest):
    d = root / name
    (d / 'data').mkdir(parents=True)
    (d / 'index.html').write_text('<html></html>')
    m = {'title': 'X', 'n': 1000, 'created': '2026-10-02'}
    m.update(manifest)
    (d / 'data' / 'manifest.json').write_text(json.dumps(m))
    return d


def test_no_viewer_no_link(mw, tmp_path):
    """A card pointing at a 404 is worse than none."""
    assert mw.manifold_info(tmp_path, 'brick') is None
    # a half-copied tree (no index.html, or no readable manifest) is not a viewer
    (tmp_path / 'brick_manifold').mkdir()
    assert mw.manifold_info(tmp_path, 'brick') is None
    (tmp_path / 'brick_manifold' / 'index.html').write_text('<html></html>')
    assert mw.manifold_info(tmp_path, 'brick') is None


def test_empty_or_malformed_manifest_is_not_linked(mw, tmp_path):
    _viewer(tmp_path, 'brick_manifold', n=0)
    assert mw.manifold_info(tmp_path, 'brick') is None
    # unparseable JSON (a half-written manifest) is skipped, not raised
    (tmp_path / 'brick_manifold' / 'data' / 'manifest.json').write_text('{')
    assert mw.manifold_info(tmp_path, 'brick') is None


def test_treasury_uses_its_existing_directory(mw, tmp_path):
    """The 10678 viewer was published as treasury_manifold/ before the
    per-field naming; its URL is already shared, so it keeps it."""
    _viewer(tmp_path, 'treasury_manifold')
    assert mw.manifold_info(tmp_path, 'gc-treasury')['href'] == 'treasury_manifold/'


def test_description_comes_from_the_manifest(mw, tmp_path):
    """The page may not claim more than the build did (review of #1028): the
    row count, parent sample and prototype status are the manifest's own."""
    _viewer(tmp_path, 'brick_manifold', title='Brick manifold (prototype)',
            n=3606678, sample={'parent': 'per-pointing quality-cut tables'},
            caveats=['colour axis unreliable in o113'])
    info = mw.manifold_info(tmp_path, 'brick')
    link = mw.manifold_link_html(info, 'v1.7-2026.09')
    assert "href='brick_manifold/'" in link
    assert '3,606,678 stars from per-pointing quality-cut tables' in link
    assert '<span class=tag>prototype</span>' in link
    assert 'colour axis unreliable in o113' in link
    assert 'Every star' not in link


def test_provenance_is_claimed_only_when_recorded(mw, tmp_path):
    """Without a recorded source release, say the viewer comes from the live
    catalogs as of its build date, never that it is this release's catalog."""
    _viewer(tmp_path, 'brick_manifold')
    plain = mw.manifold_link_html(mw.manifold_info(tmp_path, 'brick'),
                                  'v1.7-2026.09')
    assert 'pipeline catalogs as of 2026-10-02' in plain
    assert "this release's catalog files" not in plain

    _viewer(tmp_path, 'sgra_manifold',
            source={'release_version': 'v1.7-2026.09'})
    tied = mw.manifold_link_html(mw.manifold_info(tmp_path, 'sgra'),
                                 'v1.7-2026.09')
    assert "Built from this release's catalog files." in tied
    other = mw.manifold_link_html(mw.manifold_info(tmp_path, 'sgra'),
                                  'v1.8-2026.10')
    assert "this release's catalog files" not in other
    # a viewer built from another release names that release
    assert 'Built from the v1.7-2026.09 catalog files' in other


def test_field_page_carries_the_link_only_when_given(mw, tmp_path):
    _viewer(tmp_path, 'brick_manifold')
    info = mw.manifold_info(tmp_path, 'brick')
    with_link = mw.render_field_page('brick', _manifest('brick'), '',
                                     manifold=info)
    assert "href='brick_manifold/'" in with_link
    without = mw.render_field_page('brick', _manifest('brick'), '')
    assert 'brick_manifold/' not in without
    assert 'catalog manifold viewer' not in without


def test_field_with_one_viewer_per_program(mw, tmp_path):
    """The brick has two programs with disjoint filter sets (2221 o001,
    1182 o004), so it has one viewer per observation; its page links both,
    each named by its own title."""
    _viewer(tmp_path, 'brick_o001_manifold',
            title='Brick (program 2221) Manifold (prototype)')
    _viewer(tmp_path, 'brick_o004_manifold',
            title='Brick (program 1182) Manifold (prototype)')
    _viewer(tmp_path, 'brickx_o001_manifold')   # another field's prefix
    infos = mw.manifold_infos(tmp_path, 'brick')
    assert [i['href'] for i in infos] == ['brick_o001_manifold/',
                                          'brick_o004_manifold/']
    page = mw.render_field_page('brick', _manifest('brick'), '',
                                manifold=infos)
    assert 'Open the Brick (program 2221) Manifold viewer</a>' in page
    assert 'Open the Brick (program 1182) Manifold viewer</a>' in page
    assert 'Open the catalog manifold viewer' not in page


def test_single_viewer_keeps_the_plain_label(mw, tmp_path):
    _viewer(tmp_path, 'sgra_manifold', title='Sgr A Manifold (prototype)')
    infos = mw.manifold_infos(tmp_path, 'sgra')
    page = mw.render_field_page('sgra', _manifest('sgra'), '', manifold=infos)
    assert 'Open the catalog manifold viewer</a>' in page
    assert mw.manifold_infos(tmp_path, 'w51') == []
