"""Field pages link their catalog manifold viewer only when it is built."""
import importlib.util
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


def test_no_viewer_no_link(mw, tmp_path):
    """A card pointing at a 404 is worse than none."""
    assert mw.manifold_href(tmp_path, 'brick') is None
    # a directory without index.html (a half-copied tree) is not a viewer
    (tmp_path / 'brick_manifold').mkdir()
    assert mw.manifold_href(tmp_path, 'brick') is None


def test_built_viewer_is_linked(mw, tmp_path):
    (tmp_path / 'brick_manifold').mkdir()
    (tmp_path / 'brick_manifold' / 'index.html').write_text('<html></html>')
    assert mw.manifold_href(tmp_path, 'brick') == 'brick_manifold/'


def test_treasury_uses_its_existing_directory(mw, tmp_path):
    """The 10678 viewer was published as treasury_manifold/ before the
    per-field naming; its URL is already shared, so it keeps it."""
    (tmp_path / 'treasury_manifold').mkdir()
    (tmp_path / 'treasury_manifold' / 'index.html').write_text('<html></html>')
    assert mw.manifold_href(tmp_path, 'gc-treasury') == 'treasury_manifold/'


def test_field_page_carries_the_link_only_when_given(mw):
    with_link = mw.render_field_page('brick', _manifest('brick'), '',
                                     manifold_href='brick_manifold/')
    assert "href='brick_manifold/'" in with_link
    without = mw.render_field_page('brick', _manifest('brick'), '')
    assert 'manifold' not in without
