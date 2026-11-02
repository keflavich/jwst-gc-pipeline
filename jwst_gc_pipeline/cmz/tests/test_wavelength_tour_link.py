"""Field pages link their color wavelength tour when one is deployed."""
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


def test_listed_field_gets_absolute_link(mw):
    link = mw.wavelength_tour_html('brick')
    assert ("href='https://starformation.astro.ufl.edu/Aladin_tours/"
            "brick_wavelength_tour_linear.html'") in link
    assert 'color wavelength tour' in link


def test_unlisted_field_gets_nothing(mw):
    assert mw.wavelength_tour_html('gc-treasury') == ''
    assert mw.wavelength_tour_html('m92') == ''


def _manifest(field):
    return {'field': field, 'version': 'v1.7-2026.09', 'group': None,
            'release_path': f'/releases/v1.7/{field}',
            'built': '2026-09-17T12:00:00', 'mode': 'copy',
            'globus_collection_id': 'x',
            'globus_https_base': 'https://example.invalid', 'files': []}


def test_field_page_carries_the_link(mw):
    page = mw.render_field_page('sgrc', _manifest('sgrc'), '')
    assert 'sgrc_wavelength_tour_linear.html' in page
    page = mw.render_field_page('m92', _manifest('m92'), '')
    assert 'wavelength_tour' not in page
