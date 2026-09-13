"""Background layers offered by the release page's interactive sky view.

The list used to lead with `jwst_cmz_hips`, which is a SYMLINK to
`jwst_nir_hips` -- so "JWST CMZ" was the near-infrared mosaic under a name that
sounds like it covers everything, and the MIRI mosaic beside it was not offered
at all.
"""
import importlib.util
import os
import re

import pytest

_FO = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    'scripts', 'release', 'field_overview.py'))
_spec = importlib.util.spec_from_file_location('field_overview_surveys', _FO)
fo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fo)

#: The four JWST layers the page must offer, by the directory each resolves to.
REQUIRED = (
    'jwst_nir_hips',
    'jwst_miri_hips',
    'jwst_gc_treasury_hips',
    'jwst_gc_treasury_miri_hips',
)


def _urls():
    return [u for _n, u in fo.SURVEYS]


@pytest.mark.parametrize('layer', REQUIRED)
def test_each_jwst_layer_is_offered(layer):
    # exact path segment: `jwst_gc_treasury_hips` is a prefix of
    # `jwst_gc_treasury_miri_hips` under a naive substring test, so one entry
    # would satisfy the check for both.
    assert any(u.rstrip('/').endswith('/' + layer) for u in _urls()), layer


def test_the_ambiguous_cmz_alias_is_not_used():
    """`jwst_cmz_hips` is a symlink to `jwst_nir_hips`.  Offering both would
    put the same tiles on the page twice under two names, one of which implies
    it is the whole CMZ dataset."""
    assert not any('jwst_cmz_hips' in u for u in _urls())


def test_the_labels_say_which_band():
    """These are different sky at different wavelengths, not one dataset, so a
    label that omits the band is the thing that made the alias confusing."""
    labels = {n for n, _u in fo.SURVEYS}
    assert {'CMZ NIR', 'CMZ MIRI', 'Treasury NIRCam', 'Treasury MIRI'} <= labels


def test_every_jwst_layer_is_served_from_the_cors_enabled_host():
    """Aladin reads tiles into a WebGL texture, so a host that sends no
    Access-Control-Allow-Origin renders nothing -- and data.rc.ufl.edu, where
    these are built, answers 401 on top of that."""
    for url in _urls():
        if url.startswith('http'):
            assert url.startswith('https://starformation.astro.ufl.edu/avm_images/')
            assert 'data.rc.ufl.edu' not in url


def test_no_survey_names_a_hips_id_that_does_not_exist():
    """`P/Spitzer/GLIMPSE360` matched nothing at the CDS MOCServer, so that
    button drew nothing and never had."""
    assert 'P/Spitzer/GLIMPSE360' not in _urls()


def test_the_context_surveys_are_still_available():
    """Adding the JWST layers must not cost the all-sky context ones -- the
    treasury layers cover a few arcminutes and are useless for orientation."""
    assert any(not u.startswith('http') for u in _urls())
