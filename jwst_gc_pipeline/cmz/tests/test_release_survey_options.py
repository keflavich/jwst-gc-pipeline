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
    # the same F770W sky, background-matched -- offered beside the plain layer
    # so the matching can be switched against rather than assumed
    'jwst_gc_treasury_miri_bgmatch_hips',
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


# --- turning the footprint rectangles off ------------------------------------

def _rendered():
    geoms = [{'field': 'brick', 'href': 'brick.html', 'color': '#46bcd6',
              'polys': [[[266.5, -28.7], [266.6, -28.7],
                         [266.6, -28.6], [266.5, -28.6]]]}]
    return fo.section(geoms)


def test_the_viewer_can_hide_every_field_outline():
    """At the treasury layers' scale a tile is smaller than the rectangle drawn
    over it, so without this there is no way to see the imagery underneath."""
    html = _rendered()
    assert 'id=ov-outlines' in html
    assert 'Field outlines' in html


def test_one_control_reaches_every_field():
    """Each field has its OWN graphicOverlay -- that is what gives it a colour
    and a name in Aladin's layer list -- so there is no single object to
    toggle and they have to be collected."""
    html = _rendered()
    assert 'fieldOverlays.push(ov)' in html
    assert 'fieldOverlays.forEach' in html


def test_hiding_does_not_destroy_the_overlays():
    """`removeOverlay` would lose the objects, so turning them back on would
    need every polygon rebuilt."""
    html = _rendered()
    assert 'ov.show();' in html and 'ov.hide();' in html
    # a CALL, not the token -- the code comment explaining why it is avoided
    # naturally contains the word
    assert not re.search(r'\.removeOverlay\s*\(', html)


def test_the_outlines_start_visible():
    """They are the point of this map most of the time."""
    html = _rendered()
    assert re.search(r"outBtn\.className = 'on';", html)
    assert re.search(r"outBtn\.setAttribute\('aria-pressed', 'true'\)", html)
    # the variable must not be named so that it contains the survey button's
    # `b.addEventListener('click'` as a substring: an existing test splits the
    # page on that string and would land on this handler instead
    assert "ob.addEventListener('click'" not in html


def test_the_control_is_hidden_until_the_viewer_exists():
    """A control that does nothing until 1.8 MB of Aladin loads is a dead
    button; the survey switcher is unhidden in the same place for the same
    reason."""
    html = _rendered()
    assert 'id=ov-outlines class=ov-surveys hidden' in html
    assert 'outlineBar.hidden = false;' in html


def test_the_clickable_field_markers_are_not_hidden_with_the_rectangles():
    """They are the navigation to each field's page and do not cover the
    imagery; hiding them would make the cleaned-up view a dead end."""
    html = _rendered()
    hide = html.index('fieldOverlays.forEach')
    end = html.index('outlineBar.appendChild', hide)
    assert 'cat.' not in html[hide:end]
