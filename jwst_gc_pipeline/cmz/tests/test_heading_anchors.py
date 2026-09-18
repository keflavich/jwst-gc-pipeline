"""Every heading on the release site is linkable."""
import importlib.util
import os
import re

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


def test_a_heading_gets_an_id_from_its_words(mw):
    """Derived from the text, not a counter: a link keeps working when the
    heading moves and breaks when its subject changes, which is what someone
    pasting it into an email wants."""
    out = mw.anchor_headings('<h2>Astrometric offsets</h2>')
    assert out == '<h2 id="astrometric-offsets">Astrometric offsets</h2>'


def test_markup_and_entities_inside_a_heading_do_not_reach_the_id(mw):
    """`<h2>Mosaic images <span class=muted>(resampled...)</span></h2>` is the
    real shape on a field page."""
    out = mw.anchor_headings(
        '<h2>Mosaic images <span class=muted>(resampled onto a sky grid)</span></h2>')
    assert 'id="mosaic-images-resampled-onto-a-sky-grid"' in out
    assert '<span' in out, 'the heading content must survive untouched'

    ampersand = mw.anchor_headings('<h3>Command line with <code>globus</code> '
                                   '&mdash; recommended</h3>')
    assert re.search(r'id="command-line-with-globus-recommended"', ampersand)


def test_repeated_headings_get_distinct_ids(mw):
    """Two headings can legitimately read the same, and an id that silently
    collides sends every link to whichever came first."""
    out = mw.anchor_headings('<h2>Catalogs</h2><h3>x</h3><h2>Catalogs</h2>')
    assert 'id="catalogs"' in out and 'id="catalogs-2"' in out


def test_an_existing_id_is_left_alone(mw):
    """Anything that already anchors itself keeps the id it publishes, since
    someone may already be linking to it."""
    out = mw.anchor_headings('<h2 id="chosen">Astrometric offsets</h2>')
    assert out == '<h2 id="chosen">Astrometric offsets</h2>'


def test_every_heading_on_a_real_page_is_anchored(mw, tmp_path):
    """The reason this post-processes the finished page rather than adding a
    helper to ~20 `out.append("<h2>")` sites: a helper each caller has to
    remember is one half of them will not."""
    page = mw.render_index([], quicklooks=mw.QUICKLOOKS)
    headings = re.findall(r'<h[1-6]([^>]*)>', page)
    assert headings, 'the index has headings'
    missing = [h for h in headings if not re.search(r'\bid\s*=', h)]
    assert missing == [], missing


def test_every_renderer_anchors_its_headings(mw):
    """Three pages call `anchor_headings`, and one test that builds only the
    index holds one of them: dropping the call from `render_field_page` or
    from `render_help` left every test green while two thirds of the site lost
    its anchors.
    """
    manifest = {'field': 'gc-treasury', 'version': 'v1.8-2026.09',
                'group': None, 'release_path': '/releases/v1.8/gc-treasury',
                'built': '2026-09-17T12:00:00', 'mode': 'copy',
                'globus_collection_id': 'x',
                'globus_https_base': 'https://example.invalid', 'files': []}
    pages = {
        'field': mw.render_field_page('gc-treasury', manifest, ''),
        'help': mw.render_help(),
        'index': mw.render_index([], quicklooks=mw.QUICKLOOKS),
    }
    for name, page in pages.items():
        headings = re.findall(r'<h[1-6]([^>]*)>', page)
        assert headings, f'{name} has headings'
        missing = [h for h in headings if not re.search(r'\bid\s*=', h)]
        assert missing == [], (name, missing)


def test_a_heading_with_no_word_characters_still_gets_an_id(mw):
    """An id is required to be non-empty; `<h2>&mdash;</h2>` would otherwise
    produce `id=""`, which is invalid and unlinkable."""
    out = mw.anchor_headings('<h2>&mdash;</h2>')
    assert 'id="section"' in out
