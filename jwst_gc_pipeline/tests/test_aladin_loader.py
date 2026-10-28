"""The Aladin Lite fallback chain: self-hosted copies first, CDS last.

``aladin.cds.unistra.fr`` went down on 2026-10-28 (connect hung ~25 s, then
refused), and every generator that loaded Aladin Lite from a single CDS
``<script>`` tag hung on load and then rendered with ``window.A`` undefined.
``jwst_gc_pipeline.aladin_loader`` is the one place the self-hosted-first
chain is built; every static-``<script>`` generator uses it instead of
writing its own CDS tag.
"""
import importlib.util
import os

from jwst_gc_pipeline.aladin_loader import (
    ALADIN_JS_CDS, ALADIN_JS_MIRROR, ALADIN_JS_SELFHOSTED, MARKER,
    aladin_fallback_chain_html,
)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REL = os.path.join(_REPO, 'scripts', 'release')


def _load(name, path):
    import sys
    if _REL not in sys.path:
        sys.path.insert(0, _REL)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- the helper

def test_selfhosted_appears_before_mirror_and_cds():
    html = aladin_fallback_chain_html()
    assert html.index(ALADIN_JS_SELFHOSTED) < html.index(ALADIN_JS_MIRROR)
    assert html.index(ALADIN_JS_MIRROR) < html.index(ALADIN_JS_CDS)


def test_exactly_one_cds_script_url():
    html = aladin_fallback_chain_html()
    assert html.count(ALADIN_JS_CDS) == 1


def test_unavailable_stub_present():
    """If every source fails, `A.init` must stay a pending promise (so code
    written as `A.init.then(...)` stays idle instead of throwing) and the
    reader must see a visible notice."""
    html = aladin_fallback_chain_html()
    assert 'window.A = {init: new Promise(function () {}), unavailable: true}' in html
    assert 'could not be loaded' in html
    assert 'DOMContentLoaded' in html


def test_marker_present_once():
    html = aladin_fallback_chain_html()
    assert html.count(MARKER) == 1


def test_no_bare_curly_imbalance():
    """Sanity check against a typo in the hand-written JS literal: equal
    counts of every bracket pair (apart from the JS object literal, which is
    intentionally one `{` + one matching `}` -- counted below with the rest)."""
    html = aladin_fallback_chain_html()
    assert html.count('{') == html.count('}')
    assert html.count('(') == html.count(')')


# ------------------------------------------------------- generators wired in

def test_cmd_explorer_page_uses_the_chain():
    from jwst_gc_pipeline.quicklook import cmdview
    from jwst_gc_pipeline.quicklook.tests.test_cmdview import _data
    page = cmdview.render(_data())
    assert MARKER in page
    assert page.count(ALADIN_JS_CDS) == 1
    assert page.index(ALADIN_JS_SELFHOSTED) < page.index(ALADIN_JS_CDS)


def test_panner_page_uses_the_chain_and_drops_the_cds_css_link():
    from jwst_gc_pipeline.quicklook import panner
    page = panner.render_page()
    assert MARKER in page
    assert page.count(ALADIN_JS_CDS) == 1
    assert page.index(ALADIN_JS_SELFHOSTED) < page.index(ALADIN_JS_CDS)
    # the CDS aladin.css <link> is gone: the 3.8.2 UMD build injects its own
    assert 'aladin.css' not in page.lower()


def test_cmz_explorer_page_uses_the_chain():
    mw = _load('make_webpage', os.path.join(_REL, 'make_webpage.py'))
    html = mw.render_cmz_explorer('cmz/hips/CMZ_color')
    assert MARKER in html
    assert html.count(ALADIN_JS_CDS) == 1
    assert html.index(ALADIN_JS_SELFHOSTED) < html.index(ALADIN_JS_CDS)


def test_field_overview_dynamic_loader_points_at_selfhosted():
    """`field_overview.py` builds its `<script>` element in JS at runtime (on
    a button click), so it cannot splice in the static HTML chain -- but its
    one URL constant must still point at the self-hosted copy rather than at
    CDS directly."""
    fo = _load('field_overview_loader', os.path.join(_REL, 'field_overview.py'))
    assert fo.ALADIN_JS == ALADIN_JS_SELFHOSTED
