"""The Aladin Lite fallback chain: CDS first, self-hosted copies when it is down.

``aladin.cds.unistra.fr`` went down on 2026-10-28 (connect hung ~25 s, then
refused), and every generator that loaded Aladin Lite from a single CDS
``<script>`` tag hung on load and then rendered with ``window.A`` undefined.
``jwst_gc_pipeline.aladin_loader`` is the one place the chain is built; every
static-``<script>`` generator uses it instead of writing its own CDS tag.  CDS
stays the default source, so the chain is only safe with a per-source load
timeout: the node-driven tests below replay a hanging CDS against the
generated loader.
"""
import importlib.util
import os
import shutil
import subprocess

import json
import re

import pytest

from jwst_gc_pipeline.aladin_loader import (
    ALADIN_JS_CDS, ALADIN_JS_MIRROR, ALADIN_JS_SELFHOSTED, ALADIN_JS_URLS,
    ALADIN_TIMEOUT_MS, LANDING_CHAIN, LANDING_JS_CDS, LANDING_JS_MIRROR, LANDING_JS_SELFHOSTED,
    LANDING_MARKER, MARKER, aladin_fallback_chain_html, patch_hips_landing_html,
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

def test_cds_is_tried_first_then_selfhosted_then_mirror():
    html = aladin_fallback_chain_html()
    assert html.index(ALADIN_JS_CDS) < html.index(ALADIN_JS_SELFHOSTED)
    assert html.index(ALADIN_JS_SELFHOSTED) < html.index(ALADIN_JS_MIRROR)


def test_exactly_one_cds_script_url():
    html = aladin_fallback_chain_html()
    assert html.count(ALADIN_JS_CDS) == 1


def test_no_parser_blocking_script_tag():
    """A `<script src>` tag (or a document.write of one) blocks the parser
    for as long as its host takes to answer -- ~30 s when CDS hangs -- so the
    chain must load every source from JS, asynchronously."""
    html = aladin_fallback_chain_html()
    assert '<script src=' not in html
    assert 'document.write' not in html


def test_unavailable_stub_present():
    """Until a source loads, `A` is a stub whose `init` is a promise (so code
    written as `A.init.then(...)` waits instead of throwing), and the reader
    sees a visible notice if no source loads."""
    html = aladin_fallback_chain_html()
    assert 'stub.init = new Promise(' in html
    assert 'window.A = stub;' in html
    assert 'could not be loaded' in html


def test_marker_present_once():
    html = aladin_fallback_chain_html()
    assert html.count(MARKER) == 1


def test_no_bare_curly_imbalance():
    """Sanity check against a typo in the hand-written JS literal: equal
    counts of every bracket pair."""
    html = aladin_fallback_chain_html()
    assert html.count('{') == html.count('}')
    assert html.count('(') == html.count(')')


def test_url_tuple_is_cds_first():
    assert ALADIN_JS_URLS == (ALADIN_JS_CDS, ALADIN_JS_SELFHOSTED, ALADIN_JS_MIRROR)


# ------------------------------------------ the loader, replayed under node
#
# A fake DOM and a fake clock: each `<script>` the loader appends is answered
# from a per-URL script -- load (sets window.A) or error after N ms, or hang
# (never answers).  Microtasks are flushed between timer events, so the
# promise chain behaves as it would in a browser.

_HARNESS = r"""
const vm = require('vm');
let input = '';
process.stdin.on('data', d => { input += d; });
process.stdin.on('end', main);
async function main() {
  const {js, net} = JSON.parse(input);
  let now = 0, seq = 0;
  const timers = [], requested = [], warnings = [];
  let notice = false;
  const setT = (fn, ms) => { seq += 1; timers.push({at: now + ms, fn, id: seq}); return seq; };
  const clearT = id => { const i = timers.findIndex(t => t.id === id); if (i >= 0) { timers.splice(i, 1); } };
  const ctx = {console: {warn: m => warnings.push(String(m)), log() {}},
               setTimeout: setT, clearTimeout: clearT};
  const body = {
    firstChild: null,
    insertBefore(el) { el.parentNode = body; notice = true; },
    removeChild(el) { el.parentNode = null; notice = false; },
  };
  ctx.document = {
    body,
    createElement: tag => ({tag, style: {}, setAttribute() {}, parentNode: null}),
    addEventListener(ev, fn) { if (ev === 'DOMContentLoaded') { setT(fn, 0); } },
    head: {appendChild(s) {
      requested.push({url: s.src, t: now});
      const r = net[s.src] || {kind: 'hang'};
      if (r.kind === 'load') {
        setT(() => {
          ctx.A = {name: r.name, init: new Promise(res => { setT(res, r.init_ms || 0); })};
          s.onload();
        }, r.ms);
      } else if (r.kind === 'error') {
        setT(() => s.onerror(), r.ms);
      } else if (r.kind === 'empty') {
        setT(() => s.onload(), r.ms);
      }
    }},
  };
  vm.createContext(ctx);
  ctx.window = ctx;
  ctx.clock = () => now;
  vm.runInContext(js, ctx);
  vm.runInContext('var ready = null, readyAt = null;'
                  + ' A.init.then(function () { ready = A.name; readyAt = clock(); });', ctx);
  for (let step = 0; step < 1000; step++) {
    await new Promise(r => setImmediate(r));
    if (!timers.length) { break; }
    timers.sort((a, b) => a.at - b.at || a.id - b.id);
    const t = timers.shift();
    now = t.at;
    t.fn();
  }
  await new Promise(r => setImmediate(r));
  console.log(JSON.stringify({ready: ctx.ready, readyAt: ctx.readyAt, final: (ctx.A && ctx.A.name) || null,
                              unavailable: Boolean(ctx.A && ctx.A.unavailable),
                              requested, notice, warnings}));
}
"""


def _replay(net):
    node = shutil.which('node')
    if node is None:
        pytest.skip('node is not installed')
    js = re.search(r'<script>\n(.*)\n</script>', aladin_fallback_chain_html(), re.S).group(1)
    out = subprocess.run([node, '-e', _HARNESS], input=json.dumps({'js': js, 'net': net}),
                         capture_output=True, text=True, timeout=60, check=True)
    return json.loads(out.stdout)


def test_loader_uses_cds_when_it_answers():
    res = _replay({ALADIN_JS_CDS: {'kind': 'load', 'ms': 300, 'name': 'cds'}})
    assert res['ready'] == 'cds'
    assert [r['url'] for r in res['requested']] == [ALADIN_JS_CDS]
    assert not res['notice']


def test_page_code_waits_for_the_library_init():
    """`A.init.then(...)` written against the stub runs only once the loaded
    library's own `A.init` has resolved (its WebAssembly start-up), not as
    soon as the bundle has loaded."""
    res = _replay({ALADIN_JS_CDS: {'kind': 'load', 'ms': 300, 'init_ms': 500, 'name': 'cds'}})
    assert res['ready'] == 'cds'
    assert res['readyAt'] == 800


def test_loader_moves_on_when_cds_hangs():
    """The 2026-10-28 failure: CDS never answers.  The self-hosted copy is
    requested at the timeout, not after the browser's own connect timeout."""
    res = _replay({ALADIN_JS_SELFHOSTED: {'kind': 'load', 'ms': 200, 'name': 'self'}})
    assert res['ready'] == 'self'
    assert [r['url'] for r in res['requested']] == [ALADIN_JS_CDS, ALADIN_JS_SELFHOSTED]
    assert res['requested'][1]['t'] == ALADIN_TIMEOUT_MS
    assert not res['notice']


def test_loader_moves_on_at_once_when_cds_errors():
    res = _replay({ALADIN_JS_CDS: {'kind': 'error', 'ms': 50},
                   ALADIN_JS_SELFHOSTED: {'kind': 'load', 'ms': 200, 'name': 'self'}})
    assert res['ready'] == 'self'
    assert res['requested'][1]['t'] == 50


def test_a_source_that_loads_without_aladin_lite_is_skipped():
    """A 200 answer that defines no `A` (a captive portal, an HTML error
    page) leaves the stub in place; the loader must move on, not adopt the
    stub as the library and wait forever on its own `init`."""
    res = _replay({ALADIN_JS_CDS: {'kind': 'empty', 'ms': 50},
                   ALADIN_JS_SELFHOSTED: {'kind': 'load', 'ms': 200, 'name': 'self'}})
    assert res['ready'] == 'self'
    assert res['final'] == 'self'
    assert res['requested'][1]['t'] == 50


def test_late_cds_does_not_replace_the_loaded_library():
    """CDS answers after its timeout, once the self-hosted copy is running:
    the late bundle still executes and reassigns window.A, and the loader
    must put the first library back."""
    res = _replay({ALADIN_JS_CDS: {'kind': 'load', 'ms': ALADIN_TIMEOUT_MS + 900, 'name': 'cds'},
                   ALADIN_JS_SELFHOSTED: {'kind': 'load', 'ms': 200, 'name': 'self'}})
    assert res['ready'] == 'self'
    assert res['final'] == 'self'


def test_late_cds_that_beats_the_fallback_is_used():
    """If CDS answers late but before the self-hosted copy, CDS is the first
    library loaded and is kept; the mirror is never requested."""
    res = _replay({ALADIN_JS_CDS: {'kind': 'load', 'ms': ALADIN_TIMEOUT_MS + 100, 'name': 'cds'},
                   ALADIN_JS_SELFHOSTED: {'kind': 'load', 'ms': 900, 'name': 'self'}})
    assert res['ready'] == 'cds'
    assert res['final'] == 'cds'
    assert ALADIN_JS_MIRROR not in [r['url'] for r in res['requested']]


def test_late_cds_stops_the_chain():
    """CDS answers late while the self-hosted copy hangs: the self-hosted
    timeout firing afterwards must not go on to fetch the mirror."""
    res = _replay({ALADIN_JS_CDS: {'kind': 'load', 'ms': ALADIN_TIMEOUT_MS + 100, 'name': 'cds'}})
    assert res['ready'] == 'cds'
    assert [r['url'] for r in res['requested']] == [ALADIN_JS_CDS, ALADIN_JS_SELFHOSTED]


def test_loader_reaches_the_mirror():
    res = _replay({ALADIN_JS_MIRROR: {'kind': 'load', 'ms': 100, 'name': 'mirror'}})
    assert res['ready'] == 'mirror'
    assert [r['url'] for r in res['requested']] == list(ALADIN_JS_URLS)


def test_a_source_answering_after_the_notice_clears_it():
    """Every source timed out, so the notice is up; CDS then answers.  The
    page uses it and the notice goes away."""
    res = _replay({ALADIN_JS_CDS: {'kind': 'load', 'ms': 3 * ALADIN_TIMEOUT_MS + 500, 'name': 'cds'}})
    assert res['ready'] == 'cds'
    assert res['final'] == 'cds'
    assert not res['notice']


def test_loader_gives_up_with_a_notice_and_a_pending_init():
    res = _replay({url: {'kind': 'error', 'ms': 10} for url in ALADIN_JS_URLS})
    assert res['ready'] is None
    assert res['unavailable']
    assert res['notice']


# ------------------------------------------------------- generators wired in

def test_cmd_explorer_page_uses_the_chain():
    from jwst_gc_pipeline.quicklook import cmdview
    from jwst_gc_pipeline.quicklook.tests.test_cmdview import _data
    page = cmdview.render(_data())
    assert MARKER in page
    assert page.count(ALADIN_JS_CDS) == 1
    assert page.index(ALADIN_JS_CDS) < page.index(ALADIN_JS_SELFHOSTED)


def test_panner_page_uses_the_chain_and_drops_the_cds_css_link():
    from jwst_gc_pipeline.quicklook import panner
    page = panner.render_page()
    assert MARKER in page
    assert page.count(ALADIN_JS_CDS) == 1
    assert page.index(ALADIN_JS_CDS) < page.index(ALADIN_JS_SELFHOSTED)
    # the CDS aladin.css <link> is gone: the 3.8.2 UMD build injects its own
    assert 'aladin.css' not in page.lower()


def test_cmz_explorer_page_uses_the_chain():
    mw = _load('make_webpage', os.path.join(_REL, 'make_webpage.py'))
    html = mw.render_cmz_explorer('cmz/hips/CMZ_color')
    assert MARKER in html
    assert html.count(ALADIN_JS_CDS) == 1
    assert html.index(ALADIN_JS_CDS) < html.index(ALADIN_JS_SELFHOSTED)


def test_field_overview_dynamic_loader_points_at_cds_first():
    """`field_overview.py` builds its `<script>` element in JS at runtime (on
    a button click), so it cannot splice in the static HTML chain; its one
    URL constant names the first source it tries, CDS."""
    fo = _load('field_overview_loader', os.path.join(_REL, 'field_overview.py'))
    assert fo.ALADIN_JS == ALADIN_JS_CDS


def _overview_page(fo, **kw):
    geoms = [{'field': 'brick', 'href': 'brick.html', 'color': '#46bcd6',
              'polys': [[[266.5, -28.7], [266.6, -28.7],
                         [266.6, -28.6], [266.5, -28.6]]]}]
    page = fo.section(geoms, **kw)
    payload = re.search(r'<script id=ov-data type="application/json">(.*?)</script>',
                        page, re.S).group(1)
    return page, json.loads(payload)


def test_field_overview_loader_walks_every_source_in_order():
    """The overview panel's payload carries the whole CDS-first list and the
    per-source timeout; its loader moves to the next source on `onerror` or
    on the timeout, and keeps the first library that loads."""
    fo = _load('field_overview_walk', os.path.join(_REL, 'field_overview.py'))
    page, payload = _overview_page(fo)
    assert payload['aladin'] == list(ALADIN_JS_URLS)
    assert payload['aladin_timeout_ms'] == ALADIN_TIMEOUT_MS
    assert 'var sources = [].concat(data.aladin);' in page
    assert 'loadFrom(i + 1);' in page
    assert '}, data.aladin_timeout_ms);' in page
    assert 'window.A = winner;' in page
    # a single URL still works, as a one-element list
    _one, payload = _overview_page(fo, aladin_src='https://example.org/aladin.js')
    assert payload['aladin'] == ['https://example.org/aladin.js']


def test_field_overview_loader_script_parses():
    node = shutil.which('node')
    if node is None:
        pytest.skip('node is not installed')
    fo = _load('field_overview_parse', os.path.join(_REL, 'field_overview.py'))
    page, _payload = _overview_page(fo)
    scripts = re.findall(r'<script>(.*?)</script>', page, re.S)
    assert scripts
    for js in scripts:
        subprocess.run([node, '--check', '-'], input=js, text=True,
                       capture_output=True, timeout=60, check=True)


def test_landing_template_order_is_data_rc_starformation_cds():
    """data.rc serves the /orange template directly; the starformation copy
    is synced by hand and can lag (on 2026-10-28 it still loaded Aladin Lite
    self-hosted first), so data.rc comes first and CDS's unpatched template last."""
    first = [ln.strip() for ln in LANDING_CHAIN.splitlines() if ln.strip().startswith('<script src=')]
    assert first == [f'<script src="{LANDING_JS_MIRROR}"></script>']
    assert (LANDING_CHAIN.index(LANDING_JS_MIRROR) < LANDING_CHAIN.index(LANDING_JS_SELFHOSTED)
            < LANDING_CHAIN.index(LANDING_JS_CDS))


def test_each_landing_fallback_loads_only_when_the_template_is_missing():
    fallbacks = [ln.strip() for ln in LANDING_CHAIN.splitlines()
                 if LANDING_JS_SELFHOSTED in ln or LANDING_JS_CDS in ln]
    assert len(fallbacks) == 2
    assert all(ln.startswith("<script>typeof buildLandingPage !== 'undefined'"
                             " || document.write(") for ln in fallbacks)
    assert "if (typeof buildLandingPage === 'undefined') {" in LANDING_CHAIN


def test_hipsgen_build_rgb_hips_patches_the_written_index_html(monkeypatch, tmp_path):
    from jwst_gc_pipeline.cmz import hipsgen

    jar = tmp_path / 'Hipsgen.jar'
    jar.write_text('not a real jar', encoding='utf-8')
    monkeypatch.setattr(hipsgen.shutil, 'which', lambda _x: '/usr/bin/java')
    monkeypatch.setenv('HIPSGEN_JAR', str(jar))
    out_dir = tmp_path / 'rgb_hips'
    out_dir.mkdir()
    (out_dir / 'index.html').write_text(
        '<html><head><script src="https://aladin.cds.unistra.fr/hips-templates/'
        'hips-landing-page.js"></script></head><body></body></html>',
        encoding='utf-8')
    monkeypatch.setattr(hipsgen, '_run', lambda cmd: None)

    hipsgen.build_rgb_hips(str(out_dir), 'r', 'g', 'b')

    assert LANDING_MARKER in (out_dir / 'index.html').read_text(encoding='utf-8')


def test_landing_tag_match_ignores_case():
    page = ('<html><head><SCRIPT SRC="https://aladin.cds.unistra.fr/hips-templates/'
            'hips-landing-page.js"></SCRIPT></head><body></body></html>')
    patched, status = patch_hips_landing_html(page)
    assert status == 'patched'
    assert LANDING_MARKER in patched
