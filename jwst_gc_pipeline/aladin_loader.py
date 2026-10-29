"""Aladin Lite fallback chain: CDS first, self-hosted copies when CDS is down.

``aladin.cds.unistra.fr`` went down on 2026-10-28: a TCP connect to it hung for
roughly 25 s and then was refused.  Every page that loaded Aladin Lite from a
single ``<script src="https://aladin.cds.unistra.fr/...">`` tag hung for that
long on load and then rendered with ``window.A`` undefined, breaking every
feature built on it -- the sky view never showed and the rest of the page
could be blocked behind it too, depending on script order.

Verified same-build mirrors of the exact CDS 3.8.2 UMD bundle (md5
``e2210a76c05d20a897d6425e51fb2409``, confirmed to set ``window.A`` the same
way) are kept at two UF-controlled hosts:

* primary: ``https://starformation.astro.ufl.edu/vendor/aladin-lite/3.8.2/aladin.js``
* mirror:  ``https://data.rc.ufl.edu/pub/adamginsburg/vendor/aladin-lite/3.8.2/aladin.js``

CDS stays the DEFAULT, first source: it is the upstream and usually works.
The self-hosted copies are tried only when CDS fails to load or does not
answer within :data:`ALADIN_TIMEOUT_MS`.  The timeout is what makes a
CDS-first order safe: a source whose connect HANGS (the 2026-10-28 failure
mode) never fires ``onerror`` until the browser's own ~30 s connect timeout,
and a parser-inserted ``<script src>`` tag blocks the page for all of it.  So
every loader here is ASYNC and moves to the next source on ``onerror`` OR on
the timeout, whichever comes first.  A source that answers after its timeout
still executes (a started fetch cannot be cancelled), so the first library to
load is the one kept: a later copy's assignment to ``window.A`` is undone.

Every page generator that loads Aladin Lite from a STATIC ``<script>`` tag
should build it with :func:`aladin_fallback_chain_html` instead of writing its
own ``<script src=".../aladin.js">`` tag (and should drop any CDS
``aladin.css`` ``<link>`` alongside it: the 3.8.2 UMD build injects its own
CSS, and the ``<link>`` would otherwise block rendering while CDS hangs).
That helper installs a stub ``window.A`` whose ``init`` promise resolves once
the first source has loaded and started, so page code that builds its view
inside an ``A.init.then(...)`` callback needs no change.  Code
that touches ``A`` OUTSIDE an ``A.init.then`` callback sees the stub; every
generator in this repo already waits on ``A.init``.

A generator that loads Aladin Lite DYNAMICALLY -- a JS-built ``<script>``
element, e.g. a button-click loader such as
``scripts/release/field_overview.py``'s overview panel -- cannot splice in
this HTML.  Such a loader should walk :data:`ALADIN_JS_URLS` in order with the
same per-source timeout and first-loaded-wins rule, as the overview panel does.

At least one test (``jwst_gc_pipeline/cmz/tests/test_release_integration.py``)
asserts the CDS URL appears in a generated page.

See ``/vendor/README.txt`` on the starformation host for the mirrors'
provenance.
"""
import json
import re

#: The Aladin Lite build every mirror and CDS serve.  Pinned, not "latest":
#: the self-hosted copies are snapshots of this one version, so "latest"
#: cannot be offered consistently across the chain.
VERSION = '3.8.2'

ALADIN_JS_SELFHOSTED = f'https://starformation.astro.ufl.edu/vendor/aladin-lite/{VERSION}/aladin.js'
ALADIN_JS_MIRROR = f'https://data.rc.ufl.edu/pub/adamginsburg/vendor/aladin-lite/{VERSION}/aladin.js'
ALADIN_JS_CDS = f'https://aladin.cds.unistra.fr/AladinLite/api/v3/{VERSION}/aladin.js'

#: Ordered CDS first (the default), then the self-hosted copies.  Every
#: loader walks it in this order.
ALADIN_JS_URLS = (ALADIN_JS_CDS, ALADIN_JS_SELFHOSTED, ALADIN_JS_MIRROR)

#: How long a source may take to load before the next one is tried.  Matches
#: the per-source timeout in the self-hosted landing template
#: (``/vendor/hips-templates/hips-landing-page.js``).
ALADIN_TIMEOUT_MS = 8000

#: Shown to the reader when every source in the chain failed.  ``A.init`` is
#: left as a promise that never resolves (rather than ``undefined``), so code
#: written as ``A.init.then(...)`` stays idle instead of throwing.
_NOTICE = ("The interactive sky viewer (Aladin Lite) could not be loaded: its "
           "servers are unreachable right now. The rest of this page still "
           "works; reload later to get the sky map back.")

#: Marker left in the generated HTML -- both so a second call is idempotent
#: (a generator that runs this helper twice does not double the chain) and so
#: a grep can find every page that carries it.
MARKER = 'aladin-fallback-chain'

_LOADER_JS = """(function () {
  var sources = %(sources)s;
  var timeoutMs = %(timeout)d;
  var stub = {unavailable: false};
  var settle;
  stub.init = new Promise(function (resolve) { settle = resolve; });
  window.A = stub;
  var winner = null;
  var notice = null;
  function giveUp() {
    stub.unavailable = true;
    var show = function () {
      notice = document.createElement('div');
      notice.setAttribute('role', 'status');
      notice.style.cssText = 'margin:8px;padding:8px 12px;border:1px solid #c90;background:#fff6e0;color:#000;font:14px sans-serif';
      notice.textContent = %(notice)s;
      document.body.insertBefore(notice, document.body.firstChild);
    };
    if (document.body) { show(); } else { document.addEventListener('DOMContentLoaded', show); }
  }
  function tryFrom(i) {
    if (i >= sources.length) { giveUp(); return; }
    var moved = false;
    function next(why) {
      if (moved || winner) { return; }
      moved = true;
      console.warn('Aladin Lite: ' + why + ' ' + sources[i] + '; trying the next source');
      tryFrom(i + 1);
    }
    var timer = setTimeout(function () {
      next('no answer within ' + timeoutMs + ' ms from');
    }, timeoutMs);
    var s = document.createElement('script');
    s.charset = 'utf-8';
    s.onerror = function () { clearTimeout(timer); next('failed to load'); };
    s.onload = function () {
      clearTimeout(timer);
      if (winner) {
        // a source that answered after its timeout: keep the first library
        window.A = winner;
        return;
      }
      if (!window.A || window.A === stub) { next('no Aladin Lite in'); return; }
      winner = window.A;
      if (notice && notice.parentNode) { notice.parentNode.removeChild(notice); }
      Promise.resolve(winner.init).then(function () { settle(); });
    };
    s.src = sources[i];
    document.head.appendChild(s);
  }
  tryFrom(0);
})();"""


def aladin_fallback_chain_html():
    """HTML that replaces a single ``<script src=".../aladin.js">`` tag.

    Loads :data:`ALADIN_JS_CDS` asynchronously; if it fails, or has not loaded
    within :data:`ALADIN_TIMEOUT_MS`, moves on to :data:`ALADIN_JS_SELFHOSTED`,
    then :data:`ALADIN_JS_MIRROR`.  Until a source has loaded, ``window.A`` is
    a stub whose ``init`` promise resolves once the loaded library's own
    ``A.init`` has, so ``A.init.then(...)`` written right after this block
    runs against the real library.  If no source loads, ``A.init`` never
    resolves and a one-line notice is shown, so the rest of the page keeps
    working instead of throwing on the first ``A.whatever`` call.
    """
    js = _LOADER_JS % {'sources': json.dumps(list(ALADIN_JS_URLS)),
                       'timeout': ALADIN_TIMEOUT_MS,
                       'notice': json.dumps(_NOTICE)}
    return (
        f'<!-- {MARKER}: Aladin Lite {VERSION} from CDS first; when CDS fails or does not answer within\n'
        f'     {ALADIN_TIMEOUT_MS // 1000} s, from the self-hosted copies.  See /vendor/README.txt. -->\n'
        f'<script>\n{js}\n</script>'
    )


# --- HiPS landing pages -------------------------------------------------------
#
# Hipsgen's ``INDEX`` action (``jwst_gc_pipeline.cmz.hipsgen``) writes an
# ``index.html`` into every HiPS directory whose own
# ``<script src="https://aladin.cds.unistra.fr/hips-templates/hips-landing-page.js">``
# tag fails the same way: the page hangs while CDS is unreachable, then
# ``buildLandingPage`` is undefined and the landing page renders blank.  The
# template is a patched copy of the CDS one, so the chain loads it from the
# self-hosted copies first and CDS last.  The self-hosted template
# (``/vendor/hips-templates/``) then loads Aladin Lite in the same order as
# :data:`ALADIN_JS_URLS` -- CDS first, with the same per-source timeout -- and
# opens the viewer on the HiPS itself.
# ``scripts/release/patch_hips_landing_page.py`` is the command-line wrapper.

LANDING_TAG = re.compile(
    r"""[ \t]*<script[^>]*\ssrc=(["'])https?://aladin\.(?:cds\.unistra\.fr|u-strasbg\.fr)"""
    r"""/hips-templates/hips-landing-page\.js\1[^>]*>\s*</script>""",
    re.IGNORECASE)

#: Left in the rewritten page -- both so a second run is a no-op and so a
#: grep can find every landing page carrying the chain.
LANDING_MARKER = "landing-fallback-chain"

LANDING_JS_SELFHOSTED = "https://starformation.astro.ufl.edu/vendor/hips-templates/hips-landing-page.js"
LANDING_JS_MIRROR = "https://data.rc.ufl.edu/pub/adamginsburg/vendor/hips-templates/hips-landing-page.js"
LANDING_JS_CDS = "https://aladin.cds.unistra.fr/hips-templates/hips-landing-page.js"

LANDING_CHAIN = f"""    <!-- {LANDING_MARKER}: patched HiPS landing template from self-hosted copies first, CDS template last;
         the template loads Aladin Lite from CDS first, self-hosted copies on failure.  See /vendor/README.txt. -->
    <script src="{LANDING_JS_SELFHOSTED}"></script>
    <script>typeof buildLandingPage !== 'undefined' || document.write('<script src="{LANDING_JS_MIRROR}"><\\/script>');</script>
    <script>typeof buildLandingPage !== 'undefined' || document.write('<script src="{LANDING_JS_CDS}"><\\/script>');</script>
    <script>
    if (typeof buildLandingPage === 'undefined') {{
      window.buildLandingPage = function () {{
        document.addEventListener('DOMContentLoaded', function () {{
          document.body.textContent = 'The interactive HiPS viewer could not be loaded: its servers are unreachable right now. The HiPS itself (properties, Norder*/ tiles in this directory) is unaffected; reload later.';
        }});
      }};
    }}
    </script>"""


def patch_hips_landing_html(text):
    """Return ``(new_text, status)``.  ``status`` is one of ``already``
    (the marker is already present), ``no-cds-tag`` (nothing to do), or
    ``patched``."""
    if LANDING_MARKER in text:
        return text, "already"
    new, n = LANDING_TAG.subn(lambda m: LANDING_CHAIN, text)
    if n == 0:
        return text, "no-cds-tag"
    if n > 1:
        raise ValueError("more than one HiPS landing-page CDS tag; patch by hand")
    return new, "patched"


def patch_hips_landing_page(path):
    """Patch ``path`` in place.  Returns the status (see :func:`patch_hips_landing_html`)."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    new, status = patch_hips_landing_html(text)
    if status == "patched":
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(new)
    return status
