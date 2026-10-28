"""Aladin Lite fallback chain: self-hosted copies first, CDS last.

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

CDS is kept as the LAST source in the chain -- both because it is the
upstream and usually works, and because at least one test
(``jwst_gc_pipeline/cmz/tests/test_release_integration.py``) asserts the CDS
URL still appears in a generated page.

Every page generator that loads Aladin Lite from a STATIC ``<script>`` tag
should build it with :func:`aladin_fallback_chain_html` instead of writing its
own ``<script src=".../aladin.js">`` tag (and should drop any CDS
``aladin.css`` ``<link>`` alongside it: the 3.8.2 UMD build injects its own
CSS, and the ``<link>`` would otherwise block rendering while CDS hangs).

A generator that loads Aladin Lite DYNAMICALLY -- a JS-built ``<script>``
element, e.g. a button-click loader such as
``scripts/release/field_overview.py``'s overview panel -- cannot splice in
this HTML.  Such a loader should still point its URL constant at
:data:`ALADIN_JS_SELFHOSTED`, and may walk :data:`ALADIN_JS_URLS` instead if
it already supports trying more than one source.

See ``/vendor/README.txt`` on the starformation host for the mirrors'
provenance.
"""

#: The Aladin Lite build every mirror and CDS serve.  Pinned, not "latest":
#: the self-hosted copies are snapshots of this one version, so "latest"
#: cannot be offered consistently across the chain.
VERSION = '3.8.2'

ALADIN_JS_SELFHOSTED = f'https://starformation.astro.ufl.edu/vendor/aladin-lite/{VERSION}/aladin.js'
ALADIN_JS_MIRROR = f'https://data.rc.ufl.edu/pub/adamginsburg/vendor/aladin-lite/{VERSION}/aladin.js'
ALADIN_JS_CDS = f'https://aladin.cds.unistra.fr/AladinLite/api/v3/{VERSION}/aladin.js'

#: Ordered self-hosted-first, CDS-last.  For a dynamic loader that can try
#: more than one source.
ALADIN_JS_URLS = (ALADIN_JS_SELFHOSTED, ALADIN_JS_MIRROR, ALADIN_JS_CDS)

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


def aladin_fallback_chain_html():
    """HTML that replaces a single ``<script src=".../aladin.js">`` tag.

    Loads :data:`ALADIN_JS_SELFHOSTED`; if ``window.A`` is not set after that
    (blocked, 404, offline host), falls through to :data:`ALADIN_JS_MIRROR`,
    then :data:`ALADIN_JS_CDS`.  If none of the three set ``window.A``, installs
    a stub ``A`` whose ``init`` promise never resolves and shows a one-line
    notice, so the rest of the page keeps working instead of throwing on the
    first ``A.whatever`` call.
    """
    return (
        f'<!-- {MARKER}: Aladin Lite {VERSION} from self-hosted copies first, so the viewer keeps working when\n'
        f'     aladin.cds.unistra.fr is down; CDS is the last source.  See /vendor/README.txt. -->\n'
        f'<script src="{ALADIN_JS_SELFHOSTED}" charset="utf-8"></script>\n'
        f'<script>window.A || document.write(\'<script src="{ALADIN_JS_MIRROR}" charset="utf-8"><\\/script>\');</script>\n'
        f'<script>window.A || document.write(\'<script src="{ALADIN_JS_CDS}" charset="utf-8"><\\/script>\');</script>\n'
        '<script>\n'
        'if (!window.A) {\n'
        '  window.A = {init: new Promise(function () {}), unavailable: true};\n'
        '  document.addEventListener(\'DOMContentLoaded\', function () {\n'
        '    var d = document.createElement(\'div\');\n'
        '    d.setAttribute(\'role\', \'status\');\n'
        '    d.style.cssText = \'margin:8px;padding:8px 12px;border:1px solid #c90;background:#fff6e0;color:#000;font:14px sans-serif\';\n'
        f'    d.textContent = {_NOTICE!r};\n'
        '    document.body.insertBefore(d, document.body.firstChild);\n'
        '  });\n'
        '}\n'
        '</script>'
    )
