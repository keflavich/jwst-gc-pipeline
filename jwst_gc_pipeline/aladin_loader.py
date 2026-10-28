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
this HTML.  Such a loader should walk :data:`ALADIN_JS_URLS` in order, moving
to the next source when one fails to load (``onerror``), as the overview panel
does.

Neither form has a load timeout of its own.  The static chain's sources are
parser-inserted ``<script>`` tags, and the dynamic loader moves on only at
``onerror``, so a source whose connect HANGS (rather than failing fast) still
delays the fallback by the browser's own connect timeout.  With the UF hosts
first this matters only when one of them hangs; in the 2026-10-28 failure
mode, CDS is reached only after both UF copies have failed.  The self-hosted
landing template (``/vendor/hips-templates/hips-landing-page.js``) does apply
a per-source timeout to the Aladin Lite load it makes itself.

See ``/vendor/README.txt`` on the starformation host for the mirrors'
provenance.
"""
import re

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


# --- HiPS landing pages -------------------------------------------------------
#
# Hipsgen's ``INDEX`` action (``jwst_gc_pipeline.cmz.hipsgen``) writes an
# ``index.html`` into every HiPS directory whose own
# ``<script src="https://aladin.cds.unistra.fr/hips-templates/hips-landing-page.js">``
# tag fails the same way: the page hangs while CDS is unreachable, then
# ``buildLandingPage`` is undefined and the landing page renders blank.  The
# self-hosted template (``/vendor/hips-templates/``) also loads Aladin Lite from
# the self-hosted copies first and opens the viewer on the HiPS itself.
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

LANDING_CHAIN = f"""    <!-- {LANDING_MARKER}: HiPS landing template from self-hosted copies first (it loads Aladin Lite from them too);
         CDS is the last source.  See /vendor/README.txt. -->
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
