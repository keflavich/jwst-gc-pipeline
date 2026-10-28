#!/usr/bin/env python
"""Point a CDS HiPS landing page (hipsgen ``INDEX``) at the self-hosted template.

``hipsgen`` (``jwst_gc_pipeline.cmz.hipsgen.build_mono_hips``) writes an
``index.html`` landing page into every HiPS directory it builds.  That page's
own ``<script src="https://aladin.cds.unistra.fr/hips-templates/hips-landing-page.js">``
tag has the same failure mode as a direct Aladin Lite CDS tag (see
``jwst_gc_pipeline/aladin_loader.py``): when ``aladin.cds.unistra.fr`` is
unreachable (it was down for an extended period starting 2026-10-28), the tag
hangs the page load and then leaves ``buildLandingPage`` undefined, so the
``<script>buildLandingPage(...)</script>`` call the template itself drives
throws and the landing page renders blank.

This script rewrites that one ``<script>`` tag in place: self-hosted
(starformation) -> data.rc mirror -> CDS, then a stub ``buildLandingPage``
that says the viewer is unavailable while leaving the HiPS directory itself
(``properties``, ``Norder*/`` tiles) untouched and reachable.  The self-hosted
template still loads Aladin Lite from the self-hosted copies first (see
``jwst_gc_pipeline/aladin_loader.py``), so any ``alScriptURL`` the landing
page passes through is unaffected.

Idempotent: a page already carrying the ``landing-fallback-chain`` marker is
left alone.  Raises if a page has more than one matching tag (patch by hand).

Usage::

    python scripts/release/patch_hips_landing_page.py HIPS_DIR/index.html [...]
"""
import re
import sys

TAG = re.compile(
    r"""[ \t]*<script[^>]*\ssrc=(["'])https?://aladin\.(?:cds\.unistra\.fr|u-strasbg\.fr)"""
    r"""/hips-templates/hips-landing-page\.js\1[^>]*>\s*</script>""")

#: Left in the rewritten page -- both so a second run is a no-op and so a
#: grep can find every landing page carrying the chain.
MARKER = "landing-fallback-chain"

SELFHOSTED = "https://starformation.astro.ufl.edu/vendor/hips-templates/hips-landing-page.js"
MIRROR = "https://data.rc.ufl.edu/pub/adamginsburg/vendor/hips-templates/hips-landing-page.js"
CDS = "https://aladin.cds.unistra.fr/hips-templates/hips-landing-page.js"

CHAIN = f"""    <!-- {MARKER}: HiPS landing template from self-hosted copies first (it loads Aladin Lite from them too);
         CDS is the last source.  See /vendor/README.txt. -->
    <script src="{SELFHOSTED}"></script>
    <script>typeof buildLandingPage !== 'undefined' || document.write('<script src="{MIRROR}"><\\/script>');</script>
    <script>typeof buildLandingPage !== 'undefined' || document.write('<script src="{CDS}"><\\/script>');</script>
    <script>
    if (typeof buildLandingPage === 'undefined') {{
      window.buildLandingPage = function () {{
        document.addEventListener('DOMContentLoaded', function () {{
          document.body.textContent = 'The interactive HiPS viewer could not be loaded: its servers are unreachable right now. The HiPS itself (properties, Norder*/ tiles in this directory) is unaffected; reload later.';
        }});
      }};
    }}
    </script>"""


def patch(text):
    """Return ``(new_text, status)``.  ``status`` is one of ``already``
    (the marker is already present), ``no-cds-tag`` (nothing to do), or
    ``patched``."""
    if MARKER in text:
        return text, "already"
    new, n = TAG.subn(lambda m: CHAIN, text)
    if n == 0:
        return text, "no-cds-tag"
    if n > 1:
        raise ValueError("more than one HiPS landing-page CDS tag; patch by hand")
    return new, "patched"


def patch_file(path):
    """Patch ``path`` in place.  Returns the status (see :func:`patch`)."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    new, status = patch(text)
    if status == "patched":
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(new)
    return status


def main(argv=None):
    paths = sys.argv[1:] if argv is None else argv
    if not paths:
        print(__doc__)
        return 1
    for path in paths:
        status = patch_file(path)
        print(f"{status}\t{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
