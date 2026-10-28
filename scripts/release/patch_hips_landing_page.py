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
import sys

from jwst_gc_pipeline.aladin_loader import (  # noqa: F401  (re-exported for callers of this script)
    LANDING_CHAIN as CHAIN, LANDING_JS_CDS as CDS, LANDING_JS_MIRROR as MIRROR,
    LANDING_JS_SELFHOSTED as SELFHOSTED, LANDING_MARKER as MARKER, LANDING_TAG as TAG,
    patch_hips_landing_html as patch, patch_hips_landing_page as patch_file)


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
