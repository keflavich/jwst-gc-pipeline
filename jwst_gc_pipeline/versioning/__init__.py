"""Pipeline versioning, provenance-fingerprinting, and rerun-skip decisions.

This package layers a *content-addressed* provenance system over the existing
``jwst_gc_pipeline.provenance`` git-commit stamp so that, for any pipeline
product, we can answer two questions cheaply:

1. **What produced this?** -- a release *tag* (``YYYY-MM-DD_PR<n>``, or the dev
   form ``YYYY-MM-DD_PR<n>_<commit>``) plus the exact code / params / parent
   ``jwst`` version / CRDS context that fed the stage.
2. **Does a stage need to be re-run?** -- decomposed per *facet* (science data
   vs. WCS vs. other header metadata) so the engine picks the MINIMAL action
   (skip / restamp / reproject-only / refit / re-reduce) instead of a blanket
   re-run.

See ``VERSIONING_PROVENANCE.md`` (same directory) for the full design, the
decision matrix, and the seeding-cascade invariant.

Sub-modules:
  * ``tags``        -- pipeline-tag resolution + the ``assert_runnable_version``
                       run guard.
  * ``fingerprint`` -- per-facet content hashes (data/wcs/meta) + code/params
                       fingerprints.
  * ``prov_sidecar``-- read/write the ``<product>.prov.json`` provenance record.
  * ``stamping``    -- emit the sidecar + mirrored FITS keys at a stage write
                       (``stamp_product`` / ``stamp_catalog`` and their
                       fail-soft ``try_stamp_*`` variants).
  * ``rerun``       -- the per-stage rerun-skip decision engine + ``plan`` CLI.
"""

from .tags import (
    UntaggedPipelineError,
    get_pipeline_tag,
    assert_runnable_version,
    is_release_tag,
    parse_tag,
)

__all__ = [
    'UntaggedPipelineError',
    'get_pipeline_tag',
    'assert_runnable_version',
    'is_release_tag',
    'parse_tag',
]
