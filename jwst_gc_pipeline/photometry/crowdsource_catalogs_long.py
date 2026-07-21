"""DEPRECATED compatibility shim (2026-07-21) — use ``catalog_long`` instead.

This module was renamed to :mod:`jwst_gc_pipeline.photometry.catalog_long`.
This shim exists ONLY so that frozen SLURM batch scripts already sitting in the
queue (submitted before the rename, running
``python -m jwst_gc_pipeline.photometry.crowdsource_catalogs_long``) do not die
with ``ModuleNotFoundError`` when they finally start.

One-cycle lifetime: REMOVE this file once the pending queue has drained and no
batch script references the old name.  Do not add new imports of this module.
"""
import sys
import warnings

warnings.warn(
    "jwst_gc_pipeline.photometry.crowdsource_catalogs_long is a deprecated "
    "compatibility shim; use jwst_gc_pipeline.photometry.catalog_long instead "
    "(module renamed 2026-07; shim will be removed once the pending SLURM "
    "queue drains).",
    DeprecationWarning,
    stacklevel=2,
)

from jwst_gc_pipeline.photometry import catalog_long as _catalog_long
from jwst_gc_pipeline.photometry.catalog_long import main  # noqa: F401

# Mirror catalog_long's full namespace (including underscore-prefixed names,
# which a star-import would miss) so any legacy
# ``from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import X``
# keeps working for every name catalog_long defines.
_shim_module = sys.modules[__name__]
for _name, _value in vars(_catalog_long).items():
    if not _name.startswith("__"):
        setattr(_shim_module, _name, _value)
del _name, _value, _shim_module

if __name__ == "__main__":
    # identical to catalog_long's own __main__ block: main() parses sys.argv
    main()
