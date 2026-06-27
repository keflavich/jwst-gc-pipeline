"""JWST photometry pipeline for crowded Galactic Center fields."""

try:
    from .version import version as __version__
except ImportError:
    __version__ = "0.0.0.dev0"

# Stamp the running pipeline git commit into the primary header (GCPIPEV) of
# every FITS file written in this process, so catalogs/images are traceable to
# the exact code that produced them.  Installs a one-time HDUList.writeto hook.
from .provenance import install_fits_provenance_hook, get_pipeline_commit
install_fits_provenance_hook()

__all__ = ["__version__", "get_pipeline_commit", "install_fits_provenance_hook"]
