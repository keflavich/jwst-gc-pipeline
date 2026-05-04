"""JWST photometry pipeline for crowded Galactic Center fields."""

try:
    from .version import version as __version__
except ImportError:
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
