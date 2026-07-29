"""Jay Anderson STDGDC distortion-correction integration (starlist-level).

See ``stdgdc.py`` for the file-format / application conventions (with peppar
citations), ``gdc_wcs.py`` for the affine-anchored corrected sky solution, and
``correct_catalog.py`` for the per-exposure catalog CLI.  Nothing in the main
pipeline imports this package; it is opt-in.
"""
from .stdgdc import STDGDC, GDCFileNotFoundError
from .gdc_wcs import GDCSkySolution

__all__ = ['STDGDC', 'GDCFileNotFoundError', 'GDCSkySolution']
