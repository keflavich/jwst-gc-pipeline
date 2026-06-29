"""Catalog helpers shared across the photometry and reduction subpackages."""
from astropy.coordinates import SkyCoord


def catalog_skycoord(cat, on_missing='raise'):
    """Extract a `~astropy.coordinates.SkyCoord` from a catalog ``Table``.

    Tries, in order: a ``sky_centroid`` or ``skycoord`` SkyCoord column, then
    ``('RA', 'DEC')`` or ``('ra', 'dec')`` float columns interpreted as FK5.

    Parameters
    ----------
    cat : astropy.table.Table
        Catalog to read coordinates from.
    on_missing : {'raise', 'none'}
        What to do when no coordinate columns are found.  ``'raise'`` raises
        ``ValueError`` (the make_reference behavior); ``'none'`` returns ``None``
        so the caller can fall back to pixel columns (the align_to_catalogs
        behavior).

    Notes
    -----
    Replaces the previously duplicated ``_extract_skycoord``
    (photometry/make_reference_from_pipeline_catalogs.py) and
    ``_get_catalog_skycoord`` (reduction/align_to_catalogs.py).  Those two
    differed only in the ``sky_centroid``/``skycoord`` lookup order and in
    raise-vs-None on miss; both are SkyCoord columns holding the same positions,
    so the unified order is immaterial for real catalogs (which carry one or the
    other, not both).
    """
    cols = cat.colnames
    if 'sky_centroid' in cols:
        return cat['sky_centroid']
    if 'skycoord' in cols:
        return cat['skycoord']
    if 'RA' in cols and 'DEC' in cols:
        return SkyCoord(cat['RA'], cat['DEC'], frame='fk5')
    if 'ra' in cols and 'dec' in cols:
        return SkyCoord(cat['ra'], cat['dec'], frame='fk5')
    if on_missing == 'none':
        return None
    raise ValueError(
        f"Could not find sky coordinates in catalog table; columns: {cols}")
