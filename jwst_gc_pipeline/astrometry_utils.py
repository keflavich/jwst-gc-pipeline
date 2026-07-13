"""Small shared astrometry helpers (masked->NaN coercion + PM propagation).

Factored out of ``reduction/build_gaia_virac2_refcat.py``,
``reduction/build_gaia_virac2_refcat_byquery.py``, and
``photometry/generate_offsets_table.py``, which each carried a byte-identical
copy of ``farr``/``prop`` (drift risk: any change to the PM-propagation formula
had to be mirrored by hand in three places).
"""
import numpy as np


# --------------------------------------------------------------------------
# RA-difference naming convention (2026-07).
#
# The recurring ~cos(dec) mix-up (relock_exposures emitted an on-sky shift that
# fix_alignment consumed as a coordinate rotation -> every Brick band landed
# ~69 mas off VIRAC2) came from an ambiguous ``dra`` that meant different things
# in different modules.  Convention, used verbatim from here on:
#
#     dra_coordinate = ra1 - ra2                 (raw RA-coordinate difference)
#     dra            = (ra1 - ra2) * cos(dec)    (on-sky angular RA offset)
#
# ``dra_coordinate`` is what ``jwst.tweakreg.utils.adjust_wcs(delta_ra=...)``
# consumes and what an offsets-table ``dra`` column MUST hold: a coordinate
# rotation whose *on-sky* effect is ``dra_coordinate * cos(dec)`` (verified
# preflight: delta_ra=-90 mas -> -78.9 mas on-sky = -90*cos(28.8)).  ``dra`` is
# that on-sky angular offset -- what you compare to an on-sky threshold /
# separation.  Keep the two names distinct so a value never crosses the
# coordinate<->on-sky boundary unlabelled:  dra = dra_coordinate * cos(dec).
def dra_coordinate(ra1, ra2):
    """Coordinate RA difference ``ra1 - ra2`` (degrees in == degrees out).

    THIS is the value fed to ``adjust_wcs(delta_ra=...)`` and written to an
    offsets table's ``dra`` column.  Its on-sky effect is ``*cos(dec)``.
    """
    return ra1 - ra2


def dra(ra1, ra2, dec):
    """On-sky angular RA offset ``(ra1 - ra2) * cos(dec)`` (2026-07 convention).

    The true-angle RA separation, i.e. ``dra_coordinate(ra1, ra2) * cos(dec)``.
    Distinct from :func:`dra_coordinate` (= ``ra1 - ra2``, the coordinate rotation
    adjust_wcs consumes) so the on-sky and coordinate quantities can never be
    silently swapped.  ``dec`` in degrees.
    """
    return (ra1 - ra2) * np.cos(np.radians(dec))


def _resolve_existing_path(path):
    """Return ``path`` if it exists, else the /blue<->/orange basepath variant.

    Per-frame catalog ``meta['FILENAME']`` records the crf under whichever
    basepath (/blue/... scratch vs /orange/... project) was live at build time;
    normalise so reprojection finds the current crf regardless.
    """
    import os
    if os.path.exists(path):
        return path
    cand = path.replace('//', '/')
    if os.path.exists(cand):
        return cand
    for a, b in (('/blue/adamginsburg/adamginsburg/jwst/', '/orange/adamginsburg/jwst/'),
                 ('/orange/adamginsburg/jwst/', '/blue/adamginsburg/adamginsburg/jwst/')):
        alt = cand.replace(a, b)
        if os.path.exists(alt):
            return alt
    raise FileNotFoundError(f"crf not found for reprojection: {path!r}")


def reproject_xy_to_world(cat, crf_path=None, xcol='x_fit', ycol='y_fit', sci_ext='SCI'):
    """Sky positions from the STABLE detector pixel centroids through the CURRENT
    crf WCS -- NOT the catalog's cached ``skycoord_centroid``.

    A per-frame ``daophot_basic`` catalog stores ``skycoord_centroid`` computed
    from the crf WCS *at catalog-build time*.  When the crf is re-drizzled with a
    different ``assign_wcs``/distortion generation the cached RA/Dec goes stale
    (measured drift up to ~48 mas between Brick reduction runs), while the
    detector ``x_fit``/``y_fit`` are generation-invariant.  Re-deriving RA/Dec
    from x/y through the live crf WCS keeps the tie solving on the SAME
    generation it is about to correct.  ``crf_path`` defaults to
    ``cat.meta['FILENAME']``.  Returns a SkyCoord.
    """
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    path = _resolve_existing_path(crf_path or cat.meta['FILENAME'])
    with fits.open(path) as hl:
        w = WCS(hl[sci_ext].header)
    return SkyCoord(w.pixel_to_world(np.asarray(cat[xcol], float),
                                     np.asarray(cat[ycol], float)))


def farr(x):
    """Coerce ``x`` to a plain float array, turning masked / non-finite to NaN."""
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, float)), np.nan), float)


def prop(ra, dec, pmra, pmde, dt):
    """Proper-motion propagate ``(ra, dec)`` [deg] by ``dt`` years.

    ``pmra``/``pmde`` are in mas/yr (pmra is the on-sky rate, i.e. already
    includes the cos(dec) factor, so it is divided back out here).  Non-finite
    proper motions are treated as zero.
    """
    pmra = np.where(np.isfinite(pmra), pmra, 0.)
    pmde = np.where(np.isfinite(pmde), pmde, 0.)
    return ra + (pmra * dt / 3.6e6) / np.cos(np.radians(dec)), dec + (pmde * dt / 3.6e6)
