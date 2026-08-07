"""Small shared astrometry helpers (masked->NaN coercion + PM propagation).

Factored out of ``reduction/build_gaia_virac2_refcat_byquery.py``,
``reduction/build_gaia_virac2_refcat_byquery.py``, and
``photometry/generate_offsets_table.py``, which each carried a byte-identical
copy of ``farr``/``prop`` (drift risk: any change to the PM-propagation formula
had to be mirrored by hand in three places).
"""
import os
import re

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
    crf WCS, superseding the catalog's cached ``skycoord_centroid``.

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


# --------------------------------------------------------------------------
# Correction provenance (2026-07-13): make an astrometric correction safe to
# apply NO MATTER WHEN by recording, with every correction,
#   (a) the BASE coordinate it was solved against,
#   (b) the offset itself (in the labelled coordinate convention), and
#   (c) the TARGET coordinate it must produce,
# and by verifying (a) before applying and (c) after.  A correction that
# cannot verify its base refuses to apply (a generation-stale table can then
# never be silently stacked on the wrong frame -- the failure mode behind the
# ~69 mas VIRAC2 offset and the brick-1182 v001 20" incident).

# Header keys that define a frame's WCS GENERATION: if any of these differ
# between the solve and the apply, the astrometric frame may have moved.
GENERATION_KEYS = ("CAL_VER", "CRDS_CTX", "DVACORR")


def wcs_fiducial(header, wcs=None, x=1024.0, y=1024.0):
    """Sky coordinate of a fiducial detector pixel through ``header``'s WCS,
    with any baked ``RAOFFSET``/``DEOFFSET`` removed -- i.e. the frame's
    SIAF-generation fiducial.  This is the per-frame BASE coordinate a
    correction is solved against and verified against at apply time.

    Returns ``(ra_deg, dec_deg)``.
    """
    from astropy.wcs import WCS
    w = wcs if wcs is not None else WCS(header)
    sky = w.pixel_to_world(float(x), float(y))
    ra0 = float(header.get("RAOFFSET", 0.0)) / 3600.0    # coordinate degrees
    de0 = float(header.get("DEOFFSET", 0.0)) / 3600.0
    return float(sky.ra.deg - ra0), float(sky.dec.deg - de0)


def generation_stamp(header):
    """The WCS-generation identity of a frame: which pipeline/reference/DVA
    state produced its astrometric frame.  Solvers record it with each
    correction; ``fix_alignment`` refuses (or warns) when the frame it is
    about to correct carries a different stamp."""
    return {k.lower(): str(header.get(k, "")) for k in GENERATION_KEYS}


def base_mismatch_mas(base_ra, base_dec, now_ra, now_dec):
    """On-sky distance (mas) between a recorded base fiducial and the frame's
    current fiducial.  > a few mas means the frame belongs to a different
    reduction generation than the one the correction was solved on."""
    cosd = np.cos(np.radians(0.5 * (float(base_dec) + float(now_dec))))
    return float(np.hypot((float(now_ra) - float(base_ra)) * cosd,
                          float(now_dec) - float(base_dec)) * 3.6e6)


def pick_refcat(cands, field=None):
    """The refcat for THIS observation, from the candidates on disk.

    A refcat may carry an observation token (``..._o028.fits``) because it was
    built for one pointing.  Selection used to be ``sorted(cands)[-1]``, which
    picks the LAST NAME ALPHABETICALLY -- and ``_o028`` sorts after the bare
    ``gaia_virac2_refcat_epoch2023.71.fits``, so on gc2211 every observation got
    o028's catalog:

        gaia_virac2_refcat_epoch2023.71.fits          <- generic, wanted
        gaia_virac2_refcat_epoch2023.71_o028.fits     <- chosen for ALL of them

    o023 was then tied against a reference built for a pointing arcminutes away
    and the m2 checkpoint measured a -9.28" per-exposure correction, which the
    magnitude limit refused to write (correctly -- the measurement was wrong,
    not the frame).  Its five observations are 0.3-17.6 arcmin apart, so a
    neighbour's refcat is not a degraded reference, it is the wrong sky.

    Order: this observation's token, else an untokened refcat, else refuse --
    picking SOME other observation's is never right, and doing it silently is
    how this cost a full o023 chain.
    """
    if not cands:
        return None
    tok = re.compile(r'_o(\d{3})\.fits$')
    tokened = {m.group(1): p for p in cands for m in [tok.search(p)] if m}
    untokened = [p for p in cands if not tok.search(p)]
    if field:
        f3 = str(field).zfill(3)
        if f3 in tokened:
            return tokened[f3]
    if untokened:
        return sorted(untokened)[-1]
    if tokened:
        raise ValueError(
            f"astrom checkpoint: the only reference catalogs under "
            f"{os.path.dirname(cands[0])} are built for other observations "
            f"({sorted(tokened)}), and this run is "
            f"{'o' + str(field).zfill(3) if field else 'untagged'}.  Tying to "
            f"another pointing's reference is not a degraded measurement, it is "
            f"the wrong sky -- build one for this observation "
            f"(build_gaia_virac2_refcat_byquery.py) or point ASTROM_REFCAT at "
            f"the right file.")
    return None
