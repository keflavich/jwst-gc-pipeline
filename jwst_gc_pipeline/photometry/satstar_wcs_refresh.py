"""Re-project a cached per-exposure satstar catalog's sky columns through the
frame's CURRENT GWCS.

The problem
-----------
``saturated_star_finding`` writes one ``*_m<N>_satstar_catalog.fits`` per
exposure and stores, for every fitted saturated star, BOTH the pixel centroid
(``xcentroid``/``ycentroid``) and the sky position it had at fit time
(``skycoord_fit``, plus the ``sat_com_ra``/``sat_com_dec`` component anchor).
That catalog is a CACHE: ``load_or_make_satstar_catalog`` returns it instead of
re-fitting, and ``merge_catalogs.load_satstar_catalog`` reads every one of them
to build the consolidated per-filter satstar catalog whose sky positions reach
the merged photometry.

A sky position is only meaningful against the WCS that produced it, and the
frame's WCS changes underneath the cache for two reasons:

* **the frame moves** -- ``fix_alignment`` bakes a new ``RAOFFSET``/``DEOFFSET``
  whenever the offsets table is corrected and the working copy is regenerated
  from ``_cal``; and
* **the frame's WCS is read differently** -- products fit before the GWCS-first
  change (2026-07-29, issue #189) had ``skycoord_fit`` computed from a plain
  ``RA---TAN`` primary header with the distortion terms dropped, which is a
  smooth position-dependent surface tens of mas across a detector.

Neither invalidates the cache: ``_satstar_recovery_signature`` keys it on the
recovery/deblend config only, deliberately, so a plain re-run does not refit
every field.  The FIT is still valid -- the star is at the same PIXEL.  Only the
projection of that pixel onto the sky has gone stale.

The fix
-------
Do not trust a stored sky column.  On every read, recompute it from the stored
pixel centroid through :func:`jwst_gc_pipeline.frame_wcs.frame_wcs` of the frame
the catalog sits beside (the GWCS -- ASTROMETRY RULE #2).  ``refresh`` is a
no-op to <0.01 mas on a cache whose frame has not moved since the fit, so it
costs nothing but the WCS read on an up-to-date field.

``sat_com_ra``/``sat_com_dec`` are stored as sky only (the component bbox centre
pixel is not kept), so there is no pixel to re-project.  They are TRANSPORTED
instead: the anchor is displaced by the same tangent-plane offset the row's own
``skycoord_fit`` just moved by.  The anchor is the bbox centre of the same
saturated component the star was fit in -- tens of pixels away at most -- and the
old-minus-new WCS difference is a smooth surface, so its gradient over that
separation is the only error.  Measured against the exact pixel transport on
brick F182M nrcb1 (411 anchors, frame displaced 2" with a 0.05 deg roll): median
0.001 mas, max 0.003 mas.

This deliberately does NOT rebuild the fit-time WCS from the header cards in the
catalog's meta.  Doing that means inverting through an `astropy.wcs.WCS` built
from a detector-frame SIP header, which ASTROMETRY RULE #2 forbids; an earlier
draft of this module did it with a linear-card whitelist that dropped
``A_ORDER``/``A_i_j``/``B_ORDER``/``B_i_j``, so a ``RA---TAN-SIP`` projection was
inverted through a distortion-free TAN and moved the anchor 54.87 mas median /
224.30 mas max (1.79 / 7.26 px) on that same UNMOVED frame.  The transport above
needs no fit-time WCS at all and is exactly zero when the frame has not moved.

Measured on brick F200W (issue #193): 40 exposures, 3279 saturated stars, stored
minus current-GWCS = +56.8 / +88.7 mas, against a +58.7 / +88.2 mas
saturated-versus-unsaturated position excess measured in the m6 catalog built
from those caches.
"""
import hashlib
import os
import re
import warnings

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

from jwst_gc_pipeline.frame_wcs import frame_wcs

__all__ = ['frame_path_for_satstar_catalog', 'refresh_satstar_skycoords',
           'satstar_frame_state_signature', 'MissingSatstarFrameWarning']

#: Sky columns rebuilt from ``xcentroid``/``ycentroid``.
_PIXEL_DERIVED_SKY_COL = 'skycoord_fit'

#: Sky-only columns transported by the same offset as ``skycoord_fit``.
_ANCHOR_RA_COL, _ANCHOR_DEC_COL = 'sat_com_ra', 'sat_com_dec'

#: How many trailing ``_token`` suffixes may be stripped looking for the frame.
_MAX_SUFFIX_TOKENS = 4

_SAT_TAIL = re.compile(r'_satstar_catalog\.fits$')


class MissingSatstarFrameWarning(UserWarning):
    """Some or all of a satstar catalog's sky columns could not be refreshed --
    no frame beside it, or sources off the detector where the GWCS is undefined
    -- so those rows are served as stored (possibly against an older WCS)."""


def frame_path_for_satstar_catalog(catalog_path):
    """The exposure a ``*_<suffix>_satstar_catalog.fits`` was fit on.

    The writer builds the name as ``frame.replace('.fits', suffix +
    '_satstar_catalog.fits')`` where ``suffix`` is the run's file suffix
    (``_m3``, ``_iter2``, ``_resbgsub_m12``, ...), so the frame is recovered by
    dropping the tail and then trailing ``_token`` suffixes until a file exists.
    Returns ``None`` when no frame is on disk.
    """
    if not _SAT_TAIL.search(str(catalog_path)):
        return None
    stem = _SAT_TAIL.sub('', str(catalog_path))
    for _ in range(_MAX_SUFFIX_TOKENS + 1):
        candidate = stem + '.fits'
        if os.path.exists(candidate):
            return candidate
        head, sep, _tok = stem.rpartition('_')
        if not sep:
            return None
        stem = head
    return None


def satstar_frame_state_signature(catalog_paths):
    """A digest of the FRAME STATE behind a set of per-exposure satstar catalogs.

    The consolidated per-filter satstar catalog is itself a cache, and its
    freshness test compares its mtime and source count against the per-exposure
    catalogs.  Those do not move when the FRAME moves: ``fix_alignment`` baking a
    corrected ``RAOFFSET`` into a regenerated working copy rewrites the exposure
    and leaves every satstar catalog untouched, so a consolidated catalog built
    before the correction keeps serving the old frame's sky.  That is the same
    staleness this module refreshes per exposure, one level up.

    So key the consolidated cache on the exposures too.  The digest covers each
    resolved frame's name, mtime and size -- stat only, no file is opened, and a
    regeneration or a re-alignment changes all three.  A catalog with no frame on
    disk contributes a ``missing`` marker rather than being skipped, so a frame
    appearing or disappearing also invalidates.

    Returns a 16-character hex string; ``''`` for an empty input.
    """
    paths = list(catalog_paths or ())
    if not paths:
        return ''
    parts = []
    for cat in sorted(str(p) for p in paths):
        frame = frame_path_for_satstar_catalog(cat)
        if frame is None:
            parts.append(f'{os.path.basename(cat)}\0missing')
            continue
        try:
            st = os.stat(frame)
        except OSError:
            parts.append(f'{os.path.basename(frame)}\0unstatable')
            continue
        parts.append(f'{os.path.basename(frame)}\0{st.st_mtime_ns}\0{st.st_size}')
    digest = hashlib.sha256('\n'.join(parts).encode('utf-8')).hexdigest()
    return digest[:16]



def _in_frame(coord, reference):
    """``coord`` expressed in ``reference``'s frame (a no-op when they match)."""
    if coord.frame.name == reference.frame.name:
        return coord
    return coord.transform_to(reference.frame)


def refresh_satstar_skycoords(table, frame_path=None, catalog_path=None,
                              wcs=None):
    """Recompute ``table``'s sky columns from its stored pixel centroids.

    Parameters
    ----------
    table : `~astropy.table.Table`
        A per-exposure satstar catalog, modified in place and returned.
    frame_path : str, optional
        The exposure to read the GWCS from.  Derived from ``catalog_path`` when
        not given.
    catalog_path : str, optional
        Where ``table`` was read from; used to locate the frame and in messages.
    wcs : optional
        An already-read frame WCS, to avoid re-reading it per stage catalog.

    Returns
    -------
    (table, shift) : tuple
        ``shift`` is the median ``(dRA*cos(dec), dDec)`` in mas that the refresh
        MOVED the stored positions by, or ``(nan, nan)`` when nothing was
        refreshed.  A field whose frames have not moved since the fit reads
        ~0.00; a stale cache reads the frame's drift.
    """
    nan_shift = (float('nan'), float('nan'))
    if table is None or len(table) == 0:
        return table, nan_shift
    if _PIXEL_DERIVED_SKY_COL not in table.colnames:
        return table, nan_shift
    if 'xcentroid' not in table.colnames or 'ycentroid' not in table.colnames:
        # Nothing to re-project from: serving the stored sky is the only option.
        return table, nan_shift

    if wcs is None:
        if frame_path is None and catalog_path is not None:
            frame_path = frame_path_for_satstar_catalog(catalog_path)
        if frame_path is None or not os.path.exists(str(frame_path)):
            warnings.warn(
                f"No exposure found beside satstar catalog {catalog_path!r}; "
                f"its sky positions are served as stored and may predate the "
                f"frame's current WCS (issue #193).",
                MissingSatstarFrameWarning)
            return table, nan_shift
        wcs = frame_wcs(frame_path)

    x = np.asarray(table['xcentroid'], dtype=float)
    y = np.asarray(table['ycentroid'], dtype=float)
    stored = SkyCoord(table[_PIXEL_DERIVED_SKY_COL])
    fresh = wcs.pixel_to_world(x, y)
    if not isinstance(fresh, SkyCoord):
        return table, nan_shift

    # A GWCS returns NaN for a pixel off the detector -- the right sentinel, but
    # an outside-FOV satstar seed IS fit at such a pixel (its star is off the
    # frame; only its diffraction spikes are on it).  Re-projecting those to NaN
    # would delete positions the off-FOV flux reconciliation needs, so keep what
    # the cache stored there and say how many.
    off_detector = ~np.isfinite(fresh.ra.deg) & np.isfinite(stored.ra.deg)
    if np.any(off_detector):
        ra = fresh.ra.deg.copy()
        dec = fresh.dec.deg.copy()
        ra[off_detector] = stored.ra.deg[off_detector]
        dec[off_detector] = stored.dec.deg[off_detector]
        fresh = SkyCoord(ra * u.deg, dec * u.deg, frame=stored.frame.name)
        warnings.warn(
            f"{int(off_detector.sum())} satstar source(s) in {catalog_path!r} "
            f"sit off the detector, where the GWCS is undefined; their stored "
            f"sky positions were kept and are NOT re-projected.",
            MissingSatstarFrameWarning)
    table[_PIXEL_DERIVED_SKY_COL] = fresh

    dra = ((fresh.ra - stored.ra).wrap_at(180 * u.deg).to(u.mas).value
           * np.cos(stored.dec.radian))
    ddec = (fresh.dec - stored.dec).to(u.mas).value
    finite = np.isfinite(dra) & np.isfinite(ddec)
    shift = ((float(np.median(dra[finite])), float(np.median(ddec[finite])))
             if np.any(finite) else nan_shift)

    # The component anchor is stored as sky only, so there is no pixel to
    # re-project.  TRANSPORT it by the tangent-plane offset the row's own
    # skycoord_fit just moved by -- see the module docstring.  Rebuilding the
    # fit-time WCS from the meta's header cards would be ASTROMETRY RULE #2's
    # forbidden SIP-header inversion, and it is not needed.
    if _ANCHOR_RA_COL in table.colnames and _ANCHOR_DEC_COL in table.colnames:
        ra = np.asarray(table[_ANCHOR_RA_COL], dtype=float)
        dec = np.asarray(table[_ANCHOR_DEC_COL], dtype=float)
        dlon, dlat = stored.spherical_offsets_to(_in_frame(fresh, stored))
        ok = (np.isfinite(ra) & np.isfinite(dec)
              & np.isfinite(dlon.deg) & np.isfinite(dlat.deg))
        if np.any(ok):
            anchor = SkyCoord(ra[ok] * u.deg, dec[ok] * u.deg,
                              frame=stored.frame.name)
            moved = anchor.spherical_offsets_by(dlon[ok], dlat[ok])
            ra[ok] = moved.ra.deg
            dec[ok] = moved.dec.deg
            table[_ANCHOR_RA_COL] = ra
            table[_ANCHOR_DEC_COL] = dec
    return table, shift
