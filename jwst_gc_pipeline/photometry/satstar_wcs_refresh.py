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
pixel is not kept), so they are round-tripped through the WCS the cache stamped
into its own meta -- which reproduces ``skycoord_fit`` from the stored pixels to
0.00 mas, so it is the exact inverse of what wrote them.

Measured on brick F200W (issue #193): 40 exposures, 3279 saturated stars, stored
minus current-GWCS = +56.8 / +88.7 mas, against a +58.7 / +88.2 mas
saturated-versus-unsaturated position excess measured in the m6 catalog built
from those caches.
"""
import os
import re
import warnings

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import units as u
from astropy import wcs as astropy_wcs

from jwst_gc_pipeline.frame_wcs import frame_wcs

__all__ = ['frame_path_for_satstar_catalog', 'refresh_satstar_skycoords',
           'MissingSatstarFrameWarning']

#: Sky columns rebuilt from ``xcentroid``/``ycentroid``.
_PIXEL_DERIVED_SKY_COL = 'skycoord_fit'

#: Sky-only columns round-tripped through the cache's stamped WCS.
_ANCHOR_RA_COL, _ANCHOR_DEC_COL = 'sat_com_ra', 'sat_com_dec'

#: Header cards the fit stamps into the satstar catalog's meta.  They describe
#: the frame WCS AT FIT TIME, which is what makes the round trip exact.
_STAMPED_WCS_KEYS = ('WCSAXES', 'CRPIX1', 'CRPIX2', 'PC1_1', 'PC1_2',
                     'PC2_1', 'PC2_2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
                     'CDELT1', 'CDELT2', 'CUNIT1', 'CUNIT2',
                     'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2',
                     'LONPOLE', 'LATPOLE', 'RADESYS')

#: How many trailing ``_token`` suffixes may be stripped looking for the frame.
_MAX_SUFFIX_TOKENS = 4

_SAT_TAIL = re.compile(r'_satstar_catalog\.fits$')


class MissingSatstarFrameWarning(UserWarning):
    """A satstar catalog had no frame beside it, so its sky columns could not
    be refreshed and are served as stored (possibly against an older WCS)."""


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


def _stamped_wcs(meta):
    """The frame WCS the satstar fit stamped into the catalog meta, or None."""
    header = fits.Header()
    for key in _STAMPED_WCS_KEYS:
        if key in meta:
            header[key] = meta[key]
    if 'CTYPE1' not in header or 'CRVAL1' not in header:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', astropy_wcs.FITSFixedWarning)
        return astropy_wcs.WCS(header, relax=True)


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
    table[_PIXEL_DERIVED_SKY_COL] = fresh

    dra = ((fresh.ra - stored.ra).wrap_at(180 * u.deg).to(u.mas).value
           * np.cos(stored.dec.radian))
    ddec = (fresh.dec - stored.dec).to(u.mas).value
    finite = np.isfinite(dra) & np.isfinite(ddec)
    shift = ((float(np.median(dra[finite])), float(np.median(ddec[finite])))
             if np.any(finite) else nan_shift)

    # The component anchor is stored as sky only; invert it through the WCS the
    # fit stamped into meta, which is exactly what projected it.
    if _ANCHOR_RA_COL in table.colnames and _ANCHOR_DEC_COL in table.colnames:
        old = _stamped_wcs(table.meta)
        if old is not None:
            ra = np.asarray(table[_ANCHOR_RA_COL], dtype=float)
            dec = np.asarray(table[_ANCHOR_DEC_COL], dtype=float)
            ok = np.isfinite(ra) & np.isfinite(dec)
            if np.any(ok):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    ax, ay = old.all_world2pix(ra[ok], dec[ok], 0)
                    anchor = wcs.pixel_to_world(np.asarray(ax, dtype=float),
                                                np.asarray(ay, dtype=float))
                if isinstance(anchor, SkyCoord):
                    ra[ok] = anchor.ra.deg
                    dec[ok] = anchor.dec.deg
                    table[_ANCHOR_RA_COL] = ra
                    table[_ANCHOR_DEC_COL] = dec
    return table, shift
