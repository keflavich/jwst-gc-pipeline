"""Opt-in mask of the saturated footprints in the merged-catalog residual i2d.

The merged-catalog residual (``*_mergedcat_residual_i2d``) is the data minus a
PSF model of every star, and the model of a saturated star is fit to its wings.
Inside the saturated core and out to about 20 pixels the model and the detector's
PSF disagree: on gc-treasury 10678 o105 F212N the model is 5-9 % too bright inside
5 px and 5-23 % too faint at 5.5-8.5 px of the isolated satstars with ``flux_fit``
of 1e6 and more, and the residual keeps a pit at the core and ring craters around
it.  A radial profile correction of the model removes about two thirds of the deep
pits and adds ring dips (``SATURATED_PIXEL_HANDLING.md``).

This module blanks the region instead.  Every row of the consolidated satstar
catalog masks a disk around its fitted position, of radius

    KFAC * sqrt(sat_area / pi) * native_pixel_scale / mosaic_pixel_scale + MARGIN

mosaic pixels, where ``sat_area`` is the saturated area of the row in native
detector pixels.  The residual i2d pixels in the disks become NaN, the way the MIRI
pit cleanup in ``build_mergedcat_residuals`` blanks coadd pits.  Only the SCI plane
of the residual i2d changes; the model i2d, the data i2d and the catalogs do not.

The switch is off unless ``MERGEDCAT_FOOTPRINT_MASK_KFAC`` is a number above 0.
The next phase of a manual run reads the residual i2d for its i2d seeds and its
smoothed background map, and the replays in ``SATURATED_PIXEL_HANDLING.md``
measured the final residual only, so the mask is scoped to the m7 pass unless
``MERGEDCAT_FOOTPRINT_MASK_PASSES`` lists others (the scope of
``GC_RESID_NAN_SATSTAR_PASSES``).
"""
import math
import os

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from jwst_gc_pipeline.photometry.frame_pixel_area import (
    SR_PER_ARCSEC2, reference_pixel_area_sr)

#: radius factor on the saturated-area radius; off unless above 0
ENV_KFAC = 'MERGEDCAT_FOOTPRINT_MASK_KFAC'

#: margin in mosaic pixels added to the radius
ENV_MARGIN = 'MERGEDCAT_FOOTPRINT_MASK_MARGIN_PX'

DEFAULT_MARGIN_PX = 2.0

#: manual passes whose residual i2d gets the mask, comma- or space-separated;
#: unset or blank gives ``DEFAULT_PASSES``
ENV_PASSES = 'MERGEDCAT_FOOTPRINT_MASK_PASSES'

#: iteration labels ``build_mergedcat_residuals`` is called with (the ``m12``
#: driver phase passes ``m2``)
PASS_LABELS = ('m2', 'm3', 'm4', 'm5', 'm6', 'm7')

DEFAULT_PASSES = ('m7',)


class FootprintMaskUnavailable(ValueError):
    """The mask cannot be built for this residual i2d: no consolidated satstar
    catalog, or one without ``skycoord_fit`` and ``sat_area``."""


def _env_number(name, default):
    raw = os.environ.get(name, '')
    if not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        value = np.nan
    if not (np.isfinite(value) and value >= 0.0):
        raise ValueError(f"{name}={raw!r}: expected a finite number >= 0")
    return value


def mask_params():
    """``(MERGEDCAT_FOOTPRINT_MASK_KFAC, MERGEDCAT_FOOTPRINT_MASK_MARGIN_PX)``.

    ``KFAC`` unset, blank or 0 is off, the default; the margin is then not read.
    The margin defaults to 2 mosaic pixels.  A value that is not a finite number
    >= 0 raises ``ValueError`` naming the variable.
    """
    kfac = _env_number(ENV_KFAC, 0.0)
    if kfac == 0.0:
        return 0.0, DEFAULT_MARGIN_PX
    return kfac, _env_number(ENV_MARGIN, DEFAULT_MARGIN_PX)


def mask_passes():
    """Passes whose residual i2d gets the mask (``MERGEDCAT_FOOTPRINT_MASK_PASSES``).

    Labels are comma- or space-separated (``m7`` or ``m6,m7``); unset or blank
    gives ``('m7',)``.  The next pass reads the residual i2d of the m2-m6 passes
    for its seeds and its background, and only the m7 residual has been replayed
    with the mask.

    Raises ``ValueError`` naming the variable when a label is not one of
    ``PASS_LABELS``; a typo would otherwise switch the mask off for every pass
    without a word.
    """
    raw = os.environ.get(ENV_PASSES, '')
    passes = tuple(tok.lower() for tok in raw.replace(',', ' ').split())
    bad = [tok for tok in passes if tok not in PASS_LABELS]
    if bad:
        raise ValueError(
            f"{ENV_PASSES}={raw!r}: {bad} not pass labels (any of "
            f"{', '.join(PASS_LABELS)}, e.g. 'm7' or 'm6,m7'; unset applies the "
            f"mask to m7 only)")
    return passes or DEFAULT_PASSES


def applies_to(iteration_label):
    """Whether the residual i2d of ``iteration_label``'s pass gets the mask.

    ``MERGEDCAT_FOOTPRINT_MASK_KFAC`` above 0 and the label in
    ``mask_passes()``.  The pass list is read only when the switch is on, so a
    stray ``MERGEDCAT_FOOTPRINT_MASK_PASSES`` cannot break a default run.  A
    caller without an iteration label (``None``) is never masked.
    """
    if mask_params()[0] <= 0.0:
        return False
    return iteration_label in mask_passes()


def native_pixel_scale_arcsec(filtername, instrument=None):
    """Side in arcsec of the detector pixel that ``sat_area`` counts.

    The nominal scale of ``frame_pixel_area``: 0.031 arcsec for the NIRCam
    short-wave filters, 0.063 for the long-wave ones, 0.11 for MIRI.
    """
    return math.sqrt(reference_pixel_area_sr(filtername, instrument)
                     / SR_PER_ARCSEC2)


def read_satstars(basepath, filtername, proposal_id, field, target=None):
    """The consolidated satstar catalog the merge of this observation read.

    Raises ``FootprintMaskUnavailable`` when the merge read the NIRCam
    ``merged_i2d`` primary catalog instead (the consolidated cache is then not
    the catalog it used), when the cache does not exist, or when it lacks
    ``skycoord_fit`` or ``sat_area``.  Nothing is written.
    """
    from jwst_gc_pipeline.photometry import merge_catalogs
    if target and merge_catalogs._obs_filters_for(target) is not None:
        try:
            primary = merge_catalogs._primary_satstar_catalog_path(
                filtername, target, basepath, proposal_id, field)
        except KeyError:
            primary = None      # the registry does not list the filter for this target
        if primary is not None:
            raise FootprintMaskUnavailable(
                f"the merge reads the primary satstar catalog {primary}, "
                f"not a consolidated cache")
    path = merge_catalogs.consolidated_satstar_cache_path(
        basepath, filtername,
        merge_catalogs.satstar_obs_scope(proposal_id, field))
    if not os.path.exists(path):
        raise FootprintMaskUnavailable(f"no consolidated satstar catalog {path}")
    tbl = Table.read(path)
    missing = [c for c in ('skycoord_fit', 'sat_area') if c not in tbl.colnames]
    if missing:
        raise FootprintMaskUnavailable(
            f"{path} has no {' or '.join(missing)} column")
    return tbl


def mask_disks(sci, x, y, radius):
    """Set to NaN the pixels of ``sci`` within ``radius[k]`` of ``(x[k], y[k])``.

    ``x`` and ``y`` are zero-based pixel coordinates (``x`` along axis 1).  A row
    with a non-finite coordinate or radius masks nothing.  Returns the number of
    finite pixels that became NaN.
    """
    ny, nx = sci.shape
    n_before = int(np.count_nonzero(np.isfinite(sci)))
    for xi, yi, ri in zip(x, y, radius):
        if not (np.isfinite(xi) and np.isfinite(yi) and np.isfinite(ri)):
            continue
        x0, x1 = max(int(math.floor(xi - ri)), 0), min(int(math.ceil(xi + ri)) + 1, nx)
        y0, y1 = max(int(math.floor(yi - ri)), 0), min(int(math.ceil(yi + ri)) + 1, ny)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        sci[y0:y1, x0:x1][np.hypot(xx - xi, yy - yi) <= ri] = np.nan
    return n_before - int(np.count_nonzero(np.isfinite(sci)))


def mask_residual_i2d(i2d_path, satstars, filtername, instrument, kfac, margin_px):
    """Blank the footprints of ``satstars`` in the SCI plane of ``i2d_path``.

    The file is updated in place.  A row without a positive finite ``sat_area``
    has no footprint and is not masked.  The SCI header records the setting
    (``FPMASKK`` radius factor, ``FPMASKM`` margin in mosaic pixels, ``FPMASKN``
    rows masked).  Returns a dict with ``rows``, ``pixels`` (finite pixels that
    became NaN), ``finite`` (finite pixels before), ``scale`` (arcsec per mosaic
    pixel) and ``radius_p50`` / ``radius_max`` (mosaic pixels).
    """
    native = native_pixel_scale_arcsec(filtername, instrument)
    with fits.open(i2d_path, mode='update', memmap=False) as hdul:
        sci = hdul['SCI'].data
        wcs = WCS(hdul['SCI'].header)
        scale = float(np.sqrt(np.prod(proj_plane_pixel_scales(wcs)[:2]))) * 3600.0
        area = np.asarray(satstars['sat_area'], dtype=float)
        rows = np.isfinite(area) & (area > 0)
        if rows.any():
            x, y = wcs.world_to_pixel(satstars['skycoord_fit'][rows])
        else:
            x = y = np.empty(0)
        radius = (kfac * np.sqrt(area[rows] / np.pi) * native / scale
                  + margin_px)
        pixels = mask_disks(sci, x, y, radius)
        finite = int(np.count_nonzero(np.isfinite(sci))) + pixels
        header = hdul['SCI'].header
        header['FPMASKK'] = (float(kfac), 'satstar footprint mask: radius factor')
        header['FPMASKM'] = (float(margin_px),
                             'satstar footprint mask: margin [mosaic px]')
        header['FPMASKN'] = (int(rows.sum()), 'satstar footprint mask: rows')
        hdul['SCI'].data = sci
        hdul.flush()
    return {'rows': int(rows.sum()), 'pixels': pixels, 'finite': finite,
            'scale': scale,
            'radius_p50': float(np.median(radius)) if len(radius) else float('nan'),
            'radius_max': float(np.max(radius)) if len(radius) else float('nan')}
