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

A second, optional setting widens the disk of the brightest rows.  Around a satstar
with ``flux_fit`` of 1e6 and more the residual keeps ring craters 12-20 pixels out
(F212N, 0.031 arcsec per mosaic pixel), beyond the saturated-area disk, so rows with
``flux_fit`` of at least ``MERGEDCAT_FOOTPRINT_MASK_RING_FLUX_MIN`` get a disk of at
least ``MERGEDCAT_FOOTPRINT_MASK_RING_RADIUS_PX`` mosaic pixels.

A third, optional setting adds the footprints of gate-REJECTED satstar fits.  A
bright star whose fit only the sidelobe cut rejects has no row in the consolidated
catalog and no model, so its residual is the whole star.  When the satstar stage ran
with ``SATSTAR_SIDELOBE_MASK_FRAC_MIN`` its rejected table marks the fits that a
mask can cover (``sidelobe_maskable``), and ``MERGEDCAT_FOOTPRINT_MASK_REJECTED``
blanks their disks, of the same radius and with the same ring-zone extension, in the
same residual i2d.  The rows are read through ``load_rejected_satstar_catalog`` (the
merge's observation scope, phase and frame variant) with the sky position
re-projected through the frame's current GWCS.

A fourth, optional setting blanks the saturated pixels of the frames themselves.  The
disks above hang on satstar rows, so a saturated star with no row (a fit that the gates
rejected and that no mask can cover), or a second star inside the blob that one row
fitted (it lies outside the disk of the host row), has no disk.
``MERGEDCAT_FOOTPRINT_MASK_DQSAT_PX`` is a dilation radius in mosaic pixels (off unless
above 0): every residual i2d pixel within that distance of a mosaic pixel on which a
DQ_SATURATED pixel of one of the resampled per-frame residuals falls becomes NaN.  The
per-frame residuals keep the DQ of the crf they were rendered from, and the union over
the frames is blanked, so a star that saturates in only some of the dithers is blanked
wherever one of them saturates.  Only connected components (8-connectivity) of at least
``MERGEDCAT_FOOTPRINT_MASK_DQSAT_MIN_PX`` (default 4) saturated detector pixels count.

A fifth, optional setting blanks a guard band along the edge of the data footprint.  A star
whose core lies beyond the edge of the mosaic, or that the edge cuts, has a wing inside the
footprint that no model row accounts for (the satstar fits need the core, and a star with no
row is not modelled), so its residual keeps a bright halo.  ``MERGEDCAT_FOOTPRINT_MASK_EDGE_PX``
is the width of the band in mosaic pixels (off unless above 0): every residual i2d pixel
within that distance of a pixel outside the data footprint becomes NaN.  The outside is the
set of NaN pixels of the residual i2d, read before any of the other masks is applied, that
are 8-connected to the border of the image; the NaN blobs of saturated cores inside the
footprint do not count, and the image border itself is the edge of the data.

A sixth, optional setting adds a second ring-zone tier for the fainter rows.  A weakly
saturated row (a few tens of saturated detector pixels) has a footprint disk of about 5
mosaic pixels, and the PSF's second bright ring lies 0.16-0.22 arcsec (5-7 F212N mosaic
pixels) from the star, just outside it, so a model that is too faint or too bright
there leaves a whole-star residual peak that the disk misses.  A second pair,
``MERGEDCAT_FOOTPRINT_MASK_RING2_FLUX_MIN`` and ``MERGEDCAT_FOOTPRINT_MASK_RING2_RADIUS_PX``,
works like the first: rows with ``flux_fit`` of at least the first get a disk of at least
the second.  A row takes the largest radius among the disk of its saturated area and the
tiers whose flux it reaches, so the second tier is normally set at a lower flux and a
smaller radius than the first.

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
from scipy import ndimage

from jwst_gc_pipeline.frame_wcs import frame_wcs
from jwst_gc_pipeline.photometry.frame_pixel_area import (
    SR_PER_ARCSEC2, reference_pixel_area_sr)

#: radius factor on the saturated-area radius; off unless above 0
ENV_KFAC = 'MERGEDCAT_FOOTPRINT_MASK_KFAC'

#: margin in mosaic pixels added to the radius
ENV_MARGIN = 'MERGEDCAT_FOOTPRINT_MASK_MARGIN_PX'

DEFAULT_MARGIN_PX = 2.0

#: ring-zone extension: rows with ``flux_fit`` of at least this get a disk of at
#: least ``ENV_RING_RADIUS_PX``; off unless above 0, and read only when ``KFAC`` is on
ENV_RING_FLUX_MIN = 'MERGEDCAT_FOOTPRINT_MASK_RING_FLUX_MIN'

#: smallest radius in mosaic pixels of a row that ``ENV_RING_FLUX_MIN`` selects
ENV_RING_RADIUS_PX = 'MERGEDCAT_FOOTPRINT_MASK_RING_RADIUS_PX'

#: second ring-zone tier, set apart from the first one: rows with ``flux_fit`` of at least
#: this get a disk of at least ``ENV_RING2_RADIUS_PX``; off unless above 0, and read only
#: when ``KFAC`` is on
ENV_RING2_FLUX_MIN = 'MERGEDCAT_FOOTPRINT_MASK_RING2_FLUX_MIN'

#: smallest radius in mosaic pixels of a row that ``ENV_RING2_FLUX_MIN`` selects
ENV_RING2_RADIUS_PX = 'MERGEDCAT_FOOTPRINT_MASK_RING2_RADIUS_PX'

#: also blank the footprints of the gate-rejected satstar fits that the satstar
#: stage flagged ``sidelobe_maskable``; an on/off switch, off by default, and read
#: only when ``KFAC`` is on
ENV_REJECTED = 'MERGEDCAT_FOOTPRINT_MASK_REJECTED'

#: dilation radius in mosaic pixels of the DQ_SATURATED footprint mask; off unless
#: above 0, and read only when ``KFAC`` is on
ENV_DQSAT_PX = 'MERGEDCAT_FOOTPRINT_MASK_DQSAT_PX'

#: smallest connected component, in detector pixels, of DQ_SATURATED pixels that
#: counts; read only when ``ENV_DQSAT_PX`` is on
ENV_DQSAT_MIN_PX = 'MERGEDCAT_FOOTPRINT_MASK_DQSAT_MIN_PX'

DEFAULT_DQSAT_MIN_PX = 4

#: width in mosaic pixels of the guard band along the edge of the data footprint; off
#: unless above 0, and read only when ``KFAC`` is on
ENV_EDGE_PX = 'MERGEDCAT_FOOTPRINT_MASK_EDGE_PX'

#: the SATURATED bit of a JWST pixel DQ array
DQ_SATURATED = 2

_SWITCH_ON = ('1', 'true', 'yes', 'on')
_SWITCH_OFF = ('0', 'false', 'no', 'off')

#: column of ``*_satstar_rejected.fits`` that ``SATSTAR_SIDELOBE_MASK_FRAC_MIN`` writes
MASKABLE_COLUMN = 'sidelobe_maskable'

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


def ring_params():
    """``(flux_min, radius_px)`` of the ring-zone extension, ``(None, 0.0)`` when off.

    On when ``MERGEDCAT_FOOTPRINT_MASK_KFAC`` is above 0 and
    ``MERGEDCAT_FOOTPRINT_MASK_RING_FLUX_MIN`` is a number above 0; neither ring
    variable is read otherwise, so a stray one cannot break a default run.  The
    flux is in the units of the ``flux_fit`` column of the consolidated satstar
    catalog.  The radius is in mosaic pixels.

    Raises ``ValueError`` naming the variable when either is not a finite number
    >= 0, and when the flux is set without a radius above 0 (a ring setting that
    would otherwise change nothing).
    """
    return _ring_tier(ENV_RING_FLUX_MIN, ENV_RING_RADIUS_PX)


def ring2_params():
    """``(flux_min, radius_px)`` of the second ring-zone tier, ``(None, 0.0)`` when off.

    The same rules as ``ring_params`` for ``MERGEDCAT_FOOTPRINT_MASK_RING2_FLUX_MIN``
    and ``MERGEDCAT_FOOTPRINT_MASK_RING2_RADIUS_PX``.  The tiers are independent: the
    second one is on without the first.
    """
    return _ring_tier(ENV_RING2_FLUX_MIN, ENV_RING2_RADIUS_PX)


def _ring_tier(flux_name, radius_name):
    """``(flux_min, radius_px)`` read from the environment variables ``flux_name`` and
    ``radius_name``; ``(None, 0.0)`` while the mask or the tier is off."""
    if mask_params()[0] <= 0.0:
        return None, 0.0
    flux_min = _env_number(flux_name, 0.0)
    if flux_min == 0.0:
        return None, 0.0
    radius = _env_number(radius_name, 0.0)
    if radius <= 0.0:
        raise ValueError(
            f"{flux_name}={flux_min:g} needs {radius_name} set to "
            f"a number above 0 (the smallest radius, in mosaic pixels, of a row "
            f"at or above that flux)")
    return flux_min, radius


def rejected_enabled():
    """Whether ``MERGEDCAT_FOOTPRINT_MASK_REJECTED`` adds the maskable rejected fits.

    Off unless ``MERGEDCAT_FOOTPRINT_MASK_KFAC`` is above 0 and the variable is
    ``1/true/yes/on``; it is not read otherwise, so a stray one cannot break a
    default run.  ``0/false/no/off`` (any case) and unset or blank are off.  Any
    other value raises ``ValueError`` naming the variable.
    """
    if mask_params()[0] <= 0.0:
        return False
    raw = os.environ.get(ENV_REJECTED, '')
    value = raw.strip().lower()
    if not value or value in _SWITCH_OFF:
        return False
    if value in _SWITCH_ON:
        return True
    raise ValueError(
        f"{ENV_REJECTED}={raw!r} is not a recognised on/off value; use one of "
        f"{'/'.join(_SWITCH_ON)} or {'/'.join(_SWITCH_OFF)} (case-insensitive), "
        f"or leave it unset (off)")


def dqsat_params():
    """``(dilate_px, min_component_px)`` of the DQ_SATURATED footprint mask.

    On when ``MERGEDCAT_FOOTPRINT_MASK_KFAC`` is above 0 and
    ``MERGEDCAT_FOOTPRINT_MASK_DQSAT_PX`` is a number above 0; ``(0.0, 4)`` when off.
    Neither variable is read otherwise (``DQSAT_MIN_PX`` only with ``DQSAT_PX`` on), so a
    stray one cannot break a default run.  The dilation is in mosaic pixels.  The
    minimum component size is a whole number of detector pixels >= 1 and defaults to
    ``DEFAULT_DQSAT_MIN_PX``.

    Raises ``ValueError`` naming the variable when ``DQSAT_PX`` is not a finite number
    >= 0 or ``DQSAT_MIN_PX`` is not a whole number >= 1.
    """
    if mask_params()[0] <= 0.0:
        return 0.0, DEFAULT_DQSAT_MIN_PX
    dilate = _env_number(ENV_DQSAT_PX, 0.0)
    if dilate == 0.0:
        return 0.0, DEFAULT_DQSAT_MIN_PX
    min_px = _env_number(ENV_DQSAT_MIN_PX, float(DEFAULT_DQSAT_MIN_PX))
    if min_px < 1.0 or min_px != math.floor(min_px):
        raise ValueError(
            f"{ENV_DQSAT_MIN_PX}={os.environ.get(ENV_DQSAT_MIN_PX)!r}: expected a "
            f"whole number of detector pixels >= 1")
    return dilate, int(min_px)


def edge_params():
    """Width in mosaic pixels of the footprint-edge guard band, ``0.0`` when off.

    On when ``MERGEDCAT_FOOTPRINT_MASK_KFAC`` is above 0 and
    ``MERGEDCAT_FOOTPRINT_MASK_EDGE_PX`` is a number above 0.  The variable is not read
    otherwise, so a stray one cannot break a default run.  Raises ``ValueError`` naming
    the variable when it is not a finite number >= 0.
    """
    if mask_params()[0] <= 0.0:
        return 0.0
    return _env_number(ENV_EDGE_PX, 0.0)


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


def _refuse_primary_catalog(filtername, target, basepath, proposal_id, field):
    """Raise ``FootprintMaskUnavailable`` when the merge reads the NIRCam
    ``merged_i2d`` primary satstar catalog for ``target`` (not checked without a
    target)."""
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


def read_satstars(basepath, filtername, proposal_id, field, target=None,
                  need_flux=False):
    """The consolidated satstar catalog the merge of this observation read.

    Raises ``FootprintMaskUnavailable`` when the merge read the NIRCam
    ``merged_i2d`` primary catalog instead (the consolidated cache is then not
    the catalog it used), when the cache does not exist, or when it lacks
    ``skycoord_fit`` or ``sat_area`` (and ``flux_fit`` with ``need_flux``, the
    ring-zone extension).  Nothing is written.
    """
    from jwst_gc_pipeline.photometry import merge_catalogs
    _refuse_primary_catalog(filtername, target, basepath, proposal_id, field)
    path = merge_catalogs.consolidated_satstar_cache_path(
        basepath, filtername,
        merge_catalogs.satstar_obs_scope(proposal_id, field))
    if not os.path.exists(path):
        raise FootprintMaskUnavailable(f"no consolidated satstar catalog {path}")
    tbl = Table.read(path)
    required = ('skycoord_fit', 'sat_area') + (('flux_fit',) if need_flux else ())
    missing = [c for c in required if c not in tbl.colnames]
    if missing:
        raise FootprintMaskUnavailable(
            f"{path} has no {' or '.join(missing)} column")
    return tbl


def read_maskable_rejected(basepath, filtername, proposal_id, field, target=None,
                           phase=None, frames=None, need_flux=False):
    """The gate-rejected satstar fits of this observation flagged maskable.

    Rows of the ``*_satstar_rejected.fits`` that ``load_rejected_satstar_catalog``
    reads for the merge (``proposal_id`` / ``field`` observation scope, one
    ``phase`` per exposure, the frame variants of ``frames``) with
    ``sidelobe_maskable`` true, their ``skycoord_fit`` re-projected through the
    frame's current GWCS.  A file without the column contributes no row.  Nothing
    is written.

    Raises ``FootprintMaskUnavailable`` when the merge reads the primary satstar
    catalog, when there are no rejected files, or when the rejected rows lack
    ``skycoord_fit``, ``sat_area`` or ``sidelobe_maskable`` (and ``flux_fit``
    with ``need_flux``, the ring-zone extension).  The last one means the satstar
    stage ran without ``SATSTAR_SIDELOBE_MASK_FRAC_MIN``, so no row was judged.
    """
    from jwst_gc_pipeline.photometry import merge_catalogs
    from jwst_gc_pipeline.photometry.satstar_phase_selection import (
        normalise_frame_variants, satstar_frame_variant)
    _refuse_primary_catalog(filtername, target, basepath, proposal_id, field)
    variants = normalise_frame_variants(
        {satstar_frame_variant(f) for f in (frames or ())} - {''})
    kwargs = {'target': target} if target else {}
    tbl = merge_catalogs.load_rejected_satstar_catalog(
        filtername, basepath=basepath, proposal_id=proposal_id, field=field,
        phase=phase, frame_variants=variants, refresh_sky=True, **kwargs)
    if tbl is None:
        raise FootprintMaskUnavailable(
            f"no gate-rejected satstar files for {filtername}")
    required = (('skycoord_fit', 'sat_area', MASKABLE_COLUMN)
                + (('flux_fit',) if need_flux else ()))
    missing = [c for c in required if c not in tbl.colnames]
    if missing:
        raise FootprintMaskUnavailable(
            f"the rejected satstar table of {filtername} has no "
            f"{' or '.join(missing)} column")
    flag = np.ma.filled(np.ma.asarray(tbl[MASKABLE_COLUMN]), False).astype(bool)
    return tbl[flag]


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


def _footprint_disks(satstars, wcs, native, scale, kfac, margin_px, tiers):
    """``(x, y, radius, tier_rows)`` of the footprint disks of ``satstars``.

    Pixel coordinates and radii in mosaic pixels of the rows that have a positive
    finite ``sat_area``.  ``tiers`` is a sequence of ``(flux_min, radius_px)`` pairs
    (``flux_min`` None for a tier that is off): a row whose ``flux_fit`` reaches
    ``flux_min`` has a radius of at least ``radius_px``.  ``tier_rows`` has one count
    per tier, of the rows that reach it (0 for a tier that is off).
    """
    area = np.asarray(satstars['sat_area'], dtype=float)
    rows = np.isfinite(area) & (area > 0)
    if rows.any():
        x, y = wcs.world_to_pixel(satstars['skycoord_fit'][rows])
    else:
        x = y = np.empty(0)
    radius = kfac * np.sqrt(area[rows] / np.pi) * native / scale + margin_px
    tier_rows = []
    for flux_min, radius_px in tiers:
        if flux_min is None:
            tier_rows.append(0)
            continue
        with np.errstate(invalid='ignore'):
            reached = np.asarray(satstars['flux_fit'], dtype=float)[rows] >= flux_min
        radius = np.where(reached, np.maximum(radius, radius_px), radius)
        tier_rows.append(int(reached.sum()))
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float), radius, tier_rows


def mask_residual_i2d(i2d_path, satstars, filtername, instrument, kfac, margin_px,
                      ring_flux_min=None, ring_radius_px=0.0, rejected=None,
                      ring2_flux_min=None, ring2_radius_px=0.0):
    """Blank the footprints of ``satstars`` in the SCI plane of ``i2d_path``.

    The file is updated in place.  A row without a positive finite ``sat_area``
    has no footprint and is not masked.  With ``ring_flux_min`` set, a row whose
    ``flux_fit`` is at least that has a disk of at least ``ring_radius_px`` mosaic
    pixels (a row with no ``flux_fit`` or a NaN one is not widened); ``ring2_flux_min``
    and ``ring2_radius_px`` are a second tier of the same kind, and a row takes the
    largest radius among its footprint disk and the tiers it reaches.  ``rejected``
    (``read_maskable_rejected``) adds the disks of those rows, of the same radius
    and with the same ring-zone extension.  The SCI header records the setting
    (``FPMASKK`` radius factor, ``FPMASKM`` margin in mosaic pixels, ``FPMASKN``
    rows of ``satstars`` masked, ``FPREJN`` rows of ``rejected`` masked when it is
    given, and with the ring extension ``FPRINGF`` flux, ``FPRINGR`` radius in
    mosaic pixels, ``FPRINGN`` rows of both tables at or above that flux; the second
    tier writes ``FPRNG2F``, ``FPRNG2R`` and ``FPRNG2N`` the same way).  Returns
    a dict with ``rows`` (of ``satstars``), ``rejected_rows`` (0 without
    ``rejected``), ``ring_rows`` (rows at or above ``ring_flux_min``, 0 without
    the ring extension), ``ring2_rows`` (the same for the second tier),
    ``pixels`` (finite pixels that became NaN), ``finite``
    (finite pixels before), ``scale`` (arcsec per mosaic pixel) and
    ``radius_p50`` / ``radius_max`` (mosaic pixels, over every disk).

    Raises ``FootprintMaskUnavailable``, with the file untouched, when
    ``ring_flux_min`` or ``ring2_flux_min`` is set and ``satstars`` or ``rejected``
    has no ``flux_fit`` column.
    """
    tiers = ((ring_flux_min, ring_radius_px), (ring2_flux_min, ring2_radius_px))
    flux_readers = [name for name, flux in ((ENV_RING_FLUX_MIN, ring_flux_min),
                                            (ENV_RING2_FLUX_MIN, ring2_flux_min))
                    if flux is not None]
    if flux_readers:
        for name, tbl in (('satstar', satstars), ('rejected satstar', rejected)):
            if tbl is not None and 'flux_fit' not in tbl.colnames:
                raise FootprintMaskUnavailable(
                    f"the {name} catalog has no flux_fit column, which the "
                    f"ring-zone extension ({', '.join(flux_readers)}) reads")
    native = native_pixel_scale_arcsec(filtername, instrument)
    with fits.open(i2d_path, mode='update', memmap=False) as hdul:
        sci = hdul['SCI'].data
        wcs = WCS(hdul['SCI'].header)
        scale = float(np.sqrt(np.prod(proj_plane_pixel_scales(wcs)[:2]))) * 3600.0
        x, y, radius, tier_rows = _footprint_disks(
            satstars, wcs, native, scale, kfac, margin_px, tiers)
        n_rows = len(x)
        n_rejected = 0
        if rejected is not None:
            rx, ry, rradius, rtier_rows = _footprint_disks(
                rejected, wcs, native, scale, kfac, margin_px, tiers)
            n_rejected = len(rx)
            x, y = np.concatenate([x, rx]), np.concatenate([y, ry])
            radius = np.concatenate([radius, rradius])
            tier_rows = [a + b for a, b in zip(tier_rows, rtier_rows)]
        ring_rows, ring2_rows = tier_rows
        pixels = mask_disks(sci, x, y, radius)
        finite = int(np.count_nonzero(np.isfinite(sci))) + pixels
        header = hdul['SCI'].header
        header['FPMASKK'] = (float(kfac), 'satstar footprint mask: radius factor')
        header['FPMASKM'] = (float(margin_px),
                             'satstar footprint mask: margin [mosaic px]')
        header['FPMASKN'] = (n_rows, 'satstar footprint mask: rows')
        if rejected is not None:
            header['FPREJN'] = (n_rejected,
                                'satstar footprint mask: rejected rows')
        if ring_flux_min is not None:
            header['FPRINGF'] = (float(ring_flux_min),
                                 'satstar footprint mask: ring flux_fit min')
            header['FPRINGR'] = (float(ring_radius_px),
                                 'satstar footprint mask: ring radius [mosaic px]')
            header['FPRINGN'] = (ring_rows, 'satstar footprint mask: ring rows')
        if ring2_flux_min is not None:
            header['FPRNG2F'] = (float(ring2_flux_min),
                                 'satstar footprint mask: ring 2 flux_fit min')
            header['FPRNG2R'] = (float(ring2_radius_px),
                                 'satstar footprint mask: ring 2 rad [mosaic px]')
            header['FPRNG2N'] = (ring2_rows, 'satstar footprint mask: ring 2 rows')
        hdul['SCI'].data = sci
        hdul.flush()
    return {'rows': n_rows, 'rejected_rows': n_rejected, 'ring_rows': ring_rows,
            'ring2_rows': ring2_rows, 'pixels': pixels, 'finite': finite, 'scale': scale,
            'radius_p50': float(np.median(radius)) if len(radius) else float('nan'),
            'radius_max': float(np.max(radius)) if len(radius) else float('nan')}


def dqsat_footprint(frame_paths, wcs, shape, min_comp_px=DEFAULT_DQSAT_MIN_PX):
    """The mosaic pixels on which a DQ_SATURATED pixel of ``frame_paths`` falls.

    Each path is a per-frame product with a ``DQ`` extension and a WCS (read with
    ``frame_wcs``: the GWCS when the file has one).  Of its DQ_SATURATED pixels only
    the connected components (8-connectivity) of at least ``min_comp_px`` pixels
    count.  A counted pixel goes through the frame WCS and the mosaic ``wcs`` (an
    ``astropy.wcs.WCS`` of the i2d) to the mosaic grid of ``shape`` and is rounded to
    the nearest mosaic pixel; a pixel that lands outside the grid, or has no finite
    sky position, is dropped.

    Returns ``(flagged, components, pixels)``: the boolean image of the flagged
    mosaic pixels, and the number of components that counted and of detector pixels
    in them, summed over the frames.  Raises ``FootprintMaskUnavailable`` when a file
    has no ``DQ`` extension.
    """
    ny, nx = shape
    flagged = np.zeros(shape, dtype=bool)
    components = pixels = 0
    for path in frame_paths:
        with fits.open(path, memmap=False) as hdul:
            dq = hdul['DQ'].data if 'DQ' in hdul else None
        if dq is None:
            raise FootprintMaskUnavailable(f"{path} has no DQ extension")
        labels, n = ndimage.label((dq & DQ_SATURATED) > 0,
                                  structure=np.ones((3, 3), dtype=int))
        if n == 0:
            continue
        area = np.bincount(labels.ravel(), minlength=n + 1)
        counts = area >= min_comp_px
        counts[0] = False
        if not counts.any():
            continue
        yy, xx = np.nonzero(counts[labels])
        ra, dec = frame_wcs(path).pixel_to_world_values(xx, yy)
        mx, my = wcs.world_to_pixel_values(ra, dec)
        finite = np.isfinite(mx) & np.isfinite(my)
        ix = np.rint(mx[finite]).astype(int)
        iy = np.rint(my[finite]).astype(int)
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        flagged[iy[inside], ix[inside]] = True
        components += int(counts.sum())
        pixels += int(yy.size)
    return flagged, components, pixels


def dilate_disk(flagged, radius):
    """The pixels within ``radius`` (Euclidean, in pixels) of a True pixel of ``flagged``.

    The same set as ``distance_transform_edt(~flagged) <= radius``, stamped from the
    flagged pixels, so the cost follows their number and not the size of the image.
    """
    ny, nx = flagged.shape
    iy, ix = np.nonzero(flagged)
    mask = np.zeros_like(flagged)
    reach = int(math.floor(radius))
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            y, x = iy + dy, ix + dx
            inside = (y >= 0) & (y < ny) & (x >= 0) & (x < nx)
            mask[y[inside], x[inside]] = True
    return mask


def mask_residual_dqsat(i2d_path, frame_paths, dilate_px,
                        min_comp_px=DEFAULT_DQSAT_MIN_PX):
    """Blank the DQ_SATURATED footprint of ``frame_paths`` in the SCI plane of ``i2d_path``.

    The file is updated in place: every pixel within ``dilate_px`` mosaic pixels of a
    mosaic pixel that ``dqsat_footprint`` flags becomes NaN.  The SCI header records
    the setting (``FPDQSATD`` dilation in mosaic pixels, ``FPDQSATM`` smallest
    component in detector pixels, ``FPDQSATN`` frames).  Returns a dict with
    ``frames``, ``components`` and ``saturated_pixels`` (detector pixels in the
    counted components, summed over the frames), ``flagged`` (mosaic pixels they fall
    on), ``pixels`` (finite pixels that became NaN) and ``finite`` (finite pixels
    before).

    Raises ``FootprintMaskUnavailable``, with the file untouched, when a frame has no
    ``DQ`` extension.  The frames are read before the i2d is opened for update, so any
    error reading them leaves the file alone as well.
    """
    sci_header = fits.getheader(i2d_path, 'SCI')
    wcs = WCS(sci_header)
    flagged, components, saturated_pixels = dqsat_footprint(
        frame_paths, wcs, (int(sci_header['NAXIS2']), int(sci_header['NAXIS1'])),
        min_comp_px)
    mask = dilate_disk(flagged, dilate_px)
    with fits.open(i2d_path, mode='update', memmap=False) as hdul:
        sci = hdul['SCI'].data
        finite = int(np.count_nonzero(np.isfinite(sci)))
        sci[mask] = np.nan
        pixels = finite - int(np.count_nonzero(np.isfinite(sci)))
        header = hdul['SCI'].header
        header['FPDQSATD'] = (float(dilate_px),
                              'DQ_SAT footprint mask: dilation [mosaic px]')
        header['FPDQSATM'] = (int(min_comp_px),
                              'DQ_SAT footprint mask: min component [det px]')
        header['FPDQSATN'] = (len(frame_paths), 'DQ_SAT footprint mask: frames')
        hdul['SCI'].data = sci
        hdul.flush()
    return {'frames': len(frame_paths), 'components': components,
            'saturated_pixels': saturated_pixels, 'flagged': int(flagged.sum()),
            'pixels': pixels, 'finite': finite}


def footprint_outside(sci):
    """The pixels of ``sci`` outside the data footprint: the NaN ones 8-connected to the border.

    The mosaic footprint is a rotated rectangle in a bounding image, so the pixels outside
    it form one NaN region that touches the border of the image.  A NaN blob inside the
    footprint (a saturated core that no frame covered) does not touch the border and is
    not outside; one that touches the outside region is part of it.
    """
    labels, n = ndimage.label(~np.isfinite(sci), structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return np.zeros(sci.shape, dtype=bool)
    on_border = np.unique(np.concatenate(
        [labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    on_border = on_border[on_border > 0]
    return np.isin(labels, on_border)


def edge_band(outside, width):
    """The pixels within ``width`` (Euclidean, in pixels) of a True pixel of ``outside``.

    The border of the image is the edge of the data too: the band is measured from
    ``outside`` padded by one outside pixel all around.  It is stamped from the boundary of
    that region only (a pixel of ``outside`` with an 8-neighbour that is not) and joined
    with ``outside`` itself, which gives the set ``distance_transform_edt(~padded) <= width``
    cropped to the image: the nearest outside pixel of a pixel that is not outside always
    has a neighbour that is not.  The cost follows the length of the boundary, not the size
    of the image.
    """
    ny, nx = outside.shape
    padded = np.pad(outside, 1, constant_values=True)
    boundary = padded & ~ndimage.binary_erosion(
        padded, structure=np.ones((3, 3), dtype=bool), border_value=1)
    return outside | dilate_disk(boundary, width)[1:ny + 1, 1:nx + 1]


def mask_residual_edge(i2d_path, edge_px):
    """Blank a guard band of ``edge_px`` mosaic pixels along the data footprint edge of ``i2d_path``.

    The file is updated in place: every pixel within ``edge_px`` of ``footprint_outside``
    of the SCI plane as it is on entry becomes NaN, so call this before the other masks
    (a disk or a saturated blob that touches the outside region would otherwise count as
    part of it).  The SCI header records the setting (``FPEDGEPX`` width in mosaic pixels,
    ``FPEDGEN`` pixels counted as outside the footprint).  Returns a dict with ``outside``
    (those pixels), ``pixels`` (finite pixels that became NaN) and ``finite`` (finite
    pixels before).
    """
    with fits.open(i2d_path, mode='update', memmap=False) as hdul:
        sci = hdul['SCI'].data
        outside = footprint_outside(sci)
        mask = edge_band(outside, edge_px)
        finite = int(np.count_nonzero(np.isfinite(sci)))
        sci[mask] = np.nan
        pixels = finite - int(np.count_nonzero(np.isfinite(sci)))
        header = hdul['SCI'].header
        header['FPEDGEPX'] = (float(edge_px),
                              'footprint edge guard band: width [mosaic px]')
        header['FPEDGEN'] = (int(outside.sum()),
                             'footprint edge guard band: outside pixels')
        hdul['SCI'].data = sci
        hdul.flush()
    return {'outside': int(outside.sum()), 'pixels': pixels, 'finite': finite}
