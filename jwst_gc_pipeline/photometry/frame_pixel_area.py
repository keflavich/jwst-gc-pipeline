"""Per-frame pixel area for the single-frame merges (#1238).

A per-frame ``flux_fit`` is a sum of MJy/sr over that frame's native pixels, so
its flux in Jy is ``flux_fit * PIXAR_SR`` with the area of THAT detector.  The
eight NIRCam short-wave detectors span ``PIXAR_SR`` = 2.222-2.312e-14 sr (4 %);
the long-wave pair differs by 0.4 %.  ``combine_singleframe`` averages the
frames in image units and ``merge_daophot`` converts once, so every frame was
converted with one area and each SW detector carried an offset of up to 34 mmag
relative to its own area (measured on 10678 F212N: predicted and measured
per-detector offsets agree to 2 mmag rms).

The remedy implemented here rescales every frame's flux columns to one
reference area BEFORE they are averaged, and records that area in the table
meta as ``PIXAREF`` so the later conversion uses it.  The reference is a
constant per (instrument, filter) and does not depend on which frames a merge
holds, so the consolidated satstar cache, the per-observation treasury tiles
and the cross-observation m7 products all sit on the same scale.  Any constant
gives the right flux in Jy provided the conversion uses it.

Only pixel sums are rescaled.  ``local_bkg`` and the ``resbkg_*`` columns are
surface brightnesses (MJy/sr per pixel) and are left alone.
"""
import math
import os

import numpy as np
from astropy.io import fits

#: Meta key carrying the reference area, as a plain float in deg^2 (the unit of
#: ``pixelscale_deg2``).  One key only: the cross-band header loop writes
#: ``{filter}{key[:4]}``, so a second ``PIXA*`` key would overwrite this one.
PIXAREF_KEY = 'PIXAREF'

SR_PER_ARCSEC2 = (math.pi / (180.0 * 3600.0)) ** 2
DEG2_PER_SR = (180.0 / math.pi) ** 2

#: Columns that are sums of MJy/sr over native pixels, per catalogue flavour.
DAOPHOT_FLUX_COLUMNS = ('flux_fit', 'flux_err', 'flux_unc', 'flux_init')
CROWDSOURCE_FLUX_COLUMNS = ('flux', 'dflux', 'fluxiso')

#: Nominal pixel side in arcsec of the reference grid.
_NOMINAL_PIXSCALE_ARCSEC = {
    'NIRCAM_SW': 0.031,
    'NIRCAM_LW': 0.063,
    'MIRI': 0.11,
    'NIRISS': 0.0656,
}

#: Meta keys that hold a pixel side length in arcsec, in order of preference.
_PIXSCALE_KEYS = ('pixscale', 'PIXSCALE', 'pixscale_as', 'PIXSCALE_AS')


class FramePixelAreaError(ValueError):
    """The pixel area of a frame, or the reference area of a filter, is unknown."""


def _filter_number(filtername):
    digits = ''.join(ch for ch in str(filtername) if ch.isdigit())
    if not digits:
        raise FramePixelAreaError(f"cannot read a wavelength from filter {filtername!r}")
    return int(digits)


def reference_pixel_area_sr(filtername, instrument=None):
    """Reference pixel area in sr for one filter.

    ``instrument`` is the ``INSTRUME`` header value; without it the instrument
    follows from the filter name (F560W and longer are MIRI, the rest NIRCam).
    NIRCam filters below F240 are short-wave, the others long-wave.
    """
    inst = str(instrument).upper().strip() if instrument else ''
    num = _filter_number(filtername)
    if inst in ('', 'NIRCAM'):
        if not inst and num >= 500:
            inst = 'MIRI'
        else:
            inst = 'NIRCAM_SW' if num < 240 else 'NIRCAM_LW'
    if inst not in _NOMINAL_PIXSCALE_ARCSEC:
        raise FramePixelAreaError(
            f"no reference pixel area for instrument {instrument!r}, filter {filtername!r}")
    return _NOMINAL_PIXSCALE_ARCSEC[inst] ** 2 * SR_PER_ARCSEC2


def _pixscale_from_meta(meta):
    for key in _PIXSCALE_KEYS:
        if meta is not None and key in meta:
            val = float(getattr(meta[key], 'value', meta[key]))
            if np.isfinite(val) and val > 0:
                return val
    return None


def frame_pixel_area(filename=None, meta=None):
    """``(area_sr, instrument, source)`` of one frame.

    The area is ``PIXAR_SR`` from the SCI header (or the primary header) of
    ``filename``.  Without the keyword or the file it is the square of the
    table's ``pixscale`` in arcsec, which matches ``PIXAR_SR`` to 0.2 mmag.
    ``instrument`` is ``INSTRUME`` when the header was read, else ``None``.
    Raises :class:`FramePixelAreaError` when neither source exists.
    """
    instrument = None
    if filename is not None and os.path.exists(str(filename)):
        with fits.open(str(filename)) as hdul:
            prim = hdul[0].header
            sci = hdul['SCI'].header if 'SCI' in hdul else prim
            instrument = prim.get('INSTRUME', sci.get('INSTRUME'))
            for hdr in (sci, prim):
                if 'PIXAR_SR' in hdr and float(hdr['PIXAR_SR']) > 0:
                    return float(hdr['PIXAR_SR']), instrument, 'PIXAR_SR'
    pixscale = _pixscale_from_meta(meta)
    if pixscale is not None:
        return pixscale ** 2 * SR_PER_ARCSEC2, instrument, 'pixscale'
    raise FramePixelAreaError(
        f"no PIXAR_SR in {filename!r} and no pixscale in the table meta; "
        f"cannot convert this frame's flux to one pixel area")


def flux_columns_for(tbl):
    """The pixel-sum columns present in ``tbl`` (crowdsource tables carry ``qf``)."""
    names = CROWDSOURCE_FLUX_COLUMNS if 'qf' in tbl.colnames else DAOPHOT_FLUX_COLUMNS
    return [c for c in names if c in tbl.colnames]


def scale_flux_columns(tbl, factor, columns=None):
    """Multiply the flux columns of ``tbl`` by ``factor`` in place, keeping dtypes."""
    for name in (flux_columns_for(tbl) if columns is None else columns):
        col = tbl[name]
        tbl[name] = (col * np.float64(factor)).astype(col.dtype)


def stamp_reference_area(tbl, area_ref_sr):
    """Record ``area_ref_sr`` in ``tbl.meta`` (``PIXAREF`` in deg^2; pixel side keys)."""
    area_deg2 = float(area_ref_sr * DEG2_PER_SR)
    tbl.meta[PIXAREF_KEY] = area_deg2
    side_arcsec = math.sqrt(area_ref_sr / SR_PER_ARCSEC2)
    found = [k for k in _PIXSCALE_KEYS if k in tbl.meta]
    for key in (found or ['pixscale']):
        tbl.meta[key] = side_arcsec


def apply_frame_pixel_area(tables, filtername):
    """Rescale every frame's flux columns to the reference area, in place.

    Each table's fluxes are multiplied by ``A_frame / A_ref`` and its meta gets
    ``PIXAREF``.  A table that already carries ``PIXAREF`` is skipped, so a
    second call changes nothing.  Returns the reference area in sr, or ``None``
    when every table was already stamped.
    """
    todo = [t for t in tables if PIXAREF_KEY not in t.meta]
    if not todo:
        return None
    # A frame with no sources has nothing to rescale, so its area is not needed;
    # it still gets the stamp, because the product inherits the first frame's meta.
    full = [t for t in todo if len(t) > 0]
    resolved = [frame_pixel_area(t.meta.get('FILENAME'), t.meta) for t in full]
    instruments = {str(r[1]).upper() for r in resolved if r[1]}
    if len(instruments) > 1:
        raise FramePixelAreaError(
            f"frames of {filtername} come from several instruments: {sorted(instruments)}")
    area_ref = reference_pixel_area_sr(filtername, next(iter(instruments), None))
    for tbl, (area, _inst, _src) in zip(full, resolved):
        scale_flux_columns(tbl, area / area_ref)
    for tbl in todo:
        stamp_reference_area(tbl, area_ref)
    if resolved:
        areas = np.array([r[0] for r in resolved])
        print(f"apply_frame_pixel_area: {filtername}: {len(full)} frame(s) rescaled to "
              f"{math.sqrt(area_ref / SR_PER_ARCSEC2):.4f}\" pixels; frame/reference "
              f"area ratio {areas.min() / area_ref:.4f}-{areas.max() / area_ref:.4f}",
              flush=True)
    return area_ref


def apply_satstar_frame_pixel_area(tables, frames, filtername):
    """Rescale per-exposure satstar tables to the reference area, in place.

    ``frames`` holds the exposure each table was fit on (``None`` where it is
    not on disk).  The scaling is all or nothing: when any exposure's area is
    unknown no table is touched, because a consolidated catalogue that mixes
    scaled and unscaled rows has no single conversion.  Returns the reference
    area in sr, or ``None`` when nothing was scaled.
    """
    missing = [i for i, f in enumerate(frames) if f is None]
    if missing:
        print(f"WARNING: satstar {filtername}: {len(missing)} of {len(frames)} "
              f"per-exposure catalogs have no frame on disk; fluxes stay in "
              f"native-pixel units, without per-detector pixel areas (#1238)",
              flush=True)
        return None
    try:
        resolved = [frame_pixel_area(f, t.meta) for f, t in zip(frames, tables)]
    except FramePixelAreaError as ex:
        print(f"WARNING: satstar {filtername}: {ex}; fluxes stay in native-pixel "
              f"units, without per-detector pixel areas (#1238)", flush=True)
        return None
    instruments = {str(r[1]).upper() for r in resolved if r[1]}
    area_ref = reference_pixel_area_sr(filtername, next(iter(instruments), None)
                                       if len(instruments) == 1 else None)
    for tbl, (area, _inst, _src) in zip(tables, resolved):
        scale_flux_columns(tbl, area / area_ref,
                           [c for c in DAOPHOT_FLUX_COLUMNS if c in tbl.colnames])
    return area_ref


def frame_area_sr(filename):
    """``PIXAR_SR`` of one frame in sr, from its own header (no table fallback)."""
    return frame_pixel_area(filename)[0]
