"""Pixel-area correction of per-frame point-source fluxes (opt-in, issue #1151).

The problem
-----------
The PSF fit runs on the crf ``SCI`` array, which is in MJy/sr, with a unit-sum
ePSF, so ``flux_fit`` is the star's locally summed surface brightness.  The
merge turns it into Jy with ONE constant pixel solid angle per catalogue
(``wcs.proj_plane_pixel_area()``, ``merge_catalogs.merge_daophot`` and
``merge_crowdsource``; satstars in ``replace_saturated``).  The flat field makes
a uniform sky read uniform in MJy/sr, so a pixel's true solid angle is
``PIXAR_SR * AREA(x, y)``, where ``AREA`` is the crf extension the jwst photom
step attaches from the CRDS ``area`` reference.  A star on a larger-than-nominal
pixel therefore reads faint by ``2.5 log10(AREA(x, y))``.

GDC_EXPERIMENT_REPORT.md section 12 sized the omission at 7-11 mmag rms per
NIRCam SW detector (pk-pk 27-52 mmag).  Against the wd2 dolphot catalogue the
within-detector gradients of ours - dolphot follow ``AREA`` in every band, and
dividing it out lowers the robust scatter of unsaturated stars by 10-30 % in
most bands (#1151).

The fix
-------
``PHOT_PIXEL_AREA=1`` multiplies each per-frame catalogue's flux columns by
``AREA`` at the star's fitted pixel before the per-frame catalogues are
averaged (``merge_individual_frames``), and does the same for every
per-exposure satstar catalogue before the consolidated satstar catalogue is
built (``load_satstar_catalog``).  The downstream constant-area calibration is
unchanged, so the merged flux becomes ``flux_fit * AREA * Omega_mosaic``; the
remaining constant ``Omega_mosaic / PIXAR_SR`` differs from 1 by 1-2e-4
(0.1-0.2 mmag) and is left alone.

``AREA`` is read at the nearest pixel.  It varies by about 1e-5 per pixel, so
interpolation would change nothing at the mmag level.  Positions off the array
take the nearest edge pixel (satstars fitted from an off-detector seed); a
non-finite position or a non-finite or non-positive ``AREA`` value keeps a
factor of 1.
"""
import os

import numpy as np
from astropy.io import fits

PIXEL_AREA_ENV = 'PHOT_PIXEL_AREA'

#: Flux-like columns of the per-frame catalogues: daophot (``flux_fit``,
#: ``flux_err``, ``flux_init``) and crowdsource (``flux``, ``dflux``,
#: ``fluxiso``).  Surface-brightness columns (``local_bkg`` and the residual
#: background columns) are per pixel and stay as they are.
FLUX_COLUMNS = ('flux_fit', 'flux_err', 'flux_init', 'flux', 'dflux', 'fluxiso')

#: Fitted-position columns of the per-frame catalogues, in order of preference.
POSITION_COLUMNS = (('x_fit', 'y_fit'), ('x_0', 'y_0'),
                    ('xcentroid', 'ycentroid'), ('x', 'y'))

#: Frame-pixel position columns of a per-exposure satstar catalogue.  Its
#: ``x_fit``/``y_fit`` are in the local fit cutout (about 81 px for every star),
#: and ``xcentroid``/``ycentroid`` (equal to ``x_0``/``y_0``) carry the fitted
#: position in the frame.
SATSTAR_POSITION_COLUMNS = (('xcentroid', 'ycentroid'), ('x_0', 'y_0'))

#: Table meta key that records the correction, so it is never applied twice.
META_KEY = 'PIXAREA'


class PixelAreaError(RuntimeError):
    """The pixel-area correction was requested but cannot be applied."""


def pixel_area_enabled():
    """Whether ``PHOT_PIXEL_AREA`` asks for the correction (default off).
    Any of 1/true/yes/on (case-insensitive) enables it."""
    return os.environ.get(PIXEL_AREA_ENV, '0').strip().lower() \
        in ('1', 'true', 'yes', 'on')


def area_at(area, x, y):
    """``area`` at the nearest pixel of each (x, y), with the fallbacks of the
    module docstring.  ``x`` is the column (axis 1) coordinate."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    factor = np.ones(x.shape, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return factor
    ny, nx = area.shape
    ix = np.clip(np.round(x[ok]).astype(int), 0, nx - 1)
    iy = np.clip(np.round(y[ok]).astype(int), 0, ny - 1)
    value = np.asarray(area[iy, ix], dtype=float)
    factor[ok] = np.where(np.isfinite(value) & (value > 0), value, 1.0)
    return factor


def _position_columns(tbl, candidates=POSITION_COLUMNS):
    for xcol, ycol in candidates:
        if xcol in tbl.colnames and ycol in tbl.colnames:
            return xcol, ycol
    raise PixelAreaError(
        f"{PIXEL_AREA_ENV}: no pixel position columns in the catalogue "
        f"(looked for {candidates}); columns are {tbl.colnames}")


def pixel_area_factor(frame, x, y):
    """``AREA(x, y)`` of the crf ``frame`` at the given pixel positions."""
    if not os.path.exists(frame):
        raise FileNotFoundError(
            f"{PIXEL_AREA_ENV}=1 needs the frame the catalogue was fit on to "
            f"read its AREA extension, and {frame} does not exist")
    with fits.open(frame, memmap=True) as hdul:
        if 'AREA' not in hdul:
            raise PixelAreaError(
                f"{PIXEL_AREA_ENV}=1 but {frame} has no AREA extension (the "
                f"jwst photom step attaches it from the CRDS area reference)")
        area = hdul['AREA'].data
        if 'SCI' in hdul:
            sci_shape = (hdul['SCI'].header.get('NAXIS2'),
                         hdul['SCI'].header.get('NAXIS1'))
            if sci_shape != area.shape:
                raise PixelAreaError(
                    f"{frame}: AREA shape {area.shape} differs from SCI shape "
                    f"{sci_shape}")
        return area_at(area, x, y)


def apply_pixel_area(tbl, frame, position_columns=POSITION_COLUMNS):
    """Multiply the flux columns of one frame's catalogue by ``AREA`` at each
    row's fitted pixel, in place.  ``position_columns`` lists the candidate
    (x, y) column pairs in frame pixels, first present pair wins.  Returns the
    per-row factor, or None when the table already carries the correction
    (``meta['PIXAREA']``)."""
    if tbl.meta.get(META_KEY):
        return None
    if len(tbl) == 0:
        tbl.meta[META_KEY] = 1
        return np.ones(0)
    xcol, ycol = _position_columns(tbl, position_columns)
    factor = pixel_area_factor(frame, tbl[xcol], tbl[ycol])
    for col in FLUX_COLUMNS:
        if col in tbl.colnames:
            # Column * ndarray keeps the column's unit and mask
            tbl[col] = tbl[col] * factor
    tbl.meta[META_KEY] = 1
    return factor


def apply_pixel_area_to_frames(tables, frame_key='FILENAME', label=''):
    """``apply_pixel_area`` for every per-frame catalogue, each on the frame
    named by its ``meta[frame_key]``.  Prints one summary line."""
    factors = []
    for tbl in tables:
        if frame_key not in tbl.meta:
            raise PixelAreaError(
                f"{PIXEL_AREA_ENV}=1 but a per-frame catalogue has no "
                f"meta[{frame_key!r}] naming its frame")
        f = apply_pixel_area(tbl, tbl.meta[frame_key])
        if f is not None and len(f):
            factors.append(f)
    if factors:
        mag = 2.5 * np.log10(np.concatenate(factors))
        print(f"pixel-area correction{' ' + label if label else ''}: "
              f"{len(tables)} frames, {mag.size} rows, 2.5 log10(AREA) "
              f"p5/p50/p95 = {np.percentile(mag, 5):+.4f}/"
              f"{np.percentile(mag, 50):+.4f}/{np.percentile(mag, 95):+.4f} mag",
              flush=True)
