"""Artificial-star injection into ``--cutout-region`` frames (``--inject-stars``).

The reference-field regression tests (``photometry/reference_fields``) need a
truth list: stars whose positions and fluxes are known exactly, embedded in
REAL crowding, REAL extended emission and REAL noise, and carried through the
WHOLE manual chain (m12 -> m7: per-frame fits, merge, vetting, residual
mosaics, i2d re-detection).  ``artificial_stars.py`` injects into one frame and
runs only the m1 recipe, which says nothing about the vetting and
re-detection stages where faint stars are actually lost.

So the injection happens where every later stage sees it: in the cropped
frame copy that ``_prepare_cutout_input`` writes under
``<basepath>/cutouts/<label>/``.  The cutout data i2d, the residual mosaics and
every per-frame fit are all built from those copies, so an injected star is
indistinguishable from a real one to the pipeline.

* Positions are SKY positions (one table for every frame and filter), mapped
  through each frame's GWCS, so the same star lands on the same sky position in
  every dither and every band.
* Fluxes are per filter, ``flux_jy_<FILTER>`` columns, converted with the
  frame's ``PIXAR_SR`` exactly as ``merge_catalogs`` converts back
  (``flux_jy = flux_img * PIXAR_SR * 1e6``).
* The PSF is the frame's own per-detector fitting grid, re-origined to the
  cutout -- the grid the fit uses.  The test therefore measures detection and
  VETTING losses, not PSF-model mismatch (an injected star is a perfect PSF).
* Photon noise is added with the frame's effective gain (``VAR_POISSON``), and
  ``ERR``/``VAR_POISSON`` are updated in quadrature.
* The noise draw is seeded by the injection seed and the ORIGINAL frame name,
  so every phase (each re-crops and re-injects) writes byte-identical pixels.

Injection is refused for a full-frame run: it rewrites frame pixels, and only
a cutout run writes those to a disposable copy.
"""
import os
import zlib

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy import units as u

#: Half-size of the injected PSF stamp (detector pixels).  The fitting grids
#: are fovp101: 101 detector pixels across (202 grid pixels at oversampling 2),
#: so a half-width of 50 holds the whole grid; a 51x51 stamp (half-width 25)
#: holds 97.4% of an F182M grid PSF's flux.
STAMP_HALF = 50


def flux_column(filtername):
    """The injection-table column carrying ``filtername``'s flux (Jy)."""
    return f'flux_jy_{filtername.upper()}'


def read_injection_table(path):
    """Read and check an injection table: ``ra``, ``dec`` (deg) and one or
    more ``flux_jy_<FILTER>`` columns."""
    tbl = Table.read(path)
    missing = [c for c in ('ra', 'dec') if c not in tbl.colnames]
    if missing:
        raise ValueError(f"injection table {path} lacks column(s) {missing}")
    if not any(c.startswith('flux_jy_') for c in tbl.colnames):
        raise ValueError(f"injection table {path} has no flux_jy_<FILTER> column")
    return tbl


def frame_seed(seed, original_filename):
    """Deterministic per-frame noise seed: same seed + same exposure -> same
    pixels in every phase."""
    return (int(seed) * 1000003 + zlib.crc32(
        os.path.basename(original_filename).encode())) % (2**32)


def inject_table_into_frame(filename, table, grid, filtername, rng,
                            stamp_half=STAMP_HALF):
    """Add ``table``'s stars to the frame FILE ``filename`` in place.

    ``grid`` must be in ``filename``'s pixel coordinates (for a cutout, the
    re-origined grid).  Stars whose centre falls outside the array, or on a
    non-finite pixel, are skipped.  Returns the number injected.
    """
    from jwst_gc_pipeline.frame_wcs import frame_wcs
    from jwst_gc_pipeline.photometry.artificial_stars import (
        inject_stars, estimate_effective_gain)

    col = flux_column(filtername)
    if col not in table.colnames:
        raise ValueError(f"injection table has no {col} column "
                         f"(has {[c for c in table.colnames if c.startswith('flux_jy_')]})")
    ww = frame_wcs(filename)
    with fits.open(filename, mode='update') as hdul:
        sci = np.asarray(hdul['SCI'].data, dtype=float)
        err = np.asarray(hdul['ERR'].data, dtype=float)
        has_vp = 'VAR_POISSON' in hdul
        vp = np.asarray(hdul['VAR_POISSON'].data, dtype=float) if has_vp else None
        pixar_sr = float(hdul['SCI'].header['PIXAR_SR'])
        ny, nx = sci.shape

        sky = SkyCoord(np.asarray(table['ra'], float) * u.deg,
                       np.asarray(table['dec'], float) * u.deg)
        xs, ys = ww.world_to_pixel(sky)
        xs = np.asarray(xs, float)
        ys = np.asarray(ys, float)
        ix = np.rint(xs).astype(int)
        iy = np.rint(ys).astype(int)
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        inside[inside] &= np.isfinite(sci[iy[inside], ix[inside]])
        flux_img = np.asarray(table[col], float) / (1e6 * pixar_sr)
        keep = inside & np.isfinite(flux_img) & (flux_img > 0)

        gain = estimate_effective_gain(sci, vp) if has_vp else None
        sci_new = sci.copy()
        err_new = inject_stars(sci_new, err.copy(), grid, xs[keep], ys[keep],
                               flux_img[keep], rng, gain_eff=gain,
                               stamp_half=stamp_half)
        # inject_stars leaves NaN pixels NaN (NaN + x); keep the DQ story intact
        hdul['SCI'].data = sci_new.astype(hdul['SCI'].data.dtype)
        hdul['ERR'].data = err_new.astype(hdul['ERR'].data.dtype)
        if has_vp and gain:
            hdul['VAR_POISSON'].data = (vp + (err_new**2 - err**2)).astype(
                hdul['VAR_POISSON'].data.dtype)
        hdr = hdul['SCI'].header
        hdr['INJNSTAR'] = (int(keep.sum()), 'artificial stars injected (--inject-stars)')
        hdr['INJGAIN'] = (float(gain) if gain else 0.0, 'effective gain for injected shot noise')
        hdul.flush()
    return int(keep.sum())
