"""Common ePSF loader for the transdimensional (PCAT) cutout experiment.

Convention
----------
The common PSF files (``epsf_<tag>_os5.fits``, ``gridded_<tag>_os5.fits``,
``psf_<tag>_os1.fits``; built by ``outputs/pcat_cutout/psf/scripts/build_epsf.py``
with stpsf 2.2.0) store an **ePSF**, i.e. a pixel-integrated PSF:

* ``epsf[j, i]`` (oversampling ``N = OVERSAMP = 5``) is the fraction of a star's
  total flux that lands in one detector pixel whose centre sits at sub-pixel
  offset ``((i - c) / N, (j - c) / N)`` detector pixels from the star, with
  ``c = shape // 2`` (the star sits at the array centre, on a pixel centre).
* It is the 5x5 sub-pixel box **sum** of stpsf's ``OVERDIST`` extension
  (rotation + SIAF distortion + Gaussian charge diffusion + IPC/PPC, the IPC
  applied once, on the oversampled grid with stpsf's comb kernel).  Sampling
  it at sub-pixels ``2::5`` reproduces stpsf's ``DET_DIST`` exactly.  The array
  is identical (to 5e-16 of the peak away from the edges) to what stpsf's own
  ``psf_grid`` writes, so it plugs into photutils ``GriddedPSFModel`` directly.
* Normalisation: ``epsf.sum() / N**2`` = the encircled energy inside the stpsf
  field of view (header ``NORMSUM``), with the stpsf ``normalize='first'``
  convention (unit flux entering the telescope, distortion-area factor
  included).  For F182M / NRCB3 at (145, 846) the flux=1 model holds 0.658,
  0.817, 0.893, 0.942 and 0.968 of the flux in 5x5, 11x11, 25x25, 51x51 and
  101x101 px boxes; a 201 px build holds 0.980 (``outputs/pcat_cutout/psf/
  epsf_checks.json``).  A ``GriddedPSFModel`` with ``flux=1`` summed over all
  detector pixels returns ``NORMSUM``; a fitted ``flux`` is therefore the total
  flux in image units (MJy/sr * pix for a ``crf``), extrapolated by the model
  beyond the fit window.
* Pixel integration at N = 5 is a 5-point midpoint rule on stpsf's point
  samples.  It puts the centred peak 0.4-0.5 % above the converged pixel
  integral (Richardson extrapolation of N = 3, 5 and of N = 6, 10), which
  lowers fitted fluxes by about 0.2 % (``psf/os_convergence.json``).  stpsf's
  own even-N grids use a trapezoid rule: N = 2 gives a 3.1 % lower peak and
  1.7 % higher fitted fluxes than this ePSF.
* ``stpsf.utils.to_griddedpsfmodel`` pairs plane ``i`` with key ``DET_YX{i}``,
  which transposes multi-PSF grids written by stpsf 2.x / webbpsf >= 1.3 (the
  PSF computed at (x, y) lands at (y, x); 12 of 16 planes of the production
  4x4 grid and 2 of 4 planes of peppar's 2x2 grid move,
  ``outputs/pcat_cutout/psf/grids_info.json``).  For real grids use
  ``jwst_gc_pipeline.photometry.psf_grid_io.load_stpsf_grid`` (as
  ``to_gridded_psf_model`` does).  The single-position ``gridded_*`` files
  hold identical planes, so both readers agree on them.
* Validation on frame P (``outputs/pcat_cutout/psf/tables.md``): the stars are
  broader than this ePSF (Gaussian fit FWHM 1.99 px vs 1.96 px); the variant
  built with 3 mas Gaussian jitter (``*_e16_jit3_*``) lowers chi2 for 75 % of
  the validation stars and returns fluxes 0.26 % higher.
* The detector-sampled stamp ``psf_<tag>_os1.fits`` is ``DET_DIST`` (star on the
  centre of pixel ``N//2``), with the same normalisation (sum = ``NORMSUM``).

Functions
---------
load_common_epsf(path) -> (array, oversample)
to_gridded_psf_model(path) -> photutils.psf.GriddedPSFModel
detector_stamp(epsf, oversample, size, dx=0, dy=0) -> detector-sampled stamp
"""
from collections import OrderedDict

import numpy as np
from astropy.io import fits
from astropy.nddata import NDData

__all__ = ['load_common_epsf', 'to_gridded_psf_model', 'detector_stamp']

#: detector corners used for the 2x2 identical-copy grid, as (x, y)
_CORNERS_XY = ((0.0, 0.0), (2047.0, 0.0), (0.0, 2047.0), (2047.0, 2047.0))


def load_common_epsf(path):
    """Return ``(epsf, oversample)`` from an ``epsf_*`` or ``gridded_*`` file.

    For a gridded (3D) file the planes must be identical (position-independent
    common PSF); the first plane is returned.
    """
    with fits.open(path) as hdul:
        data = np.array(hdul[0].data, dtype=np.float64)
        hdr = hdul[0].header
        if 'OVERSAMP' not in hdr:
            raise KeyError(f"{path}: no OVERSAMP key")
        oversample = int(hdr['OVERSAMP'])
    if data.ndim == 3:
        if not np.all(data == data[:1]):
            raise ValueError(f'{path}: planes differ; use to_gridded_psf_model for a real grid')
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f'{path}: expected a 2D ePSF, got shape {data.shape}')
    return data, oversample


def to_gridded_psf_model(path):
    """photutils ``GriddedPSFModel`` from a common-PSF file.

    * ``epsf_*`` (2D) or ``gridded_*`` (identical planes): a 2x2 grid of identical
      copies at the detector corners (position-independent model).
    * Any other stpsf grid (planes differ) is handed to the pipeline's
      plane-order-aware loader ``jwst_gc_pipeline.photometry.psf_grid_io``.
    """
    from photutils.psf import GriddedPSFModel
    with fits.open(path) as hdul:
        data = np.array(hdul[0].data, dtype=np.float64)
        hdr = hdul[0].header.copy(strip=True)
    if data.ndim == 3 and not np.all(data == data[:1]):
        from jwst_gc_pipeline.photometry.psf_grid_io import load_stpsf_grid
        return load_stpsf_grid(path)
    epsf, oversample = load_common_epsf(path)
    for key in ('COMMENT', 'HISTORY', ''):
        hdr.remove(key, ignore_missing=True, remove_all=True)
    meta = OrderedDict((c.keyword.lower(), (c.value, c.comment)) for c in hdr.cards
                       if not c.keyword.startswith('DET_YX'))
    # photutils sorts grid points by (y, x); copies are identical so order is moot
    meta['grid_xypos'] = [tuple(p) for p in _CORNERS_XY]
    meta['oversampling'] = oversample
    cube = np.repeat(epsf[np.newaxis], len(_CORNERS_XY), axis=0)
    return GriddedPSFModel(NDData(cube, meta=meta))


def detector_stamp(epsf, oversample, size, dx=0.0, dy=0.0):
    """Detector-sampled ``size x size`` stamp of a star at offset (dx, dy) px
    from the centre of pixel ``size // 2``, by cubic-spline interpolation of the
    ePSF (the same interpolation photutils ``GriddedPSFModel`` uses).

    The stamp holds flux fractions per pixel (unit-flux star).
    """
    from scipy.interpolate import RectBivariateSpline
    n = epsf.shape[0]
    c = (n - 1) / 2.0
    sub = (np.arange(n) - c) / oversample          # sub-pixel offsets in det px
    spl = RectBivariateSpline(sub, sub, epsf, kx=3, ky=3, s=0)
    pix = np.arange(size) - size // 2
    return spl(pix - dy, pix - dx, grid=True)
