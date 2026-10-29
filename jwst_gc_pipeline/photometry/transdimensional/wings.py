"""Fixed PSF wings of bright sources outside the sampler's ``nc x nc`` stamp.

The sampler renders each source with an ``nc x nc`` stamp centred on pixel
``(ceil(y), ceil(x))`` (:class:`.kernel.PSFRenderer`).  For ``nc = 31`` the
stamp holds EE(31) = 0.91 of the total flux of the F182M ePSF; the light
outside (0.06 within the 101 px ePSF, 0.07 within 201 px) is left in the data
and the sampler fits it with faint spurious sources or with the background.

The two-stage scheme used here: a pilot run gives positions and fluxes of the
bright sources; :func:`wing_image` renders, for each of them, the full ePSF
(cubic spline, the interpolant of :class:`.mocks.TruthRenderer`) over its
field of view with the sampler's stamp footprint set to zero, so that
``stamp + wings`` is the full ePSF.  The wing image is passed to
:func:`.pcat_sampler.run_pcat` as ``fixed_model`` for the second run.
"""
import numpy as np

from .psf_poly import _spline

__all__ = ['wing_image', 'select_wing_sources']


def wing_image(x, y, f, epsf, oversample, nc, shape, half=None):
    """Image of the PSF light outside the sampler stamp for sources ``(x, y, f)``.

    Parameters
    ----------
    x, y : array pixel coordinates of the sources (0-based, array convention
        of the sampler image).
    f : total fluxes (the sampler's flux convention: ``f * ePSF``).
    epsf, oversample : the ePSF (``psf_poly.load_epsf`` output).
    nc : sampler stamp size (odd).
    shape : ``(ny, nx)`` of the output image.
    half : half-width (px) of the rendered region; default and maximum is the
        ePSF field of view, ``floor((n // 2) / oversample - 1)``.

    Returns the ``(ny, nx)`` wing image (zero inside each source's stamp
    footprint ``[ceil(y) - nc//2, ceil(y) + nc//2] x [ceil(x) - nc//2, ...]``).
    """
    if nc % 2 != 1:
        raise ValueError('nc must be odd')
    spl = _spline(epsf, oversample)
    hmax = int(np.floor((epsf.shape[0] // 2) / oversample - 1))
    h = hmax if half is None else int(min(half, hmax))
    rad = nc // 2
    if h <= rad:
        raise ValueError(f'ePSF half-width {h} px does not extend beyond the stamp (rad {rad})')
    ny, nx = shape
    img = np.zeros(shape)
    for xi, yi, fi in zip(np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float)),
                          np.atleast_1d(np.asarray(f, float))):
        cx, cy = int(np.ceil(xi)), int(np.ceil(yi))
        x0, x1 = max(cx - h, 0), min(cx + h + 1, nx)
        y0, y1 = max(cy - h, 0), min(cy + h + 1, ny)
        if x0 >= x1 or y0 >= y1:
            continue
        st = fi * spl(np.arange(y0, y1) - yi, np.arange(x0, x1) - xi, grid=True)
        # zero the sampler's stamp footprint
        bx0, bx1 = max(cx - rad, x0), min(cx + rad + 1, x1)
        by0, by1 = max(cy - rad, y0), min(cy + rad + 1, y1)
        if bx0 < bx1 and by0 < by1:
            st[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] = 0.0
        img[y0:y1, x0:x1] += st
    return img


def select_wing_sources(cond, fmin, prob_min=0.5, band=0):
    """Indices of condensed sources (``condense`` output) that get fixed wings:
    ``prob >= prob_min`` and posterior median flux ``>= fmin``."""
    f = np.asarray(cond['flux'])[band]
    p = np.asarray(cond['prob'])
    return np.flatnonzero((p >= prob_min) & np.isfinite(f) & (f >= fmin))
