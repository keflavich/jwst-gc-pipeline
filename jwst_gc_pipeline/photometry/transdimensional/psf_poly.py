"""Polynomial sub-pixel PSF tables for the PCAT renderer.

PCAT represents a detector-sampled PSF stamp of ``nc x nc`` pixels by a
polynomial in the sub-pixel offsets ``(dx, dy) in [0, 1)^2`` for every stamp
pixel (``cf`` has shape ``(k, nc*nc)``; PCAT-DE ``psf_poly_fit``).  This module
builds ``cf`` from the common ePSF (``outputs/pcat_cutout/psf/epsf_*_os5.fits``,
see :mod:`.psf`).

Phase convention (established by ``tests/test_psf_render.py``)
---------------------------------------------------------------
The renderer (:class:`.kernel.PSFRenderer`, identical to PCAT
``image_model_eval``) puts stamp pixel ``(a, b)`` on image pixel
``(ceil(y) - rad + a, ceil(x) - rad + b)`` with ``rad = nc // 2`` and evaluates
the polynomial at ``dx = ceil(x) - x``, ``dy = ceil(y) - y``.  The image pixel
centre then sits at offset ``u = (b - rad + dx, a - rad + dy)`` from the star,
so ``cf[:, a*nc + b]`` must reproduce ``ePSF(u)``, i.e. the ePSF array value at
oversampled index ``c + N*u`` (``c = shape // 2``, ``N`` = oversampling), the
convention photutils ``GriddedPSFModel`` uses.  With ``nbin = N`` the node
samples are exact ePSF array entries: ``psf0[j, i] = epsf[c - N*rad + j,
c - N*rad + i]``.

Fitting
-------
``fit='nodes'`` least-squares fits each stamp pixel to its ``(nbin+1)^2``
node samples (PCAT-DE).  ``fit='dense'`` fits to the cubic-spline interpolant
of the ePSF (``RectBivariateSpline``, kx = ky = 3: the interpolation photutils
uses) on ``(nsub+1)^2`` sub-pixel points, which targets the GriddedPSFModel
surface between the nodes as well.

The stamp is **not** renormalised: a flux-1 source deposits ``EE(nc)``, the
fraction of the total flux inside the ``nc x nc`` box, so fitted fluxes are
total fluxes (the convention of the common ePSF).
"""
import numpy as np
from astropy.io import fits

from .kernel import basis_terms, design_rows

__all__ = ['encircled_energy', 'build_cf', 'load_epsf', 'cf_from_file', 'render_stamp_poly']


def load_epsf(path):
    """Return ``(epsf, oversample, header)`` of a 2D common-ePSF file."""
    with fits.open(path) as hdul:
        data = np.array(hdul[0].data, dtype=np.float64)
        hdr = hdul[0].header.copy()
    if data.ndim == 3:
        if not np.all(data == data[:1]):
            raise ValueError(f'{path}: planes differ (position-dependent grid)')
        data = data[0]
    return data, int(hdr['OVERSAMP']), hdr


def encircled_energy(epsf, oversample, nc):
    """Fraction of the total flux inside the ``nc x nc`` box of a star centred
    on a pixel (sum of the ePSF at the integer-offset nodes)."""
    c = epsf.shape[0] // 2
    rad = nc // 2
    idx = c + oversample * np.arange(-rad, rad + 1)
    if idx.min() < 0 or idx.max() >= epsf.shape[0]:
        raise ValueError(f'nc={nc} exceeds the ePSF field of view')
    return float(epsf[np.ix_(idx, idx)].sum())


def _spline(epsf, oversample):
    from scipy.interpolate import RectBivariateSpline
    n = epsf.shape[0]
    c = n // 2
    sub = (np.arange(n) - c) / oversample
    return RectBivariateSpline(sub, sub, epsf, kx=3, ky=3, s=0)


def build_cf(epsf, oversample, nc, basis='quartic15', fit='nodes', nsub=20, nbin=None):
    """Coefficient table ``cf`` (float32, ``(k, nc*nc)``) for the renderer.

    The defaults (``quartic15`` fitted to the ePSF nodes) are the sampler's
    (``run_pcat``, ``BandData``).  Returns ``(cf, info)``; ``info`` records the
    inputs and ``EE(nc)``.
    """
    if nc % 2 != 1:
        raise ValueError('nc must be odd')
    rad = nc // 2
    terms = basis_terms(basis)
    nbin = oversample if nbin is None else int(nbin)
    if fit == 'nodes':
        npts = nbin
        if nbin == oversample:
            c = epsf.shape[0] // 2
            i0 = c - oversample * rad
            idx = i0 + np.arange(nc * nbin + 1)
            if idx.min() < 0 or idx.max() >= epsf.shape[0]:
                raise ValueError('stamp exceeds the ePSF field of view')
            grid = epsf[np.ix_(idx, idx)]
        else:
            spl = _spline(epsf, oversample)
            u = (np.arange(nc * nbin + 1) - nbin * rad) / nbin
            grid = spl(u, u, grid=True)
    elif fit == 'dense':
        npts = int(nsub)
        spl = _spline(epsf, oversample)
        u = (np.arange(nc * npts + 1) - npts * rad) / npts
        grid = spl(u, u, grid=True)
    else:
        raise ValueError(f'unknown fit {fit!r}')
    t = np.arange(npts + 1) / npts
    dyg, dxg = np.meshgrid(t, t, indexing='ij')
    A = design_rows(dxg.ravel(), dyg.ravel(), terms).astype(np.float64)
    pinv = np.linalg.pinv(A)                       # (k, (npts+1)^2)
    # gather every stamp pixel's (npts+1)^2 samples: rows a*npts + l, cols b*npts + m
    a = np.arange(nc)
    rows = a[:, None] * npts + np.arange(npts + 1)[None, :]   # (nc, npts+1)
    blocks = grid[rows[:, None, :, None], rows[None, :, None, :]]   # (nc, nc, l, m)
    blocks = blocks.reshape(nc, nc, (npts + 1) ** 2)
    cf = np.einsum('kp,abp->kab', pinv, blocks).reshape(len(terms), nc * nc)
    info = dict(nc=int(nc), rad=int(rad), basis=basis, nterms=len(terms), fit=fit,
                npts=int(npts), oversample=int(oversample),
                ee_nc=encircled_energy(epsf, oversample, nc),
                total_epsf=float(epsf.sum() / oversample ** 2))
    return np.ascontiguousarray(cf, dtype=np.float32), info


def cf_from_file(path, nc, basis='quartic15', fit='nodes', nsub=20):
    """``build_cf`` on an ePSF file; returns ``(cf, info)`` with the file name."""
    epsf, os_, hdr = load_epsf(path)
    cf, info = build_cf(epsf, os_, nc, basis=basis, fit=fit, nsub=nsub)
    info['psf_file'] = str(path)
    for key in ('NORMSUM', 'FILTER', 'DETECTOR', 'DET_X', 'DET_Y'):
        if key in hdr:
            info[key.lower()] = hdr[key]
    return cf, info


def render_stamp_poly(cf, nc, terms, dx, dy):
    """``nc x nc`` stamp for sub-pixel offsets ``(dx, dy)`` (unit flux)."""
    dd = design_rows(np.atleast_1d(dx), np.atleast_1d(dy), terms).astype(np.float64)
    return (dd @ cf.astype(np.float64)).reshape(nc, nc)
