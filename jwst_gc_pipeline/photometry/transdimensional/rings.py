"""Flag faint condensed sources on the diffraction rings of bright sources.

When the ePSF core and first rings differ from the data at the per-cent level
(on F182M frame P the residual rms chi peaks at r = 3-4 px of the brightest
stars), the sampler fits the ring residual with faint sources at the ring
radius.  Those sources have high prevalence (the residual is real signal) and
flux ratios of about 0.2-2 % of the bright star, at distances that match the
local maxima of the ePSF azimuthal profile (for the F182M nrcb3 e16 ePSF in
0.1 px annuli: 3.05 and 3.25 px for the first ring, then 5.45-6.05 px and
7.05-7.55 px).

:func:`psf_ring_radii` measures the ring radii from the ePSF itself;
:func:`flag_psf_rings` flags (does not remove) condensed sources with flux
below ``max_ratio`` times that of a bright source (flux >= ``f_bright``) and a
distance from it within ``halfwidth`` of a ring radius.  A real faint star at a
ring position of a bright star receives the same flag, so the flag marks
sources whose existence the sampler cannot separate from a PSF ring residual.
"""
import numpy as np

__all__ = ['psf_ring_radii', 'flag_psf_rings']


def psf_ring_radii(epsf, oversample, rmin=1.5, rmax=8.0, dr=0.1):
    """Radii (detector px) of the local maxima of the ePSF azimuthal mean.

    ``epsf``: oversampled ePSF array (centre at the array centre, as written by
    the ePSF builder and read by :func:`.psf_poly.load_epsf`).  The azimuthal
    mean is taken in ``dr``-px annuli between ``rmin`` (outside the core) and
    ``rmax``; returns the annulus centres of strict local maxima, ascending.
    """
    epsf = np.asarray(epsf, float)
    ny, nx = epsf.shape
    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(xx - (nx - 1) / 2.0, yy - (ny - 1) / 2.0) / float(oversample)
    edges = np.arange(rmin, rmax + dr / 2, dr)
    idx = np.digitize(r.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < edges.size - 1)
    s = np.bincount(idx[ok], weights=epsf.ravel()[ok], minlength=edges.size - 1)
    n = np.bincount(idx[ok], minlength=edges.size - 1)
    prof = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    rc = 0.5 * (edges[1:] + edges[:-1])
    k = np.flatnonzero((prof[1:-1] > prof[:-2]) & (prof[1:-1] > prof[2:])) + 1
    return rc[k]


def flag_psf_rings(x, y, flux, ring_radii, f_bright=1000.0, halfwidth=0.6, max_ratio=0.03):
    """Flag sources on a PSF ring of a brighter source.

    Parameters
    ----------
    x, y, flux : condensed-source positions (px) and fluxes (posterior median).
    ring_radii : ring radii in px (:func:`psf_ring_radii`).
    f_bright : only sources with flux >= ``f_bright`` act as parents.
    halfwidth : a source within ``halfwidth`` px of ``ring_radii[k]`` from a
        parent is on ring k.
    max_ratio : flagged sources have flux < ``max_ratio`` * parent flux.

    Returns ``(flag, parent, ring)``: boolean flag per source, the index of the
    brightest parent that flags it (-1 if none) and the ring index (-1).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    flux = np.asarray(flux, float)
    rr = np.asarray(ring_radii, float)
    n = x.size
    flag = np.zeros(n, bool)
    parent = np.full(n, -1, int)
    ring = np.full(n, -1, int)
    if n == 0 or rr.size == 0:
        return flag, parent, ring
    rmax = rr.max() + halfwidth
    par = np.flatnonzero(flux >= f_bright)
    for j in par[np.argsort(-flux[par])]:
        d = np.hypot(x - x[j], y - y[j])
        cand = np.flatnonzero((d < rmax) & (flux < max_ratio * flux[j]) & ~flag)
        if cand.size == 0:
            continue
        dd = np.abs(d[cand, None] - rr[None, :])
        on = dd.min(axis=1) <= halfwidth
        c = cand[on]
        flag[c] = True
        parent[c] = j
        ring[c] = np.argmin(dd[on], axis=1)
    return flag, parent, ring
