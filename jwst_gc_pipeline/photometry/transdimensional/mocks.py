"""Mock images and truth matching for testing the PCAT sampler.

Mock sources are drawn with the cubic-spline interpolant of the full ePSF
(the interpolation photutils ``GriddedPSFModel`` uses, see :mod:`.psf`), over
the ePSF's whole field of view, so mocks carry the PSF wings beyond the
sampler's ``nc x nc`` stamp and do not share the sampler's polynomial
renderer.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .psf_poly import _spline

__all__ = ['TruthRenderer', 'power_law_fluxes', 'make_mock', 'match_truth', 'psf_snr_factor']


class TruthRenderer:
    """Render sources with the ePSF spline (unit flux = ePSF normalisation)."""

    def __init__(self, epsf, oversample, half=None):
        self.spl = _spline(epsf, oversample)
        hmax = (epsf.shape[0] // 2) / oversample - 1
        self.half = int(np.floor(hmax if half is None else min(half, hmax)))

    def render(self, x, y, f, shape, out=None):
        ny, nx = shape
        img = np.zeros(shape) if out is None else out
        h = self.half
        for xi, yi, fi in zip(np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(f)):
            cx, cy = int(np.round(xi)), int(np.round(yi))
            x0, x1 = max(cx - h, 0), min(cx + h + 1, nx)
            y0, y1 = max(cy - h, 0), min(cy + h + 1, ny)
            if x0 >= x1 or y0 >= y1:
                continue
            px = np.arange(x0, x1) - xi
            py = np.arange(y0, y1) - yi
            img[y0:y1, x0:x1] += fi * self.spl(py, px, grid=True)
        return img


def psf_snr_factor(epsf, oversample):
    """``sqrt(sum P^2)`` of a pixel-centred source: S/N = f * factor / sigma for
    a PSF fit with known position on white noise of rms ``sigma``."""
    c = epsf.shape[0] // 2
    nodes = epsf[c % oversample::oversample, c % oversample::oversample]
    return float(np.sqrt(np.sum(nodes ** 2)))


def power_law_fluxes(n, fmin, slope, rng, fmax=None):
    """``n`` fluxes from ``p(f) ~ f^-slope`` on ``[fmin, fmax]``."""
    u = rng.uniform(size=n)
    a = 1.0 - slope
    if fmax is None:
        return fmin * (1 - u) ** (1.0 / a)
    lo, hi = fmin ** a, fmax ** a
    return (lo + u * (hi - lo)) ** (1.0 / a)


def make_mock(shape, x, y, f, renderer, sigma, bkg=0.0, rng=None):
    """Image = sources + constant ``bkg`` + white noise ``sigma``; returns
    ``(image, weight)`` with weight = 1/sigma^2."""
    rng = np.random.default_rng() if rng is None else rng
    img = renderer.render(x, y, f, shape) + bkg + rng.normal(0, sigma, shape)
    w = np.full(shape, 1.0 / sigma ** 2)
    return img, w


def match_truth(x, y, tx, ty, radius=1.0):
    """One-to-one matching of a catalog to the truth within ``radius`` px
    (minimum total distance, ``linear_sum_assignment`` over candidate pairs).

    Returns ``(i_cat, i_truth, dist)`` arrays.
    """
    x, y, tx, ty = map(np.asarray, (x, y, tx, ty))
    if x.size == 0 or tx.size == 0:
        z = np.zeros(0, int)
        return z, z, np.zeros(0)
    tree = cKDTree(np.column_stack([tx, ty]))
    pairs = tree.query_ball_point(np.column_stack([x, y]), radius)
    rows, cols = [], []
    for i, js in enumerate(pairs):
        rows += [i] * len(js)
        cols += list(js)
    if not rows:
        z = np.zeros(0, int)
        return z, z, np.zeros(0)
    rows = np.array(rows)
    cols = np.array(cols)
    ur, ri = np.unique(rows, return_inverse=True)
    uc, ci = np.unique(cols, return_inverse=True)
    big = 1e6
    cost = np.full((ur.size, uc.size), big)
    d = np.hypot(x[rows] - tx[cols], y[rows] - ty[cols])
    cost[ri, ci] = d
    r, c = linear_sum_assignment(cost)
    ok = cost[r, c] < big
    ic, it = ur[r[ok]], uc[c[ok]]
    return ic, it, np.hypot(x[ic] - tx[it], y[ic] - ty[it])
