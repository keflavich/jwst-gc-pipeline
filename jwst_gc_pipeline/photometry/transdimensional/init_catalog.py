"""Starting catalog for the PCAT sampler from matched-filter peaks.

PCAT's burn-in from an empty catalog needs of order ``N / (births accepted per
proposal)`` proposals; in a crowded field a starting catalog shortens it.  This
module finds peaks in the PSF-matched-filtered image iteratively (detect,
subtract, repeat), with fluxes from the matched filter.  The starting catalog
only sets the chain's initial state; the sampler's moves decide the
posterior.
"""
import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve

from . import kernel as _kernel
from .psf_poly import render_stamp_poly

__all__ = ['peak_catalog']


def peak_catalog(image, weight, cf, nc, basis='quartic15', nsig=5.0, niter=3, bkg=None,
                 min_sep=1.5, fmin=None, lib=None, max_new=None):
    """Iterative matched-filter detection.

    Parameters
    ----------
    image, weight : (ny, nx) data and inverse variance (0 = masked).
    cf, nc, basis : PSF table of the sampler.
    nsig : detection threshold in matched-filter S/N.
    bkg : constant background to subtract (default: median of good pixels).
    min_sep : minimum peak separation (px) within one iteration.
    fmin : flux floor; fainter peaks are dropped.

    Returns ``dict(x, y, f, bkg)`` with 0-based pixel positions.
    """
    terms = _kernel.basis_terms(basis)
    good = (weight > 0) & np.isfinite(image) & np.isfinite(weight)
    img = np.where(good, image, 0.0).astype(np.float64)
    if bkg is None:
        bkg = float(np.median(img[good]))
    w = np.where(good, weight, 0.0)
    P = render_stamp_poly(cf, nc, terms, 0.0, 0.0)
    P2 = P * P
    rad = nc // 2
    rend = _kernel.PSFRenderer(cf, nc, terms, lib=lib)
    xs, ys, fs = [], [], []
    model = np.zeros((img.shape[0] + 2 * (rad + 2), img.shape[1] + 2 * (rad + 2)), np.float32)
    sl = (slice(rad + 2, rad + 2 + img.shape[0]), slice(rad + 2, rad + 2 + img.shape[1]))
    fsz = int(np.ceil(2 * min_sep)) | 1
    for _ in range(niter):
        r = (img - bkg - model[sl]) * good
        num = fftconvolve(r * w, P[::-1, ::-1], mode='same')
        den = fftconvolve(w, P2[::-1, ::-1], mode='same')
        with np.errstate(divide='ignore', invalid='ignore'):
            flux = np.where(den > 0, num / den, 0.0)
            snr = np.where(den > 0, num / np.sqrt(den), 0.0)
        peak = (snr == ndimage.maximum_filter(snr, size=fsz)) & (snr > nsig) & good
        py, px = np.nonzero(peak)
        if px.size == 0:
            break
        if max_new is not None and px.size > max_new:
            o = np.argsort(-snr[py, px])[:max_new]
            py, px = py[o], px[o]
        # sub-pixel refinement: 1D parabola through the S/N map
        ny, nx = snr.shape
        dx = np.zeros(px.size)
        dy = np.zeros(px.size)
        inn = (px > 0) & (px < nx - 1) & (py > 0) & (py < ny - 1)
        a, b, c = snr[py[inn], px[inn] - 1], snr[py[inn], px[inn]], snr[py[inn], px[inn] + 1]
        den_x = a - 2 * b + c
        dx[inn] = np.where(den_x < 0, 0.5 * (a - c) / den_x, 0.0)
        a, c = snr[py[inn] - 1, px[inn]], snr[py[inn] + 1, px[inn]]
        den_y = a - 2 * b + c
        dy[inn] = np.where(den_y < 0, 0.5 * (a - c) / den_y, 0.0)
        x = px + np.clip(dx, -0.5, 0.5)
        y = py + np.clip(dy, -0.5, 0.5)
        f = flux[py, px]
        keep = f > (0 if fmin is None else fmin)
        x, y, f = x[keep], y[keep], f[keep]
        if x.size == 0:
            break
        xs.append(x)
        ys.append(y)
        fs.append(f)
        rend.render(x + rad + 2, y + rad + 2, f, model)
    if not xs:
        return dict(x=np.zeros(0), y=np.zeros(0), f=np.zeros(0), bkg=bkg)
    return dict(x=np.concatenate(xs), y=np.concatenate(ys), f=np.concatenate(fs), bkg=bkg)
