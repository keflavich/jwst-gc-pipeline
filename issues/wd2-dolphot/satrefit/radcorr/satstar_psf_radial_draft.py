"""Empirical radial correction of the satstar PSF model (issue #1142).

On wd2 F150W and F200W the STPSF grids under-predict the near wings
(3-11 px) of every star, saturated or not, by about 10 %.  The daophot fit of
an unsaturated star (5x5 box) weights the core and barely sees this.  The
masked-core satstar fit leans on those wing pixels, so it raises the
amplitude to match them: uncapped SW satstar fluxes read 0.05-0.14 mag
bright against dolphot, growing with brightness.  An offline refit of the
production fit (Q_integ/satrefit) with the PSF multiplied by
(1 + Delta_unsat(r)), Delta_unsat measured on isolated unsaturated stars and
not renormalised, plus a free constant background, brought F150W from
-0.047 to -0.015 mag and F200W from -0.066 to -0.020 mag (all-star median,
catalog cap kept).

``measure_radial_correction`` measures Delta(r) in the frame being fitted.
The calibrators are isolated bright unsaturated stars, each fitted the way
the daophot stage fits it (5x5 box, ``LocalBackground`` annulus and aperture
radius as ``cataloging``), so that a satstar amplitude fitted with
P (1 + Delta) is on the daophot flux scale.  Per radial bin,

    Delta_i(r) = sum(data - bg) / sum(F_i P) - 1

over the bin's usable pixels, with ``bg`` the sigma-clipped median of a far
annulus (1.8-2.4" by default, as the radial-profile study), and Delta(r) is
the median over calibrators.  ``RadialCorrection`` interpolates it and tapers
it to zero at ``rmax``.
"""
import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.table import Table
from photutils.background import LocalBackground
from scipy import ndimage
from scipy.spatial import cKDTree

from ..photometry.psf_fitting import _make_psfphotometry


def default_bin_edges(rmax_pix):
    """Radial bin edges in pixels: 0.5 px steps to 4 px, then 5, 6, 8, 10,
    12, 15, 18, 22, 26, ... (the radial-profile study's edges), ending at
    ``rmax_pix``."""
    edges = list(np.arange(0.0, 4.01, 0.5))
    for e in (5, 6, 8, 10, 12, 15, 18, 22, 26, 30, 36, 43, 50):
        if e < rmax_pix:
            edges.append(float(e))
    if edges[-1] < rmax_pix:
        edges.append(float(rmax_pix))
    return np.array(edges)


class RadialCorrection:
    """Delta(r) as a callable of the radius in pixels.

    Linear interpolation between the bin centres ``r_pix``; below the first
    centre the first value; from the last centre the correction tapers
    linearly to zero at ``rmax_pix`` and is zero beyond.  Non-finite bins are
    dropped."""

    def __init__(self, r_pix, delta, rmax_pix):
        r_pix = np.asarray(r_pix, float)
        delta = np.asarray(delta, float)
        keep = np.isfinite(r_pix) & np.isfinite(delta) & (r_pix < rmax_pix)
        if not keep.any():
            raise ValueError("RadialCorrection: no finite bin inside rmax")
        self.rmax_pix = float(rmax_pix)
        self.r = np.concatenate([r_pix[keep], [self.rmax_pix]])
        self.d = np.concatenate([delta[keep], [0.0]])

    def __call__(self, r):
        return np.interp(np.asarray(r, float), self.r, self.d,
                         left=self.d[0], right=0.0)

    def __repr__(self):
        pts = ', '.join(f'{r:.2f}:{d:+.3f}' for r, d in zip(self.r, self.d))
        return f'RadialCorrection({pts})'


def _select_calibrators(data, good, sat, *, fwhm_pix, edge, iso_pix,
                        iso_frac, sat_iso_pix, n_max):
    """Brightest isolated local maxima, in descending peak order.

    A calibrator has no saturated pixel within ``sat_iso_pix``, a clean 5x5
    core, and no other local maximum brighter than ``iso_frac`` times its
    own peak within ``iso_pix``."""
    ny, nx = data.shape
    work = np.where(good, data, -np.inf)
    size = 2 * int(np.ceil(fwhm_pix)) + 1
    peak = (work == ndimage.maximum_filter(work, size=size, mode='constant',
                                           cval=-np.inf)) & good
    peak[:edge] = False
    peak[ny - edge:] = False
    peak[:, :edge] = False
    peak[:, nx - edge:] = False
    _, med, sig = sigma_clipped_stats(data[good][::7], sigma=3.0, maxiters=5)
    yy, xx = np.nonzero(peak & (data > med + 20.0 * max(sig, 1e-30)))
    if yy.size == 0:
        return np.zeros(0, int), np.zeros(0, int)
    vals = data[yy, xx]
    order = np.argsort(vals)[::-1]
    yy, xx, vals = yy[order], xx[order], vals[order]
    dsat = ndimage.distance_transform_edt(~sat) if sat.any() else None
    tree = cKDTree(np.column_stack([xx, yy]))
    sel = []
    for k in range(yy.size):
        y, x, v = yy[k], xx[k], vals[k]
        if dsat is not None and dsat[y, x] < sat_iso_pix:
            continue
        if not good[y - 2:y + 3, x - 2:x + 3].all():
            continue
        nbr = tree.query_ball_point([x, y], iso_pix)
        if any(j != k and vals[j] > iso_frac * v for j in nbr):
            continue
        sel.append(k)
        if len(sel) >= n_max:
            break
    sel = np.asarray(sel, int)
    return yy[sel], xx[sel]


def measure_radial_correction(data, err, sat, bad, psf_grid, *, fwhm_pix,
                              pixscale, rmax_arcsec,
                              bg_annulus_arcsec=(1.8, 2.4), bin_edges_pix=None,
                              n_max=400, n_min=30, iso_arcsec=0.6,
                              iso_frac=0.05, sat_iso_arcsec=1.0):
    """Measure Delta(r) of ``psf_grid`` on isolated unsaturated stars.

    Parameters
    ----------
    data, err : 2D arrays
        The frame as the satstar fit sees it and its 1-sigma errors.
    sat : 2D bool
        Saturated pixels (any group); calibrators keep ``sat_iso_arcsec``
        away from them.
    bad : 2D bool
        Unusable pixels (NaN, DO_NOT_USE, saturated, zero).
    psf_grid : `~photutils.psf.GriddedPSFModel`
        The satstar PSF grid in DETECTOR coordinates.
    fwhm_pix, pixscale : float
        PSF FWHM in pixels and pixel scale in arcsec.
    rmax_arcsec : float
        Outer radius of the correction.

    Returns
    -------
    dict with ``r_pix`` (bin centres), ``delta``, ``n`` (calibrators per
    bin), ``n_cal``, ``edges``, ``rmax_pix``, and ``correction`` (a
    `RadialCorrection`, or None when fewer than ``n_min`` calibrators have a
    profile).
    """
    data = np.asarray(data, float)
    good = (~np.asarray(bad, bool)) & np.isfinite(data)
    sat = np.asarray(sat, bool)
    rmax_pix = float(rmax_arcsec) / pixscale
    r_bg_in, r_bg_out = (float(a) / pixscale for a in bg_annulus_arcsec)
    edges = (default_bin_edges(rmax_pix) if bin_edges_pix is None
             else np.asarray(bin_edges_pix, float))
    nb = len(edges) - 1
    out = dict(r_pix=0.5 * (edges[:-1] + edges[1:]),
               delta=np.full(nb, np.nan), n=np.zeros(nb, int), n_cal=0,
               edges=edges, rmax_pix=rmax_pix, correction=None)
    half = int(np.ceil(r_bg_out)) + 1
    yc, xc = _select_calibrators(
        data, good, sat, fwhm_pix=fwhm_pix, edge=half + 1,
        iso_pix=float(iso_arcsec) / pixscale, iso_frac=iso_frac,
        sat_iso_pix=float(sat_iso_arcsec) / pixscale, n_max=n_max)
    if yc.size < n_min:
        return out

    # daophot-like fit (cataloging: fit_shape 5x5, aperture 2 FWHM,
    # LocalBackground(max(6, 2.5 FWHM), +max(4, FWHM)))
    ap = 2.0 * fwhm_pix
    lb_in = max(6, int(round(ap + 0.5 * fwhm_pix)))
    lb_out = lb_in + max(4, int(round(fwhm_pix)))
    phot = _make_psfphotometry(
        localbkg_estimator=LocalBackground(lb_in, lb_out),
        psf_model=psf_grid, fitter=LevMarLSQFitter(), fit_shape=(5, 5),
        aperture_radius=ap, progress_bar=False)
    init = Table({'x': xc.astype(float), 'y': yc.astype(float)})
    fit_mask = ~good
    res = phot(np.where(good, data, 0.0), mask=fit_mask,
               error=np.where(good, err, 1e10), init_params=init)
    xf = np.asarray(res['x_fit'], float)
    yf = np.asarray(res['y_fit'], float)
    ff = np.asarray(res['flux_fit'], float)

    rin = int(np.ceil(rmax_pix)) + 1
    prof = []
    for x, y, f in zip(xf, yf, ff):
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(f) and f > 0):
            continue
        ix, iy = int(round(x)), int(round(y))
        if not (half <= ix < data.shape[1] - half and half <= iy < data.shape[0] - half):
            continue
        sub = data[iy - half:iy + half + 1, ix - half:ix + half + 1]
        gsub = good[iy - half:iy + half + 1, ix - half:ix + half + 1]
        gy, gx = np.mgrid[iy - half:iy + half + 1, ix - half:ix + half + 1]
        rr = np.hypot(gx - x, gy - y)
        bsel = gsub & (rr >= r_bg_in) & (rr < r_bg_out)
        if bsel.sum() < 50:
            continue
        _, bg, _ = sigma_clipped_stats(sub[bsel], sigma=3.0, maxiters=5)
        c = slice(half - rin, half + rin + 1)
        model = psf_grid.evaluate(gx[c, c].astype(float), gy[c, c].astype(float),
                                  f, x, y)
        d = sub[c, c] - bg
        g = gsub[c, c]
        r = rr[c, c]
        row = np.full(nb, np.nan)
        for k in range(nb):
            inb = (r >= edges[k]) & (r < edges[k + 1])
            use = inb & g
            if use.sum() < max(1, 0.5 * inb.sum()):
                continue
            sm = float(np.sum(model[use]))
            if sm > 0:
                row[k] = float(np.sum(d[use])) / sm - 1.0
        prof.append(row)
    if len(prof) < n_min:
        out['n_cal'] = len(prof)
        return out
    prof = np.array(prof)
    out['n_cal'] = len(prof)
    out['n'] = np.sum(np.isfinite(prof), axis=0)
    with np.errstate(all='ignore'):
        delta = np.nanmedian(np.where(out['n'] >= n_min, prof, np.nan), axis=0)
    out['delta'] = delta
    if np.isfinite(delta).any():
        out['correction'] = RadialCorrection(out['r_pix'], delta, rmax_pix)
    return out
