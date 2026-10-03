"""Saturated-star flux with a free per-exposure scattered halo (issue #1013).

The LW halo of a saturated star changes from exposure to exposure by ~10% rms
of its local intensity (up to -21% for the 10678 obs-061 target), coherently
over r ~ 15-100 px, while its diffraction spikes do not change (#993, #1013).
A masked-core PSF fit gets its amplitude from exactly those wings, so the
fitted flux inherits the halo change one-for-one.

This module fits, instead of ``F * P + B``::

    m(x) = F * P(x) + F * sum_k c_k b_k(r) * Pbar(r) + B

* ``P`` is the PSF model on the fit pixels (unit total flux);
* ``Pbar(r)`` is its azimuthal median at radius r: the smooth halo of the
  model, under and between the spikes alike;
* ``b_k(r)`` are piecewise-linear hats in log r, one per knot of ``knots``,
  all zero outside ``knots[0] <= r <= knots[-1]``, so the halo freedom is
  confined to that range;
* ``B`` is a constant background.

The halo terms can rescale the smooth halo freely, so inside the halo range
the flux ``F`` is set only by the part of ``P`` the halo terms cannot mimic:
the diffraction spikes (``P - Pbar`` along them; pupil diffraction, the
part of the LW PSF that does not change between exposures, #993), plus any
pixels outside the halo range.

The problem is linear in ``(F, F c_k, B)`` and is solved by weighted least
squares, with optional iterative clipping of field-star pixels.  With
``knots=None`` (no halo terms) the same code gives the standard ``F P + B``
fit, so the two can be compared on identical pixels and weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_KNOTS = (15.0, 35.0, 80.0, 150.0)
DEFAULT_SPIKE_CONTRAST = 2.0


@dataclass
class HaloModeFit:
    """Result of :func:`fit_flux_with_halo_modes`."""
    flux: float
    flux_err: float
    bkg: float
    halo: np.ndarray = field(default_factory=lambda: np.zeros(0))
    halo_err: np.ndarray = field(default_factory=lambda: np.zeros(0))
    knots: tuple = ()
    chi2_dof: float = np.nan
    npix: int = 0
    nspike: int = 0
    model: np.ndarray | None = None


def azimuthal_median(psf, r, rbin=1.0):
    """The PSF's azimuthal median in ``rbin``-px annuli, evaluated at every
    pixel: its smooth (spike-free) halo.  ``psf`` and ``r`` share a shape."""
    psf = np.asarray(psf, float)
    k = np.floor(np.asarray(r, float) / rbin).astype(int)
    k = np.clip(k, 0, None)
    flat_k, flat_p = k.ravel(), psf.ravel()
    order = np.argsort(flat_k, kind='stable')
    ks, ps = flat_k[order], flat_p[order]
    bounds = np.flatnonzero(np.diff(ks)) + 1
    med = np.empty(k.max() + 1)
    med[:] = np.nan
    for grp_k, grp_p in zip(np.split(ks, bounds), np.split(ps, bounds)):
        med[grp_k[0]] = np.median(grp_p)
    return med[k]


def spike_mask_from_psf(psf, r, contrast=DEFAULT_SPIKE_CONTRAST, rbin=1.0):
    """Pixels where the PSF exceeds ``contrast`` x its azimuthal median at that
    radius: the diffraction spikes (and nothing else, for a JWST PSF beyond a
    few px)."""
    return np.asarray(psf, float) > contrast * azimuthal_median(psf, r, rbin=rbin)


def log_hats(r, knots):
    """Piecewise-linear hats in log r on ``knots``: one per interior knot
    plus the two ends, each zero outside [knots[0], knots[-1]].  Returns
    ``(..., len(knots))``.  The hats sum to one inside the range."""
    lr = np.log(np.clip(np.asarray(r, float), 1e-3, None))
    lk = np.log(np.asarray(knots, float))
    out = np.zeros(lr.shape + (len(lk),))
    inside = (lr >= lk[0]) & (lr <= lk[-1])
    for j in range(len(lk)):
        h = np.zeros_like(lr)
        if j > 0:
            m = (lr >= lk[j - 1]) & (lr <= lk[j])
            h[m] = (lr[m] - lk[j - 1]) / (lk[j] - lk[j - 1])
        if j < len(lk) - 1:
            m = (lr >= lk[j]) & (lr <= lk[j + 1])
            h[m] = np.maximum(h[m], (lk[j + 1] - lr[m]) / (lk[j + 1] - lk[j]))
        out[..., j] = np.where(inside, h, 0.0)
    return out


def fit_flux_with_halo_modes(data, err, mask, psf, x0, y0, *, knots=DEFAULT_KNOTS,
                             spike_contrast=DEFAULT_SPIKE_CONTRAST, rmax=None, rbin=1.0,
                             clip=5.0, niter=3, return_model=False):
    """Fit ``F P + F sum_k c_k b_k Pbar + B`` to ``data``.

    Parameters
    ----------
    data, err : 2-D arrays
        Image and 1-sigma errors; non-finite pixels are ignored.
    mask : 2-D bool array
        True = do not use (saturated core and its buffer, bad DQ).
    psf : 2-D array
        PSF model evaluated on the same pixel grid as ``data`` at the star's
        position, normalised to unit total flux.
    x0, y0 : float
        Star position on that grid (for the radii of the halo terms).
    knots : sequence of float or None
        Radii [px] of the log-r hats.  ``None`` or empty: the standard
        ``F P + B`` fit.
    spike_contrast : float
        Threshold of :func:`spike_mask_from_psf`; only used to report how
        many fitted pixels lie on the spikes (``nspike``).
    rmax : float, optional
        Use only pixels with r <= rmax.
    clip, niter : float, int
        Iterative rejection of pixels (field stars) whose normalised residual
        departs from the median by more than ``clip`` x its robust (MAD)
        scatter, floored at 1; ``niter=1`` disables it.

    Returns
    -------
    HaloModeFit
        ``flux`` is F (the flux in the units of ``data`` x pixel, since
        ``psf`` has unit sum); ``halo`` the c_k.
    """
    data = np.asarray(data, float)
    err = np.asarray(err, float)
    psf = np.asarray(psf, float)
    yy, xx = np.indices(data.shape)
    r = np.hypot(xx - x0, yy - y0)
    good = (~np.asarray(mask, bool)) & np.isfinite(data) & np.isfinite(err) & (err > 0) & np.isfinite(psf)
    if rmax is not None:
        good &= r <= rmax
    pbar = azimuthal_median(np.where(np.isfinite(psf), psf, 0.0), r, rbin=rbin)
    spikes = psf > spike_contrast * pbar
    use_halo = knots is not None and len(knots) > 0
    if use_halo:
        hats = log_hats(r, knots)
        cols = [psf] + [pbar * hats[..., k] for k in range(hats.shape[-1])] + [np.ones_like(psf)]
    else:
        cols = [psf, np.ones_like(psf)]
    keep = good.copy()
    for it in range(max(niter, 1)):
        idx = np.flatnonzero(keep)
        w = 1.0 / err.ravel()[idx]
        A = np.stack([c.ravel()[idx] for c in cols], 1)
        # a halo hat with no unmasked off-spike pixel is unconstrained: drop it
        live = np.any(A != 0, axis=0)
        p = np.zeros(A.shape[1])
        cov = np.full((A.shape[1], A.shape[1]), np.nan)
        Aw = A[:, live] * w[:, None]
        sol, *_ = np.linalg.lstsq(Aw, data.ravel()[idx] * w, rcond=None)
        p[live] = sol
        cov_live = np.linalg.pinv(Aw.T @ Aw)
        cov[np.ix_(live, live)] = cov_live
        model = sum(pi * c for pi, c in zip(p, cols))
        res = (data - model) / err
        if it == niter - 1 or clip is None:
            break
        # clip against the residuals' robust scatter, not the formal errors: a
        # model-mismatch chi2/pix of ~8 (real LW halos), or a first pass pulled
        # by bright field stars, would otherwise clip most of the good pixels
        rk = res[keep]
        scale = max(1.0, 1.4826 * float(np.median(np.abs(rk - np.median(rk)))))
        newkeep = good & (np.abs(res - np.median(rk)) < clip * scale)
        if newkeep.sum() == keep.sum():
            break
        keep = newkeep
    F = p[0]
    dof = max(int(keep.sum()) - int(live.sum()), 1)
    chi2 = float(np.sum(res[keep] ** 2) / dof)
    if use_halo:
        g, ge = p[1:-1], np.sqrt(np.diag(cov))[1:-1]
        halo = g / F if F != 0 else np.full_like(g, np.nan)
        halo_err = ge / abs(F) if F != 0 else np.full_like(g, np.nan)
    else:
        halo, halo_err = np.zeros(0), np.zeros(0)
    return HaloModeFit(flux=float(F), flux_err=float(np.sqrt(cov[0, 0])), bkg=float(p[-1]),
                       halo=halo, halo_err=halo_err, knots=tuple(knots) if use_halo else (),
                       chi2_dof=chi2, npix=int(keep.sum()), nspike=int((keep & spikes).sum()),
                       model=model if return_model else None)


def satstar_halo_knots(r_core, rmax, *, r_min=15.0, ratio=2.0):
    """Knots for a saturated star whose masked core reaches ``r_core`` px,
    fitted out to ``rmax`` px: geometric (factor ~``ratio``) from
    ``max(r_min, r_core)`` to ``rmax``.  ``None`` when that range is too
    short (< 1.3x) to hold a halo term."""
    lo = max(float(r_min), float(r_core) if np.isfinite(r_core) else 0.0)
    if not (np.isfinite(rmax) and rmax >= 1.3 * lo):
        return None
    n = max(2, int(np.ceil(np.log(rmax / lo) / np.log(ratio))) + 1)
    return tuple(float(k) for k in np.geomspace(lo, rmax, n))


def halo_mode_flux_ratio(data, err, mask, psf, x0, y0, *, r_core, rmax, **kw):
    """The flux ratio ``F_halo / F_standard`` on identical pixels and weights,
    with knots from :func:`satstar_halo_knots`.

    Being a ratio, it does not depend on the PSF's normalisation or on how
    the caller's own fit treated the background, so it can be transferred to
    a flux measured elsewhere (``flux_fit * ratio``).  Returns
    ``(ratio, standard_fit, halo_fit)``; ``ratio`` is NaN when no halo range
    fits inside ``rmax`` or either fit is not positive.
    """
    knots = satstar_halo_knots(r_core, rmax)
    if knots is None:
        return np.nan, None, None
    s = fit_flux_with_halo_modes(data, err, mask, psf, x0, y0, knots=None, rmax=rmax, **kw)
    h = fit_flux_with_halo_modes(data, err, mask, psf, x0, y0, knots=knots, rmax=rmax, **kw)
    if not (s.flux > 0 and h.flux > 0):
        return np.nan, s, h
    return h.flux / s.flux, s, h


def satstar_halo_mode_ratios(cutout, err, mask, psf_model, positions, *, r_core, rmax):
    """:func:`halo_mode_flux_ratio` for each fitted star of one satstar cutout.

    ``psf_model`` is the photutils PSF model the production fit used (a
    ``GriddedPSFModel``), evaluated at unit flux at each ``(x, y)`` of
    ``positions`` (cutout pixel coordinates).  Returns an array of ratios
    (NaN where no halo range fits inside ``rmax``).
    """
    yy, xx = np.mgrid[0:cutout.shape[0], 0:cutout.shape[1]]
    out = np.full(len(positions), np.nan)
    for k, (x, y) in enumerate(positions):
        psf = psf_model.evaluate(xx, yy, 1.0, float(x), float(y))
        out[k], _, _ = halo_mode_flux_ratio(cutout, err, mask, psf, float(x), float(y),
                                            r_core=r_core, rmax=rmax)
    return out


# Wide-radius variant (#1013 follow-up, scripts/analysis/satstar_halo_modes/
# wide_radius_production.py).  At the production box (r <= 40.5 px) the halo
# modes do not help; fitted to r <= 200 px on the FULL frame, for cores of
# >= 300 saturated px, they remove the NRCBLONG x = 250-550 column deficit
# (in band / outside: production flux_fit 0.884 +- 0.008, wide halo-mode flux
# 0.982 +- 0.006, 54 stars; obs 041/061/075/086 of 10678).  Below ~300 px the
# 200-px stamp is dominated by crowding and the dither scatter gets WORSE.
WIDE_RMAX = 200.0
WIDE_AREA_MIN = 300
WIDE_DQ_BAD = 3          # DO_NOT_USE | SATURATED
WIDE_DQ_DILATE = 3


def satstar_halo_mode_ratios_wide(data, err, dq, psf_model, positions, sat_area, *,
                                  rmax=WIDE_RMAX, area_min=WIDE_AREA_MIN):
    """:func:`halo_mode_flux_ratio` on a ``2 rmax + 1`` stamp of the full frame.

    ``data``, ``err``, ``dq`` are full-frame arrays and ``positions`` the
    fitted DETECTOR ``(x, y)`` of each star; ``sat_area`` its saturated-core
    area [px] (one value or one per star).  Pixels with ``dq & 3`` (dilated by
    3 px), non-finite data or non-positive error are not used; other stars are
    left to the fit's clipping.  ``r_core = sqrt(sat_area / pi) + 3``.

    Returns the ratio per star; NaN for ``sat_area < area_min`` and wherever
    :func:`halo_mode_flux_ratio` gives NaN.
    """
    from scipy import ndimage
    area = np.broadcast_to(np.asarray(sat_area, float), (len(positions),))
    ny, nx = data.shape
    h = int(np.ceil(rmax)) + 2
    out = np.full(len(positions), np.nan)
    for k, (x, y) in enumerate(positions):
        if not (np.isfinite(x) and np.isfinite(y) and area[k] >= area_min):
            continue
        ix, iy = int(round(float(x))), int(round(float(y)))
        y1, y2, x1, x2 = max(iy - h, 0), min(iy + h + 1, ny), max(ix - h, 0), min(ix + h + 1, nx)
        if y2 <= y1 or x2 <= x1:
            continue
        d = np.asarray(data[y1:y2, x1:x2], float)
        e = np.asarray(err[y1:y2, x1:x2], float)
        bad = ndimage.binary_dilation((np.asarray(dq[y1:y2, x1:x2]).astype(np.int64) & WIDE_DQ_BAD) > 0,
                                      iterations=WIDE_DQ_DILATE)
        bad |= ~np.isfinite(d) | ~(e > 0)
        yy, xx = np.mgrid[y1:y2, x1:x2]
        psf = psf_model.evaluate(xx, yy, 1.0, float(x), float(y))
        out[k], _, _ = halo_mode_flux_ratio(np.where(bad, 0.0, d), np.where(bad, 1.0, e), bad, psf,
                                            float(x) - x1, float(y) - y1,
                                            r_core=float(np.sqrt(area[k] / np.pi)) + 3.0, rmax=rmax)
    return out
