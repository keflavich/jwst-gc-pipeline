"""Group-aware ``qfit`` for the members of a jointly fitted close group (#1245).

photutils defines ``qfit = sum|resid in own box| / flux_fit`` for each source,
where the residual vector is the member's own 5x5 segment of the group fit.
In a grouped fit the brighter member takes most of the blended flux, so the
fainter member's denominator shrinks while its box still holds the partner's
PSF-mismatch residual.  Its qfit roughly doubles (PK F480M: 0.57 raw against
0.31 for the ungrouped MA fit of the same stars) and the merged-catalogue
vetting, which gates on the member's own-flux qfit, drops it.

``qfit_group = qfit * flux / F_eff`` replaces the member's flux by the flux
that lands in its own fit box, ``F_eff = flux + sum_j S_ij``, where ``S_ij`` is
the light of partner ``j`` (fitted PSF model, clipped at zero) summed over the
fit box of member ``i``.  A singleton has no partners, so
``qfit_group == qfit`` and every threshold keeps its meaning for it.

Limits (opt-in switch, unvalidated on data):

* The discount for the faint member is ``F / (F + S)``.  With a Gaussian PSF of
  FWHM 2.5 px, a partner at or inside 1 FWHM and a flux ratio of 0.10 / 0.15 /
  0.30 / 0.55 / 1.0, the factor is 0.13 / 0.18 / 0.31 / 0.45 / 0.60 (0.59 /
  0.68 / 0.81 / 0.89 / 0.94 at 1.5 FWHM).  Against the flat ``qfit_max`` 0.2 a
  member then passes with an own qfit up to about 1.5 at ratio 0.10 and about
  0.65 at ratio 0.30, so at the default ``min_ratio`` 0.1 a spurious faint
  member next to a bright star (a PSF-ring fit) can be admitted.  Raise
  ``min_ratio`` (0.3 is the first value to try) before trusting a run.
* The extended-emission vetting also feeds ``qfit_group`` to the
  ``qfit_confident`` bypass of ``_emission_keep_nircam``, which keeps a row
  REGARDLESS of its S/N, so a grouped member admitted that way never meets the
  ``local_snr_min`` floor.
* The MIRI hard cap rescales ``flux_fit`` without rescaling ``qfit``, so for a
  capped row ``qfit * flux`` is no longer the residual sum and the row is
  treated as an ordinary member.  The switch is NIRCam-oriented and off by
  default.

The estimator is a heuristic: it assumes the residual sum is unchanged by the
re-normalisation.  ``flux`` is the member's total fitted flux while ``S_ij`` is
the in-box light of the partner, so ``F_eff`` is conservative (low) and
``qfit_group`` errs on the high side.
"""
import numpy as np

__all__ = ['FLAG_MAXITERS', 'compute_group_qfit']

#: photutils ``flags`` bit 8: the LevMar fit ran out of function evaluations.
FLAG_MAXITERS = 8

#: Partners further than this (px) from a member add no light to its box: the
#: model images are rendered on 21x21 stamps (10 px) and the box adds 2 px.
_PARTNER_RADIUS_PIX = 12.0


def _column(table, name, default=np.nan):
    """Float array of ``table[name]`` with masked entries as NaN, or an array
    of ``default`` when the column is absent."""
    if name not in table.colnames:
        return np.full(len(table), default, dtype=float)
    return np.asarray(np.ma.filled(np.ma.asarray(table[name], dtype=float),
                                   np.nan), dtype=float)


def compute_group_qfit(results, psf_model, shape, *, min_ratio=0.1,
                       excluded=None, fit_shape=(5, 5),
                       radius_pix=_PARTNER_RADIUS_PIX):
    """``qfit_group`` for every row of a per-frame ``PSFPhotometry`` result.

    Parameters
    ----------
    results : `~astropy.table.Table`
        The surviving rows, after the post-fit dedup and the near-saturation
        and satstar-artifact drops, so that ``F_eff`` is built from the
        partners that remain in the catalogue.  Needs ``qfit``, ``flux_fit``,
        ``x_fit``, ``y_fit`` and ``group_id``; ``flux_err``, ``flags``,
        ``x_init``, ``y_init`` and ``forced_refit`` are used when present.
    psf_model : model
        Anything with ``evaluate(x, y, flux, x_0, y_0)``, as the fit used.
    shape : tuple of int
        ``(ny, nx)`` of the fitted frame; fit-box pixels outside it carry no
        light, as in the fit.
    min_ratio : float, optional
        A member whose flux is below ``min_ratio`` times the brightest member
        of its group keeps its own ``qfit``.  The faintest members of a group
        are the ones most likely to be a PSF-ring fit.  0 disables the guard.
    excluded : bool array of shape ``shape``, optional
        Pixels the fit did not use (masked or bad); their partner light is not
        counted.
    fit_shape : (int, int), optional
        The fit box, ``(5, 5)`` in production.
    radius_pix : float, optional
        Partners further than this from the member are ignored.

    Returns
    -------
    ndarray of float, shape (len(results),)
        ``qfit_group``.  It equals ``qfit`` for a singleton, for a member below
        the flux-ratio guard, for a row with flag bit 8 or a non-finite
        ``flux_err``, for an overshoot-refit row (its ``qfit`` belongs to the
        free fit, not to the refit flux) and wherever the inputs are not
        finite.  For a non-negative ``qfit`` (always the case for a positive
        flux, since qfit is a sum of absolute residuals over the flux) it never
        exceeds ``qfit``; a negative ``qfit`` would move toward zero, and the
        vetting takes the minimum with ``qfit`` to guard against it.
    """
    n = len(results)
    qfit = _column(results, 'qfit')
    out = qfit.copy()
    needed = ('qfit', 'flux_fit', 'x_fit', 'y_fit', 'group_id')
    if n < 2 or any(c not in results.colnames for c in needed):
        return out

    flux = _column(results, 'flux_fit')
    xfit = _column(results, 'x_fit')
    yfit = _column(results, 'y_fit')
    # photutils centres the box on the (rounded) seed position
    xbox = np.where(np.isfinite(_column(results, 'x_init')),
                    _column(results, 'x_init'), xfit)
    ybox = np.where(np.isfinite(_column(results, 'y_init')),
                    _column(results, 'y_init'), yfit)
    ferr = _column(results, 'flux_err', default=1.0)
    flags = np.nan_to_num(_column(results, 'flags', default=0.0)).astype(np.int64)
    gid = np.ma.filled(np.ma.asarray(results['group_id']), -1).astype(np.int64)
    if 'forced_refit' in results.colnames:
        refit = np.ma.filled(np.ma.asarray(results['forced_refit'], dtype=bool),
                             False)
    else:
        refit = np.zeros(n, dtype=bool)

    ny, nx = int(shape[0]), int(shape[1])
    hy, hx = int(fit_shape[0]) // 2, int(fit_shape[1]) // 2
    dy, dx = np.mgrid[-hy:hy + 1, -hx:hx + 1].astype(float)
    usable = np.isfinite(flux) & (flux > 0) & np.isfinite(xfit) & np.isfinite(yfit)

    order = np.argsort(gid, kind='stable')
    bounds = np.flatnonzero(np.diff(gid[order])) + 1
    for members in np.split(order, bounds):
        # a masked group_id (filled with -1) names no group
        if members.size < 2 or gid[members[0]] < 0:
            continue
        live = members[usable[members]]
        if live.size < 2:
            continue
        fmax = float(flux[live].max())
        for i in live:
            if (not np.isfinite(qfit[i]) or not np.isfinite(ferr[i])
                    or (flags[i] & FLAG_MAXITERS) or refit[i]
                    or flux[i] < min_ratio * fmax):
                continue
            ix, iy = int(round(xbox[i])), int(round(ybox[i]))
            xx, yy = ix + dx, iy + dy
            inside = (xx >= 0) & (xx < nx) & (yy >= 0) & (yy < ny)
            if not inside.any():
                continue
            if excluded is not None:
                inside = inside & ~np.asarray(
                    excluded[np.clip(yy, 0, ny - 1).astype(int),
                             np.clip(xx, 0, nx - 1).astype(int)], dtype=bool)
            if not inside.any():
                continue
            partner = 0.0
            for j in live:
                if j == i or np.hypot(xfit[j] - xfit[i],
                                      yfit[j] - yfit[i]) > radius_pix:
                    continue
                stamp = np.asarray(
                    psf_model.evaluate(xx, yy, float(flux[j]), xfit[j], yfit[j]),
                    dtype=float)
                partner += float(np.clip(stamp[inside], 0.0, None).sum())
            if partner > 0:
                out[i] = qfit[i] * flux[i] / (flux[i] + partner)
    return out
