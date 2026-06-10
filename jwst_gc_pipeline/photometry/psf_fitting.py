"""PSF-fitting wrappers, accelerators, and post-fit deduplication.

Factored out of ``crowdsource_catalogs_long.py`` (2026-06-09 restructure) so the
new ``cataloging.py`` manual-iteration path and the legacy path share one
implementation.  Contents:

* photutils 2.x <-> 3.x kwarg-compatibility shims and constants
* ``_make_psfphotometry`` / ``_make_iterative_psfphotometry`` constructors
* ``CachingGriddedPSFModel`` (stamp-memoizing GriddedPSFModel)
* ``forced_psf_photometry`` (closed-form linear flux at fixed position)
* ``_make_model_image`` (version-safe make_model_image)
* ``_dedup_close_sources`` (greedy spatial deduplication)

The legacy parallel-chunked workers (``_parallel_psfphotometry`` etc.),
``_FakePhot``, and ``WrappedPSFModel`` remain in ``crowdsource_catalogs_long.py``
for now (tangled with crowdsource / the iterative chunked logic); they import
the helpers here.
"""
import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree

import photutils as _photutils
from packaging.version import Version as _PUVersion
from photutils.psf import PSFPhotometry, IterativePSFPhotometry, GriddedPSFModel

# ---------------------------------------------------------------------------
# Photutils 2.x <-> 3.x compatibility shims.
#
# In photutils 3.0 several keyword arguments were renamed:
#   * PSFPhotometry / IterativePSFPhotometry: ``localbkg_estimator`` ->
#     ``local_bkg_estimator``
#   * make_model_image / make_residual_image: ``include_localbkg`` ->
#     ``include_local_bkg``
# Both old names are retained as deprecation-warning aliases until 4.0,
# but the deprecation warnings flood the per-frame logs.  These small
# wrappers detect the installed version once and dispatch to the right
# kwarg, so the same source works on 2.3.0 and 3.0+.
# ---------------------------------------------------------------------------
_PHOTUTILS_GE_3 = _PUVersion(_photutils.__version__.split('+')[0]) >= _PUVersion('3.0.0.dev')
_LOCAL_BKG_KW = 'local_bkg_estimator' if _PHOTUTILS_GE_3 else 'localbkg_estimator'
_INCLUDE_LOCAL_BKG_KW = 'include_local_bkg' if _PHOTUTILS_GE_3 else 'include_localbkg'


def _make_psfphotometry(*, localbkg_estimator, **kwargs):
    """Construct a PSFPhotometry using whichever local-bkg kwarg the
    installed photutils accepts."""
    return PSFPhotometry(**{_LOCAL_BKG_KW: localbkg_estimator}, **kwargs)


def _make_iterative_psfphotometry(*, localbkg_estimator, **kwargs):
    """Construct an IterativePSFPhotometry using whichever local-bkg
    kwarg the installed photutils accepts."""
    return IterativePSFPhotometry(**{_LOCAL_BKG_KW: localbkg_estimator},
                                  **kwargs)


# ---------------------------------------------------------------------------
# Forced photometry & caching PSF model (experimental low-level fits)
#
# 1.  ``CachingGriddedPSFModel`` -- subclass of ``GriddedPSFModel`` that
#     memoizes the rendered PSF stamp on (x_0, y_0, stamp pixel grid).
#     Drop-in replacement for ``GriddedPSFModel`` in any photutils call;
#     accelerates LM fits where the inner finite-difference Jacobian
#     perturbs ONLY the flux parameter (a frequent late-iteration case)
#     or where (x_0, y_0) are pinned via ``xy_bounds``.
#
# 2.  ``forced_psf_photometry()`` -- standalone closed-form linear flux
#     solve at fixed (x_0, y_0).  Bypasses photutils' LM entirely:
#         f = sum(d*p*w) / sum(p^2*w),   sigma_f = 1 / sqrt(sum(p^2*w))
#     ~80x faster than the full LM path on iter2/iter3 chains where
#     positions come from a union seed catalog.
# ---------------------------------------------------------------------------
class CachingGriddedPSFModel(GriddedPSFModel):
    """``GriddedPSFModel`` that memoizes the rendered stamp on
    (x_0, y_0, x-grid, y-grid).

    Cache hits when LM finite-differences ONLY flux; misses on x_0/y_0
    perturbation.  For a 3-param FD Jacobian this saves the 1 of 4 evals
    that re-renders the PSF with unchanged position but perturbed flux,
    plus all subsequent rendering of the same stamp during flux-only
    sub-iterations.  For position-pinned fits (xy_bounds=(0,0) or
    forced phot), every evaluate after the first is a cache hit.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._psf_cache_key = None
        self._psf_cache_unit_stamp = None
        # diagnostic counters; reset via ``reset_cache_stats``
        self._psf_cache_hits = 0
        self._psf_cache_misses = 0

    def reset_cache_stats(self):
        self._psf_cache_hits = 0
        self._psf_cache_misses = 0

    @property
    def cache_stats(self):
        return {'hits': self._psf_cache_hits,
                'misses': self._psf_cache_misses}

    def evaluate(self, x, y, flux, x_0, y_0):
        # x/y are the eval pixel grids; use shape + first/last corner +
        # source position as the cache key.  Cheap to compute, robust
        # against int<->float drift because we cast to float.
        try:
            x_first = float(x.flat[0])
            x_last = float(x.flat[-1])
            y_first = float(y.flat[0])
            y_last = float(y.flat[-1])
        except (TypeError, AttributeError):
            # Scalar inputs (unusual) bypass the cache.
            return super().evaluate(x, y, flux, x_0, y_0)
        key = (float(x_0), float(y_0), x.shape,
               x_first, x_last, y_first, y_last)
        if self._psf_cache_key == key:
            self._psf_cache_hits += 1
            unit_stamp = self._psf_cache_unit_stamp
        else:
            self._psf_cache_misses += 1
            unit_stamp = super().evaluate(x, y, 1.0, x_0, y_0)
            self._psf_cache_key = key
            self._psf_cache_unit_stamp = unit_stamp
        return float(flux) * unit_stamp


def forced_psf_photometry(image, psf_model, init_params, *,
                          error=None, mask=None,
                          fit_shape=(5, 5),
                          aperture_radius=4,
                          nonnegative=False):
    """Closed-form linear flux solve at the (fixed) positions in
    ``init_params``.  Bypasses photutils LM entirely.

    When and why it is used
    -----------------------
    Used in ONE place: the *overshoot-refit* step of the manual-iteration
    pipeline (``cataloging._manual_phot_pass``, with
    ``--manual-overshoot-action=refit``, the default).  After each single-pass
    BASIC ``PSFPhotometry`` fit, an overshoot check
    (``_filter_or_flag_model_overshoot``) flags any source whose rendered model
    PEAK exceeds ``--manual-overshoot-ratio`` x the local DATA peak.  That is
    physically impossible for a single positive PSF and is the signature of the
    free-position fit having let the centroid WALK OFF the star and settle at an
    inflated-flux minimum (the failure mode that motivated replacing
    ``IterativePSFPhotometry``).  Each flagged source is then re-fit here.

    Two properties make this the right tool for that refit:

    * **Position pinned at the trusted SEED** (``x_init``/``y_init``), not the
      drifted ``x_fit``.  The whole failure is the centroid wandering, so the
      flux must be re-measured where the star actually is (the seed), not where
      the bad fit drifted to.  With position fixed, the model is LINEAR in flux,
      so the weighted least-squares flux has the exact closed form
      ``f = sum(d*p*w) / sum(p^2*w)``  (``sigma_f = 1/sqrt(sum(p^2*w))``) -- no
      iteration, no chance of re-drifting.
    * **~80x faster** than spinning up a fixed-position LM fit per source, which
      matters because the seed-driven passes can flag dozens of sources/frame.

    The flux is returned at the seed position; downstream the caller snaps
    ``x_fit``/``y_fit`` back to the seed and clears the overshoot flag.

    ``nonnegative`` clamps the solved flux at 0.  The unconstrained
    closed-form ``f = sum(d*p*w) / sum(p^2*w)`` goes NEGATIVE wherever the
    local data under the PSF is net-negative (neighbour / satstar / smoothed-
    background over-subtraction).  A single strictly-positive PSF cannot have
    negative flux, so for the photometry refit path this must be clamped:
    a negative solve means "no positive source consistent with the (over-
    subtracted) data here" -> flux 0 (then dropped by the non-positive ban).
    Left ``False`` by default so other callers keep the signed estimator.

    Parameters
    ----------
    image : 2D ndarray
    psf_model : GriddedPSFModel (or compatible) supporting
        ``evaluate(x_grid, y_grid, flux=1.0, x_0, y_0)``.
    init_params : Table with columns ``x_init``, ``y_init``
        (``flux_init`` optional and ignored).
    error : 2D ndarray of 1-sigma uncertainties (same shape as image).
        ``None`` falls back to unit weights.
    mask : bool 2D ndarray; ``True`` = bad pixel, excluded.
    fit_shape : (ny, nx) stamp around each source (must be odd-ish; 5x5
        matches production).
    aperture_radius : kept for signature compatibility; not used here.

    Returns
    -------
    Table with columns ``x_init``, ``y_init``, ``x_fit``, ``y_fit``
    (== x_init/y_init), ``flux_fit``, ``flux_err``, ``n_pixels_fit``.
    """
    ny_fit, nx_fit = int(fit_shape[0]), int(fit_shape[1])
    half_y, half_x = ny_fit // 2, nx_fit // 2
    img_ny, img_nx = image.shape

    x_init = np.asarray(init_params['x_init'], dtype=float)
    y_init = np.asarray(init_params['y_init'], dtype=float)
    n = len(x_init)
    flux_fit = np.full(n, np.nan, dtype=np.float64)
    flux_err = np.full(n, np.nan, dtype=np.float64)
    npix_fit = np.zeros(n, dtype=np.int32)

    # Build the (ny_fit x nx_fit) pixel grid offsets once.
    dy, dx = np.mgrid[-half_y:half_y + 1, -half_x:half_x + 1].astype(float)

    use_err = error is not None
    if mask is None:
        mask_arr = np.zeros(image.shape, dtype=bool)
    else:
        mask_arr = np.asarray(mask, dtype=bool)

    for i in range(n):
        x0 = x_init[i]
        y0 = y_init[i]
        ix = int(round(x0))
        iy = int(round(y0))
        y_lo, y_hi = iy - half_y, iy + half_y + 1
        x_lo, x_hi = ix - half_x, ix + half_x + 1
        if (y_lo < 0 or x_lo < 0 or y_hi > img_ny or x_hi > img_nx):
            continue

        data = image[y_lo:y_hi, x_lo:x_hi]
        mstamp = mask_arr[y_lo:y_hi, x_lo:x_hi]
        if mstamp.all():
            continue

        # Evaluate PSF on the absolute-coord grid centred on the source.
        xx = ix + dx
        yy = iy + dy
        psf_stamp = np.asarray(
            psf_model.evaluate(xx, yy, 1.0, x0, y0),
            dtype=np.float64,
        )

        if use_err:
            estamp = error[y_lo:y_hi, x_lo:x_hi]
            w = np.where((estamp > 0) & np.isfinite(estamp),
                         1.0 / estamp**2, 0.0)
        else:
            w = np.ones_like(data, dtype=np.float64)
        w = np.where(mstamp, 0.0, w)

        # Linear LS solution: data ~ flux * psf
        sxx = float((psf_stamp * psf_stamp * w).sum())
        if sxx <= 0:
            continue
        sxy = float((data * psf_stamp * w).sum())
        _f = sxy / sxx
        flux_fit[i] = max(_f, 0.0) if nonnegative else _f
        flux_err[i] = 1.0 / np.sqrt(sxx)
        npix_fit[i] = int((w > 0).sum())

    out = Table()
    out['x_init'] = x_init
    out['y_init'] = y_init
    out['flux_init'] = (np.asarray(init_params['flux_init'], dtype=float)
                        if 'flux_init' in init_params.colnames
                        else np.full(n, np.nan))
    out['x_fit'] = x_init
    out['y_fit'] = y_init
    out['flux_fit'] = flux_fit
    out['flux_err'] = flux_err
    out['n_pixels_fit'] = npix_fit
    return out


def _make_model_image(phot_obj, shape, *, psf_shape=None, include_local_bkg=False):
    """Call ``phot_obj.make_model_image`` with the version-appropriate
    include-local-bkg kwarg."""
    return phot_obj.make_model_image(
        shape, psf_shape=psf_shape,
        **{_INCLUDE_LOCAL_BKG_KW: include_local_bkg})


def _dedup_close_sources(xy, flux, min_sep_pix, quality=None,
                         flux_agreement_frac=0.10, is_saturated=None):
    """
    Greedy spatial deduplication of sources closer than ``min_sep_pix``.

    For each cluster of entries within ``min_sep_pix`` of each other, keep one
    representative:
      * If the fluxes of the cluster agree within ``flux_agreement_frac``
        (i.e. (fmax - fmin) / fmax <= flux_agreement_frac), the entries are
        treated as the same source and the brightest is kept.
      * Otherwise the fluxes disagree, which usually means the fits converged
        to genuinely different solutions (contamination, split binary, bad
        init). In that case:
            - if ``quality`` is provided (smaller = better, e.g. qfit), keep
              the entry with the best (smallest) quality;
            - if no quality is provided, fall back to the brightest.

    Parameters
    ----------
    xy : array, shape (N, 2)
        Positions (in pixels) to cluster on.
    flux : array, shape (N,)
        Flux estimate for each entry.  Must be finite for any entry to be
        considered; non-finite entries are kept as-is (not clustered).
    min_sep_pix : float
        Minimum allowed separation between kept entries, in pixels.
    quality : array, shape (N,), optional
        Per-entry quality score where smaller is better (e.g. photutils qfit).
        Used only for breaking ties when cluster fluxes disagree.
    flux_agreement_frac : float, optional
        Relative flux tolerance below which cluster members are treated as
        duplicates of the same source.

    Returns
    -------
    keep : ndarray of bool, shape (N,)
        True for entries to retain.
    n_disagree : int
        Number of clusters where fluxes disagreed beyond ``flux_agreement_frac``
        (i.e. cases where duplicate removal may have dropped a distinct
        neighbour rather than a pure duplicate).
    """
    n = xy.shape[0]
    keep = np.ones(n, dtype=bool)
    if n < 2:
        return keep, 0

    # Build KD-tree over all entries with finite positions.  Eligibility
    # for SEEDING a cluster (i.e. potentially claiming nearby duplicates)
    # is finite position AND (finite flux OR is_saturated=True).  But any
    # entry with finite position can be REMOVED by a seed — so NaN-flux
    # non-saturated duplicates near a saturated seed get removed.
    finite_xy   = np.all(np.isfinite(xy), axis=1)
    finite_flux = np.isfinite(flux)
    if is_saturated is None:
        is_sat_arr = np.zeros(n, dtype=bool)
    else:
        is_sat_arr = np.asarray(is_saturated, dtype=bool)

    finite_idx = np.where(finite_xy)[0]
    if finite_idx.size < 2:
        return keep, 0
    xy_f = xy[finite_idx]
    kd_full = cKDTree(xy_f)
    # Map global index -> position in finite_idx (or -1 if not finite_xy)
    pos_in_finite = np.full(n, -1, dtype=np.int64)
    pos_in_finite[finite_idx] = np.arange(finite_idx.size)

    # Only seed clusters from entries that are finite-flux OR is_saturated.
    eligible    = finite_xy & (finite_flux | is_sat_arr)
    if np.sum(eligible) < 1:
        return keep, 0
    elig_idx = np.where(eligible)[0]
    xy_e     = xy[elig_idx]
    flux_e   = flux[elig_idx]
    sat_e    = is_sat_arr[elig_idx]

    # Sort eligible entries so the cluster-seeding priority is:
    #   1) is_saturated=True (these win the dedup tiebreak regardless of
    #      flux — they're the real bright stars)
    #   2) finite-flux (brightest first)
    #   3) NaN-flux non-saturated (last; only kept if no neighbour exists)
    # Build a composite priority key: is_sat first, then flux desc.
    _flux_for_sort = np.where(np.isfinite(flux_e), flux_e, -np.inf)
    local_sort_order = np.lexsort((-_flux_for_sort, ~sat_e))

    n_disagree = 0
    for li in local_sort_order:
        i = elig_idx[li]
        if not keep[i]:
            continue
        # Query the FULL KD-tree so the seed can claim any nearby
        # entry (including NaN-flux non-saturated duplicates).
        local_neighbours = kd_full.query_ball_point(xy_e[li], min_sep_pix)
        neighbours = [int(finite_idx[lj]) for lj in local_neighbours
                      if int(finite_idx[lj]) != i and keep[int(finite_idx[lj])]]
        if not neighbours:
            continue

        # Collect the full cluster (seed + neighbours).
        cluster = [i] + neighbours
        cluster_sat = np.asarray([is_sat_arr[k] for k in cluster], dtype=bool)
        # is_saturated entries win the cluster regardless of flux —
        # they encode known positions of bright stars and we must not
        # let nearby duplicate non-saturated NaN-flux seeds outvote them.
        if np.any(cluster_sat):
            sat_members = [k for k, s in zip(cluster, cluster_sat) if s]
            # Among saturated entries, prefer the one with finite flux
            # (brightest), or just the first if none are finite.
            sat_flux = np.asarray([flux[k] for k in sat_members], dtype=float)
            if np.any(np.isfinite(sat_flux)):
                winner = sat_members[int(np.nanargmax(sat_flux))]
            else:
                winner = sat_members[0]
        else:
            cluster_flux = np.asarray([flux[k] for k in cluster], dtype=float)
            # Restrict flux-agreement computation to finite entries.
            finite_clu = np.isfinite(cluster_flux)
            if not np.any(finite_clu):
                # No finite flux in cluster — fall back to keeping seed.
                winner = i
            else:
                fvals = cluster_flux[finite_clu]
                fmin = float(fvals.min())
                fmax = float(fvals.max())
                if fmax <= 0 or (fmax - fmin) / fmax <= flux_agreement_frac:
                    # Fluxes agree: treat as same source, keep brightest (= seed i).
                    winner = i
                else:
                    # Fluxes disagree: fits converged together but to different
                    # solutions.  Prefer best-quality entry if we have quality info.
                    n_disagree += 1
                    if quality is not None:
                        cluster_q = np.asarray([quality[k] for k in cluster], dtype=float)
                        # Smaller quality is better; non-finite treated as worst.
                        cluster_q = np.where(np.isfinite(cluster_q), cluster_q, np.inf)
                        winner = cluster[int(np.argmin(cluster_q))]
                    else:
                        winner = i  # brightest

        for k in cluster:
            if k != winner:
                keep[k] = False

    return keep, n_disagree
