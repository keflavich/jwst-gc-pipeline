"""Manual-iteration PSF photometry (Hosek-style), the start of the photometry
restructure (2026-06-09).

Motivation
----------
The legacy ``crowdsource_catalogs_long.do_photometry_step`` uses photutils
``IterativePSFPhotometry`` for iter2+, which is numerically unstable for
isolated bright stars: with free position + LevMar + internal maxiters
re-detection the fit can walk the centroid and settle on a ~2x inflated-flux
minimum whose *model peak exceeds the data peak* (physically impossible for a
single positive PSF; qfit does NOT catch it).  See
``NOTES_star_vs_extended_emission.md`` (C/D case) and ``PSFPhotometryPlan2026-06-09.md``.

This module replaces the iterative fitter with **manual iterations of
single-pass** ``PSFPhotometry``, making each daofind -> fit -> residual ->
reseed cycle explicit, and adds a physical model/data-peak overshoot QC.

Recipe (cutout-first):
  per-frame iter1 -> iter2 (no merge between)
  -> merge + extended-emission filter + smoothed-bg map
  -> iter3 (bg-subtracted) -> merge + recompute bg
  -> iter4 -> cross-band stringent seed (multifilter) -> iter5

Built alongside the legacy path (gated by ``--manual-iterations``); the old
code is untouched and will be deprecated only if this proves out.

This module is BASIC-only (single-pass ``PSFPhotometry``); all output filenames
carry iteration labels ``m1..m5`` so products never collide with the legacy
``iter2/iter3/iter4`` ones.
"""
import numpy as np
from astropy.table import Table

from photutils.background import LocalBackground
from photutils.detection import DAOStarFinder
from astropy.modeling.fitting import LevMarLSQFitter

from jwst_gc_pipeline.photometry.naming import _iteration_token, _bgsub_token
from jwst_gc_pipeline.photometry.psf_fitting import (
    _make_psfphotometry, _make_model_image, _dedup_close_sources,
    forced_psf_photometry,
)

# Seed-building, local-stats, and satstar-filter helpers still live in the
# legacy module (factoring of seeds.py / image_stats.py / satstar_filters.py is
# deferred to a follow-up; these are imported here so cataloging.py uses the one
# canonical implementation).  Importing the legacy module at load time is fine:
# it does NOT import this module at top level (only lazily in its dispatch
# branch), so there is no import cycle.
from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import (
    SeededFinder,
    _combine_seed_and_satstars,
    _augment_seed_catalog_with_detections_sky,
    compute_local_noise_map,
    annotate_and_filter_by_local_snr,
    _filter_near_saturation,
    _filter_satstar_artifacts,
)


# ---------------------------------------------------------------------------
# Physical overshoot QC: the fix qfit misses
# ---------------------------------------------------------------------------
def _filter_or_flag_model_overshoot(phot_obj, modsky, data, *,
                                    ratio=1.2, action='flag', label='',
                                    box_half=1):
    """Flag / drop fits whose rendered model peak exceeds the local
    background-subtracted DATA peak by more than ``ratio``.

    A single positive PSF cannot peak above the data; ``model_peak >> data_peak``
    means the LevMar fit walked to a spurious (typically ~2x) inflated-flux
    minimum -- the C/D inflation that qfit does not catch (qfit is *lowered* by
    the inflated denominator).  See NOTES_star_vs_extended_emission.md.

    Always annotates ``phot_obj.results`` with ``model_data_peak_ratio`` (float)
    and ``model_overshoot`` (bool).  When ``action == 'drop'`` the overshooting
    rows are removed from ``phot_obj.results`` as well.  ``action == 'flag'``
    (default) only annotates -- never silently deletes on a heuristic.  For
    ``action == 'refit'`` the caller (_manual_phot_pass) handles the forced-photometry
    re-fit, since that needs the PSF model and error array.

    Returns the boolean overshoot mask (aligned to ``phot_obj.results``).
    """
    res = phot_obj.results
    n = len(res)
    if n == 0:
        return np.zeros(0, dtype=bool)
    xf = np.asarray(res['x_fit'], dtype=float)
    yf = np.asarray(res['y_fit'], dtype=float)
    lbk = (np.asarray(res['local_bkg'], dtype=float)
           if 'local_bkg' in res.colnames else np.zeros(n, dtype=float))
    lbk = np.where(np.isfinite(lbk), lbk, 0.0)

    ny, nx = data.shape
    ratio_arr = np.full(n, np.nan, dtype=float)
    model_pk = np.full(n, np.nan, dtype=float)
    data_pk = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if not (np.isfinite(xf[i]) and np.isfinite(yf[i])):
            continue
        ix = int(round(xf[i]))
        iy = int(round(yf[i]))
        ylo, yhi = max(0, iy - box_half), min(ny, iy + box_half + 1)
        xlo, xhi = max(0, ix - box_half), min(nx, ix + box_half + 1)
        if ylo >= yhi or xlo >= xhi:
            continue
        mbox = modsky[ylo:yhi, xlo:xhi]
        dbox = data[ylo:yhi, xlo:xhi]
        if not np.any(np.isfinite(mbox)) or not np.any(np.isfinite(dbox)):
            continue
        mpk = float(np.nanmax(mbox))
        dpk = float(np.nanmax(dbox)) - lbk[i]
        model_pk[i] = mpk
        data_pk[i] = dpk
        if dpk > 0:
            ratio_arr[i] = mpk / dpk

    over = np.isfinite(ratio_arr) & (ratio_arr > ratio)
    res['model_data_peak_ratio'] = ratio_arr
    res['model_overshoot'] = over
    n_over = int(np.sum(over))
    if n_over > 0:
        print(f"[{label}] model/data-peak overshoot (>{ratio:g}x): {n_over}/{n} "
              f"fits flagged (max ratio {np.nanmax(ratio_arr):.2f}); action={action}",
              flush=True)
    if action == 'drop' and n_over > 0:
        phot_obj.results = res[~over]
        phot_obj.__dict__.pop('_model_image_params', None)
    return over


# ---------------------------------------------------------------------------
# Atomic single-pass fit + post-fit cleanup
# ---------------------------------------------------------------------------
def _manual_phot_pass(*, data, mask, err, bad, dao_psf_model, init_params,
                      aperture_radius_pix, localbkg_inner, localbkg_outer,
                      grouper, options, dq, satstar_model_subtracted,
                      label, xy_bounds_pix=None,
                      overshoot_ratio=1.2, overshoot_action='flag'):
    """One single-pass ``PSFPhotometry`` fit seeded by ``init_params``, followed
    by the standard post-fit cleanup chain (mirrors the legacy BASIC block):

      dedup (1.0 px, qfit tiebreak) -> near-saturation filter ->
      satstar-wing rejection -> model/data-peak overshoot QC ->
      render model image.

    Returns ``(result_table, modsky, phot_obj)``.  Does no file I/O.
    """
    extra = {}
    if xy_bounds_pix is not None:
        extra['xy_bounds'] = (xy_bounds_pix, xy_bounds_pix)
    phot = _make_psfphotometry(
        finder=None,
        localbkg_estimator=LocalBackground(localbkg_inner, localbkg_outer),
        grouper=grouper if getattr(options, 'group', False) else None,
        psf_model=dao_psf_model,
        fitter=LevMarLSQFitter(),
        fit_shape=(5, 5),
        aperture_radius=aperture_radius_pix,
        progress_bar=False,
        **extra,
    )
    result = phot(data, mask=mask, init_params=init_params,
                  error=np.where(bad, 1e10, err))

    # --- post-fit dedup (1.0 px, qfit tiebreak) ---
    xfit = np.asarray(result['x_fit'], dtype=float)
    yfit = np.asarray(result['y_fit'], dtype=float)
    flux = np.asarray(result['flux_fit'], dtype=float)
    qfit = (np.asarray(result['qfit'], dtype=float)
            if 'qfit' in result.colnames else None)
    keep, n_disagree = _dedup_close_sources(
        xy=np.column_stack([xfit, yfit]), flux=flux,
        min_sep_pix=1.0, quality=qfit)
    if int(len(keep) - np.sum(keep)) > 0:
        phot.results = phot.results[keep]
        if (phot.init_params is not None
                and len(phot.init_params) == len(keep)):
            phot.init_params = phot.init_params[keep]
        phot.__dict__.pop('_model_image_params', None)
        print(f"[{label}] post-fit dedup: {len(keep)} -> {int(np.sum(keep))} "
              f"({n_disagree} flux-disagree clusters)", flush=True)

    # --- near-saturation + satstar-wing rejection ---
    _filter_near_saturation(phot, dq, max_sat_dist_pix=1.0, label=label)
    _filter_satstar_artifacts(phot, satstar_model_subtracted, err,
                              sig_K=float(options.satstar_artifact_sigK),
                              ratio_cut=float(options.satstar_artifact_ratio),
                              label=label)

    # --- render model, then physical overshoot QC ---
    modsky = _make_model_image(phot, data.shape, psf_shape=(21, 21),
                               include_local_bkg=False)
    over = _filter_or_flag_model_overshoot(
        phot, modsky, data, ratio=overshoot_ratio,
        action=('flag' if overshoot_action == 'refit' else overshoot_action),
        label=label)

    if overshoot_action == 'refit' and np.any(over):
        # Re-fit the overshooting sources as forced photometry at their SEED
        # position (flux free, position PINNED to the trusted seed) -- the
        # cap-free fix for the free-position inflation.  Crucially we pin at the
        # SEED position (x_init/y_init), NOT the drifted x_fit: the whole failure
        # mode is the centroid walking off the star, so refitting at the drifted
        # position would give a wrong (off-source) flux.  Replace rows in-place
        # and snap x_fit/y_fit back to the seed.
        res = phot.results
        xseed = (np.asarray(res['x_init'], dtype=float) if 'x_init' in res.colnames
                 else np.asarray(res['x_fit'], dtype=float))
        yseed = (np.asarray(res['y_init'], dtype=float) if 'y_init' in res.colnames
                 else np.asarray(res['y_fit'], dtype=float))
        seed = Table()
        seed['x_init'] = xseed[over]
        seed['y_init'] = yseed[over]
        forced = forced_psf_photometry(data, dao_psf_model, seed,
                                       error=np.where(bad, 1e10, err),
                                       mask=mask, fit_shape=(5, 5))
        if 'forced_refit' not in res.colnames:
            res['forced_refit'] = np.zeros(len(res), dtype=bool)
        idx = np.where(over)[0]
        for j, gi in enumerate(idx):
            res['flux_fit'][gi] = forced['flux_fit'][j]
            res['x_fit'][gi] = xseed[gi]
            res['y_fit'][gi] = yseed[gi]
            if 'flux_err' in res.colnames:
                res['flux_err'][gi] = forced['flux_err'][j]
            # The source is now corrected (forced at the trusted seed position);
            # clear the overshoot flag so downstream vetting does not drop a real
            # star, but record that it was force-refit for diagnostics.
            res['model_overshoot'][gi] = False
            res['forced_refit'][gi] = True
        phot.__dict__.pop('_model_image_params', None)
        print(f"[{label}] refit {len(idx)} overshooting sources as forced "
              f"photometry at the seed position (flux free)", flush=True)
        modsky = _make_model_image(phot, data.shape, psf_shape=(21, 21),
                                   include_local_bkg=False)

    return phot.results, modsky, phot


# ---------------------------------------------------------------------------
# Seed building: daofind(image) + previous catalog + satstars
# ---------------------------------------------------------------------------
def _build_manual_seed(*, detection_image, nan_replaced_data, mask, ww, fwhm_pix,
                       satstar_table, prev_catalog,
                       local_snr_threshold, roundlo, roundhi, sharplo, sharphi,
                       preferred_seed_skycoord_col=None,
                       dedup_min_sep_pix=None, label=''):
    """Build ``seeded_init_params`` for one manual pass.

    daofind(detection_image) [local-noise-map threshold] -> local-S/N filter ->
    combine (prev_catalog + satstars) -> augment with the fresh detections (sky
    matched) -> SeededFinder -> dedup.  ``prev_catalog=None`` for the first pass.

    Mirrors the legacy seed logic but omits the iter3 union-snap / per-filter
    inject blocks (those caused the spurious-seed provenance documented in the
    NOTES); the manual recipe seeds directly from the previous merged catalog.
    """
    if dedup_min_sep_pix is None:
        dedup_min_sep_pix = 0.5 * fwhm_pix

    # daofind on the detection image with a local-noise-map floor threshold
    noise_map = compute_local_noise_map(detection_image, smooth_sigma_pix=3.0)
    finite = np.isfinite(noise_map) & (noise_map > 0)
    if not np.any(finite):
        raise ValueError(f"[{label}] local noise map has no positive finite values")
    threshold = float(np.nanmin(noise_map[finite]))
    daofind = DAOStarFinder(threshold=threshold, fwhm=fwhm_pix,
                            roundlo=roundlo, roundhi=roundhi,
                            sharplo=sharplo, sharphi=sharphi)
    detections = daofind(detection_image, mask=mask)
    if detections is None:
        detections = Table()
    detections, snr_stats = annotate_and_filter_by_local_snr(
        detections, noise_map, snr_threshold=local_snr_threshold)
    print(f"[{label}] daofind: {snr_stats['input_count']} -> "
          f"{snr_stats['kept_count']} after local-S/N>={local_snr_threshold}",
          flush=True)

    # seed base = previous catalog (if any) + satstars
    seed = _combine_seed_and_satstars(prev_catalog, satstar_table)

    # augment with the fresh detections that are NOT already represented
    seed = _augment_seed_catalog_with_detections_sky(
        seed, detections, ww=ww,
        match_radius_pix=max(1.0, 0.5 * fwhm_pix),
        preferred_seed_skycoord_col=preferred_seed_skycoord_col)

    # project to pixel positions + flux_init via SeededFinder
    finstars = SeededFinder(seed, ww=ww,
                            preferred_skycoord_col=preferred_seed_skycoord_col)(
        nan_replaced_data, mask=mask)
    seeded = Table()
    seeded['x_init'] = np.asarray(finstars['x_init'], dtype=float)
    seeded['y_init'] = np.asarray(finstars['y_init'], dtype=float)
    seeded['flux_init'] = np.asarray(finstars['flux_init'], dtype=float)
    if 'is_saturated' in finstars.colnames:
        seeded['is_saturated'] = np.asarray(finstars['is_saturated'], dtype=bool)

    # dedup seeds (brightest / saturated wins within dedup_min_sep_pix)
    if len(seeded) > 1:
        keep, n_dis = _dedup_close_sources(
            xy=np.column_stack([seeded['x_init'], seeded['y_init']]),
            flux=np.asarray(seeded['flux_init'], dtype=float),
            min_sep_pix=dedup_min_sep_pix, quality=None,
            is_saturated=(np.asarray(seeded['is_saturated'], dtype=bool)
                          if 'is_saturated' in seeded.colnames else None))
        n_removed = int(len(keep) - np.sum(keep))
        if n_removed > 0:
            seeded = seeded[keep]
            print(f"[{label}] seed dedup: removed {n_removed} within "
                  f"{dedup_min_sep_pix:.2f} px", flush=True)
    return seeded


# ---------------------------------------------------------------------------
# Merged-catalog extended-emission vetting (step 9)
# ---------------------------------------------------------------------------
def _filter_extended_emission(catalog, data_i2d_image=None, ww_i2d=None, *,
                              qfit_max=0.2, peak_over_bkg=20.0,
                              local_snr_min=5.0, keep_flags=(1,),
                              drop_overshoot=True, label=''):
    """First-pass star-vs-extended-emission vetting of a MERGED catalog.

    Keep a source iff it looks like a confident star:
        (qfit <= qfit_max)
        OR (flags in keep_flags)                  # central-saturation real star
        OR (peak_SB > peak_over_bkg * local_bkg)  # bright real star
    AND (local_snr >= local_snr_min where available)
    AND (not model_overshoot, if that column exists and drop_overshoot).

    ``peak_SB`` needs a pixel value: pass the merged data i2d image + its WCS to
    sample a 3x3-box max at each source; otherwise the peak-SB criterion is
    skipped.  All thresholds are FIRST-PASS (one cutout) and CLI-tunable.

    Returns the filtered Table (does not write; the caller persists the
    ``_vetted`` file).
    """
    t = catalog
    n = len(t)
    if n == 0:
        return t

    qf = (np.asarray(t['qfit'], dtype=float)
          if 'qfit' in t.colnames else np.full(n, np.inf))
    flg = (np.asarray(t['flags'], dtype=float)
           if 'flags' in t.colnames else np.full(n, np.nan))
    lbk = (np.asarray(t['local_bkg'], dtype=float)
           if 'local_bkg' in t.colnames else np.zeros(n))

    # local S/N: prefer an explicit column, else flux/flux_err
    if 'local_snr' in t.colnames:
        snr = np.asarray(t['local_snr'], dtype=float)
    elif 'flux' in t.colnames and 'flux_err' in t.colnames:
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = (np.asarray(t['flux'], dtype=float)
                   / np.asarray(t['flux_err'], dtype=float))
    elif 'flux_fit' in t.colnames and 'flux_err' in t.colnames:
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = (np.asarray(t['flux_fit'], dtype=float)
                   / np.asarray(t['flux_err'], dtype=float))
    else:
        snr = np.full(n, np.inf)

    # peak surface brightness from the data i2d (3x3 box max), if provided
    peaksb = np.full(n, np.nan, dtype=float)
    if data_i2d_image is not None and ww_i2d is not None and 'skycoord' in t.colnames:
        from astropy.coordinates import SkyCoord
        sc = t['skycoord']
        if not isinstance(sc, SkyCoord):
            sc = SkyCoord(sc)
        xx, yy = ww_i2d.world_to_pixel(sc)
        ny, nx = data_i2d_image.shape
        for i in range(n):
            if not (np.isfinite(xx[i]) and np.isfinite(yy[i])):
                continue
            ix, iy = int(round(float(xx[i]))), int(round(float(yy[i])))
            ylo, yhi = max(0, iy - 1), min(ny, iy + 2)
            xlo, xhi = max(0, ix - 1), min(nx, ix + 2)
            if ylo < yhi and xlo < xhi:
                box = data_i2d_image[ylo:yhi, xlo:xhi]
                if np.any(np.isfinite(box)):
                    peaksb[i] = float(np.nanmax(box))

    star_like = (
        (qf <= qfit_max)
        | np.isin(flg, np.asarray(keep_flags, dtype=float))
        | (np.isfinite(peaksb) & (lbk > 0) & (peaksb > peak_over_bkg * lbk))
    )
    keep = star_like & (np.isfinite(snr) & (snr >= local_snr_min) | ~np.isfinite(snr))
    if drop_overshoot and 'model_overshoot' in t.colnames:
        keep = keep & ~np.asarray(t['model_overshoot'], dtype=bool)

    n_keep = int(np.sum(keep))
    print(f"[{label}] extended-emission filter: {n} -> {n_keep} "
          f"(qfit<={qfit_max}, flags in {keep_flags}, peakSB>{peak_over_bkg}x bkg, "
          f"snr>={local_snr_min})", flush=True)
    return t[keep]
