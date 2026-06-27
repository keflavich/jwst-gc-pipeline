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

from jwst_gc_pipeline.photometry.naming import (
    _iteration_token, _bgsub_token,
    residual_to_smoothed_bg_i2d, smoothed_bg_to_detection_i2d, vetted_to_i2dseed)
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
from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as _L
from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import (
    SeededFinder,
    _combine_seed_and_satstars,
    _augment_seed_catalog_with_detections_sky,
    compute_local_noise_map,
    annotate_and_filter_by_local_snr,
    _filter_near_saturation,
    _filter_satstar_artifacts,
)

import os
import types
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from astropy.io import fits
from astropy import wcs
from astropy import units as u
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans, convolve_fft
from astropy.nddata import NDData
from photutils.background import Background2D, MedianBackground, MMMBackground
from photutils.psf import SourceGrouper


# ---------------------------------------------------------------------------
# Physical overshoot QC: the fix qfit misses
# ---------------------------------------------------------------------------
def _filter_or_flag_model_overshoot(phot_obj, modsky, data, *,
                                    ratio=1.2, action='flag', label='',
                                    box_half=1, flag_nonpositive_data=False):
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
        elif flag_nonpositive_data and np.isfinite(mpk) and mpk > 0:
            # dpk<=0 BLIND SPOT (MIRI): near a bright/saturated neighbour the
            # sampled local_bkg can exceed the faint local data peak, so the
            # bkg-subtracted dpk goes <=0 and the source escapes the guard --
            # this is exactly how a group-fit-degenerate member kept a runaway
            # flux (5.94e6 on ~560 data) and drizzled to a -459k mosaic pit.
            # A real positive source must have data above background, so dpk<=0
            # with a positive model peak is spurious.  Use the RAW data peak
            # (floored at 1) as the denominator so a runaway model is still
            # caught while a genuinely faint source (mpk ~ its small peak) is
            # not over-flagged.
            _draw = float(np.nanmax(dbox))
            ratio_arr[i] = mpk / max(_draw, 1.0)

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
                      overshoot_ratio=1.2, overshoot_action='flag',
                      overshoot_cap_target=1.0,
                      miri_dpk_guard=False,
                      satstar_excl_xy=None, satstar_excl_pix=0.0,
                      near_sat_dist_pix=1.0,
                      miri_prominence_snr=0.0, prominence_bg_box=0,
                      prominence_data_i2d=None, prominence_ww_i2d=None,
                      frame_ww=None):
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

    # Empty seed (0 sources): a source-poor frame -- common at long MIRI
    # wavelengths (w51 F2100W at 21um is mostly extended emission, so a single
    # dither's residual detection can be empty).  photutils PSFPhotometry can't
    # be called with zero init_params (its LocalBackground builds a
    # CircularAnnulus from an empty positions array and raises), so short-circuit
    # to an empty per-frame catalog + zero model.  Without this one empty frame
    # aborts the whole filter (run_manual_pipeline treats any frame error as
    # fatal).  Carry init_params' columns plus the standard PSFPhotometry output
    # schema so save_photutils_results writes a valid (0-row) catalog.
    n_seed = 0 if init_params is None else len(init_params)
    if n_seed == 0:
        print(f"[{label}] empty seed (0 sources): skipping PSF fit, emitting "
              f"empty per-frame catalog", flush=True)
        res = Table(init_params, copy=True) if init_params is not None else Table()
        for _c, _dt in (('id', 'i8'), ('group_id', 'i8'), ('group_size', 'i8'),
                        ('local_bkg', 'f8'), ('x_init', 'f8'), ('y_init', 'f8'),
                        ('flux_init', 'f8'), ('x_fit', 'f8'), ('y_fit', 'f8'),
                        ('flux_fit', 'f8'), ('x_err', 'f8'), ('y_err', 'f8'),
                        ('flux_err', 'f8'), ('npixfit', 'i8'), ('qfit', 'f8'),
                        ('cfit', 'f8'), ('flags', 'i8'), ('iter_detected', 'i8')):
            if _c not in res.colnames:
                res[_c] = np.zeros(0, dtype=_dt)
        modsky = np.zeros_like(data, dtype=float)
        return res, modsky, None

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
    _filter_near_saturation(phot, dq, max_sat_dist_pix=near_sat_dist_pix,
                            label=label)
    _filter_satstar_artifacts(phot, satstar_model_subtracted, err,
                              sig_K=float(options.satstar_artifact_sigK),
                              ratio_cut=float(options.satstar_artifact_ratio),
                              label=label)

    # --- MIRI satstar-coincidence drop ---
    # The satstar channel OWNS saturated stars (fit + subtracted before this
    # daophot pass).  A daophot fit landing within satstar_excl_pix of a satstar
    # catalog entry double-counts that star: the satstar model AND the daophot
    # model are both subtracted, gouging a deep (-1e5) over-subtraction pit at
    # the saturated core (verified: satstar 3.3e6 + daophot 3.5e5 -> -141k).
    # The saturated core itself is masked, so any daophot "source" within ~1.5
    # FWHM of a satstar is a wing / centroid-scatter spurious fit, not a real
    # companion (a real companion that close to a bright saturated star is not
    # separable anyway).  Unlike _filter_satstar_artifacts (which only drops
    # fits DIMMER than the satstar wing), this catches the BRIGHT duplicate.
    # MIRI-only; NIRCam unaffected (satstar_excl_pix stays 0).
    if (satstar_excl_xy is not None and satstar_excl_pix > 0
            and phot.results is not None and len(phot.results)):
        _r = phot.results
        _xs = np.asarray(_r['x_fit'], dtype=float)
        _ys = np.asarray(_r['y_fit'], dtype=float)
        _sx = np.asarray(satstar_excl_xy[:, 0], dtype=float)
        _sy = np.asarray(satstar_excl_xy[:, 1], dtype=float)
        _drop = np.zeros(len(_r), dtype=bool)
        _r2 = float(satstar_excl_pix) ** 2
        for _i in range(len(_r)):
            if not (np.isfinite(_xs[_i]) and np.isfinite(_ys[_i])):
                continue
            _d2 = (_sx - _xs[_i]) ** 2 + (_sy - _ys[_i]) ** 2
            if _d2.size and float(np.min(_d2)) < _r2:
                _drop[_i] = True
        _nd = int(_drop.sum())
        if _nd:
            phot.results = _r[~_drop]
            if (phot.init_params is not None
                    and len(phot.init_params) == len(_r)):
                phot.init_params = phot.init_params[~_drop]
            phot.__dict__.pop('_model_image_params', None)
            print(f"[{label}] satstar-coincidence drop: {len(_r)} -> "
                  f"{len(_r) - _nd} ({_nd} daophot fits within "
                  f"{satstar_excl_pix:.1f}px of a satstar)", flush=True)

    # --- MIRI prominence reject: kill false sources on extended emission ---
    # The detection local-S/N is peak/high-pass-noise; the high-pass removes
    # the smooth MIRI emission from the NOISE but the peak still sits on the
    # emission pedestal, so a "source" whose flux ~= the local emission still
    # scores high S/N and is detected+fit, then over-subtracted into a negative
    # residual (user 2026-06-15: most modelled stars are false, flux at star ~=
    # flux in background).  Require the fitted source's DATA peak to rise at
    # least ``miri_prominence_snr`` * local_noise ABOVE the local median
    # background (the emission) -- i.e. a real PROMINENCE, not just flux above
    # pixel noise.  In low-emission regions bg~=0 so this reduces to ordinary
    # peak/noise and real stars pass freely; it only bites on bright emission.
    # MIRI-only; NIRCam leaves miri_prominence_snr=0 (off).
    if (miri_prominence_snr > 0
            and phot.results is not None and len(phot.results)):
        # Prominence = (data peak in core) - (median in an annulus), over the
        # annulus MAD.  The annulus (4-10 px) measures the LOCAL emission +
        # its fluctuation; a real point source's core sits far above it
        # (validated: hand-selected real F770W stars median ~126, 10th pct ~40),
        # while a false source on flat emission has core ~= annulus (~1-3).  Do
        # NOT use a high-pass noise map for the denominator: the source inflates
        # its own local high-pass variance, crushing real stars' S/N too.
        _r = phot.results
        # Measure prominence on the DEEP data i2d (frame-INVARIANT) when it is
        # supplied: a per-frame measurement is noisy (a real star can dip below
        # K in one exposure -> lost; a false bump can fluctuate above K in one
        # exposure -> the merge re-unions it back).  One value per source on the
        # deep coadd both keeps every real star and removes every false one.
        # Falls back to the per-frame ``data`` if the i2d is unavailable.
        if (prominence_data_i2d is not None and frame_ww is not None
                and prominence_ww_i2d is not None):
            _img = prominence_data_i2d
            try:
                _sk = frame_ww.pixel_to_world(np.asarray(_r['x_fit'], dtype=float),
                                              np.asarray(_r['y_fit'], dtype=float))
                _mx, _my = prominence_ww_i2d.world_to_pixel(_sk)
                _xs = np.asarray(_mx, dtype=float); _ys = np.asarray(_my, dtype=float)
            except Exception:
                _img = data
                _xs = np.asarray(_r['x_fit'], dtype=float)
                _ys = np.asarray(_r['y_fit'], dtype=float)
        else:
            _img = data
            _xs = np.asarray(_r['x_fit'], dtype=float)
            _ys = np.asarray(_r['y_fit'], dtype=float)
        _ny, _nx = _img.shape
        _H = 10
        _yo, _xo = np.mgrid[-_H:_H + 1, -_H:_H + 1]
        _rr = np.hypot(_xo, _yo)
        _cm = _rr < 1.5
        _am = (_rr >= 4) & (_rr <= _H)
        _dropp = np.zeros(len(_r), dtype=bool)
        for _i in range(len(_r)):
            if not (np.isfinite(_xs[_i]) and np.isfinite(_ys[_i])):
                continue
            _ix = int(round(_xs[_i])); _iy = int(round(_ys[_i]))
            if not (_H <= _ix < _nx - _H and _H <= _iy < _ny - _H):
                continue
            _st = _img[_iy - _H:_iy + _H + 1, _ix - _H:_ix + _H + 1]
            _core = np.nanmax(_st[_cm])
            _ann = _st[_am]
            _bg = np.nanmedian(_ann)
            _mad = 1.4826 * np.nanmedian(np.abs(_ann - _bg))
            if not (np.isfinite(_core) and np.isfinite(_bg) and _mad > 0):
                continue
            if (_core - _bg) / _mad < miri_prominence_snr:
                _dropp[_i] = True
        _ndp = int(_dropp.sum())
        if _ndp:
            phot.results = _r[~_dropp]
            if (phot.init_params is not None
                    and len(phot.init_params) == len(_r)):
                phot.init_params = phot.init_params[~_dropp]
            phot.__dict__.pop('_model_image_params', None)
            print(f"[{label}] prominence reject: {len(_r)} -> {len(_r) - _ndp} "
                  f"({_ndp} fits with core < {miri_prominence_snr:g}*annulus-MAD "
                  f"above local bg; false emission sources)", flush=True)

    # --- render model, then physical overshoot QC ---
    modsky = _make_model_image(phot, data.shape, psf_shape=(21, 21),
                               include_local_bkg=False)
    over = _filter_or_flag_model_overshoot(
        phot, modsky, data, ratio=overshoot_ratio,
        action=('flag' if overshoot_action == 'refit' else overshoot_action),
        label=label, flag_nonpositive_data=miri_dpk_guard)

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
                                       mask=mask, fit_shape=(5, 5),
                                       nonnegative=True)
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

    # --- HARD CAP residual overshoot (2026-06-23, MIRI) ---
    # Even after the seed refit, a mildly EXTENDED source (low concentration --
    # a bump on bright emission, or a source broader than the PSF) keeps
    # model_peak > data_peak and leaves a NEGATIVE POCKMARK in the residual
    # (2526 cloud-c filament: daophot model 449 on data 279 -> -170).  A single
    # positive PSF physically cannot peak above the data, so rescale any
    # still-overshooting fit's flux down until its model peak == the local data
    # peak.  Corrects the amplitude without dropping the source (it stays in the
    # catalog at its true, data-limited brightness).  MIRI-gated via the same
    # dpk-guard flag so NIRCam is unchanged.
    if (miri_dpk_guard and overshoot_cap_target and overshoot_cap_target > 0
            and len(phot.results)):
        modsky = _make_model_image(phot, data.shape, psf_shape=(21, 21),
                                   include_local_bkg=False)
        _capov = _filter_or_flag_model_overshoot(
            phot, modsky, data, ratio=overshoot_ratio, action='flag',
            label=f'{label}:cap', flag_nonpositive_data=miri_dpk_guard)
        if np.any(_capov):
            _res = phot.results
            _r = np.asarray(_res['model_data_peak_ratio'], dtype=float)
            _fl = np.asarray(_res['flux_fit'], dtype=float)
            _scale = np.where(_capov & np.isfinite(_r) & (_r > 0),
                              overshoot_cap_target / _r, 1.0)
            _res['flux_fit'] = _fl * _scale
            phot.__dict__.pop('_model_image_params', None)
            modsky = _make_model_image(phot, data.shape, psf_shape=(21, 21),
                                       include_local_bkg=False)
            print(f"[{label}] capped {int(np.sum(_capov))} overshooting fits to "
                  f"{overshoot_cap_target:g}x the local data peak", flush=True)

    # --- ban non-positive-flux (negative-peak) sources ---
    # A PSF is strictly positive, so flux_fit <= 0 is a negative-peak model: it
    # ADDS flux to the residual rather than subtracting a star, and (because the
    # next iteration seeds from this catalog) it breeds more spurious negatives
    # at over-subtracted spots -- the pillar_with_satstar iter6 explosion.
    # Negatives are PRODUCED by the closed-form forced refit (above): its
    # unconstrained f = sum(d*p*w)/sum(p^2*w) goes negative wherever the local
    # data is net-negative (neighbour / satstar / smoothed-bg over-subtraction,
    # hence the m5+ onset).  We now clamp that refit (nonnegative=True), and the
    # bounded LevMar group fit respects flux.min=0; this ban is the final safety
    # net (e.g. an exact-0 clamp lands here) so non-positives never enter the
    # catalog or the seed.
    res = phot.results
    fpos = np.asarray(res['flux_fit'], dtype=float) > 0
    n_neg = int(len(fpos) - np.sum(fpos))
    if n_neg > 0:
        phot.results = res[fpos]
        if (phot.init_params is not None
                and len(phot.init_params) == len(fpos)):
            phot.init_params = phot.init_params[fpos]
        phot.__dict__.pop('_model_image_params', None)
        modsky = _make_model_image(phot, data.shape, psf_shape=(21, 21),
                                   include_local_bkg=False)
        print(f"[{label}] dropped {n_neg} non-positive-flux (negative-peak) "
              f"sources", flush=True)

    return phot.results, modsky, phot


# ---------------------------------------------------------------------------
# Seed building: daofind(image) + previous catalog + satstars
# ---------------------------------------------------------------------------
def _subset_seed_to_frame(seed_table, ww, data_shape, fwhm_pix,
                          preferred_skycoord_col=None, margin_fwhm=3.0,
                          label=''):
    """Restrict a (possibly full-mosaic) seed catalog to the sources that land on
    THIS frame, plus a PSF-wing margin, BEFORE the per-frame seed build.

    The previous-phase merged seed catalog spans the WHOLE field (millions of
    rows for dense SW filters).  ``SeededFinder`` clips it to the frame at the
    end anyway, but only AFTER ``_combine_seed_and_satstars`` +
    ``_augment_seed_catalog_with_detections_sky`` + the sky->pixel projection
    carry the entire catalog (all columns) through memory -- which is the
    dominant per-frame, per-worker RAM (tens of GB for F115W/F200W: the m5 OOM
    at MaxRSS ~825 GB across 32 workers, 2026-06-23).  Subsetting to the frame
    footprint here cuts that ~100x.  Off-frame seeds cannot produce in-frame
    init params, so the fit is unchanged (the margin keeps stars whose wings
    fall onto the frame).  Row count is the cost driver, not column count, so we
    keep all columns (a few-k rows x ~100 cols is a few MB).
    """
    if seed_table is None or len(seed_table) == 0:
        return seed_table
    # Debug/benchmark escape hatch (default OFF -> subset active): set
    # _SEED_SUBSET_DISABLE=1 to keep the full-field seed (used to A/B the
    # memory savings).  Not for production.
    if os.environ.get('_SEED_SUBSET_DISABLE'):
        return seed_table
    resolved = _L._resolve_seed_skycoords(seed_table, ww=ww,
                                          preferred_skycoord_col=preferred_skycoord_col)
    x, y = ww.world_to_pixel(resolved['skycoord'])
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    ny, nx = int(data_shape[0]), int(data_shape[1])
    margin = max(50.0, float(margin_fwhm) * float(fwhm_pix))
    keep = (np.isfinite(x) & np.isfinite(y)
            & (x > -margin) & (x < nx + margin)
            & (y > -margin) & (y < ny + margin))
    n0 = len(resolved); nk = int(np.sum(keep))
    if nk < n0:
        print(f"[{label}] seed footprint subset: {n0} -> {nk} sources on frame "
              f"({nx}x{ny}, +{margin:.0f}px margin)", flush=True)
    return resolved[keep]


def _build_manual_seed(*, detection_image, nan_replaced_data, mask, ww, fwhm_pix,
                       satstar_table, prev_catalog,
                       local_snr_threshold, roundlo, roundhi, sharplo, sharphi,
                       preferred_seed_skycoord_col=None,
                       dedup_min_sep_pix=None, label='', apply_snr_filter=True,
                       struct_x=0.0, struct_y=0.0, coarse_bg_box=0):
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

    # Clip the previous (full-mosaic) seed catalog to this frame's footprint
    # before combine/augment/projection -- the dominant per-frame, per-worker
    # memory for dense SW fields (see _subset_seed_to_frame).
    if prev_catalog is not None and len(prev_catalog):
        prev_catalog = _subset_seed_to_frame(
            prev_catalog, ww, np.asarray(nan_replaced_data).shape, fwhm_pix,
            preferred_skycoord_col=preferred_seed_skycoord_col, label=label)

    # MIRI early-phase coarse background subtraction (AG 2026-06-13): the F770W
    # pedestal is huge (image min ~200 MJy/sr) but the bright stars are HIGH
    # contrast above the *local* background -- they just sit on the pedestal.
    # Subtract a coarse median (box >~ 5x FWHM so stars are not blurred into it)
    # from the DETECTION image only (the fit still uses the unaltered frame), so
    # daofind sees the stars.  Validated on the F770W hand-selected catalog:
    # 51px coarse-sub recovers 24/34 (the unsaturated ones; saturated -> satstar).
    # m5/m6 do NOT use this -- they detect on the trustworthy star-subtracted
    # background-subtracted residual instead (coarse_bg_box=0 there).
    if coarse_bg_box and coarse_bg_box > 0:
        from scipy.ndimage import median_filter as _medfilt
        _good = np.isfinite(detection_image)
        _fill = np.where(_good, detection_image, np.nanmedian(detection_image[_good]))
        detection_image = detection_image - _medfilt(_fill, size=int(coarse_bg_box))
        print(f"[{label}] coarse-bg detection: subtracted {int(coarse_bg_box)}px median",
              flush=True)

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
    if apply_snr_filter:
        detections, snr_stats = annotate_and_filter_by_local_snr(
            detections, noise_map, snr_threshold=local_snr_threshold)
        print(f"[{label}] daofind: {snr_stats['input_count']} -> "
              f"{snr_stats['kept_count']} after local-S/N>={local_snr_threshold}",
              flush=True)
    else:
        # First (unseeded) pass: do not apply the local-S/N cut.  On raw data the
        # local-noise map is high everywhere (source variance dominates), so the
        # cut would drop every real star (legacy iter1 uses daofind's own
        # threshold and does not post-filter on local S/N).  The fit + dedup +
        # overshoot QC handle any spurious detections; iter2 refines on the
        # residual where the local-S/N cut is meaningful.
        detections, _ = annotate_and_filter_by_local_snr(
            detections, noise_map, snr_threshold=0.0)
        print(f"[{label}] daofind: {len(detections)} detections "
              f"(no local-S/N cut on first pass)", flush=True)

    # structure-noise prune on the per-frame daofind detections (AG method):
    # this is the dominant junk source -- without it the per-frame daofind
    # injects emission bumps into the merged catalog at every phase.  Prune on
    # the detection image (raw/residual/bg-sub per phase) with the per-phase
    # (struct_x, struct_y) schedule.  err=None -> mad_std noise floor.
    if len(detections) and (struct_x or struct_y):
        xdt, ydt = _L._best_available_xy(detections)
        skeep = _structure_noise_keep(detection_image, None, xdt, ydt,
                                      struct_x=struct_x, struct_y=struct_y)
        n_b = len(detections)
        detections = detections[skeep]
        print(f"[{label}] per-frame struct-noise prune (x={struct_x},y={struct_y}): "
              f"{n_b} -> {len(detections)}", flush=True)

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
def _emission_keep_nircam(star_like, snr, local_snr_min, qfit_confident=None):
    """NIRCam star-vs-emission keep-mask.

    Keep a star-like source unless its local S/N is measurable and below the
    floor.  (Unmeasurable S/N -> kept; the star_like test already gated it.)

    ``qfit_confident`` (qfit <= qfit_max) sources are kept REGARDLESS of S/N.
    The formal S/N = flux / flux_err is unreliable for a source fit in a GROUP:
    a near-degenerate joint normal matrix inflates flux_err by 100-1000x (e.g.
    sickle F480M close pairs: flux 6200 / flux_err 8076 -> S/N 0.8, and 4402 /
    144220 -> S/N 0.0) while the FLUX and qfit (0.016) stay excellent.  Without
    this exemption the S/N floor silently DROPS a perfectly-fit bright star from
    the vetted catalog -> it is removed from the next phase's seed AND from the
    vetted residual mosaic, so it reappears as a strong unsubtracted source and
    is never re-fit.  An extended-emission bump has a BAD qfit (poor PSF match),
    so qfit<=qfit_max cannot re-admit emission; the model-overshoot drop still
    applies downstream.
    """
    snr_ok = (np.isfinite(snr) & (snr >= local_snr_min)) | ~np.isfinite(snr)
    keep = star_like & snr_ok
    if qfit_confident is not None:
        keep = keep | np.asarray(qfit_confident, dtype=bool)
    return keep


def _emission_keep_miri(prominence, min_prominence):
    """MIRI star-vs-emission keep-mask: deep-i2d prominence is the SOLE cut.

    A source must rise far enough above the local emission on THIS obs's data
    i2d.  NaN prominence (off-i2d / other-obs footprint, or unmeasurable near an
    edge) is dropped -- it will be vetted by the obs that owns its footprint.
    """
    return np.isfinite(prominence) & (prominence >= min_prominence)


def _filter_extended_emission(catalog, data_i2d_image=None, ww_i2d=None, *,
                              qfit_max=0.2, peak_over_bkg=20.0,
                              min_prominence=0.0,
                              local_snr_min=5.0, keep_flags=(1,),
                              snr_high_keep=20.0, qfit_high_keep_max=0.4,
                              drop_overshoot=True, struct_x=0.0, struct_y=0.0,
                              label=''):
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

    # peak surface brightness (3x3 box max) AND annulus-MAD PROMINENCE from the
    # data i2d, if provided.  Prominence = (core peak r<1.5) - (median in a
    # 4-10px annulus) over the annulus MAD: it measures whether the "star" rises
    # above the LOCAL EMISSION, which is the single robust discriminator on the
    # DEEP coadd (validated: hand-selected real F770W stars median ~126, 10th
    # pct ~40; false emission sources ~1-3).  This is the MERGED-catalog cut --
    # measured ONCE at each final source position, so it is immune to the
    # per-frame fit-position scatter that let false sources survive the
    # per-frame cut and get re-unioned by the merge.
    peaksb = np.full(n, np.nan, dtype=float)
    prominence = np.full(n, np.nan, dtype=float)
    if data_i2d_image is not None and ww_i2d is not None and 'skycoord' in t.colnames:
        from astropy.coordinates import SkyCoord
        sc = t['skycoord']
        if not isinstance(sc, SkyCoord):
            sc = SkyCoord(sc)
        xx, yy = ww_i2d.world_to_pixel(sc)
        ny, nx = data_i2d_image.shape
        _H = 10
        _yo, _xo = np.mgrid[-_H:_H + 1, -_H:_H + 1]
        _rr = np.hypot(_xo, _yo)
        _cm = _rr < 1.5
        _am = (_rr >= 4) & (_rr <= _H)
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
            if (min_prominence > 0 and _H <= ix < nx - _H
                    and _H <= iy < ny - _H):
                st = data_i2d_image[iy - _H:iy + _H + 1, ix - _H:ix + _H + 1]
                core = np.nanmax(st[_cm])
                ann = st[_am]
                bg = np.nanmedian(ann)
                mad = 1.4826 * np.nanmedian(np.abs(ann - bg))
                if np.isfinite(core) and np.isfinite(bg) and mad > 0:
                    prominence[i] = (core - bg) / mad

    # BRIGHT-ISOLATED keep (Mechanism 2): a real bright star whose qfit sits just
    # above qfit_max (e.g. sickle F480M star3: flux 6381, S/N 201, qfit 0.282,
    # peakSB only 16.8x bkg -> failed BOTH qfit and peakSB) was vetted out and
    # left strong in the residual.  A high-S/N source that is fit as a SINGLETON
    # (group_size==1, so flux_err -- hence S/N -- is NOT inflated by joint-fit
    # covariance degeneracy) with a still-PSF-like qfit (< qfit_high_keep_max) is
    # a confident star.  Extended-emission knots have BAD qfit (>~0.5, poor PSF
    # match), so this cannot re-admit emission; the group_size==1 gate keeps the
    # S/N trustworthy (a degenerate group inflates flux_err -> low fake S/N).
    gsz = (np.asarray(t['group_size'], dtype=float)
           if 'group_size' in t.colnames else np.ones(n))
    bright_isolated = (
        np.isfinite(snr) & (snr >= snr_high_keep)
        & np.isfinite(qf) & (qf < qfit_high_keep_max)
        & (gsz <= 1)
    )
    star_like = (
        (qf <= qfit_max)
        | np.isin(flg, np.asarray(keep_flags, dtype=float))
        | (np.isfinite(peaksb) & (lbk > 0) & (peaksb > peak_over_bkg * lbk))
        | bright_isolated
    )
    # MIRI and NIRCam use DIFFERENT keep-logics (see _emission_keep_miri /
    # _emission_keep_nircam).  min_prominence>0 selects the MIRI path.
    if min_prominence > 0:
        # MIRI: the deep-i2d PROMINENCE is the clean discriminator (real F770W
        # stars >=40, false emission <5), so use it ALONE.  The NIRCam-tuned
        # star_like (qfit/peakSB) + snr cuts REJECT bright real MIRI stars that
        # sit on extended emission (peakSB/local_bkg < 20 because the emission
        # raises local_bkg; qfit > 0.2 for bright/saturated) -- they dropped 10
        # of 36 hand-selected real stars.  Keep = prominent enough on THIS obs's
        # data_i2d AND not a runaway-overshoot model.  OFF-I2D SOURCES ARE
        # DROPPED: this vetting runs per-obs against one obs's data_i2d, so an
        # off-i2d source belongs to a different obs's footprint and will be
        # vetted (and kept) by that obs's own run -- keeping it here would carry
        # it through UNVETTED (that was the 74%-false o001-footprint leak when a
        # single all-obs vetting saw only o002's i2d).  The per-obs vetted
        # catalogs are then vstack-combined downstream into the un-tokened
        # all-obs catalog, each footprint cleaned by its own data_i2d.
        keep = _emission_keep_miri(prominence, min_prominence)
        n_prom = int(np.sum(np.isfinite(prominence) & (prominence < min_prominence)))
        n_off = int(np.sum(~np.isfinite(prominence)))
        print(f"[{label}] MIRI prominence gate: kept prominence>={min_prominence:g} on "
              f"this obs i2d (dropped {n_prom} false + {n_off} off-i2d/other-obs); "
              f"star_like/snr BYPASSED; per-obs vetted -> combined downstream", flush=True)
    else:
        # qfit-confident sources are kept regardless of the formal flux/flux_err
        # S/N (inflated to ~0 by group-fit covariance degeneracy for close pairs
        # -- see _emission_keep_nircam): a well-fit star must never be dropped
        # from the vetted catalog/residual on a broken uncertainty.
        keep = _emission_keep_nircam(star_like, snr, local_snr_min,
                                     qfit_confident=(qf <= qfit_max))

    # overshoot drop is shared by both instrument paths (was duplicated).
    if drop_overshoot and 'model_overshoot' in t.colnames:
        keep = keep & ~np.asarray(t['model_overshoot'], dtype=bool)

    # structure-noise prune on the MERGED catalog (AG 2026-06-13): rejects
    # emission-bump sources that the m12 round (which has no prune) carries
    # forward.  Requires the data i2d to measure peak vs local structure noise.
    n_struct = 0
    if ((struct_x or struct_y) and data_i2d_image is not None
            and ww_i2d is not None and 'skycoord' in t.colnames):
        skeep = _structure_noise_keep(data_i2d_image, None,
                                      np.asarray(xx), np.asarray(yy),
                                      struct_x=struct_x, struct_y=struct_y)
        n_struct = int(np.sum(keep & ~skeep))
        keep = keep & skeep

    n_keep = int(np.sum(keep))
    print(f"[{label}] extended-emission filter: {n} -> {n_keep} "
          f"(qfit<={qfit_max}, flags in {keep_flags}, peakSB>{peak_over_bkg}x bkg, "
          f"snr>={local_snr_min}, struct-prune dropped {n_struct} @x={struct_x},y={struct_y})",
          flush=True)
    return t[keep]


# ---------------------------------------------------------------------------
# Per-frame frame setup (shared load / bg / PSF / mask / satstar)
# ---------------------------------------------------------------------------
# Per-filter SATURATED data-floor (MJy/sr) for --saturation-data-floor auto mode.
# JUMP/persistence artifacts get mis-tagged SATURATED on UNsaturated sources, and
# masking those drops seeded real stars from every per-frame fit (W51 F480M).
# Only mask a SATURATED pixel when its data exceeds the per-filter floor; genuine
# saturated cores exceed it and stay masked / owned by the satstar channel.
# Empirical from W51 per-frame crf SATURATED-pixel data distributions (p99 of the
# real-saturation plateau).  Filter not listed (incl. all MIRI) -> 0 = mask all
# SATURATED (original behaviour).  Override per-run with --saturation-data-floor.
_NIRCAM_SAT_DATA_FLOOR = {
    'f140m': 5000., 'f162m': 5000., 'f182m': 4000., 'f187n': 8000., 'f210m': 4000.,
    'f335m': 2500., 'f360m': 2500., 'f405n': 5000., 'f410m': 2500., 'f480m': 5000.,
}


def _resolve_each_suffix(options, filtername):
    """Per-filter input per-exposure-crf suffix.

    ``--each-suffix-overrides=F187N:destreak_o007_crf,F210M:destreak_o007_crf``
    lets specific filters read a DIFFERENT per-exposure crf than the global
    ``--each-suffix``.  Sickle: SW F187N/F210M want ``destreak`` (1/f streaks are
    worse than the destreak artifacts), LW F335M/F470N/F480M want ``align``
    (destreak corrupts the bright extended emission).  Falls back to
    ``options.each_suffix`` for any filter not listed.
    """
    default = options.each_suffix
    raw = getattr(options, 'each_suffix_overrides', None)
    if not raw:
        return default
    for pair in str(raw).split(','):
        pair = pair.strip()
        if ':' not in pair:
            continue
        key, val = pair.split(':', 1)
        if key.strip().upper() == str(filtername).upper():
            return val.strip()
    return default


def _load_ramp_group0(crf_path):
    """Load the ramp first read (group-0) for a per-frame crf/cal file from its
    sibling Detector1 ``_ramp.fits`` (same detector pixel grid, no reprojection).

    The crf is named ``jw..._{detector}_<suffix>_crf.fits`` (suffix e.g.
    ``align_o007`` / ``destreak_o007``); the ramp is ``jw..._{detector}_ramp.fits``.
    Returns the 2-D first-read array (DN) or None if no ramp is found / not
    NIRCam.  Used by the ZEROFRAME saturated-rim recovery (#2)."""
    import re
    ramp = re.sub(r'(_nrc[ab](?:long|[1-4]))_.*\.fits$', r'\1_ramp.fits',
                  str(crf_path))
    if ramp == str(crf_path) or not os.path.exists(ramp):
        return None
    try:
        sci = fits.getdata(ramp, extname='SCI')
    except (OSError, KeyError, ValueError):
        return None
    if sci is None:
        return None
    if sci.ndim == 4:      # (nints, ngroups, ny, nx)
        g0 = sci[0, 0]
    elif sci.ndim == 3:    # (ngroups, ny, nx)
        g0 = sci[0]
    else:
        g0 = sci
    return np.asarray(g0, dtype=float)


def _prepare_frame_for_photometry(options, filtername, module, field, basepath,
                                  filename, proposal_id, *, exposurenumber,
                                  visit_id, vgroup_id, bg_boxsizes, use_webbpsf,
                                  pupil, resbg_path, satstar_label,
                                  satstar_flux_overrides=None,
                                  satstar_flux_drops=None):
    """Load a frame and produce everything a manual pass needs: scaled
    aperture/annulus, filename tokens, data + error + mask, the (cutout-
    re-origined) PSF grid, the SourceGrouper, and the satstar-subtracted
    ``nan_replaced_data`` + ``satstar_model_subtracted``.

    Reuses the legacy primitives (``_L.*``).  Input reads use the ORIGINAL
    ``basepath`` (PSF cache, satstar psfs); outputs go to ``out_basepath``
    (the cutout tree when ``--cutout-region`` is set), returned in the context.

    Mirrors the setup half of ``do_photometry_step`` but omits the legacy
    iteration-label seed inference and diagnostics; the manual path builds its
    own seeds and (for cutouts) PNG diagnostics are suppressed anyway.
    """
    fwhm_tbl = Table.read(_L.FWHM_TABLE)
    row = fwhm_tbl[fwhm_tbl['Filter'] == filtername]
    fwhm_pix = float(row['PSF FWHM (pixel)'][0])
    aperture_radius_pix = 2.0 * fwhm_pix
    localbkg_inner = max(6, int(round(aperture_radius_pix + 0.5 * fwhm_pix)))
    localbkg_outer = localbkg_inner + max(4, int(round(fwhm_pix)))

    desat = '_unsatstar' if options.desaturated else ''
    bgsub = _bgsub_token(options)
    epsf_ = "_epsf" if options.epsf else ""
    exposure_ = f'_exp{exposurenumber:05d}' if exposurenumber is not None else ''
    visitid_ = f'_visit{int(visit_id):03d}' if visit_id is not None else ''
    vgroupid_, _vgnum = _L.normalize_vgroup_id(vgroup_id)
    blur_ = "_blur" if options.blur else ""
    group = "_group" if options.group else ""

    # cutout prep (rewrites filename to the cropped copy; redirects outputs)
    cutout_label = ''
    out_basepath = basepath
    cx0, cy0 = 0, 0
    if getattr(options, 'cutout_region', ''):
        cutout_label, filename, out_basepath, cx0, cy0 = _L._prepare_cutout_input(
            filename, basepath, filtername, options)
    cutout_active = bool(cutout_label)

    fh, im1, data, wht, err, instrument, telescope, obsdate = _L.load_data(filename)
    inst_token = instrument.lower()
    ww = wcs.WCS(im1[1].header)

    background_map = None
    original_data = data.copy()  # manual residuals are built vs the pristine data

    if options.bgsub:
        bkg = Background2D(data, box_size=bg_boxsizes[filtername.lower()],
                           bkg_estimator=MedianBackground())
        background_map = bkg.background
        zeros = data == 0
        data = data - bkg.background
        data[zeros] = 0

    if resbg_path:
        # Subtract a reprojected smoothed-residual background mosaic (m3/m4/m5).
        from reproject import reproject_interp
        if not os.path.exists(resbg_path):
            raise ValueError(f"manual resbg subtraction needs {resbg_path} to exist")
        with fits.open(resbg_path) as bgh:
            bg_hdu = bgh['SCI'] if 'SCI' in [h.name for h in bgh] else bgh[0]
            bg_wcs = wcs.WCS(bg_hdu.header)
            # MEMORY: crop the full-mosaic bg (~8766x11574) to just this frame's footprint
            # (+margin) BEFORE loading/reprojecting.  Reprojecting the whole mosaic in every
            # parallel worker was the OOM driver; a ~2k cutout is ~20x smaller per op.  Fall
            # back to the full mosaic only if the crop can't be computed.
            ny, nx = data.shape
            try:
                foot = ww.calc_footprint(axes=(nx, ny))            # sky corners (deg), shape (4,2)
                bx, by = bg_wcs.world_to_pixel_values(foot[:, 0], foot[:, 1])
                m = 64
                x0 = max(int(np.floor(np.nanmin(bx))) - m, 0)
                x1 = min(int(np.ceil(np.nanmax(bx))) + m, bg_hdu.shape[1])
                y0 = max(int(np.floor(np.nanmin(by))) - m, 0)
                y1 = min(int(np.ceil(np.nanmax(by))) + m, bg_hdu.shape[0])
                if x1 - x0 < 2 or y1 - y0 < 2:
                    raise ValueError("degenerate bg crop")
                bg_data = bg_hdu.section[y0:y1, x0:x1].astype(float)  # .section = read only the cutout
                bg_wcs = bg_wcs[y0:y1, x0:x1]
            except Exception as _e:
                print(f"[manual] bg crop failed ({_e}); reprojecting full mosaic", flush=True)
                bg_data = bg_hdu.data.astype(float)
        bg_reproj, _ = reproject_interp((bg_data, bg_wcs), ww, shape_out=data.shape)
        del bg_data
        bg_finite = np.where(np.isfinite(bg_reproj), bg_reproj, 0.0)
        zeros = data == 0
        data = data - bg_finite
        data[zeros] = 0
        background_map = bg_finite
        print(f"[manual] subtracted reprojected smoothed-bg {os.path.basename(resbg_path)} "
              f"(sum={float(np.nansum(bg_finite)):.3e})", flush=True)

    data = data.astype('float32')

    grid, _psf_model = _L.get_psf_model(
        filtername, proposal_id, field, module=module, use_webbpsf=use_webbpsf,
        use_grid=options.each_exposure, blur=options.blur, target=options.target,
        obsdate=obsdate, basepath='/blue/adamginsburg/adamginsburg/jwst/',
        psf_cache_dir=os.path.join(basepath, 'psfs'), instrument=instrument)
    dao_psf_model = grid
    if cutout_active and (cx0 or cy0):
        shifted_xy = [(gx - cx0, gy - cy0) for (gx, gy) in dao_psf_model.grid_xypos]
        dao_psf_model = type(dao_psf_model)(NDData(
            np.asarray(dao_psf_model.data),
            meta={'grid_xypos': shifted_xy,
                  'oversampling': dao_psf_model.oversampling}))
        print(f"[manual] CUTOUT: re-origined PSF grid by (-{cx0}, -{cy0})", flush=True)
    dao_psf_model.flux.min = 0

    dq, weight, bad = _L.get_uncertainty(
        err, data, wht=wht, dq=im1['DQ'].data if 'DQ' in im1 else None)

    # SourceGrouper.min_separation is a grouping RADIUS: sources closer than it
    # are fit jointly.  Default 2*FWHM leaves pairs >2*FWHM apart as singletons,
    # so close blends (e.g. sickle S2 + neighbor at 2.36*FWHM) are fit
    # independently and over-subtract in the valley between them.  Raise the
    # multiplier (--manual-group-min-sep-fwhm) to jointly fit wider pairs.
    _grp_mult = float(getattr(options, 'manual_group_min_sep_fwhm', 2.0))
    _grp_sep = _grp_mult * fwhm_pix
    # resolve_max_group_size rejects the ambiguous 0; None == 'unlimited' (no cap).
    _max_group_size = _L.resolve_max_group_size(
        getattr(options, 'max_group_size', 'unlimited'))
    if _max_group_size is not None:
        grouper = _L.CappedSourceGrouper(_grp_sep, max_size=_max_group_size)
    else:
        grouper = SourceGrouper(_grp_sep)

    kernel = Gaussian2DKernel(x_stddev=fwhm_pix / 2.355)
    mask = np.isnan(data) | bad
    dqarr = im1['DQ'].data if 'DQ' in im1 else None
    if dqarr is not None:
        is_saturated = (dqarr & _L.dqflags.pixel['SATURATED']) != 0
        # A real saturated core sits at/near the detector saturation level; but
        # JUMP/persistence artifacts get mis-tagged SATURATED on UNsaturated
        # sources (e.g. W51 F480M star RA=290.929589 Dec=+14.504683: DQ=6
        # SATURATED|JUMP_DET on a 7x7 island peaking ~355, real F480M saturation
        # >1e4).  Masking those drops a seeded real star from EVERY frame's fit.
        # With --saturation-data-floor > 0, only treat a SATURATED pixel as
        # un-fittable when its data actually exceeds the floor; default 0 keeps
        # the original behaviour (mask all SATURATED).
        sat_floor = float(getattr(options, 'saturation_data_floor', -1.0))
        if sat_floor < 0:  # auto: per-filter default (0 for unlisted -> mask all)
            sat_floor = _NIRCAM_SAT_DATA_FLOOR.get(filtername.lower(), 0.0)
        if sat_floor > 0:
            is_saturated = is_saturated & (np.nan_to_num(data) > sat_floor)
        data_ = data.copy()
        data_[is_saturated] = np.nan
        mask |= is_saturated
        mask |= (dqarr & _L._bad_dq_bitmask(instrument)) != 0
    else:
        data_ = data
    # MIRI detector-edge detection guard: daofind otherwise fires on the sharp
    # good/bad boundary GRADIENT at the edge-glow rim, injecting spurious "stars"
    # at the footprint border into the per-frame -> MERGED catalog.  These are
    # dropped by the per-obs vetting (which checks the data_i2d footprint) but
    # the model_i2d is built from the MERGED fits, so they render as huge edge
    # blobs (brick F2550W: 74131-peak model sources hugging the border).  Dilate
    # the BORDER-CONNECTED masked region inward by a margin so detection/fit
    # cannot trigger within it; INTERIOR NaN holes (saturated cores) are left
    # untouched so real stars there survive.  A genuine star within the margin of
    # the detector edge is measured in the overlapping neighbour tile's interior.
    _edge_margin = int(os.environ.get('MIRI_EDGE_DETECT_MARGIN', 8))
    if 'miri' in inst_token and _edge_margin > 0:
        from scipy.ndimage import binary_dilation as _bdil, label as _lab
        _lbl, _nl = _lab(mask)
        if _nl:
            _bset = set(np.unique(np.concatenate(
                [_lbl[0, :], _lbl[-1, :], _lbl[:, 0], _lbl[:, -1]]))) - {0}
            if _bset:
                _border = np.isin(_lbl, list(_bset))
                mask = mask | _bdil(_border, iterations=_edge_margin)
    nan_replaced_data = interpolate_replace_nans(data_, kernel, convolve=convolve_fft,
                                                 allow_huge=True)

    # --- saturated-star fit + subtract (per phase, namespaced by satstar_label) ---
    satstar_table = None
    satstar_model_subtracted = None
    fit_outside = getattr(options, 'fit_satstar_outside_fov', None)
    if fit_outside is None:
        fit_outside = not cutout_active
    if fit_outside:
        outside_star_pixels, outside_locked = _L.load_outside_fov_satstar_pixels(
            basepath, ww, data_shape=nan_replaced_data.shape, max_offset_arcsec=32.0)
    else:
        outside_star_pixels, outside_locked = [], False
    # Off-FOV forced-source position search radius (px).  0 when fully LOCKED
    # (hand-verified, no search).  Otherwise wide enough to reach the in-frame
    # diffraction spikes from the seed: a Spitzer prior is ~50 mas accurate but
    # the per-FRAME WCS can be ~0.3" (5 px) off the true star (field astrometry),
    # so a +/-5 px search hits its boundary and the spike-constrained amplitude
    # is never found (the over-sub clamp then has to rescue it).  +/-9 px (~0.6")
    # covers the observed frame-WCS offset with margin so the grid locks onto the
    # spikes and the amplitude is constrained by them, not just clamped.
    forced_grid_search_radius = 0 if outside_locked else 9
    satstar_file_suffix = f'{bgsub}{_iteration_token(satstar_label)}'
    # MIRI: feed the DEEP coadded data_i2d to the satstar seed gate so the
    # extended-emission phantom rejection (prominence + faint-core) is measured
    # on the noise-averaged coadd, not this single frame.  A per-frame measure
    # lets a phantom escape via one frame's noise spike (the cross-frame satstar
    # merge then keeps it).  This is the MANUAL-pipeline path (do_photometry_step
    # is the legacy non-manual path); both must plumb the coadd.  NIRCam: the
    # gate is MIRI-only inside get_saturated_stars, so this is a no-op there.
    _seed_gate_image = _seed_gate_wcs = None
    if module == 'mirimage':
        _di2d_path = (f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-'
                      f'o{field}_t001_{_L._inst_token(filtername)}_{pupil}-'
                      f'{filtername.lower()}-{module}_data_i2d.fits')
        if os.path.exists(_di2d_path):
            try:
                with fits.open(_di2d_path) as _dih:
                    _ext = 'SCI' if 'SCI' in [h.name for h in _dih] else 0
                    _seed_gate_image = _dih[_ext].data.astype(float)
                    _seed_gate_wcs = wcs.WCS(_dih[_ext].header)
            except Exception as _gex:
                print(f"[manual] satstar seed gate: could not load coadd "
                      f"{_di2d_path}: {_gex}", flush=True)
        else:
            print(f"[manual] satstar seed gate: coadd data_i2d not found "
                  f"({_di2d_path}); gate falls back to per-frame data", flush=True)
    satstar_table = _L.load_or_make_satstar_catalog(
        filename, path_prefix=f'{basepath}/psfs',
        use_merged_psf_for_merged=(module == 'merged'),
        overwrite=bool(outside_star_pixels),
        outside_star_pixels=outside_star_pixels, outside_star_fit_box=512,
        forced_grid_search_radius=forced_grid_search_radius,
        flux_overrides=satstar_flux_overrides,
        flux_drops=satstar_flux_drops,
        oversub_clamp_percentile=float(getattr(
            options, 'satstar_oversub_clamp_percentile', 10.0)),
        file_suffix=satstar_file_suffix,
        seed_gate_image=_seed_gate_image, seed_gate_wcs=_seed_gate_wcs,
        # ZEROFRAME deblend of merged saturated cores (gc2211 crowded GC fields).
        # Opt-in via --deblend-satstars; auto-degrades to legacy where the frame
        # has no sibling _ramp.fits ZEROFRAME.  See gc2211-zeroframe-satcore-deblend.
        deblend_with_zeroframe=bool(getattr(options, 'deblend_satstars', False)))
    ext_model = filename.replace('.fits', f'{satstar_file_suffix}_extended_satstar_model.fits')
    sat_model = filename.replace('.fits', f'{satstar_file_suffix}_satstar_model.fits')
    if os.path.exists(ext_model):
        sat_model = ext_model
    if os.path.exists(sat_model):
        try:
            sm = fits.getdata(sat_model).astype(float)
        except (OSError, ValueError) as exc:
            print(f"[manual] could not read satstar model {sat_model}: {exc}", flush=True)
        else:
            if sm.shape == nan_replaced_data.shape:
                finite_model = np.where(np.isfinite(sm), sm, 0.0)
                if dqarr is not None:
                    was_sat = (dqarr & _L.dqflags.pixel['SATURATED']) != 0
                    # ZEROFRAME saturated-RIM recovery (#2, opt-in): the most-
                    # saturated stars leave a positive ring because the cal-frame
                    # rim is brighter-fatter-INFLATED above the true flux; the
                    # ramp first read (group-0) samples the true profile wherever
                    # it is unsaturated.  Replace the rim with R*group0 (de-
                    # inflated) so model subtraction collapses the ring; the deep
                    # core (group-0 also saturated) falls back to model-replace.
                    _zf_done = False
                    if getattr(options, 'satstar_zeroframe_recover', False):
                        _g0 = _load_ramp_group0(filename)
                        if _g0 is not None and _g0.shape == data.shape:
                            from jwst_gc_pipeline.reduction.saturated_star_finding import (
                                zeroframe_recover_saturated)
                            _rec, _rim, _deep, _R = zeroframe_recover_saturated(
                                np.asarray(data, dtype=float), dqarr, _g0,
                                sat_dilate=int(getattr(
                                    options, 'satstar_zeroframe_dilate', 3)))
                            if np.isfinite(_R):
                                # rim -> recovered truth; remaining sat -> model.
                                nan_replaced_data = np.where(
                                    _rim, _rec,
                                    np.where(was_sat, finite_model, nan_replaced_data))
                                _zf_done = True
                                print(f"[manual] zeroframe rim recovery: R={_R:.3f} "
                                      f"rim={int(_rim.sum())} deepcore={int(_deep.sum())}",
                                      flush=True)
                    if not _zf_done:
                        nan_replaced_data = np.where(was_sat, finite_model, nan_replaced_data)
                nan_replaced_data = nan_replaced_data - finite_model
                satstar_model_subtracted = finite_model
                print(f"[manual] subtracted satstar model {os.path.basename(sat_model)} "
                      f"(sum={float(np.nansum(finite_model)):.3e})", flush=True)

    return types.SimpleNamespace(
        fwhm_pix=fwhm_pix, aperture_radius_pix=aperture_radius_pix,
        localbkg_inner=localbkg_inner, localbkg_outer=localbkg_outer,
        desat=desat, bgsub=bgsub, epsf_=epsf_, blur_=blur_, group=group,
        exposure_=exposure_, visitid_=visitid_, vgroupid_=vgroupid_,
        inst_token=inst_token, im1=im1, ww=ww, data=data, err=err, bad=bad,
        dqarr=dqarr, mask=mask, nan_replaced_data=nan_replaced_data,
        dao_psf_model=dao_psf_model, grouper=grouper,
        satstar_table=satstar_table, satstar_model_subtracted=satstar_model_subtracted,
        original_data=original_data, background_map=background_map,
        out_basepath=out_basepath, filename=filename,
        proposal_id=proposal_id, field=field, filtername=filtername,
        module=module, pupil=pupil)


def _save_manual_pass(ctx, result, modsky, options, iteration_label, detector):
    """Write the per-frame catalog (``save_photutils_results``) + residual +
    model for one manual pass, with the manual ``_m{N}`` iteration token.  The
    residual is built against the pristine (pre-bg-subtraction) data minus the
    satstar model minus the source model, so the smoothed-bg / mergedcat steps
    see star-subtracted residuals with the extended background retained.
    """
    iter_ = _iteration_token(iteration_label)
    bp = ctx.out_basepath
    saved = _L.save_photutils_results(
        result, ctx.ww, ctx.filename, im1=ctx.im1, detector=detector,
        basepath=bp, filtername=ctx.filtername, module=ctx.module,
        desat=ctx.desat, bgsub=ctx.bgsub, blur=options.blur,
        exposure_=ctx.exposure_, visitid_=ctx.visitid_, vgroupid_=ctx.vgroupid_,
        basic_or_iterative='basic', options=options, epsf_=ctx.epsf_,
        group=ctx.group, psf=None, background_map=ctx.background_map,
        iteration_label=iteration_label)

    base = (ctx.original_data if ctx.satstar_model_subtracted is None
            else ctx.original_data - ctx.satstar_model_subtracted)
    residual = base - modsky
    stub = (f'{bp}/{ctx.filtername}/pipeline/jw0{ctx.proposal_id}-o{ctx.field}_t001_'
            f'{ctx.inst_token}_{ctx.pupil}-{ctx.filtername.lower()}-{ctx.module}'
            f'{ctx.visitid_}{ctx.vgroupid_}{ctx.exposure_}{ctx.desat}{ctx.bgsub}'
            f'{ctx.epsf_}{ctx.blur_}{ctx.group}{iter_}_daophot_basic')
    _L.save_residual_datamodel(ctx.filename, f'{stub}_residual.fits', residual)
    _L.save_residual_datamodel(ctx.filename, f'{stub}_model.fits', modsky)
    return saved


def do_photometry_step_manual(options, filtername, module, detector, field, basepath,
                              filename, proposal_id, *, manual_phase,
                              exposurenumber=None, visit_id=None, vgroup_id=None,
                              bg_boxsizes=None, use_webbpsf=False, pupil='clear',
                              prev_seed_catalog=None, resbg_path=None,
                              satstar_flux_overrides=None,
                              satstar_flux_drops=None):
    """Clean per-frame driver for the manual-iteration path.

    ``manual_phase`` in {'m12','m3','m4','m5','m6','m7'}:
      * 'm12' runs iter1 (unseeded daofind) then iter2 (daofind(residual1) +
        same-frame iter1 catalog), no merge between -- saved as m1, m2.
      * 'm3' (iter3) and 'm4' (iter4) run a single pass on the RAW frames,
        seeded by per-frame daofind + the i2d-augmented catalog (daofind on the
        raw i2d for m3, the source-subtracted residual i2d for m4); m4 builds the
        first background map.
      * 'm5'/'m6'/'m7' (iter5/6/7) run a single pass on the background-subtracted
        data, seeded by daofind + the projected previous merged / cross-band
        catalog.
    """
    overshoot_ratio = float(getattr(options, 'manual_overshoot_ratio', 1.2))
    overshoot_action = str(getattr(options, 'manual_overshoot_action', 'refit'))
    iter2_snr = float(getattr(options, 'manual_iter2_local_snr', 3.0))
    first_snr = float(getattr(options, 'local_snr_threshold', 5.0))
    # PER-FRAME struct prune is DISABLED: the (struct_x,struct_y) values are
    # tuned on the deep i2d coadd; a single exposure has far higher structure
    # noise, so the same cut drops every detection (m2: 87->0).  The structure
    # prune is applied only where it is tuned -- on the i2d daofind in
    # _build_i2d_augmented_seed (run_manual_pipeline passes struct_x/y there).
    struct_x = 0.0
    struct_y = 0.0
    # Coarse-background detection (MIRI early phases only).  run_manual_pipeline
    # sets options.coarse_bg_box=51 for m12/m3/m4 and =0 for m5/m6 (where the
    # star-subtracted background is trustworthy).  Unlike the struct prune, a
    # coarse median subtraction is safe per-frame: it only shifts the detection
    # image pedestal, it does not reject sources.
    coarse_bg_box = int(getattr(options, 'coarse_bg_box', 0))

    ctx = _prepare_frame_for_photometry(
        options, filtername, module, field, basepath, filename, proposal_id,
        exposurenumber=exposurenumber, visit_id=visit_id, vgroup_id=vgroup_id,
        bg_boxsizes=bg_boxsizes, use_webbpsf=use_webbpsf, pupil=pupil,
        resbg_path=resbg_path, satstar_label=manual_phase,
        satstar_flux_overrides=satstar_flux_overrides,
        satstar_flux_drops=satstar_flux_drops)

    # MIRI-only: enable the dpk<=0 overshoot blind-spot guard (group-fit
    # degeneracy near bright/saturated neighbours).  NIRCam unaffected.
    try:
        _instrume = str(ctx.im1['SCI'].header.get('INSTRUME',
                        ctx.im1[0].header.get('INSTRUME', ''))).upper()
    except Exception:
        _instrume = ''
    _is_miri = 'MIRI' in _instrume

    # MIRI: load the DEEP data i2d once so the prominence cut is measured on the
    # coadd (frame-invariant), not single noisy exposures.  Path mirrors
    # _data_i2d_path in run_manual_pipeline.
    _prom_i2d = None
    _prom_ww_i2d = None
    if _is_miri:
        try:
            _i2dp = (f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}'
                     f'_t001_{_L._inst_token(filtername)}_{pupil}-'
                     f'{filtername.lower()}-{module}_data_i2d.fits')
            if os.path.exists(_i2dp):
                _ih = fits.open(_i2dp)
                _prom_i2d = _ih['SCI'].data
                _prom_ww_i2d = wcs.WCS(_ih['SCI'].header)
        except Exception as _e:
            print(f"[manual] prominence i2d load failed ({_e}); "
                  f"falling back to per-frame data", flush=True)
            _prom_i2d = None

    # MIRI satstar-coincidence exclusion: daophot fits within 1.5 FWHM of a
    # satstar catalog entry are double-counts (see _manual_phot_pass).  Build
    # the satstar pixel positions once.  NIRCam: leave disabled (pix=0).
    _satstar_xy = None
    _satstar_excl_pix = 0.0
    if _is_miri and ctx.satstar_table is not None and len(ctx.satstar_table):
        _sst = ctx.satstar_table
        if 'x_fit' in _sst.colnames and 'y_fit' in _sst.colnames:
            _sxv = np.asarray(_sst['x_fit'], dtype=float)
            _syv = np.asarray(_sst['y_fit'], dtype=float)
            _ok = np.isfinite(_sxv) & np.isfinite(_syv)
            if _ok.any():
                _satstar_xy = np.column_stack([_sxv[_ok], _syv[_ok]])
                _satstar_excl_pix = 1.5 * ctx.fwhm_pix

    def _pass(seed, label):
        return _manual_phot_pass(
            data=ctx.nan_replaced_data, mask=ctx.mask, err=ctx.err, bad=ctx.bad,
            dao_psf_model=ctx.dao_psf_model, init_params=seed,
            aperture_radius_pix=ctx.aperture_radius_pix,
            localbkg_inner=ctx.localbkg_inner, localbkg_outer=ctx.localbkg_outer,
            grouper=ctx.grouper, options=options, dq=ctx.dqarr,
            satstar_model_subtracted=(ctx.satstar_model_subtracted
                                      if ctx.satstar_model_subtracted is not None
                                      else np.zeros(ctx.data.shape)),
            label=label, xy_bounds_pix=None,
            overshoot_ratio=overshoot_ratio, overshoot_action=overshoot_action,
            miri_dpk_guard=_is_miri,
            satstar_excl_xy=_satstar_xy, satstar_excl_pix=_satstar_excl_pix,
            near_sat_dist_pix=(1.5 * ctx.fwhm_pix if _is_miri else 1.0),
            miri_prominence_snr=(float(getattr(options, 'miri_prominence_snr', 5.0))
                                 if _is_miri else 0.0),
            prominence_bg_box=0,
            prominence_data_i2d=_prom_i2d, prominence_ww_i2d=_prom_ww_i2d,
            frame_ww=ctx.ww)

    if manual_phase == 'm12':
        seed1 = _build_manual_seed(
            detection_image=ctx.nan_replaced_data, nan_replaced_data=ctx.nan_replaced_data,
            mask=ctx.mask, ww=ctx.ww, fwhm_pix=ctx.fwhm_pix,
            satstar_table=ctx.satstar_table, prev_catalog=None,
            local_snr_threshold=first_snr, roundlo=-1.0, roundhi=1.0,
            sharplo=0.30, sharphi=1.40, dedup_min_sep_pix=0.5 * ctx.fwhm_pix,
            label='m1', apply_snr_filter=False, coarse_bg_box=coarse_bg_box)
        res1, modsky1, _ = _pass(seed1, 'm1')
        saved1 = _save_manual_pass(ctx, res1, modsky1, options, 'm1', detector)
        base1 = (ctx.original_data if ctx.satstar_model_subtracted is None
                 else ctx.original_data - ctx.satstar_model_subtracted)
        residual1 = base1 - modsky1
        seed2 = _build_manual_seed(
            detection_image=residual1, nan_replaced_data=ctx.nan_replaced_data,
            mask=ctx.mask, ww=ctx.ww, fwhm_pix=ctx.fwhm_pix,
            satstar_table=ctx.satstar_table, prev_catalog=saved1,
            local_snr_threshold=iter2_snr, roundlo=-0.3, roundhi=0.3,
            sharplo=0.50, sharphi=1.00, dedup_min_sep_pix=0.5 * ctx.fwhm_pix,
            label='m2', struct_x=struct_x, struct_y=struct_y,
            coarse_bg_box=coarse_bg_box)
        res2, modsky2, _ = _pass(seed2, 'm2')
        _save_manual_pass(ctx, res2, modsky2, options, 'm2', detector)
        return res2

    # m3 / m4 / m5: single seeded pass
    prev = Table.read(prev_seed_catalog) if prev_seed_catalog else None
    snr = iter2_snr
    seed = _build_manual_seed(
        detection_image=ctx.nan_replaced_data, nan_replaced_data=ctx.nan_replaced_data,
        mask=ctx.mask, ww=ctx.ww, fwhm_pix=ctx.fwhm_pix,
        satstar_table=ctx.satstar_table, prev_catalog=prev,
        local_snr_threshold=snr, roundlo=-0.3, roundhi=0.3,
        sharplo=0.50, sharphi=1.00, dedup_min_sep_pix=0.5 * ctx.fwhm_pix,
        label=manual_phase, struct_x=struct_x, struct_y=struct_y,
        coarse_bg_box=coarse_bg_box)
    res, modsky, _ = _pass(seed, manual_phase)
    _save_manual_pass(ctx, res, modsky, options, manual_phase, detector)
    return res


def _build_source_masked_bg(mc_i2d_path, vetted_catalog_path, filtername, *,
                            mask_radius_fwhm=2.0, median_size=3):
    """Build the smoothed background map from a mergedcat residual i2d with the
    fitted SOURCE CORES MASKED OUT before smoothing.

    Why: the plain smoothed residual is biased NEGATIVE at every star core (each
    fit leaves a small negative hole there).  Subtracting that negative bg from
    the data ADDS flux at the core, so the next iteration's fit inflates, deepens
    the hole, and the bg gets more negative -- a positive-feedback loop that makes
    faint stars go progressively negative with iteration (NOTES_star_vs_extended_emission.md;
    sickle low_background sources 2026-06-09).  Masking the source disks and
    interpolating over them makes the bg represent the DIFFUSE background only
    (~0 in a star field, the true extended emission in pillar fields), so it no
    longer feeds the source holes back into the fit.

    Writes ``<mc_i2d>_..._smoothed_bg_i2d.fits`` (same name the plain smoother
    would produce) and returns its path.
    """
    from astropy.io import fits as _fits
    from astropy.wcs import WCS as _WCS
    from astropy.coordinates import SkyCoord as _SkyCoord
    from astropy.convolution import (Gaussian2DKernel as _G2D,
                                     interpolate_replace_nans as _irn,
                                     convolve_fft as _cfft)
    from scipy import ndimage as _ndi

    fh = _fits.open(mc_i2d_path)
    hdu = fh['SCI'] if 'SCI' in [h.name for h in fh] else fh[0]
    w = _WCS(hdu.header)
    d = hdu.data.astype(float)
    fov = np.isfinite(d)

    # FWHM in i2d pixels
    ftab = Table.read(_L.FWHM_TABLE)
    fwhm_as = float(ftab[ftab['Filter'] == filtername]['PSF FWHM (arcsec)'][0])
    pixscale_as = float(np.sqrt(np.abs(np.linalg.det(w.pixel_scale_matrix))) * 3600.0)
    fwhm_px = fwhm_as / pixscale_as
    R = max(2.0, mask_radius_fwhm * fwhm_px)

    work = d.copy()
    try:
        t = Table.read(vetted_catalog_path)
        if 'skycoord' in t.colnames and len(t) > 0:
            sc = t['skycoord']
            if not isinstance(sc, _SkyCoord):
                sc = _SkyCoord(sc)
            xs, ys = w.world_to_pixel(sc)
            ny, nx = d.shape
            Ri = int(np.ceil(R))
            for xc, yc in zip(np.atleast_1d(xs), np.atleast_1d(ys)):
                if not (np.isfinite(xc) and np.isfinite(yc)):
                    continue
                ix, iy = int(round(float(xc))), int(round(float(yc)))
                y0, y1 = max(0, iy - Ri), min(ny, iy + Ri + 1)
                x0, x1 = max(0, ix - Ri), min(nx, ix + Ri + 1)
                if y0 >= y1 or x0 >= x1:
                    continue
                yy, xx = np.mgrid[y0:y1, x0:x1]
                disk = (xx - xc) ** 2 + (yy - yc) ** 2 <= R ** 2
                work[y0:y1, x0:x1][disk] = np.nan
    except Exception as ex:
        print(f"[bg] source-masking failed ({ex}); smoothing unmasked residual",
              flush=True)

    # interpolate over the masked source disks (fill from surrounding diffuse bg)
    if np.any(np.isnan(work) & fov):
        kern = _G2D(x_stddev=max(1.0, fwhm_px))
        work = _irn(work, kern, convolve=_cfft, allow_huge=True)
    sm = _ndi.median_filter(np.nan_to_num(work), size=int(median_size), mode='nearest')
    sm[~fov] = np.nan
    out = residual_to_smoothed_bg_i2d(mc_i2d_path)
    _fits.PrimaryHDU(data=sm.astype('float32'), header=hdu.header).writeto(out, overwrite=True)
    print(f"[bg] wrote source-masked smoothed bg {os.path.basename(out)} "
          f"(masked {0 if 'sc' not in dir() else len(np.atleast_1d(xs))} sources, "
          f"R={R:.1f}px)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Cross-band stringent seed (step 18; multifilter only)
# ---------------------------------------------------------------------------
def _build_crossband_seed(cut_bp, modules, filternames, options, *,
                          max_sep_mas=10.0, min_filters=2, snr_min=5.0,
                          qfit_max=0.2):
    """Cross-filter seed for m7 (iter7): sources detected (well) in >= min_filters
    filters within max_sep_mas, each with S/N > snr_min and good qfit.

    NOTE: the stringent cross-match (>=2 filters, <10 mas, S/N>5 each) is a TODO;
    the immediate cutout tests are single-filter so this path is not exercised.
    For now it unions the per-filter vetted m6 skycoords (like the legacy union
    seed) so the multifilter path is runnable.  Replace with
    ``merge_catalogs.merge_catalogs(..., max_offset=max_sep_mas*u.mas)`` + the
    >=min_filters / snr / qfit cut before relying on m5 scientifically.
    """
    from astropy.table import vstack as _vstack
    desat = '_unsatstar' if options.desaturated else ''
    bgsub = ('_bgsub' if options.bgsub else '') + '_resbgsub'
    blur_ = '_blur' if options.blur else ''
    # Per-obs token for gc2211 (prop 2211): the m6 vetted catalogs + this seed
    # are per-obs (see _obs_suffix usage in run_manual_pipeline).  Empty elsewhere.
    _obssuf = _L.obs_token(getattr(options, 'proposal_id', None),
                           getattr(options, 'field', None))
    tbls = []
    for module in modules:
        for filt in filternames:
            p = (f'{cut_bp}/catalogs/{filt.lower()}_{module}_indivexp_merged'
                 f'{desat}{bgsub}{blur_}_m6_dao_basic{_obssuf}_vetted.fits')
            if os.path.exists(p):
                t = Table.read(p)
                if 'skycoord' in t.colnames:
                    tbls.append(Table({'skycoord': t['skycoord']}))
    if not tbls:
        raise ValueError(f"m7 crossband seed: no vetted m6 catalogs under {cut_bp}/catalogs/")
    union = _vstack(tbls, metadata_conflicts='silent')
    out = f'{cut_bp}/catalogs/crossband_seed_manual{_obssuf}.fits'
    union.write(out, overwrite=True)
    print(f"[m7] wrote crossband seed {out} (n={len(union)}); "
          f"TODO stringent >= {min_filters}-filter/{max_sep_mas}mas/SNR>{snr_min} cut",
          flush=True)
    return out


# ---------------------------------------------------------------------------
# Detection on the merged i2d -> augmented seed for the next per-frame round
# ---------------------------------------------------------------------------
def _structure_noise_keep(data, err, xpix, ypix, *, struct_x=0.0, struct_y=0.0,
                          smooth_box=51, struct_box=51):
    """Structure-noise prune (AG 2026-06-13): keep a detection only if its data
    peak rises above the local extended-emission baseline by a combination of
    photon/read noise AND emission-structure noise:

        peak > smoothed_mean + struct_x * real_noise + struct_y * structure_noise

    smoothed_mean   = median-filtered data on ``smooth_box`` (>~5x FWHM, so it is
                      the large-scale background, NOT the filaments);
    real_noise      = propagated per-pixel ERR;
    structure_noise = local RMS of (data - smoothed) over ``struct_box`` -- the
                      "noise" introduced by emission structure.  This rejects
                      filament/PAH bumps (high structure noise) while keeping real
                      point sources (peak >> both noises).  Tuned high-purity on
                      sickle F770W: (struct_x, struct_y) ~ (5, 8).

    Returns a boolean keep-mask aligned with (xpix, ypix).  No-op if both x,y=0.
    """
    if struct_x == 0.0 and struct_y == 0.0:
        return np.ones(len(xpix), dtype=bool)
    from scipy.ndimage import median_filter, uniform_filter
    from astropy.stats import mad_std
    good = np.isfinite(data)
    filled = np.where(good, data, np.nanmedian(data[good]))
    smoothed = median_filter(filled, size=smooth_box)
    hp = filled - smoothed
    struct_noise = np.sqrt(np.clip(uniform_filter(hp**2, size=struct_box), 0, None))
    rn = (np.asarray(err, dtype=float) if err is not None
          else np.full_like(data, mad_std(hp[good])))
    ny, nx = data.shape
    xi = np.clip(np.round(np.asarray(xpix)).astype(int), 0, nx - 1)
    yi = np.clip(np.round(np.asarray(ypix)).astype(int), 0, ny - 1)
    thresh = smoothed[yi, xi] + struct_x * rn[yi, xi] + struct_y * struct_noise[yi, xi]
    return data[yi, xi] > thresh


def _build_i2d_augmented_seed(detection_i2d_path, prev_vetted_path, filtername, *,
                              local_snr_min=5.0, roundlo=-0.5, roundhi=0.5,
                              sharplo=0.4, sharphi=1.2, bg_subtract_path=None,
                              struct_x=0.0, struct_y=0.0, coarse_bg_box=0, label=''):
    """daofind on the merged i2d co-add, unioned with the previous vetted merged
    catalog, written as a seed catalog (``skycoord`` + ``flux``) for the next
    per-frame PSF-photometry round (the plan's iter3 seed = daofind(i2d) +
    previous catalog).

    ``detection_i2d_path`` is the detection co-add for this iteration, which gets
    progressively cleaner so sources hidden in one stage surface in the next:
      * iter3: raw data i2d
      * iter4: source-subtracted residual i2d (mergedcat residual)
      * iter5: residual i2d with the diffuse background also subtracted
               (``bg_subtract_path``)
    This recovers significant sources (peak/ERR S/N >> 5) that are NOT local
    maxima in the raw co-add because they sit on the wings of brighter stars or
    on nebulosity -- daofind cannot find those at ANY threshold, and lowering the
    threshold only injects noise/nebulosity false positives in crowded fields.
    Subtracting the brighter stars (and then the background) makes them clean
    local maxima with no threshold change -- region-safe.  See possible_stars.reg.

    PSF photometry is *never* run on the i2d -- the fit always runs on the raw
    (or background-subtracted) frames.  This step is detection-only.

    The i2d cutouts are drizzled at the native detector scale (0.063"/px), so the
    per-frame pixel FWHM applies unchanged.  Returns the seed-catalog path.
    """
    from astropy.coordinates import SkyCoord

    ftab = Table.read(_L.FWHM_TABLE)
    fwhm_pix = float(ftab[ftab['Filter'] == filtername]['PSF FWHM (pixel)'][0])

    with fits.open(detection_i2d_path) as dh:
        names = [h.name for h in dh]
        sci = dh['SCI'] if 'SCI' in names else dh[0]
        data = np.asarray(sci.data, dtype=float)
        ww_i2d = wcs.WCS(sci.header)
        wht = (np.asarray(dh['WHT'].data, dtype=float) if 'WHT' in names else None)
        err = (np.asarray(dh['ERR'].data, dtype=float) if 'ERR' in names else None)
    mask = ~np.isfinite(data)
    if wht is not None:
        mask |= ~np.isfinite(wht) | (wht <= 0)
    # iter5: also subtract the diffuse background so faint sources sitting on
    # nebulosity become local maxima (same grid as the residual i2d)
    if bg_subtract_path and os.path.exists(bg_subtract_path):
        with fits.open(bg_subtract_path) as bh:
            bn = [h.name for h in bh]
            bsci = bh['SCI'] if 'SCI' in bn else bh[0]
            bg = np.asarray(bsci.data, dtype=float)
        if bg.shape == data.shape:
            data = data - np.where(np.isfinite(bg), bg, 0.0)
        else:
            print(f"[{label}] bg-subtract skipped: shape {bg.shape} != {data.shape}",
                  flush=True)
    elif coarse_bg_box and coarse_bg_box > 0:
        # MIRI early phases (m3/m4, no trustworthy bg map yet): coarse median
        # subtraction so high-local-contrast stars on the huge F770W pedestal
        # become detectable (AG 2026-06-13).  m5/m6 take the bg_subtract_path
        # branch above instead (the real star-subtracted background).
        from scipy.ndimage import median_filter as _medfilt
        _g = np.isfinite(data) & ~mask
        _f = np.where(_g, data, np.nanmedian(data[_g]))
        data = data - _medfilt(_f, size=int(coarse_bg_box))
        print(f"[{label}] coarse-bg detection: subtracted {int(coarse_bg_box)}px median",
              flush=True)
    pixscale_as = float(np.sqrt(np.abs(np.linalg.det(ww_i2d.pixel_scale_matrix))) * 3600.0)

    # Detection threshold from the local-noise floor (as _build_manual_seed),
    # but the S/N CUT uses the i2d's propagated ERR -- not the empirical
    # local-scatter map.  On the deep co-add the local-scatter map is
    # source-variance dominated, so a local-S/N>=5 cut rejects every real star
    # (the same trap that emptied the raw-frame iter1 seed); peak/ERR is the
    # correct, propagated detection significance here.
    noise_map = compute_local_noise_map(np.where(mask, np.nan, data), smooth_sigma_pix=3.0)
    finite = np.isfinite(noise_map) & (noise_map > 0)
    if not np.any(finite):
        raise ValueError(f"[{label}] i2d local noise map has no positive finite values")
    threshold = float(np.nanmin(noise_map[finite]))
    daofind = DAOStarFinder(threshold=threshold, fwhm=fwhm_pix,
                            roundlo=roundlo, roundhi=roundhi,
                            sharplo=sharplo, sharphi=sharphi)
    det = daofind(np.where(mask, 0.0, data), mask=mask)
    if det is None:
        det = Table()
    n_raw = len(det)
    if len(det):
        if err is not None:
            snr_map = np.array(err, dtype=float)
            snr_map[~np.isfinite(snr_map) | (snr_map <= 0)] = np.inf
            det, _ = annotate_and_filter_by_local_snr(
                det, snr_map, snr_threshold=local_snr_min)
            snr_basis = f'ERR S/N>={local_snr_min}'
        else:
            det, _ = annotate_and_filter_by_local_snr(det, noise_map, snr_threshold=0.0)
            snr_basis = 'no ERR -> no S/N cut'
    else:
        snr_basis = '(none)'
    print(f"[{label}] i2d daofind: {n_raw} -> {len(det)} after {snr_basis}",
          flush=True)

    # structure-noise prune: reject extended-emission bumps (MIRI; AG method)
    if len(det) and (struct_x or struct_y):
        xd0, yd0 = _L._best_available_xy(det)
        skeep = _structure_noise_keep(np.where(mask, np.nan, data), err, xd0, yd0,
                                      struct_x=struct_x, struct_y=struct_y)
        n_before = len(det)
        det = det[skeep]
        print(f"[{label}] struct-noise prune (x={struct_x},y={struct_y}): "
              f"{n_before} -> {len(det)}", flush=True)

    # previous vetted merged catalog -> sky
    prev = _L._resolve_seed_skycoords(Table.read(prev_vetted_path))
    prev_sky = prev['skycoord']
    if not isinstance(prev_sky, SkyCoord):
        prev_sky = SkyCoord(prev_sky)
    # the fitted flux: per-frame catalogs use 'flux_fit', the MERGED/vetted
    # catalog uses 'flux'.  Falling through to ones() poisons the next fit --
    # bright stars then start at flux_init=1.0 (orders of magnitude too low) and
    # the free-position LevMar diverges (positions run off the frame), spawning
    # phantom catalog sources.  Try both real columns before the unit fallback.
    prev_flux = None
    for _fc in ('flux_fit', 'flux'):
        if _fc in prev.colnames:
            prev_flux = np.asarray(prev[_fc], dtype=float)
            break
    if prev_flux is None:
        prev_flux = np.ones(len(prev), dtype=float)
        print(f"[{label}] WARNING: no flux/flux_fit column in {os.path.basename(prev_vetted_path)}; "
              f"seeding flux_init=1.0", flush=True)
    # never seed from negative-peak (flux<=0) sources
    _pos = prev_flux > 0
    if not np.all(_pos):
        prev_sky = prev_sky[_pos]
        prev_flux = prev_flux[_pos]
        print(f"[{label}] dropped {int(np.sum(~_pos))} non-positive-flux prev seeds",
              flush=True)

    # i2d detections -> sky, keep only those NOT already in the previous catalog
    n_new = 0
    if len(det):
        # photutils 2.x emits xcentroid/ycentroid, 3.x x_centroid/y_centroid
        xd, yd = _L._best_available_xy(det)
        det_sky = ww_i2d.pixel_to_world(np.asarray(xd, dtype=float),
                                        np.asarray(yd, dtype=float))
        det_flux = (np.asarray(det['flux'], dtype=float)
                    if 'flux' in det.colnames else np.ones(len(det), dtype=float))
        if len(prev_sky):
            _, sep, _ = det_sky.match_to_catalog_sky(prev_sky)
            match_as = max(1.0, 0.5 * fwhm_pix) * pixscale_as
            fresh = sep.arcsec > match_as
        else:
            fresh = np.ones(len(det_sky), dtype=bool)
        n_new = int(np.sum(fresh))
        if n_new:
            all_sky = SkyCoord([prev_sky, det_sky[fresh]]) if len(prev_sky) else det_sky[fresh]
            all_flux = np.concatenate([prev_flux, det_flux[fresh]])
        else:
            all_sky, all_flux = prev_sky, prev_flux
    else:
        all_sky, all_flux = prev_sky, prev_flux

    out = Table()
    out['skycoord'] = all_sky
    out['flux'] = all_flux
    outpath = vetted_to_i2dseed(prev_vetted_path)
    out.write(outpath, overwrite=True)
    print(f"[{label}] i2d-augmented seed: {len(prev_sky)} prev + {n_new} new i2d "
          f"-> {len(out)} ({os.path.basename(outpath)})", flush=True)
    return outpath


# ---------------------------------------------------------------------------
# Orchestrator (mirrors _run_cutout_pipeline; manual phases, basic-only)
# ---------------------------------------------------------------------------
def _run_one_frame_manual(args):
    """Pickleable per-frame worker for ProcessPoolExecutor parallelism in
    ``run_manual_pipeline``.  ``args`` is a dict whose keys mirror the kwargs of
    ``do_photometry_step_manual``.  Returns ``(filename, ok, err_str_or_None)``.

    Only ``_L.CutoutNoOverlap`` is a legitimate non-fatal outcome (a frame that
    does not overlap a requested cutout region).  EVERY other failure is reported
    back as ``(False, err)`` and the caller HARD-CRASHES the whole run: silently
    dropping any exposure from any phase irrecoverably corrupts all later steps
    and the final catalog/mosaic.  Transient I/O (e.g. a truncated FITS read on a
    busy shared filesystem) is retried a few times first so a flaky read does not
    spuriously abort a long run; a persistent failure still aborts.
    """
    import time as _time
    filename = args['filename']
    _N_RETRY = 4
    last_err = None
    for _attempt in range(_N_RETRY):
        try:
            do_photometry_step_manual(
                args['options'], args['filtername'], args['module'], args['detector'],
                args['field'], args['basepath'], filename, args['proposal_id'],
                manual_phase=args['manual_phase'],
                exposurenumber=args['exposurenumber'],
                visit_id=args['visit_id'], vgroup_id=args['vgroup_id'],
                bg_boxsizes=args['bg_boxsizes'],
                use_webbpsf=args['use_webbpsf'], pupil=args['pupil'],
                prev_seed_catalog=args['prev_seed_catalog'],
                resbg_path=args['resbg_path'],
                satstar_flux_overrides=args.get('satstar_flux_overrides'),
                satstar_flux_drops=args.get('satstar_flux_drops'))
            return (filename, True, None)
        except _L.CutoutNoOverlap as ex:
            return (filename, False, f'no-overlap: {ex}')
        except (OSError, IOError) as ex:
            # Transient shared-filesystem read errors ("Header missing END card",
            # truncated reads, stale NFS handles) -- retry with backoff.
            last_err = f'{type(ex).__name__}: {ex}\n{traceback.format_exc()}'
            if _attempt < _N_RETRY - 1:
                _time.sleep(2.0 * (_attempt + 1))
                continue
            return (filename, False, last_err)
        except Exception as ex:
            return (filename, False,
                    f'{type(ex).__name__}: {ex}\n{traceback.format_exc()}')


def _clean_offfov_dups_and_offfield(merged, filt, data_i2d_path, basepath, *,
                                    dedup_arcsec=1.0, fov_pad_psf=5.0):
    """Post-merge cleanup of off-FOV satstar artifacts (per-filter merged catalog).

    A) Collapse ``replaced_saturated`` rows clustering within ``dedup_arcsec`` to a
       SINGLE representative (the median-flux member).  An off-FOV bright star is
       fit PER FRAME and the (degenerate) positions scatter wider than the 0.15"
       satstar dedup, leaving many rows at one physical location -- the catalog
       must have exactly ONE entry per off-FOV star (sickle F480M m7 had 12).
    B) Drop NON-satstar rows projecting > ``fov_pad_psf`` PSF widths OUTSIDE this
       filter's data FOV.  These are m7 cross-band-seed artifacts: the shared
       seed unions positions from ALL filters, so positions covered only by other
       (e.g. SW) filters fall outside this LW filter's FOV, get fit at off-field
       locations, and are inserted as garbage (qfit up to 1e4, up to 56" out).
       replaced_saturated rows are KEPT (a real off-FOV star legitimately sits
       off-field -- but as ONE row after A).

    Returns (cleaned_table, n_dedup_removed, n_offfield_removed).
    """
    if merged is None or len(merged) == 0:
        return merged, 0, 0
    from astropy.coordinates import SkyCoord, search_around_sky
    sc = (SkyCoord(merged['skycoord']) if 'skycoord' in merged.colnames
          else SkyCoord(merged['ra'], merged['dec'], unit='deg'))
    fcol = ('flux' if 'flux' in merged.colnames
            else ('flux_fit' if 'flux_fit' in merged.colnames else None))
    rs = (np.asarray(merged['replaced_saturated'], dtype=bool)
          if 'replaced_saturated' in merged.colnames else np.zeros(len(merged), bool))
    drop = np.zeros(len(merged), bool)

    # ---- A: dedup replaced_saturated clusters to one representative ----
    if rs.any() and fcol is not None:
        sat_idx = np.where(rs)[0]
        ssc = sc[sat_idx]
        i1, i2, _, _ = search_around_sky(ssc, ssc, dedup_arcsec * u.arcsec)
        parent = list(range(len(sat_idx)))
        def _find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for a, b in zip(i1, i2):
            parent[_find(int(a))] = _find(int(b))
        groups = {}
        for i in range(len(sat_idx)):
            groups.setdefault(_find(i), []).append(i)
        flux = np.asarray(merged[fcol], dtype=float)
        for g in groups.values():
            if len(g) <= 1:
                continue
            gf = flux[sat_idx[g]]
            keep = g[int(np.argmin(np.abs(gf - np.nanmedian(gf))))]
            for j in g:
                if j != keep:
                    drop[sat_idx[j]] = True
    n_dedup = int(drop.sum())

    # ---- B: drop non-satstar rows far outside the FOV ----
    n_off = 0
    if data_i2d_path and os.path.exists(data_i2d_path):
        try:
            with fits.open(data_i2d_path) as dh:
                _s = dh['SCI'] if 'SCI' in [x.name for x in dh] else dh[0]
                ww = wcs.WCS(_s.header); ny, nx = _s.data.shape
            px = float(ww.proj_plane_pixel_scales()[0].to('arcsec').value)
            fw = 0.16
            try:
                _ft = Table.read(f'{basepath}/reduction/fwhm_table.ecsv')
                _row = _ft[_ft['Filter'] == filt.upper()]
                if len(_row):
                    fw = float(_row['PSF FWHM (arcsec)'][0])
            except Exception:
                pass
            pad = fov_pad_psf * fw / px
            x, y = ww.world_to_pixel(sc)
            outside = ((x < -pad) | (x > nx - 1 + pad)
                       | (y < -pad) | (y > ny - 1 + pad))
            offfield = outside & (~rs) & (~drop)
            drop |= offfield
            n_off = int(offfield.sum())
        except Exception as ex:
            print(f"  off-field FOV filter skipped ({ex})", flush=True)
    return merged[~drop], n_dedup, n_off


def _reconstruct_smoothed_bg_path(cut_bp, proposal_id, field, module, filt,
                                  label, options, pupil):
    """Rebuild the on-disk smoothed-bg i2d path for a completed phase ``label``.

    Used by --manual-start-phase to recover the previous phase's background map
    (the only cross-phase state) when starting partway through.  Mirrors the
    name build_mergedcat_residuals + _build_source_masked_bg write:
    ``...-{filt}-{module}{desat}{bgsub}{group}_{label}_daophot_basic_mergedcat_residual_smoothed_bg_i2d.fits``
    (m5/m6/m7 are resbgsub).
    """
    desat = '_unsatstar' if options.desaturated else ''
    bgsub = ('_bgsub' if options.bgsub else '') + (
        '_resbgsub' if label in ('m5', 'm6', 'm7') else '')
    group_ = '_group' if options.group else ''
    inst = _L._inst_token(filt)
    return (f'{cut_bp}/{filt}/pipeline/jw0{proposal_id}-o{field}_t001_{inst}_'
            f'{pupil}-{filt.lower()}-{module}{desat}{bgsub}{group_}_{label}_'
            f'daophot_basic_mergedcat_residual_smoothed_bg_i2d.fits')


def _reconstruct_resid_i2d_path(cut_bp, proposal_id, field, module, filt,
                                label, options, pupil):
    """The mergedcat residual i2d (the next phase's detection image) sits next to
    the smoothed-bg i2d, differing only by the ``_smoothed_bg`` infix.  Derive it
    from ``_reconstruct_smoothed_bg_path`` so per-frame SLURM jobs can rebuild
    ``resid_i2d_for_next`` from disk (it is otherwise in-memory only)."""
    bg = _reconstruct_smoothed_bg_path(cut_bp, proposal_id, field, module, filt,
                                       label, options, pupil)
    return smoothed_bg_to_detection_i2d(bg)


def _satstar_reconciled_path(cut_bp, module, filt):
    """On-disk persistence for m12's cross-frame-reconciled out-of-FOV satstar
    fluxes (``satstar_overrides``) + drops (``satstar_drops``).  In a monolithic
    run these live only in memory and are forwarded to m3..m7; persisting them
    lets a per-frame m3..m7 fan-out worker (a fresh process) reconstruct them."""
    return (f'{cut_bp}/catalogs/{filt.lower()}_{module}_'
            f'satstar_reconciled_m12.fits')


def _persist_reconciled_satstars(path, ovr, drp):
    """Write the reconciled satstar overrides/drops as one flat table.
    ``ovr`` = list of (SkyCoord, flux); ``drp`` = list of SkyCoord (flux=NaN)."""
    from astropy.table import Table
    import numpy as _np
    ras, decs, fluxes, kinds = [], [], [], []
    for sc, fl in (ovr or []):
        ras.append(float(sc.ra.deg)); decs.append(float(sc.dec.deg))
        fluxes.append(float(fl)); kinds.append('override')
    for sc in (drp or []):
        ras.append(float(sc.ra.deg)); decs.append(float(sc.dec.deg))
        fluxes.append(_np.nan); kinds.append('drop')
    Table({'ra': _np.array(ras, dtype='float64'),
           'dec': _np.array(decs, dtype='float64'),
           'flux': _np.array(fluxes, dtype='float64'),
           'kind': _np.array(kinds if kinds else [], dtype='U8')}
          ).write(path, overwrite=True)


def _reconstruct_reconciled_satstars(path):
    """Inverse of :func:`_persist_reconciled_satstars`; returns ``(ovr, drp)`` in
    the in-memory shapes m3..m7 expect, or ``([], [])`` if the file is absent."""
    import os as _os
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import astropy.units as _u
    import numpy as _np
    if not _os.path.exists(path):
        return [], []
    t = Table.read(path)
    ovr, drp = [], []
    for row in t:
        sc = SkyCoord(float(row['ra']) * _u.deg, float(row['dec']) * _u.deg)
        if str(row['kind']) == 'drop' or not _np.isfinite(row['flux']):
            drp.append(sc)
        else:
            ovr.append((sc, float(row['flux'])))
    return ovr, drp


def _reconstruct_prev_merged(merged_path):
    """Rebuild ``prev_merged_for`` = (SkyCoord, iter_found array) from a prior
    phase's on-disk merged catalog (it carries the ``iter_found`` column), so a
    per-frame finalize job can tag provenance identically to a monolithic run.
    Returns ``None`` if the catalog or its columns are unavailable."""
    import os as _os
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import numpy as _np
    if not _os.path.exists(merged_path):
        return None
    t = Table.read(merged_path)
    if 'skycoord' not in t.colnames or 'iter_found' not in t.colnames:
        return None
    return (SkyCoord(t['skycoord']), _np.asarray(t['iter_found']))


def _resolve_crossband_ref_filter(options, filternames):
    """Pick the astrometric reference filter for the cross-band merge.

    Explicit ``--manual-crossband-ref-filter`` wins (must be one of the run's
    filters).  Otherwise auto-select the reddest BROAD/MEDIUM band (filter not
    ending in 'N'); fall back to the reddest band overall.  The reddest
    broad/medium band is the most complete, highest-S/N detection list, so it
    anchors the merged coordinate frame best (sickle -> F480M).
    """
    explicit = (getattr(options, 'manual_crossband_ref_filter', '') or '').strip()
    names = list(filternames)
    if explicit:
        match = [f for f in names if f.upper() == explicit.upper()]
        if not match:
            raise ValueError(
                f"--manual-crossband-ref-filter={explicit!r} is not one of the "
                f"run filters {names}")
        return match[0]

    def _wl(f):
        digits = ''.join(ch for ch in f if ch.isdigit())[:3]
        return int(digits) if digits else 0
    broad = [f for f in names if not f.upper().endswith('N')]
    pool = broad if broad else names
    pick = max(pool, key=_wl)
    print(f"[crossband] ref_filter auto-selected: {pick} (reddest "
          f"{'broad/medium' if broad else 'band'} of {names}); override with "
          f"--manual-crossband-ref-filter", flush=True)
    return pick


def run_manual_pipeline(options, modules, filternames, nvisits, proposal_id,
                        target, field, basepath, crowdsource_default_kwargs,
                        bg_boxsizes):
    """In-process manual-iteration cutout pipeline (parallels
    ``_run_cutout_pipeline``).  Phases m12 -> m3 -> m4 -> m5 -> m6 (-> m7 if
    multifilter); BASIC-only.  m12 = per-frame iter1+iter2.  The i2d-detection
    co-add gets progressively cleaner (PSFPhotometryPlan2026-06-09):
      m3 (iter3): daofind(raw i2d),                 fit RAW frames
      m4 (iter4): daofind(residual i2d),            fit RAW frames; builds 1st bg
      m5 (iter5): daofind(residual i2d - bg),       fit bg-subtracted frames
      m6 (iter6): daofind(residual i2d - bg),       fit bg-subtracted frames
      m7 (iter7): cross-filter seed,                fit bg-subtracted frames
    After each phase: merge per-frame catalogs, tag iteration-found provenance,
    vet (extended-emission filter), build the vetted mergedcat residual i2d, and
    smooth it into the background map fed to the next phase.
    """
    import copy
    from astropy.coordinates import SkyCoord
    from jwst_gc_pipeline.photometry import merge_catalogs as _merge_catalogs

    cut_bp = _L._cutout_out_basepath(basepath, options)
    os.makedirs(os.path.join(cut_bp, 'catalogs'), exist_ok=True)
    pupil = 'clear'
    multifilter = len(filternames) > 1
    phases = ['m12', 'm3', 'm4', 'm5', 'm6']
    if multifilter:
        phases.append('m7')

    # MIRI tuning (2026-06-13).  MIRI imaging sits on dominant, structured
    # extended emission with low star/background contrast, so it needs a
    # different threshold schedule from NIRCam (same pipeline, small tweaks):
    #   (1) no cross-filter extra-deep search (drop m7) -- the bright extended
    #       background makes the union-of-filters seed unreliable;
    #   (2) higher detection thresholds in the early raw-image rounds
    #       (m12/m3/m4) so only high-confidence sources enter;
    #   (3) more aggressive (lower) thresholds on the background-subtracted
    #       rounds (m5/m6), where point sources finally stand out;
    #   (4) relaxed qfit vetting (extended background degrades qfit).
    # Applied per-phase to opts_phase below; auto-on for all-MIRI runs unless
    # --no-miri-tuning is given.
    is_miri = all(_L._instrument_from_filter(f) == 'MIRI' for f in filternames)
    miri_tuning = is_miri and not getattr(options, 'no_miri_tuning', False)
    if miri_tuning and 'm7' in phases:
        phases.remove('m7')   # tweak (1)

    print(f"MANUAL PIPELINE: phases={phases} filters={filternames} "
          f"modules={modules} miri_tuning={miri_tuning}", flush=True)

    # Extended-emission-dominated NIRCam fields (W51/Sickle/WD2): bright PAH/dust
    # nebulosity -- especially at long wavelengths -- spawns many spurious
    # DAOStarFinder detections that explode the source count and over-subtract.
    # Default the structure-noise prune to (1, 2): validated on the W51-main
    # F405N cutout to remove ~15% of detections (preferentially in bright
    # emission, dropped/kept SB ratio ~1.85) at ~zero net real-source loss vs the
    # SW F210M catalog.  Explicit --manual-struct-noise-x/-y override this.  MIRI
    # is skipped (its per-phase miri_tuning schedule sets these itself).
    EXTENDED_EMISSION_TARGETS = ('w51', 'sickle', 'wd2')
    if (not miri_tuning and str(target).lower() in EXTENDED_EMISSION_TARGETS
            and float(getattr(options, 'struct_x', 0.0)) == 0.0
            and float(getattr(options, 'struct_y', 0.0)) == 0.0):
        options.struct_x = 1.0
        options.struct_y = 2.0
        print(f"  [{target}] extended-emission NIRCam field: defaulting "
              f"structure-noise prune to struct_x=1.0 struct_y=2.0 "
              f"(override with --manual-struct-noise-x/-y).", flush=True)

    def _merged_path(label, module, filt, resbgsub):
        desat = '_unsatstar' if options.desaturated else ''
        bgsub = ('_bgsub' if options.bgsub else '') + ('_resbgsub' if resbgsub else '')
        blur_ = '_blur' if options.blur else ''
        return (f'{cut_bp}/catalogs/{filt.lower()}_{module}_indivexp_merged'
                f'{desat}{bgsub}{blur_}_{label}_dao_basic.fits')

    def _data_i2d_path(module, filt):
        return (f'{cut_bp}/{filt}/pipeline/jw0{proposal_id}-o{field}_t001_'
                f'{_L._inst_token(filt)}_{pupil}-{filt.lower()}-{module}_data_i2d.fits')

    frame_cache = {}
    bg_for_next = {}      # (module, filt) -> smoothed-bg path for the next phase
    resid_i2d_for_next = {}  # (module, filt) -> mergedcat residual i2d (detection image)
    prev_merged_for = {}  # (module, filt) -> (SkyCoord, iter_found array) for provenance
    # (module, filt) -> [(SkyCoord, flux)] cross-frame reconciled out-of-field
    # satstar fluxes, computed from m12's per-frame satstar catalogs and forwarded
    # to every later phase so far-detector (spike-less) fits are pinned to the
    # nearest-detector flux instead of mis-fitting scattered light.
    satstar_overrides = {}
    # (module, filt) -> [SkyCoord] out-of-field satstars that disagreed across
    # frames but had no detector diversity (single detector, spikes never in
    # FOV).  No trustworthy flux exists, so these are DROPPED (skipped, not
    # contributed) on every later phase to avoid a fake-background model.
    satstar_drops = {}
    overlap_total = 0

    # Optional partial start (e.g. an m7-only "finalize" job that reuses the
    # m12..m6 products written by per-filter jobs, so the big multifilter run can
    # be split into small queue-friendly per-filter jobs + one finalize).  The
    # ONLY cross-phase state a later phase consumes is the background map
    # (bg_for_next); reconstruct it from the previous phase's on-disk smoothed-bg
    # i2d.  m7's cross-band seed reads the previous VETTED catalogs from disk, and
    # frames re-scan because the start phase becomes phases[0].  satstar overrides
    # from m12's reconcile are not reconstructed, but the off-FOV path now uses
    # the Spitzer-prior _spitzer.reg + the model<=data clamp, which load fresh
    # each phase, so off-FOV handling is unaffected.
    # --- per-frame fan-out controls (option C) ---------------------------------
    # These let one phase run as N independent per-exposure SLURM tasks
    # (--manual-skip-finalize) followed by a single barrier job
    # (--manual-finalize-only), chained across phases.  All default-off, so a
    # monolithic run is byte-for-byte unchanged.
    stop_after = (getattr(options, 'manual_stop_after_phase', '') or '').strip()
    skip_finalize = bool(getattr(options, 'manual_skip_finalize', False))
    finalize_only = bool(getattr(options, 'manual_finalize_only', False))
    if skip_finalize and finalize_only:
        raise ValueError("--manual-skip-finalize and --manual-finalize-only are "
                         "mutually exclusive.")
    _shard = (getattr(options, 'manual_frame_shard', '') or '').strip()
    shard_i, shard_n = 0, 1
    if _shard:
        try:
            shard_i, shard_n = (int(x) for x in _shard.split('/'))
        except (ValueError, TypeError):
            raise ValueError(f"--manual-frame-shard must be 'I/N'; got {_shard!r}")
        if not (shard_n >= 1 and 0 <= shard_i < shard_n):
            raise ValueError(f"--manual-frame-shard 'I/N' needs N>=1 and 0<=I<N; "
                             f"got {_shard!r}")
    sharded = shard_n > 1
    # Per-frame completion markers: a fan-out worker writes one per (frame,phase)
    # on success; the finalize job verifies every candidate frame has its marker
    # before merging (so a silently dropped exposure HARD-CRASHES, never slips
    # through -- mirrors the in-memory frame-completeness guard).
    _marker_dir = os.path.join(cut_bp, 'catalogs', '_perframe_markers')
    if skip_finalize or finalize_only:
        os.makedirs(_marker_dir, exist_ok=True)

    def _marker_path(filename, module, filt, phase, kind='ok'):
        # kind: 'ok' (fit produced output) or 'nooverlap' (legit cutout miss).
        return os.path.join(
            _marker_dir,
            f'{os.path.basename(filename)}.{filt.lower()}.{module}.{phase}.{kind}')

    orig_last_phase = phases[-1]

    start_phase = (getattr(options, 'manual_start_phase', '') or '').strip()
    if start_phase:
        if start_phase not in phases:
            raise ValueError(f"--manual-start-phase={start_phase!r} not in phases "
                             f"{phases} (multifilter={multifilter})")
        _si = phases.index(start_phase)
        if _si > 0:
            _prev = phases[_si - 1]
            _prev_label = 'm2' if _prev == 'm12' else _prev
            _prev_resbgsub = _prev in ('m5', 'm6', 'm7')
            for module in modules:
                for filt in filternames:
                    _bg = _reconstruct_smoothed_bg_path(
                        cut_bp, proposal_id, field, module, filt, _prev, options, pupil)
                    if not os.path.exists(_bg):
                        raise FileNotFoundError(
                            f"--manual-start-phase={start_phase}: required {_prev} "
                            f"smoothed-bg for {filt}/{module} is missing (expected "
                            f"{_bg}).  Run the earlier per-filter phases first.")
                    bg_for_next[(module, filt)] = _bg
                    # mergedcat residual i2d (detection image for m4..m6 seed)
                    _ri = _reconstruct_resid_i2d_path(
                        cut_bp, proposal_id, field, module, filt, _prev, options, pupil)
                    if os.path.exists(_ri):
                        resid_i2d_for_next[(module, filt)] = _ri
                    # iter_found provenance from the prior merged catalog
                    _pm = _reconstruct_prev_merged(
                        _merged_path(_prev_label, module, filt, _prev_resbgsub))
                    if _pm is not None:
                        prev_merged_for[(module, filt)] = _pm
                    # m12-reconciled out-of-FOV satstar overrides/drops
                    _ov, _dr = _reconstruct_reconciled_satstars(
                        _satstar_reconciled_path(cut_bp, module, filt))
                    if _ov:
                        satstar_overrides[(module, filt)] = _ov
                    if _dr:
                        satstar_drops[(module, filt)] = _dr
                    print(f"manual [start={start_phase}]: reused {_prev} state for "
                          f"{filt}/{module} (bg, resid_i2d={'y' if os.path.exists(_ri) else 'n'}, "
                          f"prev_merged={'y' if _pm is not None else 'n'}, "
                          f"satstar_ovr={len(_ov)}, satstar_drop={len(_dr)})", flush=True)
        phases = phases[_si:]
        print(f"MANUAL PIPELINE: partial start at {start_phase}; phases now "
              f"{phases}", flush=True)

    if stop_after:
        if stop_after not in phases:
            raise ValueError(f"--manual-stop-after-phase={stop_after!r} not in "
                             f"remaining phases {phases}")
        phases = phases[:phases.index(stop_after) + 1]
        print(f"MANUAL PIPELINE: stop after {stop_after}; phases now {phases}",
              flush=True)

    for phase in phases:
        # iter3 (m3) and iter4 (m4) fit the RAW frames; m4 produces the first
        # background map.  Background subtraction in the fit begins at iter5 (m5).
        resbgsub = phase in ('m5', 'm6', 'm7')
        merge_label = 'm2' if phase == 'm12' else phase
        opts_phase = copy.copy(options)
        opts_phase.iteration_label = merge_label
        opts_phase.seed_catalog = ''
        opts_phase.use_iter3_residual_bg = resbgsub

        if miri_tuning:
            # tweak (2): raise thresholds on the raw-image rounds (m12/m3/m4)
            # tweak (3): lower them on the background-subtracted rounds (m5/m6)
            # tweak (4): relax qfit vetting everywhere for MIRI
            base_first = float(getattr(options, 'local_snr_threshold', 5.0))
            base_iter2 = float(getattr(options, 'manual_iter2_local_snr', 3.0))
            base_extsnr = float(getattr(options, 'manual_ext_local_snr_min', 5.0))
            base_qfit = float(getattr(options, 'manual_ext_qfit_max', 0.2))
            if phase in ('m12', 'm3', 'm4'):
                opts_phase.local_snr_threshold = max(base_first, 8.0)
                opts_phase.manual_iter2_local_snr = max(base_iter2, 6.0)
                opts_phase.manual_ext_local_snr_min = max(base_extsnr, 8.0)
                # high-purity structure-noise prune on the raw rounds
                opts_phase.struct_x = 5.0
                opts_phase.struct_y = 8.0
                # coarse-bg detection: the raw rounds detect on a 51px-median-
                # subtracted image so bright stars on the very high (but smooth)
                # MIRI background are not lost to the global pedestal.  51px beats
                # 101px (median peak/ERR 35.7 vs 18.5); captures 24/34
                # hand-selected unsaturated stars, the rest via the satstar path.
                opts_phase.coarse_bg_box = 51
            elif phase in ('m5', 'm6'):
                opts_phase.manual_iter2_local_snr = min(base_iter2, 3.0)
                opts_phase.manual_ext_local_snr_min = min(base_extsnr, 3.0)
                # background already subtracted -> structure noise is lower;
                # relax the prune to recover faint point sources (tweak 3)
                opts_phase.struct_x = 3.0
                opts_phase.struct_y = 4.0
                # bg already subtracted & trustworthy -> no coarse-bg detection
                opts_phase.coarse_bg_box = 0
            opts_phase.manual_ext_qfit_max = max(base_qfit, 0.4)
            print(f"  [miri_tuning {phase}] first_snr="
                  f"{getattr(opts_phase,'local_snr_threshold',5.0)} iter2_snr="
                  f"{opts_phase.manual_iter2_local_snr} ext_snr_min="
                  f"{opts_phase.manual_ext_local_snr_min} qfit_max="
                  f"{opts_phase.manual_ext_qfit_max}", flush=True)

        for module in modules:
            for filt in filternames:
                prev_seed = None
                resbg_path = None
                # Vetted catalog is PER-OBS tokened (_o{field}) -- each obs vetted
                # vs its own data_i2d, then combined into the un-tokened all-obs
                # catalog (see the vet + combine block below).  Enabled for:
                #   - MIRI multi-obs targets (cloudef obs2+5), and
                #   - gc2211 (prop 2211): 5 NIRCam pointings share ONE basepath and
                #     reuse visit/vgroup/exp tuples; without per-obs vetting the
                #     all-obs merge would pool every obs and a single vetting pass
                #     would carry sources outside each obs's footprint.  Pairs with
                #     the _o{field} per-frame catalog token + the _o* all-obs merge
                #     glob in merge_catalogs (see obs_token()).
                # Single-obs NIRCam targets keep _vtok='' (unchanged behavior).
                _miri_field = (module == 'mirimage'
                               or _L._instrument_from_filter(filt) == 'MIRI')
                _multiobs = str(proposal_id) == '2211'
                _vtok = f'_o{field}' if (_miri_field or _multiobs) else ''
                # gc2211: the COMBINED (post-vet) catalog is also per-obs (no cross-
                # obs vstack).  MIRI: combined stays un-tokened (all-obs).
                _combsuf = f'_o{field}' if _multiobs else ''
                # m3..m6 seed = vetted previous catalog UNION daofind on a
                # progressively cleaner i2d (per PSFPhotometryPlan2026-06-09):
                #   iter3(m3): raw i2d                        fit RAW frames
                #   iter4(m4): residual i2d (m3)              fit RAW frames
                #   iter5(m5): residual i2d (m4) - bg (m4)    fit bg-sub frames
                #   iter6(m6): residual i2d (m5) - bg (m5)    fit bg-sub frames
                #   iter7(m7): cross-filter seed             fit bg-sub frames
                # resid_i2d_for_next/bg_for_next hold the PRIOR phase's products.
                vetted_prev = None
                det_i2d = None
                bg_sub = None
                if phase == 'm3':
                    vetted_prev = _merged_path('m2', module, filt, False).replace('.fits', f'{_vtok}_vetted.fits')
                    det_i2d = _data_i2d_path(module, filt)          # raw i2d
                    resbg_path = None                                # RAW frames
                elif phase == 'm4':
                    vetted_prev = _merged_path('m3', module, filt, False).replace('.fits', f'{_vtok}_vetted.fits')
                    det_i2d = resid_i2d_for_next.get((module, filt))  # m3 residual
                    resbg_path = None                                # RAW frames
                elif phase == 'm5':
                    vetted_prev = _merged_path('m4', module, filt, False).replace('.fits', f'{_vtok}_vetted.fits')
                    det_i2d = resid_i2d_for_next.get((module, filt))  # m4 residual
                    bg_sub = bg_for_next.get((module, filt))          # minus m4 bg
                    resbg_path = bg_for_next.get((module, filt))      # fit on bg-sub frames
                elif phase == 'm6':
                    vetted_prev = _merged_path('m5', module, filt, True).replace('.fits', f'{_vtok}_vetted.fits')
                    det_i2d = resid_i2d_for_next.get((module, filt))  # m5 residual
                    bg_sub = bg_for_next.get((module, filt))          # minus m5 bg
                    resbg_path = bg_for_next.get((module, filt))      # fit on bg-sub frames
                elif phase == 'm7':
                    prev_seed = _build_crossband_seed(cut_bp, modules, filternames, options)
                    resbg_path = bg_for_next.get((module, filt))      # bg from m6

                if phase in ('m3', 'm4', 'm5', 'm6'):
                    det_i2d = det_i2d or _data_i2d_path(module, filt)  # fallback
                    try:
                        prev_seed = _build_i2d_augmented_seed(
                            det_i2d, vetted_prev, filt,
                            local_snr_min=float(getattr(
                                opts_phase, 'manual_ext_local_snr_min', 5.0)),
                            bg_subtract_path=bg_sub,
                            # Structure-noise prune / coarse-bg detection.  For
                            # MIRI the miri_tuning block sets opts_phase.struct_x/y
                            # and coarse_bg_box per phase (5/8 & 51 on raw rounds,
                            # 3/4 on bg-subtracted rounds).  For NIRCam they come
                            # straight from options (the --manual-struct-noise-x/-y
                            # and --manual-coarse-bg-box flags), defaulting to 0 so
                            # the default NIRCam pipeline is unchanged; set them to
                            # enable the prune on extended-emission fields.
                            struct_x=float(getattr(opts_phase, 'struct_x', 0.0)),
                            struct_y=float(getattr(opts_phase, 'struct_y', 0.0)),
                            coarse_bg_box=int(getattr(opts_phase, 'coarse_bg_box', 0)),
                            label=f'{phase}:{filt}')
                    except Exception as ex:
                        print(f"manual [{phase}]: i2d-augmented seed failed ({ex}); "
                              f"using {os.path.basename(vetted_prev)}", flush=True)
                        prev_seed = vetted_prev

                # candidate frames (scan on first phase, cache thereafter)
                if phase == phases[0]:
                    candidate_frames = []
                    for visitid in range(1, nvisits[proposal_id][target] + 1):
                        candidate_frames.extend(sorted(_L.get_filenames(
                            basepath, filt, proposal_id, field,
                            visitid=f'{visitid:03d}',
                            each_suffix=_resolve_each_suffix(options, filt),
                            module=module, pupil='clear', allow_empty=True)))
                else:
                    candidate_frames = frame_cache.get((module, filt), [])

                # MIRI FAKE-STAR FIX (2026-06-23): the satstar seed gate's
                # strongest, FIELD-GENERAL phantom rejection (coadd prominence /
                # core / concentration -- RELATIVE metrics that separate a fake on
                # smooth emission from a real saturated star regardless of field
                # brightness) needs the deep detection coadd ``_data_i2d``.  On a
                # field's FIRST run that coadd did not exist yet (it used to be
                # built only by the later mosaic step), so the gate fell back to
                # per-frame data and FAKE bright satstars survived (2526 cloud-c
                # filament: satstar flux 4.4e6 on data ~340; prom 1.6/conc 1.2 on
                # the coadd would have rejected it outright).  Build the coadd
                # ONCE up front from the input frames so EVERY per-frame satstar
                # fit gets the coadd gate.  Cheap vs the fits; skipped if present.
                if (phase == phases[0] and module == 'mirimage'
                        and candidate_frames):
                    _det_i2d = _data_i2d_path(module, filt)
                    if not os.path.exists(_det_i2d):
                        try:
                            _L.mosaic_cutout_input_data(
                                basepath, filt, proposal_id, field, module,
                                label=phase, pupil='clear',
                                input_files=candidate_frames)
                            print(f"[manual] built detection coadd for satstar "
                                  f"gate: {_det_i2d}", flush=True)
                        except Exception as _cex:
                            print(f"[manual] could not build detection coadd "
                                  f"{_det_i2d} ({_cex}); satstar gate falls back "
                                  f"to per-frame data", flush=True)

                # Build per-frame work units.  Per-frame fits within a phase
                # are independent (phase-level dependencies only); parallelize
                # them across N workers when ``--parallel-workers > 1``.
                frame_args = []
                for filename in candidate_frames:
                    exposure_id = filename.split("_")[2]
                    visit_id = filename.split("_")[0][-3:]
                    vgroup_id = filename.split("_")[1]
                    file_detector = filename.split("_")[3]
                    # JOINT multi-obs runs (field like '002-998'): two observations
                    # can share visit+vgroup+exposure -- e.g. sgrb2 obs998 ("redo")
                    # reused obs002's mosaic tile numbers, so both map to tile 02101
                    # visit 001.  The per-frame output identity is named by the
                    # joint FIELD token (o002-998 for both), so the obs is otherwise
                    # lost and the two frames collide (overwrite).  Fold the per-frame
                    # observation number into vgroup_id so the name + collision key
                    # stay unique; the merge globs vgroup* so it still finds both.
                    # Single-obs runs (no '-') are unchanged.
                    if '-' in str(field):
                        obs_id = filename.split("_")[0][-6:-3]
                        vgroup_id = f'{obs_id}{vgroup_id}'
                    # Per-frame products MUST be named by the actual DETECTOR, never
                    # the (coarser) requested module.  Otherwise SW exposures whose
                    # 4 detectors (nrcb1-4) share visit+vgroup+exp collapse to one
                    # filename (module='nrcb') and 3 of 4 are silently overwritten,
                    # holing the residual/model mosaics and dropping detections from
                    # the merge.  The merged-module products downstream still glob
                    # all detector variants for module='nrcb'.
                    file_module = file_detector
                    frame_args.append({
                        'options': opts_phase, 'filtername': filt,
                        'module': file_module, 'detector': file_detector,
                        'field': field, 'basepath': basepath,
                        'filename': filename, 'proposal_id': proposal_id,
                        'manual_phase': phase,
                        'exposurenumber': int(exposure_id),
                        'visit_id': visit_id, 'vgroup_id': vgroup_id,
                        'bg_boxsizes': bg_boxsizes,
                        'use_webbpsf': True, 'pupil': pupil,
                        'prev_seed_catalog': prev_seed,
                        'resbg_path': resbg_path,
                        'satstar_flux_overrides': satstar_overrides.get((module, filt)),
                        'satstar_flux_drops': satstar_drops.get((module, filt)),
                    })

                # Collision guard: every frame's per-frame output identity
                # (visit+vgroup+exp+detector) MUST be unique, else two frames write
                # the same file and one is silently overwritten (= data loss).
                _ident = {}
                for a in frame_args:
                    k = (a['visit_id'], a['vgroup_id'], a['exposurenumber'], a['module'])
                    _ident.setdefault(k, []).append(a['filename'])
                _collisions = {k: v for k, v in _ident.items() if len(v) > 1}
                if _collisions:
                    _msg = '\n'.join(
                        f"    {k} <- " + ', '.join(os.path.basename(f) for f in v)
                        for k, v in _collisions.items())
                    raise RuntimeError(
                        f"manual [{phase}] {filt}/{module}: per-frame output name "
                        f"COLLISION -- {len(_collisions)} (visit,vgroup,exp,detector) "
                        f"keys map to >1 input frame; outputs would silently "
                        f"overwrite each other.  Aborting:\n{_msg}")

                max_workers = max(1, int(getattr(options, 'parallel_workers', 1) or 1))
                _is_cutout = bool(getattr(options, 'cutout_region', ''))
                overlapping_now = []
                no_overlap = []   # frames that legitimately miss a cutout region
                failures = []     # (filename, err) -- ANY of these aborts the run
                # Full candidate set (pre-shard) for the finalize completeness check.
                all_frames = [a['filename'] for a in frame_args]

                if finalize_only:
                    # Barrier job: do NOT fit.  Require every candidate frame to
                    # carry a completion marker written by a fan-out worker; a
                    # missing marker = a silently dropped exposure -> HARD-CRASH.
                    _missing_marker = []
                    for fn in all_frames:
                        _det = fn.split('_')[3]   # per-frame products keyed by detector
                        if os.path.exists(_marker_path(fn, _det, filt, phase, 'ok')):
                            overlapping_now.append(fn)
                        elif os.path.exists(_marker_path(fn, _det, filt, phase, 'nooverlap')):
                            no_overlap.append((fn, 'no-overlap (marker)'))
                        else:
                            _missing_marker.append(fn)
                    if _missing_marker:
                        raise RuntimeError(
                            f"manual [{phase}] {filt}/{module}: --manual-finalize-only "
                            f"found {len(_missing_marker)} candidate frame(s) with NO "
                            f"completion marker -- the per-frame fan-out did not finish "
                            f"them (a dropped exposure corrupts the catalog).  Re-run "
                            f"the missing shard(s):\n"
                            + '\n'.join(f"    {os.path.basename(m)}" for m in sorted(_missing_marker)))
                    print(f"manual [{phase}] {filt}/{module}: finalize-only verified "
                          f"{len(overlapping_now)} frame markers "
                          f"({len(no_overlap)} no-overlap)", flush=True)
                else:
                    # Fan-out / monolithic FIT.  In sharded mode keep only this
                    # task's slice of frames (index % shard_n == shard_i).
                    if sharded:
                        frame_args = [a for j, a in enumerate(frame_args)
                                      if j % shard_n == shard_i]
                        print(f"manual [{phase}] {filt}/{module}: frame-shard "
                              f"{shard_i}/{shard_n} -> {len(frame_args)} of "
                              f"{len(all_frames)} frames", flush=True)

                    def _on_result(filename, ok, err):
                        if ok:
                            overlapping_now.append(filename)
                            if skip_finalize or finalize_only:
                                open(_marker_path(filename, filename.split('_')[3],
                                                  filt, phase, 'ok'), 'w').close()
                        elif err and err.startswith('no-overlap'):
                            no_overlap.append((filename, err))
                            if skip_finalize or finalize_only:
                                open(_marker_path(filename, filename.split('_')[3],
                                                  filt, phase, 'nooverlap'), 'w').close()
                        else:
                            failures.append((filename, err))

                    if max_workers > 1 and len(frame_args) > 1:
                        n_workers = min(max_workers, len(frame_args))
                        print(f"manual [{phase}]: fitting {len(frame_args)} frames "
                              f"with {n_workers} parallel workers", flush=True)
                        with ProcessPoolExecutor(max_workers=n_workers) as ex:
                            futures = {ex.submit(_run_one_frame_manual, a): a['filename']
                                       for a in frame_args}
                            for fut in as_completed(futures):
                                filename, ok, err = fut.result()
                                _on_result(filename, ok, err)
                    else:
                        for a in frame_args:
                            filename, ok, err = _run_one_frame_manual(a)
                            _on_result(filename, ok, err)

                # HARD-CRASH on ANY frame failure.  Silently dropping even one
                # exposure from any phase irrecoverably corrupts every later
                # step and the final catalog/mosaic (missing-data holes), so a
                # failed frame must abort the whole run -- never be skipped.
                if failures:
                    _msg = '\n'.join(f"    {os.path.basename(fn)}: {er}"
                                     for fn, er in failures)
                    raise RuntimeError(
                        f"manual [{phase}] {filt}/{module}: {len(failures)} frame(s) "
                        f"FAILED -- aborting (a dropped exposure corrupts all later "
                        f"phases and the final catalog).  Fix the cause and re-run:\n"
                        f"{_msg}")
                # A non-overlapping frame is legitimate ONLY for a cutout region.
                # In a full-frame run every input exposure must overlap and be fit.
                if no_overlap and not _is_cutout:
                    _msg = '\n'.join(f"    {os.path.basename(fn)}: {er}"
                                     for fn, er in no_overlap)
                    raise RuntimeError(
                        f"manual [{phase}] {filt}/{module}: {len(no_overlap)} frame(s) "
                        f"reported NO OVERLAP in a full-frame run -- every input "
                        f"exposure must be processed.  Aborting:\n{_msg}")

                if phase == phases[0]:
                    frame_cache[(module, filt)] = overlapping_now
                    overlap_total += len(overlapping_now)
                else:
                    # Every later phase MUST process the exact same set of frames
                    # established in the first phase.  A frame that succeeded then
                    # but is missing now means a silent mid-pipeline drop -> abort.
                    _expected = set(frame_cache.get((module, filt), []))
                    _got = set(overlapping_now)
                    _missing = _expected - _got
                    if _missing:
                        raise RuntimeError(
                            f"manual [{phase}] {filt}/{module}: {len(_missing)} frame(s) "
                            f"processed in {phases[0]} did NOT produce output in {phase} "
                            f"-- a mid-pipeline drop corrupts the catalog.  Aborting:\n"
                            + '\n'.join(f"    {os.path.basename(m)}" for m in sorted(_missing)))
                if not overlapping_now:
                    # An empty slice is legitimate ONLY for a fan-out shard with
                    # more tasks than frames; that worker simply has nothing to do.
                    if sharded and skip_finalize:
                        print(f"manual [{phase}] {filt}/{module}: frame-shard "
                              f"{shard_i}/{shard_n} has no frames; nothing to fit.",
                              flush=True)
                        continue
                    if _is_cutout:
                        raise ValueError(
                            f"--cutout-region overlapped none of the {filt}/{module} "
                            f"frames in phase {phase}.")
                    raise ValueError(
                        f"no {filt}/{module} frames produced output in phase {phase}.")

                # Fan-out worker: the (sharded) per-frame fits are done and their
                # completion markers written.  Skip the per-phase barrier
                # (reconcile/merge/vet/residual/bg); a single --manual-finalize-only
                # job runs it once every shard has finished.
                if skip_finalize:
                    continue

                # --- cross-frame out-of-field satstar reconciliation (after m12) ---
                # m12 fit each frame's out-of-field bright stars independently;
                # gather those per-frame satstar catalogs and pick, per star, the
                # flux from the detector whose footprint is CLOSEST (it caught the
                # diffraction spikes).  Forward as overrides to m3..m7.
                if phase == 'm12':
                    from jwst_gc_pipeline.reduction.saturated_star_finding import (
                        reconcile_outside_fov_satstar_fluxes)
                    _ss_suffix = f'{_bgsub_token(opts_phase)}{_iteration_token("m12")}'
                    _per_frame = []
                    for _fr in overlapping_now:
                        _sp = _fr.replace('.fits', f'{_ss_suffix}_satstar_catalog.fits')
                        # A missing satstar catalog is legitimate (the frame had no
                        # saturated stars).  A catalog that EXISTS but can't be read
                        # is corruption -- do not swallow it.
                        if not os.path.exists(_sp):
                            continue
                        _per_frame.append((_fr, Table.read(_sp)))
                    try:
                        _ovr, _drp = reconcile_outside_fov_satstar_fluxes(
                            _per_frame,
                            match_radius=float(getattr(
                                options, 'satstar_reconcile_radius_arcsec', 1.0)) * u.arcsec,
                            disagree_factor=float(getattr(
                                options, 'satstar_reconcile_disagree_factor', 2.0)),
                            min_detector_diversity=float(getattr(
                                options, 'satstar_reconcile_min_diversity_arcsec',
                                30.0)) * u.arcsec)
                    except Exception as _ex:
                        print(f"manual [m12]: satstar reconciliation failed: {_ex}",
                              flush=True)
                        _ovr, _drp = [], []
                    if _ovr:
                        satstar_overrides[(module, filt)] = _ovr
                        print(f"manual [m12]: reconciled {len(_ovr)} out-of-field "
                              f"satstar flux(es) for {filt}/{module}; will pin on "
                              f"m3..m7", flush=True)
                    if _drp:
                        satstar_drops[(module, filt)] = _drp
                        print(f"manual [m12]: DROPPING {len(_drp)} out-of-field "
                              f"satstar(s) for {filt}/{module} (no detector "
                              f"diversity -> no trustworthy flux); will skip on "
                              f"m3..m7", flush=True)
                    # Persist the reconciled overrides/drops so a per-frame m3..m7
                    # fan-out (a fresh process) can reconstruct them from disk; in a
                    # monolithic run this file is simply an extra (harmless) artifact.
                    try:
                        _persist_reconciled_satstars(
                            _satstar_reconciled_path(cut_bp, module, filt), _ovr, _drp)
                    except Exception as _pex:
                        print(f"manual [m12]: persisting reconciled satstars failed "
                              f"for {filt}/{module}: {_pex}", flush=True)

                # merge per-frame catalogs (BASIC only)
                _merge_catalogs.merge_individual_frames(
                    module=module, filtername=filt.lower(), progid=proposal_id,
                    method='dao', suffix='_basic', target=target, basepath=cut_bp,
                    iteration_label=merge_label, bgsub=options.bgsub,
                    desat=options.desaturated, epsf=options.epsf, blur=options.blur,
                    resbgsub=resbgsub, group=getattr(options, 'group', False),
                    fwhm_basepath=basepath,
                    # parallelize the otherwise-serial dense-field merge: pass the
                    # worker count so combine auto-spatial-chunks when the source
                    # volume is large (>1M detections).
                    n_spatial_chunks=int(getattr(options, 'merge_spatial_chunks', 1) or 1),
                    merge_workers=max(1, int(getattr(options, 'parallel_workers', 1) or 1)))

                # data i2d once (m12), for peak-SB in the vetting step.  Cutout
                # runs resample the per-frame crops (globbed by label); full-frame
                # runs resample the original overlapping frames passed explicitly.
                # Gated on the actual m12 (not phases[0]) so a single-phase finalize
                # job for a LATER phase reuses m12's data i2d instead of rebuilding
                # it; monolithically phases[0] IS m12, so behaviour is unchanged.
                if phase == 'm12':
                    try:
                        if getattr(options, 'cutout_region', ''):
                            _L.mosaic_cutout_input_data(
                                cut_bp, filt, proposal_id, field, module,
                                _L._cutout_label_for(options), pupil=pupil)
                        else:
                            _L.mosaic_cutout_input_data(
                                cut_bp, filt, proposal_id, field, module, '',
                                pupil=pupil,
                                input_files=frame_cache.get((module, filt), []))
                    except Exception as ex:
                        print(f"manual: data i2d build failed: {ex}", flush=True)

                # vet the merged catalog -> per-obs tokened _o{field}_vetted.fits.
                # The merged catalog is all-obs (merge globs all frames); the
                # VETTED is per-obs (each vetted vs its own data_i2d) so footprints
                # don't overwrite and off-i2d sources aren't carried unvetted.
                # A combine step (below) vstacks per-obs vetted -> un-tokened
                # all-obs _vetted.fits (the final science catalog + m7 seed).
                merged_path = _merged_path(merge_label, module, filt, resbgsub)
                vetted_path = merged_path.replace('.fits', f'{_vtok}_vetted.fits')
                # gc2211 (prop 2211): the "combined" catalog is PER-OBS (each obs is
                # its own target -- do NOT cross-combine obs), so token it _o{field}.
                # MIRI multi-obs keeps the un-tokened all-obs combine (cloudef obs2+5
                # ARE the same field).  _combsuf is set next to _vtok above.
                combined_vetted_path = merged_path.replace('.fits', f'{_combsuf}_vetted.fits')
                try:
                    merged = Table.read(merged_path)
                    # post-merge off-FOV cleanup: (A) one row per off-FOV satstar
                    # (the per-frame fits scatter wider than the 0.15" satstar
                    # dedup), (B) drop NON-satstar rows that fall >5 PSF widths
                    # outside this filter's FOV (m7 cross-band-seed unions all
                    # filters' positions -> off-this-FOV garbage).  See
                    # _clean_offfov_dups_and_offfield.
                    merged, _ndup, _noff = _clean_offfov_dups_and_offfield(
                        merged, filt, _data_i2d_path(module, filt), basepath,
                        dedup_arcsec=float(getattr(options, 'offfov_dedup_arcsec', 1.0)),
                        fov_pad_psf=float(getattr(options, 'offfield_fov_pad_psf', 5.0)))
                    if _ndup or _noff:
                        print(f"manual [{phase}] {filt}/{module}: off-FOV cleanup "
                              f"removed {_ndup} duplicate satstar row(s) + {_noff} "
                              f"off-field cross-band-seed artifact(s) -> n={len(merged)}",
                              flush=True)
                        merged.write(merged_path, overwrite=True)
                    # --- iteration-found provenance: the first phase a source
                    # appears in (matched across phases by sky position) ---
                    _iter_num = {'m2': 2, 'm3': 3, 'm4': 4, 'm5': 5, 'm6': 6, 'm7': 7}[merge_label]
                    _msc = SkyCoord(merged['skycoord'])
                    _ifound = np.full(len(merged), _iter_num, dtype=int)
                    _prev = prev_merged_for.get((module, filt))
                    if _prev is not None and len(merged):
                        _psc, _pif = _prev
                        _ftab = Table.read(_L.FWHM_TABLE)
                        _fw = float(_ftab[_ftab['Filter'] == filt]['PSF FWHM (arcsec)'][0])
                        _idx, _sep, _ = _msc.match_to_catalog_sky(_psc)
                        _m = _sep.arcsec < 0.5 * _fw   # within ~0.5 FWHM = same source
                        _ifound[_m] = np.asarray(_pif)[np.asarray(_idx)][_m]
                    merged['iter_found'] = _ifound
                    merged.write(merged_path, overwrite=True)
                    prev_merged_for[(module, filt)] = (_msc, _ifound)

                    d_i2d, ww_i2d = None, None
                    dpath = _data_i2d_path(module, filt)
                    if os.path.exists(dpath):
                        with fits.open(dpath) as dh:
                            hdu = dh['SCI'] if 'SCI' in [h.name for h in dh] else dh[0]
                            d_i2d = hdu.data.astype(float)
                            ww_i2d = wcs.WCS(hdu.header)
                    # MIRI: required deep-i2d prominence gate kills false emission
                    # sources that pass the qfit OR-branch.  NIRCam: off (0).
                    _miri_field = (module == 'mirimage'
                                   or _L._instrument_from_filter(filt) == 'MIRI')
                    vetted = _filter_extended_emission(
                        merged, data_i2d_image=d_i2d, ww_i2d=ww_i2d,
                        qfit_max=float(getattr(opts_phase, 'manual_ext_qfit_max', 0.2)),
                        peak_over_bkg=float(getattr(opts_phase, 'manual_ext_peak_over_bkg', 20.0)),
                        min_prominence=(float(getattr(opts_phase, 'miri_prominence_snr', 5.0))
                                        if _miri_field else 0.0),
                        local_snr_min=float(getattr(opts_phase, 'manual_ext_local_snr_min', 5.0)),
                        snr_high_keep=float(getattr(opts_phase, 'manual_ext_snr_high_keep', 20.0)),
                        qfit_high_keep_max=float(getattr(opts_phase, 'manual_ext_qfit_high_keep_max', 0.4)),
                        struct_x=0.0, struct_y=0.0,  # prune at detection, not here
                        label=f'{phase}:{filt}')
                    vetted.write(vetted_path, overwrite=True)
                    # MIRI: combine per-obs tokened vetted catalogs into the
                    # un-tokened ALL-OBS vetted (the final science catalog + the
                    # m7 cross-band seed read at _build_crossband_seed).  Globs
                    # every `_o*_vetted` sibling so it's incremental: after o001
                    # it holds o001; after o002 it holds o001+o002 (deduped by
                    # sky position, brighter/first wins in the overlap).  Each
                    # footprint was already cleaned by its own obs' data_i2d.
                    if _vtok:
                        try:
                            import glob as _glob
                            from astropy.table import vstack as _vstack
                            from astropy.coordinates import SkyCoord as _SkyCoord
                            # JOINT run ('-' in field): one vetting pass against
                            # the joint (both-obs) coadd is authoritative -- do
                            # NOT vstack stale per-obs `_o001_vetted`/`_o002_vetted`
                            # siblings from prior separate runs (the `_o*_vetted`
                            # wildcard would catch them and double/triple-count).
                            # JOINT run, or gc2211 (each obs a distinct target):
                            # use ONLY this obs's vetted -- never cross-combine obs.
                            # MIRI multi-obs (cloudef) still vstacks the _o*_vetted
                            # siblings into the all-obs catalog.
                            if '-' in str(field) or _multiobs:
                                _sibs = [vetted_path]
                            else:
                                _sibs = sorted(_glob.glob(
                                    merged_path.replace('.fits', '_o*_vetted.fits')))
                            _tabs = []
                            for _sp in _sibs:
                                try:
                                    _tabs.append(Table.read(_sp))
                                except Exception:
                                    continue
                            if _tabs:
                                _comb = _tabs[0] if len(_tabs) == 1 else _vstack(
                                    _tabs, metadata_conflicts='silent')
                                if len(_tabs) > 1 and 'skycoord' in _comb.colnames:
                                    _csc = _SkyCoord(_comb['skycoord'])
                                    _keepc = _dedup_close_sources(
                                        xy=np.column_stack([_csc.ra.deg, _csc.dec.deg]),
                                        flux=(np.asarray(_comb['flux'], dtype=float)
                                              if 'flux' in _comb.colnames else None),
                                        min_sep_pix=0.11 / 3600.0, quality=None)[0]
                                    _comb = _comb[_keepc]
                                _comb.write(combined_vetted_path, overwrite=True)
                                print(f"manual [{phase}]: combined {len(_sibs)} per-obs "
                                      f"vetted -> {os.path.basename(combined_vetted_path)} "
                                      f"({len(_comb)} all-obs sources)", flush=True)
                                # CARTA-friendly export: the science catalog uses
                                # an astropy SkyCoord MIXIN column (serialized as
                                # dotted ``skycoord.ra``/``skycoord.dec`` with mixin
                                # metadata) that CARTA rejects ("Catalog type not
                                # supported").  Write a sibling with plain float
                                # ra/dec (deg) columns that CARTA loads directly.
                                try:
                                    from astropy.coordinates import SkyCoord as _SC2
                                    _cart = Table()
                                    if 'skycoord' in _comb.colnames:
                                        _sc2 = _SC2(_comb['skycoord'])
                                    else:
                                        _sc2 = _SC2(_comb['ra'], _comb['dec'], unit='deg')
                                    _cart['ra'] = np.asarray(_sc2.ra.deg, dtype='float64')
                                    _cart['dec'] = np.asarray(_sc2.dec.deg, dtype='float64')
                                    for _cc in ('flux', 'flux_err', 'qfit', 'cfit',
                                                'flags', 'is_saturated',
                                                'replaced_saturated', 'iter_found'):
                                        if _cc in _comb.colnames:
                                            _cart[_cc] = np.asarray(_comb[_cc])
                                    _cart.write(combined_vetted_path.replace(
                                        '.fits', '_carta.fits'), overwrite=True)
                                except Exception as _cex2:
                                    print(f"manual [{phase}]: CARTA catalog export "
                                          f"failed: {_cex2}", flush=True)
                        except Exception as _cex:
                            print(f"manual [{phase}]: vetted combine failed: {_cex}",
                                  flush=True)
                except Exception as ex:
                    print(f"manual [{phase}]: vetting failed ({ex}); using unvetted "
                          f"merged catalog as seed", flush=True)
                    vetted_path = merged_path

                # build vetted mergedcat residual i2d, smooth -> bg for next phase
                try:
                    outpaths = _L.build_mergedcat_residuals(
                        cut_bp, basepath, vetted_path, filt, proposal_id, field,
                        module, opts_phase, frame_cache.get((module, filt), []),
                        merge_label, ['basic'], pupil=pupil, satstar_label=phase)
                    mc_i2d = outpaths.get('basic')
                    if mc_i2d and os.path.exists(mc_i2d):
                        # this phase's residual i2d is the detection image for the
                        # next phase's i2d-augmented seed (blended-source recovery)
                        resid_i2d_for_next[(module, filt)] = mc_i2d
                        # source-masked smoothed bg (breaks the bg<-source-hole
                        # feedback loop that drives faint stars negative with
                        # iteration); falls back to the plain smoother on error
                        try:
                            bg_for_next[(module, filt)] = _build_source_masked_bg(
                                mc_i2d, vetted_path, filt,
                                median_size=int(getattr(options, 'manual_residual_bg_median_size', 3)))
                        except Exception as ex:
                            print(f"manual [{phase}]: source-masked bg failed ({ex}); "
                                  f"using plain smoother", flush=True)
                            bg_for_next[(module, filt)] = _L._cutout_smooth_residual_bg(mc_i2d)
                        print(f"manual [{phase}]: smoothed bg for next phase = "
                              f"{bg_for_next[(module, filt)]}", flush=True)
                except Exception as ex:
                    print(f"manual [{phase}]: mergedcat residual / bg build failed: {ex}",
                          flush=True)

    # -------------------------------------------------------------------
    # Cross-band merge (final step, multifilter only).  Port of the legacy
    # merge_catalogs.merge_daophot -> merge_catalogs cross-filter match: one
    # multi-filter photometry table (per-band fluxes/mags matched within
    # max_offset, anchored on ref_filter) from the per-filter m7 VETTED
    # catalogs.  Needs the per-filter reduction i2d mosaics (WCS/pixelscale),
    # which exist for full-frame runs but not cutouts -> cutouts keep the
    # per-filter catalogs + the union seed only.
    # -------------------------------------------------------------------
    last_phase = phases[-1]
    # The cross-band merge belongs ONLY to the job that completes the FINAL phase's
    # barrier: never a fan-out worker (skip_finalize), never a partial-phase job
    # (last_phase != the run's true final phase).  Monolithically both hold, so
    # this is unchanged.
    _do_crossband = (multifilter and not skip_finalize
                     and last_phase == orig_last_phase)
    if multifilter and not _do_crossband:
        print(f"manual [{last_phase}]: cross-band merge SKIPPED (partial/fan-out "
              f"job; runs in the {orig_last_phase} finalize)", flush=True)
    if _do_crossband and not getattr(options, 'cutout_region', ''):
        ref_filter = _resolve_crossband_ref_filter(options, filternames)
        for module in modules:
            print(f"manual [{last_phase}]: CROSS-BAND MERGE (module={module}, "
                  f"ref_filter={ref_filter}, filters={list(filternames)})", flush=True)
            _merge_catalogs.merge_daophot(
                module=module, daophot_type='basic', indivexp=True,
                desat=bool(options.desaturated), bgsub=bool(options.bgsub),
                blur=bool(options.blur), resbgsub=True,
                iteration_label=last_phase, target=target, basepath=cut_bp,
                ref_filter=ref_filter.lower(),
                filternames_override=[f.lower() for f in filternames],
                field=field,
                vetted=True)
            _xbsuf = _L.obs_token(proposal_id, field)
            _xb = (f'{cut_bp}/catalogs/basic_{module}_indivexp_photometry_tables_'
                   f'merged_resbgsub_{last_phase}{_xbsuf}.fits')
            print(f"manual [{last_phase}]: CROSS-BAND MERGE done (module={module}) "
                  f"-> {_xb}", flush=True)
    elif _do_crossband:
        print(f"manual [{last_phase}]: cross-band merge SKIPPED for cutout run "
              f"(no full-frame per-filter i2d mosaics); per-filter {last_phase} "
              f"catalogs + crossband seed only", flush=True)

    print(f"MANUAL PIPELINE DONE: {overlap_total} overlapping frames, "
          f"phases={phases}", flush=True)
