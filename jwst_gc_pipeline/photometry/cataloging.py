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
                       dedup_min_sep_pix=None, label='', apply_snr_filter=True):
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


# ---------------------------------------------------------------------------
# Per-frame frame setup (shared load / bg / PSF / mask / satstar)
# ---------------------------------------------------------------------------
def _prepare_frame_for_photometry(options, filtername, module, field, basepath,
                                  filename, proposal_id, *, exposurenumber,
                                  visit_id, vgroup_id, bg_boxsizes, use_webbpsf,
                                  pupil, resbg_path, satstar_label):
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
            bg_data = bg_hdu.data.astype(float)
        bg_reproj, _ = reproject_interp((bg_data, bg_wcs), ww, shape_out=data.shape)
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

    _max_group_size = int(getattr(options, 'max_group_size', 0) or 0)
    if _max_group_size > 0:
        grouper = _L.CappedSourceGrouper(2 * fwhm_pix, max_size=_max_group_size)
    else:
        grouper = SourceGrouper(2 * fwhm_pix)

    kernel = Gaussian2DKernel(x_stddev=fwhm_pix / 2.355)
    mask = np.isnan(data) | bad
    dqarr = im1['DQ'].data if 'DQ' in im1 else None
    if dqarr is not None:
        is_saturated = (dqarr & _L.dqflags.pixel['SATURATED']) != 0
        data_ = data.copy()
        data_[is_saturated] = np.nan
        mask |= is_saturated
        mask |= (dqarr & _L._bad_dq_bitmask(instrument)) != 0
    else:
        data_ = data
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
    forced_grid_search_radius = 0 if outside_locked else 5
    satstar_file_suffix = f'{bgsub}{_iteration_token(satstar_label)}'
    satstar_table = _L.load_or_make_satstar_catalog(
        filename, path_prefix=f'{basepath}/psfs',
        use_merged_psf_for_merged=(module == 'merged'),
        overwrite=bool(outside_star_pixels),
        outside_star_pixels=outside_star_pixels, outside_star_fit_box=512,
        forced_grid_search_radius=forced_grid_search_radius,
        file_suffix=satstar_file_suffix)
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
                              prev_seed_catalog=None, resbg_path=None):
    """Clean per-frame driver for the manual-iteration path.

    ``manual_phase`` in {'m12','m3','m4','m5'}:
      * 'm12' runs iter1 (unseeded daofind) then iter2 (daofind(residual1) +
        same-frame iter1 catalog), no merge between -- saved as m1, m2.
      * 'm3'/'m4'/'m5' run a single pass on the (bg-subtracted) data, seeded by
        daofind + the projected previous merged/cross-band catalog.
    """
    overshoot_ratio = float(getattr(options, 'manual_overshoot_ratio', 1.2))
    overshoot_action = str(getattr(options, 'manual_overshoot_action', 'refit'))
    iter2_snr = float(getattr(options, 'manual_iter2_local_snr', 3.0))
    first_snr = float(getattr(options, 'local_snr_threshold', 5.0))

    ctx = _prepare_frame_for_photometry(
        options, filtername, module, field, basepath, filename, proposal_id,
        exposurenumber=exposurenumber, visit_id=visit_id, vgroup_id=vgroup_id,
        bg_boxsizes=bg_boxsizes, use_webbpsf=use_webbpsf, pupil=pupil,
        resbg_path=resbg_path, satstar_label=manual_phase)

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
            overshoot_ratio=overshoot_ratio, overshoot_action=overshoot_action)

    if manual_phase == 'm12':
        seed1 = _build_manual_seed(
            detection_image=ctx.nan_replaced_data, nan_replaced_data=ctx.nan_replaced_data,
            mask=ctx.mask, ww=ctx.ww, fwhm_pix=ctx.fwhm_pix,
            satstar_table=ctx.satstar_table, prev_catalog=None,
            local_snr_threshold=first_snr, roundlo=-1.0, roundhi=1.0,
            sharplo=0.30, sharphi=1.40, dedup_min_sep_pix=0.5 * ctx.fwhm_pix,
            label='m1', apply_snr_filter=False)
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
            label='m2')
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
        label=manual_phase)
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
    out = mc_i2d_path.replace('_residual_i2d.fits', '_residual_smoothed_bg_i2d.fits')
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
    """Cross-filter seed for m5: sources detected (well) in >= min_filters
    filters within max_sep_mas, each with S/N > snr_min and good qfit.

    NOTE: the stringent cross-match (>=2 filters, <10 mas, S/N>5 each) is a TODO;
    the immediate cutout tests are single-filter so this path is not exercised.
    For now it unions the per-filter vetted m4 skycoords (like the legacy union
    seed) so the multifilter path is runnable.  Replace with
    ``merge_catalogs.merge_catalogs(..., max_offset=max_sep_mas*u.mas)`` + the
    >=min_filters / snr / qfit cut before relying on m5 scientifically.
    """
    from astropy.table import vstack as _vstack
    desat = '_unsatstar' if options.desaturated else ''
    bgsub = ('_bgsub' if options.bgsub else '') + '_resbgsub'
    blur_ = '_blur' if options.blur else ''
    tbls = []
    for module in modules:
        for filt in filternames:
            p = (f'{cut_bp}/catalogs/{filt.lower()}_{module}_indivexp_merged'
                 f'{desat}{bgsub}{blur_}_m4_dao_basic_vetted.fits')
            if os.path.exists(p):
                t = Table.read(p)
                if 'skycoord' in t.colnames:
                    tbls.append(Table({'skycoord': t['skycoord']}))
    if not tbls:
        raise ValueError(f"m5 crossband seed: no vetted m4 catalogs under {cut_bp}/catalogs/")
    union = _vstack(tbls, metadata_conflicts='silent')
    out = f'{cut_bp}/catalogs/crossband_seed_manual.fits'
    union.write(out, overwrite=True)
    print(f"[m5] wrote crossband seed {out} (n={len(union)}); "
          f"TODO stringent >= {min_filters}-filter/{max_sep_mas}mas/SNR>{snr_min} cut",
          flush=True)
    return out


# ---------------------------------------------------------------------------
# Orchestrator (mirrors _run_cutout_pipeline; manual phases, basic-only)
# ---------------------------------------------------------------------------
def run_manual_pipeline(options, modules, filternames, nvisits, proposal_id,
                        target, field, basepath, crowdsource_default_kwargs,
                        bg_boxsizes):
    """In-process manual-iteration cutout pipeline (parallels
    ``_run_cutout_pipeline``).  Phases m12 -> m3 -> m4 (-> m5 if multifilter);
    BASIC-only.  After each phase: merge per-frame catalogs, vet the merged
    catalog (extended-emission filter), build the vetted mergedcat residual i2d,
    smooth it into the background map fed to the next phase.
    """
    import copy
    from jwst_gc_pipeline.photometry import merge_catalogs as _merge_catalogs

    cut_bp = _L._cutout_out_basepath(basepath, options)
    os.makedirs(os.path.join(cut_bp, 'catalogs'), exist_ok=True)
    pupil = 'clear'
    multifilter = len(filternames) > 1
    phases = ['m12', 'm3', 'm4']
    if multifilter:
        phases.append('m5')
    print(f"MANUAL PIPELINE: phases={phases} filters={filternames} "
          f"modules={modules}", flush=True)

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
    bg_for_next = {}   # (module, filt) -> smoothed-bg path for the next phase
    overlap_total = 0

    for phase in phases:
        resbgsub = phase in ('m3', 'm4', 'm5')
        merge_label = 'm2' if phase == 'm12' else phase
        opts_phase = copy.copy(options)
        opts_phase.iteration_label = merge_label
        opts_phase.seed_catalog = ''
        opts_phase.use_iter3_residual_bg = resbgsub

        for module in modules:
            for filt in filternames:
                prev_seed = None
                resbg_path = None
                if phase == 'm3':
                    prev_seed = _merged_path('m2', module, filt, False).replace(
                        '.fits', '_vetted.fits')
                    resbg_path = bg_for_next.get((module, filt))
                elif phase == 'm4':
                    prev_seed = _merged_path('m3', module, filt, True).replace(
                        '.fits', '_vetted.fits')
                    resbg_path = bg_for_next.get((module, filt))
                elif phase == 'm5':
                    prev_seed = _build_crossband_seed(cut_bp, modules, filternames, options)
                    resbg_path = bg_for_next.get((module, filt))

                # candidate frames (scan on first phase, cache thereafter)
                if phase == phases[0]:
                    candidate_frames = []
                    for visitid in range(1, nvisits[proposal_id][target] + 1):
                        candidate_frames.extend(sorted(_L.get_filenames(
                            basepath, filt, proposal_id, field,
                            visitid=f'{visitid:03d}', each_suffix=options.each_suffix,
                            module=module, pupil='clear')))
                else:
                    candidate_frames = frame_cache.get((module, filt), [])

                overlapping_now = []
                for filename in candidate_frames:
                    exposure_id = filename.split("_")[2]
                    visit_id = filename.split("_")[0][-3:]
                    vgroup_id = filename.split("_")[1]
                    file_detector = filename.split("_")[3]
                    file_module = file_detector if module == 'merged' else module
                    try:
                        do_photometry_step_manual(
                            opts_phase, filt, file_module, file_detector, field,
                            basepath, filename, proposal_id,
                            manual_phase=phase, exposurenumber=int(exposure_id),
                            visit_id=visit_id, vgroup_id=vgroup_id,
                            bg_boxsizes=bg_boxsizes, use_webbpsf=True, pupil=pupil,
                            prev_seed_catalog=prev_seed, resbg_path=resbg_path)
                    except _L.CutoutNoOverlap as ex:
                        print(f"manual [{phase}]: skip non-overlapping {filename} ({ex})",
                              flush=True)
                        continue
                    overlapping_now.append(filename)

                if phase == phases[0]:
                    frame_cache[(module, filt)] = overlapping_now
                    overlap_total += len(overlapping_now)
                if not overlapping_now:
                    raise ValueError(
                        f"--cutout-region overlapped none of the {filt}/{module} "
                        f"frames in phase {phase}.")

                # merge per-frame catalogs (BASIC only)
                _merge_catalogs.merge_individual_frames(
                    module=module, filtername=filt.lower(), progid=proposal_id,
                    method='dao', suffix='_basic', target=target, basepath=cut_bp,
                    iteration_label=merge_label, bgsub=options.bgsub,
                    desat=options.desaturated, epsf=options.epsf, blur=options.blur,
                    resbgsub=resbgsub, fwhm_basepath=basepath)

                # data i2d once (m12), for peak-SB in the vetting step
                if phase == phases[0]:
                    try:
                        _L.mosaic_cutout_input_data(
                            cut_bp, filt, proposal_id, field, module,
                            _L._cutout_label_for(options), pupil=pupil)
                    except Exception as ex:
                        print(f"manual: data i2d build failed: {ex}", flush=True)

                # vet the merged catalog -> _vetted.fits
                merged_path = _merged_path(merge_label, module, filt, resbgsub)
                vetted_path = merged_path.replace('.fits', '_vetted.fits')
                try:
                    merged = Table.read(merged_path)
                    d_i2d, ww_i2d = None, None
                    dpath = _data_i2d_path(module, filt)
                    if os.path.exists(dpath):
                        with fits.open(dpath) as dh:
                            hdu = dh['SCI'] if 'SCI' in [h.name for h in dh] else dh[0]
                            d_i2d = hdu.data.astype(float)
                            ww_i2d = wcs.WCS(hdu.header)
                    vetted = _filter_extended_emission(
                        merged, data_i2d_image=d_i2d, ww_i2d=ww_i2d,
                        qfit_max=float(getattr(options, 'manual_ext_qfit_max', 0.2)),
                        peak_over_bkg=float(getattr(options, 'manual_ext_peak_over_bkg', 20.0)),
                        local_snr_min=float(getattr(options, 'manual_ext_local_snr_min', 5.0)),
                        label=f'{phase}:{filt}')
                    vetted.write(vetted_path, overwrite=True)
                except Exception as ex:
                    print(f"manual [{phase}]: vetting failed ({ex}); using unvetted "
                          f"merged catalog as seed", flush=True)
                    vetted_path = merged_path

                # build vetted mergedcat residual i2d, smooth -> bg for next phase
                try:
                    outpaths = _L.build_mergedcat_residuals(
                        cut_bp, basepath, vetted_path, filt, proposal_id, field,
                        module, opts_phase, frame_cache.get((module, filt), []),
                        merge_label, ['basic'], pupil=pupil)
                    mc_i2d = outpaths.get('basic')
                    if mc_i2d and os.path.exists(mc_i2d):
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

    print(f"MANUAL PIPELINE DONE: {overlap_total} overlapping frames, "
          f"phases={phases}", flush=True)
