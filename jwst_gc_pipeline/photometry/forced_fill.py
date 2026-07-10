"""m8 forced cross-band photometry fill.

The cross-band merge (``merge_catalogs.merge_daophot``) builds the merged m7
catalog by *mutual-nearest cross-match* of the per-band catalogs, each detected
independently.  A source detected in band A but not independently detected in
band B within 0.1" is left ``mask_B = True`` -- a phantom non-detection.  These
dominate the bright end of color diagrams (e.g. bright F187N stars with "no"
F405N), which is unphysical: the star is there, it just was not in band B's
independent detection list (crowding, threshold, sub-0.1" astrometry).

m8 fixes this: at every source's reference position (``skycoord_ref``), for each
band where it is a non-detection (and not saturated), force-fit the flux with
the closed-form fixed-position solver
(:func:`jwst_gc_pipeline.photometry.psf_fitting.forced_psf_photometry`) against
that band's m7 per-frame data + PSF + error -- the SAME frames m7 photometered.
Per-frame forced fluxes are combined inverse-variance.  The result either
recovers a real flux (kills the phantom) or yields a genuine per-source noise
limit.

Calibration is taken from the band's own firm detections in the merged table
(``conv = median(flux_jy / flux)``; Vega zero-point likewise), so m8 fluxes are
on exactly the same system as m7 -- no zero-point / pixel-area re-derivation.

This runs after the m7 cross-band merge for every multifilter run (see
``run_manual_pipeline``), producing ``..._merged_resbgsub_m8.fits``.
"""
import os
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

from jwst_gc_pipeline.photometry.psf_fitting import forced_psf_photometry

ABMAG_OFFSET = 8.90  # -2.5 log10(flux_jy) + 8.90  (AB)


def _frame_args_from_filename(filename, *, options, filt, field, basepath,
                              proposal_id, bg_boxsizes, pupil, resbg_path,
                              satstar_overrides, satstar_drops, module,
                              satstar_label='m7'):
    """Replicate run_manual_pipeline's per-frame arg construction (the tokens
    parsed out of the exposure filename) so we prepare the frame identically.

    ``satstar_label`` selects which phase's satstar products to subtract; m8
    has none of its own, so it reuses the final real phase (m7)."""
    exposure_id = filename.split("_")[2]
    visit_id = filename.split("_")[0][-3:]
    vgroup_id = filename.split("_")[1]
    file_detector = filename.split("_")[3]
    if '-' in str(field):
        obs_id = filename.split("_")[0][-6:-3]
        vgroup_id = f'{obs_id}{vgroup_id}'
    return dict(
        options=options, filtername=filt, module=file_detector, field=field,
        basepath=basepath, filename=filename, proposal_id=proposal_id,
        exposurenumber=int(exposure_id), visit_id=visit_id, vgroup_id=vgroup_id,
        bg_boxsizes=bg_boxsizes, use_webbpsf=True, pupil=pupil,
        resbg_path=resbg_path, satstar_label=satstar_label,
        satstar_flux_overrides=(satstar_overrides or {}).get((module, filt)),
        satstar_flux_drops=(satstar_drops or {}).get((module, filt)),
    )


def _band_calibration(tbl, filt):
    """(conv_jy, vega_zp) from this band's firm detections in the merged table.

    conv_jy : flux_jy = flux * conv_jy   (raw image flux -> Jy)
    vega_zp : mag_vega = -2.5 log10(flux_jy / vega_zp)
    Returns (None, None) if there are too few detections to calibrate.
    """
    if f'flux_{filt}' not in tbl.colnames or f'flux_jy_{filt}' not in tbl.colnames:
        return None, None
    flux = np.asarray(getattr(tbl[f'flux_{filt}'], 'filled', lambda v: tbl[f'flux_{filt}'])(np.nan), dtype=float)
    fjy = np.asarray(getattr(tbl[f'flux_jy_{filt}'], 'filled', lambda v: tbl[f'flux_jy_{filt}'])(np.nan), dtype=float)
    mask = np.asarray(getattr(tbl[f'mask_{filt}'], 'filled', lambda v: tbl[f'mask_{filt}'])(True), dtype=bool) \
        if f'mask_{filt}' in tbl.colnames else np.zeros(len(tbl), dtype=bool)
    good = (~mask) & np.isfinite(flux) & np.isfinite(fjy) & (flux > 0) & (fjy > 0)
    if good.sum() < 20:
        return None, None
    conv_jy = float(np.nanmedian(fjy[good] / flux[good]))
    vega_zp = None
    if f'mag_vega_{filt}' in tbl.colnames:
        mv = np.asarray(getattr(tbl[f'mag_vega_{filt}'], 'filled', lambda v: tbl[f'mag_vega_{filt}'])(np.nan), dtype=float)
        gv = good & np.isfinite(mv)
        if gv.sum() >= 20:
            vega_zp = float(np.nanmedian(fjy[gv] / 10 ** (-0.4 * mv[gv])))
    return conv_jy, vega_zp


def _satstar_partner_guard(tbl, filt, ref, targets,
                           guard_radius=0.5 * u.arcsec):
    """Veto fill targets that are saturated-star rows with a real ``filt``
    detection nearby -- filling them measures the star-subtracted residual.

    A row that is ``replaced_saturated``/``is_saturated`` in ANY band but a
    "non-detection" in ``filt`` is almost always the same physical star as a
    nearby independent ``filt`` detection that the cross-band merge failed to
    associate (saturated-core centroids scatter ~0.2"; Brick F182M: median
    0.214" to the partner detection, 78% within 0.5").  Force-fitting at the
    satstar position then measures the residual AFTER that detection's flux
    was subtracted -- ~5 mag too faint -- and plants a wildly wrong color on
    the CMD bright end.  Those rows should be merged by the catalog dedup
    (``dedup_catalog.dedup_merged_catalog`` ``sat_link_radius``), not filled.

    Returns the boolean veto mask aligned to ``tbl`` (True = do NOT fill).
    """
    n = len(tbl)
    is_sat_any = np.zeros(n, bool)
    for c in tbl.colnames:
        if c.startswith('replaced_saturated_') or c.startswith('is_saturated_'):
            col = tbl[c]
            is_sat_any |= np.asarray(col.filled(False) if hasattr(col, 'filled')
                                     else col, dtype=bool)
    cand = targets & is_sat_any
    if not cand.any():
        return np.zeros(n, bool)
    # independent firm detections in this band: finite mag, not masked
    mag = tbl[f'mag_vega_{filt}'] if f'mag_vega_{filt}' in tbl.colnames else None
    if mag is None or f'mask_{filt}' not in tbl.colnames:
        return np.zeros(n, bool)
    mag = np.asarray(mag.filled(np.nan) if hasattr(mag, 'filled') else mag,
                     dtype=float)
    msk = tbl[f'mask_{filt}']
    msk = np.asarray(msk.filled(True) if hasattr(msk, 'filled') else msk,
                     dtype=bool)
    det = np.isfinite(mag) & ~msk
    if not det.any():
        return np.zeros(n, bool)
    ci = np.where(cand)[0]
    _, sep, _ = ref[ci].match_to_catalog_sky(ref[det])
    veto = np.zeros(n, bool)
    veto[ci[sep < guard_radius]] = True
    return veto


def forced_fill_band(tbl, filt, frames, *, prepare_frame, frame_arg_builder,
                     nsigma=3.0, fit_shape=(5, 5),
                     satstar_partner_guard=0.5 * u.arcsec, verbose=True):
    """Force-fit ``filt`` at the reference position of every source that is a
    non-detection (and not saturated) in ``filt``, combine across ``frames``,
    and write the results back into ``tbl`` in place.

    Adds/sets columns: ``forced_filled_{filt}`` (bool), ``forced_snr_{filt}``,
    and -- for sources reaching SNR>=nsigma -- updates ``flux_{filt}``,
    ``flux_jy_{filt}``, ``mag_ab_{filt}``, ``mag_vega_{filt}``,
    ``emag_ab_{filt}`` and clears ``mask_{filt}``.  Sub-threshold sources keep
    ``mask_{filt}=True`` but record the measured flux (a real per-source limit).
    """
    n = len(tbl)
    fl = filt.lower()  # merged-catalog columns are lowercase; frames keep orig case
    filt = fl
    if f'mask_{filt}' not in tbl.colnames or 'skycoord_ref' not in tbl.colnames:
        if verbose:
            print(f"  [m8 {filt}] no mask/skycoord_ref column; skip", flush=True)
        return 0

    conv_jy, vega_zp = _band_calibration(tbl, filt)
    if conv_jy is None:
        if verbose:
            print(f"  [m8 {filt}] too few detections to calibrate; skip", flush=True)
        return 0

    mask = np.asarray(tbl[f'mask_{filt}'].filled(True) if hasattr(tbl[f'mask_{filt}'], 'filled')
                      else tbl[f'mask_{filt}'], dtype=bool)
    sat = np.zeros(n, dtype=bool)
    # NOTE 'near_saturated_{filt}_{filt}' (doubled suffix) was checked here
    # since aef909b -- that column never exists, so near-saturated rows were
    # silently NOT excluded from the fill.  Fixed to the real column name.
    for sc in (f'is_saturated_{filt}', f'near_saturated_{filt}',
               f'replaced_saturated_{filt}'):
        if sc in tbl.colnames:
            c = tbl[sc]
            sat |= np.asarray(c.filled(False) if hasattr(c, 'filled') else c, dtype=bool)

    ref = tbl['skycoord_ref']
    if not isinstance(ref, SkyCoord):
        ref = SkyCoord(ref)
    targets = mask & (~sat) & np.isfinite(ref.ra.deg) & np.isfinite(ref.dec.deg)
    if satstar_partner_guard is not None and satstar_partner_guard > 0:
        veto = _satstar_partner_guard(tbl, filt, ref, targets,
                                      guard_radius=satstar_partner_guard)
        if veto.any() and verbose:
            print(f"  [m8 {filt}] satstar-partner guard: skipping "
                  f"{int(veto.sum())} saturated-row target(s) with a real "
                  f"{filt} detection within {satstar_partner_guard} "
                  f"(fill would measure the subtracted residual)", flush=True)
        targets &= ~veto
    tgt_idx = np.where(targets)[0]
    if tgt_idx.size == 0:
        if verbose:
            print(f"  [m8 {filt}] no fill targets", flush=True)
        return 0

    # inverse-variance accumulators over frames
    wsum = np.zeros(tgt_idx.size)      # sum 1/var
    wfsum = np.zeros(tgt_idx.size)     # sum flux/var
    nseen = np.zeros(tgt_idx.size, dtype=int)
    ra = ref.ra.deg[tgt_idx]
    dec = ref.dec.deg[tgt_idx]

    for filename in frames:
        try:
            kw = frame_arg_builder(filename)
            ctx = prepare_frame(**kw)
        except Exception as ex:
            if verbose:
                print(f"  [m8 {filt}] prep failed {os.path.basename(filename)}: {ex}", flush=True)
            continue
        ny, nx = ctx.nan_replaced_data.shape
        xpix, ypix = ctx.ww.all_world2pix(ra, dec, 0)
        inb = (xpix > 2) & (xpix < nx - 3) & (ypix > 2) & (ypix < ny - 3) & \
              np.isfinite(xpix) & np.isfinite(ypix)
        if not inb.any():
            continue
        seed = Table({'x_init': xpix[inb], 'y_init': ypix[inb]})
        res = forced_psf_photometry(
            ctx.nan_replaced_data, ctx.dao_psf_model, seed,
            error=ctx.err, mask=ctx.mask, fit_shape=fit_shape, nonnegative=False)
        f = np.asarray(res['flux_fit'], dtype=float)
        e = np.asarray(res['flux_err'], dtype=float)
        ok = np.isfinite(f) & np.isfinite(e) & (e > 0)
        loc = np.where(inb)[0][ok]
        var = e[ok] ** 2
        wsum[loc] += 1.0 / var
        wfsum[loc] += f[ok] / var
        nseen[loc] += 1

    fitted = nseen > 0
    flux = np.full(tgt_idx.size, np.nan)
    ferr = np.full(tgt_idx.size, np.nan)
    flux[fitted] = wfsum[fitted] / wsum[fitted]
    ferr[fitted] = 1.0 / np.sqrt(wsum[fitted])
    snr = np.where(ferr > 0, flux / ferr, np.nan)

    # write back ---------------------------------------------------------
    def _ensure(col, dtype, fill):
        if col not in tbl.colnames:
            tbl[col] = np.full(n, fill, dtype=dtype)
    _ensure(f'forced_filled_{filt}', bool, False)
    _ensure(f'forced_snr_{filt}', float, np.nan)

    detected = fitted & np.isfinite(snr) & (snr >= nsigma) & (flux > 0)
    fjy = flux * conv_jy

    # unmask columns we may write (MaskedColumn -> plain so assignment sticks)
    for col in (f'flux_{filt}', f'flux_jy_{filt}', f'mag_ab_{filt}',
                f'mag_vega_{filt}', f'emag_ab_{filt}', f'mask_{filt}'):
        if col in tbl.colnames and hasattr(tbl[col], 'filled'):
            tbl[col] = tbl[col].filled(tbl[col].fill_value)

    ridx = tgt_idx
    tbl[f'forced_filled_{filt}'][ridx[fitted]] = True
    tbl[f'forced_snr_{filt}'][ridx[fitted]] = snr[fitted]
    if f'flux_{filt}' in tbl.colnames:
        tbl[f'flux_{filt}'][ridx[fitted]] = flux[fitted]
    if f'flux_jy_{filt}' in tbl.colnames:
        tbl[f'flux_jy_{filt}'][ridx[fitted]] = fjy[fitted]
    if f'emag_ab_{filt}' in tbl.colnames:
        with np.errstate(divide='ignore', invalid='ignore'):
            tbl[f'emag_ab_{filt}'][ridx[fitted]] = 1.0857 * (ferr[fitted] / flux[fitted])
    # magnitudes only for positive flux
    posfit = fitted & (flux > 0)
    if f'mag_ab_{filt}' in tbl.colnames:
        tbl[f'mag_ab_{filt}'][ridx[posfit]] = -2.5 * np.log10(fjy[posfit]) + ABMAG_OFFSET
    if vega_zp is not None and f'mag_vega_{filt}' in tbl.colnames:
        tbl[f'mag_vega_{filt}'][ridx[posfit]] = -2.5 * np.log10(fjy[posfit] / vega_zp)
    # clear mask only for SNR>=nsigma (now a real detection)
    if f'mask_{filt}' in tbl.colnames:
        tbl[f'mask_{filt}'][ridx[detected]] = False

    if verbose:
        print(f"  [m8 {filt}] targets={tgt_idx.size} fitted={int(fitted.sum())} "
              f"recovered(SNR>={nsigma:.0f})={int(detected.sum())} "
              f"conv_jy={conv_jy:.3e} vega_zp={'%.3e' % vega_zp if vega_zp else 'NA'}",
              flush=True)
    return int(detected.sum())


def run_forced_crossband_fill(merged_path, out_path, *, filternames, module,
                              frames_for, prepare_frame, frame_arg_builder_for,
                              nsigma=3.0, fit_shape=(5, 5), verbose=True):
    """Load the merged m7 catalog, force-fill every band, write the m8 catalog.

    Parameters
    ----------
    merged_path : path to the merged m7 ``...resbgsub_m7[...].fits``
    out_path : path to write the m8 catalog
    filternames : list of (real) filter names to fill
    module : detector module token (for logging only)
    frames_for : callable ``filt -> [frame filenames]``
    prepare_frame : callable matching ``_prepare_frame_for_photometry`` kwargs,
        returning the per-frame context object
    frame_arg_builder_for : callable ``filt -> (filename -> kwargs dict)``
    """
    if verbose:
        print(f"[m8] forced cross-band fill: {os.path.basename(merged_path)} "
              f"(module={module})", flush=True)
    tbl = Table.read(merged_path)
    total = 0
    for filt in filternames:
        if f'mask_{filt.lower()}' not in tbl.colnames:
            continue
        frames = list(frames_for(filt))
        if not frames:
            if verbose:
                print(f"  [m8 {filt}] no frames available; skip", flush=True)
            continue
        total += forced_fill_band(
            tbl, filt, frames, prepare_frame=prepare_frame,
            frame_arg_builder=frame_arg_builder_for(filt),
            nsigma=nsigma, fit_shape=fit_shape, verbose=verbose)
    tbl.write(out_path, overwrite=True)
    if verbose:
        print(f"[m8] wrote {out_path}  ({total} phantom non-detections recovered)",
              flush=True)
    return out_path
