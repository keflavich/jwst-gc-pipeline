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

A sub-threshold fit (SNR < ``nsigma``) is NOT reported as a detection: the raw
point-estimate flux is kept in ``flux_{filt}``/``flux_jy_{filt}`` (useful for
QA / stacking, and unbiased -- clamping it would bias a later stack), but
``mask_{filt}`` stays ``True`` and no magnitude is written.  Alongside it,
``flux_ulim_{filt}`` / ``mag_vega_ulim_{filt}`` / ``mag_ab_ulim_{filt}`` record
the actual ``nsigma``-scaled upper limit (``nsigma * ferr``, from the SAME
inverse-variance-combined per-frame fit error used for ``forced_snr_{filt}``).
``ferr`` comes from the crf ``ERR`` plane (Poisson, read noise and flat error),
so it follows the Poisson noise of nebular flux under the source but carries no
term for structure left in the background-subtracted frame: neither the
residual-background median-filter box (``--residual-bg-median-size``, which sets
what is subtracted, not a variance) nor unresolved neighbours enter it.  On
bright extended emission ``ferr`` is therefore a lower bound on the local noise
(#1120; #1284 measured the fixed-position scatter at ~10x ``ferr`` on o107
F480M).  ``--forced-fill-bkg-scatter`` adds that term (see below).  Without
this column the only place the per-source noise level survived was
``forced_snr_{filt}`` (flux/ferr), from which a limit cannot be safely
recovered when the point estimate itself is near zero or negative.

``is_upper_limit_{filt}`` is an explicit per-band bool, set ``True`` exactly
for a fitted-but-sub-threshold row (the only rows where ``flux_ulim_{filt}``
is a bound rather than a detection-depth diagnostic) and ``False`` everywhere
else -- a firm detection, a row never targeted (independently detected,
saturated, or vetoed by the satstar-partner guard), and a row where the fill
was attempted but never converged on any frame (``fitted`` is ``False``, so
there is no measurement of any kind, not even a limit). Reading
``mask_{filt} & forced_filled_{filt} & isfinite(flux_ulim_{filt})`` recovers
the same set, but the flag exists so a consumer does not have to reconstruct
that combination correctly at every call site.

IMPORTANT for any consumer: ``flux_ulim_{filt}``/``mag_ab_ulim_{filt}``/
``mag_vega_ulim_{filt}`` are ALSO populated for a firm detection (a depth
diagnostic there, same as a sub-threshold row) -- a finite ``mag_ab_ulim``
is NOT by itself evidence that a row is a limit.  Gate on
``is_upper_limit_{filt}``, never on ``isfinite(mag_ab_ulim_{filt})`` alone.

With ``bkg_scatter`` (``--forced-fill-bkg-scatter``, default off) each
source's per-frame error stamp becomes ``sqrt(ERR**2 + sigma**2)``, where
``sigma`` is ``1.4826 * MAD`` of the valid, finite pixels of the frame's
satstar-subtracted data in an annulus of 2.5 to 4.5 FWHM about the source (the
estimator of ``SATSTAR_ERR_BKG_SCATTER``,
``psf_fitting.annulus_scatter_sigma``).  A source whose annulus has fewer than
20 usable pixels keeps the ERR-only error in that frame.  ``ferr``,
``forced_snr_{filt}``, ``flux_ulim_{filt}`` and the detection cut then all rest
on the inflated error, and ``forced_bkg_sigma_{filt}`` records the mean
``sigma`` over the frames that measured it (image units, NaN when none did).
The term is white and per pixel, so it under-corrects scatter that is
correlated over the PSF footprint.  It is an error term only: an annulus median
offset still enters ``flux_{filt}`` directly, since the closed-form fit has no
background parameter.  Overlapping frames share their background structure but
are still combined as independent, so ``ferr`` shrinks as ``1/sqrt(N)`` for a
shared sky.

With ``nbrsub`` (``--forced-fill-nbrsub``, default off) each frame is fitted
with its own m7 star-only model removed (``{stub}_daophot_basic_model.fits``,
the model the m7 pass wrote with the same PSF), so the other stars of the band
no longer add light to the 5x5 box.  Without it a neighbour half a FWHM away
puts about 0.72 of its flux into the fit and the fill reports it as the
target's (#1284).  The m7 sources of the frame's unvetted catalogue within 40 mas
of a target are added back for that target, so a same-star counterpart that the
merge left unpaired stays in the target's flux, and so do the rows that m7
vetting removed.  ``forced_nbrsub_{filt}`` and ``forced_near_m7_sep_{filt}``
record this.  A frame with no usable model file is fitted without subtraction
and the path is printed.

With ``nrand`` (``--forced-fill-nrand``, default 0) the combined error is scaled
by ``k``, the half-width of the 16th to 84th percentile range of flux/ferr at
random blank positions, taken as at least 1 (#1284).  The per-frame errors are
combined as independent although their scatter is shared between dithers, so
``k`` is measured on the combined values.  Run it with ``nbrsub`` (and
``bkg_scatter`` where used) so that ``k`` is the residual factor.
``forced_err_scale_{filt}`` records ``k`` and ``forced_snr_raw_{filt}`` the
signal-to-noise before the scaling.

Calibration is taken from the band's own firm detections in the merged table
(``conv = median(flux_jy / flux)``; Vega zero-point likewise), so m8 fluxes are
on exactly the same system as m7 -- no zero-point re-derivation.  Each frame's
fitted flux is first multiplied by that frame's own ``PIXAR_SR`` (relative to
the conversion), so a detector's pixel area enters once, whatever area the m7
table used (#1238).

This runs after the m7 cross-band merge for every multifilter run (see
``run_manual_pipeline``), producing ``..._merged_resbgsub_m8.fits``.
"""
import os
import traceback
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

from jwst_gc_pipeline.photometry import frame_pool
from jwst_gc_pipeline.photometry.frame_pixel_area import frame_area_sr
from jwst_gc_pipeline.photometry.naming import frame_identity
from jwst_gc_pipeline.photometry.psf_fitting import (
    annulus_scatter_sigma, forced_psf_photometry)

ABMAG_OFFSET = 8.90  # -2.5 log10(flux_jy) + 8.90  (AB)


def _frame_args_from_filename(filename, *, options, filt, field, basepath,
                              proposal_id, bg_boxsizes, pupil, resbg_path,
                              satstar_overrides, satstar_drops, module,
                              satstar_label='m7'):
    """Replicate run_manual_pipeline's per-frame arg construction (the tokens
    parsed out of the exposure filename) so we prepare the frame identically.

    ``satstar_label`` selects which phase's satstar products to subtract; m8
    has none of its own, so it reuses the final real phase (m7)."""
    # `get_filenames` returns FULL PATHS, so the tokens come from the basename.
    # Splitting the path shifted every index for a field directory with an
    # underscore -- gc2211_o023 read visit '211' and detector '00001' -- which is
    # the defect #472 fixed in run_manual_pipeline and left standing here.
    visit_id, vgroup_id, exposure_id, file_detector = frame_identity(
        filename, field=field)
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


# Annulus of the local-background scatter term, in FWHM, and the fewest usable
# annulus pixels (the satstar estimator's min_pixels).
BKG_SCATTER_INNER_FWHM = 2.5
BKG_SCATTER_OUTER_FWHM = 4.5
BKG_SCATTER_MIN_PIXELS = 20

# Neighbour subtraction: an m7 source within this distance of a target is added
# back to the fitted image, so a same-star m7 counterpart that the cross-band
# merge failed to pair stays in the target's flux (#1284).  The model stamp is
# the 21x21 that ``cataloging._subset_model_image`` renders.
NBRSUB_ADDBACK_MAS = 40.0
NBRSUB_STAMP_HALF = 10

# Empirical error scale: random positions drawn 1 to 2 FWHM from random table
# rows, and the fewest fitted ones that give a scale (#1284).
NRAND_OFFSET_FWHM = (1.0, 2.0)
NRAND_MIN_FIT = 50


def _pixel_scale_arcsec(ww, nx, ny):
    """Pixel scale (arcsec) of ``ww`` at the frame centre, from the sky
    separation of two adjacent pixels."""
    cx, cy = 0.5 * nx, 0.5 * ny
    ra, dec = ww.all_pix2world([cx, cx + 1.0], [cy, cy], 0)
    return float(SkyCoord(ra[0] * u.deg, dec[0] * u.deg).separation(
        SkyCoord(ra[1] * u.deg, dec[1] * u.deg)).arcsec)


def _load_m7_frame_model(model_path, shape, filt):
    """The m7 star-only model image of one frame, non-finite pixels set to 0,
    or ``None`` (with the path printed) when the file is missing, unreadable or
    not ``shape``.  The path carries the PSF token, so a fill run with a
    different PSF from the m7 run finds no file and falls back."""
    if not os.path.exists(model_path):
        print(f"  [m8 {filt}] neighbour subtraction: no m7 model {model_path}; "
              f"this frame is fitted without it", flush=True)
        return None
    try:
        model = np.asarray(fits.getdata(model_path, extname='SCI'), dtype=float)
    except (OSError, ValueError, KeyError) as ex:
        print(f"  [m8 {filt}] neighbour subtraction: cannot read m7 model "
              f"{model_path}: {ex}; this frame is fitted without it", flush=True)
        return None
    if model.shape != tuple(shape):
        print(f"  [m8 {filt}] neighbour subtraction: m7 model {model_path} has "
              f"shape {model.shape}, frame {tuple(shape)}; this frame is fitted "
              f"without it", flush=True)
        return None
    return np.where(np.isfinite(model), model, 0.0)


def _read_m7_frame_sources(cat_path, filt):
    """``(x, y, flux)`` of the finite rows of one frame's m7 per-frame
    catalogue (the unvetted one), or ``None`` with the path printed."""
    if not os.path.exists(cat_path):
        print(f"  [m8 {filt}] neighbour subtraction: no m7 frame catalogue "
              f"{cat_path}; nothing is added back on this frame", flush=True)
        return None
    try:
        cat = Table.read(cat_path)
        x = np.asarray(cat['x_fit'], dtype=float)
        y = np.asarray(cat['y_fit'], dtype=float)
        flux = np.asarray(cat['flux_fit'], dtype=float)
    except (OSError, ValueError, KeyError) as ex:
        print(f"  [m8 {filt}] neighbour subtraction: cannot read m7 frame "
              f"catalogue {cat_path}: {ex}; nothing is added back on this "
              f"frame", flush=True)
        return None
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(flux)
    return x[ok], y[ok], flux[ok]


def _model_stamp(psf_model, x0, y0, flux, shape, half=NBRSUB_STAMP_HALF):
    """``(ylo, yhi, xlo, xhi, stamp)`` of one source rendered on a
    ``(2*half+1)`` square stamp clipped to ``shape``, or ``None`` when the
    stamp misses the image."""
    ix, iy = int(round(x0)), int(round(y0))
    ylo, yhi = max(iy - half, 0), min(iy + half + 1, shape[0])
    xlo, xhi = max(ix - half, 0), min(ix + half + 1, shape[1])
    if ylo >= yhi or xlo >= xhi:
        return None
    yy, xx = np.mgrid[ylo:yhi, xlo:xhi].astype(float)
    stamp = np.asarray(psf_model.evaluate(xx, yy, float(flux), x0, y0),
                       dtype=float)
    return ylo, yhi, xlo, xhi, stamp


def _forced_fit_frame(filename, filt, ra, dec, *, prepare_frame,
                      frame_arg_builder, fit_shape, conv_jy=None,
                      frame_area_sr=None, bkg_scatter=False,
                      nbrsub=False, frame_products=None,
                      addback_mas=NBRSUB_ADDBACK_MAS):
    """Prepare one frame and force-fit every target that lands on it.

    Returns ``(loc, flux, var)`` for the fits with a finite flux and a positive
    error, ``loc`` indexing the target arrays ``ra``/``dec``, or ``None`` when
    the frame contributes nothing: no target lands on it, or its preparation
    raised one of the exceptions a missing or malformed frame raises (printed
    with its traceback).  Anything else propagates.

    With ``frame_area_sr`` (a callable ``filename -> PIXAR_SR`` in sr) and
    ``conv_jy``, the fitted flux and error are multiplied by
    ``PIXAR_SR * 1e6 / conv_jy``.  Their later product with ``conv_jy`` is then
    the frame's flux in Jy with its own detector's pixel area (#1238), whatever
    area the m7 conversion used.

    With ``bkg_scatter`` each source's error stamp gains the annulus scatter of
    the fitted image in quadrature (``psf_fitting.annulus_scatter_sigma``), and
    the tuple gains the per-source scatter in image units (NaN where
    unmeasured) as its fourth element.

    With ``nbrsub`` the frame's own m7 star-only model is subtracted from the
    fitted image, so the other stars of the frame no longer add light to the
    5x5 box (#1284).  ``frame_products(ctx, frame_kwargs)`` returns the m7
    ``(model_path, catalogue_path)`` of the frame.  Each m7 source in the
    catalogue within ``addback_mas`` of a target is added back to that target's
    fit, so a counterpart of the target that the merge left unpaired remains in
    its flux.  A frame whose model file is unusable is fitted unsubtracted.  The
    tuple gains a last element ``(subtracted, near_sep_mas)``: a bool per fit,
    True where the model was removed, and the separation in mas to the nearest
    added-back source (NaN when none).  Without ``nbrsub`` the fit is
    unchanged.
    """
    # A dropped frame here silently weakens the inverse-variance combine
    # (no-silent-frame-drops rule): catch only the exceptions frame prep
    # can legitimately raise (filename-token parsing -> ValueError /
    # IndexError; missing or unreadable frame products / headers ->
    # OSError (incl. FileNotFoundError) / KeyError), ALWAYS print the
    # full traceback, and let anything else propagate.
    try:
        kw = frame_arg_builder(filename)
        ctx = prepare_frame(**kw)
    except (OSError, KeyError, ValueError, IndexError) as ex:
        print(f"  [m8 {filt}] prep failed {os.path.basename(filename)}: {ex}\n"
              f"{traceback.format_exc()}", flush=True)
        return None
    ny, nx = ctx.nan_replaced_data.shape
    # The cross-band source list spans the whole mosaic, so for any single
    # frame most reference positions fall outside its footprint.  With the
    # GWCS (ctx.ww is a FrameWCS) those return NaN and are dropped by the
    # ``inb`` test below.  With a SIP WCS the iterative inverse cannot
    # converge there and raises NoConvergence unless ``quiet=True`` --
    # which aborted the entire W51 m8 fill (#187).  Pass quiet for the
    # SIP-fallback case; FrameWCS ignores it (its inverse is exact).
    xpix, ypix = ctx.ww.all_world2pix(ra, dec, 0, quiet=True)
    inb = (xpix > 2) & (xpix < nx - 3) & (ypix > 2) & (ypix < ny - 3) & \
          np.isfinite(xpix) & np.isfinite(ypix)
    if not inb.any():
        return None
    seed = Table({'x_init': xpix[inb], 'y_init': ypix[inb]})
    image = ctx.nan_replaced_data
    subtracted = False
    if nbrsub:
        model_path, cat_path = frame_products(ctx, kw)
        model = _load_m7_frame_model(model_path, image.shape, filt)
        if model is not None:
            image = image - model
            subtracted = True
    bkg_sigma = None
    if bkg_scatter:
        fwhm = float(ctx.fwhm_pix)
        # keep the annulus clear of the fit stamp whatever the FWHM
        inner = max(BKG_SCATTER_INNER_FWHM * fwhm,
                    0.5 * max(fit_shape) + 1.5)
        outer = max(BKG_SCATTER_OUTER_FWHM * fwhm, inner + 2.0)
        bkg_sigma = annulus_scatter_sigma(
            image, ~np.asarray(ctx.mask, dtype=bool),
            xpix[inb], ypix[inb], inner, outer,
            min_pixels=BKG_SCATTER_MIN_PIXELS)
    fit_kw = dict(error=ctx.err, mask=ctx.mask, fit_shape=fit_shape,
                  nonnegative=False)
    res = forced_psf_photometry(image, ctx.dao_psf_model, seed,
                                bkg_sigma=bkg_sigma, **fit_kw)
    f = np.asarray(res['flux_fit'], dtype=float)
    e = np.asarray(res['flux_err'], dtype=float)
    near_sep = np.full(f.size, np.nan)
    if subtracted:
        src = _read_m7_frame_sources(cat_path, filt)
        if src is not None and src[0].size:
            from scipy.spatial import cKDTree
            sx, sy, sflux = src
            pix_mas = 1000.0 * _pixel_scale_arcsec(ctx.ww, nx, ny)
            radius = addback_mas / pix_mas
            tree = cKDTree(np.c_[sx, sy])
            near = tree.query_ball_point(np.c_[xpix[inb], ypix[inb]], radius)
            for i, nb in enumerate(near):
                if not nb:
                    continue
                stamps = [_model_stamp(ctx.dao_psf_model, sx[j], sy[j],
                                       sflux[j], image.shape) for j in nb]
                stamps = [st for st in stamps if st is not None]
                if not stamps:
                    continue
                near_sep[i] = pix_mas * float(np.min(np.hypot(
                    sx[nb] - seed['x_init'][i], sy[nb] - seed['y_init'][i])))
                ylo = min(st[0] for st in stamps)
                yhi = max(st[1] for st in stamps)
                xlo = min(st[2] for st in stamps)
                xhi = max(st[3] for st in stamps)
                saved = image[ylo:yhi, xlo:xhi].copy()
                for y0, y1, x0, x1, stamp in stamps:
                    image[y0:y1, x0:x1] += stamp
                one = forced_psf_photometry(
                    image, ctx.dao_psf_model, seed[i:i + 1],
                    bkg_sigma=None if bkg_sigma is None else bkg_sigma[i:i + 1],
                    **fit_kw)
                image[ylo:yhi, xlo:xhi] = saved
                f[i] = float(one['flux_fit'][0])
                e[i] = float(one['flux_err'][0])
    if frame_area_sr is not None and conv_jy is not None:
        scale = float(frame_area_sr(filename)) * 1e6 / conv_jy
        f = f * scale
        e = e * scale
    ok = np.isfinite(f) & np.isfinite(e) & (e > 0)
    out = [np.where(inb)[0][ok], f[ok], e[ok] ** 2]
    if bkg_scatter:
        out.append(bkg_sigma[ok])
    if nbrsub:
        out.append((np.full(int(ok.sum()), subtracted), near_sep[ok]))
    return tuple(out)


def _nominal_fwhm_arcsec(filt):
    """Diffraction-limited FWHM (arcsec) of a JWST filter from the wavelength
    in its name (``f480m`` -> 4.8 um, ``f1130w`` -> 11.3 um), 1.03 lambda/D with
    D = 6.5 m.  Sets the offsets of the random positions only."""
    digits = ''.join(ch for ch in filt if ch.isdigit())
    wave_um = float(digits) / 100.0
    return 1.03 * wave_um * 1e-6 / 6.5 * 206264.806


def _random_positions(ref, rows, exclude, n, fwhm_arcsec, exclude_mas, seed):
    """``n`` random sky positions displaced 1 to 2 FWHM from random ``rows`` of
    ``ref`` (so they sample the crowding of real targets), none within
    ``exclude_mas`` of a position in ``exclude``.  Returns ``(ra, dec)`` in
    degrees; fewer than ``n`` when too few candidates survive the exclusion."""
    rng = np.random.default_rng(seed)
    lo, hi = NRAND_OFFSET_FWHM
    ncand = 3 * n
    base = ref[rng.choice(rows, size=ncand)]
    sep = rng.uniform(lo, hi, ncand) * fwhm_arcsec
    cand = base.directional_offset_by(
        rng.uniform(0.0, 2.0 * np.pi, ncand) * u.rad, sep * u.arcsec)
    keep = np.ones(ncand, dtype=bool)
    if len(exclude):
        _, d2d, _ = cand.match_to_catalog_sky(exclude)
        keep = d2d.arcsec > exclude_mas / 1000.0
    cand = cand[keep][:n]
    return cand.ra.deg, cand.dec.deg


def _empirical_error_scale(flux, ferr):
    """``(k, median, nfit)`` of ``flux / ferr`` over blank positions.

    ``k = max(1, 0.5 * (p84 - p16))``: the scatter of the signal-to-noise at
    positions that hold no target, which is 1 when ``ferr`` is the whole
    noise.  Fewer than ``NRAND_MIN_FIT`` finite fits give ``k = 1`` and a NaN
    median.  The blank positions can hold real sources below the m7 threshold,
    so ``k`` leans conservative."""
    flux = np.asarray(flux, dtype=float)
    ferr = np.asarray(ferr, dtype=float)
    good = np.isfinite(flux) & np.isfinite(ferr) & (ferr > 0)
    if good.sum() < NRAND_MIN_FIT:
        return 1.0, np.nan, int(good.sum())
    ratio = flux[good] / ferr[good]
    p16, p50, p84 = np.percentile(ratio, [16.0, 50.0, 84.0])
    return max(1.0, 0.5 * float(p84 - p16)), float(p50), int(good.sum())


def forced_fill_band(tbl, filt, frames, *, prepare_frame, frame_arg_builder,
                     nsigma=3.0, fit_shape=(5, 5),
                     satstar_partner_guard=0.5 * u.arcsec, verbose=True,
                     n_procs=1, frame_area_sr=None, bkg_scatter=False,
                     nbrsub=False, frame_products=None,
                     addback_mas=NBRSUB_ADDBACK_MAS, nrand=0, rand_seed=0):
    """Force-fit ``filt`` at the reference position of every source that is a
    non-detection (and not saturated) in ``filt``, combine across ``frames``,
    and write the results back into ``tbl`` in place.

    ``n_procs > 1`` prepares and fits the frames in that many forked processes
    (``frame_pool.map_frames``, #1230).  The per-frame results are summed in
    ``frames`` order either way, so the result does not depend on ``n_procs``.

    Adds/sets columns: ``forced_filled_{filt}`` (bool), ``forced_snr_{filt}``,
    ``flux_ulim_{filt}``, ``mag_vega_ulim_{filt}`` (when a Vega zero-point is
    available), ``mag_ab_ulim_{filt}`` -- these four are the ``nsigma``-scaled
    per-source upper limit, written for every fitted target regardless of SNR
    -- and ``is_upper_limit_{filt}`` (bool) -- and -- for sources reaching
    SNR>=nsigma -- updates ``flux_{filt}``, ``flux_jy_{filt}``, ``mag_ab_{filt}``,
    ``mag_vega_{filt}``, ``emag_ab_{filt}`` and clears ``mask_{filt}``.
    Sub-threshold sources keep ``mask_{filt}=True``, keep the raw measured flux
    in ``flux_{filt}`` (a point estimate, not a bound -- may be negative or
    zero), get their bound from ``flux_ulim_{filt}``/``mag_*_ulim_{filt}``
    instead, and are flagged ``is_upper_limit_{filt}=True`` so a consumer does
    not have to re-derive "fitted but not detected" from the other columns.

    ``frame_area_sr`` is a callable ``filename -> PIXAR_SR`` (sr).  When given,
    each frame's fitted flux and error are brought to Jy with that frame's own
    pixel area before the inverse-variance combine (#1238); ``None`` keeps the
    raw image units, which assumes one pixel area for all frames.

    ``bkg_scatter`` adds each source's local annulus scatter to the per-frame
    error in quadrature (module docstring) and writes ``forced_bkg_sigma_{filt}``.
    Off (the default), the output is unchanged and that column is not created.

    ``nbrsub`` fits each frame with its own m7 star-only model removed and the
    m7 sources within ``addback_mas`` of a target added back
    (``_forced_fit_frame``); ``frame_products`` is the callable that names the
    m7 model and catalogue of a frame.  It writes ``forced_nbrsub_{filt}`` (True
    where every contributing frame was subtracted) and
    ``forced_near_m7_sep_{filt}`` (mas to the nearest added-back source, the
    smallest over the frames; NaN when none).  Off (the default), these columns
    are not created and the fit is unchanged.

    ``nrand > 0`` fits ``nrand`` random positions with the targets (through the
    same frames, ``nbrsub`` and ``bkg_scatter`` included) and scales the
    combined error of every target by ``k = max(1, 0.5 * (p84 - p16))`` of the
    random positions' flux/error.  ``forced_snr_{filt}``, the ``nsigma`` cut,
    ``flux_ulim_{filt}`` and ``emag_ab_{filt}`` then use the scaled error.  The
    columns ``forced_err_scale_{filt}`` (``k``) and ``forced_snr_raw_{filt}``
    (flux over the unscaled error) are written.  The frames' scatter is
    correlated between dithers, which the inverse-variance combine ignores, so
    ``k`` is measured on the combined values.  ``rand_seed`` fixes the draw.
    ``nrand = 0`` (the default) leaves the output unchanged.
    """
    if nbrsub and frame_products is None:
        raise ValueError('nbrsub needs frame_products, the callable that names '
                         "a frame's m7 model and catalogue")
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

    if frame_area_sr is not None and f'{filt[1:-1]}PIXA'.upper() not in tbl.meta:
        print(f"  [m8 {filt}] WARNING: the m7 header has no {filt[1:-1].upper()}PIXA "
              f"(PIXAREF): its firm rows were merged without per-detector pixel "
              f"areas, so filled rows are per-detector correct and firm rows are "
              f"not until m7 is rebuilt (#1238)", flush=True)

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

    ra = ref.ra.deg[tgt_idx]
    dec = ref.dec.deg[tgt_idx]
    ntgt = tgt_idx.size
    if nrand > 0:
        # blank positions for the empirical error scale, kept clear of this
        # band's firm detections (the independent m7 rows)
        rows = np.where(np.isfinite(ref.ra.deg) & np.isfinite(ref.dec.deg))[0]
        det = ref[(~mask) & np.isfinite(ref.ra.deg) & np.isfinite(ref.dec.deg)]
        rra, rdec = _random_positions(
            ref, rows, det, int(nrand), _nominal_fwhm_arcsec(filt),
            addback_mas, rand_seed)
        ra = np.r_[ra, rra]
        dec = np.r_[dec, rdec]
    # inverse-variance accumulators over frames (targets, then random positions)
    ntot = ra.size
    wsum = np.zeros(ntot)      # sum 1/var
    wfsum = np.zeros(ntot)     # sum flux/var
    nseen = np.zeros(ntot, dtype=int)
    bsum = np.zeros(ntot)      # sum of the frames' annulus scatter
    bseen = np.zeros(ntot, dtype=int)
    nsub = np.zeros(ntot, dtype=int)   # frames fitted with the m7 model removed
    near_min = np.full(ntot, np.nan)   # nearest added-back m7 source, mas

    def _fit(filename):
        return _forced_fit_frame(filename, filt, ra, dec,
                                 prepare_frame=prepare_frame,
                                 frame_arg_builder=frame_arg_builder,
                                 fit_shape=fit_shape, conv_jy=conv_jy,
                                 frame_area_sr=frame_area_sr,
                                 bkg_scatter=bkg_scatter, nbrsub=nbrsub,
                                 frame_products=frame_products,
                                 addback_mas=addback_mas)

    frames = list(frames)
    if verbose and n_procs > 1 and len(frames) > 1:
        print(f"  [m8 {filt}] {len(frames)} frames across "
              f"{min(int(n_procs), len(frames))} processes "
              f"(--forced-fill-procs)", flush=True)
    for fit in frame_pool.map_frames(_fit, frames, n_procs):
        if fit is None:
            continue
        loc, f, var = fit[:3]
        wsum[loc] += 1.0 / var
        wfsum[loc] += f / var
        nseen[loc] += 1
        if bkg_scatter:
            bs = fit[3]
            got = np.isfinite(bs)
            bsum[loc[got]] += bs[got]
            bseen[loc[got]] += 1
        if nbrsub:
            was_sub, sep = fit[-1]
            nsub[loc] += was_sub
            near_min[loc] = np.fmin(near_min[loc], sep)

    fitted_all = nseen > 0
    flux_all = np.full(ntot, np.nan)
    ferr_all = np.full(ntot, np.nan)
    flux_all[fitted_all] = wfsum[fitted_all] / wsum[fitted_all]
    ferr_all[fitted_all] = 1.0 / np.sqrt(wsum[fitted_all])
    k_scale = 1.0
    if nrand > 0:
        k_scale, rand_median, nrand_fit = _empirical_error_scale(
            flux_all[ntgt:], ferr_all[ntgt:])
        if verbose:
            if nrand_fit < NRAND_MIN_FIT:
                print(f"  [m8 {filt}] error scale: only {nrand_fit} random "
                      f"positions fitted (< {NRAND_MIN_FIT}); k = 1", flush=True)
            else:
                print(f"  [m8 {filt}] error scale k={k_scale:.2f} from "
                      f"{nrand_fit} random positions (median flux/ferr "
                      f"{rand_median:+.2f})", flush=True)
    fitted = fitted_all[:ntgt]
    flux = flux_all[:ntgt]
    ferr_raw = ferr_all[:ntgt]
    ferr = k_scale * ferr_raw
    snr = np.where(ferr > 0, flux / ferr, np.nan)
    bkg_mean = np.full(ntgt, np.nan)
    bkg_mean[bseen[:ntgt] > 0] = bsum[:ntgt][bseen[:ntgt] > 0] / bseen[:ntgt][bseen[:ntgt] > 0]
    sub_all = fitted & (nsub[:ntgt] == nseen[:ntgt])
    near_sep = near_min[:ntgt]
    with np.errstate(divide='ignore', invalid='ignore'):
        snr_raw = np.where(ferr_raw > 0, flux / ferr_raw, np.nan)

    # write back ---------------------------------------------------------
    def _ensure(col, dtype, fill):
        if col not in tbl.colnames:
            tbl[col] = np.full(n, fill, dtype=dtype)
    _ensure(f'forced_filled_{filt}', bool, False)
    _ensure(f'forced_snr_{filt}', float, np.nan)
    _ensure(f'flux_ulim_{filt}', float, np.nan)
    _ensure(f'mag_ab_ulim_{filt}', float, np.nan)
    _ensure(f'mag_vega_ulim_{filt}', float, np.nan)
    _ensure(f'is_upper_limit_{filt}', bool, False)
    if bkg_scatter:
        _ensure(f'forced_bkg_sigma_{filt}', float, np.nan)
    if nbrsub:
        _ensure(f'forced_nbrsub_{filt}', bool, False)
        _ensure(f'forced_near_m7_sep_{filt}', float, np.nan)
    if nrand > 0:
        _ensure(f'forced_err_scale_{filt}', float, np.nan)
        _ensure(f'forced_snr_raw_{filt}', float, np.nan)

    detected = fitted & np.isfinite(snr) & (snr >= nsigma) & (flux > 0)
    not_detected = fitted & ~detected
    fjy = flux * conv_jy

    # per-source nsigma upper limit -- from ferr (the SAME inverse-variance
    # per-frame fit error as forced_snr), not from flux/snr: flux can be <= 0
    # for a sub-threshold fit, which would make a flux/snr-derived bound
    # meaningless or wrong-signed.  Defined for every fitted row, detections
    # included (a useful depth diagnostic there too).
    ulim_flux = np.full(tgt_idx.size, np.nan)
    ulim_flux[fitted] = nsigma * ferr[fitted]
    ulim_fjy = ulim_flux * conv_jy
    with np.errstate(divide='ignore', invalid='ignore'):
        ulim_mag_ab = np.where(np.isfinite(ulim_fjy) & (ulim_fjy > 0),
                               -2.5 * np.log10(ulim_fjy) + ABMAG_OFFSET, np.nan)
        ulim_mag_vega = (np.where(np.isfinite(ulim_fjy) & (ulim_fjy > 0),
                                  -2.5 * np.log10(ulim_fjy / vega_zp), np.nan)
                         if vega_zp is not None else np.full(tgt_idx.size, np.nan))

    # unmask columns we may write (MaskedColumn -> plain so assignment sticks)
    for col in (f'flux_{filt}', f'flux_jy_{filt}', f'mag_ab_{filt}',
                f'mag_vega_{filt}', f'emag_ab_{filt}', f'mask_{filt}',
                f'flux_ulim_{filt}', f'mag_ab_ulim_{filt}', f'mag_vega_ulim_{filt}',
                f'is_upper_limit_{filt}'):
        if col in tbl.colnames and hasattr(tbl[col], 'filled'):
            tbl[col] = tbl[col].filled(tbl[col].fill_value)

    ridx = tgt_idx
    # Reset every column this call owns, for EVERY current target row, before
    # writing this pass's results.  Without this a rerun over the fill's own
    # output (or a partial overlay applied twice) leaves a PRIOR pass's
    # is_upper_limit/flux_ulim/forced_snr stale on a row this pass does NOT
    # fit this time (its frames became unavailable) or DOES promote to a
    # firm detection -- `fitted`/`not_detected` only ever SET True, so
    # without a reset they can only accumulate, never clear.  `mask_{filt}`
    # needs no equivalent reset: a row this call cleared to a detection on a
    # prior pass reads mask=False from the table and is excluded from
    # `targets` (and so from `ridx`) on the next call, so it is never
    # revisited here.
    tbl[f'forced_filled_{filt}'][ridx] = False
    tbl[f'forced_snr_{filt}'][ridx] = np.nan
    tbl[f'flux_ulim_{filt}'][ridx] = np.nan
    tbl[f'mag_ab_ulim_{filt}'][ridx] = np.nan
    tbl[f'mag_vega_ulim_{filt}'][ridx] = np.nan
    tbl[f'is_upper_limit_{filt}'][ridx] = False
    if bkg_scatter:
        tbl[f'forced_bkg_sigma_{filt}'][ridx] = np.nan
    if nbrsub:
        tbl[f'forced_nbrsub_{filt}'][ridx] = False
        tbl[f'forced_near_m7_sep_{filt}'][ridx] = np.nan
    if nrand > 0:
        tbl[f'forced_err_scale_{filt}'][ridx] = np.nan
        tbl[f'forced_snr_raw_{filt}'][ridx] = np.nan
    if f'flux_{filt}' in tbl.colnames:
        tbl[f'flux_{filt}'][ridx] = np.nan
    if f'flux_jy_{filt}' in tbl.colnames:
        tbl[f'flux_jy_{filt}'][ridx] = np.nan
    if f'mag_ab_{filt}' in tbl.colnames:
        tbl[f'mag_ab_{filt}'][ridx] = np.nan
    if f'mag_vega_{filt}' in tbl.colnames:
        tbl[f'mag_vega_{filt}'][ridx] = np.nan
    if f'emag_ab_{filt}' in tbl.colnames:
        tbl[f'emag_ab_{filt}'][ridx] = np.nan

    tbl[f'forced_filled_{filt}'][ridx[fitted]] = True
    tbl[f'forced_snr_{filt}'][ridx[fitted]] = snr[fitted]
    tbl[f'flux_ulim_{filt}'][ridx[fitted]] = ulim_flux[fitted]
    tbl[f'mag_ab_ulim_{filt}'][ridx[fitted]] = ulim_mag_ab[fitted]
    tbl[f'mag_vega_ulim_{filt}'][ridx[fitted]] = ulim_mag_vega[fitted]
    tbl[f'is_upper_limit_{filt}'][ridx[not_detected]] = True
    if bkg_scatter:
        tbl[f'forced_bkg_sigma_{filt}'][ridx[fitted]] = bkg_mean[fitted]
    if nbrsub:
        tbl[f'forced_nbrsub_{filt}'][ridx[fitted]] = sub_all[fitted]
        tbl[f'forced_near_m7_sep_{filt}'][ridx[fitted]] = near_sep[fitted]
    if nrand > 0:
        tbl[f'forced_err_scale_{filt}'][ridx[fitted]] = k_scale
        tbl[f'forced_snr_raw_{filt}'][ridx[fitted]] = snr_raw[fitted]
    if f'flux_{filt}' in tbl.colnames:
        tbl[f'flux_{filt}'][ridx[fitted]] = flux[fitted]
    if f'flux_jy_{filt}' in tbl.colnames:
        tbl[f'flux_jy_{filt}'][ridx[fitted]] = fjy[fitted]
    # magnitudes AND emag only for positive flux (a flux<=0 row keeps mag NaN,
    # so writing its emag would leave an absurd error on a NaN magnitude)
    posfit = fitted & (flux > 0)
    if f'emag_ab_{filt}' in tbl.colnames:
        with np.errstate(divide='ignore', invalid='ignore'):
            tbl[f'emag_ab_{filt}'][ridx[posfit]] = 1.0857 * (ferr[posfit] / flux[posfit])
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
                              nsigma=3.0, fit_shape=(5, 5), verbose=True,
                              n_procs=1, frame_area_sr=frame_area_sr,
                              bkg_scatter=False, nbrsub=False,
                              frame_products=None, nrand=0):
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
    n_procs : forked processes per band's frame loop (``forced_fill_band``)
    frame_area_sr : callable ``frame filename -> PIXAR_SR`` (sr); the default
        reads the frame's own header (#1238).  ``None`` fills in raw image units.
    bkg_scatter : add the local annulus scatter to the fit errors
        (``forced_fill_band``); default off.
    nbrsub : fit each frame with its m7 star-only model removed
        (``forced_fill_band``); default off.
    frame_products : callable ``(ctx, frame_kwargs) -> (m7 model path, m7
        frame catalogue path)``; required with ``nbrsub``.
    nrand : random positions for the empirical error scale
        (``forced_fill_band``); default 0 = none.
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
            nsigma=nsigma, fit_shape=fit_shape, verbose=verbose,
            n_procs=n_procs, frame_area_sr=frame_area_sr,
            bkg_scatter=bkg_scatter, nbrsub=nbrsub,
            frame_products=frame_products, nrand=nrand)
    tbl.write(out_path, overwrite=True)
    if verbose:
        print(f"[m8] wrote {out_path}  ({total} phantom non-detections recovered)",
              flush=True)
    return out_path
