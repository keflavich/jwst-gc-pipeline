"""Artificial-star (injection-recovery) completeness tests.

Injects gridded-PSF point sources at known Vega magnitudes into COPIES of
per-exposure ``crf`` frames, re-runs the pipeline's first-pass (m1-style)
daofind + daophot-BASIC detection/photometry path on both the original and the
injected copy, and records per-star recovery so completeness can be measured
as a function of magnitude and of local surface brightness (the selection
function P(detected | mag, Sigma) needed for dust mapping / depth calibration).

Fidelity to the production vetted-catalog path (``cataloging.py`` m1):

* PSF: the same per-detector webbpsf cache grids the production per-frame
  path loads (``nircam_{det}_{filt}_fovp101_samp2_npsf16.fits``), used for
  BOTH injection and fitting so the flux scale is exactly self-consistent.
* detection: ``DAOStarFinder(threshold=min(local_noise), fwhm=fwhm_pix,
  roundlo=-1, roundhi=1, sharplo=0.30, sharphi=1.40)`` followed by the
  ``annotate_and_filter_by_local_snr`` local-S/N >= 5 filter -- the m1 recipe
  (``cataloging.py`` line ~2030) with the local noise map from
  ``compute_local_noise_map``.
* photometry: ``PSFPhotometry(finder=None, LocalBackground(inner, outer),
  fit_shape=(5,5), aperture_radius=2*fwhm)`` with error=1e10 on bad pixels --
  the ``_fit_and_clean_frame`` construction.

Deliberate simplifications (single-frame, no multi-pass): no saturated-star
model subtraction, no m2..m6 residual re-detection passes, and no cross-frame
nmatch vetting.  The measured completeness therefore corresponds to the
single-exposure BASIC catalogs; the final vetted catalog requires detection
in >=2 frames, so P_vetted ~ P_basic^2-ish at the faint limit (frames overlap
~4x).  Recovery criterion: a fitted source within ``match_radius_pix`` whose
recovered magnitude is within 0.75 mag of the injected one (the classic
DAOPHOT artificial-star convention; the mag gate stops a pre-existing bright
star from masquerading as a recovery).

Originals are never modified: frames are copied into ``--workdir`` first.

Usage (one frame per task; see submit_artificial_star_completeness.sh)::

    python -m jwst_gc_pipeline.photometry.artificial_stars run \
        --band F212N --detector nrca1 --n-stars 1250 --seed 0 \
        --workdir /blue/adamginsburg/adamginsburg/jwst/brick/artificial_star_tests

    python -m jwst_gc_pipeline.photometry.artificial_stars analyze \
        --workdir /blue/adamginsburg/adamginsburg/jwst/brick/artificial_star_tests
"""
import argparse
import glob
import os
import shutil

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy import wcs as astropy_wcs
from astropy.modeling.fitting import LevMarLSQFitter
from photutils.background import LocalBackground
from photutils.detection import DAOStarFinder
from scipy import ndimage
from scipy.spatial import cKDTree

# Pipeline building blocks (the m1 detection path + PSFPhotometry shim).
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS

# Detection parameters SINGLE-SOURCED from production (review #130 item 1):
# m1/m2 S/N and the m2 shape bounds come from MANUAL_DEFAULTS; the m1 daofind
# shape bounds mirror the literals in cataloging.py (m1 recipe,
# 'sharplo=0.30, sharphi=1.40' / crowdsource daofind_roundlo/hi defaults) and
# are pinned against drift by test_artificial_stars_params_match_production.
M1_SNR = MANUAL_DEFAULTS['local_snr_threshold']
M2_SNR = MANUAL_DEFAULTS['manual_iter2_local_snr']
M1_ROUNDLO, M1_ROUNDHI, M1_SHARPLO, M1_SHARPHI = -1.0, 1.0, 0.30, 1.40
M2_ROUNDLO = MANUAL_DEFAULTS['manual_resid_roundlo']
M2_ROUNDHI = MANUAL_DEFAULTS['manual_resid_roundhi']
M2_SHARPLO = MANUAL_DEFAULTS['manual_resid_sharplo']
M2_SHARPHI = MANUAL_DEFAULTS['manual_resid_sharphi']

from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import (
    compute_local_noise_map, annotate_and_filter_by_local_snr,
    _bad_dq_bitmask)
from jwst_gc_pipeline.photometry.psf_fitting import (
    _make_psfphotometry, _make_model_image)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DETECTORS = ['nrca1', 'nrca2', 'nrca3', 'nrca4',
             'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']

#: Per-band frame source + injected-magnitude grid.  The mag ranges span the
#: number-count turnover for each band (F212N ~19.5, F200W ~21 single-frame).
BANDS = {
    'F212N': dict(
        pipedir='/blue/adamginsburg/adamginsburg/jwst/brick/F212N/pipeline',
        frame_glob='jw02221001001_*_{det}_destreak_o001_crf.fits',
        mag_min=15.0, mag_max=21.0,
        proposal='2221',
    ),
    'F200W': dict(
        pipedir='/blue/adamginsburg/adamginsburg/jwst/brick/F200W/pipeline',
        frame_glob='jw01182004001_*_{det}_destreak_o004_crf.fits',
        mag_min=16.0, mag_max=22.0,
        proposal='1182',
    ),
}

#: SVO Vega zero points (Jy) -- same source (astroquery SvoFps 'JWST' list,
#: filterID JWST/NIRCam.*) as merge_catalogs.py uses for mag_vega.  Hardcoded
#: so SLURM tasks don't depend on the SVO service being up.
VEGA_ZEROPOINT_JY = {
    'F212N': 674.83167374035,
    'F200W': 757.65380461946,
}

PSF_DIR = '/blue/adamginsburg/adamginsburg/jwst/brick/psfs'
DEFAULT_WORKDIR = '/blue/adamginsburg/adamginsburg/jwst/brick/artificial_star_tests'

#: PSF FWHM in pixels (fwhm_table.ecsv values used by the pipeline).
FWHM_PIX = {'F212N': 2.341, 'F200W': 2.141}

#: DAOPHOT-convention recovery gates.
MATCH_RADIUS_PIX = 1.5
MAG_TOLERANCE = 0.75

#: minimum separation between injected stars (avoid self-crowding; the test
#: measures crowding from the REAL field, not from the injected population).
MIN_INJECT_SEP_PIX = 12.0

EDGE_BUFFER_PIX = 16


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def load_psf_grid(band, detector):
    """The per-detector webbpsf cache grid the production per-frame path uses."""
    from stpsf.utils import to_griddedpsfmodel
    fn = os.path.join(PSF_DIR,
                      f'nircam_{detector}_{band.lower()}_fovp101_samp2_npsf16.fits')
    if not os.path.exists(fn):
        raise FileNotFoundError(fn)
    grid = to_griddedpsfmodel(fn)
    if isinstance(grid, list):
        grid = grid[0]
    return grid


def mag_to_imflux(mag_vega, band, pixar_sr):
    """Vega mag -> total PSF flux in image units (MJy/sr summed over pixels).

    Inverse of merge_catalogs: flux_jy = flux_img * PIXAR_SR * 1e6;
    mag_vega = -2.5 log10(flux_jy / ZP).
    """
    flux_jy = VEGA_ZEROPOINT_JY[band] * 10 ** (-0.4 * np.asarray(mag_vega))
    return flux_jy / (1e6 * pixar_sr)


def imflux_to_mag(flux_img, band, pixar_sr):
    flux_jy = np.asarray(flux_img, dtype=float) * 1e6 * pixar_sr
    with np.errstate(invalid='ignore', divide='ignore'):
        return -2.5 * np.log10(flux_jy / VEGA_ZEROPOINT_JY[band])


def draw_positions(rng, ny, nx, n_stars, valid_mask,
                   min_sep=MIN_INJECT_SEP_PIX, edge=EDGE_BUFFER_PIX,
                   max_tries=200):
    """Uniform positions on valid pixels with a minimum mutual separation."""
    xs, ys = [], []
    tree, tree_n, buffer = None, 0, []   # amortized: rebuild every 64 accepts
    tries = 0
    while len(xs) < n_stars and tries < max_tries * n_stars:
        tries += 1
        x = rng.uniform(edge, nx - 1 - edge)
        y = rng.uniform(edge, ny - 1 - edge)
        if not valid_mask[int(round(y)), int(round(x))]:
            continue
        if tree is not None and tree.query([x, y], k=1)[0] < min_sep:
            continue
        if buffer and np.hypot(np.array(buffer)[:, 0] - x,
                               np.array(buffer)[:, 1] - y).min() < min_sep:
            continue
        xs.append(x)
        ys.append(y)
        buffer.append([x, y])
        if len(buffer) >= 64:
            pts = ([] if tree is None else tree.data.tolist()) + buffer
            tree = cKDTree(pts)
            tree_n = len(pts)
            buffer = []
    return np.array(xs), np.array(ys)


def inject_stars(data, err, grid, xs, ys, fluxes, rng, gain_eff=None,
                 stamp_half=25):
    """Add PSF stars (with approximate Poisson scatter) to data in place;
    returns the updated err array (err added in quadrature)."""
    ny, nx = data.shape
    err2 = err ** 2
    for x, y, flux in zip(xs, ys, fluxes):
        x0 = max(0, int(np.floor(x)) - stamp_half)
        x1 = min(nx, int(np.floor(x)) + stamp_half + 1)
        y0 = max(0, int(np.floor(y)) - stamp_half)
        y1 = min(ny, int(np.floor(y)) + stamp_half + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        stamp = grid.evaluate(xx, yy, flux, x, y)
        stamp = np.clip(stamp, 0, None)
        if gain_eff is not None and gain_eff > 0:
            # approximate per-pixel Poisson scatter of the injected star
            electrons = stamp * gain_eff
            stamp_noisy = rng.poisson(np.clip(electrons, 0, None)) / gain_eff
            var_add = stamp / gain_eff
        else:
            stamp_noisy = stamp
            var_add = 0.0
        data[y0:y1, x0:x1] += stamp_noisy
        err2[y0:y1, x0:x1] += var_add
    return np.sqrt(err2)


def estimate_effective_gain(sci, var_poisson):
    """Median effective gain g such that var_poisson ~ sci / g (per-pixel,
    image units).  Used only to give injected stars approximate shot noise."""
    good = (np.isfinite(sci) & np.isfinite(var_poisson)
            & (sci > 0) & (var_poisson > 0))
    if good.sum() < 1000:
        return None
    g = np.median(sci[good] / var_poisson[good])
    return float(g) if np.isfinite(g) and g > 0 else None


# ---------------------------------------------------------------------------
# m1-style detection + BASIC PSF photometry (pipeline-parameter mirror)
# ---------------------------------------------------------------------------

def _daofind_snr(image, badmask, fwhm_pix, *, roundlo, roundhi, sharplo,
                 sharphi, snr_threshold, label=''):
    """daofind at the min-local-noise floor + the local-S/N filter (the
    pipeline's per-pass detection primitive)."""
    masked = np.where(badmask, np.nan, image)
    noise_map = compute_local_noise_map(masked, smooth_sigma_pix=3.0)
    finite = np.isfinite(noise_map) & (noise_map > 0)
    if not finite.any():
        raise ValueError(f'[{label}] no valid noise pixels')
    gmin = float(np.nanmin(noise_map[finite]))
    det = DAOStarFinder(threshold=gmin, fwhm=fwhm_pix,
                        roundlo=roundlo, roundhi=roundhi,
                        sharplo=sharplo, sharphi=sharphi)(
                            np.nan_to_num(masked), mask=badmask)
    if det is None or len(det) == 0:
        return Table()
    det, stats = annotate_and_filter_by_local_snr(
        det, noise_map, snr_threshold=snr_threshold)
    print(f'[{label}] daofind {stats["input_count"]} -> '
          f'{stats["kept_count"]} after local S/N>={snr_threshold:g}',
          flush=True)
    return det


def _det_xy(det):
    xcol = 'xcentroid' if 'xcentroid' in det.colnames else 'x_centroid'
    ycol = 'ycentroid' if 'ycentroid' in det.colnames else 'y_centroid'
    return (np.asarray(det[xcol], dtype=float),
            np.asarray(det[ycol], dtype=float))


def detect_and_fit(data, err, dq, grid, fwhm_pix, instrument='NIRCAM',
                   label=''):
    """Two-pass (m1 + m2-style residual) daofind + daophot-BASIC fit with the
    pipeline's parameters.  Returns the fitted photutils result table.

    The residual pass matters: ``compute_local_noise_map`` is a high-pass
    statistic, so an ISOLATED star inflates its own local noise and saturates
    at local S/N ~4.7 no matter how bright it is -- below the m1 threshold of
    5.  In production such stars are picked up by the residual passes at
    S/N >= 3 (``manual_iter2_local_snr``); without this second pass the
    bright-end completeness would be badly underestimated.
    """
    badbits = _bad_dq_bitmask(instrument)
    badmask = (~np.isfinite(data)) | ((dq.astype(np.uint32) & badbits) != 0)

    # --- pass 1: m1 parameters (S/N >= 5, wide shape bounds) ---
    det1 = _daofind_snr(data, badmask, fwhm_pix,
                        roundlo=M1_ROUNDLO, roundhi=M1_ROUNDHI,
                        sharplo=M1_SHARPLO, sharphi=M1_SHARPHI,
                        snr_threshold=M1_SNR, label=f'{label} m1')

    aperture_radius_pix = 2.0 * fwhm_pix
    localbkg_inner = max(6, int(round(aperture_radius_pix + 0.5 * fwhm_pix)))
    localbkg_outer = localbkg_inner + max(4, int(round(fwhm_pix)))

    def _fit(init):
        phot = _make_psfphotometry(
            finder=None,
            localbkg_estimator=LocalBackground(localbkg_inner, localbkg_outer),
            grouper=None,
            psf_model=grid,
            fitter=LevMarLSQFitter(),
            fit_shape=(5, 5),
            aperture_radius=aperture_radius_pix,
            progress_bar=False,
        )
        result = phot(np.nan_to_num(data), mask=badmask, init_params=init,
                      error=np.where(badmask, 1e10, err))
        return phot, result

    x1, y1 = _det_xy(det1) if len(det1) else (np.array([]), np.array([]))
    init1 = Table({'x_init': x1, 'y_init': y1})
    if len(init1) == 0:
        raise ValueError(f'[{label}] no m1 detections')
    phot1, res1 = _fit(init1)

    # --- pass 2: re-detect on the star-subtracted residual (m2 parameters:
    #     S/N >= 3, manual_resid_* shape bounds) ---
    psf_npix = 2 * int(np.ceil(3.5 * fwhm_pix)) + 1
    model = _make_model_image(phot1, data.shape, psf_shape=(psf_npix, psf_npix))
    residual = data - model
    det2 = _daofind_snr(residual, badmask, fwhm_pix,
                        roundlo=M2_ROUNDLO, roundhi=M2_ROUNDHI,
                        sharplo=M2_SHARPLO, sharphi=M2_SHARPHI,
                        snr_threshold=M2_SNR, label=f'{label} m2')

    if len(det2):
        x2, y2 = _det_xy(det2)
        # keep only residual detections not already fit (dedup 0.5*fwhm)
        tree = cKDTree(np.column_stack([x1, y1]))
        d, _ = tree.query(np.column_stack([x2, y2]))
        new = d > 0.5 * fwhm_pix
        print(f'[{label}] residual pass adds {new.sum()} new sources',
              flush=True)
        if new.any():
            init_union = Table({'x_init': np.concatenate([x1, x2[new]]),
                                'y_init': np.concatenate([y1, y2[new]])})
            _, result = _fit(init_union)
        else:
            result = res1
    else:
        result = res1

    ok = np.isfinite(np.asarray(result['x_fit'], dtype=float))
    result = result[ok]
    print(f'[{label}] fitted {len(result)} sources', flush=True)
    return result


# ---------------------------------------------------------------------------
# Per-frame run
# ---------------------------------------------------------------------------

def pick_frame(band, detector):
    cfg = BANDS[band]
    pat = os.path.join(cfg['pipedir'], cfg['frame_glob'].format(det=detector))
    frames = sorted(glob.glob(pat))
    if not frames:
        raise FileNotFoundError(f'no frames match {pat}')
    return frames[0]


def run_frame(band, detector, workdir, n_stars=1250, seed=0, smoke=False):
    cfg = BANDS[band]
    frame = pick_frame(band, detector)
    stem = os.path.basename(frame).replace('.fits', '')
    outdir = os.path.join(workdir, band)
    os.makedirs(outdir, exist_ok=True)
    label = f'{band} {detector}'

    # never touch the original: work on a copy
    frame_copy = os.path.join(outdir, os.path.basename(frame))
    if not os.path.exists(frame_copy):
        shutil.copy(frame, frame_copy)

    with fits.open(frame_copy) as hdul:
        sci = np.asarray(hdul['SCI'].data, dtype=float)
        err = np.asarray(hdul['ERR'].data, dtype=float)
        dq = np.asarray(hdul['DQ'].data)
        var_poisson = np.asarray(hdul['VAR_POISSON'].data, dtype=float)
        pixar_sr = float(hdul['SCI'].header['PIXAR_SR'])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ww = astropy_wcs.WCS(hdul['SCI'].header)

    if smoke:
        sci = sci[:512, :512].copy()
        err = err[:512, :512].copy()
        dq = dq[:512, :512].copy()
        var_poisson = var_poisson[:512, :512].copy()
        n_stars = min(n_stars, 100)

    fwhm_pix = FWHM_PIX[band]
    grid = load_psf_grid(band, detector)
    rng = np.random.default_rng(seed * 100 + DETECTORS.index(detector))

    # --- baseline catalog (original frame, identical path) ---
    base = detect_and_fit(sci, err, dq, grid, fwhm_pix, label=f'{label} base')
    base_xy = np.column_stack([np.asarray(base['x_fit'], dtype=float),
                               np.asarray(base['y_fit'], dtype=float)])
    base_tree = cKDTree(base_xy)
    base_flux = np.asarray(base['flux_fit'], dtype=float)

    # --- injection ---
    ny, nx = sci.shape
    valid = np.isfinite(sci)
    xs, ys = draw_positions(rng, ny, nx, n_stars, valid)
    mags = rng.uniform(cfg['mag_min'], cfg['mag_max'], size=len(xs))
    fluxes = mag_to_imflux(mags, band, pixar_sr)

    # local surface brightness (Sigma proxy) from the ORIGINAL frame:
    # 31-px median-filtered SCI sampled at the injected positions.
    sb_map = ndimage.median_filter(np.nan_to_num(sci), size=31)
    noise_map = compute_local_noise_map(np.where(valid, sci, np.nan),
                                        smooth_sigma_pix=3.0)
    ix = np.clip(np.rint(xs).astype(int), 0, nx - 1)
    iy = np.clip(np.rint(ys).astype(int), 0, ny - 1)
    local_sb = sb_map[iy, ix]
    local_noise = noise_map[iy, ix]

    gain_eff = estimate_effective_gain(sci, var_poisson)
    sci_inj = sci.copy()
    err_inj = inject_stars(sci_inj, err.copy(), grid, xs, ys, fluxes, rng,
                           gain_eff=gain_eff)

    tag = '_smoke' if smoke else ''
    inj_fn = os.path.join(outdir, f'{stem}_artstar_seed{seed}{tag}.fits')
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(sci_inj.astype(np.float32), name='SCI'),
                  fits.ImageHDU(err_inj.astype(np.float32), name='ERR')]
                 ).writeto(inj_fn, overwrite=True)

    # --- recovery run (identical path on the injected frame) ---
    rec = detect_and_fit(sci_inj, err_inj, dq, grid, fwhm_pix,
                         label=f'{label} inj')
    rec_xy = np.column_stack([np.asarray(rec['x_fit'], dtype=float),
                              np.asarray(rec['y_fit'], dtype=float)])
    rec_tree = cKDTree(rec_xy)
    rec_flux = np.asarray(rec['flux_fit'], dtype=float)
    rec_mag = imflux_to_mag(rec_flux, band, pixar_sr)

    # --- match ---
    d_rec, i_rec = rec_tree.query(np.column_stack([xs, ys]),
                                  distance_upper_bound=MATCH_RADIUS_PIX)
    matched = np.isfinite(d_rec) & (d_rec <= MATCH_RADIUS_PIX)
    mag_out = np.full(len(xs), np.nan)
    flux_out = np.full(len(xs), np.nan)
    sep_pix = np.full(len(xs), np.nan)
    qfit_out = np.full(len(xs), np.nan)
    qfit_col = (np.asarray(rec['qfit'], dtype=float)
                if 'qfit' in rec.colnames else None)
    mag_out[matched] = rec_mag[i_rec[matched]]
    flux_out[matched] = rec_flux[i_rec[matched]]
    sep_pix[matched] = d_rec[matched]
    if qfit_col is not None:
        qfit_out[matched] = qfit_col[i_rec[matched]]
    recovered = matched & (np.abs(mag_out - mags) <= MAG_TOLERANCE)

    # pre-existing-source context at the injection site
    d_base, i_base = base_tree.query(np.column_stack([xs, ys]))
    base_mag_near = imflux_to_mag(base_flux[np.clip(i_base, 0,
                                                    len(base_flux) - 1)],
                                  band, pixar_sr)

    truth = Table()
    truth['x'] = xs
    truth['y'] = ys
    sky = ww.pixel_to_world(xs, ys)
    truth['ra'] = sky.ra.deg
    truth['dec'] = sky.dec.deg
    truth['mag_in'] = mags
    truth['flux_in'] = fluxes
    truth['mag_out'] = mag_out
    truth['flux_out'] = flux_out
    truth['sep_pix'] = sep_pix
    truth['qfit'] = qfit_out
    truth['matched'] = matched
    truth['recovered'] = recovered
    truth['local_sb'] = local_sb
    truth['local_noise'] = local_noise
    truth['d_nearest_base_pix'] = d_base
    truth['mag_nearest_base'] = base_mag_near
    truth.meta['band'] = band
    truth.meta['detector'] = detector
    truth.meta['frame'] = frame
    truth.meta['seed'] = seed
    truth.meta['nstars'] = len(xs)
    truth.meta['gaineff'] = gain_eff if gain_eff is not None else -1
    truth.meta['pixarsr'] = pixar_sr
    truth.meta['matchrad'] = MATCH_RADIUS_PIX
    truth.meta['magtol'] = MAG_TOLERANCE
    truth.meta['smoke'] = smoke

    truth_fn = os.path.join(outdir, f'{stem}_artstar_seed{seed}{tag}_truth.fits')
    truth.write(truth_fn, overwrite=True)
    base.write(os.path.join(outdir, f'{stem}_artstar{tag}_basecat.fits'),
               overwrite=True)
    rec.write(os.path.join(outdir, f'{stem}_artstar_seed{seed}{tag}_reccat.fits'),
              overwrite=True)

    frac = recovered.mean() if len(recovered) else np.nan
    print(f'[{label}] injected {len(xs)}, matched {matched.sum()}, '
          f'recovered {recovered.sum()} ({100 * frac:.1f}%)  -> {truth_fn}',
          flush=True)
    return truth_fn


# ---------------------------------------------------------------------------
# Aggregation / analysis
# ---------------------------------------------------------------------------

def completeness_curve(mags, recovered, bins):
    idx = np.digitize(mags, bins) - 1
    nbin = len(bins) - 1
    n_inj = np.zeros(nbin, dtype=int)
    n_rec = np.zeros(nbin, dtype=int)
    for b in range(nbin):
        sel = idx == b
        n_inj[b] = sel.sum()
        n_rec[b] = recovered[sel].sum()
    with np.errstate(invalid='ignore', divide='ignore'):
        frac = np.where(n_inj > 0, n_rec / np.maximum(n_inj, 1), np.nan)
        # binomial (Wald) error; fine for reporting
        efrac = np.where(n_inj > 0,
                         np.sqrt(np.clip(frac * (1 - frac), 0, None)
                                 / np.maximum(n_inj, 1)), np.nan)
    return n_inj, n_rec, frac, efrac


def mag_at_completeness(bin_centers, frac, level):
    """First (brightest-to-faintest) crossing of ``level`` by interpolation."""
    good = np.isfinite(frac)
    bc, fr = bin_centers[good], frac[good]
    for i in range(1, len(fr)):
        if fr[i - 1] >= level > fr[i]:
            return float(np.interp(level, [fr[i], fr[i - 1]],
                                   [bc[i], bc[i - 1]]))
    return np.nan


def analyze(workdir, bin_width=0.25):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    summary_rows = []
    fig, axes = plt.subplots(1, len(BANDS), figsize=(6 * len(BANDS), 4.5),
                             squeeze=False)
    for iband, band in enumerate(sorted(BANDS)):
        fns = sorted(glob.glob(os.path.join(workdir, band,
                                            '*_truth.fits')))
        fns = [fn for fn in fns if 'smoke' not in os.path.basename(fn)]
        if not fns:
            print(f'{band}: no truth tables found, skipping')
            continue
        tbls = [Table.read(fn) for fn in fns]
        allt = Table(np.concatenate([np.asarray(t) for t in tbls]))
        mags = np.asarray(allt['mag_in'], dtype=float)
        rec = np.asarray(allt['recovered'], dtype=bool)
        sb = np.asarray(allt['local_sb'], dtype=float)

        cfg = BANDS[band]
        bins = np.arange(cfg['mag_min'], cfg['mag_max'] + bin_width / 2,
                         bin_width)
        bc = 0.5 * (bins[:-1] + bins[1:])
        n_inj, n_rec, frac, efrac = completeness_curve(mags, rec, bins)
        m50 = mag_at_completeness(bc, frac, 0.5)
        m90 = mag_at_completeness(bc, frac, 0.9)

        # completeness split by local surface-brightness terciles
        q1, q2 = np.nanpercentile(sb, [33.3, 66.7])
        tiers = [('low_sb', sb <= q1), ('mid_sb', (sb > q1) & (sb <= q2)),
                 ('high_sb', sb > q2)]
        ax = axes[0][iband]
        ax.errorbar(bc, frac, yerr=efrac, color='k', lw=2, label='all')
        for name, sel in tiers:
            _, _, f_t, _ = completeness_curve(mags[sel], rec[sel], bins)
            ax.plot(bc, f_t, alpha=0.7, label=name)
            m50_t = mag_at_completeness(bc, f_t, 0.5)
            summary_rows.append(dict(band=band, subset=name,
                                     n_injected=int(sel.sum()),
                                     n_recovered=int(rec[sel].sum()),
                                     m50=m50_t,
                                     m90=mag_at_completeness(bc, f_t, 0.9),
                                     sb_lo=(float(np.nanmin(sb[sel]))
                                            if sel.any() else np.nan),
                                     sb_hi=(float(np.nanmax(sb[sel]))
                                            if sel.any() else np.nan)))
        summary_rows.append(dict(band=band, subset='all',
                                 n_injected=int(len(mags)),
                                 n_recovered=int(rec.sum()),
                                 m50=m50, m90=m90,
                                 sb_lo=float(np.nanmin(sb)),
                                 sb_hi=float(np.nanmax(sb))))
        ax.axhline(0.5, color='gray', ls=':')
        ax.set_xlabel(f'{band} Vega mag (injected)')
        ax.set_ylabel('recovery fraction')
        ax.set_title(f'{band}: m50={m50:.2f}, m90={m90:.2f}')
        ax.legend(loc='lower left', fontsize=8)
        ax.set_ylim(0, 1.05)

        curve = Table(dict(mag=bc, n_injected=n_inj, n_recovered=n_rec,
                           completeness=frac, e_completeness=efrac))
        curve.meta['band'] = band
        curve.meta['m50'] = m50
        curve.meta['m90'] = m90
        curve_fn = os.path.join(workdir, f'completeness_curve_{band}.ecsv')
        curve.write(curve_fn, overwrite=True, format='ascii.ecsv')
        print(f'{band}: {len(mags)} injected, m50={m50:.2f}, m90={m90:.2f} '
              f'-> {curve_fn}')

    if summary_rows:
        summ = Table(rows=summary_rows)
        summ_fn = os.path.join(workdir, 'completeness_summary.ecsv')
        summ.write(summ_fn, overwrite=True, format='ascii.ecsv')
        fig.tight_layout()
        fig_fn = os.path.join(workdir, 'completeness_curves.png')
        fig.savefig(fig_fn, dpi=150)
        print(f'summary -> {summ_fn}\nfigure  -> {fig_fn}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)

    runp = sub.add_parser('run', help='inject+recover one frame')
    runp.add_argument('--band', required=True, choices=sorted(BANDS))
    runp.add_argument('--detector', required=True, choices=DETECTORS)
    runp.add_argument('--n-stars', type=int, default=1250)
    runp.add_argument('--seed', type=int, default=0)
    runp.add_argument('--workdir', default=DEFAULT_WORKDIR)
    runp.add_argument('--smoke', action='store_true',
                      help='512x512 cutout, <=100 stars (fast validation)')

    anap = sub.add_parser('analyze', help='aggregate completeness curves')
    anap.add_argument('--workdir', default=DEFAULT_WORKDIR)
    anap.add_argument('--bin-width', type=float, default=0.25)

    args = ap.parse_args()
    if args.cmd == 'run':
        run_frame(args.band, args.detector, args.workdir,
                  n_stars=args.n_stars, seed=args.seed, smoke=args.smoke)
    else:
        analyze(args.workdir, bin_width=args.bin_width)


if __name__ == '__main__':
    main()
