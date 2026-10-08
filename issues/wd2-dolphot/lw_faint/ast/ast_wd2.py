"""Injection-recovery test of the jwst-gc-pipeline fitting path on wd2 frames.

Wrapper around jwst_gc_pipeline.photometry.artificial_stars (read-only import).
Supplies wd2 frames, PSF grids, FWHM, Vega zero points.  One frame per call;
the SAME injected stars are fit in two variants:
  (a) 'raw'   : injected crf as in the module (LocalBackground annulus on crf)
  (b) 'resbg' : after subtracting the production smoothed-residual background
                (m6 smoothed_bg mosaic reprojected onto the frame exactly as
                cataloging.py does before the m7 fit) from the injected frame.

Usage: python ast_wd2.py run --band F410M --det nrcblong --exp 1 [--smoke]
"""
import argparse
import glob
import os
import warnings

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy import wcs as astropy_wcs
from scipy import ndimage
from scipy.spatial import cKDTree

import jwst_gc_pipeline.photometry.artificial_stars as AS
from jwst_gc_pipeline.frame_wcs import frame_wcs
from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import compute_local_noise_map

ROOT = '/orange/adamginsburg/jwst/wd2'
OUT = os.path.join(ROOT, 'dolphot_benchmark/X_lw_scatter/ast')
PSF_DIR = os.path.join(ROOT, 'psfs')

# SVO zero points (Jy), the values merge_catalogs.py uses (jfilts ZeroPoint)
ZP_JY = {'F410M': 208.75045059896, 'F405N': 206.96939908611,
         'F277W': 430.1399020248, 'F200W': 757.65380461946}
# fwhm_table.ecsv, PSF FWHM (pixel)
FWHM = {'F410M': 2.179, 'F405N': 2.165, 'F277W': 1.444, 'F200W': 2.141}
MAGRANGE = {'F410M': (16, 23), 'F405N': (15, 22), 'F277W': (16, 23), 'F200W': (16, 23)}
PIPEDIR = {b: f'{ROOT}/{b}/pipeline' for b in ZP_JY}
# smoothed residual bg m7 subtracts: the m6 smoothed_bg mosaic of the production
# run behind matched_Q_mainfcbg (tree_mainfcbg) when it exists, else the main tree
BGTREE = f'{ROOT}/dolphot_benchmark/Q_integ/tree_mainfcbg'
PROPOSAL_PREFIX = 'jw03523005001'
VISIT = {'F410M': '16101', 'F405N': '06101', 'F277W': '10101', 'F200W': '12101'}
SEED_BASE = 20261008

AS.VEGA_ZEROPOINT_JY.update(ZP_JY)


def psf_detector(det):
    return det[:4] + '5' if det.endswith('long') else det


def load_grid(band, det):
    from stpsf.utils import to_griddedpsfmodel
    fn = f'{PSF_DIR}/nircam_{psf_detector(det)}_{band.lower()}_fovp101_samp2_npsf16.fits'
    grid = to_griddedpsfmodel(fn)
    if isinstance(grid, list):
        grid = grid[0]
    grid.flux.min = 0
    return grid, fn


def bg_path(band):
    p = (f'{BGTREE}/{band}/pipeline/jw03523-o005_t001_nircam_clear-{band.lower()}-merged_'
         f'resbgsub_m6_daophot_basic_mergedcat_residual_smoothed_bg_i2d.fits')
    if not os.path.exists(p):
        p = (f'{PIPEDIR[band]}/jw03523-o005_t001_nircam_clear-{band.lower()}-merged_'
             f'resbgsub_m6_daophot_basic_mergedcat_residual_smoothed_bg_i2d.fits')
    return p


def reproject_bg(path, ww, shape, cutslice):
    """Replicates cataloging.py: crop mosaic to frame footprint, reproject_interp."""
    from reproject import reproject_interp
    ny, nx = shape
    with fits.open(path) as bgh:
        hdu = bgh['SCI'] if 'SCI' in [h.name for h in bgh] else bgh[0]
        bw = astropy_wcs.WCS(hdu.header)
        foot = ww.calc_footprint(axes=(nx, ny))
        bx, by = bw.world_to_pixel_values(foot[:, 0], foot[:, 1])
        m = 64
        x0 = max(int(np.floor(np.nanmin(bx))) - m, 0)
        x1 = min(int(np.ceil(np.nanmax(bx))) + m, hdu.shape[1])
        y0 = max(int(np.floor(np.nanmin(by))) - m, 0)
        y1 = min(int(np.ceil(np.nanmax(by))) + m, hdu.shape[0])
        bdat = hdu.section[y0:y1, x0:x1].astype(float)
        bw = bw[y0:y1, x0:x1]
    rep, _ = reproject_interp((bdat, bw), getattr(ww, 'fits_wcs', ww), shape_out=(ny, nx))
    rep = np.where(np.isfinite(rep), rep, 0.0)
    return rep[cutslice]


def run(band, det, exp, nstars, smoke, outdir):
    os.makedirs(outdir, exist_ok=True)
    frame = f'{PIPEDIR[band]}/{PROPOSAL_PREFIX}_{VISIT[band]}_{exp:05d}_{det}_align_o005_crf.fits'
    stem = f'{band}_{det}_exp{exp}' + ('_smoke' if smoke else '')
    with fits.open(frame) as h:   # read-only, in-memory copies
        sci = np.asarray(h['SCI'].data, float)
        err = np.asarray(h['ERR'].data, float)
        dq = np.asarray(h['DQ'].data)
        var_p = np.asarray(h['VAR_POISSON'].data, float)
        pixar = float(h['SCI'].header['PIXAR_SR'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ww = frame_wcs(h)
    full_shape = sci.shape
    bg_full = reproject_bg(bg_path(band), ww, full_shape, (slice(None), slice(None)))
    pixarea_wcs = float(ww.proj_plane_pixel_area().to('sr').value) if hasattr(ww, 'proj_plane_pixel_area') else np.nan
    sl = (slice(1500, 2012), slice(500, 1012)) if smoke else (slice(None), slice(None))
    y0, x0 = sl[0].start or 0, sl[1].start or 0
    sci, err, dq, var_p, bg = (a[sl].copy() for a in (sci, err, dq, var_p, bg_full))
    if smoke:
        nstars = min(nstars, 100)
    fwhm = FWHM[band]
    grid, psf_fn = load_grid(band, det)
    seed = SEED_BASE + 100 * list(ZP_JY).index(band) + 10 * exp + int(det.startswith('nrcb'))
    rng = np.random.default_rng(seed)
    label = f'{band} {det} exp{exp}'
    print(f'[{label}] frame={frame} seed={seed} pixar_sr={pixar:.4e} wcs_area_sr={pixarea_wcs:.4e}', flush=True)

    # baseline (raw frame) for nearest pre-existing source
    base = AS.detect_and_fit(sci, err, dq, grid, fwhm, label=f'{label} base')
    bxy = np.column_stack([np.asarray(base['x_fit'], float), np.asarray(base['y_fit'], float)])
    btree = cKDTree(bxy)
    bflux = np.asarray(base['flux_fit'], float)

    ny, nx = sci.shape
    valid = np.isfinite(sci)
    xs, ys = AS.draw_positions(rng, ny, nx, nstars, valid)
    lo, hi = MAGRANGE[band]
    mags = rng.uniform(lo, hi, size=len(xs))
    # production flux scale: flux_jy = flux_img * pixel_area_sr(wcs) * 1e6
    pix_for_flux = pixarea_wcs if np.isfinite(pixarea_wcs) else pixar
    fluxes = AS.mag_to_imflux(mags, band, pix_for_flux)
    sb_map = ndimage.median_filter(np.nan_to_num(sci), size=31)
    ix = np.clip(np.rint(xs).astype(int), 0, nx - 1)
    iy = np.clip(np.rint(ys).astype(int), 0, ny - 1)
    local_sb = sb_map[iy, ix]
    local_bg_map = bg[iy, ix]
    gain = AS.estimate_effective_gain(sci, var_p)
    sci_inj = sci.copy()
    err_inj = AS.inject_stars(sci_inj, err.copy(), grid, xs, ys, fluxes, rng, gain_eff=gain)

    dbase, ibase = btree.query(np.column_stack([xs, ys]))
    mag_base_near = AS.imflux_to_mag(bflux[ibase], band, pix_for_flux)

    variants = {'raw': sci_inj}
    zeros = sci_inj == 0
    sub = sci_inj - bg
    sub[zeros] = 0
    variants['resbg'] = sub
    sky = ww.pixel_to_world(xs + x0, ys + y0)
    for vname, img in variants.items():
        rec = AS.detect_and_fit(img.astype('float32'), err_inj, dq, grid, fwhm, label=f'{label} {vname}')
        rxy = np.column_stack([np.asarray(rec['x_fit'], float), np.asarray(rec['y_fit'], float)])
        rflux = np.asarray(rec['flux_fit'], float)
        rmag = AS.imflux_to_mag(rflux, band, pix_for_flux)
        rtree = cKDTree(rxy)
        d, i = rtree.query(np.column_stack([xs, ys]), distance_upper_bound=AS.MATCH_RADIUS_PIX)
        matched = np.isfinite(d)
        mag_out = np.full(len(xs), np.nan)
        flux_out = np.full(len(xs), np.nan)
        qfit = np.full(len(xs), np.nan)
        mag_out[matched] = rmag[i[matched]]
        flux_out[matched] = rflux[i[matched]]
        if 'qfit' in rec.colnames:
            qfit[matched] = np.asarray(rec['qfit'], float)[i[matched]]
        recovered = matched & (np.abs(mag_out - mags) <= AS.MAG_TOLERANCE)
        t = Table()
        t['x'] = xs; t['y'] = ys
        t['ra'] = sky.ra.deg; t['dec'] = sky.dec.deg
        t['mag_in'] = mags; t['flux_in'] = fluxes
        t['mag_out'] = mag_out; t['flux_out'] = flux_out; t['qfit'] = qfit
        t['matched'] = matched; t['recovered'] = recovered
        t['local_sb'] = local_sb; t['smooth_bg_at_star'] = local_bg_map
        t['d_nearest_base_pix'] = dbase; t['mag_nearest_base'] = mag_base_near
        t.meta.update(dict(band=band, detector=det, exp=exp, variant=vname, seed=seed,
                           nstars=len(xs), gaineff=gain if gain else -1, pixarsr=pixar,
                           zpjy=ZP_JY[band], fwhm=fwhm, psf=psf_fn, frame=frame,
                           bgmosaic=bg_path(band), nbase=len(base)))
        fn = f'{outdir}/{stem}_{vname}_truth.fits'
        t.write(fn, overwrite=True)
        dm = (mag_out - mags)[matched]
        sel = matched & (mags > 19) & (mags < 21)
        print(f'[{label} {vname}] matched {matched.sum()}/{len(xs)} recovered {recovered.sum()}; '
              f'median dm (19-21, all matched) = {np.nanmedian((mag_out - mags)[sel]):+.3f}  -> {fn}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--band', required=True)
    ap.add_argument('--det', required=True)
    ap.add_argument('--exp', type=int, default=1)
    ap.add_argument('--n-stars', type=int, default=1000)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--outdir', default=f'{OUT}/truth')
    a = ap.parse_args()
    run(a.band, a.det, a.exp, a.n_stars, a.smoke, a.outdir if not a.smoke else f'{OUT}/smoke')
