"""Inject known-flux stars spanning saturation regimes, recover them with the
production satstar path, and measure recovered/injected flux vs saturation depth.

Pipeline per frame:
  1. load a real _crf (SCI MJy/sr, ERR, DQ, VAR_POISSON) + header (PHOTMJSR,
     NGROUPS, TGROUP, DETECTOR); gain from CRDS.
  2. draw injection positions on background pixels away from real stars/saturation.
  3. assign a magnitude ladder (unsaturated -> deeply saturated).
  4. for each star: build the TRUE count-rate scene (e-/s) from the PSF grid on a
     cutout (+ local background rate), run the CRDS saturation forward model
     (saturation_forward_model.simulate_cal) -> the star's cal-level SCI/DQ/VAR
     WITH nonlinearity + hard-saturation + charge-migration artefacts, and add it
     onto the frame (SCI in MJy/sr, DQ OR-ed, VAR added).
  5. run get_saturated_stars on the injected frame (GWCS-correct header per rule #2).
  6. match recovered satstars to injected positions (sky), compute
     recovered_mag - injected_mag vs saturation depth (sat_area / group-0-lost),
     write a table + figure.

Units: SCI[MJy/sr] = DN/s * PHOTMJSR ; e-/s = DN/s * gain. So
  rate_e_s = (stamp_MJy_sr / PHOTMJSR) * gain, and the star's forward-modelled
  SCI (DN/s) maps back with * PHOTMJSR.
"""
import os
import sys
import glob

os.environ.setdefault('STPSF_PATH', '/orange/adamginsburg/jwst/stpsf-data/')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from jwst_gc_pipeline.photometry import artificial_stars as art
from jwst_gc_pipeline.photometry import saturation_forward_model as sfm
from jwst.datamodels import dqflags

P = dqflags.pixel
STAMP = 60          # cutout half-size for the forward model (px)


def _hdr_val(hduls, key, default=None):
    for h in hduls:
        if key in h.header:
            return h.header[key]
    return default


def gain_median(detector):
    refs = sfm.load_detector_refs(detector, box=(1000, 1050, 1000, 1050))
    return float(np.median(refs['gain']))


def build_valid_mask(sci, dq, fwhm_pix):
    from scipy import ndimage
    sat = (dq & P['SATURATED']) != 0
    dnu = (dq & P['DO_NOT_USE']) != 0
    bad = sat | dnu | ~np.isfinite(sci)
    # avoid existing bright sources: dilate high pixels
    hi = np.nan_to_num(sci) > np.nanpercentile(sci, 97)
    bad |= ndimage.binary_dilation(hi | sat, iterations=int(max(6, 3 * fwhm_pix)))
    return ~bad


def _slice_refs(full, box):
    y0, y1, x0, x1 = box
    return dict(gain=full['gain'][y0:y1, x0:x1],
                fullwell=full['fullwell'][y0:y1, x0:x1],
                coeffs=full['coeffs'][:, y0:y1, x0:x1],
                detector=full['detector'])


def inject_one(frame_sci, frame_dq, frame_var, full_refs, grid, x, y, flux_img,
               photmjsr, gain_med, ngroups, tgroup, f_bf, local_bkg_rate_e,
               ewr=None):
    """Forward-model one star and add it onto the frame in place. Returns
    (sat_area, group0_lost). ``ewr``=(rcent,ratio) applies the empirical real/STPSF
    wing profile so injected wings match the frame (so the recovery's wing-selfcal
    is appropriate -- see empirical_wings.py)."""
    ny, nx = frame_sci.shape
    x0 = max(0, int(round(x)) - STAMP); x1 = min(nx, int(round(x)) + STAMP + 1)
    y0 = max(0, int(round(y)) - STAMP); y1 = min(ny, int(round(y)) + STAMP + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    stamp = np.clip(grid.evaluate(xx, yy, flux_img, x, y), 0, None)   # MJy/sr
    if ewr is not None:
        from jwst_gc_pipeline.photometry.empirical_wings import apply_wing_ratio
        stamp = np.clip(apply_wing_ratio(stamp, xx, yy, x, y, ewr[0], ewr[1]), 0, None)
    # true count-rate scene (e-/s): star + local background
    rate_e = stamp / photmjsr * gain_med + local_bkg_rate_e
    cutrefs = _slice_refs(full_refs, (y0, y1, x0, x1))     # slice preloaded refs
    sim = sfm.simulate_cal(rate_e, cutrefs, ngroups=ngroups, tgroup=tgroup,
                           f_bf=f_bf)
    # star-only cal contribution (subtract the background we added in)
    star_dns = sim['sci'] - local_bkg_rate_e / gain_med    # DN/s
    star_mjy = star_dns * photmjsr                          # MJy/sr
    # add onto frame; propagate DQ + variance; NaN where truly lost
    reg_sci = frame_sci[y0:y1, x0:x1]
    reg_sci += np.where(np.isfinite(star_mjy), star_mjy, np.nan)
    frame_dq[y0:y1, x0:x1] |= sim['dq']
    add_var = np.nan_to_num((sim['var_poisson'] * photmjsr ** 2))
    frame_var[y0:y1, x0:x1] += add_var
    sat_area = int(((sim['dq'] & P['SATURATED']) != 0).sum())
    g0lost = int(((sim['dq'] & P['DO_NOT_USE']) != 0).sum())
    return sat_area, g0lost


def baseline_unsat(sci, err, dq, grid, inj, band, pixar_sr, fwhm_pix):
    """Unsaturated baseline: forced PSF photometry at the injected positions of
    the UNSATURATED stars (sat_area==0). dmag should be ~0 -- validates the
    harness end (injection+recovery unbiased where nothing saturates)."""
    from photutils.psf import PSFPhotometry
    from astropy.table import Table as _T
    sel = np.asarray(inj['sat_area']) == 0
    if sel.sum() < 3:
        return np.full(len(inj), np.nan)
    init = _T({'x_0': np.asarray(inj['x'])[sel], 'y_0': np.asarray(inj['y'])[sel]})
    phot = PSFPhotometry(psf_model=grid, fit_shape=(5, 5),
                         aperture_radius=max(3, 2 * fwhm_pix))
    badmask = (dq & (P['DO_NOT_USE'] | P['SATURATED'])) != 0
    res = phot(np.nan_to_num(sci), error=err, mask=badmask, init_params=init)
    rec_mag = art.imflux_to_mag(np.asarray(res['flux_fit'], float), band, pixar_sr)
    out = np.full(len(inj), np.nan)
    out[np.where(sel)[0]] = rec_mag - np.asarray(inj['mag_inj'])[sel]
    return out


def run(crf, band, detector, field, psfs, out, f_bf=0.3, n_stars=200,
        mag_lo=8.5, mag_hi=14.5, seed=0, empirical_wings=True):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    fh = fits.open(crf)
    sci = np.asarray(fh['SCI'].data, float)
    err = np.asarray(fh['ERR'].data, float) if 'ERR' in fh else np.sqrt(np.abs(sci))
    dq = np.asarray(fh['DQ'].data, np.uint32)
    var = (np.asarray(fh['VAR_POISSON'].data, float) if 'VAR_POISSON' in fh
           else err ** 2)
    photmjsr = float(_hdr_val(fh, 'PHOTMJSR', 1.0))
    ngroups = int(_hdr_val(fh, 'NGROUPS', 7))
    tgroup = float(_hdr_val(fh, 'TGROUP', 10.0))
    gain_med = gain_median(detector)
    pixar_sr = float(_hdr_val(fh, 'PIXAR_SR', 2.29e-14))
    fwhm_pix = {'F200W': 2.0}.get(band, 2.0)
    print(f'[inj] {os.path.basename(crf)} band={band} det={detector} '
          f'NGROUPS={ngroups} TGROUP={tgroup} PHOTMJSR={photmjsr:.4g} gain={gain_med:.2f} '
          f'f_bf={f_bf}', flush=True)

    grid = art.load_psf_grid(band, detector.lower())
    valid = build_valid_mask(sci, dq, fwhm_pix)
    xs, ys = art.draw_positions(rng, *sci.shape, n_stars, valid)
    if len(xs) == 0:
        raise SystemExit('[inj] no valid injection positions')
    mags = np.linspace(mag_lo, mag_hi, len(xs))
    rng.shuffle(mags)
    local_bkg_rate_e = np.nanmedian(np.clip(sci, 0, None)) / photmjsr * gain_med
    full_refs = sfm.load_detector_refs(detector)      # load CRDS refs ONCE (full det)
    ewr = None
    if empirical_wings:
        from jwst_gc_pipeline.photometry.empirical_wings import measure_empirical_wing_ratio
        rcent, ratio = measure_empirical_wing_ratio(sci, dq, grid, fwhm_pix=fwhm_pix)
        ewr = (rcent, ratio) if rcent is not None else None
        print(f'[inj] empirical wings: {"measured" if ewr else "UNAVAILABLE -> STPSF wings"}',
              flush=True)

    inj = []
    for x, y, mag in zip(xs, ys, mags):
        flux_img = float(art.mag_to_imflux(mag, band, pixar_sr))
        sa, g0 = inject_one(sci, dq, var, full_refs, grid,
                            x, y, flux_img, photmjsr, gain_med, ngroups, tgroup,
                            f_bf, local_bkg_rate_e, ewr=ewr)
        inj.append((x, y, mag, flux_img, sa, g0))
    inj = Table(rows=inj, names=('x', 'y', 'mag_inj', 'flux_inj', 'sat_area', 'g0lost'))
    print(f'[inj] injected {len(inj)} stars; sat_area range {inj["sat_area"].min()}'
          f'..{inj["sat_area"].max()}', flush=True)

    # write injected frame + recover
    fh['SCI'].data = sci.astype(np.float32)
    fh['DQ'].data = dq
    if 'VAR_POISSON' in fh:
        fh['VAR_POISSON'].data = var.astype(np.float32)
    if 'ERR' in fh:
        fh['ERR'].data = np.sqrt(np.abs(var)).astype(np.float32)
    # GWCS-correct primary header BEFORE writing (rule #2: read GWCS, not SIP)
    hdr = fh[0].header
    if 'CRPIX1' not in hdr:
        from jwst_gc_pipeline.frame_wcs import frame_wcs as _fw
        from jwst_gc_pipeline.reduction.fits_wcs_sync import sync_header_to_gwcs
        sync_header_to_gwcs(hdr, _fw(fh).gwcs, fh['SCI'].data.shape, label='inject')
    injfits = f'{out}/injected_{field}_{band}_{detector}.fits'
    fh.writeto(injfits, overwrite=True)

    from jwst_gc_pipeline.reduction import saturated_star_finding as ssf
    cat = ssf.get_saturated_stars(fits.open(injfits), path_prefix=psfs, plot=False,
                                  use_merged_psf_for_merged=False)
    if cat is None or not len(cat):
        raise SystemExit('[inj] recovery returned no satstars')

    # match recovered -> injected (sky). match_to_catalog_sky forbids NaN in the
    # catalog, and some recovered satstars have unmeasurable (NaN) skycoord_fit,
    # so match only against the finite-position subset and map indices back.
    from jwst_gc_pipeline.frame_wcs import frame_wcs
    w = frame_wcs(fh).gwcs
    ra_i, dec_i = w(inj['x'].astype(float), inj['y'].astype(float))
    sc_i_all = SkyCoord(np.asarray(ra_i, float) * u.deg, np.asarray(dec_i, float) * u.deg)
    sc_r_all = SkyCoord(cat['skycoord_fit'])
    fin_r = np.isfinite(sc_r_all.ra.deg) & np.isfinite(sc_r_all.dec.deg)
    fin_i = np.isfinite(sc_i_all.ra.deg) & np.isfinite(sc_i_all.dec.deg)
    rec_mag = np.full(len(inj), np.nan)
    sep_mas = np.full(len(inj), np.nan)
    if fin_r.sum() and fin_i.sum():
        cat_fin = cat[fin_r]
        idx, sep, _ = sc_i_all[fin_i].match_to_catalog_sky(sc_r_all[fin_r])
        rf = np.asarray(cat_fin['flux_fit'][idx], float)
        rm = art.imflux_to_mag(rf, band, pixar_sr)
        ii = np.where(fin_i)[0]
        rec_mag[ii] = rm
        sep_mas[ii] = sep.mas
    ok = sep_mas < 150.0   # 0.15"
    rec_mag = np.where(ok, rec_mag, np.nan)
    inj['matched'] = ok
    inj['mag_rec'] = rec_mag
    inj['dmag'] = inj['mag_rec'] - inj['mag_inj']       # +ve = recovered too faint
    inj['sep_mas'] = sep_mas
    # unsaturated baseline (forced daophot-style fit; should be ~0)
    try:
        inj['dmag_baseline'] = baseline_unsat(sci, err, dq, grid, inj, band,
                                              pixar_sr, fwhm_pix)
        _b = inj['dmag_baseline'][np.isfinite(inj['dmag_baseline'])]
        if len(_b):
            print(f'[inj] unsat baseline: N={len(_b)} dmag_med={np.median(_b)*1000:+.0f}mmag '
                  f'(should be ~0)', flush=True)
    except (ValueError, RuntimeError, KeyError) as e:
        print(f'[inj] baseline skipped: {e}', flush=True)
        inj['dmag_baseline'] = np.full(len(inj), np.nan)
    inj.write(f'{out}/inj_recovery_{field}_{band}_{detector}.ecsv', overwrite=True)
    print(f'[inj] matched {int(ok.sum())}/{len(inj)} within 0.15"', flush=True)

    _plot(inj, field, band, detector, ngroups, out)
    return inj


def _plot(inj, field, band, detector, ngroups, out):
    m = inj['matched'] & np.isfinite(inj['dmag'])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    a = ax[0]
    sat = np.asarray(inj['sat_area'], float)
    a.scatter(np.asarray(inj['mag_inj'])[m], np.asarray(inj['dmag'])[m] * 1000,
              c=sat[m], cmap='viridis', s=25)
    a.axhline(0, color='0.5'); a.axhline(20, color='r', ls=':'); a.axhline(-20, color='r', ls=':')
    a.set_xlabel('injected mag (Vega)'); a.set_ylabel('recovered - injected (mmag)')
    a.set_title(f'(a) recovery bias vs brightness')
    a.set_ylim(-300, 300)
    a = ax[1]
    order = np.argsort(sat[m])
    a.scatter(sat[m], np.asarray(inj['dmag'])[m] * 1000, s=25, color='tab:purple')
    a.axhline(0, color='0.5'); a.axhline(20, color='r', ls=':'); a.axhline(-20, color='r', ls=':')
    a.set_xlabel('saturated-pixel area (depth)'); a.set_ylabel('recovered - injected (mmag)')
    a.set_title('(b) recovery bias vs saturation depth'); a.set_ylim(-300, 300)
    fig.suptitle(f'Saturated-star recovery bias — {field} {band} {detector} '
                 f'(NGROUPS={ngroups})', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = f'{out}/inj_recovery_{field}_{band}_{detector}.png'
    fig.savefig(p, dpi=120); print(f'[inj] wrote {p}', flush=True)


if __name__ == '__main__':
    import json
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    run(**cfg)
