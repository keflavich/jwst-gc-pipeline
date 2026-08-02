"""Calibrate the charge-migration coupling f_bf against the observed #210
fit-flux-vs-footprint sensitivity of gc2211 (NGROUPS=2, charge_migration OFF).

For a given f_bf: inject bright stars into a gc2211 F200W frame via the forward
model, recover twice -- fit_shape=31 and fit_shape=81 -- and measure
R = flux81/flux31 for the matched injected saturated stars. The charge-migration
excess (kept because NGROUPS=2) inflates the wings, so R rises with f_bf. The
target is the real gc2211 R(size31 vs 81) ~ 1.10-1.18 (docs/reports/
SATSTAR_FIT_FOOTPRINT.md matched-r_core table). Run over f_bf and pick the match.

Usage: calibrate_fbf.py <f_bf>
"""
import os
import sys
import json

os.environ.setdefault('STPSF_PATH', '/orange/adamginsburg/jwst/stpsf-data/')
os.environ.setdefault('CRDS_PATH', '/orange/adamginsburg/jwst/crds')

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

from jwst_gc_pipeline.photometry import artificial_stars as art
from jwst_gc_pipeline.photometry import saturation_forward_model as sfm
from jwst_gc_pipeline.photometry import saturated_injection_recovery as sir

GC = '/orange/adamginsburg/jwst/gc2211'
CRF = (f'{GC}/F200W/pipeline/jw02211023001_02201_00001_nrca1_destreak_'
       f'jw02211-o023_20260515t014922_image3_00001_crf.fits')
PSFS = f'{GC}/psfs/'
BAND, DET, FIELD = 'F200W', 'NRCA1', 'gc2211'
OUT = os.environ.get('FBF_OUT', '/blue/adamginsburg/adamginsburg/tmp/satinject_fbfcal')


def _recover_flux(injfits, fh, inj_x, inj_y, size):
    from jwst_gc_pipeline.reduction import saturated_star_finding as ssf
    from jwst_gc_pipeline.frame_wcs import frame_wcs
    cat = ssf.get_saturated_stars(fits.open(injfits), path_prefix=PSFS, plot=False,
                                  use_merged_psf_for_merged=False, size=size)
    if cat is None or not len(cat):
        return np.full(len(inj_x), np.nan)
    w = frame_wcs(fh).gwcs
    ra, dec = w(inj_x.astype(float), inj_y.astype(float))
    sc_i = SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg)
    sc_r = SkyCoord(cat['skycoord_fit'])
    fin = np.isfinite(sc_r.ra.deg) & np.isfinite(sc_r.dec.deg)
    out = np.full(len(inj_x), np.nan)
    if fin.sum():
        idx, sep, _ = sc_i.match_to_catalog_sky(sc_r[fin])
        f = np.asarray(cat[fin]['flux_fit'][idx], float)
        out = np.where(sep.arcsec < 0.15, f, np.nan)
    return out


def run(f_bf, n_stars=120, seed=3):
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(seed)
    fh = fits.open(CRF)
    sci = np.asarray(fh['SCI'].data, float)
    err = np.asarray(fh['ERR'].data, float) if 'ERR' in fh else np.sqrt(np.abs(sci))
    dq = np.asarray(fh['DQ'].data, np.uint32)
    var = np.asarray(fh['VAR_POISSON'].data, float) if 'VAR_POISSON' in fh else err ** 2
    photmjsr = float(sir._hdr_val(fh, 'PHOTMJSR', 1.93))
    ngroups = int(sir._hdr_val(fh, 'NGROUPS', 2))
    tgroup = float(sir._hdr_val(fh, 'TGROUP', 53.68))
    pixar_sr = float(sir._hdr_val(fh, 'PIXAR_SR', 2.29e-14))
    gain_med = sir.gain_median(DET)
    grid = art.load_psf_grid(BAND, DET.lower())
    valid = sir.build_valid_mask(sci, dq, 2.0)
    xs, ys = art.draw_positions(rng, *sci.shape, n_stars, valid)
    mags = rng.uniform(9.0, 12.5, len(xs))     # all saturate at NGROUPS=2
    lb = np.nanmedian(np.clip(sci, 0, None)) / photmjsr * gain_med
    full_refs = sfm.load_detector_refs(DET)
    for x, y, mag in zip(xs, ys, mags):
        flux = float(art.mag_to_imflux(mag, BAND, pixar_sr))
        sir.inject_one(sci, dq, var, full_refs, grid, x, y, flux, photmjsr,
                       gain_med, ngroups, tgroup, f_bf, lb)
    fh['SCI'].data = sci.astype(np.float32); fh['DQ'].data = dq
    if 'VAR_POISSON' in fh:
        fh['VAR_POISSON'].data = var.astype(np.float32)
    hdr = fh[0].header
    if 'CRPIX1' not in hdr:
        from jwst_gc_pipeline.frame_wcs import frame_wcs as _fw
        from jwst_gc_pipeline.reduction.fits_wcs_sync import sync_header_to_gwcs
        sync_header_to_gwcs(hdr, _fw(fh).gwcs, fh['SCI'].data.shape, label='fbfcal')
    injfits = f'{OUT}/inj_gc2211_fbf{f_bf}.fits'
    fh.writeto(injfits, overwrite=True)

    f31 = _recover_flux(injfits, fh, xs, ys, 31)
    f81 = _recover_flux(injfits, fh, xs, ys, 81)
    m = np.isfinite(f31) & np.isfinite(f81) & (f31 > 0) & (f81 > 0)
    R = f81[m] / f31[m]
    res = dict(f_bf=f_bf, n_matched=int(m.sum()),
               R_median=float(np.median(R)) if m.sum() else float('nan'),
               R_std=float(np.std(R)) if m.sum() else float('nan'))
    json.dump(res, open(f'{OUT}/fbfcal_{f_bf}.json', 'w'))
    print(f'[fbfcal] f_bf={f_bf}: R(81/31)_median={res["R_median"]:.3f} '
          f'(N={res["n_matched"]}) target~1.10-1.18', flush=True)
    return res


if __name__ == '__main__':
    run(float(sys.argv[1]) if len(sys.argv) > 1 else 0.0)
