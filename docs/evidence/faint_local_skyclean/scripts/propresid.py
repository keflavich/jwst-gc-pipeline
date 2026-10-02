"""Residual stamps for a catalog other than the one the production residual subtracted.

The production m6 residual i2d is data - model(production m6 vetted catalog),
resampled from the per-frame residuals.  For a stamp, the residual of another
catalog C built from the same m6 fit table is

    R_C = R_prod + s * sum_{prod \\ C} f_i P(x - x_i) - s * sum_{C \\ prod} f_i P(x - x_i)

with P the effective i2d PSF in catalog-flux units.  P is stacked from the
production model image M = data - R_prod itself (noise-free, and it carries the
per-frame PSF, the resampling and the catalog flux scale): stamps of M around
isolated production-kept stars, background-subtracted with the 0.8-1.0" annulus
median, divided by catalog flux and median-combined on a 2x oversampled grid of
offsets from the catalog position.  calibrate() then fits one scale s on other
production-kept stars (s near 1 means P is in catalog-flux units) and reports
frac_rms = rms(M - s*PSF sum)/rms(M), the fraction of the production model the
PSF sum does not reproduce.

Production membership is by position: the production vetted catalog is a
row subset of the m6 fit table, so its stars sit at the same coordinates.
"""
import os

import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy import wcs
from scipy.ndimage import map_coordinates
from scipy.stats import binned_statistic_2d

R = '/orange/adamginsburg/jwst'
PROD = {'brick': f'{R}/brick/catalogs/f182m_merged_o001_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits',
        'sgrb2': f'{R}/sgrb2/catalogs/f187n_merged_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits',
        'w51': f'{R}/w51/catalogs/f187n_merged_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits'}
OVERSAMPLE = 2
HALF_PSF = 1.0  # arcsec: PSF model footprint half-width


def sky(t):
    s = t['skycoord']
    return s if isinstance(s, SkyCoord) else SkyCoord(s)


def isolated(i, sc_all, flux, pb, r=0.8, frac=0.3):
    """No other production-kept star brighter than frac * flux[i] within r arcsec."""
    sep = sc_all[pb].separation(sc_all[i]).arcsec
    return not np.any((sep > 0.01) & (sep < r) & (flux[pb] > frac * flux[i]))


def psf_stars(flux, prod, sat):
    """Unsaturated production-kept stars between the 60th and 95th flux percentile.

    Saturated stars' catalog flux comes from the satstar fit and their i2d
    cores are masked, so their model image per unit catalog flux differs
    from an ordinary star's; the brightest unsaturated stars sit close to
    saturation.
    """
    ok = prod & np.isfinite(flux) & ~sat
    lo, hi = np.nanpercentile(flux[ok], [60, 95])
    return ok & (flux > lo) & (flux < hi)


def empirical_psf(sc_all, flux, prod, data, resid, sat, n=400, half=1.0, seed=5):
    """Median-stacked effective PSF from the production model image (see module docstring)."""
    (dd, wd), (rd, wr) = data, resid
    scale = wcs.utils.proj_plane_pixel_scales(wd)[0] * 3600
    npx = int(round(half / scale))
    rng = np.random.default_rng(seed)
    cand = rng.permutation(np.flatnonzero(psf_stars(flux, prod, sat)))
    pb = np.flatnonzero(prod & np.isfinite(flux))
    dx_all, dy_all, v_all, used = [], [], [], []
    for i in cand:
        if len(used) >= n:
            break
        if not isolated(i, sc_all, flux, pb):
            continue
        try:
            cd = Cutout2D(dd, sc_all[i], (2 * npx + 1, 2 * npx + 1), wcs=wd, mode='partial', fill_value=np.nan)
            cr = Cutout2D(rd, sc_all[i], (2 * npx + 1, 2 * npx + 1), wcs=wr, mode='partial', fill_value=np.nan)
        except (ValueError, wcs.NoConvergence):
            continue
        if cd.data.shape != cr.data.shape:
            continue
        m = np.asarray(cd.data, float) - np.asarray(cr.data, float)
        x, y = cd.wcs.world_to_pixel(sc_all[i])
        yy, xx = np.indices(m.shape, dtype=float)
        r = np.hypot(xx - x, yy - y) * scale
        ann = np.isfinite(m) & (r > 0.8) & (r < 1.0)
        if not np.isfinite(m[r < 0.15]).all() or ann.sum() < 50:
            continue
        ok = np.isfinite(m) & (r < half)
        dx_all.append((xx - x)[ok]); dy_all.append((yy - y)[ok])
        v_all.append((m[ok] - np.median(m[ann])) / flux[i])
        used.append(i)
    dx, dy, v = map(np.concatenate, (dx_all, dy_all, v_all))
    nb = 2 * npx * OVERSAMPLE + 1
    edges = (np.arange(nb + 1) - (nb - 1) / 2 - 0.5) / OVERSAMPLE
    psf = binned_statistic_2d(dy, dx, v, 'median', bins=[edges, edges]).statistic
    return np.nan_to_num(psf, nan=0.0), used


def prod_member(field, sc_all):
    """Boolean per m6-table row: in the production vetted catalog (same position within 1 mas)."""
    ref = sky(Table.read(PROD[field]))
    idx, sep, _ = ref.match_to_catalog_sky(sc_all)
    m = np.zeros(len(sc_all), bool)
    m[idx[sep.to_value('mas') < 1]] = True
    return m, int((sep.to_value('mas') >= 1).sum())


def render(shape, xs, ys, fluxes, psf):
    """Sum of fluxes[i] * psf centred at (xs[i], ys[i]) on an image of shape (pixel coords)."""
    out = np.zeros(shape)
    c = (psf.shape[0] - 1) / 2
    yy, xx = np.indices(shape, dtype=float)
    for x, y, f in zip(xs, ys, fluxes):
        if not np.isfinite(f):
            continue
        u = (xx - x) * OVERSAMPLE + c
        v = (yy - y) * OVERSAMPLE + c
        out += f * map_coordinates(psf, [v, u], order=1, mode='constant', cval=0.0)
    return out


def stamp_model(cut_wcs, shape, sc_all, flux, mask, centre, psf, pad=HALF_PSF):
    """Model of the masked sources within the stamp (plus a PSF half-width) on a cutout grid."""
    scale = wcs.utils.proj_plane_pixel_scales(cut_wcs)[0] * 3600
    near = np.flatnonzero(mask & (sc_all.separation(centre).arcsec < pad + np.hypot(*shape) / 2 * scale))
    if near.size == 0:
        return np.zeros(shape)
    x, y = cut_wcs.world_to_pixel(sc_all[near])
    return render(shape, x, y, flux[near], psf)


def calibrate(psf, sc_all, flux, prod, data, resid, sat, n=150, half=1.5, seed=3, iso=False, exclude=()):
    """Fit s on stamps around moderately bright production-kept stars.

    iso=False: random stars (crowded stamps); iso=True: held-out isolated stars
    (not in ``exclude``, the stack), the case the gallery model is for.
    """
    (dd, wd), (rd, wr) = data, resid
    pb = np.flatnonzero(prod & np.isfinite(flux))
    excl = set(int(e) for e in exclude)
    rng = np.random.default_rng(seed)
    cand = np.flatnonzero(psf_stars(flux, prod, sat))
    if iso:
        pick = [i for i in rng.permutation(cand)[:20 * n]
                if int(i) not in excl and isolated(i, sc_all, flux, pb)][:n]
    else:
        pick = rng.choice(cand, min(n, cand.size), replace=False)
    px = half / (wcs.utils.proj_plane_pixel_scales(wd)[0] * 3600)
    A, B = [], []
    for i in pick:
        try:
            cd = Cutout2D(dd, sc_all[i], (2 * px, 2 * px), wcs=wd, mode='partial', fill_value=np.nan)
            cr = Cutout2D(rd, sc_all[i], (2 * px, 2 * px), wcs=wr, mode='partial', fill_value=np.nan)
        except (ValueError, wcs.NoConvergence):
            continue
        if cd.data.shape != cr.data.shape:
            continue
        model = np.asarray(cd.data, float) - np.asarray(cr.data, float)
        pm = stamp_model(cd.wcs, model.shape, sc_all, flux, prod, sc_all[i], psf)
        ok = np.isfinite(model) & np.isfinite(pm)
        if ok.sum() < 0.8 * model.size:
            continue
        # per-stamp constant absorbs any background difference between data and residual
        A.append(pm[ok] - pm[ok].mean())
        B.append(model[ok] - model[ok].mean())
    a, b = np.concatenate(A), np.concatenate(B)
    s = float(np.dot(a, b) / np.dot(a, a))
    frac = float(np.sqrt(np.mean((b - s * a) ** 2)) / np.sqrt(np.mean(b ** 2)))
    return s, frac, len(A)


def main(field):
    """Print the calibration for one field (used to fill the gallery captions)."""
    import json
    here = os.path.dirname(os.path.abspath(__file__))
    from added_gallery import CFG, load
    band = CFG[field]['band']
    base = Table.read(f'{here}/out/{field}_{band}_seed.fits')
    prov = json.loads(base.meta['PROVJSON'])
    sc_all = sky(base)
    prod, nmiss = prod_member(field, sc_all)
    kb = np.asarray(base['kept'], bool)
    print(f'{field}: production vetted rows {prod.sum()} (unmatched {nmiss}); base kept {kb.sum()}; '
          f'base-only {np.sum(kb & ~prod)}, production-only {np.sum(prod & ~kb)}')
    flux = np.asarray(base['flux'], float)
    data, resid = load(prov['data_i2d']), load(CFG[field]['resid'])
    sat = np.asarray(base['is_saturated'], bool)
    psf, used = empirical_psf(sc_all, flux, prod, data, resid, sat)
    np.save(f'{here}/out/{field}_{band}_epsf.npy', psf)
    s_iso, frac_iso, n_iso = calibrate(psf, sc_all, flux, prod, data, resid, sat, iso=True, exclude=used, seed=7)
    s, frac, n = calibrate(psf, sc_all, flux, prod, data, resid, sat)
    # close pairs: kept rows within 2 px of each other (possible duplicate fits)
    kp = sc_all[prod]
    _, d2, _ = kp.match_to_catalog_sky(kp, nthneighbor=2)
    close = float(np.mean(d2.arcsec < 0.0625))
    print(f'{field}: effective PSF from {len(used)} isolated stars, sum {psf.sum() / OVERSAMPLE ** 2:.3f}; '
          f'held-out isolated: s = {s_iso:.3f}, rms(model - s*PSF sum)/rms(model) = {frac_iso:.3f} ({n_iso} stamps); '
          f'random stamps: s = {s:.3f}, frac {frac:.3f} ({n}); kept rows with another within 2 px: {close:.3f}')
    with open(f'{here}/out/{field}_{band}_epsf.json', 'w') as fh:
        json.dump(dict(n_stack=len(used), s_iso=s_iso, frac_rms_iso=frac_iso, n_iso=n_iso,
                       s_random=s, frac_rms_random=frac, n_random=n, close_pair_frac=close), fh)
    return s_iso, frac_iso


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
