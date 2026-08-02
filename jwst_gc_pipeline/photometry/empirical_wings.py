"""Empirical-wing injection: make injected stars carry the frame's REAL wing
profile, not the (too-narrow) STPSF model wings.

Why: the wing-selfcal ON/OFF test showed the injection recovery bias is a harness
artifact -- injected stars have STPSF wings, but the recovery's wing-selfcal is
calibrated on REAL stars (broader wings) and divides flux down, over-correcting
the STPSF injections (hard-sat bias +1161 mmag ON vs +6 OFF). To measure the TRUE
recovery bias, the injected stars must have real wings.

``measure_empirical_wing_ratio`` stacks bright isolated UNSATURATED stars in the
frame, builds their median radial profile, and returns the multiplicative ratio
real/STPSF vs radius (normalised in the core so total core flux is preserved).
``apply_wing_ratio`` multiplies an STPSF stamp by that radial ratio so the
injected star has real wings while keeping the injected total-flux definition.
"""
import numpy as np


def measure_empirical_wing_ratio(sci, dq, grid, fwhm_pix=2.0, *, rmax=40,
                                 core_r=2.0, nmax=40, edge=110, peak_lo_frac=0.02,
                                 peak_hi_frac=0.6):
    """Return (r_edges_centres, ratio(r)) = median real/STPSF radial profile ratio,
    core-normalised (ratio->1 at r<=core_r). Bright ISOLATED unsaturated stars only.
    """
    from photutils.detection import DAOStarFinder
    from astropy.stats import sigma_clipped_stats
    from jwst.datamodels import dqflags
    P = dqflags.pixel
    bad = ~np.isfinite(sci) | ((dq & (P['SATURATED'] | P['DO_NOT_USE'])) != 0)
    _, med, std = sigma_clipped_stats(sci[~bad], sigma=3.0, maxiters=5)
    dao = DAOStarFinder(threshold=15 * std, fwhm=fwhm_pix, exclude_border=True)
    cat = dao(np.nan_to_num(sci - med), mask=bad)
    if cat is None or len(cat) == 0:
        return None, None
    xc = 'xcentroid' if 'xcentroid' in cat.colnames else 'x_centroid'
    yc = 'ycentroid' if 'ycentroid' in cat.colnames else 'y_centroid'
    x, y, peak = (np.asarray(cat[xc], float), np.asarray(cat[yc], float),
                  np.asarray(cat['peak'], float))
    ny, nx = sci.shape
    pk_hi = peak_hi_frac * np.nanmax(sci)
    pk_lo = peak_lo_frac * np.nanmax(sci)
    cand = np.where((peak > pk_lo) & (peak < pk_hi) & (x > edge) & (x < nx - edge)
                    & (y > edge) & (y < ny - edge))[0]
    # isolation
    sel = []
    for i in cand:
        d2 = (x - x[i]) ** 2 + (y - y[i]) ** 2
        d2[i] = np.inf
        if not np.any((d2 < (2 * rmax) ** 2) & (peak > 0.1 * peak[i])):
            sel.append(i)
    if not sel:
        return None, None
    sel = np.array(sel)[np.argsort(peak[np.array(sel)])[::-1]][:nmax]
    redges = np.arange(0, rmax + 1, 1.0)
    rcent = 0.5 * (redges[:-1] + redges[1:])
    real_prof, model_prof = [], []
    yy, xx = np.mgrid[0:2 * rmax + 1, 0:2 * rmax + 1]
    for i in sel:
        ix, iy = int(round(x[i])), int(round(y[i]))
        if ix - rmax < 0 or ix + rmax >= nx or iy - rmax < 0 or iy + rmax >= ny:
            continue
        cut = sci[iy - rmax:iy + rmax + 1, ix - rmax:ix + rmax + 1].astype(float)
        cut = cut - np.nanmedian(cut[-5:, :])           # crude local bkg
        mod = grid.evaluate(xx + ix - rmax, yy + iy - rmax, 1.0, x[i], y[i])
        rad = np.hypot(xx - rmax, yy - rmax)
        rp = np.array([np.nanmedian(cut[(rad >= a) & (rad < b)]) for a, b in zip(redges[:-1], redges[1:])])
        mp = np.array([np.nanmedian(mod[(rad >= a) & (rad < b)]) for a, b in zip(redges[:-1], redges[1:])])
        core = rcent <= core_r
        if np.nansum(rp[core]) <= 0 or np.nansum(mp[core]) <= 0:
            continue
        real_prof.append(rp / np.nansum(rp[core]))       # normalise in the core
        model_prof.append(mp / np.nansum(mp[core]))
    if len(real_prof) < 3:
        return None, None
    real_med = np.nanmedian(real_prof, axis=0)
    model_med = np.nanmedian(model_prof, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = real_med / model_med
    ratio[rcent <= core_r] = 1.0                          # core preserved
    ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, 1.0)
    # enforce monotonic-ish, clip runaway
    ratio = np.clip(ratio, 0.3, 20.0)
    print(f'[empwing] stacked {len(real_prof)} stars; ratio real/STPSF at '
          f'r=5/10/20/35 = ' + '/'.join(
              f'{np.interp(rr, rcent, ratio):.2f}' for rr in (5, 10, 20, 35)), flush=True)
    return rcent, ratio


def apply_wing_ratio(stamp, xx, yy, xc, yc, rcent, ratio):
    """Multiply an STPSF stamp by the radial real/STPSF ratio (extrapolate flat
    beyond the measured range). Keeps the core (ratio=1) so injected total flux
    is dominated by the intended value; wings are boosted to the real profile."""
    if rcent is None:
        return stamp
    r = np.hypot(xx - xc, yy - yc)
    rr = np.interp(r, rcent, ratio, left=1.0, right=ratio[-1])
    return stamp * rr
