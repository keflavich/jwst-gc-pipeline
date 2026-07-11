"""Measure empirical PSF wing / STPSF-model wing ratio vs radius.

For each bright unsaturated star:
  * cut a stamp, mask DQ-flagged pixels
  * subtract a local background (robust median in a 55-70 px annulus)
  * refine the centroid on the core
  * compute an azimuthal-MEDIAN radial profile (robust to neighbors in a
    crowded field)
  * evaluate the GriddedPSFModel at the same detector position + sub-pixel
    phase and compute the same profile
  * normalize model to data on the CORE (r in [1.5, 6] px)
  * per-star ratio curve  C_i(r) = data(r) / (A_core * model(r))
Stack C_i(r) with a median across stars.

Usage: python measure_wing_ratio.py <detector> <filter> [nframes]
e.g.   python measure_wing_ratio.py nrca1 f182m 8
"""
import sys
import glob
import warnings
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import maximum_filter, median_filter

warnings.filterwarnings('ignore')

SCRATCH = ('/blue/adamginsburg/adamginsburg/tmp/claude-3663/'
           '-blue-adamginsburg-adamginsburg-repos-jwst-gc-pipeline/'
           'd046d21c-902e-4ed6-b230-4b20ac93f54c/scratchpad/agent_empsf')

DET = sys.argv[1] if len(sys.argv) > 1 else 'nrca1'
FILT = sys.argv[2] if len(sys.argv) > 2 else 'f182m'
NFRAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 8

FRAME_GLOB = (f'/blue/adamginsburg/adamginsburg/jwst/brick/{FILT.upper()}/'
              f'pipeline/jw02221001001_*_{DET}_destreak_o001_crf.fits')
PSF_FN = (f'/blue/adamginsburg/adamginsburg/jwst/brick/psfs/'
          f'nircam_{DET}_{FILT}_fovp101_samp2_npsf16.fits')

PEAK_MIN, PEAK_MAX = 300.0, 3500.0     # MJy/sr, unsaturated bright stars
HALF = 55                              # stamp half-size (px); profile to r=50
EDGE = HALF + 5
BG_R_IN, BG_R_OUT = 55.0, 70.0         # local background annulus
CORE_NORM_R = (1.5, 6.0)               # radii used to scale model to data
R_MAX = 46.0                           # stay inside model 101-px FOV edge

# radial bins: 0.5 px inside r=8, 1 px to 30, 2 px beyond
r_edges = np.concatenate([np.arange(0, 8, 0.5),
                          np.arange(8, 30, 1.0),
                          np.arange(30, R_MAX + 2, 2.0)])
r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])


def _clipped_median(vals, sigma=2.5, iters=4):
    """Sigma-clipped median (same estimator for data and model annuli)."""
    v = vals[np.isfinite(vals)]
    for _ in range(iters):
        if len(v) < 3:
            break
        med = np.median(v)
        sd = 1.4826 * np.median(np.abs(v - med))
        if sd == 0:
            break
        keep = np.abs(v - med) < sigma * sd
        if keep.all():
            break
        v = v[keep]
    return np.median(v) if len(v) else np.nan


def radial_profile(stamp, mask_bad, cx, cy):
    """Azimuthal clipped-median in each radial bin (bg-subtracted stamp)."""
    yy, xx = np.mgrid[0:stamp.shape[0], 0:stamp.shape[1]]
    rr = np.hypot(xx - cx, yy - cy)
    prof = np.full(len(r_mid), np.nan)
    npix = np.zeros(len(r_mid), int)
    good = np.isfinite(stamp) & ~mask_bad
    for i in range(len(r_mid)):
        sel = (rr >= r_edges[i]) & (rr < r_edges[i + 1]) & good
        n = sel.sum()
        if n >= 3:
            prof[i] = _clipped_median(stamp[sel])
            npix[i] = n
    return prof, npix


def refine_centroid(stamp, x0, y0, box=7):
    """Flux-weighted centroid on the core box."""
    h = box // 2
    ix, iy = int(round(x0)), int(round(y0))
    sub = stamp[iy - h:iy + h + 1, ix - h:ix + h + 1]
    if sub.shape != (box, box) or not np.all(np.isfinite(sub)):
        sub = np.nan_to_num(sub, nan=0.0)
        if sub.size == 0:
            return x0, y0
    sub = np.clip(sub, 0, None)
    yy, xx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    tot = sub.sum()
    if tot <= 0:
        return x0, y0
    return ix - h + (sub * xx).sum() / tot, iy - h + (sub * yy).sum() / tot


def main():
    from stpsf.utils import to_griddedpsfmodel
    grid = to_griddedpsfmodel(PSF_FN)
    if isinstance(grid, list):
        grid = grid[0]
    grid.flux = 1.0

    frames = sorted(glob.glob(FRAME_GLOB))[:NFRAMES]
    print(f'{DET} {FILT}: using {len(frames)} frames')

    all_ratio, all_dprof, all_mprof, star_meta = [], [], [], []

    for fn in frames:
        with fits.open(fn) as hdul:
            sci = hdul['SCI'].data.astype(float)
            dq = hdul['DQ'].data
        bad = (dq & 1).astype(bool) | ~np.isfinite(sci)  # DO_NOT_USE
        sat = (dq & 2).astype(bool)                       # SATURATED

        # smooth background field for detection thresholding
        med = np.nanmedian(sci)

        # local maxima above PEAK_MIN
        scif = np.where(bad, med, sci)
        mx = maximum_filter(scif, size=5)
        peaks = (scif == mx) & (scif > PEAK_MIN + med)
        py, px = np.nonzero(peaks)
        keep = ((px > EDGE) & (px < 2048 - EDGE) & (py > EDGE) &
                (py < 2048 - EDGE))
        py, px = py[keep], px[keep]
        pk = scif[py, px]

        # all significant maxima (for isolation test)
        nb_thresh = 20.0
        nbpk = (scif == mx) & (scif > nb_thresh + med)
        ny, nx = np.nonzero(nbpk)
        nv = scif[ny, nx]

        nstar_this = 0
        for xi, yi, pki in zip(px, py, pk):
            if not (PEAK_MIN <= pki - med <= PEAK_MAX):
                continue
            # saturation anywhere within 5 px of core -> reject
            if sat[yi - 5:yi + 6, xi - 5:xi + 6].any():
                continue
            # isolation: no neighbor >5% of peak within 12 px
            d2 = (nx - xi) ** 2 + (ny - yi) ** 2
            near = (d2 > 4) & (d2 < 12 ** 2)
            if np.any(nv[near] - med > 0.05 * (pki - med)):
                continue

            st = sci[yi - HALF - 20:yi + HALF + 21,
                     xi - HALF - 20:xi + HALF + 21].copy()
            stbad = bad[yi - HALF - 20:yi + HALF + 21,
                        xi - HALF - 20:xi + HALF + 21]
            c0 = HALF + 20
            cx, cy = refine_centroid(np.where(stbad, np.nan, st), c0, c0)

            # local background: robust median in annulus
            yy, xx2 = np.mgrid[0:st.shape[0], 0:st.shape[1]]
            rr = np.hypot(xx2 - cx, yy - cy)
            bgsel = (rr > BG_R_IN) & (rr < BG_R_OUT) & ~stbad & np.isfinite(st)
            if bgsel.sum() < 200:
                continue
            _, bgmed, bgstd = sigma_clipped_stats(st[bgsel], sigma=2.5)
            st -= bgmed

            dprof, dnpix = radial_profile(st, stbad, cx, cy)

            # model stamp at same detector position + subpixel phase
            xdet = xi - c0 + cx
            ydet = yi - c0 + cy
            myy, mxx = np.mgrid[0:st.shape[0], 0:st.shape[1]]
            mst = grid.evaluate(mxx + (xi - c0), myy + (yi - c0),
                                1.0, xdet, ydet)
            mprof, _ = radial_profile(mst, np.zeros_like(mst, bool), cx, cy)

            # normalize model to data on the core
            csel = ((r_mid >= CORE_NORM_R[0]) & (r_mid <= CORE_NORM_R[1]) &
                    np.isfinite(dprof) & np.isfinite(mprof) & (mprof > 0))
            if csel.sum() < 4:
                continue
            A = np.median(dprof[csel] / mprof[csel])
            if A <= 0:
                continue
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = dprof / (A * mprof)

            all_ratio.append(ratio)
            all_dprof.append(dprof / A)
            all_mprof.append(mprof)
            star_meta.append((fn.split('/')[-1], xdet, ydet, pki - med,
                              bgmed, bgstd, A))
            nstar_this += 1
        print(f'  {fn.split("/")[-1]}: {nstar_this} stars')

    all_ratio = np.array(all_ratio)
    all_dprof = np.array(all_dprof)
    all_mprof = np.array(all_mprof)
    n = len(all_ratio)
    print(f'total stars: {n}')
    if n < 5:
        print('TOO FEW STARS')
        return

    med_ratio = np.nanmedian(all_ratio, axis=0)
    mad = 1.4826 * np.nanmedian(
        np.abs(all_ratio - med_ratio[None, :]), axis=0)
    err = mad / np.sqrt(np.maximum(np.sum(np.isfinite(all_ratio), axis=0), 1))

    np.savez(f'{SCRATCH}/wingratio_{DET}_{FILT}.npz',
             r_mid=r_mid, r_edges=r_edges,
             med_ratio=med_ratio, mad=mad, err=err,
             all_ratio=all_ratio, all_dprof=all_dprof, all_mprof=all_mprof,
             meta=np.array(star_meta, dtype=object))

    print('\n  r[px]   data/model   err')
    for rq in [2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 48]:
        i = np.argmin(np.abs(r_mid - rq))
        print(f'  {r_mid[i]:5.1f}   {med_ratio[i]:8.3f}   {err[i]:6.3f}')


if __name__ == '__main__':
    main()
