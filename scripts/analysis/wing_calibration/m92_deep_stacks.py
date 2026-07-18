"""M92 sparse-field deep-mask wing-calibration campaign — STAGE 1: stacks.

Build sub-pixel-registered empirical (data) and STPSF-model 2D PSF stacks
from bright unsaturated isolated stars in the M92 GO-1334 crf frames
(zero background, thousands of calibrators).  Same estimators as the Brick
measurement (see measure_wing_ratio.py / README.md):

  * stamp, DQ mask, local annulus background (55-70 px)
  * core flux-weighted centroid refinement
  * azimuthal clipped-median radial profile (for the core normalization A)
  * GriddedPSFModel evaluated at the same detector position + subpixel phase
  * model normalized to data on core r in [1.5, 6] px
  * recentered 2D stamps, median-stacked; ALSO per-frame median stacks so
    stage 2 (m92_deep_calcurve.py) can estimate frame-to-frame scatter.

Purpose: extend the masked-core wing-calibration curves C(r_mask) beyond
r=10 px (target 20-25 px).  In crowded fields the production self-cal
(saturated_star_finding.apply_wing_selfcal) has no calibrators at r>10 px
and CLAMPS the correction there (np.interp on clipped radii), leaving a
residual ~ -0.25 mag for the deepest-mask satstar stratum
(docs/reports/SATURATED_STAR_PHOTOMETRY_ARTICLE.md section 6 item 1).

Usage: python m92_deep_stacks.py <filter> <detector> [nframes] [outdir]
e.g.   python m92_deep_stacks.py f090w nrca1 4 /orange/adamginsburg/jwst/m92/wingcal_deep_stacks
"""
import os
import sys
import glob
import warnings
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import maximum_filter, shift as ndshift

warnings.filterwarnings('ignore')

FIELD = 'm92'
FILT = sys.argv[1] if len(sys.argv) > 1 else 'f090w'
DET = sys.argv[2] if len(sys.argv) > 2 else 'nrca1'
NFRAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 4
OUTDIR = (sys.argv[4] if len(sys.argv) > 4
          else '/orange/adamginsburg/jwst/m92/wingcal_deep_stacks')

FRAME_GLOB = (f'/orange/adamginsburg/jwst/{FIELD}/{FILT.upper()}/'
              f'pipeline/jw0*_{DET}_destreak_o001_crf.fits')
PSF_FN = (f'/orange/adamginsburg/jwst/{FIELD}/psfs/'
          f'nircam_{DET}_{FILT}_fovp101_samp2_npsf16.fits')

# m92 F090W/F150W: brightest unsaturated peaks ~1500-1700 MJy/sr
PEAK_MIN, PEAK_MAX = 250.0, 1300.0
HALF = 55                              # stamp half-size (px)
PAD = 20                               # extra padding -> stamps 151x151
EDGE = HALF + PAD + 5
BG_R_IN, BG_R_OUT = 55.0, 70.0         # local background annulus
CORE_NORM_R = (1.5, 6.0)               # radii used to scale model to data
R_MAX = 46.0                           # stay inside model 101-px FOV edge

r_edges = np.concatenate([np.arange(0, 8, 0.5),
                          np.arange(8, 30, 1.0),
                          np.arange(30, R_MAX + 2, 2.0)])
r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])


def _clipped_median(vals, sigma=2.5, iters=4):
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
    yy, xx = np.mgrid[0:stamp.shape[0], 0:stamp.shape[1]]
    rr = np.hypot(xx - cx, yy - cy)
    prof = np.full(len(r_mid), np.nan)
    good = np.isfinite(stamp) & ~mask_bad
    for i in range(len(r_mid)):
        sel = (rr >= r_edges[i]) & (rr < r_edges[i + 1]) & good
        if sel.sum() >= 3:
            prof[i] = _clipped_median(stamp[sel])
    return prof


def refine_centroid(stamp, x0, y0, box=7):
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

    os.makedirs(OUTDIR, exist_ok=True)
    frames = sorted(glob.glob(FRAME_GLOB))[:NFRAMES]
    print(f'{FIELD} {DET} {FILT}: using {len(frames)} frames', flush=True)
    if not frames:
        print('NO FRAMES FOUND')
        return

    all_ratio, star_meta = [], []
    frame_names, nstars_frames = [], []
    dstack_frames, mstack_frames = [], []   # per-frame median stacks
    all_dstack, all_mstack = [], []         # per-star stacks (all frames)

    for fn in frames:
        with fits.open(fn) as hdul:
            sci = hdul['SCI'].data.astype(float)
            dq = hdul['DQ'].data
        bad = (dq & 1).astype(bool) | ~np.isfinite(sci)  # DO_NOT_USE
        sat = (dq & 2).astype(bool)                       # SATURATED

        med = np.nanmedian(sci)
        scif = np.where(bad, med, sci)
        mx = maximum_filter(scif, size=5)
        peaks = (scif == mx) & (scif > PEAK_MIN + med)
        py, px = np.nonzero(peaks)
        keep = ((px > EDGE) & (px < 2048 - EDGE) & (py > EDGE) &
                (py < 2048 - EDGE))
        py, px = py[keep], px[keep]
        pk = scif[py, px]

        nb_thresh = 20.0
        nbpk = (scif == mx) & (scif > nb_thresh + med)
        ny, nx = np.nonzero(nbpk)
        nv = scif[ny, nx]

        dstack, mstack = [], []
        for xi, yi, pki in zip(px, py, pk):
            if not (PEAK_MIN <= pki - med <= PEAK_MAX):
                continue
            if sat[yi - 5:yi + 6, xi - 5:xi + 6].any():
                continue
            # isolation: no neighbor >5% of peak within 12 px
            d2 = (nx - xi) ** 2 + (ny - yi) ** 2
            near = (d2 > 4) & (d2 < 12 ** 2)
            if np.any(nv[near] - med > 0.05 * (pki - med)):
                continue

            c0 = HALF + PAD
            st = sci[yi - c0:yi + c0 + 1, xi - c0:xi + c0 + 1].copy()
            stbad = bad[yi - c0:yi + c0 + 1, xi - c0:xi + c0 + 1]
            if st.shape != (2 * c0 + 1, 2 * c0 + 1):
                continue
            cx, cy = refine_centroid(np.where(stbad, np.nan, st), c0, c0)

            yy, xx2 = np.mgrid[0:st.shape[0], 0:st.shape[1]]
            rr = np.hypot(xx2 - cx, yy - cy)
            bgsel = (rr > BG_R_IN) & (rr < BG_R_OUT) & ~stbad & np.isfinite(st)
            if bgsel.sum() < 200:
                continue
            _, bgmed, bgstd = sigma_clipped_stats(st[bgsel], sigma=2.5)
            st -= bgmed

            dprof = radial_profile(st, stbad, cx, cy)

            xdet = xi - c0 + cx
            ydet = yi - c0 + cy
            myy, mxx = np.mgrid[0:st.shape[0], 0:st.shape[1]]
            mst = grid.evaluate(mxx + (xi - c0), myy + (yi - c0),
                                1.0, xdet, ydet)
            mprof = radial_profile(mst, np.zeros(st.shape, bool), cx, cy)

            csel = ((r_mid >= CORE_NORM_R[0]) & (r_mid <= CORE_NORM_R[1]) &
                    np.isfinite(dprof) & np.isfinite(mprof) & (mprof > 0))
            if csel.sum() < 4:
                continue
            A = np.median(dprof[csel] / mprof[csel])
            if A <= 0:
                continue
            with np.errstate(divide='ignore', invalid='ignore'):
                all_ratio.append(dprof / (A * mprof))
            star_meta.append((fn.split('/')[-1], xdet, ydet, pki - med,
                              bgmed, bgstd, A))

            # ---- 2D stacks: shift to common center, normalize by A ----
            dx, dy = cx - c0, cy - c0
            stn = np.where(stbad, np.nan, st) / A
            dsh = ndshift(np.nan_to_num(stn, nan=0.0), (-dy, -dx), order=1,
                          mode='constant', cval=0.0)
            wsh = ndshift(np.isfinite(stn).astype(float), (-dy, -dx),
                          order=1, mode='constant', cval=0.0)
            dsh = np.where(wsh > 0.7, dsh / np.maximum(wsh, 1e-6), np.nan)
            msh = ndshift(mst, (-dy, -dx), order=1, mode='constant', cval=0.0)
            dstack.append(dsh)
            mstack.append(msh)
        print(f'  {fn.split("/")[-1]}: {len(dstack)} stars', flush=True)
        frame_names.append(fn.split('/')[-1])
        nstars_frames.append(len(dstack))
        if len(dstack) >= 3:
            dstack_frames.append(np.nanmedian(np.array(dstack), axis=0))
            mstack_frames.append(np.nanmedian(np.array(mstack), axis=0))
        else:
            dstack_frames.append(np.full((2 * (HALF + PAD) + 1,) * 2, np.nan))
            mstack_frames.append(np.full((2 * (HALF + PAD) + 1,) * 2, np.nan))
        all_dstack.extend(dstack)
        all_mstack.extend(mstack)

    n = len(all_dstack)
    print(f'total stars: {n}', flush=True)
    if n < 20:
        print('TOO FEW STARS')
        return

    med_dstack = np.nanmedian(np.array(all_dstack), axis=0)
    med_mstack = np.nanmedian(np.array(all_mstack), axis=0)
    all_ratio = np.array(all_ratio)
    med_ratio = np.nanmedian(all_ratio, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(all_ratio - med_ratio[None, :]), axis=0)
    err = mad / np.sqrt(np.maximum(np.sum(np.isfinite(all_ratio), axis=0), 1))

    out = f'{OUTDIR}/wingratio_{FIELD}_{DET}_{FILT}.npz'
    np.savez_compressed(
        out, r_mid=r_mid, r_edges=r_edges,
        med_ratio=med_ratio, mad=mad, err=err,
        med_dstack=med_dstack, med_mstack=med_mstack,
        dstack_frames=np.array(dstack_frames),
        mstack_frames=np.array(mstack_frames),
        frame_names=np.array(frame_names),
        nstars_frames=np.array(nstars_frames),
        meta=np.array(star_meta, dtype=object),
        psf_fn=PSF_FN)
    print(f'wrote {out}', flush=True)

    print('\n  r[px]   data/model(median)   err')
    for rq in [3, 5, 7, 10, 12, 15, 20, 25, 30, 40, 45]:
        i = np.argmin(np.abs(r_mid - rq))
        print(f'  {r_mid[i]:5.1f}   {med_ratio[i]:8.3f}   {err[i]:9.3f}')


if __name__ == '__main__':
    main()
