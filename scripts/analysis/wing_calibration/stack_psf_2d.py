"""Build 2D median stacks of real bright-star stamps and matched STPSF-model
stamps, through IDENTICAL machinery, to visualize PSF-wing mismatch.

Per star (peak 300-3500 MJy/sr, unsaturated, isolated):
  * cut 151x151 stamp, mask DO_NOT_USE, subtract LocalBackground-style robust
    annulus median (r=55-70 px)
  * refine flux-weighted centroid (iterative, 7-px box)
  * evaluate GriddedPSFModel at the same detector position AND sub-pixel phase
  * fitted flux A = lsq scale of model to data on core (r<6 px); data /= A
  * bilinear (order=1) shift of BOTH stamps to the common center using the
    SAME centroid estimator on each; crop to 101x101
Median-stack data stamps and model stamps identically.

Sanity checks stored:
  * NULL stack: model*A + real ERR-map Gaussian noise, pushed through the
    IDENTICAL machinery (noisy centroid, annulus bg, flux fit, shift, stack).
    If noise/registration/normalization biases could fake a wing excess, they
    would appear here; null/model must be ~1.
  * single model shifted by the median centroid uncertainty vs unshifted
    (mis-centering scale)

Usage: python stack_psf_2d.py <filt> <det>   e.g. f182m nrca1
"""
import sys
import glob
import warnings
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import maximum_filter, shift as ndshift

warnings.filterwarnings('ignore')

SCRATCH = ('/blue/adamginsburg/adamginsburg/tmp/claude-3663/'
           '-blue-adamginsburg-adamginsburg-repos-jwst-gc-pipeline/'
           'd046d21c-902e-4ed6-b230-4b20ac93f54c/scratchpad/agent_calcurves')

FILT = sys.argv[1] if len(sys.argv) > 1 else 'f182m'
DET = sys.argv[2] if len(sys.argv) > 2 else 'nrca1'

# nrcalong frames use the nrca5 PSF grid
PSFDET = {'nrcalong': 'nrca5', 'nrcblong': 'nrcb5'}.get(DET, DET)

FRAME_GLOB = (f'/blue/adamginsburg/adamginsburg/jwst/brick/{FILT.upper()}/'
              f'pipeline/jw02221*_{DET}_destreak_o001_crf.fits')
PSF_FN = (f'/blue/adamginsburg/adamginsburg/jwst/brick/psfs/'
          f'nircam_{PSFDET}_{FILT}_fovp101_samp2_npsf16.fits')

PEAK_MIN, PEAK_MAX = 300.0, 3500.0   # MJy/sr above local median, unsaturated
CUT = 75                             # working stamp half-size -> 151x151
FINAL_HALF = 50                      # final stack half-size  -> 101x101
EDGE = CUT + 5
BG_R_IN, BG_R_OUT = 55.0, 70.0       # local background annulus (px)
CORE_R = 6.0                         # flux-fit core radius (px)
MAX_FRAMES = 16
MAX_STARS = 250


def refine_centroid(stamp, x0, y0, box=7, iters=3):
    """Iterative flux-weighted centroid on a core box (NaN-tolerant)."""
    cx, cy = float(x0), float(y0)
    h = box // 2
    for _ in range(iters):
        ix, iy = int(round(cx)), int(round(cy))
        sub = stamp[iy - h:iy + h + 1, ix - h:ix + h + 1]
        if sub.shape != (box, box):
            return cx, cy
        sub = np.clip(np.nan_to_num(sub, nan=0.0), 0, None)
        tot = sub.sum()
        if tot <= 0:
            return cx, cy
        yy, xx = np.mgrid[0:box, 0:box]
        cx = ix - h + (sub * xx).sum() / tot
        cy = iy - h + (sub * yy).sum() / tot
    return cx, cy


def shift_stamp(st, bad, dy, dx):
    """Bilinear shift with validity-weighted renormalization; NaN where the
    interpolation drew >30% weight from bad pixels."""
    val = (~bad & np.isfinite(st)).astype(float)
    stf = np.where(val > 0, st, 0.0)
    ss = ndshift(stf, (dy, dx), order=1, mode='constant', cval=0.0)
    vv = ndshift(val, (dy, dx), order=1, mode='constant', cval=0.0)
    return np.where(vv > 0.7, ss / np.maximum(vv, 1e-9), np.nan)


def register(stamp, bad):
    """Centroid with the shared estimator, shift centroid to (CUT, CUT),
    crop to 101x101. Returns (cropped, cx, cy) or (None, cx, cy)."""
    cx, cy = refine_centroid(np.where(bad, np.nan, stamp), CUT, CUT)
    if abs(cx - CUT) > 1.6 or abs(cy - CUT) > 1.6:
        return None, cx, cy
    sh = shift_stamp(stamp, bad, CUT - cy, CUT - cx)
    c = sh[CUT - FINAL_HALF:CUT + FINAL_HALF + 1,
           CUT - FINAL_HALF:CUT + FINAL_HALF + 1]
    return c, cx, cy


def main():
    from stpsf.utils import to_griddedpsfmodel
    grid = to_griddedpsfmodel(PSF_FN)
    if isinstance(grid, list):
        grid = grid[0]

    frames = sorted(glob.glob(FRAME_GLOB))[:MAX_FRAMES]
    print(f'{FILT} {DET}: {len(frames)} frames available', flush=True)

    rng = np.random.default_rng(abs(hash(FILT + DET)) % 2**32)
    dstamps, mstamps, nstamps, meta, sigcens = [], [], [], [], []
    nofalse = np.zeros((2 * CUT + 1, 2 * CUT + 1), bool)

    for fn in frames:
        if len(dstamps) >= MAX_STARS:
            break
        with fits.open(fn) as hdul:
            sci = hdul['SCI'].data.astype(float)
            err = hdul['ERR'].data.astype(float)
            dq = hdul['DQ'].data
        bad = (dq & 1).astype(bool) | ~np.isfinite(sci)
        sat = (dq & 2).astype(bool)
        med = np.nanmedian(sci)
        scif = np.where(bad, med, sci)

        mx = maximum_filter(scif, size=5)
        peaks = (scif == mx) & (scif > PEAK_MIN + med)
        py, px = np.nonzero(peaks)
        keep = ((px > EDGE) & (px < sci.shape[1] - EDGE) &
                (py > EDGE) & (py < sci.shape[0] - EDGE))
        py, px = py[keep], px[keep]
        pk = scif[py, px]

        # all significant maxima, for the isolation test
        nbpk = (scif == mx) & (scif > 20.0 + med)
        ny, nx = np.nonzero(nbpk)
        nv = scif[ny, nx] - med
        sy, sx = np.nonzero(sat)

        n_this = 0
        for xi, yi, pki in zip(px, py, pk):
            if len(dstamps) >= MAX_STARS:
                break
            p = pki - med
            if not (PEAK_MIN <= p <= PEAK_MAX):
                continue
            # no saturated pixel within 20 px (sat halos corrupt wings)
            if len(sx) and np.any((sx - xi) ** 2 + (sy - yi) ** 2 < 20 ** 2):
                continue
            # isolation: no >5% neighbor within 12 px, no >30% within 30 px
            d2 = (nx - xi) ** 2 + (ny - yi) ** 2
            if np.any((d2 > 4) & (d2 < 12 ** 2) & (nv > 0.05 * p)):
                continue
            if np.any((d2 > 4) & (d2 < 30 ** 2) & (nv > 0.30 * p)):
                continue

            sl = np.s_[yi - CUT:yi + CUT + 1, xi - CUT:xi + CUT + 1]
            st = sci[sl].copy()
            stbad = bad[sl]
            ste = err[sl]
            # clean core required for centroid + flux fit
            if stbad[CUT - 3:CUT + 4, CUT - 3:CUT + 4].any():
                continue

            cx, cy = refine_centroid(np.where(stbad, np.nan, st), CUT, CUT)
            if abs(cx - CUT) > 1.6 or abs(cy - CUT) > 1.6:
                continue

            # local background
            yy, xx2 = np.mgrid[0:st.shape[0], 0:st.shape[1]]
            rr = np.hypot(xx2 - cx, yy - cy)
            bgsel = (rr > BG_R_IN) & (rr < BG_R_OUT) & ~stbad & np.isfinite(st)
            if bgsel.sum() < 500:
                continue
            _, bgmed, bgstd = sigma_clipped_stats(st[bgsel], sigma=2.5)
            st -= bgmed

            # model at the same detector position + sub-pixel phase
            xdet = xi - CUT + cx
            ydet = yi - CUT + cy
            myy, mxx = np.mgrid[0:st.shape[0], 0:st.shape[1]]
            mst = grid.evaluate(mxx + (xi - CUT), myy + (yi - CUT),
                                1.0, xdet, ydet)

            # fitted flux: lsq scale of model to data on the core
            csel = (rr < CORE_R) & ~stbad & np.isfinite(st) & (mst > 0)
            if csel.sum() < 30:
                continue
            A = np.sum(st[csel] * mst[csel]) / np.sum(mst[csel] ** 2)
            if A <= 0:
                continue

            # NULL stamp: model*A + real ERR noise, treated exactly like data
            nse = np.where(np.isfinite(ste) & ~stbad, ste, 0.0)
            nstamp = mst * A + rng.normal(size=st.shape) * nse
            _, nbg, _ = sigma_clipped_stats(nstamp[bgsel], sigma=2.5)
            nstamp = nstamp - nbg
            An = (np.sum(nstamp[csel] * mst[csel]) /
                  np.sum(mst[csel] ** 2))
            if An <= 0:
                continue

            dreg, dcx, dcy = register(st / A, stbad)
            mreg, mcx, mcy = register(mst, nofalse)
            nreg, _, _ = register(nstamp / An, stbad)
            if dreg is None or mreg is None or nreg is None:
                continue
            if not np.isfinite(dreg[FINAL_HALF, FINAL_HALF]):
                continue

            # centroid uncertainty from the ERR extension (core box)
            h = 3
            iy0, ix0 = int(round(cy)), int(round(cx))
            csub = st[iy0 - h:iy0 + h + 1, ix0 - h:ix0 + h + 1]
            esub = ste[iy0 - h:iy0 + h + 1, ix0 - h:ix0 + h + 1]
            F = np.nansum(np.clip(csub, 0, None))
            xxc, yyc = np.mgrid[0:2 * h + 1, 0:2 * h + 1]
            if F > 0 and np.all(np.isfinite(esub)):
                scx = np.sqrt(np.nansum(esub ** 2 *
                                        (xxc - h - (cx - ix0)) ** 2)) / F
                scy = np.sqrt(np.nansum(esub ** 2 *
                                        (yyc - h - (cy - iy0)) ** 2)) / F
                sigcens.append(0.5 * (scx + scy))

            dstamps.append(dreg.astype(np.float32))
            mstamps.append(mreg.astype(np.float32))
            nstamps.append(nreg.astype(np.float32))
            meta.append((fn.split('/')[-1], float(xdet), float(ydet),
                         float(p), float(A), float(bgmed), float(bgstd)))
            n_this += 1
        print(f'  {fn.split("/")[-1]}: +{n_this} (total {len(dstamps)})',
              flush=True)

    n = len(dstamps)
    print(f'{FILT} {DET}: N stars = {n}', flush=True)
    if n < 10:
        print('TOO FEW STARS', flush=True)
        return

    dstamps = np.array(dstamps)
    mstamps = np.array(mstamps)
    nstamps = np.array(nstamps)
    data_stack = np.nanmedian(dstamps, axis=0)
    model_stack = np.nanmedian(mstamps, axis=0)
    null_stack = np.nanmedian(nstamps, axis=0)

    # reference: a single perfectly-centered model (for context only)
    xm = np.median([m[1] for m in meta])
    ym = np.median([m[2] for m in meta])
    cyy, cxx = np.mgrid[0:101, 0:101]
    ref_model = grid.evaluate(cxx + xm - 50.0, cyy + ym - 50.0, 1.0, xm, ym)

    # --- sanity check 2: mis-centering by the median centroid uncertainty ---
    sig_cen = float(np.median(sigcens)) if sigcens else np.nan
    if np.isfinite(sig_cen):
        sh_model = grid.evaluate(cxx + xm - 50.0 - sig_cen,
                                 cyy + ym - 50.0, 1.0, xm, ym)
    else:
        sh_model = ref_model.copy()

    np.savez_compressed(
        f'{SCRATCH}/stack2d_{FILT}_{DET}.npz',
        data_stack=data_stack, model_stack=model_stack,
        null_stack=null_stack,
        dstamps=dstamps, mstamps=mstamps,
        ref_model=ref_model, shifted_model=sh_model,
        sig_cen=sig_cen, n_stars=n,
        meta=np.array(meta, dtype=object))
    print(f'saved stack2d_{FILT}_{DET}.npz  (sig_cen={sig_cen:.4f} px)',
          flush=True)


if __name__ == '__main__':
    main()
