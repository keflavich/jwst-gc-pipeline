"""Fit-domain wing calibration: per-annulus LSQ amplitude ratio.

For each bright star, per radial annulus:  a(r) = sum(d*m)/sum(m^2)
(the amplitude a 2D LSQ fit would assign using only that annulus), with a
3-sigma residual clip against the core-anchored prediction to reject
neighbor stars. Core amplitude A = LSQ over r<6 px. C_eff(r) = median over
stars of a(r)/A.

Usage: python measure_fitdomain.py nrca1 f182m 12
"""
import sys
import glob
import warnings
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import maximum_filter

warnings.filterwarnings('ignore')

SCRATCH = ('/blue/adamginsburg/adamginsburg/tmp/claude-3663/'
           '-blue-adamginsburg-adamginsburg-repos-jwst-gc-pipeline/'
           'd046d21c-902e-4ed6-b230-4b20ac93f54c/scratchpad/agent_empsf')
DET = sys.argv[1] if len(sys.argv) > 1 else 'nrca1'
FILT = sys.argv[2] if len(sys.argv) > 2 else 'f182m'
NFRAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 12

FRAME_GLOB = (f'/blue/adamginsburg/adamginsburg/jwst/brick/{FILT.upper()}/'
              f'pipeline/jw02221001001_*_{DET}_destreak_o001_crf.fits')
PSF_FN = (f'/blue/adamginsburg/adamginsburg/jwst/brick/psfs/'
          f'nircam_{DET}_{FILT}_fovp101_samp2_npsf16.fits')

PEAK_MIN, PEAK_MAX = 800.0, 3500.0
HALF = 55
EDGE = HALF + 5
BG_R_IN, BG_R_OUT = 55.0, 70.0
R_MAXP = 46.0
r_edges = np.concatenate([np.arange(0, 12, 1.0),
                          np.arange(12, R_MAXP + 2, 2.0)])
r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])


def refine_centroid(stamp, x0, y0, box=7):
    h = box // 2
    ix, iy = int(round(x0)), int(round(y0))
    sub = np.nan_to_num(stamp[iy - h:iy + h + 1, ix - h:ix + h + 1], nan=0.0)
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

    frames = sorted(glob.glob(FRAME_GLOB))[:NFRAMES]
    print(f'{DET} {FILT}: fit-domain calib, {len(frames)} frames')
    ratios, Alist = [], []

    for fn in frames:
        with fits.open(fn) as hdul:
            sci = hdul['SCI'].data.astype(float)
            dq = hdul['DQ'].data
        bad = (dq & 1).astype(bool) | ~np.isfinite(sci)
        sat = (dq & 2).astype(bool)
        med = np.nanmedian(sci)
        scif = np.where(bad, med, sci)
        mx = maximum_filter(scif, size=5)
        py, px = np.nonzero((scif == mx) & (scif > PEAK_MIN + med))
        keep = ((px > EDGE) & (px < 2048 - EDGE) & (py > EDGE) &
                (py < 2048 - EDGE))
        py, px = py[keep], px[keep]
        pk = scif[py, px] - med
        # neighbor list
        nby, nbx = np.nonzero((scif == mx) & (scif > 20 + med))
        nv = scif[nby, nbx] - med

        n_this = 0
        for xi, yi, pki in zip(px, py, pk):
            if pki > PEAK_MAX:
                continue
            if sat[yi - 5:yi + 6, xi - 5:xi + 6].any():
                continue
            d2 = (nbx - xi) ** 2 + (nby - yi) ** 2
            near = (d2 > 4) & (d2 < 12 ** 2)
            if np.any(nv[near] > 0.05 * pki):
                continue

            st = sci[yi - HALF - 20:yi + HALF + 21,
                     xi - HALF - 20:xi + HALF + 21].copy()
            stbad = bad[yi - HALF - 20:yi + HALF + 21,
                        xi - HALF - 20:xi + HALF + 21]
            c0 = HALF + 20
            cx, cy = refine_centroid(np.where(stbad, np.nan, st), c0, c0)
            yy, xx2 = np.mgrid[0:st.shape[0], 0:st.shape[1]]
            rr = np.hypot(xx2 - cx, yy - cy)
            bgsel = (rr > BG_R_IN) & (rr < BG_R_OUT) & ~stbad & np.isfinite(st)
            if bgsel.sum() < 200:
                continue
            _, bgmed, _ = sigma_clipped_stats(st[bgsel], sigma=2.5)
            d_im = st - bgmed

            xdet, ydet = xi - c0 + cx, yi - c0 + cy
            m_im = grid.evaluate(xx2 + (xi - c0), yy + (yi - c0),
                                 1.0, xdet, ydet)

            good = np.isfinite(d_im) & ~stbad
            core = good & (rr >= 1.0) & (rr < 6.0)
            if core.sum() < 20:
                continue
            A = np.sum(d_im[core] * m_im[core]) / np.sum(m_im[core] ** 2)
            if A <= 0:
                continue
            # residual clip against core-anchored prediction (kill neighbors)
            res = d_im - A * m_im
            _, rmed, rstd = sigma_clipped_stats(res[good & (rr > 6)],
                                                sigma=3.0)
            okpix = good & (np.abs(res - rmed) < 4 * rstd)

            a_r = np.full(len(r_mid), np.nan)
            for j in range(len(r_mid)):
                sel = okpix & (rr >= r_edges[j]) & (rr < r_edges[j + 1])
                if sel.sum() < 8:
                    continue
                denom = np.sum(m_im[sel] ** 2)
                if denom <= 0:
                    continue
                a_r[j] = np.sum(d_im[sel] * m_im[sel]) / denom
            ratios.append(a_r / A)
            Alist.append(A)
            n_this += 1
        print(f'  {fn.split("/")[-1]}: {n_this} stars')

    ratios = np.array(ratios)
    Alist = np.array(Alist)
    print(f'total {len(ratios)} stars')
    Ceff = np.nanmedian(ratios, axis=0)
    nfin = np.sum(np.isfinite(ratios), axis=0)
    err = (1.4826 * np.nanmedian(np.abs(ratios - Ceff), axis=0)
           / np.sqrt(np.maximum(nfin, 1)))
    print('\n  r[px]   C_eff    err')
    for j in range(len(r_mid)):
        print(f'  {r_mid[j]:5.1f}   {Ceff[j]:6.3f}   {err[j]:6.3f}')

    np.savez(f'{SCRATCH}/fitdomain_{DET}_{FILT}.npz',
             r_mid=r_mid, r_edges=r_edges, Ceff=Ceff, err=err,
             ratios=ratios, A=Alist)


if __name__ == '__main__':
    main()
