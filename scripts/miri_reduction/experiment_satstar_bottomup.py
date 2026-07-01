#!/usr/bin/env python
"""EXPERIMENT (2026-06-14): bottom-up (lower-envelope) satstar amplitude vs LSQ.

Problem: the current MIRI satstar fits OVER-subtract badly (ssr_ratio median ~45;
models 17-28x the local data; residual pits to -156000 MJy/sr).  LSQ on the
masked-core + wings overshoots, partly because the inner wings sit on extended
emission.

Idea (user): fit from the bottom up -- raise the amplitude only until the model
PSF matches/just touches the data at the INNER wings, so it can never
over-subtract.  Combine with 2D local-background subtraction so the wing fit
isn't pulled up by extended emission.

This prototypes the amplitude math on REAL saturated stars from one crf frame
(no pipeline changes), comparing:
  A_lsq        : inverse-variance LSQ over unmasked pixels (what the pipeline does)
  A_env(p)     : p-th percentile of (data-bg)/psf over inner-wing pixels
                 (p small -> lower envelope, no over-subtraction)
and reporting, for each, ssr_ratio (resid^2 vs (data-median)^2 in inner region)
and the most-negative residual.  Goal: a rule that keeps EVERY saturated star
(100% fit) with ssr ~1 and no large negatives.
"""
import os
import sys
import glob
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from astropy.io import fits
from scipy.ndimage import label, binary_dilation, median_filter, distance_transform_edt
from stpsf.utils import to_griddedpsfmodel

os.environ.setdefault("CRDS_PATH", "/orange/adamginsburg/jwst/brick/crds/")
from jwst.datamodels import dqflags

CRF = sys.argv[1] if len(sys.argv) > 1 else \
    sorted(glob.glob('/orange/adamginsburg/jwst/sickle/F770W/pipeline/'
                     'jw03958002001_*_o002_crf.fits'))[1]
PSFGRID = '/orange/adamginsburg/jwst/sickle/psfs/miri_mirim_f770w_fovp101_samp4_npsf16.fits'
PAD = 60          # cutout half-size
INNER_LO = 3      # inner-wing annulus: just outside the dilated sat mask
BG_MEDFILT = 25   # 2D local-bg median-filter box (extended emission)


def psf_cutout(grid, xc, yc, ny, nx):
    """Evaluate the gridded PSF (unit total flux) centred at (xc,yc)."""
    y, x = np.mgrid[0:ny, 0:nx]
    grid.x_0 = xc; grid.y_0 = yc; grid.flux = 1.0
    return grid(x, y)


def ssr(resid, data, fitpix):
    a = np.sum(resid[fitpix] ** 2)
    b = np.sum((data[fitpix] - np.median(data[fitpix])) ** 2)
    return a / max(b, 1e-30)


def main():
    grid = to_griddedpsfmodel(PSFGRID)
    if isinstance(grid, list):
        grid = grid[0]
    fh = fits.open(CRF)
    data = fh['SCI'].data.astype(float)
    dq = fh['DQ'].data
    err = fh['ERR'].data.astype(float)
    sat = (dq & dqflags.pixel['SATURATED']) > 0
    lab, n = label(sat)
    sizes = np.bincount(lab.ravel())
    # brightest saturated cores = largest sat area
    order = np.argsort(sizes[1:])[::-1] + 1
    print(f'frame {os.path.basename(CRF)}: {n} saturated components', flush=True)
    print(f'{"star":>4} {"satpx":>5} {"A_lsq":>9} {"ssr_lsq":>8} {"minres_lsq":>10} '
          f'| {"A_env10":>9} {"ssr_e10":>8} {"minres_e10":>10} '
          f'| {"A_env25":>9} {"ssr_e25":>8} {"minres_e25":>10}', flush=True)
    for k, lid in enumerate(order[:8]):
        ys, xs = np.where(lab == lid)
        yc, xc = ys.mean(), xs.mean()
        y0, y1 = int(yc - PAD), int(yc + PAD)
        x0, x1 = int(xc - PAD), int(xc + PAD)
        if y0 < 0 or x0 < 0 or y1 > data.shape[0] or x1 > data.shape[1]:
            continue
        cut = data[y0:y1, x0:x1].copy()
        ecut = err[y0:y1, x0:x1].copy()
        satc = (lab[y0:y1, x0:x1] == lid)
        satdil = binary_dilation(satc, iterations=3)
        ny, nx = cut.shape
        # 2D local background (extended emission) from a masked median filter
        fillv = np.where(satdil | ~np.isfinite(cut), np.nan, cut)
        bg = median_filter(np.nan_to_num(fillv, nan=np.nanmedian(fillv)), size=BG_MEDFILT)
        d = cut - bg
        psf = psf_cutout(grid, xc - x0, yc - y0, ny, nx)
        # fit pixels: unmasked, finite, inside r<15px (inner region for ssr)
        yy, xx = np.mgrid[0:ny, 0:nx]
        r = np.hypot(xx - (xc - x0), yy - (yc - y0))
        unmasked = (~satdil) & np.isfinite(cut) & (cut != 0)
        inner = unmasked & (r < 25)
        # inner-wing annulus for the envelope amplitude
        dist = distance_transform_edt(~satdil)
        wing = unmasked & (dist >= INNER_LO) & (r < 25) & (psf > psf.max() * 1e-4)
        if wing.sum() < 8 or inner.sum() < 8:
            print(f'  skip star{k}: satpx={int(sizes[lid])} inner={inner.sum()} '
                  f'wing={wing.sum()} psfmax={psf.max():.2e}', flush=True)
            continue
        w = 1.0 / np.maximum(ecut, 1e-6) ** 2
        A_lsq = np.sum((w * d * psf)[unmasked]) / max(np.sum((w * psf * psf)[unmasked]), 1e-30)
        ratios = (d[wing] / psf[wing])
        ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
        out = [k, int(sizes[lid])]
        for A, tag in [(A_lsq, 'lsq'),
                       (np.percentile(ratios, 10), 'e10'),
                       (np.percentile(ratios, 25), 'e25')]:
            model = A * psf
            res = d - model
            out += [A, ssr(res, d, inner), float(np.nanmin(res[inner]))]
        print('%4d %5d %9.2e %8.1f %10.0f | %9.2e %8.1f %10.0f | %9.2e %8.1f %10.0f'
              % tuple(out), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
