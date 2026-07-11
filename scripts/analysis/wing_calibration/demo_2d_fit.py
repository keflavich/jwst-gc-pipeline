"""2D prototype: CorrectedGriddedPSFModel in a real masked-core PSF fit.

Wraps GriddedPSFModel.evaluate with a radial multiplicative correction
C(r) (r = detector-px distance from x_0,y_0). Fits real bright unsaturated
stars from one frame with photutils PSFPhotometry twice each:
  (a) full stamp, uncorrected model      -> truth flux
  (b) core masked to r<R_MASK px, uncorrected model  -> production bias
  (c) core masked, corrected model       -> prototype
Reports flux(b)/flux(a) and flux(c)/flux(a) in mag.

Usage: python demo_2d_fit.py nrca1 f182m [R_MASK] [nstars]
"""
import sys
import warnings
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from scipy.ndimage import maximum_filter
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

SCRATCH = ('/blue/adamginsburg/adamginsburg/tmp/claude-3663/'
           '-blue-adamginsburg-adamginsburg-repos-jwst-gc-pipeline/'
           'd046d21c-902e-4ed6-b230-4b20ac93f54c/scratchpad/agent_empsf')
DET = sys.argv[1] if len(sys.argv) > 1 else 'nrca1'
FILT = sys.argv[2] if len(sys.argv) > 2 else 'f182m'
R_MASK = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
NSTARS = int(sys.argv[4]) if len(sys.argv) > 4 else 40

# use a frame NOT in the 12 used for calibration (nrca1 used frames 1-12)
FRAME = (f'/blue/adamginsburg/adamginsburg/jwst/brick/{FILT.upper()}/pipeline/'
         f'jw02221001001_07101_00018_{DET}_destreak_o001_crf.fits')
PSF_FN = (f'/blue/adamginsburg/adamginsburg/jwst/brick/psfs/'
          f'nircam_{DET}_{FILT}_fovp101_samp2_npsf16.fits')

from stpsf.utils import to_griddedpsfmodel
from photutils.psf import PSFPhotometry


def make_corrected_model(grid, r_c, C_c):
    """Subclass the *class of grid* so photutils copy semantics survive."""
    corr = interp1d(r_c, C_c, bounds_error=False,
                    fill_value=(C_c[0], C_c[-1]))

    class CorrectedGriddedPSFModel(type(grid)):
        """GriddedPSFModel with radial multiplicative wing correction."""
        _radial_corr = staticmethod(corr)

        def evaluate(self, x, y, flux, x_0, y_0):
            out = super().evaluate(x, y, flux, x_0, y_0)
            r = np.hypot(np.asanyarray(x) - x_0, np.asanyarray(y) - y_0)
            return out * self._radial_corr(r)

    new = grid.copy()
    new.__class__ = CorrectedGriddedPSFModel
    return new


def main():
    grid = to_griddedpsfmodel(PSF_FN)
    if isinstance(grid, list):
        grid = grid[0]

    cdat = np.load(f'{SCRATCH}/wing_correction_{DET}_{FILT}.npz')
    cgrid = make_corrected_model(grid, cdat['spline_r'], cdat['spline_C'])

    with fits.open(FRAME) as hdul:
        sci = hdul['SCI'].data.astype(float)
        err = hdul['ERR'].data.astype(float)
        dq = hdul['DQ'].data
    bad = (dq & 1).astype(bool) | ~np.isfinite(sci)
    sat = (dq & 2).astype(bool)
    med = np.nanmedian(sci)
    scif = np.where(bad, med, sci)
    mx = maximum_filter(scif, size=5)
    py, px = np.nonzero((scif == mx) & (scif > 300 + med))
    keep = (px > 60) & (px < 1988) & (py > 60) & (py < 1988)
    py, px = py[keep], px[keep]
    pk = scif[py, px] - med
    order = np.argsort(-pk)
    py, px, pk = py[order], px[order], pk[order]

    HALF = 35  # fit window half-size (r<=30 wing + margin)
    rows = []
    for xi, yi, pki in zip(px, py, pk):
        if len(rows) >= NSTARS or pki > 3500:
            if pki > 3500:
                continue
            break
        if sat[yi - 5:yi + 6, xi - 5:xi + 6].any():
            continue
        # isolation: no neighbor peak >10% within 15 px
        d2 = (px - xi) ** 2 + (py - yi) ** 2
        near = (d2 > 4) & (d2 < 15 ** 2)
        if np.any(pk[near] > 0.10 * pki):
            continue

        st = sci[yi - HALF:yi + HALF + 1, xi - HALF:xi + HALF + 1]
        er = err[yi - HALF:yi + HALF + 1, xi - HALF:xi + HALF + 1]
        bd = bad[yi - HALF:yi + HALF + 1, xi - HALF:xi + HALF + 1]
        yy, xx = np.mgrid[0:st.shape[0], 0:st.shape[1]]
        rr = np.hypot(xx - HALF, yy - HALF)
        bsel = (rr > 30) & ~bd & np.isfinite(st)
        if bsel.sum() < 100:
            continue
        _, bg, _ = sigma_clipped_stats(st[bsel], sigma=2.5)
        stz = np.where(bd, np.nan, st - bg)

        init = Table({'x_0': [float(xi)], 'y_0': [float(yi)],
                      'flux_0': [float(pki * 30)]})

        def dofit(model, mask_core):
            m = ~np.isfinite(stz)
            if mask_core:
                m |= rr < R_MASK
            # global-coords stamp fit
            phot = PSFPhotometry(model, fit_shape=(2 * HALF + 1,) * 2,
                                 aperture_radius=5)
            xg = np.arange(xi - HALF, xi + HALF + 1)
            yg = np.arange(yi - HALF, yi + HALF + 1)
            # photutils works in array coords; shift init to stamp coords
            t = Table({'x_0': [float(HALF)], 'y_0': [float(HALF)],
                       'flux_0': [float(pki * 30)]})
            try:
                res = phot(np.nan_to_num(stz, nan=0.0), mask=m,
                           error=np.where(np.isfinite(er) & (er > 0), er,
                                          np.nanmedian(er)),
                           init_params=t)
            except (ValueError, RuntimeError):
                return np.nan
            return float(res['flux_fit'][0])

        f_truth = dofit(grid, False)
        f_unc = dofit(grid, True)
        f_cor = dofit(cgrid, True)
        if not (np.isfinite(f_truth) and f_truth > 0):
            continue
        rows.append((xi, yi, pki, f_truth, f_unc, f_cor))

    rows = np.array(rows)
    print(f'{DET} {FILT} frame 09101_00013: {len(rows)} stars fit, '
          f'R_MASK={R_MASK}')
    dm_unc = -2.5 * np.log10(rows[:, 4] / rows[:, 3])
    dm_cor = -2.5 * np.log10(rows[:, 5] / rows[:, 3])

    def s(v):
        v = v[np.isfinite(v)]
        return (f'median {np.median(v):+.3f} mag, '
                f'MAD-sigma {1.4826 * np.median(np.abs(v - np.median(v))):.3f}'
                f'  (n={len(v)})')
    print('masked-core fit vs full-fit truth:')
    print(f'  uncorrected model: {s(dm_unc)}')
    print(f'  corrected model:   {s(dm_cor)}')
    np.savez(f'{SCRATCH}/demo2d_{DET}_{FILT}_rmask{int(R_MASK)}.npz',
             rows=rows, dm_unc=dm_unc, dm_cor=dm_cor)


if __name__ == '__main__':
    main()
