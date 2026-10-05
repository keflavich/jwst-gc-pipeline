"""Cutouts of benchmark stars whose satstar seed came from an off-centre NaN-VAR_POISSON clump
(_refine_coms_by_data "genuine core" branch).  Left: crf (asinh), cyan = SATURATED outline,
red squares = NaN-variance pixels carrying OUTLIER or bad-pixel DQ bits, orange squares = other
NaN-variance pixels; markers: x = seed used (x_init), + = eroded-core seed (the fallback),
o = dolphot position.  Middle / right: satstar residual of the two arms with the fit position.
usage: python fig_seed.py OUT.png A B"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import simple_norm
from scipy import ndimage
from stdatamodels.jwst.datamodels import dqflags
out, A, B = sys.argv[1:4]
sys.argv = sys.argv[:1]
import score_rc as S
EXCL = 0
for k in ('OUTLIER', 'DEAD', 'HOT', 'WARM', 'LOW_QE', 'RC', 'TELEGRAPH', 'NO_LIN_CORR', 'NO_SAT_CHECK',
          'NO_GAIN_VALUE', 'NO_FLAT_FIELD', 'UNRELIABLE_BIAS', 'OTHER_BAD_PIXEL', 'REFERENCE_PIXEL'):
    EXCL |= dqflags.pixel[k]
STARS = [('F187N', 'jw03523005001_06101_00002_nrcb1', 57.50, 756.44),
         ('F250M', 'jw03523005001_04101_00001_nrcalong', 1603.49, 803.00),
         ('F212N', 'jw03523005001_04101_00002_nrca1', 1589.43, 1208.39)]
hw = 9
fig, axes = plt.subplots(len(STARS), 3, figsize=(9.0, 3.1 * len(STARS)), squeeze=False)
for row, (band, fr, x, y) in zip(axes, STARS):
    F = S.frame_data(band, fr)
    rx, ry = F['w'].world_to_pixel(F['rsc'])
    i = np.argmin(np.hypot(rx - x, ry - y))
    rm = F['rm'][i]
    f = fits.open(f'{S.R}/{band}/pipeline/{fr}_align_o005_crf.fits')
    xi, yi = int(round(x)), int(round(y))
    sl = (slice(yi - hw, yi + hw + 1), slice(xi - hw, xi + hw + 1))
    ext = (xi - hw - 0.5, xi + hw + 0.5, yi - hw - 0.5, yi + hw + 0.5)
    crf = f['SCI'].data.astype(float)[sl]
    dq = f['DQ'].data[sl].astype(np.int64)
    sat = (dq & 2) > 0
    nanv = np.isnan(f['VAR_POISSON'].data[sl]) & sat
    bad = nanv & ((dq & EXCL) != 0)
    e = ndimage.binary_erosion(sat, iterations=2)
    ey, ex = ndimage.center_of_mass(e)
    ey += yi - hw; ex += xi - hw
    pk = np.nanpercentile(crf, 99.5)
    ax = row[0]
    ax.imshow(crf, origin='lower', cmap='gray', extent=ext,
              norm=simple_norm(crf, 'asinh', vmin=np.nanpercentile(crf, 1), vmax=pk))
    yy, xx = np.mgrid[yi - hw:yi + hw + 1, xi - hw:xi + hw + 1]
    ax.contour(xx, yy, sat, levels=[0.5], colors='c', linewidths=0.7)
    for m, c in ((bad, 'r'), (nanv & ~bad, 'orange')):
        for py, px in zip(*np.where(m)):
            ax.add_patch(plt.Rectangle((px + xi - hw - 0.5, py + yi - hw - 0.5), 1, 1, fill=False, ec=c, lw=1.2))
    t = Table.read(f'{S.H}/tree_{A}/{band}/pipeline/{fr}_align_o005_crf_rctest_satstar_catalog.fits')
    j = np.argmin(np.hypot(np.asarray(t['x_0'], float) - x, np.asarray(t['y_0'], float) - y))
    sx = float(t['x_init'][j] + t['x_0'][j] - t['x_fit'][j]); sy = float(t['y_init'][j] + t['y_0'][j] - t['y_fit'][j])
    ax.plot(sx, sy, 'x', color='m', ms=9, mew=2, label=f'seed used ({np.hypot(sx - rx[i], sy - ry[i]):.1f} px)')
    ax.plot(ex, ey, '+', color='lime', ms=10, mew=2, label=f'eroded seed ({np.hypot(ex - rx[i], ey - ry[i]):.1f} px)')
    ax.plot(rx[i], ry[i], 'o', mfc='none', mec='y', ms=8, mew=1.5, label='dolphot')
    ax.legend(fontsize=6, loc='upper left', framealpha=0.6)
    ax.set_title(f'{band} {fr.split("_", 1)[1]}\ndolphot {rm:.2f}; DQ of red px: {sorted(set(dq[bad].tolist()))}', fontsize=7)
    lim = 0.1 * pk
    for ax, arm in ((row[1], A), (row[2], B)):
        r = fits.getdata(f'{S.H}/tree_{arm}/{band}/pipeline/{fr}_align_o005_crf_rctest_satstar_residual.fits').astype(float)[sl]
        med = np.nanmedian(r)
        ax.imshow(r, origin='lower', cmap='RdBu_r', vmin=med - lim, vmax=med + lim, extent=ext)
        ms, _ = S.arm_mags(band, fr, arm, F)
        t = Table.read(f'{S.H}/tree_{arm}/{band}/pipeline/{fr}_align_o005_crf_rctest_satstar_catalog.fits')
        j = np.argmin(np.hypot(np.asarray(t['x_0'], float) - x, np.asarray(t['y_0'], float) - y))
        ax.plot(float(t['x_0'][j]), float(t['y_0'][j]), 'k.', ms=6)
        ax.plot(rx[i], ry[i], 'o', mfc='none', mec='y', ms=8, mew=1.5)
        m = ms[i]
        s = f'{m:.2f} (dm {m - rm:+.2f})' if np.isfinite(m) else 'no row'
        ax.set_title(f'residual {arm}: {s}\nfit flags {int(t["flags"][j])}, qfit {float(t["qfit"][j]):.2f}', fontsize=7)
    for ax in row:
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f'satstar seeds on off-centre NaN-variance clumps ({A} vs {B}; residual scale: median +-10% of crf 99.5th pct)', fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 1 - 0.1 / len(STARS)))
fig.savefig(out, dpi=100)
print('wrote', out)
